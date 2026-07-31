"""Stage A: pretrain the frame encoder on goal grounding alone.

No temporal model, no policy, no actions — one frame in, one token out, occupancy
decoded from that token. That is deliberate: it isolates the single question this
rebuild turns on, which is whether one 512-d token can hold as much spatial structure
as the four view tokens it replaces.

    python -m contra_encoder.train
    python -m contra_encoder.train train.steps=500 loader.num_workers=0   # smoke

**The gate is `peak_hit` and `pck16` per family, plus entity `dice` — not
``point_err_px``.** That was the original gate and it was wrong: ``points_to_target``
collapses a frame's goal centroids to their mean, and boss goals have 4.57 components
spread ~34 px on 98.7% of frames, so the target names an empty spot and the error
*grows* as a predictor sharpens. Measured: boss went 2.6 px to 8.8 px across one run
while ``peak_hit`` reached 0.999 and the three single-centroid families improved 7-19x.

``point_err_px`` is still reported — it is the evaluator's pinned statistic — but only
over ``n_goal_points == 1`` frames, with ``multi_goal_frac`` saying how much was
dropped. Do **not** gate on ``exist_acc`` either: the goal is visible on 100.0% of kill
and boss val frames, so a constant predictor scores 100%.

``[val]`` lines are what decides whether stage B is worth starting.

Metrics land in ``metrics.csv`` beside the checkpoints; the encoder is written with its
own config embedded so ``load_pretrained_encoder`` can rebuild it without being told the
architecture.
"""

from __future__ import annotations

import csv
import math
import os
import signal
import time
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from contra_encoder.data import FAMILIES, build_datamodule, flatten_window
from contra_encoder.net import EncoderConfig, build_encoder
from contra_policy.loss import GoalHeatmapLoss, point_err_px


# ── metric plumbing ───────────────────────────────────────────────────────────

class CSVLogger:
    """Append-only CSV with a growing header, matching the RL trainer's logger."""

    def __init__(self, path: str):
        self.path, self.keys = path, []

    def log(self, row: Dict[str, float]) -> None:
        new = [k for k in row if k not in self.keys]
        if new:
            old = self._read()
            self.keys += new
            with open(self.path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.keys, restval="")
                w.writeheader()
                w.writerows(old)
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.keys, restval="").writerow(row)

    def _read(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return list(csv.DictReader(fh))


#: Screen-pixel radii for PCK. 8 px is roughly one NES sprite — "found the thing";
#: 16 px is "in the right neighbourhood".
PCK_RADII = (8.0, 16.0)


def per_family_grounding(pred: Dict[str, torch.Tensor],
                         batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Grounding quality, pooled and split by family.

    Which of these to trust, and why they are not interchangeable:

    ``peak_hit``          **the gate.** Does the argmax *cell* land inside a target
                          blob. Defined for any number of goal components, so it is the
                          one localisation number comparable across all four families.
    ``pck16``             fraction localised within 16 screen px. Outlier-proof
                          companion to the gate.
    ``point_err_px``      the evaluator's pinned statistic — but **only reported over
                          single-centroid frames**, see below. ``pck8`` likewise.
    ``exist_acc``         **degenerate; do not gate on it.** On val the goal is visible
                          on 100.0% of kill and 100.0% of boss frames (95.4% item,
                          38.8% traverse), so a constant "visible" predictor scores
                          100% on the two families that matter.

    **Why point statistics exclude multi-component goals.** ``points_to_target``
    collapses a frame's goal centroids to their *mean*. With one centroid that is the
    thing's location. Boss goals span all live components — 4.57 per frame on average,
    up to 7, spread ~34 px from their mean, on 98.7% of frames — so the "target point"
    names a spot where nothing is, and error against it *grows as a predictor gets
    sharper*: a blurry map's centre of mass sits near the cloud's centre, a confident
    one sits on a component. That is not a hypothesis. The first trained encoder went
    2.6 px at step 3000 to 8.8 px at step 20000 on boss while `peak_hit` reached 0.999
    and every single-centroid family improved 7-19x.

    So point statistics are masked to ``n_goal_points == 1``, which drops ~99% of boss
    frames and none elsewhere. ``multi_goal_frac`` reports how much was excluded, so a
    thin boss sample can never be mistaken for a good one. ``point_err_px`` itself is
    untouched — it is a frozen interface shared with ``contra_nes_evaluation``; what
    changed is only which frames we average it over.
    """
    out: Dict[str, float] = {}
    err = point_err_px(pred["point"], batch["point"])            # (N,)
    vis = batch["exist"] > 0
    acc = ((pred["exist"].squeeze(-1) > 0).float() == batch["exist"]).float()
    fam = batch["family"]
    hit = _peak_hit(pred["goal_heatmap"], batch["goal_heatmap"])
    # Shards written before 2026-07-31 carry no centroid count; assume single, which is
    # what every family except boss actually is.
    n_pts = batch.get("n_goal_points")
    single = torch.ones_like(vis) if n_pts is None else (n_pts == 1)

    def block(prefix: str, sel: torch.Tensor) -> None:
        out[f"{prefix}exist_acc"] = float(acc[sel].mean())
        v = sel & vis
        if not bool(v.any()):
            return
        # Defined for any goal shape → the gate.
        out[f"{prefix}peak_hit"] = float(hit[v].mean())
        out[f"{prefix}multi_goal_frac"] = float((~single[v]).float().mean())
        p = v & single
        if not bool(p.any()):
            return
        e = err[p]
        out[f"{prefix}point_frames"] = float(p.sum())
        out[f"{prefix}point_err_px"] = float(e.mean())
        out[f"{prefix}point_err_px_p50"] = float(e.median())
        for r in PCK_RADII:
            out[f"{prefix}pck{int(r)}"] = float((e <= r).float().mean())

    block("", torch.ones_like(vis))
    for i, name in enumerate(FAMILIES):
        sel = fam == i
        if not bool(sel.any()):
            continue
        out[f"{name}/frames"] = float(sel.sum())
        block(f"{name}/", sel)
    return out


#: Channel order of the entity target, matching ``ContraCrossViewDataset.ENTITY_CLASSES``
#: and ``env.entity.HEATMAP_CLASSES``.
ENTITY_CLASSES = ("player", "player_bullets", "enemies", "enemy_bullets")


def entity_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 10.0
                ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weighted BCE over the four entity occupancy channels, plus a per-class read-out.

    Same shape of objective as ``GoalHeatmapLoss`` — cell-weighted by
    ``1 + pos_weight * target`` and normalised — but *not* the same module, because that
    one is tied to the ``(B, T, ...)`` masked contract and to the goal's ``point`` /
    ``exist`` readout, neither of which applies here.

    Per-class ``peak_hit`` is reported because the classes are wildly unequal: on boss
    frames there are ~4.9 enemy bullets of ~2 px each against exactly 1 player sprite, so
    a pooled number would be carried by the easy class. That per-class split is the whole
    reason this target is worth having.
    """
    bce = F.binary_cross_entropy_with_logits(pred.float(), target, reduction="none")
    cell_w = 1.0 + pos_weight * target
    per_class = (bce * cell_w).flatten(2).mean(-1) / (1.0 + pos_weight * 0.5)   # (N, C)
    loss = per_class.mean()

    metrics: Dict[str, torch.Tensor] = {"entity_loss": loss.detach()}
    with torch.no_grad():
        prob = torch.sigmoid(pred.float())
        n, c = per_class.shape
        for i, name in enumerate(ENTITY_CLASSES[:c]):
            metrics[f"entity/{name}/loss"] = per_class[:, i].mean()
            present = target[:, i].flatten(1).max(-1).values > 0.5
            if not bool(present.any()):
                continue
            p, t = prob[:, i][present], target[:, i][present]
            metrics[f"entity/{name}/dice"] = soft_dice(p, t)
            metrics[f"entity/{name}/mse_skill"] = mse_skill(p, t)
            metrics[f"entity/{name}/peak_hit"] = _peak_hit(pred[:, i][present], t).mean()
    return loss, metrics


def soft_dice(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-9
              ) -> torch.Tensor:
    """``2Σpt / (Σp² + Σt²)`` on the raw probabilities — no threshold anywhere.

    The metric to read first. These maps are 95-98% empty, so anything scored over all
    cells equally is dominated by how well the background was predicted, which is
    trivial. Dice only counts overlap against positive mass, so an all-zero prediction
    scores 0.000 and a perfect one 1.000, with "found half the mass" landing near 0.5 —
    a scale that means the same thing for `player` (1 instance) and `enemy_bullets`
    (~5 tiny ones).

    Squared denominator (the V-Net form), not the more familiar ``Σp + Σt``. Our targets
    are *soft* Gaussians, not binary masks, and with a linear denominator an exactly
    correct prediction scores ``Σt²/Σt`` — 0.90 on a typical blob, not 1.0. A metric
    whose ceiling moves with the target's shape is unreadable; this one peaks at 1.0 for
    any target.
    """
    num = 2.0 * (pred * target).flatten(1).sum(-1)
    den = (pred ** 2).flatten(1).sum(-1) + (target ** 2).flatten(1).sum(-1)
    return (num / den.clamp_min(eps)).mean()


def mse_skill(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-9
              ) -> torch.Tensor:
    """``1 - MSE(pred) / MSE(all-zeros)`` — MSE, made readable.

    Plain MSE is not wrong here, it is *unscaled*: predicting nothing anywhere already
    scores 0.0021 (player) to 0.0065 (enemies), and the whole useful range sits below
    that, differently per class. Referencing it to the all-zero predictor turns it into
    the fraction of the target's energy actually explained — 0 for predicting nothing,
    1 for exact, and negative for a prediction worse than silence (e.g. confident mass
    in the wrong place, which plain Dice cannot go negative to signal).
    """
    mse = ((pred - target) ** 2).flatten(1).mean(-1)
    base = (target ** 2).flatten(1).mean(-1)
    return (1.0 - mse / base.clamp_min(eps)).mean()


def _peak_hit(pred_heat: torch.Tensor, target_heat: torch.Tensor,
              thresh: float = 0.5) -> torch.Tensor:
    """Per-frame: does the predicted argmax cell fall inside the target blob?

    ``goal_mask`` renders a Gaussian with peak 1.0 at each centroid, so ``> 0.5`` is
    "within about a sigma of a real goal". Uses the *cell* argmax rather than the
    soft-argmax, which is the whole point — it cannot be fooled by mass split between
    two locations averaging out to a plausible-looking coordinate.
    """
    n, a, _ = pred_heat.shape
    flat_idx = pred_heat.reshape(n, -1).argmax(dim=-1)
    return (target_heat.reshape(n, -1).gather(1, flat_idx[:, None]).squeeze(1)
            > thresh).float()


def _mean_of(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Frame-weighted where a count is available, plain mean otherwise."""
    keys = {k for r in rows for k in r}
    out: Dict[str, float] = {}
    for k in sorted(keys):
        vals = [r[k] for r in rows if k in r]
        if k.endswith("/frames"):
            out[k] = float(sum(vals))
        else:
            out[k] = float(np.mean(vals))
    return out


# ── the loop ──────────────────────────────────────────────────────────────────

class EncoderTrainer:
    def __init__(self, args: DictConfig, run_dir: str):
        self.args, self.run_dir = args, run_dir
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[args.precision]
        if self.device.type != "cuda":
            self.autocast_dtype = None

        cfg = EncoderConfig(**OmegaConf.to_container(args.encoder, resolve=True))
        self.encoder = build_encoder(cfg).to(self.device)
        n_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_frozen = sum(p.numel() for p in self.encoder.parameters() if not p.requires_grad)
        print(f"[enc] {n_train/1e6:.1f}M trainable + {n_frozen/1e6:.1f}M frozen · "
              f"1 token of {cfg.hiddim} per frame · heatmap {cfg.n_classes}x"
              f"{cfg.aux_size}x{cfg.aux_size} decoded from that token", flush=True)
        if cfg.entity_classes > 0:
            print(f"[enc] entity head: {cfg.entity_classes} classes "
                  f"({', '.join(ENTITY_CLASSES[:cfg.entity_classes])}) "
                  f"@ sigma {float(args.loss.entity_sigma_px)}px, "
                  f"weight {float(args.loss.entity_weight)}", flush=True)

        self.objective = GoalHeatmapLoss(float(args.loss.heatmap_weight),
                                         float(args.loss.pos_weight)).to(self.device)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.encoder.parameters() if p.requires_grad],
            lr=float(args.train.learning_rate),
            weight_decay=float(args.train.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_scale)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.autocast_dtype is torch.float16))

        self.data = build_datamodule(
            shard_dir=args.shard_dir, families=list(args.families),
            image_size=cfg.image_size, aux_size=cfg.aux_size,
            sigma_px=float(args.sigma_px), win_len=int(args.loader.win_len),
            batch_size=int(args.loader.batch_size),
            num_workers=int(args.loader.num_workers),
            prefetch_factor=int(args.loader.prefetch_factor),
            cache_dir=args.cache_dir, seed=int(args.seed),
            want_entities=cfg.entity_classes > 0,
            entity_sigma_px=float(args.loss.entity_sigma_px))

        self.logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.wandb = None
        if args.logger == "wandb":
            import wandb
            self.wandb = wandb.init(project=args.project,
                                    config=OmegaConf.to_container(args, resolve=True))
        self.step = 0
        self._last_val_step = -1

    def _lr_scale(self, step: int) -> float:
        warm = int(self.args.train.warmup_steps)
        total = int(self.args.train.steps)
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if self.args.train.lr_decay == "cosine":
            p = (step - warm) / max(1, total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
        return 1.0

    # -- one optimisation step ---------------------------------------------

    def _forward(self, frames: Dict[str, torch.Tensor]):
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null())
        with ctx:
            out = self.encoder(frames["image"], frames["goal_image"],
                               frames["goal_mask"], frames["interaction"])
            # GoalHeatmapLoss works on (B, T, ...) with a mask; a frame batch is T=1
            # with an all-ones mask. Reusing it rather than reimplementing the weighted
            # BCE is what keeps this loss identical to the one BC optimises.
            latents = {"goal_heatmap": out["goal_heatmap"].unsqueeze(1),
                       "point": out["point"].unsqueeze(1),
                       "exist": out["exist"].unsqueeze(1)}
            target = {"goal_heatmap": frames["goal_heatmap"].unsqueeze(1),
                      "point": frames["point"].unsqueeze(1),
                      "exist": frames["exist"].unsqueeze(1),
                      "mask": torch.ones_like(frames["exist"]).unsqueeze(1)}
            loss, metrics = self.objective(latents, target)

            if "entity_heatmap" in out and "entity_heatmap" in frames:
                e_loss, e_metrics = entity_loss(
                    out["entity_heatmap"], frames["entity_heatmap"],
                    pos_weight=float(self.args.loss.entity_pos_weight))
                loss = loss + float(self.args.loss.entity_weight) * e_loss
                metrics.update(e_metrics)
        return loss, metrics, out

    def train_step(self, batch: Dict) -> Optional[Dict[str, float]]:
        frames = flatten_window(batch, self.device)
        if not frames:
            return None
        self.encoder.train()
        loss, metrics, out = self._forward(frames)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gn = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(),
                                           float(self.args.train.max_grad_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        row = {k: float(v) for k, v in metrics.items()}
        row.update({"loss": float(loss.detach()), "grad_norm": float(gn),
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "frames": float(frames["image"].shape[0])})
        with torch.no_grad():
            row.update(per_family_grounding(out, frames))
        return row

    @torch.no_grad()
    def validate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        """``max_batches=0`` (or None -> config) walks the *entire* val set.

        The in-run curve uses a fixed 50-batch subset: same frames every time, so the
        trend is low-variance and comparable across steps. But that subset is 6.4% of
        val and only 613 of the 10,128 val boss frames, which is too thin to decide the
        5.3 px gate on. The closing pass walks all 781 batches (~3 min).
        """
        limit = int(self.args.train.val_batches) if max_batches is None else max_batches
        self.encoder.eval()
        rows: List[Dict[str, float]] = []
        for i, batch in enumerate(self.data.val_dataloader()):
            if limit and i >= limit:
                break
            frames = flatten_window(batch, self.device)
            if not frames:
                continue
            loss, metrics, out = self._forward(frames)
            row = {"loss": float(loss)}
            # Carry the objective's own metrics through — they hold every entity number
            # (4 of the 5 predicted channels). Dropping them made `[val]` report only
            # goal grounding, so the entity head was trained but never validated.
            row.update({k: float(v) for k, v in metrics.items()})
            row.update(per_family_grounding(out, frames))
            rows.append(row)
        return _mean_of(rows)

    def run(self) -> None:
        total = int(self.args.train.steps)
        t0 = time.time()
        try:
            while self.step < total:
                for batch in self.data.train_dataloader():
                    if self.step >= total:
                        break
                    row = self.train_step(batch)
                    self.step += 1
                    if row is None:
                        continue
                    if self.step % int(self.args.train.log_every) == 0:
                        row["step"] = self.step
                        row["frames_per_s"] = row["frames"] * int(
                            self.args.train.log_every) / max(1e-9, time.time() - t0)
                        t0 = time.time()
                        self._emit(row, "train")
                    if self.step % int(self.args.train.val_every) == 0:
                        v = self.validate()
                        v["step"] = self.step
                        self._emit(v, "val")
                        self._last_val_step = self.step
                    if self.step % int(self.args.train.save_every) == 0:
                        self.save()
        finally:
            # Only if the loop did not just validate at this exact step — otherwise a
            # run whose length is a multiple of `val_every` reports the same numbers
            # twice, which reads like a discrepancy.
            # The gate: whole val set, however the run ended. Worth ~3 minutes at the
            # end of a 1.6-hour run, and it is the number stage B is decided on.
            v = self.validate(max_batches=int(self.args.train.final_val_batches))
            v["step"] = self.step
            self._emit(v, "val_full")
            self.save(final=True)
            if self.wandb is not None:
                self.wandb.finish()

    # -- output -------------------------------------------------------------

    def _emit(self, row: Dict[str, float], phase: str) -> None:
        self.logger.log({**row, "phase": phase})
        if self.wandb is not None:
            self.wandb.log({f"{phase}/{k}": v for k, v in row.items() if k != "step"},
                           step=self.step)
        head = [f"{k}={row[k]:.4g}" for k in
                ("loss", "peak_hit", "pck16", "point_err_px", "point_err_px_p50")
                if k in row]
        line = f"[{phase} {self.step}/{int(self.args.train.steps)}] " + " ".join(head)
        if "frames_per_s" in row:
            line += f" frames/s={row['frames_per_s']:.0f}"
        print(line, flush=True)
        # The gate lives on this line: boss point error against the policy's 5.3 px,
        # with pck8 beside it so a mean dragged up by a tail of confusions is visible.
        # peak_hit is the gate: defined for any number of goal components. point_err
        # is shown beside it only where it is meaningful, with `-` where a family's
        # goals are multi-component and the mean-of-centroids target is ill-defined.
        fam = []
        for f in FAMILIES:
            if f"{f}/peak_hit" not in row:
                continue
            pe = (f"{row[f'{f}/point_err_px']:.1f}px" if row.get(f"{f}/point_err_px")
                  and row.get(f"{f}/multi_goal_frac", 0) < 0.5 else "-")
            fam.append(f"{f}({row[f'{f}/peak_hit']:.2f}/{pe})")
        if fam:
            print("    peak_hit/err: " + " ".join(fam), flush=True)
        ent = [f"{c}({row[f'entity/{c}/dice']:.2f})" for c in ENTITY_CLASSES
               if f"entity/{c}/dice" in row]
        if ent:
            # Dice, not peak_hit: the maps are 95-98% empty, so a single-guess hit rate
            # says nothing about whether ~5 enemy bullets were all found. Per class,
            # because a pooled number is carried by `player` — one large always-present
            # sprite — while `enemy_bullets` is the class that would help boss survival.
            print("    entity dice: " + " ".join(ent), flush=True)

    def save(self, final: bool = False) -> str:
        tag = "final" if final else f"{self.step:06d}"
        path = os.path.join(self.run_dir, "checkpoints", f"encoder-{tag}.pt")
        self.encoder.save(path, step=self.step,
                          train_config=OmegaConf.to_container(self.args, resolve=True))
        print(f"[enc] saved {path}", flush=True)
        return path


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ── entry point ───────────────────────────────────────────────────────────────

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _install_signal_handlers()
    _seed_everything(int(args.seed))

    run_dir = os.getcwd()          # hydra has already chdir'd here
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as fh:
        fh.write(OmegaConf.to_yaml(args, resolve=True))
    EncoderTrainer(args, run_dir=run_dir).run()


def _install_signal_handlers() -> None:
    """SIGTERM as a normal exit, so ``run``'s ``finally`` saves and validates.

    Same reasoning as ``train_rl.py``: Python's default disposition terminates without
    running ``finally``, which loses the checkpoint. ``timeout`` and ``kill`` both send
    SIGTERM, so this is the ordinary way a long run ends.
    """
    def _exit(signum, _frame):
        print(f"\n[enc] {signal.Signals(signum).name} received — saving and stopping",
              flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _seed_everything(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
