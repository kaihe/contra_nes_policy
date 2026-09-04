"""Stage A: pretrain the frame encoder on goal grounding alone.

No temporal model, no policy, no actions — one frame in, one token out, occupancy
decoded from that token. That is deliberate: it isolates the single question this
rebuild turns on, which is whether one 512-d token can hold as much spatial structure
as the four view tokens it replaces.

    python -m contra_encoder.train
    python -m contra_encoder.train train.steps=500 loader.num_workers=0   # smoke

**The gate is per-class entity `dice`, split by family**, against 0001's baseline
(`enemies` 0.96, `enemy_bullets` 0.91). Goal grounding is *not* measured here: since
0002 the encoder is goal-agnostic and goal matching belongs to the policy's attention,
so `point_err_px` and `peak_hit` only become measurable again at stage B. That is a
deliberate loss of diagnostic power, recorded in 0002 §3.

Both image kinds are trained: agent frames and the episode's goal frame, which is a
real frame with the target painted on it. `goal_frame_idx` gives its index, so it
carries the same 4-class entity target as any other frame — the encoder sees the
orange-marker distribution rather than meeting it first at stage B.

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


#: Channel order of the entity target, matching ``ContraCrossViewDataset.ENTITY_CLASSES``
#: and ``env.entity.HEATMAP_CLASSES``.
ENTITY_CLASSES = ("player", "player_bullets", "enemies", "enemy_bullets")


def entity_loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: float = 10.0
                ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Weighted BCE over the four entity occupancy channels, plus a per-class read-out.

    Cell-weighted by
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
              f"1 token of {cfg.hiddim} per image · goal-agnostic", flush=True)
        print(f"[enc] entity head: {cfg.entity_classes} classes "
              f"({', '.join(ENTITY_CLASSES[:cfg.entity_classes])}) @ "
              f"{cfg.aux_size}x{cfg.aux_size}, sigma {float(args.loss.entity_sigma_px)}px",
              flush=True)
        print(f"[enc] reconstruction: "
              f"{'on, weight ' + str(float(args.loss.recon_weight)) if cfg.reconstruct else 'off (0002 §4 ablation)'}",
              flush=True)
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
        """One encode over agent frames **and** goal frames, concatenated.

        Goal frames go through the same function and carry the same 4-class target, so
        they are simply more rows in the batch — which is the whole point of 0002. One
        concatenated forward rather than two keeps the GPU batch large; the split is
        only needed to report them separately, since a goal frame is ~1 row per window
        against ~32 agent frames and would otherwise vanish into the mean.
        """
        img, goal = frames["image"], frames.get("goal_image")
        n_frame = img.shape[0]
        has_goal = goal is not None and "goal_entity_heatmap" in frames
        x = torch.cat([img, goal], 0) if has_goal else img

        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null())
        with ctx:
            out = self.encoder(x)
            target = frames["entity_heatmap"]
            if has_goal:
                target = torch.cat([target, frames["goal_entity_heatmap"]], 0)
            loss, metrics = entity_loss(
                out["entity_heatmap"], target,
                pos_weight=float(self.args.loss.entity_pos_weight))

            if has_goal:
                # Reported, not weighted differently: a goal frame is a frame, and its
                # share of the loss is its share of the batch.
                with torch.no_grad():
                    _gl, gm = entity_loss(out["entity_heatmap"][n_frame:],
                                          frames["goal_entity_heatmap"])
                    metrics.update({f"goal_{k}": v for k, v in gm.items()
                                    if k.endswith("/dice")})

            if "reconstruction" in out:
                recon = F.mse_loss(out["reconstruction"], x.float() / 255.0)
                loss = loss + float(self.args.loss.recon_weight) * recon
                metrics["recon_mse"] = recon.detach()
                # PSNR is how the old dreamer AE reported this; keep it comparable.
                metrics["recon_psnr"] = 10.0 * torch.log10(1.0 / recon.detach().clamp_min(1e-9))
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
        row.update(self._per_family(out, frames))
        return row

    @torch.no_grad()
    def _per_family(self, out, frames) -> Dict[str, float]:
        """Entity dice split by family — the gate.

        Per family because the classes are not equally hard and neither are the
        families: boss frames carry ~4.9 enemy bullets of ~2 px each against exactly one
        player sprite, so a pooled number is carried by the easy class in the easy family.
        """
        row: Dict[str, float] = {}
        pred, tgt, fam = out["entity_heatmap"], frames["entity_heatmap"], frames["family"]
        for i, name in enumerate(FAMILIES):
            sel = fam == i
            if not bool(sel.any()):
                continue
            row[f"{name}/frames"] = float(sel.sum())
            _l, m = entity_loss(pred[:sel.shape[0]][sel], tgt[sel])
            for k, v in m.items():
                if k.endswith("/dice"):
                    row[f"{name}/{k.split('/')[1]}"] = float(v)
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
            row.update({k: float(v) for k, v in metrics.items()})
            row.update(self._per_family(out, frames))
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
                ("loss", "recon_psnr", "entity_loss") if k in row]
        line = f"[{phase} {self.step}/{int(self.args.train.steps)}] " + " ".join(head)
        if "frames_per_s" in row:
            line += f" frames/s={row['frames_per_s']:.0f}"
        print(line, flush=True)
        # Per class, because a pooled number is carried by `player` — one large
        # always-present sprite — while `enemy_bullets` is ~2 px and is the class that
        # would help boss survival. `g:` is the goal frame's own dice, which lags if the
        # painted marker has pushed goal frames out of distribution.
        ent = [f"{c}({row[f'entity/{c}/dice']:.2f})" for c in ENTITY_CLASSES
               if f"entity/{c}/dice" in row]
        if ent:
            g = [f"{row[f'goal_entity/{c}/dice']:.2f}" for c in ENTITY_CLASSES
                 if f"goal_entity/{c}/dice" in row]
            print("    entity dice: " + " ".join(ent)
                  + ("   g: " + " ".join(g) if g else ""), flush=True)
        # The gate: entity dice per family, against 0001's enemies 0.96 / e_bullets 0.91.
        fam = [f"{f}({row[f'{f}/enemies']:.2f}/{row[f'{f}/enemy_bullets']:.2f})"
               for f in FAMILIES if f"{f}/enemies" in row and f"{f}/enemy_bullets" in row]
        if fam:
            print("    by family (enemies/e_bullets): " + " ".join(fam), flush=True)

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
