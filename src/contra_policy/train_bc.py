"""Stage B: behaviour cloning, one whole episode per sequence.

    python -m contra_policy.train_bc
    python -m contra_policy.train_bc train.steps=200 loader.num_workers=0   # smoke

**The gate is `val/bc_acc` against 0.76**, the number the previous windowed recurrent
policy reached. That is what decides whether 0002's bet holds: a plain causal
transformer over whole episodes learning this as well as a Transformer-XL over a
32-step window with carried memory.

It is a *proxy*, deliberately. The number that matters is task completion against
72.8%, which needs closed-loop rollout in ``contra_nes_evaluation`` and cannot be
measured here. The two can disagree — the `prev_action` ablation barely moved accuracy
while collapsing boss completion 8.8% → 1.8% — so a passing `bc_acc` licenses the next
step, not the conclusion.

Per-family accuracy matters more than the pooled figure: `traverse` is 65% of training
steps and the family already handled best, so a pooled move says little about where it
came from.

The encoder is **frozen by default**. That makes the causal core the only thing that
changed, so a bad number is unambiguously its fault rather than a co-adaptation between
two things moving at once. Unfreeze once the bet is confirmed.
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
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from contra_policy.dataset import (FAMILIES, ContraCrossViewDataset, LengthGroupedSampler,
                                   load_or_build_index, pad_episodes, shard_paths)
from contra_policy.loss import ContraObjective, action_class_weights
from contra_policy.model import PREFIX, PolicyConfig, build_policy


class CSVLogger:
    """Append-only CSV with a growing header."""

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


def _mean_of(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = {k for r in rows for k in r}
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in sorted(keys)}


def _model_tokens(batch: Dict) -> int:
    """Dense causal-transformer tokens in one padded BC batch.

    This deliberately follows build-nanogpt's throughput convention: count tensor
    positions the backbone computes, including padding, rather than only positions that
    reach the loss. Each episode contributes the ``interaction`` and ``goal`` prefix
    tokens followed by the batch-padded frame sequence. ``frames`` remains the useful,
    unpadded counterpart, so the two rates expose padding overhead instead of hiding it.
    """
    batch_size, padded_frames = batch["image"].shape[:2]
    return int(batch_size) * (int(padded_frames) + PREFIX)


class BCTrainer:
    def __init__(self, args: DictConfig, run_dir: str):
        self.args, self.run_dir = args, run_dir
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[args.precision]
        if self.device.type != "cuda":
            self.autocast_dtype = None

        # -- data ------------------------------------------------------------
        shard_dir = os.path.expanduser(args.shard_dir)
        fams = list(args.families)
        train_idx = load_or_build_index(shard_paths(shard_dir, fams, "train"), args.cache_dir)
        val_idx = load_or_build_index(shard_paths(shard_dir, fams, "val"), args.cache_dir)
        ds_kw = dict(whole_episode=True, image_size=int(args.image_size),
                     sigma_px=float(args.sigma_px), aux_size=int(args.policy.aux_size),
                     prev_action_keep_prob=0.0, seed=int(args.seed))
        self.train_ds = ContraCrossViewDataset(train_idx, **ds_kw)
        self.val_ds = ContraCrossViewDataset(val_idx, **ds_kw)
        # -1 because the last frame of an episode has no action taken *from* it.
        self.train_len = [max(1, e["length"] - 1) for e in train_idx]
        self.val_len = [max(1, e["length"] - 1) for e in val_idx]

        counts = np.zeros(21, dtype=np.int64)
        for ep in train_idx:
            counts += np.asarray(ep["action_counts"], dtype=np.int64)
        longest = max(self.train_len + self.val_len)
        print(f"[bc] {len(train_idx)} train / {len(val_idx)} val episodes · "
              f"longest {longest} frames", flush=True)

        # -- model -----------------------------------------------------------
        pcfg = PolicyConfig(**OmegaConf.to_container(args.policy, resolve=True))
        self.policy = build_policy(pcfg).to(self.device)
        if longest + 2 > self.policy.context:
            raise ValueError(
                f"longest episode is {longest} frames + 2 prefix = {longest + 2}, over "
                f"context {self.policy.context}. Raise policy.core.context — it is a "
                f"config value, not a trained parameter.")
        n_tr = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        n_fr = sum(p.numel() for p in self.policy.parameters() if not p.requires_grad)
        print(f"[bc] policy {n_tr/1e6:.2f}M trainable + {n_fr/1e6:.2f}M frozen "
              f"({'frozen' if pcfg.freeze_encoder else 'TRAINABLE'} encoder) · "
              f"context {self.policy.context}", flush=True)

        modal = int(counts.argmax())
        print(f"[bc] modal action is index {modal} at {counts[modal]/counts.sum():.1%} "
              f"of train steps — a constant predictor would score that as bc_acc, so "
              f"watch bc_bal_acc and pred_modal_frac beside it", flush=True)
        w = action_class_weights(counts, alpha=float(args.loss.action_loss_alpha))
        self.objective = ContraObjective(
            bc_weight=float(args.loss.bc_weight),
            heatmap_weight=float(args.loss.heatmap_weight),
            heatmap_pos_weight=float(args.loss.heatmap_pos_weight),
            label_smoothing=float(args.loss.label_smoothing),
            class_weights=None if w is None else w.to(self.device),
            families=tuple(FAMILIES), modal_action=modal).to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=float(args.train.learning_rate), weight_decay=float(args.train.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_scale)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.autocast_dtype is torch.float16))

        self.logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.step = 0
        self.best = -1.0

    # -- plumbing -----------------------------------------------------------

    def _lr_scale(self, step: int) -> float:
        warm, total = int(self.args.train.warmup_steps), int(self.args.train.steps)
        if warm > 0 and step < warm:
            return (step + 1) / warm
        if self.args.train.lr_decay == "cosine":
            p = (step - warm) / max(1, total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
        return 1.0

    def _loader(self, ds, lengths, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds, collate_fn=pad_episodes,
            batch_sampler=LengthGroupedSampler(
                lengths, int(self.args.loader.batch_size),
                pool_batches=int(self.args.loader.pool_batches),
                seed=int(self.args.seed), shuffle=shuffle),
            num_workers=int(self.args.loader.num_workers),
            pin_memory=True,
            prefetch_factor=int(self.args.loader.prefetch_factor)
            if int(self.args.loader.num_workers) > 0 else None)

    def _to_device(self, batch: Dict) -> Dict:
        out = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in batch.items() if k != "cross_view"}
        out["cross_view"] = {k: v.to(self.device, non_blocking=True)
                             for k, v in batch["cross_view"].items()}
        return out

    def _forward(self, batch: Dict):
        cv = batch["cross_view"]
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null())
        with ctx:
            latents = self.policy(batch["image"], cv["cross_view_image"],
                                  cv["cross_view_obj_id"])
            loss, metrics = self.objective(latents, batch)
        return loss, metrics

    # -- the loop -----------------------------------------------------------

    def train_step(self, batch: Dict) -> Dict[str, float]:
        self.policy.train()
        if self.policy.cfg.freeze_encoder:
            self.policy.encoder.eval()      # keep frozen norms in inference mode
        batch = self._to_device(batch)
        loss, metrics = self._forward(batch)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        gn = torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            float(self.args.train.max_grad_norm))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        row = {k: float(v) for k, v in metrics.items()}
        row.update({"grad_norm": float(gn), "lr": self.optimizer.param_groups[0]["lr"],
                    "frames": float(batch["mask"].sum()),
                    "tokens": float(_model_tokens(batch)),
                    "pad_frac": 1.0 - float(batch["mask"].mean())})
        return row

    @torch.no_grad()
    def validate(self, max_batches: int = 0) -> Dict[str, float]:
        self.policy.eval()
        rows: List[Dict[str, float]] = []
        for i, batch in enumerate(self._loader(self.val_ds, self.val_len, shuffle=False)):
            if max_batches and i >= max_batches:
                break
            _loss, metrics = self._forward(self._to_device(batch))
            rows.append({k: float(v) for k, v in metrics.items()})
        return _mean_of(rows)

    def run(self) -> None:
        total = int(self.args.train.steps)
        t0, seen_frames, seen_tokens = time.time(), 0.0, 0.0
        try:
            while self.step < total:
                for batch in self._loader(self.train_ds, self.train_len, shuffle=True):
                    if self.step >= total:
                        break
                    row = self.train_step(batch)
                    self.step += 1
                    seen_frames += row["frames"]
                    seen_tokens += row["tokens"]
                    if self.step % int(self.args.train.log_every) == 0:
                        # CUDA launches asynchronously. Synchronise before stopping the
                        # clock or throughput measures Python enqueue speed, not GPU work.
                        if self.device.type == "cuda":
                            torch.cuda.synchronize(self.device)
                        elapsed = max(1e-9, time.time() - t0)
                        row["step"] = self.step
                        row["frames_per_s"] = seen_frames / elapsed
                        row["tokens_per_sec"] = seen_tokens / elapsed
                        t0, seen_frames, seen_tokens = time.time(), 0.0, 0.0
                        self._emit(row, "train")
                    if self.step % int(self.args.train.val_every) == 0:
                        self._run_val(int(self.args.train.val_batches))
        finally:
            # Whole val set however the run ended — this is the gate.
            self._run_val(0, tag="val_full")
            self.save(final=True)

    def _run_val(self, batches: int, tag: str = "val") -> None:
        v = self.validate(batches)
        v["step"] = self.step
        self._emit(v, tag)
        acc = v.get("bc_acc", -1.0)
        if acc > self.best:
            self.best = acc
            self.save(best=True)

    def _emit(self, row: Dict[str, float], phase: str) -> None:
        self.logger.log({**row, "phase": phase})
        head = [f"{k}={row[k]:.4g}" for k in
                ("loss", "bc_acc", "bc_bal_acc", "bc_nonmodal_acc", "pred_modal_frac",
                 "point_err_px") if k in row]
        line = f"[{phase} {self.step}/{int(self.args.train.steps)}] " + " ".join(head)
        for k in ("tokens_per_sec", "frames_per_s", "pad_frac"):
            if k in row:
                line += f" {k}={row[k]:.3g}"
        print(line, flush=True)
        # Per family, because traverse is 65% of steps and the family already handled
        # best — a pooled move says little about where it came from.
        fam = [f"{f}({row[f'{f}/bc_acc']:.3f})" for f in FAMILIES
               if f"{f}/bc_acc" in row]
        if fam:
            print("    bc_acc: " + " ".join(fam)
                  + "   [gate 0.76; constant-R scores 0.68]", flush=True)

    def save(self, final: bool = False, best: bool = False) -> str:
        tag = "final" if final else ("best" if best else f"{self.step:06d}")
        path = os.path.join(self.run_dir, "checkpoints", f"policy-{tag}.pt")
        self.policy.save(path, step=self.step, best_bc_acc=self.best,
                         train_config=OmegaConf.to_container(self.args, resolve=True))
        return path


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@hydra.main(config_path=".", config_name="config_bc", version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _install_signal_handlers()
    _seed_everything(int(args.seed))
    run_dir = os.getcwd()
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as fh:
        fh.write(OmegaConf.to_yaml(args, resolve=True))
    BCTrainer(args, run_dir=run_dir).run()


def _install_signal_handlers() -> None:
    """SIGTERM as a normal exit, so `run`'s `finally` validates and saves.

    Python's default disposition terminates without running `finally`, which loses both
    the final checkpoint and the gate measurement.
    """
    def _exit(signum, _frame):
        print(f"\n[bc] {signal.Signals(signum).name} received — validating and saving",
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
