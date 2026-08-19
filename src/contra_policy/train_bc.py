"""Stage B: behaviour cloning, one whole episode per sequence.

    python -m contra_policy.train_bc
    python -m contra_policy.train_bc train.steps=200 loader.num_workers=0   # smoke

The base-policy contract is deliberately GPT-like: masked action cross-entropy is the
only objective, and its optimisation telemetry matches ``build-nanogpt``. Closed-loop
task completion remains an evaluation job; offline diagnostic heads and accuracies do
not belong to this trainer. See ``doc/0006-design-action-only-base-policy.md``.

The encoder is **frozen by default**. That makes the causal core the only thing that
changed, so a bad number is unambiguously its fault rather than a co-adaptation between
two things moving at once. Unfreeze once the bet is confirmed.
"""

from __future__ import annotations

import csv
import math
import json
import os
import random
import signal
import time
from collections import Counter
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from contra_policy.dataset import (FAMILIES, ContraCrossViewDataset,
                                   FixedFamilyBatchSampler, LengthGroupedSampler,
                                   load_or_build_index, pad_episodes, scaling_release,
                                   shard_paths)
from contra_policy.action_space import ACTION_NAMES
from contra_policy.loss import BehaviorCloneLoss
from contra_policy.model import PREFIX, PolicyConfig, build_policy
from contra_policy.datahouse import (DatahouseTokens, datahouse_index,
                                     shard_prefix, split_uids)
from contra_policy.token_cache import (CachedEpisodeDataset, TokenCache,
                                       encoder_fingerprint)

# The dataset's most common action, and the one `tail_ce` excludes. Derived from the
# frozen action space rather than written as an integer, so a reordering there cannot
# silently redefine the metric.
MODAL_ACTION = ACTION_NAMES.index("R")


def wsd_onset(total_steps: int, decay_frac: float) -> int:
    """First step of the cooldown in a warmup-stable-decay schedule.

    WSD is this repo's schedule, and the reason is reusability rather than final loss —
    the two land in about the same place. **Cosine bakes the total budget into the
    schedule**: its LR reaches 0 exactly at ``train.steps``, so a finished run cannot be
    extended and every budget costs a run from scratch. Chinchilla flags this directly —
    the cosine cycle must match the training duration or the result degrades — and it is
    why a 40k cosine checkpoint cannot become an 80k run.

    WSD holds the LR *flat* after warmup and only anneals over the last ``decay_frac`` of
    the budget. Any checkpoint taken in that stable phase is a legitimate starting point
    for **any** longer budget, because the schedule there does not depend on the total. A
    20k/40k/80k ladder is then one trunk plus short anneals rather than three full runs.

    The cooldown is linear to ``final_lr_frac``. (1-sqrt) is reported slightly better in
    the literature and is a one-line change if that ever matters here; linear is kept
    because it is the shape most results were measured with.
    """
    return max(0, int(total_steps) - int(round(float(decay_frac) * int(total_steps))))


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


def _weighted_tail(out: Dict[str, float], rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Re-aggregate ``tail_ce`` over the dataset rather than over batches.

    Non-modal steps are ~22% of frames and cluster by family — a `traverse` batch and a
    `boss` batch contribute very different counts — so the unweighted batch mean that
    ``_mean_of`` produces is not the tail CE of the validation set. Weight each batch by
    its own non-modal step count, and report the total rather than a mean of counts.
    """
    w = np.array([r.get("tail_n", 0.0) for r in rows], dtype=float)
    if w.sum() <= 0:
        return out
    v = np.array([r.get("tail_ce", 0.0) for r in rows], dtype=float)
    return {**out, "tail_ce": float((v * w).sum() / w.sum()), "tail_n": float(w.sum())}


def _model_tokens(batch: Dict) -> int:
    """Dense causal-transformer tokens in one padded BC batch.

    This deliberately follows build-nanogpt's throughput convention: count tensor
    positions the backbone computes, including padding, rather than only positions that
    reach the loss. Each episode contributes the ``interaction`` and ``goal`` prefix
    tokens followed by the batch-padded frame sequence. ``frames`` remains the useful,
    unpadded counterpart, so the two rates expose padding overhead instead of hiding it.
    """
    key = "image" if "image" in batch else "tokens"
    batch_size, padded_frames = batch[key].shape[:2]
    return int(batch_size) * (int(padded_frames) + PREFIX)


def _require_family_counts(index: List[dict], expected: Dict, split: str) -> None:
    """Refuse a silently incomplete or accidentally duplicated release."""
    actual = Counter(ep["family"] for ep in index)
    for family, count in expected.items():
        if actual[family] != int(count):
            raise ValueError(
                f"{split} family {family!r}: expected {int(count)} episodes, "
                f"resolved {actual[family]}")


def _timed_train_iteration(trainer, batches, loader, clock=time.perf_counter):
    """Acquire and train one batch, returning end-to-end elapsed time and iterator.

    The clock deliberately starts before ``next(batches)``. Karpathy's throughput
    includes input acquisition; excluding it hides exactly the loader stalls an
    efficiency experiment needs to find.
    """
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    t0 = clock()
    try:
        batch = next(batches)
    except StopIteration:
        batches = iter(loader)
        batch = next(batches)
    row, tokens = trainer.train_step(batch)
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
    return row, tokens, max(1e-9, clock() - t0), batches


class BCTrainer:
    def __init__(self, args: DictConfig, run_dir: str):
        self.args, self.run_dir = args, run_dir
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[args.precision]
        if self.device.type != "cuda":
            self.autocast_dtype = None

        # -- data ------------------------------------------------------------
        # Data owns the tokens now (doc/0016 §2, `kaihe/contra_nes_policy#7`), so this
        # branch has no shard tars, no `cache/` and no family schedule to assert: the
        # datahouse slice *is* the dataset. Everything below the branch is shared.
        dh_cfg = args.get("datahouse", {}) or {}
        self.datahouse = bool(dh_cfg.get("root"))
        if self.datahouse:
            if not bool(args.policy.freeze_encoder):
                raise ValueError(
                    "datahouse requires policy.freeze_encoder=true — a trainable encoder "
                    "cannot be served from precomputed tokens")
            enc_sha = encoder_fingerprint(str(args.policy.encoder_ckpt))
            store = DatahouseTokens(
                str(dh_cfg.root), level=int(dh_cfg.get("level", 1)),
                task=str(dh_cfg.get("task", "boss")),
                # str or list: a multi-weapon cell mixes start states (datahouse.py).
                weapon=OmegaConf.to_container(dh_cfg.weapon, resolve=True)
                if not isinstance(dh_cfg.get("weapon", "spread"), str)
                else str(dh_cfg.weapon),
                encoder_sha256=enc_sha, image_size=int(args.image_size))
            train_uids, val_uids = split_uids(store, int(dh_cfg.get("val_every", 20)))
            # A shard prefix cuts the training tier only: the holdout is carved by uid digest
            # over the whole store first, so every prefix validates on the same episodes.
            prefix = dh_cfg.get("shard_counts") or dh_cfg.get("shard_count")
            if prefix is not None and not isinstance(prefix, (int, float)):
                prefix = OmegaConf.to_container(prefix, resolve=True)
            train_uids = shard_prefix(store, train_uids, prefix)
            train_idx = datahouse_index(store, train_uids)
            val_idx = datahouse_index(store, val_uids)
            self.token_cache = True
            self.train_ds = CachedEpisodeDataset(store, train_uids)
            self.val_ds = CachedEpisodeDataset(store, val_uids)
            train_frames = sum(store.length(u) for u in train_uids)
            # What this run trains on, not what the store holds: a shard prefix reads a
            # fraction of the slice, and `tools/scaling_report.py` names the tier from this
            # file rather than guessing a size from the config.
            json.dump({"source": "datahouse", "slice": list(store.slice[:2]) + [
                        list(store.slice[2])],
                       "train_episodes": len(train_uids), "val_episodes": len(val_uids),
                       "train_frames": int(train_frames),
                       "store_episodes": len(store), "store_frames": int(store.declared[1]),
                       "shard_count": prefix,
                       "val_every": int(dh_cfg.get("val_every", 20))},
                      open("dataset.json", "w"), indent=2)
            print(f"[bc] datahouse {store.slice} · {len(train_uids)} train / "
                  f"{len(val_uids)} val episodes ({train_frames} train frames) · "
                  f"encoder {enc_sha[:12]} · provisional split 1/"
                  f"{int(dh_cfg.get('val_every', 20))}"
                  + (f" · train shard prefix {prefix}" if prefix else ""), flush=True)
        else:
            shard_dir = os.path.expanduser(args.shard_dir)
            fams = list(args.families)
            overrides = {k: os.path.expanduser(str(v))
                         for k, v in dict(args.get("shard_overrides", {})).items()}
            scaling = args.get("boss_scaling", {})
            release = None
            if bool(scaling.get("enabled", False)):
                if "boss" not in fams:
                    raise ValueError("boss_scaling requires 'boss' in families")
                release = scaling_release(
                    str(scaling.manifest), int(scaling.shard_count),
                    str(scaling.validation_sha256))
                other = [f for f in fams if f != "boss"]
                train_tars = (shard_paths(shard_dir, other, "train", overrides)
                              + release["train"])
                val_tars = (shard_paths(shard_dir, other, "val", overrides)
                            + release["val"])
                print(f"[bc] boss scaling D{int(scaling.shard_count)} · "
                      f"{release['train_episodes']} episodes / "
                      f"{release['train_frames']} decisions", flush=True)
            else:
                train_tars = shard_paths(shard_dir, fams, "train", overrides)
                val_tars = shard_paths(shard_dir, fams, "val", overrides)
            train_idx = load_or_build_index(train_tars, args.cache_dir)
            val_idx = load_or_build_index(val_tars, args.cache_dir)
            expected = args.get("expected_episodes", {})
            expected_train = dict(expected.get("train", {}))
            expected_val = dict(expected.get("val", {}))
            if release is not None:
                expected_train["boss"] = release["train_episodes"]
                expected_val["boss"] = release["val_episodes"]
            _require_family_counts(train_idx, expected_train, "train")
            _require_family_counts(val_idx, expected_val, "val")
            # Precomputed frozen-encoder tokens replace the decode+encode legs entirely.
            # `train_index` still comes from the tars, because the family schedule and the
            # release assertions above are contracts about the *shards*; the cache only
            # changes where the pixels' representation comes from.
            tc_cfg = args.get("token_cache", {}) or {}
            self.token_cache = bool(tc_cfg.get("train"))
            if self.token_cache and not bool(args.policy.freeze_encoder):
                raise ValueError(
                    "token_cache requires policy.freeze_encoder=true — a trainable encoder "
                    "cannot be served from a cache of its own past outputs")
            ds_kw = dict(whole_episode=True, image_size=int(args.image_size),
                         sigma_px=float(args.sigma_px), aux_size=int(args.policy.aux_size),
                         prev_action_keep_prob=0.0, seed=int(args.seed))
            if self.token_cache:
                enc_sha = encoder_fingerprint(str(args.policy.encoder_ckpt))
                caches = {split: TokenCache(str(tc_cfg[split]), encoder_sha256=enc_sha,
                                            image_size=int(args.image_size))
                          for split in ("train", "val")}
                self.train_ds = CachedEpisodeDataset(caches["train"],
                                                     [e["uid"] for e in train_idx])
                self.val_ds = CachedEpisodeDataset(caches["val"],
                                                   [e["uid"] for e in val_idx])
                print(f"[bc] token cache · {len(caches['train'])} train / "
                      f"{len(caches['val'])} val episodes · encoder {enc_sha[:12]}",
                      flush=True)
            else:
                self.train_ds = ContraCrossViewDataset(train_idx, **ds_kw)
                self.val_ds = ContraCrossViewDataset(val_idx, **ds_kw)
        self.train_index = train_idx
        # -1 because the last frame of an episode has no action taken *from* it.
        self.train_len = [max(1, e["length"] - 1) for e in train_idx]
        self.val_len = [max(1, e["length"] - 1) for e in val_idx]

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

        self.objective = BehaviorCloneLoss(
            diagnostics=False, modal_action=MODAL_ACTION).to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=float(args.train.learning_rate), weight_decay=float(args.train.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_scale)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.autocast_dtype is torch.float16))

        self.logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.step = 0
        self.best = math.inf
        self.draw_counts: Counter = Counter()
        self.token_counts: Counter = Counter()
        self._resume(str(args.resume_from) if args.get("resume_from") else None)

    # -- plumbing -----------------------------------------------------------

    def _lr_scale(self, step: int) -> float:
        """Multiplier on the base LR. ``wsd`` is this repo's default — see :func:`wsd_phase`."""
        warm, total = int(self.args.train.warmup_steps), int(self.args.train.steps)
        if warm > 0 and step < warm:
            return (step + 1) / warm
        decay = str(self.args.train.lr_decay)
        if decay == "cosine":
            p = (step - warm) / max(1, total - warm)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, p))))
        if decay == "wsd":
            onset = wsd_onset(total, float(self.args.train.get("decay_frac", 0.1)))
            if step < onset:
                return 1.0                      # the stable phase: LR is flat, so branchable
            p = (step - onset) / max(1, total - onset)
            floor = float(self.args.train.get("final_lr_frac", 0.0))
            return floor + (1.0 - floor) * (1.0 - min(1.0, max(0.0, p)))
        return 1.0

    def _loader(self, ds, lengths, shuffle: bool) -> DataLoader:
        # `or {}` because a boss-only config disables the schedule with `family_draws: null`
        # — OmegaConf merges mappings, so `{}` would silently inherit the parent's.
        family_draws = dict(self.args.loader.get("family_draws", {}) or {})
        if shuffle and family_draws:
            batch_sampler = FixedFamilyBatchSampler(
                self.train_index, lengths, int(self.args.loader.batch_size),
                family_draws, num_batches=int(self.args.train.steps),
                pool_batches=int(self.args.loader.pool_batches),
                seed=int(self.args.seed), start_batch=self.step)
        else:
            batch_sampler = LengthGroupedSampler(
                lengths, int(self.args.loader.batch_size),
                pool_batches=int(self.args.loader.pool_batches),
                seed=int(self.args.seed), shuffle=shuffle)
        return DataLoader(
            ds, collate_fn=pad_episodes,
            batch_sampler=batch_sampler,
            num_workers=int(self.args.loader.num_workers),
            pin_memory=True,
            prefetch_factor=int(self.args.loader.prefetch_factor)
            if int(self.args.loader.num_workers) > 0 else None)

    def _to_device(self, batch: Dict) -> Dict:
        out = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v)
               for k, v in batch.items() if k != "cross_view"}
        if "cross_view" in batch:
            out["cross_view"] = {k: v.to(self.device, non_blocking=True)
                                 for k, v in batch["cross_view"].items()}
        return out

    def _forward(self, batch: Dict):
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null())
        with ctx:
            if self.token_cache:
                latents = self.policy.forward_tokens(
                    batch["tokens"], batch["goal_token"], batch["interaction"])
            else:
                cv = batch["cross_view"]
                latents = self.policy(batch["image"], cv["cross_view_image"],
                                      cv["cross_view_obj_id"])
            loss, metrics = self.objective(latents, batch)
        return loss, metrics

    # -- the loop -----------------------------------------------------------

    def train_step(self, batch: Dict) -> tuple[Dict[str, float], int]:
        self.policy.train()
        if self.policy.cfg.freeze_encoder:
            self.policy.encoder.eval()      # keep frozen norms in inference mode
        accounting = list(zip(batch["family"].tolist(), batch["seq_len"].tolist()))
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
        # Count only a completed optimizer step. An exception midway through the
        # forward/backward path must not make a resumed checkpoint double-count it.
        for family_id, seq_len in accounting:
            family = FAMILIES[int(family_id)]
            self.draw_counts[family] += 1
            self.token_counts[family] += int(seq_len)

        row = {k: float(v) for k, v in metrics.items()}
        row.update({"grad_norm": float(gn), "lr": self.optimizer.param_groups[0]["lr"]})
        return row, _model_tokens(batch)

    @torch.no_grad()
    def validate(self, max_batches: int = 0) -> Dict[str, float]:
        self.policy.eval()
        rows: List[Dict[str, float]] = []
        for i, batch in enumerate(self._loader(self.val_ds, self.val_len, shuffle=False)):
            if max_batches and i >= max_batches:
                break
            _loss, metrics = self._forward(self._to_device(batch))
            rows.append({k: float(v) for k, v in metrics.items()})
        return _weighted_tail(_mean_of(rows), rows)

    def run(self) -> None:
        total = int(self.args.train.steps)
        loader = self._loader(self.train_ds, self.train_len, shuffle=True)
        batches = iter(loader)
        try:
            while self.step < total:
                # Match build-nanogpt's wall-clock boundary: batch acquisition is part
                # of the step. Starting after `next()` hid decoder/loader stalls and
                # made tokens_per_sec a model-only number rather than train throughput.
                row, tokens, elapsed, batches = _timed_train_iteration(
                    self, batches, loader)
                self.step += 1
                if self.step % int(self.args.train.log_every) == 0:
                    row["step"] = self.step
                    row["step_ms"] = elapsed * 1000.0
                    row["tokens_per_sec"] = tokens / elapsed
                    row.update(self._accounting_metrics())
                    self._emit(row, "train")
                if self.step % int(self.args.train.val_every) == 0:
                    self._run_val(int(self.args.train.val_batches))
                periodic = self.step in {
                    int(x) for x in self.args.train.get("save_steps", [])}
                trunk = (
                    bool(self.args.train.get("save_wsd_trunk", False))
                    and str(self.args.train.lr_decay) == "wsd"
                    and self.step == wsd_onset(
                        total, float(self.args.train.get("decay_frac", 0.1))))
                if periodic or trunk:
                    self.save()
        finally:
            # Whole val set however the run ended — this is the gate.
            self._run_val(0, tag="val_full")
            self.save(final=True)

    def _run_val(self, batches: int, tag: str = "val") -> None:
        v = self.validate(batches)
        v["step"] = self.step
        v.update(self._accounting_metrics())
        self._emit(v, tag)
        val_loss = v["loss"]
        if val_loss < self.best:
            self.best = val_loss
            if bool(self.args.train.get("save_best", True)):
                self.save(best=True)

    def _emit(self, row: Dict[str, float], phase: str) -> None:
        self.logger.log({**row, "phase": phase})
        head = [f"{k}={row[k]:.4g}" for k in
                ("loss", "tail_ce", "lr", "grad_norm", "step_ms", "tokens_per_sec")
                if k in row]
        line = f"[{phase} {self.step}/{int(self.args.train.steps)}] " + " ".join(head)
        print(line, flush=True)

    def save(self, final: bool = False, best: bool = False) -> str:
        tag = "final" if final else ("best" if best else f"{self.step:06d}")
        path = os.path.join(self.run_dir, "checkpoints", f"policy-{tag}.pt")
        self.policy.save(
            path, step=self.step, best_val_loss=self.best,
            train_config=OmegaConf.to_container(self.args, resolve=True),
            optimizer=self.optimizer.state_dict(), scheduler=self.scheduler.state_dict(),
            scaler=self.scaler.state_dict(), family_draws=dict(self.draw_counts),
            family_tokens=dict(self.token_counts), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            numpy_rng=np.random.get_state(), python_rng=random.getstate())
        print(f"[bc] saved {path}", flush=True)
        return path

    def _accounting_metrics(self) -> Dict[str, float]:
        return {**{f"draws/{f}": float(self.draw_counts[f]) for f in FAMILIES},
                **{f"valid_tokens/{f}": float(self.token_counts[f]) for f in FAMILIES}}

    def _resume(self, path: Optional[str]) -> None:
        """Restore an exact optimizer/schedule/sampling continuation checkpoint."""
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        required = {"policy", "optimizer", "scheduler", "scaler", "step",
                    "train_config"}
        missing = sorted(required - set(ckpt))
        if missing:
            raise ValueError(f"BC checkpoint cannot resume; missing {missing}")
        previous = ckpt["train_config"]
        current = OmegaConf.to_container(self.args, resolve=True)

        def get(cfg, keys):
            for key in keys:
                cfg = cfg.get(key) if isinstance(cfg, dict) else None
            return cfg

        guards = [("seed",), ("loader", "family_draws"), ("boss_scaling", "shard_count")]
        # Extending the budget is the whole point of WSD, so `train.steps` is only a guard
        # under a schedule that cannot be extended. Under WSD it is checked differently
        # below: the constraint is not "same total" but "resumed inside the stable phase".
        old_decay, new_decay = get(previous, ("train", "lr_decay")), get(current, ("train", "lr_decay"))
        extendable = old_decay == "wsd" and new_decay == "wsd"
        if not extendable:
            guards.append(("train", "steps"))
        for keys in guards:
            old, new = get(previous, keys), get(current, keys)
            if old != new:
                dotted = ".".join(keys)
                hint = ""
                if dotted == "train.steps":
                    hint = (f" — schedule is {old_decay!r}, whose LR is defined against the "
                            f"total, so it cannot be extended. Use lr_decay: wsd.")
                raise ValueError(f"resume changes {dotted}: {old!r} -> {new!r}{hint}")

        old_total = int(get(previous, ("train", "steps")) or 0)
        new_total = int(get(current, ("train", "steps")) or 0)
        # Only when the budget actually changes. Resuming the *same* run after a crash must
        # work from any step, cooldown included — that is ordinary resumption, not a branch.
        if extendable and new_total != old_total:
            old_frac = float(get(previous, ("train", "decay_frac")) or 0.1)
            onset = wsd_onset(old_total, old_frac)
            resumed_at = int(ckpt["step"])
            if resumed_at > onset:
                raise ValueError(
                    f"cannot extend from step {resumed_at}: its run began cooling down at "
                    f"{onset} (of {old_total}), so the LR there is already annealed and the "
                    f"weights are not a stable-phase trunk. Branch from a checkpoint at or "
                    f"before step {onset}, or train the longer budget from scratch.")
            if new_total < resumed_at:
                raise ValueError(f"resume target {new_total} is before the checkpoint's "
                                 f"step {resumed_at}")
        self.policy.load_state_dict(ckpt["policy"], strict=True)
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.scaler.load_state_dict(ckpt["scaler"])
        self.step = int(ckpt["step"])
        self.best = float(ckpt.get("best_val_loss", math.inf))
        self.draw_counts.update(ckpt.get("family_draws", {}))
        self.token_counts.update(ckpt.get("family_tokens", {}))
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"])
        if torch.cuda.is_available() and ckpt.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        if "numpy_rng" in ckpt:
            np.random.set_state(ckpt["numpy_rng"])
        if "python_rng" in ckpt:
            random.setstate(ckpt["python_rng"])
        print(f"[bc] resumed {path} at step {self.step}", flush=True)


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
