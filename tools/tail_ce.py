"""Tail CE for checkpoints that already exist — step 1 of ``doc/0010``.

    python -m tools.tail_ce runs/bc/2026-08-05/15-49-28/checkpoints/policy-{003000,010000,final}.pt

Total validation CE bottoms at step 3,000 and then triples, yet the step-3,000
checkpoint plays 12.9 pp *worse* on the 846 suite than the fully overfit final. The
proposed explanation is that total CE is frequency-weighted over an action distribution
that is 78% ``R``, while survival depends on the rare frames. Tail CE — the same
cross-entropy restricted to steps whose target is not ``R`` — should therefore keep
falling where total CE rises.

This runs before spending two hours on the dropout sweep, because if tail CE does *not*
invert, the mechanism in 0010 §1 is wrong and the doc's framing needs revision first.
No training: the three checkpoints already have closed-loop numbers to correlate against.

The shard selection, image size and sigma come out of each checkpoint's own stored
``train_config``, never from a caller's config, so a checkpoint is always scored on the
validation set its run actually used.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from contra_policy.dataset import (ContraCrossViewDataset, LengthGroupedSampler,
                                   load_or_build_index, pad_episodes, scaling_release,
                                   shard_paths)
from contra_policy.loss import BehaviorCloneLoss
from contra_policy.model import load_policy
from contra_policy.train_bc import MODAL_ACTION, _weighted_tail


def val_shards(cfg: Dict) -> List[str]:
    """The validation tars of the run that produced this checkpoint."""
    shard_dir = os.path.expanduser(cfg["shard_dir"])
    fams = list(cfg["families"])
    overrides = {k: os.path.expanduser(str(v))
                 for k, v in dict(cfg.get("shard_overrides") or {}).items()}
    scaling = cfg.get("boss_scaling") or {}
    if not bool(scaling.get("enabled", False)):
        return shard_paths(shard_dir, fams, "val", overrides)
    release = scaling_release(str(scaling["manifest"]), int(scaling["shard_count"]),
                              str(scaling["validation_sha256"]))
    other = [f for f in fams if f != "boss"]
    return shard_paths(shard_dir, other, "val", overrides) + release["val"]


@torch.no_grad()
def score(path: str, device: torch.device, batch_size: int, max_batches: int) -> Dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["train_config"]
    idx = load_or_build_index(val_shards(cfg), cfg["cache_dir"])
    ds = ContraCrossViewDataset(
        idx, whole_episode=True, image_size=int(cfg["image_size"]),
        sigma_px=float(cfg["sigma_px"]), aux_size=int(cfg["policy"]["aux_size"]),
        prev_action_keep_prob=0.0, seed=int(cfg["seed"]))
    lengths = [max(1, e["length"] - 1) for e in idx]

    policy = load_policy(path).to(device).eval()
    objective = BehaviorCloneLoss(diagnostics=False, modal_action=MODAL_ACTION).to(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[cfg["precision"]]
    if device.type != "cuda":
        dtype = None

    loader = DataLoader(
        ds, collate_fn=pad_episodes, num_workers=0,
        batch_sampler=LengthGroupedSampler(
            lengths, batch_size, pool_batches=int(cfg["loader"]["pool_batches"]),
            seed=int(cfg["seed"]), shuffle=False))

    rows: List[Dict[str, float]] = []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in batch.items() if k != "cross_view"} | {
            "cross_view": {k: v.to(device) for k, v in batch["cross_view"].items()}}
        ctx = (torch.autocast("cuda", dtype=dtype) if dtype is not None
               else torch.autocast("cpu", enabled=False))
        with ctx:
            latents = policy(batch["image"], batch["cross_view"]["cross_view_image"],
                             batch["cross_view"]["cross_view_obj_id"])
            _loss, metrics = objective(latents, batch)
        rows.append({k: float(v) for k, v in metrics.items()})

    keys = {k for r in rows for k in r}
    out = _weighted_tail(
        {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}, rows)
    out["step"] = int(ckpt.get("step", -1))
    out["checkpoint"] = os.path.basename(path)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=0, help="0 = whole validation set")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"{'checkpoint':22s} {'step':>7s} {'val CE':>9s} {'tail CE':>9s} {'tail n':>10s}")
    for path in args.checkpoints:
        r = score(os.path.expanduser(path), device, args.batch_size, args.max_batches)
        print(f"{r['checkpoint']:22s} {r['step']:7d} {r['loss']:9.4f} "
              f"{r.get('tail_ce', float('nan')):9.4f} {r.get('tail_n', 0):10.0f}",
              flush=True)


if __name__ == "__main__":
    main()
