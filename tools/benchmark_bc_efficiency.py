"""Reproducible end-to-end BC efficiency benchmark for doc/0007.

Each invocation measures one configuration in a fresh process. Run variants from the
same checkpoint and seed so they see the same episode order and start from identical
weights. Warm-up is excluded from the summary (and therefore absorbs compilation),
while every measured step includes DataLoader acquisition, host-to-device transfer,
encoder, core, backward and optimizer work.

Example::

    python tools/benchmark_bc_efficiency.py \
      --checkpoint runs/bc/2026-08-04/15-19-58/checkpoints/policy-final.pt \
      --label eager-padded --warmup 10 --steps 100
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from contra_policy.train_bc import (BCTrainer, _model_tokens, _seed_everything,
                                    _useful_model_tokens)


def percentile(values, q: float) -> float:
    ordered = sorted(values)
    return float(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))])


def next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", help="optional JSON output path")
    ap.add_argument("--compile-core", action="store_true")
    ap.add_argument("--attention-layout", choices=("padded", "varlen"), default="padded")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("efficiency benchmark requires CUDA")
    torch.set_float32_matmul_precision("high")
    _seed_everything(args.seed)
    if args.compile_core:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = OmegaConf.load(os.path.join(root, "src/contra_policy/config_bc.yaml"))
    cfg.shard_dir = os.path.expanduser(str(cfg.shard_dir))
    cfg.cache_dir = os.path.join(root, "cache")
    cfg.policy.encoder_ckpt = os.path.join(
        root, "runs/encoder/2026-07-31/18-00-11/checkpoints/encoder-final.pt")
    cfg.loader.batch_size = args.batch_size
    cfg.loader.num_workers = args.num_workers
    cfg.seed = args.seed
    cfg.performance = {"compile_core": args.compile_core,
                       "compile_dynamic": True,
                       "attention_layout": args.attention_layout}

    os.makedirs(os.path.join(root, "tmp"), exist_ok=True)
    run_dir = tempfile.mkdtemp(prefix=f"bench-{args.label}-", dir=os.path.join(root, "tmp"))
    trainer = BCTrainer(cfg, run_dir)
    checkpoint = torch.load(os.path.expanduser(args.checkpoint), map_location="cpu",
                            weights_only=False)
    trainer.policy.load_state_dict(checkpoint["policy"], strict=True)

    loader = trainer._loader(trainer.train_ds, trainer.train_len, shuffle=True)
    batches = iter(loader)

    # Warm-up includes graph compilation and allocator growth but is not scored.
    compile_start = time.perf_counter()
    for _ in range(args.warmup):
        batch, batches = next_batch(batches, loader)
        trainer.train_step(batch)
    torch.cuda.synchronize(trainer.device)
    warmup_seconds = time.perf_counter() - compile_start
    torch.cuda.reset_peak_memory_stats(trainer.device)

    elapsed, losses, grad_norms = [], [], []
    useful_tokens = dense_tokens = 0
    start = time.perf_counter()
    for _ in range(args.steps):
        torch.cuda.synchronize(trainer.device)
        t0 = time.perf_counter()
        batch, batches = next_batch(batches, loader)
        useful_tokens += _useful_model_tokens(batch)
        dense_tokens += _model_tokens(batch)
        row, _ = trainer.train_step(batch)
        torch.cuda.synchronize(trainer.device)
        elapsed.append(time.perf_counter() - t0)
        losses.append(row["loss"])
        grad_norms.append(row["grad_norm"])
    wall = time.perf_counter() - start

    result = {
        "label": args.label,
        "checkpoint": os.path.abspath(args.checkpoint),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(trainer.device),
        "compile_core": args.compile_core,
        "attention_layout": args.attention_layout,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "warmup_steps": args.warmup,
        "warmup_seconds": warmup_seconds,
        "measured_steps": args.steps,
        "wall_seconds": wall,
        "step_ms_median": statistics.median(elapsed) * 1000.0,
        "step_ms_p90": percentile(elapsed, 0.9) * 1000.0,
        "useful_tokens_per_sec": useful_tokens / wall,
        "dense_tokens_per_sec": dense_tokens / wall,
        "padding_fraction": 1.0 - useful_tokens / dense_tokens,
        "loss_mean": float(np.mean(losses)),
        "loss_last": losses[-1],
        "grad_norm_mean": float(np.mean(grad_norms)),
        "peak_cuda_gb": torch.cuda.max_memory_allocated(trainer.device) / 1e9,
    }
    if args.compile_core:
        result["compile_unique_graphs"] = int(
            torch._dynamo.utils.counters["stats"]["unique_graphs"])
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if args.output:
        with open(args.output, "w") as fh:
            json.dump(result, fh, indent=2, sort_keys=True)
            fh.write("\n")


if __name__ == "__main__":
    main()
