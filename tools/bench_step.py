"""Throughput of one training step, for doc/0019 — measure a config, do not train a model.

    python tools/bench_step.py --config-name config_bc_scaling_lr \
        policy.core.d_model=1024 policy.core.n_layer=8 policy.core.n_head=16 \
        policy.core.n_kv_head=16 bench.label=XXL-A

Runs `bench.steps` real training steps through the real `BCTrainer` — same sampler,
collate, autocast, backward and optimizer — and reports the **median** `step_ms` over
`bench.warmup`..`bench.steps`. Median rather than mean because a page fault on first
touch of a shard is a 200 ms outlier that no amount of averaging removes.

Three things it deliberately does not do, all of which would corrupt the number:

    no validation   `BCTrainer.run`'s `finally` scores the whole holdout, which is 5-12
                    min at these sizes and is the very cost this doc is trying to price
    no checkpoints  a 1.29 GB XXL save inside the timed window is a disk benchmark
    no resume       `bench.steps` is short enough that a warm cache is the point

**Memory is measured, not assumed.** doc/0019 §2 forbids raising batch speculatively:
this box is a 19 GB WSL VM and the collate builds padded batches in host memory, so an
OOM there takes the machine down and the queue with it. Peak VRAM comes from
`torch.cuda.max_memory_allocated`; peak host RSS is sampled across the *process tree*
(a `num_workers>0` arm pays in children, which `ru_maxrss` of self would miss) by a
background thread, since the peak lands between any two step boundaries.

`padded/real` is the padding overhead the length-grouped sampler leaves behind — the
share of computed positions that are mask. It is reported per arm because batch size
and `loader.pool_batches` both move it, and it is the denominator that makes a
tokens/sec comparison across batch sizes honest.
"""

from __future__ import annotations

import json
import os
import threading
import time

import hydra
import numpy as np
import psutil
import torch
from omegaconf import DictConfig, OmegaConf

from contra_policy.model import PREFIX
from contra_policy.train_bc import (BCTrainer, _seed_everything,
                                    _timed_train_iteration)


class _RSSSampler:
    """Peak host memory of this process tree, sampled off the training thread.

    Reports **both** RSS and PSS, because they answer different questions and only one of
    them is the right ceiling test. Summing RSS over parent and workers double-counts every
    shared page — and DataLoader workers are forked, so they share the interpreter, the
    torch libraries and the mmapped token shards. PSS divides each shared page by the number
    of processes mapping it, so the tree's PSS is what actually has to fit in the 19 GB VM.
    RSS is kept because it is the number the earlier arms were gated on.
    """

    def __init__(self, interval: float = 0.05, pss_every: float = 1.0):
        self.proc = psutil.Process()
        self.interval, self.peak, self._stop = interval, 0, threading.Event()
        self.peak_pss, self._pss_due = 0, 0.0
        self.pss_every = pss_every
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _tree(self):
        yield self.proc
        for child in self.proc.children(recursive=True):
            yield child

    def _sample(self) -> int:
        total = 0
        for p in self._tree():
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass                      # a worker exiting mid-sample is not an error
        return total

    def _sample_pss(self) -> int:
        """Sum of Pss from smaps_rollup — shared pages counted once, split across mappers."""
        total = 0
        for p in self._tree():
            try:
                with open(f"/proc/{p.pid}/smaps_rollup") as fh:
                    for line in fh:
                        if line.startswith("Pss:"):
                            total += int(line.split()[1]) * 1024
                            break
            except (OSError, ValueError):
                pass
        return total

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.peak = max(self.peak, self._sample())
                now = time.perf_counter()
                if now >= self._pss_due:
                    self._pss_due = now + self.pss_every
                    self.peak_pss = max(self.peak_pss, self._sample_pss())
            except psutil.Error:
                pass

    def __enter__(self):
        self.peak = self._sample()
        self.peak_pss = self._sample_pss()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=1.0)
        return False


def _real_tokens(batch) -> int:
    """Unpadded positions the backbone computes, prefix included — `_model_tokens`'s floor."""
    return int(batch["seq_len"].sum()) + PREFIX * int(batch["seq_len"].shape[0])


@hydra.main(config_path="../src/contra_policy", config_name="config_bc_scaling_lr",
            version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(args.seed))
    bench = args.get("bench", {}) or {}
    steps = int(bench.get("steps", 400))
    warmup = int(bench.get("warmup", 100))
    label = str(bench.get("label", "bench"))
    out_path = bench.get("out")

    run_dir = os.getcwd()                 # hydra.run.dir — a scratch dir, not runs/
    trainer = BCTrainer(args, run_dir=run_dir)

    # `_timed_train_iteration` hands back only the padded token count, and re-drawing a
    # batch to measure the unpadded one would advance the sampler. Record it as a side
    # effect of the step the trainer actually takes.
    real_tokens: list = []
    _step = trainer.train_step

    def counting_step(batches):
        real_tokens.append(sum(_real_tokens(batch) for batch in batches))
        return _step(batches)

    trainer.train_step = counting_step

    # Weights + grads + optimizer state, before a single activation exists. §2 predicts a
    # larger batch by scaling *only* the activation term, and that needs this measured.
    fixed_vram = (torch.cuda.memory_allocated(trainer.device) / 2**30
                  if trainer.device.type == "cuda" else 0.0)

    ms, padded = [], []
    loader = trainer._loader(trainer.train_ds, trainer.train_len, shuffle=True)
    batches = iter(loader)
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)

    t_wall = time.perf_counter()
    with _RSSSampler() as rss:
        for i in range(steps):
            row, tokens, elapsed, batches = _timed_train_iteration(trainer, batches, loader)
            trainer.step += 1
            ms.append(elapsed * 1000.0)
            padded.append(tokens)
            if (i + 1) % 100 == 0:
                print(f"[bench {label}] {i + 1}/{steps} "
                      f"median {np.median(ms[warmup:] or ms):.2f} ms", flush=True)
    wall = time.perf_counter() - t_wall

    keep = slice(warmup, steps)
    m = np.array(ms[keep])
    pad_tok = float(np.mean(padded[keep]))
    real_tok = float(np.mean(real_tokens[keep]))
    vram = (torch.cuda.max_memory_allocated(trainer.device) / 2**30
            if trainer.device.type == "cuda" else 0.0)
    result = {
        "label": label,
        "d_model": int(trainer.policy.core.cfg.d_model),
        "n_layer": int(trainer.policy.core.cfg.n_layer),
        "params_m": sum(p.numel() for p in trainer.policy.parameters()
                        if p.requires_grad) / 1e6,
        "batch_size": int(args.loader.batch_size),
        "num_workers": int(args.loader.num_workers),
        "pool_batches": int(args.loader.pool_batches),
        "steps": steps, "warmup": warmup,
        "step_ms_median": float(np.median(m)),
        "step_ms_p10": float(np.percentile(m, 10)),
        "step_ms_p90": float(np.percentile(m, 90)),
        "tokens_per_sec": pad_tok / (float(np.median(m)) / 1000.0),
        "real_tokens_per_sec": real_tok / (float(np.median(m)) / 1000.0),
        "episodes_per_sec": int(args.loader.batch_size) / (float(np.median(m)) / 1000.0),
        "padded_tokens_per_step": pad_tok,
        "real_tokens_per_step": real_tok,
        "padding_waste": 1.0 - real_tok / pad_tok,
        "peak_vram_gib": vram,
        "fixed_vram_gib": fixed_vram,
        "activation_vram_gib": max(0.0, vram - fixed_vram),
        "peak_rss_gib": rss.peak / 2**30,
        "peak_pss_gib": rss.peak_pss / 2**30,
        "wall_sec": wall,
    }
    print(json.dumps(result, indent=2), flush=True)
    if out_path:
        # hydra has chdir'd into the run dir; resolve against where the user invoked us so
        # a sweep appends to one file rather than one file per arm.
        path = os.path.expanduser(str(out_path))
        if not os.path.isabs(path):
            path = os.path.join(hydra.utils.get_original_cwd(), path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
