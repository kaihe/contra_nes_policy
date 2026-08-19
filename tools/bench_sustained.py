"""Does step time drift under sustained load? — doc/0019.

    python tools/bench_sustained.py --config-name config_bc_bench \
        policy.core.d_model=1024 policy.core.n_layer=8 policy.core.n_head=16 \
        policy.core.n_kv_head=16 bench.steps=4000 bench.label=XXL-sustained

`bench_step.py` reports one median over 300 steps, which answers "how fast is a step"
but not "how fast is a step **after 40 minutes**". Those differ here, and the difference
is the largest single effect in doc/0019: this box is an **80 W power-capped mobile
4090**, and the measured ladder shows XXL at 101.7 ms/step over a 6.7 h D40k run against
71.9 ms over a 1 h D10k run — same model, same batch, same data layout per step.

If that gap is heat, then `ms/step` is a function of run *length* and every per-size
recipe built on a 400-step median is quoting a cold number. So this reports the median
per block of `bench.block` steps alongside SM clock, power draw and temperature sampled
from `nvidia-smi`, and the drift from first block to last is the answer.

M is the control: it should not throttle, because it never approaches the cap.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from contra_policy.train_bc import (BCTrainer, _seed_everything,
                                    _timed_train_iteration)


class _GPUSampler:
    """SM clock / power / temperature, sampled off the training thread."""

    # The violation counters are cumulative microseconds of *enforced* throttling, so a
    # per-block delta attributes a slow block to the power cap or to heat rather than
    # leaving it to be inferred from clocks.
    QUERY = ("utilization.gpu,clocks.sm,power.draw,temperature.gpu,"
             "clocks_throttle_reasons.sw_power_cap,"
             "clocks_throttle_reasons.hw_thermal_slowdown")

    def __init__(self, interval: float = 2.0):
        self.interval, self.rows, self._stop = interval, [], threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                out = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={self.QUERY}",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                f = out.split(",")
                util, sm, pw, temp = [float(x) for x in f[:4]]
                # "Active"/"Not Active" for the two reasons that matter on this box.
                capped = 1.0 if "not active" not in f[4].strip().lower() else 0.0
                hot = 1.0 if "not active" not in f[5].strip().lower() else 0.0
                self.rows.append((time.perf_counter(), util, sm, pw, temp, capped, hot))
            except Exception:
                pass                      # a dropped sample must not stop the run

    def since(self, t0):
        return [r for r in self.rows if r[0] >= t0]

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3.0)
        return False


@hydra.main(config_path="../src/contra_policy", config_name="config_bc_bench",
            version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(args.seed))
    bench = args.get("bench", {}) or {}
    steps = int(bench.get("steps", 4000))
    block = int(bench.get("block", 200))
    label = str(bench.get("label", "sustained"))
    out_path = bench.get("out")

    trainer = BCTrainer(args, run_dir=os.getcwd())
    loader = trainer._loader(trainer.train_ds, trainer.train_len, shuffle=True)
    batches = iter(loader)

    blocks, ms = [], []
    t_start = time.perf_counter()
    print(f"{'block':>6} {'step':>7} {'min':>6} {'ms/step':>8} {'SM MHz':>7} "
          f"{'W':>6} {'degC':>5} {'capped':>7}", flush=True)
    with _GPUSampler() as gpu:
        for i in range(steps):
            t_block = time.perf_counter() if i % block == 0 else t_block
            _row, _tok, elapsed, batches = _timed_train_iteration(trainer, batches, loader)
            trainer.step += 1
            ms.append(elapsed * 1000.0)
            if (i + 1) % block == 0:
                seg = np.array(ms[-block:])
                g = gpu.since(t_block)
                sm = float(np.median([r[2] for r in g])) if g else float("nan")
                pw = float(np.median([r[3] for r in g])) if g else float("nan")
                tp = float(np.max([r[4] for r in g])) if g else float("nan")
                cap = float(np.mean([r[5] for r in g]) * 100) if g else float("nan")
                mins = (time.perf_counter() - t_start) / 60
                rec = {"block": len(blocks) + 1, "step": i + 1, "minutes": mins,
                       "step_ms_median": float(np.median(seg)),
                       "sm_mhz": sm, "watts": pw, "temp_c": tp, "power_capped_pct": cap}
                blocks.append(rec)
                print(f"{rec['block']:>6} {i + 1:>7} {mins:>6.1f} "
                      f"{rec['step_ms_median']:>8.2f} {sm:>7.0f} {pw:>6.1f} {tp:>5.0f} "
                      f"{cap:>6.0f}%", flush=True)

    first, last = blocks[0]["step_ms_median"], blocks[-1]["step_ms_median"]
    result = {"label": label, "d_model": int(trainer.policy.core.cfg.d_model),
              "batch_size": int(args.loader.batch_size), "steps": steps, "block": block,
              "first_block_ms": first, "last_block_ms": last,
              "drift_pct": (last / first - 1) * 100,
              "best_block_ms": min(b["step_ms_median"] for b in blocks),
              "worst_block_ms": max(b["step_ms_median"] for b in blocks),
              "minutes": blocks[-1]["minutes"], "blocks": blocks}
    print(json.dumps({k: v for k, v in result.items() if k != "blocks"}, indent=2),
          flush=True)
    if out_path:
        path = os.path.expanduser(str(out_path))
        if not os.path.isabs(path):
            path = os.path.join(hydra.utils.get_original_cwd(), path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
