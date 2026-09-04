"""What validation and checkpointing actually cost — doc/0019 §2's two "free" savings.

    python tools/bench_val.py --config-name config_bc_bench \
        policy.core.d_model=1024 policy.core.n_layer=8 policy.core.n_head=16 \
        policy.core.n_kv_head=16 bench.label=XXL

doc/0019 §2 claims both savings are already measured and need no arm. They were measured
*indirectly*, by subtracting `median step_ms x steps` from a run's wall clock, which
attributes every non-step second to validation and saving — including page-cache misses,
throttling and queue overhead. This measures them directly instead:

    val     one whole-holdout pass (`val_batches: 0`, 1,976 episodes), timed. The run
            config scores this every 500 steps, so the per-run cost is this x steps/500.
    save    one `Trainer.save()`, timed, with the file size it produced. Intermediate
            checkpoints carry optimizer state; `--weights-only` prices dropping it.

The two are reported against the same run's step time, because the decision §4 has to
make is not "is validation slow" but "what fraction of the ladder is it".
"""

from __future__ import annotations

import json
import os
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from contra_policy.train_bc import (BCTrainer, _seed_everything,
                                    _timed_train_iteration)


@hydra.main(config_path="../src/contra_policy", config_name="config_bc_bench",
            version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _seed_everything(int(args.seed))
    bench = args.get("bench", {}) or {}
    label = str(bench.get("label", "val"))
    out_path = bench.get("out")
    trainer = BCTrainer(args, run_dir=os.getcwd())

    # A few real steps first: an untouched model would time validation against a cold
    # allocator and a cold page cache, which is not the state a run validates from.
    loader = trainer._loader(trainer.train_ds, trainer.train_len, shuffle=True)
    batches = iter(loader)
    step_ms = []
    for _ in range(120):
        _r, _t, elapsed, batches = _timed_train_iteration(trainer, batches, loader)
        trainer.step += 1
        step_ms.append(elapsed * 1000.0)
    step_ms_med = float(np.median(step_ms[20:]))

    val_s = []
    for _ in range(int(bench.get("val_repeats", 3))):
        torch.cuda.synchronize(trainer.device)
        t0 = time.perf_counter()
        trainer.validate(0)                      # whole holdout, as `val_batches: 0` does
        torch.cuda.synchronize(trainer.device)
        val_s.append(time.perf_counter() - t0)

    t0 = time.perf_counter()
    path = trainer.save()
    save_s = time.perf_counter() - t0
    size_gib = os.path.getsize(path) / 2**30

    # Weights-only, for the intermediate-checkpoint proposal in §2.
    wpath = path.replace(".pt", "-weights.pt")
    t0 = time.perf_counter()
    torch.save({"policy": trainer.policy.state_dict(), "step": trainer.step}, wpath)
    wsave_s = time.perf_counter() - t0
    wsize_gib = os.path.getsize(wpath) / 2**30

    steps = int(args.train.steps)
    val_every = int(args.train.val_every)
    n_val = steps // val_every
    n_save = len(list(args.train.get("save_steps", []))) + 2      # + best + final
    val_med = float(np.median(val_s))
    result = {
        "label": label, "d_model": int(trainer.policy.core.cfg.d_model),
        "batch_size": int(args.loader.batch_size),
        "step_ms": step_ms_med,
        "val_seconds": val_med, "val_repeats": [round(v, 2) for v in val_s],
        "val_episodes": len(trainer.val_ds),
        "save_seconds": save_s, "save_gib": size_gib,
        "weights_only_seconds": wsave_s, "weights_only_gib": wsize_gib,
        # What they cost the run the config actually describes.
        "run_steps": steps, "val_every": val_every,
        "train_minutes": steps * step_ms_med / 1000 / 60,
        "val_minutes": n_val * val_med / 60,
        "save_minutes": n_save * save_s / 60,
    }
    result["val_pct_of_run"] = 100 * result["val_minutes"] / (
        result["train_minutes"] + result["val_minutes"] + result["save_minutes"])
    print(json.dumps(result, indent=2), flush=True)
    os.remove(wpath)
    if out_path:
        p = os.path.expanduser(str(out_path))
        if not os.path.isabs(p):
            p = os.path.join(hydra.utils.get_original_cwd(), p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as fh:
            fh.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
