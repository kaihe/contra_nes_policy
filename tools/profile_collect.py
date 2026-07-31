"""Dev tool: where does a rollout decision's wall time actually go?

Collection is 58% of a training update, and the obvious fix — more emulator worker
processes — costs 2.65 GB each on a 20 GB box. Before paying that, it is worth knowing
which term dominates a decision, because the three candidates want *different* fixes:

``gpu_forward``    one batched policy forward serving all slots. Fixed cost per
                   decision-batch, amortised over ``batch_size``. If this dominates,
                   more emulators buy nothing — the GPU is already the floor.
``emu_step``       ``env.step()`` x ``skip`` NES frames. Genuinely serial on one
                   emulator, so this is the term that N parallel emulators divide.
``emu_rewind``     ``rewind_state()`` — swapping a slot's savestate in before its
                   step. **Pure overhead of sharing one emulator across N slots.**
                   Dedicated emulators would not pay it at all, so it is a lower bound
                   on what an emulator-per-worker split saves, independent of
                   parallelism.
``observe``        frame grab + resize + host-to-device.
``episode_setup``  ``_start`` / ``_finish`` — savestate load, prompt build, episode
                   assembly. Per *episode*, not per decision.

    python tools/profile_collect.py --steps 600
    python tools/profile_collect.py --steps 600 --batch-size 32

The GPU forward is timed with ``torch.cuda.synchronize()`` around it. That removes any
CPU/GPU overlap, so the reported ``gpu_forward`` is the honest serial cost rather than
whatever the async queue hid; the loop already forces a sync immediately afterwards
(``int(step["action"][i])`` on a device tensor), so little overlap exists to lose.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

DEFAULT_CKPT = ("~/code/contra_nes_policy/runs/2026-07-28/18-01-29/weights/"
                "weight-epoch=18-step=30000.ckpt")


class Timers:
    """Wall-clock totals and call counts, keyed by label."""

    def __init__(self) -> None:
        self.total: Dict[str, float] = defaultdict(float)
        self.calls: Dict[str, int] = defaultdict(int)

    def add(self, key: str, dt: float, n: int = 1) -> None:
        self.total[key] += dt
        self.calls[key] += n


def instrument(collector, timers: Timers, sync: bool) -> None:
    """Wrap the collector's seams in timers, in place."""
    import util.replay as replay

    actor = collector.actor

    # -- GPU forward -------------------------------------------------------
    real_act = actor.act

    def timed_act(obs):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = real_act(obs)
        if sync:
            torch.cuda.synchronize()
        timers.add("gpu_forward", time.perf_counter() - t0)
        return out

    actor.act = timed_act

    # -- observation build -------------------------------------------------
    real_observe = collector._observe

    def timed_observe(slots):
        t0 = time.perf_counter()
        out = real_observe(slots)
        timers.add("observe", time.perf_counter() - t0)
        return out

    collector._observe = timed_observe

    # -- emulator: rewind vs step -----------------------------------------
    # `_step` calls rewind_state then env.step x skip. Patch the module symbol the
    # method resolves (it imports inside the function body, so the module attribute
    # is what matters) and the env's own step.
    real_rewind = replay.rewind_state

    def timed_rewind(env, state):
        t0 = time.perf_counter()
        out = real_rewind(env, state)
        timers.add("emu_rewind", time.perf_counter() - t0)
        return out

    replay.rewind_state = timed_rewind

    env = collector._env
    real_env_step = env.step

    def timed_env_step(action):
        t0 = time.perf_counter()
        out = real_env_step(action)
        timers.add("emu_step", time.perf_counter() - t0)
        return out

    env.step = timed_env_step

    # -- whole decision, and per-episode setup ----------------------------
    real_step = collector._step

    def timed_step(slot, action):
        t0 = time.perf_counter()
        out = real_step(slot, action)
        timers.add("decision_total", time.perf_counter() - t0)
        return out

    collector._step = timed_step

    for name in ("_start", "_finish"):
        real = getattr(collector, name)

        def make(real_fn):
            def timed(*a, **kw):
                t0 = time.perf_counter()
                out = real_fn(*a, **kw)
                timers.add("episode_setup", time.perf_counter() - t0)
                return out
            return timed

        setattr(collector, name, make(real))


def report(timers: Timers, decisions: int, episodes: int, wall: float) -> None:
    print(f"\n{decisions} decisions · {episodes} episodes · {wall:.1f} s wall "
          f"({decisions / wall:.0f} decisions/s)\n")

    # emu_step fires once per NES frame (skip per decision); the rest once per
    # decision or per batch. Report both the share of wall and the per-decision cost.
    order = ["emu_rewind", "emu_step", "observe", "gpu_forward", "episode_setup"]
    print(f"{'component':16} {'total s':>9} {'% wall':>8} {'calls':>9} "
          f"{'us/call':>10} {'ms/decision':>12}")
    print("-" * 70)
    accounted = 0.0
    for k in order:
        t = timers.total.get(k, 0.0)
        n = timers.calls.get(k, 0)
        if n == 0:
            continue
        accounted += t
        print(f"{k:16} {t:9.2f} {100*t/wall:7.1f}% {n:9} "
              f"{1e6*t/n:10.1f} {1e3*t/decisions:12.3f}")
    print("-" * 70)
    print(f"{'accounted':16} {accounted:9.2f} {100*accounted/wall:7.1f}%")
    print(f"{'unaccounted':16} {wall-accounted:9.2f} {100*(wall-accounted)/wall:7.1f}%"
          "   (python loop, bookkeeping, GC)")

    emu = timers.total.get("emu_step", 0.0) + timers.total.get("emu_rewind", 0.0)
    gpu = timers.total.get("gpu_forward", 0.0)
    print(f"\nemulator (step+rewind) {emu:.2f} s   vs   gpu forward {gpu:.2f} s"
          f"   ->  ratio {emu/max(gpu,1e-9):.2f}x")

    rewind = timers.total.get("emu_rewind", 0.0)
    print(f"\nIf each slot owned its own emulator, `emu_rewind` disappears entirely:")
    print(f"  {rewind:.2f} s of {wall:.1f} s = {100*rewind/wall:.1f}% of collection, "
          f"before any parallelism.")
    if gpu > 0:
        floor = gpu + timers.total.get("observe", 0.0)
        print(f"  Perfectly parallel emulators would leave gpu+observe = {floor:.2f} s "
              f"as the floor -> {decisions/max(floor,1e-9):.0f} decisions/s ceiling.")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--task-root", default="~/code/contra_nes_data/game_trace/tasks")
    p.add_argument("--shard-dir", default="~/code/contra_nes_data/game_trace/hf")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--families", default="kill,item,traverse,boss")
    p.add_argument("--steps", type=int, default=600, help="decisions to collect")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--precision", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-sync", action="store_true",
                   help="skip cuda.synchronize around the forward (understates GPU)")
    args = p.parse_args(argv)

    for k in ("ckpt", "task_root", "shard_dir"):
        setattr(args, k, os.path.expanduser(getattr(args, k)))

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from contra_policy.model import CrossViewContraRocket
    from contra_policy.rl import checkpoint as ckpt_io
    from contra_policy.rl.rollout import EpisodeCollector
    from contra_policy.rl.tasks import TaskCatalog, TaskSampler

    model_cfg = ckpt_io.model_config_from_checkpoint(args.ckpt)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CrossViewContraRocket(**model_cfg).to(device)
    ckpt_io.load_policy_weights(model, args.ckpt)
    model.eval()

    catalog = TaskCatalog(task_root=args.task_root, shard_dir=args.shard_dir,
                          families=tuple(args.families.split(",")), split="train",
                          image_size=int(model_cfg["image_size"]), sigma_px=12.0,
                          cache_dir=args.cache_dir)
    catalog.assert_split("train")
    sampler = TaskSampler(catalog, 0.7, 0.3, None, seed=args.seed)

    collector = EpisodeCollector(
        model, catalog, sampler, batch_size=args.batch_size, budget_mult=2.0,
        min_budget=24, image_size=int(model_cfg["image_size"]), device=device,
        temperature=1.0, precision=args.precision, seed=args.seed,
        reward={"success": 1.0, "death": 0.0, "timeout": 0.0, "step": 0.0,
                "truncated": 0.0},
        max_episode_steps=0, collect_goal_points=True, owner="profile_collect")

    timers = Timers()
    collector.open()                       # open the emulator before wrapping env.step
    instrument(collector, timers, sync=not args.no_sync)

    print(f"[profile] batch_size={args.batch_size} precision={args.precision} "
          f"warming up...", flush=True)
    collector.collect(64, 1)               # page in caches, cuDNN autotune, JIT
    timers.total.clear()
    timers.calls.clear()

    print(f"[profile] collecting {args.steps} decisions...", flush=True)
    t0 = time.perf_counter()
    episodes = collector.collect(args.steps, args.episodes)
    wall = time.perf_counter() - t0

    decisions = sum(len(e) for e in episodes)
    report(timers, decisions, len(episodes), wall)
    collector.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
