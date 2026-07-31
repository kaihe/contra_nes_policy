"""Dev tool: prove the RL collector and ``contra_nes_evaluation`` agree, task by task.

``contra_nes_evaluation`` must stay deletable, so this repo does not import it — which
means the RL collector is a **second implementation of the same stepping order**. That
is a real risk: a one-step-off budget, a success/death tie broken the other way, or a
cross-view prompt rebuilt from the shard that does not match the one the harness
renders from the emulator, would all show up as a reward that quietly disagrees with
the metric the run is judged by.

This script pins the two against each other on real tasks. It is a *tool*, not part of
the package and not part of the test suite: it is the only place ``contra_eval`` is
imported, and nothing under ``src/`` or ``tests/`` depends on it existing.

    python tools/parity_vs_evaluation.py --limit-per-family 10

Both sides run **greedy** (``temperature=0``) in fp32, which makes each episode a
deterministic function of its task and the weights — so anything other than exact
agreement on the outcome is a genuine divergence, not sampling noise.

The two differ deliberately in exactly one place, and that is half the point of the
check: the harness rebuilds the cross-view prompt by replaying the task through the
emulator, while the collector reads it out of the shard the policy trained on. If the
join or the mask rendering drifted, the outcomes would part company here.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

DEFAULT_CKPT = ("~/code/contra_nes_policy/runs/2026-07-28/18-01-29/weights/"
                "weight-epoch=18-step=30000.ckpt")


def pick_tasks(catalog, limit_per_family: int, seed: int) -> List:
    """A fixed, seeded subset of the training split, spread over families and labels."""
    rng = np.random.default_rng(seed)
    out = []
    for family in sorted(catalog.by_family):
        labels = sorted(catalog.by_family[family])
        per_label = max(1, limit_per_family // len(labels))
        for label in labels:
            group = sorted(catalog.by_family[family][label], key=lambda t: t.uid)
            take = min(per_label, len(group))
            idx = rng.choice(len(group), take, replace=False)
            out.extend(group[i] for i in sorted(idx))
    return sorted(out, key=lambda t: (t.family, t.label, t.uid))


def run_collector(model, catalog, tasks, args) -> Dict[str, dict]:
    from contra_policy.rl.rollout import EpisodeCollector
    from contra_policy.rl.tasks import TaskSampler

    sampler = TaskSampler(catalog, 1.0, 0.0, seed=args.seed)
    t0 = time.time()
    with EpisodeCollector(model, catalog, sampler, batch_size=args.batch_size,
                          budget_mult=args.budget_mult, min_budget=args.min_budget,
                          image_size=args.image_size, device=torch.device(args.device),
                          temperature=0.0, precision="fp32", seed=args.seed,
                          owner="parity-collector") as col:
        episodes = col.collect(0, 0, tasks=list(tasks))
    dt = time.time() - t0
    steps = sum(len(e) for e in episodes)
    print(f"[collector] {len(episodes)} episodes, {steps} steps in {dt:.1f}s "
          f"({steps/max(dt,1e-9):.1f} steps/s)")
    return {e.uid: {"outcome": e.outcome, "steps": len(e), "budget": e.budget,
                    "actions": e.actions.tolist()} for e in episodes}


def run_evaluation(tasks, args) -> Dict[str, dict]:
    from contra_eval.policies import build_policy
    from contra_eval.rollout import Runner
    from contra_eval.tasks import TaskSpec

    specs = [TaskSpec(path=t.path, kind=t.family, label=t.label, uid=t.uid,
                      split=t.split) for t in tasks]
    policy = build_policy("ckpt", batch_size=args.batch_size,
                          ckpt=os.path.expanduser(args.ckpt), device=args.device,
                          temperature=0.0, seed=args.seed, use_prev_action=False,
                          precision="fp32")
    runner = Runner(policy, image_size=args.image_size, sigma_px=args.sigma_px,
                    batch_size=args.batch_size, budget_mult=args.budget_mult,
                    min_budget=args.min_budget, keep_actions=True)
    t0 = time.time()
    results = runner.run(specs)
    dt = time.time() - t0
    steps = sum(r.steps for r in results)
    print(f"[contra_eval] {len(results)} episodes, {steps} steps in {dt:.1f}s "
          f"({steps/max(dt,1e-9):.1f} steps/s, includes prompt replay)")
    return {r.uid: {"outcome": r.outcome, "steps": r.steps, "budget": r.budget,
                    "actions": r.actions} for r in results}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--task-root", default="~/code/contra_nes_data/game_trace/tasks")
    p.add_argument("--shard-dir", default="~/code/contra_nes_data/game_trace/hf")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--families", default="kill,item,traverse,boss")
    p.add_argument("--limit-per-family", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--budget-mult", type=float, default=2.0)
    p.add_argument("--min-budget", type=int, default=24)
    p.add_argument("--sigma-px", type=float, default=12.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from contra_policy.model import CrossViewContraRocket
    from contra_policy.rl import checkpoint as ckpt_io
    from contra_policy.rl.tasks import TaskCatalog

    model_cfg = ckpt_io.model_config_from_checkpoint(args.ckpt)
    args.image_size = int(model_cfg["image_size"])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = CrossViewContraRocket(**model_cfg).to(device)
    ckpt_io.load_policy_weights(model, args.ckpt)
    model.eval()

    catalog = TaskCatalog(task_root=args.task_root, shard_dir=args.shard_dir,
                          families=tuple(args.families.split(",")), split="train",
                          image_size=args.image_size, sigma_px=args.sigma_px,
                          cache_dir=args.cache_dir)
    tasks = pick_tasks(catalog, args.limit_per_family, args.seed)
    print(f"[parity] {len(tasks)} tasks · greedy · fp32 · budget {args.budget_mult}x "
          f"(min {args.min_budget})")

    mine = run_collector(model, catalog, tasks, args)
    del model
    torch.cuda.empty_cache()
    theirs = run_evaluation(tasks, args)

    same_outcome = same_steps = same_actions = 0
    disagreements = []
    for t in tasks:
        a, b = mine.get(t.uid), theirs.get(t.uid)
        if a is None or b is None:
            disagreements.append((t.uid, "missing", str(a is None), str(b is None)))
            continue
        if a["budget"] != b["budget"]:
            disagreements.append((t.uid, "budget", a["budget"], b["budget"]))
        if a["outcome"] == b["outcome"]:
            same_outcome += 1
        else:
            disagreements.append((t.uid, "outcome", a["outcome"], b["outcome"]))
        same_steps += a["steps"] == b["steps"]
        same_actions += a["actions"] == b["actions"]

    n = len(tasks)
    print(f"\n[parity] outcome agreement {same_outcome}/{n} "
          f"({same_outcome/n:.1%})")
    print(f"[parity] step-count agreement {same_steps}/{n} ({same_steps/n:.1%})")
    print(f"[parity] identical action sequence {same_actions}/{n} "
          f"({same_actions/n:.1%})")
    if disagreements:
        print("\n[parity] disagreements:")
        for uid, what, a, b in disagreements[:20]:
            print(f"  {uid:<28} {what:<8} collector={a!s:<10} contra_eval={b!s}")
    return 0 if same_outcome == n else 1


if __name__ == "__main__":
    sys.exit(main())
