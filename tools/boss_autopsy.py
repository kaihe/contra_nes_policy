"""Why does the policy die at ~40 decisions when the expert survives to ~87?

    python -m tools.boss_autopsy --ckpt <policy.pt> --repeats 16

Every boss measurement so far is an aggregate: success %, mean damage, mean steps. They
all say the same thing — ~9% success, ~40 decisions, ~90% death — across BC at four data
scales, a dropout sweep, sparse GRPO, 10 h of graded GRPO, and binary specialty GRPO. The
one constant nothing has moved is **survival**: ~40 decisions against the 87 an expert
needs on Spread.

Aggregates cannot say *why*, and the three candidate causes need completely different
fixes:

**perception**   the killing bullet is not in the representation when it matters. 0001
                 measured `enemy_bullets` at 0.97 dice on boss frames, but averaged over
                 all frames — never at the moment of death.
**action space** the bullet is visible with only 1-2 decisions of warning, and at 20 Hz
                 (50 ms per decision) no policy could dodge it.
**control**      the bullet is visible for 10+ decisions and the policy does not move.

This separates them, quantitatively, over hundreds of rollouts rather than by eye. The
game-side quantities all come from the data repo's *interpreted* accessors — no `ADDR_*`
knowledge crosses into this repo, same rule `KillBossMaker.boss_hp` follows.

Output is a table plus a per-step CSV. Frames are not saved; if a picture is wanted after
the numbers point somewhere, `env.entity.annotate(frame, ram)` draws one.
"""

from __future__ import annotations

import argparse
import collections
import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch

from contra_policy.model import load_policy
from contra_policy.rl.rollout import EpisodeCollector
from contra_policy.rl.tasks import TaskCatalog, TaskSampler

#: Screen-space distance, in pixels, at which an enemy bullet counts as a live threat.
#: The player sprite is ~16 px wide, so this is roughly two body widths — close enough
#: that a dodge is required, far enough that one is still possible.
THREAT_PX = 32.0


class StepRecorder:
    """Per-decision game state, keyed by episode.

    Installed as :class:`EpisodeCollector`'s ``on_step`` hook, which fires after the
    emulator has advanced and therefore sees the RAM the action produced — the only place
    that state exists.
    """

    def __init__(self) -> None:
        self.rows: List[dict] = []
        self._n: Dict[tuple, int] = collections.defaultdict(int)

    def __call__(self, slot, ram: np.ndarray, action: int) -> None:
        from env.entity import (is_grounded, nearest_enemy_bullet_dist,
                                nearest_enemy_dist, player_pos)

        from contra_policy.action_space import vectors_to_indices

        key = (slot.task.uid, slot.group_id)
        t = self._n[key]
        self._n[key] += 1
        px, py = (float(v) for v in player_pos(ram))
        # The expert's action at the same decision index. `seg.actions` holds button
        # vectors, not indices, so it goes through the frozen action table rather than
        # being cast — a silent mis-decode here would invent a comparison.
        expert = -1
        if t < len(slot.seg.actions):
            expert = int(vectors_to_indices(
                np.asarray(slot.seg.actions[t], dtype=np.uint8)[None])[0])
        self.rows.append({
            "uid": slot.task.uid, "episode": slot.group_id, "t": t,
            "action": int(action), "expert_action": expert,
            "player_x": px, "player_y": py,
            "grounded": int(bool(is_grounded(ram))),
            "bullet_dist": float(nearest_enemy_bullet_dist(ram)),
            "enemy_dist": float(nearest_enemy_dist(ram)),
        })


def replay_expert(catalog: TaskCatalog) -> List[dict]:
    """The expert's own per-step state, by replaying their action sequence.

    The index-matched comparison — policy action at t against expert action at t — is
    weak, because by step t the two trajectories are in different states. This replays
    ``seg.actions`` from the same save-state and records the *expert's* bullet distances,
    so the comparison can be conditioned on the threat actually faced. Deterministic, so
    one pass per task is the whole distribution.
    """
    from util.replay import make_env, rewind_state

    from contra_policy.action_space import vectors_to_indices
    from contra_policy.rl.rollout import claim_emulator, release_emulator

    from env.entity import (is_grounded, nearest_enemy_bullet_dist,
                            nearest_enemy_dist, player_pos)

    claim_emulator("boss_autopsy.expert")
    env = make_env()
    rows: List[dict] = []
    try:
        for task in catalog.tasks:
            seg = catalog.segment(task)
            rewind_state(env, seg.initial_state)
            idx = vectors_to_indices(np.asarray(seg.actions, dtype=np.uint8))
            for t, vec in enumerate(seg.actions):
                for _ in range(seg.skip):
                    env.step(np.asarray(vec, dtype=np.uint8))
                ram = env.unwrapped.get_ram().copy()
                px, py = (float(v) for v in player_pos(ram))
                rows.append({
                    "uid": task.uid, "episode": -1, "t": t,
                    "action": int(idx[t]), "expert_action": int(idx[t]),
                    "player_x": px, "player_y": py,
                    "grounded": int(bool(is_grounded(ram))),
                    "bullet_dist": float(nearest_enemy_bullet_dist(ram)),
                    "enemy_dist": float(nearest_enemy_dist(ram)),
                })
    finally:
        env.close()
        release_emulator()
    return rows


def state_matched(policy_rows: List[dict], expert_rows: List[dict]) -> None:
    """Action mix conditioned on the threat actually faced, not on decision index."""
    from contra_policy.action_space import ACTION_NAMES

    jump = {i for i, n in enumerate(ACTION_NAMES) if "J" in n}
    up = {i for i, n in enumerate(ACTION_NAMES) if "U" in n}
    right = {i for i, n in enumerate(ACTION_NAMES) if n == "R"}
    bins = [(0, 16), (16, 32), (32, 64), (64, float("inf"))]

    print("\n4. state-matched: action mix by the distance of the nearest enemy bullet")
    print(f"   {'bullet px':>12s}  {'who':7s} {'n':>6s} {'jump':>7s} {'up':>7s} {'R':>7s}")
    for lo, hi in bins:
        for who, rs in (("policy", policy_rows), ("expert", expert_rows)):
            a = [r["action"] for r in rs if lo <= r["bullet_dist"] < hi]
            if not a:
                continue
            label = f"{lo}-{hi:.0f}" if np.isfinite(hi) else f"{lo}+"
            print(f"   {label:>12s}  {who:7s} {len(a):6d} "
                  f"{np.mean([x in jump for x in a]):7.1%}"
                  f"{np.mean([x in up for x in a]):7.1%}"
                  f"{np.mean([x in right for x in a]):7.1%}")


def summarize(rows: List[dict], outcomes: Dict[tuple, str]) -> None:
    """The three-hypothesis table."""
    by_ep: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in rows:
        by_ep[(r["uid"], r["episode"])].append(r)

    deaths = [sorted(v, key=lambda r: r["t"])
              for k, v in by_ep.items() if outcomes.get(k) == "death"]
    if not deaths:
        print("no deaths recorded — nothing to autopsy")
        return

    print(f"\n=== {len(deaths)} deaths, {len(by_ep)} episodes ===\n")

    # 1. Was the killing threat visible in advance?
    print("1. warning time — nearest enemy-bullet distance (px) before death")
    print(f"   {'t-':>4s} " + " ".join(f"{q:>7s}" for q in ("p10", "median", "p90")) +
          "   share within threat range")
    for back in (1, 2, 3, 5, 10):
        d = [ep[-back]["bullet_dist"] for ep in deaths
             if len(ep) >= back and np.isfinite(ep[-back]["bullet_dist"])]
        if not d:
            continue
        near = float(np.mean([x <= THREAT_PX for x in d]))
        print(f"   {back:>4d} " + " ".join(f"{np.percentile(d, q):7.1f}"
                                           for q in (10, 50, 90)) + f"   {near:>12.0%}")

    # 2. How long was the threat live before it landed?
    #
    # Anchored on the closest approach in the final window, NOT on the last recorded step.
    # A bullet that connects despawns, so the final frame shows the next-nearest bullet
    # instead — median 50 px against 13.9 px one decision earlier. Counting back from the
    # last step therefore reported "77% never had a bullet close", which is an artifact of
    # the killing bullet already being gone.
    lead, contact = [], []
    for ep in deaths:
        d = [r["bullet_dist"] for r in ep]
        tail = d[-8:] if len(d) >= 8 else d
        ci = len(d) - len(tail) + int(np.argmin(tail))
        contact.append(d[ci])
        n = 0
        for i in range(ci, -1, -1):
            if not np.isfinite(d[i]) or d[i] > THREAT_PX:
                break
            n += 1
        lead.append(n)
    print(f"\n2. warning before contact (<= {THREAT_PX:.0f} px, 50 ms per decision)")
    print(f"   contact distance: p10 {np.percentile(contact, 10):.1f} · "
          f"median {np.median(contact):.1f} · p90 {np.percentile(contact, 90):.1f} px")
    print(f"   warning window  : median {np.median(lead):.0f} · mean {np.mean(lead):.1f} · "
          f"p90 {np.percentile(lead, 90):.0f} decisions · "
          f"zero {np.mean([x == 0 for x in lead]):.0%} · "
          f">=3 {np.mean([x >= 3 for x in lead]):.0%}")

    # 3. Did the policy respond while the threat was live?
    from contra_policy.action_space import ACTION_NAMES
    jump = {i for i, n in enumerate(ACTION_NAMES) if "J" in n}
    under, free = [], []
    for ep in deaths:
        for r in ep:
            (under if r["bullet_dist"] <= THREAT_PX else free).append(r)

    up = {i for i, n in enumerate(ACTION_NAMES) if "U" in n}
    right = {i for i, n in enumerate(ACTION_NAMES) if n == "R"}

    def rates(rs: List[dict], key: str) -> str:
        acts = [r[key] for r in rs if r[key] >= 0]
        if not acts:
            return "n/a"
        return (f"jump {np.mean([a in jump for a in acts]):5.1%}"
                f"   up {np.mean([a in up for a in acts]):5.1%}"
                f"   R {np.mean([a in right for a in acts]):5.1%}   n={len(acts):5d}")

    # The expert's action at the same decision *index* — not the same state, since their
    # trajectory diverges. Read it as "what a working policy does around here", and note
    # that a bias showing in both the threat and clear rows is the policy's, not an
    # artifact of the mismatch.
    print("\n3. action mix, policy vs expert at the same decision index")
    for label, rs in (("threat", under), ("clear", free)):
        print(f"   {label:6s} policy : {rates(rs, 'action')}")
        print(f"   {label:6s} expert : {rates(rs, 'expert_action')}")

    print("\nreading:")
    print("  contact preceded by <2 decisions of warning -> action space / 20 Hz")
    print("  bullet never tracked before contact         -> perception")
    print("  warning present and the action mix diverges -> control")
    print("  note the expert runs at the same 20 Hz with the same action table, so an")
    print("  adequate warning window makes action space an existence-proof failure.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val"])
    ap.add_argument("--weapon", default="Spread")
    ap.add_argument("--rapid", default="true")
    ap.add_argument("--repeats", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="tmp/boss-autopsy")
    ap.add_argument("--shard-dir", default="~/code/contra_nes_data/game_trace/hf")
    ap.add_argument("--task-root", default="~/code/contra_nes_data/game_trace/tasks")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(os.path.expanduser(args.ckpt)).to(device).eval()

    task_filter = {"weapon": [args.weapon], "rapid": args.rapid.lower() == "true"}
    catalog = TaskCatalog(task_root=args.task_root, shard_dir=args.shard_dir,
                          families=["boss"], split=args.split, cache_dir="cache",
                          task_filter=task_filter)
    rec = StepRecorder()
    collector = EpisodeCollector(
        policy, catalog, TaskSampler(catalog), batch_size=args.batch_size,
        device=device, collect_goal_points=False, owner="boss_autopsy", on_step=rec,
        # Named, not bypassed: this rolls held-out tasks on purpose and takes no
        # gradient. `EpisodeCollector` defaults to "train" so no trainer can do this
        # by accident.
        require_split=args.split)

    groups = [[t] for t in catalog.tasks for _ in range(args.repeats)]
    print(f"[autopsy] {len(catalog.tasks)} {args.split} tasks x {args.repeats} "
          f"= {len(groups)} rollouts", flush=True)
    eps = collector.collect_groups(groups)
    collector.close()

    outcomes = {(e.task_uid, e.group_id): e.outcome for e in eps}
    won = sum(1 for e in eps if e.outcome == "success")
    print(f"[autopsy] success {won}/{len(eps)} = {won / max(1, len(eps)):.1%}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "steps.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rec.rows[0]) + ["outcome"])
        w.writeheader()
        for r in rec.rows:
            w.writerow({**r, "outcome": outcomes.get((r["uid"], r["episode"]), "")})
    print(f"[autopsy] wrote {len(rec.rows)} steps to {path}")

    summarize(rec.rows, outcomes)

    print("\n[autopsy] replaying the expert for the state-matched comparison...", flush=True)
    exp = replay_expert(catalog)
    print(f"[autopsy] {len(exp)} expert steps over {len(catalog.tasks)} tasks")
    state_matched(rec.rows, exp)


if __name__ == "__main__":
    main()
