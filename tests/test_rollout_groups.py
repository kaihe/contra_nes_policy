"""Group identity across repeated collection calls, and the memory it costs.

These pin the three faults that took down run ``2026-08-02/11-48-03`` after 8 updates:

1. ``collect_groups`` restarted group ids at 0 on every call, so ``collect_filtered``
   pooled unrelated tasks into one group — GRPO's baseline stopped being same-task —
   and its ``n_kept >= want`` exit became unreachable, so every update ran to the
   oversample cap.
2. The collection-side ``zero_variance_group_frac`` shared a key with the post-filter
   one and was overwritten, hiding a real 0.59 behind a logged 0.0.
3. Whole ``Episode`` objects were retained purely to report success rates, at ~18 MB
   each and 512 per update.

The invariant that ties 1 together and would have caught it first: **one group id means
one task**.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from contra_policy.rl.buffer import Episode, EpisodeOutcome
from contra_policy.rl.trainer import GRPOTrainer


def _task(uid, fam="kill"):
    return SimpleNamespace(uid=uid, family=fam)


class _FakeCollector:
    """Mimics ``EpisodeCollector.collect_groups``' id contract without an emulator.

    Group ``i`` of this call gets id ``base_gid + i`` — the same
    ``enumerate(groups, start=base_gid)`` the real one uses.
    """

    def __init__(self, rewards):
        self.rewards = list(rewards)   # one reward pattern per group, cycled
        self.calls = []

    def collect_groups(self, groups, base_gid=0):
        self.calls.append(base_gid)
        out = []
        for i, g in enumerate(groups):
            pattern = self.rewards[(base_gid + i) % len(self.rewards)]
            for k, task in enumerate(g):
                out.append(Episode(
                    task_uid=task.uid, family=task.family, group_id=base_gid + i,
                    frames=np.zeros((3, 4, 4, 3), np.uint8),
                    goal_image=np.zeros((4, 4, 3), np.uint8), interaction=0,
                    actions=np.zeros(3, np.int64), logprobs=np.zeros(3, np.float32),
                    reward=float(pattern[k % len(pattern)]),
                    outcome="success" if pattern[k % len(pattern)] else "death"))
        return out


class _FakeSampler:
    def __init__(self, group_size=4):
        self.group_size, self.n = group_size, 0

    def sample_groups(self, n_groups):
        groups = []
        for _ in range(n_groups):
            self.n += 1
            groups.append([_task(f"task-{self.n}")] * self.group_size)
        return groups


def _trainer(rewards, *, want=16, at_once=8, factor=4.0, memory_limit_gb=0.0):
    """A stub carrying only what ``collect_filtered`` touches, with the real method
    bound to it — so the code under test is the shipped one, not a copy."""
    stub = SimpleNamespace(
        args=OmegaConf.create({
            "rollout": {"groups_per_update": want, "collect_groups_at_once": at_once,
                        "max_oversample_factor": factor, "filter_groups": True},
            "train": {"memory_limit_gb": memory_limit_gb},
        }),
        groups=_FakeSampler(),
        collector=_FakeCollector(rewards),
    )
    stub.collect_filtered = GRPOTrainer.collect_filtered.__get__(stub)
    stub._check_memory = GRPOTrainer._check_memory.__get__(stub)
    stub.outcome_stats = GRPOTrainer.outcome_stats.__get__(stub)
    return stub


# ── one group id means one task ──────────────────────────────────────────────

def test_group_ids_are_unique_across_calls():
    """The regression. Ids from successive ``collect_groups`` calls must not collide."""
    t = _trainer([[1, 0, 0, 0]])          # every group has spread, nothing filtered
    kept, _, _ = t.collect_filtered()

    by_gid = {}
    for e in kept:
        by_gid.setdefault(e.group_id, set()).add(e.task_uid)
    assert all(len(uids) == 1 for uids in by_gid.values()), \
        f"a group id spans several tasks: { {g: u for g, u in by_gid.items() if len(u) > 1} }"


def test_collect_advances_base_gid_by_the_batch_size():
    t = _trainer([[1, 0, 0, 0]], want=16, at_once=8)
    t.collect_filtered()
    assert t.collector.calls == [0, 8], \
        "base_gid must advance by collect_groups_at_once between calls"


def test_the_loop_stops_at_want_rather_than_the_cap():
    """With ids fixed, 16 usable groups are reached in two draws of 8 — not 64."""
    t = _trainer([[1, 0, 0, 0]], want=16, at_once=8, factor=4.0)
    kept, outcomes, stats = t.collect_filtered()
    assert stats["groups_kept"] == 16
    assert stats["groups_drawn"] == 16
    assert stats["oversample_factor"] == 1.0
    assert len(outcomes) == 64            # 16 groups x G=4, not the 4x cap


def test_zero_variance_groups_still_force_oversampling():
    """Half the groups all-succeed, so twice as many draws are needed. The cap holds."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8, factor=4.0)
    _, _, stats = t.collect_filtered()
    assert stats["groups_kept"] == 8
    assert stats["groups_drawn"] == 16
    assert stats["collect/zero_variance_group_frac"] == pytest.approx(0.5)


def test_the_collection_stat_is_namespaced_away_from_the_advantage_one():
    """Both dicts land in one CSV row; sharing the key masked the stop-early signal."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    _, _, stats = t.collect_filtered()
    assert "collect/zero_variance_group_frac" in stats
    assert "zero_variance_group_frac" not in stats


# ── discarded episodes are not retained ──────────────────────────────────────

def test_rolled_episodes_are_returned_without_frames():
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    kept, outcomes, _ = t.collect_filtered()
    assert len(outcomes) > len(kept)                       # some were filtered out
    assert all(isinstance(o, EpisodeOutcome) for o in outcomes)
    assert not any(hasattr(o, "frames") for o in outcomes)


def test_outcome_stats_reports_over_everything_rolled():
    """Filtering is an update-side decision; it must not flatter the success rate."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    kept, outcomes, _ = t.collect_filtered()
    stats = t.outcome_stats(outcomes)
    assert stats["episodes"] == len(outcomes)
    assert stats["success"] == pytest.approx(np.mean([o.reward for o in outcomes]))
    # The survivors alone would read 0.25; everything rolled reads 0.625.
    assert stats["success"] > np.mean([e.reward for e in kept])


def test_episode_outcome_preserves_what_reporting_needs():
    e = Episode("u", "boss", 3, np.zeros((7, 4, 4, 3), np.uint8),
                np.zeros((4, 4, 3), np.uint8), 0, np.zeros(7, np.int64),
                np.zeros(7, np.float32), 1.0, "success")
    o = EpisodeOutcome.of(e)
    assert (o.family, o.outcome, o.reward) == ("boss", "success", 1.0)
    assert len(o) == len(e) == 7


# ── the memory guard ─────────────────────────────────────────────────────────

def test_memory_guard_raises_before_the_guest_swaps():
    t = _trainer([[1, 0, 0, 0]], memory_limit_gb=1e-6)
    with pytest.raises(MemoryError, match="host memory at"):
        t.collect_filtered()


def test_memory_guard_is_off_at_zero():
    t = _trainer([[1, 0, 0, 0]], memory_limit_gb=0.0)
    t.collect_filtered()          # must not raise
