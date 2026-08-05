"""The fixed train probe: an unbiased counterpart to the sampler-selected metrics.

Why it exists, concretely. Run ``2026-08-03/09-23-22`` logged boss climbing 0.093 ->
0.338 on train while the same checkpoints scored 0.035 -> 0.105 on val. The gap was not
task difficulty — train and val boss match on weapon mix (within 4.2 pp) and expert
length (median 140.5 vs 138.0). It was the difficulty tournament selecting boss tasks
near p=0.5 and tracking that frontier upward, so the logged rate rose partly because the
denominator moved.

The probe removes that: same tasks every time, uniform within family, no filtering, no
difficulty weighting, and no feedback into the sampler.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from contra_policy.rl.buffer import Episode, EpisodeOutcome
from contra_policy.rl.trainer import GRPOTrainer

FAMS = ("kill", "item", "traverse", "boss")


def _task(uid, fam, label="L"):
    return SimpleNamespace(uid=uid, family=fam, label=label)


class _Catalog:
    """Families of unequal size, like the real one (boss 466 vs traverse 3693)."""

    def __init__(self, sizes):
        self.by_family = {
            f: {"L": [_task(f"{f}-{i}", f) for i in range(n)]} for f, n in sizes.items()}


class _Collector:
    """Rolls each group; success is a fixed per-task property, so a probe over the same
    tasks must return the same number every time."""

    def __init__(self, wins):
        self.wins = wins           # set of uids that succeed
        self.calls = []

    def collect_groups(self, groups, base_gid=0):
        self.calls.append([t.uid for g in groups for t in g])
        out = []
        for i, g in enumerate(groups):
            for t in g:
                won = t.uid in self.wins
                out.append(Episode(
                    task_uid=t.uid, family=t.family, group_id=base_gid + i,
                    frames=np.zeros((5, 4, 4, 3), np.uint8),
                    goal_image=np.zeros((4, 4, 3), np.uint8), interaction=0,
                    actions=np.zeros(5, np.int64), logprobs=np.zeros(5, np.float32),
                    reward=float(won), outcome="success" if won else "death",
                    task_label=t.label))
        return out


def _trainer(sizes, wins=(), per=8, repeats=1, seed=12345):
    stub = SimpleNamespace(
        args=OmegaConf.create({"probe": {"tasks_per_family": per, "repeats": repeats,
                                         "seed": seed, "every": 10}}),
        catalog=_Catalog(sizes),
        collector=_Collector(set(wins)),
        _probe_gid=10 ** 9,
    )
    stub._build_probe = GRPOTrainer._build_probe.__get__(stub)
    stub.run_probe = GRPOTrainer.run_probe.__get__(stub)
    stub.probe_tasks = stub._build_probe()
    return stub


# ── task selection ───────────────────────────────────────────────────────────

def test_the_probe_is_family_stratified():
    t = _trainer({"kill": 200, "item": 50, "traverse": 900, "boss": 60}, per=8)
    n = {f: sum(1 for x in t.probe_tasks if x.family == f) for f in FAMS}
    assert n == {"kill": 8, "item": 8, "traverse": 8, "boss": 8}, \
        "boss must get the same n as traverse despite being 15x rarer"


def test_a_small_family_is_capped_at_its_size():
    t = _trainer({"kill": 200, "item": 3, "traverse": 900, "boss": 60}, per=8)
    assert sum(1 for x in t.probe_tasks if x.family == "item") == 3


def test_the_task_set_is_identical_across_runs():
    a = _trainer({"kill": 200, "item": 50, "traverse": 900, "boss": 60})
    b = _trainer({"kill": 200, "item": 50, "traverse": 900, "boss": 60})
    assert [x.uid for x in a.probe_tasks] == [x.uid for x in b.probe_tasks]


def test_a_different_seed_gives_a_different_set():
    a = _trainer({"kill": 200, "item": 50, "traverse": 900, "boss": 60}, seed=1)
    b = _trainer({"kill": 200, "item": 50, "traverse": 900, "boss": 60}, seed=2)
    assert [x.uid for x in a.probe_tasks] != [x.uid for x in b.probe_tasks]


def test_tasks_per_family_zero_disables_the_probe():
    t = _trainer({"kill": 200, "boss": 60}, per=0)
    assert t.probe_tasks == [] and t.run_probe() == {}


# ── measurement ──────────────────────────────────────────────────────────────

def test_the_probe_is_repeatable_on_a_fixed_policy():
    """The property that makes a trend readable: no task-selection variance."""
    t = _trainer({"kill": 40, "item": 40, "traverse": 40, "boss": 40})
    t.collector.wins = {x.uid for x in t.probe_tasks[::3]}
    assert t.run_probe() == t.run_probe()
    assert t.collector.calls[0] == t.collector.calls[1]


def test_success_is_reported_per_family_with_intervals():
    t = _trainer({"kill": 40, "item": 40, "traverse": 40, "boss": 40}, per=8)
    t.collector.wins = {x.uid for x in t.probe_tasks if x.family == "kill"}
    out = t.run_probe()
    assert out["probe/kill/success"] == 1.0
    assert out["probe/boss/success"] == 0.0
    assert out["probe/kill/ci_lo"] < 1.0 and out["probe/boss/ci_hi"] > 0.0
    assert out["probe/episodes"] == 32


def test_macro_average_does_not_let_a_big_family_hide_a_small_one():
    """The probe is family-balanced, so macro and pooled agree here — but macro is what
    is reported, because the *training* mix is not balanced and pooling would track it."""
    t = _trainer({"kill": 40, "item": 40, "traverse": 40, "boss": 40}, per=8)
    t.collector.wins = {x.uid for x in t.probe_tasks if x.family != "boss"}
    out = t.run_probe()
    assert out["probe/macro"] == pytest.approx(0.75)
    assert out["probe/boss/success"] == 0.0


def test_repeats_roll_each_task_more_than_once():
    t = _trainer({"kill": 10, "boss": 10}, per=4, repeats=3)
    out = t.run_probe()
    assert out["probe/episodes"] == 4 * 2 * 3
    assert len(t.collector.calls[0]) == 24


def test_probe_group_ids_cannot_collide_with_training_ids():
    """Probe episodes carry group ids too; they must not be mistaken for a training
    group if anything downstream buckets by id."""
    t = _trainer({"kill": 40, "boss": 40}, per=8)
    t.run_probe()
    before = t._probe_gid
    t.run_probe()
    assert before >= 10 ** 9 and t._probe_gid > before


def test_the_probe_does_not_feed_the_difficulty_sampler():
    """It must not perturb the sampler it exists to measure around. `run_probe` has no
    reference to `self.groups`, so a stub without that attribute must still work."""
    t = _trainer({"kill": 40, "boss": 40}, per=8)
    assert not hasattr(t, "groups")
    t.run_probe()          # would raise if it tried to observe


# ── success means the goal, not a positive reward ────────────────────────────

def test_success_counts_the_outcome_not_a_positive_reward():
    """Since doc/0005 §2 a losing boss episode scores `progress_coef * damage`, so
    `reward > 0` means "dealt some damage", not "won". The first graded run reported
    boss=0.76 and probe boss=0.90 against a true rate near 0.10 before this was caught."""
    t = _trainer({"boss": 40}, per=8)
    winners = {t.probe_tasks[0].uid}          # 1 of 8 tasks actually wins

    def graded(groups, base_gid=0):
        out = []
        for i, g in enumerate(groups):
            for task in g:
                won = task.uid in winners
                out.append(Episode(
                    task_uid=task.uid, family=task.family, group_id=base_gid + i,
                    frames=np.zeros((5, 4, 4, 3), np.uint8),
                    goal_image=np.zeros((4, 4, 3), np.uint8), interaction=0,
                    actions=np.zeros(5, np.int64), logprobs=np.zeros(5, np.float32),
                    reward=1.0 if won else 0.25,   # every loser still scores > 0
                    outcome="success" if won else "death", task_label=task.label))
        return out

    t.collector.collect_groups = graded
    out = t.run_probe()
    assert out["probe/boss/success"] == pytest.approx(1 / 8)   # not 1.0
    # the graded signal is still reported, just not as "success"
    assert out["probe/reward_mean"] == pytest.approx((1.0 + 7 * 0.25) / 8)


def test_outcome_stats_agrees_with_the_probe_on_what_success_means():
    """Both reporting paths must read `outcome`, or the two lines in the log disagree."""
    stub = SimpleNamespace()
    stub.outcome_stats = GRPOTrainer.outcome_stats.__get__(stub)
    eps = [EpisodeOutcome(family="boss", outcome="death", reward=0.4, n_steps=5),
           EpisodeOutcome(family="boss", outcome="success", reward=1.0, n_steps=5)]
    out = stub.outcome_stats(eps)
    assert out["success"] == pytest.approx(0.5)      # not 1.0
    assert out["reward_mean"] == pytest.approx(0.7)
    assert out["boss/success"] == pytest.approx(0.5)
