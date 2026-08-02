"""Biasing task selection toward groups that will carry gradient.

The measured problem (run ``2026-08-02/11-48-03``): the policy scores ~83% on *train*
tasks, so 59% of groups came back all-success and 58% of the rollout budget produced no
gradient. The weight is P(a group of G is not all-agreeing) — exactly the chance it
survives `filter_groups` — which peaks at p=0.5 and falls off at *both* ends.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from contra_policy.rl.tasks import DifficultyTracker, GroupSampler


def _t(uid, label=None):
    return SimpleNamespace(uid=uid, label=label if label is not None else uid)


def _ep(uid, reward, fam="kill", label="L"):
    return SimpleNamespace(task_uid=uid, reward=float(reward), family=fam,
                           task_label=label)


class _StubSampler:
    """Cycles a fixed task list, so tournament selection is the only thing choosing."""

    def __init__(self, uids):
        self.tasks = [SimpleNamespace(uid=u, family="kill", label=u) for u in uids]
        self.i = 0

    def sample(self):
        t = self.tasks[self.i % len(self.tasks)]
        self.i += 1
        return t

    def state(self):
        return {"i": self.i}

    def load_state(self, s):
        self.i = s["i"]


# ── the weight ───────────────────────────────────────────────────────────────

def test_weight_peaks_at_half_and_falls_off_both_ends():
    d = DifficultyTracker(group_size=8, min_weight=0.0)
    d.observe("mid", "mid", 50, 100)        # p ~ 0.5
    d.observe("easy", "easy", 83, 100)       # p ~ 0.83, the measured train rate
    d.observe("hard", "hard", 3, 100)        # p ~ 0.035, boss
    assert d.weight(_t("mid")) > d.weight(_t("easy")) > d.weight(_t("hard"))
    assert d.weight(_t("mid")) == pytest.approx(1 - 2 * 0.5 ** 8, abs=0.02)


def test_an_always_succeeding_task_is_as_useless_as_an_always_failing_one():
    """Two-sidedness — the property doc/0004 §5 records and G was chosen without."""
    d = DifficultyTracker(group_size=8, min_weight=0.0)
    d.observe("never", "never", 0, 200)
    d.observe("always", "always", 200, 200)
    assert d.weight(_t("never")) == pytest.approx(d.weight(_t("always")), abs=1e-6)


def test_survival_weighting_is_gentler_on_boss_than_bernoulli_variance():
    """Why `1 - p^G - (1-p)^G` rather than `4p(1-p)`: at the tails the latter punishes
    boss harder, and boss is the family we most need to keep sampling."""
    d = DifficultyTracker(group_size=8, min_weight=0.0)
    d.observe("boss", "boss", 35, 1000)
    d.observe("easy", "easy", 830, 1000)
    p_boss, p_easy = d.p_hat("boss", "boss"), d.p_hat("easy", "easy")
    var_ratio = (4 * p_easy * (1 - p_easy)) / (4 * p_boss * (1 - p_boss))
    assert d.weight(_t("easy")) / d.weight(_t("boss")) < var_ratio


def test_an_unseen_task_reads_one_half_and_is_explored_first():
    d = DifficultyTracker(group_size=8)
    d.observe("known", "known", 100, 100)
    assert d.p_hat("never-seen", "never-seen") == pytest.approx(0.5)
    assert d.weight(_t("never-seen")) > d.weight(_t("known"))


def test_min_weight_keeps_hopeless_tasks_reachable():
    """Without a floor, a task the policy always fails could never be rediscovered."""
    d = DifficultyTracker(group_size=8, min_weight=0.05)
    d.observe("hopeless", "hopeless", 0, 10_000)
    assert d.weight(_t("hopeless")) == pytest.approx(0.05)


# ── the hierarchical prior ───────────────────────────────────────────────────

def test_an_unseen_task_inherits_its_labels_measured_rate():
    """The reason the bias works at all. 6438 tasks and ~32 drawn per update means most
    tasks are never seen twice — but 13 labels at 495 tasks each converge in a few
    updates, so an unseen task starts from a real number instead of 0.5."""
    d = DifficultyTracker(group_size=8)
    for i in range(60):                      # many *different* tasks, one label
        d.observe(f"seen-{i}", "kill_turret", 7, 8)
    assert d.p_hat("brand-new", "kill_turret") == pytest.approx(0.875, abs=0.03)
    assert d.p_hat("brand-new", "unseen_label") == pytest.approx(0.5)


def test_task_evidence_overrides_the_label_prior():
    d = DifficultyTracker(group_size=8, decay=1.0)
    for i in range(60):
        d.observe(f"seen-{i}", "L", 8, 8)    # the label is easy
    d.observe("odd-one", "L", 0, 40)         # this task is not
    assert d.p_label("L") > 0.9
    assert d.p_hat("odd-one", "L") < 0.1


def test_the_label_prior_makes_easy_tasks_downweighted_before_they_are_seen():
    """Without it, every unseen task reads p=0.5 and the tournament is a coin flip."""
    d = DifficultyTracker(group_size=8, min_weight=0.0)
    for i in range(60):
        d.observe(f"easy-{i}", "easy_label", 8, 8)
        d.observe(f"mid-{i}", "mid_label", 4, 8)
    assert d.weight(_t("fresh", "mid_label")) > 2 * d.weight(_t("fresh", "easy_label"))


# ── estimates track a moving policy ──────────────────────────────────────────

def test_decay_lets_a_new_estimate_outvote_an_old_one():
    d = DifficultyTracker(group_size=8, decay=0.9)
    for _ in range(50):
        d.observe("t", "t", 8, 8)                 # long history of always succeeding
    assert d.p_hat("t", "t") > 0.9
    for _ in range(20):
        d.observe("t", "t", 0, 8)                 # the policy got worse at it
    assert d.p_hat("t", "t") < 0.2


def test_decay_bounds_the_effective_sample_size():
    d = DifficultyTracker(group_size=8, decay=0.9)
    for _ in range(1000):
        d.observe("t", "t", 4, 8)
    assert d.n["t"] == pytest.approx(8 / (1 - 0.9), rel=0.01)   # ~80, not 8000


def test_observe_episodes_aggregates_a_group_into_one_update():
    d = DifficultyTracker(group_size=4, decay=1.0)
    d.observe_episodes([_ep("a", 1), _ep("a", 0), _ep("a", 1), _ep("b", 0)])
    assert (d.s["a"], d.n["a"]) == (2.0, 3.0)
    assert (d.s["b"], d.n["b"]) == (0.0, 1.0)


# ── tournament selection ─────────────────────────────────────────────────────

def test_tournament_shifts_draws_toward_informative_tasks():
    """The stub alternates, so every tournament of 4 holds 2 easy and 2 mid and the
    pick rate is exactly ``w_mid / (w_mid + w_easy)`` — 0.747 at these rates, against
    0.5 unbiased. Pinning the analytic value rather than a threshold, so a change in
    the weight function shows up here as a number rather than a pass."""
    d = DifficultyTracker(group_size=8, min_weight=0.05)
    d.observe("easy", "easy", 950, 1000)     # p ~ 0.95, mostly all-success groups
    d.observe("mid", "mid", 500, 1000)      # p ~ 0.5
    gs = GroupSampler(_StubSampler(["easy", "mid"]), group_size=8,
                      difficulty=d, candidates=4, seed=0)
    picks = [g[0].uid for g in gs.sample_groups(4000)]

    w_mid, w_easy = d.weight(_t("mid")), d.weight(_t("easy"))
    expected = w_mid / (w_mid + w_easy)
    assert expected == pytest.approx(0.747, abs=0.01)
    assert picks.count("mid") / len(picks) == pytest.approx(expected, abs=0.02)


def test_candidates_of_one_disables_the_bias():
    d = DifficultyTracker(group_size=8)
    d.observe("easy", "easy", 1000, 1000)
    d.observe("mid", "mid", 500, 1000)
    gs = GroupSampler(_StubSampler(["easy", "mid"]), group_size=8,
                      difficulty=d, candidates=1, seed=0)
    picks = [g[0].uid for g in gs.sample_groups(1000)]
    assert picks.count("mid") / len(picks) == pytest.approx(0.5, abs=0.05)


def test_no_tracker_leaves_sampling_untouched():
    gs = GroupSampler(_StubSampler(["a", "b"]), group_size=4, seed=0)
    picks = [g[0].uid for g in gs.sample_groups(100)]
    assert picks.count("a") == 50
    gs.observe([_ep("a", 1)])        # must be a no-op, not an error
    assert gs.stats() == {}


def test_a_group_is_still_g_copies_of_one_task():
    """The premise the bias must not break: every member of a group is the same task."""
    d = DifficultyTracker(group_size=8)
    gs = GroupSampler(_StubSampler(["a", "b", "c"]), group_size=8,
                      difficulty=d, candidates=3, seed=0)
    for g in gs.sample_groups(50):
        assert len(g) == 8 and len({t.uid for t in g}) == 1


# ── resumption ───────────────────────────────────────────────────────────────

def test_state_round_trips_the_difficulty_estimates():
    d = DifficultyTracker(group_size=8)
    d.observe("a", "a", 3, 8)
    gs = GroupSampler(_StubSampler(["a", "b"]), 8, difficulty=d, candidates=2, seed=1)
    saved = gs.state()

    d2 = DifficultyTracker(group_size=8)
    gs2 = GroupSampler(_StubSampler(["a", "b"]), 8, difficulty=d2, candidates=2, seed=9)
    gs2.load_state(saved)
    assert d2.p_hat("a", "a") == pytest.approx(d.p_hat("a", "a"))
    assert [g[0].uid for g in gs2.sample_groups(20)] == \
           [g[0].uid for g in gs.sample_groups(20)]


def test_load_state_tolerates_a_pre_difficulty_checkpoint():
    """Old checkpoints stored the bare TaskSampler state, not a nested dict."""
    gs = GroupSampler(_StubSampler(["a", "b"]), 8,
                      difficulty=DifficultyTracker(8), candidates=2, seed=0)
    gs.load_state({"i": 7})          # must not raise
    assert gs.sampler.i == 7
