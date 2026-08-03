"""The graded terminal reward — doc/0005 §2.

A group is G rollouts of one task, and its advantage is
``(r_i - mean) / std``. When every member gets the same reward the group contributes no
gradient at all: 43-75% of boss groups at G=8, and 53% of every rollout budget measured
across three runs. Grading failures by damage dealt gives those groups real spread.

The property that keeps it honest is that the ranges stay **disjoint** — a failure can
never outrank a success, however much damage it did.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from contra_policy.rl.buffer import group_advantages
from contra_policy.rl.rollout import EpisodeCollector


def _slot(outcome, hp_peak=100, hp_last=100):
    """`hp_peak`, not the value at step 0: a boss task starts at the reveal, before the
    boss occupies an enemy slot, so HP reads 0 for the first several steps."""
    return SimpleNamespace(outcome=outcome, hp0=hp_peak, hp_peak=hp_peak, hp_last=hp_last)


def _reward_fn(**reward):
    """`_reward_for` bound to a stub carrying only the reward table it reads."""
    base = {"success": 1.0, "death": 0.0, "timeout": 0.0, "step": 0.0,
            "truncated": 0.0, "progress_coef": 0.0}
    stub = SimpleNamespace(reward={**base, **reward})
    return EpisodeCollector._reward_for.__get__(stub, EpisodeCollector)


# ── grading ──────────────────────────────────────────────────────────────────

def test_a_failure_is_graded_by_the_fraction_of_hp_removed():
    r = _reward_fn(progress_coef=0.5)
    assert r(_slot("death", 100, 100)) == pytest.approx(0.0)    # no damage
    assert r(_slot("death", 100, 60)) == pytest.approx(0.20)    # 40% removed
    assert r(_slot("death", 100, 0)) == pytest.approx(0.50)     # everything, still died


def test_grading_is_relative_to_the_peak_hp():
    """Mid-fight tasks begin with the boss already damaged; absolute HP removed would
    understate them. The fraction makes tasks comparable wherever they start."""
    r = _reward_fn(progress_coef=0.5)
    full = r(_slot("death", 100, 50))     # half of a full fight
    part = r(_slot("death", 40, 20))      # half of what remained
    assert full == pytest.approx(part)


def test_no_failure_can_outscore_a_success():
    """The disjointness property. Without it the objective stops being 'kill the boss'."""
    r = _reward_fn(progress_coef=0.5)
    best_failure = max(r(_slot(o, 100, 0)) for o in ("death", "timeout", "truncated"))
    assert best_failure < r(_slot("success", 100, 0))


def test_success_is_never_graded():
    """A win is a win — the speed term (doc/0005 §3) is the only thing that will ever
    differentiate successes, and it is not this."""
    r = _reward_fn(progress_coef=0.5)
    assert r(_slot("success", 100, 100)) == r(_slot("success", 100, 0)) == 1.0


def test_zero_coefficient_reproduces_the_binary_reward():
    r = _reward_fn(progress_coef=0.0)
    assert r(_slot("death", 100, 0)) == 0.0
    assert r(_slot("success", 100, 100)) == 1.0


def test_families_without_a_progress_signal_are_untouched():
    """`hp0 = -1` means the maker exposes no accessor — kill/item/traverse stay binary."""
    r = _reward_fn(progress_coef=0.5)
    assert r(_slot("death", -1, -1)) == 0.0


def test_hp_that_somehow_rises_cannot_produce_negative_reward():
    r = _reward_fn(progress_coef=0.5)
    assert r(_slot("death", 100, 140)) == 0.0


# ── the point of the exercise ────────────────────────────────────────────────

def test_an_all_failing_group_now_carries_gradient():
    """The measurement this exists to fix: eight identical zeros produce nothing."""
    binary = [0.0] * 8
    _, stats = group_advantages(binary, [0] * 8)
    assert stats["zero_variance_group_frac"] == 1.0
    assert stats["adv_abs_mean"] == 0.0

    r = _reward_fn(progress_coef=0.5)
    graded = [r(_slot("death", 100, hp)) for hp in (98, 69, 95, 56, 92, 88, 62, 97)]
    adv, stats = group_advantages(graded, [0] * 8)
    assert stats["zero_variance_group_frac"] == 0.0
    assert stats["adv_abs_mean"] > 0.5
    # and it ranks them the way the damage did
    assert np.argmax(adv) == int(np.argmax(graded))


def test_a_group_mixing_wins_and_losses_still_favours_the_wins():
    r = _reward_fn(progress_coef=0.5)
    rewards = [r(_slot("success", 100, 0)), r(_slot("death", 100, 10)),
               r(_slot("death", 100, 90)), r(_slot("death", 100, 100))]
    adv, _ = group_advantages(rewards, [0] * 4)
    assert adv[0] == max(adv)
    assert adv[1] > adv[2] > adv[3]      # graded failures are ordered by damage


def test_dying_before_the_boss_spawns_scores_zero():
    """A boss task starts at the reveal, so HP reads 0 until the boss occupies a slot.
    An episode that never gets there has no peak to measure damage against, and made no
    progress — measured: all 24 sampled tasks read hp=0 at step 0 and peaked at 25-64."""
    r = _reward_fn(progress_coef=0.5)
    assert r(_slot("death", hp_peak=0, hp_last=0)) == 0.0
