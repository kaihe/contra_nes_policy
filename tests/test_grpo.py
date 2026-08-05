"""GRPO: the group baseline, and the objective built on it.

The property worth pinning hardest is the *degenerate group* — G rollouts that all
succeed or all fail have no advantage spread and contribute no gradient. That is not a
bug, it is what "nothing to learn here yet" looks like, and at boss's ~3.5% success it
will be most boss groups. It has to be measured rather than discovered.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from contra_policy.rl.buffer import (Episode, GroupBatch, group_advantages,
                                     iter_minibatches)
from contra_policy.rl.grpo import GRPOConfig, grpo_loss


def _ep(t, gid, reward, fam="kill", uid="u"):
    return Episode(uid, fam, gid, np.zeros((t, 8, 8, 3), np.uint8),
                   np.zeros((8, 8, 3), np.uint8), 0, np.zeros(t, np.int64),
                   np.zeros(t, np.float32), reward, "x")


# ── the group baseline ───────────────────────────────────────────────────────

def test_advantage_is_own_reward_minus_the_group():
    adv, _ = group_advantages([1, 0, 0, 0], [0, 0, 0, 0])
    assert adv[0] > 0 and all(a < 0 for a in adv[1:])
    assert abs(float(adv.sum())) < 1e-5, "a group's advantages must sum to zero"


def test_a_group_that_all_fails_contributes_nothing():
    """The central open question for GRPO here, made visible instead of inferred."""
    adv, st = group_advantages([0, 0, 0, 0], [0, 0, 0, 0])
    assert np.allclose(adv, 0.0)
    assert st["zero_variance_group_frac"] == 1.0


def test_a_group_that_all_succeeds_also_contributes_nothing():
    adv, st = group_advantages([1, 1, 1], [0, 0, 0])
    assert np.allclose(adv, 0.0)
    assert st["zero_variance_group_frac"] == 1.0


def test_degenerate_fraction_counts_groups_not_episodes():
    # three groups, only the middle one degenerate
    r = [1, 0, 0, 0, 0, 0, 1, 1, 0]
    g = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    _adv, st = group_advantages(r, g)
    assert st["groups"] == 3.0
    assert st["zero_variance_group_frac"] == pytest.approx(1 / 3)


def test_groups_are_scored_independently():
    """A hard task's group must not be baselined against an easy task's."""
    r = [1, 0], [1, 1]
    adv, _ = group_advantages([1, 0, 1, 1], [0, 0, 1, 1])
    assert adv[0] > 0 and adv[1] < 0          # group 0 separates
    assert adv[2] == 0.0 and adv[3] == 0.0    # group 1 is degenerate, not dragged by 0


def test_unnormalised_mode_keeps_the_raw_difference():
    adv, _ = group_advantages([1, 0, 0, 0], [0] * 4, normalise=False)
    assert float(adv[0]) == pytest.approx(0.75)
    assert float(adv[1]) == pytest.approx(-0.25)


# ── batching ─────────────────────────────────────────────────────────────────

def test_batch_pads_to_its_own_maximum_and_masks_the_rest():
    eps = [_ep(3, 0, 1.0), _ep(7, 0, 0.0)]
    adv, _ = group_advantages([e.reward for e in eps], [e.group_id for e in eps])
    b = GroupBatch(eps, adv)
    assert b.image.shape[:2] == (2, 7)
    assert b.mask[0].tolist() == [1] * 3 + [0] * 4
    assert b.steps == 10


def test_one_advantage_per_episode_not_per_step():
    """A terminal reward says nothing about which step earned it; broadcasting rather
    than materialising per-step values keeps that honest."""
    eps = [_ep(4, 0, 1.0), _ep(4, 0, 0.0)]
    adv, _ = group_advantages([1.0, 0.0], [0, 0])
    b = GroupBatch(eps, adv)
    assert b.advantage.shape == (2,)


def test_minibatches_never_split_an_episode():
    eps = [_ep(3 + i, i // 4, float(i % 2)) for i in range(12)]
    adv, _ = group_advantages([e.reward for e in eps], [e.group_id for e in eps])
    rng = np.random.default_rng(0)
    seen = 0
    for mb in iter_minibatches(eps, adv, 4, rng):
        seen += len(mb)
        # every row's mask is a contiguous prefix — no fragment starts mid-episode
        for row in mb.mask:
            r = row.tolist()
            assert r == sorted(r, reverse=True)
    assert seen == len(eps)


# ── the objective ────────────────────────────────────────────────────────────

def _batch(b=4, t=6, adv=None):
    eps = [_ep(t, i // 2, float(i % 2)) for i in range(b)]
    a = np.array(adv if adv is not None else [1.0, -1.0] * (b // 2), dtype=np.float32)
    return GroupBatch(eps, a)


def test_a_zero_advantage_batch_produces_no_policy_gradient():
    """The degenerate group, end to end: all-failure groups must not move the policy."""
    batch = _batch(adv=[0.0, 0.0, 0.0, 0.0])
    logits = torch.randn(4, 6, 21, requires_grad=True)
    cfg = GRPOConfig(kl_coef=0.0, entropy_coef=0.0)
    loss, m = grpo_loss(logits, batch, cfg)
    assert float(m["policy_loss"]) == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert logits.grad.abs().sum() == pytest.approx(0.0, abs=1e-6)


def test_positive_advantage_raises_the_taken_action_logprob():
    batch = _batch(adv=[1.0, 1.0, 1.0, 1.0])
    logits = torch.zeros(4, 6, 21, requires_grad=True)
    loss, _ = grpo_loss(logits, batch, GRPOConfig(kl_coef=0.0, entropy_coef=0.0))
    loss.backward()
    taken = logits.grad[:, :, 0]              # action 0 was taken everywhere
    assert (taken < 0).all(), "gradient should push the taken action's logit up"


def test_the_clip_bounds_the_ratio_term():
    batch = _batch(adv=[5.0, 5.0, 5.0, 5.0])
    batch.old_logprob = torch.full_like(batch.old_logprob, -20.0)   # huge ratio
    logits = torch.zeros(4, 6, 21, requires_grad=True)
    _loss, m = grpo_loss(logits, batch, GRPOConfig(clip_ratio=0.2, kl_coef=0.0,
                                                   entropy_coef=0.0))
    assert float(m["clip_frac"]) == 1.0


def test_reference_kl_is_zero_against_itself_and_positive_otherwise():
    batch = _batch()
    logits = torch.randn(4, 6, 21)
    cfg = GRPOConfig(kl_coef=0.02, entropy_coef=0.0)
    _l, same = grpo_loss(logits, batch, cfg, ref_logits=logits.clone())
    _l, diff = grpo_loss(logits, batch, cfg, ref_logits=torch.randn(4, 6, 21))
    assert float(same["kl_ref"]) == pytest.approx(0.0, abs=1e-5)
    assert float(diff["kl_ref"]) > 0.1


def test_approx_kl_is_non_negative():
    """The k3 estimator, not the naive mean of (old - new) which is noisy around zero
    and can report a negative KL."""
    batch = _batch()
    for _ in range(5):
        _l, m = grpo_loss(torch.randn(4, 6, 21), batch, GRPOConfig())
        assert float(m["approx_kl"]) >= 0.0


def test_padded_steps_never_reach_the_loss():
    a = _ep(6, 0, 1.0), _ep(2, 0, 0.0)
    adv, _ = group_advantages([1.0, 0.0], [0, 0])
    batch = GroupBatch(list(a), adv)
    logits = torch.zeros(2, 6, 21, requires_grad=True)
    loss, _ = grpo_loss(logits, batch, GRPOConfig(kl_coef=0.0, entropy_coef=0.0))
    loss.backward()
    # rows beyond the second episode's 2 real steps must be untouched
    assert logits.grad[1, 2:].abs().sum() == pytest.approx(0.0, abs=1e-7)


# ── group filtering (dynamic sampling) ───────────────────────────────────────

def test_filter_drops_groups_whose_members_agree():
    """Group filtering: a group with zero reward variance has zero advantage
    everywhere, so it moves the policy not at all. Both tails qualify."""
    from contra_policy.rl.buffer import filter_groups

    eps = ([_ep(3, 0, 1.0), _ep(3, 0, 0.0)]          # mixed  -> kept
           + [_ep(3, 1, 0.0), _ep(3, 1, 0.0)]        # all fail    -> dropped
           + [_ep(3, 2, 1.0), _ep(3, 2, 1.0)])       # all succeed -> dropped
    kept, st = filter_groups(eps)
    assert {e.group_id for e in kept} == {0}
    assert st["groups_collected"] == 3.0 and st["groups_kept"] == 1.0
    assert st["zero_variance_group_frac"] == pytest.approx(2 / 3)
    assert st["episodes_discarded"] == 4.0


def test_filtering_changes_the_scale_of_the_update_not_only_its_cost():
    """Why filtering is not merely an optimisation.

    The loss is a mean over the batch, so unfiltered zero-advantage episodes divide the
    signal down — the effective step size would track how hard the current task mix
    happens to be rather than staying fixed.
    """
    from contra_policy.rl.buffer import filter_groups

    mixed = [_ep(4, 0, 1.0), _ep(4, 0, 0.0)]
    dead = [_ep(4, 1, 0.0), _ep(4, 1, 0.0)]
    logits = torch.zeros(2, 4, 21)

    def grad_norm(eps):
        adv, _ = group_advantages([e.reward for e in eps], [e.group_id for e in eps])
        lg = logits.clone().requires_grad_(True)
        b = GroupBatch(eps, adv)
        loss, _ = grpo_loss(lg, b, GRPOConfig(kl_coef=0.0, entropy_coef=0.0))
        loss.backward()
        return float(lg.grad.abs().sum())

    kept, _ = filter_groups(mixed + dead)
    filtered = grad_norm(kept)
    # Same signal, diluted by two dead episodes -> the update is literally smaller.
    unfiltered_all = mixed + dead
    adv, _ = group_advantages([e.reward for e in unfiltered_all],
                              [e.group_id for e in unfiltered_all])
    lg = torch.zeros(4, 4, 21, requires_grad=True)
    loss, _ = grpo_loss(lg, GroupBatch(unfiltered_all, adv),
                        GRPOConfig(kl_coef=0.0, entropy_coef=0.0))
    loss.backward()
    assert float(lg.grad.abs().sum()) < filtered


def test_a_group_smaller_than_two_cannot_be_baselined():
    from contra_policy.rl.buffer import filter_groups

    kept, st = filter_groups([_ep(3, 0, 1.0)])
    assert kept == [] and st["groups_kept"] == 0.0
