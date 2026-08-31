from types import SimpleNamespace

import numpy as np
import pytest
import torch

from contra_policy.rl.buffer import Episode, PPOBatch
from contra_policy.rl.ppo import PPOConfig, explained_variance, gae, ppo_loss


def _episode(n=3):
    e = Episode(
        task_uid="u", family="boss", group_id=0,
        frames=np.zeros((n, 4, 4, 3), np.uint8),
        goal_image=np.zeros((4, 4, 3), np.uint8), interaction=0,
        actions=np.arange(n, dtype=np.int64) % 2,
        logprobs=np.full(n, -np.log(2), np.float32),
        reward=1.0, outcome="success")
    e.values = np.full(n, 0.5, np.float32)
    e.advantages = np.linspace(-1, 1, n, dtype=np.float32)
    e.value_targets = np.ones(n, np.float32)
    return e


def test_gae_propagates_binary_terminal_outcome():
    win_adv, win_target = gae(1.0, np.full(3, 0.5), gamma=1.0, lam=1.0)
    lose_adv, lose_target = gae(0.0, np.full(3, 0.5), gamma=1.0, lam=1.0)

    np.testing.assert_allclose(win_adv, 0.5)
    np.testing.assert_allclose(win_target, 1.0)
    np.testing.assert_allclose(lose_adv, -0.5)
    np.testing.assert_allclose(lose_target, 0.0)


def test_gae_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        gae(1.0, np.array([]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        gae(1.0, np.ones(2), lam=1.1)


def test_ppo_batch_keeps_timestep_advantages_and_padding():
    short, long = _episode(2), _episode(4)
    batch = PPOBatch([short, long])

    assert batch.advantage.shape == (2, 4)
    assert batch.old_value.shape == (2, 4)
    assert batch.value_target.shape == (2, 4)
    assert batch.mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]
    assert not batch.advantage[0, 2:].any()


def test_ppo_loss_has_unit_ratio_and_zero_reference_kl_at_collection_policy():
    batch = PPOBatch([_episode(3)])
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    values = torch.zeros(1, 3, requires_grad=True)  # sigmoid -> neutral 0.5
    cfg = PPOConfig(entropy_coef=0.0, temperature=1.0)

    loss, metrics = ppo_loss(logits, values, batch, cfg, ref_logits=logits.detach())

    assert float(metrics["ratio_mean"]) == pytest.approx(1.0)
    assert float(metrics["approx_kl"]) == pytest.approx(0.0, abs=1e-7)
    assert float(metrics["kl_ref"]) == pytest.approx(0.0, abs=1e-7)
    assert float(metrics["value_loss"]) == pytest.approx(0.25)
    loss.backward()
    assert logits.grad is not None and values.grad is not None


def test_explained_variance_distinguishes_predictor_from_constant():
    target = np.array([0, 0, 1, 1], dtype=np.float32)
    assert explained_variance(target, target) == pytest.approx(1.0)
    assert explained_variance(target, np.full(4, 0.5)) == pytest.approx(0.0)
