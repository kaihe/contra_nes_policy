"""Replay targets and joint objective for design 0029."""

import numpy as np
import torch

from contra_policy.alphazero import (AlphaZeroBatch, SearchEpisode, alphazero_loss,
                                     evaluate_epoch)


def episode(length=3):
    return SearchEpisode(
        frames=np.zeros((length, 8, 8, 3), np.uint8),
        policy_targets=np.full((length, 2), 0.5, np.float32),
        rewards=np.array([1.0, -0.5, 3.0][:length], np.float32),
        motion=np.zeros((length, 2), np.float32),
        weapon=np.zeros(length, np.int64), rapid=np.zeros(length, np.float32),
        progress=np.linspace(0, 1, length, dtype=np.float32),
        progress_mask=np.ones(length, np.float32),
        interaction=4, outcome="success")


def test_terminal_success_is_copied_to_every_decision():
    assert episode().success_targets.tolist() == [1.0, 1.0, 1.0]


def test_batch_padding_does_not_create_valid_targets():
    batch = AlphaZeroBatch([episode(3), episode(2)], torch.device("cpu"))
    assert batch.mask.tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch.value_target[0].tolist() == [1.0, 1.0, 1.0]


def test_joint_loss_reaches_every_head():
    batch = AlphaZeroBatch([episode()], torch.device("cpu"))
    out = {
        "pi_logits": torch.randn(1, 3, 2, requires_grad=True),
        "vpred": torch.randn(1, 3, requires_grad=True),
        "motion": torch.randn(1, 3, 2, requires_grad=True),
        "weapon_logits": torch.randn(1, 3, 6, requires_grad=True),
        "rapid_logit": torch.randn(1, 3, requires_grad=True),
        "progress_logit": torch.randn(1, 3, requires_grad=True),
    }
    loss, metrics = alphazero_loss(out, batch)
    loss.backward()
    assert set(metrics) == {"loss", "policy_loss", "value_loss", "motion_loss",
                            "weapon_loss", "rapid_loss", "progress_loss"}
    assert all(tensor.grad is not None for tensor in out.values())


def test_evaluate_epoch_does_not_create_gradients():
    class Model(torch.nn.Module):
        def forward(self, images, goal, interaction):
            del goal, interaction
            b, t = images.shape[:2]
            zero = self.weight.expand(b, t)
            return {"pi_logits": zero.unsqueeze(-1).expand(b, t, 2), "vpred": zero,
                    "motion": zero.unsqueeze(-1).expand(b, t, 2),
                    "weapon_logits": zero.unsqueeze(-1).expand(b, t, 6),
                    "rapid_logit": zero, "progress_logit": zero}

        weight = torch.nn.Parameter(torch.tensor(0.0))

    model = Model()
    metrics = evaluate_epoch(model, [episode()], device=torch.device("cpu"), batch_episodes=1)
    assert "policy_loss" in metrics
    assert model.weight.grad is None
