"""Searched-episode records and joint policy/value/state training for design 0029."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SearchEpisode:
    frames: np.ndarray
    policy_targets: np.ndarray
    rewards: np.ndarray
    motion: np.ndarray
    weapon: np.ndarray
    rapid: np.ndarray
    interaction: int
    outcome: str

    def __post_init__(self) -> None:
        n = len(self.frames)
        if not all(len(x) == n for x in (self.policy_targets, self.rewards, self.motion,
                                         self.weapon, self.rapid)):
            raise ValueError("all episode targets must align with frames")

    @property
    def returns(self) -> np.ndarray:
        return reward_to_go(self.rewards)


def reward_to_go(rewards: np.ndarray) -> np.ndarray:
    """Undiscounted sum of rewards from every decision through termination."""
    values = np.asarray(rewards, dtype=np.float32)
    return np.flip(np.cumsum(np.flip(values), dtype=np.float32)).copy()


class AlphaZeroBatch:
    """Pad complete searched episodes without cutting their causal histories."""

    def __init__(self, episodes: Sequence[SearchEpisode], device: torch.device):
        if not episodes:
            raise ValueError("cannot build an empty AlphaZero batch")
        b, t = len(episodes), max(len(e.frames) for e in episodes)
        image_shape = episodes[0].frames.shape[1:]
        actions = episodes[0].policy_targets.shape[-1]
        self.images = torch.zeros((b, t, *image_shape), dtype=torch.uint8, device=device)
        self.policy_target = torch.zeros((b, t, actions), device=device)
        self.value_target = torch.zeros((b, t), device=device)
        self.motion = torch.zeros((b, t, 2), device=device)
        self.weapon = torch.zeros((b, t), dtype=torch.long, device=device)
        self.rapid = torch.zeros((b, t), device=device)
        self.mask = torch.zeros((b, t), device=device)
        self.interaction = torch.tensor([e.interaction for e in episodes], device=device)
        for i, episode in enumerate(episodes):
            n = len(episode.frames)
            self.images[i, :n] = torch.from_numpy(episode.frames).to(device)
            self.policy_target[i, :n] = torch.from_numpy(episode.policy_targets).to(device)
            self.value_target[i, :n] = torch.from_numpy(episode.returns).to(device)
            self.motion[i, :n] = torch.from_numpy(episode.motion).to(device)
            self.weapon[i, :n] = torch.from_numpy(episode.weapon).to(device)
            self.rapid[i, :n] = torch.from_numpy(episode.rapid).to(device)
            self.mask[i, :n] = 1


@dataclass(frozen=True)
class LossWeights:
    policy: float = 1.0
    value: float = 1.0
    motion: float = 0.1
    weapon: float = 0.1
    rapid: float = 0.1


def alphazero_loss(out: dict[str, torch.Tensor], batch: AlphaZeroBatch,
                   weights: LossWeights = LossWeights()
                   ) -> tuple[torch.Tensor, dict[str, float]]:
    """Joint soft-policy, reward-to-go, and decoded-state objective."""
    mask, denom = batch.mask, batch.mask.sum().clamp(min=1)
    mean = lambda x: (x * mask).sum() / denom
    policy = mean(-(batch.policy_target * F.log_softmax(out["pi_logits"].float(), -1)).sum(-1))
    # Dense mc_search returns are tens of points on the fixed Laser episode. Smooth L1
    # keeps a randomly initialized value head from overwhelming policy distillation.
    value = mean(F.smooth_l1_loss(out["vpred"].float(), batch.value_target,
                                  reduction="none"))
    motion = mean(F.smooth_l1_loss(out["motion"].float(), batch.motion, reduction="none").mean(-1))
    weapon = mean(F.cross_entropy(out["weapon_logits"].float().transpose(1, 2),
                                  batch.weapon, reduction="none"))
    rapid = mean(F.binary_cross_entropy_with_logits(out["rapid_logit"].float(), batch.rapid,
                                                     reduction="none"))
    total = (weights.policy * policy + weights.value * value + weights.motion * motion
             + weights.weapon * weapon + weights.rapid * rapid)
    metrics = {"loss": float(total.detach()), "policy_loss": float(policy.detach()),
               "value_loss": float(value.detach()), "motion_loss": float(motion.detach()),
               "weapon_loss": float(weapon.detach()), "rapid_loss": float(rapid.detach())}
    return total, metrics


def train_epoch(model, episodes: Sequence[SearchEpisode], optimizer,
                *, device: torch.device, batch_episodes: int = 4,
                weights: LossWeights = LossWeights(), seed: int = 0) -> dict[str, float]:
    """Train once over one iteration's episodes, batching whole episodes."""
    model.train()
    order = np.random.default_rng(seed).permutation(len(episodes))
    sums, batches = {}, 0
    for start in range(0, len(order), batch_episodes):
        selected = [episodes[int(i)] for i in order[start:start + batch_episodes]]
        batch = AlphaZeroBatch(selected, device)
        out = model(batch.images, None, batch.interaction)
        loss, metrics = alphazero_loss(out, batch, weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batches += 1
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + value
    return {key: value / max(1, batches) for key, value in sums.items()}
