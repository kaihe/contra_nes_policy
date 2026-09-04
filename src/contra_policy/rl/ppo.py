"""Binary-return PPO with a learned causal-history value baseline.

The reward stays terminal and verifiable. GAE does not invent intermediate rewards; it
uses changes in the old critic's prediction to distribute a conditional baseline across
timesteps. Experiment 0028 tests whether that estimate is better than GRPO's one scalar
per trajectory, and the critic must first beat a constant predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from contra_policy.rl.grpo import masked_mean


@dataclass
class PPOConfig:
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    kl_coef: float = 0.02
    entropy_coef: float = 0.01
    target_kl: float = 0.02
    max_grad_norm: float = 1.0
    temperature: float = 1.0


def gae(reward: float, values: np.ndarray, gamma: float = 1.0,
        lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Return unnormalised GAE and lambda-return targets for one complete episode."""
    v = np.asarray(values, dtype=np.float32)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError("values must be a non-empty 1-D episode")
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError("gamma and lambda must lie in [0, 1]")
    rewards = np.zeros_like(v)
    rewards[-1] = float(reward)
    next_v = np.concatenate([v[1:], np.zeros(1, dtype=np.float32)])
    delta = rewards + gamma * next_v - v
    advantage = np.zeros_like(v)
    carry = 0.0
    for t in range(len(v) - 1, -1, -1):
        carry = float(delta[t]) + gamma * lam * carry
        advantage[t] = carry
    return advantage, advantage + v


def explained_variance(target: np.ndarray, pred: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    var = float(np.var(target))
    return 0.0 if var < 1e-12 else 1.0 - float(np.var(target - pred)) / var


def ppo_loss(logits: torch.Tensor, values: torch.Tensor, batch, cfg: PPOConfig,
             ref_logits: Optional[torch.Tensor] = None
             ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Clipped PPO actor loss plus lambda-return value regression and reference KL."""
    if cfg.temperature <= 0:
        raise ValueError("PPO requires a positive sampling temperature")
    mask = batch.mask
    logp_all = F.log_softmax(logits.float() / cfg.temperature, dim=-1)
    logp = logp_all.gather(-1, batch.action.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(logp - batch.old_logprob)
    unclipped = ratio * batch.advantage
    clipped = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * batch.advantage
    policy_loss = -masked_mean(torch.min(unclipped, clipped), mask)

    value_prob = values.float().sigmoid()
    value_loss = masked_mean((value_prob - batch.value_target) ** 2, mask)
    probs = logp_all.exp()
    entropy = -(probs * logp_all).sum(-1)
    loss = (policy_loss + cfg.value_coef * value_loss
            - cfg.entropy_coef * masked_mean(entropy, mask))

    metrics: Dict[str, torch.Tensor] = {"value_loss": value_loss.detach()}
    if ref_logits is not None and cfg.kl_coef > 0:
        ref_logp = F.log_softmax(ref_logits.float() / cfg.temperature, dim=-1)
        kl_ref = (probs * (logp_all - ref_logp)).sum(-1)
        kl_ref_mean = masked_mean(kl_ref, mask)
        metrics["kl_ref"] = kl_ref_mean.detach()
        loss = loss + cfg.kl_coef * kl_ref_mean

    with torch.no_grad():
        d = batch.old_logprob - logp
        metrics.update({
            "policy_loss": policy_loss.detach(),
            "entropy": masked_mean(entropy, mask),
            "approx_kl": masked_mean(torch.exp(-d) - 1 + d, mask),
            "clip_frac": masked_mean(
                ((ratio - 1).abs() > cfg.clip_ratio).float(), mask),
            "ratio_mean": masked_mean(ratio, mask),
            "value_brier": masked_mean(
                (value_prob - batch.reward.unsqueeze(1)) ** 2, mask),
        })
    metrics["loss"] = loss.detach()
    return loss, metrics
