"""The GRPO objective: a clipped ratio on group-relative advantages, plus a reference KL.

Three terms, and one of them is not optional here:

``policy``     the PPO clipped surrogate, unchanged — GRPO differs from PPO in where the
               advantage comes from, not in how it is used.
``kl_ref``     KL against a frozen copy of the BC policy. **Measured need**: the previous
               PPO run left this at zero and `item` regressed 76.5% → 71.1% over 500
               updates while boss improved. RL on a verifiable reward will happily trade
               away a family the reward does not mention.
``entropy``    the usual bonus. Watched rather than trusted: entropy collapsed 0.24 →
               0.12 in the last 120 updates of the previous run, which is the signature
               of falling back onto the action prior.

What is absent: no value loss, no GAE, no bootstrapping. The advantage is one scalar per
episode from :func:`~contra_policy.rl.buffer.group_advantages`, broadcast across its
steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class GRPOConfig:
    clip_ratio: float = 0.2
    #: Weight on KL(pi || pi_bc). Zero reproduces the forgetting measured last time; the
    #: RLHF default is a small positive number and that is what this is.
    kl_coef: float = 0.02
    entropy_coef: float = 0.01
    #: Stop the update early once the batch's mean KL to the *behaviour* policy exceeds
    #: this. Distinct from `kl_coef`, which pulls towards the frozen BC reference.
    target_kl: float = 0.02
    max_grad_norm: float = 1.0


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def grpo_loss(logits: torch.Tensor, batch, cfg: GRPOConfig,
              ref_logits: Optional[torch.Tensor] = None
              ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """``logits`` (B, T, A) from the current policy; ``batch`` a
    :class:`~contra_policy.rl.buffer.GroupBatch`.

    ``ref_logits`` are the frozen BC policy's, on the same frames. Omitting them drops
    the KL term — permitted so an ablation can be run, not because it is a sensible
    default.
    """
    mask = batch.mask
    logp_all = F.log_softmax(logits.float(), dim=-1)
    logp = logp_all.gather(-1, batch.action.unsqueeze(-1)).squeeze(-1)

    ratio = torch.exp(logp - batch.old_logprob)
    # One advantage per episode, broadcast over its steps: a terminal reward carries no
    # information about which step earned it, and pretending otherwise is what the
    # critic was doing.
    if getattr(batch, "episode_index", None) is not None:
        adv = batch.advantage[batch.episode_index]
    else:
        adv = batch.advantage.unsqueeze(-1).expand_as(ratio)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv
    policy_loss = -masked_mean(torch.min(unclipped, clipped), mask)

    probs = logp_all.exp()
    entropy = -(probs * logp_all).sum(-1)
    loss = policy_loss - cfg.entropy_coef * masked_mean(entropy, mask)

    metrics: Dict[str, torch.Tensor] = {}
    if ref_logits is not None and cfg.kl_coef > 0:
        ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
        # Forward KL(pi || pi_ref) summed over actions — the full-distribution form, not
        # the sampled-action estimate, because we already have every logit and the exact
        # value is cheaper to interpret.
        kl_ref = (probs * (logp_all - ref_logp)).sum(-1)
        kl_ref_m = masked_mean(kl_ref, mask)
        loss = loss + cfg.kl_coef * kl_ref_m
        metrics["kl_ref"] = kl_ref_m.detach()

    with torch.no_grad():
        # k3 estimator of KL to the *behaviour* policy: unbiased and non-negative, unlike
        # the naive mean of (old - new) which is noisy around zero and can go negative.
        d = batch.old_logprob - logp
        approx_kl = masked_mean(torch.exp(d) - 1 - d, mask)
        metrics.update({
            "policy_loss": policy_loss.detach(),
            "entropy": masked_mean(entropy, mask),
            "approx_kl": approx_kl,
            "clip_frac": masked_mean(
                ((ratio - 1.0).abs() > cfg.clip_ratio).float(), mask),
            "ratio_mean": masked_mean(ratio, mask),
        })
    metrics["loss"] = loss.detach()
    return loss, metrics
