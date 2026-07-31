"""The clipped PPO objective, plus the auxiliaries that protect the BC policy.

Everything here operates on **one chunk** of a recurrent minibatch — a ``(B, L)``
block of consecutive steps from ``B`` episodes, with a validity mask for the episodes
that already ended. The chunk is the unit because the trainer replays each episode in
order with carried memory and backprops chunk by chunk, so the loss must be reducible
by summation over chunks with no cross-chunk term.

Metrics are returned as **sums over valid steps** rather than means, so the trainer
can divide once at the end and get the same number a single large batch would give.
Averaging per-chunk means instead would silently weight a 4-step tail chunk the same
as a full 32-step one.

Reward shaping is deliberately absent. If credit assignment turns out to be
inadequate, the only shaping that leaves the optimal policy unchanged is
potential-based::

    r_shaped = r_task + beta * (gamma * potential(s') - potential(s))

with ``potential`` coming from an authoritative progress API in ``contra_nes_data``
and forced to zero at every terminal state so the telescoping sum cancels. No such
API exists today (``TaskMaker`` defines ``goal_reached`` but nothing continuous), so
shaping is not implemented rather than invented locally — see the run report.

Never rewarded, under any configuration: predicted heatmap confidence, agreement with
the expert action, ``bc_acc``, staying alive per step, moving right, or firing. The
heatmap and BC objectives appear below as *auxiliary losses on privileged labels*,
which is a different thing from an environment reward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from contra_policy.loss import GoalHeatmapLoss


@dataclass
class PPOConfig:
    """Every PPO knob. All of these are configuration; none is a literal in the code."""

    learning_rate: float = 1.0e-5
    gamma: float = 1.0
    gae_lambda: float = 1.0
    clip_ratio: float = 0.1
    value_coef: float = 0.5
    value_clip: float = 0.2          # 0 disables value clipping
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    ppo_epochs: int = 2
    minibatch_episodes: int = 4
    seq_len: int = 32
    normalize_advantages: bool = True
    #: Coefficient on KL(pi || pi_bc) against a frozen copy of the initialisation.
    bc_kl_coef: float = 0.0
    #: Coefficient on continued goal-heatmap supervision over on-policy frames.
    heatmap_coef: float = 0.0
    heatmap_pos_weight: float = 10.0


def _masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum()


def policy_loss(logprob: torch.Tensor, old_logprob: torch.Tensor,
                advantage: torch.Tensor, mask: torch.Tensor, clip_ratio: float
                ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Clipped surrogate, summed over valid steps.

    ``approx_kl`` is Schulman's low-variance estimator ``E[(r - 1) - log r]``, which
    is non-negative and much better behaved than ``E[-log r]`` when the ratio is close
    to 1 — the regime the whole update is supposed to stay in.
    """
    log_ratio = logprob - old_logprob
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
    loss = -_masked_sum(torch.min(unclipped, clipped), mask)
    with torch.no_grad():
        metrics = {
            "approx_kl": _masked_sum((ratio - 1.0) - log_ratio, mask),
            "clip_frac": _masked_sum(
                ((ratio - 1.0).abs() > clip_ratio).float(), mask),
            "ratio": _masked_sum(ratio, mask),
        }
    return loss, metrics


def value_loss(vpred: torch.Tensor, old_value: torch.Tensor, returns: torch.Tensor,
               mask: torch.Tensor, clip: float) -> torch.Tensor:
    """Squared error, optionally clipped to a trust region around the old value.

    Value clipping matters more than usual here: the critic starts from a head that
    received **no gradient at all** during behaviour cloning, so its first predictions
    are an untrained linear readout and the unclipped error can be large enough to
    dominate the policy term on the very first updates.
    """
    unclipped = (vpred - returns) ** 2
    if clip <= 0:
        return 0.5 * _masked_sum(unclipped, mask)
    v_clipped = old_value + torch.clamp(vpred - old_value, -clip, clip)
    clipped = (v_clipped - returns) ** 2
    return 0.5 * _masked_sum(torch.max(unclipped, clipped), mask)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1)


def reference_kl(logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """``KL(pi_current || pi_bc)`` per step, exactly (21 actions — no sampling needed).

    The direction is the standard "stay near the reference" penalty: it is finite only
    where the current policy puts mass, so it discourages the policy from moving
    probability onto actions the BC policy considers implausible without forcing it to
    keep every action the BC policy liked.
    """
    logp = torch.log_softmax(logits, dim=-1)
    ref_logp = torch.log_softmax(ref_logits, dim=-1)
    return (logp.exp() * (logp - ref_logp)).sum(-1)


class PPOObjective:
    """Assembles the chunk loss and its metric sums."""

    def __init__(self, cfg: PPOConfig):
        self.cfg = cfg
        self.heatmap = GoalHeatmapLoss(1.0, cfg.heatmap_pos_weight)

    def __call__(self, latents: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor],
                 ref_logits: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        cfg = self.cfg
        mask = batch["mask"]
        count = mask.sum()

        logits = latents["pi_logits"].float()
        vpred = latents["vpred"].squeeze(-1).float()
        logprob = torch.log_softmax(logits, dim=-1).gather(
            -1, batch["action"].unsqueeze(-1)).squeeze(-1)

        pi_loss, metrics = policy_loss(logprob, batch["old_logprob"], batch["advantage"],
                                       mask, cfg.clip_ratio)
        v_loss = value_loss(vpred, batch["old_value"], batch["returns"], mask,
                            cfg.value_clip)
        entropy = _masked_sum(categorical_entropy(logits), mask)

        total = pi_loss + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
        metrics.update({"policy_loss": pi_loss.detach(), "value_loss": v_loss.detach(),
                        "entropy": entropy.detach()})

        if cfg.bc_kl_coef > 0.0 and ref_logits is not None:
            bc_kl = _masked_sum(reference_kl(logits, ref_logits.float()), mask)
            total = total + cfg.bc_kl_coef * bc_kl
            metrics["bc_kl"] = bc_kl.detach()

        if cfg.heatmap_coef > 0.0 and "goal_heatmap" in batch:
            # GoalHeatmapLoss reduces as a masked *mean*; rescale to a sum so it
            # composes with the rest of this function's summed terms.
            hm_loss, hm_metrics = self.heatmap(latents, batch)
            total = total + cfg.heatmap_coef * hm_loss * count
            metrics["heatmap_loss"] = hm_loss.detach() * count
            metrics["point_err_px"] = hm_metrics["point_err_px"].detach() * count
            metrics["exist_acc"] = hm_metrics["exist_acc"].detach() * count

        return total, metrics, count


def explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """``1 - Var(returns - values) / Var(returns)``; 0 means "no better than the mean".

    Reported rather than the raw value loss because with a binary episodic return the
    value loss is bounded by 0.25 whatever the critic does, so its absolute size says
    almost nothing about whether the critic is informative.
    """
    values = np.asarray(values, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    var = returns.var()
    if var < 1e-12:
        return float("nan")
    return float(1.0 - (returns - values).var() / var)
