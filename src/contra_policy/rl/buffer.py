"""Episodes → group-relative advantages → padded batches.

The contract between generation and the objective. Everything the loss sees comes from
here, and nothing here knows what a policy is.

**Why there is no critic.** With ``gamma=1.0``, no bootstrapping and a binary episodic
reward, ``V(s)`` is exactly "probability of success from here" — the hardest possible
regression target and the one the previous PPO run only partly learned
(``explained_variance`` ~0.33 after starting negative, so two thirds of the return
variance went into the advantages as noise). GRPO estimates the same baseline
empirically: G rollouts of one task, advantage = own return minus the group's.

Every step of an episode gets that same scalar. There is no per-step credit assignment,
which is honest — with a terminal-only reward there is no information about *which*
action mattered, and GAE's apparent per-step signal was interpolating a critic that did
not know either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch


@dataclass
class Episode:
    """One complete rollout. Frames are the encoder's input, not its output — the
    trainer re-encodes, because the encoder may be being fine-tuned."""

    task_uid: str
    family: str
    group_id: int
    frames: np.ndarray          # (T, S, S, 3) uint8
    goal_image: np.ndarray      # (S, S, 3) uint8
    interaction: int
    actions: np.ndarray         # (T,) int64 — what was sampled
    logprobs: np.ndarray        # (T,) float32 — under the behaviour policy
    reward: float               # terminal, binary
    outcome: str                # success | death | timeout
    goal_heatmap: Optional[np.ndarray] = None   # (T, A, A) float32, privileged aux

    def __len__(self) -> int:
        return int(self.frames.shape[0])


def group_advantages(rewards: Sequence[float], group_ids: Sequence[int],
                     normalise: bool = True, eps: float = 1e-4
                     ) -> tuple[np.ndarray, Dict[str, float]]:
    """Per-episode advantage: own reward minus its group's mean.

    ``normalise`` divides by the group's standard deviation (GRPO as published).
    Turning it off gives the RLOO-style unnormalised difference, which some find more
    stable when groups are small — kept as a switch rather than a fork.

    A group whose members all share a reward has zero spread and therefore zero
    advantage: it contributes no gradient. That is not a bug to paper over — it is what
    "nothing to learn from this task right now" looks like, and at boss's ~3.5% success
    it will be most boss groups. The returned stats make the rate visible.
    """
    r = np.asarray(rewards, dtype=np.float64)
    g = np.asarray(group_ids)
    adv = np.zeros_like(r)
    degenerate = 0
    groups = np.unique(g)
    for gid in groups:
        sel = g == gid
        vals = r[sel]
        centred = vals - vals.mean()
        if normalise:
            sd = vals.std()
            if sd < eps:
                degenerate += 1
                continue                      # leave at zero: no signal in this group
            centred = centred / sd
        elif np.abs(centred).max() < eps:
            degenerate += 1
        adv[sel] = centred

    stats = {
        "groups": float(len(groups)),
        # The number to watch. High means most groups are all-success or all-failure and
        # the effective batch is far smaller than it looks.
        "degenerate_group_frac": float(degenerate) / max(1, len(groups)),
        "reward_mean": float(r.mean()),
        "adv_abs_mean": float(np.abs(adv).mean()),
    }
    return adv.astype(np.float32), stats


class GroupBatch:
    """Padded tensors for one optimisation step, plus the mask everything reduces over.

    Shapes mirror the BC path (`pad_episodes`) on purpose, so the objective and the
    policy see the same contract in both stages.
    """

    def __init__(self, episodes: Sequence[Episode], advantages: np.ndarray,
                 device: Optional[torch.device] = None):
        b = len(episodes)
        t = max(len(e) for e in episodes)
        s = episodes[0].frames.shape[1]

        image = np.zeros((b, t, s, s, 3), dtype=np.uint8)
        action = np.zeros((b, t), dtype=np.int64)
        logprob = np.zeros((b, t), dtype=np.float32)
        mask = np.zeros((b, t), dtype=np.float32)
        for i, e in enumerate(episodes):
            n = len(e)
            image[i, :n] = e.frames
            action[i, :n] = e.actions
            logprob[i, :n] = e.logprobs
            mask[i, :n] = 1.0

        self.image = torch.from_numpy(image)
        self.goal_image = torch.from_numpy(
            np.stack([e.goal_image for e in episodes]))
        self.interaction = torch.tensor([e.interaction for e in episodes],
                                        dtype=torch.long)
        self.action = torch.from_numpy(action)
        self.old_logprob = torch.from_numpy(logprob)
        self.mask = torch.from_numpy(mask)
        # One scalar per episode, broadcast over its steps at use. Broadcasting rather
        # than materialising keeps it obvious that there is no per-step credit here.
        self.advantage = torch.from_numpy(advantages.astype(np.float32))
        self.family = [e.family for e in episodes]
        self.reward = torch.tensor([e.reward for e in episodes], dtype=torch.float32)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "GroupBatch":
        for k in ("image", "goal_image", "interaction", "action", "old_logprob",
                  "mask", "advantage", "reward"):
            setattr(self, k, getattr(self, k).to(device, non_blocking=True))
        return self

    @property
    def steps(self) -> int:
        return int(self.mask.sum())

    def __len__(self) -> int:
        return int(self.mask.shape[0])


def iter_minibatches(episodes: Sequence[Episode], advantages: np.ndarray,
                     minibatch_episodes: int, rng: np.random.Generator,
                     device: Optional[torch.device] = None):
    """Shuffle **episodes**, never steps.

    An episode is one sequence to the causal core; splitting it across minibatches would
    hand the model a fragment whose prefix it never saw. Groups are deliberately *not*
    kept together — advantages are already computed, so a minibatch needs no group
    structure, and mixing them avoids a batch that is entirely one task.

    Episodes are sorted by length within each minibatch so padding tracks the batch
    maximum rather than the global one; at a 100-frame mean against 1038-frame budgets
    that is most of the compute.
    """
    order = rng.permutation(len(episodes))
    for start in range(0, len(order), minibatch_episodes):
        idx = order[start:start + minibatch_episodes]
        idx = idx[np.argsort([len(episodes[i]) for i in idx], kind="stable")]
        yield GroupBatch([episodes[i] for i in idx], advantages[idx], device=device)
