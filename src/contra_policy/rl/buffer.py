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
    #: The task's fine-grained class ("red_turret", "pick_laser"). Carried so the
    #: difficulty sampler can pool 495 tasks per label into one usable estimate — see
    #: :class:`~contra_policy.rl.tasks.DifficultyTracker`.
    task_label: str = ""

    def __len__(self) -> int:
        return int(self.frames.shape[0])


@dataclass
class EpisodeOutcome:
    """What reporting needs from an episode, without the frames.

    An :class:`Episode` at 256px carries ``T × 256 × 256 × 3`` bytes — ~18 MB at the
    measured 96-step mean. Success rates are reported over *everything* rolled including
    the groups filtering discarded, so holding whole Episodes just to count them cost
    ~9 GB per update and took the VM down (run ``2026-08-02/11-48-03``). This is the
    same information at ~100 bytes.
    """

    family: str
    outcome: str
    reward: float
    n_steps: int

    @classmethod
    def of(cls, e: "Episode") -> "EpisodeOutcome":
        return cls(family=e.family, outcome=e.outcome, reward=e.reward, n_steps=len(e))

    def __len__(self) -> int:
        return self.n_steps


def group_advantages(rewards: Sequence[float], group_ids: Sequence[int],
                     normalise: bool = True, eps: float = 1e-4
                     ) -> tuple[np.ndarray, Dict[str, float]]:
    """Per-episode advantage: own reward minus its group's mean.

    ``normalise`` divides by the group's standard deviation (GRPO as published).
    Turning it off gives the RLOO-style unnormalised difference, which some find more
    stable when groups are small — kept as a switch rather than a fork.

    A group whose members all share a reward has zero spread and therefore zero
    advantage: it contributes no gradient. Known in the RLVR literature as a
    *zero-variance* group; the mitigation is *dynamic sampling* / *prompt filtering* —
    see :func:`filter_groups`.

    Both tails do it. ``P(zero variance) = p**G + (1-p)**G``, so a task the policy
    always fails is as useless as one it always solves: boss at 3.5% gives 87% at G=4,
    and `kill` on *train* tasks measured ~92% success, giving 66%. The returned stats
    make the rate visible per update rather than inferred from a flat curve.
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
        "zero_variance_group_frac": float(degenerate) / max(1, len(groups)),
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
                 device: Optional[torch.device] = None, layout: str = "padded"):
        b = len(episodes)
        self.layout = layout
        self.n_episodes = b
        lengths = np.asarray([len(e) for e in episodes], dtype=np.int64)
        if layout == "varlen":
            image = np.concatenate([e.frames for e in episodes], axis=0)
            action = np.concatenate([e.actions for e in episodes], axis=0)
            logprob = np.concatenate([e.logprobs for e in episodes], axis=0)
            mask = np.ones(int(lengths.sum()), dtype=np.float32)
            self.episode_index = torch.from_numpy(
                np.repeat(np.arange(b, dtype=np.int64), lengths))
        elif layout == "padded":
            t = int(lengths.max())
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
            self.episode_index = None
        else:
            raise ValueError(f"unknown attention layout {layout!r}")

        self.image = torch.from_numpy(image)
        self.goal_image = torch.from_numpy(
            np.stack([e.goal_image for e in episodes]))
        self.interaction = torch.tensor([e.interaction for e in episodes],
                                        dtype=torch.long)
        self.action = torch.from_numpy(action)
        self.old_logprob = torch.from_numpy(logprob)
        self.mask = torch.from_numpy(mask)
        self.seq_len = torch.from_numpy(lengths)
        # One scalar per episode, broadcast over its steps at use. Broadcasting rather
        # than materialising keeps it obvious that there is no per-step credit here.
        self.advantage = torch.from_numpy(advantages.astype(np.float32))
        self.family = [e.family for e in episodes]
        self.reward = torch.tensor([e.reward for e in episodes], dtype=torch.float32)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device) -> "GroupBatch":
        for k in ("image", "goal_image", "interaction", "action", "old_logprob",
                  "mask", "advantage", "reward", "seq_len", "episode_index"):
            value = getattr(self, k)
            if value is not None:
                setattr(self, k, value.to(device, non_blocking=True))
        return self

    @property
    def steps(self) -> int:
        return int(self.mask.sum())

    def __len__(self) -> int:
        return self.n_episodes


def iter_minibatches(episodes: Sequence[Episode], advantages: np.ndarray,
                     minibatch_episodes: int, rng: np.random.Generator,
                     device: Optional[torch.device] = None,
                     layout: str = "padded"):
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
        yield GroupBatch([episodes[i] for i in idx], advantages[idx], device=device,
                         layout=layout)


def filter_groups(episodes: Sequence[Episode], eps: float = 1e-4
                  ) -> tuple[List[Episode], Dict[str, float]]:
    """Drop groups whose members all got the same reward — *group filtering*.

    **What it is.** GRPO's advantage is ``(r_i − mean(r_group)) / std(r_group)``. When a
    group's rewards are all equal every numerator is zero, so every advantage is zero and
    the group moves the policy not at all. It still cost G rollouts, and rollouts are the
    expensive part of the loop. Filtering removes them before the update so the gradient
    is computed only over groups that carry signal.

    **Why it is not merely an optimisation.** Keeping them changes the *scale* of the
    update, not just its cost: the mean over a batch that is 70% zeros is 0.3x the mean
    over the survivors, so the effective learning rate silently tracks how hard the
    current task mix happens to be. Filtering makes the step size mean the same thing
    from update to update.

    **What it is not.** It does not make an impossible task learnable — a boss group that
    always fails is dropped, not solved. It saves the *update* from being diluted; the
    wasted rollouts are recovered only by sampling better tasks (a curriculum) or by
    oversampling until enough groups survive (:meth:`GRPOTrainer.collect_filtered`).

    Returns the surviving episodes and stats including the fraction dropped — which is
    the number to watch, because a high value means the rollout budget is going
    somewhere useless even though the loss looks healthy.
    """
    by_group: Dict[int, List[Episode]] = {}
    for e in episodes:
        by_group.setdefault(e.group_id, []).append(e)

    kept: List[Episode] = []
    n_zero = 0
    for gid, members in by_group.items():
        r = np.asarray([m.reward for m in members], dtype=np.float64)
        if len(members) < 2 or r.std() < eps:
            n_zero += 1
            continue
        kept.extend(members)

    n_groups = len(by_group)
    return kept, {
        "groups_collected": float(n_groups),
        "groups_kept": float(n_groups - n_zero),
        "zero_variance_group_frac": float(n_zero) / max(1, n_groups),
        "episodes_discarded": float(len(episodes) - len(kept)),
    }
