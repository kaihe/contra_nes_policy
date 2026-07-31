"""The stored episode, its returns and advantages, and how it is fed back to PPO.

One :class:`Episode` is a *complete recurrent trajectory*: every observation the
policy saw, every action it sampled, the log-probability and value it reported at the
time, the reward, and the outcome. Nothing is stored per-transition-in-isolation —
the recurrent core makes transition ``t`` meaningless without ``0..t-1``, so the
episode is the atom of both storage and optimisation.

Credit assignment
-----------------
The default objective is undiscounted and unbootstrapped: ``gamma=1``,
``gae_lambda=1``, reward only at the terminal step. That collapses to something worth
stating plainly, because it is what makes a sparse binary signal survive a 500-step
boss episode::

    return[t]    = 1 if the episode succeeded else 0      (every t)
    advantage[t] = return[t] - value[t]

At ``gamma=0.99`` a success 180 decisions away is worth 0.16 at the start of a boss
episode, and a low ``lambda`` erases it the same way. The variance reduction that a
discount would normally buy comes from the critic instead: every state in a winning
episode targets 1, every state in a losing one targets 0, and the advantage is
"did this episode do better than this state's prior odds".

Terminals and bootstrapping
---------------------------
Success, death and **budget exhaustion** are all true terminals with no bootstrap —
running out of the task's real evaluation budget is a failure worth exactly 0, not a
state whose value should be estimated. The only thing that may bootstrap is an
*artificial* cut: the collector giving up on an over-long episode
(``max_episode_steps``). Such an episode carries ``terminal=False`` and the critic's
estimate of the state it was cut at, and :func:`compute_gae` uses it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from contra_policy.dataset import FAMILIES
from contra_policy.goal import goal_mask, points_to_target

#: Outcomes an episode can end with. The first three are true terminals; ``truncated``
#: is the artificial collector cut and is the only one that bootstraps.
OUTCOMES = ("success", "death", "timeout", "truncated")


@dataclass
class Episode:
    """One complete rollout of one task, in the order the policy experienced it."""

    family: str
    label: str
    uid: str
    interaction: int
    #: (S, S, 3) uint8 goal frame and (S, S) uint8 goal blob — one per episode, not
    #: per step. A Contra task has exactly one goal, and materialising it per timestep
    #: would make the prompt 32x the size of the trajectory it belongs to.
    goal_image: np.ndarray
    goal_mask: np.ndarray
    #: (T, S, S, 3) uint8 — the agent view *before* each action. This dominates the
    #: rollout's memory: 196 KB per step at 256 px.
    obs: np.ndarray
    actions: np.ndarray            # (T,) int64, sampled
    prev_actions: np.ndarray       # (T,) int64, the action that produced this frame
    logprobs: np.ndarray           # (T,) float32, under the behaviour policy
    values: np.ndarray             # (T,) float32, the critic at collection time
    rewards: np.ndarray            # (T,) float32
    outcome: str
    terminal: bool                 # False only for an artificial collector cut
    bootstrap_value: float         # V(s_T); meaningful only when terminal is False
    budget: int
    expert_steps: int
    died: bool
    #: Per-step authoritative goal position(s) in PPU coordinates, read from live RAM
    #: at the frame the policy saw, and whether any was visible. Privileged *labels*
    #: for the optional grounding auxiliary — never policy input, never reward.
    goal_points: List[List[List[int]]] = field(default_factory=list)
    goal_visible: Optional[np.ndarray] = None      # (T,) bool

    # filled by compute_returns
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return int(len(self.actions))

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def family_index(self) -> int:
        return FAMILIES.index(self.family)

    def nbytes(self) -> int:
        return int(self.obs.nbytes + self.goal_image.nbytes + self.goal_mask.nbytes)


# ── returns and advantages ────────────────────────────────────────────────────

def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float,
                terminal: bool = True, bootstrap_value: float = 0.0
                ) -> Tuple[np.ndarray, np.ndarray]:
    """``(advantages, returns)`` for one episode.

    There are no intra-episode ``done`` flags to handle: an :class:`Episode` is by
    construction a single uninterrupted trajectory, and the only question at its end
    is whether the next state is worth 0 (a true terminal) or ``bootstrap_value`` (an
    artificial cut).
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    t_total = len(rewards)
    advantages = np.zeros(t_total, dtype=np.float64)
    next_value = 0.0 if terminal else float(bootstrap_value)
    last = 0.0
    for t in range(t_total - 1, -1, -1):
        v_next = values[t + 1] if t + 1 < t_total else next_value
        delta = rewards[t] + gamma * v_next - values[t]
        last = delta + gamma * lam * last
        advantages[t] = last
    return advantages.astype(np.float32), (advantages + values).astype(np.float32)


def compute_returns(episodes: Sequence[Episode], gamma: float, lam: float) -> None:
    """Fill ``advantages`` / ``returns`` on every episode, in place."""
    for ep in episodes:
        ep.advantages, ep.returns = compute_gae(
            ep.rewards, ep.values, gamma, lam,
            terminal=ep.terminal, bootstrap_value=ep.bootstrap_value)


def normalize_advantages(episodes: Sequence[Episode], eps: float = 1e-8) -> Tuple[float, float]:
    """Standardise advantages over the whole rollout batch, in place.

    Across the batch rather than per minibatch: with a binary episodic return the
    per-minibatch mean is dominated by how many successes happened to land in it, so
    normalising there would rescale the gradient by the luck of the shuffle.
    """
    flat = np.concatenate([ep.advantages for ep in episodes]) if episodes else np.zeros(0)
    if flat.size == 0:
        return 0.0, 1.0
    mean, std = float(flat.mean()), float(flat.std())
    for ep in episodes:
        ep.advantages = ((ep.advantages - mean) / (std + eps)).astype(np.float32)
    return mean, std


# ── minibatching, without breaking time ───────────────────────────────────────

def iter_minibatches(episodes: Sequence[Episode], minibatch_episodes: int,
                     rng: np.random.Generator) -> Iterator[List[Episode]]:
    """Shuffle **episodes** into minibatches — never the transitions inside them.

    A transition-level shuffle is the standard PPO minibatch and is exactly wrong
    here: the recurrent core would be handed step 41 of one episode next to step 7 of
    another with a memory belonging to neither. Episodes are the shuffle unit, and
    each one is replayed in order inside :func:`iter_chunks`.
    """
    order = rng.permutation(len(episodes))
    for start in range(0, len(order), minibatch_episodes):
        yield [episodes[i] for i in order[start:start + minibatch_episodes]]


def iter_chunks(episodes: Sequence[Episode], seq_len: int) -> Iterator[Tuple[int, int]]:
    """``(start, end)`` step ranges covering the longest episode in the minibatch.

    Chunks are consecutive and start at 0, so replaying them in order with carried
    memory reconstructs each episode's recurrent state exactly — no stored memory and
    no burn-in approximation. Episodes shorter than the longest one are masked off in
    the tail chunks by :func:`build_chunk`.
    """
    longest = max((len(ep) for ep in episodes), default=0)
    for start in range(0, longest, seq_len):
        yield start, min(start + seq_len, longest)


def build_chunk(episodes: Sequence[Episode], start: int, end: int, *,
                device: torch.device, aux_size: int = 0, sigma_px: float = 12.0,
                first: bool = False) -> Dict:
    """Steps ``[start, end)`` of each episode as one padded ``(B, L, …)`` batch.

    The returned dict is split in two: ``model`` is exactly what
    :meth:`CrossViewContraRocket.forward` consumes, and the rest are PPO targets.

    ``prev_action_dropout`` is all zeros, unconditionally. The evaluated
    configuration forces the learned "unknown" previous-action embedding
    (``--no-prev-action``), and optimising under a different input than the one the
    policy will be evaluated with would make every ratio in the PPO objective a ratio
    between two different policies.

    ``goal_heatmap`` / ``exist`` / ``point`` are only built when ``aux_size > 0``;
    rendering a Gaussian per step is wasted work when the grounding auxiliary is off.
    """
    b, length = len(episodes), end - start
    s = episodes[0].obs.shape[1]

    image = np.zeros((b, length, s, s, 3), dtype=np.uint8)
    goal_image = np.zeros((b, s, s, 3), dtype=np.uint8)
    goal_msk = np.zeros((b, s, s), dtype=np.uint8)
    interaction = np.zeros((b, length), dtype=np.int64)
    prev_action = np.zeros((b, length), dtype=np.int64)
    action = np.zeros((b, length), dtype=np.int64)
    old_logprob = np.zeros((b, length), dtype=np.float32)
    old_value = np.zeros((b, length), dtype=np.float32)
    advantage = np.zeros((b, length), dtype=np.float32)
    returns = np.zeros((b, length), dtype=np.float32)
    mask = np.zeros((b, length), dtype=np.float32)
    family = np.zeros(b, dtype=np.int64)

    want_aux = aux_size > 0
    heatmap = np.zeros((b, length, aux_size, aux_size), dtype=np.float32) if want_aux else None
    exist = np.zeros((b, length), dtype=np.float32) if want_aux else None
    point = np.zeros((b, length, 2), dtype=np.float32) if want_aux else None

    for i, ep in enumerate(episodes):
        goal_image[i] = ep.goal_image
        goal_msk[i] = ep.goal_mask
        family[i] = ep.family_index
        n = max(0, min(end, len(ep)) - start)
        if n <= 0:
            # Past the end of a short episode: the interaction id must still be a valid
            # embedding row, and the mask is zero, so nothing here reaches a loss.
            interaction[i, :] = ep.interaction
            continue
        sl = slice(start, start + n)
        image[i, :n] = ep.obs[sl]
        interaction[i, :] = ep.interaction
        prev_action[i, :n] = ep.prev_actions[sl]
        action[i, :n] = ep.actions[sl]
        old_logprob[i, :n] = ep.logprobs[sl]
        old_value[i, :n] = ep.values[sl]
        # None before compute_returns has run — the case the behaviour-stat refresh
        # pass is in, which needs the observations but none of the PPO targets.
        if ep.advantages is not None:
            advantage[i, :n] = ep.advantages[sl]
        if ep.returns is not None:
            returns[i, :n] = ep.returns[sl]
        mask[i, :n] = 1.0
        if want_aux and ep.goal_visible is not None:
            for k in range(n):
                j = start + k
                if not ep.goal_visible[j] or not ep.goal_points[j]:
                    continue
                exist[i, k] = 1.0
                point[i, k], _bbox = points_to_target(ep.goal_points[j])
                heatmap[i, k] = goal_mask(ep.goal_points[j], aux_size, sigma_px)

    def t(arr, dtype):
        return torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)

    model_input = {
        "image": t(image, torch.uint8),
        "cross_view": {
            "cross_view_image": t(goal_image, torch.uint8),
            "cross_view_obj_mask": t(goal_msk, torch.uint8),
            "cross_view_obj_id": t(interaction, torch.long),
        },
        "prev_action": t(prev_action, torch.long),
        # Always the learned "unknown" embedding — see the docstring.
        "prev_action_dropout": torch.zeros((b, length), dtype=torch.float32, device=device),
    }
    if first:
        flag = torch.zeros((b, length), dtype=torch.bool, device=device)
        flag[:, 0] = True
        model_input["first"] = flag

    batch = {
        "model": model_input,
        "action": t(action, torch.long),
        "old_logprob": t(old_logprob, torch.float32),
        "old_value": t(old_value, torch.float32),
        "advantage": t(advantage, torch.float32),
        "returns": t(returns, torch.float32),
        "mask": t(mask, torch.float32),
        "family": t(family, torch.long),
    }
    if want_aux:
        batch["goal_heatmap"] = t(heatmap, torch.float32)
        batch["exist"] = t(exist, torch.float32)
        batch["point"] = t(point, torch.float32)
    return batch


# ── rollout-level statistics ──────────────────────────────────────────────────

def rollout_stats(episodes: Sequence[Episode]) -> Dict[str, float]:
    """Outcome rates, per-family and per-label completion, episode *and* step counts.

    Both counts are reported because they diverge sharply: a boss episode runs several
    hundred decisions and an item episode a few dozen, so a family that is 15% of
    episode starts can be 30% of the transitions the gradient actually sees.
    """
    out: Dict[str, float] = {}
    if not episodes:
        return out
    n = len(episodes)
    steps = sum(len(ep) for ep in episodes)
    out["episodes"] = float(n)
    out["steps"] = float(steps)
    out["mean_episode_len"] = steps / n
    out["reward"] = float(np.mean([ep.rewards.sum() for ep in episodes]))
    out["return"] = float(np.mean([ep.returns[0] if ep.returns is not None else
                                   ep.rewards.sum() for ep in episodes]))
    for name in OUTCOMES:
        out[name] = sum(ep.outcome == name for ep in episodes) / n
    out["completion"] = out["success"]
    # `death` is the outcome; `died` is whether the player died at all, which can
    # differ if the collector is ever configured to continue after a death.
    out["died"] = sum(ep.died for ep in episodes) / n

    by_family: Dict[str, List[Episode]] = {}
    by_label: Dict[Tuple[str, str], List[Episode]] = {}
    for ep in episodes:
        by_family.setdefault(ep.family, []).append(ep)
        by_label.setdefault((ep.family, ep.label), []).append(ep)
    for family, group in by_family.items():
        out[f"{family}/completion"] = sum(e.success for e in group) / len(group)
        out[f"{family}/death"] = sum(e.outcome == "death" for e in group) / len(group)
        out[f"{family}/episodes"] = float(len(group))
        out[f"{family}/steps"] = float(sum(len(e) for e in group))
        out[f"{family}/episode_share"] = len(group) / n
        out[f"{family}/step_share"] = sum(len(e) for e in group) / max(1, steps)
    for (family, label), group in by_label.items():
        out[f"{family}.{label}/completion"] = sum(e.success for e in group) / len(group)
        out[f"{family}.{label}/episodes"] = float(len(group))
    out["macro_completion"] = float(np.mean(
        [sum(e.success for e in g) / len(g) for g in by_label.values()]))
    return out
