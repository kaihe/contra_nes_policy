"""Environment-independent MCTS with policy-guided PUCT selection.

One simulation grows at most one permanent node. From that node it samples a
temporary policy rollout to a real terminal and backs the binary result through
the permanent path. The environment and neural policy are protocols so all tree
invariants can be tested without Stable Retro or CUDA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Protocol, Sequence

import numpy as np


class Terminal(str, Enum):
    ACTIVE = "active"
    SUCCESS = "success"
    DEATH = "death"
    TIMEOUT = "timeout"

    @property
    def value01(self) -> float:
        if self is Terminal.ACTIVE:
            raise ValueError("a non-terminal state has no Monte Carlo value")
        return float(self is Terminal.SUCCESS)


@dataclass(frozen=True)
class Transition:
    """Environment result after one policy action."""

    emu_state: bytes
    observation: np.ndarray
    state_data: Any
    terminal: Terminal = Terminal.ACTIVE


@dataclass
class Edge:
    action_id: int
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    successes: int = 0
    deaths: int = 0
    timeouts: int = 0
    child: Optional["Node"] = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    emu_state: bytes
    frame_token: Any
    state_data: Any
    previous_action: int
    steps: int
    terminal: Terminal = Terminal.ACTIVE
    parent: Optional["Node"] = field(default=None, repr=False)
    incoming_action: Optional[int] = None
    expanded: bool = False
    edges: Dict[int, Edge] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchConfig:
    simulations_per_move: int = 16
    max_live_nodes: int = 2048
    c_puct: float = 1.5
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.simulations_per_move < 1:
            raise ValueError("simulations_per_move must be positive")
        if self.max_live_nodes < 2:
            raise ValueError("max_live_nodes must be at least 2")
        if self.c_puct < 0:
            raise ValueError("c_puct must be nonnegative")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class PolicyTarget:
    """Compact output at one committed decision; training is outside design 0030."""

    step: int
    previous_action: int
    legal_mask: np.ndarray
    priors: np.ndarray
    visits: np.ndarray
    probabilities: np.ndarray
    successes: np.ndarray
    deaths: np.ndarray
    timeouts: np.ndarray
    chosen_action: int


class SearchPolicy(Protocol):
    num_actions: int
    action_names: Sequence[str]

    def encode(self, observation: np.ndarray) -> Any: ...

    def priors(self, frame_tokens: Sequence[Any], previous_action: int) -> np.ndarray: ...


class SearchEnvironment(Protocol):
    def legal_mask(self, node: Node) -> np.ndarray: ...

    def step(self, node: Node, action_id: int) -> Transition: ...


class SearchTree:
    """Persistent tree for one committed episode."""

    def __init__(self, root: Node, policy: SearchPolicy, environment: SearchEnvironment,
                 config: SearchConfig = SearchConfig(), *, model_id: str = ""):
        self.root = root
        self.policy = policy
        self.environment = environment
        self.config = config
        self.model_id = model_id
        self.committed_prefix: list[Any] = []
        self.completed_simulations = 0
        self.live_nodes = 1
        self.created_nodes = 1
        self.rng = np.random.default_rng(config.seed)
        if root.terminal is Terminal.ACTIVE and not root.expanded:
            self._expand(root, self.context(root))

    def context(self, node: Node) -> list[Any]:
        suffix: list[Any] = []
        cur: Optional[Node] = node
        while cur is not None:
            suffix.append(cur.frame_token)
            cur = cur.parent
        suffix.reverse()
        return [*self.committed_prefix, *suffix]

    def _expand(self, node: Node, context: Sequence[Any]) -> None:
        if node.terminal is not Terminal.ACTIVE:
            return
        mask = np.asarray(self.environment.legal_mask(node), dtype=bool)
        priors = np.asarray(
            self.policy.priors(context, node.previous_action), dtype=np.float64)
        if mask.shape != (self.policy.num_actions,) or priors.shape != mask.shape:
            raise ValueError("legal mask and priors must match the policy action space")
        if not mask.any():
            raise RuntimeError("environment returned no legal action")
        if not np.isfinite(priors).all() or np.any(priors < 0):
            raise ValueError("policy priors must be finite and nonnegative")
        probs = np.where(mask, priors, 0.0)
        total = float(probs.sum())
        probs = probs / total if total > 0 else mask.astype(np.float64) / mask.sum()
        node.edges = {int(i): Edge(int(i), float(probs[i])) for i in np.flatnonzero(mask)}
        node.expanded = True

    def select_edge(self, node: Node) -> Edge:
        parent_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(max(1, parent_visits))

        def rank(edge: Edge) -> tuple[float, int]:
            score = (edge.mean_value
                     + self.config.c_puct * edge.prior * scale / (1 + edge.visits))
            return score, -edge.action_id

        return max(node.edges.values(), key=rank)

    def simulate(self) -> bool:
        """Run one selection/expansion/rollout/backup; return false at node limit."""
        if self.root.terminal is not Terminal.ACTIVE:
            return False
        node = self.root
        path: list[Edge] = []

        while True:
            edge = self.select_edge(node)
            path.append(edge)
            if edge.child is None:
                if self.live_nodes >= self.config.max_live_nodes:
                    return False
                transition = self.environment.step(node, edge.action_id)
                child = Node(
                    emu_state=transition.emu_state,
                    frame_token=self.policy.encode(transition.observation),
                    state_data=transition.state_data,
                    previous_action=edge.action_id,
                    steps=node.steps + 1,
                    terminal=transition.terminal,
                    parent=node,
                    incoming_action=edge.action_id,
                )
                edge.child = child
                self.live_nodes += 1
                self.created_nodes += 1
                node = child
                break
            node = edge.child
            if node.terminal is not Terminal.ACTIVE:
                break

        context = self.context(node)
        if node.terminal is Terminal.ACTIVE:
            self._expand(node, context)
            outcome = self._terminal_rollout(node, context)
        else:
            outcome = node.terminal
        self.backup(path, outcome)
        self.completed_simulations += 1
        return True

    def _terminal_rollout(self, start: Node, context: list[Any]) -> Terminal:
        node = start
        first = True
        while node.terminal is Terminal.ACTIVE:
            mask = np.asarray(self.environment.legal_mask(node), dtype=bool)
            if first:
                # Expansion already evaluated this leaf. Reuse its immutable priors
                # instead of paying for the same transformer forward twice.
                probs = np.zeros(self.policy.num_actions, dtype=np.float64)
                for action, edge in node.edges.items():
                    probs[action] = edge.prior
                first = False
            else:
                probs = np.asarray(
                    self.policy.priors(context, node.previous_action), dtype=np.float64)
            probs = np.where(mask, probs, 0.0)
            total = float(probs.sum())
            probs = probs / total if total > 0 else mask.astype(np.float64) / mask.sum()
            if self.config.temperature != 1.0:
                probs = np.power(probs, 1.0 / self.config.temperature)
                probs /= probs.sum()
            action = int(self.rng.choice(self.policy.num_actions, p=probs))
            transition = self.environment.step(node, action)
            node = Node(
                emu_state=transition.emu_state,
                frame_token=self.policy.encode(transition.observation),
                state_data=transition.state_data,
                previous_action=action,
                steps=node.steps + 1,
                terminal=transition.terminal,
            )
            context.append(node.frame_token)
        return node.terminal

    @staticmethod
    def backup(path: Sequence[Edge], outcome: Terminal) -> None:
        if outcome is Terminal.ACTIVE:
            raise ValueError("cannot back up a non-terminal outcome")
        value = outcome.value01
        for edge in path:
            edge.visits += 1
            edge.value_sum += value
            if outcome is Terminal.SUCCESS:
                edge.successes += 1
            elif outcome is Terminal.DEATH:
                edge.deaths += 1
            else:
                edge.timeouts += 1

    def search(self, simulations: Optional[int] = None) -> int:
        target = simulations or self.config.simulations_per_move
        done = 0
        while done < target and self.simulate():
            done += 1
        return done

    def root_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.zeros(self.policy.num_actions, dtype=bool)
        visits = np.zeros(self.policy.num_actions, dtype=np.int64)
        for action, edge in self.root.edges.items():
            mask[action] = True
            visits[action] = edge.visits
        total = int(visits.sum())
        if total == 0:
            raise RuntimeError("cannot form a policy target before a simulation")
        return mask, visits, visits.astype(np.float64) / total

    def commit(self) -> PolicyTarget:
        """Choose the most-visited root edge, preserve its subtree, and re-root."""
        mask, visits, probabilities = self.root_target()
        priors = np.zeros(self.policy.num_actions, dtype=np.float64)
        successes = np.zeros(self.policy.num_actions, dtype=np.int64)
        deaths = np.zeros(self.policy.num_actions, dtype=np.int64)
        timeouts = np.zeros(self.policy.num_actions, dtype=np.int64)
        for action, edge in self.root.edges.items():
            priors[action] = edge.prior
            successes[action] = edge.successes
            deaths[action] = edge.deaths
            timeouts[action] = edge.timeouts
        chosen = min(
            (edge for edge in self.root.edges.values() if edge.visits == visits.max()),
            key=lambda edge: edge.action_id,
        )
        if chosen.child is None:
            raise RuntimeError("the most-visited edge has no child")
        old_root = self.root
        target = PolicyTarget(old_root.steps, old_root.previous_action, mask, priors, visits,
                              probabilities, successes, deaths, timeouts, chosen.action_id)
        self.committed_prefix.append(old_root.frame_token)
        self.root = chosen.child
        self.root.parent = None
        self.root.incoming_action = None
        self.live_nodes = self._count_nodes(self.root)
        return target

    @staticmethod
    def _count_nodes(root: Node) -> int:
        count, stack = 0, [root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(edge.child for edge in node.edges.values()
                         if edge.child is not None)
        return count
