"""PUCT search with exact transitions and terminal-success value backup."""

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


@dataclass(frozen=True)
class Transition:
    emu_state: bytes
    observation: np.ndarray
    state_data: Any
    reward: float
    terminal: Terminal = Terminal.ACTIVE


@dataclass(frozen=True)
class Evaluation:
    priors: np.ndarray
    value: float


@dataclass
class Edge:
    action_id: int
    prior: float
    reward: Optional[float] = None
    visits: int = 0
    value_sum: float = 0.0
    child: Optional["Node"] = None

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass
class Node:
    emu_state: bytes
    observation: np.ndarray
    frame_token: Any
    state_data: Any
    previous_action: int
    steps: int
    terminal: Terminal = Terminal.ACTIVE
    parent: Optional["Node"] = field(default=None, repr=False)
    expanded: bool = False
    edges: Dict[int, Edge] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchConfig:
    simulations_per_move: int = 16
    max_live_nodes: int = 2048
    c_puct: float = 1.5
    visit_temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.simulations_per_move < 1 or self.max_live_nodes < 2:
            raise ValueError("simulation and node budgets must be positive")
        if self.c_puct < 0 or self.visit_temperature <= 0:
            raise ValueError("PUCT and temperatures must be positive")


@dataclass(frozen=True)
class PolicyTarget:
    step: int
    legal_mask: np.ndarray
    priors: np.ndarray
    visits: np.ndarray
    probabilities: np.ndarray
    chosen_action: int
    chosen_reward: float
    observation: np.ndarray


class SearchPolicy(Protocol):
    num_actions: int

    def encode(self, observation: np.ndarray) -> Any: ...
    def evaluate(self, frame_tokens: Sequence[Any], previous_action: int) -> Evaluation: ...


class SearchEnvironment(Protocol):
    def legal_mask(self, node: Node) -> np.ndarray: ...
    def step(self, node: Node, action_id: int) -> Transition: ...


class SearchTree:
    """Persistent single-agent tree; one simulation creates at most one permanent node."""

    def __init__(self, root: Node, policy: SearchPolicy, environment: SearchEnvironment,
                 config: SearchConfig = SearchConfig()):
        self.root, self.policy, self.environment, self.config = root, policy, environment, config
        self.committed_tokens: list[Any] = []
        self.live_nodes = 1
        self.completed_simulations = 0
        self.rng = np.random.default_rng(config.seed)
        if root.terminal is Terminal.ACTIVE:
            self._expand(root, self.context(root))

    def context(self, node: Node) -> list[Any]:
        suffix = []
        cur: Optional[Node] = node
        while cur is not None:
            suffix.append(cur.frame_token)
            cur = cur.parent
        return [*self.committed_tokens, *reversed(suffix)]

    def _expand(self, node: Node, context: Sequence[Any]) -> Evaluation:
        evaluation = self.policy.evaluate(context, node.previous_action)
        mask = np.asarray(self.environment.legal_mask(node), dtype=bool)
        priors = np.asarray(evaluation.priors, dtype=np.float64)
        if mask.shape != (self.policy.num_actions,) or priors.shape != mask.shape:
            raise ValueError("legal mask and priors must match the action space")
        if not mask.any() or not np.isfinite(priors).all() or np.any(priors < 0):
            raise ValueError("invalid legal mask or policy priors")
        probs = np.where(mask, priors, 0.0)
        probs = probs / probs.sum() if probs.sum() else mask.astype(float) / mask.sum()
        node.edges = {int(i): Edge(int(i), float(probs[i])) for i in np.flatnonzero(mask)}
        node.expanded = True
        return evaluation

    def select_edge(self, node: Node) -> Edge:
        parent_visits = sum(edge.visits for edge in node.edges.values())
        scale = math.sqrt(max(1, parent_visits))
        return max(node.edges.values(), key=lambda edge: (
            edge.mean_value + self.config.c_puct * edge.prior * scale / (1 + edge.visits),
            -edge.action_id))

    def simulate(self) -> bool:
        if self.root.terminal is not Terminal.ACTIVE:
            return False
        node, path = self.root, []
        while True:
            edge = self.select_edge(node)
            path.append(edge)
            if edge.child is None:
                if self.live_nodes >= self.config.max_live_nodes:
                    return False
                transition = self.environment.step(node, edge.action_id)
                edge.reward = float(transition.reward)
                child = Node(transition.emu_state, transition.observation,
                             self.policy.encode(transition.observation), transition.state_data,
                             edge.action_id, node.steps + 1, transition.terminal, parent=node)
                edge.child = child
                self.live_nodes += 1
                node = child
                break
            node = edge.child
            if node.terminal is not Terminal.ACTIVE:
                break

        if node.terminal is Terminal.ACTIVE:
            evaluation = self._expand(node, self.context(node))
            leaf_value = float(evaluation.value)
        else:
            leaf_value = 1.0 if node.terminal is Terminal.SUCCESS else 0.0
        self.backup(path, leaf_value)
        self.completed_simulations += 1
        return True

    @staticmethod
    def backup(path: Sequence[Edge], leaf_value: float) -> None:
        value = float(leaf_value)
        for edge in reversed(path):
            if edge.reward is None:
                raise RuntimeError("cannot back up an edge without an exact reward")
            edge.visits += 1
            edge.value_sum += value

    def search(self, simulations: Optional[int] = None) -> int:
        target, done = simulations or self.config.simulations_per_move, 0
        while done < target and self.simulate():
            done += 1
        return done

    def root_target(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = np.zeros(self.policy.num_actions, dtype=bool)
        visits = np.zeros(self.policy.num_actions, dtype=np.int64)
        for action, edge in self.root.edges.items():
            mask[action], visits[action] = True, edge.visits
        if visits.sum() == 0:
            raise RuntimeError("search must run before forming a root target")
        scaled = visits.astype(np.float64) ** (1.0 / self.config.visit_temperature)
        return mask, visits, scaled / scaled.sum()

    def commit(self, *, sample: bool = True) -> PolicyTarget:
        mask, visits, probabilities = self.root_target()
        priors = np.zeros(self.policy.num_actions, dtype=np.float64)
        for action, edge in self.root.edges.items():
            priors[action] = edge.prior
        chosen = (int(self.rng.choice(self.policy.num_actions, p=probabilities)) if sample
                  else int(np.flatnonzero(visits == visits.max())[0]))
        edge = self.root.edges[chosen]
        if edge.child is None:
            raise RuntimeError("chosen action has no expanded child")
        old_root = self.root
        target = PolicyTarget(old_root.steps, mask, priors, visits, probabilities, chosen,
                              float(edge.reward), old_root.observation.copy())
        self.committed_tokens.append(old_root.frame_token)
        self.root = edge.child
        self.root.parent = None
        self.live_nodes = self._count_nodes(self.root)
        return target

    @staticmethod
    def _count_nodes(root: Node) -> int:
        count, stack = 0, [root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(e.child for e in node.edges.values() if e.child is not None)
        return count
