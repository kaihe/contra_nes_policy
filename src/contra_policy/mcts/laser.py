"""Exact fixed-state Laser environment and searched-episode generator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contra_policy.action_space import actions_np
from contra_policy.alphazero import SearchEpisode
from contra_policy.mcts.core import Node, SearchConfig, SearchTree, Terminal, Transition
from contra_policy.mcts.policy import TorchSearchPolicy
from contra_policy.rl.rollout import (IDLE_ACTION, budget_for, claim_emulator,
                                      classify_step, release_emulator)
from contra_policy.rl.tasks import resize_to_input

MOTION_SCALE = 32.0


@dataclass(frozen=True)
class LaserState:
    ram: np.ndarray
    motion: np.ndarray
    weapon: int
    rapid: bool


def decode_state(previous_ram: np.ndarray, current_ram: np.ndarray) -> LaserState:
    """Decode design-0029 auxiliary targets; motion is clipped to one action chunk."""
    from env.constant import ADDR_WEAPON
    from env.entity import ADDR_PLAYER_Y, player_x

    dx = np.clip((player_x(current_ram) - player_x(previous_ram)) / MOTION_SCALE, -1, 1)
    dy = np.clip((int(current_ram[ADDR_PLAYER_Y]) - int(previous_ram[ADDR_PLAYER_Y]))
                 / MOTION_SCALE, -1, 1)
    raw = int(current_ram[ADDR_WEAPON])
    weapon = raw & 0x0F
    if weapon > 5:
        weapon = 5
    return LaserState(np.asarray(current_ram).copy(), np.array([dx, dy], np.float32),
                      weapon, bool(raw & 0x10))


class LaserEnvironment:
    """One fixed boss task with exact restoration and full level-1 search reward."""

    def __init__(self, catalog, task, *, image_size: int = 256,
                 budget_mult: float = 2.0, min_budget: int = 24):
        from agent.action_mask import legal_mask
        from agent.reward import compute_reward
        from agent.sampler import ActionSampler
        from env.event import make_terminal_events
        from task_maker.kill_boss import KillBossMaker
        from util.replay import make_env

        claim_emulator("AlphaZeroLaser")
        self.env = make_env()
        self.seg, self.maker = catalog.segment(task), KillBossMaker()
        self.image_size = image_size
        self.budget = budget_for(self.seg, budget_mult, min_budget)
        self.vectors = actions_np(np.uint8)
        sampler = ActionSampler.for_level(1)
        self.reward_config, self._compute_reward = sampler.reward_config, compute_reward
        self._legal_mask = legal_mask
        self.die = next((e for e in make_terminal_events() if e.tag == "die"), None)
        if self.die is None:
            self.close()
            raise RuntimeError("terminal event configuration has no die event")

    def close(self) -> None:
        if getattr(self, "env", None) is not None:
            self.env.close()
            self.env = None
            release_emulator()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _screen(self) -> np.ndarray:
        screen = np.ascontiguousarray(self.env.unwrapped.get_screen())
        return np.ascontiguousarray(resize_to_input(screen, self.image_size))

    def initial_transition(self) -> Transition:
        from util.replay import rewind_state

        rewind_state(self.env, self.seg.initial_state)
        state = self.env.em.get_state()
        ram = self.env.unwrapped.get_ram().copy()
        for _ in range(self.seg.skip):
            self.env.step(self.vectors[IDLE_ACTION])
        observation = self._screen()
        rewind_state(self.env, state)
        decoded = decode_state(ram, ram)
        return Transition(state, observation, decoded, 0.0, Terminal.ACTIVE)

    def legal_mask(self, node: Node) -> np.ndarray:
        previous = self.vectors[node.previous_action]
        return self._legal_mask(self.vectors, node.state_data.ram, previous)

    def step(self, node: Node, action_id: int) -> Transition:
        from util.replay import rewind_state

        rewind_state(self.env, node.emu_state)
        previous = node.state_data.ram
        vector = self.vectors[action_id]
        for _ in range(self.seg.skip):
            self.env.step(vector)
        current = self.env.unwrapped.get_ram().copy()
        reward = self._compute_reward(previous, current, self.reward_config, action=vector)
        steps = node.steps + 1
        reached = bool(self.maker.goal_reached(self.seg, previous, current))
        died = bool(self.die.trigger(previous, current))
        done, outcome, _ = classify_step(reached, died, steps, self.budget)
        terminal = Terminal(outcome) if done else Terminal.ACTIVE
        return Transition(self.env.em.get_state(), self._screen(),
                          decode_state(previous, current), float(reward), terminal)


def generate_episode(model, catalog, task, *, device, simulations: int = 16,
                     bootstrap: bool = False, sample: bool = True, seed: int = 0,
                     precision: str = "bf16", image_size: int = 256) -> SearchEpisode:
    """Generate one complete searched episode from the task's exact initial state."""
    prompt = catalog.prompt(task)
    policy = TorchSearchPolicy(model, prompt.interaction, device=device, precision=precision)
    targets, rewards, motion, weapon, rapid = [], [], [], [], []
    with LaserEnvironment(catalog, task, image_size=image_size) as environment:
        initial = environment.initial_transition()
        root = Node(initial.emu_state, initial.observation, policy.encode(initial.observation),
                    initial.state_data, IDLE_ACTION, 0)
        tree = SearchTree(root, policy, environment,
                          SearchConfig(simulations_per_move=simulations,
                                       bootstrap_rollouts=bootstrap, seed=seed))
        while tree.root.terminal is Terminal.ACTIVE:
            if tree.search() == 0:
                raise RuntimeError("MCTS exhausted its live-node budget")
            state = tree.root.state_data
            target = tree.commit(sample=sample)
            targets.append(target)
            rewards.append(target.chosen_reward)
            motion.append(state.motion)
            weapon.append(state.weapon)
            rapid.append(state.rapid)
        outcome = tree.root.terminal.value
    return SearchEpisode(
        frames=np.stack([t.observation for t in targets]),
        policy_targets=np.stack([t.probabilities for t in targets]).astype(np.float32),
        rewards=np.asarray(rewards, np.float32), motion=np.asarray(motion, np.float32),
        weapon=np.asarray(weapon, np.int64), rapid=np.asarray(rapid, np.float32),
        interaction=prompt.interaction, outcome=outcome)
