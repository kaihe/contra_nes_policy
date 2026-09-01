"""Stable-Retro adapter and command-line runner for the design-0030 Laser search."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch

from contra_policy.action_space import ACTION_NAMES, actions_np
from contra_policy.mcts.core import Node, SearchConfig, SearchTree, Terminal, Transition
from contra_policy.mcts.policy import TorchSearchPolicy
from contra_policy.model import load_policy
from contra_policy.rl.rollout import (IDLE_ACTION, budget_for, claim_emulator,
                                      classify_step, release_emulator)
from contra_policy.rl.tasks import TaskCatalog, resize_to_input


DEFAULT_CHECKPOINT = (
    "runs/ppo/2026-08-27/laser-critic-14-56-52/checkpoints/ppo-final.pt")
DEFAULT_UID = "win_level1_20260630171218_i8"


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class LaserEnvironment:
    """Exact one-action transitions for one fixed boss task."""

    def __init__(self, catalog: TaskCatalog, task, *, image_size: int = 256,
                 budget_mult: float = 2.0, min_budget: int = 24):
        from agent.action_mask import legal_mask
        from env.event import make_terminal_events
        from task_maker.kill_boss import KillBossMaker
        from util.replay import make_env

        claim_emulator("LaserMCTS")
        self.env = make_env()
        self._legal_mask = legal_mask
        self.seg = catalog.segment(task)
        self.maker = KillBossMaker()
        self.image_size = image_size
        self.budget = budget_for(self.seg, budget_mult, min_budget)
        self.vectors = actions_np(np.uint8)
        self.die = next((event for event in make_terminal_events() if event.tag == "die"), None)
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

    def __exit__(self, *exc) -> None:
        self.close()

    def initial_transition(self) -> Transition:
        from util.replay import rewind_state

        rewind_state(self.env, self.seg.initial_state)
        state = self.env.em.get_state()
        ram = self.env.unwrapped.get_ram().copy()
        idle = self.vectors[IDLE_ACTION]
        for _ in range(self.seg.skip):
            self.env.step(idle)
        screen = np.ascontiguousarray(self.env.unwrapped.get_screen())
        rewind_state(self.env, state)
        return Transition(state, np.ascontiguousarray(resize_to_input(screen, self.image_size)),
                          ram, Terminal.ACTIVE)

    def legal_mask(self, node: Node) -> np.ndarray:
        previous = self.vectors[node.previous_action]
        return self._legal_mask(self.vectors, node.state_data, previous)

    def step(self, node: Node, action_id: int) -> Transition:
        from util.replay import rewind_state

        rewind_state(self.env, node.emu_state)
        previous_ram = np.asarray(node.state_data)
        for _ in range(self.seg.skip):
            self.env.step(self.vectors[action_id])
        current_ram = self.env.unwrapped.get_ram().copy()
        steps = node.steps + 1
        reached = bool(self.maker.goal_reached(self.seg, previous_ram, current_ram))
        died = bool(self.die.trigger(previous_ram, current_ram))
        done, outcome, _ = classify_step(reached, died, steps, self.budget)
        terminal = Terminal(outcome) if done else Terminal.ACTIVE
        state = self.env.em.get_state()
        screen = np.ascontiguousarray(self.env.unwrapped.get_screen())
        observation = np.ascontiguousarray(resize_to_input(screen, self.image_size))
        return Transition(state, observation, current_ram, terminal)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    model = load_policy(checkpoint, map_location="cpu")
    catalog = TaskCatalog(
        task_root=args.task_root, shard_dir=args.shard_dir, families=["boss"], split="train",
        image_size=args.image_size, require_prompt=False,
        task_filter={"uid": [args.task_uid]}, expected_tasks=1,
    )
    task = catalog.tasks[0]
    prompt = catalog.prompt(task)
    cfg = SearchConfig(args.simulations, args.max_live_nodes, args.c_puct,
                       args.temperature, args.seed)
    targets = []
    try:
        with LaserEnvironment(catalog, task, image_size=args.image_size) as environment:
            policy = TorchSearchPolicy(model, prompt.interaction, device=device,
                                       precision=args.precision)
            initial = environment.initial_transition()
            replayed = environment.initial_transition()
            if (initial.emu_state != replayed.emu_state
                    or not np.array_equal(initial.observation, replayed.observation)
                    or not np.array_equal(initial.state_data, replayed.state_data)):
                raise RuntimeError("restoring the task twice did not reproduce its root state")
            root = Node(initial.emu_state, policy.encode(initial.observation),
                        initial.state_data, IDLE_ACTION, 0)
            tree = SearchTree(root, policy, environment, cfg,
                              model_id=_sha256(checkpoint))
            while tree.root.terminal is Terminal.ACTIVE:
                completed = tree.search()
                if completed == 0:
                    raise RuntimeError("search reached the live-node limit before a backup")
                target = tree.commit()
                targets.append({
                    "step": target.step,
                    "previous_action": target.previous_action,
                    "legal_mask": target.legal_mask.tolist(),
                    "visits": target.visits.tolist(),
                    "probabilities": target.probabilities.tolist(),
                    "chosen_action": target.chosen_action,
                    "chosen_action_name": ACTION_NAMES[target.chosen_action],
                })
            result = {
                "task_uid": task.uid,
                "model_id": tree.model_id,
                "outcome": tree.root.terminal.value,
                "steps": tree.root.steps,
                "completed_simulations": tree.completed_simulations,
                "config": asdict(cfg),
                "targets": targets,
            }
    finally:
        catalog.close()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)
    return result


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--task-uid", default=DEFAULT_UID)
    parser.add_argument("--task-root", default="~/code/contra_nes_data/game_trace/tasks")
    parser.add_argument("--shard-dir", default="~/code/contra_nes_data/game_trace/hf")
    parser.add_argument("--output", default="runs/mcts/laser-search.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--max-live-nodes", type=int, default=2048)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    result = run(parse_args())
    print(json.dumps({k: result[k] for k in ("outcome", "steps", "completed_simulations")},
                     indent=2))


if __name__ == "__main__":
    main()
