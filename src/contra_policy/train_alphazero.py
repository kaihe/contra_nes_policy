"""Run the design-0029 fixed-state Laser AlphaZero loop."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from contra_policy.alphazero import train_epoch
from contra_policy.mcts.laser import evaluate_policy_episode, generate_episode
from contra_policy.model import initialize_alphazero_policy
from contra_policy.rl.tasks import TaskCatalog


def run(args: argparse.Namespace) -> str:
    device = torch.device(args.device)
    model = initialize_alphazero_policy(args.checkpoint, seed=args.seed).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=args.lr, weight_decay=args.weight_decay)
    os.makedirs(args.output, exist_ok=True)
    catalog = TaskCatalog(task_root=args.task_root, shard_dir=args.shard_dir,
                          families=["boss"], split="train", image_size=args.image_size,
                          require_prompt=False, task_filter={"uid": [args.task_uid]},
                          expected_tasks=1)
    task = catalog.tasks[0]
    metrics_path = os.path.join(args.output, "metrics.jsonl")
    try:
        for generation in range(args.generations):
            episodes = []
            for episode_index in range(args.episodes_per_generation):
                episode = generate_episode(
                    model, catalog, task, device=device, simulations=args.simulations,
                    sample=True,
                    seed=args.seed + generation * args.episodes_per_generation + episode_index,
                    precision=args.precision, image_size=args.image_size)
                episodes.append(episode)
            train_metrics = {}
            for epoch in range(args.epochs):
                train_metrics = train_epoch(
                    model, episodes, optimizer, device=device,
                    batch_episodes=args.batch_episodes,
                    seed=args.seed + generation * args.epochs + epoch)
            evaluations = [evaluate_policy_episode(
                model, catalog, task, device=device,
                seed=args.seed + 1_000_000 + generation * args.eval_episodes + i,
                precision=args.precision, image_size=args.image_size)
                for i in range(args.eval_episodes)]
            row = {"generation": generation, "episodes": len(episodes),
                   "return_mean": float(np.mean([e.rewards.sum() for e in episodes])),
                   "search_win_rate": float(np.mean([
                       e.outcome == "success" for e in episodes])),
                   "policy_eval_episodes": len(evaluations),
                   "policy_win_rate": float(np.mean([
                       outcome == "success" for outcome, _ in evaluations])),
                   "policy_return_mean": float(np.mean([
                       reward for _, reward in evaluations])),
                   **train_metrics}
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            model.save(os.path.join(args.output, f"generation-{generation:04d}.pt"),
                       generation=generation, optimizer=optimizer.state_dict(), metrics=row)
            print(json.dumps(row), flush=True)
    finally:
        catalog.close()
    return os.path.join(args.output, f"generation-{args.generations - 1:04d}.pt")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task-uid", default="win_level1_20260630171218_i8")
    parser.add_argument("--task-root", default="~/code/contra_nes_data/game_trace/tasks")
    parser.add_argument("--shard-dir", default="~/code/contra_nes_data/game_trace/hf")
    parser.add_argument("--output", default="runs/alphazero/laser-fixed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--episodes-per-generation", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-episodes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
