"""Run the design-0029 fixed-state Laser AlphaZero loop."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from contra_policy.alphazero import evaluate_epoch, train_epoch
from contra_policy.mcts.laser import evaluate_policy_episode, generate_episode
from contra_policy.model import initialize_alphazero_policy
from contra_policy.rl.tasks import TaskCatalog


def run(args: argparse.Namespace) -> str:
    if not 0 < args.validation_episodes < args.episodes_per_generation:
        raise ValueError("validation episodes must be between zero and episodes per generation")
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
    epochs_path = os.path.join(args.output, "epochs.jsonl")
    best_path = os.path.join(args.output, "best-policy.pt")
    final_path = os.path.join(args.output, "final-policy.pt")
    start_time = time.monotonic()
    deadline = start_time + args.max_wall_minutes * 60 if args.max_wall_minutes > 0 else None

    def evaluate_raw():
        return [evaluate_policy_episode(
            model, catalog, task, device=device,
            seed=args.seed + 1_000_000 + i,
            precision=args.precision, image_size=args.image_size)
            for i in range(args.eval_episodes)]

    def evaluation_fields(evaluations):
        return {
            "policy_eval_episodes": len(evaluations),
            "policy_wins": int(sum(outcome == "success" for outcome, _ in evaluations)),
            "policy_win_rate": float(np.mean([
                outcome == "success" for outcome, _ in evaluations])),
            "policy_return_mean": float(np.mean([reward for _, reward in evaluations])),
        }

    with open(os.path.join(args.output, "resolved_config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, sort_keys=True)
    try:
        baseline_eval = evaluate_raw()
        baseline = {"generation": -1, "kind": "initial_policy",
                    "elapsed_seconds": time.monotonic() - start_time,
                    **evaluation_fields(baseline_eval)}
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(baseline) + "\n")
        best_wins = baseline["policy_wins"]
        model.save(best_path, generation=-1, metrics=baseline)
        print(json.dumps(baseline), flush=True)
        last_generation = -1
        for generation in range(args.generations):
            if deadline is not None and time.monotonic() >= deadline:
                break
            episodes = []
            for episode_index in range(args.episodes_per_generation):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                episode = generate_episode(
                    model, catalog, task, device=device, simulations=args.simulations,
                    sample=True,
                    seed=args.seed + generation * args.episodes_per_generation + episode_index,
                    precision=args.precision, image_size=args.image_size)
                episodes.append(episode)
            if len(episodes) != args.episodes_per_generation:
                break
            split_rng = np.random.default_rng(args.seed + 500_000 + generation)
            order = split_rng.permutation(len(episodes))
            validation = [episodes[int(i)] for i in order[:args.validation_episodes]]
            training = [episodes[int(i)] for i in order[args.validation_episodes:]]
            pre_train = evaluate_epoch(model, training, device=device,
                                       batch_episodes=args.batch_episodes)
            pre_validation = evaluate_epoch(model, validation, device=device,
                                            batch_episodes=args.batch_episodes)
            train_metrics, validation_metrics = pre_train, pre_validation
            best_validation, stale_epochs = pre_validation["policy_loss"], 0
            for epoch in range(args.epochs):
                train_metrics = train_epoch(
                    model, training, optimizer, device=device,
                    batch_episodes=args.batch_episodes,
                    seed=args.seed + generation * args.epochs + epoch)
                validation_metrics = evaluate_epoch(
                    model, validation, device=device, batch_episodes=args.batch_episodes)
                epoch_row = {"generation": generation, "epoch": epoch,
                             "elapsed_seconds": time.monotonic() - start_time,
                             **{f"train_{k}": v for k, v in train_metrics.items()},
                             **{f"validation_{k}": v
                                for k, v in validation_metrics.items()}}
                with open(epochs_path, "a") as fh:
                    fh.write(json.dumps(epoch_row) + "\n")
                if validation_metrics["policy_loss"] < best_validation:
                    best_validation, stale_epochs = validation_metrics["policy_loss"], 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= args.early_stop_patience:
                        break
            evaluations = evaluate_raw()
            row = {"generation": generation, "episodes": len(episodes),
                   "train_episodes": len(training), "validation_episodes": len(validation),
                   "epochs_completed": epoch + 1,
                   "elapsed_seconds": time.monotonic() - start_time,
                   "return_mean": float(np.mean([e.rewards.sum() for e in episodes])),
                   "search_win_rate": float(np.mean([
                       e.outcome == "success" for e in episodes])),
                   **evaluation_fields(evaluations),
                   **{f"pre_train_{k}": v for k, v in pre_train.items()},
                   **{f"pre_validation_{k}": v for k, v in pre_validation.items()},
                   **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"validation_{k}": v for k, v in validation_metrics.items()}}
            promoted = row["policy_wins"] > best_wins
            row["promoted"] = promoted
            if promoted:
                best_wins = row["policy_wins"]
                model.save(best_path, generation=generation, optimizer=optimizer.state_dict(),
                           metrics=row)
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
            last_generation = generation
        model.save(final_path, generation=last_generation,
                   optimizer=optimizer.state_dict(), elapsed_seconds=time.monotonic() - start_time)
    finally:
        catalog.close()
    return final_path


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
    parser.add_argument("--validation-episodes", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--max-wall-minutes", type=float, default=0)
    parser.add_argument("--batch-episodes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
