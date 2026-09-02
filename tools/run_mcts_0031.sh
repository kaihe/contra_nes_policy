#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_dir"

for guide in ppo bigram; do
  out_dir="runs/mcts/0031/${guide}-${guide}-s16"
  mkdir -p "$out_dir"
  for seed in $(seq 0 29); do
    output="$out_dir/seed-${seed}.json"
    if [[ -s "$output" ]]; then
      continue
    fi
    python -u -m contra_policy.mcts.laser \
      --guide "$guide" \
      --seed "$seed" \
      --simulations 16 \
      --output "$output"
  done
done
