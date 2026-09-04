#!/bin/bash
# The doc/0019 throughput arms: every arm at every size, one process each.
#
#   bash tools/bench_arms.sh <out.jsonl> <repeats> <arm> [arm ...]
#
# Two protocol rules, both learned the hard way on this box:
#
#   exclusivity   this is an 80 W power-capped mobile 4090. A second CUDA context does
#                 not halve throughput, it roughly quarters it: an XXL arm measured
#                 171 ms/step beside another XXL process and 74 ms/step alone. The guard
#                 below refuses to start while any train_bc or bench_step is alive, which
#                 is the same reason doc/0019 §2 rejects arm F.
#   repeats       clocks swing 900-2415 MHz under the power cap, so one 400-step median
#                 is not reproducible to better than a few percent. Each arm runs
#                 `repeats` times in fresh processes and the report takes the median of
#                 the medians.
#
# A fresh process per repeat is not incidental — peak VRAM and peak RSS are per-process
# high-water marks, and a second arm in the same interpreter reports the first's peak.
set -u
cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1

OUT="${1:?usage: bench_arms.sh <out.jsonl> <repeats> <arm>...}"; shift
# Absolute: hydra chdir's each run into its own dir, so a relative `bench.out` would
# scatter one JSONL per arm instead of appending to one.
mkdir -p "$(dirname "$OUT")"; OUT="$(readlink -f "$OUT")"
REPEATS="${1:?usage: bench_arms.sh <out.jsonl> <repeats> <arm>...}"; shift
SCRATCH="${BENCH_SCRATCH:-/tmp/bench-0019}"
SIZES="${BENCH_SIZES:-M L XL XXL}"

size_args() {
    case "$1" in
        M)   echo "policy.core.d_model=512  policy.core.n_layer=4 policy.core.n_head=8  policy.core.n_kv_head=8"  ;;
        L)   echo "policy.core.d_model=640  policy.core.n_layer=5 policy.core.n_head=10 policy.core.n_kv_head=10" ;;
        XL)  echo "policy.core.d_model=768  policy.core.n_layer=6 policy.core.n_head=12 policy.core.n_kv_head=12" ;;
        XXL) echo "policy.core.d_model=1024 policy.core.n_layer=8 policy.core.n_head=16 policy.core.n_kv_head=16" ;;
    esac
}

# One declared change from the shipped config per arm — doc/0019 §2.
arm_args() {
    case "$1" in
        A)  echo "loader.batch_size=16 loader.num_workers=0" ;;
        B)  echo "loader.batch_size=16 loader.num_workers=2" ;;
        C)  echo "loader.batch_size=32 loader.num_workers=0" ;;
        D)  echo "loader.batch_size=64 loader.num_workers=0" ;;
        BC) echo "loader.batch_size=32 loader.num_workers=2" ;;
        BD) echo "loader.batch_size=64 loader.num_workers=2" ;;
        # The worker sweep. The `num_workers: 0-2` ceiling this project carries comes from
        # the RL rollout path, which decodes frames into pixel tensors; this path mmaps
        # precomputed tokens and the box has 32 cores, so the ceiling is re-tested here
        # rather than inherited.
        W4)  echo "loader.batch_size=32 loader.num_workers=4"  ;;
        W8)  echo "loader.batch_size=32 loader.num_workers=8"  ;;
        W16) echo "loader.batch_size=32 loader.num_workers=16" ;;
        *)  echo "" ;;
    esac
}

wait_for_gpu() {
    while pgrep -f "contra_policy.train_bc|tools/bench_step.py" > /dev/null; do
        echo "[arms] another CUDA process is alive — waiting"; sleep 30
    done
}

for arm in "$@"; do
    args="$(arm_args "$arm")"
    [ -z "$args" ] && { echo "[arms] unknown arm $arm"; continue; }
    for size in $SIZES; do
        for rep in $(seq 1 "$REPEATS"); do
            label="$size-$arm-r$rep"
            if grep -qF "\"label\": \"$label\"" "$OUT" 2>/dev/null; then
                echo "[arms] skip $label — already in $OUT"; continue
            fi
            wait_for_gpu
            echo "[arms] === $label ==="
            python tools/bench_step.py --config-name config_bc_bench \
                $(size_args "$size") $args \
                bench.label="$label" bench.out="$OUT" \
                hydra.run.dir="$SCRATCH/$label" 2>&1 \
                | grep -Ev "UserWarning|_C._set_float32" | tail -3
        done
    done
done
echo "[arms] done"
