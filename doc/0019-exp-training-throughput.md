# Which batch size and worker count is fastest for each model size?

## 1. Goal

Every cell from [0013](0013-exp-scaling-model.md) to [0018](0018-exp-learning-rate.md) trained at
**batch 16 with `num_workers: 0`**, at every width, and neither was ever swept. The D40k C160k
ladder costs **12 h 36** on that config, and XXL alone is 53% of it.

**What batch size and worker count is fastest at each size?** Those are the two axes here;
nothing else is varied. The decision it drives is the config the next ladder runs at, and whether
that is one setting for the whole ladder or one per size — a per-size batch makes a size
comparison also a batch comparison, so it costs a new baseline row.

## 2. Setup

**These are throughput measurements, not training runs.** Each arm runs the real
`BCTrainer.train_step` — same forward, backward, clip and optimizer — for a fixed number of
cycles, with no validation and no checkpoints, and reports the median `step_ms`. No arm trains
long enough for its CE to mean anything and none is kept.

Common to every arm: `config_bc_bench.yaml` (inheriting 0018's `config_bc_scaling_lr.yaml`), the
datahouse 13-shard D10k prefix, frozen encoder, bf16, seed 0, `pool_batches: 32`. The only keys
that move are `loader.batch_size` and `loader.num_workers`.

| arm | batch | workers | cycles |
|---|---:|---:|---:|
| A (control) | 16 | 0 | 400 |
| B | 16 | 2 | 400 |
| C | 32 | 0 | 400 |
| D | 64 | 0 | 400 |
| BC | 32 | 2 | 400 |
| W16 | 32 | 16 | 120 |

Every arm runs at all four sizes, twice, in fresh processes; §3 reports the median of the per-run
medians and the spread between repeats. Two rules the hardware forces: the GPU is an **RTX 4090
Laptop capped at 80 W**, so clocks swing and one run is not reproducible to better than a few
percent — hence the repeats; and **no two CUDA processes run at once**, measured at 170.9 ms/step
beside a second process against 72.0 ms alone.

**W16 ran at 120 cycles, not 400**, after the 400-cycle grid was already complete. Short windows
inflate `step_ms` — a 40-cycle probe read 57.6 ms at M where 400 reads 34.8 — so W16's numbers are
if anything pessimistic against the rest of the table.

**Batch was never raised speculatively.** Batch 64 was predicted from two measured batches at
2.20 / 3.45 / 4.96 / 9.05 GiB for M / L / XL / XXL against a 70%-of-16 GB budget, then run; XXL
measured 9.05 GiB, matching exactly.

**The `num_workers: 0-2` ceiling this repo carries does not apply here.** It was measured on the
RL rollout path, which decodes frames into pixel tensors; this path mmaps precomputed tokens.
Summed RSS across forked workers also over-counts every shared page — at w16 it reads 11.42 GiB
against a true **PSS of 2.91 GiB**, less than the w2 arm's RSS. Worker count was swept to 16 on
that basis.

**Not varied:** the sampler (`pool_batches: 32`), validation cadence, checkpoint policy, learning
rate, data tier, and everything else 0018 fixed. **Not run:** batch 64 with workers, the remaining
corner of the grid.

## 3. Evaluation metrics

From `python tools/bench_report.py runs/bench/arms.jsonl --ladder 160000`; every individual run is
one JSON line in `runs/bench/arms.jsonl`. `C160k` prices a full ladder at **matched episodes, not
matched cycles**, since batch 32 reaches the same exposure in half the steps.

| size | batch | workers | ms/step | spread | repeats | episodes/s | peak VRAM | C160k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M | 16 | 0 | 34.79 | ±3% | 2 | 460 | 0.78 | 93 min |
| M | 16 | 2 | 32.41 | ±18% | 2 | 494 | 0.78 | 86 min |
| M | 32 | 0 | 40.74 | ±3% | 2 | 786 | 1.25 | 54 min |
| M | 64 | 0 | 51.32 | ±34% | 2 | 1,247 | 2.26 | **34 min** |
| M | 32 | 2 | 28.78 | ±4% | 2 | 1,112 | 1.25 | 38 min |
| M | 32 | 16 | 33.00 | ±2% | 2 | 970 | 1.09 | 44 min |
| L | 16 | 0 | 45.32 | ±10% | 2 | 353 | 1.25 | 121 min |
| L | 16 | 2 | 38.64 | ±8% | 2 | 414 | 1.25 | 103 min |
| L | 32 | 0 | 40.20 | ±1% | 2 | 796 | 1.98 | 54 min |
| L | 64 | 0 | 65.01 | ±12% | 2 | 984 | 3.45 | 43 min |
| L | 32 | 2 | 31.29 | ±1% | 2 | 1,023 | 1.98 | **42 min** |
| L | 32 | 16 | 39.52 | ±10% | 2 | 810 | 1.74 | 53 min |
| XL | 16 | 0 | 53.01 | ±13% | 2 | 302 | 1.86 | 141 min |
| XL | 16 | 2 | 41.63 | ±10% | 2 | 384 | 1.86 | 111 min |
| XL | 32 | 0 | 55.04 | ±16% | 2 | 581 | 2.89 | 73 min |
| XL | 64 | 0 | 87.09 | ±5% | 2 | 735 | 4.94 | **58 min** |
| XL | 32 | 2 | 48.26 | ±3% | 2 | 663 | 2.89 | 64 min |
| XL | 32 | 16 | 46.24 | — | 1 | 692 | 2.55 | 62 min |
| XXL | 16 | 0 | 71.96 | ±10% | 2 | 222 | 3.61 | 192 min |
| XXL | 16 | 2 | 66.45 | ±0% | 2 | 241 | 3.61 | 177 min |
| XXL | 32 | 0 | 112.63 | ±1% | 2 | 284 | 5.42 | 150 min |
| XXL | 64 | 0 | 237.71 | — | 1 | 269 | 9.05 | 158 min |
| XXL | 32 | 2 | 86.67 | ±3% | 2 | 369 | 5.42 | 116 min |
| XXL | 32 | 16 | 81.12 | ±2% | 3 | 395 | 4.81 | **108 min** |

XXL at batch 64 is a single repeat; the second crashed. It is reported as one run rather than
dropped.

**Whole-ladder wall clock**, all four sizes at C160k-equivalent exposure:

| config | ladder | vs control |
|---|---:|---:|
| batch 16, 0 workers (control) | 9.11 h | — |
| batch 16, 2 workers | 7.96 h | −13% |
| batch 32, 0 workers | 5.52 h | −39% |
| batch 64, 0 workers | 4.90 h | −46% |
| batch 32, 16 workers | 4.44 h | −51% |
| **batch 32, 2 workers** | **4.33 h** | **−52%** |
| fastest config per size | 4.16 h | −54% |

**Fastest per size:**

| size | batch | workers | C160k | vs control |
|---|---:|---:|---:|---:|
| M | 64 | 0 | 34 min | −63% |
| L | 32 | 2 | 42 min | −65% |
| XL | 64 | 0 | 58 min | −59% |
| XXL | 32 | 16 | 108 min | −44% |

M's batch-64 lead over batch 32 with 2 workers (34 against 38 min) is smaller than that arm's
±34% spread. XL's lead over 16 workers (58 against 62 min) is not.

## 4. Conclusion

Use batch 32 with `num_workers: 2` for all the following experiments.
