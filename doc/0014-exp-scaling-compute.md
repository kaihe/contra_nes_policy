# Does doubling compute lift the model ladder at fixed data?

## 1. Goal

[0013](0013-exp-scaling-model.md) scaled the core at D10k / 20,000 cycles and concluded that size
alone buys nothing there, and that XXL is **under-trained** — its win rate was still climbing at
the end of the budget. This set of runs tests exactly that: hold the data at
`boss-spread-10k-v1` D13 and the four core sizes fixed, and **double the training budget from
20,000 to 40,000 cycles**.

**The question:** at each model size, does twice the compute raise the boss win rate, and does
the size ranking change when every cell is trained longer? **The decision it drives:** whether
the next round of compute goes into longer runs at the sizes we already have, or into more data
([0015](0015-exp-scaling-data.md)) — the two halves of the Chinchilla trade that 0013 named but
could not separate.

## 2. Setup

Two rows of four at matched size and matched data; the budget is the axis. Head dim 64, aspect
ratio d/n_layer 128, so width and depth scale together and a cell is named by its width.

**Common to all runs:** boss-only Spread, `boss-spread-10k-v1` D13 prefix (13 shards / 9,900
episodes / 770,679 frames, one start state `win_level1_20260701015306_i371`), validation the
release's own 100-task holdout shard `29fd4017…cc9ae0`, batch 16, LR **3e-4**, AdamW, 500
warmup, bf16, dropout 0.2, `aux_size: 0`, `value_head: false`, frozen stage-A encoder
(`encoder-final.pt`, sha `f36041bc…1923c`), precomputed encoder tokens,
`config_bc_scaling.yaml`, seed 0.

| row | run | params | cycles | epochs | decay | checkpoints kept | dir |
|---|---|---:|---:|---:|---|---|---|
| C20k | `M-D10k-C20k-cos` | 12.86M | 20,000 | 32.3 | cosine | 3k, 10k, best, final | `runs/scaling/m-d13-s0` |
| C20k | `L-D10k-C20k-cos` | 25.13M | 20,000 | 32.3 | cosine | 3k, 10k, best, final | `runs/scaling/l-d13-s0` |
| C20k | `XL-D10k-C20k-cos` | 42.89M | 20,000 | 32.3 | cosine | 3k, 10k, best, final | `runs/scaling/xl-d13-s0` |
| C20k | `XXL-D10k-C20k-cos` | 101.76M | 20,000 | 32.3 | cosine | 3k, 10k, best, final | `runs/scaling/xxl-d13-s0` |
| C40k | `M-D10k-C40k` | 12.86M | 40,000 | 64.6 | wsd | 3k, 10k, 20k, 30k, best, final | `runs/scaling/M-D10k-C40k` |
| C40k | `L-D10k-C40k` | 25.13M | 40,000 | 64.6 | wsd | 3k, 10k, 20k, 30k, best, final | `runs/scaling/L-D10k-C40k` |
| C40k | `XL-D10k-C40k` | 42.89M | 40,000 | 64.6 | wsd | 3k, 10k, 20k, 30k, best, final | `runs/scaling/XL-D10k-C40k` |
| C40k | `XXL-D10k-C40k` | 101.76M | 40,000 | 64.6 | wsd | 3k, 10k, 20k, 30k, best, final | `runs/scaling/XXL-D10k-C40k` |
| control | `XXL-D10k-C40k-cos` | 101.76M | 40,000 | 64.6 | cosine | 3k, 10k, 20k, 30k, best, final | `runs/scaling/xxl-d13-s0-40k` |

The C20k row is [0013](0013-exp-scaling-model.md)'s ladder, unchanged and re-read here on the
budget axis. `-cos` marks the schedule; seed 0 and WSD are defaults and stay out of the name.

**The schedule is tangled with the budget, and the control row is how it gets untangled.** The
C20k row is cosine because cosine was the default when it ran; the C40k row is WSD because
cosine defines its LR against `train.steps` and so cannot be extended — 0013's first
conclusion. That makes a bare C20k-vs-C40k comparison two changes, not one.
`XXL-D10k-C40k-cos` is the same size and budget as `XXL-D10k-C40k` under the old schedule, so
at XXL the budget can be read against cosine alone (C20k-cos → C40k-cos) and the schedule
against fixed budget (C40k-cos → C40k-wsd). At M, L and XL the two remain confounded, and §3
must be read that way.

**A 20k checkpoint of a C40k run is not a C20k run.** Under WSD, step 20,000 of 40,000 sits
mid-stable-phase at full LR with no cooldown behind it; a C20k cosine run at step 20,000 is
fully annealed. The rows compare finals to finals, not step counts to step counts.

**Not varied:** seed (0 only), LR (3e-4, unswept across a 7.9x parameter span, as in 0013),
data (D10k throughout — the data axis is [0015](0015-exp-scaling-data.md)). **Not run:** XS and S
at either budget, an 80k row, and any held-out boss evaluation — see §3. WSD makes any
stable-phase checkpoint of the C40k row a valid trunk if an 80k row is ever wanted, which is
the practical reason the schedule changed at all.

## 3. Evaluation metrics

**Cross-entropy and cost**, from `tools/scaling_report.py runs/scaling` over each run's
`metrics.csv`. `train CE` is the tail-average of the last 20 logged train rows; `val CE (best)`
is the minimum over the run, `@step` where it occurred; ms/step is a 4090 laptop with the token
cache on.

| row | run | train CE | val CE (final) | val CE (best) | @step | ms/step | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| C20k | `M-D10k-C20k-cos` | 0.1819 | 1.3627 | 0.5855 | 4,000 | 29.2 | 9.7 min |
| C20k | `L-D10k-C20k-cos` | 0.1746 | 1.3866 | 0.5907 | 3,000 | 35.4 | 11.8 min |
| C20k | `XL-D10k-C20k-cos` | 0.2189 | 1.2827 | 0.5900 | 5,000 | 48.2 | 16.1 min |
| C20k | `XXL-D10k-C20k-cos` | 0.3921 | 0.7746 | 0.5883 | 5,000 | 71.1 | 23.7 min |
| C40k | `M-D10k-C40k` | 0.1024 | 1.9225 | 0.5903 | 3,000 | 30.2 | 21.7 min |
| C40k | `L-D10k-C40k` | 0.0899 | 2.0748 | 0.5923 | 3,000 | 38.5 | 28.2 min |
| C40k | `XL-D10k-C40k` | 0.0905 | 2.1234 | 0.5879 | 6,000 | 50.8 | 41.6 min |
| C40k | `XXL-D10k-C40k` | 0.3122 | 1.0423 | 0.5823 | 13,000 | 73.6 | 53.6 min |
| control | `XXL-D10k-C40k-cos` | 0.2353 | 1.3454 | 0.5881 | 8,000 | 73.0 | 48.7 min |

The C40k row cost 2 h 25 min for the four (wall clock from `queue/state.tsv` and the run logs);
C20k wall is ms/step x cycles.

**Closed-loop success**, from `contra_nes_evaluation` docs
[0015](../../contra_nes_evaluation/doc/0015-scaling-single-spread.md) (the C20k row) and
[0016](../../contra_nes_evaluation/doc/0016-c40k-d10k.md) (the C40k row and the control),
carrying their label unchanged: these are **in-distribution / memorization probes** on the
single *train* start state that sources all 9,900 training episodes — not a val boss rate, and
not comparable to the 57-task mixed-weapon probes (~3–14%) or the specialty val Spread probe
(9.5%, eval 0014). n = 100 rollouts per checkpoint, T = 1.0, seed 0, 2x expert budget, bf16,
batch 8; Wilson 95% intervals, half-width ~8–10 pp near 50% and ~5–7 pp near 85%. Timeout 0 and
saw-boss 100% in every cell.

Every cell below reads **success % [Wilson 95%] · mean damage %** — for example
`86% [77.9, 91.5] · 91.8` means 86 of 100 rollouts killed the boss, a true rate of 77.9–91.5%
is consistent with that, and the average rollout removed 91.8% of the boss's starting HP
(`damage_removed / boss_hp_start`, counting a win as 100% and averaging over failures too, so
the 14 losses here averaged ~41% each). Every number has a run directory under
`contra_nes_evaluation/runs/`, named mechanically from the run and the checkpoint role:

| row | directory |
|---|---|
| C20k | `0812-scaling-single-spread/<size>-{best,final}-n100` (lower-case size) |
| C40k | `0813-c40k-grid/<run>-{best,final,010000,020000,030000}-n100` |
| control | `0812-xxl-40k-single-spread/{best,20k,30k,final}-n100` |

### Finals — the comparison this document rests on

Read finals against finals, per §2: a 20k checkpoint of a C40k run is not a C20k run.

| size | C20k-cos final | C40k-wsd final | Δ | C40k-cos final (control) |
|---|---|---|---:|---|
| M | 69% [59.4, 77.2] · 82.1 | **86%** [77.9, 91.5] · 91.8 | +17 pp | — |
| L | 58% [48.2, 67.2] · 76.5 | **89%** [81.4, 93.7] · 93.7 | +31 pp | — |
| XL | 60% [50.2, 69.1] · 75.0 | **88%** [80.2, 93.0] · 94.2 | +28 pp | — |
| XXL | 55% [45.2, 64.4] · 75.4 | 59% [49.2, 68.1] · 74.6 | +4 pp | 71% [61.5, 79.0] · 83.3 |

Δ carries **both** changes — budget and cosine → WSD. Only XXL has the control that separates
them, and at that matched 40,000-cycle budget cosine (71%) leads WSD (59%).

### Best-val-CE checkpoints — a different checkpoint, not a different run

The step each was selected at is in the CE table above; it is repeated here because it varies
by an order of magnitude across cells.

| size | C20k-cos best | C40k-wsd best | C40k-cos best (control) |
|---|---|---|---|
| M | 59% @ 4k [49.2, 68.1] · 74.2 | 32% @ 3k [23.7, 41.7] · 56.0 | — |
| L | 38% @ 3k [29.1, 47.8] · 62.3 | 35% @ 3k [26.4, 44.7] · 55.2 | — |
| XL | 51% @ 5k [41.3, 60.6] · 69.2 | 58% @ 6k [48.2, 67.2] · 73.2 | — |
| XXL | 46% @ 5k [36.6, 55.7] · 62.7 | 45% @ 13k [35.6, 54.8] · 66.6 | 46% @ 8k [36.6, 55.7] · 65.5 |

No best-val checkpoint beats its own run's final, and in the C40k row it trails by 14 to 54 pp.
These rows select on val CE, which 0010 established is not a proxy for play; they are reported
so the budget axis is not read off a checkpoint rule that silently changes with it.

### XXL only — the trajectory

XXL is the one size with intermediate checkpoints evaluated, on both schedules.

| step | C40k-wsd | C40k-cos (control) |
|---|---|---|
| 10k | 39% [30.0, 48.8] · 61.8 | — |
| 20k | 45% [35.6, 54.8] · 67.0 | 46% [36.6, 55.7] · 67.7 |
| 30k | 55% [45.2, 64.4] · 71.2 | 60% [50.2, 69.1] · 74.1 |
| 40k (final) | 59% [49.2, 68.1] · 74.6 | 71% [61.5, 79.0] · 83.3 |

Both curves are monotone and neither has flattened at 40,000 cycles.

**No held-out boss rate exists for any run here**, for the same two structural reasons as 0013:
the 57-task mixed-v2 set is out of distribution for Spread-only models (eval 0015 §1 declined
to run it), and the release's own 100-task holdout ships no task `.npz` start states, so it
supports CE only (data
[0003](../../contra_nes_data/doc/0003-incremental-spread-scaling.md)). Every rate above is a
train-state probe.

## 4. Conclusion

1. More computation makes the model memorize the training traces better.
2. The XXL model is still under-trained at C40k.
3. The best-val-CE checkpoint does not mean the most capable model, for now.
