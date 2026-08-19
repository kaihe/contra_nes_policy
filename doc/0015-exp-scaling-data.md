# Does doubling boss data from 10k to 20k lift the model ladder?

## 1. Goal

The other two axes are measured: [0013](0013-exp-scaling-model.md) scaled the core at fixed data and
budget, [0014](0014-exp-scaling-compute.md) doubled the budget at fixed data. This set moves the
third — **the data** — holding size and budget fixed. [0009](0009-exp-boss-data-scaling.md) earlier
found the boss curve flat across four data scales, topping out around 2,500 episodes;
`boss-spread-20k-v1` doubles the training set to 19,900 episodes / 1,567,785 frames, so the
question is asked again on a Spread-only release twice the size.

**The question:** at each model size, does the win rate rise when the data doubles, and does the
size ranking change? **The decision it drives:** whether to keep commissioning larger Spread
releases, or to stop buying data and unfreeze the encoder — the last untested component once
capacity (0013), compute (0014) and data (here) have each been moved. A win rate that rises with
data at any size argues for more data; a flat pair of rows leaves the encoder as the only
remaining suspect.

## 2. Setup

Two rows of four, at matched model size and matched compute — the only thing that differs
between rows is which release the episodes came from.

**Common to all eight runs:** batch 16, **40,000 cycles**, LR 3e-4, AdamW, 500 warmup, WSD
with a 10% cooldown, bf16, dropout 0.2, `aux_size: 0`, `value_head: false`, frozen stage-A
encoder (`encoder-final.pt`, sha `f36041bc…1923c`), precomputed encoder tokens, seed 0.
Checkpoints at 3k / 10k / 20k / 30k, plus best and final. Head dim 64, aspect ratio
d/n_layer 128.

| row | data | config | run | params | dir |
|---|---|---|---|---:|---|
| A | D10k | `config_bc_scaling.yaml` | `M-D10k-C40k` | 12.86M | `runs/scaling/M-D10k-C40k` |
| A | D10k | `config_bc_scaling.yaml` | `L-D10k-C40k` | 25.13M | `runs/scaling/L-D10k-C40k` |
| A | D10k | `config_bc_scaling.yaml` | `XL-D10k-C40k` | 42.89M | `runs/scaling/XL-D10k-C40k` |
| A | D10k | `config_bc_scaling.yaml` | `XXL-D10k-C40k` | 101.76M | `runs/scaling/XXL-D10k-C40k` |
| B | D20k | `config_bc_scaling_20k.yaml` | `M-D20k-C40k` | 12.86M | `runs/scaling20k/M-D20k-C40k` |
| B | D20k | `config_bc_scaling_20k.yaml` | `L-D20k-C40k` | 25.13M | `runs/scaling20k/L-D20k-C40k` |
| B | D20k | `config_bc_scaling_20k.yaml` | `XL-D20k-C40k` | 42.89M | `runs/scaling20k/XL-D20k-C40k` |
| B | D20k | `config_bc_scaling_20k.yaml` | `XXL-D20k-C40k` | 101.76M | `runs/scaling20k/XXL-D20k-C40k` |

Row A is the control, and it is the same four runs as [0014](0014-exp-scaling-compute.md)'s C40k
row — one set of runs, read once on the budget axis and once here on the data axis. Row B ran
from `queue/jobs.txt` via `tools/run_queue.sh`, smallest cell first, 2 h 41 min for the four.
All eight runs completed with exit 0.

**The two datasets:**

| | D10k = `boss-spread-10k-v1` | D20k = `boss-spread-20k-v1` |
|---|---|---|
| episodes / frames | 9,900 / 770,679 | 19,900 / 1,567,785 |
| epochs at 40,000 cycles | 64.6 | **32.2** |
| validation | `29fd4017…cc9ae0`, 100 tasks | `84cc5c50…6f9bfc`, its own 100 tasks |
| overlap | — | **zero shared episode uids** |
| start state | `…_i371` | **the same `…_i371`** |

Three consequences, and they set what §3 may compare:

- **D20k is not a superset of D10k.** It is an independent regeneration from the same start
  state, so row B is not row A plus more; it is a different draw, twice as large.
- **Compute is fixed, so more data means fewer passes** — 64.6 epochs against 32.2. That is
  the intended shape of a fixed-compute comparison, not a confound to correct for.
- **The two rows share one start state, so their closed-loop probes are the same task.** Win
  rate is therefore comparable across the data axis. **CE is not** — each release scores
  against its own validation shard — so CE may be read across sizes *within* a row and never
  between rows.

**Not varied:** seed (0 only), LR (3e-4, unswept across a 7.9x parameter span, as in 0013),
budget (40,000 cycles throughout — the budget axis is 0014), schedule (WSD everywhere, so both
rows are internally and mutually comparable). **Not run:** XS and S, and any held-out boss
evaluation — see §3.

**Infrastructure.** All 28 tars of the 20k release were re-hashed against the manifest before
use; every one matched. Its encoder-token cache was built in 18.0 min → 1.63 GB, every episode
decoding to its declared length, frame total equal to the manifest, and 15 random episodes
agreeing with a live encoder forward to **1.940e-3** against a 5e-3 tolerance.

## 3. Evaluation metrics

**Cross-entropy**, complete for both rows, from `tools/scaling_report.py` over `runs/scaling`
and `runs/scaling20k`. `train CE` is the tail-average of the last 20 logged train rows;
`val CE (best)` is the minimum over the run and `@step` where it occurred. Each row is scored
against its own release's holdout, so these compare **across sizes within a row** — never
between rows.

| row | run | train CE | val CE (final) | val CE (best) | @step | ms/step | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| A | `M-D10k-C40k` | 0.1024 | 1.9225 | 0.5903 | 3,000 | 30.2 | 21.7 min |
| A | `L-D10k-C40k` | 0.0899 | 2.0748 | 0.5923 | 3,000 | 38.5 | 28.2 min |
| A | `XL-D10k-C40k` | 0.0905 | 2.1234 | 0.5879 | 6,000 | 50.8 | 41.6 min |
| A | `XXL-D10k-C40k` | 0.3122 | 1.0423 | 0.5823 | 13,000 | 73.6 | 53.6 min |
| B | `M-D20k-C40k` | 0.3036 | 1.0371 | 0.5744 | 7,000 | 28.3 | 19.9 min |
| B | `L-D20k-C40k` | 0.3178 | 1.0043 | 0.5731 | 10,000 | 38.9 | 26.5 min |
| B | `XL-D20k-C40k` | 0.4048 | 0.7962 | 0.5662 | 12,000 | 58.8 | 40.6 min |
| B | `XXL-D20k-C40k` | 0.5100 | 0.5858 | 0.5647 | 27,000 | 86.1 | 74.4 min |

**Closed-loop success**, from `contra_nes_evaluation` docs
[0016](../../contra_nes_evaluation/doc/0016-c40k-d10k.md) (row A) and
[0017](../../contra_nes_evaluation/doc/0017-c40k-d20k.md) (row B), carrying their label
unchanged: these are **in-distribution / memorization probes** on the single *train* start
state that both releases were generated from — not a val boss rate, and not comparable to the
57-task mixed-weapon probes (~3–14%) or the specialty val Spread probe (9.5%, eval 0014).
n = 100 rollouts per checkpoint, T = 1.0, seed 0, 2x expert budget, bf16, batch 8; Wilson 95%
intervals, half-width ~8–10 pp near 50% and ~5–7 pp near 85%. Timeout 0 and saw-boss 100% in
every cell. **Unlike CE, these rates are comparable across rows** — one start state, one
protocol.

Every cell reads **success % [Wilson 95%] · mean damage %**, where mean damage is
`damage_removed / boss_hp_start` averaged over all 100 rollouts, counting a win as 100%. Run
directories are `contra_nes_evaluation/runs/0813-c40k-grid/<run>-<role>-n100`, with role in
`{best, final, 010000, 020000, 030000}`.

### Finals — the comparison this document rests on

| size | A: D10k final | B: D20k final | Δ |
|---|---|---|---:|
| M | **86%** [77.9, 91.5] · 91.8 | 66% [56.3, 74.5] · 81.1 | **−20 pp** |
| L | **89%** [81.4, 93.7] · 93.7 | 65% [55.3, 73.6] · 79.2 | **−24 pp** |
| XL | **88%** [80.2, 93.0] · 94.2 | 63% [53.2, 71.8] · 78.0 | **−25 pp** |
| XXL | 59% [49.2, 68.1] · 74.6 | 61% [51.2, 70.0] · 74.4 | +2 pp |

At M, L and XL the intervals are disjoint and mean damage falls with the win rate. At XXL —
the one cell that was not clearing the start on D10k — the two rows are indistinguishable.

### Best-val-CE checkpoints — the opposite ordering

| size | A: D10k best | B: D20k best |
|---|---|---|
| M | 32% @ 3k [23.7, 41.7] · 56.0 | 40% @ 7k [30.9, 49.8] · 66.9 |
| L | 35% @ 3k [26.4, 44.7] · 55.2 | 55% @ 10k [45.2, 64.4] · 74.1 |
| XL | 58% @ 6k [48.2, 67.2] · 73.2 | 62% @ 12k [52.2, 70.9] · 78.3 |
| XXL | 45% @ 13k [35.6, 54.8] · 66.6 | **68%** @ 27k [58.3, 76.3] · 81.7 |

Row B's best-val checkpoint beats row A's at every size, and rises monotonically with size
(40 → 55 → 62 → 68%) — a ranking that does not survive to the finals above. The best-to-final
gap also reverses: +26 / +10 / +1 / −7 pp on D20k against +54 / +54 / +30 / +14 pp on D10k, so
`XXL-D20k-C40k` is the only run in either document whose high-water mark is not its final.

### XXL only — the trajectory

| step | A: D10k | B: D20k |
|---|---|---|
| 10k | 39% [30.0, 48.8] · 61.8 | 50% [40.4, 59.6] · 68.0 |
| 20k | 45% [35.6, 54.8] · 67.0 | 56% [46.2, 65.3] · 71.5 |
| 27k (B best) | — | 68% [58.3, 76.3] · 81.7 |
| 30k | 55% [45.2, 64.4] · 71.2 | 65% [55.3, 73.6] · 80.2 |
| 40k (final) | 59% [49.2, 68.1] · 74.6 | 61% [51.2, 70.0] · 74.4 |

D20k XXL leads at every intermediate step and ends level with D10k XXL.

Two gaps are structural rather than oversights. Neither release ships task `.npz` start states,
so their holdout shards support **CE only** — a held-out closed-loop rate is impossible without
a data-side change. And the 57-task mixed-v2 set is out of distribution for Spread-only models,
which is why eval 0015 declined to run it for the 0013 ladder.

## 4. Conclusion

1. A larger dataset at fixed computation damages memorization in the small-sized models.
2. The XXL model is much better at generalization.
3. Best val CE is actually a good indicator. It did not work in our previous experiments
   because:
   a. the small dataset is not sufficient to describe the true distribution;
   b. the task happens to be a deterministic one, on which the model can memorize and copy a
      winning trace.
