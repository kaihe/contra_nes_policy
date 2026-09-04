# Can a larger GPT core beat the ~10% boss win-rate ceiling?

## 1. Goal

**Check whether a GPT policy can win the boss fight at a rate higher than the previous
ceilings, by scaling the core.** Every prior attempt has stalled around 10%:

| where | boss win rate |
|---|---|
| [0004](0004-exp-grpo.md) phase 2, GRPO | 3.5% → **10.5%** |
| [0011](0011-exp-boss-grpo.md), graded GRPO, held-out | 7.5 / 11.0 / 7.5% against an 8.5% init |
| [0009](0009-exp-boss-data-scaling.md), BC across four data scales | flat, ~90% death at every scale |
| eval [0014](../../contra_nes_evaluation/doc/0014-spread-specialty-boss-grpo.md), binary GRPO on Spread | **9.5% = init**, McNemar p = 1.0 |

Optimizer and data have both been eliminated as the reason, and the 12.86M causal core
inherited VPT's 4x512 shape in [0002](0002-design-gpt-policy.md) without ever being priced. This set
of runs varies **one thing only — the size of the core** — holding the data at
`boss-spread-10k-v1` D13 and the budget at 20,000 cycles, and asks whether capacity is what
has been holding the win rate at that ceiling. **The decision it drives:** whether the next
round of compute goes into a larger core, or into the remaining suspects — the frozen encoder,
the schedule, and more data.

## 2. Setup

Four cells, one axis. Head dim is held at 64 and aspect ratio d/n_layer at 128, so width and
depth scale together and a cell is named by its width alone.

**Common to all four:** boss-only Spread, `boss-spread-10k-v1` D13 prefix (13 shards / 9,900
episodes / 770,679 frames, one start state `win_level1_20260701015306_i371`), validation the
release's own 100-task holdout shard `29fd4017…cc9ae0`, **20,000 cycles** at batch 16 (32.3
epochs), LR **3e-4**, AdamW, 500 warmup, **cosine** decay, bf16, dropout 0.2, `aux_size: 0`,
`value_head: false`, frozen stage-A encoder (`encoder-final.pt`, sha `f36041bc…1923c`),
precomputed encoder tokens, `config_bc_scaling.yaml`, seed 0. Checkpoints kept at 3k and 10k,
plus best and final.

Runs are named `<model>-D<data>-C<cycles>`; these predate the scheme and keep the directory
names that eval 0015 cites by path. `-cos` marks the schedule, since WSD later became the
default.

| run | d_model | n_layer | n_head | params | dir |
|---|---:|---:|---:|---:|---|
| `XS-D10k-C20k-cos` | 256 | 2 | 4 | 1.74M | **not run** |
| `S-D10k-C20k-cos` | 384 | 3 | 6 | 5.52M | **not run** |
| `M-D10k-C20k-cos` | 512 | 4 | 8 | 12.86M | `runs/scaling/m-d13-s0` |
| `L-D10k-C20k-cos` | 640 | 5 | 10 | 25.13M | `runs/scaling/l-d13-s0` |
| `XL-D10k-C20k-cos` | 768 | 6 | 12 | 42.89M | `runs/scaling/xl-d13-s0` |
| `XXL-D10k-C20k-cos` | 1024 | 8 | 16 | 101.76M | `runs/scaling/xxl-d13-s0` |

XS and S were not run: the axis was read at the top end first, on the reasoning that a ceiling
is disproved by exceeding it, not by confirming that smaller cores also fail. **The planned
per-size LR sweep `{1e-4, 3e-4, 1e-3}` was also not run** — all four cells train at 3e-4,
unswept across a 7.9x parameter span, which is this set's largest known weakness.

Two code changes were required before any cell but M could train:

- `model.py` pinned `d_model` to the encoder's `hiddim`. Added
  `in_proj = nn.Linear(512, d_core, bias=False)`, and `nn.Identity()` when `d_core == 512`, so
  M adds no parameters and every pre-ladder checkpoint keeps its state-dict shape.
- The evaluation harness needed the same fix: `contra_eval.policies.CheckpointPolicy` read the
  encoder width, and L/XL/XXL crashed on the first `act` until it used core `d_model` and
  applied `in_proj` (eval [0015](../../contra_nes_evaluation/doc/0015-scaling-single-spread.md) §1).

Training ran on precomputed frozen-encoder tokens (`src/contra_policy/token_cache.py`), which
removes the decode and encode legs from the step. The cache is keyed on the encoder checkpoint
sha256 and `image_size`, and `tests/test_token_cache.py` verifies it equals a live encoder
forward to 1.946e-3.

**Not in this set.** Every D10k run at 40,000 cycles — the same four sizes under WSD, and XXL
under cosine — belongs to [0014](0014-exp-scaling-compute.md), the budget axis: once the budget
changes, the comparison is no longer about size alone. The data axis is
[0015](0015-exp-scaling-data.md).

## 3. Evaluation metrics

**Cross-entropy and step cost**, from `tools/scaling_report.py runs/scaling` over each run's
`metrics.csv`. `train CE` is the tail-average of the last 20 logged train rows; `val CE (best)`
is the minimum over the run, `@step` where it occurred; ms/step is a 4090 laptop with the token
cache on.

| run | params | train CE | val CE (final) | val CE (best) | @step | ms/step | wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `M-D10k-C20k-cos` | 12.86M | 0.1819 | 1.3627 | 0.5855 | 4,000 | 29.2 | 9.7 min |
| `L-D10k-C20k-cos` | 25.13M | 0.1746 | 1.3866 | 0.5907 | 3,000 | 35.4 | 11.8 min |
| `XL-D10k-C20k-cos` | 42.89M | 0.2189 | 1.2827 | 0.5900 | 5,000 | 48.2 | 16.1 min |
| `XXL-D10k-C20k-cos` | 101.76M | 0.3921 | 0.7746 | 0.5883 | 5,000 | 71.1 | 23.7 min |

Wall clock is ms/step x 20,000; the ladder cost 61 min end to end.

**Closed-loop success**, from `contra_nes_evaluation` doc
[0015](../../contra_nes_evaluation/doc/0015-scaling-single-spread.md), carrying its label
unchanged: these are **in-distribution / memorization probes** on the single *train* start
state that sources all 9,900 training episodes — not a val boss rate, and not comparable to the
57-task mixed-weapon probes (~3–14%) or the specialty val Spread probe (9.5%, eval 0014).
n = 100 rollouts per checkpoint, T = 1.0, seed 0, 2x expert budget, bf16; Wilson 95% intervals,
half-width ~8–10 pp near 50%. Timeout 0 and saw-boss 100% in every cell.

| run | checkpoint | success | Wilson 95% | mean dmg | source |
|---|---|---:|---|---:|---|
| `M-D10k-C20k-cos` | best (4k) | 59% | [49.2, 68.1] | 74.2% | `runs/0812-scaling-single-spread/m-best-n100` |
| `M-D10k-C20k-cos` | final (20k) | 69% | [59.4, 77.2] | 82.1% | `…/m-final-n100` |
| `L-D10k-C20k-cos` | best (3k) | 38% | [29.1, 47.8] | 62.3% | `…/l-best-n100` |
| `L-D10k-C20k-cos` | final (20k) | 58% | [48.2, 67.2] | 76.5% | `…/l-final-n100` |
| `XL-D10k-C20k-cos` | best (5k) | 51% | [41.3, 60.6] | 69.2% | `…/xl-best-n100` |
| `XL-D10k-C20k-cos` | final (20k) | 60% | [50.2, 69.1] | 75.0% | `…/xl-final-n100` |
| `XXL-D10k-C20k-cos` | best (5k) | 46% | [36.6, 55.7] | 62.7% | `…/xxl-best-n100` |
| `XXL-D10k-C20k-cos` | final (20k) | 55% | [45.2, 64.4] | 75.4% | `…/xxl-final-n100` |

**Not measured — including the metric §1 names.** No **held-out** boss win rate exists for any
checkpoint in this document, so nothing here is on the same axis as the ~10% ceilings in §1,
which are all val-task rates. Two separate gaps produce that:

- **The 57-task mixed-v2 boss val set**, and its 13 Spread+rapid subset: eval 0015 §1 judged it
  out of distribution for Spread-only models and did not run it.
- **The release's own 100-task holdout**, closed-loop: needs task `.npz` start states that
  `boss-spread-10k-v1` does not ship (data
  [0003](../../contra_nes_data/doc/0003-incremental-spread-scaling.md): converted task files are
  build intermediates, not release artifacts). The holdout supports CE only.

## 4. Conclusion

1. We need to switch to WSD, to enable training to be picked up.
2. The XXL model is able to lower val CE, meaning that the unseen boss-fight traces are making
   more sense to it. But based on the game rollout test, the M-sized model is able to memorize
   some traces so precisely that it wins better by repeating from memory.
3. The XXL model also has the potential to win better as training goes on from 5k to 20k, so
   the XXL model is under-trained.
4. Model size scaling up does not give us an advantage on D10k C20k. We need to scale up data
   and computation at the same time, as the Chinchilla law suggested.
