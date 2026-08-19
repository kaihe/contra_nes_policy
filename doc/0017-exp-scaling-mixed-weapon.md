# Does adding a second weapon improve boss-fight performance?

## 1. Goal

Every run from [0013](0013-exp-scaling-model.md) to [0016](0016-exp-scaling-joint.md) trained on
**Spread only, from one start state** — `win_level1_20260701015306_i371` — and was scored on that
same fight. Model size, budget, data and the last two together have now all been tried inside
that box; the best checkpoint remains `L-D10k-C40k` at **89%** [81.4, 93.7], and 0016's D40k
cells came in under it at 73 / 83 / 77 / 66%.

The datahouse ships a second store: **40,000 Laser boss episodes from a different start state**
(`win_level1_20260630171218_i8`). These cells add them, doubling the training set to 80,000
episodes at the same 160,000-cycle budget.

**Does a second weapon and a second start state help the boss fight, or does diluting the probe's
own distribution cost play?** [0012](0012-exp-spread-grpo.md) found every boss success ever recorded
came from Spread or Laser, so the mechanics may transfer; or the model may simply get half as
much of the fight it is measured on, at half the exposure.

**The decision:** whether the training corpus stays a single fight or grows across weapons and
start states. A gain argues for commissioning the remaining weapons and starts; a loss says keep
it narrow and spend on the frozen encoder instead.

## 2. Setup

Four cells, one per size — identical to [0016](0016-exp-scaling-joint.md)'s ladder in every respect
except the training set, so the pair is read cell against cell.

Common to all four: boss-only, batch 16, **160,000 cycles**, LR 3e-4, AdamW, 500 warmup, WSD with
a 10% cooldown, bf16, dropout 0.2, `aux_size: 0`, `value_head: false`, frozen stage-A encoder
(`encoder-final.pt`, sha `f36041bc…1923c`), datahouse tokens, seed 0, `config_bc_scaling_80k.yaml`.
Head dim 64, aspect ratio d/n_layer 128. Checkpoints every 10,000 cycles plus best and final.

| run | params | d_model | n_layer | ms/step | state | dir |
|---|---:|---:|---:|---:|---|---|
| `M-D80k-C160k` | 12.86M | 512 | 4 | 32.0 | done, 95 min | `runs/scaling80k/M-D80k-C160k` |
| `L-D80k-C160k` | 25.13M | 640 | 5 | 42.9 | done, 157 min | `runs/scaling80k/L-D80k-C160k` |
| `XL-D80k-C160k` | 42.89M | 768 | 6 | 50.2 | **stopped at 130,000**, resumable | `runs/scaling80k/XL-D80k-C160k` |
| `XXL-D80k-C160k` | 101.76M | 1024 | 8 | — | **not run** | — |

XL was killed to free the machine; `policy-partial-130328.pt` is the SIGTERM save and is not a
final. Its queue entry carries `resume_from=…/policy-130000.pt`.

**This is not "D40k with more of the same data."** Read against 0016 it moves **three things at
once**, and §3 must be read that way:

| | Spread | Laser |
|---|---|---|
| episodes / frames | 40,000 / 3,134,341 | 40,000 / 4,180,163 |
| mean / max frames | 78.4 / 242 | 104.5 / 311 |
| start state | `win_level1_20260701015306_i371` | `win_level1_20260630171218_i8` |

- **How much data** — 80,000 episodes against 40,000.
- **What the data is** — half is now a weapon and a start state 0016 never trained on.
- **How many passes** — 32.0 epochs against 64.0, because the budget is held fixed.

So a Spread rate below 0016's does not distinguish "more data does not help" from "half the data
is off probe" or "half the exposure". Longest episode is 311 frames, inside
`policy.core.context: 1024`, so nothing is truncated. Laser's longer episodes make a batch ~17%
more tokens.

Validation is the provisional **1-in-40** uid-digest holdout, ~2,005 episodes — matched to
0016's 1,976 so the CE noise floor is the same. Not the fixed split data
[0004](../../contra_nes_data/doc/0004-tokenized-datahouse.md) promises, so **CE is comparable
inside this document only**; win rate is unaffected.

**Not varied:** seed (0), LR (3e-4, unswept), schedule (WSD), model shapes, budget.
**Not run:** a `D80k-C320k` cell that would restore 64.0 epochs and separate dilution from
exposure; Regular and Flamethrower, whose stores are empty and which
[0012](0012-exp-spread-grpo.md) measured at 0 wins in 316 rollouts.

## 3. Evaluation metrics

Closed-loop success from eval [0021](../../contra_nes_evaluation/doc/0021-d80k-mixed-finals.md),
`runs/0818-d80k-{spread,laser}/{M,L}-<role>-n100`: n = 100 rollouts, T = 1.0, seed 0, 2x expert
budget, bf16, batch 8, train split, Wilson 95% in brackets. **In-distribution for both starts** —
D80k trained on each — so this is a memorization probe, not a val boss rate. XL and XXL are not
scored.

**Spread start** (`full_spread.state` ≡ `win_level1_20260701015306_i371`), against 0016's
spread-only cells at the same budget:

| size | D40k-C160k final (0016) | D80k-C160k final |
|---|---|---|
| M | 73% [63.6, 80.7] | 67% [57.3, 75.4] |
| L | 83% [74.5, 89.1] | 75% [65.7, 82.5] |

**Laser start** (`full_laser.state` ≡ `win_level1_20260630171218_i8`) — first evidence about one
core playing two fights. No 0016 control exists; those cells never saw this start.

| size | D80k-C160k final |
|---|---|
| M | 23% [15.8, 32.2] |
| L | 30% [21.9, 39.6] |

Trajectory, both starts:

| step | M Spread | L Spread | M Laser | L Laser |
|---|---|---|---|---|
| 40k | 62% | 56% | 31% | 19% |
| 80k | 51% | 53% | 23% | 29% |
| 120k | 48% | 58% | 30% | 32% |
| 160k (final) | **67%** | **75%** | **23%** | **30%** |

Cross-entropy is not reported here; `tools/scaling_report.py runs/scaling80k` prints train CE,
val CE final and best with the step, per the split caveat in §2.

## 4. Conclusion

_Pending — M and L collected on both starts, XL and XXL not finished, awaiting discussion._
