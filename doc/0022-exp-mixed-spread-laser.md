# Can one model retain Spread and Laser win rates when trained on both?

## 1. Goal

Separate D10k models reach much higher success on Spread than Laser, and [0021](0021-exp-laser-data-scaling.md)
shows that Laser remains the harder fight as its data scales. A deployable policy should handle
both weapon/start-state distributions without requiring a separate checkpoint for each.

**Can one model trained on D10k Spread plus D10k Laser retain the win rates of separately trained
models, and does that require doubling compute from C20k to C40k?** C20k holds total compute fixed;
C40k restores the exposure each weapon receives in its separate D10k/C20k run.

## 2. Setup

Four model sizes crossed with two compute budgets. Every cell trains on the first 13 of 53 Spread
shards plus the first 18 of 70 Laser shards: **19,101 episodes / 1,751,079 frames**, comprising
9,330 Spread and 9,771 Laser episodes. The uid-digest holdout is carved from both full stores first,
giving the same **3,969 validation episodes** to every cell; train/val overlap is zero.

Common to all cells: mixed Spread + Laser level-1 boss datahouse tokens, batch 32, AdamW, LR
**1e-4**, weight decay 0.01, 500 warmup, WSD with 10% cooldown, bf16, dropout 0.2,
`aux_size: 0`, `value_head: false`, frozen stage-A encoder (`encoder-final.pt`, sha
`f36041bc…1923c`), seed 0, `config_bc_mixed_d10.yaml`, validation every 500 on the whole holdout,
and two checkpoints: the branchable WSD trunk at cooldown onset plus final. Closed-loop evaluation
uses only final.

| run | params | d_model | n_layer | cycles | epochs | state | dir |
|---|---:|---:|---:|---:|---:|---|---|
| `M-D20k-C20k-mixed-lr1e-4` | 12.86M | 512 | 4 | 20,000 | 33.5 | running | `runs/mixed_d10/M-D20k-C20k-mixed-lr1e-4` |
| `L-D20k-C20k-mixed-lr1e-4` | 25.13M | 640 | 5 | 20,000 | 33.5 | queued | `runs/mixed_d10/L-D20k-C20k-mixed-lr1e-4` |
| `XL-D20k-C20k-mixed-lr1e-4` | 42.89M | 768 | 6 | 20,000 | 33.5 | deferred to night | `runs/mixed_d10/XL-D20k-C20k-mixed-lr1e-4` |
| `XXL-D20k-C20k-mixed-lr1e-4` | 101.76M | 1024 | 8 | 20,000 | 33.5 | deferred to night | `runs/mixed_d10/XXL-D20k-C20k-mixed-lr1e-4` |
| `M-D20k-C40k-mixed-lr1e-4` | 12.86M | 512 | 4 | 40,000 | 67.0 | queued | `runs/mixed_d10/M-D20k-C40k-mixed-lr1e-4` |
| `L-D20k-C40k-mixed-lr1e-4` | 25.13M | 640 | 5 | 40,000 | 67.0 | queued | `runs/mixed_d10/L-D20k-C40k-mixed-lr1e-4` |
| `XL-D20k-C40k-mixed-lr1e-4` | 42.89M | 768 | 6 | 40,000 | 67.0 | deferred to night | `runs/mixed_d10/XL-D20k-C40k-mixed-lr1e-4` |
| `XXL-D20k-C40k-mixed-lr1e-4` | 101.76M | 1024 | 8 | 40,000 | 67.0 | deferred to night | `runs/mixed_d10/XXL-D20k-C40k-mixed-lr1e-4` |

Closed-loop evaluation uses both training starts: Spread
`win_level1_20260701015306_i371` and Laser `win_level1_20260630171218_i8`. These are
in-distribution / memorization probes, not cross-start generalisation. The Laser-only D10k/C20k
1e-4 controls in [0020](0020-exp-laser-model-scaling.md) are recipe-matched. A complete recipe-matched
four-size Spread-only control does not yet exist: 0018 has only L/XXL and used batch 16/C40k,
while 0015 used legacy releases and LR 3e-4. Spread comparisons must retain that limitation.

## 3. Evaluation metrics

Cross-entropy will come from `python tools/scaling_report.py runs/mixed_d10` on the shared
3,969-episode mixed holdout.

| cell | cycles | train CE | val CE best | @step | val CE final | source |
|---|---:|---:|---:|---:|---:|---|
| M | 20,000 | — | — | — | — | run not started |
| L | 20,000 | — | — | — | — | run not started |
| XL | 20,000 | — | — | — | — | run not started |
| XXL | 20,000 | — | — | — | — | run not started |
| M | 40,000 | — | — | — | — | run not started |
| L | 40,000 | — | — | — | — | run not started |
| XL | 40,000 | — | — | — | — | run not started |
| XXL | 40,000 | — | — | — | — | run not started |

Closed-loop success will report **final checkpoints only**, separately on the Spread and Laser
starts, using n = 100, T = 1.0, seed 0, 2x expert budget, bf16, batch 8 and Wilson 95% intervals.

| cell | cycles | Spread final | Spread Wilson 95% | Laser final | Laser Wilson 95% | source |
|---|---:|---:|---|---:|---|---|
| M | 20,000 | — | — | — | — | not requested |
| L | 20,000 | — | — | — | — | not requested |
| XL | 20,000 | — | — | — | — | not requested |
| XXL | 20,000 | — | — | — | — | not requested |
| M | 40,000 | — | — | — | — | not requested |
| L | 40,000 | — | — | — | — | not requested |
| XL | 40,000 | — | — | — | — | not requested |
| XXL | 40,000 | — | — | — | — | not requested |

## 4. Conclusion

_Pending — experiment not yet run._
