# Does the frame-difference token improve Laser boss policy success?

## 1. Goal

Data experiment 0019 sharply improved projectile Dice and reduced empty-frame projectile
FPR by replacing static RGB tokens with signed frame-difference tokens. The remaining
question is whether that representation improvement helps the causal policy win.

Compare one matched L/D10k/C20k candidate with the existing static-encoder Laser baseline.
Only the frozen 512-D image representation may change; this decides whether to adopt the
temporal encoder for later policy scaling.

## 2. Setup

Use the same ordered 40,000-episode Laser boss store and uid-digest holdout as policy
experiments 0020 and 0021. Carve validation from the full store first, then train on the
same first 18-shard D10k prefix: 9,771 training episodes, 1,020,299 training frames, and
1,993 validation episodes. Re-encode the entire store in a separate datahouse root so
episode identities, actions, shard boundaries, and splits remain matched while encoder
tokens cannot mix.

Common policy recipe: L core with `d_model=640`, 5 layers, 10 heads, 25.13M policy
parameters; batch 32; 20,000 cycles; AdamW at `1e-4`; weight decay 0.01; 500-step warmup;
WSD with cooldown from 18,000 to 20,000; bf16; dropout 0.2; frozen encoder; no value or
auxiliary head; seed 0; full validation every 500 cycles; final checkpoint retained.

| run | image state | encoder | data | cycles | state | dir |
|---|---|---|---:|---:|---|---|
| `L-D10k-C20k-laser-lr1e-4` | current RGB | static 512-D `f36041bc…1923c` | 9,771 | 20,000 | existing baseline | `runs/laser/L-D10k-C20k-laser-lr1e-4` |
| `L-D10k-C20k-laser-frame-diff-lr1e-4` | `[RGB(t), RGB(t)-RGB(t-1)]` | 512-D `6ebacbfa…ac670` | 9,771 | 20,000 | done, 18 min | `runs/laser-motion/L-D10k-C20k-laser-frame-diff-lr1e-4` |

The candidate datahouse uses native 224×240 consecutive frames. The first observation and
the separate goal image use zero delta; every later decision frame uses the preceding frame
from the same episode. Pairs never cross an episode boundary. The published policy checkpoint
must retain this temporal preprocessing contract so closed-loop evaluation derives identical
tokens from live RGB history.

## 3. Evaluation metrics

Report behavioral-cloning CE on the complete shared 1,993-episode holdout and closed-loop
Laser-start success for `policy-final.pt`. Closed-loop evaluation must reproduce evaluation
0024: `full_laser.state`, 100 rollouts, temperature 1.0, seed 0, twice the expert budget,
bf16, and batch 8. This remains an in-distribution memorization probe, not held-out-start
generalization.

| metric | static baseline | temporal candidate | source |
|---|---:|---:|---|
| best validation CE | 0.7009 at 2,000 | 0.7130 at 2,000 | policy `metrics.csv` / `tools/scaling_report.py` |
| final validation CE | 2.1465 | 2.1579 | same |
| final Laser success | 19/100, 19% [12.5, 27.8] | pending | evaluation 0024 and matched candidate run |
| mean boss damage fraction | 38.7% | pending | evaluation summary |

The primary gate is candidate final success above the baseline Wilson interval's 27.8%
upper bound. Also report the Wilson 95% interval, death/timeout counts, boss-seen rate,
mean damage fraction, and action entropy. CE is diagnostic and cannot substitute for
closed-loop success.

| recorded fact | source |
|---|---|
| static policy recipe, data counts, and CE | policy experiments 0020 and 0021; baseline run artifacts |
| static closed-loop result and protocol | evaluation result 0024 |
| temporal encoder result and input contract | data experiments 0018 and 0019 |
| candidate dataset identity | temporal datahouse catalog, encoder spec, and candidate `dataset.json` |

## 4. Conclusion

_Pending — closed-loop evaluation not yet run._
