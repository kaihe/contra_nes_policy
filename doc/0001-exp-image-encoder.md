# Can one image token preserve the spatial information of four?

## 1. Goal

The policy's six tokens per decision make task-length context expensive: a 32-decision
window contains only 2.1% of training tasks, while the median boss task lasts 281
decisions. At 1,024 decisions, the old layout requires 6,144 tokens and 0.805 GB of KV
memory at batch 16.

This experiment tests whether a 512-dimensional image token can retain entity structure
as well as the four-token encoder. If it can, the policy will use
`[interaction, goal_token, image_token × N]`, reducing that context to 1,026 tokens and
0.134 GB of KV memory, with goal matching left to temporal attention.

## 2. Setup

Both encoders trained for 20,000 steps on the full validation set of 85,054 frames. Agent
and goal frames share the same convolutional trunk and entity supervision. The 16.6M-parameter
goal-agnostic model maps each image to one 512-dimensional token, from which it decodes
32×32 heatmaps for player, player bullets, enemies, and enemy bullets.

The entity head reads only the token, forcing spatial information through the bottleneck.
A 1×1 convolution reduces 1,024 channels to 256 before flattening, and entity targets use
a 6 px sigma. Reconstruction is implemented but disabled with `reconstruct: false`.

| run | conditioning | params removed | steps | validation | dir |
|---|---|---:|---:|---|---|
| goal-conditioned baseline | FiLM from goal token | — | 20,000 | full, 782 batches | `runs/encoder/2026-07-31/11-20-50/` |
| goal-agnostic | none; goal matching downstream | 3.65M | 20,000 | `phase=val_full` | `runs/encoder/2026-07-31/18-00-11/` |

## 3. Evaluation metrics

| model/checkpoint | player dice | player-bullet dice | enemy dice | enemy-bullet dice | source |
|---|---:|---:|---:|---:|---|
| goal-conditioned, step 20,000 | 0.984 | 0.634 | 0.960 | 0.910 | `runs/encoder/2026-07-31/11-20-50/`, full validation |
| goal-agnostic, step 5,000 | 0.977 | 0.660 | 0.960 | 0.923 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |
| goal-agnostic, step 20,000 | 0.99 | 0.70 | 0.98 | 0.97 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |

| goal-agnostic frame set | player dice | player-bullet dice | enemy dice | enemy-bullet dice | source |
|---|---:|---:|---:|---:|---|
| goal frames | 0.99 | 0.70 | 0.98 | 0.98 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |

| family | enemy dice | enemy-bullet dice | source |
|---|---:|---:|---|
| kill | 0.98 | 0.96 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |
| item | 0.97 | 0.93 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |
| traverse | 0.98 | 0.97 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |
| boss | 0.99 | 0.97 | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full` |

| diagnostic | value | source |
|---|---:|---|
| boss `point_err_px`, `n_goal_points == 1` | 0.43 px | full validation sweep; 128 eligible frames |
| boss mean target components per frame | 4.57 | full validation sweep; 10,128 frames |
| boss multi-component frames | 98.7% | full validation sweep; 10,128 frames |

## 4. Conclusion

The one-token image encoder is good enough.
