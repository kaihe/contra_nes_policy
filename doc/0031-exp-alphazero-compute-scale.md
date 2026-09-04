# Does a one-to-two-hour AlphaZero run convert more search compute into policy improvement?

## 1. Goal

The 16- and 32-simulation prototypes show that additional MCTS compute improves the
searched policy, but neither run reliably transfers that gain to the raw network. The
next run tests whether scaling current-generation samples and optimizer work produces
measurable, well-behaved policy distillation within a one-to-two-hour prototype budget.

The scale-up is successful only if within-update training and validation losses decline,
raw-policy performance improves over its pre-training baseline, and throughput remains
stable. This decides whether to scale the AlphaZero loop further or debug its policy and
value targets before spending more compute.

## 2. Setup

Common to all cells: fixed Laser boss state `win_level1_20260630171218_i8`, GPT-policy
initialization from `L-D10k-C20k-laser-null-goal-proj-tuned/checkpoints/policy-final.pt`,
terminal-success value from 0030, iteration-local samples, bf16, AdamW at `2e-6`, batch of
two complete episodes, seed 400, and stochastic root-visit action selection.

| run | status | updates | episodes/update | simulations/action | epochs/update | raw eval/update | target wall time | dir |
|---|---|---:|---:|---:|---:|---:|---:|---|
| S16 prototype | complete | 10 | 16 | 16 | 2 | 16 | not logged | `runs/alphazero/laser-terminal-v-i10-e16-s16` |
| S32 prototype | complete | 10 | 16 | 32 | 2 | 16 | not logged | `runs/alphazero/laser-terminal-v-i10-e16-s32` |
| S32 scale | planned | 10 | 32 | 32 | up to 8 with early stopping | 64 | 1–2 h | `runs/alphazero/laser-terminal-v-i10-e32-s32` |

For the planned cell, split each update's 32 newly searched episodes into 24 training
episodes and eight fixed validation episodes. Record loss before optimization and after
every epoch. Stop an update when validation policy loss fails to improve for two epochs;
do not reuse episodes from earlier updates. Evaluate the unmodified initialization before
update zero and evaluate every candidate with the same 64 action-sampling seeds.

Save no per-generation checkpoints. Maintain `best-policy.pt` only when the candidate's
paired raw-policy result passes the declared promotion gate, write it atomically, and save
one `final-policy.pt` after update nine. Keep `metrics.jsonl`, the resolved configuration,
per-epoch loss rows, validation summaries, elapsed time, episode counts, and RNG seeds.
The wall-time limit stops generation cleanly after the current episode or optimizer step;
it does not leave a partial checkpoint.

## 3. Evaluation metrics

| completed cell | search wins | search win rate | raw-policy wins | raw-policy win rate | best update search | best update raw | source |
|---|---:|---:|---:|---:|---:|---:|---|
| S16 prototype | 81/160 | 50.63% | 29/160 | 18.13% | 81.25% | 37.50% | `runs/alphazero/laser-terminal-v-i10-e16-s16/metrics.jsonl` |
| S32 prototype | 99/160 | 61.88% | 31/160 | 19.38% | 87.50% | 31.25% | `runs/alphazero/laser-terminal-v-i10-e16-s32/metrics.jsonl` |
| S32 scale | pending | pending | pending | pending | pending | pending | `runs/alphazero/laser-terminal-v-i10-e32-s32/metrics.jsonl` |

| scale-up gate | measurement | pass condition | source |
|---|---|---|---|
| compute utilization | searched episodes and MCTS simulations per wall-clock hour | no unexplained throughput decline across updates | resolved config and per-update timing log |
| policy optimization | training and validation visit-target cross-entropy by epoch | both decline within updates; validation does not systematically regress | per-epoch loss log |
| search imitation | top-action agreement and probability assigned to MCTS-selected action | post-update exceeds pre-update on held-out roots | validation summary |
| terminal value | validation binary cross-entropy and Brier score | improves over the constant-rate baseline | validation summary |
| policy improvement | paired raw-policy wins before training and at each accepted update | best candidate exceeds initialization; report count and Wilson interval | fixed-seed raw evaluation log |
| search improvement | MCTS-enhanced wins per update | report count and Wilson interval separately from raw policy | `metrics.jsonl` |
| stability | best-to-final raw-policy change | final does not regress because failed candidates are not promoted | promotion log |
| auxiliary progress | masked boss-progress validation error | declines without worsening policy validation loss | validation summary |
| storage | checkpoint files | only `best-policy.pt` and `final-policy.pt` exist | run directory listing |

## 4. Conclusion

_Pending — experiment not yet run._
