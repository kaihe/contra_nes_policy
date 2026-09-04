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
| S32 scale | complete | 10 | 24 | 32 | up to 6 with early stopping | 64 | 90.57 min measured | `runs/alphazero/laser-terminal-v-i10-e24-s32` |

For the scale cell, split each update's 24 newly searched episodes into 18 training
episodes and six fixed validation episodes. Record loss before optimization and after
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
| S32 scale | 159/240 | 66.25% | 150/640 | 23.44% | 87.50% | 26.56% | `runs/alphazero/laser-terminal-v-i10-e24-s32/metrics.jsonl` |

| scale-up measurement | result | source |
|---|---|---|
| wall time | 90.57 min | final `elapsed_seconds` in `metrics.jsonl` |
| search throughput | 159.00 complete searched episodes/hour | 240 episodes and final elapsed time in `metrics.jsonl` |
| optimizer epochs | 49 total; updates 1, 2, and 8 stopped early | `epochs.jsonl` |
| mean validation policy CE, before to after | 0.25585 to 0.25520 | pre/post fields in `metrics.jsonl` |
| mean validation value BCE, before to after | 0.64140 to 0.61922 | pre/post fields in `metrics.jsonl` |
| mean validation progress error, before to after | 0.08478 to 0.08367 | pre/post fields in `metrics.jsonl` |
| paired raw-policy initialization | 15/64, 23.44%, 95% Wilson interval 14.75–35.13% | generation -1 in `metrics.jsonl` |
| paired raw-policy best | 17/64, 26.56%, 95% Wilson interval 17.30–38.48%, update 5 | generation 5 and promotion fields in `metrics.jsonl` |
| paired raw-policy final | 15/64, 23.44%, 95% Wilson interval 14.75–35.13% | generation 9 in `metrics.jsonl` |
| aggregate searched policy | 159/240, 66.25%, 95% Wilson interval 60.05–71.93% | all generation rows in `metrics.jsonl` |
| best searched update | 21/24, 87.50%, 95% Wilson interval 69.00–95.66%, update 6 | generation 6 in `metrics.jsonl` |
| search-imitation top-action agreement | not recorded | metric absent from `metrics.jsonl` and `epochs.jsonl` |
| terminal-value Brier score | not recorded | metric absent from `metrics.jsonl` and `epochs.jsonl` |
| checkpoint files | `best-policy.pt` and `final-policy.pt` | run directory listing |

## 4. Conclusion

_Pending — metrics collected, awaiting discussion._
