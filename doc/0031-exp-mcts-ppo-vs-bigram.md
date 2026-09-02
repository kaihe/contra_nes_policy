# Does PPO guide Laser MCTS better than the mc_search bigram?

## 1. Goal

Design 0030 uses the image-conditioned PPO u158 policy for both PUCT priors and terminal
rollouts. The cheaper `mc_search` bigram uses only the previous action and legal-action mask,
so it is the relevant non-visual search control.

Compare the two complete search guides under the same simulation budget. Use PPO for search
data generation only if its committed win rate justifies its wall-clock cost; otherwise use
the bigram control or revisit the search budget before policy distillation.

## 2. Setup

Common to both cells: exact `full_laser.state` task `win_level1_20260630171218_i8`; 30
complete episodes with seeds 0–29; binary terminal value (`success=1`, `death/timeout=0`);
216-decision limit; 16 new simulations per committed decision; `c_puct=1.5`; rollout
temperature 1.0; most-visited commit with fixed action-ID tie break; no root noise or
recovery; Stable-Retro transitions; version-3 records from design 0030.

| run | tree prior | terminal rollout | action support | initialization | state | dir |
|---|---|---|---|---|---|---|
| `PPO-PPO-S16-n30` | masked PPO probabilities | PPO sampling | policy 21-action table | PPO u158 `runs/ppo/2026-08-27/laser-critic-14-56-52/checkpoints/ppo-final.pt` | planned | `runs/mcts/0031/ppo-ppo-s16` |
| `BIGRAM-BIGRAM-S16-n30` | masked Level-1 bigram row | bigram sampling | published Level-1 15-action table | `contra_nes_data/src/agent/priors/level1.yaml` | planned | `runs/mcts/0031/bigram-bigram-s16` |

The bigram artifact was counted from 525 winning Level-1 traces containing 652,722 actions.
Its trimmed action table is part of the published `mc_search` setup, so this is an end-to-end
guide comparison rather than a probability-only ablation. Both cells apply the same stateful
fire/jump legal-mask rules before renormalizing their own action support.

Episode timing starts after checkpoint/prior, task, emulator, and initial root setup, and ends
at committed success, death, or timeout. It includes policy/prior evaluation, PUCT, temporary
rollouts, emulator steps, commit, and re-root; initialization and JSON writing are reported
separately. Equal-simulation quality and measured wins per search-hour answer the accuracy and
throughput questions without adding mixed-prior cells.

## 3. Evaluation metrics

| cell | committed wins / n | win rate with Wilson 95% interval | death | timeout | mean committed steps | source |
|---|---:|---:|---:|---:|---:|---|
| PPO / PPO | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | `runs/mcts/0031/ppo-ppo-s16` summary |
| Bigram / Bigram | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | `runs/mcts/0031/bigram-bigram-s16` summary |

| cell | mean seconds / episode | median | p90 | search hours | wins / search-hour | initialization seconds | source |
|---|---:|---:|---:|---:|---:|---:|---|
| PPO / PPO | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | same run summary |
| Bigram / Bigram | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | _Pending_ | same run summary |

| cell | new simulations | temporary emulator decisions | mean legal actions / root | mean committed visit share | source |
|---|---:|---:|---:|---:|---|
| PPO / PPO | _Pending_ | _Pending_ | _Pending_ | _Pending_ | same run summary |
| Bigram / Bigram | _Pending_ | _Pending_ | _Pending_ | _Pending_ | same run summary |

## 4. Conclusion

_Pending — experiment not yet run._
