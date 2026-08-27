# Can binary-reward GRPO improve the frozen-projection Laser policy?

## 1. Goal

The frozen-projection L/D10k/C20k policy wins 25/100 rollouts from the published Laser
evaluation start, while adapting the encoder projection and doubling BC compute did not
improve it ([0024](0024-exp-unfrozen-laser-encoder.md)). The remaining question is whether
closed-loop optimization can improve this policy without changing its visual representation.

Run a bounded binary-reward GRPO feasibility cell on that exact start. Continue with a
broader RL experiment only if success rises without uncontrolled reference-policy drift;
otherwise do not spend a longer budget on this single-start recipe.

## 2. Setup

Common configuration: `src/contra_policy/rl/laser_projection.yaml`; seed 0; bf16; rollout
temperature 1.0; group size 8; 16 usable groups per update; one optimization epoch;
minibatch 8 episodes; AdamW at `5e-6`; policy clipping 0.2; reference KL coefficient 0.02;
entropy coefficient 0.01; and per-update target KL 0.02. Dropout is disabled during both
collection and optimization so stored behavior log-probabilities and current-policy
log-probabilities describe the same policy.

| run | initialization | task data | reward | updates / wall clock | state | dir |
|---|---|---|---|---:|---|---|
| `L-D10k-C20k-laser-proj-frozen-GRPO100` | `runs/laser-projection/L-D10k-C20k-laser-null-goal-proj-frozen/checkpoints/policy-final.pt` | canonical `boss` task `win_level1_20260630171218_i8` | 1 win, 0 death/timeout; no shaping | 87 / 0.918 h; KL-guard stop | done | `runs/grpo/2026-08-26/laser-projection-16-40-25` |

The task is the same `full_laser.state` start used by evaluations 0030 and 0031. The
policy retains the frozen convolutional trunk and projection, L core (`d_model=640`, five
layers, ten heads), and learned null goal from BC. No legacy goal-image shard is loaded.
Save checkpoints after updates 25, 50, 75, and 100; exact resume must restore policy,
optimizer, sampler, difficulty history, and all random-number generator states. Stop early
if the rolling reference KL reaches 0.10. A fixed probe runs at update 1 and every ten
updates with 32 repeats of the same task.

## 3. Evaluation metrics

| pre-run measurement | value | source |
|---|---:|---|
| BC success on the exact Laser start | 25/100, 25% [17.5%, 34.3%] | evaluation [0030](../../contra_nes_evaluation/doc/0030-result-laser-projection-adaptation.md), frozen static baseline |
| one-update smoke success | 5/16, 31.25% | real-emulator smoke run `/tmp/contra-grpo-update-otu0xd`, update 1 collection |
| smoke zero-variance group fraction | 0/2, 0% | same smoke run |
| smoke policy ratio mean | 0.999893 | same smoke run |
| smoke approximate behavior KL | 0.000190 | same smoke run |
| smoke reference KL | 0.0000445 | same smoke run |

| closed-loop metric | init | u25 | u50 | u75 | u87 final | source |
|---|---:|---:|---:|---:|---:|---|
| fixed-start success, n=100 with Wilson 95% interval | 25/100, 25% [17.5%, 34.3%] | **49/100, 49% [39.4%, 58.7%]** | 45/100, 45% [35.6%, 54.8%] | 38/100, 38% [29.1%, 47.8%] | 40/100, 40% [30.9%, 49.8%] | evaluation [0032](../../contra_nes_evaluation/doc/0032-result-laser-projection-grpo.md) |
| death / timeout | 74 / 1 | **51 / 0** | 55 / 0 | 62 / 0 | 60 / 0 | evaluation 0032 |
| boss seen | 100% | 100% | 100% | 100% | 100% | evaluation 0032 |
| mean boss damage | 47.1% | **61.8%** | 61.4% | 55.8% | 58.8% | evaluation 0032 |
| mean episode steps | 78.9 | 79.7 | 79.2 | 75.0 | 78.7 | evaluation 0032 |
| action entropy | 2.304 bits | 2.456 bits | 2.371 bits | 2.392 bits | 2.448 bits | evaluation 0032 |

| training metric | u25 collection | u50 collection | u75 collection | u87 collection | source |
|---|---:|---:|---:|---:|---|
| rollout success | 42.97% (55/128) | 46.88% (90/192) | 49.22% (63/128) | 39.84% (51/128) | run `metrics.csv`; collection precedes the numbered optimizer update |
| reference KL | 0.0371 | 0.0759 | 0.1001 | 0.1225 | run `metrics.csv`, `kl_ref` |
| behavior approximate KL | 0.000270 | 0.000417 | 0.000333 | 0.000380 | run `metrics.csv`, `approx_kl` |
| entropy | 0.2961 nats | 0.2921 nats | 0.2824 nats | 0.3036 nats | run `metrics.csv`, `entropy` |
| raw zero-variance groups | 0/16 | 1/24 | 0/16 | 0/16 | run `metrics.csv`, `groups_drawn` and `collect/zero_variance_group_frac` |
| elapsed wall clock | 0.293 h | 0.573 h | 0.810 h | 0.918 h | run `metrics.csv`, `elapsed_hours` |
| ten-update mean reference KL at stop | — | — | — | 0.1008 | run `metrics.csv`, updates 78–87; guard threshold 0.100 |

## 4. Conclusion

Fixing the previous GRPO bugs proves that post-training is effective: it raised the Laser
boss-fight win rate from 25% to 49%. However, as the policy drifted from the reference
policy, the win rate appeared to decline, reaching 40% at the final update.
