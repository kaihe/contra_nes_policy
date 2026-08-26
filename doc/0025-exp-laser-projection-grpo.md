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
| `L-D10k-C20k-laser-proj-frozen-GRPO100` | `runs/laser-projection/L-D10k-C20k-laser-null-goal-proj-frozen/checkpoints/policy-final.pt` | canonical `boss` task `win_level1_20260630171218_i8` | 1 win, 0 death/timeout; no shaping | 100 / 1.5 h maximum | planned | `runs/grpo/<launch-date>/laser-projection-<launch-time>` |

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

| experiment metric | init | u25 | u50 | u75 | u100 | source |
|---|---:|---:|---:|---:|---:|---|
| fixed-start success, n=100 with Wilson 95% interval | 25% [17.5%, 34.3%] | pending | pending | pending | pending | evaluation 0030 for init; matched evaluation handoff for GRPO checkpoints |
| death / timeout | 74 / 1 | pending | pending | pending | pending | same matched evaluation |
| mean boss damage | 47.1% | pending | pending | pending | pending | same matched evaluation |
| fixed training-probe success, n=32 | pending | pending | pending | pending | pending | run `metrics.csv`, `probe/success_rate` |
| zero-variance group fraction | pending | pending | pending | pending | pending | run `metrics.csv`, `collect/zero_variance_group_frac` |
| reference KL | 0 | pending | pending | pending | pending | run `metrics.csv`, `train/kl_ref` |
| policy entropy | pending | pending | pending | pending | pending | run `metrics.csv`, `train/entropy` |

## 4. Conclusion

_Pending — experiment not yet run._
