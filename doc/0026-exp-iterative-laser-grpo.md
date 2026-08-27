# Can a second GRPO stage improve the 49% Laser policy?

## 1. Goal

Corrected binary-reward GRPO raised fixed-start Laser success from 25% to 49% at update
25, then regressed as the policy moved farther from its BC reference
([0025](0025-exp-laser-projection-grpo.md)). The project currently prioritizes proving
that the complete visual-policy-post-training stack can master this one task.

Promote the validated u25 policy to both initialization and frozen reference, then test
whether a lower-rate second GRPO stage can exceed 49%. Extend iterative post-training only
if a saved checkpoint improves the matched closed-loop result.

## 2. Setup

Common configuration: `src/contra_policy/rl/laser_projection_stage2.yaml`; seed 0; bf16;
the exact `full_laser.state` task; binary reward without shaping; temperature 1.0; group
size 8; 16 usable groups per update; one epoch; minibatch 8 episodes; AdamW at `2e-6`;
clip ratio 0.2; reference KL coefficient 0.02; entropy coefficient 0.01; and target KL
0.02. Both the trainable policy and frozen reference load stage-one u25, resetting the
stage-local reference KL to zero.

| run | policy and reference initialization | stage-two updates / wall clock | state | dir |
|---|---|---:|---|---|
| `L-D10k-C20k-laser-GRPO25-GRPO200` | `runs/grpo/2026-08-26/laser-projection-16-40-25/checkpoints/grpo-000025.pt` | 200 / 2.5 h maximum | planned | `runs/grpo/<launch-date>/laser-projection-stage2-<launch-time>` |

The image encoder, frozen projection, L core, learned null goal, task, reward, and rollout
protocol remain unchanged from 0025. Save stage-two checkpoints every 25 updates and stop
if the ten-update mean KL from u25 reaches 0.10. The original u25 file remains immutable;
stage-two update numbers start at zero and do not continue stage one's optimizer, sampler,
or random-number-generator state.

## 3. Evaluation metrics

| closed-loop metric | stage-one u25 reference | stage-two u25 | u50 | u75 | u100 | u125 | u150 | u175 | u200 | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed-start success, n=100 with Wilson 95% interval | 49/100, 49% [39.4%, 58.7%] | pending | pending | pending | pending | pending | pending | pending | pending | evaluation [0032](../../contra_nes_evaluation/doc/0032-result-laser-projection-grpo.md) for reference; matched evaluation handoff for stage two |
| death / timeout | 51 / 0 | pending | pending | pending | pending | pending | pending | pending | pending | same matched evaluation |
| mean boss damage | 61.8% | pending | pending | pending | pending | pending | pending | pending | pending | same matched evaluation |
| action entropy | 2.456 bits | pending | pending | pending | pending | pending | pending | pending | pending | same matched evaluation |

## 4. Conclusion

_Pending — experiment not yet run._
