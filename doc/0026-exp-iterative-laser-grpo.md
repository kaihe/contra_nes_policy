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
| `L-D10k-C20k-laser-GRPO25-GRPO200` | `runs/grpo/2026-08-26/laser-projection-16-40-25/checkpoints/grpo-000025.pt` | 200 / 1.966 h | done | `runs/grpo/2026-08-27/laser-projection-stage2-09-29-52` |

The image encoder, frozen projection, L core, learned null goal, task, reward, and rollout
protocol remain unchanged from 0025. Save stage-two checkpoints every 25 updates and stop
if the ten-update mean KL from u25 reaches 0.10. The original u25 file remains immutable;
stage-two update numbers start at zero and do not continue stage one's optimizer, sampler,
or random-number-generator state.

## 3. Evaluation metrics

| closed-loop metric | stage-one u25 reference | stage-two u25 | u50 | u75 | u100 | u125 | u150 | u175 | u200 | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed-start success, n=100 with Wilson 95% interval | 49/100, 49% [39.4%, 58.7%] | 40/100, 40% [30.9%, 49.8%] | 46/100, 46% [36.6%, 55.7%] | 48/100, 48% [38.5%, 57.7%] | 46/100, 46% [36.6%, 55.7%] | **51/100, 51% [41.3%, 60.6%]** | 48/100, 48% [38.5%, 57.7%] | 47/100, 47% [37.5%, 56.7%] | 50/100, 50% [40.4%, 59.6%] | evaluation [0033](../../contra_nes_evaluation/doc/0033-result-iterative-laser-grpo.md) |
| death / timeout | 51 / 0 | 60 / 0 | 54 / 0 | 52 / 0 | 54 / 0 | **49 / 0** | 52 / 0 | 53 / 0 | 50 / 0 | evaluation 0033 |
| boss seen | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | evaluation 0033 |
| mean boss damage | 61.8% | 57.0% | 59.9% | 62.5% | 62.8% | **67.7%** | 65.3% | 63.3% | 63.9% | evaluation 0033 |
| mean episode steps | 79.7 | 77.8 | 77.4 | 76.0 | 77.4 | 78.3 | 81.0 | 79.3 | 75.6 | evaluation 0033 |
| action entropy | 2.456 bits | 2.463 bits | 2.416 bits | 2.434 bits | 2.474 bits | 2.470 bits | 2.423 bits | 2.421 bits | 2.372 bits | evaluation 0033 |

## 4. Conclusion

The additional 200 GRPO updates produced no improvement over the previous policy.
