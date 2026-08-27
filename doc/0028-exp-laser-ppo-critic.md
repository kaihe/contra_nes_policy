# Can a learned critic improve the 49% Laser policy?

## 1. Goal

Corrected GRPO raised fixed-start Laser success to 49%, but a second 200-update stage
changed the policy without improving it ([0026](0026-exp-iterative-laser-grpo.md)). Its
one episode-level advantage cannot distinguish useful actions from incidental actions in
the same winning trajectory.

Test the state-value critic and GAE specified by [0027](0027-design-ppo-critic.md) while
keeping the visual policy, task, and binary reward fixed. Continue actor–critic work only
if the critic beats a constant predictor and a saved PPO checkpoint exceeds 49% in matched
closed-loop evaluation.

## 2. Setup

Common configuration: `src/contra_policy/rl/laser_ppo.yaml`; stage-one GRPO u25 initializes
the actor and frozen reference; a neutral value head initializes at 0.5; exact
`full_laser.state` task; seed 0; bf16; binary terminal reward; `gamma=1.0`; GAE
`lambda=0.95`; rollout temperature 1.0; 128 complete episodes per update; one epoch;
minibatch 8 episodes; actor LR `2e-6`; value-head LR `1e-4`; PPO clip 0.2; value
coefficient 0.5; reference KL coefficient 0.02; entropy coefficient 0.01.

| run | initialization | critic warmup | updates / wall clock | state | dir |
|---|---|---|---:|---|---|
| `L-D10k-C20k-laser-GRPO25-PPO200` | stage-one `grpo-000025.pt` | 512 train + 128 fresh validation rollouts; three head-only epochs per 128-roll chunk; actor frozen | 200 / 2.5 h maximum | running | `runs/ppo/2026-08-27/laser-critic-14-56-52` |

Warmup must achieve positive validation explained variance and lower Brier score than the
constant train-success predictor before actor updates begin. Save checkpoints every 25
updates; stop at ten-update reference KL 0.10. Checkpoints include value head, optimizer,
sampler, RNG, elapsed-time, and warmup state for exact resume. The actor uses complete
episodes with no reward-dependent filtering; evaluation ignores the value head.

## 3. Evaluation metrics

| critic gate | constant | warmup final train chunk | warmup validation | source |
|---|---:|---:|---:|---|
| Brier score | 0.2854 | 0.2126 | **0.2260** | run `metrics.csv`; fresh 128-roll validation is compared with the training-success constant |
| explained variance | 0 | 0.102 | **0.122** | same PPO warmup metrics |

| closed-loop metric | u25 reference | PPO u25 | u50 | u75 | u100 | u125 | u150 | u175 | u200 | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed-start success, n=100 with Wilson 95% interval | 49/100, 49% [39.4%, 58.7%] | pending | pending | pending | pending | pending | pending | pending | pending | evaluation [0032](../../contra_nes_evaluation/doc/0032-result-laser-projection-grpo.md) for reference; matched evaluation handoff for PPO |
| death / timeout | 51 / 0 | pending | pending | pending | pending | pending | pending | pending | pending | same matched evaluation |
| mean boss damage | 61.8% | pending | pending | pending | pending | pending | pending | pending | pending | same matched evaluation |
| action entropy | 2.456 bits | pending | pending | pending | pending | pending | pending | pending | pending | same matched evaluation |

## 4. Conclusion

_Pending — experiment in progress._
