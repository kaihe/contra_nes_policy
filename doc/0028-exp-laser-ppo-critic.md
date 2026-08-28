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
| `L-D10k-C20k-laser-GRPO25-PPO200` | stage-one `grpo-000025.pt` | 512 train + 128 fresh validation rollouts; three head-only epochs per 128-roll chunk; actor frozen | 158 / 1.597 h; KL-guard stop | done | `runs/ppo/2026-08-27/laser-critic-14-56-52` |

Warmup must achieve positive validation explained variance and lower Brier score than the
constant train-success predictor before actor updates begin. Save checkpoints every 25
updates; stop at ten-update reference KL 0.10. Checkpoints include value head, optimizer,
RNG, elapsed-time, and completed warmup metrics for exact post-gate resume. The task is
fixed rather than sampled. The actor uses complete episodes with no reward-dependent
filtering; evaluation ignores the value head.

## 3. Evaluation metrics

| critic gate | constant | warmup final train chunk | warmup validation | source |
|---|---:|---:|---:|---|
| Brier score | 0.2854 | 0.2126 | **0.2260** | run `metrics.csv`; fresh 128-roll validation is compared with the training-success constant |
| explained variance | 0 | 0.102 | **0.122** | same PPO warmup metrics |

| closed-loop metric | u25 reference | PPO u25 | u50 | u75 | u100 | u125 | u150 | u158 final | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fixed-start success, n=100 with Wilson 95% interval | 49/100, 49% [39.4%, 58.7%] | 40/100, 40% [30.9%, 49.8%] | 54/100, 54% [44.3%, 63.4%] | 43/100, 43% [33.7%, 52.8%] | 50/100, 50% [40.4%, 59.6%] | 52/100, 52% [42.3%, 61.5%] | 52/100, 52% [42.3%, 61.5%] | **59/100, 59% [49.2%, 68.1%]** | evaluation [0034](../../contra_nes_evaluation/doc/0034-result-laser-critic-ppo.md) |
| death / timeout | 51 / 0 | 60 / 0 | 46 / 0 | 57 / 0 | 50 / 0 | 48 / 0 | 48 / 0 | **41 / 0** | evaluation 0034 |
| boss seen | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | evaluation 0034 |
| mean boss damage | 61.8% | 56.8% | 67.8% | 60.7% | 65.8% | 63.8% | 66.3% | **72.5%** | evaluation 0034 |
| mean episode steps | 79.7 | 76.9 | 81.5 | 76.7 | 78.2 | 77.1 | 79.1 | 79.1 | evaluation 0034 |
| action entropy | 2.456 bits | 2.435 bits | 2.454 bits | 2.450 bits | 2.461 bits | 2.480 bits | 2.465 bits | 2.474 bits | evaluation 0034 |

| training window | rollout success | reference KL | critic explained variance | value Brier | source |
|---|---:|---:|---:|---:|---|
| updates 1–20 | 36.4% | 0.0034 | 0.099 | 0.221 | run `metrics.csv`, arithmetic mean over the window |
| updates 81–100 | 48.0% | 0.0560 | 0.117 | 0.216 | same run metrics |
| updates 141–158 | 54.4% | 0.0979 | 0.097 | 0.210 | same run metrics |

## 4. Conclusion

_Pending — metrics collected, awaiting discussion._
