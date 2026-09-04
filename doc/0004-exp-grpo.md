# Can GRPO train the policy and improve boss completion?

## 1. Goal

GRPO replaces the critic with a group baseline, but equal-outcome groups produce zero
gradient. The first question is whether the stack improves `kill`, `item`, and `traverse`
without regressing `item`; the second is whether sparse boss success leaves enough usable
groups to improve `boss`.

Two phases separate those questions. Phase 1 tests the machinery on the three easier
families; phase 2 restarts from the same BC checkpoint with all four families. The decision
is whether GRPO is a useful policy-training recipe and whether boss progress should come
from GRPO configuration or from better boss data.

## 2. Setup

Common to the completed phases: start from `policy-final.pt` (67.5% pooled validation),
use same-task grouped rollouts and reference KL, and evaluate fixed checkpoints through
`contra_nes_evaluation`. Phase 2 restarts from BC rather than continuing phase 1, avoiding
three-family drift as a confound. Per-family training rates from the difficulty sampler are
not fixed-distribution estimates; the fixed probe is the comparable training metric.

| run | families | group size | purpose | state | dir |
|---|---|---:|---|---|---|
| preliminary phase 1 | kill, item, traverse | 4 | end-to-end rollout check | invalid after 8 updates: group-id, metric-key, and memory bugs | `runs/grpo/2026-08-02/11-48-03` |
| phase 1 | kill, item, traverse | 4 | stack movement and item retention | completed | `runs/grpo/2026-08-02/15-22-32` |
| phase 2 | kill, item, traverse, boss | 8 for boss | boss movement from matched BC start | completed | `runs/grpo/2026-08-03/09-23-22` |

The preregistered checks were kill reaching roughly 75–80%, item staying within 2 points,
boss zero-variance groups near 0.75 at G=8, and boss not improving materially (defined as
remaining at or below 10%). Phase 1's historical comparison was ROCKET PPO's per-family
gain; phase 2 was compared with phase 1 at matched updates and with the shared BC start.

## 3. Evaluation metrics

| policy | pooled val | kill val | item val | boss val | source |
|---|---:|---:|---:|---:|---|
| BC start | 67.5% | 66.9% | 66.7% | 3.5% | `contra_nes_evaluation/doc/0005-gpt-bc.md` |
| GRPO phase 1 | 70.6% | — | 66.7% | excluded | `contra_nes_evaluation/doc/0006-grpo.md` |
| GRPO phase 2 | 71.6% | 74.4% at u075 | 72.2% | 10.5% final | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` |
| ROCKET BC reference | 72.8% | — | — | 8.8% | evaluation baseline cited by `0006` and `0008` |

| prediction/diagnostic | expected | measured | source |
|---|---:|---:|---|
| kill validation | 75–80% | 74.4% | `contra_nes_evaluation/doc/0008-grpo-with-boss.md`, phase-2 u075 |
| item change from BC | within ±2 pp | 0.0 pp phase 1; +5.5 pp phase 2 | `contra_nes_evaluation/doc/0006-grpo.md`, `0008-grpo-with-boss.md` |
| boss zero-variance groups, G=8 | ~0.75 | 0.43 | `runs/grpo/2026-08-03/09-23-22` |
| boss validation completion | ≤10% | 10.5% | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` |
| easy-family zero-variance groups | <0.30 stop threshold | 0.50–0.67, mean 0.59 | `runs/grpo/2026-08-02/11-48-03/metrics.csv`; diagnostic only |

| boss diagnostic | value | source |
|---|---:|---|
| validation deaths | 55/57 | `contra_nes_evaluation/runs/0801-gpt-bc-final/episodes.csv` |
| median death step | 40 (14% of budget) | `contra_nes_evaluation/runs/0801-gpt-bc-final/episodes.csv` |
| distinct source traces | 466/466 train tasks | boss train shard `src_trace` |
| recorded weapon metadata | none | episode JSON keys |

## 4. Conclusion

_Pending — metrics collected, awaiting discussion._
