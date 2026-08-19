# Grade failures by boss damage and wins by speed

Status: Accepted
Supersedes: —

Depends on [0004](0004-exp-grpo.md) and `kaihe/contra_nes_data#2`.

**Question.** Binary terminal reward wastes GRPO groups when every rollout succeeds or
every rollout fails. Can graded terminal rewards recover both tails without changing the
one-advantage-per-episode training contract?

**Answer.** Yes, in two independently measured phases. Grade failed boss episodes by
damage dealt, then grade successful episodes by completion speed. Success must always
outrank failure. Keep group advantage normalisation enabled: disabling it reduced boss
learning in the attempted confirmation. HP grading is implemented and reduces pooled
zero-variance groups from 0.514 to about 0.27, but its matched closed-loop confirmation
and the speed term remain undone.

## 1. Evidence

Binary reward discards groups whose outcomes agree because
`advantage = (reward - group_mean) / group_std` becomes zero. Across the initial GRPO
runs, 53–58% of rolled episodes were filtered this way:

| run | families | zero variance | episodes | used |
|---|---|---:|---:|---:|
| `2026-08-02/11-48-03` | 3 | 0.58 | 4,096 | 1,736 |
| `2026-08-02/15-22-32` | 3 | 0.53 | 30,080 | 14,176 |
| `2026-08-03/09-23-22` | 4, boss 50% | 0.53 | 30,080 | 14,096 |

Difficulty sampling cannot solve this alone: 12 of 14 easy-family labels have training
success above 0.84, so few tasks lie near the minimum-degeneracy rate of 0.5. Boss has
the opposite problem: it succeeds about 10% on train, and failures die near the
approach-to-engage transition while receiving the same reward as near-wins.

On identical BC-policy rollouts, HP grading restored every all-failure boss group without
affecting binary `kill` rewards:

| family | groups | binary zero variance | graded zero variance | success rate |
|---|---:|---:|---:|---:|
| boss | 6 × G=8 | 0.67 | 0.00 | 0.12 |
| kill | 3 × G=8 | 0.67 | 0.67 | 0.71 |

Two subsequent graded runs reduced pooled zero variance to 0.278 and 0.270, but stopped at
updates 77 and 61 and received no closed-loop evaluation. They do not satisfy the planned
matched confirmation against the 100-update control.

## 2. Design

Phase 1 grades only failed boss episodes:

```text
success: reward = 1.0
failure: reward = 0.5 * (boss_hp_start - boss_hp_end) / boss_hp_start
```

`boss_hp_start` is task metadata representing full-reveal HP. It cannot be the RAM value
at episode step 0, when the boss is not yet active, or each rollout's observed peak, which
would let an early death normalise against a smaller staged spawn. The reward remains one
terminal scalar; no RAM address or HP observation enters the policy.

Phase 2 grades successful episodes:

```text
success: reward = 1.0 - 0.1 * steps / budget
failure: phase-1 reward
```

The ranges remain disjoint: failures are in `[0, 0.5]`, successes in `[0.9, 1.0]`.
Budget normalisation handles weapon-dependent episode lengths and lets speed break ties
among winners without preferring a fast death.

Keep `normalise_advantages: true`. The unnormalised attempt
`runs/grpo/2026-08-03/18-54-05` discounted all-failure boss groups by roughly 10×:

| metric | graded, unnormalised | binary control |
|---|---:|---:|
| boss rollout success, updates 76–100 | 0.109 | 0.309 |
| fixed boss probe | 0.12 → 0.06 | — |
| `adv_abs_mean` | 0.197 | 0.829 |
| `kl_ref` | 0.016 | 0.030 |

That run changed grading and normalisation together, so it cannot attribute the loss to
grading. A standard-deviation floor is a separate future experiment.

## 3. Rejected alternatives

| proposal | reason rejected |
|---|---|
| symmetric step penalty | ranks a fast death above longer survival, reinforcing the boss failure mode |
| flat per-step penalty | budgets span roughly 180–776 decisions across weapons |
| per-step returns-to-go | breaks the one-scalar-per-episode GRPO contract without adding information |
| disable advantage normalisation | confounded attempt learned boss much worse; test a std floor separately |
| use rollout-observed peak HP | staged boss spawn lets early deaths receive an artificially small denominator |
| grade ordinary `kill` by HP | its degeneracy is on the success tail, handled by the speed term |
| increase group size | rollout cost grows linearly while tail degeneracy falls slowly |
| condition the policy on HP or weapon | reward shaping needs neither and should not change the input contract |

## 4. Risks and gates

| risk | gate |
|---|---|
| damage farming without winning | successes stay above 0.9; fixed-probe boss completion must exceed 10.5% |
| speed term trades wins for speed | `kill` and `traverse` fixed-probe success must not fall |
| partial credit inflates training success | report fixed-probe success and zero variance, not sampler-selected train success |
| HP sum includes surviving boss objects | compare only within a same-task group; group centring removes task scale |

## 5. Sequencing

1. Run the matched 100-update HP-grading confirmation: same BC start, 50% boss sampling,
   G, KL, learning rate, and normalisation as `2026-08-03/09-23-22`; change only
   `reward.progress_coef: 0.5`.
2. Confirm only if boss zero variance falls below 0.2 and held-out boss completion exceeds
   10.5%. Otherwise the recovered gradient is not useful enough to ship.
3. Generate diversified boss data through `kaihe/contra_nes_data#2`, independent of the
   reward result.
4. Implement and measure the speed term only after phase 1 is settled; gate overall zero
   variance below 0.30 without reducing easy-family fixed-probe completion.

## Appendix — provenance

| claim | source |
|---|---|
| zero variance and rollout waste | `runs/grpo/*/metrics.csv` |
| easy-family label success | `runs/grpo/2026-08-02/15-22-32/metrics.csv`, update 100 |
| boss baseline and completion | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` |
| binary/graded rollout comparison | six boss and three kill G=8 groups from BC `policy-final` |
| unnormalised result | `runs/grpo/2026-08-03/18-54-05` |
| boss HP semantics | `contra_nes_data/src/agent/boss_search.py` and its boss curriculum doc |
| zero-gradient contract | `tests/test_grpo.py::test_a_zero_advantage_batch_produces_no_policy_gradient` |
