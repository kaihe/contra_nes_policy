# Regularize with dropout only as a gated probe; replace val CE with tail CE

Status: Proposed
Supersedes: —
Depends on: [0009](0009-boss-data-scaling.md) (D8 run and its curve), [0006](0006-action-only-base-policy.md) (fixed recipe)

**Question.** Validation CE bottoms at step 3,000 and then triples by step 20,000, so
the action-only policy is plainly overfitting. Does regularizing that away — dropout
0.1/0.2/0.3 on the D8 cell — improve closed-loop play?

**Answer.** Probably not, and the doc says so before the runs: the checkpoint at the CE
minimum plays **12.9 pp worse** on the full 846 suite than the fully overfit final one
(52.6% vs 65.5%, non-overlapping CIs), so validation CE is *anti-correlated* with
closed-loop success across this range. Run the three cells anyway — it is two hours and
the prediction is falsifiable — but gate every cell on whether the overfit rise actually
shrank, and treat the real deliverable as the **replacement offline proxy**: validation
CE restricted to frames where the expert action is not the modal `R`. Total val CE is
frequency-weighted by a 78%-`R` action distribution and therefore measures the wrong
thing.

---

## 1. Why — the evidence

### The model overfits, hard

Validation CE by step, from `metrics.csv` of the 0009 cells (`val_every: 1000`):

| step | D1 | D8 |
|---:|---:|---:|
| 1 000 | 0.759 | 0.765 |
| **3 000** | **0.713** | **0.703** |
| 10 000 | 0.971 | 0.881 |
| 20 000 | 2.100 | 1.754 |

Train CE at step 20,000 is **0.051** (D1) and **0.078** (D8) with `dropout: 0.0`. The
12.86M core memorizes the training set and val CE rises to 3× its minimum. This is a
variance problem, not a capacity problem.

### Correcting 0009 and evaluation 0011

Evaluation [0011](../../contra_nes_evaluation/doc/0011-boss-data-scaling.md) §4 cites
"offline full-val CE falls D1→D8 (2.13 → 1.76)" and calls it a proxy that improved
without translating. **At the CE minimum the two runs are 0.713 and 0.703** — a 0.01 nat
difference. Eight times the boss data bought essentially *no* generalization; what it
changed was the *rate* of memorization, so the runs differ at step 20,000 only because
they sit at different points on their own overfitting curves. The proxy never improved,
so "proxy improved, completion did not" does not describe this experiment.

### Less overfitting plays worse

D8 step 3,000 (`policy-003000.pt`, `= policy-best.pt`, the run's CE minimum) evaluated
against D8 final on the frozen 846 suite:

| checkpoint | val CE | pooled 846 | 95% CI | death | timeout | progress |
|---|---:|---:|---|---:|---:|---:|
| D8 3k (CE-optimal) | **0.703** | **52.6%** | [49.2, 55.9] | 37% | 11% | 0.849 |
| D8 final (overfit) | 1.754 | **65.5%** | [62.2, 68.6] | 23% | 12% | 0.910 |

−12.9 pp on n = 846 with non-overlapping intervals. The CE-optimal checkpoint is not
marginally worse; it is a different quality of policy. `steps_vs_expert` is unchanged
(1.021 vs 1.024) — it is not slower, it dies partway.

### It is not a sampling-sharpness artifact

The 3k model is measurably less decisive (rollout action entropy 1.061 vs 0.893 nats;
2.89 vs 2.44 effective actions), which invites the explanation "hedging at 20 Hz is
jitter, and jitter dies." Temperature 0 is the causal test of that — same weights, only
sampling changed:

| policy | T | entropy | pooled 846 | death |
|---|---:|---:|---:|---:|
| GPT BC final | 1.0 | 0.912 | 67.5% | 23% |
| GPT BC final | 0.0 | 0.686 | 66.0% | 25% |
| GRPO u075 | 1.0 | 0.954 | 71.6% | 20% |
| GRPO u075 | 0.0 | 0.720 | 69.4% | 21% |

Sharpening a fixed model is worth **−1.5 to −2.2 pp**, not −12.9. So entropy is a
symptom, not the cause, and "make the policy more confident" is not the lever.

### The mechanism that fits

Validation CE is a frequency-weighted average over held-out expert frames, and **78% of
those frames are `R`** (`action_distribution` of the 846 run). By step 3,000 the model
predicts common frames well — that is the CE minimum. Continued training memorizes
training episodes, so total val CE rises, while the model keeps improving on the *rare*
decision points: jump timing, dodges, the frame where holding `R` kills you. Survival
depends almost entirely on those. The model gets worse on the average frame and better
at staying alive.

This predicts the outcome of the sweep: dropout regularizes toward the frequency-weighted
average, and the average is not what is failing.

## 2. The design

### Cells

Three new runs on **D8** (`boss_scaling.shard_count: 8`), so no dropout effect is
confounded with data starvation. Only `model.core.dropout` changes.

| cell | dropout | steps | run |
|---|---:|---:|---|
| control | 0.0 | 20 000 | existing `runs/bc/2026-08-05/15-49-28` |
| R1 | 0.1 | 20 000 | new |
| R2 | 0.2 | 20 000 | new |
| R3 | 0.3 | 20 000 | new |

Everything else is bit-identical to 0009: seed 0, `lr 3e-4`, cosine over 20,000 with 500
warmup, `weight_decay 0.01`, `batch_size 4`, `family_draws` (kill 2290 / item 455 /
traverse 3693 / boss 666), `save_steps [3000, 10000]`. At ~103 ms/step that is ~34 min
of training and 5.5 min of evaluation per cell — **~2 hours total**.

### Metrics, per cell

1. **Minimum val CE** and the step it occurred at.
2. **Final val CE** at step 20,000.
3. **Overfit rise `Δ = CE(20k) − CE(min)`.** Control is Δ = 1.051 (D8), 1.387 (D1).
4. **Pooled success on the full 846 suite** at the final checkpoint, plus death.
5. **Tail CE** (below) at every `val_every` tick.

### The engagement gate

A cell is interpretable only if the regularizer did what regularizers do. Read Δ before
reading anything else:

| Δ at 20k | reading |
|---|---|
| ≤ 0.4 | engaged; its pooled 846 number is a real test of "less overfitting → better play" |
| 0.4 – 0.8 | partial; report, do not conclude from it alone |
| ≥ 0.8 | **void cell** — dropout did not bite at this rate. Raise the rate; do not record "dropout does not help" |

Without this gate a flat pooled result is ambiguous between "regularization does not help
closed-loop" (interesting) and "dropout 0.1 on a four-layer core does nothing"
(not interesting).

### Decision rule

Primary comparison is pooled 846 at final versus the control's 65.5%, paired McNemar on
the 846 shared `uid`s. n = 846 resolves roughly ±3–4 pp; the 57-task boss subset resolves
nothing (±5 pp of pure RNG noise — see 0011 §2), so read boss off the full-suite run and
never from a separate `--kinds boss` eval.

| observation | decision |
|---|---|
| an engaged cell beats control by ≥ 3 pp, p < 0.05 | overfitting *was* binding; adopt that rate and re-examine the CE proxy |
| all engaged cells within ±3 pp | regularization is not the lever; close this branch and do not revisit dropout, weight decay or augmentation on the CE axis |
| engaged cells regress ≥ 3 pp | confirms the anti-correlation; the CE axis is actively misleading and 0006's recipe should keep `dropout: 0.0` |
| no cell reaches Δ ≤ 0.4 | inconclusive on regularization; report as a null instrumentation result |

### Tail CE — the durable deliverable

Define **tail CE** as validation cross-entropy restricted to frames whose expert action is
not the modal action `R`, computed on the same frozen full-val shard alongside total val
CE, at every `val_every` tick.

If the mechanism in §1 is right, tail CE keeps *falling* from 3k to 20k while total val CE
triples. Compute it first on the **existing** D8 checkpoints (3k / 10k / final) — a few
minutes of forward passes, no training — because those three already have closed-loop
numbers to correlate against (52.6% / — / 65.5%).

A proxy that tracks closed-loop where total CE inverts replaces a 5.5-minute evaluation
with a validation pass, and every experiment after this one gets cheaper to steer. That is
a larger payoff than the dropout answer itself, and the three sweep cells supply further
points to validate it against.

### Registered predictions

Recorded before the runs, in the style of [0004](0004-grpo-experiment-plan.md) §4:

| prediction | resolves against |
|---|---|
| R1/R2 land within ±3 pp of control; R3 is flat or negative | pooled 846 |
| dropout 0.1 fails the Δ ≤ 0.4 gate | Δ per cell |
| tail CE falls monotonically 3k → 20k on the existing D8 checkpoints | tail CE table |
| no cell exceeds GRPO u075's 71.6% | pooled 846 |

## 3. What was rejected, and why

**Scaling the model up.** The obvious next step after 0009, and wrong: train CE is already
0.051. Capacity reduces bias, and there is no bias left to reduce — more parameters at
fixed data widen the gap. If the capacity question needs closing, the cheap decisive test
is the *opposite* direction: one ~3M-parameter cell (`n_layer: 2` or half `d_model`) on
D4. If a 4× smaller model matches closed-loop, capacity is ruled out for ~40 minutes.

**More boss data.** Answered by 0009 / evaluation 0011: pooled D1→D8 is +0.2 pp (p = 0.95)
and boss stays in a 0–15% band at ~90% death. §1 above strengthens that null — the data
did not even improve generalization at the CE minimum.

**Shorter training / early stopping.** Measured, not argued: 52.6% pooled. Note the
existing 3k checkpoint is also snapshotted at 96% of peak LR (cosine is only 12.8% through
decay at step 3,000), so a properly annealed 3,000-step run is a *different* model and
would score higher. It is not worth running: it would have to recover 12.9 pp to reach
parity with a checkpoint already in hand.

**Temperature sharpening.** Measured: −1.5 to −2.2 pp at T = 0 for both BC and GRPO.

**Dropout 0.4.** On a four-layer core it will mostly underfit, and an underfit cell cannot
distinguish the two hypotheses this sweep exists to separate. Add it only if 0.3 trends
positive.

**Boss-only evaluation of the sweep.** 57 tasks, 2–4 successes, ±5 pp RNG noise. It would
reproduce exactly the uninterpretable table 0011 §2 already produced.

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate |
|---|---|---|
| dropout does not engage at any tested rate | four layers, `d_model` small, weight decay already 0.01 | Δ ≤ 0.4; if unmet, escalate the rate rather than conclude |
| a cell wins on noise | single seed, ±3–4 pp resolution at n = 846 | paired McNemar p < 0.05, not point estimates |
| tail CE is dominated by one rare action | `URJ`, `LF` are < 0.3% of frames | report tail CE per action bucket as well as pooled |
| conclusions leak into checkpoint choice | 0011 §6 forbids selecting on closed-loop | cells, rates and the Δ gate are fixed by this doc before any run |
| the whole CE axis is a dead end | already 12.9 pp of evidence for that | this doc's predictions table; if all four resolve as written, close the branch |

## 5. Sequencing

1. Compute **tail CE** on the existing D8 3k / 10k / final checkpoints. No training. If it
   does not invert relative to total val CE, the §1 mechanism is wrong and this doc's
   framing needs revision before spending the two hours.
2. Add `tail_ce` to the BC validation loop next to the existing val CE, with a test that
   masks a known frame set and checks the restricted mean.
3. Run R1/R2/R3. Record min val CE, final val CE, Δ.
4. Evaluate the three finals on the full 846 suite; apply the Δ gate, then the decision
   rule.
5. Update this doc's `Status` and resolve the predictions table. If the outcome is the
   predicted null, say so plainly and close the regularization branch.

**Not blocked on another repo.** The independent boss lever — graded boss-HP reward — is
already unblocked at `kaihe/contra_nes_policy#1` (data side done: `KillBossMaker.boss_hp`,
`boss_hp_start` in boss JSON) and belongs in its own doc, not this one. Nothing here
addresses the 91% boss death rate; this sweep is about the offline proxy.

---

## Appendix — provenance

| claim | source |
|---|---|
| val CE curves, train CE, Δ | `runs/bc/2026-08-05/{12-03-04,15-49-28}/metrics.csv`, rows with `phase=val` |
| fixed recipe, `dropout: 0.0`, family draws, save steps | `runs/bc/2026-08-05/15-49-28/resolved_config.yaml` |
| D8 3k pooled 52.6%, death 37%, entropy, `progress` | `contra_nes_evaluation/runs/0805-boss-scale-D8-003000-full/report.json` |
| D8 final pooled 65.5% | `contra_nes_evaluation/runs/0805-boss-scale-D8-final-full/report.json` |
| 78% `R` action share; 846 = 454 traverse / 281 kill / 57 boss / 54 item | `contra_nes_evaluation/runs/0805-boss-scale-D4-final-full/report.json` (`overall.action_distribution`, `kinds`) |
| T = 0 vs T = 1 for BC and GRPO | `contra_nes_evaluation/runs/{0801-gpt-bc-final,0802-gpt-bc-final-t0,0803-grpo-000075,0802-grpo-000075-t0}/report.json` |
| boss-only 57 is ±5 pp noise; D1→D8 pooled +0.2 pp p = 0.95 | `contra_nes_evaluation/doc/0011-boss-data-scaling.md` §2, §3 |
| step time ~103 ms, eval wall clock 5.5 min | `metrics.csv` `step_ms`; `report.json` `meta.wall_clock` |
| boss-HP accessor ready | `kaihe/contra_nes_policy#1` |

Action entropy above is `-Σ p log p` over `overall.action_distribution`, i.e. the entropy
of actions *taken during rollout*, not the model's per-frame predictive entropy.
