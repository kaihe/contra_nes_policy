# Can regularisation or reweighted validation CE predict closed-loop play?

Status: Implemented
Supersedes: —
Depends on: [0009](0009-exp-boss-data-scaling.md) (the D1–D8 runs this measures), [0006](0006-design-action-only-base-policy.md) (fixed recipe)

**Question.** Validation CE bottoms at step 3,000 and then triples by step 20,000, so the
action-only policy is plainly overfitting. Does regularizing that away improve closed-loop
play — and can a reweighted CE be made into an offline proxy that tracks play, so the
project stops steering by a number that disagrees with its own evaluation?

**Answer.** **No to both, and the second is measured.** The checkpoint at the CE minimum
plays **12.9 pp worse** than the fully overfit final (52.6% vs 65.5% pooled on 846,
non-overlapping CIs). The proposed fix — *tail CE*, the same cross-entropy restricted to
the ~22% of steps whose target is not the modal `R` — **is a constant multiple of total
CE**: the ratio sits at 2.10–2.29 across both ends of training and a 4× data range, so it
carries no independent information whatsoever. Held-out imitation loss degrades
*uniformly* across the action distribution while play improves. That kills not just tail CE
but every reweighting of CE over the expert distribution, and BC budget should not be spent
on the offline-CE axis.

**The dropout sweep ran anyway** (§5). Best cell is **0.2 at 69.0% pooled**, +3.5 pp over
the control — a soft positive that this doc predicted would not happen, so the prediction
is recorded as failed in §3. It does not rescue the offline proxy: no cell cleared the
`Δ ≤ 0.4` gate, the dose response is incoherent (0.3 has the lowest Δ *and* the worst
play), the p-value is uncorrected across three comparisons, and all three new runs
reproduce the constant tail/total ratio to three significant figures.

---

## 1. Why — the evidence

### The model overfits, hard

Validation CE by step, from `metrics.csv` of the 0009 cells (`val_every: 1000`, 60-batch
subset):

| step | D1 | D8 |
|---:|---:|---:|
| 1 000 | 0.759 | 0.765 |
| **3 000** | **0.713** | **0.703** |
| 10 000 | 0.971 | 0.881 |
| 20 000 | 2.100 | 1.754 |

Train CE at step 20,000 is **0.051** (D1) and **0.078** (D8) with `dropout: 0.0`. The
12.86M core memorizes the training set and val CE rises to 3× its minimum. This is a
variance problem, not a capacity problem — which is why §4 rejects scaling the model up.

### Less overfitting plays worse

D8 step 3,000 (`policy-003000.pt` = `policy-best.pt`, the run's CE minimum) against D8
final on the frozen 846 suite:

| checkpoint | val CE | pooled 846 | 95% CI | death | timeout | progress |
|---|---:|---:|---|---:|---:|---:|
| D8 3k (CE-optimal) | **0.703** | **52.6%** | [49.2, 55.9] | 37% | 11% | 0.849 |
| D8 final (overfit) | 1.754 | **65.5%** | [62.2, 68.6] | 23% | 12% | 0.910 |

−12.9 pp on n = 846 with non-overlapping intervals. `steps_vs_expert` is unchanged
(1.021 vs 1.024) — the CE-optimal policy is not slower, it dies partway.

### It is not a sampling-sharpness artifact

The 3k model is less decisive (rollout action entropy 1.061 vs 0.893 nats; 2.89 vs 2.44
effective actions), which invites "hedging at 20 Hz is jitter, and jitter dies."
Temperature 0 is the causal test — same weights, only sampling changed:

| policy | T | entropy | pooled 846 | death |
|---|---:|---:|---:|---:|
| GPT BC final | 1.0 | 0.912 | 67.5% | 23% |
| GPT BC final | 0.0 | 0.686 | 66.0% | 25% |
| GRPO u075 | 1.0 | 0.954 | 71.6% | 20% |
| GRPO u075 | 0.0 | 0.720 | 69.4% | 21% |

Sharpening a fixed model is worth **−1.5 to −2.2 pp**, not −12.9. Entropy is a symptom,
not the cause.

### The proposed mechanism, and its refutation

The hypothesis was: validation CE is a frequency-weighted average and **78% of steps are
`R`**, so it is dominated by frames every policy handles. Continued training memorizes,
raising total CE, while the model keeps improving on the *rare* decisions survival depends
on. If true, CE restricted to non-`R` targets would keep falling where total CE rises.

Measured on the whole validation set (28,273 non-modal steps per checkpoint,
`tools/tail_ce.py`):

| cell | val CE | tail CE | **tail / total** | pooled 846 |
|---|---:|---:|---:|---:|
| D8 3k | 0.7091 | 1.6244 | **2.291** | 52.6% |
| D8 10k | 0.8995 | 2.0582 | **2.288** | — |
| D8 final | 1.7615 | 3.7041 | **2.103** | 65.5% |
| D1 final | 2.1305 | 4.8529 | **2.278** | 65.2% |
| D2 final | 1.9294 | 4.1376 | **2.145** | 64.5% |
| D4 final | 1.8223 | 3.8406 | **2.108** | 67.1% |

**The hypothesis is false, twice over.**

1. **Tail CE rises with total CE**, 1.62 → 2.06 → 3.70 over the D8 run — it more than
   doubles where it was predicted to fall.
2. **The ratio is constant at 2.10–2.29** across both ends of training and a 4× data
   range. Tail CE is total CE times a constant; it carries no independent information.
   The rare frames degrade at essentially the same rate as the common ones.

At matched training compute — the four finals, all 20,000 steps, all fully annealed — tail
CE spans 1.15 nats while pooled spans 2.6 pp with no ordering (r = −0.41 on n = 4, and the
pooled spread is itself inside evaluation noise).

**Independently replicated by the dropout sweep.** The three regularized runs were trained
after this finding and reproduce the ratio without being fitted to it:

| cell | tail/total at the CE minimum | tail/total at step 20,000 |
|---|---:|---:|
| dropout 0.1 | 2.384 | **2.254** |
| dropout 0.2 | 2.438 | **2.240** |
| dropout 0.3 | 2.440 | **2.243** |

Nine measurements now span two data scales, four dropout rates and three training
positions, and the ratio never leaves 2.10–2.44.

### What that generalizes to

The model does not get worse on average frames while improving on rare ones. It gets worse
at predicting held-out expert actions **uniformly across the action distribution** while
closed-loop play improves 12.9 pp.

So the problem is not *which* frames the loss emphasises. **No reweighting of
cross-entropy over the expert distribution can yield a proxy that tracks play**, because
every slice of it moves the same wrong way. That retires the whole family at once: class
weights (`action_class_weights`, already in `loss.py`), focal-style weighting, label
smoothing, per-family CE — each is a reweighting of a quantity measured to be uniformly
uninformative here. Fitting held-out expert actions is simply not what closed-loop
survival measures.

This is also the general form of the failure [0003](0003-design-grpo-code-layout.md),
[0005](0005-design-graded-reward.md) and evaluation 0011 §4 each hit separately.

### Correcting evaluation 0011

Evaluation [0011](../../contra_nes_evaluation/doc/0011-boss-data-scaling.md) §4 cites
"offline full-val CE falls D1→D8 (2.13 → 1.76)" and calls it a proxy that improved without
translating. **At the CE minimum the two runs are 0.713 and 0.703** — a 0.01 nat
difference. Eight times the boss data bought essentially no generalization; it changed the
*rate* of memorization, so the runs differ at step 20,000 only because they sit at
different points on their own overfitting curves. The proxy never improved, so "proxy
improved, completion did not" does not describe that experiment. A handoff issue on the
evaluation repo is owed.

## 2. What was built

`tail_ce` ships despite the negative result, because the negative result is what it
measured and re-deriving it later would cost another six validation passes.

| piece | where |
|---|---|
| `tail_ce_metrics(ce, target, mask, modal_action)` | `src/contra_policy/loss.py` |
| `MODAL_ACTION`, derived from `ACTION_NAMES.index("R")` | `src/contra_policy/train_bc.py` |
| `_weighted_tail` — aggregates by non-modal step count, not by batch | `src/contra_policy/train_bc.py` |
| `tools/tail_ce.py` — scores existing checkpoints, no retraining | `tools/` |
| 7 tests, incl. one pinning that logging it leaves the loss bit-identical | `tests/test_tail_ce.py` |

Two properties worth keeping:

- It is computed under `no_grad` and never enters the returned loss, so a run that logs it
  optimizes bit-identically to one that does not. Cells stay comparable to the existing
  dropout-0.0 control.
- It aggregates weighted by non-modal step count. Those counts cluster by family, so the
  unweighted batch mean `_mean_of` produces is not the tail CE of the validation set.

`tools/tail_ce.py` takes shard selection from each checkpoint's own stored `train_config`,
so a checkpoint is always scored on the validation set its run actually used. It reproduces
the D8 run's own `val_full` at step 20,000 exactly (1.7615), which is what validates the
measurement.

## 3. How the registered predictions resolved

Recorded before the runs, in the style of [0004](0004-exp-grpo.md) §4:

| prediction | outcome |
|---|---|
| tail CE falls monotonically 3k → 20k on the existing D8 checkpoints | **falsified** — it rises 1.62 → 2.06 → 3.70 |
| R1/R2 land within ±3 pp of control; R3 flat or negative | **mixed** — 0.1 at +2.5 pp and 0.3 at +0.1 pp were right; **0.2 at +3.5 pp was outside the band** |
| dropout 0.1 fails the Δ ≤ 0.4 gate | **confirmed** — Δ = 0.789 |
| no cell exceeds GRPO u075's 71.6% | **confirmed** — best cell 69.0% |

Two confirmed, one mixed, one falsified — and the falsified one was load-bearing. The
mechanism in §1 was wrong, and the stance it implied ("regularization cannot help because
it acts on an anti-correlated proxy") was too strong: dropout 0.2 produced a soft positive
this doc argued against. What survives is the narrower, better-measured claim — offline CE
does not *predict* play, which is a statement about steering, not about whether
regularization can ever help.

## 4. What was rejected, and why

**Tail CE as an offline proxy.** Measured dead: constant 2.1–2.3× ratio to total CE. Do not
propose "weight the loss toward the decisions that matter" again without first explaining
why every slice of held-out CE degraded together here.

**Scaling the model up.** The obvious next step after 0009, and wrong: train CE is already
0.051. Capacity reduces bias, and there is no bias left to reduce — more parameters at
fixed data widen the gap. If the capacity question needs closing, the cheap decisive test
is the *opposite* direction: one ~3M-parameter cell (`n_layer: 2` or half `d_model`) on D4.
If a 4× smaller model matches closed-loop, capacity is ruled out for ~40 minutes.

**More boss data.** Answered by 0009 / evaluation 0011: pooled D1→D8 is +0.2 pp (p = 0.95),
boss stays in a 0–15% band at ~90% death. §1 strengthens that null — the data did not even
improve generalization at the CE minimum.

**Shorter training / early stopping.** Measured, not argued: 52.6% pooled. The existing 3k
checkpoint is also snapshotted at 96% of peak LR (cosine is only 12.8% through decay at
step 3,000), so a properly annealed 3,000-step run is a *different* model and would score
higher — but it would have to recover 12.9 pp to reach a checkpoint already in hand.

**Temperature sharpening.** Measured: −1.5 to −2.2 pp at T = 0 for both BC and GRPO.

**Boss-only evaluation of any of this.** 57 tasks, 2–4 successes, ±5 pp RNG noise. Read
boss off the full-suite runs.

## 5. The dropout sweep, and what it returned

Run 2026-08-06 on D8, seed 0, 20,000 steps, `policy.core.dropout` the only change.
Closed-loop from evaluation [0012](../../contra_nes_evaluation/doc/0012-d8-dropout-sweep.md):

| cell | min val CE | final val CE | **Δ** | band | pooled 846 | vs control | p |
|---|---:|---:|---:|---|---:|---:|---:|
| 0.0 control | 0.703 | 1.754 | 1.051 | — | 65.5% | — | — |
| 0.1 | 0.708 | 1.496 | **0.789** | partial | 68.0% | +2.5 pp | 0.17 |
| **0.2** | 0.708 | 1.368 | **0.660** | partial | **69.0%** | **+3.5 pp** | **0.050** |
| 0.3 | 0.705 | 1.288 | **0.583** | partial | 65.6% | +0.1 pp | 1.0 |

Dropout engaged monotonically. **No cell reached `Δ ≤ 0.4`**, so by this doc's own gate
every cell is "partial" and none is a clean test of "would curing overfit help play."

### Dropout probably cannot clear the gate at this placement

The Δ decrements are **0.262 → 0.129 → 0.077**, roughly halving each step. Fitted
geometrically that asymptotes at **Δ ≈ 0.47–0.51** — above the 0.4 threshold. Evaluation
0012 §6.2 recommends escalating to rate 0.4–0.5 to clear the gate; on this placement that
will not work, it will just underfit. Dropout reaches only attention weights and the
SwiGLU output (`causal.py:135`, `:149`) — nothing on the residual stream, nothing on the
embeddings, and the encoder is frozen. Clearing `Δ ≤ 0.4` needs **broader placement**, an
architecture change, not a larger number. Three points fitted geometrically, so treat it
as a strong hint; it is enough that the 0.4/0.5 cells are not worth funding as specified.

### How much to believe 0.2

Not much, as a finding; quite a lot, as a checkpoint.

- **Three comparisons, best p = 0.050 uncorrected.** Family-wise error under the null is
  `1 − 0.95³ ≈ 14%`; a Bonferroni threshold would be 0.0167.
- **The dose response is incoherent.** Δ falls monotonically 0.789 → 0.660 → 0.583 while
  pooled goes 68.0 → 69.0 → **65.6**. The most-regularized cell is the worst and sits
  exactly at the control. A real regularization effect should not collapse like that.
- **Boss is untouched**: 5.3–8.8% on the 846 suite, 4.0–8.5% on 0012's 200-resample probe,
  with the same short-survival and low-damage histograms at every rate.
- **Single seed**, finals only.

So dropout 0.2's 69.0% is the best *point estimate* in the project for pure BC and a
reasonable RL init — [0011](0011-exp-boss-grpo.md) §2 gates it on a seed-1 replication before
spending 10 GPU-hours on it — but it is not evidence that regularization is the lever, and
it does not reopen the offline-CE axis.

## 6. What would reopen this

| finding | what it would overturn |
|---|---|
| an offline metric correlating with pooled across ≥ 6 checkpoints spanning ≥ 10 pp | §1's claim that no offline proxy works; would restore cheap steering |
| a properly annealed short run beating 65.5% | the "early stopping is measured dead" line in §4 |
| a ~3M cell matching D4 closed-loop | closes the capacity question; makes the rejected model-size sweep formally dead rather than argued |
| dropout at Δ ≤ 0.4 beating control by ≥ 3 pp | §5's reading of the sweep — but it needs broader dropout placement, since rate alone asymptotes at Δ ≈ 0.5 |
| dropout 0.2's +3.5 pp replicating at a second seed | §5's "not a finding"; would make regularization a live axis again |

## 7. What is next, and it is not this

Nothing in this doc addresses the **91% boss death rate**, which survives every
intervention tried: BC at four data scales, all three checkpoint positions, both
temperatures, and GRPO. The lever with an actual mechanism is the graded boss-HP reward —
data side is done (`KillBossMaker.boss_hp`, `boss_hp_start` in boss JSON) and the request
is open at `kaihe/contra_nes_policy#1`, with the design already in
[0005](0005-design-graded-reward.md). It converts ~91% of boss rollouts from a zero into a graded
signal, and it needs its own doc.

---

## Appendix — provenance

| claim | source |
|---|---|
| val CE curves, train CE, Δ | `runs/bc/2026-08-05/{12-03-04,15-49-28}/metrics.csv`, rows with `phase=val` |
| fixed recipe, `dropout: 0.0`, family draws, save steps | `runs/bc/2026-08-05/15-49-28/resolved_config.yaml` |
| tail CE, all six checkpoints, 28,273 non-modal steps | `python -m tools.tail_ce runs/bc/2026-08-05/{12-03-04,13-19-43,14-36-21,15-49-28}/checkpoints/policy-*.pt` (2026-08-06) |
| D8 3k pooled 52.6%, death 37%, entropy, `progress` | `contra_nes_evaluation/runs/0805-boss-scale-D8-003000-full/report.json` |
| D8 final pooled 65.5%; D1/D2/D4 finals 65.2/64.5/67.1% | `contra_nes_evaluation/runs/0805-boss-scale-D{1,2,4,8}-final-full/report.json` |
| 78% `R` action share | same reports, `overall.action_distribution` |
| T = 0 vs T = 1 for BC and GRPO | `contra_nes_evaluation/runs/{0801-gpt-bc-final,0802-gpt-bc-final-t0,0803-grpo-000075,0802-grpo-000075-t0}/report.json` |
| boss-only 57 is ±5 pp noise; D1→D8 pooled +0.2 pp p = 0.95 | `contra_nes_evaluation/doc/0011-boss-data-scaling.md` §2, §3 |
| step time ~103 ms, eval wall clock 5.5 min | `metrics.csv` `step_ms`; `report.json` `meta.wall_clock` |
| dropout sweep Δ, min/final val CE, tail/total ratios | `runs/bc/2026-08-06/dropout-{0.1,0.2,0.3}/metrics.csv`, rows with `phase=val` |
| dropout closed-loop, McNemar, boss probe | `contra_nes_evaluation/doc/0012-d8-dropout-sweep.md` §2–§4; handoff `kaihe/contra_nes_evaluation#4` |
| boss-HP accessor ready | `kaihe/contra_nes_policy#1` |

Action entropy above is `-Σ p log p` over `overall.action_distribution`, i.e. the entropy
of actions *taken during rollout*, not the model's per-frame predictive entropy.
