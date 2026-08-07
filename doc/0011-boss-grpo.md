# Settle whether boss is RL-solvable: ten hours of graded-reward GRPO, boss only

Status: Implemented
Supersedes: [0008](0008-fourteen-hour-grpo.md)
Depends on: [0005](0005-graded-reward.md) (the reward, already built), [0004](0004-grpo-experiment-plan.md) (the stack)

**Question.** Boss has resisted every intervention: 8× data, all three checkpoint
positions, both temperatures, a dropout sweep, and 100 updates of sparse-reward GRPO. It
sits at 3.5–8.8% with ~91–96% death throughout. **Is boss solvable by RL at all** with
this architecture and a dense reward — or is something else the binding constraint?

**Answer.** Spend 10 hours finding out, boss-only, and treat the result as a feasibility
verdict rather than a product change. This is a **capability question, not a metric
question**: what the other three families do while it runs is out of scope, recorded once
at the end and never used as a gate. The setup is not new machinery — `progress_coef` is
in `rollout.py`, the matched configuration is fixed by 0005 §2, and the two runs that used
it died at updates **61 and 77** against the control's 100, defunded to pay for a data
campaign that measured null. Boss is the only family where grading has leverage: **75–79%**
of boss rollouts deal non-zero HP damage against **4.0–8.5%** that succeed, so ~61% of
boss groups currently contribute nothing at G = 8.

The verdict is read on **two axes — train-start success and held-out success** — because
"RL can fit boss" and "RL produces a boss-general policy" are different claims and only
the first is a feasibility proof. A negative needs a *plateau*, not merely a low endpoint.

**Measured outcome (2026-08-06/07).** Ran the full 10 h → **1,619 updates** at 22.2 s/update
(3.4× this doc's ~480 estimate; boss episodes are short because the policy dies fast).
**The 2×2 lands in the low/low cell, and all three §4 risks fired.**

- **Train:** probe boss flat at ~4.3–5.8% for ~600 updates against a 3.1% init, then
  collapse to ~0.1%. The single u540 probe of 11.5% is noise — neighbours read 1.0–8.3%
  and the u490–620 mean is 4.6%. Do not cite it.
- **Held-out** (eval [0013](../../contra_nes_evaluation/doc/0013-boss-only-grpo-precollapse.md)):
  u525/u550/u575 at 7.5 / 11.0 / 7.5% against an 8.5% BC init — u550's edge is 0.84σ and
  its neighbours sit below the init. Full 846 on u550 is 64.3% (−4.7 pp, p = 0.005), the
  expected specialist cost.
- **How it collapsed:** cumulative `kl_ref` 0.017 → **0.245** with no bound on total drift
  (the per-update `target_kl` fired on 1,500 of 1,619 updates and was therefore no guard);
  entropy 0.875 → 0.407; and **chip-without-kill** — damage per failed episode 11.9% →
  13.9% while wins went 4.9% → 0.1%.
- **The finding that outlived the run:** weapon stratification of the held-out probe shows
  **~40% of boss tasks are unwinnable** (Regular/Flamethrower: 0 wins in 316 rollouts) and
  every boss success ever recorded came from Spread or Laser. The graded reward also
  **defeated the difficulty sampler** — hopeless groups gained damage spread, survived
  filtering, and stopped being detectable as hopeless. [0012](0012-spread-grpo.md) is the
  successor experiment: binary reward, Spread + rapid only.

---

## 1. Why — the evidence

### The reward works on the metric it was built for

`collect/zero_variance_group_frac`, the fraction of rollout budget producing no gradient:

| run | `progress_coef` | updates | mean zero-variance |
|---|---|---:|---:|
| `runs/grpo/2026-08-03/09-23-22` (control) | none | 100 | **0.514** |
| `runs/grpo/2026-08-04/09-45-33` | 0.5 | 77 | **0.278** |
| `runs/grpo/2026-08-04/18-57-23` | 0.5 | 61 | **0.270** |

51% → 27%. 0005 §5 pre-registered failure at > 0.4; this passes with room. The mechanism
is not in doubt.

### Both graded runs were cut short, and we know why

0009 §4 records it: *"The attempted 0008 run was stopped at update 61 to fund data
generation."* That data campaign became 0009/0011-eval — D1→D8, pooled **+0.2 pp**
(p = 0.95), boss unchanged, val CE floor moved 0.01 nats.

So the only lever with a working mechanism was defunded to pay for the one that measured
null, and it was never evaluated closed-loop at all. The best result in the project,
**71.6% pooled** (eval 0008, from `runs/grpo/2026-08-03/09-23-22`), belongs to the
*ungraded* control.

### Boss is where grading has leverage

From eval 0012 §4's 200-resample probe (200 draws from the 57 val boss tasks):

| signal | share of boss rollouts carrying gradient |
|---|---:|
| sparse success | **4.0–8.5%** |
| graded HP damage | **75–79%** (all but the zero-damage 21–25%) |

At ~6% success and G = 8, `0.94⁸ ≈ 0.61` of boss groups are all-failure and contribute
exactly zero under a binary reward. Under grading the observed damage spread makes a
degenerate group rare. No other family has that gap — traverse, kill and item already sit
at 66–75% and their groups carry signal.

### The failure is early death, not weak damage

Also from 0012 §4: median boss episode is **37–41 decision steps ≈ 1.9–2.1 s** at 20 Hz,
against a boss expert mean of **177.7 steps ≈ 8.9 s**. More than half of all attempts end
in the first `[19, 48)`-step bin. The policy is not losing long fights; it is dying before
the fight starts. Damage grading rewards exactly the margin that is missing.

### RL is not training on the four search states

`rl/tasks.py` joins tasks to shards on `(family, uid)`: **6,904 train tasks = 6,438
non-boss + 466 published boss**. The 2,034 four-state search episodes are a BC-only
release (`boss-mixed-v2`) and never enter an RL worker. The start-coverage concern that
applies to BC boss data does not apply here — RL rolls out 466 distinct published starts.

## 2. The design

### Fixed configuration

Everything is the 0005 §2 matched configuration. `progress_coef` and boss share are the
only departures from the `09-23-22` control.

| component | value | why |
|---|---|---|
| init | best available BC base (see below) | RL compounds from a better start |
| `reward.progress_coef` | **0.5** | 0005 §2; ranges stay disjoint below 0.5 |
| `normalise_advantages` | **true** | matches control; 0005 records that flipping this *and* grading together failed and had to be discarded |
| `rollout.group_size` | 8 | unchanged |
| boss rollout share | **1.0** | this document's variable |
| budget | **10 hours wall-clock**, resumable | not a fixed update count |
| KL, lr, clip, `minibatch_episodes` | unchanged from `09-23-22` | one variable at a time |

### Budget arithmetic

Measured 51–60 s/update at 50% boss share. Boss episodes run ~1.8× the average length
(177.7 vs 100.5 expert steps), so at 100% boss expect **~75 s/update → ~480 updates in 10
hours**. Plan the run wall-clock-bounded with true resumption (0008's requirement, which
survives into this doc); do not hard-code an update count that a slower rollout would
silently truncate.

At ~301 episodes/update ≈ 37 groups, 480 updates visits each of the 466 boss tasks about
**38 times**. That is the run's main scientific risk, gated in §4.

### The base checkpoint

Two candidates, and the choice is pre-registered rather than made after looking:

1. **dropout 0.2 final, 69.0%** — best BC point estimate, but +3.5 pp at p = 0.050
   uncorrected across three comparisons (family-wise ≈ 14%), one seed, and its dose
   response is incoherent (dropout 0.3 has lower Δ and worse play).
2. **D4 final, 67.1%** — the 0009 sweep's best, similar pedigree to the 67.5% the control
   started from.

**Gate:** re-run dropout 0.2 at seed 1 (~40 min). If pooled lands within 2 pp of 69.0%,
use it. Otherwise use D4 final. Spending 10 GPU-hours on an unreplicated 3.5 pp is not a
trade worth making.

### Measurement

Primary and secondary are both boss; the guardrail is the rest of the suite.

| metric | source | cadence | role |
|---|---|---|---|
| **held-out boss success + HP-damage histogram** | eval 0012 §4 probe, 200 resamples from the 57 val tasks, seed 0 | every probe | primary |
| **train-start boss success** | training probe over the 466 published starts | continuous | primary |
| median / distribution of decision steps | 0012 probe | every probe | shape of the failure |
| `collect/zero_variance_group_frac` | training metrics | continuous | run health |
| policy entropy, KL to reference | training metrics | continuous | run health |
| full 846 pooled + per-family | standard harness | **once, at the end** | recorded cost, not a gate |

The 200-resample probe is the reason to run this now rather than earlier: when the graded
runs were cut, boss could only be measured as 57 binary tasks at ±5 pp, which could not
have resolved the effect this run is looking for.

Checkpoint every ~50 updates. Select on a **held-out** basis per 0008 — never by looking
at closed-loop results first.

### Decision rule

Because this is a feasibility question, the verdict is a 2×2 on *learnability* versus
*generalization*, not a single number. "Solved" means substantially above the ~9% ceiling
every prior method hit.

| train starts | held-out | verdict |
|---|---|---|
| high | high | **Boss is RL-solvable.** Question settled; decide separately how to re-mix families |
| high | low | **Learnable but not general.** RL fits boss from the 466 starts and does not transfer to 57 held-out ones — the constraint is *start-state coverage*, which makes diversified boss generation the justified next spend |
| low | low | **Not solvable this way.** Dense reward, 10 h and undivided budget did not move it; gradient supply was never the bottleneck, and the next question is representation or task design |
| low | high | incoherent — check the probe before believing it |

**A negative requires a plateau, not a low endpoint.** Read boss success against update
number: a run still climbing at the 10-hour cut proves solvability-in-progress and argues
for more wall clock, whereas a flat trace for the last ~150 updates is the real negative.
Record which of the two it is; without that the run cannot answer the question it exists
to answer.

## 3. What was rejected, and why

**A fixed ~1,000-update count (0008's framing).** Superseded here. 0008 was written before
boss episode cost was measured; at 100% boss, 1,000 updates is ~21 hours, not 14. Wall
clock is the honest budget and resumption makes it safe.

**Mixed-family rollouts at the 50% share.** That is the control (`09-23-22`) and it
already produced 71.6% pooled / 8.8% boss. Repeating it with grading would spread the
recovered gradient across three families that do not need it.

**More boss BC data.** Measured null across 8× (0009 / eval 0011), and the searched data
is 81% from four start states. Not this document's axis.

**Grouping boss tasks by weapon.** Already the case: a group is G rollouts of *one* task
(`rollout.py:411`, `:478`), so weapon, start state and `boss_hp_start` are constant within
every group and across-group difficulty is normalized away.

**Unnormalised advantages.** 0005 records the attempt: combined with grading it changed
two variables at once, boss got worse, and the run was discarded. `normalise_advantages`
stays `true`.

**A symmetric step penalty.** 0005 §8 — it scores a fast death above a long survival,
which is precisely inverted for a policy whose failure *is* dying at ~22% of the expert's
episode length.

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate |
|---|---|---|
| **overfitting the 466 train starts** | ~38 visits per task over 480 updates | the §2 decision rule's second row — this *is* the primary scientific axis, not a side risk |
| non-boss regression | one shared causal core, 0% non-boss rollouts | **out of scope.** Measured once at the end for the record; it must **not** trigger a restart or a replay share. A specialist that abandons the other families is the intended shape of this experiment |
| **chip-without-kill** | grading pays partial damage | reward ranges are disjoint at `progress_coef ≤ 0.5` (0005 §2); watch the `(89, 100]` damage bin, not the mean |
| **entropy collapse over 10 h** | 10× longer than any completed run | log policy entropy; T=1 rollout entropy below ~0.6 nats is a red flag (T=0 sits at 0.69 and costs 1.5–2.2 pp) |
| **the run dies mid-way again** | it has happened twice | true resumption tested *before* launch, not after |
| boss simply does not move | every prior intervention has failed | the §2 decision rule's third row — a clean negative at 10 h is a real result |

## 5. Sequencing

1. **Verify the base** — dropout 0.2 at seed 1, ~40 min. Decides init per §2.
2. **Test resumption** — kill and resume a short graded run, confirm optimizer, scheduler
   and sampler state continue exactly. This is what killed the two prior attempts'
   usefulness; do not launch a 10-hour job without it.
3. **Launch**, boss share 1.0, `progress_coef 0.5`, wall-clock 10 h, checkpoint each ~50
   updates.
4. **Probe** with the 0012 200-resample script plus the full 846 guardrail at each saved
   checkpoint.
5. **Hand results to eval** as a handoff issue; update this doc's `Status` and resolve the
   §2 decision rule.

Not blocked on another repo. `KillBossMaker.boss_hp` and `boss_hp_start` are already
consumed (`kaihe/contra_nes_policy#1`).

---

## Appendix — provenance

| claim | source |
|---|---|
| zero-variance 0.514 → 0.278 / 0.270 | `runs/grpo/{2026-08-03/09-23-22,2026-08-04/09-45-33,2026-08-04/18-57-23}/metrics.csv`, `collect/zero_variance_group_frac` |
| graded runs stopped at 61 / 77 updates | same files, row counts; reason in `doc/0009` §4 |
| 71.6% pooled is the ungraded control | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` (`CKPT=…/grpo/2026-08-03/09-23-22`) |
| 75–79% non-zero damage vs 4.0–8.5% success; median 37–41 steps | `contra_nes_evaluation/doc/0012-d8-dropout-sweep.md` §4 |
| boss expert mean 177.7 steps; overall 100.5 | `contra_nes_evaluation/runs/0805-boss-scale-D4-final-full/report.json` |
| 6,904 train tasks = 6,438 non-boss + 466 published boss | `src/contra_policy/rl/tasks.py` docstring; `load_or_build_index` counts |
| 51–60 s/update at 50% boss share, `group_size: 8` | run directory wall clock ÷ update count; `resolved_config.yaml` |
| groups are one task; weapon constant within a group | `src/contra_policy/rl/rollout.py:411`, `:478`; `doc/0005` §1 |
| dropout 0.2 = 69.0%, p = 0.050; D4 = 67.1% | `contra_nes_evaluation/doc/0012` §3, `doc/0011` §3 |
| 20 Hz control rate | one decision = 50 ms |
