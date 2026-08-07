# Prove GRPO can move boss at all: binary reward, Spread + rapid only

Status: Proposed
Supersedes: —
Depends on: [0011](0011-boss-grpo.md) (the run this diagnoses), [0005](0005-graded-reward.md) (the reward this switches **off**)

**Question.** Every boss result in this project sits at 7–11% with ~90% death, across BC
at four data scales, a dropout sweep, sparse GRPO and graded GRPO. Weapon stratification
now shows why the aggregate cannot move: **~40% of boss tasks are unwinnable** by a policy
that survives ~2 s, and **every boss success ever recorded came from Spread or Laser**. So
the open question is no longer "does boss improve" — it is **can GRPO improve anything on
boss at all**, measured where a win is actually reachable?

**Answer.** Run GRPO on the **Spread + rapid-fire subset only**, with the **plain binary
reward** (`progress_coef: 0.0`). Two things make this the decisive test rather than another
null. First, win density does the job the graded reward was invented for: at Spread's 17%
success, only **22.5%** of groups come back degenerate at G = 8, against **61%** on the
pooled boss mix — and it *improves* as the policy improves (5.8% at p = 0.30), a virtuous
cycle instead of the graded reward's runaway. Second, it removes the 40% of tasks where no
reachable behavior wins, which is where 0011's gradient was going.

If boss does not move here, it will not move anywhere, and the bottleneck is the policy or
the representation rather than the optimizer. The binding constraint is measurement: only
**13 val tasks** carry this weapon, so the protocol below is built around that.

---

## 1. Why — the evidence

### Weapon decides the fight; the policy's survival does not vary

200-resample boss probe on the BC init (eval [0013](../../contra_nes_evaluation/doc/0013-boss-only-grpo-precollapse.md)), joined to shard metadata:

| weapon | draws | median steps | expert steps | survived | mean damage | **success** |
|---|---:|---:|---:|---:|---:|---:|
| Spread | 41 | 37 | 87.2 | **52.9%** | 29.9% | **17.1%** |
| Laser | 80 | 39 | 117.8 | 43.0% | 23.0% | 12.5% |
| Regular | 73 | 37 | **283.0** | 19.7% | 8.6% | **0.0%** |
| Flamethrower | 6 | 56 | **346.8** | 17.4% | 11.4% | **0.0%** |

The policy survives ~37–56 decisions — about 2 s — **regardless of weapon**. Fight length
varies 3.2×. Damage is just DPS × survival, so it tracks the survived fraction exactly.

**Regular and Flamethrower are 0.0% across all four checkpoints evaluated — zero wins in
316 rollouts.** 23 of 57 val tasks are structurally out of reach, capping boss success near
60% before any policy question is asked. The universal 7–11% ceiling is a mixture statistic
over an unwinnable 40%.

### Binary reward is sufficient here, and self-reinforcing

`P(zero-variance group) = p^G + (1−p)^G` at G = 8:

| regime | p | degenerate groups |
|---|---:|---:|
| pooled boss (0011) | 0.06 | **61.0%** |
| **Spread + rapid (this doc)** | 0.17 | **22.5%** |
| if it reaches | 0.30 | 5.8% |
| ideal | 0.50 | 0.8% |

0005 was written because 53% of the rollout budget produced no gradient. On this subset the
binary reward starts at 22.5% waste and **falls as the policy improves**. The shaping that
0005 introduced to fix gradient supply is not needed where win density already supplies it.

### The graded reward should be switched off, not tuned

0011 ran it for 10 h and produced three findings, all arguing against carrying it here:

1. **Chip-without-kill.** Damage per failed episode rose 11.9% → 13.9% while wins fell
   4.9% → 0.1%. In a group with no win, the aggressive behavior that could reach a kill
   dies earlier with less damage and scores *below* the cautious one — so the only path to
   a win is actively punished. At p = 0.06, 99% of groups had no win.
2. **It defeated the difficulty sampler.** `difficulty_bias` routes budget away from tasks
   whose groups carry no gradient, keyed on group variance. Under the binary reward a
   hopeless boss group was eight identical zeros → filtered → down-weighted. Under grading
   the same group has damage spread → survives filtering → *looks productive*. The
   headline `zero_variance_group_frac` 0.514 → 0.002 also means **hopeless tasks stopped
   being detectable as hopeless**.
3. **No lift even before collapse.** Train probe flat ~4–6% for ~600 healthy updates;
   held-out u525/u550/u575 at 7.5 / 11.0 / 7.5% against an 8.5% init (0013).

## 2. The design

### Task restriction

| split | Spread + rapid | of total |
|---|---:|---:|
| train | **120** | 466 |
| val | **13** | 57 |

All 13 Spread val tasks carry `rapid: true`, so "Spread + rapid" and "Spread" are the same
val set; the pairing only matters on train (120 of 126 Spread train tasks).

Selection is by **metadata assertion, not globbing**, in the spirit of 0009's manifest
discipline: read `weapon` and `rapid` from the shard JSON — the same source eval joins on —
and assert exactly 120 train / 13 val at startup. A mismatch is a hard failure, because a
silently smaller pool would look like a successful run on an easier set.

### Configuration

Departures from the 0011 boss-only run, each with its reason:

| setting | value | why |
|---|---|---|
| `reward.progress_coef` | **0.0** | binary. §1: win density supplies the variance |
| task filter | Spread + rapid | removes the 40% where no win is reachable |
| `families` | `[boss]` | unchanged from 0011 |
| **cumulative kl_ref stop** | **0.10** | *new*. 0011 had only a per-update `target_kl`, which fired on 1,500 of 1,619 updates while cumulative drift ran to 0.245 unchecked |
| `train.updates` | **500** | bounded by updates, not wall clock — 0011's 10 h bought 1,619 updates, 3.4× its design |
| `train.max_hours` | 4.0 | backstop only; ~22 s/update ⇒ 500 updates ≈ 3 h |
| `save_every` | 25 | unchanged |
| `group_size`, `kl_coef`, lr | unchanged | one variable at a time; the control at 100 updates was healthy here |

### Measurement, built around 13 tasks

This is the experiment's weak point and the protocol has to earn its conclusion.

- **Fixed 13 × 16 repeats = 208 rollouts** per checkpoint, not 200 draws with replacement.
  Every task gets equal weight and the same 13 starts appear in every cell.
- **Report per-task success as a 13-row table.** A rise concentrated in one or two tasks is
  not a capability gain, and the aggregate cannot distinguish those.
- **Cluster-aware intervals** — 13 clusters, not 208 independent samples. A naive binomial
  CI here is roughly 4× too narrow.
- **Re-measure the BC init on this exact protocol.** The 17.1% init figure comes from 41
  draws inside a 200-draw pooled probe and is not a control for this design.
- Track `collect/zero_variance_group_frac` (should start ~0.22 and *fall*), policy entropy,
  and cumulative `kl_ref`.
- Train-side: the unbiased probe over the 120 train tasks, every 10 updates.

### Decision rule

Predeclared. "Init" is the BC base re-measured on the 13 × 16 protocol.

| observation | verdict |
|---|---|
| held-out Spread success rises ≥ 15 pp over init, spread across ≥ 5 of the 13 tasks | **GRPO works on boss.** The prior nulls were reachability and reward-shaping problems, not optimizer failure. Extend to Laser, then reconsider Regular |
| rises ≥ 15 pp but concentrated in ≤ 2 tasks | memorization of specific starts, not capability — report as such, do not extend |
| train probe rises while held-out is flat | overfitting the 120 train starts; the constraint is start-state coverage |
| both flat after 500 updates at `zero_var` ≈ 0.2 | **GRPO does not move boss on this stack**, with gradient supply healthy and every reachability excuse removed. The next question is representation or survival, not RL |
| `kl_ref` hits the 0.10 stop early | configuration, not result — re-run with lower lr; do not report the truncated run as a null |

## 3. What was rejected, and why

**Keeping the graded reward "just in case".** §1.3: it produced chip-without-kill, hid
hopeless tasks from the difficulty sampler, and delivered no lift in 600 healthy updates.
Carrying it would leave two variables moving and make a null unattributable.

**Training on all four weapons.** 40% of groups would be on tasks where no reachable
behavior wins. That is where 0011's budget went.

**Restricting to Spread without `rapid`.** Only 6 train tasks — too few to train on.

**Wall-clock budget.** 0011's lesson: the same 10 h bought 3.4× the intended updates
because boss episodes are short. Updates are the honest unit when episode length is stable.

**A survival bonus.** It is probably the right long-term reward — the measured failure is a
fixed ~2 s survival against 87–347 decisions needed, and 0005 §8 only ever rejected a
step *penalty*, never a bonus. But it is a second variable, and this experiment exists to
isolate whether GRPO moves anything at all. It belongs in the successor doc.

**Adding Laser now.** Laser wins exist (12.5%) so it is a reasonable second cell, but its
fight is 1.35× longer and mixing it in makes a null harder to attribute. Extend on success.

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate |
|---|---|---|
| **13 val tasks cannot resolve the effect** | cluster-limited, not sample-limited | per-task table + cluster-aware CI; a verdict needs ≥ 5 of 13 tasks moving |
| overfitting 120 train starts | ~67 visits per task over 500 updates | train probe vs held-out; §2's third decision row |
| KL runaway repeats | it did in 0011 | cumulative `kl_ref` stop at 0.10 — the guard 0011 lacked |
| entropy collapse | 0.875 → 0.407 in 0011 | log entropy; below 0.6 is a warning, and T = 0 already costs 1.5–2.2 pp |
| the subset is *too* easy and results do not generalize | Spread is the strongest weapon | that is the point — this is a feasibility floor, not a shippable policy. Extension to Laser is the generalization test |
| binary reward starves the update after all | 22.5% degenerate at p = 0.17 | `zero_var` > 0.4 sustained means the win density assumption is wrong |

## 5. Sequencing

1. **Add the metadata task filter** — `weapon` / `rapid` read from shard JSON, with hard
   assertions of 120 train / 13 val. Test that the filter selects exactly those uids.
2. **Add the cumulative `kl_ref` stop.** 0011 had no bound on total drift.
3. **Re-measure the BC init** on the 13 × 16 protocol. This is the control and it does not
   exist yet.
4. **Run 500 updates** (~3 h), checkpointing every 25.
5. **Evaluate** held-out on the 13 × 16 protocol, with the per-task table.
6. Update this doc's `Status`, resolve the decision rule, and hand results to eval.

Not blocked on another repo — the metadata is already in the shards eval reads.

---

## Appendix — provenance

| claim | source |
|---|---|
| weapon × survival × success table | `contra_nes_evaluation/runs/0806-dropout-0.2-boss-progress-200/boss_progress.csv` joined to `contra_nes_data/game_trace/hf/boss-val-00000.tar` JSON on `uid` |
| 0 wins in 316 Regular/Flamethrower rollouts | same join over the four probe runs in eval `doc/0013` |
| Spread+rapid counts 120 train / 13 val | `boss-{train,val}-00000.tar` JSON, `weapon` × `rapid` |
| degenerate-group rates | `p^8 + (1−p)^8` at the measured success rates |
| chip-without-kill; kl_ref 0.245; entropy 0.407; 1500/1619 early stops | `runs/grpo/2026-08-06/boss-only-19-15-52/metrics.csv` |
| held-out pre-collapse 7.5 / 11.0 / 7.5 vs init 8.5 | eval `doc/0013` §2 |
| 22.2 s/update | 0011 run wall clock ÷ 1,619 updates |
| expert lengths by weapon | `expert_steps` in the probe CSV; cf. `doc/0005` §3 |
