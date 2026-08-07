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
**13 val tasks** carry this weapon, so but the boss starts are near-identical, so the 120-task train probe carries the verdict
and held-out is a sanity check.

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
| `train.updates` | **100** | a fixed budget matched to the runs that worked — the control `09-23-22` moved boss 3.5% → 10.5% in exactly 100 updates and ended at `kl_ref` ~0.030. 0011's collapse was at 1,619. At 100% boss share this carries ~2× the control's per-update boss gradient |
| `train.max_hours` | 1.0 | backstop only; ~22 s/update ⇒ 100 updates ≈ 40 min, cheap enough to repeat across seeds |
| `save_every` | 25 | unchanged |
| `group_size`, `kl_coef`, lr | unchanged | one variable at a time; the control at 100 updates was healthy here |

### Measurement: the 120-task train probe is primary

The earlier draft of this section built everything around the 13 held-out tasks. That was
wrong, for a reason worth writing down: **the boss tasks are near-identical starts.** All
120 Spread+rapid train tasks sit at `start_x` 3102–3104 — three distinct values, because
the level-1 boss room is at a fixed map position — with `boss_hp_start` inside a 6-point
band (59–64). They come from 120 distinct playthroughs, so they are not duplicates, but
they are 120 samples of *one encounter*, not 120 different problems.

Two consequences:

- **Held-out adds little.** A val Spread task is the same fight from an equivalently
  similar state. There is no domain shift to generalize across, so the
  memorization-versus-capability distinction that motivated the 13 × 16 protocol is much
  weaker than assumed.
- **The train probe is the better measurement.** 120 tasks against 13, already unbiased
  by construction — fixed tasks, uniform within family, no filtering, no difficulty
  weighting, and results never fed back to the sampler.

So: **train probe every 10 updates as the primary signal; the 13 held-out tasks once at
the end as a sanity check, not a gate.**

### Progress metrics, not just outcome

Binary success is the wrong primary metric on a pool this size, and switching to the
binary *reward* makes that worse: `reward` folds damage in only when `progress_coef > 0`,
so at 0.0 the scalar keeps no trace of how far a rollout got. Damage and survival are
therefore recorded independently of the reward:

| metric | why |
|---|---|
| `probe/damage_lost` | boss HP removed on **failed** episodes. Pooling wins in would just re-report success |
| `probe/len_lost` | steps before death on failures — the quantity the whole diagnosis points at: the policy dies at a weapon-independent ~37–56 decisions against 87 needed on Spread |
| `probe/tasks_won` | how many of the 120 tasks were won at least once — separates a broad gain from one lucky start |
| `probe/damage_mean`, `probe/saw_boss` | context for the above |

These move before success does. 0011 measured damage per failure going 11.9% → 13.9%
while success sat at 0.1% — the wrong direction, but *measurable* where the binary metric
was pinned. On 120 near-identical starts that sensitivity is the difference between an
interpretable run and another null.

Also tracked: `collect/zero_variance_group_frac` (should start ~0.22 and *fall*), policy
entropy, and `kl_ref` against the 0.10 stop.

### Decision rule

Predeclared. "Init" is the BC base re-measured on the 13 × 16 protocol.

| observation | verdict |
|---|---|
| train probe success rises ≥ 15 pp, with `tasks_won` rising and `len_lost` up | **GRPO works on boss.** The prior nulls were reachability and reward-shaping problems, not optimizer failure. Confirm on the 13 held-out, then extend to Laser |
| success flat but `len_lost` and `damage_lost` rise | RL is moving survival without converting to kills yet — the leading indicators fired. Extend the update budget rather than concluding |
| success rises but `tasks_won` barely moves | a lucky start, not capability — report as such, do not extend |
| train rises ≥ 15 pp while the 13 held-out stay flat | surprising given the starts are near-identical; suspect the probe or the split before believing it |
| both flat after 100 updates at `zero_var` ≈ 0.2 | **GRPO does not move boss on this stack**, with gradient supply healthy and every reachability excuse removed. The next question is representation or survival, not RL |
| `kl_ref` hits the 0.10 stop early | configuration, not result — re-run with lower lr; do not report the truncated run as a null |

## 3. What was rejected, and why

**Keeping the graded reward "just in case".** §1.3: it produced chip-without-kill, hid
hopeless tasks from the difficulty sampler, and delivered no lift in 600 healthy updates.
Carrying it would leave two variables moving and make a null unattributable.

**Training on all four weapons.** 40% of groups would be on tasks where no reachable
behavior wins. That is where 0011's budget went.

**Restricting to Spread without `rapid`.** Only 6 train tasks — too few to train on.

**A wall-clock budget.** 0011's lesson: the same 10 h bought 3.4× the intended updates
because boss episodes are short. Updates are the honest unit, and a budget matched to a run
that is known to have worked beats an open-ended session — 0011 also showed that a long run
with an unbounded failure mode simply finds it.

**A survival bonus.** It is probably the right long-term reward — the measured failure is a
fixed ~2 s survival against 87–347 decisions needed, and 0005 §8 only ever rejected a
step *penalty*, never a bonus. But it is a second variable, and this experiment exists to
isolate whether GRPO moves anything at all. It belongs in the successor doc.

**Adding Laser now.** Laser wins exist (12.5%) so it is a reasonable second cell, but its
fight is 1.35× longer and mixing it in makes a null harder to attribute. Extend on success.

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate |
|---|---|---|
| **binary success is noise-limited on this pool** | ~17% success over 120 near-identical starts | `len_lost` and `damage_lost` are the primary readouts; success is confirmatory |
| overfitting 120 train starts | ~13 visits per task over 100 updates | weak here — the starts differ only in boss HP (59–64) and timing, so there is little task-specific structure to memorize. `tasks_won` is the check |
| KL runaway repeats | it did in 0011 | cumulative `kl_ref` stop at 0.10 — the guard 0011 lacked |
| entropy collapse | 0.875 → 0.407 in 0011 | log entropy; below 0.6 is a warning, and T = 0 already costs 1.5–2.2 pp |
| the subset is *too* easy and results do not generalize | Spread is the strongest weapon | that is the point — this is a feasibility floor, not a shippable policy. Extension to Laser is the generalization test |
| binary reward starves the update after all | 22.5% degenerate at p = 0.17 | `zero_var` > 0.4 sustained means the win density assumption is wrong |

## 5. Sequencing

1. **Add the metadata task filter** — `weapon` / `rapid` read from shard JSON, with hard
   assertions of 120 train / 13 val. Test that the filter selects exactly those uids.
2. **Add the cumulative `kl_ref` stop.** 0011 had no bound on total drift.
3. **Re-measure the BC init** on the same 120-task probe. This is the control and it does
   not exist yet — every published init figure comes from eval's pooled 200-draw probe.
4. **Run 100 updates** (~40 min), checkpointing every 25. Repeat on seeds 0/1/2 — at this
   budget three seeds cost two hours and give the variance estimate 13 val tasks cannot.
5. **Evaluate** the 13 held-out tasks once, at the end, as a sanity check on the train gain.
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
