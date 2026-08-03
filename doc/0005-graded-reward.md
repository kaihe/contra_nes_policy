# Graded rewards: recovering the half of every rollout budget that teaches nothing

Status: Proposed
Supersedes: —
Depends on: [0004](0004-grpo-experiment-plan.md) (the runs this is measured against),
`kaihe/contra_nes_data#2` (the HP accessor phase 1 needs)

**Question.** Our reward is binary and terminal: 1 for success, 0 for everything else.
Across three GRPO runs, **53% of rolled episodes produced no gradient** — their group's
members all agreed, so every advantage was zero and the group was filtered out. Can a
graded reward recover them, and what would it cost?

**Answer.** Two changes, attacking **opposite tails**, run in sequence rather than
together. **Phase 1: HP grading** gives failing boss rollouts partial credit for damage
dealt, rescuing all-failure groups. **Phase 2: a speed term** grades successes by how
fast they finished, rescuing all-success groups. Both keep one scalar per episode, so
nothing in the GRPO stack changes. A **symmetric** step penalty — the obvious version —
is rejected: it scores a fast death above a long survival, which inverts the exact
signal boss needs.

---

## 1. The measurement this exists to fix

`zero_variance_group_frac` has never gone below 0.46 in any run:

| run | families | zero_var | episodes rolled | used | waste |
|---|---|---|---|---|---|
| `2026-08-02/11-48-03` | 3 | 0.58 | 4,096 | 1,736 | 58% |
| `2026-08-02/15-22-32` | 3 | 0.53 | 30,080 | 14,176 | 53% |
| `2026-08-03/09-23-22` | 4, boss 50% | 0.53 | 30,080 | 14,096 | 53% |

A group is G rollouts of *one* task. Its advantage is
`(r_i − mean(r_group)) / std(r_group)`, so when every member gets the same reward the
numerators are all zero and the group moves the policy not at all. With a binary reward
that happens whenever the policy solves a task reliably **or** fails it reliably —
`P(zero variance) = p^G + (1−p)^G`, which is near 1 at both ends and minimal at p = 0.5.

We have already spent one experiment trying to fix this by *sampling*: the
`DifficultyTracker` biases task selection toward p ≈ 0.5. It cannot work alone, because
**tasks near p = 0.5 barely exist**. Measured per-label success on the three easy
families: 0.718, 0.726, 0.755, 0.766, 0.846, 0.879, 0.935, 0.936, 0.942, 0.942, 0.942,
0.958, 0.958, 0.979. Twelve of fourteen labels sit above 0.84. The sampler can only
reorder what the catalog contains.

Grading changes the reward instead of the sampling, and does not depend on a p ≈ 0.5
population existing.

## 2. Phase 1 — HP grading (the low tail)

```
success:  reward = 1.0
failure:  reward = α · (HP_start − HP_end) / HP_start          α = 0.5
```

Damage *removed*, so higher is better. Still terminal: two RAM reads, one at `_start`
and one at `_finish`, giving one scalar per episode. `buffer.py`'s contract — one
advantage per episode, broadcast over its steps, no per-step credit — is untouched.

**Why boss first.** It is where the low tail lives. Boss scores 3.5% on val and ~10% on
train, so at G = 8 between 43% and 75% of boss groups are all-failure. Today those look
like this:

```
[0, 0, 0, 0, 0, 0, 0, 0]  ->  std 0  ->  filtered, 8 rollouts discarded
```

and with grading like this:

```
[0.02, 0.31, 0.05, 0.44, 0.08, 0.12, 0.38, 0.03]  ->  real spread
```

The group now says *the rollouts that took the boss to 40% got further than the ones
that died on approach* — which is exactly what the binary reward destroys, and exactly
the distinction our failure mode turns on. Measured over 69 boss deaths at `u075`, the
policy dies at **27% of the expert's episode** (val: 32%), at the approach→engage
transition. Every one of those currently scores identically to a rollout that nearly
won.

**Normalising by the anchor is what makes mid-fight tasks work.** Tasks generated from a
savestate partway through a fight begin with the boss already damaged; absolute damage
would understate them. This is why `boss_hp_start` is requested as task metadata.

### The anchor is the observed peak, not the value at step 0

Found on implementation, and it would have silently produced a no-op. A boss task begins
at the boss *reveal*, **before** the boss occupies an active enemy slot — so `boss_hp`
reads **0** at step 0 and only climbs once it spawns. Measured over 24 rollouts: `hp0`
was 0 for **all 24**, peaking at 25-64, and successes ended at 28-30 against deaths at
59-64.

Anchoring on the initial read scored every episode as having dealt no damage —
`progress_coef` had literally no effect. The anchor is therefore `hp_peak`, the maximum
observed during the episode. An episode that dies before the boss ever spawns has
`hp_peak == 0` and scores zero, which is correct: it made no progress.

### Measured on real rollouts before the run

Four boss groups of G=8 at `grpo-000075`:

| | binary | graded |
|---|---|---|
| `zero_variance_group_frac` | **0.50** | **0.00** |
| failures with non-zero credit | 0% | 96% |
| largest failure reward | 0.0 | 0.275 |

The two all-failure groups — discarded outright under the binary reward — came back with
real spread (`[0.039 … 0.164]` and `[0.0 … 0.141]`). Ranges stayed disjoint.

### The confirmation run

A matched repeat of `runs/grpo/2026-08-03/09-23-22`: **same BC init, same 50% boss
sampling, same G, KL, lr and 100 updates**, adding only `reward.progress_coef: 0.5`. That
run is the control, and its checkpoints are already evaluated
(`contra_nes_evaluation/doc/0008-grpo-with-boss.md`), so the comparison is against
published numbers rather than a re-run.

Not bit-identical: the fixed `probe` was added afterwards and consumes rollouts, which
shifts the sampling stream. It trains on nothing and feeds the sampler nothing, so the
comparison is statistically clean but not seed-exact.

**Confirmed if** boss `zero_var` falls below 0.2 *and* val boss beats 10.5% at matched
updates. **Refuted if** val boss is unchanged — which would mean the recovered gradient
is real but uninformative, and the boss path is entirely a data problem.

Either way the next step is the diversified data from `kaihe/contra_nes_data#2`; this run
decides whether graded reward ships alongside it or is dropped.

**Dependency.** `boss_hp` does not exist yet. `env/utility.py:boss_enemy_present()` in
the data repo already walks exactly the right slots (`ADDR_ENEMY_HP`, filtered by
`BOSS_ENEMY_TYPES_BY_LEVEL`); it needs a sibling that sums instead of testing, exposed as
a method on `KillBossMaker` — which `rollout.py:376` already imports and calls every
step. Requested in `kaihe/contra_nes_data#2`. **No `ADDR_*` knowledge crosses into this
repo, and the policy's input contract is unchanged — this is reward shaping, not a new
observation.**

## 3. Phase 2 — the speed term (the high tail)

```
success:  reward = 1 − β · (steps / budget)                    β = 0.1
failure:  unchanged
```

**The high tail is the bigger half.** `kill`, `item` and `traverse` run at 83–90% on
train, and any task the policy solves reliably returns eight identical 1.0s. Twelve of
the fourteen labels above are over 0.84. A speed term gives those groups variance
without needing any task to be hard.

**Normalised by budget, not a flat per-step rate.** Budgets are 2× expert length, and
expert length by weapon runs Spread 90 → Flamethrower 321 decisions, so budgets span
roughly 180 → 776. A flat α that is sensible for one is meaningless for the other.
Normalising also *guarantees* the ordering below.

**Ranges stay disjoint**, which is the property that keeps the objective honest:

| | range |
|---|---|
| failure (phase 1) | [0, 0.5] |
| success (phase 2) | [0.9, 1.0] |

Every success outranks every failure at any α ≤ 0.5, β ≤ 0.1. The speed term can only
break ties *among winners*; it can never trade a win for a fast loss.

## 4. Sequencing

1. **Phase 1 confirmation run** — §2, matched against `2026-08-03/09-23-22`.
2. **Diversified boss data** — `kaihe/contra_nes_data#2`, independent of the outcome
   above and the actual path for boss (§7).
3. **Phase 2, the speed term** — after, and only measured once phase 1 has settled.

Phases 1 and 2 are independently falsifiable and touch different families. Boss is ~50%
of rollouts under the current sampling and the easy families are the rest, so shipping
both together would leave a change in `zero_var` unattributable — and `zero_var` is the
whole point. Phase 1 turned out to have no external blocker after all: the data repo
landed `KillBossMaker.boss_hp` before this was written.

## 5. What we expect, written down in advance

| prediction | why | falsified by |
|---|---|---|
| phase 1 cuts boss `zero_var` from ~0.6 to < 0.2 | measured 0.50 → 0.00 on four groups before the run | boss `zero_var` > 0.4 |
| phase 1 raises **val** boss above 10.5% | the failure signal becomes informative | val boss ≤ 10.5% at matched updates |
| phase 2 cuts overall `zero_var` below 0.30 | 12 of 14 labels are > 0.84, all rescued | `zero_var` > 0.40 |
| neither changes **pooled** val by more than ~2 pp | this buys sample efficiency, not a better objective | pooled moves > 5 pp either way |

The last row is the honest one. Two GRPO recipes have now landed at ~71% pooled
([0006](../../contra_nes_evaluation/doc/0006-grpo.md),
[0008](../../contra_nes_evaluation/doc/0008-grpo-with-boss.md)). Recovering wasted
rollouts should make the same objective reachable in fewer episodes; it is not a reason
to expect a different ceiling. If pooled jumps, the reasoning here is wrong somewhere.

## 6. Risks

| risk | gate |
|---|---|
| **reward hacking** — plink from safety, farm damage, never commit | successes ≥ 0.9 strictly dominate partials ≤ 0.5; watch for boss episodes lengthening while success is flat, visible in the probe within ~20 updates |
| **β too high** trades success for speed | start at 0.1; `kill`/`traverse` must not fall on the probe |
| **partial credit inflates train numbers** | report `zero_var` and probe success, never pooled train success — see §7 |
| boss HP is multi-component and may not decrease monotonically | sum over boss slots; verify on expert traces before trusting the signal |

## 7. What this cannot conclude

**Whether the boss strategy is learnable from the current data.** Grading makes the
existing failures *informative*; it does not add a demonstration of a safer route. The
policy dies on its own expert's line — measured, median 2.2px from the nearest point,
though a mismatched task's line is 3.6px away, so that test is weak because every level-1
boss fight happens in the same ~100px room. The data path is
`kaihe/contra_nes_data#2` and is independent of this doc.

**Anything from train success.** Both changes make partial progress score above zero, so
train success rises mechanically. The gate is the val harness, and in-run the fixed
`probe` (`config_grpo.yaml`), which samples uniformly and is not touched by the
difficulty sampler.

## 8. What was rejected

| proposal | why not |
|---|---|
| **symmetric step penalty** (`−α` per step, all outcomes) | scores a fast death above a long survival. Our boss failure *is* dying early, at 27% of the expert's episode; this would optimise directly against the fix. The warning is already in `rollout.py:_finish`. |
| flat per-step penalty rather than budget-normalised | budgets span 180 → 776 decisions across weapons; one constant cannot suit both |
| per-step shaped rewards with returns-to-go | breaks `buffer.py`'s one-scalar-per-episode contract and reintroduces the credit assignment GRPO exists to avoid. A terminal graded score gets the same information for none of the machinery. |
| grading `kill` by enemy HP as well | `live_slots()` makes it easy, but `kill` runs at 85% — its degeneracy is on the *success* side, which phase 2 already handles |
| raising G instead | tails fall slowly and cost is linear; [0004](0004-grpo-experiment-plan.md) §5 |
| conditioning the policy on HP or weapon | would need a new input token. Rejected by the author: reward shaping needs neither, and the policy's contract stays `[interaction, goal, frame…]` |

---

## Appendix — provenance

| claim | source |
|---|---|
| `zero_var` and waste per run | `runs/grpo/*/metrics.csv` |
| per-label success on the easy families | `sampler/p_label/*`, `runs/grpo/2026-08-02/15-22-32/metrics.csv` u100 |
| boss val 3.5% → 10.5% | `contra_nes_evaluation/doc/0008-grpo-with-boss.md` |
| policy dies at 27% of the expert's episode | 69 boss deaths at `u075`, replayed trajectories in world coordinates |
| 2.2px matched vs 3.6px mismatched | same analysis, null control against a different task's expert path |
| expert length by weapon (90 / 108 / 242 / 321) | 523 boss task `.npz`, replayed to read `ADDR_WEAPON` |
| `step: 0.0` is a dead config key | `rollout.py:552` reads only `self.reward[outcome]` |
| zero gradient from a zero-variance group | `tests/test_grpo.py::test_a_zero_advantage_batch_produces_no_policy_gradient` |
