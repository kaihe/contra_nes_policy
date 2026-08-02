# GRPO in two phases: what each one is allowed to conclude

Status: Proposed
Supersedes: —
Depends on: [0002](0002-gpt-policy.md) (the policy), [0003](0003-grpo-code-layout.md) (the stack)

**Question.** GRPO replaces the critic with a group baseline. Two things are unknown and
they are not the same question: does the stack work, and does it help `boss`? What does
each phase measure, and what is it *not* allowed to claim?

**Answer.** **Phase 1** runs three families and asks only whether the machinery
produces movement — the bar is PPO's historical gain, not 100%. **Phase 2** restarts
from the *same BC checkpoint* with all four and asks whether `boss` moves. Neither
phase can answer the boss question on its own, because the binding constraint is
`degenerate_group_frac`, not the algorithm — and that number is readable in phase 1.

---

## 1. What is actually uncertain

| question | how it gets answered | phase |
|---|---|---|
| does the GRPO stack train at all? | per-family completion moves within ~50-100 updates | 1 |
| does the group baseline beat the critic we deleted? | gain vs PPO's historical per-family gain | 1 |
| does the reference KL stop forgetting? | `item` does **not** regress | 1 |
| is `boss` gradient-starved? | `degenerate_group_frac` on boss groups | 1 or 2 — it is a metric, not an experiment |
| does `boss` improve with GRPO? | phase 2 vs phase 1 at matched updates | 2 |
| *can* boss improve at 3.5% base rate? | **neither** — see §5 | — |

## 2. Phase 1 — does the machinery work

**Setup.** `kill`, `item`, `traverse`. From `policy-final.pt` (67.5% pooled). G=4 —
those families succeed 67-76%, so all-failure groups are rare (`0.33**4 ≈ 1%`) and G=4
buys a usable baseline cheaply. Boss is excluded **for iteration speed only**: its
episodes are long (budget p50 281 against kill's 104) and ~75% of its groups would
yield nothing, so it slows the loop while the code is new. That is not an isolation
argument.

**The bar is PPO's historical gain, not 100%.** The `expert` policy scores 100%, so
every task is solvable, but no learned policy has come close. What GRPO has to beat:

| family | BC now | PPO gained (ROCKET) | phase-1 bar |
|---|---|---|---|
| kill | 66.9% | 69.0 → 81.5 (**+12.5**) | ~78% |
| traverse | 76.0% | 83.5 → 87.7 (+4.2) | ~80% |
| item | 66.7% | 70.4 → 70.4 (**0.0**) | **does not regress** |

`item` is the interesting row. PPO gained nothing there *and* the raw BC number fell
76.5% → 71.1% during training with `bc_kl_coef: 0.0`. So "item holds" is a real result
for the reference-KL term, not a null one.

**Stop early if.** `degenerate_group_frac` > 0.3 on these families (the baseline is not
working), `approx_kl` pinned at `target_kl` from update 1 (the step size is wrong), or
entropy below ~0.12 (the previous run's collapse signature).

## 3. Phase 2 — does boss move

**Setup.** All four families, **from the same BC checkpoint as phase 1** — not from
phase 1's output.

That branch point is the design decision. Continuing phase 1's policy would confound a
boss effect with three-family drift, and drift is measured, not hypothetical: `item`
regressed 76.5% → 71.1% over 500 PPO updates. Restarting makes "did boss move" a clean
comparison against phase 1's per-family numbers at matched updates.

**Report per family, at matched updates.** Phase 2's non-boss numbers should land near
phase 1's; a gap there means adding boss changed the other families, which is itself
worth knowing (the sampler's family mix shifts when a fourth family enters).

## 4. What we expect, written down in advance

Recording predictions so the result can surprise us:

| prediction | why | falsified by |
|---|---|---|
| phase 1 kill reaches ~75-80% | PPO gained +12.5 from a similar base | no movement in 100 updates |
| `item` holds within 2 pp | the reference KL is the fix for what broke it | item falls > 5 pp |
| boss `degenerate_group_frac` ≈ 0.75 at G=8 | `0.965**8` at the measured 3.5% | anything below 0.5 |
| **boss does not improve materially** | see §5 | boss > 10% |

The last row is the one worth being explicit about. PPO already made boss *worse*
(8.8% → 5.3%). If GRPO moves it, that is a genuine surprise and the reasoning in §5 is
wrong.

## 5. What neither phase can conclude

**Whether boss is fixable at a 3.5% base rate.** All-failure groups produce exactly zero
gradient — pinned by a test — so at G=8 roughly three quarters of boss groups contribute
nothing:

| p(success) | G=4 | G=8 | G=16 | G=32 |
|---|---|---|---|---|
| **3.5%** (today) | 87% | 75% | 57% | 32% |
| 20% | 41% | **17%** | 3% | 0% |
| 35% | 18% | **3%** | 0% | 0% |

All the leverage is in the base rate; none is in G. A larger G buys a linear rollout
cost for a shrinking share of usable groups.

### Degeneracy is two-sided — measured after writing the rollout

`P(degenerate) = p^G + (1−p)^G`. A group is useless when its members **agree**, and they
agree at *both* ends:

| p(success) | G=4 | G=8 | G=16 |
|---|---|---|---|
| 3.5% (boss) | 87% | 75% | 57% |
| 30% | 25% | 6% | 0% |
| **50%** | **12%** | **1%** | **0%** |
| 90% | 66% | 43% | 19% |
| 95% | 81% | 66% | 44% |

A first end-to-end rollout on `kill` **train** tasks scored ~92% success (12 episodes,
3 groups — small, but the direction is clear) and **67% of groups were degenerate
because they all succeeded.** That is nearly as wasteful as boss, from the opposite
side, and it was not anticipated: §2's G=4 was chosen from val success rates of 67-76%,
but GRPO trains on *train* tasks, where the policy is much stronger.

This changes phase 1. Options, in the order I would try them:

1. **Filter degenerate groups before the update** — standard practice in LLM RLVR
   ("prompt filtering"): a group whose members agree contributes nothing, so drop it and
   spend the rollout budget elsewhere. Cheap, and `degenerate_group_frac` already
   measures the waste.
2. **Bias the task sampler toward p ≈ 0.5** — keep a per-task running success estimate
   and oversample the uncertain ones. This is a curriculum by another name, and it is
   the same mechanism the boss request asks the data repo for.
3. **Raise G** — the weakest lever, since the tails fall slowly and cost is linear.

### Measured on the three easy families: 0.59, not the 0.25 the table predicts

The first phase-1 attempt (`runs/grpo/2026-08-02/11-48-03`) ran 8 updates before a bug
took the VM down. Its usable numbers: at ~83% pooled success, the true zero-variance
fraction on `kill`/`item`/`traverse` was **0.50–0.67, mean 0.59** — well above the 0.3
stop-early threshold in §2, and from the *high* tail, as this section anticipated for
`kill` but not for all three.

So the §2 setup collects roughly **2.4x more rollouts than it uses**, on the families
chosen for being cheap. Option 2 above (bias the sampler toward p ≈ 0.5) is therefore
not a boss-only measure; it is the phase-1 economics too.

Three faults kept this invisible at the time and are fixed with regression tests in
`tests/test_rollout_groups.py`: group ids restarted at 0 on every collection call so
unrelated tasks were pooled into one group (destroying the same-task baseline and making
the oversample loop's exit unreachable); the collection-side
`zero_variance_group_frac` shared a CSV key with the post-filter one, which is zero by
construction, and was overwritten by it; and whole episodes were retained just to report
success rates, at ~18 MB each and 512 per update — ~9 GB, which is what exhausted the
20 GB guest. **The 8 updates of training are not interpretable**, because the advantages
were computed across pooled tasks.

**And the boss failure looks like a data problem, not an algorithm one.** Measured on the 57
val boss episodes: **55 deaths, median 40 steps — 14% of the budget**, against an expert
that needs ~140. The policy dies *on approach*, two seconds in. That is what imitating a
`push_right`-shaped charge looks like: MC search survives the dangerous line with
frame-perfect timing, and a policy with any execution error does not.

Supporting: the 466 boss tasks come from 466 *distinct* source traces, so this is not a
few-traces problem — it is 466 samples of **one strategy**, because they share one
generator with one reward. Sample diversity is not strategy diversity. And weapon is not
recorded anywhere in the episode JSON, so gun coverage cannot even be measured today.

So the boss path runs through the data repo, not through a GRPO configuration:
no-push-reward traces, weapon metadata and coverage, and reverse-curriculum savestates
along each trace. Tracked separately; **not** gated on either phase here.

## 6. Sequencing

1. Write `rollout.py` and `trainer.py` ([0003](0003-grpo-code-layout.md) §5).
2. **Phase 1**, ~100 updates first. Check the stop-early conditions before committing to
   a long run.
3. Phase 1 long run to the bar in §2.
4. **Phase 2** from the BC checkpoint, all four families, matched updates.
5. Boss data request to `contra_nes_data` — independent of 2-4, and the actual boss path.

---

## Appendix — provenance

| claim | source |
|---|---|
| BC per-family baseline (67.5% pooled) | `contra_nes_evaluation/doc/0005-gpt-bc.md` |
| PPO's historical per-family gain | `contra_nes_evaluation/runs/0729-e18-noprev` vs `0730-rl-u375-noprev` |
| item regression without reference KL | 500-update PPO run, first vs last 100 updates |
| boss deaths at median 40 steps / 14% of budget | `contra_nes_evaluation/runs/0801-gpt-bc-final/episodes.csv`, 57 boss rows |
| 466 boss tasks from 466 distinct traces | `src_trace` over the boss train shard |
| weapon not recorded | episode JSON keys — no weapon field |
| degenerate-group table | `(1 - p)**G` at the measured 3.5% |
| zero gradient from a degenerate group | `tests/test_grpo.py::test_a_zero_advantage_batch_produces_no_policy_gradient` |
| zero-variance fraction 0.59 on the three easy families | `runs/grpo/2026-08-02/11-48-03/metrics.csv`, recovered as `1 − (episodes_used/G)/groups_drawn` |
| the three faults in that run | `tests/test_rollout_groups.py` |
