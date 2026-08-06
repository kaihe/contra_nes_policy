# Measure boss-data scaling before increasing the policy or RL budget

Status: Implemented
Supersedes: —
Depends on: [0006](0006-action-only-base-policy.md) (fixed model and BC recipe)

**Question.** Does giving the current action-only GPT more distinct, verified boss
demonstrations improve held-out boss play, or would a larger model and longer RL run
only scale a data bottleneck we have not resolved?

**Answer.** Measure the data curve first. Train the unchanged 12.86M-parameter policy
at the four nested `boss-mixed-v2` prefixes (58k, 116k, 232k and 464k boss decision
frames), with three seeds per prefix. Hold the model, optimizer steps, learning-rate
schedule, non-boss data, and *family draw counts* fixed. This last control matters:
natural episode sampling would raise boss from 8.7% to 43.1% of training decisions as
the prefix grows, confounding unique data with simple boss upsampling. Evaluate fixed
steps on the byte-identical 57-task boss validation split, and evaluate every final
checkpoint on the unchanged 846-task suite. Only choose a model-size sweep after this
curve says which data scale is worth carrying forward.

**Measured outcome (2026-08-05).** Ran as a reduced matrix — **seed 0 only**, four cells,
not the 12 predeclared here. The curve is flat: pooled 846 finals are 65.2 / 64.5 / 67.1 /
65.5% (D1/D2/D4/D8), D1→D8 paired +0.2 pp (p = 0.95); boss stays in a 0–15% band at ~90%
death at every scale. This is §3's predeclared row *"all four remain near 3.5% boss
success → do not scale model or RL compute on this premise"*, and that decision stands.
D4 final (67.1%) is the best base checkpoint the sweep produced. Full results:
[evaluation 0011](../../contra_nes_evaluation/doc/0011-boss-data-scaling.md). One caveat
that doc gets wrong: the offline CE gain it reports (2.13 → 1.76) is measured at step
20,000, but **at the CE minimum D1 and D8 are 0.713 and 0.703** — the data bought no
generalization at all, only a slower approach to overfitting. See
[0010](0010-dropout-regularization.md) §1.

Use `boss-mixed-v2`, not `boss-pure-v1`, for the primary result. Both contain the new
search wins, but pure-v1 derives all 2,034 examples from four fixed emulator states and
is therefore a labelled fixed-start/OOD ablation. Mixed-v2 retains the 466 published
train starts and is the release handed to policy for historical comparison.

---

## 1. Why data comes first

The first action-only run changed architecture and boss data together. It trained on
`boss-full-v1` (466 published + 200 searched demonstrations) and did not improve held-
out boss success:

| checkpoint | boss train | boss val success | pooled val success |
|---|---:|---:|---:|
| prior GPT BC final | 466 published | 3.5% | 67.5% |
| action-only base-v2 final | 466 published + 200 searched | 3.5% | 63.0% |

That result does not establish a data-scaling law: it has one added-data point, one
seed, and an architecture change. It does establish two requirements for the next
experiment. Closed-loop success, not minimum validation CE, is the gate; and the
action-only architecture must remain bit-identical across every cell.

The new release supplies a genuine nested curve:

| prefix | boss shards | boss episodes | boss decisions | multiple of D1 |
|---|---:|---:|---:|---:|
| D1 | 1 | 313 | 58,015 | 1.00x |
| D2 | 2 | 625 | 115,997 | 2.00x |
| D4 | 4 | 1,250 | 232,009 | 4.00x |
| D8 | 8 | 2,500 | 464,019 | 8.00x |

The prefixes are cumulative, frame-balanced and approximately weapon-stratified. The
57-example validation tar remains byte-identical with SHA-256
`131835e34c55f75ded04410976730600a866744b40a8525bfc4d7f9ab952ecad`.

### The sampling confound

The unchanged non-boss train set contains 6,438 episodes and 612,684 decisions. The
current loader visits every episode once per shuffled pass. Merely pointing it at a
larger boss prefix therefore changes the optimization mixture:

| prefix | natural boss episode share | natural boss decision share |
|---|---:|---:|
| D1 | 4.6% | 8.7% |
| D2 | 8.9% | 15.9% |
| D4 | 16.3% | 27.5% |
| D8 | 28.0% | 43.1% |

A positive curve under that loader could mean “the model saw boss more often,” not
“more unique boss traces helped.” The experiment therefore fixes the number of draws
from each family and changes only which distinct boss episodes those draws can cover.

## 2. Experimental design

### Data and deterministic prefix selection

Read the exact cumulative filenames from
`boss-mixed-v2/manifest.json::train_scaling_prefixes`; do not glob the directory and
truncate an incidental ordering. All non-boss train and validation shards remain the
ones used by 0006. Startup must assert the expected boss episode count for the selected
prefix and the 57-example validation hash.

`boss-pure-v1` is excluded from the main matrix. If the mixed curve is flat, its
matching generated-only prefixes may be run later to diagnose whether the published
starts or the four fixed search states dominate the result, with every report labelled
fixed-start/OOD.

### Fixed family schedule

Define one reference sampling cycle with the family counts from the 0006 training set:

| family | draws per reference cycle |
|---|---:|
| kill | 2,290 |
| item | 455 |
| traverse | 3,693 |
| boss | 666 |
| **total** | **7,104** |

Within a family, sample uniformly and deterministically from the selected prefix. A
smaller boss prefix cycles/shuffles to supply 666 draws; a larger prefix is sampled
without replacement across successive cycles so coverage is even over the full run.
Then apply the existing pooled length grouping. This preserves approximately the
0006 boss optimization share while keeping padding efficiency and all non-boss exposure
constant. Log realized episodes and valid action tokens per family to verify the
control rather than assuming it.

### Fixed model and optimization compute

Every cell uses the resolved 0006 architecture and recipe:

| component | fixed value |
|---|---|
| trainable model | action-only causal core, 12.86M parameters |
| encoder | same frozen Stage-A checkpoint |
| core | 4 layers, 8 heads, context 1024, dropout 0 |
| batch | 4 whole episodes, existing padded/length-grouped path |
| optimization | 20,000 steps, AdamW, lr 3e-4, cosine decay, 500 warmup |
| precision | bf16 |
| objective | masked action cross-entropy only |
| seeds | 0, 1, 2 for initialization and data order |

Twenty thousand steps means 80,000 episode draws in every cell. Because the family
schedule and length distributions are fixed, optimizer decisions are comparable; the
run must report actual valid action tokens as the final check. Do not switch to fixed
epochs: that would give larger datasets proportionally more optimizer compute and make
the data effect inseparable from compute scaling.

Save checkpoints at steps 3,000, 10,000 and 20,000. Those positions are fixed before
looking: the previous action-only run reached minimum validation CE at 3,000 while its
20,000-step checkpoint played better, so neither CE-best nor final alone represents the
learning trajectory.

The primary matrix is therefore 4 data prefixes x 3 seeds = 12 training runs. Model
size is one value; RL is absent.

## 3. Evaluation and decision rule

For every saved checkpoint:

1. Report boss validation action CE on the unchanged shard.
2. Run closed-loop evaluation on all 57 boss validation tasks with the existing
   temperature 1.0, frame skip, success predicate and 2x expert budget.
3. For every 20,000-step final, run the complete unchanged 846-task suite and report
   pooled, macro and per-family success, death and timeout.

The primary comparison is D8-final minus D1-final in boss closed-loop success, pooled
over the three predeclared seeds and paired by task UID. Report a task/seed bootstrap
95% interval rather than treating 171 correlated observations as independent. Plot
boss success and boss validation CE against log2 boss decision frames at all three
fixed steps.

Interpret the result as follows:

| observation | decision |
|---|---|
| D8 beats D1 and the intermediate prefixes rise | data scaling works; carry the smallest unsaturated prefix into the model-size experiment |
| D4 and D8 are indistinguishable | data saturates near D4; use D4 rather than paying for D8 in the parameter sweep |
| all four remain near 3.5% boss success | same-distribution searched demonstrations are not fixing closed-loop boss play; do not scale model or RL compute on this premise |
| boss rises but pooled/non-boss success falls | the shared policy has a capacity or interference problem; model scaling becomes justified, but the regression is part of the result |
| CE improves while closed-loop stays flat | imitation fit improved without behavioral transfer; do not call it data scaling success |

No checkpoint may be selected because of a single lucky 57-task observation. The
three fixed steps and three seeds are reported together. Full-suite regression larger
than 3 percentage points versus the same-seed D1 mean is a guardrail failure even if
boss improves.

## 4. What is deliberately deferred

**Parameter scaling.** Crossing four data sizes with several model sizes immediately
would be expensive and ambiguous. First locate the data regime. A following document
will define parameter counts and compute matching after 0009 selects D4, D8, or rejects
the premise.

**Long GRPO.** The attempted 0008 run was stopped at update 61 to fund data generation.
Its probe improved early but did not answer the fourteen-hour question. On-policy
compute should start from the best base checkpoint produced by the data/model stages,
not be scaled concurrently with them.

**Pure-generated training as the headline.** Pure-v1 has useful controlled actions but
only four initial emulator states. Success or failure on the 57-start validation set
would mix data scale with a large start-state distribution shift.

**Changing the optimizer with scale.** Learning-rate retuning, longer schedules and
fixed-epoch training may improve an individual cell, but each introduces a second
independent variable. They belong after the fixed-compute curve.

## 5. Risks and gates

| risk | why it is plausible | gate |
|---|---|---|
| searched traces are correlated | 2,034 wins originate from four search states | mixed-v2 retains published starts; report manifest nearest-distance distribution and reserve pure-v1 for OOD labelling |
| family sampling drifts with prefix size | natural decision share spans 8.7-43.1% | exact reference-cycle counts plus logged per-family draw/token totals |
| one seed creates a false curve | boss has only 57 validation tasks and stochastic rollouts | three predeclared train/eval seeds; paired task/seed uncertainty |
| lower CE is mistaken for better play | 0006 CE-best played 6.1 pp worse than final | closed-loop boss success is primary; CE is diagnostic only |
| boss gains erase general play | all families share the causal core | complete 846-task evaluation of every final; 3 pp guardrail |
| a live release silently changes | directory globs can consume later shards | manifest filenames, episode assertions and validation SHA check |

## 6. Sequencing

1. Add manifest-prefix selection and the fixed-family length-grouped sampler; test exact
   D1/D2/D4/D8 membership, reference-cycle counts and deterministic resumption.
2. Add periodic checkpoint saving at steps 3,000 and 10,000 plus per-family draw/token
   counters. Run the full unit suite and a one-step real-shard smoke test.
3. Train all 12 cells without changing the recipe after seeing an intermediate result.
4. Evaluate the three fixed checkpoints on boss validation and every final on the full
   suite; publish paired uncertainty and the scaling plots in evaluation.
5. Update this document with the measured curve and mark it Implemented. Then write the
   parameter-scaling design using the selected data prefix; do not preselect model
   sizes here.

---

## Appendix — provenance

| claim | source |
|---|---|
| mixed-v2 prefix counts, frames, weapons and shard hashes | `/home/kaihe/code/contra_nes_data/game_trace/releases/boss-mixed-v2/manifest.json` |
| mixed-v2 is the primary comparison release; pure-v1 is fixed-start/OOD | `kaihe/contra_nes_policy#2`; `contra_nes_data/doc/0001-boss-search-curriculum.md` sections 8-10 |
| 466 published + 2,034 searched demonstrations; no exact duplicates | the same mixed-v2 manifest |
| validation has 57 episodes and the stated SHA-256 | the same mixed-v2 manifest |
| non-boss counts and decisions | `load_or_build_index` over the three published non-boss train shards on 2026-08-05 |
| natural boss episode/decision shares | the preceding counts crossed with the four manifest prefixes |
| action-only base-v2 results and CE/checkpoint mismatch | `/home/kaihe/code/contra_nes_evaluation/doc/0010-action-only-base-v2.md` |
| fixed model and optimizer recipe | `src/contra_policy/config_bc.yaml`; `runs/bc/2026-08-04/15-19-58/resolved_config.yaml` |

