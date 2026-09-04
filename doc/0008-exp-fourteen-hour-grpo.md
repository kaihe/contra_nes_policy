# Does GRPO continue learning over fourteen hours?

Status: Superseded by 0011
Supersedes: —
Depends on: [0004](0004-exp-grpo.md) (GRPO stack and probes),
[0005](0005-design-graded-reward.md) (current reward),
[0006](0006-design-action-only-base-policy.md) (initial policy)

**Question.** The existing GRPO experiments stop after roughly 100 updates. Is their
plateau a real learning limit, or does the action-only GPT policy need a much longer
on-policy run before it learns robust closed-loop play?

**Answer.** Run the current graded-reward, normalised-advantage GRPO recipe for at most
14 wall-clock hours, starting once from
`runs/bc/2026-08-04/15-19-58/checkpoints/policy-final.pt`. At the measured 49.3 seconds
per update this is about 1,000 updates, 208,000 rollouts and 14.8 million decisions.
Change duration only: do not tune reward, sampling, learning rate, group size or KL
during the run. Add resumable checkpoints and a wall-clock limit before launch. Judge
learning from the fixed probe and held-out closed-loop evaluation across checkpoints,
not sampled training success and not the final checkpoint by default.

**Execution note (2026-08-04).** The run was intentionally stopped after update 61 to
reallocate the remaining compute to MC-search data generation. Its fixed probe showed
an early, noisy gain, but 61 updates do not answer this document's fourteen-hour
question. The resulting `grpo-final.pt` is a pilot artifact, not evidence for or
against long-horizon GRPO. The active sequence is now data scale, model scale, then RL
scale; see [0009](0009-exp-boss-data-scaling.md).

---

## 1. Why this experiment

The longest comparable recent runs contain 77-100 updates and last 1.0-1.7 hours:

| run | updates | mean seconds/update | total measured time |
|---|---:|---:|---:|
| `2026-08-03/09-23-22` | 100 | 60.9 | 1.69 h |
| `2026-08-03/18-54-05` | 100 | 48.9 | 1.36 h |
| `2026-08-04/09-45-33` | 77 | 49.3 | 1.05 h |

Those runs are long enough to validate the machinery but not to distinguish an early
plateau from slow accumulation. The latest 77-update run moved `kl_ref` from 0.001 to
about 0.05 while its fixed probe oscillated rather than trending cleanly. A tenfold
longer horizon makes “it needs more updates” falsifiable.

The current checkpoint format is insufficient for an unattended run. It saves policy
weights and metadata, but not AdamW moments, update index, RNGs, difficulty-sampler
state or rollout-generator state. A restart would therefore be a new optimization
process that merely inherited weights. The sampler and actor already expose `state()` /
`load_state()`; the trainer must persist and restore them.

## 2. Fixed experimental recipe

The run changes only its horizon and uses the current checked-in defaults:

| component | fixed value |
|---|---|
| initialization and frozen KL reference | BC `2026-08-04/15-19-58/policy-final.pt` |
| families | kill, item, traverse, boss |
| reward | success 1.0; boss failure HP progress coefficient 0.5 |
| advantages | group-normalised |
| group size / usable groups | 8 / 16 |
| sampling | current 50% boss mixture plus difficulty tournament |
| optimizer | AdamW, lr `5e-6`, one epoch, minibatch 8 |
| regularization | reference KL 0.02, entropy 0.01, target behaviour KL 0.02 |
| probe | 48 fixed tasks/family every 10 updates |
| checkpoints | every 25 updates plus final |

No hyperparameter may change after looking at an intermediate result. If a safety gate
fires, stop and record the failure; do not repair the same run in place.

## 3. Time budget and resumability

Add two execution controls:

```yaml
train:
  updates: 1200       # safety ceiling; wall time is the primary stop
  max_hours: 14.0
  resume_from: null
```

`max_hours` is checked after every complete update, so the run may exceed 14 hours by
one update. SIGTERM and SIGINT keep their graceful-save behavior.

A resumable GRPO checkpoint must contain:

- policy and optimizer state;
- completed update and cumulative active training seconds;
- NumPy trainer RNG plus Python, NumPy, Torch CPU and CUDA RNG states;
- `GroupSampler.state()` including its task sampler and difficulty tracker;
- rollout actor generator state;
- the fully resolved training configuration.

Resume must continue in the same run directory and append to `metrics.csv`. It must
refuse changes to optimization, reward, sampling, task, model or probe configuration;
only the wall-time ceiling may be extended. A deterministic interruption test compares
an uninterrupted short run with a save/reload continuation.

At 49.285 seconds/update, the primary estimate is:

```text
14 hours / 49.285 seconds = 1,023 updates
```

The 1,200-update ceiling protects against a much faster task mix while leaving wall
time as the actual experimental variable.

## 4. What “learns how to play” means

The difficulty-biased training success rate is not the answer: its task distribution
moves as the policy changes. Read these measurements instead:

1. **Fixed train probe:** record an initialization baseline before update 1, then every
   10 updates. Report macro and all four family rates with Wilson intervals.
2. **Held-out closed-loop evaluation:** after training, evaluate the BC initialization
   and checkpoints nearest updates 250, 500, 750 and final on the identical validation
   task set and seeds. Evaluation runs sequentially after training because the emulator
   permits only one owner process.
3. **Learning dynamics:** plot probe macro/families, entropy, `kl_ref`, `approx_kl`,
   zero-variance fraction, oversampling and decisions collected against both update and
   wall time.

A positive result requires held-out macro completion at least 5 percentage points above
the BC initialization, with no family falling more than 5 points. A family-specific
result is still reportable when one family improves by at least 5 points and the other
three remain within 2 points, but it is not evidence of general gameplay improvement.
Select the best checkpoint by held-out macro under these constraints; the final
checkpoint has no privileged status.

A flat or negative result after the full budget is also conclusive: this GRPO recipe
does not become effective merely by waiting ten times longer.

## 5. Safety gates

| risk | stop condition |
|---|---|
| numerical failure | non-finite loss, gradient norm, KL or logits |
| policy collapse | entropy below 0.10 at three consecutive fixed probes |
| catastrophic forgetting | probe macro down 10 points from initialization at two consecutive probes |
| no usable signal | oversample cap hit on three consecutive updates |
| runaway drift | `kl_ref > 0.20` at three consecutive probes |
| host-memory pressure | existing system usage guard exceeds 17 GB |
| context overflow | any rollout reaches the configured context limit; stop rather than truncate |

Every stop path saves a resumable checkpoint and the reason. Monitoring must alert on a
missing metric update for 15 minutes, but monitoring does not alter the run.

## 6. What was rejected

**Set `updates: 1000` and call it fourteen hours.** Update cost varies with episode
length and oversampling. Wall time is the intended treatment, so it must be represented
directly; update count is only a ceiling.

**Continue an older GRPO checkpoint.** That confounds duration with its earlier base
policy and recipe. This run starts once from the latest completed action-only BC model.

**Tune while it runs.** A 14-hour adaptive sequence cannot say whether duration helped.
Interesting failures become inputs to a later numbered experiment.

**Use sampled training success as the headline.** Difficulty sampling deliberately
changes its denominator. The fixed probe measures train tasks consistently; held-out
evaluation measures generalisation.

**Always report the final checkpoint.** Long RL can improve and then forget. Periodic
checkpoints exist so model selection follows the declared held-out criterion.

## 7. Sequencing

1. Implement and test full trainer resumption and `train.max_hours`.
2. Add the pre-update fixed probe so the exact BC initialization is row zero.
3. Run a 2-update interruption/resume smoke test and verify continuity.
4. Record the held-out BC baseline, then launch the 14-hour run with no config changes.
5. Evaluate checkpoints 250/500/750/final in `contra_nes_evaluation` and select by §4.
6. Update this document to `Implemented` with elapsed time, actual updates, stop reason,
   rollout count, learning curves and held-out results.

The evaluation handoff is filed only when the long run produces checkpoints; no other
repository is blocked on the implementation prerequisites.

---

## Appendix — provenance

| claim | source |
|---|---|
| current GRPO initialization | `src/contra_policy/rl/config_grpo.yaml` at `2be7859` |
| 49.285 seconds/update, 207.79 episodes/update, 14,847 decisions/update | `runs/grpo/2026-08-04/09-45-33/metrics.csv`, 77 rows |
| other measured run durations | `seconds` column in the run directories listed in §1 |
| current checkpoints omit optimizer and runtime state | `GRPOTrainer.save` before 0008 |
| sampler and rollout RNG state hooks already exist | `rl/tasks.py` and `rl/rollout.py` `state`/`load_state` methods |
| fixed probe is post-update and first runs at update 1 | `GRPOTrainer.run` before 0008 |
| one-emulator ownership constraint | `rl/rollout.py::claim_emulator` |
| latest completed action-only BC policy | `runs/bc/2026-08-04/15-19-58` |
