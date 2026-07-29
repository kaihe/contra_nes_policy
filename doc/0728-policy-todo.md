# Policy repo work plan — after the 2026-07-28 evaluation

Baseline to beat: **56.5% pooled task completion** (63.5% with `--no-prev-action`),
`val/bc_acc` 0.76, from `runs/2026-07-28/12-20-29`, evaluated in
`contra_nes_evaluation/doc/0728.md`.

## Status — 2026-07-28

Tiers 0-3 are **implemented**. Tier 4 (RL) is not started.

| item | state | default |
|---|---|---|
| 0.1 `save_last` | landed | always on |
| 0.2 per-family metrics | landed | always on |
| 0.2b representative val subset | landed (new — see below) | always on |
| 1.1 `index_bias` | landed, folded into 2.1 | always on |
| 1.2 prev-action off | `config_noprev.yaml` | — |
| 2.1 goal heatmap head | landed, replaces exist/point/bbox | always on |
| 3.1 action class weighting | landed | `action_loss_alpha: 0.0` (off) |
| 3.2 memory carry | landed | `carry_memory: false` (off) |
| 3.3 family balancing | landed | `family_balance_alpha: 0.0` (off) |

The three Tier-3 knobs default to **off** so the prev-action run measures one variable,
per this document's own framing. Turn them on one at a time afterwards.

**Found while landing 0.2 — validation was measured on a skewed subset.** The shard
index sorts by tar path, so the val set is contiguous by family (boss, item, kill,
traverse) and `limit_val_batches` takes a *prefix*. At the shipped
`limit_val_batches: 50` / `batch_size: 16` that is 800 of 3125 windows: all 345 boss,
all 98 item, 357 of 658 kill, and **zero traverse** — which is 65% of the val set. Every
`val/` number in `0728.md` and `0726.md`, including the headline `bc_acc` 0.91 → 0.76,
was computed without a single traverse window. The val loader now takes a fixed seeded
permutation, so the subset is representative *and* stable across passes and runs. This
does not touch the closed-loop numbers, which come from the evaluator independently.

## Framing

The `boss` 0/57 result is **not** taken as a context-window problem. BC is per-step
supervised, so a 140-decision episode does not require the attention window to span it.
What boss actually needs is recovery behaviour from states no winning trace contains,
and the dataset is 523/523 wins. That is an RL problem, not a BC or a window-size one,
and no amount of BC data engineering reaches it. The plan below therefore spends its
BC effort on signal quality and its long-term effort on standing up RL fine-tuning.

Read every change closed-loop. `val/bc_acc` has already been shown to move opposite to
task completion once (0.91 → 0.76 while completion went 8.8% → 56.5%).

## Frozen interfaces

The evaluator pins these and aborts or silently misreports on drift. Nothing below may
change them:

- `goal.py` — `goal_mask` / `interaction_id` / `ppu_to_norm`, and how `goal.png` is consumed.
- `action_space.py` — the 21 logits are positional, checked against `contra_nes_data/src/agent/baseline.yaml`.
- The `point_err_px` formula, `loss.py:99-100`.

---

## Tier 0 — measurement hygiene

Land both before running any experiment. Until they do, no two runs are comparable.

### 0.1 Checkpoint the final weights

`save_freq: 2000` against a 7,959-step run means the last weights snapshot is step
6000 — **75% of training**. The 0728 evaluation was run on an undertrained checkpoint
and understates the result.

- Add `save_last=True` to `WeightsOnlyCheckpoint` in `train.py:90`, or set `save_freq`
  to divide the step count.
- **Done when:** a completed run leaves a snapshot at the final step.

### 0.2 Per-family validation metrics

Everything is currently pooled, so `traverse` (65.2% of training steps) dominates every
number, and `exist_acc` is meaningless on `kill`/`boss` where the goal is visible on
**100.0%** of frames — a constant "visible" predictor scores 100% there.

- Log `val/bc_acc`, `val/point_err_px`, `exist` per family. Needs the family tag on the
  batch; `dataset.py` can carry it from the shard name.
- Stop reporting `exist_acc` on families whose base rate is 100%, or report base rate
  alongside it.
- **Done when:** val logs break down by family, and no metric is reported against a
  degenerate base rate without that base rate next to it.

---

## Tier 1 — cheap isolated experiments

One training run each, one variable each.

### 1.1 Fix `index_bias`

Open since `0728-policy-request.md`, still unlanded.

With `num_view_tokens=4` and `use_prev_action=True`, the per-timestep block is
`[v0 v1 v2 v3, interaction, prev_action]`. The interaction token sits at `-2`, but
`model.py:103-108` starts `index_bias` at `-2` and decrements to `-3`, which lands on
the **last view token**. Because attention is causal over the flattened sequence, that
token precedes the interaction token and can never attend to it — so the aux head has
never known whether the task is kill, pick, avoid, traverse or boss.

- The initial value should be `-1`, decrementing to `-2`.
- `tests/test_pipeline.py::test_forward_shapes_and_token_layout` asserts `-3` and must
  be updated with it.
- **Hypothesis:** lifts `exist`/grounding most on `kill`, where distinguishing "kill
  this" from "avoid this" is exactly what the interaction token carries.
- **Note:** if 2.1 lands first, this is subsumed — the new aux head should read the
  interaction token by construction. Do whichever comes first, not both.

### 1.2 Decide the prev-action token

Dropping it at rollout is worth **+7.0 pp pooled (56.5 → 63.5) and +14.1 pp macro
(45.9 → 60.0)**, concentrated on `item` (`pick_spread` 25.0 → 66.7) and `kill`
(50.2 → 65.1). That is the largest free win currently on the table, and it is a
rollout-time feedback loop: the policy conditions on its own last action and locks in.

- Train one run at `prev_action_keep_prob: 0.10` and one with `use_prev_action: false`.
- Evaluate each both ways.
- **Done when:** there is a number saying whether the token earns its place. If it does
  not, delete it — it also costs a token per timestep, i.e. 1/6 of the sequence length.

---

## Tier 2 — replace the aux head with an entity heatmap

### 2.1 Goal heatmap head (no data-repo dependency)

Replaces the scalar `exist` + 2-D `point` + 4-D `bbox` regression with dense spatial
supervision. This fixes the degeneracy at its root rather than working around it: on a
frame where the goal is present, every pixel outside the blob is a **negative example**,
so a family with 100% goal visibility still supplies plenty of negatives, and a frame
with no goal is an all-zero target rather than a single masked-off scalar.

Target rendering is already written and needs no new code:
`goal.py:78 goal_mask(points, size, sigma_px)` returns exactly this map from a frame's
centroids and returns all-zero for an empty point list. It is the same function that
renders the cross-view goal channel, so target and prompt stay consistent by
construction.

Design:

- **Head:** `hiddim → S_aux²` (or a small deconv stack) off the interaction token,
  `S_aux = 16` or `32`. At `S_aux=32`, that is 512→1024, ~0.5M params.
- **Target:** `goal_mask(centroids[start+k], S_aux, sigma_px)` per timestep, built in
  `dataset.py` alongside the existing aux targets. These are **not** shifted with the
  actions — `centroids[j]` is read from the same RAM sample as `frames[j]`.
- **Loss:** per-pixel BCE-with-logits or MSE against the soft Gaussian, masked by the
  window `mask`. No separate `exist` term.
- **Keep `point_err_px` comparable:** derive `point` by soft-argmax over the predicted
  heatmap into normalised `[0,1]` coords and feed it to the **unchanged** formula at
  `loss.py:99-100`. `exist` can be reported as the heatmap max. The evaluator's pinned
  number then stays meaningful across the change.
- **Done when:** `point_err_px` is reported on the same scale as before and on-policy
  grounding (currently 22.3 px on the policy's own states, vs 15.9 teacher) improves.

### 2.2 Multi-class entity heatmap (needs the data repo)

The natural extension, and the frozen backbone is already primed for it: per
`encoder.py:8-10` the Contra autoencoder was pretrained with a **four-class entity
occupancy aux loss — player / player bullets / enemies / enemy bullets** — precisely so
2-4px sprites survive into the embedding. Predicting those four channels re-uses a
representation that already encodes them.

**Blocked on data:** the episode `.json` currently ships only `goal_points`,
`centroids` and `visibility`. Four-class occupancy needs the exporter to emit per-frame
entity positions by class. File that as a data-repo request; do 2.1 first, since it
needs nothing.

---

## Tier 3 — BC quality

### 3.1 Action class imbalance

The policy under-aims: `UR` is **10.0% of training steps (69,213)** but only 2.2% of
rollout actions; `URF` 2.7% → 0.5%. This is not data scarcity — `UR` is the second most
common action in the dataset — it is imbalance against `R` at 68.2%.

- Try inverse-frequency or focal weighting on the `pi_head` cross-entropy.
- Note 9 of the 21 logits are effectively dead (`LJ`, `UJ`, `DJ`, `UL`, `ULJ`, `ULF`
  never occur; `LF`/`U`/`UF` under 0.1%). Leave the table alone — it is frozen — but do
  not let those classes absorb weighting mass.
- **Done when:** the rollout `UR`/`URF` share approaches the expert's 11.1% / 3.2%.

### 3.2 Carry memory across windows

`lit.py:56` calls the policy with no `memory`, and `dataset.py` windows are
non-overlapping, so every training window starts from `initial_state` — zeros with
`state_mask` all-False, which `masked_attention.py:97` masks away entirely. Inside a
window, timestep 0 predicts from **1 frame** of context and timestep 31 from 32; the
average is ~16 steps. At rollout every step gets the full 32.

The `mem_len: 32` slots are therefore never populated during training at all. This is a
genuine train/inference gap, and it will matter more under RL.

- Needs an episode-ordered sampler (parallel episode streams, as VPT does) so
  consecutive windows of one episode land in the same batch lane, with memory passed
  between them. Conflicts with the current global shuffle — this is the expensive item
  in this tier.
- **Done when:** `mem_len` is non-zero in training and rollout `point_err_px`
  degradation (15.9 → 22.3) narrows.

### 3.3 Family-balanced sampling

`traverse` is 65.2% of training steps and is the family the policy already handles best
(68.3%); `item` is 2.8%. Most gradient currently goes to the solved task.

- A `WeightedRandomSampler` over the window table, weighted by family, is a few lines
  in `ContraDataModule` and costs nothing to try.
- **Done when:** macro completion (currently 45.9% vs 56.5% pooled) closes toward pooled.

---

## Tier 4 — RL fine-tuning

### 4.1 Stand up the RL path

This is where `boss`, the 27% death rate, and recovery from off-distribution states get
solved. BC cannot reach them: every source trace is a win, so no failure state is ever
demonstrated.

The scaffolding is deliberately already in place — `model.py:130-136` carries a
`value_head` that receives no gradient under BC (which is why the multi-GPU strategy is
`ddp_find_unused_parameters_true`), and `act()` at `model.py:242` is a working
single-step sampler with carried memory.

Open questions to settle before writing code:

- **Where does the loop live?** The emulator stepping and `goal_reached` scoring exist
  in `contra_nes_evaluation`. Either that harness grows a training mode or this repo
  takes a dependency on it. Do not fork the rollout code — it is the thing that keeps
  evaluation honest.
- **Reward:** each task's own `goal_reached` is a sparse terminal signal. Decide
  whether to shape with the aux grounding signal or the recorded solution length.
- **Algorithm:** PPO on the BC init is the conventional choice and matches the existing
  `value_head`.
- **Done when:** a BC-initialised policy improves boss completion above 0% and the
  death rate falls below the BC policy's 27%.

---

## Suggested order

1. **0.1, 0.2** — measurement hygiene, land together, no training run needed.
2. **1.2** — prev-action decision. Largest known free win (+7 pp pooled, +14 macro).
3. **2.1** — goal heatmap aux head. Subsumes 1.1.
4. **3.1, 3.3** — cheap BC quality, can share a run if measured per-family.
5. **3.2** — memory carry. Expensive, and a prerequisite worth having before RL.
6. **4.1** — RL fine-tuning; the only path to `boss` and to recovery behaviour.

Run 1.1 standalone only if 2.1 is going to be delayed.
