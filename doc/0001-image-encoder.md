# One token per frame: the Contra image encoder

Status: Implemented — §2 (the design) superseded by [0002](0002-symmetric-encoder.md)
Supersedes: —

> **0002 changes the architecture**: the encoder becomes goal-agnostic and goal matching
> moves to the policy's temporal attention. Everything below still stands as the record
> of what was built and measured — in particular §3 (rejected alternatives) and §4 (the
> `point_err_px` false alarm), which are architecture-independent.

**Question.** The policy spends 6 tokens per decision (`[view×4, interaction,
prev_action]`), which makes a task-length context unaffordable. Can one token per frame
carry as much as four, and can goal grounding move out of the temporal transformer into
the encoder?

**Answer.** Yes, on every family. A single 512-d token, with occupancy decoded back out
of it, reaches `peak_hit` **0.982–0.999** and `pck16` **0.983–1.000** across all four
families on the full val set, and **0.43–2.19 px** point error wherever that statistic
is well-defined. Entity occupancy is recovered at dice **0.96 player / 0.96 enemies /
0.91 enemy_bullets**.

An earlier draft of this doc reported boss "plateauing at 8.8 px" and called the gate
marginal. **That was a broken metric, not a broken encoder** — see §4. Boss is in fact
the strongest family.

Built as `src/contra_encoder/`; the policy is not yet retokenised.

---

## 1. Why — the evidence

**The context is far too short.** A 32-decision window fully contains **2.1%** of
training tasks:

| family | p50 budget | p90 | max |
|---|---|---|---|
| kill | 104 | 240 | 304 |
| item | 80 | 135 | 330 |
| traverse | 208 | 388 | 666 |
| **boss** | **281** | **610** | **1038** |

Boss is the family the wider effort exists to fix (91% death, 8.8% completion) and its
median task is 281 decisions — 14 seconds against a 1.6 s window.

**A longer window is not free.** Profiling one training update: optimiser 41.7%, rollout
inference 34.2%, emulator 12.7% — **76% is GPU**. Extending to 1024 decisions naively is
32× the per-step attention.

**Tokens per decision are the lever.**

| layout | tokens @ N=1024 | KV @ batch 16 |
|---|---|---|
| current `[view×4, interact, prev] × N` | 6,144 | 0.805 GB |
| **`[interact, goal, img × N]`** | **1,026** | **0.134 GB** |

A 6× reduction, which turns whole-episode context from a 32× cost into roughly 5×.

## 2. The design

```python
encode_goal(goal_image, goal_mask, interaction) -> (B, 512)              # once per chunk
encode_frame(frame, goal_token)                 -> (B, 512), (B, C, 32, 32)
```

- **One shared conv trunk** for frame and goal, as the policy does. A second trunk was
  11.2M duplicated parameters learning the same features from less data.
- **1×1 channel reduction before flattening.** `Linear(1024×16, 512)` is 8.4M parameters
  per projection — more than the trunk feeding it. Reducing 1024→256 first costs 0.26M
  and makes the projection 2.4M. Total **20.3M**, down from 50.5M in the first draft.
- **The frame encoder stays goal-conditioned**, via FiLM on the single goal token. A
  per-frame heatmap answers "where is *the goal entity*", which is unanswerable without
  the goal. What got cheaper is conditioning on one token instead of spatial
  cross-attention over the goal's whole patch grid.
- **Two heads, both decoded from the token**: a 1-channel goal map (feeding the pinned
  `point_err_px`) and a 4-channel entity map (`player`, `player_bullets`, `enemies`,
  `enemy_bullets`). Decoding from the *token*, not the conv map, is what forces spatial
  structure to survive the compression — there is a test asserting no gradient path
  around it.
- **`prev_action` is deleted, not merged.** It was already a constant
  (`prev_action_dropout` all zeros), and the ablation on the same checkpoint shows
  feeding it collapses boss 8.8% → 1.8% and doubles grounding error 5.3 → 12.1 px.

Entity sigma is **6 px, not the goal's 12**: at A=32 a 12 px sigma is 1.66 cells, which
smears a boss frame's ~4.9 enemy bullets into one blob.

## 3. What was rejected, and why

**Migrating to Stable-Baselines3.** Three of four blockers are in SB3's core training
loop: no recurrent state in `collect_rollouts`, `RolloutBuffer` shuffles *transitions*
(which hands a recurrent core a memory from another trajectory), and fixed `n_steps`
windows bootstrap at the boundary where we need complete unbootstrapped episodes with a
per-task budget. A migration ends up overriding everything SB3's PPO *is*, keeping only
`SubprocVecEnv` — whose entire target is the 12.7% of wall the emulator occupies, so
its ceiling is 1.15×.

**An LLM-scale backbone (Qwen-class) with vLLM.** We are GPU-bound at 76%; this
multiplies the dominant cost ~14× in parameters and ~32× in context. vLLM does not
reduce FLOPs, and its continuous batching wins on long generations — we emit one token
per environment step with a stateful emulator in the loop. It also discards the BC
checkpoint the whole run initialises from.

**Raw per-frame RAM in the shards.** This repo holds no RAM addresses at all; its
`get_ram()` calls pass an opaque array to interpreters in `contra_nes_data`. Shipping
RAM would force that knowledge across the boundary. Per-class positions in the existing
JSON are 20× smaller and consistent with how `centroids` already works.

**Merging `prev_action` into the image token.** Superseded by the ablation — the signal
is harmful whether or not it is recoverable from pixels.

**`exist_acc` as a gate.** Degenerate: the goal is visible on 100.0% of kill and 100.0%
of boss val frames, so a constant "visible" predictor scores 100% on both families that
matter.

**Plain MSE as the entity metric.** Not wrong, but unscaled — these maps are 95–98%
empty, so predicting *nothing* already scores 0.0021 (player) to 0.0065 (enemies), and
the baseline differs per class. Replaced by `dice` (bounded, comparable) and
`mse_skill` = `1 − MSE/MSE(zeros)` (the same quantity, referenced to that baseline).

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate | outcome |
|---|---|---|---|
| one token cannot hold enough spatial structure | four view tokens do it today; the heatmap needs 32×32 | `peak_hit` and `pck16` per family | **passed — 0.982–0.999 / 0.983–1.000** |
| per-frame grounding loses temporal smoothing | NES renders ≤8 sprites/scanline and flickers deliberately | same | **no sign of it** — boss scores highest |
| unfreezing the trunk destabilises BC | it has been frozen for every run to date | `val/bc_acc` vs the 72.8% baseline | not yet tested (stage B) |
| longer window blows host RAM | 20 GB WSL VM already binds | peak PSS under `tools/rss_guard.py` | **passed — 2.5 GB** |

### The `point_err_px` false alarm

`point_err_px` was the original gate and it was the wrong instrument. `points_to_target`
collapses a frame's goal centroids to their **mean**. Measured over the val split:

| family | centroids/frame | % multi-component | spread from their mean |
|---|---|---|---|
| kill / item / traverse | 1.00 | 0% | — |
| **boss** | **4.57** (max 7) | **98.7%** | **34.2 px** |

Boss is the only family where the "target point" names a spot where nothing is. Error
against it therefore **grows as a predictor gets sharper**: a blurry map's centre of
mass sits near the cloud's centre, a confident one sits on a component. That is exactly
what the run did — boss read 2.6 px at step 3000 and 8.8 px at step 20000, while
`peak_hit` climbed to 0.999 and `pck16` to 1.000.

Restricting the statistic to `n_goal_points == 1` settles it: on the 128 boss frames
where `point` is well-defined, the error is **0.43 px at pck8 = 1.00** — the best of any
family.

**Fix:** `ContraCrossViewDataset` now emits `n_goal_points`; point statistics are masked
to single-centroid frames and `multi_goal_frac` reports what was excluded, so a thin
sample can never pass as authoritative. `point_err_px` itself is untouched — it is a
frozen interface shared with `contra_nes_evaluation`; only the aggregation changed.

**Full val, corrected gate:**

| family | frames | peak_hit | pck16 | point frames | err_px | multi% |
|---|---|---|---|---|---|---|
| kill | 16,964 | 0.995 | 0.995 | 16,964 | 0.84 | 0% |
| item | 2,271 | 0.982 | 0.983 | 2,216 | 2.19 | 0% |
| traverse | 54,902 | 0.994 | 0.999 | 21,035 | 1.84 | 0% |
| boss | 10,071 | **0.999** | **1.000** | 128 | **0.43** | 99% |

Entity dice, per family (scored separately — the run's `validate` was dropping the
objective's metrics, since fixed):

| family | player | player_bullets | enemies | enemy_bullets |
|---|---|---|---|---|
| kill | 0.984 | 0.634 | 0.968 | 0.915 |
| item | 0.972 | 0.632 | 0.919 | 0.854 |
| traverse | 0.986 | 0.632 | 0.959 | 0.920 |
| boss | 0.956 | 0.735 | **0.991** | 0.902 |

## 5. Sequencing

1. **Stage A — encoder pretraining.** Done; `src/contra_encoder/`. Gate passed on
   every family once the metric was corrected (§4). Superseded by 0002, which retrains
   a goal-agnostic encoder.
2. **Stage B — retokenise `contra_policy.model`** to `[interact, goal, img × N]`,
   delete the `prev_action` machinery, rewire the heads. BC at `seq_len: 32` first, so
   a grounding regression surfaces in a cheap run. Gate: completion vs 72.8%.
3. **Stage C — `win_len: 256` with a per-chunk goal prefix, `maxlen` toward 1024.**
   The invariant that makes this safe: `maxlen ≥ chunk token length` means the chunk's
   own goal prefix can never be evicted by `clipped_causal`.
### Boss val power — considered and dropped

An earlier draft called the 57-task boss val split a blocker: separating 8.8% from
12.9% at 80% power needs ~920 tasks per arm, so it is ~16× underpowered, and the first
RL run's boss gain (4.9% → 12.9% on training rollouts, disjoint CIs) could not be
confirmed on held-out data.

**Not pursued.** Two reasons, and the second is the stronger:

- **RL manufactures boss experience.** BC is only the initialisation. A 500-update run
  already generated 2,093 boss episodes from 466 train tasks; at 5,000 updates that is
  ~21,000. Training data was never the constraint.
- **Boss is a single label.** All 466 train and 57 val tasks are `boss_level1` — one
  encounter, different starting states. "Generalising across boss tasks" is therefore a
  much narrower claim than for `traverse` (many level positions) or `kill` (five enemy
  types), and held-out tasks carry correspondingly less information.

With 2,000+ boss episodes on training rollouts against 57 val tasks, the **training
rollouts are the better-powered estimate**. Val's remaining job is detecting gross
overfitting, not measuring boss performance.

Revisit if either changes:

- **a run past ~5,000 updates** — per-task repetition reaches 45×, and at 180× (20,000
  updates) memorising 466 savestates becomes plausible;
- **a second boss exists** — then `boss` stops being one label and cross-boss
  generalisation is a real question that same-encounter tasks cannot answer.

---

## Appendix — provenance

| claim | source |
|---|---|
| task budget distribution | `len(actions)` over 6,904 train `.npz`, `budget = max(24, ceil(2×n))` |
| 76% GPU / 12.7% emulator | `tools/profile_collect.py`, applied to the measured 12.7 s collect + 9.1 s optimise split |
| token and KV figures | `2 × layers × tokens × hidden × 2 bytes × batch`, at 4 layers / 512 hidden / batch 16 |
| `prev_action` ablation | `contra_nes_evaluation/runs/0729-e18` vs `0729-e18-noprev`, same checkpoint |
| goal visible 100% kill/boss | full val sweep of `visibility` over 85,054 frames |
| entity target consistency | val sweep: 100.0% of visible kill/item/boss goals appear in `entities.enemies` |
| MSE baselines per class | 630 real target frames; `mean(target²)` for an all-zero predictor |
| encoder parameter counts | `sum(p.numel())` per submodule, `EncoderConfig()` defaults |
| val curve and final gate | `runs/encoder/2026-07-31/11-20-50/`, full val (782 batches, 85,054 frames) |
| boss centroid counts | val sweep of `len(centroids[j])` where `visibility[j]`, 10,128 boss frames |
| entity dice per family | `encoder-final.pt` re-scored on 200 val batches with the fixed metric path |
| peak host RAM 2.5 GB | `tools/rss_guard.py` peak group PSS |
