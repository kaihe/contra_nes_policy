# The image encoder: one token per image

Status: Implemented
Supersedes: —

**Question.** The policy spends 6 tokens per decision
(`[view×4, interaction, prev_action]`), which makes a task-length context
unaffordable. Can one token per frame carry as much as four?

**Answer.** Yes. `src/contra_encoder/` is a single symmetric function —

```python
encode(image) -> token, (entity_heatmap, [reconstruction])
```

— applied to agent frames and goal frames alike, with 4-class entity occupancy decoded
back out of the 512-d token. It recovers the entity structure a 4-token encoder held,
at **16.6M parameters**, and it knows nothing about goals: goal matching belongs to the
policy's temporal attention.

---

## 1. Why

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

**A longer window is not free.** Profiling one training update: optimiser 41.7%,
rollout inference 34.2%, emulator 12.7% — **76% is GPU**. Extending to 1024 decisions
naively is 32× the per-step attention.

**Tokens per decision are the lever.**

| layout | tokens @ N=1024 | KV @ batch 16 |
|---|---|---|
| current `[view×4, interact, prev] × N` | 6,144 | 0.805 GB |
| **`[interact, goal, img × N]`** | **1,026** | **0.134 GB** |

A 6× reduction, which turns whole-episode context from a 32× cost into roughly 5×.

## 2. The design

One conv trunk, one projection, one function — whatever image it is given:

```python
encode(image)      -> token (512)
entity_head(token) -> (4, 32, 32)   player / player_bullets / enemies / enemy_bullets
recon_head(token)  -> (3, 256, 256) optional, see §6
```

- **Occupancy is decoded from the token, never from the conv map.** That is what forces
  spatial structure through the 512-d bottleneck rather than letting it live in a
  feature map the head could read around. A test asserts no gradient path bypasses the
  token, *and* that the path exists when it is attached — otherwise it passes vacuously.
- **Goal frames need no special handling.** `goal.png` is a real episode frame with the
  target painted into the RGB — sampled at the goal points it reads (225, 110, 18)
  against an image mean of (56, 70, 14). An image with the answer drawn on it is still
  an image.
- **Goal frames are supervised, not merely encoded.** `goal_frame_idx` locates the
  frame, so `entities[cls][goal_frame_idx]` labels it exactly as any other. Both kinds
  go through one concatenated forward; the goal frame's dice is reported separately
  because it is ~1 row per window against ~32 and would otherwise vanish into the mean.
- **1×1 channel reduction before flattening.** `Linear(1024×16, 512)` is 8.4M
  parameters per projection, more than the trunk feeding it. Reducing 1024→256 first
  costs 0.26M and makes the projection 2.4M.
- **Entity sigma is 6 px, not the goal blob's 12.** At A=32 a 12 px sigma is 1.66 cells,
  which smears a boss frame's ~4.9 enemy bullets into one blob.

The policy then builds `[interaction, goal_token, img_token × N]` and computes goal
grounding on the **temporal** output, where `model.py` already computes it.

## 3. What was tried and rejected

**A goal-conditioned encoder.** The first implemented version modulated the frame
encode by a goal token via FiLM, so the encoder itself could answer "where is the goal
in this frame". It worked — see §5 — but it was the weaker half of the design. FiLM
produces one scale and one shift per channel, broadcast over the whole spatial grid: it
can say "attend to turret-ish features", never "look *there*". Attention between a
frame token and a goal token expresses the comparison directly, and already exists
downstream. Removing it deleted `mask_backbone`, `goal_reduce`, `goal_proj` and `film`
— **3.65M parameters** that existed only to answer a question the policy answers
better.

What that cost: stage A can no longer measure goal grounding at all, so a grounding
regression would first surface at stage B tangled with the retokenisation, the BC
retrain and the unfrozen trunk. Entity dice is the proxy — it measures the same spatial
content the goal head was reading.

**Migrating to Stable-Baselines3.** Three of four blockers are in SB3's core training
loop: no recurrent state in `collect_rollouts`, `RolloutBuffer` shuffles *transitions*
(handing a recurrent core a memory from another trajectory), and fixed `n_steps`
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

**`prev_action`, in any form.** It was already a constant (`prev_action_dropout` all
zeros), and the ablation on the same checkpoint is decisive: feeding it collapses boss
**8.8% → 1.8%** and doubles grounding error **5.3 → 12.1 px**. Merging it into the
image token was considered and dropped for the same reason — the signal is harmful
whether or not it is recoverable from pixels.

**`exist_acc` as a gate.** Degenerate: the goal is visible on 100.0% of kill and 100.0%
of boss val frames, so a constant "visible" predictor scores 100% on both families that
matter.

**Plain MSE as the entity metric.** Not wrong, but unscaled — these maps are 95–98%
empty, so predicting *nothing* already scores 0.0021 (player) to 0.0065 (enemies), and
the baseline differs per class. Replaced by `dice` (bounded, comparable across classes)
and `mse_skill` = `1 − MSE/MSE(zeros)` — the same quantity, referenced to that
baseline. They are complementary: dice cannot go negative, so it scores silence and
confident error identically; `mse_skill` distinguishes them.

## 4. The `point_err_px` false alarm

Worth keeping because the failure mode is one to recognise again: **a metric that
degrades as the model improves.**

`point_err_px` was the original gate. `points_to_target` collapses a frame's goal
centroids to their **mean**:

| family | centroids/frame | % multi-component | spread from their mean |
|---|---|---|---|
| kill / item / traverse | 1.00 | 0% | — |
| **boss** | **4.57** (max 7) | **98.7%** | **34.2 px** |

Boss is the only family where the target names a spot where nothing is. Error against
it therefore *grows as a predictor sharpens*: a blurry map's centre of mass sits near
the cloud's centre, a confident one sits on a component. The first trained encoder read
2.6 px at step 3000 and 8.8 px at step 20000 on boss, while `peak_hit` climbed to 0.999
and `pck16` to 1.000. Restricting to `n_goal_points == 1` settled it — on the 128 boss
frames where `point` is well-defined the error was **0.43 px**, the best of any family.

`ContraCrossViewDataset` now emits `n_goal_points` so any consumer can mask
multi-component frames. `point_err_px` itself is untouched: it is a frozen interface
shared with `contra_nes_evaluation`, and only the aggregation changed.

## 5. Results

**Goal-conditioned encoder** (20,000 steps, full val, 85,054 frames) — the design §3
rejected, kept as the baseline the current one had to match:

| family | peak_hit | pck16 | entity `enemies` | entity `enemy_bullets` |
|---|---|---|---|---|
| kill | 0.995 | 0.995 | 0.968 | 0.915 |
| item | 0.982 | 0.983 | 0.919 | 0.854 |
| traverse | 0.994 | 0.999 | 0.959 | 0.920 |
| boss | 0.999 | 1.000 | 0.991 | 0.902 |

**Goal-agnostic encoder** (20,000 steps, full val) — beats it on every class, and had
already matched it by step 5,000:

| | player | player_bullets | enemies | enemy_bullets |
|---|---|---|---|---|
| goal-conditioned baseline | 0.984 | 0.634 | 0.960 | 0.910 |
| step 5000 | 0.977 | 0.660 | 0.960 | 0.923 |
| **final (20000)** | **0.99** | **0.70** | **0.98** | **0.97** |

Per family at the end: `kill 0.98/0.96 · item 0.97/0.93 · traverse 0.98/0.97 ·
boss 0.99/0.97` (enemies / enemy_bullets). **Boss scores highest**, and its
`enemy_bullets` — ~4.9 sprites of ~2 px per frame, the class most relevant to
surviving a boss fight — went 0.910 → 0.97.

Goal-frame dice is `0.99 / 0.70 / 0.98 / 0.98`, indistinguishable from the agent-frame
column. The painted marker did not push goal frames out of distribution, which was the
open risk when we chose to supervise them.

`player_bullets` is the weak class in both designs (0.70 against ~0.98 for the others).
Not diagnosed. It is the one class the policy plausibly does not need — the player's own
bullets are a consequence of its actions, not a thing to react to.

## 6. Open: is reconstruction worth it?

Implemented behind `reconstruct: false`. A decoder is 7M at `recon_depth: 16` (27.97M
at full width — larger than the entire encoder).

The precedent is thinner than it looks. `contra_agent/dreamer/train_ae.py` trains recon
+ entity together, but its claim is that a **recon-*only*** encoder "goes entity-blind"
— an argument for adding the entity head, not evidence reconstruction helps once you
have one. Its decoder also had a second job there (the world model renders).

Settle by ablation: entity-only vs entity+recon, on entity dice and stage-B completion.

## 7. Sequencing

1. **Stage A — encoder pretraining.** Done.
2. **Stage B — retokenise `contra_policy.model`** to `[interact, goal, img × N]`,
   delete the `prev_action` machinery, restore the goal heatmap head on the temporal
   output. BC at `seq_len: 32` first, so a grounding regression surfaces in a cheap
   run. Gate: completion vs the 72.8% BC baseline. **This is where goal grounding is
   measured for the first time under the goal-agnostic design.**
3. **Stage C — `win_len: 256` with a per-chunk goal prefix, `maxlen` toward 1024.**
   The invariant that makes it safe: `maxlen ≥ chunk token length` means the chunk's
   own goal prefix can never be evicted by `clipped_causal`.
4. **Reconstruction ablation** (§6), independent of the above.

### Boss val power — considered and dropped

An earlier draft called the 57-task boss val split a blocker: separating 8.8% from
12.9% at 80% power needs ~920 tasks per arm, so it is ~16× underpowered, and the first
RL run's boss gain (4.9% → 12.9% on training rollouts, disjoint CIs) could not be
confirmed on held-out data.

**Not pursued.** Two reasons, the second stronger:

- **RL manufactures boss experience.** BC is only the initialisation. A 500-update run
  already generated 2,093 boss episodes from 466 train tasks; at 5,000 updates that is
  ~21,000. Training data was never the constraint.
- **Boss is a single label.** All 466 train and 57 val tasks are `boss_level1` — one
  encounter, different starting states. "Generalising across boss tasks" is a much
  narrower claim than for `traverse` or `kill`, so held-out tasks carry less
  information than the count suggests.

With 2,000+ boss episodes on training rollouts against 57 val tasks, the **training
rollouts are the better-powered estimate**. Val's remaining job is detecting gross
overfitting.

Revisit if either changes: **a run past ~5,000 updates** (per-task repetition reaches
45×, and 180× at 20,000, where memorising 466 savestates becomes plausible), or **a
second boss** (then `boss` stops being one label).

---

## Appendix — provenance

| claim | source |
|---|---|
| task budget distribution | `len(actions)` over 6,904 train `.npz`, `budget = max(24, ceil(2×n))` |
| 76% GPU / 12.7% emulator | `tools/profile_collect.py` over the measured 12.7 s collect + 9.1 s optimise split |
| token and KV figures | `2 × layers × tokens × hidden × 2 bytes × batch`, at 4 layers / 512 hidden / batch 16 |
| blob painted into `goal.png` | pixel sample at `ppu_to_norm(goal_points)` |
| `goal_frame_idx` correctness | matches pixel-matched frame index on 8/8 episodes across all families |
| `prev_action` ablation | `contra_nes_evaluation/runs/0729-e18` vs `0729-e18-noprev`, same checkpoint |
| goal visible 100% kill/boss | full val sweep of `visibility`, 85,054 frames |
| boss centroid counts | val sweep of `len(centroids[j])` where `visibility[j]`, 10,128 boss frames |
| MSE baselines per class | 630 real target frames, `mean(target²)` for an all-zero predictor |
| removed parameter counts | `sum(p.numel())` per submodule |
| `ConvDecoder` 27.97M | `dreamer.models.ConvDecoder(256, depth=32, feat_dim=1024)` |
| goal-conditioned results | `runs/encoder/2026-07-31/11-20-50/`, full val (782 batches) |
| goal-agnostic results | `runs/encoder/2026-07-31/18-00-11/`, `phase=val_full`, whole val set |
