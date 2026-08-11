# Scale the core and the data: a 6x5 boss-only Spread grid

Status: Proposed
Supersedes: —
Depends on: [0009](0009-boss-data-scaling.md) (flat data curve at 2,500 episodes),
[0010](0010-dropout-regularization.md) (rejected model scaling at 666), [0012](0012-spread-grpo.md) (Spread-only scope)

**Question.** GRPO is now eliminated — eval [0014](../../contra_nes_evaluation/doc/0014-spread-specialty-boss-grpo.md)
finished at 9.5%, exactly its init. Data was eliminated at 2,500 boss episodes
([0009](0009-boss-data-scaling.md)). A 9,900-episode Spread release just landed. Is the
bottleneck capacity, or data beyond that scale?

**Answer.** One grid: **6 core sizes (1.6M–101M)** x the **5 nested prefixes already in
`boss-spread-10k-v1` (762–9,900 episodes)**, boss-only. Cache the frozen encoder's tokens
first — preprocessing is 97% of a step, so caching turns ~50 GPU-hours into ~3, which is
what buys a grid instead of a line.

**Caveat governing every table below.** All 9,900 traces share **one start state**
(`…_i371`) and validation is a holdout from it, so the **57-task mixed-v2 boss set is
co-primary**.

---

## 1. Why

Everything but capacity and data-past-2,500 has been eliminated:

| tried | where | boss result |
|---|---|---|
| BC, 4 data scales (58k → 464k frames) | [0009](0009-boss-data-scaling.md) | flat, ~90% death at every scale |
| dropout 0.0–0.3 | [0010](0010-dropout-regularization.md) | boss unmoved |
| sparse / graded GRPO | [0004](0004-grpo-experiment-plan.md), [0011](0011-boss-grpo.md) | 10.5%, then collapse |
| **binary GRPO, Spread + rapid** | [0012](0012-spread-grpo.md) / eval 0014 | **9.5% = init.** McNemar p = 1.0 |

Model size was never priced. Per frame on the 4090, batch 16 x 78 = 1,248 frames:

| leg | per frame | per batch |
|---|---:|---:|
| PyAV decode (PNG-in-MKV, 240x224) | 0.192 ms | 123 ms |
| cv2 resize → 256 | 0.097 ms | 62 ms |
| **frozen encoder → 512-d token** | **0.229 ms** | **286 ms** |
| trainable core @ 12.85M | — | 8.5 ms |
| trainable core @ 101M | — | 40.5 ms |

CPU overlaps GPU, so a step is GPU-bound at ~295 ms and **the core is 2.9% of it** — 7.9x the
parameters costs **+11% wall clock**. The 12.85M core inherited VPT's 4x512 shape in
[0002](0002-gpt-policy.md) and nothing since has questioned it. The encoder is frozen with no
pixel augmentation, so training re-encodes each frame identically 32x (D13) to 420x (D1);
§2.3 caches that away.

**The regime.** 770,679 frames / 12.85M params = 0.060 frames per parameter against
Chinchilla's ~20. Even a projected 40k release (39,900 episodes, ~3.1M frames, 0.242) leaves
this ~80x over-parameterized — hence §4's flat prediction, and a ladder that goes **down** as
well as up.

## 2. Design

### 2.1 The ladder

Constant head dim 64, aspect ratio d/L = 128, so width and depth scale together.

| cell | d_model | n_layer | n_head | params | x current |
|---|---:|---:|---:|---:|---:|
| XS | 256 | 2 | 4 | 1.61M | 0.14x |
| S | 384 | 3 | 6 | 5.31M | 0.43x |
| **M (current)** | **512** | **4** | **8** | **12.85M** | **1.00x** |
| L | 640 | 5 | 10 | 24.79M | 1.95x |
| XL | 768 | 6 | 12 | 42.48M | 3.34x |
| XXL | 1024 | 8 | 16 | 101.20M | 7.92x |

`model.py:92` pins `d_model` to the encoder's `hiddim`, so only M is expressible today. Add
`in_proj = nn.Linear(512, d_core, bias=False)`, or `nn.Identity()` when `d_core == 512` — no
parameters at M, so every existing checkpoint keeps its state-dict shape. That equality is a
test, not a hope.

### 2.2 Data cells and the fixed recipe

Read by `shard_count` from `manifest.json::train_scaling_prefixes`, never by glob. The top
cell is **13 shards, not 16** — plot against log2(frames).

| cell | shards | episodes | frames | epochs @ 20k steps x batch 16 |
|---|---:|---:|---:|---:|
| D1 | 1 | 762 | 59,305 | 420 |
| D2 | 2 | 1,524 | 118,610 | 210 |
| D4 | 4 | 3,048 | 237,219 | 105 |
| D8 | 8 | 6,093 | 474,297 | 52.5 |
| D13 | 13 | 9,900 | 770,679 | 32.3 |

Fixed across all 30 cells: `families: [boss]`, batch **16** (Spread episodes average 77.8
frames, so batch 4 is only ~312 tokens/step), 20,000 steps, AdamW, cosine, 500 warmup, bf16,
`aux_size: 0`, `value_head: false`, `dropout: 0.2`, frozen stage-A encoder, validation SHA
`29fd4017…cc9ae0` asserted at startup. Checkpoints at 3,000 / 10,000 / 20,000, fixed before
looking.

**LR is swept, not inherited** — one LR across a 63x parameter span is the standard way to
manufacture a false flat curve. `{1e-4, 3e-4, 1e-3}` per size at D13 for 3,000 steps, pick by
train loss, then hold; 18 runs, ~15 min cached. Publish the six chosen LRs.

0009's `family_draws` control is dropped — it holds the boss share of a *mixture* constant and
there is no mixture here. Recorded because it could not have been kept anyway: at 666 boss
draws per cycle it covers 76% of 9,900 episodes and **19% of 40k**, capping this very axis.

### 2.3 The token cache

One offline pass writes a `(T+1, 512)` bf16 array per episode (frames + goal token), memmapped
with a uid → (offset, length) index: **790 MB** for the 10k release, built in under ten
minutes, and `num_workers` drops to 0. The key **must** include the encoder checkpoint sha256
and `image_size` — a mismatch is a hard error, never a silent rebuild. Training-time only;
eval and GRPO still encode live.

Requested as an optional release sidecar on `kaihe/contra_nes_data#6`, but built here first
and the issue says declining is fine: §3 names unfreezing the encoder as the likely successor,
and a sidecar baked into an immutable release is bound to one encoder forever.

### 2.4 Evaluation — two axes

**In-distribution** is the release's own 100-task holdout, same start state; a clean curve
there is *not* evidence of a scaling law for play. **Transfer** is the 57-task mixed-v2 boss
val (SHA `1318…ecad`), reported as all 57, the **13 Spread+rapid** tasks, and the 23
Regular/Flamethrower tasks 0012 showed are unwinnable (floor check, out of the headline).

Primary comparison: **transfer / Spread subset, D13 XXL vs D13 M**, paired by task uid, 200
draws with replacement as in eval 0014. The 846-task suite is not run — a boss-only specialist
cannot play the other families, which is the price of this scope.

**Selection is by fixed step, never validation CE**: 0010 measured the CE-optimal checkpoint
playing 12.9 pp *worse* than the overfit final. CE is a diagnostic and the memorization-rate
signal the model axis is about; it selects nothing.

Staging: CE on all 30 cells x 3 seeds (~9 GPU-h); closed loop on 13 cells at seed 0 — the 6
sizes at D13, the 5 data cells at the selected size, the D1/D13 x XS/XXL corners. Top two
cells re-run at seeds 1 and 2 before any number is called a result.

### 2.5 The 20k / 40k extension

Predeclared, blocked on data, gated on nothing — if the later snapshots never arrive, the five
cells above still resolve every §4 prediction. When they land the axis extends to 7 points
with no recipe change, provided they keep the same 100-task validation and prefix scheme and
use **more than one start state**.

## 3. Rejected

- **Depth-only scaling** — free, no in-projection. But it caps at 38.5M and gets there at
  aspect ratio 43, and a flat curve from a badly shaped ladder is uninterpretable.
- **Unfreezing the encoder** — the obvious suspect, and 74% of the compute. Rejected as a
  *second variable*; it is the successor experiment, and this cache sharpens it.
- **muP** — the principled fix for LR-vs-width. Costs per-tensor plumbing and a coordinate
  check, against 18 runs that take 15 minutes. Revisit past 101M.
- **Keeping the 4-family mixture** — preserves comparability with every pooled number (65–69%)
  and the 846 guardrail. But the release is Spread-only boss with `baseline_train_episodes: 0`,
  so the mixture's boss slot would come from a different distribution than its own validation.
  Clean or comparable, not both; this takes clean.
- **Selecting on validation CE** — measured backwards in this project (0010). Listed because
  every scaling-law paper does it and someone will reintroduce it.

## 4. Predictions, registered before running

| axis | prediction | falsified by |
|---|---|---|
| model, at D13 | **flat or non-monotone** on transfer-Spread; XS within noise of M | XXL beats M by >5 pp, non-overlapping CIs |
| model, in-distribution | monotone — bigger cores memorize one state faster | flat here too, which indicts the encoder immediately |
| data, D1 → D13 | in-distribution rises, **transfer flat**, reproducing 0009 at 15x | transfer-Spread rises with log2(frames) |
| the gap | in-distribution exceeds transfer-Spread by **>20 pp** at every cell | a small gap, meaning one start state confounds less than feared |

If all four hold, neither capacity nor same-state data is the bottleneck, and the frozen
encoder becomes the primary suspect for the first time.

## 5. Risks and gates

| risk | gate |
|---|---|
| one start state makes every curve a memorization curve | transfer-Spread co-primary; the in-distribution/transfer gap is a reported quantity |
| cache diverges from the live encoder | key = encoder sha + image_size; test asserts cached == live to bf16 tolerance on 64 held-out frames |
| big cells undertrained, faking a flat model axis | per-size LR sweep (§2.2); six chosen LRs published |
| n = 13 cannot resolve the primary comparison | reported as "underpowered at n=13", never as "no effect" |
| 105–420 epochs on D1/D2 is pathological | train CE per cell; < 0.01 is labelled saturated on the plot, not dropped |
| batch 4 → 16 breaks comparison with 0009/0010 | it does; the anchor for this doc is M at D13, trained here |
| a live release changes underneath the grid | manifest filenames, episode assertions, both validation SHAs checked at startup |

## 6. Sequencing

1. **Token cache** — builder, memmap reader, key validation. Gate: builds in <15 min, ≤1 GB,
   cached D1 trains ≥20x faster. Report that ratio on `contra_nes_data#6`, which is blocked on
   it and on policy publishing `encoder-final.pt` (sha `f36041bc…1923c`; `runs/` is gitignored).
2. **In-projection** (§2.1) + the state-dict-identity test. Gate: `pytest tests/ -q` green, an
   existing checkpoint loads unchanged.
3. **Boss-only config.** Gate: a 100-step D1 smoke run reproduces manifest episode counts exactly.
4. **LR sweep.** Gate: publish the six LRs before the grid. Stop if a best LR sits at an endpoint.
5. **The 30-cell grid**, then the §2.4 closed-loop staging. No recipe changes mid-grid.
6. Update this doc with the measured grid, mark Implemented, record how each §4 prediction resolved.

---

## Appendix — provenance

| claim | source |
|---|---|
| 9,900 episodes / 770,679 frames / 5 prefixes / Spread-only / `generated_only` | `~/code/contra_nes_data/game_trace/releases/boss-spread-10k-v1/manifest.json` |
| all 9,900 uids share one source state; 100-task holdout SHA `29fd4017…cc9ae0` | same manifest — `train_shards[*].uids` split on `__` gives 1 distinct prefix |
| shards hold `.obs.mkv` + `.actions.npy` + `.goal.png` + `.json`, **no precomputed features** | `tar tf` over the 10k val shard and the legacy `game_trace/hf/boss-train-00000.tar` |
| decode 0.192 / resize 0.097 ms per frame, 77.85 frames/episode | PyAV + cv2 over 30 episodes / 2,270 frames, blobs preloaded, `cv2.setNumThreads(0)` |
| encoder 0.229 ms/frame; core 8.5 / 40.5 ms per 1,248-frame batch | microbenchmark, 4090 laptop 16 GB, bf16 autocast, 12 iters after 4 warmup |
| core parameter counts | `4d² + 3d·h + 2d` per layer, `h` from the SwiGLU rule at `causal.py:144` |
| `d_model` pinned to encoder `hiddim` | `src/contra_policy/model.py:92` |
| no pixel augmentation in the loader | grep for augment/jitter/flip over `dataset.py` — only sampling RNG |
| train CE 0.051 @ 666 episodes; CE-optimal plays −12.9 pp | [0010](0010-dropout-regularization.md) §1 |
| flat mixed-v2 curve; val SHA `1318…ecad` | [0009](0009-boss-data-scaling.md) |
| Spread GRPO finished at 9.5% = init, n=13, McNemar p = 1.0 | [eval 0014](../../contra_nes_evaluation/doc/0014-spread-specialty-boss-grpo.md) |
| Regular/Flamethrower 0 wins in 316 rollouts; 13 Spread+rapid val tasks | [0012](0012-spread-grpo.md) §1 |
| "20k"/"40k" name candidate traces; validation fixed at 100 per release | `~/code/contra_nes_data/doc/0003-incremental-spread-scaling.md` |
