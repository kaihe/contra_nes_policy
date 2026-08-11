# Scale the core, not just the data: a 6x5 boss-only Spread grid on cached encoder tokens

Status: Proposed
Supersedes: —
Depends on: [0009](0009-boss-data-scaling.md) (the flat mixed-v2 data curve this
re-runs at 15x the boss data), [0010](0010-dropout-regularization.md) (which **rejected**
model scaling — §1 says why that rejection expires), [0012](0012-spread-grpo.md) (the
Spread-only scope this inherits)

**Question.** Boss has sat at 7–11% success through four BC data scales, a dropout sweep,
sparse GRPO, graded GRPO and — as of eval [0014](../../contra_nes_evaluation/doc/0014-spread-specialty-boss-grpo.md)
— Spread-only binary-reward GRPO, which finished at **9.5%, exactly its init**. The
optimizer has now been eliminated. Data has been eliminated at the 2,500-episode scale.
That leaves **capacity** and **more data than has ever been tried**, and a 9,900-episode
Spread release just landed. Which one moves?

**Answer.** Run both axes as one grid, boss-only and Spread-only, on **cached frozen-encoder
tokens**. Six core sizes spanning **1.6M → 101M** parameters (0.14x to 7.9x the current
12.85M) crossed with the five nested data prefixes already in the release
(**762 → 9,900** episodes). The cache is what makes this affordable: the frozen encoder is
**~74% of a training step** and the core is 14 ms of it, so precomputing 790 MB of bf16
tokens once turns a 1-hour cell into a **3–14 minute** cell and the whole 30-cell grid into
about **4 GPU-hours**. Model scaling was never expensive — it was never measured.

**This doc predicts the model axis is flat and says so before running it** (§4). 0010
measured train CE at 0.051 with 666 boss episodes: a variance problem, not a capacity
problem. The premise that expires is the *data* half — at 9,900 episodes the memorization
headroom is 15x larger, and a capacity claim made at 666 episodes does not carry. If the
model axis is flat at 9,900 too, that is a real result and it points at the frozen encoder,
not the core.

**The binding caveat, stated once and carried into every table:** all 9,900 traces share
**one source emulator state** (`win_level1_20260701015306_i371`), and the 100-task
validation split is a holdout *from that same state*. This is narrower than `boss-pure-v1`,
which 0009 already labelled fixed-start/OOD at four states. A curve read only on the
100-task holdout measures how fast each cell memorizes one boss encounter. The
**57-task mixed-v2 boss validation is therefore co-primary**, not a footnote — see §2.4.

---

## 1. Why — the evidence

### Everything except capacity and data-beyond-2,500 has now been eliminated

| what was tried | where | boss result |
|---|---|---|
| BC at 4 boss data scales (58k → 464k frames) | [0009](0009-boss-data-scaling.md) | flat, ~90% death at every scale |
| dropout 0.0–0.3 | [0010](0010-dropout-regularization.md) | best cell +3.5 pp pooled, boss unmoved |
| sparse GRPO | [0004](0004-grpo-experiment-plan.md) | 3.5% → 10.5% |
| graded GRPO, 10 h / 1,619 updates | [0011](0011-boss-grpo.md) | collapse; held-out 7.5/11.0/7.5% vs 8.5% init |
| **binary GRPO, Spread + rapid only** | [0012](0012-spread-grpo.md) / eval 0014 | **9.5% = init, exactly.** McNemar p = 1.0 |

0012 was built as the decisive optimizer test: *"if boss does not move here, it will not
move anywhere, and the bottleneck is the policy or the representation."* It did not move.
That sentence is now a standing instruction, and this doc executes it.

### Model scaling costs almost nothing, which is why nobody noticed it was untested

Measured on the 4090 laptop (16 GB), one batch of 4 x 521 frames = 2,084 tokens:

| component | time | share |
|---|---:|---:|
| frozen encoder forward (chunk 256) | **477.4 ms** | — |
| core fwd+bwd+opt @ 12.85M (d512 L4) | 13.8 ms | 2.8% of encoder |
| core fwd+bwd+opt @ 101M (d1024 L8) | 67.7 ms | 14% of encoder |
| core fwd+bwd+opt @ 198M (d1280 L10) | 133.1 ms | 28% of encoder |

At the realized mean batch of a 0010 run (~633 padded tokens/step, 195 ms mean step) the
encoder is **~74% of wall clock**. Going 12.85M → 101M adds ~16 ms to a 195 ms step: **+8%**.
The 12.85M core is small because it inherited VPT's 4x512 shape in [0002](0002-gpt-policy.md),
not because a larger one was priced and rejected.

The same arithmetic says the *right* engineering move is to delete the encoder from the
inner loop entirely. It is frozen (`freeze_encoder: true`), and the loader applies **no
pixel augmentation** — every frame maps to exactly one deterministic 512-d token forever.

### The regime, in tokens per parameter

| release | episodes | decision frames | frames/param @ 12.85M |
|---|---:|---:|---:|
| mixed-v2 D8 (0009's largest) | 2,500 | 464,019 | 0.036 |
| **boss-spread-10k-v1** | 9,900 | 770,679 | 0.060 |
| projected 20k | ~19,800 | ~1.54M | 0.120 |
| projected 40k | ~39,600 | ~3.08M | 0.240 |

Chinchilla's compute-optimal ratio is ~20 tokens/parameter. Even at 40k traces this project
is **~80x over-parameterized** at the *current* size. That is the quantitative reason §4
predicts the model axis is flat, and the reason the data axis is predicted to be the live
one. It is also why the ladder goes **down** as well as up: if 12.85M is already past
saturation, the interesting number is where the curve stops being flat below it.

## 2. The design

### 2.1 The cache

One offline pass writes, per episode, a `(T+1, 512)` bf16 array — `T` frame tokens plus the
one goal-frame token — keyed by episode uid, alongside actions and lengths. Sizes:

| release | frames | cache (bf16) |
|---|---:|---:|
| boss-spread-10k-v1 | 770,679 | **790 MB** |
| projected 40k | ~3.08M | ~3.2 GB |

Stored as a memmapped `.npy` plus a uid → (offset, length) index, so the 20 GB WSL VM never
holds more than a batch (see memory note in `doc/README.md` open questions). The cache key
must include the **encoder checkpoint sha256** and `image_size`; a mismatch is a hard error,
not a silent rebuild. Training reads tokens directly and skips `encode_images` entirely.

This is a training-time optimization only. Closed-loop evaluation and GRPO still run the
encoder live — they see frames the cache has never met.

### 2.2 The model ladder

Constant head dim 64, aspect ratio d/L = 128 held fixed so depth and width scale together
and RoPE numerics are identical across cells.

| cell | d_model | n_layer | n_head | core params | x current |
|---|---:|---:|---:|---:|---:|
| XS | 256 | 2 | 4 | 1.61M | 0.14x |
| S | 384 | 3 | 6 | 5.31M | 0.43x |
| **M (current)** | **512** | **4** | **8** | **12.85M** | **1.00x** |
| L | 640 | 5 | 10 | 24.79M | 1.95x |
| XL | 768 | 6 | 12 | 42.48M | 3.34x |
| XXL | 1024 | 8 | 16 | 101.20M | 7.92x |

**The one architectural change.** `model.py:92` pins `d_model` to the encoder's `hiddim`,
so no cell but M is currently expressible. Add an in-projection:

```python
d_enc  = self.encoder.cfg.hiddim                    # 512, fixed by stage A
d_core = int(cfg.core.get("d_model") or d_enc)      # the experiment variable
self.in_proj = (nn.Identity() if d_core == d_enc
                else nn.Linear(d_enc, d_core, bias=False))
```

applied to frame and goal tokens; `interaction`, `pi_head`, `value_head` and `aux_head`
become `d_core`-wide. `nn.Identity()` at `d_core == 512` carries **no parameters**, so every
existing checkpoint keeps its exact state-dict shape and the M cell stays bit-identical to
the 0006/0009/0010 anchor. That equality is a test, not a hope
(`test_identity_projection_preserves_state_dict`).

### 2.3 Data cells and the fixed recipe

Read from `boss-spread-10k-v1/manifest.json::train_scaling_prefixes` by `shard_count` —
never by directory glob. Note the top cell is **13 shards, not 16**: 1.63x the 8-shard cell,
not 2x. Plot against log2(frames), not cell index.

| cell | shards | episodes | frames | epochs @ 20k steps x batch 16 |
|---|---:|---:|---:|---:|
| D1 | 1 | 762 | 59,305 | 420 |
| D2 | 2 | 1,524 | 118,610 | 210 |
| D4 | 4 | 3,048 | 237,219 | 105 |
| D8 | 8 | 6,093 | 474,297 | 52.5 |
| D13 | 13 | 9,900 | 770,679 | 32.3 |

Fixed across all 30 cells: `families: [boss]`, batch **16** episodes (raised from 4 — Spread
episodes average **77.8** frames against the mixed set's ~103, so batch 4 is only ~312
tokens/step), 20,000 steps, AdamW, cosine decay, 500 warmup, bf16, `aux_size: 0`,
`value_head: false`, `dropout: 0.2` (0010's best cell), frozen stage-A encoder.
Validation SHA-256 `29fd40177d9277931d1a115d8a36171c7eacc1a02c17cf1a4f7093ac94cc9ae0`
asserted at startup. Checkpoints at steps 3,000, 10,000 and 20,000, fixed before looking.

0009's `family_draws` control is **dropped, not broken**: it exists to hold the boss share
of a four-family mixture constant, and there is no mixture here. Its failure mode is worth
recording anyway — at 666 boss draws per cycle, a 20,000-step run touches 7,500 draws, which
covers 76% of 9,900 episodes and would cover **19% of 40k**. The control that made 0009
sound would have silently capped the data axis this doc exists to measure.

**Learning rate is swept, not inherited.** A too-high LR at large width is the single most
common way a model-size sweep manufactures a false flat curve. Before the grid, sweep
`{1e-4, 3e-4, 1e-3}` per model size at D13 for 3,000 steps and pick by train loss; then hold
that LR fixed for every data cell of that size. Publish the six chosen LRs. 18 short runs,
~20 minutes total on the cache.

### 2.4 Evaluation — two axes, and the second one is the real one

Every checkpoint is read on both:

1. **In-distribution (100-task generated holdout).** Action CE plus closed-loop success on
   the release's own holdout. Same start state as training. This axis will produce a clean
   curve; a clean curve here is **not** evidence of a scaling law for play.
2. **Transfer (57-task mixed-v2 boss validation,** SHA `1318…ecad`**).** Published starts,
   all four weapons. Reported three ways: all 57; the **13 Spread+rapid** tasks 0012
   identified as the specialty pool; and the 23 Regular/Flamethrower tasks that 0012 showed
   are structurally unwinnable (expected 0.0%, included as a floor check, excluded from the
   headline).

The primary comparison is **transfer, Spread subset, D13 XXL vs D13 M**, paired by task uid
with a task/seed bootstrap interval. The 846-task suite is **not** run: a boss-only Spread
specialist cannot play kill/item/traverse, and reporting a pooled number for it would be
meaningless. This is the price of the boss-only scope and it is paid knowingly.

**Checkpoint selection is by fixed step, never by validation CE.** 0010 measured the
CE-optimal checkpoint playing **12.9 pp worse** than the overfit final, and killed every
reweighting of CE as a proxy. CE is reported as a diagnostic and as the *memorization rate*
signal the model axis is actually about; it selects nothing.

Cost staging, because rollouts are the expensive half: CE on all 30 cells x 3 seeds; closed
loop on the 6 model cells at D13, the 5 data cells at the selected size, and the D1/D13 x
XS/XXL corners — 13 cells, seed 0. The two best cells are then re-run at seeds 1 and 2
before any number is called a result.

### 2.5 The 20k and 40k extension

Predeclared here, blocked on data. `boss-spread-10k-v1` is the only release that exists;
`contra_nes_data` [0003](../../contra_nes_data/doc/0003-incremental-spread-scaling.md)
commits to further snapshots under the same contract. When they land, the data axis extends
to 7 points (762 → 39,600) at the size selected by phase A, with **no recipe change**.

Three things must hold for the extension to be comparable, and they are requests on the data
repo, not assumptions: the **same 100-task validation split**, the **same deterministic
prefix scheme**, and — the one that matters most — **more than one source start state**.
A 40k release that is still one state scales memorization, not competence. No handoff issue
is filed yet; see §5 step 6.

## 3. What was rejected

**Depth-only scaling.** Free — no in-projection, no code change. Rejected because at fixed
d=512 the ladder runs out at L=12 (38.5M) and gets there with aspect ratio 512/12 = 43,
far off the ~128 that every reference decoder uses. A flat curve from a badly-shaped ladder
is uninterpretable, and the projection costs 0.13–0.52M parameters.

**Unfreezing the encoder as part of this experiment.** It is the obvious suspect — §1's
elimination argument points at the representation, and the encoder is 74% of the compute
that a scaling experiment is trying to spend on the core. Rejected as a *second variable*.
If the model axis is flat at D13, "unfreeze and re-run the ladder" is 0014's headline, and
the cache built here is exactly what makes the comparison sharp. Doing both at once means a
positive result cannot be attributed.

**muP.** The principled fix for LR-vs-width, and it would replace §2.3's sweep with a
transfer rule. Rejected on cost: it needs per-tensor multiplier plumbing through
`causal.py` and a coordinate check to verify, against 18 runs that take 20 minutes on cached
tokens. Revisit if the ladder is extended past 101M, where the sweep gets expensive.

**Keeping the four-family mixture.** Preserves comparability with every pooled number this
project has (65–69%) and keeps the 846-task regression guardrail. Rejected because the
release is Spread-only boss with `baseline_train_episodes: 0`, so the mixture's boss slot
would be filled by data from a different distribution than its own validation set, and
§2.3's coverage arithmetic caps the data axis at 19% by 40k. The scaling question gets a
clean answer or a comparable one, not both; this doc takes clean.

**Selecting on validation CE.** See §2.4. Recorded here because it is the single most likely
thing for a future reader to reintroduce — every scaling-law paper selects on held-out loss,
and in this project that is measured backwards.

## 4. Predictions, registered before running

Written down first so a flat result is a result rather than a disappointment.

| axis | prediction | what would falsify it |
|---|---|---|
| **model, at D13** | **Flat or non-monotone** on transfer-Spread. XS (1.6M) within noise of M. | XXL beats M by >5 pp on transfer-Spread with non-overlapping CIs |
| **model, in-distribution** | **Monotone improvement.** Larger cores memorize one start state faster — CE falls, closed-loop-on-holdout rises | in-distribution flat too, which would indict the encoder immediately |
| **data, D1 → D13** | in-distribution rises; **transfer flat**, reproducing 0009 at 15x the data | transfer-Spread rises monotonically with log2(frames) |
| **the gap** | in-distribution success exceeds transfer-Spread success by **>20 pp** at every cell | a small gap, which would mean the single start state is less of a confound than §0 claims |

The honest reading if all four hold: neither capacity nor same-state data is the bottleneck,
and the frozen encoder becomes the primary suspect for the first time in the project.

## 5. Risks, and the gate on each

| risk | why it is plausible | gate |
|---|---|---|
| single start state makes every curve a memorization curve | 9,900/9,900 uids share `..._i371`; validation is a holdout from it | transfer-Spread (n=13) is co-primary; the in-distribution/transfer gap is a reported quantity, not a footnote |
| cache silently diverges from the live encoder | GRPO and eval run the encoder on frames the cache never saw | cache key = encoder ckpt sha256 + image_size; a test asserts cached tokens equal a live forward to bf16 tolerance on 64 held-out frames |
| big cells are undertrained, faking a flat model axis | one LR for a 63x parameter span | per-size 3-point LR sweep (§2.3); the six chosen LRs are published |
| n = 13 cannot resolve the primary comparison | 0012 already hit this; eval 0014 reports 19/200 both arms | 200 draws with replacement as in 0014; a null is reported as "underpowered at n=13", never as "no effect" |
| 105–420 epochs on D1/D2 is pathological, not informative | fixed-step budget over a 13x data span | train CE reported per cell; a cell at train CE < 0.01 is labelled saturated on the plot rather than dropped |
| batch 4 → 16 breaks comparison with 0009/0010 | it does | the M cell is not claimed comparable to those runs; the anchor for this doc is M at D13, trained here |
| a live release changes underneath the grid | directory globs consume later shards | manifest filenames, episode-count assertions, both validation SHAs checked at startup |

## 6. Sequencing

1. **Token cache.** Builder + memmap reader + cache-key validation. Test: cached tokens
   match a live encoder forward; a wrong encoder sha raises. Gate: full 10k cache builds in
   under an hour and is ≤ 1 GB.
2. **In-projection.** `model.py` change per §2.2, plus the state-dict-identity test at
   d=512. Gate: `pytest tests/ -q` green, and an existing checkpoint loads unchanged.
3. **Boss-only config.** `config_bc_scaling.yaml` — `families: [boss]`, the 10k manifest,
   batch 16, no `family_draws`. Gate: a 100-step smoke run on D1 reproduces episode counts
   from the manifest exactly.
4. **LR sweep**, 6 sizes x 3 LRs x 3,000 steps at D13. Gate: publish the six chosen LRs
   before starting the grid. Do not proceed if the best LR sits at a sweep endpoint.
5. **The 30-cell grid**, 3 seeds for CE, then the §2.4 closed-loop staging. Do not change
   the recipe after seeing an intermediate cell.
6. **Extension.** Only after §4's predictions have resolved: file the `contra_nes_data`
   handoff issue for 20k/40k with the three §2.5 requirements, the multi-start-state one
   argued from whatever the in-distribution/transfer gap turns out to be. Deferred
   deliberately — the gap is the evidence that makes the request specific.
7. Update this doc with the measured grid, mark it Implemented, record how each §4
   prediction resolved.

---

## Appendix — provenance

| claim | source |
|---|---|
| 9,900 episodes / 770,679 frames / 5 nested prefixes / Spread-only / `generated_only` | `~/code/contra_nes_data/game_trace/releases/boss-spread-10k-v1/manifest.json` |
| all 9,900 uids share one source state `win_level1_20260701015306_i371` | same manifest, `train_shards[*].uids` split on `__`, 1 distinct prefix |
| 100-task holdout, SHA `29fd4017…cc9ae0`, `kind: generated_holdout` | same manifest, `validation` |
| encoder 477.4 ms / core 13.8–133.1 ms, peak memory per ladder cell | microbenchmark, 4090 laptop 16 GB, B=4 T=521 bf16 autocast, 12 iters after 4 warmup |
| 195 ms mean step, 3,244 tokens/s, 8.2M valid tokens, 7,500 boss draws @ 20k steps | `runs/bc/2026-08-06/dropout-0.2/metrics.csv` |
| core parameter counts | `CausalGPTConfig` + `SwiGLU` hidden rule (`causal.py:144`), `4d² + 3d·h + 2d` per layer |
| `d_model` pinned to encoder `hiddim` | `src/contra_policy/model.py:92` |
| no pixel augmentation anywhere in the loader | grep over `src/contra_policy/dataset.py` for augment/jitter/flip — only sampling RNG |
| train CE 0.051 @ 666 boss episodes; CE-optimal plays −12.9 pp | [0010](0010-dropout-regularization.md) §1 |
| mixed-v2 flat curve, 2,500 episodes / 464,019 frames, val SHA `1318…ecad` | [0009](0009-boss-data-scaling.md) §2, appendix |
| Spread+rapid GRPO finished at 9.5% = init, n=13, McNemar p = 1.0 | [evaluation 0014](../../contra_nes_evaluation/doc/0014-spread-specialty-boss-grpo.md) |
| Regular/Flamethrower 0 wins in 316 rollouts; 13 Spread+rapid val tasks | [0012](0012-spread-grpo.md) §1 |
| further Spread snapshots use the same 100-task validation and prefix scheme | `~/code/contra_nes_data/doc/0003-incremental-spread-scaling.md` §"Scaling contract" |
