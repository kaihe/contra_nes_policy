# A plain causal transformer over whole episodes

Status: Proposed
Supersedes: —
Depends on: [0001](0001-image-encoder.md) — one token per image is what makes this fit.

**Question.** The policy's temporal core is VPT's `ResidualRecurrentBlocks`: a
Transformer-XL with a learned relative-position basis, a clipped-causal mask, carried
KV memory, and per-chunk `first` bookkeeping. All of that exists to simulate a long
context from a 32-step window. With 1 token per frame, does an episode just *fit*?

**Answer.** Yes, and the machinery becomes unnecessary. Sequence:

```
[interaction, goal, frame_1, frame_2, …, frame_510]      512 tokens
```

A plain causal decoder — Llama-shaped, as in the minimind starter configs: RMSNorm,
RoPE, SwiGLU, GQA — with the action predicted at each frame position. **One forward per
episode.** No carried memory, no chunking, no truncated BPTT, no `first` flag, no
`index_bias`.

It is also **6.7× cheaper**: 131k query-key pairs for a 510-frame episode against
885k for the current 32-chunk-plus-32-memory scheme.

---

## 1. Why this is now possible

0001 made each frame one token. That is the whole enabling change — at 6 tokens/frame a
510-frame episode is 3,062 tokens and full causal attention is 4.7M pairs, which is why
the windowed scheme existed. At 1 token/frame it is 512 tokens and 131k pairs.

**Context coverage**, measured:

| context | BC demos covered | RL episodes covered |
|---|---|---|
| 254 frames | 96.0% | ~78% |
| **510 frames** | **99.99%** (1 of 6,904 exceeds it; max 519) | **96.0%** |
| 1022 frames | 100% | **100%** (max budget 1038) |

So **512 is the right size for stage B (BC)** and **1024 for RL**, because an RL episode
runs to the full budget — roughly twice the expert demonstration it was derived from.
The context length is a config value, not a design commitment; the layout is unchanged.

## 2. The design

**Tokens.** `interaction` is an embedding lookup (5 interaction kinds + "no goal").
`goal` and every `frame_t` are `contra_encoder.encode(image)` — the same function, which
is what 0001 bought. Prefix first so every frame attends to the goal, at a relative
distance that RoPE handles natively.

**Heads**, all reading the transformer output at frame position *t*:

| head | output | purpose |
|---|---|---|
| `pi` | 21 logits | the action taken *from* frame *t* |
| `value` | scalar | PPO critic |
| `goal_heatmap` | 32×32 | grounding, back where 0001 moved it from — the model has attended to the goal token, so this is now a comparison rather than a broadcast |

**Actions are predicted, never fed back.** The sequence contains no action tokens. This
is not an oversight and not a Decision-Transformer: the ablation in 0001 shows feeding
the previous action collapses boss 8.8% → 1.8% and doubles grounding error. Interleaving
`[…, frame_t, action_t, frame_t+1, …]` would reintroduce exactly that signal, and double
the sequence length to boot.

**Episode = sequence.** Padding and an attention mask handle variable length. Nothing
crosses an episode boundary, so the `first` flag and the memory-reset logic disappear
rather than being ported.

## 3. What this deletes

| gone | why it existed |
|---|---|
| carried KV memory (`mem_len`) | to see past a 32-step window — the window is now the episode |
| `seq_len` chunking + truncated BPTT | to fit that window in memory; gradient no longer truncates |
| `first` flag, `state_mask` | to stop memory leaking across episode boundaries |
| `clipped_causal` band mask | to equalise context across positions in a chunk |
| `bandify` relative-position basis | replaced by RoPE |
| `index_bias` arithmetic | head placement within a 6-token block; there is one token now |

`index_bias` has already caused one silent bug — the aux head read a token that causally
*preceded* the interaction token, so it "never knew whether the task was kill, pick,
avoid, traverse or boss" (`model.py:108-114`). Deleting the arithmetic removes the class
of error rather than re-deriving it for a new layout.

## 4. What was rejected

**Porting `ResidualRecurrentBlocks` to a longer window.** It would work, but every
mechanism above is scaffolding for a short context. Keeping them while the episode fits
in one sequence is paying complexity for nothing — and the relative-position basis is a
learned `(10, mem_len × tokens)` parameter whose shape changes with the horizon, so
extending it is surgery rather than a config change.

**Interleaving action tokens (Decision-Transformer style).** See §2 — the `prev_action`
ablation rules it out on measured grounds, not aesthetic ones.

**An LLM-scale backbone.** Already rejected in 0001 §3 and unchanged here: we are
GPU-bound at 76%, and this is about *removing* work. The proposal is minimind's
*architecture* — a small Llama — not its scale. The current core is 12.77M and there is
no reason for the replacement to be larger.

## 5. Risks, and the metric that gates each

| risk | why plausible | gate |
|---|---|---|
| full-episode attention loses the locality the window enforced | `clipped_causal` guaranteed every step saw exactly 32 decisions; now step 500 sees 500 | BC `val/bc_acc` and completion vs the 72.8% baseline |
| RoPE at 512 without the learned basis grounds worse | the basis was trained for this data | `point_err_px` on single-centroid frames, vs 0001's 0.84 px kill / 2.19 px traverse |
| a 4% RL truncation at 512 biases against long boss episodes | boss p90 is 610 | run RL at 1024, not 512 |
| padding wastes compute — mean demo is 100 frames against 512 | 80% of a padded batch is mask | length-bucketed batching if it bites |

**The comparison that decides it:** BC completion against **72.8%**, at `seq_len: 32`
equivalent settings first, so a regression is attributable to the architecture rather
than to the longer context.

## 6. Sequencing

1. **Write the core** — a small Llama-style causal transformer, config-shaped
   (`n_layer`, `n_head`, `d_model`, `context`), with an attention mask. No memory, no
   chunking.
2. **Retokenise `model.py`** to `[interaction, goal, frame × N]`, heads on frame
   positions, delete the machinery in §3.
3. **BC at context 512.** Gate: completion vs 72.8%.
4. **RL at context 1024**, once BC holds. `rollout.max_episode_steps` and the budget
   already bound episodes below it.

Step 3 is the real gate. Steps 1-2 are mechanical; the question is only whether a plain
causal model over whole episodes learns this task as well as a windowed recurrent one.

---

## Appendix — provenance

| claim | source |
|---|---|
| BC episode lengths | `ep["length"]` over 6,904 train episodes; max 519, 1 exceeds 510 |
| RL budget coverage | `max(24, ceil(2 × len(actions)))` over 6,904 task `.npz`; max 1038 |
| attention pair counts | causal `n(n+1)/2` at 512 tokens vs `ceil(T/32) × 192 × 384 × 0.75` for the chunked scheme |
| current core 12.77M | `sum(p.numel())` over `CrossViewContraRocket.recurrent`, 4 layers / 512 hid / 8 heads |
| `prev_action` ablation | 0001 §3 — `contra_nes_evaluation/runs/0729-e18` vs `-noprev` |
| `index_bias` bug | `src/contra_policy/model.py:108-114` |
