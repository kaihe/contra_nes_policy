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
[interaction, goal, frame_1, frame_2, …, frame_N]        N + 2 tokens, N up to 1022
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

BC demos are roughly half the budget of the task they came from, so 512 would cover BC
but truncate 4% of RL episodes. **Train at 1024 throughout** — §3 shows the capacity is
nearly free once batching is length-bucketed, and using one length for BC and RL means
nothing is ever asked to run past what it saw.

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

## 3. Short episodes, and why length never needs extrapolating

Two questions that turn out to have one answer.

**Most episodes are far shorter than the context.** Mean demo is 100 frames, p50 102,
p90 150 — against 512. Padded naively that is mostly mask:

| scheme | batch 8 | batch 32 |
|---|---|---|
| pad to fixed 512 | 80% waste | 80% |
| pad to fixed 1024 | 90% waste | 90% |
| pad to batch max, random batches | 49% | 65% |
| **pad to batch max, length-bucketed** | **0%** | **1%** |

So: **bucket by length and pad to the batch maximum.** Sort episodes, batch neighbours,
pad to the longest in the batch. Waste falls to ~1% at batch 32 with no change to the
model. Sequence *packing* — several episodes per row behind a block-diagonal mask —
reaches 0% too, but reintroduces exactly the document-boundary bookkeeping §4 deletes,
for one percentage point.

Attention for a median 102-frame episode, causal:

| | query-key pairs |
|---|---|
| padded to 512 | 132,355 |
| padded to 1024 | 526,851 |
| **actual length (104 tokens)** | **5,460** |

**And that answers extrapolation: there is none.** Cost tracks *actual* length, not
context capacity, so capacity is nearly free — set it to **1024 from the start**, train
BC and RL at the same length, and never ask the model to run past what it saw.

Our lengths are bounded by data, not by user input: `budget = max(24, ceil(2 ×
expert_steps))`, **max 1038** over all 6,904 tasks. Unlike text generation there is no
open-ended case, so RoPE scaling, NTK/YaRN interpolation and ALiBi are all answers to a
question we do not have. If a future level ships longer tasks, retrain at the new bound
— which is a config change, because RoPE has **no learned positional parameter**.

That last point is a quiet advantage over the current core. `bandify`'s `b_nd` is a
learned `(10, mem_len × num_step_tokens)` tensor: its *shape* depends on the horizon, so
lengthening the context there means reshaping a trained parameter. RoPE makes context
length a config value.

## 4. What this deletes

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

The `first` flag deserves a note: it is deleted only because an episode is now one
sequence. If sequence packing is ever adopted (§3), the boundary problem returns and so
must a mask that expresses it.

## 5. What was rejected

**Porting `ResidualRecurrentBlocks` to a longer window.** It would work, but every
mechanism above is scaffolding for a short context. Keeping them while the episode fits
in one sequence is paying complexity for nothing — and the relative-position basis is a
learned `(10, mem_len × tokens)` parameter whose shape changes with the horizon, so
extending it is surgery rather than a config change.

**Interleaving action tokens (Decision-Transformer style).** See §2 — the `prev_action`
ablation rules it out on measured grounds, not aesthetic ones.

**Training at a short context (128) and cutting long traces into fragments**, each
re-prefixed with `[interaction, goal]` so it stays self-contained. This is the LLM
staged short→long recipe, and the construction is right — the prefix is exactly the
invariant that keeps a fragment interpretable. It is rejected on measured cost.

Aggregate attention over all 6,904 BC episodes (692,179 frames):

| scheme | query-key pairs | vs full-episode |
|---|---|---|
| full episode, no cut | 48,710,573 | 1.00× |
| fragments of 128 | 36,938,636 | **1.32×** |
| fragments of 256 | 45,781,266 | 1.06× |
| fragments of 512 | 48,705,986 | 1.00× |

**Cutting to 128 saves 24%** — the quadratic term never gets going, because the mean
episode is 100 frames and the tail is thin. And the 24% is paid by the wrong family:

| family | fits one 126-frame fragment | mean fragments | p90 length |
|---|---|---|---|
| kill | 93.1% | 1.07 | 120 |
| item | 99.3% | 1.01 | 68 |
| traverse | 88.0% | 1.17 | 194 |
| **boss** | **42.7%** | **1.80** | **305** |

57% of boss episodes would be cut — the family this work exists to fix, and the one
whose attack cycles were the argument for long context. Trading whole-episode context
on boss for 24% of a cost that is already negligible inverts the priority.

Why the LLM recipe does not transfer: it exists because long documents are *scarce* in
pretraining corpora (so paying 32k for the whole run wastes capacity) and because 32k²
is expensive at trillions of tokens. Here the long episodes are exactly the ones we care
about, and the whole dataset costs 48.7M pairs — a rounding error beside the conv trunk.

Worth noting it is not a regression against today's model: 126 frames is 4× the current
32-decision window. It is simply strictly worse than full-episode for 32% more cost.

**Revisit if** a future level ships budgets in the thousands, where the quadratic term
would start to bite, or if activation memory becomes binding — it is not today
(`flex_attention` makes the attention term O(n), and stage A peaked at 2.5 GB of 16).

**An LLM-scale backbone.** Already rejected in 0001 §3 and unchanged here: we are
GPU-bound at 76%, and this is about *removing* work. The proposal is minimind's
*architecture* — a small Llama — not its scale. The current core is 12.77M and there is
no reason for the replacement to be larger.

## 6. Risks, and the metric that gates each

| risk | why plausible | gate |
|---|---|---|
| full-episode attention loses the locality the window enforced | `clipped_causal` guaranteed every step saw exactly 32 decisions; now step 500 sees 500 | BC `val/bc_acc` and completion vs the 72.8% baseline |
| RoPE at 512 without the learned basis grounds worse | the basis was trained for this data | `point_err_px` on single-centroid frames, vs 0001's 0.84 px kill / 2.19 px traverse |
| a 4% RL truncation at 512 biases against long boss episodes | boss p90 is 610 | run RL at 1024, not 512 |
| length-bucketed batching correlates episodes within a batch | neighbours in a sorted order share length, and length correlates with family — traverse is long, item short | per-family mix per batch; watch `val/bc_acc` split by family |

**The comparison that decides it:** BC completion against **72.8%**, at `seq_len: 32`
equivalent settings first, so a regression is attributable to the architecture rather
than to the longer context.

## 7. Sequencing

1. **Write the core** — a small Llama-style causal transformer, config-shaped
   (`n_layer`, `n_head`, `d_model`, `context`), with an attention mask. No memory, no
   chunking.
2. **Retokenise `model.py`** to `[interaction, goal, frame × N]`, heads on frame
   positions, delete the machinery in §3.
3. **Length-bucketed batching** in the loader — the one thing that makes a 1024 context
   affordable for a 100-frame median episode.
4. **BC at context 1024.** Gate: completion vs 72.8%.
5. **RL at the same 1024**, once BC holds — no length change, so nothing to extrapolate.

Step 3 is the real gate. Steps 1-2 are mechanical; the question is only whether a plain
causal model over whole episodes learns this task as well as a windowed recurrent one.

---

## Appendix — provenance

| claim | source |
|---|---|
| BC episode lengths | `ep["length"]` over 6,904 train episodes; max 519, 1 exceeds 510 |
| RL budget coverage | `max(24, ceil(2 × len(actions)))` over 6,904 task `.npz`; max 1038 |
| attention pair counts | causal `n(n+1)/2` at 512 tokens vs `ceil(T/32) × 192 × 384 × 0.75` for the chunked scheme |
| episode length distribution | 6,904 train demos: mean 100, p50 102, p90 150, max 519 |
| padding waste | 400 sampled batches per scheme; bucketed = sort then batch neighbours |
| fragmentation cost | causal `n(n+1)/2` summed per episode; fragments of `ctx` = `floor(L/(ctx-2))` full rows plus the remainder, each with a 2-token prefix |
| current core 12.77M | `sum(p.numel())` over `CrossViewContraRocket.recurrent`, 4 layers / 512 hid / 8 heads |
| `prev_action` ablation | 0001 §3 — `contra_nes_evaluation/runs/0729-e18` vs `-noprev` |
| `index_bias` bug | `src/contra_policy/model.py:108-114` |
