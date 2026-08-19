# Use a plain causal transformer for whole episodes

Status: Implemented
Supersedes: —

Depends on [0001](0001-exp-image-encoder.md), which reduces each image to one token.

**Question.** Once an episode fits in context, should the policy replace its
Transformer-XL-style recurrent core with a plain causal decoder over the whole episode?

**Answer.** Yes. Use `[interaction, goal, frame_1, …, frame_N]`, one forward per episode,
and predict actions at frame positions. A small Llama-shaped decoder (RMSNorm, RoPE,
SwiGLU, GQA) removes carried memory, chunking, truncated BPTT, and position bookkeeping.
The first BC run reached 67.5% completion: the architecture works, but did not beat ROCKET
BC's 72.8%, and whole-episode context did not improve boss completion.

## 1. Evidence

One-token frames make full-episode attention cheaper than the old windowed scheme. A
510-frame episode uses 512 tokens and 131k causal query-key pairs, versus 885k pairs for
32-step chunks with 32-step memory. Length-bucketed batches keep padding near 1%.

| context | BC demos covered | RL episodes covered |
|---:|---:|---:|
| 254 frames | 96.0% | ~78% |
| 510 frames | 99.99% | 96.0% |
| 1022 frames | 100% | 100% |

The implemented model was evaluated closed-loop on 846 validation tasks:

| policy | completion | Wilson 95% | boss completion | source |
|---|---:|---:|---:|---|
| ROCKET BC | 72.8% | — | 8.8% | evaluation baseline |
| plain causal, final | 67.5% | [64.3, 70.6] | 3.5% | `contra_nes_evaluation/doc/0005-gpt-bc.md` |
| plain causal, best `bc_acc` checkpoint | 53.3% | — | — | step 4,000; `bc_acc=0.739` |

The final model trailed ROCKET by 5.3 percentage points (paired McNemar p=0.0029).
`val/bc_acc` was therefore the wrong selection gate: its best checkpoint lost 14.2 points
to the final checkpoint. `point_err_px` tracked closed-loop completion better.

## 2. Design

The sequence is `[interaction, goal, frame × N]`, capped at 1,024 tokens. `interaction`
is an embedding lookup; goal and frame tokens come from the shared encoder in 0001. All
heads read the transformer output at frame position `t`:

| head | output | purpose |
|---|---|---|
| `pi` | 21 logits | action from frame `t` |
| `value` | scalar | PPO critic |
| `goal_heatmap` | 32×32 | grounding after goal/frame attention |

Actions are predicted but never fed back. The previous-action ablation in 0001 reduced
boss completion from 8.8% to 1.8% and doubled grounding error, so action tokens would add
a harmful signal while doubling sequence length.

Each episode is one independently padded sequence. Episodes are sorted by length, batched
with neighbours, and padded only to the batch maximum. Context capacity is fixed at 1,024
for both BC and RL, and cost follows actual batch length. The largest configured RL budget
is 1,038 frames, slightly above the 1,022-frame payload; observed episodes fit, but the cap
can truncate that theoretical tail.

The implementation removes `mem_len`, `seq_len` chunking, carried KV state, `first`,
`state_mask`, `clipped_causal`, `bandify`, and `index_bias`. RoPE makes context length a
configuration value instead of the shape of a learned position parameter.

## 3. Rejected alternatives

**Keep `ResidualRecurrentBlocks`.** Its memory, band mask, learned relative-position
basis, and reset bookkeeping exist to simulate long context from short windows. Retaining
them after episodes fit preserves complexity and makes horizon changes reshape a learned
parameter.

**Interleave action tokens.** This restores the previous-action signal rejected by the
measured ablation and doubles sequence length.

**Train at context 128 and fragment long episodes.** It saves 24% of attention pairs but
cuts 57.3% of boss demonstrations, the family for which long context was intended.

| scheme | query-key pairs over 6,904 demos | relative cost |
|---|---:|---:|
| full episodes | 48.71M | 1.00× |
| 128-token fragments | 36.94M | 0.76× |
| 256-token fragments | 45.78M | 0.94× |

**Sequence packing.** It removes the final ~1% padding but reintroduces episode-boundary
masks and bookkeeping. Reconsider only if activation memory becomes limiting.

**LLM-scale backbone.** The goal is the small Llama architecture, not its scale. Training
is already GPU-bound, and the 12.77M-parameter recurrent core does not justify a larger
replacement.

## 4. Caveats

- Whole-episode context did not validate the boss hypothesis: boss completion fell from
  8.8% to 3.5%.
- Full causal attention no longer enforces equal local context at every position.
- `bc_acc` must not select checkpoints; closed-loop completion is the policy gate, with
  `point_err_px` retained as the useful offline diagnostic.
- A future level with multi-thousand-frame budgets may require fragmentation or packing.

## Appendix — provenance

| claim | source |
|---|---|
| BC lengths | 6,904 train episodes; mean 100, p50 102, p90 150, max 519 |
| RL coverage | `max(24, ceil(2 × len(actions)))`; max 1,038 |
| attention costs | causal `n(n+1)/2`; old chunked-mask calculation from measured layout |
| padding waste | 400 sampled batches per batching scheme |
| fragmentation cost | causal pairs summed over all 6,904 demonstrations |
| current core size | `sum(p.numel())` over `CrossViewContraRocket.recurrent` |
| closed-loop result | `contra_nes_evaluation/doc/0005-gpt-bc.md` |
