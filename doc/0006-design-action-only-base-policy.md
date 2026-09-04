# Use action cross-entropy alone for the base policy

Status: Implemented
Supersedes: —

**Question.** What should the base-policy checkpoint learn, report, and read now that
post-training does not use the grounding or value heads and `boss-full-v1` is available?

**Answer.** Train one causal GPT whose only learned output is 21-way action logits. Use
masked action cross-entropy, select checkpoints by minimum validation loss, and report
standard GPT optimisation telemetry. Route boss data to all three `boss-full-v1` training
shards while keeping the 57-episode boss validation shard unchanged. Preserve legacy
head construction only when loading old checkpoints.

## 1. Evidence

The heatmap head supplied auxiliary grounding supervision but is not consumed by GRPO;
BC has no value target, and GRPO has no critic. Keeping either head in a new base checkpoint
adds unused parameters and obscures whether held-out action cross-entropy is improving.

The new boss release increases full-fight coverage without changing validation:

| boss data | episodes | decision frames |
|---|---:|---:|
| published baseline | 466 | 79,495 |
| added full-fight wins | 200 | — |
| `boss-full-v1` | 666 | 120,055 |

Its three training shards contain 39,985, 40,035, and 40,035 frames. The frozen
57-episode validation shard has SHA-256
`131835e34c55f75ded04410976730600a866744b40a8525bfc4d7f9ab952ecad`.

## 2. Design

The input stays `[interaction, goal, frame_1, ..., frame_T]`; removing the grounding
output does not remove goal conditioning. New base checkpoints set `aux_size: 0` and
`value_head: false`, so `ContraPolicy.forward` returns only:

```text
pi_logits: (batch, time, 21)
```

Training minimises masked action cross-entropy with no heatmap BCE, value loss, class
reweighting, or label smoothing. Missing head-control fields retain the legacy architecture
so old checkpoints still load strictly; the new configuration explicitly disables both
heads and creates a new checkpoint lineage.

| phase | metric | definition |
|---|---|---|
| train | `loss` | masked action cross-entropy |
| train | `lr` | current optimiser learning rate |
| train | `grad_norm` | global norm returned by gradient clipping |
| train | `step_ms` | CUDA-synchronised interval time |
| train | `tokens_per_sec` | `B × (T_padded + 2)` dense positions per second |
| validation | `loss` | mean masked action cross-entropy |

Best-checkpoint selection uses minimum validation loss. Closed-loop completion remains an
external evaluation metric and is not replaced by offline loss or accuracy.

`shard_dir` remains the source for `kill`, `item`, and `traverse`. The boss family points
to `boss-full-v1/hf`; discovery accepts every `<family>-<split>-*.tar`, and the cache
fingerprint covers the complete path list. The release replaces rather than appends to the
old boss shard because it already contains those 466 episodes.

## 3. Rejected alternatives

| proposal | reason rejected |
|---|---|
| retain heatmap head at zero loss weight | still allocates and computes an unused 32×32 output and misstates the checkpoint contract |
| retain value head | BC has no value target and GRPO does not use a learned critic |
| keep diagnostic geometry and action metrics in training | they do not select this checkpoint; run them separately against pinned artifacts |
| append `boss-full-v1` to the old shard | duplicates all 466 baseline boss episodes |

## 4. Verification and remaining gate

Implementation verification resolved exactly 666 boss training and 57 boss validation
episodes, 7,104/846 episodes overall, passed 131 tests, and completed one real-shard GPU
training step plus one validation batch. Timing synchronises CUDA, and compatibility tests
cover legacy checkpoint-shaped models.

The implementation is complete; policy usefulness is not established by the smoke test.
A full training run and closed-loop evaluation in `contra_nes_evaluation` must decide
whether the new checkpoint improves the policy or harms non-boss families.

## Appendix — provenance

| claim | source |
|---|---|
| telemetry contract | `ref/build-nanogpt/train_gpt2.py:385-407,503-516` |
| model and objective contract | `src/contra_policy/model.py`, `loss.py`, and `train_bc.py` |
| release membership and hashes | `contra_nes_data/game_trace/releases/boss-full-v1/manifest.json` |
| release rationale | `contra_nes_data/doc/0001-boss-search-curriculum.md` |
| resolved episode counts | real index smoke; cache fingerprints `856541e18109970a`, `74f1d9241b4180e1` |
| implementation verification | `pytest -q`; one-batch `BCTrainer` GPU smoke |
