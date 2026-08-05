# Train the base policy as an action-only GPT on the full-fight boss release

Status: Implemented
Supersedes: —

**Question.** What should the next base-policy checkpoint learn, report, and read now
that the grounding auxiliary head has no role in post-training and the larger boss
demonstration release exists?

**Answer.** Train one causal GPT with one learned output: the 21-way action logits.
Optimise masked action cross-entropy only; do not construct the heatmap or value heads
for this checkpoint. Report the same optimisation telemetry as Karpathy's
`build-nanogpt`: train loss, learning rate, gradient norm, step time, dense
tokens/second, and validation loss. Replace the published boss training shard with all
three shards from `boss-full-v1`, while keeping the other families in the ordinary HF
release and the boss validation set byte-identical.

---

## 1. Why — the evidence

The heatmap head was introduced as auxiliary grounding supervision. It does not produce
an action token, is not consumed by GRPO, and makes a nominally GPT-style base model
optimise a second dense objective. `point_err_px`, goal visibility, action accuracy,
balanced accuracy, non-modal accuracy, predicted-modal fraction and per-family accuracy
can help diagnose a representation, but none is the training objective. Keeping all of
them in the base trainer blurred the simplest question: is held-out action
cross-entropy going down at healthy compute throughput?

The new data release materially changes boss coverage:

| boss train data | episodes | decision frames |
|---|---:|---:|
| published baseline | 466 | 79,495 |
| added full-fight wins | 200 | — |
| `boss-full-v1` total | 666 | 120,055 |

The 666 episodes are frame-balanced over three shards (39,985, 40,035 and 40,035
frames). The validation shard remains the same 57 episodes and has SHA-256
`131835e34c55f75ded04410976730600a866744b40a8525bfc4d7f9ab952ecad`.

## 2. The design

### 2.1 Model and objective

The token sequence stays `[interaction, goal, frame_1, ..., frame_T]`. The goal image is
still an input token: removing an output head does not remove goal conditioning. The
new checkpoint sets `aux_size: 0` and disables the value head, so `ContraPolicy.forward`
returns only:

```text
pi_logits: (batch, time, 21)
```

The loss is masked cross-entropy over recorded actions. There is no heatmap BCE, class
reweighting, label smoothing, value loss, or geometric readout in the baseline config.

For checkpoint compatibility, absent head-control fields retain the legacy architecture:
old checkpoints rebuild their heatmap and value heads and continue to load strictly.
The new config explicitly disables both heads and starts a new base checkpoint; it does
not reinterpret old weights as the new architecture.

### 2.2 Metric contract

Training emits only the quantities in `build-nanogpt/train_gpt2.py`:

| phase | metric | definition |
|---|---|---|
| train | `loss` | masked action cross-entropy |
| train | `lr` | current optimiser learning rate |
| train | `grad_norm` | global norm returned by gradient clipping |
| train | `step_ms` | wall time for the measured training interval, CUDA-synchronised |
| train | `tokens_per_sec` | dense GPT positions computed per second, including padding and the two prefix tokens |
| validation | `loss` | mean masked action cross-entropy |

`step` and `phase` remain identifiers in the CSV, not model-quality metrics. Best
checkpoint selection changes from maximum validation action accuracy to minimum
validation loss. Closed-loop task completion stays in `contra_nes_evaluation`; it must
not be replaced by offline accuracy.

### 2.3 Data routing

`shard_dir` remains the default source for `kill`, `item`, and `traverse`. A
family-specific boss override points at:

```text
~/code/contra_nes_data/game_trace/releases/boss-full-v1/hf
```

Shard discovery accepts every matching `<family>-<split>-*.tar`, rather than assuming
only `00000`. Thus train resolves three boss shards and validation resolves the single
frozen boss shard. The index cache already fingerprints the complete path list.

## 3. What was rejected, and why

**Keep the heatmap head but give its loss weight zero.** This still allocates parameters,
computes a 32×32 logit map per frame, exposes unused outputs to downstream code, and
makes the checkpoint contract look more capable than the trained policy is. Legacy
checkpoints get compatibility; new checkpoints should describe what they actually do.

**Keep value prediction in the base model.** Behaviour cloning has no value target, and
GRPO does not use a learned critic. An untrained value head is dead checkpoint state.

**Keep diagnostic action and geometry metrics in the trainer.** Those metrics answered
earlier representation questions, but they enlarge the monitoring surface without
changing this run's action-CE decision. If a future ablation needs them, compute them in
an evaluation job against a pinned checkpoint and dataset.

**Append the new boss shards to the old boss shard.** `boss-full-v1` already contains the
466 baseline episodes. Appending would duplicate every baseline boss episode and
silently change its weight.

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate |
|---|---|---|
| the simpler objective does not learn | auxiliary grounding previously contributed gradient | validation action CE must improve below its step-0 value; closed-loop evaluation decides usefulness |
| loading old policy checkpoints breaks | old state dicts contain `aux_head` and `value_head` | strict round-trip test for a legacy checkpoint-shaped model |
| the boss override misses shards or duplicates baseline | the release has three train tars and already includes the baseline | resolved boss index is exactly 666 train / 57 validation episodes |
| throughput reporting is misleading | CUDA is asynchronous and batches are padded | synchronize before timing and count `B × (T_padded + 2)` dense positions |
| more boss data harms the other families | the natural step mix changes | the external full-family rollout report remains the acceptance gate after BC |

## 5. Sequencing

1. Commit this decision before implementation.
2. Add optional-head compatibility and pin old/new forward and checkpoint behavior.
3. Reduce the base trainer to action CE and the metric contract above; pin metric keys
   and minimum-validation-loss checkpoint selection.
4. Add multi-shard family overrides; require the configured release to resolve 666
   boss train and 57 boss validation episodes.
5. Run the unit suite and a short training smoke test. Only then mark this doc
   `Implemented` and commit the implementation.
6. Train the full checkpoint and evaluate closed-loop completion in
   `contra_nes_evaluation`. That result, not this implementation commit, decides whether
   the new base policy is better.

Implementation verification on 2026-08-04 resolved 666/57 boss train/validation
episodes from the configured paths, passed 131 tests, and completed one GPU training
step plus one validation batch against the real shards. Full training and closed-loop
evaluation remain future experiment work, not implementation gates.

---

## Appendix — provenance

| claim | source |
|---|---|
| Karpathy reports loss, LR, norm, step duration, tokens/sec and validation loss | `ref/build-nanogpt/train_gpt2.py:385-407,503-516` |
| the current policy always constructs action, value and 32×32 heatmap heads | `src/contra_policy/model.py` before this branch's implementation commit |
| the current BC objective combines action CE and heatmap BCE | `src/contra_policy/loss.py` and `src/contra_policy/train_bc.py` before this branch's implementation commit |
| the release has 666 train episodes in three shards and a frozen 57-episode validation shard | `/home/kaihe/code/contra_nes_data/doc/0001-boss-search-curriculum.md`, §7 |
| shard membership, hashes and validation SHA | `/home/kaihe/code/contra_nes_data/game_trace/releases/boss-full-v1/manifest.json` |
| `boss-full-v1` is built on data commit `db62a0c` | `git -C /home/kaihe/code/contra_nes_data rev-parse HEAD` on 2026-08-04 |
| configured routing resolves 7,104 train / 846 validation episodes, including 666/57 boss | real index smoke on 2026-08-04; cache fingerprints `856541e18109970a` and `74f1d9241b4180e1` |
| implementation passes 131 tests and a real GPU train/validation step | `pytest -q`; one-batch `BCTrainer` smoke on 2026-08-04 |
