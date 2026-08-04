# Compile and varlen training experiment — rejected

Status: Implemented
Supersedes: —

**Question.** Which training-efficiency changes should the next base-policy run test
if every optimization must also be useful in on-policy RL, where frame tokens cannot
be precomputed?

**Answer.** We implemented and independently measured dynamic core compilation and
boundary-safe varlen FlashAttention. Neither improved 500-step end-to-end BC throughput
by the required 10%, and their combination was slightly slower. Both experimental code
paths were therefore removed; base-policy and GRPO training remain eager and padded.
The corrected end-to-end throughput timer stays, because restoring the earlier timer
would restore a misleading GPU-only number rather than restore performance.

---

## 1. Why — the evidence

The new policy has one shared expensive path in BC and GRPO:

```text
images -> frozen encoder -> causal core -> action logits
```

BC and GRPO updates both batch complete variable-length episodes. Rollout inference
also runs the same causal core over variable-length active histories. A core compiler
and a boundary-aware variable-length attention layout can therefore transfer across
stages; offline token caching cannot.

Several Karpathy optimizations are already present: TF32, BF16 autocast,
`scaled_dot_product_attention`, gradient clipping, pinned/prefetched loading and
`zero_grad(set_to_none=True)`. `torch.compile` and fused AdamW are not enabled.

The current `boss-full-v1` BC epoch has 7,104 episodes. With batch size 4 and the
current length-grouped sampler, one deterministic epoch measures:

| work | current padding cost |
|---|---:|
| dense token/linear positions | 4.39% overhead, 4.21% of compute positions wasted |
| causal attention pairs | 12.74% overhead, 11.30% of pairs wasted |

This revisits, but does not yet overturn, 0002 §3. The original release measured about
1% padding under a different data mix and batching setup. The larger boss release and
the actual batch-4 sampler raise the measured waste, but not enough to assume that a
new attention backend will pay for its complexity.

The pre-experiment `tokens_per_sec` timer began after the DataLoader yielded a batch.
Karpathy starts timing before `next_batch()`. Commit `4a27bcf` corrected the boundary so
the retained headline rate includes loading, transfer, encoder, core, backward and
optimizer work.

## 2. The experiment as run

### 2.1 Factor A — dynamic core compilation

The experiment added an explicit configuration switch:

```yaml
performance:
  compile_core: false
  compile_dynamic: true
```

It compiled only the causal core after device placement. The raw `ContraPolicy` remained
the checkpoint owner, so compilation added no `_orig_mod` checkpoint keys. Dynamic
compilation handled the different time dimensions produced by batch-local padding.

The eager and compiled paths consumed identical batches in identical order. First-use
compilation was excluded from steady-state throughput and reported separately. The
implementation covered both intended callers:

- BC forward/backward/update over whole episodes.
- GRPO rollout inference and padded optimizer minibatches.

### 2.2 Factor B — boundary-safe varlen attention

The tested packed representation concatenated complete episode segments:

```text
[interaction_A, goal_A, frames_A]
[interaction_B, goal_B, frames_B]
...
```

It carried `cu_seqlens = [0, L_A, L_A + L_B, ...]`, maximum segment length, reset
position IDs, frame/action offsets and a loss mask. A variable-length causal attention
kernel uses those boundaries directly, so episode B cannot attend to episode A and no
dense block-diagonal `T x T` mask is materialized. Apply the same representation to raw
frames before encoding; packing must remove padded encoder work too, not only padded
core work.

This is the SFT-style design used by current NeMo/Megatron packed training: sequence
boundaries are passed to a THD/varlen attention kernel rather than expressed as one
dense custom mask. Sources:

- <https://docs.nvidia.com/nemo-framework/user-guide/25.02/sft_peft/packed_sequence.html>
- <https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/packed-sequences.html>
- <https://github.com/dao-ailab/flash-attention>

The implementation covered BC, GRPO optimizer minibatches and ragged rollout histories;
it was not a BC-only packer.

### 2.3 Benchmark matrix and metrics

The four runs used the same fixed batch schedule and checkpoint:

| run | core | attention layout | purpose |
|---|---|---|---|
| E | eager | padded | baseline |
| C | compiled dynamic | padded | isolated compile effect |
| V | eager | varlen packed | isolated packing effect |
| CV | compiled dynamic | varlen packed | interaction and final candidate |

BC was the first performance gate. Only a candidate that passed it would advance to
GRPO-update, rollout-inference and closed-loop measurements. The BC comparison reported:

- end-to-end tokens or decisions per second;
- steady-state step latency median and p90;
- GPU utilization and peak allocated memory;
- compile startup time and recompilation count;
- padding fraction or packed occupancy;
- loss and gradient norm on the fixed batches.

No candidate passed, so the later RL and closed-loop benchmarks were deliberately not
run; doing them could not change the base-policy decision.

Compilation and varlen packing were gated independently. The 10% threshold kept the
measured single-digit gains and their maintenance burden out of the main path.

## 3. What was rejected, and why

**Precompute frozen encoder tokens.** This would likely accelerate offline BC, but an
on-policy rollout generates new frames and must encode them online. GRPO also re-encodes
collected frames during updates under the current shared policy contract. Maintaining a
second token-only dataset and model entry point would make BC throughput less
representative of the post-training system. This experiment deliberately ignores it.

**Naively concatenate tasks with an EOT-style separator.** Text pretraining can treat
documents as a stream; SFT tasks and Contra episodes are independent. A learned
separator is not an access-control boundary and would allow goal, interaction and frame
state to leak between episodes.

**Use the existing dense block-diagonal mask as the performance implementation.** It is
a useful correctness reference, but attention is still launched over the square of the
whole pack and may lose the ordinary causal FlashAttention path. The candidate must use
a genuine varlen kernel whose work is the sum of the segment squares.

**Compile the whole image policy first.** The frozen encoder uses a Python branch and a
variable-count chunk loop based on `batch x time`. That is a noisier compiler target
than the causal core and risks hiding the value of compilation behind graph breaks.

**Change the active base-policy run.** Compilation changes process construction and
cannot be enabled safely inside a running job. The eager/padded run active when the
experiment began remained the baseline and finished unchanged.

## 4. Risks, and the metric that gates each

| risk | why it is plausible | gate |
|---|---|---|
| dynamic compilation repeatedly recompiles | episode lengths change every batch | after warm-up, no new graph for 500 consecutive steps |
| compiled checkpoints become incompatible | wrapping a module can prefix state keys with `_orig_mod` | eager and compiled modes save identical key sets and strictly cross-load |
| packing leaks across tasks | one physical tensor contains several episodes | changing every token in one segment changes no logits in any other segment |
| packed position semantics differ | RoPE positions must restart at each episode | packed logits match individually evaluated episode logits within BF16 tolerance |
| varlen loses rather than gains throughput | present padding waste is only 4.21% | keep only at >=10% steady-state end-to-end improvement in a target phase |
| speed changes optimization | different kernels change reduction order | fixed-batch loss/gradient checks pass and short-run validation loss stays within 0.01 |
| BC-only optimization adds a dead RL path | batching contracts currently differ | feature is incomplete until BC, GRPO update and rollout benchmarks exist |

## 5. Sequence completed

1. The original eager/padded run finished and remains the behavioral baseline.
2. The timer was corrected and the 500-step eager baseline was measured.
3. Checkpoint-transparent dynamic compilation was implemented, tested and rejected at
   +1.9% end-to-end throughput.
4. Padding was measured at only 4.21% of dense compute positions.
5. Boundary-safe varlen attention was implemented and tested alone and with compile;
   both configurations missed the gate.
6. Closed-loop evaluation was unnecessary because neither candidate passed the speed
   gate. The experimental paths were removed in `e1965b4`.

## 6. Result and rollback

The fixed comparison used the same checkpoint, seed and batch schedule on an RTX 4090
Laptop GPU, with 20 warm-up steps and 500 measured updates. Timing included batch
acquisition, transfer, encoder, core, backward and optimizer work.

| run | useful tokens/s | versus eager/padded |
|---|---:|---:|
| eager + padded | 2,229 | baseline |
| compiled + padded | 2,271 | +1.9% |
| eager + varlen | 2,269 | +1.8% |
| compiled + varlen | 2,206 | -1.0% |

All candidates missed the 10% gate. The implementation was subsequently rolled back,
including its private PyTorch 2.9 varlen-operator dependency, configuration switches,
packing contracts and benchmark harness. This is a rejected experiment, not unfinished
work. Raw benchmark JSON remains in the ignored local `tmp/` directory where available.

The two full training runs on 2026-08-04 confirm that the lower post-0007 logged rate is
a timer correction rather than a real slowdown. The old run completed 20,000 steps in
about 68 minutes; the new eager/padded run advanced at the same roughly 4.88 steps/s.
The old ~4,000 tokens/s started timing after the DataLoader yielded, whereas the retained
timer includes acquisition and exposes bursty loader stalls.

No other repository is blocked on this experiment, so no handoff issue is required.

---

## Appendix — provenance

| claim | source |
|---|---|
| current BC path is eager, BF16/TF32, ordinary AdamW and SDPA | `src/contra_policy/train_bc.py`, `src/contra_policy/causal.py` at `e1965b4` |
| BC and GRPO update both pad whole episodes | `pad_episodes` and `rl.buffer.GroupBatch` at `446676b` |
| rollout core input is a padded batch of active histories | `rl.rollout.BatchedPolicyRunner._core_over_histories` at `446676b` |
| 7,104 train episodes; 725,635 usable frame positions | `cache/shard_index_v2_856541e18109970a.json`, boss-full-v1 configuration |
| 4.21% dense padding waste | deterministic `LengthGroupedSampler(batch_size=4, pool_batches=32, seed=0)` epoch: 725,635 valid / 757,500 dense frames |
| 11.30% padded causal-attention pairs are avoidable | same epoch, `sum(L_i^2)` against `sum(batch_size * max(L)^2)`, including two prefix tokens |
| pre-0007 throughput excluded DataLoader latency | timer placement at `446676b`; corrected by `4a27bcf` |
| boundary-aware SFT packing uses cumulative lengths and varlen attention | NVIDIA NeMo/Megatron documentation and official FlashAttention repository linked in §2.2 |
