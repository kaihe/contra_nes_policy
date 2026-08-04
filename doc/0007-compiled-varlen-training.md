# Compile the shared policy core, then gate boundary-safe varlen packing

Status: Implemented (experimental paths retained, defaults rejected)
Supersedes: —

**Question.** Which training-efficiency changes should the next base-policy run test
if every optimization must also be useful in on-policy RL, where frame tokens cannot
be precomputed?

**Answer.** We ran two controlled optimizations in order. First compile the causal core
with dynamic sequence length while leaving batching unchanged. Then test true packed
episodes through a variable-length FlashAttention kernel using cumulative sequence
boundaries. Keep each change only if it improves steady-state end-to-end throughput by
at least 10% without changing task isolation, checkpoint format, loss, or closed-loop
behavior. Neither cleared that gate in the 500-step BC measurement, so both remain
explicitly opt-in and the next base-policy run stays eager/padded. Do not build a
cached-encoder-token path: it accelerates BC only and therefore optimizes a data path
RL cannot use.

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

The current `tokens_per_sec` timer begins after the DataLoader yields a batch. Karpathy
starts timing before `next_batch()`. Before any A/B, move the timer boundary so the
headline rate includes loading, transfer, encoder, core, backward and optimizer work.

## 2. The experiment

### 2.1 Factor A — dynamic core compilation

Add an explicit configuration switch:

```yaml
performance:
  compile_core: false
  compile_dynamic: true
```

Compile only the causal core after device placement. The raw `ContraPolicy` remains the
checkpoint owner, so enabling compilation must not add `_orig_mod` keys or otherwise
change its state dict. Dynamic compilation is required because batch-local padding
produces a different time dimension from one batch to the next.

The eager and compiled paths consume identical batches in identical order. Exclude
first-use compilation from steady-state throughput, but report its wall-clock cost and
the number of recompilations separately. Test both:

- BC forward/backward/update over whole episodes.
- GRPO rollout inference and padded optimizer minibatches.

### 2.2 Factor B — boundary-safe varlen attention

The packed representation concatenates complete episode segments:

```text
[interaction_A, goal_A, frames_A]
[interaction_B, goal_B, frames_B]
...
```

It carries `cu_seqlens = [0, L_A, L_A + L_B, ...]`, maximum segment length, reset
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

The implementation must cover BC, GRPO optimizer minibatches and ragged rollout
histories before it is considered shared infrastructure. A BC-only packer is not the
goal of this experiment.

### 2.3 Benchmark matrix and metrics

Run the same fixed batch schedule from the same checkpoint:

| run | core | attention layout | purpose |
|---|---|---|---|
| E | eager | padded | baseline |
| C | compiled dynamic | padded | isolated compile effect |
| V | eager | varlen packed | isolated packing effect |
| CV | compiled dynamic | varlen packed | interaction and final candidate |

Report separately for BC update, GRPO update and rollout inference:

- end-to-end tokens or decisions per second;
- steady-state step latency median and p90;
- GPU utilization and peak allocated memory;
- compile startup time and recompilation count;
- padding fraction or packed occupancy;
- loss and gradient norm on the fixed batches.

Compilation and varlen packing are accepted independently. A 10% speed threshold keeps
small benchmark noise and maintenance-heavy single-digit gains out of the main path.

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
cannot be enabled safely inside a running job. The current eager/padded run remains the
baseline and finishes unchanged.

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

## 5. Sequencing

1. Let the current eager/padded run finish. Its artifacts remain the behavioral
   baseline; do not restart it for this proposal.
2. Correct the timer to include batch acquisition and record 500 steady-state eager
   steps. Gate: reproducible median throughput within 3% across two repeats.
3. Implement checkpoint-transparent dynamic core compilation and tests. Run C against
   E. Keep it only at >=10% improvement with no post-warm-up recompilation storm.
4. Profile padding and attention time independently in BC, GRPO update and rollout.
   Continue to varlen work only where at least 10% end-to-end headroom is attributable
   to padding or padded attention.
5. Implement the varlen layout and boundary-equivalence tests, then run V and CV. Do
   not substitute the existing dense block mask for the varlen kernel.
6. Run a short fixed-token training comparison and closed-loop evaluation before making
   either switch a default. Update this doc to `Implemented` only for the optimizations
   that pass their gates; record rejected results here rather than deleting them.

## 6. Implementation and measured result (2026-08-04)

Both execution paths are implemented behind the same keys in BC and GRPO. Compilation
wraps an unregistered callable, leaving the raw eager core as checkpoint owner; strict
cross-load and absence of `_orig_mod` keys are tested. Varlen batches concatenate raw
frames before the encoder, carry `seq_len`/`cu_seqlens`, reset RoPE positions per
episode and call the autograd-enabled ATen varlen FlashAttention operator. The same
layout is wired through BC, GRPO minibatches and rollout token histories. PyTorch 2.9
does not yet expose the public `torch.nn.attention.varlen` API, so the CUDA call is an
explicit version-sensitive boundary in `causal.py`; CPU uses an independent-segment
reference implementation for correctness tests.

The controlled BC benchmark used the same checkpoint, seed, batch schedule, batch size
4, BF16, RTX 4090 Laptop GPU, 20 warm-up steps and 500 measured update steps. Every
timing includes DataLoader acquisition, transfer, encoder, core, backward and optimizer.

| run | useful tok/s | vs E | median ms | p90 ms | peak GB | padding | graphs |
|---|---:|---:|---:|---:|---:|---:|---:|
| E eager/padded | 2,229 | — | 106.9 | 431.1 | 2.747 | 4.19% | — |
| C compiled/padded | 2,271 | +1.9% | 105.8 | 449.6 | 2.747 | 4.19% | 1 |
| V eager/varlen | 2,269 | +1.8% | 117.8 | 364.9 | 2.710 | 0% | — |
| CV compiled/varlen | 2,206 | -1.0% | 113.3 | 392.0 | 2.710 | 0% | 2 |

Compile warm-up cost 14.8 seconds for C and 12.7 seconds for CV. Mean loss stayed
within 0.001 across variants (0.1840-0.1850); mean gradient norm stayed within 0.04
(1.76-1.80). The complete test suite passes, including packed-versus-independent core
and whole-policy equivalence, task-boundary isolation, packed GRPO advantage mapping and
checkpoint transparency.

**Decision.** Reject both as defaults for the next base-policy run. C and V each gain
about 2%, far below the 10% maintenance gate, and CV is slightly slower. The 100-step
exploration misleadingly suggested roughly 20% gains because periodic loader stalls
dominated such a short wall-clock sample; the planned 500-step result is authoritative.
The opt-in paths stay available to benchmark GRPO update and rollout phases, but no RL
speed or closed-loop-quality claim is made yet. GPU utilization was not sampled during
this run; peak allocation is reported above.

No other repository is blocked on this experiment, so no handoff issue is required.

---

## Appendix — provenance

| claim | source |
|---|---|
| current BC path is eager, BF16/TF32, ordinary AdamW and SDPA | `src/contra_policy/train_bc.py`, `src/contra_policy/causal.py` at `446676b` |
| BC and GRPO update both pad whole episodes | `pad_episodes` and `rl.buffer.GroupBatch` at `446676b` |
| rollout core input is a padded batch of active histories | `rl.rollout.BatchedPolicyRunner._core_over_histories` at `446676b` |
| 7,104 train episodes; 725,635 usable frame positions | `cache/shard_index_v2_856541e18109970a.json`, boss-full-v1 configuration |
| 4.21% dense padding waste | deterministic `LengthGroupedSampler(batch_size=4, pool_batches=32, seed=0)` epoch: 725,635 valid / 757,500 dense frames |
| 11.30% padded causal-attention pairs are avoidable | same epoch, `sum(L_i^2)` against `sum(batch_size * max(L)^2)`, including two prefix tokens |
| current throughput excludes DataLoader latency | timer placement in `src/contra_policy/train_bc.py:226-240` |
| boundary-aware SFT packing uses cumulative lengths and varlen attention | NVIDIA NeMo/Megatron documentation and official FlashAttention repository linked in §2.2 |
