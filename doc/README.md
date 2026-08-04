# Design docs — `contra_nes_policy`

Conventions in the `contra-nes-workflow` skill. In short: `NNNN-topic.md`, sequential
and never dated, every doc carries a `Status:` header, and this index is the first
thing to read. A doc whose status is stale is worse than no doc.

One doc per **topic**, not per revision: a superseded design belongs in its successor's
"what was rejected" section, where a reader meets it in context, rather than in a
separate file they must reconcile.

Requests to another repo are **issues on that repo**, not docs here — see the
`contra-nes-handoff` skill.

| doc | status | what it decides |
|---|---|---|
| [0007 Compile, then gate varlen packing](0007-compiled-varlen-training.md) | Proposed | benchmark checkpoint-transparent dynamic compilation first, then boundary-safe varlen FlashAttention across BC and RL; explicitly reject cached encoder tokens because they cannot accelerate on-policy rollout. |
| [0006 Action-only base policy](0006-action-only-base-policy.md) | Implemented | train one action-only causal GPT with masked cross-entropy, report only Karpathy's optimisation metrics, and replace the boss slice with all 666 episodes from the full-fight release while preserving old-checkpoint compatibility. |
| [0005 Graded rewards](0005-graded-reward.md) | Proposed | 53% of every rollout budget produces no gradient because the group's members all agree. Two phases attacking opposite tails: HP grading rescues all-failure boss groups, a budget-normalised speed term rescues all-success ones. Rejects the symmetric step penalty, which scores a fast death above a long survival. |
| [0004 GRPO in two phases](0004-grpo-experiment-plan.md) | Implemented | phase 1 validates the stack on three families; phase 2 branches from the *same BC checkpoint* to test boss. Both ran: pooled val ~71% for each, boss 3.5% → 10.5%, which falsifies this doc's own "boss does not improve" prediction. §4 records how every prediction resolved. |
| [0003 How to organise a GRPO stack](0003-grpo-code-layout.md) | Proposed | borrow slime's decomposition (generation as a subsystem), SB3's buffer-as-contract, LLaMA-Factory's config discipline. Rejects SB3's class hierarchy and stage-dispatch. |
| [0002 A plain causal transformer over whole episodes](0002-gpt-policy.md) | Implemented | Llama-shaped causal core over `[interaction, goal, frame × 510]`; the episode becomes the sequence, so carried memory, chunking, truncated BPTT and `index_bias` all delete. 6.7× cheaper attention. Depends on 0001. |
| [0001 The image encoder](0001-image-encoder.md) | Implemented | `src/contra_encoder/`: one symmetric `encode(image)` for frames and goal frames alike, 4-class occupancy decoded from a 512-d token, goal matching left to the policy's attention. Records the rejected alternatives (goal-conditioning, SB3, an LLM backbone, `prev_action`) and the `point_err_px` false alarm. |

## Open questions not yet in a doc

- **`rollout.batch_size`.** Measured +23% end-to-end going 16 → 64 (collect throughput
  205 → 305 decisions/s, GPU 2.86 → 1.73 ms/decision), for one config line. Needs
  `rollout.steps` re-tuned for the larger overshoot, and host RAM checked at batch 64.
- **The optimiser phase.** 41.7% of a training update and never profiled.
  `recompute_old_logprobs` and `ppo_epochs: 2` are each a full forward pass, and
  `minibatch_episodes: 4` at `seq_len: 32` peaked at 2.6 GB of a 16 GB card.
