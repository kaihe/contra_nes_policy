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
| [0013 Scaling grid](0013-scaling-grid.md) | Proposed | GRPO eliminated (eval 0014: 9.5% = init), so cross 6 core sizes (1.6M–101M) with the 5 prefixes of `boss-spread-10k-v1` (762–9,900 episodes), boss-only. Caching the frozen encoder's tokens makes it ~3 GPU-h instead of ~50. Predicts the model axis is flat. One start state, so mixed-v2 transfer is co-primary. |
| [0012 Prove GRPO moves boss](0012-spread-grpo.md) | Proposed | weapon stratification shows ~40% of boss tasks are unwinnable (Regular/Flamethrower: 0 wins in 316 rollouts) and every boss success ever recorded came from Spread or Laser. So test the optimizer where a win is reachable: **binary** reward (`progress_coef: 0.0`), Spread + rapid only (120 train / 13 val). Win density supplies the group variance the graded reward was invented for — 22.5% degenerate groups vs 61% pooled, falling as the policy improves. Adds the cumulative KL stop 0011 lacked. |
| [0011 Is boss RL-solvable?](0011-boss-grpo.md) | Implemented | ran 10 h → 1,619 updates and landed in the **low/low** cell: train probe flat ~4–6% then collapse, held-out 7.5/11.0/7.5% vs an 8.5% init. All three registered risks fired — cumulative `kl_ref` → 0.245 unbounded, entropy → 0.41, and chip-without-kill (damage per failure up, wins → 0). Grading did cut wasted rollout budget 51% → 27%, but that also hid hopeless tasks from the difficulty sampler, which keys on group variance. Records why a boss verdict must be read on two axes (train-start vs held-out) and why a negative needs a plateau rather than a low endpoint. Supersedes 0008; successor is [0012](0012-spread-grpo.md). |
| [0010 No offline CE proxy](0010-dropout-regularization.md) | Implemented | the CE-optimal checkpoint plays 12.9 pp *worse* than the overfit one, and the proposed fix — CE on the non-`R` frames — turns out to be a constant 2.1–2.3× multiple of total CE at both ends of training and across 4× data. Held-out imitation loss degrades uniformly while play improves, so *no* reweighting of CE can proxy closed-loop success. The sweep ran: best cell dropout 0.2 at 69.0% (+3.5 pp, p = 0.050 uncorrected across three comparisons, incoherent dose response), no cell cleared the Δ ≤ 0.4 gate, and rate alone asymptotes at Δ ≈ 0.5 so escalating cannot clear it. Rejects model scaling — train CE is already 0.051. |
| [0009 Boss-data scaling](0009-boss-data-scaling.md) | Implemented | measured four nested mixed-v2 boss-data prefixes at fixed model size, compute and family exposure (seed 0 only, not three). Curve is flat — pooled +0.2 pp D1→D8, boss ~90% death at every scale — so the predeclared "do not scale model or RL compute on this premise" row fired. D4 final (67.1%) is the best base checkpoint it produced. |
| [0008 Fourteen-hour GRPO](0008-fourteen-hour-grpo.md) | Superseded by [0011](0011-boss-grpo.md) | hold the current recipe fixed for a wall-clock-limited ~1,000-update run from the latest action-only BC policy; require true resumption, fixed probes and held-out checkpoint selection before claiming that longer training teaches gameplay. |
| [0007 Compile and varlen experiment](0007-compiled-varlen-training.md) | Implemented | both optimizations missed the 10% end-to-end gate and were rolled back; eager/padded remains the shared BC/GRPO path, while the corrected loader-inclusive timer stays. |
| [0006 Action-only base policy](0006-action-only-base-policy.md) | Implemented | train one action-only causal GPT with masked cross-entropy, report only Karpathy's optimisation metrics, and replace the boss slice with all 666 episodes from the full-fight release while preserving old-checkpoint compatibility. |
| [0005 Graded rewards](0005-graded-reward.md) | Accepted | 53% of every rollout budget produces no gradient because the group's members all agree. Two phases attacking opposite tails: HP grading rescues all-failure boss groups, a budget-normalised speed term rescues all-success ones. **Phase 1 is built and cuts wasted budget 0.514 → 0.27, but its matched confirmation run is still owed** (both graded runs were cut at updates 61/77 and never evaluated closed-loop); phase 2 is not built. Rejects the symmetric step penalty, which scores a fast death above a long survival. |
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
