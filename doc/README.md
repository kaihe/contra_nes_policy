# Design docs — `contra_nes_policy`

Conventions in the `contra-nes-workflow` skill. In short: `NNNN-topic.md`, sequential
and never dated, every doc carries a `Status:` header, and this index is the first
thing to read. A doc whose status is stale is worse than no doc.

Requests to another repo are **issues on that repo**, not docs here — see the
`contra-nes-handoff` skill.

| doc | status | what it decides |
|---|---|---|
| [0001 One token per frame: the image encoder](0001-image-encoder.md) | Implemented — §2 superseded by 0002 | `src/contra_encoder/`, one 512-d token per frame with occupancy decoded from it; `prev_action` deleted; the gate is `peak_hit`/`pck16`, **not** `point_err_px`; SB3 and an LLM backbone rejected |
| [0002 A goal-agnostic encoder](0002-symmetric-encoder.md) | Proposed | one symmetric `encode(image)` for frames and goals alike; goal matching moves to the policy's attention; 3.65M of goal-specific machinery deleted; reconstruction behind an ablation |

## Open questions not yet in a doc

- **`rollout.batch_size`.** Measured +23% end-to-end going 16 → 64 (collect throughput
  205 → 305 decisions/s, GPU 2.86 → 1.73 ms/decision), for one config line. Needs
  `rollout.steps` re-tuned for the larger overshoot, and host RAM checked at batch 64.
- **The optimiser phase.** 41.7% of a training update and never profiled.
  `recompute_old_logprobs` and `ppo_epochs: 2` are each a full forward pass, and
  `minibatch_episodes: 4` at `seq_len: 32` peaked at 2.6 GB of a 16 GB card.
