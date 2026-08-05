"""GRPO fine-tuning of the causal policy.

Organised per ``doc/0003-grpo-code-layout.md``: generation is a subsystem with a
boundary, the buffer is the contract, and the objective sees nothing else.

    tasks    task catalog and the group sampler — GRPO's premise
    rollout  emulator -> complete Episodes
    buffer   Episodes -> group-relative advantages -> padded batches
    grpo     the objective: clipped ratio + reference KL
    trainer  the loop
"""
