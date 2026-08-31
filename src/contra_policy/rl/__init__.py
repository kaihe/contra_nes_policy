"""On-policy post-training of the causal policy.

Organised per ``doc/0003-design-grpo-code-layout.md``: generation is a subsystem with a
boundary, the buffer is the contract, and the objective sees nothing else.

    tasks    task catalog and the group sampler — GRPO's premise
    rollout  emulator -> complete Episodes
    buffer   Episodes -> GRPO or timestep PPO batches
    grpo     the objective: clipped ratio + reference KL
    ppo      critic, GAE, clipped actor and value objectives
    trainer / ppo_trainer  the loops
"""
