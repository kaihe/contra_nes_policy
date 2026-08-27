# Add a state-value critic for binary-reward PPO

Status: Proposed

**Question.** Stage-one GRPO raised fixed-start Laser success from 25% to 49%, but a
second 200-update stage changed the policy without improving success. Should the next
post-training method learn a state value and use GAE, and what must make that test valid?

**Answer.** Add one scalar value head to the causal policy and train actor–critic PPO on
complete episodes with the unchanged binary terminal reward. The critic predicts eventual
success from each pre-action history; GAE turns successive prediction errors into
timestep-specific advantages. This is a variance-reduction hypothesis, not a claim that a
critic discovers causal actions. Critic calibration, policy-ratio parity, reference KL,
exact resumption, and matched closed-loop evaluation are hard gates.

---

## The critic predicts success from causal history

At action timestep `t`, the existing transformer hidden state `h_t` already represents the
visible frame and preceding actions. Attach `value_head: Linear(d_model, 1)` and define

```text
V(h_t) = sigmoid(value_head(h_t)) ≈ P(eventual success | h_t, policy)
```

The sigmoid makes the binary-return interpretation explicit. Initialize its weights and
bias to zero, so an untrained critic predicts 0.5 rather than introducing arbitrary
advantages. The actor and frozen reference both initialize from stage-one GRPO u25; the
reference needs only action logits and never owns a value head.

The critic shares the causal core with the actor. During critic warmup the core and action
head are frozen, so only the new head learns. During joint PPO, value gradients may update
the shared core; log actor and critic gradient norms separately so interference is visible.

## Complete episodes define returns without reward shaping

Keep the objective used by evaluations 0032–0033:

```text
r_t = 0 for non-terminal transitions
r_t = 1 when the boss is defeated
r_t = 0 on death or timeout
gamma = 1.0
```

With complete episodes and `gamma=1`, every Monte Carlo value target is the episode's
binary outcome. This measures win probability and does not prefer a quick win over a slow
one. There is no timeout bootstrap: the task budget defines timeout as a terminal failure,
matching closed-loop evaluation.

For stored old-policy values, compute backward:

```text
delta_t = r_t + gamma * V_old(h_{t+1}) - V_old(h_t)
A_t = delta_t + gamma*lambda*delta_(t+1) + ...
lambda = 0.95
value_target_t = stop_gradient(A_t_raw + V_old(h_t))
```

Set `V_old(h_{T+1}) = 0` after every terminal outcome. GAE may vary advantages across
timesteps, but terminal-only data still cannot prove which action caused a win. Its testable
benefit is lower-variance conditional baselining than one scalar for an entire GRPO episode.

## Rollouts store behavior values and remain strictly on-policy

Extend `Episode` with one float32 behavior value per sampled action. The collector obtains
logits and values in the same dropout-disabled forward pass, then stores sampled action,
behavior log-probability, and `V_old(h_t)`. The buffer pads complete episodes and emits a
mask, returns, GAE advantages, old log-probabilities, and old values.

One update collects at least 128 complete episodes from the exact Laser savestate. There is
no group filtering or adaptive stopping on reward: PPO consumes successes and failures
alike. Before optimization, recomputation must give policy ratio mean 1 and value predictions
equal to stored values within bf16 tolerance. One epoch avoids repeatedly fitting the same
on-policy sample; minibatches contain whole episodes so causal histories are never cut.

## Separate actor and critic losses expose failure

The actor retains the corrected clipped-ratio objective and reference regularization:

```text
L_actor  = -mean(min(ratio*A, clip(ratio, 0.8, 1.2)*A))
           + beta_kl*KL(policy || frozen_u25) - beta_H*entropy
L_value  = mean((V(h_t) - value_target_t)^2)
L_total  = L_actor + value_coef*L_value
```

Keep an unnormalized advantage copy for the value target, then normalize the actor's
advantages once over real timesteps in the collected batch. Start actor LR at
`2e-6`, critic-head LR at `1e-4`, `value_coef=0.5`, `beta_kl=0.02`, and entropy coefficient
`0.01`. Clip the combined gradient norm at 1.0 and retain the 0.10 ten-update reference-KL
guard. These are starting values for a smoke test, not conclusions to tune after viewing
closed-loop results.

Log policy loss, value BCE, Brier score, explained variance, actor/critic gradient norms,
behavior KL, reference KL, entropy, clip fraction, and success. A falling value loss without
better held-out Brier score is critic overfit, not successful credit assignment.

## Critic warmup must beat a constant predictor

Collect 512 u25 episodes once, split them deterministically 80/20 by episode index, and
train only the value head. Compare against a constant predictor equal to the training-set
success rate. Proceed to actor updates only when validation Brier score beats that constant
and explained variance is positive; otherwise stop the experiment before policy compute.

The previous deleted PPO critic reached explained variance only about 0.33. That result is
the prior against this design, not evidence to ignore: the new test uses a 49%-successful
single-task policy, complete histories, complete episodes, and explicit critic validation.
If it cannot pass the warmup gate in this easier setting, restore GRPO and do not rebuild a
larger critic.

## Checkpoints make critic training exactly resumable

Save actor/core weights, value-head weights, both optimizer states, update count, elapsed
time, task sampler state, rollout and minibatch RNG states, and Python/NumPy/Torch/CUDA RNG
states. Save the frozen u25 reference identity and reject a resume whose reference differs.
Old BC and GRPO checkpoints remain loadable as actor initialization; a missing value head is
initialized to the neutral 0.5 predictor only for a new PPO run, never during resume.

Save every 25 updates for a 200-update feasibility run. Evaluation uses the exact 100-roll
Laser protocol from 0033. The experiment succeeds only if a predeclared checkpoint improves
the 49% reference beyond evaluation noise while critic validation remains calibrated. Flat
success with positive explained variance means better baselining is insufficient; invalid
ratios, critic validation worse than constant, or uncontrolled KL invalidate the run.

---

## Provenance and auditability

| claim | source |
|---|---|
| corrected GRPO raises Laser success 25% to 49% | evaluation `doc/0032-result-laser-projection-grpo.md` |
| second-stage u125 is 51% with overlapping interval; no measured gain | evaluation `doc/0033-result-iterative-laser-grpo.md` |
| second stage moves to KL 0.058 without improvement | `runs/grpo/2026-08-27/laser-projection-stage2-09-29-52/metrics.csv` |
| prior PPO critic reaches explained variance about 0.33 | `src/contra_policy/rl/tasks.py`, retained measurement from the deleted PPO run |
| current ratio, dropout, KL and resume invariants | [0003](0003-design-grpo-code-layout.md) §6 |
