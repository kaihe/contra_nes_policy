# Train one Contra policy-value network through emulator MCTS

Status: Proposed

**Question.** How should Contra combine the existing CNN/GPT policy, exact emulator
search, the `mc_search` reward, and decoded game state into one iterative policy
improvement system?

**Answer.** Keep the CNN image encoder and causal GPT backbone, and train one network
with policy, value, `dx`/`dy`, weapon, and rapid-fire heads. Every MCTS node owns an exact
emulator savestate plus its observation history. PUCT uses the network's action
distribution as its prior and its value prediction at newly expanded leaves. Root visit
counts supervise the policy head; the realized `mc_search` reward-to-go supervises the
value head; decoded emulator state supervises all auxiliary heads. Repeated search,
episode generation, and joint network training form the policy-improvement loop.

---

## One CNN/GPT trunk emits policy, value, and game-state predictions

The existing CNN encodes each game image and the causal GPT consumes the resulting image
tokens in decision order. Preserve the existing action timing: one decision is one legal
controller action held for the task's frame skip. The network consumes only information
available to the deployed policy: image history and the existing causal context. Exact
emulator bytes and decoded RAM labels never become network inputs.

The final hidden state feeds six outputs:

| head | target | loss | search role |
|---|---|---|---|
| policy | normalized MCTS root visits over legal actions | cross-entropy | PUCT prior |
| value | cumulative future `mc_search` reward | regression | leaf evaluation |
| `dx`, `dy` | player displacement since the preceding decision | normalized regression | none |
| weapon | decoded current weapon class | categorical cross-entropy | none |
| rapid fire | decoded current rapid-fire flag | binary cross-entropy | none |

The state heads are jointly trained auxiliary outputs, not a separate stage. They force
the shared representation to retain motion and equipment information useful to policy
and value prediction. Search does not use their predictions because the emulator already
provides exact transitions and labels. Auxiliary loss weights must be declared before a
run and kept small enough that policy and value validation do not regress.

The existing GPT-policy checkpoint initializes the CNN, GPT, and action head. The value,
`dx`/`dy`, weapon, and rapid-fire heads are newly and randomly initialized; no PPO critic
or previous auxiliary-head weights are loaded. Record the initialization seed. The value
head begins untrained and improves from completed searched episodes.

## Every search node restores an exact emulator and causal policy state

A tree node contains:

```text
emulator savestate
current resized image and decoded RAM state
causal image/action history or an equivalent GPT cache
previous action, decision count, and terminal status
per-action immediate reward, prior, visit count, value sum, and child reference
```

The savestate is the authoritative state for branch restoration and advancement. Decoded
RAM supplies legal-action masks, terminal/reward observations, and auxiliary labels.
Neither is a learned world model. Sibling branches must restore their own savestate and
causal context; observations generated on one branch cannot enter another.

The complete savestate stays in the live tree but need not enter the training dataset.
Each saved sample contains the policy observation/history, legal mask, visit target,
reward-to-go, decoded auxiliary targets, task identity, and decision index. A state
hash may be retained for replay auditing. All label decoders must be versioned and tested
against representative airborne, grounded, weapon, and rapid-fire states.

## PUCT turns network priors and leaf values into policy targets

At every real decision, MCTS runs a fixed number of simulations. Each simulation selects
edges using backed-up mean value plus prior-weighted exploration, restores and advances
the emulator, records that transition's exact `mc_search` reward, and expands at most one
new leaf. A non-terminal leaf is evaluated once by the network. Backup adds each selected
edge's immediate reward to the downstream leaf value. A terminal leaf has zero future
value after its terminal transition reward. Contra is single-player, so backup never
changes the value sign.

After the simulation budget, normalized root visits are the improved action distribution.
Episode generation samples from that distribution during exploration and may choose its
most-visited action during evaluation. The chosen child's subtree is retained for the
next decision. Exploration noise, visit temperature, simulation budget, and PUCT constant
are run configuration, not hidden defaults.

Every generation uses the same AlphaZero expansion protocol: expand one new leaf, evaluate
its policy and value once, and back up its value together with exact edge rewards. There
are no terminal policy rollouts. Generation zero therefore searches with a pretrained
action prior and a random value head; completed real episodes provide the first value and
auxiliary targets.

## The value predicts `mc_search` reward-to-go under searched play

For a non-terminal state, value means the expected sum of future per-decision rewards from
that state to game end when subsequent actions follow the current MCTS-improved policy.
The initial design is undiscounted. A training sample at decision `t` receives the sum of
the rewards generated by decisions `t` through the terminal decision. The state after the
terminal transition has value zero.

Use the complete `mc_search` reward configuration without removing its dense or
generation-specific terms: advancement, enemy and boss damage, item events, level
completion, death, and per-button hold costs. Consequently value is a search-utility
estimate, not win probability. The same reward implementation and weights must score tree
transitions, construct training returns, and report episode return. Also report win rate
separately so optimization of dense reward cannot be mistaken for better completion.

The policy named in this definition is the root-visit distribution, not the raw network
prior. The network learns the policy produced by the preceding search generation; the
next generation uses that approximation to guide a stronger search.

## Iterations generate episodes, train jointly, and gate promotion

One iteration performs:

```text
sample a declared training start state
run MCTS at each decision and store the root sample
advance the emulator with an action from root visits
finish the episode and compute each sample's reward-to-go
add samples to a bounded replay buffer
train all heads jointly on shuffled recent samples
compare the candidate with the accepted network using fresh RNG seeds
promote only after the candidate passes the declared gates
```

The first loop always restores the same Laser boss savestate. This deliberately tests
whether search generation, reward backup, replay, and joint improvement work under the
smallest controlled state distribution; it does not test state or weapon generalization.
Evaluation also starts from that state but uses fresh episode/search RNG seeds and no
training exploration noise. A later design revision may introduce separate state banks
only after this loop works.

Required gates are deterministic state restoration, correct masking and visit backup,
auxiliary accuracy above constant baselines, value fit against realized reward-to-go,
policy imitation of search visits, and improvement in both closed-loop return and reported
win rate over the accepted network. Report both emulator decisions and wall-clock time.
Tree reuse, cached image
tokens, per-node GPT caches, and batched leaf evaluation are permitted optimizations only
when they preserve search results within declared numerical tolerance.

---

## Provenance and auditability

| claim | source |
|---|---|
| current policy uses a CNN image encoder and causal GPT with action and grounding outputs | `src/contra_policy/model.py`; [0002](0002-design-gpt-policy.md) |
| current MCTS nodes already retain emulator bytes, RAM state, and per-edge PUCT statistics | `src/contra_policy/mcts/core.py` |
| emulator branches restore savestates and advance one frame-skipped action | `src/contra_policy/mcts/laser.py` |
| current reward and terminal classification are implemented by the rollout stack | `src/contra_policy/rl/rollout.py`; [0005](0005-design-graded-reward.md) |
| policy/value networks, PUCT, root visits, and final outcomes form AlphaZero's loop | Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*, arXiv:1712.01815 |
