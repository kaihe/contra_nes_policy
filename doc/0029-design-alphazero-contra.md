# Adapt AlphaZero's search-improvement loop to Contra

Status: Proposed

**Question.** How does AlphaZero turn a policy, value estimate, exact game rules, and
search into a stronger agent, and which parts transfer to Contra's fixed-start Laser boss
fight? Can the existing Monte Carlo search become the first planner by replacing its
history-free rollout prior with the learned policy?

**Answer.** Treat the NES emulator as the exact dynamics model and the current causal
policy network as the search guide. Proceed directly to the persistent PUCT tree
specified by [0030](0030-design-ppo-guided-mcts.md): the PPO policy already supplies a useful
closed-loop prior, so a separate bigram-versus-policy rollout gate would not answer the
remaining question. PPO supplies priors and terminal rollout actions; root visit counts
supervise the next policy. Contra has no opponent or useful self-play symmetry, so training is repeated
single-agent policy improvement from a controlled task-state distribution.

---

## AlphaZero alternates search improvement and network distillation

AlphaZero's network consumes a complete game state and emits two predictions:

```text
policy prior p(a | s): which legal actions deserve search effort
state value v(s):      expected final game outcome from this state
```

For every real move, MCTS starts at the current state and repeats four operations:

1. **Select:** descend through existing nodes using PUCT. An edge is attractive when its
   backed-up mean value is high, its policy prior is high, or it has been visited less than
   competing edges.
2. **Expand:** on reaching an unexpanded state, query the network once and create legal
   child edges with its policy probabilities as priors.
3. **Evaluate:** use the network's value prediction at that leaf. AlphaZero does not need a
   random rollout to the end of the game for every simulation.
4. **Back up:** propagate the leaf value through every selected edge, updating visit count,
   cumulative value, and mean value.

After a fixed simulation budget, the root visit-count distribution is the search-improved
policy. Early in training, sampling from it and adding root Dirichlet noise preserves
exploration; evaluation play can choose its most-visited action. A complete game stores one
record per decision:

```text
(state, root visit-count distribution, final outcome)
```

The network is then trained to imitate the visit-count distribution and predict the final
outcome. Search therefore improves the network's targets, and the improved network makes
later search cheaper and stronger. This is policy iteration through planning and
distillation, not PPO: there is no likelihood ratio, GAE target, or clipped policy update.

## The emulator is Contra's exact model but the state is not the raw frame

AlphaZero is often called model-based because search applies known game rules to generate
successor states. Contra already has the stronger equivalent of a learned world model: an
emulator savestate can be cloned and advanced exactly. Predicting future RGB frames would
add approximation error without removing the need to model RAM, collision, timing, and
hidden game state.

The policy does not observe the full emulator state, however. It acts from a causal history
of images and previous actions. A Contra search node must therefore contain both:

```text
exact planning state: emulator savestate
policy state:         causal observation/action history (or an equivalent cache)
```

Every speculative edge advances both states. Sibling branches must clone the policy context
so one branch cannot leak observations into another. Two emulator savestates that look alike
are not safely mergeable unless their policy histories are also equivalent. This makes
Contra a partially observed planning problem from the network's perspective, even though
the emulator transition itself is deterministic.

## Contra replaces self-play with fixed-task policy improvement

Go, chess, and shogi have two alternating players, a compact legal-action boundary, and a
natural terminal outcome from the current player's perspective. Contra has one learning
agent, fixed enemies, simultaneous real-time dynamics, frame-skipped controller inputs, and
long delayed consequences. There is no opponent network to self-play against.

The corresponding loop is:

```text
sample a training start state
-> run policy-guided terminal-rollout search at each decision
-> execute an action from the improved root distribution
-> finish the episode
-> train policy on root visits
-> repeat with the updated network
```

The start-state distribution becomes part of the objective. Training only from
`full_laser.state` optimizes that one fight, which is acceptable for the current component
validation goal but does not establish general Contra play. Later work should use a bank of
states spanning fight phases and player conditions. Search actions initially remain the
existing discrete controller combinations held for the existing frame skip; macro-actions
are deferred until the fixed action version exposes its measured branching and depth cost.

## Existing MC search is a policy-guided rollout baseline, not a tree

The data repository's search samples several fixed-length action sequences, scores each
sequence with a shaped search reward, commits a random-sized prefix of the best sequence,
and backtracks when the best sequence dies. Its action generator is a legal-action mask plus
a previous-action bigram prior. It keeps one committed history, not per-edge visit counts or
backed-up action values. Consequently it is greedy receding-horizon Monte Carlo search.

The smallest possible bridge would replace only the proposal distribution:

```text
current image/action history -> policy probabilities
policy probabilities * legal-action mask -> sampled rollout action
emulator + existing search reward -> rollout score
```

Always taking the policy argmax would make candidate sequences collapse to the same path.
The baseline must sample at a declared temperature and retain explicit exploration. Compare
the existing bigram generator and policy-guided generator with identical start states,
rollouts per decision, horizon, seeds, emulator-step budget, and reward configuration. The
gate is more wins found per emulator step and per wall-clock hour, not merely a higher shaped
score. Neural inference cost and batched branch evaluation must be reported separately.

That bridge is no longer a prerequisite: PPO already reaches 59/100 in independent
closed-loop evaluation, and the next unknown is whether persistent tree search improves its
decisions. It remains a useful debugging baseline if PUCT fails. Such a baseline may reuse
the search reward because its purpose is efficient trace discovery.
That reward includes progress, damage, death, items, and button cleanliness, so its score is
not a probability of eventually winning. It must not silently become the critic target.

## PUCT search uses policy priors, backed-up values, and visit targets

Contra should now implement the persistent tree directly, as specified by
[0030](0030-design-ppo-guided-mcts.md). Each node owns legal edges; each edge stores the policy
prior, visit count, cumulative
value, and mean value. Selection uses a PUCT score conceptually equivalent to:

```text
selection score = backed-up mean value
                + exploration constant * policy prior
                  * sqrt(parent visits) / (1 + edge visits)
```

The policy concentrates simulations on plausible controller inputs while the visit term
still tests uncertain alternatives. In the first implementation, the PPO policy samples a
rollout from each new leaf to a real terminal result. A win backs up `1`; death or timeout
backs up `0`. No critic or shaped immediate reward enters search. Because Contra is
single-agent, values keep the same sign at every depth—there is no alternating opponent
perspective as in board-game AlphaZero.

The initial tree should advance one current action per edge and reuse the chosen child's
subtree at the next real decision. Search output is the normalized root visit counts, not
the single best sampled sequence. Distillation trains the action head against those soft
targets. Critic and PPO losses are excluded from this first experiment so any gain can be
attributed to terminal-rollout planning targets.

## Search validity is gated before policy retraining

Search can consume too much emulator time or overfit noisy rollout results while appearing
to improve its internal score. Evaluation therefore uses these gates:

| gate | comparison | pass condition |
|---|---|---|
| leaf evaluation | PPO terminal rollouts | backups are binary terminal outcomes and throughput is affordable |
| tree improvement | raw policy versus PUCT-selected actions | matched closed-loop win rate improves before any distillation |

Only then should searched root distributions become training labels. A held-out state bank
must measure whether distillation reproduces search choices and whether the distilled policy
improves without online search. The planner is unsuccessful if gains require reward-specific
behavior that lowers actual wins, if inference removes the emulator-step saving, or if tree
search cannot look far enough to distinguish actions within the available budget.

---

## Provenance and auditability

| claim | source |
|---|---|
| AlphaZero uses one policy/value network, PUCT MCTS, root visit targets, and terminal outcome targets | [Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*](https://arxiv.org/abs/1712.01815) |
| AlphaGo Zero expands leaves with policy priors, evaluates them with value, and backs values through search | [Silver et al., *Mastering the game of Go without human knowledge*](https://www.nature.com/articles/nature24270) |
| current search samples prior-guided masked rollouts and greedily selects the highest reward | `contra_nes_data/src/agent/sampler.py`; `contra_nes_data/src/agent/mc_search.py` |
| search reward contains progress, combat, items, terminal events, and button costs | `contra_nes_data/src/agent/reward.py` |
| current policy critic predicts binary success and PPO u158 reaches 59/100 | [0027](0027-design-ppo-critic.md); [0028](0028-exp-laser-ppo-critic.md) |
