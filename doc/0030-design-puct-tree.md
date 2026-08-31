# Build a branch-safe PUCT tree with PPO-guided terminal rollouts

Status: Proposed

**Question.** How should Contra construct a persistent PUCT tree using the trained PPO
policy as its action prior and rollout policy, and how should that tree train a new policy?

**Answer.** Freeze the best PPO checkpoint for one generation round. It supplies action
priors inside the tree and samples actions from each new leaf to a real terminal result.
Stable-retro returns exact transitions; boss defeat backs up `1`, while death or timeout
backs up `0`. No critic or immediate shaped reward is used. Normalized root visits supervise
the next policy. Promote a trained candidate only after independent closed-loop evaluation.

---

## The frozen PPO checkpoint starts each generation round

The first search model is the evaluated PPO checkpoint from experiment 0028. Its action head
provides `P(action | history)` both for tree priors and rollout sampling. The checkpoint
remains frozen while a dataset round is generated, so every edge prior and rollout comes
from one identifiable policy. Its critic, PPO ratios, GAE, clipping, and PPO reference model
have no role in this first search loop.

At expansion, mask illegal controller actions, renormalize the remaining probabilities, and
store them as edge priors. A zero-mass legal set falls back to uniform legal priors. Root
Dirichlet noise and visit-temperature sampling are configurable exploration mechanisms, but
the first correctness run disables noise and chooses the most-visited root action so tree
behavior is deterministic under a fixed seed.

## A node binds an emulator snapshot to one causal history

A node represents the state before the next action. The classes separate environment state,
per-action statistics, and tree-level history ownership:

```text
Node
  emu_state:        exact stable-retro savestate bytes
  frame_token:      token for this node's observation only
  parent:           parent node, absent at root
  incoming_action: action taken from parent, absent at root
  previous_action: last action, required by the legal-action mask
  depth:            decisions below the current root
  terminal:         success, death, timeout, or non-terminal
  expanded:         whether the network has evaluated this node
  edges:            action_id -> Edge

Edge
  action_id:        discrete policy action held for one environment skip
  prior:            masked policy probability P
  visits:           selection count N
  value_sum:        backed-up value sum W
  mean_value:       Q = W / N, or zero before the first visit
  child:            resulting Node, created lazily

SearchTree
  root:             node for the current real decision
  committed_prefix: immutable frame tokens preceding the current root
  simulations:      completed-backup count for this root
  model_id:         frozen PPO checkpoint identity
```

`Node` stores one token rather than copying the full causal sequence into every descendant.
Leaf evaluation reconstructs its input as:

```text
SearchTree.committed_prefix + root-to-leaf frame tokens
```

Parent links and incoming actions make paths auditable; the current policy itself consumes
frame tokens, while `previous_action` is needed by the search legal-action mask. Children
share only immutable prefixes, and siblings never mutate common history. `Edge` owns `N`,
`W`, and `Q` because those statistics describe choosing one action from one parent, not the
child state in isolation. `SearchTree` owns operations such as `select_path`, `expand_leaf`,
`backup`, `root_policy`, and `advance_root`; `Node` and `Edge` remain data containers.

The initial implementation does not clone a transformer KV cache: the current rollout actor
caches image tokens but reruns the causal core over the full prefix. Batched leaf evaluation
should be implemented first; a branchable KV cache is a later optimization gated by profile
data.

The child node is created by restoring the parent's `emu_state`, applying the edge action,
capturing the resulting observation and savestate, and appending its encoded frame token.
Success is resolved before death and timeout, matching PPO collection and evaluation.

## Tree construction alternates simulations with real action commitment

Create the initial root by loading the task savestate, peeking its first valid observation
without advancing the game clock, encoding that frame once, and attaching the committed
history prefix. Evaluate and expand the root before the first selection. Then construct the
tree for one real decision as follows:

```text
repeat until simulation budget is exhausted:
  1. start at the current root
  2. select existing edges by PUCT until reaching an unexpanded or terminal node
  3. for an edge without a child, restore its parent savestate and execute its action
  4. capture the child savestate, resulting frame token, and terminal status
  5. evaluate the policy once and expand a non-terminal child
  6. sample PPO actions from that child until win, death, or timeout
  7. back up 1 for win or 0 for death/timeout through traversed tree edges

normalize root visits -> save policy target -> commit selected action -> re-root
```

Every simulation restores emulator state from the selected edge's parent; speculative
transitions never modify the committed environment state. Repeated simulations grow and
update the same tree. If an edge already owns a child, selection reuses that child and its
statistics rather than recreating the transition. After commitment, the chosen child becomes
the root and its known descendants remain available for the next decision.

The construction loop stops on committed success, death, or timeout. Tree depth is measured
in policy decisions, while emulator cost is counted in skipped NES frames. A maximum node
count and maximum search depth bound memory and latency independently of the simulation
budget.

Standard PUCT does not rewind after commitment. An optional bounded recovery controller may
retain recent committed roots for trace generation. After the committed trajectory actually
dies, it may restore an ancestor's emulator state and policy prefix, record the abandoned
continuation as `0`, increase that ancestor's search budget, and choose another edge. Several
zero-valued rollouts alone do not prove that a root is dead and do not trigger recovery.

## Terminal Monte Carlo rollouts supply every backed-up value

One simulation begins at the root and repeatedly selects the legal edge with the greatest:

```text
score = Q + c_puct * P * sqrt(parent visits) / (1 + N)
```

On the first unexpanded non-terminal node, evaluate the frozen policy once and store its
masked action priors. Then continue from that node by sampling the same PPO policy until a
real terminal result. Rollout states are transient and are not added to the tree; each
simulation adds only its newly expanded tree node. Boss defeat has value `1`; death or
timeout has value `0`. Back up that scalar through every traversed tree edge, incrementing
`N` and adding it to `W`. Contra is single-player, so backup never reverses the sign.

The emulator supplies exact transitions and terminal events, not the future value of a
non-terminal node. Repeated terminal rollouts estimate that value empirically. One failed
rollout therefore adds `0` but does not prove its edge is dead; a different sampled
continuation may win. Only an action whose immediate child is concretely terminal can be
excluded as terminal.

Emulator steps remain sequential within a worker because stable-retro permits one emulator
instance per process; multiple workers may run independent simulations and return their
terminal outcomes to one tree owner. Logs must separate tree nodes, rollout emulator steps,
policy calls, terminal outcomes, and wall time. Terminal rollout cost is the primary scaling
gate.

## Root visits supervise the next policy

After a fixed simulation budget, convert the root edge visits into a probability vector over
the complete policy action space; illegal and unvisited actions receive zero. Save:

```text
frame-token history, previous actions, legal mask, normalized root visits
```

Recovery records abandoned and replacement continuations as separate attempts. A later win
must not relabel states that existed only on an abandoned branch. Root visit targets may
include that branch's backed-up `0` evidence.

Training initializes a candidate from the frozen generator and minimizes policy
cross-entropy against normalized root visits. The first experiment does not train or use the
critic, and it never backs up the shaped `mc_search` reward. Candidate training reads stored
tokens initially, so encoder weights remain frozen and search data do not require repeated
image encoding.

## Correctness gates precede scaling and model promotion

The first smoke test uses one fixed Laser start state, a small depth/budget grid, and exact
replay checks. Required invariants are:

| gate | required observation |
|---|---|
| branch isolation | sibling order does not change their observations, priors, or values |
| deterministic replay | replaying a root-to-leaf action path reproduces its RAM and terminal state |
| backup arithmetic | terminal outcomes produce expected `N`, `W`, and `Q` without sign reversal |
| reward purity | every backup is exactly `0` or `1`; no shaped reward enters search |
| legal actions | masked actions receive neither prior mass nor visits |
| budget accounting | reported simulations equal completed backups; emulator and network calls are counted |
| search improvement | PUCT beats direct frozen-policy play under the same start-state and episode budget |

Only after these pass should generated visits train a candidate. Compare the candidate,
generator, and generator-plus-PUCT with independent closed-loop trials. Promote the candidate
as the next frozen generator only when it does not regress the declared task metric; retain
the previous generator and dataset manifest so a bad round is reversible.

---

## Provenance and auditability

| claim | source |
|---|---|
| root visits can supervise policy improvement | [0029](0029-design-alphazero-contra.md) and Silver et al., [AlphaZero](https://arxiv.org/abs/1712.01815) |
| current PPO model exposes action logits for priors and rollout sampling | `src/contra_policy/model.py` |
| rollout inference caches frame tokens but recomputes the full causal prefix | `src/contra_policy/rl/rollout.py`, `TokenHistoryActor` |
| success-before-death and the task budget define terminal outcomes | `src/contra_policy/rl/rollout.py`, `classify_step` |
| stable-retro permits one emulator per process in the current collector | `src/contra_policy/rl/rollout.py`, `claim_emulator` |
| emulator savestates and action stepping already exist in search | `contra_nes_data/src/agent/mc_search.py`; `contra_nes_data/src/agent/sampler.py` |
