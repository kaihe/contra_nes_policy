# Build a branch-safe PUCT tree from the PPO policy and critic

Status: Proposed

**Question.** How should Contra construct a persistent PUCT tree using the trained PPO
policy as its initial action prior and leaf evaluator, and how should that tree produce
supervision for a new policy?

**Answer.** Freeze the best PPO checkpoint for one generation round. Each root search runs
PUCT over exact emulator transitions while every node preserves the matching causal policy
history. New leaves receive legal-action priors and a win-probability estimate from the
frozen network; simulations back that value up without opponent sign changes. The normalized
root visit counts supervise the next policy, while completed episode outcomes supervise its
value head. Promote a trained candidate only after independent closed-loop evaluation.

---

## The frozen PPO checkpoint starts each generation round

The first search model is the evaluated PPO checkpoint from experiment 0028. Its action head
provides `P(action | history)` and its sigmoid value head estimates eventual task success.
Both remain frozen while a dataset round is generated, so every sample in that round has one
identifiable search policy. PPO ratios, GAE, clipping, and the PPO reference model have no
role in this training loop.

At expansion, mask illegal controller actions, renormalize the remaining probabilities, and
store them as edge priors. A zero-mass legal set falls back to uniform legal priors. Root
Dirichlet noise and visit-temperature sampling are configurable exploration mechanisms, but
the first correctness run disables noise and chooses the most-visited root action so tree
behavior is deterministic under a fixed seed.

## A node binds an emulator snapshot to one causal history

A node represents the state before the next action and owns:

```text
emu_state:        exact stable-retro savestate bytes
frame_tokens:     immutable causal token sequence through this observation
previous_actions: action sequence used to reach the node
terminal:         success, death, timeout, or non-terminal
edges:            legal actions and their search statistics
```

The action history is retained for audit and reconstruction; the current model consumes the
causal frame-token sequence rather than a recurrent hidden state. Children may share their
parent's immutable prefix and append one token. Siblings must never mutate shared history.
The initial implementation does not clone a transformer KV cache: the current rollout actor
caches image tokens but reruns the causal core over the full prefix. Batched leaf evaluation
should be implemented first; a branchable KV cache is a later optimization gated by profile
data.

An edge means holding one discrete policy action for the existing environment skip. It owns:

```text
action, prior P, visit count N, value sum W, mean value Q = W / N
```

The child node is created by restoring the parent's `emu_state`, applying the edge action,
capturing the resulting observation and savestate, and appending its encoded frame token.
Success is resolved before death and timeout, matching PPO collection and evaluation.

## PUCT selection, expansion, and backup preserve win probability

One simulation begins at the root and repeatedly selects the legal edge with the greatest:

```text
score = Q + c_puct * P * sqrt(parent visits) / (1 + N)
```

On the first unexpanded non-terminal node, evaluate the frozen network once. Store its masked
policy priors and use `sigmoid(value_logit)` as the leaf value. A terminal success has value
`1`; death or timeout has value `0`. Back up that same scalar through every traversed edge,
incrementing `N` and adding it to `W`. Contra is single-player, so backup never reverses the
sign as two-player AlphaZero does.

Network requests from newly reached leaves should be accumulated into a GPU batch. Emulator
steps remain sequential within a worker because stable-retro permits one emulator instance
per process; multiple workers may search independent roots. Each unique node is evaluated
once. Logs must separate emulator steps, encoder calls, core calls, and wall time so a higher
visit budget cannot hide unusable throughput.

## Root visits become policy targets and the chosen child becomes the next root

After a fixed simulation budget, convert the root edge visits into a probability vector over
the complete policy action space; illegal and unvisited actions receive zero. Save:

```text
frame-token history, previous actions, legal mask, normalized root visits
```

Choose the real environment action from that distribution, advance the committed episode,
and reuse the selected child and its descendants as the next tree. Discard sibling subtrees.
Continue until success, death, or the evaluation-matched decision budget. Then attach the
same binary episode outcome to every saved decision record.

Training initializes a candidate from the frozen generator and minimizes policy
cross-entropy against root visits plus binary value loss against the completed outcome.
Search values are not critic labels, and the shaped Monte Carlo reward is not used. Candidate
training reads stored tokens initially, so encoder weights remain frozen and search data do
not require repeated image encoding.

## Correctness gates precede scaling and model promotion

The first smoke test uses one fixed Laser start state, a small depth/budget grid, and exact
replay checks. Required invariants are:

| gate | required observation |
|---|---|
| branch isolation | sibling order does not change their observations, priors, or values |
| deterministic replay | replaying a root-to-leaf action path reproduces its RAM and terminal state |
| backup arithmetic | hand-built trees produce expected `N`, `W`, and `Q` without sign reversal |
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
| root visits supervise policy and terminal outcomes supervise value | [0029](0029-design-alphazero-contra.md) and Silver et al., [AlphaZero](https://arxiv.org/abs/1712.01815) |
| current PPO model exposes action logits and a scalar value logit | `src/contra_policy/model.py` |
| rollout inference caches frame tokens but recomputes the full causal prefix | `src/contra_policy/rl/rollout.py`, `TokenHistoryActor` |
| success-before-death and the task budget define terminal outcomes | `src/contra_policy/rl/rollout.py`, `classify_step` |
| stable-retro permits one emulator per process in the current collector | `src/contra_policy/rl/rollout.py`, `claim_emulator` |
| emulator savestates and action stepping already exist in search | `contra_nes_data/src/agent/mc_search.py`; `contra_nes_data/src/agent/sampler.py` |
