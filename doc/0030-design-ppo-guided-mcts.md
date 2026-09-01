# Build PPO-guided MCTS for the Laser boss fight

Status: Proposed

**Question.** How should Contra use the trained PPO policy to construct MCTS trees that
produce policy-improvement data for the Laser boss fight?

**Answer.** Freeze PPO for one generation round. At every committed state, use PUCT and
terminal PPO rollouts to turn its action prior into a stronger root-visit distribution, then
save that distribution as one training target. The tree is a temporary teacher: re-rooting
may discard explored branches after their outcomes affect visits. The objective is neither a
large final tree nor a few winning traces, but many state-to-search-policy records from many
episodes. Policy optimization is specified separately after search data pass correctness and
closed-loop improvement gates.

---

## The frozen PPO checkpoint starts each generation round

The first search model is the evaluated PPO checkpoint from experiment 0028. Its action head
provides `P(action | history)` both for tree priors and rollout sampling. The checkpoint
remains frozen while a dataset round is generated, so every edge prior and rollout comes
from one identifiable policy. Its critic, PPO ratios, GAE, clipping, and PPO reference model
have no role in this first search loop.

The first Laser implementation pins one small, matched setup:

| setting | initial value |
|---|---|
| generator | `runs/ppo/2026-08-27/laser-critic-14-56-52/checkpoints/ppo-final.pt` (u158) |
| task | `win_level1_20260630171218_i8`, exact `full_laser.state` start |
| policy input | 256px frame, learned null goal, causal frame/action history |
| action execution | the task's 21-action vocabulary; one action per task-defined skip |
| episode limit | 216 policy decisions, matching evaluation's `max(24, 2 * 108)` |
| search limit | 16 completed simulations per committed decision; 2,048 live nodes |
| sampling | temperature 1.0, seed 0, no root noise |
| commit | greatest root visit count; fixed action-id tie break |
| recovery | disabled for the first correctness run |

Initialization is explicit:

```text
load frozen PPO checkpoint and task metadata
create emulator and restore full_laser.state
capture the initial observation without advancing the emulator
encode that frame and start an empty committed_prefix
create root(previous_action = task start value, emu_state = restored state)
evaluate root once, apply the legal-action mask, and create its 21-action edge table
```

Before search begins, restore and peek the state a second time in the same emulator and
require identical savestate, RAM, and observation. Stable Retro permits only one emulator
per process; the repeated restore still catches a mismatched or unstable task before search.

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
caches image tokens but reruns the causal core over the full prefix. Batch policy inference
for newly expanded leaves first; a branchable KV cache is a later optimization gated by
profile data.

The child node is created by restoring the parent's `emu_state`, applying the edge action,
capturing the resulting observation and savestate, and appending its encoded frame token.
Success is resolved before death and timeout, matching PPO collection and evaluation.

## One loop searches a state, saves its target, and commits one action

Load the task state, capture its first frame, and expand the root with PPO action priors.
Then run one loop until the committed episode wins, dies, or times out:

```text
while episode is active:

  run 16 simulations from the current root:

    1. Start at the root.

    2. Follow existing edges with the greatest PUCT score:

         score = Q + c_puct * P * sqrt(parent visits) / (1 + N)

       P = frozen PPO prior
       N = visits to this edge
       Q = mean terminal result backed up through this edge

    3. Stop at the first edge with no child, or at an existing terminal child.
       For a missing child, execute its one action and add one permanent node.

    4. If the child is non-terminal, evaluate PPO there and create its legal edges.

    5. Use an existing terminal child's result directly. Otherwise, from the new
       child sample temporary PPO actions until:

         boss win     -> value 1
         death/timeout -> value 0

       These rollout states are temporary and are not added to the tree.

    6. Back up the value through the permanent edges selected in steps 2-3.

  after 16 simulations:

    7. Normalize the current root's edge visits.
       Save them as one policy-training target.

    8. Commit the most-visited root action in the real episode.

    9. Make its child the new root and discard unused sibling branches.
```

One simulation adds at most one permanent node. Its terminal rollout may execute many
temporary actions, but those actions only contribute the backed-up `0` or `1`. A failed
rollout is evidence, not proof that its edge can never win.

One committed decision produces one durable record:

```text
task and generator IDs
search seed and configuration
causal context or reproducible action prefix
legal-action mask and frozen-PPO prior
raw root visits and normalized visit target
rollout terminal counts and committed action
```

Re-rooting moves the old root's frame token into `committed_prefix`, retains the selected
child's subtree, and prunes only the old root and its unselected branches. The prefix remains
the causal policy context; the root record separately preserves what search learned at the
old state. Thus a 75-decision episode produces 75 training targets, not one winning trace.
Keep records from losing episodes too. Run many independently seeded episodes to build a
dataset of `(causal context, search visit distribution)` pairs. The next design owns policy
loss, dataset weighting, and candidate promotion.

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
| search improvement | PPO-guided MCTS beats direct frozen PPO under the same start state and episode budget |

The implementation lives in `src/contra_policy/mcts/`. `core.py` owns generic tree mechanics;
`policy.py` adapts the frozen causal policy; and `laser.py` owns Stable-Retro task execution.
Only after these gates pass should a separate policy-update design consume generated visits.

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
