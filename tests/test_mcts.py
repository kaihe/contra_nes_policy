"""Correctness gates for design-0030 terminal-success MCTS."""

import numpy as np

from contra_policy.mcts import Evaluation, Node, SearchConfig, SearchTree, Terminal, Transition


class FakePolicy:
    num_actions = 2

    def encode(self, observation):
        return int(np.asarray(observation).item())

    def evaluate(self, frame_tokens, previous_action):
        del frame_tokens, previous_action
        return Evaluation(np.array([0.75, 0.25]), value=0.4)


class FakeEnvironment:
    def __init__(self):
        self.transitions = []

    def legal_mask(self, node):
        return np.ones(2, dtype=bool)

    def step(self, node, action_id):
        self.transitions.append((node.steps, action_id))
        step = node.steps + 1
        terminal = Terminal.SUCCESS if step >= 3 else Terminal.ACTIVE
        reward = 2.0 if action_id == 0 else -1.0
        return Transition(f"s{step}".encode(), np.array(step), step, reward, terminal)


def make_tree(**overrides):
    env = FakeEnvironment()
    root = Node(b"s0", np.array(0), 0, 0, previous_action=0, steps=0)
    cfg = SearchConfig(**{"simulations_per_move": 4, "seed": 2, **overrides})
    return SearchTree(root, FakePolicy(), env, cfg), env


def test_leaf_success_probability_is_backed_up_without_dense_reward():
    tree, _ = make_tree()
    assert tree.simulate()
    edge = tree.root.edges[0]
    assert edge.reward == 2.0
    assert edge.visits == 1
    assert edge.value_sum == 0.4


def test_backup_preserves_one_success_probability_without_sign_flip():
    tree, _ = make_tree()
    first, second = tree.root.edges[0], tree.root.edges[1]
    first.reward, second.reward = 2.0, -1.0
    tree.backup([first, second], leaf_value=0.4)
    assert second.value_sum == 0.4
    assert first.value_sum == 0.4


def test_terminal_success_backs_up_one():
    tree, _ = make_tree()
    tree.root.steps = 2
    assert tree.simulate()
    assert tree.root.edges[0].value_sum == 1.0


def test_root_visits_form_masked_policy_target_and_commit_reuses_child():
    tree, _ = make_tree()
    tree.search(4)
    old = tree.root
    target = tree.commit(sample=False)
    assert target.visits.sum() == 4
    assert np.isclose(target.probabilities.sum(), 1.0)
    assert target.chosen_action == int(np.argmax(target.visits))
    assert tree.root is old.edges[target.chosen_action].child
    assert tree.root.parent is None


def test_node_limit_stops_before_unbacked_expansion():
    tree, _ = make_tree(max_live_nodes=2)
    assert tree.simulate()
    assert not tree.simulate()
