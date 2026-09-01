"""Correctness gates for the environment-independent design-0030 MCTS core."""

from __future__ import annotations

import numpy as np

from contra_policy.mcts import Node, SearchConfig, SearchTree, Terminal, Transition
from contra_policy.mcts.laser import target_record


class FakePolicy:
    num_actions = 3

    def encode(self, observation):
        return int(np.asarray(observation).item())

    def priors(self, frame_tokens):
        del frame_tokens
        return np.array([0.6, 0.3, 0.1])


class FakeEnvironment:
    """Action 0 wins after two steps; the others fail immediately."""

    def __init__(self):
        self.transitions = []

    def legal_mask(self, node):
        return np.array([True, True, node.steps == 0])

    def step(self, node, action_id):
        self.transitions.append((node.emu_state, action_id))
        step = node.steps + 1
        if action_id != 0:
            terminal = Terminal.DEATH
        elif step >= 2:
            terminal = Terminal.SUCCESS
        else:
            terminal = Terminal.ACTIVE
        state = f"{node.emu_state.decode()}-{action_id}".encode()
        return Transition(state, np.array(step), step, terminal)


def make_tree(**overrides):
    policy, environment = FakePolicy(), FakeEnvironment()
    root = Node(b"s0", 0, 0, previous_action=0, steps=0)
    cfg = SearchConfig(**{"simulations_per_move": 8, "seed": 3, **overrides})
    return SearchTree(root, policy, environment, cfg, model_id="fake"), environment


def test_expansion_masks_and_normalizes_policy_priors():
    tree, _ = make_tree()
    assert set(tree.root.edges) == {0, 1, 2}
    assert np.isclose(sum(edge.prior for edge in tree.root.edges.values()), 1.0)

    tree.simulate()  # action 0 creates the non-terminal child
    child = tree.root.edges[0].child
    assert set(child.edges) == {0, 1}
    assert np.isclose(child.edges[0].prior, 2 / 3)
    assert np.isclose(child.edges[1].prior, 1 / 3)


def test_one_simulation_adds_only_one_permanent_node_and_backs_up_win():
    tree, environment = make_tree()
    assert tree.simulate()
    assert tree.live_nodes == 2
    edge = tree.root.edges[0]
    assert edge.visits == 1
    assert edge.value_sum == 1.0
    assert edge.child is not None
    # One permanent transition plus at least one transient rollout transition.
    assert len(environment.transitions) >= 2
    assert all(child.child is None for child in edge.child.edges.values())


def test_backup_is_binary_and_has_no_two_player_sign_flip():
    tree, _ = make_tree()
    path = [tree.root.edges[0], tree.root.edges[1]]
    tree.backup(path, Terminal.SUCCESS)
    assert [(edge.visits, edge.value_sum) for edge in path] == [(1, 1.0), (1, 1.0)]
    assert [(edge.successes, edge.deaths, edge.timeouts) for edge in path] == [
        (1, 0, 0), (1, 0, 0)]

    tree.backup(path, Terminal.TIMEOUT)
    assert [(edge.visits, edge.value_sum) for edge in path] == [(2, 1.0), (2, 1.0)]
    assert [(edge.successes, edge.deaths, edge.timeouts) for edge in path] == [
        (1, 0, 1), (1, 0, 1)]


def test_root_visits_form_policy_target():
    tree, _ = make_tree()
    assert tree.search(6) == 6
    mask, visits, probabilities = tree.root_target()
    assert mask.tolist() == [True, True, True]
    assert visits.sum() == 6
    assert np.isclose(probabilities.sum(), 1.0)
    assert np.all(probabilities[~mask] == 0)


def test_commit_re_roots_and_preserves_only_selected_subtree_context():
    tree, _ = make_tree()
    tree.search(6)
    old_root = tree.root
    target = tree.commit()
    assert target.chosen_action == 0
    assert isinstance(target.chosen_action, int)
    assert np.isclose(target.priors.sum(), 1.0)
    assert np.array_equal(target.successes + target.deaths + target.timeouts,
                          target.visits)
    assert tree.root is old_root.edges[0].child
    assert tree.root.parent is None
    assert tree.root.incoming_action is None
    assert tree.committed_prefix == [old_root.frame_token]
    assert tree.context(tree.root) == [old_root.frame_token, tree.root.frame_token]
    assert tree.live_nodes == tree._count_nodes(tree.root)

    record = target_record(target, [2, 1])
    assert record["action_prefix"] == [2, 1]
    assert len(record["ppo_prior"]) == tree.policy.num_actions
    assert set(record["terminal_counts"]) == {"success", "death", "timeout"}
    totals = np.sum(list(record["terminal_counts"].values()), axis=0)
    assert np.array_equal(totals, record["visits"])


def test_same_action_at_different_nodes_has_independent_edge_statistics():
    tree, _ = make_tree()
    tree.simulate()
    root_edge = tree.root.edges[0]
    child_edge = root_edge.child.edges[0]
    assert root_edge is not child_edge
    assert root_edge.visits == 1
    assert child_edge.visits == 0


def test_node_limit_stops_before_unbacked_expansion():
    tree, _ = make_tree(max_live_nodes=2)
    assert tree.simulate()
    before = tree.completed_simulations
    assert not tree.simulate()
    assert tree.completed_simulations == before
