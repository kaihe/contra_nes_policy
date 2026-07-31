"""RL fine-tuning tests: the invariants that fail silently rather than crash.

Nothing here checks that PPO "works" — that is what a training curve is for. These
target the things that produce a plausible-looking curve while being wrong:

* an episode scored as a death on the step it actually completed the task;
* a budget-exhausted episode bootstrapped from the critic, inventing value where the
  objective says there is none;
* recurrent memory leaking from one episode into the next through a recycled slot;
* a minibatch that shuffles transitions out of temporal order and hands the recurrent
  core a memory belonging to a different trajectory;
* a held-out task reaching a rollout worker;
* the previous-action token arriving as anything other than the learned "unknown"
  embedding, which is the branch the 72.8% baseline was measured in;
* a weights file this repo can load and ``contra_nes_evaluation`` cannot.

Tests needing the emulator, the shards or the BC checkpoint skip cleanly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from contra_policy.dataset import shard_paths
from contra_policy.model import CrossViewContraRocket
from contra_policy.rl import checkpoint as ckpt_io
from contra_policy.rl.ppo import PPOConfig, PPOObjective, policy_loss, value_loss
from contra_policy.rl.rollout import (EpisodeCollector, RolloutActor, RolloutObservation,
                                      budget_for, claim_emulator, classify_step,
                                      release_emulator)
from contra_policy.rl.tasks import TaskCatalog, TaskSampler
from contra_policy.rl.trajectory import (Episode, build_chunk, compute_gae,
                                         compute_returns, iter_chunks, iter_minibatches,
                                         normalize_advantages, rollout_stats)

SHARD_DIR = os.path.expanduser("~/code/contra_nes_data/game_trace/hf")
TASK_ROOT = os.path.expanduser("~/code/contra_nes_data/game_trace/tasks")
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
FAMILIES = ("kill",)          # one family is enough and keeps the index build cheap

HAVE_SHARDS = all(os.path.exists(p) for p in shard_paths(SHARD_DIR, FAMILIES, "train")
                  + shard_paths(SHARD_DIR, FAMILIES, "val"))
HAVE_TASKS = os.path.isdir(os.path.join(TASK_ROOT, "kill"))
try:                                        # the emulator lives in contra_nes_data
    import util.replay                      # noqa: F401

    HAVE_EMULATOR = True
except Exception:                           # pragma: no cover
    HAVE_EMULATOR = False

needs_data = pytest.mark.skipif(not (HAVE_SHARDS and HAVE_TASKS),
                                reason="task .npz files or shards are not on this machine")
needs_emulator = pytest.mark.skipif(not HAVE_EMULATOR,
                                    reason="stable_retro / contra_nes_data not installed")

TINY_MODEL = dict(image_size=64, view_depth=8, mask_depth=4, minres=4,
                  view_backbone_ckpt=None, hiddim=64, num_heads=4, num_layers=2,
                  timesteps=8, mem_len=8, num_view_tokens=2, aux_size=16)
TINY_SEQ_LEN = TINY_MODEL["timesteps"]


def tiny_model(seed: int = 0) -> CrossViewContraRocket:
    torch.manual_seed(seed)
    return CrossViewContraRocket(**TINY_MODEL).eval()


def fake_episode(length: int, outcome: str = "success", *, family: str = "kill",
                 label: str = "sniper", size: int = 4, seed: int = 0,
                 terminal: bool = True, bootstrap: float = 0.0) -> Episode:
    """A synthetic trajectory whose observations encode their own step index.

    ``obs[t]`` is uniformly ``t % 256``, which is what makes the sequence-order tests
    able to assert that step ``t`` really arrived at position ``t``.
    """
    rng = np.random.default_rng(seed)
    rewards = np.zeros(length, dtype=np.float32)
    if outcome == "success" and length:
        rewards[-1] = 1.0
    obs = np.stack([np.full((size, size, 3), t % 256, dtype=np.uint8)
                    for t in range(length)])
    return Episode(
        family=family, label=label, uid=f"u{seed}", interaction=0,
        goal_image=np.zeros((size, size, 3), np.uint8),
        goal_mask=np.zeros((size, size), np.uint8),
        obs=obs,
        actions=rng.integers(0, 21, size=length).astype(np.int64),
        prev_actions=np.zeros(length, dtype=np.int64),
        logprobs=np.full(length, -1.0, dtype=np.float32),
        values=np.zeros(length, dtype=np.float32),
        rewards=rewards, outcome=outcome, terminal=terminal,
        bootstrap_value=bootstrap, budget=length, expert_steps=length,
        died=outcome == "death",
        goal_points=[[] for _ in range(length)],
        goal_visible=np.zeros(length, dtype=bool))


# ── terminal semantics ────────────────────────────────────────────────────────

def test_success_is_checked_before_death():
    # The step that clears a boss can also register a hit on the player. Scoring it as
    # a death would train the policy away from the winning move, and would disagree
    # with the evaluation harness, which resolves the tie the same way.
    done, outcome, died = classify_step(goal_reached=True, died=True, steps=5, budget=10)
    assert (done, outcome) == (True, "success")
    # ...and the death is still recorded, so the `died` rate stays honest.
    assert died is True


def test_death_ends_the_episode_when_no_goal_was_reached():
    assert classify_step(False, True, 5, 10)[:2] == (True, "death")


def test_budget_exhaustion_is_terminal():
    assert classify_step(False, False, 10, 10)[:2] == (True, "timeout")
    assert classify_step(False, False, 9, 10)[0] is False


def test_success_wins_over_budget_exhaustion_on_the_same_step():
    assert classify_step(True, False, 10, 10)[1] == "success"


def test_budget_matches_the_evaluation_harness_formula():
    class _Seg:
        actions = list(range(30))

    # contra_nes_evaluation: max(min_budget, ceil(budget_mult * expert_steps))
    assert budget_for(_Seg(), 2.0, 24) == 60
    _Seg.actions = list(range(5))
    assert budget_for(_Seg(), 2.0, 24) == 24


# ── returns, advantages, bootstrapping ────────────────────────────────────────

def test_undiscounted_binary_return_is_flat_over_a_successful_episode():
    # gamma = lambda = 1, reward only at the end, an untrained critic reporting 0:
    # every state in a win targets 1 and its advantage is 1.
    rewards = np.array([0, 0, 0, 1], dtype=np.float32)
    values = np.zeros(4, dtype=np.float32)
    adv, ret = compute_gae(rewards, values, gamma=1.0, lam=1.0, terminal=True)
    assert np.allclose(ret, 1.0)
    assert np.allclose(adv, 1.0)


def test_failed_episode_returns_zero_everywhere():
    adv, ret = compute_gae(np.zeros(6, np.float32), np.zeros(6, np.float32), 1.0, 1.0)
    assert np.allclose(ret, 0.0) and np.allclose(adv, 0.0)


def test_advantage_is_return_minus_value_at_gamma_lambda_one():
    rng = np.random.default_rng(0)
    values = rng.normal(size=12).astype(np.float32)
    rewards = np.zeros(12, dtype=np.float32)
    rewards[-1] = 1.0
    adv, ret = compute_gae(rewards, values, 1.0, 1.0)
    assert np.allclose(ret, 1.0, atol=1e-5)
    assert np.allclose(adv, 1.0 - values, atol=1e-5)


def test_gae_matches_a_hand_computed_discounted_trajectory():
    # delta_t = r_t + g*V(s_{t+1}) - V(s_t); A_t = delta_t + g*l*A_{t+1}
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    g, lam = 0.9, 0.8
    d2 = 1.0 + g * 0.0 - 0.75
    d1 = 0.0 + g * 0.75 - 0.25
    d0 = 0.0 + g * 0.25 - 0.5
    a2 = d2
    a1 = d1 + g * lam * a2
    a0 = d0 + g * lam * a1
    adv, ret = compute_gae(rewards, values, g, lam, terminal=True)
    assert np.allclose(adv, [a0, a1, a2], atol=1e-6)
    assert np.allclose(ret, np.array([a0, a1, a2]) + values, atol=1e-6)


@pytest.mark.parametrize("outcome", ["success", "death", "timeout"])
def test_true_terminals_never_bootstrap(outcome):
    # A terminal episode carries terminal=True, so a stale bootstrap value on it must
    # have no effect whatsoever — success, death and running out of the task's real
    # evaluation budget are all worth exactly what the reward says.
    ep = fake_episode(5, outcome, terminal=True, bootstrap=99.0)
    adv_a, ret_a = compute_gae(ep.rewards, ep.values, 1.0, 1.0,
                               terminal=True, bootstrap_value=99.0)
    adv_b, ret_b = compute_gae(ep.rewards, ep.values, 1.0, 1.0,
                               terminal=True, bootstrap_value=0.0)
    assert np.allclose(adv_a, adv_b) and np.allclose(ret_a, ret_b)
    expected = 1.0 if outcome == "success" else 0.0
    assert np.allclose(ret_a, expected)


def test_artificial_chunk_boundary_does_bootstrap():
    # The only cut that may bootstrap: the collector gave up on an over-long episode,
    # so the tail is unknown rather than worth zero.
    rewards = np.zeros(4, dtype=np.float32)
    values = np.zeros(4, dtype=np.float32)
    adv, ret = compute_gae(rewards, values, 1.0, 1.0, terminal=False, bootstrap_value=0.7)
    assert np.allclose(ret, 0.7)
    adv0, ret0 = compute_gae(rewards, values, 1.0, 1.0, terminal=True)
    assert np.allclose(ret0, 0.0)          # ...and a true terminal does not


def test_advantage_normalisation_is_over_the_whole_batch():
    eps = [fake_episode(4, "success", seed=1), fake_episode(6, "death", seed=2)]
    compute_returns(eps, 1.0, 1.0)
    normalize_advantages(eps)
    flat = np.concatenate([e.advantages for e in eps])
    assert abs(flat.mean()) < 1e-5 and abs(flat.std() - 1.0) < 1e-4


# ── PPO objective ─────────────────────────────────────────────────────────────

def test_policy_loss_clips_a_positive_advantage_above_the_ratio_ceiling():
    # ratio = e^{0.5} ≈ 1.649, well past 1 + clip. With a positive advantage the
    # objective must take the clipped branch, so the loss is -(1 + clip) * A and the
    # gradient through the ratio is zero.
    old = torch.zeros(1, 1)
    logprob = torch.full((1, 1), 0.5, requires_grad=True)
    adv = torch.ones(1, 1)
    mask = torch.ones(1, 1)
    loss, metrics = policy_loss(logprob, old, adv, mask, clip_ratio=0.1)
    assert torch.allclose(loss, torch.tensor(-1.1))
    assert float(metrics["clip_frac"]) == 1.0
    loss.backward()
    assert float(logprob.grad.abs().sum()) == 0.0


def test_policy_loss_is_unclipped_inside_the_trust_region():
    old = torch.zeros(1, 1)
    logprob = torch.full((1, 1), 0.05, requires_grad=True)
    loss, metrics = policy_loss(logprob, old, torch.ones(1, 1), torch.ones(1, 1), 0.1)
    assert torch.allclose(loss, -torch.exp(torch.tensor(0.05)))
    assert float(metrics["clip_frac"]) == 0.0
    loss.backward()
    assert float(logprob.grad.abs().sum()) > 0.0


def test_negative_advantage_clips_on_the_other_side():
    # ratio far *below* 1 with a negative advantage: min() picks the clipped branch,
    # so the loss saturates at -(1 - clip) * A rather than running away.
    old = torch.zeros(1, 1)
    logprob = torch.full((1, 1), -0.5, requires_grad=True)
    loss, _ = policy_loss(logprob, old, -torch.ones(1, 1), torch.ones(1, 1), 0.1)
    assert torch.allclose(loss, torch.tensor(0.9))
    loss.backward()
    assert float(logprob.grad.abs().sum()) == 0.0


def test_masked_steps_contribute_nothing_to_the_policy_loss():
    logprob = torch.tensor([[0.3, 5.0]])
    old = torch.zeros(1, 2)
    adv = torch.tensor([[1.0, 1000.0]])
    full = policy_loss(logprob, old, adv, torch.ones(1, 2), 0.1)[0]
    masked = policy_loss(logprob, old, adv, torch.tensor([[1.0, 0.0]]), 0.1)[0]
    assert not torch.allclose(full, masked)
    only = policy_loss(logprob[:, :1], old[:, :1], adv[:, :1], torch.ones(1, 1), 0.1)[0]
    assert torch.allclose(masked, only)


def test_value_clipping_bounds_the_error_around_the_old_value():
    vpred = torch.tensor([[10.0]])
    old = torch.zeros(1, 1)
    returns = torch.ones(1, 1)
    unclipped = value_loss(vpred, old, returns, torch.ones(1, 1), clip=0.0)
    clipped = value_loss(vpred, old, returns, torch.ones(1, 1), clip=0.2)
    assert float(unclipped) == pytest.approx(0.5 * 81.0)
    # max(unclipped, clipped) keeps the pessimistic branch: it never *hides* a large
    # error, it only stops the update from chasing it in one step.
    assert float(clipped) == pytest.approx(0.5 * 81.0)
    # ...and when the prediction moves the wrong way inside the region, the clipped
    # branch is the one that binds.
    close = value_loss(torch.tensor([[0.9]]), old, returns, torch.ones(1, 1), clip=0.2)
    assert float(close) == pytest.approx(0.5 * (0.2 - 1.0) ** 2)


def test_objective_metric_sums_scale_with_the_valid_step_count():
    cfg = PPOConfig(entropy_coef=0.0, value_coef=0.0)
    obj = PPOObjective(cfg)
    b, t = 2, 4
    latents = {"pi_logits": torch.zeros(b, t, 21), "vpred": torch.zeros(b, t, 1)}
    batch = {"mask": torch.ones(b, t), "action": torch.zeros(b, t, dtype=torch.long),
             "old_logprob": torch.full((b, t), -np.log(21.0)),
             "old_value": torch.zeros(b, t), "advantage": torch.ones(b, t),
             "returns": torch.zeros(b, t)}
    loss, metrics, count = obj(latents, batch)
    assert float(count) == b * t
    assert float(metrics["ratio"]) == pytest.approx(b * t, rel=1e-4)
    assert float(loss) == pytest.approx(-b * t, rel=1e-4)


# ── recurrent sequence handling ───────────────────────────────────────────────

def test_minibatches_shuffle_episodes_not_transitions():
    eps = [fake_episode(6, seed=i) for i in range(8)]
    rng = np.random.default_rng(0)
    seen = [ep for mb in iter_minibatches(eps, 3, rng) for ep in mb]
    assert len(seen) == len(eps)
    assert {id(e) for e in seen} == {id(e) for e in eps}


def test_chunks_are_consecutive_and_start_at_zero():
    eps = [fake_episode(5), fake_episode(17)]
    ranges = list(iter_chunks(eps, 8))
    assert ranges == [(0, 8), (8, 16), (16, 17)]
    assert ranges[0][0] == 0
    assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))


def test_chunk_tensors_carry_steps_in_temporal_order():
    # obs[t] is filled with t, so the frames coming back out of build_chunk say
    # exactly which step landed at which position. Any transition-level shuffle would
    # show up here immediately.
    eps = [fake_episode(10, seed=0), fake_episode(6, seed=1)]
    compute_returns(eps, 1.0, 1.0)
    dev = torch.device("cpu")
    for lo, hi in iter_chunks(eps, 4):
        batch = build_chunk(eps, lo, hi, device=dev)
        image = batch["model"]["image"].numpy()
        for i, ep in enumerate(eps):
            n = max(0, min(hi, len(ep)) - lo)
            for k in range(n):
                assert image[i, k].min() == image[i, k].max() == (lo + k) % 256
            assert float(batch["mask"][i, :n].sum()) == n
            assert float(batch["mask"][i, n:].sum()) == 0.0


def test_padding_past_a_short_episode_is_masked_off():
    eps = [fake_episode(3), fake_episode(9)]
    compute_returns(eps, 1.0, 1.0)
    batch = build_chunk(eps, 8, 9, device=torch.device("cpu"))
    assert float(batch["mask"][0].sum()) == 0.0        # episode 0 ended at step 3
    assert float(batch["mask"][1].sum()) == 1.0


def test_first_flag_is_set_only_on_the_opening_chunk():
    eps = [fake_episode(20)]
    compute_returns(eps, 1.0, 1.0)
    dev = torch.device("cpu")
    opening = build_chunk(eps, 0, 8, device=dev, first=True)
    later = build_chunk(eps, 8, 16, device=dev, first=False)
    assert bool(opening["model"]["first"][0, 0]) is True
    assert not bool(opening["model"]["first"][0, 1:].any())
    assert "first" not in later["model"]


def test_chunked_replay_reproduces_a_single_long_forward():
    # The optimiser replays an episode as ordered chunks with carried memory. That has
    # to be the same computation as one long forward, or the log-probs PPO recomputes
    # are not the log-probs of the policy that acted.
    model = tiny_model()
    b, t, s = 1, TINY_SEQ_LEN, TINY_MODEL["image_size"]
    torch.manual_seed(1)
    full_input = {
        "image": torch.randint(0, 255, (b, t, s, s, 3), dtype=torch.uint8),
        "cross_view": {
            "cross_view_image": torch.randint(0, 255, (b, s, s, 3), dtype=torch.uint8),
            "cross_view_obj_mask": torch.randint(0, 255, (b, s, s), dtype=torch.uint8),
            "cross_view_obj_id": torch.zeros(b, t, dtype=torch.long),
        },
        "prev_action": torch.zeros(b, t, dtype=torch.long),
        "prev_action_dropout": torch.zeros(b, t),
    }
    with torch.no_grad():
        full, _ = model(full_input, None)
        memory = None
        pieces = []
        half = t // 2
        for lo, hi in ((0, half), (half, t)):
            part = dict(full_input)
            part["image"] = full_input["image"][:, lo:hi]
            part["prev_action"] = full_input["prev_action"][:, lo:hi]
            part["prev_action_dropout"] = full_input["prev_action_dropout"][:, lo:hi]
            part["cross_view"] = dict(full_input["cross_view"])
            part["cross_view"]["cross_view_obj_id"] = \
                full_input["cross_view"]["cross_view_obj_id"][:, lo:hi]
            out, memory = model(part, memory)
            memory = [m.detach() for m in memory]
            pieces.append(out["pi_logits"])
    assert torch.allclose(full["pi_logits"], torch.cat(pieces, dim=1), atol=1e-4)


# ── the "unknown" previous-action embedding ───────────────────────────────────

def test_chunk_always_selects_the_unknown_prev_action_embedding():
    eps = [fake_episode(5)]
    eps[0].prev_actions[:] = 7            # a real action history is stored...
    compute_returns(eps, 1.0, 1.0)
    batch = build_chunk(eps, 0, 5, device=torch.device("cpu"))
    # ...and is never used: dropout=0 routes every step to the learned "unknown"
    # embedding, which is the configuration the 72.8% baseline was measured in.
    assert float(batch["model"]["prev_action_dropout"].abs().sum()) == 0.0


def test_actor_always_selects_the_unknown_prev_action_embedding():
    model = tiny_model()
    actor = RolloutActor(model, 2, device=torch.device("cpu"), seed=0, precision="fp32")
    s = TINY_MODEL["image_size"]
    obs = RolloutObservation(
        image=np.zeros((2, s, s, 3), np.uint8), goal_image=np.zeros((2, s, s, 3), np.uint8),
        goal_mask=np.zeros((2, s, s), np.uint8), interaction=np.zeros(2, np.int64),
        prev_action=np.full(2, 13, np.int64), active=np.ones(2, bool))
    model_input = actor._model_input(obs)
    assert float(model_input["prev_action_dropout"].abs().sum()) == 0.0
    assert model_input["prev_action_dropout"].shape == (2, 1)


def test_dropout_zero_makes_the_prev_action_value_irrelevant():
    # The strong form of the invariant: with the unknown embedding selected, two
    # different action histories must produce identical logits.
    model = tiny_model()
    b, t, s = 1, 4, TINY_MODEL["image_size"]
    base = {
        "image": torch.zeros(b, t, s, s, 3, dtype=torch.uint8),
        "cross_view": {
            "cross_view_image": torch.zeros(b, s, s, 3, dtype=torch.uint8),
            "cross_view_obj_mask": torch.zeros(b, s, s, dtype=torch.uint8),
            "cross_view_obj_id": torch.zeros(b, t, dtype=torch.long),
        },
        "prev_action_dropout": torch.zeros(b, t),
    }
    with torch.no_grad():
        a, _ = model({**base, "prev_action": torch.zeros(b, t, dtype=torch.long)}, None)
        c, _ = model({**base, "prev_action": torch.full((b, t), 20, dtype=torch.long)}, None)
    assert torch.allclose(a["pi_logits"], c["pi_logits"])


# ── recurrent memory across episode boundaries ────────────────────────────────

def test_actor_reset_clears_only_the_named_slots():
    model = tiny_model()
    actor = RolloutActor(model, 4, device=torch.device("cpu"), seed=0, precision="fp32")
    for tensor in actor.memory:
        if tensor.dtype.is_floating_point:
            tensor.copy_(torch.randn_like(tensor))
        else:
            tensor.fill_(True)
    actor.reset([1, 3])
    fresh = model.recurrent.initial_state(1)
    for tensor, new in zip(actor.memory, fresh):
        assert torch.equal(tensor[1], new[0])
        assert torch.equal(tensor[3], new[0])
        assert not torch.equal(tensor[0], new[0])      # untouched slots keep their state


# ── one emulator per process ──────────────────────────────────────────────────

def test_a_process_may_claim_only_one_emulator():
    release_emulator()
    claim_emulator("first")
    try:
        with pytest.raises(RuntimeError, match="one instance per process"):
            claim_emulator("second")
    finally:
        release_emulator()
    claim_emulator("after release")        # releasing makes the slot reusable
    release_emulator()


# ── the training split, and only the training split ───────────────────────────

@needs_data
def test_catalog_holds_only_the_requested_split():
    train = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=FAMILIES,
                        split="train", image_size=64, cache_dir=CACHE, verbose=False)
    assert train.tasks and all(t.split == "train" for t in train.tasks)
    train.assert_split("train")
    train.close()


@needs_data
def test_no_validation_task_can_enter_an_rl_worker():
    train = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=FAMILIES,
                        split="train", image_size=64, cache_dir=CACHE, verbose=False)
    val = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=FAMILIES,
                      split="val", image_size=64, cache_dir=CACHE, verbose=False)
    assert val.tasks
    assert not ({t.key for t in train.tasks} & {t.key for t in val.tasks})
    # A val catalog is refused outright...
    with pytest.raises(RuntimeError, match="held-out"):
        val.assert_split("train")
    # ...and the sampler cannot reach one, because it only ever indexes its own tasks.
    sampler = TaskSampler(train, 0.7, 0.3, seed=0)
    assert all(sampler.sample().split == "train" for _ in range(200))
    train.close()
    val.close()


@needs_data
@needs_emulator
def test_collector_refuses_a_held_out_task():
    val = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=FAMILIES,
                      split="val", image_size=64, cache_dir=CACHE, verbose=False)
    train = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=FAMILIES,
                        split="train", image_size=64, cache_dir=CACHE, verbose=False)
    # Construction alone is refused for a val catalog.
    with pytest.raises(RuntimeError, match="held-out"):
        EpisodeCollector(tiny_model(), val, TaskSampler(train, 1.0, 0.0),
                         image_size=64, owner="refuse-val-catalog")
    val.close()
    train.close()


@needs_data
def test_sampling_mixture_lifts_the_weak_families():
    families = ("kill", "item", "traverse", "boss")
    if not all(os.path.exists(p) for p in shard_paths(SHARD_DIR, families, "train")):
        pytest.skip("not every family shard is on this machine")
    cat = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=families,
                      split="train", image_size=64, cache_dir=CACHE, verbose=False)
    natural = TaskSampler(cat, 1.0, 0.0, seed=0).expected_family_mix()
    mixed = TaskSampler(cat, 0.7, 0.3, seed=0).expected_family_mix()
    # The point of the mixture: roughly double the weak families without abandoning
    # the natural distribution or starving the strong one.
    assert mixed["boss"] > 1.5 * natural["boss"]
    assert mixed["item"] > 1.5 * natural["item"]
    assert mixed["traverse"] > 0.5 * natural["traverse"]
    assert abs(sum(mixed.values()) - 1.0) < 1e-6
    cat.close()


# ── collection against the real emulator ──────────────────────────────────────

@pytest.fixture
def train_catalog():
    if not (HAVE_SHARDS and HAVE_TASKS):
        pytest.skip("task .npz files or shards are not on this machine")
    cat = TaskCatalog(task_root=TASK_ROOT, shard_dir=SHARD_DIR, families=FAMILIES,
                      split="train", image_size=64, cache_dir=CACHE, verbose=False)
    yield cat
    cat.close()


@needs_emulator
def test_frozen_policy_collection_does_not_mutate_the_weights(train_catalog):
    model = tiny_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sampler = TaskSampler(train_catalog, 1.0, 0.0, seed=0)
    with EpisodeCollector(model, train_catalog, sampler, batch_size=2, image_size=64,
                          min_budget=6, budget_mult=0.05,
                          owner="test-frozen") as col:
        episodes = col.collect(min_steps=1, min_episodes=2)
    assert episodes
    after = model.state_dict()
    for k, v in before.items():
        assert torch.equal(v, after[k]), f"{k} moved during collection"
    assert all(p.grad is None for p in model.parameters())


@needs_emulator
def test_budget_exhaustion_produces_a_terminal_timeout(train_catalog):
    model = tiny_model()
    sampler = TaskSampler(train_catalog, 1.0, 0.0, seed=0)
    # A random tiny model will not complete anything; a 6-step budget guarantees it.
    with EpisodeCollector(model, train_catalog, sampler, batch_size=2, image_size=64,
                          min_budget=6, budget_mult=0.01, owner="test-timeout") as col:
        episodes = col.collect(min_steps=1, min_episodes=4)
    timeouts = [e for e in episodes if e.outcome == "timeout"]
    assert timeouts, "expected at least one budget-exhausted episode"
    for ep in timeouts:
        assert ep.terminal is True          # a real budget is a terminal, not a cut
        assert len(ep) == ep.budget == 6
        assert float(ep.rewards.sum()) == 0.0
    compute_returns(timeouts, 1.0, 1.0)
    assert all(np.allclose(e.returns, 0.0) for e in timeouts)


@needs_emulator
def test_artificial_cut_is_not_terminal_and_carries_a_critic_value(train_catalog):
    model = tiny_model()
    sampler = TaskSampler(train_catalog, 1.0, 0.0, seed=0)
    with EpisodeCollector(model, train_catalog, sampler, batch_size=2, image_size=64,
                          min_budget=200, budget_mult=5.0, max_episode_steps=4,
                          owner="test-truncate") as col:
        episodes = col.collect(min_steps=1, min_episodes=2)
    cut = [e for e in episodes if e.outcome == "truncated"]
    assert cut, "max_episode_steps should have cut at least one episode"
    for ep in cut:
        assert ep.terminal is False
        assert len(ep) == 4
        assert np.isfinite(ep.bootstrap_value)
    compute_returns(cut, 1.0, 1.0)
    # The tail comes from the critic, which is exactly what a true terminal must not do.
    assert all(np.allclose(e.returns, e.bootstrap_value, atol=1e-5) for e in cut)


@needs_emulator
def test_memory_resets_between_episodes_in_a_recycled_slot(train_catalog):
    # One slot, the same task twice, greedy actions. If the second episode inherited
    # the first one's attention memory it would act differently from step 0.
    model = tiny_model()
    task = train_catalog.tasks[0]
    sampler = TaskSampler(train_catalog, 1.0, 0.0, seed=0)
    with EpisodeCollector(model, train_catalog, sampler, batch_size=1, image_size=64,
                          temperature=0.0, min_budget=12, budget_mult=0.05,
                          precision="fp32", owner="test-memory-reset") as col:
        episodes = col.collect(min_steps=0, min_episodes=0, tasks=[task, task])
    assert len(episodes) == 2
    a, b = episodes
    assert a.uid == b.uid
    assert np.array_equal(a.actions, b.actions)
    assert np.allclose(a.logprobs, b.logprobs, atol=1e-5)
    assert np.array_equal(a.obs, b.obs)


@needs_emulator
def test_collected_episodes_carry_the_prompt_the_dataset_would_have_built(train_catalog):
    from contra_policy.goal import goal_mask
    import json

    model = tiny_model()
    task = train_catalog.tasks[0]
    sampler = TaskSampler(train_catalog, 1.0, 0.0, seed=0)
    with EpisodeCollector(model, train_catalog, sampler, batch_size=1, image_size=64,
                          min_budget=3, budget_mult=0.01, owner="test-prompt") as col:
        ep = col.collect(0, 0, tasks=[task])[0]
    meta = json.loads(train_catalog._read_member(task, "json"))
    expected = (goal_mask(meta["goal_points"], 64, 12.0) * 255.0).round().astype(np.uint8)
    assert np.array_equal(ep.goal_mask, expected)
    assert ep.interaction >= 0
    assert ep.goal_image.shape == (64, 64, 3)


@needs_emulator
def test_rollout_stats_report_episodes_and_transitions_per_family(train_catalog):
    eps = [fake_episode(10, "success", family="boss", label="boss_level1", seed=1),
           fake_episode(2, "death", family="item", label="pick_laser", seed=2)]
    compute_returns(eps, 1.0, 1.0)
    stats = rollout_stats(eps)
    # A family can be half the episode starts and most of the transitions; both are
    # reported because they are what the sampling mixture has to be judged on.
    assert stats["boss/episode_share"] == 0.5
    assert stats["boss/step_share"] == pytest.approx(10 / 12)
    assert stats["completion"] == 0.5
    assert stats["macro_completion"] == 0.5
    assert stats["boss.boss_level1/completion"] == 1.0


# ── checkpoints ───────────────────────────────────────────────────────────────

def test_weights_only_checkpoint_loads_through_the_policy_loader(tmp_path):
    model = tiny_model(seed=3)
    config = {"model": dict(TINY_MODEL), "seed": 0}
    path = ckpt_io.save_weights_only(str(tmp_path / "w.ckpt"), model, config)

    # These four lines are exactly what contra_nes_evaluation's CheckpointPolicy does:
    # read hyper_parameters.model, drop the pretrained-encoder path, rebuild, then
    # load the `policy.`-prefixed state dict strictly. Keeping the contract pinned
    # here means a weights file this repo writes is always one that harness can read.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert "hyper_parameters" in payload and "model" in payload["hyper_parameters"]
    assert all(k.startswith("policy.") for k in payload["state_dict"])
    rebuilt = CrossViewContraRocket(**ckpt_io.model_config_from_checkpoint(path))
    ckpt_io.load_policy_weights(rebuilt, path)
    for k, v in model.state_dict().items():
        assert torch.equal(v, rebuilt.state_dict()[k])


def test_weights_only_checkpoint_refuses_to_drop_the_model_block(tmp_path):
    with pytest.raises(ValueError, match="model"):
        ckpt_io.save_weights_only(str(tmp_path / "w.ckpt"), tiny_model(), {"seed": 0})


def test_resumable_checkpoint_round_trips_state_and_counters(tmp_path):
    model = tiny_model(seed=4)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)
    # Take a real step so the optimizer carries moment estimates worth restoring.
    loss = sum((p * p).sum() for p in params)
    loss.backward()
    opt.step()
    sched.step()
    counters = {"update": 17, "episodes": 512, "steps": 40000, "rollouts": 17}
    sampler = _StubSampler()
    path = ckpt_io.save_resumable(
        str(tmp_path / "rl.pt"), model=model, optimizer=opt, scheduler=sched,
        counters=counters, config={"model": dict(TINY_MODEL)}, sampler=sampler)

    fresh = tiny_model(seed=99)
    fresh_params = [p for p in fresh.parameters() if p.requires_grad]
    fresh_opt = torch.optim.AdamW(fresh_params, lr=1e-3)
    fresh_sched = torch.optim.lr_scheduler.LambdaLR(fresh_opt, lambda s: 0.5)
    restored_sampler = _StubSampler()
    got = ckpt_io.load_resumable(path, model=fresh, optimizer=fresh_opt,
                                 scheduler=fresh_sched, sampler=restored_sampler)
    # Counters continue rather than restart — that is the whole point of resuming.
    assert got == counters
    assert fresh_sched.last_epoch == sched.last_epoch
    assert restored_sampler.loaded == sampler.state()
    for k, v in model.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k])
    # The two samplers now draw the same stream.
    assert np.array_equal(sampler.rng.random(8), restored_sampler.rng.random(8))


class _StubSampler:
    """A TaskSampler's RNG contract, without needing the shards to build one."""

    def __init__(self):
        self.rng = np.random.default_rng(11)
        self.rng.random(5)
        self.loaded = None

    def state(self):
        return {"rng": self.rng.bit_generator.state}

    def load_state(self, state):
        self.loaded = state
        self.rng.bit_generator.state = state["rng"]


@pytest.mark.skipif(
    not os.path.exists(os.path.expanduser(
        "~/code/contra_nes_policy/runs/2026-07-28/18-01-29/weights/"
        "weight-epoch=18-step=30000.ckpt")),
    reason="the BC initialisation checkpoint is not on this machine")
def test_bc_checkpoint_supplies_the_architecture():
    path = os.path.expanduser("~/code/contra_nes_policy/runs/2026-07-28/18-01-29/"
                              "weights/weight-epoch=18-step=30000.ckpt")
    cfg = ckpt_io.model_config_from_checkpoint(path)
    # Read from the checkpoint, never retyped in config_rl.yaml: the architecture has
    # to match the weights, and a config file is the wrong place for that invariant.
    assert cfg["view_backbone_ckpt"] is None
    assert cfg["use_prev_action"] is True     # the unknown-embedding branch needs it
    model = CrossViewContraRocket(**cfg)
    ckpt_io.load_policy_weights(model, path)


# ── worker lifetime ───────────────────────────────────────────────────────────
# A collector worker holds a CUDA context, a copy of the policy and an emulator —
# several GB. `daemon=True` only reaps it on the parent's *clean* exit, because it is
# implemented by an atexit hook, so SIGTERM (what `timeout` sends) or SIGKILL used to
# leak the whole set. On a 20 GB WSL VM two leaked sets are enough to drive the box
# into swap, which is how the first multi-worker benchmark took the machine down.

_PDEATHSIG_PARENT = """
import os, sys, time
import torch.multiprocessing as mp
sys.path.insert(0, {src!r})
from contra_policy.rl.workers import _die_with_parent

def child(parent_pid, q):
    _die_with_parent(parent_pid)
    q.put(os.getpid())
    time.sleep(300)

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(os.getpid(), q), daemon=True)
    p.start()
    print(q.get(timeout=120), flush=True)
    time.sleep(300)
"""


def test_a_worker_does_not_outlive_a_sigkilled_parent(tmp_path):
    import subprocess
    import time

    src = os.path.join(os.path.dirname(__file__), "..", "src")
    script = tmp_path / "pdeathsig_parent.py"
    script.write_text(_PDEATHSIG_PARENT.format(src=os.path.abspath(src)))

    parent = subprocess.Popen([os.sys.executable, str(script)], stdout=subprocess.PIPE,
                              text=True)
    try:
        child_pid = int(parent.stdout.readline().strip())
        assert _alive(child_pid), "the child never started"

        # SIGKILL: no finally, no atexit, nothing the parent can do on its way out.
        # Only the kernel's PR_SET_PDEATHSIG can reap the child from here.
        parent.kill()
        parent.wait(timeout=30)

        deadline = time.time() + 15
        while time.time() < deadline and _alive(child_pid):
            time.sleep(0.2)
        assert not _alive(child_pid), (
            f"worker {child_pid} outlived its SIGKILLed parent — every such orphan "
            f"holds a CUDA context and a policy replica until the box is rebooted")
    finally:
        for pid in (parent.pid, locals().get("child_pid")):
            if pid:
                try:
                    os.kill(pid, 9)
                except (OSError, TypeError):
                    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                          # pragma: no cover
        return True
    # A killed child stays as a zombie until reaped; its parent is dead, so init reaps
    # it, but check the state rather than race that.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(") ", 1)[1].split()[0] != "Z"
    except OSError:                                  # pragma: no cover
        return False


# ── host-memory preflight ─────────────────────────────────────────────────────

class _Cfg(dict):
    """Attribute access over nested dicts, like the OmegaConf node the trainer sees."""

    def __getattr__(self, k):
        v = self[k]
        return _Cfg(v) if isinstance(v, dict) else v


def _mem_args(**over):
    cfg = _Cfg({"host_ram_budget_gb": 12.0,
                "rollout": {"num_workers": 0, "steps": 2048},
                "critic_warmup": {"steps": 2048},
                "ppo": {"bc_kl_coef": 0.0}})
    cfg["rollout"] = _Cfg({**cfg["rollout"], **over.pop("rollout", {})})
    cfg.update(over)
    return cfg


def test_memory_estimate_matches_the_measured_peaks():
    from contra_policy.rl.trainer import estimate_peak_host_gb

    # Measured on this box with tools/rss_guard.py (peak group PSS), two-update smoke
    # at steps=2048, batch_size=16. The estimator is what stands between a bad config
    # and a swapping VM, so it has to stay anchored to real numbers.
    assert estimate_peak_host_gb(_mem_args(), 256) == pytest.approx(3.54, abs=0.4)
    assert estimate_peak_host_gb(
        _mem_args(rollout={"num_workers": 2}), 256) == pytest.approx(8.83, abs=0.4)


def test_preflight_refuses_a_config_that_would_swap_the_box():
    from contra_policy.rl.trainer import preflight_host_memory

    # 4 workers measured >15 GB before the first update finished — the configuration
    # that took the VM down. It must not be reachable by editing one number.
    with pytest.raises(MemoryError, match="host_ram_budget_gb|available right now"):
        preflight_host_memory(_mem_args(rollout={"num_workers": 4}), 256)


def test_preflight_can_be_disabled_but_is_not_by_default():
    from contra_policy.rl.trainer import preflight_host_memory

    args = _mem_args(rollout={"num_workers": 4})
    args["host_ram_budget_gb"] = 0.0
    assert preflight_host_memory(args, 256) > 12      # returns the estimate, no raise
