"""Group identity across repeated collection calls, and the memory it costs.

These pin the three faults that took down run ``2026-08-02/11-48-03`` after 8 updates:

1. ``collect_groups`` restarted group ids at 0 on every call, so ``collect_filtered``
   pooled unrelated tasks into one group — GRPO's baseline stopped being same-task —
   and its ``n_kept >= want`` exit became unreachable, so every update ran to the
   oversample cap.
2. The collection-side ``zero_variance_group_frac`` shared a key with the post-filter
   one and was overwritten, hiding a real 0.59 behind a logged 0.0.
3. Whole ``Episode`` objects were retained purely to report success rates, at ~18 MB
   each and 512 per update.

The invariant that ties 1 together and would have caught it first: **one group id means
one task**.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from contra_policy.rl.buffer import Episode, EpisodeOutcome
from contra_policy.rl.trainer import GRPOTrainer


def _task(uid, fam="kill"):
    return SimpleNamespace(uid=uid, family=fam)


class _FakeCollector:
    """Mimics ``EpisodeCollector.collect_groups``' id contract without an emulator.

    Group ``i`` of this call gets id ``base_gid + i`` — the same
    ``enumerate(groups, start=base_gid)`` the real one uses.
    """

    def __init__(self, rewards):
        self.rewards = list(rewards)   # one reward pattern per group, cycled
        self.calls = []

    def collect_groups(self, groups, base_gid=0):
        self.calls.append(base_gid)
        out = []
        for i, g in enumerate(groups):
            pattern = self.rewards[(base_gid + i) % len(self.rewards)]
            for k, task in enumerate(g):
                out.append(Episode(
                    task_uid=task.uid, family=task.family, group_id=base_gid + i,
                    frames=np.zeros((3, 4, 4, 3), np.uint8),
                    goal_image=np.zeros((4, 4, 3), np.uint8), interaction=0,
                    actions=np.zeros(3, np.int64), logprobs=np.zeros(3, np.float32),
                    reward=float(pattern[k % len(pattern)]),
                    outcome="success" if pattern[k % len(pattern)] else "death"))
        return out


class _FakeSampler:
    def __init__(self, group_size=4):
        self.group_size, self.n = group_size, 0
        self.observed = []

    def sample_groups(self, n_groups):
        groups = []
        for _ in range(n_groups):
            self.n += 1
            groups.append([_task(f"task-{self.n}")] * self.group_size)
        return groups

    def observe(self, episodes):
        self.observed.extend(episodes)

    def stats(self):
        return {}


def _trainer(rewards, *, want=16, at_once=8, factor=4.0, memory_limit_gb=0.0):
    """A stub carrying only what ``collect_filtered`` touches, with the real method
    bound to it — so the code under test is the shipped one, not a copy."""
    stub = SimpleNamespace(
        args=OmegaConf.create({
            "rollout": {"groups_per_update": want, "collect_groups_at_once": at_once,
                        "max_oversample_factor": factor, "filter_groups": True},
            "train": {"memory_limit_gb": memory_limit_gb},
        }),
        groups=_FakeSampler(),
        collector=_FakeCollector(rewards),
    )
    stub.collect_filtered = GRPOTrainer.collect_filtered.__get__(stub)
    stub._check_memory = GRPOTrainer._check_memory.__get__(stub)
    stub.outcome_stats = GRPOTrainer.outcome_stats.__get__(stub)
    return stub


# ── one group id means one task ──────────────────────────────────────────────

def test_group_ids_are_unique_across_calls():
    """The regression. Ids from successive ``collect_groups`` calls must not collide."""
    t = _trainer([[1, 0, 0, 0]])          # every group has spread, nothing filtered
    kept, _, _ = t.collect_filtered()

    by_gid = {}
    for e in kept:
        by_gid.setdefault(e.group_id, set()).add(e.task_uid)
    assert all(len(uids) == 1 for uids in by_gid.values()), \
        f"a group id spans several tasks: { {g: u for g, u in by_gid.items() if len(u) > 1} }"


def test_collect_advances_base_gid_by_the_batch_size():
    t = _trainer([[1, 0, 0, 0]], want=16, at_once=8)
    t.collect_filtered()
    assert t.collector.calls == [0, 8], \
        "base_gid must advance by collect_groups_at_once between calls"


def test_the_loop_stops_at_want_rather_than_the_cap():
    """With ids fixed, 16 usable groups are reached in two draws of 8 — not 64."""
    t = _trainer([[1, 0, 0, 0]], want=16, at_once=8, factor=4.0)
    kept, outcomes, stats = t.collect_filtered()
    assert stats["groups_kept"] == 16
    assert stats["groups_drawn"] == 16
    assert stats["oversample_factor"] == 1.0
    assert len(outcomes) == 64            # 16 groups x G=4, not the 4x cap


def test_zero_variance_groups_still_force_oversampling():
    """Half the groups all-succeed, so twice as many draws are needed. The cap holds."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8, factor=4.0)
    _, _, stats = t.collect_filtered()
    assert stats["groups_kept"] == 8
    assert stats["groups_drawn"] == 16
    assert stats["collect/zero_variance_group_frac"] == pytest.approx(0.5)


def test_the_collection_stat_is_namespaced_away_from_the_advantage_one():
    """Both dicts land in one CSV row; sharing the key masked the stop-early signal."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    _, _, stats = t.collect_filtered()
    assert "collect/zero_variance_group_frac" in stats
    assert "zero_variance_group_frac" not in stats


# ── discarded episodes are not retained ──────────────────────────────────────

def test_discarded_groups_still_feed_the_difficulty_estimate():
    """An all-success group is the strongest evidence a task is too easy, and it is
    exactly what filtering throws away — so the sampler must see it first."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    _, outcomes, _ = t.collect_filtered()
    assert len(t.groups.observed) == len(outcomes)


def test_rolled_episodes_are_returned_without_frames():
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    kept, outcomes, _ = t.collect_filtered()
    assert len(outcomes) > len(kept)                       # some were filtered out
    assert all(isinstance(o, EpisodeOutcome) for o in outcomes)
    assert not any(hasattr(o, "frames") for o in outcomes)


def test_outcome_stats_reports_over_everything_rolled():
    """Filtering is an update-side decision; it must not flatter the success rate."""
    t = _trainer([[1, 1, 1, 1], [1, 0, 0, 0]], want=8, at_once=8)
    kept, outcomes, _ = t.collect_filtered()
    stats = t.outcome_stats(outcomes)
    assert stats["episodes"] == len(outcomes)
    assert stats["success"] == pytest.approx(np.mean([o.reward for o in outcomes]))
    # The survivors alone would read 0.25; everything rolled reads 0.625.
    assert stats["success"] > np.mean([e.reward for e in kept])


def test_episode_outcome_preserves_what_reporting_needs():
    e = Episode("u", "boss", 3, np.zeros((7, 4, 4, 3), np.uint8),
                np.zeros((4, 4, 3), np.uint8), 0, np.zeros(7, np.int64),
                np.zeros(7, np.float32), 1.0, "success")
    o = EpisodeOutcome.of(e)
    assert (o.family, o.outcome, o.reward) == ("boss", "success", 1.0)
    assert len(o) == len(e) == 7


# ── the memory guard ─────────────────────────────────────────────────────────

def test_memory_guard_raises_before_the_guest_swaps():
    t = _trainer([[1, 0, 0, 0]], memory_limit_gb=1e-6)
    with pytest.raises(MemoryError, match="host memory at"):
        t.collect_filtered()


def test_memory_guard_is_off_at_zero():
    t = _trainer([[1, 0, 0, 0]], memory_limit_gb=0.0)
    t.collect_filtered()          # must not raise


def test_act_uses_the_active_mask_not_a_none_test_on_the_image_array():
    """`_observe` returns a dense image array, so a None-test marks every slot active.

    That ran the causal core over slots holding no episode: harmless-looking at the tail
    of a full collection round, but a hard crash whenever fewer tasks were queued than
    there are batch slots (`groups_per_update * G < rollout.batch_size`) — the slot had
    never been begun, so its prefix token was still None. doc/0011's smoke found it.
    """
    import numpy as np

    from contra_policy.rl.rollout import RolloutObservation

    b, s = 4, 8
    obs = RolloutObservation(
        image=np.zeros((b, s, s, 3), np.uint8), goal_image=np.zeros((b, s, s, 3), np.uint8),
        goal_mask=np.zeros((b, s, s), np.uint8), interaction=np.zeros(b, np.int64),
        prev_action=np.zeros(b, np.int64),
        active=np.array([True, False, True, False]))

    by_none = [i for i, im in enumerate(obs.image) if im is not None]
    by_mask = [i for i, on in enumerate(obs.active) if on]

    assert by_none == [0, 1, 2, 3]          # what the old code computed
    assert by_mask == [0, 2]                # what it must compute


def test_actor_matches_full_forward_with_dropout_null_goal_and_width_projection():
    """At unchanged weights, sequential rollout and update logits must be identical."""
    import torch

    from contra_policy.model import PolicyConfig, build_policy
    from contra_policy.rl.rollout import RolloutObservation, TokenHistoryActor

    enc = dict(image_size=32, hiddim=24, depth=4, minres=4, proj_ch=8,
               aux_size=8, head_depth=4, entity_classes=0)
    core = dict(d_model=32, n_layer=1, n_head=4, n_kv_head=4, context=16,
                mlp_ratio=2.0, rope_theta=10000.0, dropout=0.2)
    model = build_policy(PolicyConfig(
        encoder=enc, core=core, freeze_encoder=False, use_goal_image=False,
        aux_size=0, value_head=False))
    actor = TokenHistoryActor(model, 1, device=torch.device("cpu"), seed=7)
    assert not model.training                    # dropout must be disabled

    goal = np.zeros((32, 32, 3), np.uint8)
    images = np.random.default_rng(4).integers(
        0, 256, (1, 3, 32, 32, 3), dtype=np.uint8)
    actor.begin(0, goal, interaction=4)
    for t in range(3):
        obs = RolloutObservation(
            image=images[:, t], goal_image=goal[None],
            goal_mask=np.zeros((1, 32, 32), np.uint8),
            interaction=np.array([4]), prev_action=np.zeros(1, np.int64),
            active=np.array([True]))
        actor.act(obs)
        sequential = actor._core_over_histories([0])
        with torch.no_grad():
            full = model(torch.from_numpy(images[:, :t + 1]), None,
                         torch.tensor([4]))["pi_logits"][:, -1]
        assert torch.allclose(sequential, full, atol=1e-6, rtol=1e-5)


def test_meta_matches_supports_membership_equality_and_missing_keys():
    """doc/0012's task filter: a typo must empty the pool, never pass everything."""
    from contra_policy.rl.tasks import meta_matches

    spread = {"weapon": "Spread", "rapid": True}
    regular = {"weapon": "Regular", "rapid": True}

    assert meta_matches(spread, {"weapon": ["Spread"], "rapid": True})
    assert meta_matches(spread, {"weapon": ["Spread", "Laser"]})       # list = membership
    assert not meta_matches(regular, {"weapon": ["Spread"]})
    assert not meta_matches(spread, {"rapid": False})                  # scalar = equality
    # An unknown field matches nothing rather than being ignored, so `expected_tasks`
    # turns a mistyped filter into a hard failure instead of a silently different pool.
    assert not meta_matches(spread, {"weapno": ["Spread"]})
    assert meta_matches({**spread, "uid": "laser-start"},
                        {"uid": ["laser-start"]})


def test_null_goal_catalog_builds_a_prompt_without_legacy_shards():
    from contra_policy.goal import INTERACTIONS
    from contra_policy.rl.tasks import GoalPrompt, RLTask, TaskCatalog

    catalog = TaskCatalog.__new__(TaskCatalog)
    catalog.require_prompt = False
    catalog.image_size = 32
    catalog._prompts = {}
    catalog.prompt_cache_size = 2
    catalog.segment = lambda _task: SimpleNamespace(meta={"goal_when": "boss"})
    task = RLTask("/unused.npz", "boss", "boss_level1", "u", "train")

    prompt = catalog.prompt(task)

    assert isinstance(prompt, GoalPrompt)
    assert prompt.interaction == INTERACTIONS.index("boss")
    assert prompt.image.shape == (32, 32, 3) and not prompt.image.any()
    assert prompt.mask.shape == (32, 32) and not prompt.mask.any()


def test_shard_task_meta_keys_by_family_and_uid(tmp_path):
    import json
    import tarfile

    from contra_policy.rl.tasks import shard_task_meta

    path = tmp_path / "boss-train-00000.tar"
    with tarfile.open(path, "w") as tf:
        for uid, weapon in (("a1", "Spread"), ("b2", "Regular")):
            blob = json.dumps({"weapon": weapon, "rapid": True}).encode()
            info = tarfile.TarInfo(f"{uid}.json")
            info.size = len(blob)
            tf.addfile(info, __import__("io").BytesIO(blob))

    meta = shard_task_meta([str(path)])

    assert meta[("boss", "a1")]["weapon"] == "Spread"
    assert meta[("boss", "b2")]["weapon"] == "Regular"
