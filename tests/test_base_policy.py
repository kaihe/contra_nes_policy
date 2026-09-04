"""The action-only base-policy contract from doc/0006."""

from __future__ import annotations

import torch

from contra_policy.dataset import shard_paths
from contra_policy.loss import BehaviorCloneLoss
from contra_policy.model import (PolicyConfig, build_policy,
                                 initialize_alphazero_policy, load_policy)
from contra_policy.train_bc import _require_family_counts


ENCODER = dict(image_size=64, hiddim=32, depth=4, minres=4, proj_ch=16,
               aux_size=8, head_depth=4, entity_classes=0)
CORE = dict(n_layer=1, n_head=4, n_kv_head=4, context=16,
            mlp_ratio=2.0, rope_theta=10000.0, dropout=0.0)


def _config(**overrides):
    values = dict(encoder=ENCODER, core=CORE, freeze_encoder=False, encode_chunk=8)
    values.update(overrides)
    return PolicyConfig(**values)


def test_action_only_policy_has_one_output_and_no_dead_heads():
    policy = build_policy(_config(aux_size=0, value_head=False)).eval()
    images = torch.randint(0, 256, (2, 3, 64, 64, 3), dtype=torch.uint8)
    goals = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)

    with torch.no_grad():
        out = policy(images, goals, torch.tensor([0, 1]))

    assert set(out) == {"pi_logits"}
    assert out["pi_logits"].shape == (2, 3, 21)
    assert not any(k.startswith(("aux_head.", "value_head."))
                   for k in policy.state_dict())


def test_null_goal_skips_goal_image_without_changing_output_shape():
    policy = build_policy(_config(aux_size=0, value_head=False,
                                  use_goal_image=False)).eval()
    images = torch.randint(0, 256, (2, 3, 64, 64, 3), dtype=torch.uint8)
    interaction = torch.tensor([4, 4])

    with torch.no_grad():
        a = policy(images, None, interaction)["pi_logits"]
        b = policy(images, torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
                   interaction)["pi_logits"]

    assert a.shape == (2, 3, 21)
    assert torch.equal(a, b)


def test_reduced_features_reproduce_the_encoder_projection_path():
    policy = build_policy(_config(aux_size=0, value_head=False,
                                  freeze_encoder=True, train_projection=True,
                                  use_goal_image=False)).eval()
    # The tiny test encoder's reduced map has 256 values; production uses 256x4x4.
    features = torch.randn(2, 3, 16, 4, 4)
    interaction = torch.tensor([4, 4])

    with torch.no_grad():
        got = policy.forward_reduced_features(features, interaction)["pi_logits"]
        frames = policy.encoder.token_ln(
            policy.encoder.proj(features.flatten(2).reshape(6, -1))).view(2, 3, -1)
        goal = policy.null_goal.unsqueeze(0).expand(2, -1)
        expected = policy._heads(frames, goal, interaction, None)["pi_logits"]

    assert torch.equal(got, expected)
    assert all(p.requires_grad for p in policy.encoder.proj.parameters())
    assert all(p.requires_grad for p in policy.encoder.token_ln.parameters())
    assert not any(p.requires_grad for p in policy.encoder.view_backbone.parameters())


def test_legacy_head_defaults_still_round_trip_strictly(tmp_path):
    policy = build_policy(_config())
    path = policy.save(str(tmp_path / "legacy.pt"))

    loaded = load_policy(path)

    assert loaded.value_head is not None
    assert loaded.aux_head is not None
    assert set(loaded.state_dict()) == set(policy.state_dict())


def test_alphazero_heads_have_the_declared_shapes():
    policy = build_policy(_config(aux_size=0, value_head=True, state_heads=True)).eval()
    images = torch.randint(0, 256, (2, 3, 64, 64, 3), dtype=torch.uint8)
    goals = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)

    with torch.no_grad():
        out = policy(images, goals, torch.tensor([0, 1]))

    assert set(out) == {"pi_logits", "vpred", "motion", "weapon_logits", "rapid_logit",
                        "progress_logit"}
    assert out["vpred"].shape == (2, 3)
    assert out["motion"].shape == (2, 3, 2)
    assert out["weapon_logits"].shape == (2, 3, 6)
    assert out["rapid_logit"].shape == (2, 3)
    assert out["progress_logit"].shape == (2, 3)


def test_alphazero_initialization_transfers_only_policy_weights(tmp_path):
    source = build_policy(_config(aux_size=0, value_head=False, state_heads=False))
    path = source.save(str(tmp_path / "gpt.pt"))

    first = initialize_alphazero_policy(path, seed=7)
    second = initialize_alphazero_policy(path, seed=7)

    assert torch.equal(first.pi_head.weight, source.pi_head.weight)
    assert torch.equal(first.core.blocks[0].attn.q.weight,
                       source.core.blocks[0].attn.q.weight)
    assert torch.equal(first.value_head.weight, second.value_head.weight)
    assert first.value_head is not None and first.motion_head is not None
    assert first.aux_head is None


def test_action_ce_can_suppress_all_diagnostic_metrics():
    objective = BehaviorCloneLoss(diagnostics=False)
    logits = torch.randn(2, 4, 21, requires_grad=True)
    batch = {"action": torch.randint(0, 21, (2, 4)),
             "mask": torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.float32)}

    loss, metrics = objective({"pi_logits": logits}, batch)
    loss.backward()

    assert set(metrics) == {"loss"}
    assert logits.grad is not None


def test_shard_override_discovers_all_numbered_shards(tmp_path):
    default = tmp_path / "default"
    boss = tmp_path / "boss-release"
    default.mkdir()
    boss.mkdir()
    (default / "kill-train-00000.tar").touch()
    for number in range(3):
        (boss / f"boss-train-{number:05d}.tar").touch()

    got = shard_paths(str(default), ("kill", "boss"), "train", {"boss": str(boss)})

    assert got == [str(default / "kill-train-00000.tar"),
                   str(boss / "boss-train-00000.tar"),
                   str(boss / "boss-train-00001.tar"),
                   str(boss / "boss-train-00002.tar")]


def test_expected_release_count_fails_closed():
    index = [{"family": "boss"}] * 665

    try:
        _require_family_counts(index, {"boss": 666}, "train")
    except ValueError as exc:
        assert "expected 666 episodes, resolved 665" in str(exc)
    else:
        raise AssertionError("an incomplete boss release was accepted")
