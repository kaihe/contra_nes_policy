"""The action-only base-policy contract from doc/0006."""

from __future__ import annotations

import torch

from contra_policy.dataset import shard_paths
from contra_policy.loss import BehaviorCloneLoss
from contra_policy.model import PolicyConfig, build_policy, load_policy
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


def test_legacy_head_defaults_still_round_trip_strictly(tmp_path):
    policy = build_policy(_config())
    path = policy.save(str(tmp_path / "legacy.pt"))

    loaded = load_policy(path)

    assert loaded.value_head is not None
    assert loaded.aux_head is not None
    assert set(loaded.state_dict()) == set(policy.state_dict())


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
