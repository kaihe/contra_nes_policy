"""Encoder tests: the invariants that would fail quietly rather than crash.

Since `doc/0002-symmetric-encoder.md` the encoder is **goal-agnostic** — one
`encode(image)` for an agent frame and a goal frame alike — and the only occupancy
target is the 4-class entity map. What has to be pinned:

* occupancy is decoded from the **token**, never from the conv map; otherwise the token
  could discard spatial structure and the premise of "one token per frame" goes
  untested;
* the encoder really is symmetric — no branch, no goal-only parameters;
* goal frames carry real entity targets via `goal_frame_idx`, so the design is
  symmetric in supervision and not only in architecture;
* a checkpoint rebuilds its own architecture, so config drift cannot silently load
  wrong weights;
* `flatten_window` keeps members aligned, and the entity metrics stay readable on maps
  that are 95-98% empty.

`heatmap_readout` is still tested here even though the encoder no longer calls it: it
is the policy's readout at stage B and `point_err_px` is pinned by
`contra_nes_evaluation`.

Tests needing the shards skip cleanly.
"""

from __future__ import annotations

import os

import pytest
import torch

from contra_encoder import EncoderConfig, build_encoder, load_pretrained_encoder
from contra_encoder.data import flatten_window
from contra_encoder.heads import HeatmapHead, ReconstructionHead, heatmap_readout

SHARD_DIR = os.path.expanduser("~/code/contra_nes_data/game_trace/hf")
BC_CKPT = os.path.expanduser("~/code/contra_nes_policy/runs/2026-07-28/18-01-29/"
                             "weights/weight-epoch=18-step=30000.ckpt")
# Small everywhere so the suite stays fast; image_size must remain minres*2^k.
SMALL = dict(image_size=64, hiddim=32, depth=4, minres=4, proj_ch=16,
             aux_size=32, head_depth=8, recon_depth=4)


def _enc(**over):
    return build_encoder(EncoderConfig(**{**SMALL, **over}))


def _img(b, size):
    return torch.randint(0, 255, (b, size, size, 3), dtype=torch.uint8)


# ── the premise: occupancy flows through the token ───────────────────────────

def test_occupancy_is_decoded_only_from_the_token():
    """No gradient path from the entity map to the trunk that bypasses the token.

    This is the experiment the rebuild rests on. If the map could be read off the conv
    features directly, the token would be free to discard sprite positions and the loss
    would never notice.
    """
    enc = _enc()
    x = _img(2, enc.cfg.image_size)
    token = enc.encode(x)

    enc.entity_head(token.detach()).sum().backward()
    leaked = [n for n, p in enc.view_backbone.named_parameters()
              if p.grad is not None and p.grad.abs().sum() > 0]
    assert not leaked, f"entity map reached the trunk without the token: {leaked}"

    # With the token attached it must reach the trunk, or the test above is vacuous.
    enc.zero_grad(set_to_none=True)
    enc(x)["entity_heatmap"].sum().backward()
    assert [n for n, p in enc.view_backbone.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0]


# ── symmetry ─────────────────────────────────────────────────────────────────

def test_encoder_has_no_goal_specific_parameters():
    """0002 deleted the mask trunk, the second projection and the conditioning layer.

    A goal image is a real episode frame with the target painted into its RGB, so it
    needs no special path — and the policy's attention is a better goal matcher than
    FiLM channel modulation ever was.
    """
    enc = _enc()
    for gone in ("mask_backbone", "goal_reduce", "goal_proj", "film",
                 "heatmap_head", "interaction"):
        assert not hasattr(enc, gone), f"{gone} should have been removed by 0002"


def test_encode_takes_one_image_and_is_the_same_function_for_both_kinds():
    enc = _enc().eval()
    s = enc.cfg.image_size
    frame, goal = _img(3, s), _img(3, s)
    with torch.no_grad():
        a, b = enc.encode(frame), enc.encode(goal)
    assert a.shape == b.shape == (3, enc.cfg.hiddim)
    # Same weights, no branch: encoding both together equals encoding them apart.
    with torch.no_grad():
        both = enc.encode(torch.cat([frame, goal], 0))
    assert torch.allclose(both[:3], a, atol=1e-5)
    assert torch.allclose(both[3:], b, atol=1e-5)


def test_forward_shapes_and_optional_heads():
    enc = _enc()
    b, A = 3, enc.cfg.aux_size
    out = enc(_img(b, enc.cfg.image_size))
    assert out["token"].shape == (b, enc.cfg.hiddim)
    assert out["entity_heatmap"].shape == (b, 4, A, A)
    assert "reconstruction" not in out           # 0002 §4: off pending the ablation

    r = _enc(reconstruct=True)
    o = r(_img(2, r.cfg.image_size))
    # Reconstruction comes back in the input's layout, so a caller can compare it to
    # image/255 without transposing.
    assert o["reconstruction"].shape == (2, r.cfg.image_size, r.cfg.image_size, 3)
    assert 0.0 <= float(o["reconstruction"].min())
    assert float(o["reconstruction"].max()) <= 1.0


def test_entity_head_can_be_disabled():
    enc = _enc(entity_classes=0)
    assert enc.entity_head is None
    assert "entity_heatmap" not in enc(_img(2, enc.cfg.image_size))


def test_the_conv_trunk_is_trainable_by_default():
    """The policy freezes it; this package exists to train it. With a frozen trunk the
    loss would reach only the projection layers."""
    assert EncoderConfig().freeze_view_backbone is False
    assert all(p.requires_grad for p in _enc().view_backbone.parameters())


# ── head shapes ──────────────────────────────────────────────────────────────

def test_heads_reject_a_grid_that_is_not_base_times_a_power_of_two():
    HeatmapHead(dim=16, grid=32, base=4)
    ReconstructionHead(dim=16, size=64, depth=4, base=4)
    with pytest.raises(ValueError, match="power of two"):
        HeatmapHead(dim=16, grid=30, base=4)
    with pytest.raises(ValueError, match="power of two"):
        ReconstructionHead(dim=16, size=60, depth=4, base=4)


# ── checkpoints rebuild their own architecture ───────────────────────────────

def test_checkpoint_round_trip_restores_architecture_and_weights(tmp_path):
    enc = _enc(proj_ch=8, reconstruct=True)
    path = str(tmp_path / "encoder.pt")
    enc.save(path)

    back = load_pretrained_encoder(path)
    assert back.cfg.to_dict() == enc.cfg.to_dict()
    enc.eval(), back.eval()
    x = _img(2, enc.cfg.image_size)
    with torch.no_grad():
        assert torch.equal(enc(x)["token"], back(x)["token"])


def test_freeze_on_load_disables_every_gradient(tmp_path):
    enc = _enc()
    path = str(tmp_path / "encoder.pt")
    enc.save(path)
    assert not any(p.requires_grad
                   for p in load_pretrained_encoder(path, freeze=True).parameters())


# ── the policy's readout, still pinned ───────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(BC_CKPT),
                    reason="the BC initialisation checkpoint is not on this machine")
def test_heatmap_readout_is_identical_to_the_policys():
    from contra_policy.model import CrossViewContraRocket
    from contra_policy.rl import checkpoint as ckpt_io

    policy = CrossViewContraRocket(**ckpt_io.model_config_from_checkpoint(BC_CKPT))
    torch.manual_seed(0)
    for shape in [(5, 1, 32, 32), (2, 7, 32, 32)]:
        heat = torch.randn(*shape) * 3
        p_ref, e_ref = policy.heatmap_readout(heat)
        p_new, e_new = heatmap_readout(heat)
        # Bit-identical, not close: point_err_px is pinned by contra_nes_evaluation.
        assert torch.equal(p_ref, p_new) and torch.equal(e_ref, e_new)


def test_soft_argmax_has_no_half_cell_offset():
    """`goal_mask` places a blob at cx = x_norm * A exactly, so the readout inverts that.
    A half-cell offset would bias every prediction by 0.5/A — about 3.7 px at A=32."""
    A = 32
    for (col, row) in [(0, 0), (7, 21), (31, 31)]:
        heat = torch.full((1, A, A), -30.0)
        heat[0, row, col] = 30.0
        point, _ = heatmap_readout(heat)
        assert point[0, 0].item() == pytest.approx(col / A, abs=1e-4)
        assert point[0, 1].item() == pytest.approx(row / A, abs=1e-4)


# ── entity metrics, on maps that are 95-98% empty ────────────────────────────

def _blob_target(n=3, A=32):
    t = torch.zeros(n, A, A)
    t[:, 10, 10] = 1.0
    t[:, 10, 11] = 0.5          # soft, as goal_mask renders it
    t[:, 20, 20] = 1.0
    return t


def test_soft_dice_peaks_at_one_on_a_soft_target():
    """The reason for the squared denominator: with `Σp + Σt` an exactly correct
    prediction scores Σt²/Σt — about 0.90 — so the ceiling would move with the blob."""
    from contra_encoder.train import soft_dice

    t = _blob_target()
    assert float(soft_dice(t, t)) == pytest.approx(1.0, abs=1e-6)
    assert float(soft_dice(torch.zeros_like(t), t)) == pytest.approx(0.0, abs=1e-6)


def test_both_metrics_are_zero_for_predicting_nothing():
    """The baseline plain MSE hides: an all-zero prediction already scores 0.002-0.007."""
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    z = torch.zeros_like(t)
    assert float(soft_dice(z, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(mse_skill(z, t)) == pytest.approx(0.0, abs=1e-6)


def test_mse_skill_distinguishes_wrong_place_from_predicting_nothing():
    """Where the two complement each other: dice cannot go negative, so it scores
    silence and confident error identically. A model hallucinating bullets is worse
    than one predicting none."""
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    wrong = torch.zeros_like(t)
    wrong[:, 30, 30] = 1.0
    assert float(soft_dice(wrong, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(mse_skill(wrong, t)) < -0.1
    assert float(mse_skill(torch.zeros_like(t), t)) == pytest.approx(0.0, abs=1e-6)


def test_metrics_rank_partial_detection_between_nothing_and_perfect():
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    A = t.shape[-1]
    half = torch.where(torch.arange(A).view(1, A, 1) < 15, t, torch.zeros_like(t))
    for fn in (soft_dice, mse_skill):
        z, h, p = (float(fn(torch.zeros_like(t), t)), float(fn(half, t)), float(fn(t, t)))
        assert z < h < p, f"{fn.__name__} does not rank partial between none and perfect"


def test_flooding_the_map_is_punished_by_both():
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    flood = torch.ones_like(t)
    assert float(soft_dice(flood, t)) < 0.05
    assert float(mse_skill(flood, t)) < 0.0


def test_entity_loss_reports_every_class_separately():
    """Pooled would be carried by `player` — one big always-present sprite — while
    `enemy_bullets` is ~2 px and is the class that would help boss survival."""
    from contra_encoder.train import ENTITY_CLASSES, entity_loss

    n, A = 6, 32
    target = torch.zeros(n, 4, A, A)
    target[:, :, 10, 10] = 1.0
    loss, m = entity_loss(torch.randn(n, 4, A, A), target)
    assert torch.isfinite(loss)
    for c in ENTITY_CLASSES:
        assert f"entity/{c}/dice" in m and f"entity/{c}/mse_skill" in m


def test_entity_loss_skips_a_class_absent_from_every_frame():
    from contra_encoder.train import entity_loss

    n, A = 4, 32
    target = torch.zeros(n, 4, A, A)
    target[:, 0, 5, 5] = 1.0                      # only `player` present
    _loss, m = entity_loss(torch.randn(n, 4, A, A), target)
    assert "entity/player/dice" in m
    assert "entity/enemy_bullets/dice" not in m
    assert "entity/enemy_bullets/loss" in m       # loss is defined; all negatives


# ── window flattening ────────────────────────────────────────────────────────

def _window(b, t, size=8, A=4):
    return {
        "image": torch.arange(b * t).view(b, t, 1, 1, 1).expand(
            b, t, size, size, 3).to(torch.uint8).contiguous(),
        "cross_view": {
            "cross_view_image": torch.arange(b).view(b, 1, 1, 1).expand(
                b, size, size, 3).to(torch.uint8).contiguous(),
            "cross_view_obj_mask": torch.zeros(b, size, size, dtype=torch.uint8),
            "cross_view_obj_id": torch.arange(b).view(b, 1).expand(b, t).contiguous(),
        },
        "mask": torch.ones(b, t),
        "family": torch.arange(b),
    }


def test_flatten_window_keeps_frames_and_family_aligned():
    b, t = 3, 4
    f = flatten_window(_window(b, t))
    assert f["image"].shape[0] == b * t
    for i in range(b * t):
        assert int(f["image"][i, 0, 0, 0]) == i          # encodes (b, t) order
        assert int(f["family"][i]) == i // t


def test_flatten_window_returns_one_goal_image_per_window_not_per_frame():
    """0002: the goal is no longer paired with each frame — it is its own row."""
    b, t = 3, 4
    f = flatten_window(_window(b, t))
    assert f["goal_image"].shape[0] == b
    assert f["image"].shape[0] == b * t


def test_flatten_window_drops_padded_tail():
    b, t = 2, 5
    w = _window(b, t)
    w["mask"][:, 3:] = 0.0
    f = flatten_window(w)
    assert f["image"].shape[0] == b * 3
    assert f["family"].shape[0] == b * 3
    assert f["goal_image"].shape[0] == b        # unaffected: one per window


def test_flatten_window_returns_empty_on_an_all_padding_batch():
    w = _window(2, 3)
    w["mask"][:] = 0.0
    assert flatten_window(w) == {}


def test_flatten_window_carries_entity_targets_when_present():
    b, t, A = 2, 3, 4
    w = _window(b, t, A=A)
    w["entity_heatmap"] = torch.zeros(b, t, 4, A, A)
    w["goal_entity_heatmap"] = torch.zeros(b, 4, A, A)
    f = flatten_window(w)
    assert f["entity_heatmap"].shape == (b * t, 4, A, A)
    assert f["goal_entity_heatmap"].shape == (b, 4, A, A)


# ── goal frames are supervised too ───────────────────────────────────────────

@pytest.mark.skipif(not os.path.isdir(SHARD_DIR), reason="shards are not on this machine")
def test_goal_frames_get_real_entity_targets_from_goal_frame_idx():
    """The thing that makes 0002 symmetric in *supervision*, not only in architecture.

    `goal.png` is an episode frame with the target painted on; `goal_frame_idx` says
    which one, so `entities[cls][goal_frame_idx]` labels it exactly as any frame.
    """
    from contra_encoder.data import build_datamodule

    dm = build_datamodule(shard_dir=SHARD_DIR, families=("kill",), win_len=8,
                          batch_size=2, num_workers=0, cache_dir="cache",
                          want_entities=True)
    f = flatten_window(next(iter(dm.train_dataloader())))
    g = f["goal_entity_heatmap"]
    assert g.shape[1] == 4
    # `player` is on essentially every frame, so the goal frame's channel cannot be empty.
    assert float(g[:, 0].max()) > 0.5, "goal frame has no entity target — check goal_frame_idx"
    assert 0.0 <= float(g.min()) and float(g.max()) <= 1.0


@pytest.mark.skipif(not os.path.isdir(SHARD_DIR), reason="shards are not on this machine")
def test_goal_frame_idx_is_present_and_in_range_for_every_family():
    """It varies by family — kill/item use frame 0, traverse the last, boss ~7% in —
    so nothing may assume a constant."""
    import json

    from contra_policy.dataset import FAMILIES, load_or_build_index, shard_paths

    for fam in FAMILIES:
        idx = load_or_build_index(shard_paths(SHARD_DIR, (fam,), "val"), "cache")
        for ep in idx[:3]:
            with open(ep["tar"], "rb") as fh:
                off, size = ep["members"]["json"]
                fh.seek(off)
                meta = json.loads(fh.read(size))
            gi = meta.get("goal_frame_idx")
            assert gi is not None, f"{fam}/{ep['uid']} has no goal_frame_idx"
            assert 0 <= int(gi) < ep["length"], f"{fam}/{ep['uid']}: {gi} out of range"
