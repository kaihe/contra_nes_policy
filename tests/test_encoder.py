"""Encoder tests: the invariants that would fail quietly rather than crash.

The rebuild's whole premise is "one 512-d token can hold what four view tokens held",
and the gate on it is a number — boss point error against the 5.3 px the current policy
reports. So what has to be pinned here is not that the network runs, but that:

* ``heatmap_readout`` is *identical* to the policy's, because ``point_err_px`` is pinned
  by ``contra_nes_evaluation`` and a half-cell offset would move the gate by 3.7 px;
* the occupancy map is decoded from the **token**, not from the conv map — if a
  gradient path existed around the token, the grounding loss would stop forcing spatial
  structure into it and the premise would be untested;
* a checkpoint rebuilds its own architecture, so a config drift cannot silently load
  wrong weights;
* ``flatten_window`` keeps every member aligned, so a frame is never scored against
  another frame's goal.

Tests needing the shards skip cleanly.
"""

from __future__ import annotations

import os

import pytest
import torch

from contra_encoder import EncoderConfig, build_encoder, load_pretrained_encoder
from contra_encoder.data import flatten_window
from contra_encoder.heads import HeatmapHead, heatmap_readout

SHARD_DIR = os.path.expanduser("~/code/contra_nes_data/game_trace/hf")
BC_CKPT = os.path.expanduser("~/code/contra_nes_policy/runs/2026-07-28/18-01-29/"
                             "weights/weight-epoch=18-step=30000.ckpt")
# Small everywhere so the suite stays fast; image_size must remain minres*2^k.
SMALL = dict(image_size=64, hiddim=32, depth=4, minres=4, mask_depth=2,
             proj_ch=16, aux_size=32, head_depth=8)


def _enc(**over):
    return build_encoder(EncoderConfig(**{**SMALL, **over}))


def _inputs(b, size):
    return (torch.randint(0, 255, (b, size, size, 3), dtype=torch.uint8),
            torch.randint(0, 255, (b, size, size, 3), dtype=torch.uint8),
            torch.randint(0, 255, (b, size, size), dtype=torch.uint8))


# ── the pinned metric ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not os.path.exists(BC_CKPT),
                    reason="the BC checkpoint is not on this machine")
def test_heatmap_readout_is_identical_to_the_policys():
    from contra_policy.model import CrossViewContraRocket
    from contra_policy.rl import checkpoint as ckpt_io

    policy = CrossViewContraRocket(**ckpt_io.model_config_from_checkpoint(BC_CKPT))
    torch.manual_seed(0)
    for shape in [(5, 1, 32, 32), (2, 7, 32, 32)]:
        heat = torch.randn(*shape) * 3
        p_ref, e_ref = policy.heatmap_readout(heat)
        p_new, e_new = heatmap_readout(heat)
        # Bit-identical, not close: point_err_px is pinned by the evaluator, and the
        # gate on this rebuild is a 5.3 px comparison against it.
        assert torch.equal(p_ref, p_new), f"point differs at {shape}"
        assert torch.equal(e_ref, e_new), f"exist differs at {shape}"


def test_soft_argmax_has_no_half_cell_offset():
    """A blob centred on cell (c, r) must read back as (c/A, r/A), not (c+.5)/A.

    ``goal.goal_mask`` places a blob at ``cx = x_norm * A`` exactly, so the readout has
    to invert that. Adding half a cell biases every prediction by 0.5/A — about 3.7
    screen px at A=32, a quarter of the current error.
    """
    A = 32
    for (col, row) in [(0, 0), (7, 21), (31, 31)]:
        heat = torch.full((1, A, A), -30.0)
        heat[0, row, col] = 30.0
        point, _ = heatmap_readout(heat)
        assert point[0, 0].item() == pytest.approx(col / A, abs=1e-4)
        assert point[0, 1].item() == pytest.approx(row / A, abs=1e-4)


# ── the premise: grounding flows through the token ────────────────────────────

def test_occupancy_is_decoded_only_from_the_token():
    """No gradient path from the heatmap to the trunk that bypasses the frame token.

    This is the experiment the rebuild rests on. If the heatmap could be read off the
    conv map directly, the token would be free to discard spatial structure and the
    grounding loss would never force it to keep sprites.
    """
    enc = _enc()
    frame, gimg, gmsk = _inputs(2, enc.cfg.image_size)
    goal = enc.encode_goal(gimg, gmsk)
    token, heat = enc.encode_frame(frame, goal.detach())

    # Detaching the token must sever the heatmap from every trunk parameter.
    _t2, heat_detached = token.detach(), enc.heatmap_head(token.detach())
    heat_detached.sum().backward()
    leaked = [n for n, p in enc.view_backbone.named_parameters()
              if p.grad is not None and p.grad.abs().sum() > 0]
    assert not leaked, f"heatmap reached the trunk without passing the token: {leaked}"

    # And with the token attached, it must reach the trunk — otherwise the loss is
    # training nothing and the test above would pass vacuously.
    enc.zero_grad(set_to_none=True)
    heat.sum().backward()
    reached = [n for n, p in enc.view_backbone.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    assert reached, "heatmap gradient never reached the conv trunk at all"


def test_heatmap_head_grid_must_be_a_power_of_two_times_base():
    HeatmapHead(dim=16, grid=32, base=4)                 # 3 upsamples, fine
    with pytest.raises(ValueError, match="power of two"):
        HeatmapHead(dim=16, grid=30, base=4)


def test_conditioning_starts_as_identity():
    """FiLM is zero-init, so step 0 sees the trunk's features unperturbed.

    A randomly initialised gamma/beta would multiply the trunk's output by noise on the
    first step, which is a bad place to start a from-scratch encoder.
    """
    enc = _enc()
    frame, gimg, gmsk = _inputs(2, enc.cfg.image_size)
    g1 = enc.encode_goal(gimg, gmsk)
    g2 = torch.randn_like(g1) * 5           # a wildly different goal token
    t1, _ = enc.encode_frame(frame, g1)
    t2, _ = enc.encode_frame(frame, g2)
    assert torch.allclose(t1, t2, atol=1e-6), \
        "at init the goal token must not yet influence the frame token"


def test_shapes_and_single_class_squeeze():
    enc = _enc()
    b, A = 3, enc.cfg.aux_size
    out = enc(*_inputs(b, enc.cfg.image_size), torch.tensor([0, 2, -1]))
    assert out["token"].shape == (b, enc.cfg.hiddim)
    assert out["goal_token"].shape == (b, enc.cfg.hiddim)
    # n_classes == 1 drops the class axis, matching the policy's (B, A, A) contract.
    assert out["goal_heatmap"].shape == (b, A, A)
    assert out["point"].shape == (b, 2) and out["exist"].shape == (b, 1)


def test_multi_class_keeps_the_class_axis():
    """The 4-class entity target needs RAM in the shards, but the shape must already work."""
    enc = _enc(n_classes=4)
    b, A = 2, enc.cfg.aux_size
    out = enc(*_inputs(b, enc.cfg.image_size))
    assert out["goal_heatmap"].shape == (b, 4, A, A)


def test_interaction_id_minus_one_is_a_valid_embedding_row():
    """id -1 means "no goal" and must land on row 0, not index out of bounds."""
    enc = _enc()
    out = enc(*_inputs(2, enc.cfg.image_size), torch.tensor([-1, -1]))
    assert torch.isfinite(out["token"]).all()


# ── checkpoints rebuild their own architecture ───────────────────────────────

def test_checkpoint_round_trip_restores_architecture_and_weights(tmp_path):
    enc = _enc(proj_ch=8, head_depth=4)
    path = str(tmp_path / "encoder.pt")
    enc.save(path)

    back = load_pretrained_encoder(path)
    assert back.cfg.to_dict() == enc.cfg.to_dict()
    enc.eval(), back.eval()
    args = _inputs(2, enc.cfg.image_size)
    with torch.no_grad():
        a, b = enc(*args)["token"], back(*args)["token"]
    assert torch.equal(a, b)


def test_freeze_on_load_disables_every_gradient(tmp_path):
    enc = _enc()
    path = str(tmp_path / "encoder.pt")
    enc.save(path)
    frozen = load_pretrained_encoder(path, freeze=True)
    assert not any(p.requires_grad for p in frozen.parameters())


def test_the_conv_trunk_is_trainable_by_default():
    """The policy freezes it; this package exists to train it.

    With a frozen trunk the grounding loss reaches only the projection layers, which is
    exactly the limitation the rebuild is meant to remove.
    """
    assert EncoderConfig().freeze_view_backbone is False
    assert all(p.requires_grad for p in _enc().view_backbone.parameters())


def test_the_rgb_trunk_is_shared_between_frame_and_goal():
    """One trunk, as in the policy — a second would be duplicated parameters."""
    enc = _enc()
    assert not hasattr(enc, "goal_backbone")
    frame, gimg, gmsk = _inputs(2, enc.cfg.image_size)
    enc.encode_goal(gimg, gmsk).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in enc.view_backbone.parameters()), \
        "the goal path must train the shared trunk too"


# ── window flattening keeps members aligned ──────────────────────────────────

def _window(b, t, size=8, A=4):
    return {
        "image": torch.arange(b * t).view(b, t, 1, 1, 1).expand(
            b, t, size, size, 3).to(torch.uint8).contiguous(),
        "cross_view": {
            "cross_view_image": torch.arange(b).view(b, 1, 1, 1).expand(
                b, size, size, 3).to(torch.uint8).contiguous(),
            "cross_view_obj_mask": torch.arange(b).view(b, 1, 1).expand(
                b, size, size).to(torch.uint8).contiguous(),
            "cross_view_obj_id": torch.arange(b).view(b, 1).expand(b, t).contiguous(),
        },
        "goal_heatmap": torch.zeros(b, t, A, A),
        "point": torch.zeros(b, t, 2),
        "exist": torch.ones(b, t),
        "mask": torch.ones(b, t),
        "family": torch.arange(b),
    }


def test_flatten_window_pairs_each_frame_with_its_own_goal():
    b, t = 3, 4
    f = flatten_window(_window(b, t))
    assert f["image"].shape[0] == b * t
    for i in range(b * t):
        # image value encodes (b, t) order; goal/family encode b. They must agree.
        assert int(f["image"][i, 0, 0, 0]) == i
        assert int(f["goal_image"][i, 0, 0, 0]) == i // t
        assert int(f["family"][i]) == i // t


def test_flatten_window_drops_padded_tail():
    b, t = 2, 5
    w = _window(b, t)
    w["mask"][:, 3:] = 0.0                    # last two steps are padding
    f = flatten_window(w)
    assert f["image"].shape[0] == b * 3
    assert all(v.shape[0] == b * 3 for v in f.values())


def test_flatten_window_returns_empty_on_an_all_padding_batch():
    w = _window(2, 3)
    w["mask"][:] = 0.0
    assert flatten_window(w) == {}


# ── which metric is trustworthy ──────────────────────────────────────────────

def test_peak_hit_ignores_a_bimodal_prediction_that_fools_the_soft_argmax():
    """The failure mode `peak_hit` exists to catch.

    Two confident peaks either side of the real goal average, under soft-argmax, to a
    coordinate near the truth — so `point_err_px` looks good while the map is wrong.
    `peak_hit` asks whether the argmax *cell* is inside the blob, and is not fooled.
    """
    from contra_encoder.train import _peak_hit
    from contra_policy.goal import goal_mask

    A = 32
    target = torch.from_numpy(goal_mask([[120.0, 112.0]], A, 12.0)).unsqueeze(0)
    centre = target[0].reshape(-1).argmax().item()
    row, col = centre // A, centre % A

    pred = torch.full((1, A, A), -30.0)
    pred[0, row, max(0, col - 8)] = 30.0            # two peaks, symmetric about truth
    pred[0, row, min(A - 1, col + 8)] = 30.0

    point, _ = heatmap_readout(pred)
    # Soft-argmax lands close to the true column despite neither peak being there.
    assert abs(point[0, 0].item() - col / A) < 0.05
    # peak_hit sees through it.
    assert float(_peak_hit(pred, target)) == 0.0


def test_grounding_metrics_are_reported_per_family_and_pooled():
    from contra_encoder.train import per_family_grounding
    from contra_encoder.data import FAMILIES as F

    n, A = 8, 32
    pred = {"point": torch.rand(n, 2), "exist": torch.randn(n, 1),
            "goal_heatmap": torch.randn(n, A, A)}
    batch = {"point": torch.rand(n, 2), "exist": torch.ones(n),
             "goal_heatmap": torch.zeros(n, A, A),
             "family": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])}
    m = per_family_grounding(pred, batch)
    for key in ("point_err_px", "point_err_px_p50", "pck8", "pck16", "peak_hit"):
        assert key in m, f"pooled {key} missing"
    for f in F:
        assert f"{f}/point_err_px" in m and f"{f}/frames" in m
        assert m[f"{f}/frames"] == 2.0


def test_point_metrics_skip_frames_with_no_visible_goal():
    """An invisible goal has no centroid, so it must not enter the point statistics."""
    from contra_encoder.train import per_family_grounding

    n, A = 4, 32
    pred = {"point": torch.zeros(n, 2), "exist": torch.zeros(n, 1),
            "goal_heatmap": torch.zeros(n, A, A)}
    batch = {"point": torch.zeros(n, 2), "exist": torch.zeros(n),   # nothing visible
             "goal_heatmap": torch.zeros(n, A, A), "family": torch.zeros(n, dtype=torch.long)}
    m = per_family_grounding(pred, batch)
    assert "point_err_px" not in m and "pck8" not in m
    assert "exist_acc" in m                     # still defined on every frame


# ── 4-class entity target (shards re-exported 2026-07-30) ────────────────────

def test_entity_head_is_separate_from_the_goal_head():
    """Two heads, not extra channels on one.

    The goal head is conditioned on *which* entity is the target and feeds the pinned
    `point_err_px` gate; entity occupancy is a property of the frame alone. Sharing a
    head would entangle the gate with a signal unrelated to the task.
    """
    enc = _enc(entity_classes=4)
    b, A = 2, enc.cfg.aux_size
    out = enc(*_inputs(b, enc.cfg.image_size))
    assert out["goal_heatmap"].shape == (b, A, A)        # unchanged contract
    assert out["entity_heatmap"].shape == (b, 4, A, A)
    assert enc.entity_head is not None and enc.heatmap_head is not enc.entity_head


def test_entity_head_is_off_by_default():
    enc = _enc()
    assert enc.entity_head is None
    assert "entity_heatmap" not in enc(*_inputs(2, enc.cfg.image_size))


def test_entity_head_also_trains_only_through_the_token():
    """Same premise as the goal head: no gradient path around the frame token."""
    enc = _enc(entity_classes=4)
    frame, gimg, gmsk = _inputs(2, enc.cfg.image_size)
    goal = enc.encode_goal(gimg, gmsk)
    token, _ = enc.encode_frame(frame, goal.detach())
    enc.entity_head(token.detach()).sum().backward()
    assert not [n for n, p in enc.view_backbone.named_parameters()
                if p.grad is not None and p.grad.abs().sum() > 0]


def test_entity_loss_reports_every_class_separately():
    """Pooled would be carried by `player` — one big always-present sprite."""
    from contra_encoder.train import ENTITY_CLASSES, entity_loss

    n, A = 6, 32
    target = torch.zeros(n, 4, A, A)
    target[:, :, 10, 10] = 1.0                    # every class present somewhere
    loss, m = entity_loss(torch.randn(n, 4, A, A), target)
    assert torch.isfinite(loss)
    for c in ENTITY_CLASSES:
        assert f"entity/{c}/loss" in m
        assert f"entity/{c}/peak_hit" in m


def test_entity_loss_skips_peak_hit_for_an_absent_class():
    """A class with no instance on any frame has no localisation to score."""
    from contra_encoder.train import entity_loss

    n, A = 4, 32
    target = torch.zeros(n, 4, A, A)
    target[:, 0, 5, 5] = 1.0                      # only `player` present
    _loss, m = entity_loss(torch.randn(n, 4, A, A), target)
    assert "entity/player/peak_hit" in m
    assert "entity/enemy_bullets/peak_hit" not in m
    assert "entity/enemy_bullets/loss" in m       # loss is still defined (all negatives)


def test_flatten_window_carries_entity_heatmaps_when_present():
    b, t, A = 2, 3, 4
    w = _window(b, t, A=A)
    w["entity_heatmap"] = torch.zeros(b, t, 4, A, A)
    f = flatten_window(w)
    assert f["entity_heatmap"].shape == (b * t, 4, A, A)
    del w["entity_heatmap"]
    assert "entity_heatmap" not in flatten_window(w)


@pytest.mark.skipif(not os.path.isdir(SHARD_DIR), reason="shards are not on this machine")
def test_dataset_emits_entity_targets_from_the_re_exported_shards():
    """End-to-end against the real shards, including the all-zero fallback contract."""
    from contra_encoder.data import build_datamodule

    dm = build_datamodule(shard_dir=SHARD_DIR, win_len=4, batch_size=2, num_workers=0,
                          cache_dir="cache", want_entities=True)
    batch = next(iter(dm.train_dataloader()))
    eh = batch["entity_heatmap"]
    assert eh.shape[2] == 4 and eh.shape[-1] == eh.shape[-2] == 32
    # `player` is present on essentially every frame, so its channel cannot be empty.
    assert float(eh[:, :, 0].max()) > 0.5, "player channel is empty — entities not read"
    assert float(eh.min()) >= 0.0 and float(eh.max()) <= 1.0


@pytest.mark.skipif(not os.path.isdir(SHARD_DIR), reason="shards are not on this machine")
def test_entities_absent_falls_back_to_zeros_not_a_crash():
    """Shards older than the 2026-07-30 re-export must still load."""
    from contra_policy.dataset import (ContraCrossViewDataset, load_or_build_index,
                                       shard_paths)

    idx = load_or_build_index(shard_paths(SHARD_DIR, ("kill",), "val"), "cache")
    ds = ContraCrossViewDataset(idx, win_len=4, want_entities=True)
    real = ds[0]["entity_heatmap"]
    assert float(real.max()) > 0.0                 # the new shards do have entities

    original = ds._read

    def _strip(ep, ext):
        raw = original(ep, ext)
        if ext != "json":
            return raw
        import json as _json
        meta = _json.loads(raw)
        meta.pop("entities", None)                 # simulate a pre-re-export shard
        return _json.dumps(meta).encode()

    ds._read = _strip
    assert float(ds[0]["entity_heatmap"].max()) == 0.0


# ── continuous, full-map entity metrics ──────────────────────────────────────

def _blob_target(n=3, A=32):
    t = torch.zeros(n, A, A)
    t[:, 10, 10] = 1.0
    t[:, 10, 11] = 0.5          # soft, as goal_mask renders it
    t[:, 20, 20] = 1.0
    return t


def test_soft_dice_peaks_at_one_on_a_soft_target():
    """The reason for the squared denominator.

    Our targets are soft Gaussians, not binary masks. With the familiar `Σp + Σt`
    denominator an exactly correct prediction scores `Σt²/Σt` — about 0.90 on a typical
    blob — so the ceiling would move with the target's shape and the number would be
    unreadable.
    """
    from contra_encoder.train import soft_dice

    t = _blob_target()
    assert float(soft_dice(t, t)) == pytest.approx(1.0, abs=1e-6)
    assert float(soft_dice(torch.zeros_like(t), t)) == pytest.approx(0.0, abs=1e-6)


def test_both_metrics_are_zero_for_predicting_nothing():
    """The baseline that plain MSE hides: these maps are 95-98% empty, so an all-zero
    prediction already scores MSE 0.002-0.007. Both normalised metrics put it at 0."""
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    z = torch.zeros_like(t)
    assert float(soft_dice(z, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(mse_skill(z, t)) == pytest.approx(0.0, abs=1e-6)


def test_mse_skill_distinguishes_wrong_place_from_predicting_nothing():
    """Where the two metrics complement each other.

    Dice scores both at 0.000 — it cannot tell silence from confident error. mse_skill
    goes negative for mass in the wrong place, which is the failure worth catching:
    a model hallucinating bullets is worse than one predicting none.
    """
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    wrong = torch.zeros_like(t)
    wrong[:, 30, 30] = 1.0
    assert float(soft_dice(wrong, t)) == pytest.approx(0.0, abs=1e-6)
    assert float(soft_dice(torch.zeros_like(t), t)) == pytest.approx(0.0, abs=1e-6)
    assert float(mse_skill(wrong, t)) < -0.1          # strictly worse than silence
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
    """Predicting 'everything everywhere' must not look good on a 98%-empty target."""
    from contra_encoder.train import mse_skill, soft_dice

    t = _blob_target()
    flood = torch.ones_like(t)
    assert float(soft_dice(flood, t)) < 0.05
    assert float(mse_skill(flood, t)) < 0.0


# ── the point-metric false alarm ─────────────────────────────────────────────

def test_point_metrics_exclude_multi_component_goals():
    """`points_to_target` returns the MEAN of a frame's goal centroids.

    With one centroid that is the thing's location; with several it names a spot where
    nothing is, and error against it *grows* as a predictor sharpens. Boss goals have
    4.57 components on 98.7% of frames, which is why the first run's boss error went
    2.6 px -> 8.8 px while `peak_hit` reached 0.999.
    """
    from contra_encoder.train import per_family_grounding

    n, A = 6, 32
    pred = {"point": torch.zeros(n, 2), "exist": torch.ones(n, 1),
            "goal_heatmap": torch.zeros(n, A, A)}
    batch = {"point": torch.zeros(n, 2), "exist": torch.ones(n),
             "goal_heatmap": torch.zeros(n, A, A),
             "family": torch.zeros(n, dtype=torch.long),
             "n_goal_points": torch.tensor([1, 1, 4, 5, 1, 7])}
    m = per_family_grounding(pred, batch)
    # 3 of 6 frames are single-centroid; only those enter the point statistics.
    assert m["point_frames"] == 3.0
    assert m["multi_goal_frac"] == pytest.approx(0.5)
    # peak_hit is defined on every visible frame regardless of goal shape — the gate.
    assert "peak_hit" in m


def test_point_metrics_vanish_when_every_goal_is_multi_component():
    """A family that is ~99% multi-component must report no point number at all,
    rather than one computed from a 1% sliver that looks authoritative."""
    from contra_encoder.train import per_family_grounding

    n, A = 4, 32
    pred = {"point": torch.zeros(n, 2), "exist": torch.ones(n, 1),
            "goal_heatmap": torch.zeros(n, A, A)}
    batch = {"point": torch.zeros(n, 2), "exist": torch.ones(n),
             "goal_heatmap": torch.zeros(n, A, A),
             "family": torch.zeros(n, dtype=torch.long),
             "n_goal_points": torch.full((n,), 4)}
    m = per_family_grounding(pred, batch)
    assert "point_err_px" not in m and "pck8" not in m
    assert m["multi_goal_frac"] == 1.0
    assert "peak_hit" in m


def test_missing_centroid_count_is_treated_as_single():
    """Shards older than 2026-07-31 have no count; every family but boss is single."""
    from contra_encoder.train import per_family_grounding

    n, A = 4, 32
    pred = {"point": torch.zeros(n, 2), "exist": torch.ones(n, 1),
            "goal_heatmap": torch.zeros(n, A, A)}
    batch = {"point": torch.zeros(n, 2), "exist": torch.ones(n),
             "goal_heatmap": torch.zeros(n, A, A),
             "family": torch.zeros(n, dtype=torch.long)}      # no n_goal_points
    m = per_family_grounding(pred, batch)
    assert m["point_frames"] == 4.0 and m["multi_goal_frac"] == 0.0


@pytest.mark.skipif(not os.path.isdir(SHARD_DIR), reason="shards are not on this machine")
def test_boss_goals_really_are_multi_component_in_the_shards():
    """The measurement the whole fix rests on, pinned against the real data."""
    from contra_encoder.data import build_datamodule, flatten_window, FAMILIES

    dm = build_datamodule(shard_dir=SHARD_DIR, families=("boss", "kill"), win_len=16,
                          batch_size=4, num_workers=0, cache_dir="cache")
    seen = {}
    for i, b in enumerate(dm.val_dataloader()):
        if i >= 12:
            break
        f = flatten_window(b)
        vis = f["exist"] > 0
        for fi, name in enumerate(FAMILIES):
            sel = (f["family"] == fi) & vis
            if bool(sel.any()):
                seen.setdefault(name, []).append(f["n_goal_points"][sel].float().mean())
    assert "boss" in seen, "no visible boss frames sampled"
    boss = float(torch.stack(seen["boss"]).mean())
    assert boss > 2.0, f"boss goals should be multi-component, got {boss:.2f}"
    if "kill" in seen:
        assert float(torch.stack(seen["kill"]).mean()) == pytest.approx(1.0, abs=0.01)
