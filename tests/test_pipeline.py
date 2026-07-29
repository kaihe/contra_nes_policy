"""End-to-end smoke tests: real shards → batch → forward → loss → backward.

These run against the actual tars under ``~/code/contra_nes_data/game_trace/hf`` and
skip cleanly if they are absent. They are deliberately shape- and invariant-focused:
the failure modes that matter in this pipeline are silent (a PPU coordinate that is
never shifted into screen space, an aux target regressed on frames where the goal is
not visible, padding that quietly contributes to a mean), not crashes.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
import torch

from contra_policy.action_space import (NUM_ACTIONS,
                                        check_matches_source, indices_to_vectors,
                                        vectors_to_indices)
from contra_policy.dataset import (FAMILIES, ContraCrossViewDataset,
                                   ContraDataModule, EpisodeStreamSampler,
                                   family_weights, load_or_build_index, shard_paths)
from contra_policy.goal import (VIS_H, VIS_W, XOFF, YOFF, goal_mask, interaction_id,
                                points_to_target, ppu_to_norm)
from contra_policy.loss import ContraObjective, action_class_weights
from contra_policy.model import CrossViewContraRocket

SHARD_DIR = os.path.expanduser("~/code/contra_nes_data/game_trace/hf")
TARS = shard_paths(SHARD_DIR, ("item",), "train")
VAL_TARS = shard_paths(SHARD_DIR, ("item",), "val")
HAVE_SHARDS = all(os.path.exists(p) for p in TARS + VAL_TARS)
CACHE = os.path.join(os.path.dirname(__file__), "..", "cache")
# Families whose train+val shards are both present, for the multi-family val test.
CONFIGS_ON_DISK = tuple(
    c for c in ("kill", "item", "traverse", "boss")
    if all(os.path.exists(p) for p in shard_paths(SHARD_DIR, (c,), "train")
           + shard_paths(SHARD_DIR, (c,), "val")))

TINY_MODEL = dict(image_size=64, view_depth=8, mask_depth=4, minres=4,
                  view_backbone_ckpt=None, hiddim=64, num_heads=4, num_layers=2,
                  timesteps=8, mem_len=8, num_view_tokens=2, aux_size=16)


# ── action space ──────────────────────────────────────────────────────────────

def test_action_roundtrip():
    idx = np.arange(NUM_ACTIONS)
    assert np.array_equal(vectors_to_indices(indices_to_vectors(idx)), idx)


def test_action_table_matches_data_repo():
    if not check_matches_source():
        pytest.skip("contra_nes_data/src/agent/baseline.yaml not reachable")


def test_unknown_action_vector_is_rejected():
    # SELECT is never pressed by the searcher; mapping it to a nearest action would
    # corrupt the BC target silently, so it must raise.
    with pytest.raises(ValueError):
        vectors_to_indices(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.uint8))


# ── goal geometry ─────────────────────────────────────────────────────────────

def test_ppu_shift_maps_screen_corners_to_unit_square():
    # PPU (8, 8) is the top-left of the 240x224 overscan crop; (248, 232) is past it.
    corners = ppu_to_norm([(XOFF, YOFF), (XOFF + VIS_W, YOFF + VIS_H)])
    assert np.allclose(corners[0], [0.0, 0.0])
    assert np.allclose(corners[1], [1.0, 1.0])


def test_goal_mask_peaks_at_the_marked_point():
    size = 64
    px, py = 128, 120                       # mid-screen in PPU coords
    hm = goal_mask([(px, py)], size, sigma_px=12.0)
    peak_y, peak_x = np.unravel_index(hm.argmax(), hm.shape)
    expect = ppu_to_norm([(px, py)])[0] * size
    assert abs(peak_x - expect[0]) <= 1 and abs(peak_y - expect[1]) <= 1
    assert hm.max() == pytest.approx(1.0, abs=1e-3)


def test_empty_goal_gives_zero_mask_and_zero_targets():
    assert goal_mask([], 32).max() == 0.0
    point, bbox = points_to_target([])
    assert point.sum() == 0.0 and bbox.sum() == 0.0


def test_bbox_spans_all_boss_components():
    pts = [(50, 60), (150, 100)]            # two boss parts
    point, bbox = points_to_target(pts, half_size_px=8.0)
    n = ppu_to_norm(pts)
    assert np.allclose(point, n.mean(0))
    assert bbox[0] < n[:, 0].min() and bbox[2] > n[:, 0].max()
    assert bbox[1] < n[:, 1].min() and bbox[3] > n[:, 1].max()
    assert ((bbox >= 0) & (bbox <= 1)).all()


def test_interaction_ids_cover_the_five_task_families():
    cases = {
        "kill": {"goal_when": "first", "goal_kind": "target"},
        "pick": {"goal_when": "first", "goal_kind": "item", "kind": "pick"},
        "avoid": {"goal_when": "first", "goal_kind": "item", "kind": "avoid"},
        "traverse": {"goal_when": "last", "goal_kind": "player"},
        "boss": {"goal_when": "boss", "goal_kind": "boss"},
    }
    ids = {name: interaction_id(meta) for name, meta in cases.items()}
    assert len(set(ids.values())) == 5


# ── dataset ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    if not HAVE_SHARDS:
        pytest.skip(f"shards not found under {SHARD_DIR}")
    index = load_or_build_index(TARS, cache_dir=CACHE)
    return ContraCrossViewDataset(index, win_len=8, image_size=64, seed=0)


def test_index_lengths_are_positive(dataset):
    assert len(dataset.index) > 0
    assert all(ep["length"] > 0 for ep in dataset.index)


def test_item_shapes_and_dtypes(dataset):
    item = dataset[0]
    T, S = 8, 64
    assert item["image"].shape == (T, S, S, 3) and item["image"].dtype == torch.uint8
    cv = item["cross_view"]
    # No time axis on the goal — see test_window_payload_stays_small.
    assert cv["cross_view_image"].shape == (S, S, 3)
    assert cv["cross_view_image"].dtype == torch.uint8
    assert cv["cross_view_obj_mask"].shape == (S, S)
    assert cv["cross_view_obj_mask"].dtype == torch.uint8
    assert cv["cross_view_obj_id"].shape == (T,)
    A = dataset.aux_size
    for key, shape in [("prev_action", (T,)), ("action", (T,)), ("exist", (T,)),
                       ("point", (T, 2)), ("goal_heatmap", (T, A, A)), ("mask", (T,)),
                       ("first", (T,)), ("family", ())]:
        assert item[key].shape == shape, key
    assert item["action"].max() < NUM_ACTIONS
    assert FAMILIES[int(item["family"])] == dataset.index[dataset.windows[0][0]]["family"]


def test_cross_view_is_constant_across_the_window(dataset):
    # A Contra episode has exactly one goal, unlike ROCKET-2's per-window sampling.
    cv = dataset[0]["cross_view"]
    assert (cv["cross_view_obj_id"] == cv["cross_view_obj_id"][0]).all()


def test_window_payload_stays_small(dataset):
    """Guards the OOM that materialising the goal per timestep caused.

    A loader keeps num_workers * prefetch_factor * batch_size windows resident and
    pin_memory doubles that, so per-window bytes translate directly into GBs of host
    RAM. At 64px/T=8 the agent view alone is 393KB; the goal must be a rounding error
    on top, not another 1.5x.
    """
    item = dataset[0]
    nbytes = lambda d: sum(v.numel() * v.element_size() if torch.is_tensor(v) else nbytes(v)
                           for v in d.values())
    total = nbytes(item)
    goal = nbytes({k: v for k, v in item["cross_view"].items() if k != "cross_view_obj_id"})
    assert goal < 0.15 * total, f"cross view is {goal/total:.0%} of the window payload"


def test_prev_action_is_the_shifted_action(dataset):
    item = dataset[0]                       # window 0 → start == 0
    n = int(item["mask"].sum())
    assert torch.equal(item["prev_action"][1:n], item["action"][:n - 1])


def _raw_episode(dataset, ep_i):
    """The episode's recorded actions and centroids, straight off the tar."""
    ep = dataset.index[ep_i]
    actions = vectors_to_indices(np.load(io.BytesIO(dataset._read(ep, "actions.npy"))))
    meta = json.loads(dataset._read(ep, "json"))
    return ep, actions, meta


def test_window_indices_match_the_recorded_trajectory(dataset):
    """Pins the observation/action alignment against the raw shard.

    ``export_hf.materialize`` steps the emulator and *then* records the screen, so
    ``frames[j]`` is the state AFTER ``actions[j]``. The target for ``frames[j]`` is
    therefore the action taken *from* it, ``actions[j+1]``; pairing it with
    ``actions[j]`` instead trains inverse dynamics — the model reads the effect of the
    action off the frame and reports the cause. That bug is invisible at runtime (every
    shape and mask stays valid, and open-loop accuracy goes *up*), so assert the index
    relationship itself rather than any downstream number.

    Checked over many windows, not one: an episode often opens on a run of identical
    actions, and over such a run the correct and the off-by-one pairing agree. The
    ``discriminating`` counter below fails the test if it only ever saw such windows,
    so this can never pass vacuously.
    """
    discriminating = 0
    for which in range(min(40, len(dataset.windows))):
        ep_i, rel = dataset.windows[which]
        ep, actions, meta = _raw_episode(dataset, ep_i)
        item = dataset[which]
        start = rel * dataset.win_len
        n = int(item["mask"].sum())

        for k in range(n):
            j = start + k
            assert item["action"][k].item() == actions[j + 1], (
                f"action[{k}] of window {which} must be the action taken FROM frame {j}")
            assert item["prev_action"][k].item() == actions[j], (
                f"prev_action[{k}] of window {which} must be what PRODUCED frame {j}")
            discriminating += actions[j] != actions[j + 1]
            # Aux targets are read from the same RAM sample as the frame, so they do
            # NOT shift with the actions.
            if j < len(meta["centroids"]) and meta["visibility"][j] and meta["centroids"][j]:
                point, _bbox = points_to_target(meta["centroids"][j])
                assert np.allclose(item["point"][k].numpy(), point), f"aux point at {k}"

    assert discriminating > 0, "no window where the shift changes the target — test is vacuous"


def test_last_frame_of_an_episode_is_never_a_training_target(dataset):
    """The action taken from the final frame was never recorded, so it is dropped."""
    for ep_i, ep in enumerate(dataset.index):
        rels = [r for i, r in dataset.windows if i == ep_i]
        if not rels:
            assert ep["length"] <= 1        # degenerate episodes are skipped entirely
            continue
        assert max(rels) * dataset.win_len < ep["length"] - 1

    # …and the reachable steps are exactly length-1 per episode.
    reachable = sum(int(dataset[i]["mask"].sum()) for i in range(min(30, len(dataset))))
    assert reachable > 0


def test_no_source_trace_is_shared_between_train_and_val():
    """The split is the data repo's and must not leak across shards.

    The previous in-repo random split put every val episode's source trace in train
    too; this asserts the property the new shards are supposed to guarantee.
    """
    if not HAVE_SHARDS:
        pytest.skip(f"shards not found under {SHARD_DIR}")

    def traces(tars, split):
        out = set()
        ds = ContraCrossViewDataset(load_or_build_index(tars, cache_dir=CACHE),
                                    win_len=8, image_size=64)
        for ep in ds.index:
            meta = json.loads(ds._read(ep, "json"))
            assert meta["split"] == split, f"{ep['uid']} is {meta['split']} in a {split} shard"
            out.add(meta["src_trace"])
        return out

    assert not (traces(TARS, "train") & traces(VAL_TARS, "val"))


def test_padding_is_masked_off(dataset):
    """Any window shorter than win_len must have mask==0 on the tail."""
    short = next((i for i, (e, r) in enumerate(dataset.windows)
                  if dataset.index[e]["length"] - r * 8 < 8), None)
    if short is None:
        pytest.skip("no partial window in this shard")
    item = dataset[short]
    n = int(item["mask"].sum())
    assert n < 8
    assert item["mask"][n:].sum() == 0.0
    assert item["image"][n:].sum() == 0      # padding frames are blank, not stale


def test_aux_targets_only_where_the_goal_is_visible(dataset):
    for i in range(min(20, len(dataset))):
        item = dataset[i]
        invisible = (item["exist"] == 0)
        assert item["point"][invisible].sum() == 0.0
        # An invisible goal is an all-zero heatmap — this is the negative example the
        # scalar `exist` head never got on kill/boss.
        assert item["goal_heatmap"][invisible].sum() == 0.0


def test_heatmap_peak_agrees_with_the_point_target(dataset):
    """The heatmap and the pinned `point` metric must describe the same location.

    `point_err_px` is derived from a soft-argmax over the predicted heatmap, so if the
    target rendering and the point convention disagree the metric silently measures a
    constant offset. Checked against the *target*, where the answer is known.
    """
    checked = 0
    for i in range(min(30, len(dataset))):
        item = dataset[i]
        for k in range(int(item["mask"].sum())):
            hm = item["goal_heatmap"][k]
            if item["exist"][k] == 0 or hm.max() == 0:
                continue
            a = hm.shape[-1]
            row, col = np.unravel_index(int(hm.argmax()), hm.shape)
            # goal_mask puts a blob centre at cx = x_norm * a, so col/a recovers x.
            assert abs(col / a - float(item["point"][k, 0])) < 2.0 / a
            assert abs(row / a - float(item["point"][k, 1])) < 2.0 / a
            checked += 1
    assert checked > 0, "no visible goal in the sampled windows — test is vacuous"


def test_episode_stream_sampler_keeps_windows_consecutive(dataset):
    """Memory carry is only correct if lane j walks one episode in order.

    Asserts the two properties the carried memory depends on: within a lane, windows
    of an episode arrive consecutively and in temporal order, and `first` is true
    exactly when a lane starts a new episode.
    """
    bs = 4
    sampler = EpisodeStreamSampler(dataset, batch_size=bs, shuffle=True, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) and all(len(b) == bs for b in batches)

    prev = [None] * bs
    for batch in batches:
        for lane, flat in enumerate(batch):
            ep_i, rel = dataset.windows[flat]
            starts_episode = rel == 0
            if not starts_episode:
                # Continuing: must be the very next window of the same episode.
                assert prev[lane] == (ep_i, rel - 1), (
                    f"lane {lane} jumped to {(ep_i, rel)} from {prev[lane]}")
            assert bool(dataset[flat]["first"][0]) == starts_episode
            prev[lane] = (ep_i, rel)


def test_validation_subset_is_representative_and_stable():
    """`limit_val_batches` must not take a family-contiguous prefix.

    The index sorts by tar path, so unpermuted the val set runs boss, item, kill,
    traverse — and a prefix of 800 windows contains no traverse at all despite
    traverse being 65% of it. Asserts the permutation mixes families early, and that
    it is identical across datamodule constructions so the metric is comparable.
    """
    if not HAVE_SHARDS:
        pytest.skip(f"shards not found under {SHARD_DIR}")
    dm = ContraDataModule(shard_dir=SHARD_DIR, configs=("item",), win_len=8,
                          image_size=64, batch_size=2, num_workers=0,
                          cache_dir=CACHE, seed=0)
    order = dm._val_order(dm.val_dataset)
    assert sorted(order) == list(range(len(dm.val_dataset)))     # a permutation
    assert order != list(range(len(dm.val_dataset)))             # …and not identity
    again = ContraDataModule(shard_dir=SHARD_DIR, configs=("item",), win_len=8,
                             image_size=64, batch_size=2, num_workers=0,
                             cache_dir=CACHE, seed=0)._val_order(dm.val_dataset)
    assert order == again, "val subset must be stable across runs"

    # With more than one family available, an early slice must contain more than one.
    full = ContraDataModule(shard_dir=SHARD_DIR, configs=CONFIGS_ON_DISK, win_len=32,
                            image_size=64, batch_size=2, num_workers=0,
                            cache_dir=CACHE, seed=0)
    ds = full.val_dataset
    head = full._val_order(ds)[:400]
    fams = {ds.index[ds.windows[i][0]]["family"] for i in head}
    assert len(fams) == len(CONFIGS_ON_DISK), f"early val slice only covers {fams}"


def test_family_weights_flatten_the_mix(dataset):
    w0 = family_weights(dataset.index, 0.0)
    w1 = family_weights(dataset.index, 1.0)
    assert np.allclose(w0, w0[0])                 # alpha=0 is uniform over episodes
    assert np.isclose(w1.sum(), 1.0)
    # A rarer family must not be down-weighted relative to a common one.
    fams = [ep["family"] for ep in dataset.index]
    if len(set(fams)) > 1:
        by_fam = {f: w1[[i for i, x in enumerate(fams) if x == f]].sum()
                  for f in set(fams)}
        assert min(by_fam.values()) > 0


# ── model + loss ──────────────────────────────────────────────────────────────

def _fake_batch(b=2, t=8, s=64, per_timestep_goal=False):
    goal_shape = (b, t, s, s, 3) if per_timestep_goal else (b, s, s, 3)
    mask_shape = (b, t, s, s) if per_timestep_goal else (b, s, s)
    a = TINY_MODEL["aux_size"]
    return {
        "image": torch.randint(0, 255, (b, t, s, s, 3), dtype=torch.uint8),
        "cross_view": {
            "cross_view_image": torch.randint(0, 255, goal_shape, dtype=torch.uint8),
            "cross_view_obj_mask": torch.randint(0, 255, mask_shape, dtype=torch.uint8),
            "cross_view_obj_id": torch.randint(0, 5, (b, t)),
        },
        "prev_action": torch.randint(0, NUM_ACTIONS, (b, t)),
        "prev_action_dropout": (torch.rand(b, t) < 0.25).float(),
        "action": torch.randint(0, NUM_ACTIONS, (b, t)),
        "exist": (torch.rand(b, t) < 0.7).float(),
        "point": torch.rand(b, t, 2),
        "goal_heatmap": torch.rand(b, t, a, a),
        "mask": torch.ones(b, t),
        "first": torch.zeros(b, t, dtype=torch.bool),
        "family": torch.randint(0, len(FAMILIES), (b,)),
    }


def test_forward_shapes_and_token_layout():
    model = CrossViewContraRocket(**TINY_MODEL)
    assert model.num_step_tokens == TINY_MODEL["num_view_tokens"] + 2   # +interaction +prev_action
    # The block is [view..., interaction, prev_action], so the interaction token is
    # at -2. It was -3 (the last view token), which causal attention places *before*
    # the interaction token — the aux head could never see the task kind.
    assert model.index_bias == -2
    batch = _fake_batch()
    latents, memory = model(batch)
    b, t, a = 2, 8, TINY_MODEL["aux_size"]
    assert latents["pi_logits"].shape == (b, t, NUM_ACTIONS)
    assert latents["vpred"].shape == (b, t, 1)
    assert latents["goal_heatmap"].shape == (b, t, a, a)
    # These two are the evaluator's contract (contra_eval/policies.py reads them
    # alongside pi_logits, and applies its own sigmoid to `exist`).
    assert latents["exist"].shape == (b, t, 1)
    assert latents["point"].shape == (b, t, 2)
    assert len(memory) == 3 * TINY_MODEL["num_layers"]


def test_index_bias_lands_on_the_interaction_token():
    """Both with and without the prev-action token, not just the default config."""
    with_prev = CrossViewContraRocket(**TINY_MODEL)
    without = CrossViewContraRocket(**{**TINY_MODEL, "use_prev_action": False})
    # Block layout is [view * n, interaction] (+ prev_action). Counting from the end,
    # the interaction token is at -2 with the prev-action token and -1 without it.
    assert with_prev.index_bias == -2
    assert without.index_bias == -1
    for m in (with_prev, without):
        n_view = TINY_MODEL["num_view_tokens"]
        assert m.num_step_tokens + m.index_bias == n_view, "index_bias is off the block"


def test_heatmap_readout_inverts_the_target_rendering():
    """A one-hot heatmap at cell (row, col) must read back as (col/A, row/A).

    This is the soft-argmax convention that keeps `point_err_px` comparable with the
    pre-heatmap runs: goal_mask centres a blob at cx = x_norm * A exactly, so the
    readout must not add the usual half-cell offset.
    """
    model = CrossViewContraRocket(**TINY_MODEL)
    a = TINY_MODEL["aux_size"]
    heat = torch.full((1, 1, a, a), -30.0)
    row, col = 2, 5
    heat[0, 0, row, col] = 30.0
    point, exist = model.heatmap_readout(heat)
    assert torch.allclose(point[0, 0], torch.tensor([col / a, row / a]), atol=1e-3)
    assert exist[0, 0, 0] > 0        # a peak means "present", pre-sigmoid


def test_empty_heatmap_reads_as_absent():
    model = CrossViewContraRocket(**TINY_MODEL)
    a = TINY_MODEL["aux_size"]
    _point, exist = model.heatmap_readout(torch.full((1, 1, a, a), -30.0))
    assert exist[0, 0, 0] < 0        # sigmoid(exist) < 0.5 → absent


def test_action_class_weights_are_data_normalised():
    counts = np.array([680, 100, 90, 55, 27] + [0] * (NUM_ACTIONS - 5))
    flat = action_class_weights(counts, alpha=0.0)
    assert torch.allclose(flat, torch.ones(NUM_ACTIONS))

    w = action_class_weights(counts, alpha=0.5)
    f = torch.as_tensor(counts, dtype=torch.double) / counts.sum()
    # E_data[w] == 1, so bc_weight means the same thing at any alpha. No class here
    # hits the cap, so normalisation survives it.
    assert abs(float((f * w.double()).sum()) - 1.0) < 1e-6
    # The rare seen class is upweighted relative to the dominant one.
    assert w[4] > w[0]
    # Unseen classes must not absorb mass; they are never a target anyway.
    assert torch.allclose(w[5:], w[5:][0])


def test_ultra_rare_actions_do_not_dominate_the_weighting():
    """The real histogram's tail must not take the largest weights.

    Nine of the 21 actions are unusable — LF/U/UF have 30/22/12 examples in 692k
    steps. Weighting purely by inverse frequency hands those three the top weights in
    the table, and capping *before* the E_data[w]=1 normalisation does not stop it:
    the normalisation scales everything back up past the cap.
    """
    counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
    counts[:12] = [471904, 69213, 62175, 38385, 18885, 11206,
                   9040, 4895, 4131, 839, 736, 706]
    counts[12:15] = [30, 22, 12]                 # LF, U, UF
    w = action_class_weights(counts, alpha=0.5, max_ratio=10.0)
    assert float(w.max()) <= 10.0 + 1e-6
    # The tail must not outrank the classes the weighting actually targets.
    assert w[12:15].max() <= w.max()
    # UR (index 1) is what this exists to lift, relative to R (index 0).
    assert w[1] > w[0]
    assert 1.5 < float(w[1] / w[0]) < 6.0, "weighting is either inert or extreme"


def test_first_flag_cuts_the_carried_memory():
    """With first=True the window must ignore whatever memory it was handed."""
    torch.manual_seed(0)
    model = CrossViewContraRocket(**TINY_MODEL).eval()
    batch = _fake_batch(b=2, t=8)
    with torch.no_grad():
        _out, memory = model(batch)
        fresh = {**batch, "first": torch.ones(2, 8, dtype=torch.bool)}
        a, _ = model(fresh, memory)          # carried memory, but flagged as a reset
        b, _ = model(fresh, None)            # no memory at all
        c, _ = model(batch, memory)          # carried memory, no reset
    assert torch.allclose(a["pi_logits"], b["pi_logits"], atol=1e-5)
    assert not torch.allclose(c["pi_logits"], b["pi_logits"], atol=1e-5), (
        "carrying memory changed nothing — the memory path is dead")


def test_backward_reaches_every_trainable_parameter():
    """Only the value head may be gradient-free — it is unused by the BC objective.

    Anything else showing up here means a token or a head has been detached from the
    graph, which trains silently and wrongly.
    """
    model = CrossViewContraRocket(**TINY_MODEL)
    batch = _fake_batch()
    latents, _ = model(batch)
    loss, metrics = ContraObjective(families=FAMILIES)(latents, batch)
    loss.backward()
    dead = {n for n, p in model.named_parameters() if p.requires_grad and p.grad is None}
    assert dead == {"value_head.weight", "value_head.bias"}, f"unexpected dead params: {dead}"
    assert set(metrics) >= {"loss", "bc_loss", "bc_acc", "point_err_px", "exist_acc",
                            "heatmap_loss"}
    # Per-family split: every family present in the batch reports, with its base rate.
    for i in set(batch["family"].tolist()):
        assert f"{FAMILIES[i]}/bc_acc" in metrics
        assert f"{FAMILIES[i]}/goal_vis" in metrics


def test_per_window_goal_matches_per_timestep_goal():
    """Encoding the goal once and expanding must equal encoding T copies of it.

    This equivalence is the entire justification for the dataset shipping the goal
    without a time axis, so it is asserted rather than assumed.
    """
    torch.manual_seed(0)
    model = CrossViewContraRocket(**TINY_MODEL).eval()
    batch = _fake_batch(b=2, t=8)
    tiled = {k: v for k, v in batch.items()}
    cv = batch["cross_view"]
    tiled["cross_view"] = {
        "cross_view_image": cv["cross_view_image"].unsqueeze(1).expand(-1, 8, -1, -1, -1),
        "cross_view_obj_mask": cv["cross_view_obj_mask"].unsqueeze(1).expand(-1, 8, -1, -1),
        "cross_view_obj_id": cv["cross_view_obj_id"],
    }
    with torch.no_grad():
        a, _ = model(batch)
        b, _ = model(tiled)
    for key in ("pi_logits", "point", "exist", "goal_heatmap"):
        assert torch.allclose(a[key], b[key], atol=1e-5), key


def test_no_unused_encoder_head_is_built():
    """`ConvEncoder.head` is 16.8M params the policy never calls — from-scratch
    encoders must not allocate it."""
    model = CrossViewContraRocket(**TINY_MODEL)
    assert model.view_backbone.head is None
    assert model.mask_backbone.head is None


def test_frozen_backbone_gets_no_gradient(tmp_path):
    ckpt = os.path.expanduser("~/code/contra_agent/tmp/dreamer/ae_pretrained.pt")
    if not os.path.exists(ckpt):
        pytest.skip("pretrained encoder checkpoint not available")
    cfg = dict(TINY_MODEL, image_size=256, view_depth=32, view_backbone_ckpt=ckpt)
    model = CrossViewContraRocket(**cfg)
    assert all(not p.requires_grad for p in model.view_backbone.parameters())
    assert all(p.requires_grad for p in model.mask_backbone.parameters())
    # The unused embedding head must be dropped after loading, not carried frozen:
    # it is 16.8M params, 60% of the pretrained checkpoint.
    assert model.view_backbone.head is None
    frozen = sum(p.numel() for p in model.view_backbone.parameters())
    assert frozen < 12e6, f"{frozen/1e6:.1f}M frozen — the unused head is still attached"


def test_padding_does_not_change_the_loss():
    """A window padded to 2x length must give the same masked loss as the real part."""
    torch.manual_seed(0)
    a = TINY_MODEL["aux_size"]
    objective = ContraObjective()
    real = _fake_batch(b=1, t=4)
    latents = {"pi_logits": torch.randn(1, 4, NUM_ACTIONS), "vpred": torch.randn(1, 4, 1),
               "exist": torch.randn(1, 4, 1), "point": torch.randn(1, 4, 2),
               "goal_heatmap": torch.randn(1, 4, a, a)}
    loss_real, _ = objective(latents, real)

    # `family` has no time axis, so it must not be padded along dim 1 with the rest.
    pad = {k: (torch.cat([v, torch.zeros_like(v)], dim=1) if torch.is_tensor(v) else v)
           for k, v in real.items() if k not in ("cross_view", "family")}
    pad["family"] = real["family"]
    pad["mask"] = torch.cat([real["mask"], torch.zeros(1, 4)], dim=1)
    latents_pad = {k: torch.cat([v, torch.randn_like(v)], dim=1) for k, v in latents.items()}
    loss_pad, _ = objective(latents_pad, pad)
    assert torch.allclose(loss_real, loss_pad, atol=1e-6)
