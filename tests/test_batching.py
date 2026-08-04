"""Whole-episode batching: the sampler and collate the policy trains on.

The bug this file exists for was silent and expensive: a 60-batch validation scored 240
of 846 episodes and contained **zero** `traverse` — 65% of all decision steps — because
the index is ordered by tar path and `traverse` starts at index 392. Every reported
validation number excluded the largest family, and nothing crashed.
"""

from __future__ import annotations

import collections
import os

import pytest
import torch

from contra_policy.dataset import LengthGroupedSampler, pack_episodes, pad_episodes


def _lengths(n=800, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.integers(24, 520, n).tolist()


def test_order_is_permuted_even_when_shuffle_is_off():
    """`shuffle=False` must not mean "walk the index in order".

    The index is grouped by family, so a caller scoring a prefix of the batches would
    see whole families vanish — which is exactly what happened.
    """
    n = 400
    s = LengthGroupedSampler(_lengths(n), batch_size=4, pool_batches=32, shuffle=False)
    first = [i for b in list(s)[:20] for i in b]
    assert first != sorted(first)[:len(first)], "batches are in raw index order"
    # A prefix must draw from across the whole index, not just its head.
    assert max(first) > n * 0.6, f"first 20 batches only reach index {max(first)} of {n}"


def test_validation_order_is_identical_across_calls():
    """Representative *and* stable: a subset that changes per call makes the val trend
    unreadable, which is why plain `shuffle=True` is wrong for validation."""
    s = LengthGroupedSampler(_lengths(), batch_size=4, pool_batches=32, shuffle=False)
    assert list(s) == list(s)


def test_training_order_varies_between_epochs():
    s = LengthGroupedSampler(_lengths(), batch_size=4, pool_batches=32, shuffle=True)
    assert list(s) != list(s)


def test_every_episode_appears_exactly_once_per_epoch():
    n = 397                                     # deliberately not a multiple of 4
    s = LengthGroupedSampler(_lengths(n), batch_size=4, pool_batches=8, shuffle=True)
    seen = [i for b in s for i in b]
    assert sorted(seen) == list(range(n))


def test_batches_group_similar_lengths():
    """The point of the sampler: padding is to the batch maximum, so a batch of mixed
    lengths is mostly mask."""
    lens = _lengths(2000)
    s = LengthGroupedSampler(lens, batch_size=8, pool_batches=32, shuffle=True)
    waste = []
    for b in s:
        got = [lens[i] for i in b]
        waste.append(1 - sum(got) / (max(got) * len(got)))
    assert sum(waste) / len(waste) < 0.15, "length grouping is not reducing padding"


# ── collate ──────────────────────────────────────────────────────────────────

def _ep(t, size=4, A=4, fam=0, inter=2):
    return {
        "image": torch.full((t, size, size, 3), t, dtype=torch.uint8),
        "action": torch.arange(t) % 21,
        "mask": torch.ones(t),
        "goal_heatmap": torch.zeros(t, A, A),
        "exist": torch.ones(t),
        "point": torch.zeros(t, 2),
        "n_goal_points": torch.ones(t, dtype=torch.long),
        "cross_view": {
            "cross_view_image": torch.full((size, size, 3), fam, dtype=torch.uint8),
            "cross_view_obj_mask": torch.zeros(size, size, dtype=torch.uint8),
            "cross_view_obj_id": torch.full((t,), inter, dtype=torch.long),
        },
        "family": torch.tensor(fam),
    }


def test_pad_episodes_pads_to_the_batch_maximum_not_the_context():
    """Padding to the model's 1024 context would make ~90% of every batch mask."""
    out = pad_episodes([_ep(5), _ep(12), _ep(9)])
    assert out["image"].shape[:2] == (3, 12)
    assert out["seq_len"].tolist() == [5, 12, 9]


def test_mask_marks_exactly_the_real_steps():
    """Every loss reduces over `mask`; a padded step carries a fabricated action target
    of 0 and must never reach a gradient."""
    out = pad_episodes([_ep(3), _ep(7)])
    assert out["mask"][0].tolist() == [1] * 3 + [0] * 4
    assert out["mask"][1].tolist() == [1] * 7
    assert int(out["mask"].sum()) == 10


def test_interaction_collapses_to_one_id_per_episode():
    """The windowed loader emitted it per timestep; the policy takes a single id."""
    out = pad_episodes([_ep(4, inter=1), _ep(6, inter=3)])
    assert out["cross_view"]["cross_view_obj_id"].shape == (2,)
    assert out["cross_view"]["cross_view_obj_id"].tolist() == [1, 3]


def test_goal_image_is_one_per_episode_with_no_time_axis():
    out = pad_episodes([_ep(4, fam=1), _ep(6, fam=2)])
    assert out["cross_view"]["cross_view_image"].shape[0] == 2
    assert out["family"].tolist() == [1, 2]


def test_pack_episodes_removes_padding_and_keeps_boundaries():
    out = pack_episodes([_ep(3, fam=1), _ep(7, fam=2)])
    assert out["image"].shape[0] == 10
    assert out["action"].shape == (10,)
    assert out["seq_len"].tolist() == [3, 7]
    assert out["family"].tolist() == [1, 2]


@pytest.mark.skipif(
    not os.path.isdir(os.path.expanduser("~/code/contra_nes_data/game_trace/hf")),
    reason="shards are not on this machine")
def test_a_validation_prefix_covers_every_family_on_the_real_index():
    """The regression itself, against the real val split."""
    from contra_policy.dataset import FAMILIES, load_or_build_index, shard_paths

    sd = os.path.expanduser("~/code/contra_nes_data/game_trace/hf")
    idx = load_or_build_index(shard_paths(sd, FAMILIES, "val"), "cache")
    lengths = [max(1, e["length"] - 1) for e in idx]
    s = LengthGroupedSampler(lengths, 4, pool_batches=32, seed=0, shuffle=False)

    seen = collections.Counter()
    for i, b in enumerate(s):
        if i >= 60:
            break
        for j in b:
            seen[idx[j]["family"]] += 1
    for f in FAMILIES:
        assert seen[f] > 0, f"a 60-batch validation contains no {f} episodes"
    # And roughly in proportion — traverse is over half the split.
    total = sum(seen.values())
    true_traverse = sum(1 for e in idx if e["family"] == "traverse") / len(idx)
    assert abs(seen["traverse"] / total - true_traverse) < 0.10
