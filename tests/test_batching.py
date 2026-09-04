"""Whole-episode batching: the sampler and collate the policy trains on.

The bug this file exists for was silent and expensive: a 60-batch validation scored 240
of 846 episodes and contained **zero** `traverse` — 65% of all decision steps — because
the index is ordered by tar path and `traverse` starts at index 392. Every reported
validation number excluded the largest family, and nothing crashed.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os

import pytest
import torch

from contra_policy.dataset import (FixedFamilyBatchSampler, LengthGroupedSampler,
                                   pad_episodes, scaling_release)


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


# ── fixed-family scaling schedule ────────────────────────────────────────────

def _family_index(counts):
    index = []
    for family, count in counts.items():
        index.extend({"family": family, "uid": f"{family}-{i}"} for i in range(count))
    return index


def test_fixed_family_schedule_has_exact_reference_cycle_counts():
    index = _family_index({"kill": 7, "boss": 3})
    sampler = FixedFamilyBatchSampler(
        index, list(range(10, 20)), batch_size=2,
        family_draws={"kill": 6, "boss": 4}, num_batches=5,
        pool_batches=2, seed=4)

    seen = collections.Counter(index[i]["family"] for batch in sampler for i in batch)

    assert seen == {"kill": 6, "boss": 4}


def test_fixed_family_schedule_resume_reconstructs_the_exact_suffix():
    index = _family_index({"kill": 9, "item": 4, "boss": 6})
    kwargs = dict(index=index, lengths=list(range(19)), batch_size=2,
                  family_draws={"kill": 6, "item": 2, "boss": 4},
                  num_batches=17, pool_batches=3, seed=11)
    complete = list(FixedFamilyBatchSampler(**kwargs))
    resumed = list(FixedFamilyBatchSampler(**kwargs, start_batch=7))

    assert resumed == complete[7:]


def test_large_family_is_covered_without_replacement_across_cycles():
    index = _family_index({"boss": 8})
    sampler = FixedFamilyBatchSampler(
        index, [10] * 8, batch_size=1, family_draws={"boss": 3},
        num_batches=6, pool_batches=3, seed=0)
    picks = [batch[0] for batch in sampler]

    assert len(set(picks)) == 6


def test_partial_final_cycle_keeps_total_family_counts_fixed_across_lengths():
    index = _family_index({"kill": 7, "boss": 5})
    kwargs = dict(index=index, batch_size=2,
                  family_draws={"kill": 6, "boss": 4}, num_batches=7,
                  pool_batches=2, seed=3)
    schedules = [FixedFamilyBatchSampler(lengths=lengths, **kwargs)
                 for lengths in (list(range(12)), list(reversed(range(12))))]
    counts = [collections.Counter(index[i]["family"]
                                  for batch in sampler for i in batch)
              for sampler in schedules]

    assert counts[0] == counts[1] == {"kill": 8, "boss": 6}


def test_scaling_release_uses_manifest_membership_and_pins_validation(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    for name in ("boss-train-00000.tar", "boss-train-00001.tar"):
        (hf / name).write_bytes(b"train")
    val = hf / "boss-val-00000.tar"
    val.write_bytes(b"frozen validation")
    digest = hashlib.sha256(val.read_bytes()).hexdigest()
    manifest = {
        "train_scaling_prefixes": [
            {"shard_count": 1, "episodes": 3, "frames": 30,
             "files": ["hf/boss-train-00000.tar"]},
            {"shard_count": 2, "episodes": 6, "frames": 60,
             "files": ["hf/boss-train-00000.tar", "hf/boss-train-00001.tar"]},
        ],
        "validation": {"episodes": 2, "file": "hf/boss-val-00000.tar",
                       "sha256": digest},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))

    got = scaling_release(str(path), 1, digest)

    assert got["train"] == [str(hf / "boss-train-00000.tar")]
    assert got["train_episodes"] == 3
    assert got["val"] == [str(val)]


def test_scaling_release_rejects_a_changed_validation_tar(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    train = hf / "boss-train-00000.tar"
    val = hf / "boss-val-00000.tar"
    train.write_bytes(b"train")
    val.write_bytes(b"changed")
    expected = hashlib.sha256(b"original").hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "train_scaling_prefixes": [
            {"shard_count": 1, "episodes": 1, "frames": 1,
             "files": ["hf/boss-train-00000.tar"]}],
        "validation": {"episodes": 1, "file": "hf/boss-val-00000.tar",
                       "sha256": expected},
    }))

    with pytest.raises(ValueError, match="validation SHA mismatch"):
        scaling_release(str(path), 1, expected)


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
