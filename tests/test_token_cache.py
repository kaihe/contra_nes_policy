"""The frozen-encoder token cache — doc/0013 §2.3.

The cache exists to delete 97% of a training step, and its one dangerous failure is
silence: a cache built by a different encoder still loads, still trains, and still
produces a plausible loss curve. So the tests that matter most here are not about
throughput — they pin that a cached token *equals* a live one, and that a mismatched
encoder or image size raises instead of being tolerated.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from contra_encoder.net import build_encoder
from contra_policy.token_cache import (CACHE_VERSION, StaleCache, TokenCache,
                                       build_token_cache, encoder_fingerprint)


S, DIM = 32, 512


class _FakeDataset:
    """Stands in for ContraCrossViewDataset: deterministic pixels, no tar or codec.

    The real decode path is exercised by the shard tests; what is under test here is the
    cache's layout, keying and round-trip, and those must be checkable without 6 GB of
    shards on disk.
    """

    def __init__(self, lengths):
        self.lengths = lengths

    def _pixels(self, uid, k):
        rng = np.random.default_rng(abs(hash((uid, k))) % (2 ** 32))
        return rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)

    def goal_image(self, ep):
        return self._pixels(ep["uid"], -1)

    def frames(self, ep, start, count):
        return np.stack([self._pixels(ep["uid"], j)
                         for j in range(start, start + count)])


@pytest.fixture(scope="module")
def encoder():
    torch.manual_seed(0)
    return build_encoder(image_size=S, hiddim=DIM, depth=4, minres=4,
                         proj_ch=16, aux_size=8, head_depth=8).eval()


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory, encoder):
    path = tmp_path_factory.mktemp("enc") / "encoder-final.pt"
    encoder.save(str(path))
    return str(path)


def _index(lengths):
    return [{"uid": f"ep{i}", "tar": "shard.tar", "length": n}
            for i, n in enumerate(lengths)]


@pytest.fixture(scope="module")
def cache(tmp_path_factory, encoder, ckpt):
    lengths = [5, 1, 12, 3]
    out = str(tmp_path_factory.mktemp("cache") / "tokens")
    build_token_cache(_index(lengths), encoder, out, encoder_ckpt=ckpt, image_size=S,
                      dataset=_FakeDataset(lengths), device="cpu", log_every=0)
    return out, lengths, ckpt


# ── the property the cache exists to preserve ────────────────────────────────

def test_cached_tokens_equal_a_live_encoder_forward(cache, encoder):
    """The whole point. If this drifts, every number in the 0013 grid is wrong."""
    out, lengths, ckpt = cache
    tc = TokenCache(out, encoder_sha256=encoder_fingerprint(ckpt), image_size=S)
    ds = _FakeDataset(lengths)

    for i, n in enumerate(lengths):
        ep = {"uid": f"ep{i}", "tar": "shard.tar", "length": n}
        with torch.no_grad():
            live_goal = encoder.encode(torch.from_numpy(ds.goal_image(ep)[None]))[0]
            live_frames = encoder.encode(torch.from_numpy(ds.frames(ep, 0, n)))

        # fp16 storage: 1.9e-3 max error against the fp32 reference on real frames, and
        # the core's first matmul is bf16 (~8e-3 near 1.0), so this tolerance is far
        # tighter than anything the model can resolve. See token_cache.__doc__.
        assert np.allclose(tc.goal(f"ep{i}"), live_goal.numpy(), atol=5e-3)
        assert np.allclose(tc.frames(f"ep{i}"), live_frames.numpy(), atol=5e-3)


def test_goal_row_is_not_mistaken_for_a_frame(cache, encoder):
    """Row 0 of an episode block is the goal; an off-by-one here is silent and fatal."""
    out, lengths, _ = cache
    tc = TokenCache(out)
    ds = _FakeDataset(lengths)
    ep = {"uid": "ep0", "tar": "shard.tar", "length": lengths[0]}
    with torch.no_grad():
        live_first_frame = encoder.encode(torch.from_numpy(ds.frames(ep, 0, 1)))[0]
    assert np.allclose(tc.frames("ep0", 0, 1)[0], live_first_frame.numpy(), atol=5e-3)
    assert not np.allclose(tc.goal("ep0"), live_first_frame.numpy(), atol=5e-3)


# ── keying: the failure that would poison a whole sweep ──────────────────────

def test_a_different_encoder_raises_rather_than_loading(cache):
    out, _, _ = cache
    with pytest.raises(StaleCache, match="encoder_sha256"):
        TokenCache(out, encoder_sha256="0" * 64)


def test_a_different_image_size_raises(cache):
    out, _, _ = cache
    with pytest.raises(StaleCache, match="image_size"):
        TokenCache(out, image_size=S * 2)


def test_an_old_layout_version_raises(cache, tmp_path):
    out, _, _ = cache
    stale = tmp_path / "stale"
    stale.mkdir()
    os.symlink(os.path.join(out, "tokens.npy"), stale / "tokens.npy")
    meta = json.load(open(os.path.join(out, "meta.json")))
    meta["version"] = CACHE_VERSION - 1
    json.dump(meta, open(stale / "meta.json", "w"))
    with pytest.raises(StaleCache, match="layout"):
        TokenCache(str(stale))


def test_unchecked_construction_is_allowed_but_records_the_key(cache, ckpt):
    """No key passed = no check, so tools can inspect a cache. The key is still on disk."""
    out, _, _ = cache
    tc = TokenCache(out)
    assert tc.meta["encoder_sha256"] == encoder_fingerprint(ckpt)
    assert tc.meta["dtype"] == "float16"


# ── layout and windowing ─────────────────────────────────────────────────────

def test_layout_is_contiguous_and_has_no_unused_rows(cache):
    out, lengths, _ = cache
    tc = TokenCache(out)
    assert len(tc) == len(lengths)
    assert tc.tokens.shape == (sum(lengths) + len(lengths), DIM)
    offsets = [tc.meta["episodes"][i]["offset"] for i in range(len(lengths))]
    assert offsets == list(np.cumsum([0] + [n + 1 for n in lengths[:-1]]))


def test_windowing_matches_the_loader(cache):
    """`frames(uid, start, count)` indexes like ContraCrossViewDataset.frames."""
    out, lengths, _ = cache
    tc = TokenCache(out)
    whole = tc.frames("ep2")
    assert whole.shape == (lengths[2], DIM)
    assert np.array_equal(tc.frames("ep2", 3, 4), whole[3:7])
    assert np.array_equal(tc.frames("ep2", 10, 99), whole[10:])   # clipped, not padded
    assert tc.frames("ep2", 99, 4).shape == (0, DIM)


def test_single_frame_episode_round_trips(cache):
    out, lengths, _ = cache
    tc = TokenCache(out)
    assert tc.length("ep1") == 1
    assert tc.frames("ep1").shape == (1, DIM)


def test_rebuilding_over_an_existing_cache_refuses(cache, encoder, ckpt):
    out, lengths, _ = cache
    with pytest.raises(FileExistsError):
        build_token_cache(_index(lengths), encoder, out, encoder_ckpt=ckpt, image_size=S,
                          dataset=_FakeDataset(lengths), device="cpu", log_every=0)


def test_fingerprint_is_the_file_hash(ckpt, tmp_path):
    """A sibling repo must be able to verify the key without importing torch."""
    import hashlib
    expected = hashlib.sha256(open(ckpt, "rb").read()).hexdigest()
    assert encoder_fingerprint(ckpt) == expected
