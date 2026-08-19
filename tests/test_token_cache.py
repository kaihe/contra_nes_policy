"""The frozen-encoder token cache — see ``contra_policy.token_cache``.

The cache exists to delete 97% of a training step, and its one dangerous failure is
silence: a cache built by a different encoder still loads, still trains, and still
produces a plausible loss curve. So the tests that matter most here are not about
throughput — they pin that a cached token *equals* a live one, and that a mismatched
encoder or image size raises instead of being tolerated.
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pytest
import torch

from contra_encoder.net import build_encoder
from contra_policy.action_space import ACTION_VECTORS, NUM_ACTIONS
from contra_policy.token_cache import (CACHE_VERSION, CachedEpisodeDataset, StaleCache,
                                       TokenCache, build_token_cache,
                                       encoder_fingerprint)


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

    def actions_of(self, ep):
        """Real 21-action indices, deterministic per uid — `vectors_to_indices` is strict."""
        rng = np.random.default_rng(abs(hash(("act", ep["uid"]))) % (2 ** 32))
        return rng.integers(0, NUM_ACTIONS, size=int(ep["length"]), dtype=np.int64)

    def _read(self, ep, ext):
        """The builder's only tar accessor, so the fake shard lives entirely here."""
        if ext == "actions.npy":
            buf = io.BytesIO()
            np.save(buf, np.asarray(ACTION_VECTORS)[self.actions_of(ep)].astype(np.uint8))
            return buf.getvalue()
        if ext == "json":
            return json.dumps({"goal_when": "boss"}).encode()
        raise KeyError(ext)


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
    return [{"uid": f"ep{i}", "tar": "shard.tar", "family": "boss", "length": n}
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
    """The whole point. If this drifts, every number in the 0013 ladder is wrong."""
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


# ── the cached training path ─────────────────────────────────────────────────

def test_cached_dataset_reproduces_the_action_shift(cache):
    """frames[j] is the state AFTER actions[j], so its target is actions[j+1].

    Pins the copy of that convention in CachedEpisodeDataset against the raw array. If
    the two ever drift, a cached run trains on inverse dynamics and still looks healthy.
    """
    out, lengths, _ = cache
    tc = TokenCache(out)
    ds = CachedEpisodeDataset(tc, ["ep2"])
    item = ds[0]
    raw = tc.raw_actions("ep2")
    n = int(item["mask"].sum())
    assert n == lengths[2] - 1                      # last frame has no action FROM it
    assert np.array_equal(item["action"][:n].numpy(), raw[1:1 + n])
    assert item["tokens"].shape == (lengths[2] - 1, DIM)
    assert item["tokens"].dtype == torch.float32    # fp32: see forward_tokens' docstring


def test_padded_steps_carry_no_supervision(cache):
    """A short episode padded into a batch must contribute zero gradient."""
    from contra_policy.dataset import pad_episodes
    out, lengths, _ = cache
    ds = CachedEpisodeDataset(TokenCache(out), ["ep3", "ep2"])      # lengths 3 and 12
    batch = pad_episodes([ds[0], ds[1]])
    assert batch["tokens"].shape[1] == lengths[2] - 1               # padded to the longer
    short, long = batch["mask"][0], batch["mask"][1]
    assert short[:lengths[3] - 1].sum() == lengths[3] - 1
    assert short[lengths[3] - 1:].sum() == 0
    pad = batch["tokens"][0, lengths[3] - 1:]
    assert torch.equal(pad, torch.zeros_like(pad))
    assert long.sum() == lengths[2] - 1
    assert batch["seq_len"].tolist() == [lengths[3] - 1, lengths[2] - 1]


def test_a_length_one_episode_is_rejected_exactly_as_the_live_loader_rejects_it(cache):
    """T = max(1, L-1) = 1 but n = min(T, L-1) = 0: nothing to supervise.

    `ContraCrossViewDataset.__getitem__` asserts on this too. Pinned so the cached path
    cannot start quietly tolerating an episode the live path refuses — real episodes run
    24-1038 decisions, so hitting this means the index is wrong, not the data.
    """
    out, _, _ = cache
    ds = CachedEpisodeDataset(TokenCache(out), ["ep1"])
    with pytest.raises(AssertionError, match="no supervisable step"):
        ds[0]


# ── one cache per release, selected by uid (doc/0015 §2) ─────────────────────

def test_a_data_cell_is_a_uid_subset_of_the_release_cache(cache):
    """The property the whole data axis rests on: prefixes nest, so one cache serves all.

    Six per-cell caches for the 20k release would re-encode the same frames 2.1x for
    tokens that are identical by construction. Instead the cache is built over the longest
    prefix and `shard_count` selects the cell — so a subset must be exactly the same
    episodes, in the requested order, and must not see the ones it excluded.
    """
    out, lengths, _ = cache
    tc = TokenCache(out)
    full = CachedEpisodeDataset(tc)
    cell = CachedEpisodeDataset(tc, ["ep2", "ep0"])       # a smaller cell, reordered
    assert len(full) == len(lengths) and len(cell) == 2
    assert cell.uids == ["ep2", "ep0"]
    assert torch.equal(cell[0]["action"], full[2]["action"])
    assert torch.equal(cell[1]["tokens"], full[0]["tokens"])


def test_a_cell_the_cache_does_not_cover_raises_before_training(cache):
    """Pointing a 27-shard cell at a 13-shard cache is the mistake this design invites.

    It must fail at construction with the count, not 40 minutes in with a bare KeyError
    naming one uid — and not silently, by training the big cell on the small cell's data.
    """
    out, _, _ = cache
    with pytest.raises(StaleCache, match="not in the cache"):
        CachedEpisodeDataset(TokenCache(out), ["ep0", "ep_from_a_later_shard"])


def test_forward_tokens_equals_forward_on_the_same_pixels(encoder):
    """The two model entry points must agree, or the cache changes the model."""
    from contra_policy.model import PolicyConfig, build_policy

    torch.manual_seed(0)
    policy = build_policy(PolicyConfig(
        core={"d_model": DIM, "n_layer": 2, "n_head": 4, "n_kv_head": 4, "context": 64},
        encoder_ckpt=None, encoder=dict(image_size=S, hiddim=DIM, depth=4, minres=4,
                                        proj_ch=16, aux_size=8, head_depth=8),
        freeze_encoder=True, value_head=False, aux_size=0)).eval()

    rng = np.random.default_rng(0)
    images = torch.from_numpy(rng.integers(0, 256, (2, 6, S, S, 3), dtype=np.uint8))
    goal = torch.from_numpy(rng.integers(0, 256, (2, S, S, 3), dtype=np.uint8))
    inter = torch.tensor([0, 1])

    with torch.no_grad():
        live = policy(images, goal, inter)["pi_logits"]
        tokens = policy.encode_images(images)
        goal_tok = policy.encode_images(goal.unsqueeze(1)).squeeze(1)
        cached = policy.forward_tokens(tokens, goal_tok, inter)["pi_logits"]
    assert torch.equal(live, cached)


RELEASE = os.path.expanduser(
    "~/code/contra_nes_data/game_trace/releases/boss-spread-10k-v1/manifest.json")
VAL_SHA = "29fd40177d9277931d1a115d8a36171c7eacc1a02c17cf1a4f7093ac94cc9ae0"
VAL_CACHE = "cache/tokens/spread10k-val"


@pytest.mark.skipif(not (os.path.exists(RELEASE) and os.path.isdir(VAL_CACHE)),
                    reason="needs boss-spread-10k-v1 and its built val cache")
def test_cached_batch_matches_the_live_loader():
    """The equivalence the whole cached path rests on, on real shards.

    The synthetic tests above pin layout and keying; this one pins that a cached batch
    *is* a live batch — same targets, same mask, same padding, same loss. Everything
    discrete must be bit-identical; only the tokens may differ, and only by fp16.
    """
    from contra_policy.dataset import (ContraCrossViewDataset, load_or_build_index,
                                       pad_episodes, scaling_release)
    idx = load_or_build_index(scaling_release(RELEASE, 1, VAL_SHA)["val"],
                              cache_dir="cache")[:8]
    live = ContraCrossViewDataset(idx, whole_episode=True, image_size=256, aux_size=0,
                                  prev_action_keep_prob=0.0, seed=0)
    cached = CachedEpisodeDataset(TokenCache(VAL_CACHE, image_size=256),
                                  [e["uid"] for e in idx])

    lb = pad_episodes([live[i] for i in range(len(idx))])
    cb = pad_episodes([cached[i] for i in range(len(idx))])

    assert torch.equal(lb["action"], cb["action"])
    assert torch.equal(lb["mask"], cb["mask"])
    assert torch.equal(lb["seq_len"], cb["seq_len"])
    assert torch.equal(lb["family"], cb["family"])
    assert torch.equal(lb["cross_view"]["cross_view_obj_id"], cb["interaction"])
    assert lb["image"].shape[:2] == cb["tokens"].shape[:2]


RELEASE_20K = os.path.expanduser(
    "~/code/contra_nes_data/game_trace/releases/boss-spread-20k-v1/manifest.json")
VAL_SHA_20K = "84cc5c509bd1307048e2dee6b497691f71b6b4d3b71a093eae18e279ba6f9bfc"
TRAIN_CACHE_20K = "cache/tokens/spread20k-d27"
#: doc/0015 §2, on boss-spread-20k-v1. Note 27, not 32.
CELLS_20K = [(1, 737), (2, 1474), (4, 2949), (8, 5897), (16, 11793), (27, 19900)]


@pytest.mark.skipif(not (os.path.exists(RELEASE_20K) and os.path.isdir(TRAIN_CACHE_20K)),
                    reason="needs boss-spread-20k-v1 and its built d27 cache")
def test_the_release_cache_covers_every_data_cell():
    """The gate on the data axis: one cache, six cells, no per-cell rebuild.

    Checked against the manifest's own per-shard uid lists rather than a tar index — the
    manifest is what `scaling_release` treats as the authority for membership, and this
    must be answerable without re-indexing 12.6 GB of shards.
    """
    with open(RELEASE_20K) as fh:
        manifest = json.load(fh)
    shards = manifest["train_shards"]
    tc = TokenCache(TRAIN_CACHE_20K, image_size=256)
    for shard_count, episodes in CELLS_20K:
        uids = [u for s in shards[:shard_count] for u in s["uids"]]
        assert len(uids) == episodes, f"D{shard_count} manifest drift"
        missing = [u for u in uids if u not in tc]
        assert not missing, f"D{shard_count}: {len(missing)} uids absent from the cache"
    assert len(tc) == CELLS_20K[-1][1]


@pytest.mark.skipif(not (os.path.exists(RELEASE_20K) and os.path.isdir(TRAIN_CACHE_20K)),
                    reason="needs boss-spread-20k-v1 and its built d27 cache")
def test_the_two_releases_are_disjoint_and_must_not_share_a_cache():
    """boss-spread-20k-v1 regenerated the data; it did not extend boss-spread-10k-v1.

    Recorded as a test because the cheap reading of "the 20k release" is "the 10k release
    plus more", under which the 10k cache would serve its first 13 shards and the two
    curves would join. Zero uids overlap, so it would raise — but the reason it raises
    belongs somewhere a reader can find it.
    """
    with open(RELEASE_20K) as fh:
        uids_20k = {u for s in json.load(fh)["train_shards"] for u in s["uids"]}
    with open(RELEASE) as fh:
        uids_10k = {u for s in json.load(fh)["train_shards"] for u in s["uids"]}
    assert not (uids_10k & uids_20k)


def test_forward_tokens_rejects_a_sequence_over_context(encoder):
    from contra_policy.model import PolicyConfig, build_policy
    policy = build_policy(PolicyConfig(
        core={"d_model": DIM, "n_layer": 1, "n_head": 4, "n_kv_head": 4, "context": 8},
        encoder_ckpt=None, encoder=dict(image_size=S, hiddim=DIM, depth=4, minres=4,
                                        proj_ch=16, aux_size=8, head_depth=8),
        freeze_encoder=True, value_head=False, aux_size=0)).eval()
    with pytest.raises(ValueError, match="context"):
        policy.forward_tokens(torch.zeros(1, 16, DIM), torch.zeros(1, DIM),
                              torch.tensor([0]))


# ── the boss-only scaling config ─────────────────────────────────────────────

def test_scaling_config_actually_disables_the_family_schedule():
    """OmegaConf merges mappings, so `family_draws: {}` would inherit the parent's.

    That inheritance is invisible in the YAML and only surfaces as "fixed-family schedule
    has no episodes for ['kill', 'item', 'traverse']" once a boss-only index is built —
    after the cache, the model and the release checks have all already been paid for.
    """
    from hydra import compose, initialize_config_module

    with initialize_config_module("contra_policy", version_base=None):
        cfg = compose(config_name="config_bc_scaling")
    assert cfg.loader.family_draws is None
    assert list(cfg.families) == ["boss"]
    assert cfg.loader.batch_size == 32
    assert cfg.loader.num_workers == 2
    assert cfg.policy.freeze_encoder is True      # token_cache requires it
    assert cfg.boss_scaling.shard_count == 13
    assert dict(cfg.loader.family_draws or {}) == {}


def test_the_20k_config_keeps_the_recipe_and_changes_only_the_release():
    """The data axis moves to a new release; nothing else in the recipe may move with it.

    A data-scaling curve is only readable if every cell shares one recipe, so this pins
    that `config_bc_scaling_20k` inherits boss-only / batch 32 / no family schedule
    verbatim and differs from its parent in exactly the release, its validation SHA and
    the cache paths.
    """
    from hydra import compose, initialize_config_module

    with initialize_config_module("contra_policy", version_base=None):
        base = compose(config_name="config_bc_scaling")
        cfg = compose(config_name="config_bc_scaling_20k")

    assert "boss-spread-20k-v1" in str(cfg.boss_scaling.manifest)
    assert cfg.boss_scaling.shard_count == 27
    assert cfg.boss_scaling.validation_sha256 == VAL_SHA_20K
    assert cfg.boss_scaling.validation_sha256 != base.boss_scaling.validation_sha256
    # Unresolved: the paths interpolate `${hydra:runtime.cwd}`, which only exists inside a
    # launched job, not in a bare `compose`.
    from omegaconf import OmegaConf
    paths = OmegaConf.to_container(cfg.token_cache, resolve=False)
    assert str(paths["train"]).endswith("spread20k-d27")
    assert str(paths["val"]).endswith("spread20k-val")

    assert list(cfg.families) == ["boss"]
    assert cfg.loader.family_draws is None
    assert cfg.loader.batch_size == base.loader.batch_size == 32
    assert cfg.loader.num_workers == base.loader.num_workers == 2
    assert cfg.policy.freeze_encoder is True
    assert cfg.policy.core.dropout == base.policy.core.dropout
    assert cfg.train.steps == base.train.steps
    assert cfg.train.lr_decay == base.train.lr_decay == "wsd"


def test_0020_laser_config_pins_the_requested_recipe():
    """The size ladder must not silently inherit Spread or the old LR."""
    from hydra import compose, initialize_config_module

    with initialize_config_module("contra_policy", version_base=None):
        cfg = compose(config_name="config_bc_laser", overrides=[
            "policy.core.d_model=1024", "policy.core.n_layer=8",
            "policy.core.n_head=16", "policy.core.n_kv_head=16",
        ])

    assert cfg.datahouse.weapon == "laser"
    assert cfg.datahouse.shard_count == 18
    assert cfg.loader.batch_size == 32
    assert cfg.loader.num_workers == 2
    assert cfg.train.steps == 20_000
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert cfg.train.lr_decay == "wsd"
    assert cfg.policy.core.d_model == 1024
    assert cfg.policy.core.n_layer == 8


def test_0021_laser_20k_config_changes_only_the_data_prefix():
    from hydra import compose, initialize_config_module

    with initialize_config_module("contra_policy", version_base=None):
        base = compose(config_name="config_bc_laser")
        cfg = compose(config_name="config_bc_laser_20k")

    assert cfg.datahouse.weapon == base.datahouse.weapon == "laser"
    assert cfg.datahouse.shard_count == 36
    assert base.datahouse.shard_count == 18
    assert cfg.loader.batch_size == base.loader.batch_size == 32
    assert cfg.train.steps == base.train.steps == 20_000
    assert cfg.train.learning_rate == base.train.learning_rate == pytest.approx(1e-4)


def test_0021_laser_40k_config_uses_the_full_store():
    from hydra import compose, initialize_config_module

    with initialize_config_module("contra_policy", version_base=None):
        base = compose(config_name="config_bc_laser")
        cfg = compose(config_name="config_bc_laser_40k")

    assert cfg.datahouse.weapon == base.datahouse.weapon == "laser"
    assert cfg.datahouse.shard_count is None
    assert cfg.loader.batch_size == base.loader.batch_size == 32
    assert cfg.train.steps == base.train.steps == 20_000
    assert cfg.train.learning_rate == base.train.learning_rate == pytest.approx(1e-4)


def test_multiweapon_prefix_selects_each_weapons_own_shards():
    from contra_policy.datahouse import shard_prefix

    class Store:
        slice = (1, "boss", ("spread", "laser"))
        # Catalog order is by weapon, not by a global mixed-data ordinal.
        shard_weapons = ["laser", "laser", "laser", "spread", "spread"]
        shard_ordinals = [0, 1, 2, 0, 1]
        shards = list(range(5))
        positions = {"l0": 0, "l1": 1, "l2": 2, "s0": 3, "s1": 4}

        def shard_index(self, uid):
            return self.positions[uid]

    got = shard_prefix(Store(), list(Store.positions), {"laser": 2, "spread": 1})
    assert got == ["l0", "l1", "s0"]


def test_multiweapon_prefix_requires_one_limit_per_weapon():
    from contra_policy.datahouse import shard_prefix

    class Store:
        slice = (1, "boss", ("spread", "laser"))
        shard_weapons = ["laser", "spread"]
        shard_ordinals = [0, 0]
        shards = list(range(2))

    with pytest.raises(ValueError, match="name exactly"):
        shard_prefix(Store(), [], {"laser": 1})


def test_0022_mixed_d10_config_pins_both_weapon_prefixes_and_final_only():
    from hydra import compose, initialize_config_module

    with initialize_config_module("contra_policy", version_base=None):
        cfg = compose(config_name="config_bc_mixed_d10")

    assert list(cfg.datahouse.weapon) == ["spread", "laser"]
    assert cfg.datahouse.shard_count is None
    assert dict(cfg.datahouse.shard_counts) == {"spread": 13, "laser": 18}
    assert cfg.train.steps == 20_000
    assert cfg.train.learning_rate == pytest.approx(1e-4)
    assert list(cfg.train.save_steps) == []
    assert cfg.train.save_best is False


@pytest.mark.parametrize("shard_count,episodes", CELLS_20K)
def test_every_20k_data_cell_composes_as_a_hydra_override(shard_count, episodes):
    from hydra import compose, initialize_config_module
    with initialize_config_module("contra_policy", version_base=None):
        cfg = compose(config_name="config_bc_scaling_20k",
                      overrides=[f"boss_scaling.shard_count={shard_count}"])
    assert cfg.boss_scaling.shard_count == shard_count


# ── the model-size ladder (doc/0013 §2) ──────────────────────────────────────

LADDER = [("XS", 256, 2, 4), ("S", 384, 3, 6), ("M", 512, 4, 8),
          ("L", 640, 5, 10), ("XL", 768, 6, 12), ("XXL", 1024, 8, 16)]


def _policy(d, n_layer, n_head):
    from contra_policy.model import PolicyConfig, build_policy
    return build_policy(PolicyConfig(
        core=dict(d_model=d, n_layer=n_layer, n_head=n_head, n_kv_head=n_head,
                  context=64),
        encoder_ckpt=None, encoder=dict(image_size=S, hiddim=DIM, depth=4, minres=4,
                                        proj_ch=16, aux_size=8, head_depth=8),
        freeze_encoder=True, value_head=False, aux_size=0))


@pytest.mark.parametrize("name,d,n_layer,n_head", LADDER)
def test_every_ladder_cell_is_expressible_and_runs(name, d, n_layer, n_head):
    """`core.d_model` used to be silently overridden to the encoder's 512.

    That is the worst kind of bug for a scaling sweep: L/XL/XXL would all have trained
    at d=512 with only depth varying, every run would have succeeded, and the resulting
    flat curve would have looked like a finding.
    """
    pol = _policy(d, n_layer, n_head).eval()
    assert pol.core.cfg.d_model == d, "d_model must not be pinned to the encoder width"
    assert d // n_head == 64, "the ladder holds head_dim at 64"
    with torch.no_grad():
        out = pol.forward_tokens(torch.zeros(2, 5, DIM), torch.zeros(2, DIM),
                                 torch.tensor([0, 1]))
    from contra_policy.action_space import NUM_ACTIONS
    assert out["pi_logits"].shape == (2, 5, NUM_ACTIONS)


def test_the_M_cell_adds_no_parameters_and_keeps_its_state_dict():
    """d_core == d_enc must stay bit-compatible with every pre-ladder checkpoint."""
    import torch.nn as nn
    m = _policy(DIM, 4, 8)
    assert isinstance(m.in_proj, nn.Identity)
    assert sum(p.numel() for p in m.in_proj.parameters()) == 0
    assert not [k for k in m.state_dict() if k.startswith("in_proj")]


def test_a_wider_core_projects_rather_than_truncating():
    import torch.nn as nn
    m = _policy(1024, 8, 16)
    assert isinstance(m.in_proj, nn.Linear)
    assert m.in_proj.weight.shape == (1024, DIM) and m.in_proj.bias is None


@pytest.mark.parametrize("name,d,n_layer,n_head", LADDER)
def test_every_ladder_cell_composes_as_a_hydra_override(name, d, n_layer, n_head):
    """`core.d_model` must be a declared key, or the override errors instead of applying.

    OmegaConf refuses to create keys on a struct, so an omitted `d_model` makes
    `policy.core.d_model=1024` a hard failure — which is how the first ladder queue lost
    all three cells. Composing here costs milliseconds; finding out at launch costs a
    scheduling round trip.
    """
    from hydra import compose, initialize_config_module
    with initialize_config_module("contra_policy", version_base=None):
        cfg = compose(config_name="config_bc_scaling", overrides=[
            f"policy.core.d_model={d}", f"policy.core.n_layer={n_layer}",
            f"policy.core.n_head={n_head}", f"policy.core.n_kv_head={n_head}"])
    assert cfg.policy.core.d_model == d
    assert cfg.policy.core.n_layer == n_layer and cfg.policy.core.n_head == n_head


def test_the_unoverridden_scaling_config_is_the_M_cell():
    from hydra import compose, initialize_config_module
    with initialize_config_module("contra_policy", version_base=None):
        cfg = compose(config_name="config_bc_scaling")
    assert cfg.policy.core.d_model is None      # null = take the encoder's width
    assert cfg.policy.core.n_layer == 4 and cfg.policy.core.n_head == 8
