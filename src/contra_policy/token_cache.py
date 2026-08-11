"""Precomputed frozen-encoder tokens — the inner loop's preprocessing, done once.

A training step spends ~97% of its GPU time turning pixels into tokens and ~3% training
the thing under study::

    PyAV decode      0.192 ms/frame  ─┐ CPU, overlapped
    cv2 resize→256   0.097 ms/frame  ─┘
    encoder.encode   0.229 ms/frame   ← GPU, 286 ms of a 295 ms step
    the core @12.85M                  ← GPU, 8.5 ms

Every one of those encodes is *redundant*. The encoder is frozen, the loader applies no
pixel augmentation, and a run makes 32 (D13) to 420 (D1) passes over the data — so each
frame is decoded and encoded into the identical 512-d vector dozens to hundreds of times.
This module computes it once and memmaps the answer. See ``doc/0013-scaling-grid.md`` §2.3.

Layout, written by :func:`build_token_cache`::

    <dir>/tokens.npy     (rows, 512) float16, memmapped
    <dir>/actions.npy    (total_steps,) int64 — raw action indices, memmapped
    <dir>/meta.json      {version, encoder_sha256, image_size, dim, dtype, episodes: […]}

One episode occupies ``T + 1`` contiguous rows of ``tokens.npy`` starting at its ``offset``:
**row 0 is the goal-frame token**, rows ``1 … T`` are the frame tokens in decision order.
Keeping the goal adjacent means a batch touches one slice per episode rather than two.

``actions.npy`` holds each episode's *raw, unshifted* action indices, and ``meta`` carries
its interaction id and family. With those, :class:`CachedEpisodeDataset` reads **nothing but
this directory** — no tar, no PyAV, no cv2 — which is what lets ``num_workers`` go to 0.
They cost 8 MB against the tokens' 800 MB, and the alternative is parsing a 19 KB episode
JSON 320,000 times a run to recover one integer. The shift that turns raw actions into BC
targets is deliberately *not* baked in: it lives in one place
(``ContraCrossViewDataset.__getitem__``) and is reproduced against that, so a cached run and
a live run cannot disagree about which action a frame is supervised with.

**Storage is float16, and that is not the same choice as the model's compute dtype.**
``ContraFrameEncoder.encode`` ends in a ``LayerNorm``, which ``torch.autocast`` promotes to
fp32 — so the live token is fp32 *even inside a bf16 autocast region*, and "store what the
model sees" would mean 4 bytes. Measured against that fp32 reference on real frames:

    float16    max abs err 1.9e-03      (0.2% of the token std)
    bfloat16   max abs err 1.5e-02      (1.5%)

Same two bytes, 8x less error, and numpy-native so the memmap needs no bit reinterpretation.
It is also finer than anything downstream can represent: the core's first matmul runs in
bf16, whose resolution near 1.0 is ~8e-3. Tokens are LayerNorm outputs bounded to about
±5.1, so fp16's range is never in question.

**The cache key is the encoder checkpoint's sha256 plus the image size.** A mismatch raises;
it never silently rebuilds or falls back. Training the whole 0013 grid against a stale
representation would invalidate every cell in it, and that failure is invisible in the loss
curve — it looks like a slightly worse encoder, not like a bug.

Training-time only. Closed-loop evaluation and GRPO still run the encoder live, because they
see frames no cache has ever met.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

#: Bumped when the on-disk layout changes in a way that makes old caches unreadable.
#: v2 added ``actions.npy`` and the per-episode interaction/family, which is what let the
#: cached training path drop its tar dependency entirely.
CACHE_VERSION = 2

_META = "meta.json"
_TOKENS = "tokens.npy"
_ACTIONS = "actions.npy"
_DTYPE = np.float16


def encoder_fingerprint(ckpt_path: str) -> str:
    """sha256 of the encoder checkpoint file.

    The file, not the state dict: it is what a sibling repo can verify without importing
    torch, and it is what ``kaihe/contra_nes_data#6`` would record in a release manifest.
    """
    h = hashlib.sha256()
    with open(os.path.expanduser(ckpt_path), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class StaleCache(RuntimeError):
    """The cache on disk was not built by the encoder/preprocessing now in use."""


class TokenCache:
    """Read-only view of a built cache. Cheap to construct; safe per worker."""

    def __init__(self, path: str, encoder_sha256: Optional[str] = None,
                 image_size: Optional[int] = None):
        self.path = os.path.expanduser(path)
        with open(os.path.join(self.path, _META)) as fh:
            self.meta = json.load(fh)
        if int(self.meta.get("version", -1)) != CACHE_VERSION:
            raise StaleCache(
                f"cache at {self.path} is layout v{self.meta.get('version')}, "
                f"this build reads v{CACHE_VERSION} — rebuild it")
        self._check("encoder_sha256", encoder_sha256)
        self._check("image_size", image_size)

        self.dim = int(self.meta["dim"])
        self.episodes: List[dict] = list(self.meta["episodes"])
        self._by_uid: Dict[str, dict] = {e["uid"]: e for e in self.episodes}
        self._tokens: Optional[np.ndarray] = None      # opened lazily, see `tokens`
        self._actions: Optional[np.ndarray] = None

    def _check(self, field: str, expected) -> None:
        if expected is None:
            return
        actual = self.meta.get(field)
        if str(actual) != str(expected):
            raise StaleCache(
                f"cache at {self.path} was built with {field}={actual!r}, "
                f"but this run uses {expected!r}. Rebuild the cache; do not mix them.")

    @property
    def tokens(self) -> np.ndarray:
        """The memmap. Opened on first use so a cache can be forked to workers unopened."""
        if self._tokens is None:
            self._tokens = np.load(os.path.join(self.path, _TOKENS), mmap_mode="r")
        return self._tokens

    @property
    def actions(self) -> np.ndarray:
        """Flat int64 memmap of raw, unshifted action indices."""
        if self._actions is None:
            self._actions = np.load(os.path.join(self.path, _ACTIONS), mmap_mode="r")
        return self._actions

    def raw_actions(self, uid: str) -> np.ndarray:
        """``(L,)`` — the episode's action indices exactly as ``actions.npy`` recorded them.

        Unshifted on purpose: see the module docstring. ``L`` is the episode's declared
        length, which may exceed the number of frames that decoded.
        """
        ep = self._by_uid[uid]
        base, n = int(ep["action_offset"]), int(ep["action_len"])
        return np.asarray(self.actions[base:base + n])

    def interaction(self, uid: str) -> int:
        return int(self._by_uid[uid]["interaction"])

    def family(self, uid: str) -> str:
        return str(self._by_uid[uid]["family"])

    def __contains__(self, uid: str) -> bool:
        return uid in self._by_uid

    def __len__(self) -> int:
        return len(self._by_uid)

    def length(self, uid: str) -> int:
        """Decoded frame count for ``uid`` — excludes the goal row."""
        return int(self._by_uid[uid]["length"])

    def goal(self, uid: str) -> np.ndarray:
        """``(512,)`` — the cross-view prompt token."""
        return np.asarray(self.tokens[self._by_uid[uid]["offset"]])

    def frames(self, uid: str, start: int = 0,
               count: Optional[int] = None) -> np.ndarray:
        """``(count, 512)`` — frame tokens ``[start, start + count)``.

        Mirrors ``ContraCrossViewDataset.frames``' windowing so a cached loader and a live
        one index identically. A short episode returns what it has, exactly as a decode of
        the same range would.
        """
        ep = self._by_uid[uid]
        base = int(ep["offset"]) + 1                      # +1 skips the goal row
        n = int(ep["length"])
        stop = n if count is None else min(n, start + int(count))
        if start >= stop:
            return np.empty((0, self.dim), dtype=_DTYPE)
        return np.asarray(self.tokens[base + start:base + stop])


class CachedEpisodeDataset(Dataset):
    """Whole-episode items read entirely from a :class:`TokenCache`.

    Emits the same keys the action-only BC path consumes — ``tokens``, ``goal_token``,
    ``action``, ``mask``, ``family``, ``interaction`` — so it drops into
    :func:`~contra_policy.dataset.pad_episodes` and
    :meth:`~contra_policy.model.ContraPolicy.forward_tokens` unchanged. It carries no
    ``image``, and that is the point: nothing here opens a tar or a codec.

    **The alignment below is copied from** ``ContraCrossViewDataset.__getitem__`` **and
    pinned to it by test.** ``frames[j]`` is the state *after* ``actions[j]``, so the target
    for ``frames[j]`` is ``actions[j + 1]`` — pairing them directly would ask the model to
    do inverse dynamics (read the muzzle flash, answer "fire"), which is a causal leak and
    off-distribution at rollout. If that convention ever changes, both sites move together
    or ``test_cached_batch_matches_the_live_loader`` fails.
    """

    def __init__(self, cache: TokenCache, uids: Optional[Sequence[str]] = None):
        self.cache = cache
        self.uids = list(uids) if uids is not None else [e["uid"] for e in cache.episodes]
        # -1 because the last frame of an episode has no action taken *from* it.
        self.lengths = [max(1, cache.length(u) - 1) for u in self.uids]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from contra_policy.dataset import FAMILIES        # local: avoids a cycle

        uid = self.uids[idx]
        raw = self.cache.raw_actions(uid)
        T = self.lengths[idx]
        tokens = self.cache.frames(uid, 0, T)
        n = int(min(T, len(tokens), len(raw) - 1))
        assert n > 0, f"{uid} has no supervisable step (length={self.cache.length(uid)})"

        tok = np.zeros((T, self.cache.dim), dtype=np.float32)
        tok[:n] = tokens[:n]
        act = np.zeros(T, dtype=np.int64)
        act[:n] = raw[1:1 + n]
        mask = np.zeros(T, dtype=np.float32)
        mask[:n] = 1.0

        return {
            # float32, matching the live path exactly: `encode()` ends in a LayerNorm,
            # which autocast promotes to fp32, so a live frame token reaches the core as
            # fp32. Handing it fp16 here would also break `torch.cat` against the fp32
            # interaction embedding.
            "tokens": torch.from_numpy(tok),
            "goal_token": torch.from_numpy(
                self.cache.goal(uid).astype(np.float32)),
            "action": torch.from_numpy(act),
            "mask": torch.from_numpy(mask),
            "interaction": torch.tensor(self.cache.interaction(uid), dtype=torch.int64),
            "family": torch.tensor(FAMILIES.index(self.cache.family(uid)),
                                   dtype=torch.int64),
        }


def build_token_cache(index: Sequence[dict], encoder, out_dir: str, *,
                      encoder_ckpt: str, image_size: int = 256,
                      dataset=None, device: str = "cuda", chunk: int = 256,
                      log_every: int = 200) -> str:
    """Encode every episode in ``index`` once and write the cache to ``out_dir``.

    ``dataset`` is a :class:`~contra_policy.dataset.ContraCrossViewDataset` over the same
    shards, used purely as the decoder — going through it rather than reimplementing
    decode/resize is what guarantees a cached token equals a live one. One is built here if
    not supplied.

    Writes to a temporary directory and renames on success, so an interrupted build leaves
    no half-cache that a later run would trust.
    """
    from contra_policy.action_space import vectors_to_indices    # local: avoids a cycle
    from contra_policy.dataset import ContraCrossViewDataset
    from contra_policy.goal import interaction_id

    out_dir = os.path.expanduser(out_dir)
    if dataset is None:
        dataset = ContraCrossViewDataset(list(index), image_size=image_size,
                                         whole_episode=True)
    encoder = encoder.to(device).eval()

    rows = sum(int(ep["length"]) + 1 for ep in index)
    dim = int(encoder.cfg.hiddim)
    staging = out_dir + ".building"
    os.makedirs(staging, exist_ok=True)
    tokens_path = os.path.join(staging, _TOKENS)
    tokens = np.lib.format.open_memmap(tokens_path, mode="w+",
                                       dtype=_DTYPE, shape=(rows, dim))

    print(f"[token_cache] {len(index)} episodes, {rows - len(index)} frames "
          f"→ {rows} rows x {dim} @ fp16 = {rows * dim * 2 / 1e9:.2f} GB")

    episodes: List[dict] = []
    actions: List[np.ndarray] = []
    cursor = action_cursor = 0
    for i, ep in enumerate(index):
        T = int(ep["length"])
        goal = dataset.goal_image(ep)                       # (S, S, 3) uint8
        frames = dataset.frames(ep, 0, T)                   # (T, S, S, 3) uint8
        # A short/corrupt video yields fewer frames than `length` claims. Record what was
        # actually encoded rather than zero-padding: a cached zero token is a real frame as
        # far as the model is concerned, and would train on silence.
        got = int(len(frames))
        batch = np.concatenate([goal[None], frames[:got]], axis=0)

        with torch.no_grad():
            tok = torch.cat([
                encoder.encode(torch.from_numpy(batch[j:j + chunk]).to(device))
                for j in range(0, len(batch), chunk)]).float().cpu().numpy()
        tokens[cursor:cursor + len(tok)] = tok.astype(_DTYPE)

        act = vectors_to_indices(
            np.load(io.BytesIO(dataset._read(ep, "actions.npy")))).astype(np.int64)
        meta_json = json.loads(dataset._read(ep, "json"))
        actions.append(act)

        episodes.append({"uid": ep["uid"], "tar": ep["tar"], "family": ep["family"],
                         "offset": cursor, "length": got,
                         "action_offset": action_cursor, "action_len": int(len(act)),
                         "interaction": int(interaction_id(meta_json))})
        cursor += got + 1
        action_cursor += len(act)
        if log_every and (i + 1) % log_every == 0:
            print(f"[token_cache] {i + 1}/{len(index)} episodes", flush=True)

    np.save(os.path.join(staging, _ACTIONS),
            np.concatenate(actions) if actions else np.zeros(0, dtype=np.int64))

    tokens.flush()
    del tokens
    # `rows` assumed every episode decoded fully; trim if any came up short so the memmap
    # has no trailing uninitialised rows for a future reader to wander into.
    if cursor != rows:
        full = np.load(tokens_path, mmap_mode="r")
        trimmed = np.lib.format.open_memmap(
            tokens_path + ".trim", mode="w+", dtype=_DTYPE, shape=(cursor, dim))
        trimmed[:] = full[:cursor]
        trimmed.flush()
        del full, trimmed
        os.replace(tokens_path + ".trim", tokens_path)
        print(f"[token_cache] trimmed {rows - cursor} unused rows "
              f"({rows - cursor} frames were declared but not decoded)")

    meta = {"version": CACHE_VERSION,
            "encoder_sha256": encoder_fingerprint(encoder_ckpt),
            "encoder_ckpt": os.path.abspath(os.path.expanduser(encoder_ckpt)),
            "image_size": int(image_size),
            "dim": dim,
            "dtype": np.dtype(_DTYPE).name,
            "rows": int(cursor),
            "action_steps": int(action_cursor),
            "episodes": episodes}
    with open(os.path.join(staging, _META), "w") as fh:
        json.dump(meta, fh)

    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists — remove it to rebuild")
    os.replace(staging, out_dir)
    print(f"[token_cache] wrote {out_dir} "
          f"({os.path.getsize(os.path.join(out_dir, _TOKENS)) / 1e9:.2f} GB)")
    return out_dir
