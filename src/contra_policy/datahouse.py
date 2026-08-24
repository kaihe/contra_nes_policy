"""Read-only consumer of ``contra_nes_data``'s tokenized datahouse.

The token cache in :mod:`contra_policy.token_cache` was policy-owned: this repo ran the
frozen encoder over a release's frames and memmapped the result under ``cache/``. Data
[0004](../../contra_nes_data/doc/0004-tokenized-datahouse.md) moved that ownership — the
encoder, its preprocessing contract and every encoded token now live in ``contra_nes_data``,
and policy consumes them. See ``kaihe/contra_nes_policy#7``.

So this module has **no write path**. It does not build, extract, convert or cache anything;
it reads the data repo's tars where they lie. That is affordable because the tars are
uncompressed and each member is contiguous: an ``.npy`` inside a ``.tar`` is a byte range,
and a byte range of an ``mmap`` is a zero-copy numpy view. Indexing 53 shards costs ~15 s of
header seeks at startup and no disk.

**What is asserted before a single token is read**, all of it from the data-owned
``spec.json`` rather than from anything this repo believes:

    checkpoint_sha256   must equal the encoder this run loads — the same guarantee
                        `TokenCache` gave, and the reason a stale representation cannot
                        silently train a whole scaling axis
    tokens.dtype/width  float16 / 512
    tokens.layout       goal_then_decision_frames — row 0 is the goal, as `goal()` assumes
    input.image_size    must equal `args.image_size`
    action_alignment    raw indices; the +1 shift stays in the dataset, not here

The surface deliberately mirrors :class:`~contra_policy.token_cache.TokenCache` — ``goal``,
``frames``, ``raw_actions``, ``length``, ``interaction``, ``family``, ``__contains__``,
``__len__`` — so :class:`~contra_policy.token_cache.CachedEpisodeDataset` consumes it
unchanged and the cached and datahouse paths cannot drift in how they align actions to
frames.

**The validation split is provisional.** Data 0004 promises one fixed validation split
shared by the 10k/20k/40k tiers; until it ships, :func:`split_uids` carves a deterministic
holdout by uid digest. That is enough for doc/0016 and doc/0017, whose metrics are *within-run*
differences between two checkpoints scored on the same episodes — but it is not the official
split, so CE from these runs is not comparable to any other tier's. Replace it, do not
extend it.
"""

from __future__ import annotations

import hashlib
import io
import json
import mmap
import os
import sqlite3
import tarfile
from collections.abc import Mapping
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from contra_policy.token_cache import StaleCache

_DTYPE = np.float16
_WIDTH = 512
_LAYOUT = "goal_then_decision_frames"


def _npy_view(mm: mmap.mmap, offset: int, size: int) -> np.ndarray:
    """Zero-copy view of an ``.npy`` that starts at ``offset`` inside ``mm``."""
    head = io.BytesIO(mm[offset:offset + min(size, 256)])
    version = np.lib.format.read_magic(head)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(head)
    else:
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(head)
    if fortran:
        raise StaleCache("datahouse .npy is Fortran-ordered; the reader assumes C order")
    n = int(np.prod(shape)) if shape else 1
    return np.frombuffer(mm, dtype=dtype, count=n,
                         offset=offset + head.tell()).reshape(shape)


class DatahouseTokens:
    """Read-only view over one ``(level, task, weapon)`` slice of the datahouse."""

    def __init__(self, root: str, *, level: int = 1, task: str = "boss",
                 weapon="spread", encoder_sha256: Optional[str] = None,
                 image_size: Optional[int] = None):
        """``weapon`` may be one name or a list of them.

        A multi-weapon store is a *different experiment*, not a bigger version of the same
        one: each weapon in this datahouse was generated from its own start state, so
        combining them changes what the training set covers and dilutes the share of
        episodes that come from the probe's start. doc/0017 §2 records which cells do this.
        """
        self.path = os.path.expanduser(root)
        self.dim = _WIDTH
        weapons = [str(weapon)] if isinstance(weapon, str) else [str(w) for w in weapon]
        self.slice = (int(level), str(task), tuple(weapons))

        cat = os.path.join(self.path, "catalog.sqlite")
        if not os.path.exists(cat):
            raise StaleCache(f"no catalog.sqlite under {self.path} — is this a datahouse?")
        con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True)
        marks = ",".join("?" * len(weapons))
        rows = list(con.execute(
            f"select path, encoder_sha256, episodes, frames, weapon, ordinal from shards "
            f"where level=? and task=? and weapon in ({marks}) "
            f"order by weapon, ordinal", (int(level), str(task), *weapons)))
        found = {str(r[4]) for r in rows}
        con.close()
        if not rows:
            raise StaleCache(f"catalog has no shards for level={level} task={task} "
                             f"weapon={weapons}")
        missing = [w for w in weapons if w not in found]
        if missing:
            raise StaleCache(f"catalog has no shards for weapon(s) {missing} — asked for "
                             f"{weapons}, found {sorted(found)}. A silently smaller "
                             f"training set would look like a worse model.")

        shas = {r[1] for r in rows}
        if len(shas) != 1:
            raise StaleCache(f"shards for {self.slice} mix encoders: {sorted(shas)}")
        self.encoder_sha256 = shas.pop()
        if encoder_sha256 is not None and self.encoder_sha256 != encoder_sha256:
            raise StaleCache(
                f"datahouse tokens were encoded by {self.encoder_sha256[:12]}, but this run "
                f"loads {encoder_sha256[:12]}. These are different representations; do not "
                f"mix them.")
        self.spec = self._read_spec(self.encoder_sha256, image_size)

        self.shards = [os.path.join(self.path, r[0]) for r in rows]
        self.declared = (sum(int(r[2]) for r in rows), sum(int(r[3]) for r in rows))
        self.shard_weapons = [str(r[4]) for r in rows]
        self.shard_ordinals = [int(r[5]) for r in rows]
        self._mm: Dict[int, mmap.mmap] = {}
        self._fh: Dict[int, io.BufferedReader] = {}
        self._index()

    # -- construction ----------------------------------------------------

    def _read_spec(self, sha: str, image_size: Optional[int]) -> dict:
        """Load and enforce the data-owned encoder contract."""
        hits = [os.path.join(dp, "spec.json")
                for dp, _, fs in os.walk(self.path) if "spec.json" in fs and sha in dp]
        if not hits:
            raise StaleCache(f"no encoder bundle spec.json for {sha[:12]} under {self.path}")
        spec = json.load(open(hits[0]))
        got = spec.get("checkpoint_sha256")
        if got != sha:
            raise StaleCache(f"{hits[0]} declares {got!r}, catalog says {sha!r}")
        tok, inp = spec.get("tokens", {}), spec.get("input", {})
        if tok.get("dtype") != "float16" or int(tok.get("width", -1)) != _WIDTH:
            raise StaleCache(f"spec tokens are {tok!r}; this reader requires "
                             f"float16 x {_WIDTH}")
        if tok.get("layout") != _LAYOUT:
            raise StaleCache(f"spec token layout is {tok.get('layout')!r}, not {_LAYOUT!r} "
                             f"— `goal()` would return the wrong row")
        if (image_size is not None and "image_size" in inp and
                int(inp["image_size"]) != int(image_size)):
            raise StaleCache(f"spec was encoded at image_size={inp.get('image_size')}, "
                             f"this run uses {image_size}")
        if "height" in inp or "width" in inp:
            if int(inp.get("height", 0)) < 1 or int(inp.get("width", 0)) < 1:
                raise StaleCache(f"invalid rectangular input contract: {inp!r}")
        return spec

    def _index(self) -> None:
        """One header pass per shard. Records byte ranges; reads no token data."""
        self.episodes: List[dict] = []
        self._loc: Dict[str, Tuple[int, int, int, int, int]] = {}
        for si, tar_path in enumerate(self.shards):
            with tarfile.open(tar_path) as tf:
                members = {m.name: m for m in tf.getmembers()}
                metas = {n: m for n, m in members.items() if n.endswith(".json")}
                for name, meta in metas.items():
                    stem = name[:-len(".json")]
                    tok = members.get(stem + ".tokens.npy")
                    act = members.get(stem + ".actions.npy")
                    if tok is None or act is None:
                        raise StaleCache(f"{tar_path}:{stem} is missing tokens or actions")
                    ep = json.load(tf.extractfile(meta))
                    uid = str(ep["uid"])
                    if uid in self._loc:
                        raise StaleCache(f"duplicate uid {uid} in {tar_path}")
                    self._loc[uid] = (si, tok.offset_data, tok.size,
                                      act.offset_data, act.size)
                    self.episodes.append(
                        {"uid": uid, "length": int(ep["length"]),
                         "action_len": int(ep["action_len"]),
                         "family": str(ep["family"]),
                         "interaction": int(ep["interaction"])})
        self._by_uid = {e["uid"]: e for e in self.episodes}
        if len(self.episodes) != self.declared[0]:
            raise StaleCache(f"catalog declares {self.declared[0]} episodes for "
                             f"{self.slice}, the shards hold {len(self.episodes)}")

    # -- access ----------------------------------------------------------

    def _map(self, si: int) -> mmap.mmap:
        """Per-process mmap of one shard, opened on first touch."""
        mm = self._mm.get(si)
        if mm is None:
            fh = open(self.shards[si], "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._fh[si], self._mm[si] = fh, mm
        return mm

    def _tokens(self, uid: str) -> np.ndarray:
        si, off, size, _, _ = self._loc[uid]
        arr = _npy_view(self._map(si), off, size)
        if arr.dtype != _DTYPE or arr.ndim != 2 or arr.shape[1] != _WIDTH:
            raise StaleCache(f"{uid} tokens are {arr.shape} {arr.dtype}, expected "
                             f"(T+1, {_WIDTH}) {_DTYPE}")
        return arr

    def raw_actions(self, uid: str) -> np.ndarray:
        si, _, _, off, size = self._loc[uid]
        return np.asarray(_npy_view(self._map(si), off, size))

    def interaction(self, uid: str) -> int:
        return int(self._by_uid[uid]["interaction"])

    def family(self, uid: str) -> str:
        return str(self._by_uid[uid]["family"])

    def length(self, uid: str) -> int:
        """Decoded frame count — excludes the goal row, as `TokenCache.length` does."""
        return int(self._by_uid[uid]["length"])

    def goal(self, uid: str) -> np.ndarray:
        return np.asarray(self._tokens(uid)[0])

    def frames(self, uid: str, start: int = 0,
               count: Optional[int] = None) -> np.ndarray:
        arr = self._tokens(uid)
        n = int(self._by_uid[uid]["length"])
        stop = n if count is None else min(n, start + int(count))
        if start >= stop:
            return np.empty((0, self.dim), dtype=_DTYPE)
        return np.asarray(arr[1 + start:1 + stop])       # +1 skips the goal row

    def shard_index(self, uid: str) -> int:
        """Which shard, in catalog order, holds ``uid``."""
        return int(self._loc[uid][0])

    def __contains__(self, uid: str) -> bool:
        return uid in self._by_uid

    def __len__(self) -> int:
        return len(self._by_uid)


def split_uids(store: DatahouseTokens, val_every: int = 20) -> Tuple[List[str], List[str]]:
    """Provisional deterministic holdout: every ``val_every``-th uid by digest.

    Keyed on the uid's sha1 rather than on shard order, so the holdout is stable under
    re-sharding and independent of how many shards a cell reads. **This is not the fixed
    validation split data 0004 promises**; see the module docstring.
    """
    train, val = [], []
    for e in store.episodes:
        digest = hashlib.sha1(e["uid"].encode()).digest()
        (val if int.from_bytes(digest[:4], "big") % int(val_every) == 0 else train
         ).append(e["uid"])
    if not val:
        raise ValueError(f"val_every={val_every} selected no episodes")
    return sorted(train), sorted(val)


def shard_prefix(store: DatahouseTokens, uids: Sequence[str],
                 shard_count: Optional[int | Mapping[str, int]]) -> List[str]:
    """Restrict ``uids`` to the first ``shard_count`` shards; ``None`` keeps them all.

    This is how a smaller *training* tier is cut from a larger store while the holdout stays
    the store's own. Because :func:`split_uids` keys on the uid digest and not on shard order,
    the val set is identical for every prefix — so val CE from a 13-shard cell and a 53-shard
    cell are the same episodes, and the two are directly comparable. Train episodes from the
    dropped shards are simply not trained on; they do not leak into val, which was already
    carved before this runs.
    """
    if shard_count is None:
        return list(uids)
    if isinstance(shard_count, Mapping):
        limits = {str(k): int(v) for k, v in shard_count.items()}
        weapons = set(store.slice[2])
        if set(limits) != weapons:
            raise ValueError(f"per-weapon shard counts must name exactly {sorted(weapons)}, "
                             f"got {sorted(limits)}")
        available = {w: 0 for w in weapons}
        for weapon, ordinal in zip(store.shard_weapons, store.shard_ordinals):
            available[weapon] = max(available[weapon], ordinal + 1)
        bad = {w: n for w, n in limits.items() if n < 1 or n > available[w]}
        if bad:
            raise ValueError(f"per-weapon shard counts {bad} outside available {available}")
        return [u for u in uids
                if store.shard_ordinals[store.shard_index(u)]
                < limits[store.shard_weapons[store.shard_index(u)]]]
    n = int(shard_count)
    if n < 1 or n > len(store.shards):
        raise ValueError(f"shard_count={n} outside 1..{len(store.shards)} for {store.slice}")
    return [u for u in uids if store.shard_index(u) < n]


def datahouse_index(store: DatahouseTokens, uids: Sequence[str]) -> List[dict]:
    """The ``{uid, length, family, interaction}`` rows the trainer's samplers expect."""
    return [dict(store._by_uid[u]) for u in uids]
