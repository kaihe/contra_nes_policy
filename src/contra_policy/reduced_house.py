"""Read-only consumer for datahouse features at the frozen encoder ``reduce`` boundary.

The representation is one float16 ``(T, 256, 4, 4)`` array per episode.  The expensive
``view_backbone + reduce`` producer is immutable; policy owns and may train ``proj`` and
``token_ln``.  Actions remain unshifted, so :class:`ReducedFeatureEpisodeDataset` applies
the same ``features[i] -> actions[i + 1]`` causal contract as every other BC reader.
"""

from __future__ import annotations

import io
import json
import mmap
import os
import sqlite3
import tarfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from contra_policy.datahouse import _npy_view
from contra_policy.token_cache import StaleCache

REPRESENTATION = "reduced-view-v1"
BOUNDARY = "view_backbone+reduce"
SHAPE = (256, 4, 4)
DTYPE = np.float16


class ReducedFeatureHouse:
    """One catalog-selected reduced-feature representation, mmaped from tar members."""

    def __init__(self, root: str, *, level: int = 1, task: str = "boss",
                 weapon: str = "laser", representation: str = REPRESENTATION,
                 encoder_sha256: Optional[str] = None):
        self.path = os.path.expanduser(root)
        self.slice = (int(level), str(task), str(weapon), str(representation))
        cat = os.path.join(self.path, "catalog.sqlite")
        if not os.path.exists(cat):
            raise StaleCache(f"no catalog.sqlite under {self.path}")
        con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True)
        rows = list(con.execute(
            "select path,encoder_sha256,boundary,dtype,channels,feature_height,"
            "feature_width,ordinal,episodes,frames from feature_shards "
            "where level=? and task=? and weapon=? and representation=? order by ordinal",
            self.slice))
        con.close()
        if not rows:
            raise StaleCache(f"catalog has no feature shards for {self.slice}")
        contracts = {(str(r[1]), str(r[2]), str(r[3]), int(r[4]), int(r[5]), int(r[6]))
                     for r in rows}
        if len(contracts) != 1:
            raise StaleCache(f"feature shards mix contracts: {sorted(contracts)}")
        sha, boundary, dtype, c, h, w = contracts.pop()
        if encoder_sha256 is not None and sha != encoder_sha256:
            raise StaleCache(f"feature encoder {sha[:12]} != policy encoder "
                             f"{encoder_sha256[:12]}")
        if (boundary, dtype, (c, h, w)) != (BOUNDARY, "float16", SHAPE):
            raise StaleCache(f"feature contract is {boundary}, {dtype}, {(c,h,w)}; "
                             f"expected {BOUNDARY}, float16, {SHAPE}")
        self.encoder_sha256 = sha
        self.shards = [os.path.join(self.path, str(r[0])) for r in rows]
        self.shard_ordinals = [int(r[7]) for r in rows]
        self.declared = (sum(int(r[8]) for r in rows), sum(int(r[9]) for r in rows))
        self._fh: Dict[int, io.BufferedReader] = {}
        self._mm: Dict[int, mmap.mmap] = {}
        self._index()

    def _index(self) -> None:
        self.episodes: List[dict] = []
        self._loc: Dict[str, Tuple[int, int, int, int, int]] = {}
        for si, path in enumerate(self.shards):
            with tarfile.open(path) as archive:
                members = {m.name: m for m in archive.getmembers()}
                for name, meta_member in members.items():
                    if not name.endswith(".json") or name == "manifest.json":
                        continue
                    stem = name[:-5]
                    feat = members.get(stem + ".features.npy")
                    act = members.get(stem + ".actions.npy")
                    if feat is None or act is None:
                        raise StaleCache(f"{path}:{stem} lacks features or actions")
                    meta = json.load(archive.extractfile(meta_member))
                    uid = str(meta["uid"])
                    if uid in self._loc:
                        raise StaleCache(f"duplicate uid {uid}")
                    length = int(meta["frames"])
                    self._loc[uid] = (si, feat.offset_data, feat.size,
                                      act.offset_data, act.size)
                    self.episodes.append({"uid": uid, "length": length,
                                          "family": "boss", "interaction": 4,
                                          "shard": si})
        self._by_uid = {e["uid"]: e for e in self.episodes}
        got = (len(self.episodes), sum(e["length"] for e in self.episodes))
        if got != self.declared:
            raise StaleCache(f"shards hold {got}, catalog declares {self.declared}")

    def _map(self, si: int) -> mmap.mmap:
        mm = self._mm.get(si)
        if mm is None:
            fh = open(self.shards[si], "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._fh[si], self._mm[si] = fh, mm
        return mm

    def _array(self, uid: str, feature: bool) -> np.ndarray:
        si, fo, fs, ao, ass = self._loc[uid]
        return _npy_view(self._map(si), fo if feature else ao, fs if feature else ass)

    def features(self, uid: str, start: int = 0,
                 count: Optional[int] = None) -> np.ndarray:
        arr = self._array(uid, True)
        if arr.dtype != DTYPE or arr.ndim != 4 or tuple(arr.shape[1:]) != SHAPE:
            raise StaleCache(f"{uid} features are {arr.shape} {arr.dtype}")
        stop = len(arr) if count is None else min(len(arr), start + int(count))
        return np.asarray(arr[start:stop])

    def raw_actions(self, uid: str) -> np.ndarray:
        arr = self._array(uid, False)
        if arr.dtype != np.int64 or arr.ndim != 1:
            raise StaleCache(f"{uid} actions are {arr.shape} {arr.dtype}")
        return np.asarray(arr)

    def length(self, uid: str) -> int:
        return int(self._by_uid[uid]["length"])

    def family(self, uid: str) -> str:
        return "boss"

    def interaction(self, uid: str) -> int:
        return 4

    def shard_index(self, uid: str) -> int:
        return int(self._by_uid[uid]["shard"])

    def __contains__(self, uid: str) -> bool:
        return uid in self._by_uid

    def __len__(self) -> int:
        return len(self._by_uid)


class ReducedFeatureEpisodeDataset(Dataset):
    """Whole episodes with projection inputs and causally shifted action targets."""

    def __init__(self, house: ReducedFeatureHouse,
                 uids: Optional[Sequence[str]] = None):
        self.house = house
        self.uids = list(uids) if uids is not None else [e["uid"] for e in house.episodes]
        missing = [u for u in self.uids if u not in house]
        if missing:
            raise StaleCache(f"{len(missing)} requested episodes absent, e.g. {missing[:3]}")
        self.lengths = [max(1, house.length(u) - 1) for u in self.uids]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from contra_policy.dataset import FAMILIES

        uid, t = self.uids[idx], self.lengths[idx]
        features = self.house.features(uid, 0, t)
        actions = self.house.raw_actions(uid)
        n = int(min(t, len(features), len(actions) - 1))
        if n <= 0:
            raise StaleCache(f"{uid} has no supervisable step")
        feat = np.zeros((t, *SHAPE), dtype=np.float32)
        feat[:n] = features[:n]
        act = np.zeros(t, dtype=np.int64)
        act[:n] = actions[1:1 + n]
        mask = np.zeros(t, dtype=np.float32)
        mask[:n] = 1.0
        return {"features": torch.from_numpy(feat), "action": torch.from_numpy(act),
                "mask": torch.from_numpy(mask),
                "interaction": torch.tensor(4, dtype=torch.int64),
                "family": torch.tensor(FAMILIES.index("boss"), dtype=torch.int64)}
