"""Read-only consumer of ``contra_nes_data``'s **raw frame** shards.

The token datahouse (:mod:`contra_policy.datahouse`) serves one precomputed 512-D vector
per frame. This serves pixels to policies that train the image encoder end to end.

Frame shards deliberately live in their own catalog tables — ``frame_shards`` and
``frame_shard_episodes``, keyed by ``format`` rather than ``encoder_sha256``. Registering
them in ``shards`` would break every existing Spread run, because
:meth:`DatahouseTokens.__init__` selects on ``(level, task, weapon)`` without filtering by
encoder and raises :class:`StaleCache` on more than one. So this module is *beside*
``datahouse.py`` and shares no code path with it. See data 0012.

Two alignment rules differ from the token path and are pinned by test:

* **No goal row.** Token arrays are ``length + 1`` and readers do ``arr[1 + start]``.
  Frame arrays are ``length``, so index ``i`` is decision ``i``.
* **``frames[i]`` is the screen after action ``i``**, so the target for ``frames[i]`` is
  ``actions[i + 1]`` — the same +1 :class:`~contra_policy.token_cache.CachedEpisodeDataset`
  applies. Pairing them directly asks the model to do inverse dynamics.

The catalog selects *which* shards and in what order; every byte offset comes from each
tar's own ``manifest.json``, so a member is a seek rather than a scan.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import tarfile
from typing import Dict, List, Optional, Sequence

import av
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from contra_policy.token_cache import StaleCache

FORMAT = "png-mkv-v1"


class FrameHouse:
    """Read-only view over one ``(level, task, weapon, format)`` slice of frame shards."""

    def __init__(self, root: str, *, level: int = 1, task: str = "boss",
                 weapon: str = "spread", fmt: str = FORMAT,
                 frame_size: Optional[tuple] = None):
        self.path = os.path.expanduser(root)
        self.slice = (int(level), str(task), str(weapon), str(fmt))

        cat = os.path.join(self.path, "catalog.sqlite")
        if not os.path.exists(cat):
            raise StaleCache(f"no catalog.sqlite under {self.path} — is this a datahouse?")
        con = sqlite3.connect(f"file:{cat}?mode=ro", uri=True)
        rows = list(con.execute(
            "select path, ordinal, episodes, frames, frame_height, frame_width "
            "from frame_shards where level=? and task=? and weapon=? and format=? "
            "order by ordinal", (int(level), str(task), str(weapon), str(fmt))))
        con.close()
        if not rows:
            raise StaleCache(f"catalog has no frame shards for {self.slice}. The token "
                             f"shards are a different table; this needs a frame release.")
        geo = {(int(r[4]), int(r[5])) for r in rows}
        if len(geo) != 1:
            raise StaleCache(f"frame shards mix geometry: {sorted(geo)}")
        self.frame_size = geo.pop()
        if frame_size is not None and tuple(frame_size) != self.frame_size:
            raise StaleCache(
                f"frame shards are {self.frame_size} but the config asks for "
                f"{tuple(frame_size)}. A resized release is a different representation.")

        self.shards = [os.path.join(self.path, r[0]) for r in rows]
        self.shard_ordinals = [int(r[1]) for r in rows]
        self.declared = (sum(int(r[2]) for r in rows), sum(int(r[3]) for r in rows))
        self._fh: Dict[int, io.BufferedReader] = {}
        self._index()

    # -- construction ----------------------------------------------------

    def _index(self) -> None:
        """Read every shard's ``manifest.json``: uid, length and member byte ranges."""
        self.episodes: List[dict] = []
        self._by_uid: Dict[str, dict] = {}
        for si, path in enumerate(self.shards):
            with tarfile.open(path) as tf:
                man = json.load(tf.extractfile("manifest.json"))
            for ep in man["episodes"]:
                members = {m["name"].split(".", 1)[1]: (int(m["offset"]), int(m["size"]))
                           for m in ep["members"]}
                row = {"uid": str(ep["uid"]), "length": int(ep["frames"]),
                       "family": "boss", "interaction": -1,
                       "shard": si, "members": members}
                self.episodes.append(row)
                self._by_uid[row["uid"]] = row
        got = (len(self.episodes), sum(e["length"] for e in self.episodes))
        if got != self.declared:
            raise StaleCache(f"manifests hold {got} (episodes, frames) but the catalog "
                             f"declares {self.declared}")

    def _read(self, ep: dict, ext: str) -> bytes:
        si = ep["shard"]
        fh = self._fh.get(si)
        if fh is None:
            fh = self._fh[si] = open(self.shards[si], "rb")
        offset, size = ep["members"][ext]
        fh.seek(offset)
        return fh.read(size)

    # -- the read surface, mirroring DatahouseTokens ----------------------

    def raw_actions(self, uid: str) -> np.ndarray:
        """Unshifted 21-way baseline indices, one per frame. The +1 lives in the dataset."""
        return np.load(io.BytesIO(self._read(self._by_uid[uid], "actions.npy")))

    def frames(self, uid: str, start: int = 0,
               count: Optional[int] = None) -> np.ndarray:
        """``(count, H, W, 3)`` uint8, native geometry, no resize.

        All-intra PNG-in-MKV, so every frame is a keyframe and a seek lands exactly.
        Falls back to a whole-episode decode if the container does not seek where
        expected, which keeps correctness independent of the codec's seek behaviour.
        """
        ep = self._by_uid[uid]
        count = ep["length"] - start if count is None else int(count)
        data = io.BytesIO(self._read(ep, "obs.mkv"))
        with av.open(data) as container:
            stream = container.streams.video[0]
            if start > 0:
                target = int(start / (stream.average_rate * stream.time_base))
                container.seek(target, stream=stream, backward=True, any_frame=False)
            out, first = [], None
            for frame in container.decode(video=0):
                pos = int(round(float(frame.pts * stream.time_base * stream.average_rate)))
                if pos < start:
                    continue
                if first is None:
                    first = pos
                out.append(frame.to_ndarray(format="rgb24"))
                if len(out) == count:
                    break
        if out and first == start:
            return np.stack(out)
        data.seek(0)
        with av.open(data) as container:
            allf = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
        return np.stack(allf[start:start + count])

    def length(self, uid: str) -> int:
        return self._by_uid[uid]["length"]

    def family(self, uid: str) -> str:
        return self._by_uid[uid]["family"]

    def interaction(self, uid: str) -> int:
        return self._by_uid[uid]["interaction"]

    def shard_index(self, uid: str) -> int:
        return self._by_uid[uid]["shard"]

    def __contains__(self, uid: str) -> bool:
        return uid in self._by_uid

    def __len__(self) -> int:
        return len(self._by_uid)


class FrameEpisodeDataset(Dataset):
    """Whole-episode pixel items: ``image``, ``action``, ``mask``, ``family``.

    Emits no ``goal_token`` and no ``cross_view``, because the ViT policy is image-only —
    :func:`~contra_policy.dataset.pad_episodes` keys on ``image`` and pads the same way it
    does for the token path, so ``mask`` and ``seq_len`` mean exactly what they do there.

    The ``actions[i + 1]`` shift is copied from
    :class:`~contra_policy.token_cache.CachedEpisodeDataset` and pinned to it by test.
    """

    def __init__(self, house: FrameHouse, uids: Optional[Sequence[str]] = None,
                 image_size: int = 256):
        self.house = house
        self.image_size = int(image_size)
        self.uids = list(uids) if uids is not None else [e["uid"] for e in house.episodes]
        missing = [u for u in self.uids if u not in house]
        if missing:
            raise StaleCache(f"{len(missing)} of {len(self.uids)} requested episodes are "
                             f"not in the frame release, e.g. {missing[:3]}")
        # -1 because the last frame of an episode has no action taken *from* it.
        self.lengths = [max(1, house.length(u) - 1) for u in self.uids]

    def __len__(self) -> int:
        return len(self.uids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        from contra_policy.dataset import FAMILIES        # local: avoids a cycle

        uid = self.uids[idx]
        T = self.lengths[idx]
        raw = self.house.raw_actions(uid)
        img = self.house.frames(uid, 0, T)
        img = np.asarray([cv2.resize(frame, (self.image_size, self.image_size),
                                     interpolation=cv2.INTER_AREA) for frame in img],
                         dtype=np.uint8)
        n = int(min(T, len(img), len(raw) - 1))
        assert n > 0, f"{uid} has no supervisable step (length={self.house.length(uid)})"

        h = w = self.image_size
        image = np.zeros((T, h, w, 3), dtype=np.uint8)
        image[:n] = img[:n]
        act = np.zeros(T, dtype=np.int64)
        act[:n] = raw[1:1 + n]
        mask = np.zeros(T, dtype=np.float32)
        mask[:n] = 1.0

        return {
            "image": torch.from_numpy(image),
            "action": torch.from_numpy(act),
            "mask": torch.from_numpy(mask),
            "family": torch.tensor(FAMILIES.index(self.house.family(uid)),
                                   dtype=torch.int64),
            "interaction": torch.tensor(self.house.interaction(uid), dtype=torch.int64),
        }
