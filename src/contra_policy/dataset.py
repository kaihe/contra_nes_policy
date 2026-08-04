"""Contra task shards → ROCKET-2-shaped training windows.

The data is eight WebDataset tar shards written by ``contra_nes_data``'s
``task_maker/export_hf.py`` — one per task family (kill / item / traverse / boss)
per split::

    game_trace/hf/{kill,item,traverse,boss}-train-00000.tar
    game_trace/hf/{kill,item,traverse,boss}-val-00000.tar

The split is the *data repo's*, not ours. It is materialised three ways over there
(a ``split`` field in each episode's ``.json``, in the task ``.npz``, and a ``split``
column in ``manifest.csv``) and the shards are written pre-partitioned, so this repo
reads it off the filename and never reproduces it. An earlier version held out a
random permutation of episodes here instead; that split was leaky — every val episode
came from a source trace that also fed training, and 35% of val decision steps were
frames that also appeared inside a training episode.

Each episode is four grouped members::

    <uid>.obs.mkv       (T, 224, 240, 3) lossless frames, one per decision step
    <uid>.actions.npy   (T, 9) uint8 NES button vectors, aligned to the frames
    <uid>.goal.png      the ROCKET goal frame: reference image + colored blob
    <uid>.json          goal points, per-frame centroids + visibility, labels

Reading strategy: a one-time pass records each member's byte offset inside its tar,
after which episodes are pulled by ``seek``+``read`` with no tar walking. That buys
a map-style ``Dataset`` with true shuffling and a deterministic ``__getitem__`` —
the same contract as ROCKET-2's ``RawDataset``, which likewise chops each episode
into non-overlapping ``win_len`` windows and locates a sample by
``(episode, relative_idx)``. The index is cached on disk and keyed by the tars'
size+mtime, so it is rebuilt only when the shards change.

One deliberate divergence from ROCKET-2's ``CrossViewDataset``: there, the cross
view is *sampled* per window from elsewhere in the same episode, because a Minecraft
episode contains many interactions. A Contra task episode is defined by exactly one
goal, already rendered into ``goal.png``, so the cross view is constant across the
window and no sampling (or its retry logic) is needed.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import io
import json
import math
import os
import tarfile
from typing import Dict, List, Sequence

import av
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler

from contra_policy.action_space import ACTION_NAMES, NUM_ACTIONS, vectors_to_indices
from contra_policy.goal import goal_mask, interaction_id, points_to_target

CONFIGS = ("kill", "item", "traverse", "boss")
FAMILIES = CONFIGS                      # index order for the per-family metric split
IDLE_ACTION = ACTION_NAMES.index("_")   # fills prev_action on masked-off tail padding

# Bumped whenever build_index's schema changes, so a stale cache is rebuilt rather
# than silently read back missing the new fields.
_INDEX_VERSION = 2


# ── shard index ───────────────────────────────────────────────────────────────

def _tar_fingerprint(paths: Sequence[str]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f"{os.path.basename(p)}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def family_of(tar_path: str) -> str:
    """``…/kill-train-00000.tar`` → ``"kill"``. The shard name is the family."""
    return os.path.basename(tar_path).split("-")[0]


def build_index(tar_paths: Sequence[str]) -> List[dict]:
    """Walk each tar once and record every episode's member offsets.

    Returns one entry per episode:
    ``{tar, uid, family, length, action_counts, members: {ext: [offset, size]}}``.
    ``length`` comes from the actions array (authoritative — it is what the frames
    were rendered against), read during the walk since it is only a few hundred bytes.
    ``action_counts`` is the episode's 21-way action histogram, free to compute here
    since the array is already decoded, and what the class-weighted BC loss is built
    from without a second pass over the shards.
    """
    episodes: Dict[str, dict] = {}
    for tar_path in tar_paths:
        family = family_of(tar_path)
        with tarfile.open(tar_path) as tf:
            for m in tf:
                if not m.isfile() or "." not in m.name:
                    continue
                uid, ext = m.name.split(".", 1)
                key = f"{tar_path}::{uid}"
                ep = episodes.setdefault(
                    key, {"tar": tar_path, "uid": uid, "family": family, "members": {}})
                ep["members"][ext] = [m.offset_data, m.size]
                if ext == "actions.npy":
                    arr = np.load(io.BytesIO(tf.extractfile(m).read()))
                    ep["length"] = int(len(arr))
                    idx = vectors_to_indices(arr)
                    ep["action_counts"] = np.bincount(
                        idx, minlength=NUM_ACTIONS).astype(int).tolist()
    def complete(ep: dict) -> bool:
        have = set(ep["members"])
        return ("length" in ep and {"actions.npy", "goal.png", "json"} <= have
                and not {"obs.mkv", "obs.mp4"}.isdisjoint(have))


    out = [ep for ep in episodes.values() if complete(ep)]
    out.sort(key=lambda e: (e["tar"], e["uid"]))
    return out


def load_or_build_index(tar_paths: Sequence[str], cache_dir: str = "cache") -> List[dict]:
    """Cached :func:`build_index`, invalidated by any shard's size or mtime changing."""
    tar_paths = [os.path.expanduser(p) for p in tar_paths]
    for p in tar_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"shard not found: {p}")
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(
        cache_dir, f"shard_index_v{_INDEX_VERSION}_{_tar_fingerprint(tar_paths)}.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    print(f"[dataset] indexing {len(tar_paths)} shard(s) → {path} (one-time pass)")
    index = build_index(tar_paths)
    with open(path, "w") as fh:
        json.dump(index, fh)
    print(f"[dataset] indexed {len(index)} episodes, "
          f"{sum(e['length'] for e in index)} decision steps")
    return index


# ── dataset ───────────────────────────────────────────────────────────────────

class ContraCrossViewDataset(Dataset):
    """Non-overlapping ``win_len`` windows over episodes, in ROCKET-2 input format.

    Each item is a dict of tensors:

    ==========================  ====================  ==================================
    key                         shape                 meaning
    ==========================  ====================  ==================================
    ``image``                   (T, S, S, 3) uint8    agent view
    ``cross_view/…_image``      (S, S, 3) uint8       goal frame — no time axis
    ``cross_view/…_obj_mask``   (S, S) uint8          goal blob channel, regenerated
    ``cross_view/…_obj_id``     (T,) int64            interaction id
    ``prev_action``             (T,) int64            action that produced this frame
    ``prev_action_dropout``     (T,) float32          1 = use it, 0 = "unknown" embedding
    ``action``                  (T,) int64            BC target: action taken *from* it
    ``goal_heatmap``            (T, A, A) float32     optional when ``aux_size > 0``
    ``exist`` / ``point``       (T,) (T,2) float32    optional with the heatmap target
    ``mask``                    (T,) float32          1 on real steps, 0 on tail padding
    ``first``                   (T,) bool             True at t=0 of an episode's first
                                                      window — resets attention memory
    ``family``                  () int64              index into FAMILIES, for metrics
    ==========================  ====================  ==================================

    Windows are padded, never dropped: episodes are as short as 8 steps and dropping
    them would silently bias the mix away from short kills. ``mask`` gates every loss.

    Watch the per-window byte cost when changing this: a loader holds
    ``num_workers * prefetch_factor * batch_size`` windows in flight, doubled again by
    ``pin_memory``, so a few MB per window turns into tens of GB of host RAM.
    """

    #: Per-class entity occupancy channels, in the order the shards' ``entities`` field
    #: and ``env.entity.HEATMAP_CLASSES`` use. Position is meaningful — it is the
    #: channel order of the ``entity_heatmap`` target.
    ENTITY_CLASSES = ("player", "player_bullets", "enemies", "enemy_bullets")

    def __init__(self, index: List[dict], win_len: int = 32, image_size: int = 256,
                 sigma_px: float = 12.0, prev_action_keep_prob: float = 0.25,
                 aux_size: int = 32, seed: int = 0, want_entities: bool = False,
                 entity_sigma_px: float = 6.0, whole_episode: bool = False):
        self.index = index
        # One item = one whole episode at its natural length, unpadded. The policy's
        # context holds an entire episode, so windows are no longer the unit — and
        # padding every episode to 1024 would be 201 MB of uint8 frames per item.
        # Callers must collate with `pad_episodes`, which pads to the batch maximum.
        self.whole_episode = whole_episode
        self.want_entities = want_entities
        # Tighter than the goal's 12 px on purpose. At A=32 a 12 px sigma is 1.66 cells,
        # which smears a boss frame's ~4.9 enemy bullets into one blob; 6 px is 0.83
        # cells and keeps individual sprites separable. The goal blob keeps 12 px
        # because `goal_mask` also renders the cross-view *prompt* at that width and the
        # two must agree.
        self.entity_sigma_px = entity_sigma_px
        self.win_len = win_len
        self.image_size = image_size
        self.sigma_px = sigma_px
        self.prev_action_keep_prob = prev_action_keep_prob
        self.aux_size = aux_size
        self.seed = seed

        # Flat (episode_i, relative_idx) table so __getitem__ is O(1) and the sampler
        # sees a plain length. ROCKET-2 does the same with locate_item().
        self.windows: List[tuple[int, int]] = []
        # Flat window indices per episode, in temporal order. EpisodeStreamSampler
        # walks these to keep a lane's windows consecutive so memory can be carried.
        self.episode_windows: List[List[int]] = []
        for i, ep in enumerate(self.index):
            # The exporter records the frame *after* each action, so the action taken
            # from frames[j] is actions[j+1] and the final frame has no target. One
            # step per episode drops out — ~1% of recorded steps.
            usable = ep["length"] - 1
            if usable <= 0:
                self.episode_windows.append([])
                continue                       # degenerate single-step episode
            flat = []
            n_win = 1 if whole_episode else math.ceil(usable / win_len)
            for r in range(n_win):
                flat.append(len(self.windows))
                self.windows.append((i, r))
            self.episode_windows.append(flat)

        # Per-worker file handles. No decoded-frame cache: windows are shuffled, so an
        # episode cache would hit near-zero while costing hundreds of MB per worker.
        self._fh: Dict[str, io.BufferedReader] = {}

    def __len__(self) -> int:
        return len(self.windows)

    # -- member IO ---------------------------------------------------------

    def _read(self, ep: dict, ext: str) -> bytes:
        fh = self._fh.get(ep["tar"])
        if fh is None:
            fh = self._fh[ep["tar"]] = open(ep["tar"], "rb")
        offset, size = ep["members"][ext]
        fh.seek(offset)
        return fh.read(size)

    def _frames(self, ep: dict, start: int, count: int) -> np.ndarray:
        """Decode only frames ``[start, start + count)`` of an episode.

        ``export_hf.py`` writes the observation video with an all-intra codec (PNG or
        FFV1 in MKV), so *every* frame is a keyframe and a seek lands exactly rather
        than on some earlier GOP boundary. That makes windowed decoding both correct
        and cheap: decoding the whole episode to serve one 32-frame window costs 4.8x
        more frames than the epoch actually needs (7.5x on the boss shard), which is
        enough to starve the GPU.

        Falls back to a full decode if the container does not seek where expected, so
        a shard written with a non-intra codec still reads correctly, just slower.
        """
        ext = "obs.mkv" if "obs.mkv" in ep["members"] else "obs.mp4"
        # PyAV decodes straight from the in-memory tar slice; imageio's FFMPEG plugin
        # would need the bytes spilled to a temp file first.
        data = io.BytesIO(self._read(ep, ext))
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
                out.append(self._resize(frame.to_ndarray(format="rgb24")))
                if len(out) == count:
                    break
        if out and first == start:
            return np.stack(out)
        return self._full_decode(data, start, count)

    def _full_decode(self, data: io.BytesIO, start: int, count: int) -> np.ndarray:
        """Whole-episode decode, then slice — the correctness fallback for _frames."""
        data.seek(0)
        with av.open(data) as container:
            frames = [self._resize(f.to_ndarray(format="rgb24"))
                      for f in container.decode(video=0)]
        return np.stack(frames[start:start + count])

    def _resize(self, img: np.ndarray) -> np.ndarray:
        """224x240 NES screen → square network input.

        INTER_AREA on the way down keeps thin sprites (bullets are ~2px) from being
        dropped by nearest-neighbour sampling, which is the whole reason the
        pretrained encoder can see them.
        """
        s = self.image_size
        if img.shape[:2] == (s, s):
            return img
        return cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)

    # -- item --------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep_i, rel = self.windows[idx]
        ep = self.index[ep_i]
        meta = json.loads(self._read(ep, "json"))
        actions = vectors_to_indices(np.load(io.BytesIO(self._read(ep, "actions.npy"))))

        # Whole-episode mode sizes T to this episode rather than to a fixed window.
        # `usable` is length-1 because the last frame has no action taken *from* it.
        T = max(1, int(ep["length"]) - 1) if self.whole_episode else self.win_len
        start = 0 if self.whole_episode else rel * T
        frames = self._frames(ep, start, T)
        # -1 because frames[j] is the state AFTER actions[j]: the action taken FROM
        # the last frame of an episode was never recorded, so that frame has no target.
        n = int(min(T, len(frames), len(actions) - start - 1))
        assert n > 0, f"window {rel} of {ep['uid']} is empty (length={ep['length']})"

        image = np.zeros((T, self.image_size, self.image_size, 3), dtype=np.uint8)
        image[:n] = frames[:n]

        # Cross view: ONE goal per episode, emitted without a time axis.
        #
        # Materialising it per timestep (as ROCKET-2 must, since a Minecraft window
        # can span several interactions) would make the two cross-view tensors 14.7MB
        # of a 21MB window — 32 identical copies of a 196KB frame — and that is enough
        # to OOM a multi-worker loader. The model expands the encoded tokens instead,
        # which is both identical in result and ~2x less backbone compute, since the
        # goal frame is then encoded once per window rather than once per frame.
        # cv2.imdecode rather than imageio.imread: same pixels, ~3x cheaper, and this
        # runs once per window (the goal is re-decoded for each window of an episode).
        goal_bgr = cv2.imdecode(np.frombuffer(self._read(ep, "goal.png"), np.uint8),
                                cv2.IMREAD_COLOR)
        goal_img = self._resize(cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB))
        gmask = goal_mask(meta["goal_points"], self.image_size, self.sigma_px)
        cross_id = np.full(T, interaction_id(meta), dtype=np.int64)
        # uint8, not float32: the mask is a [0,1] blob map and quantising it to 256
        # levels costs nothing next to halving the tensor again. Scaled in the model.
        gmask_u8 = (gmask * 255.0).round().astype(np.uint8)

        # Action alignment. `export_hf.materialize` steps the emulator and *then* grabs
        # the screen, so frames[j] is the state after actions[j] was executed:
        #
        #     s0 --a0--> s1 --a1--> s2 …        frames[j] == s(j+1)
        #
        # The BC target for frames[j] is therefore the action taken *from* it,
        # actions[j+1] — not actions[j], which is the action that produced it and whose
        # effect is already on screen. Pairing those directly asks the model to do
        # inverse dynamics (read the muzzle flash, answer "fire"), which is both a
        # causal leak and off-distribution at rollout, where only the pre-action frame
        # is available. Both indices shift by one, in the same direction.
        act = np.zeros(T, dtype=np.int64)
        act[:n] = actions[start + 1:start + 1 + n]

        # prev_action[k] = the action that produced this frame = actions[start + k].
        # It always exists — even at start == 0, where actions[0] is what produced
        # frames[0] — so nothing has to be fabricated and nothing is force-dropped.
        prev = np.empty(T, dtype=np.int64)
        prev[:n] = actions[start:start + n]
        prev[n:] = IDLE_ACTION

        rng = np.random.default_rng(self.seed + idx)
        keep = (rng.random(T) < self.prev_action_keep_prob).astype(np.float32)

        # Aux targets: per-frame goal centroids, gated by the exported visibility flag.
        # These are NOT shifted with the actions above: the exporter reads RAM in the
        # same iteration it grabs the screen, so centroids[j] describes frames[j].
        #
        # `goal_heatmap` is the supervision the aux head actually trains on; `exist`
        # and `point` are kept as *evaluation* targets only, so `point_err_px` stays
        # on the scale the evaluator pins. The heatmap is rendered by the same
        # goal.goal_mask that draws the cross-view prompt channel, so target and
        # prompt agree by construction, and an invisible goal is an all-zero map —
        # which is the whole reason for the change. Scalar `exist` gave `kill` and
        # `boss` (100.0% goal-visible) literally no negative examples; a heatmap makes
        # every pixel outside the blob a negative on every frame.
        A = self.aux_size
        exist = point = heatmap = n_goal_points = None
        if A > 0:
            exist = np.zeros(T, dtype=np.float32)
            point = np.zeros((T, 2), dtype=np.float32)
            heatmap = np.zeros((T, A, A), dtype=np.float32)
            # How many centroids the goal has on this frame. `points_to_target`
            # collapses them to their mean, which is well-defined only for one target.
            n_goal_points = np.zeros(T, dtype=np.int64)
            centroids, visibility = meta["centroids"], meta["visibility"]
            for k in range(n):
                j = start + k
                if j >= len(centroids) or not visibility[j] or not centroids[j]:
                    continue
                exist[k] = 1.0
                point[k], _bbox = points_to_target(centroids[j])
                n_goal_points[k] = len(centroids[j])
                heatmap[k] = goal_mask(centroids[j], A, self.sigma_px)

        # Per-class entity occupancy, off by default so the BC path pays nothing for it.
        # `entities` is absent from shards exported before 2026-07-30; an all-zero target
        # is the honest fallback (no entities known), and `want_entities` callers should
        # check `entities_available` rather than train on silence.
        entity = goal_entity = None
        if self.want_entities:
            if A <= 0:
                raise ValueError("want_entities requires aux_size > 0")
            C = len(self.ENTITY_CLASSES)
            entity = np.zeros((T, C, A, A), dtype=np.float32)
            ent = meta.get("entities")

            def render(frame_idx: int, into: np.ndarray) -> None:
                for c, name in enumerate(self.ENTITY_CLASSES):
                    col = ent.get(name)
                    if col is None or frame_idx >= len(col) or not col[frame_idx]:
                        continue
                    into[c] = goal_mask(col[frame_idx], A, self.entity_sigma_px)

            if ent is not None:
                for k in range(n):
                    render(start + k, entity[k])
                # The goal frame's own entity target. `goal.png` IS an episode frame
                # with the target painted into it, and `goal_frame_idx` says which —
                # so a goal-agnostic encoder can supervise goal frames with exactly the
                # same signal as any other frame, rather than leaving them unlabelled.
                gi = meta.get("goal_frame_idx")
                if gi is not None:
                    goal_entity = np.zeros((C, A, A), dtype=np.float32)
                    render(int(gi), goal_entity)

        mask = np.zeros(T, dtype=np.float32)
        mask[:n] = 1.0

        # True only on the window that opens an episode. The recurrent core resets its
        # attention memory here; every other window continues the one before it, which
        # is what makes EpisodeStreamSampler's carried memory correct.
        first = np.zeros(T, dtype=bool)
        first[0] = rel == 0

        out = {
            "image": torch.from_numpy(image),
            "cross_view": {
                "cross_view_image": torch.from_numpy(goal_img),
                "cross_view_obj_mask": torch.from_numpy(gmask_u8),
                "cross_view_obj_id": torch.from_numpy(cross_id),
            },
            "prev_action": torch.from_numpy(prev),
            "prev_action_dropout": torch.from_numpy(keep),
            "action": torch.from_numpy(act),
            "mask": torch.from_numpy(mask),
            "first": torch.from_numpy(first),
            "family": torch.tensor(FAMILIES.index(ep["family"]), dtype=torch.int64),
        }
        if heatmap is not None:
            out.update({
                "exist": torch.from_numpy(exist),
                "point": torch.from_numpy(point),
                "goal_heatmap": torch.from_numpy(heatmap),
                "n_goal_points": torch.from_numpy(n_goal_points),
            })
        if entity is not None:
            out["entity_heatmap"] = torch.from_numpy(entity)
        if goal_entity is not None:
            out["goal_entity_heatmap"] = torch.from_numpy(goal_entity)
        return out


# ── datamodule ────────────────────────────────────────────────────────────────

def _worker_init(worker_id: int) -> None:
    """Pin each worker to one thread for cv2 and torch.

    Both default to one thread per core (32 here), so N workers each try to fan out
    across the whole machine and spend their time contending instead of decoding.
    A dataloader worker should be single-threaded — parallelism comes from having
    several of them.
    """
    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def shard_paths(shard_dir: str, configs: Sequence[str], split: str,
                overrides: Dict[str, str] | None = None) -> List[str]:
    """Resolve every family shard, with optional family-specific directories.

    Missing families retain the old deterministic ``00000`` path so the caller raises
    a useful ``FileNotFoundError`` instead of silently training without that family.
    """
    overrides = overrides or {}
    paths: List[str] = []
    for family in configs:
        root = os.path.expanduser(overrides.get(family, shard_dir))
        matches = sorted(glob.glob(os.path.join(root, f"{family}-{split}-*.tar")))
        paths.extend(matches or [os.path.join(root, f"{family}-{split}-00000.tar")])
    return paths


def family_weights(index: List[dict], alpha: float) -> np.ndarray:
    """Per-episode sampling weight that flattens the family mix by ``alpha``.

    The natural mix is heavily skewed — ``traverse`` is 65% of training steps and is
    also the family the policy already handles best, while ``item`` is 2.8% — so most
    gradient goes to the solved task. Weight ∝ ``(1 / family_step_share) ** alpha``:
    ``alpha=0`` leaves the mix alone, ``alpha=1`` equalises families entirely (which
    upweights ``item`` ~23x and is almost certainly too aggressive), and ``alpha=0.5``
    splits the difference. Weighting is by *steps*, not episodes, since that is what
    the gradient actually sees.
    """
    steps = collections.Counter()
    for ep in index:
        steps[ep["family"]] += ep["length"]
    total = sum(steps.values())
    w = np.array([(total / max(1, steps[ep["family"]])) ** alpha for ep in index],
                 dtype=np.float64)
    return w / w.sum()


class EpisodeStreamSampler(Sampler):
    """Batches of ``batch_size`` parallel episode streams, for carried memory.

    The default loader shuffles windows globally, so every window arrives with an
    empty attention memory. Because ``mem_len`` slots start zeroed and masked
    (``masked_attention.get_mask`` ANDs them against an all-False ``state_mask``),
    that means **training never populates the memory at all**: window position 0
    predicts from one frame of context and position 31 from thirty-two, averaging
    ~16, while at rollout every step gets the full window. This sampler closes that
    gap the way VPT does — hold ``batch_size`` lanes, each walking one episode's
    windows in order, so lane *j* of batch *b+1* continues lane *j* of batch *b* and
    the memory handed forward is the right one.

    Correctness does not depend on the lanes alone: each window also carries
    ``first``, true exactly on an episode's first window, and the recurrent core
    masks memory away there. So a boundary is always honoured even if a lane restarts.
    """

    def __init__(self, dataset: "ContraCrossViewDataset", batch_size: int,
                 weights: np.ndarray | None = None, shuffle: bool = True,
                 seed: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.weights = weights
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.streams = [w for w in dataset.episode_windows if w]
        self._length = max(1, sum(len(s) for s in self.streams) // batch_size)

    def __len__(self) -> int:
        return self._length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        n = len(self.streams)
        if self.weights is not None:
            # Sample episodes with replacement at the family-balanced rate. Drawing
            # without replacement would just reorder a fixed multiset and leave the
            # mix untouched, which is the opposite of the point.
            w = np.array([self.weights[i] for i, s in enumerate(self.dataset.episode_windows)
                          if s], dtype=np.float64)
            w = w / w.sum()
            order = list(rng.choice(n, size=self._length * self.batch_size + n, p=w))
        elif self.shuffle:
            order = list(rng.permutation(n))
        else:
            order = list(range(n))

        nxt = 0
        lanes: List[List[int]] = [[] for _ in range(self.batch_size)]
        for _ in range(self._length):
            batch = []
            for j in range(self.batch_size):
                while not lanes[j]:
                    lanes[j] = list(self.streams[order[nxt % len(order)]])
                    nxt += 1
                batch.append(lanes[j].pop(0))
            yield batch


class ContraDataModule:
    """Plain (non-Lightning) datamodule: indexes each split's shards, makes loaders.

    Kept framework-free so the loaders can also be driven from a script or a test;
    ``train.py`` hands it to Lightning via ``train_dataloader``/``val_dataloader``.
    """

    def __init__(self, shard_dir: str, configs: Sequence[str] = CONFIGS,
                 win_len: int = 32, image_size: int = 256, sigma_px: float = 12.0,
                 prev_action_keep_prob: float = 0.25, aux_size: int = 32,
                 carry_memory: bool = False, family_balance_alpha: float = 0.0,
                 batch_size: int = 4, num_workers: int = 4, prefetch_factor: int = 2,
                 cache_dir: str = "cache", seed: int = 0, want_entities: bool = False,
                 entity_sigma_px: float = 6.0):
        self.shard_dir = os.path.expanduser(shard_dir)
        self.train_tars = shard_paths(self.shard_dir, configs, "train")
        self.val_tars = shard_paths(self.shard_dir, configs, "val")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.carry_memory = carry_memory
        self.family_balance_alpha = family_balance_alpha
        self.seed = seed
        self.ds_kwargs = dict(win_len=win_len, image_size=image_size, sigma_px=sigma_px,
                              prev_action_keep_prob=prev_action_keep_prob,
                              aux_size=aux_size, seed=seed,
                              want_entities=want_entities,
                              entity_sigma_px=entity_sigma_px)
        # Two independent indexes, each cached under its own shard fingerprint.
        train_idx = load_or_build_index(self.train_tars, cache_dir)
        val_idx = load_or_build_index(self.val_tars, cache_dir)
        self.train_dataset = ContraCrossViewDataset(train_idx, **self.ds_kwargs)
        self.val_dataset = ContraCrossViewDataset(val_idx, **self.ds_kwargs)

        # Training action histogram, for the class-weighted BC loss. Free — build_index
        # already decoded every actions.npy to get the episode lengths.
        self.action_counts = np.zeros(NUM_ACTIONS, dtype=np.int64)
        for ep in train_idx:
            self.action_counts += np.asarray(ep["action_counts"], dtype=np.int64)

        win_mb = (win_len * image_size ** 2 * 3 + image_size ** 2 * 4) / 1e6
        host_gb = num_workers * prefetch_factor * batch_size * win_mb * 2 / 1e3  # x2: pin_memory
        print(f"[dataset] train {len(train_idx)} episodes / {len(self.train_dataset)} windows · "
              f"val {len(val_idx)} episodes / {len(self.val_dataset)} windows")
        print(f"[dataset] {win_mb:.1f} MB/window → ~{host_gb:.1f} GB host RAM in flight")
        steps = collections.Counter()
        for ep in train_idx:
            steps[ep["family"]] += ep["length"]
        total = sum(steps.values())
        print("[dataset] family mix (train steps): "
              + " · ".join(f"{f} {steps[f]/total:.1%}" for f in FAMILIES))
        if carry_memory:
            print("[dataset] carry_memory on: episode-stream sampler, "
                  f"{batch_size} parallel lanes")
        if family_balance_alpha:
            print(f"[dataset] family_balance_alpha={family_balance_alpha} (train only)")

    def _loader(self, dataset: "ContraCrossViewDataset", shuffle: bool,
                num_workers: int, persistent: bool) -> DataLoader:
        common = dict(
            num_workers=num_workers,
            prefetch_factor=self.prefetch_factor if num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=persistent and num_workers > 0,
            worker_init_fn=_worker_init,
        )
        # `shuffle` doubles as "this is the training loader" — val is never rebalanced.
        weights = (family_weights(dataset.index, self.family_balance_alpha)
                   if self.family_balance_alpha and shuffle else None)

        if self.carry_memory:
            # A batch_sampler owns batching entirely, so batch_size/shuffle/drop_last
            # must not also be passed. Lane order is preserved by the DataLoader even
            # with several workers, which is what makes the carried memory line up.
            # Val shuffles too (see _val_order) but never advances its epoch, so its
            # permutation is fixed and every validation pass scores the same windows.
            return DataLoader(
                dataset,
                batch_sampler=EpisodeStreamSampler(dataset, self.batch_size,
                                                   weights=weights, shuffle=True,
                                                   seed=self.seed),
                **common)

        sampler = None
        if weights is not None and shuffle:
            # Per-window weight is its episode's weight; windows of one episode are
            # interchangeable here since nothing is carried between them.
            per_window = np.array([weights[e] for e, _ in dataset.windows])
            sampler = WeightedRandomSampler(torch.from_numpy(per_window).double(),
                                            num_samples=len(dataset), replacement=True)
        elif not shuffle:
            sampler = self._val_order(dataset)
        return DataLoader(dataset, batch_size=self.batch_size,
                          shuffle=shuffle and sampler is None, sampler=sampler,
                          drop_last=shuffle, **common)

    def _val_order(self, dataset: "ContraCrossViewDataset") -> List[int]:
        """A fixed, seeded permutation of the val windows.

        In shard order the val set is contiguous by family — the index sorts by tar
        path, so it runs boss, item, kill, traverse — and ``limit_val_batches`` takes a
        *prefix*. At the shipped ``limit_val_batches: 50`` and ``batch_size: 16`` that
        is 800 of 3125 windows: all 345 boss, all 98 item, 357 of 658 kill, and **zero
        traverse**, which is 65% of the val set. Every ``val/`` number reported before
        this was computed on that skewed prefix.

        Permuting fixes the composition; seeding it (rather than shuffling per pass)
        keeps the subset identical across validations and across runs, so the metric
        moves only when the model does.
        """
        return np.random.default_rng(self.seed).permutation(len(dataset)).tolist()

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True,
                            num_workers=self.num_workers, persistent=True)

    def val_dataloader(self) -> DataLoader:
        # Not persistent, and fewer workers: validation runs for ~50 batches every
        # 1000 steps, so persistent val workers would idle >95% of the time while
        # holding prefetch buffers — that doubles the loader's resident set and
        # squeezes the page cache the training loader depends on.
        #
        # Never family-balanced: val must report the real mix, or the pooled number
        # stops meaning what the evaluator's pooled number means.
        return self._loader(self.val_dataset, shuffle=False,
                            num_workers=min(2, self.num_workers), persistent=False)


# ── whole-episode batching ────────────────────────────────────────────────────

def pad_episodes(items: List[Dict]) -> Dict:
    """Collate variable-length episodes, padding to the batch maximum.

    Not to the model's context: episodes average 100 frames against a 1024 context, so
    padding to the context would make ~90% of every batch mask. Padding to the batch
    maximum and grouping similar lengths together (:class:`LengthGroupedSampler`) brings
    that to roughly 1%.

    ``mask`` marks real steps and is what every loss must reduce over — a padded step
    has a fabricated action target of 0 and must never reach a gradient.
    """
    t = max(int(x["image"].shape[0]) for x in items)
    out: Dict = {}

    def stack(key, fill=0):
        ref = items[0][key]
        buf = torch.full((len(items), t, *ref.shape[1:]), fill, dtype=ref.dtype)
        for i, x in enumerate(items):
            n = x[key].shape[0]
            buf[i, :n] = x[key]
        return buf

    for key in ("image", "action", "mask", "goal_heatmap", "exist", "point",
                "n_goal_points", "prev_action", "prev_action_dropout", "first"):
        if key in items[0]:
            out[key] = stack(key)
    if "entity_heatmap" in items[0]:
        out["entity_heatmap"] = stack("entity_heatmap")

    # Per-episode members carry no time axis.
    out["cross_view"] = {
        "cross_view_image": torch.stack([x["cross_view"]["cross_view_image"] for x in items]),
        "cross_view_obj_mask": torch.stack(
            [x["cross_view"]["cross_view_obj_mask"] for x in items]),
        # One interaction id per episode. The windowed loader emitted it per timestep;
        # the policy takes a single id, so collapse it here rather than in the model.
        "cross_view_obj_id": torch.stack(
            [x["cross_view"]["cross_view_obj_id"][0] for x in items]),
    }
    out["family"] = torch.stack([x["family"] for x in items])
    out["seq_len"] = torch.tensor([int(x["image"].shape[0]) for x in items])
    return out


def pack_episodes(items: List[Dict]) -> Dict:
    """Collate complete episodes by concatenation; ``seq_len`` preserves boundaries."""
    out: Dict = {}
    for key in ("image", "action", "mask", "goal_heatmap", "exist", "point",
                "n_goal_points", "prev_action", "prev_action_dropout", "first"):
        if key in items[0]:
            out[key] = torch.cat([x[key] for x in items], dim=0)
    out["cross_view"] = {
        "cross_view_image": torch.stack(
            [x["cross_view"]["cross_view_image"] for x in items]),
        "cross_view_obj_mask": torch.stack(
            [x["cross_view"]["cross_view_obj_mask"] for x in items]),
        "cross_view_obj_id": torch.stack(
            [x["cross_view"]["cross_view_obj_id"][0] for x in items]),
    }
    out["family"] = torch.stack([x["family"] for x in items])
    out["seq_len"] = torch.tensor([int(x["image"].shape[0]) for x in items])
    return out


class LengthGroupedSampler(Sampler):
    """Batches of similar-length episodes, shuffled.

    Sorting globally would put every long episode in one batch and, because length
    correlates with family (traverse long, item short), make a batch nearly one family —
    which changes the gradient's family mix step to step. So sort within a *pool* of
    ``pool_batches`` batches, then shuffle the batch order: length is close enough for
    padding, and families still mix.

    **The order is always permuted, including when ``shuffle=False``.** The index is
    ordered by tar path, so walking it directly groups every family together: a caller
    scoring only the first N batches would miss whole families. That is not
    hypothetical — a 60-batch validation covered 240 of 846 episodes and contained zero
    `traverse`, which is 65% of all decision steps, because `traverse` starts at index
    392. ``shuffle`` therefore controls only whether the permutation *varies between
    epochs*: ``True`` for training, ``False`` for validation, where the subset must stay
    representative **and** identical across calls or the trend is unreadable.
    """

    def __init__(self, lengths: Sequence[int], batch_size: int,
                 pool_batches: int = 32, seed: int = 0, shuffle: bool = True):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.pool = batch_size * pool_batches
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + (self.epoch if self.shuffle else 0))
        if self.shuffle:
            self.epoch += 1
        order = rng.permutation(len(self.lengths))
        batches = []
        for start in range(0, len(order), self.pool):
            pool = order[start:start + self.pool]
            pool = pool[np.argsort([self.lengths[i] for i in pool], kind="stable")]
            batches += [pool[i:i + self.batch_size].tolist()
                        for i in range(0, len(pool), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches
