"""Where an RL episode starts, what it is asked to do, and which task gets sampled.

An RL episode needs two things that live in two different places:

* the **initial emulator save-state** and the task metadata that
  ``TaskMaker.goal_reached`` reads — these are in the task ``.npz`` under
  ``contra_nes_data/game_trace/tasks/<kind>/<label>/<uid>.npz``;
* the **cross-view prompt** the policy was trained against — the goal frame, the goal
  blob channel and the interaction id.

The prompt is taken from the shard (``<uid>.goal.png`` plus ``goal_points`` in
``<uid>.json``) rather than re-rendered. Re-rendering means replaying the whole task
through the emulator once per episode start, which is what ``contra_nes_evaluation``
must do (it starts from the ``.npz`` alone) and costs more than the rollout itself.
Reading it back out of the shard is both cheaper and *exactly* the tensor the policy
trained on, byte for byte, since :mod:`contra_policy.dataset` builds it from the same
two members with the same ``goal_mask``.

The join is ``(family, uid)``: the shard name is the family and the member stem is the
uid, which is unique within a family. Both sides carry ``split``, and both are
``contra_nes_data``'s — this module reads it and never derives it. On the shipped
dataset the join is total (6904 train tasks ↔ 6904 train shard episodes, 846 ↔ 846
val), and :meth:`TaskCatalog.__init__` reports any task it had to drop rather than
silently training on a smaller set.

**The val split never enters an RL worker.** :class:`TaskCatalog` filters on the split
it was asked for and re-checks every task; :meth:`TaskCatalog.assert_split` is called
by the collector before the first episode.
"""

from __future__ import annotations

import collections
import csv
import glob
import io
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from contra_policy.dataset import FAMILIES, load_or_build_index, shard_paths
from contra_policy.goal import goal_mask, interaction_id

DEFAULT_TASK_ROOT = "~/code/contra_nes_data/game_trace/tasks"
DEFAULT_SHARD_DIR = "~/code/contra_nes_data/game_trace/hf"


@dataclass(frozen=True)
class RLTask:
    """One trainable task: a save-state on disk, plus how to name it in a report."""

    path: str        # absolute path to the task .npz (initial state + expert actions)
    family: str      # kill | item | traverse | boss — the shard it exported into
    label: str       # fine-grained class, e.g. "red_turret", "pick_laser", "level1"
    uid: str         # filename stem; unique within a family
    split: str       # "train" | "val", assigned by contra_nes_data per source trace

    @property
    def key(self) -> Tuple[str, str]:
        return (self.family, self.uid)


@dataclass(frozen=True)
class GoalPrompt:
    """The cross-view prompt at network input resolution — the policy's goal input."""

    image: np.ndarray        # (S, S, 3) uint8, the shard's goal.png
    mask: np.ndarray         # (S, S) uint8, the goal blob regenerated from goal_points
    interaction: int         # index into contra_policy.goal.INTERACTIONS


# ── discovery ─────────────────────────────────────────────────────────────────

def _splits_from_manifest(root: str, family: str) -> Dict[str, str]:
    """``{uid: split}`` for one family, read from ``contra_nes_data``'s manifest.csv.

    The manifest is that repo's own derived index and carries the ``split`` column, so
    one file read replaces opening several thousand ``.npz`` files for a single string
    each. The ``.npz`` stays the source of truth — :func:`discover_tasks` falls back to
    it for anything the manifest does not mention.
    """
    path = os.path.join(root, family, "manifest.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "split" not in rows[0]:
        return {}
    return {os.path.splitext(os.path.basename(r["file"]))[0]: r["split"] for r in rows}


def discover_tasks(task_root: str = DEFAULT_TASK_ROOT,
                   families: Sequence[str] = FAMILIES) -> List[RLTask]:
    """Every task ``.npz`` under ``task_root/<family>/<label>/<uid>.npz``.

    The directory layout *is* the labelling — that is how the makers write it — so
    nothing has to be parsed out of the npz to bucket a report.
    """
    root = os.path.expanduser(task_root)
    out: List[RLTask] = []
    for family in families:
        splits = _splits_from_manifest(root, family)
        for path in sorted(glob.glob(os.path.join(root, family, "*", "*.npz"))):
            label = os.path.basename(os.path.dirname(path))
            uid = os.path.splitext(os.path.basename(path))[0]
            out.append(RLTask(path=path, family=family, label=label, uid=uid,
                              split=splits.get(uid, "")))
    if not out:
        raise FileNotFoundError(
            f"no tasks found under {root}/<family>/<label>/*.npz "
            f"(families={list(families)}) — is contra_nes_data checked out?")
    return [_fill_split(t) if not t.split else t for t in out]


def _fill_split(task: RLTask) -> RLTask:
    """Recover a task's split from its ``.npz`` when the manifest could not supply it."""
    from dataclasses import replace

    from task_maker.base import load_task

    return replace(task, split=str(load_task(task.path).split))


# ── catalog ───────────────────────────────────────────────────────────────────

class TaskCatalog:
    """Tasks of one split, joined to their shard prompts and grouped by family/label.

    Two small LRU caches sit in front of the two per-episode reads. Neither is an
    optimisation for its own sake: at ~20 episode starts per second a cold
    ``load_task`` (npz decompression of a save-state) and a cold PNG decode are a
    visible fraction of the collector's wall clock, and both objects are immutable.
    Sizes are in *entries*; a prompt is ``image + mask`` ≈ 260 KB at 256 px and a
    cached segment is dominated by its save-state, so the defaults are ~35 MB and
    ~20 MB respectively per worker process.
    """

    def __init__(self, task_root: str = DEFAULT_TASK_ROOT,
                 shard_dir: str = DEFAULT_SHARD_DIR,
                 families: Sequence[str] = FAMILIES, split: str = "train",
                 image_size: int = 256, sigma_px: float = 12.0,
                 cache_dir: str = "cache", prompt_cache: int = 128,
                 segment_cache: int = 256, verbose: bool = True):
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.split = split
        self.families = tuple(families)
        self.image_size = image_size
        self.sigma_px = sigma_px
        self.prompt_cache_size = prompt_cache
        self.segment_cache_size = segment_cache

        shard_dir = os.path.expanduser(shard_dir)
        index = load_or_build_index(shard_paths(shard_dir, self.families, split), cache_dir)
        self._shard: Dict[Tuple[str, str], dict] = {
            (ep["family"], ep["uid"]): ep for ep in index}

        all_tasks = discover_tasks(task_root, self.families)
        in_split = [t for t in all_tasks if t.split == split]
        self.tasks = [t for t in in_split if t.key in self._shard]
        dropped = len(in_split) - len(self.tasks)
        if not self.tasks:
            raise RuntimeError(
                f"no {split!r} task could be joined to a shard episode under {shard_dir}; "
                f"the shards and game_trace/tasks are out of sync")
        if dropped and verbose:
            print(f"[rl.tasks] {dropped} {split} task(s) have no shard episode and were "
                  f"dropped (no cross-view prompt available)")

        # family -> label -> tasks, the structure the balanced sampler walks.
        self.by_family: Dict[str, Dict[str, List[RLTask]]] = {}
        for t in self.tasks:
            self.by_family.setdefault(t.family, {}).setdefault(t.label, []).append(t)
        self.present_families = tuple(f for f in self.families if f in self.by_family)

        self._fh: Dict[str, io.BufferedReader] = {}
        self._prompts: "collections.OrderedDict[Tuple[str, str], GoalPrompt]" = \
            collections.OrderedDict()
        self._segments: "collections.OrderedDict[Tuple[str, str], object]" = \
            collections.OrderedDict()

        if verbose:
            counts = collections.Counter(t.family for t in self.tasks)
            print(f"[rl.tasks] {len(self.tasks)} {split} tasks · "
                  + " · ".join(f"{f} {counts[f]}" for f in self.present_families))

    def __len__(self) -> int:
        return len(self.tasks)

    def assert_split(self, expected: str = "train") -> None:
        """Hard guard against a held-out task reaching a rollout worker.

        Cheap, and worth running rather than trusting the constructor: an evaluation
        number computed on tasks the policy was fine-tuned on is not wrong in a way
        anyone notices from the curve.
        """
        if self.split != expected:
            raise RuntimeError(
                f"this catalog holds the {self.split!r} split but {expected!r} was "
                f"required; RL must never train on held-out tasks")
        bad = [t.uid for t in self.tasks if t.split != expected][:5]
        if bad:
            raise RuntimeError(f"tasks not in the {expected!r} split leaked into the "
                               f"catalog: {bad}")

    # -- per-task assets ---------------------------------------------------

    def _read_member(self, task: RLTask, ext: str) -> bytes:
        ep = self._shard[task.key]
        fh = self._fh.get(ep["tar"])
        if fh is None:
            fh = self._fh[ep["tar"]] = open(ep["tar"], "rb")
        offset, size = ep["members"][ext]
        fh.seek(offset)
        return fh.read(size)

    def prompt(self, task: RLTask) -> GoalPrompt:
        """The task's cross-view prompt, built exactly as ``dataset.__getitem__`` does."""
        hit = self._prompts.get(task.key)
        if hit is not None:
            self._prompts.move_to_end(task.key)
            return hit
        meta = json.loads(self._read_member(task, "json"))
        bgr = cv2.imdecode(np.frombuffer(self._read_member(task, "goal.png"), np.uint8),
                           cv2.IMREAD_COLOR)
        image = resize_to_input(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.image_size)
        mask = goal_mask(meta["goal_points"], self.image_size, self.sigma_px)
        prompt = GoalPrompt(image=np.ascontiguousarray(image),
                            mask=(mask * 255.0).round().astype(np.uint8),
                            interaction=int(interaction_id(meta)))
        self._prompts[task.key] = prompt
        while len(self._prompts) > self.prompt_cache_size:
            self._prompts.popitem(last=False)
        return prompt

    def segment(self, task: RLTask):
        """The task's ``Segment`` — save-state, expert actions, ``goal_reached`` meta.

        ``load_task`` is ``contra_nes_data``'s; the on-disk format is not re-read here.
        """
        hit = self._segments.get(task.key)
        if hit is not None:
            self._segments.move_to_end(task.key)
            return hit
        from task_maker.base import load_task

        seg = load_task(task.path)
        self._segments[task.key] = seg
        while len(self._segments) > self.segment_cache_size:
            self._segments.popitem(last=False)
        return seg

    def close(self) -> None:
        for fh in self._fh.values():
            fh.close()
        self._fh.clear()

    # -- reporting ---------------------------------------------------------

    def label_index(self) -> Dict[Tuple[str, str], int]:
        """Stable ``(family, label) -> index`` map, for per-label metric arrays."""
        keys = sorted({(t.family, t.label) for t in self.tasks})
        return {k: i for i, k in enumerate(keys)}


def resize_to_input(img: np.ndarray, image_size: int) -> np.ndarray:
    """224x240 NES screen → square network input, exactly as the dataset does it.

    INTER_AREA, not the cv2 default: bullets are ~2 px and bilinear/nearest sampling
    drops them, which is precisely the signal the frozen encoder was pretrained to
    keep. Duplicated arithmetic here would be a silent distribution shift between
    training and rollout, so this mirrors ``ContraCrossViewDataset._resize`` exactly.
    """
    if img.shape[:2] == (image_size, image_size):
        return img
    return cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)


# ── sampling ──────────────────────────────────────────────────────────────────

class TaskSampler:
    """The episode-start distribution: a mixture of the natural mix and a balanced one.

    Uniform sampling over training tasks would spend most of the gradient on
    ``traverse`` — 3693 of 6904 tasks, and already the strongest family at 83.5%
    completion — while ``item`` (455) and ``boss`` (466) get 6.6% each. The mixture
    keeps the real distribution in the loop and doubles exposure to the weak families:

    * with probability ``natural_fraction``, draw uniformly from every training task;
    * otherwise draw a **family** uniformly (times ``family_multiplier``), then a
      **label** uniformly within it, then a **task** uniformly within that label.

    The label level matters as much as the family level: ``item`` is 7 labels ranging
    from 3 to 120 tasks, so a family-uniform-then-task-uniform draw would still be 26%
    ``pick_laser`` and 0.7% ``avoid_spread``.

    No importance correction is applied, deliberately. The mixture *defines* the
    training objective as a multi-objective one; reweighting the gradient back to the
    natural distribution would undo the entire point while adding variance. Final
    evaluation stays on the unchanged natural val set, and both pooled and macro
    completion are reported, which is where the difference is supposed to show up.
    """

    def __init__(self, catalog: TaskCatalog, natural_fraction: float = 0.7,
                 balanced_family_fraction: float = 0.3,
                 family_multiplier: Optional[Dict[str, float]] = None,
                 seed: int = 0):
        total = natural_fraction + balanced_family_fraction
        if total <= 0:
            raise ValueError("natural_fraction + balanced_family_fraction must be > 0")
        self.natural_fraction = natural_fraction / total
        self.catalog = catalog
        self.families = list(catalog.present_families)
        mult = dict(family_multiplier or {})
        w = np.array([float(mult.get(f, 1.0)) for f in self.families], dtype=np.float64)
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError(f"family_multiplier must be non-negative and not all zero: {mult}")
        self.family_weights = w / w.sum()
        self.rng = np.random.default_rng(seed)

    def sample(self) -> RLTask:
        if self.rng.random() < self.natural_fraction:
            return self.catalog.tasks[int(self.rng.integers(len(self.catalog.tasks)))]
        family = self.families[int(self.rng.choice(len(self.families), p=self.family_weights))]
        labels = self.catalog.by_family[family]
        label = sorted(labels)[int(self.rng.integers(len(labels)))]
        group = labels[label]
        return group[int(self.rng.integers(len(group)))]

    # -- resumption --------------------------------------------------------

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state}

    def load_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state["rng"]

    def expected_family_mix(self) -> Dict[str, float]:
        """The episode-start share of each family under the configured mixture.

        Printed at startup so the mixture is a number in the log rather than an
        intention in a config file.
        """
        n = len(self.catalog.tasks)
        out: Dict[str, float] = {}
        for i, f in enumerate(self.families):
            natural = len(self.catalog.by_family[f]) and \
                sum(len(v) for v in self.catalog.by_family[f].values()) / n
            out[f] = (self.natural_fraction * natural
                      + (1.0 - self.natural_fraction) * float(self.family_weights[i]))
        return out
