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


class DifficultyTracker:
    """Per-task success estimate, used to sample tasks whose groups will carry signal.

    **The problem it solves.** A group only produces gradient when its G rollouts
    disagree. Measured on run ``2026-08-02/11-48-03``: the policy scores ~83% on *train*
    tasks (against 67-76% on val, which is where G was chosen from), so 59% of groups
    came back all-success and 58% of the rollout budget bought nothing.

    **The weight is the thing we actually want to maximise**, not a proxy: the
    probability that a group of size G is *not* all-agreeing,

        p_useful = 1 - p**G - (1-p)**G

    which is exactly the chance the group survives :func:`filter_groups`. It peaks at
    p=0.5 and falls off at both ends — a task solved 95% of the time is nearly as
    useless as one solved 3.5% of the time, which is the two-sidedness doc/0004 §5
    records. Using survival probability rather than the Bernoulli variance ``4p(1-p)``
    matters at the tails: at G=8 it downweights boss 3.1x relative to a 83% task, where
    ``4p(1-p)`` would downweight it 4.2x.

    **The estimate is hierarchical, and it has to be.** There are 6438 train tasks and
    an update draws ~32 of them, so after 100 updates most tasks have been seen once or
    never — a per-task estimate alone would still be at its prior when the run ends, and
    measured exactly that: 43 tasks tracked after 4 updates. But there are only **13
    labels** (495 tasks each), so a label's rate is well estimated within a few updates.
    A task therefore shrinks toward *its label's* observed rate rather than toward 0.5:

        p(label) = (s_label + w0/2) / (n_label + w0)
        p(task)  = (s_task + prior * p(label)) / (n_task + prior)

    An unseen task inherits a real measurement instead of an uninformative 0.5, so the
    bias is worth something from update ~3 rather than never. Task-level counts then
    refine it wherever they exist. ``traverse`` has a single label, so there it degrades
    to a family-level prior — still far better than 0.5.

    **Estimates must stay fresh**, because the policy is what makes a task easy and the
    policy is moving. Counts decay geometrically on each observation, so the effective
    window is the last ``G/(1-decay)`` rollouts — 80 at the defaults, and bounded, so an
    old estimate cannot outvote a new one.

    **Nothing is ever excluded.** ``min_weight`` floors the weight, so a task the policy
    currently always fails is sampled rarely rather than never. Without that floor a
    task could not be rediscovered once it became learnable — which for boss is the
    entire hope.
    """

    def __init__(self, group_size: int, decay: float = 0.9, prior: float = 1.0,
                 min_weight: float = 0.05, label_prior: float = 2.0):
        if not 0.0 < decay <= 1.0:
            raise ValueError(f"decay must be in (0, 1], got {decay}")
        self.group_size, self.decay = int(group_size), float(decay)
        self.prior, self.min_weight = float(prior), float(min_weight)
        self.label_prior = float(label_prior)
        self.s: Dict[str, float] = {}
        self.n: Dict[str, float] = {}
        self.ls: Dict[str, float] = {}
        self.ln: Dict[str, float] = {}

    def observe(self, uid: str, label: str, successes: float,
                attempts: float) -> None:
        d = self.decay
        self.s[uid] = self.s.get(uid, 0.0) * d + float(successes)
        self.n[uid] = self.n.get(uid, 0.0) * d + float(attempts)
        self.ls[label] = self.ls.get(label, 0.0) * d + float(successes)
        self.ln[label] = self.ln.get(label, 0.0) * d + float(attempts)

    def observe_episodes(self, episodes) -> None:
        """One update's rollouts, grouped by task. A group of G gives G attempts at
        once, which is a far better estimate than one rollout would be."""
        agg: Dict[tuple, list] = {}
        for e in episodes:
            a = agg.setdefault((e.task_uid, e.task_label), [0.0, 0.0])
            a[0] += float(e.reward > 0)
            a[1] += 1.0
        for (uid, label), (s, n) in agg.items():
            self.observe(uid, label, s, n)

    def p_label(self, label: str) -> float:
        """The label's success rate; 0.5 until anything of that label has been rolled.
        Decays with the same window, so it tracks the policy rather than its history."""
        s, n = self.ls.get(label, 0.0), self.ln.get(label, 0.0)
        w0 = self.label_prior
        return (s + 0.5 * w0) / (n + w0)

    def p_hat(self, uid: str, label: str = "") -> float:
        """Task rate shrunk toward its label's rate — see the class docstring."""
        s, n = self.s.get(uid, 0.0), self.n.get(uid, 0.0)
        return (s + self.prior * self.p_label(label)) / (n + self.prior)

    def weight(self, task) -> float:
        """P(a group of this task survives filtering), floored by ``min_weight``.

        Takes the task rather than a uid because the label is half the estimate.
        """
        uid = getattr(task, "uid", task)
        label = getattr(task, "label", "")
        p, g = self.p_hat(uid, label), self.group_size
        return max(self.min_weight, 1.0 - p ** g - (1.0 - p) ** g)

    def state(self) -> dict:
        return {"s": dict(self.s), "n": dict(self.n),
                "ls": dict(self.ls), "ln": dict(self.ln)}

    def load_state(self, state: dict) -> None:
        self.s, self.n = dict(state.get("s", {})), dict(state.get("n", {}))
        self.ls, self.ln = dict(state.get("ls", {})), dict(state.get("ln", {}))

    def stats(self) -> Dict[str, float]:
        out: Dict[str, float] = {"sampler/tracked_tasks": float(len(self.n)),
                                 "sampler/tracked_labels": float(len(self.ln))}
        for label in sorted(self.ln):
            out[f"sampler/p_label/{label}"] = self.p_label(label)
        return out


class GroupSampler:
    """Draws one task and yields it ``group_size`` times — GRPO's premise.

    GRPO replaces a learned critic with the *group* mean: sample G rollouts of the same
    prompt, and each one's advantage is its return minus what its siblings got. That
    only works if "the same prompt" is reproducible, which here it is — a task is a
    savestate, so every rollout in a group starts from bit-identical emulator state and
    differs only in the policy's sampling.

    Why this beats the critic we had: with ``gamma=1.0`` and a binary episodic reward,
    ``V(s)`` *is* P(success from here). The previous PPO run's critic reached
    ``explained_variance`` ~0.33 after starting negative — it explained a third of the
    return variance, and the rest went into the advantages as noise. A group mean
    estimates the same quantity directly, with no second network to train, no GAE, and
    no critic-warmup phase.

    **The known failure**: a group whose members all fail has zero advantage spread and
    contributes no gradient. Boss succeeds ~3.5% of the time, so at G=8 roughly
    ``0.965**8 = 75%`` of boss groups are all-failure. ``group_stats`` reports the
    degenerate fraction so this is visible rather than inferred — it is the central open
    question for GRPO on this task (doc/0003 §6).
    """

    def __init__(self, sampler: "TaskSampler", group_size: int,
                 difficulty: Optional["DifficultyTracker"] = None,
                 candidates: int = 1, seed: int = 0):
        if group_size < 2:
            raise ValueError(f"group_size must be at least 2 to have a baseline, "
                             f"got {group_size}")
        if candidates < 1:
            raise ValueError(f"candidates must be at least 1, got {candidates}")
        self.sampler = sampler
        self.group_size = group_size
        self.difficulty = difficulty
        self.candidates = int(candidates)
        self.rng = np.random.default_rng(seed)

    def sample_group(self) -> List["RLTask"]:
        """``group_size`` references to one task. Identical objects on purpose: the
        rollout layer restores the same savestate for each.

        With a :class:`DifficultyTracker` and ``candidates > 1`` this is *tournament
        selection*: draw that many tasks from the base mixture, then pick one with
        probability proportional to how likely its group is to survive filtering.

        Tournament rather than a global weighted draw over all 6438 tasks, for two
        reasons. It preserves the base mixture's shape — the family/label balance
        :class:`TaskSampler` exists to enforce is still what generates the candidates,
        and difficulty only reorders within them. And ``candidates`` is a single dial
        for how aggressive the bias is: 1 disables it entirely, and larger values
        concentrate harder on p~0.5 without ever reaching a deterministic argmax.
        """
        if self.difficulty is None or self.candidates == 1:
            return [self.sampler.sample()] * self.group_size
        pool = [self.sampler.sample() for _ in range(self.candidates)]
        w = np.array([self.difficulty.weight(t) for t in pool], dtype=np.float64)
        task = pool[int(self.rng.choice(len(pool), p=w / w.sum()))]
        return [task] * self.group_size

    def sample_groups(self, n_groups: int) -> List[List["RLTask"]]:
        return [self.sample_group() for _ in range(n_groups)]

    def observe(self, episodes) -> None:
        """Feed an update's rollouts back into the difficulty estimate. No-op without
        a tracker, so the trainer calls it unconditionally."""
        if self.difficulty is not None:
            self.difficulty.observe_episodes(episodes)

    def stats(self) -> Dict[str, float]:
        return self.difficulty.stats() if self.difficulty is not None else {}

    def state(self) -> dict:
        st = {"sampler": self.sampler.state(), "rng": self.rng.bit_generator.state}
        if self.difficulty is not None:
            st["difficulty"] = self.difficulty.state()
        return st

    def load_state(self, state: dict) -> None:
        # Tolerates checkpoints written before difficulty tracking existed.
        self.sampler.load_state(state.get("sampler", state))
        if "rng" in state:
            self.rng.bit_generator.state = state["rng"]
        if self.difficulty is not None and "difficulty" in state:
            self.difficulty.load_state(state["difficulty"])
