"""The discrete Contra action space — the policy's output vocabulary.

The dataset stores raw length-9 NES button vectors in bit order
``[B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]``, but every trace was
produced by the MC searcher under ``contra_nes_data``'s ``agent/baseline.yaml``,
so all 777k recorded steps fall inside a 21-action set (verified by scanning all
four shards: zero out-of-vocabulary vectors). That makes a single 21-way
categorical the right policy head — it replaces ROCKET-2's hierarchical
Minecraft head (buttons + camera bins) with the one thing Contra needs.

The action table is duplicated here rather than imported so this repo does not
depend on ``contra_nes_data`` at train time. :func:`check_matches_source` compares
against the YAML when that repo is available, so drift is caught rather than
silently mistrained on.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# Index order is the YAML insertion order of contra_nes_data/src/agent/baseline.yaml
# and must stay stable — a checkpoint's logits are meaningless if it changes.
ACTION_NAMES: tuple[str, ...] = (
    "_", "J", "F", "L", "LJ", "LF", "R", "RJ", "RF", "U", "UJ",
    "UF", "D", "DJ", "DF", "UL", "ULJ", "ULF", "UR", "URJ", "URF",
)

# NES buttons: [B, NULL, SELECT, START, UP, DOWN, LEFT, RIGHT, A]
ACTION_VECTORS: tuple[tuple[int, ...], ...] = (
    (0, 0, 0, 0, 0, 0, 0, 0, 0),   # _    idle
    (0, 0, 0, 0, 0, 0, 0, 0, 1),   # J    jump
    (1, 0, 0, 0, 0, 0, 0, 0, 0),   # F    fire
    (0, 0, 0, 0, 0, 0, 1, 0, 0),   # L
    (0, 0, 0, 0, 0, 0, 1, 0, 1),   # LJ
    (1, 0, 0, 0, 0, 0, 1, 0, 0),   # LF
    (0, 0, 0, 0, 0, 0, 0, 1, 0),   # R
    (0, 0, 0, 0, 0, 0, 0, 1, 1),   # RJ
    (1, 0, 0, 0, 0, 0, 0, 1, 0),   # RF
    (0, 0, 0, 0, 1, 0, 0, 0, 0),   # U
    (0, 0, 0, 0, 1, 0, 0, 0, 1),   # UJ
    (1, 0, 0, 0, 1, 0, 0, 0, 0),   # UF
    (0, 0, 0, 0, 0, 1, 0, 0, 0),   # D
    (0, 0, 0, 0, 0, 1, 0, 0, 1),   # DJ
    (1, 0, 0, 0, 0, 1, 0, 0, 0),   # DF
    (0, 0, 0, 0, 1, 0, 1, 0, 0),   # UL
    (0, 0, 0, 0, 1, 0, 1, 0, 1),   # ULJ
    (1, 0, 0, 0, 1, 0, 1, 0, 0),   # ULF
    (0, 0, 0, 0, 1, 0, 0, 1, 0),   # UR
    (0, 0, 0, 0, 1, 0, 0, 1, 1),   # URJ
    (1, 0, 0, 0, 1, 0, 0, 1, 0),   # URF
)

NUM_ACTIONS = len(ACTION_VECTORS)
SKIP = 3  # NES frames one decision is held for (baseline.yaml `skip`)

_VEC_TO_IDX = {v: i for i, v in enumerate(ACTION_VECTORS)}

# A 9-bit button vector is a number in [0, 512), so the whole mapping is one lookup
# table. The obvious per-row Python loop costs ~4ms per training window — small until
# you notice it runs once per window in every dataloader worker.
_BIT_WEIGHTS = (1 << np.arange(9)).astype(np.int64)
_LUT = np.full(512, -1, dtype=np.int64)
for _i, _v in enumerate(ACTION_VECTORS):
    _LUT[int(np.asarray(_v, dtype=np.int64) @ _BIT_WEIGHTS)] = _i


def actions_np(dtype=np.uint8) -> np.ndarray:
    """All button vectors as a (NUM_ACTIONS, 9) array."""
    return np.array(ACTION_VECTORS, dtype=dtype)


def vectors_to_indices(vectors: np.ndarray) -> np.ndarray:
    """(T, 9) uint8 button vectors → (T,) int64 action indices.

    Raises on an out-of-vocabulary vector: silently mapping it to some nearest
    action would corrupt the behaviour-cloning target, and the shard scan says it
    should never happen.
    """
    vectors = np.asarray(vectors)
    assert vectors.ndim == 2 and vectors.shape[1] == 9, f"expected (T, 9), got {vectors.shape}"
    out = _LUT[vectors.astype(np.int64) @ _BIT_WEIGHTS]
    bad = np.flatnonzero(out < 0)
    if len(bad):
        i = int(bad[0])
        key = tuple(int(v) for v in vectors[i])
        raise ValueError(f"button vector {key} at step {i} is not in the {NUM_ACTIONS}-action "
                         f"baseline set; the shard was exported with a different action config")
    return out


def indices_to_vectors(indices: Sequence[int], dtype=np.uint8) -> np.ndarray:
    """(T,) action indices → (T, 9) NES button vectors, for rollout in the emulator."""
    return actions_np(dtype)[np.asarray(indices, dtype=np.int64)]


def check_matches_source(yaml_path: str | None = None) -> bool:
    """True if this table still matches ``contra_nes_data``'s baseline.yaml.

    Returns False (rather than raising) when that repo isn't reachable, so this is
    safe to call from a test on a machine that only has the shards.
    """
    import yaml

    if yaml_path is None:
        yaml_path = os.path.expanduser("~/code/contra_nes_data/src/agent/baseline.yaml")
    if not os.path.exists(yaml_path):
        return False
    with open(yaml_path) as fh:
        spec = yaml.safe_load(fh)
    names = tuple(spec["actions"].keys())
    vecs = tuple(tuple(v) for v in spec["actions"].values())
    return names == ACTION_NAMES and vecs == ACTION_VECTORS and spec.get("skip", 8) == SKIP
