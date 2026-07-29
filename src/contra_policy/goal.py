"""Goal-side primitives: PPU→screen geometry, goal masks, interaction ids, aux targets.

ROCKET-2's cross-view prompt is a triple ``(cross_view_image, cross_view_obj_mask,
cross_view_obj_id)``: a reference frame, a binary mask of the object being pointed at
in that frame, and an id for the *kind* of interaction intended. Contra's shards
carry the same information in a different packing:

* ``<uid>.goal.png`` is the reference frame with the goal blob already composited in
  (``env.entity.overlay_pointers``) — this is the cross-view image.
* ``goal_points`` in the JSON are the marked PPU coordinates, so the mask channel is
  *regenerated* here rather than shipped. That keeps ``sigma`` a training knob and
  gives the mask backbone a clean single-channel input instead of asking it to
  recover the blob from the tinted RGB.
* ``goal_when``/``goal_kind``/``kind`` name the interaction; :func:`interaction_id`
  folds them into the five ids the embedding table is indexed by.

Coordinate note, and the easiest thing in this pipeline to get wrong: every position
in the shards (``goal_points``, ``centroids``) is a **PPU** coordinate on the native
256×240 NES frame, while the stored frames are stable_retro's 224×240 overscan crop.
Screen coords are therefore ``(x - 8, y - 8)``; :func:`ppu_to_norm` is the single
place that shift lives.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

PPU_W, PPU_H = 256, 240      # native NES frame
VIS_W, VIS_H = 240, 224      # stable_retro screen crop (what the shards store)
XOFF, YOFF = (PPU_W - VIS_W) // 2, (PPU_H - VIS_H) // 2   # 8, 8

# Interaction vocabulary, derived from (goal_when, goal_kind, kind) in the shard
# JSON. Order is the embedding index order — stable, like ACTION_NAMES.
INTERACTIONS: tuple[str, ...] = ("kill", "pick", "avoid", "traverse", "boss")
NUM_INTERACTIONS = len(INTERACTIONS)
NO_INTERACTION = -1   # padded / absent goal; embedded at index 0 after the +1 offset

_INTERACTION_INDEX = {n: i for i, n in enumerate(INTERACTIONS)}


def interaction_id(meta: dict) -> int:
    """Map a shard sample's metadata to an interaction id in ``[0, NUM_INTERACTIONS)``.

    The five task families in the shards are distinguished by ``goal_when`` plus,
    for the item shard, the ``kind`` field that says whether the powerup is to be
    picked up or dodged.
    """
    when, kind = meta.get("goal_when"), meta.get("kind")
    if when == "boss":
        name = "boss"
    elif when == "last":
        name = "traverse"
    elif kind in ("pick", "avoid"):
        name = kind
    else:
        name = "kill"
    return _INTERACTION_INDEX[name]


def ppu_to_norm(points: Iterable[Sequence[float]]) -> np.ndarray:
    """PPU (x, y) pairs → (N, 2) float32 in [0, 1] screen space (x first).

    Values are *not* clipped: a point can legitimately sit just off the visible
    crop, and the caller decides what to do with that (the aux loss is gated on the
    exist flag, so off-screen targets simply do not contribute).
    """
    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.empty_like(pts)
    out[:, 0] = (pts[:, 0] - XOFF) / VIS_W
    out[:, 1] = (pts[:, 1] - YOFF) / VIS_H
    return out


def goal_mask(points: Iterable[Sequence[float]], size: int, sigma_px: float = 12.0) -> np.ndarray:
    """(size, size) float32 goal channel: max of Gaussian blobs at ``points``.

    Reproduces ``env.entity.overlay_pointers``' heatmap — same PPU shift, same
    elementwise max over blobs (occupancy, not a sum) — but rendered directly at the
    network's square input resolution, so ``sigma_px`` is given in *source screen*
    pixels and scaled here. An empty point list gives an all-zero mask, which is what
    the ``goal_when="last"`` traverse episodes need when the goal is off-screen.
    """
    hm = np.zeros((size, size), dtype=np.float32)
    pts = np.asarray(list(points), dtype=np.float32).reshape(-1, 2)
    if len(pts) == 0:
        return hm
    sx, sy = size / VIS_W, size / VIS_H
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for px, py in pts:
        cx, cy = (px - XOFF) * sx, (py - YOFF) * sy
        s = sigma_px * (sx + sy) * 0.5
        np.maximum(hm, np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * s * s)), out=hm)
    return hm


def points_to_target(points: Iterable[Sequence[float]], half_size_px: float = 8.0
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame aux targets from that frame's goal centroids.

    Returns ``(point, bbox)`` — ``point`` is the (x, y) centroid mean and ``bbox`` is
    ``(x_min, y_min, x_max, y_max)``, both normalised to [0, 1] screen space.

    ROCKET-2 derives its bbox from a segmentation mask, which Contra's RAM-derived
    targets do not have: the shards give entity *positions*, not extents. Each
    centroid is therefore given a fixed ``half_size_px`` box (roughly an NES sprite)
    and the bbox is their union — so for a single enemy it is a sprite-sized box, and
    for the boss it spans all live components, which is the useful signal there.
    """
    n = ppu_to_norm(points)
    if len(n) == 0:
        return np.zeros(2, dtype=np.float32), np.zeros(4, dtype=np.float32)
    hx, hy = half_size_px / VIS_W, half_size_px / VIS_H
    point = n.mean(axis=0)
    bbox = np.array([n[:, 0].min() - hx, n[:, 1].min() - hy,
                     n[:, 0].max() + hx, n[:, 1].max() + hy], dtype=np.float32)
    return point.astype(np.float32), np.clip(bbox, 0.0, 1.0)
