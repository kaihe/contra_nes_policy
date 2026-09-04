"""Frames and grounding targets, flattened out of the policy's windowed loader.

Pretraining the encoder is a per-frame problem: no temporal model, no recurrent state,
every frame independent. But the *data* is stored per episode in tar-sliced all-intra
video, and ``contra_policy.dataset`` already solves the hard parts of reading it —
byte-offset seeks into the tar, windowed decoding that does not pay for a whole episode
to serve one window, and the exact construction of ``goal_heatmap`` / ``point`` /
``exist`` from ``centroids`` and ``visibility``.

So this wraps that loader rather than re-reading the shards. The alternative — a second
reader with a second copy of the target arithmetic — is precisely the drift that
``goal.goal_mask`` being the single renderer for both prompt and target is meant to
prevent.

Windows still come out ``(T, ...)``; :func:`flatten_window` collapses the time axis into
the batch and drops the padded tail using the window's own ``mask``. ``win_len`` here is
a *decode efficiency* knob, not a model parameter: larger windows mean fewer seeks per
frame, and the frames are shuffled into batches afterwards regardless.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch

from contra_policy.dataset import CONFIGS, FAMILIES, ContraDataModule

__all__ = ["FAMILIES", "build_datamodule", "flatten_window"]


def build_datamodule(shard_dir: str, families: Sequence[str] = CONFIGS,
                     image_size: int = 256, aux_size: int = 32, sigma_px: float = 12.0,
                     win_len: int = 32, batch_size: int = 4, num_workers: int = 2,
                     prefetch_factor: int = 2, cache_dir: str = "cache",
                     seed: int = 0, want_entities: bool = False,
                     entity_sigma_px: float = 6.0) -> ContraDataModule:
    """The policy's datamodule, configured for frame pretraining.

    ``prev_action_keep_prob=0.0`` and ``carry_memory=False``: there is no previous-action
    input in this design and no recurrent state to carry, so neither knob can affect the
    result. Pinned explicitly rather than left at their defaults so a future change to
    those defaults cannot quietly alter pretraining.

    ``family_balance_alpha=0.0``: pretraining wants the real frame mix. The families are
    already wildly unbalanced by *steps* (traverse is 45% of decisions), and grounding
    is a per-frame perceptual task where more frames genuinely is more signal.
    """
    return ContraDataModule(
        shard_dir=shard_dir, configs=list(families), win_len=win_len,
        image_size=image_size, sigma_px=sigma_px, prev_action_keep_prob=0.0,
        aux_size=aux_size, carry_memory=False, family_balance_alpha=0.0,
        batch_size=batch_size, num_workers=num_workers,
        prefetch_factor=prefetch_factor, cache_dir=cache_dir, seed=seed,
        want_entities=want_entities, entity_sigma_px=entity_sigma_px)


def flatten_window(batch: Dict, device: Optional[torch.device] = None,
                   ) -> Dict[str, torch.Tensor]:
    """``(B, T, ...)`` window batch → ``(N, ...)`` frame batch, padding removed.

    Since 0002 the encoder is goal-agnostic, so the goal image is *not* paired with
    each frame. It is returned separately as ``goal_image`` — one per window, not one
    per frame — for the reconstruction term, which is the only supervision a goal frame
    can carry (see the module docstring).

    Returns ``{}`` when a batch is entirely padding, which the caller must skip; that
    only happens for degenerate single-step episodes.
    """
    mask = batch["mask"]                                   # (B, T) float 1 where real
    b, t = mask.shape
    keep = mask.reshape(-1).bool()
    if not bool(keep.any()):
        return {}

    cv = batch["cross_view"]
    out = {
        "image": batch["image"].reshape(b * t, *batch["image"].shape[2:])[keep],
        # One per window, not per frame: goal frames are encoded by the same function
        # but carry no entity labels, so they never enter the per-frame batch.
        "goal_image": cv["cross_view_image"],
        # Per-frame family index, for the per-family metric split.
        "family": batch["family"].unsqueeze(1).expand(b, t).reshape(-1)[keep],
    }
    # Only present when the datamodule was built with want_entities.
    if "entity_heatmap" in batch:
        eh = batch["entity_heatmap"]
        out["entity_heatmap"] = eh.reshape(b * t, *eh.shape[2:])[keep]
    # The goal frame's target, one per window — same shape and semantics as a frame's,
    # because a goal frame *is* a frame.
    if "goal_entity_heatmap" in batch:
        out["goal_entity_heatmap"] = batch["goal_entity_heatmap"]
    if device is not None:
        out = {k: v.to(device, non_blocking=True) for k, v in out.items()}
    return out


def _repeat_t(x: torch.Tensor, t: int) -> torch.Tensor:
    """``(B, ...)`` → ``(B*T, ...)`` with each element repeated ``t`` times, contiguously.

    ``unsqueeze(1).expand(...)`` then ``reshape`` keeps the (b, t) ordering identical to
    every other member's ``reshape(b * t, ...)``, which is what lets one flat mask index
    all of them.
    """
    b = x.shape[0]
    return x.unsqueeze(1).expand(b, t, *x.shape[1:]).reshape(b * t, *x.shape[1:])
