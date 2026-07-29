"""Training objectives — the replacements for minestudio's ``MineCallbacks``.

ROCKET-2's ``train.py`` composes two of them::

    BehaviorCloneCallback(weight=args.objective_weight)
    PointPredictionCallback(point_weight=0.1, bbox_weight=0.1, exist_weight=0.01)

Neither ships in the open-source drop (``loss.py`` is imported but absent), and
``minestudio.offline`` is not installed, so both are written here against the same
weights. Everything is masked by the window's validity mask, and the two geometric
terms are additionally gated on ``exist``: regressing a point for a frame where the
goal entity is off-screen would train the head to hallucinate a location. That
gating matters here — 59% of ``traverse`` frames have no visible goal.

All losses reduce as ``sum / count`` over *valid* elements rather than ``.mean()``
over the padded tensor, so a batch of short episodes is not silently down-weighted
by its padding.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``x`` over entries where ``mask`` is 1, safe when nothing is valid."""
    total = mask.sum()
    return (x * mask).sum() / total.clamp(min=1.0)


def action_class_weights(counts, alpha: float = 0.0, max_ratio: float = 10.0,
                         min_share: float = 1e-3) -> torch.Tensor:
    """Per-class BC weights from the training action histogram.

    The policy under-aims: ``UR`` is 10.0% of training steps — the second most common
    action — yet only 2.2% of rollout actions, and ``URF`` 2.7% → 0.5%. That is not
    data scarcity, it is imbalance against ``R`` at 68.2%, so it belongs to the loss.

    ``w_c ∝ (median_f / f_c) ** alpha``, normalised so ``sum_c f_c * w_c == 1`` — which
    keeps the loss on the same scale as the unweighted version, so ``bc_weight`` means
    the same thing at any alpha — and finally capped at ``max_ratio``.

    Two guards, both load-bearing:

    * ``min_share`` floors the frequency used in the ratio. Nine of the 21 actions are
      unusable (6 never occur; ``LF``/``U``/``UF`` have 30/22/12 examples in 692k
      steps), and without the floor those three take the *largest* weights in the
      table purely for being rare, which is fitting noise.
    * The cap is applied **after** normalising, not before. Applied first it does
      nothing: normalisation divides by the data-weighted mean, which is well below 1
      because the dominant class is down-weighted, and that scales every weight back up
      past the cap — a max_ratio of 10 was landing weights of 50.

    ``alpha=0`` is the identity. Classes with zero count keep weight 1 and are inert:
    they are never a target.
    """
    f = torch.as_tensor(counts, dtype=torch.double)
    f = f / f.sum().clamp(min=1)
    if alpha == 0.0:
        return torch.ones_like(f, dtype=torch.float32)
    seen = f > 0
    med = f[seen].median()
    w = torch.ones_like(f)
    w[seen] = (med / f[seen].clamp(min=min_share)) ** alpha
    w = w / (f * w).sum().clamp(min=1e-12)     # E_data[w] == 1
    return w.clamp(max=max_ratio).to(torch.float32)


class BehaviorCloneLoss(nn.Module):
    """Cross-entropy of the 21-way action head against the recorded action."""

    def __init__(self, weight: float = 1.0, label_smoothing: float = 0.0,
                 class_weights: torch.Tensor | None = None):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        # Buffer, not a plain attribute, so it follows the module across devices and
        # into the checkpoint (making the weighting a recorded property of the run).
        self.register_buffer("class_weights", class_weights, persistent=True)

    def forward(self, latents: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits, target, mask = latents["pi_logits"], batch["action"], batch["mask"]
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
            reduction="none", label_smoothing=self.label_smoothing,
            weight=self.class_weights,
        ).reshape(target.shape)
        # F.cross_entropy's own 'mean' would divide by the summed class weights; the
        # masked mean here divides by valid-step count, so the weighting shifts the
        # relative pull between classes without rescaling the loss.
        loss = _masked_mean(ce, mask)
        with torch.no_grad():
            correct = (logits.argmax(-1) == target).float()
            metrics = {"bc_loss": loss.detach(),
                       "bc_acc": _masked_mean(correct, mask)}
        return self.weight * loss, metrics


def point_err_px(pred_point: torch.Tensor, target_point: torch.Tensor) -> torch.Tensor:
    """Per-element point error in 240x224 screen pixels.

    Lifted verbatim out of the old aux loss so the formula has exactly one definition.
    `contra_nes_evaluation` reports the same number on-policy and pins it; do not
    change the arithmetic here.
    """
    err = (pred_point - target_point).abs()
    return (err[..., 0] * 240.0 + err[..., 1] * 224.0) * 0.5


class GoalHeatmapLoss(nn.Module):
    """Dense goal-occupancy grounding, replacing scalar exist + point + bbox.

    The scalar version could not work on two of the four families. `kill` and `boss`
    episodes have the goal visible on **100.0%** of frames, so the `exist` head saw no
    negative example ever, and a constant "visible" predictor scored 100% — the metric
    was degenerate and the signal absent. A heatmap inverts that: on every frame where
    the goal *is* present, every cell outside the blob is a negative, and a frame
    without a goal is an all-zero map rather than one masked-off scalar.

    The target comes from ``goal.goal_mask`` — the same renderer that draws the
    cross-view prompt channel — so prompt and target cannot drift apart.

    ``pos_weight`` counters the area imbalance: at A=32 with sigma_px=12 the blob is
    roughly 1.7% of the map, so unweighted BCE is minimised well by predicting
    "empty everywhere". The weight is applied per cell as ``1 + pos_weight * target``,
    which is graded by the soft Gaussian rather than thresholded.
    """

    def __init__(self, heatmap_weight: float = 1.0, pos_weight: float = 10.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.pos_weight = pos_weight

    def forward(self, latents: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask = batch["mask"]
        target = batch["goal_heatmap"]
        pred = latents["goal_heatmap"]

        bce = F.binary_cross_entropy_with_logits(pred.float(), target, reduction="none")
        cell_w = 1.0 + self.pos_weight * target
        per_step = (bce * cell_w).flatten(2).mean(-1) / (1.0 + self.pos_weight * 0.5)
        loss = _masked_mean(per_step, mask)

        with torch.no_grad():
            exist_t = batch["exist"]
            visible = mask * exist_t              # geometry is only defined here
            exist_acc = ((latents["exist"].squeeze(-1) > 0).float() == exist_t).float()
            metrics = {
                "heatmap_loss": loss.detach(),
                "exist_acc": _masked_mean(exist_acc, mask),
                "point_err_px": _masked_mean(
                    point_err_px(latents["point"], batch["point"]), visible),
            }
        return self.heatmap_weight * loss, metrics


class ContraObjective(nn.Module):
    """The full training objective: BC + the cross-view grounding aux head."""

    def __init__(self, bc_weight: float = 1.0, heatmap_weight: float = 1.0,
                 heatmap_pos_weight: float = 10.0, label_smoothing: float = 0.0,
                 class_weights: torch.Tensor | None = None,
                 families: tuple[str, ...] = ()):
        super().__init__()
        self.bc = BehaviorCloneLoss(weight=bc_weight, label_smoothing=label_smoothing,
                                    class_weights=class_weights)
        self.aux = GoalHeatmapLoss(heatmap_weight, heatmap_pos_weight)
        self.families = families

    def forward(self, latents: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
                ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bc_loss, metrics = self.bc(latents, batch)
        aux_loss, aux_metrics = self.aux(latents, batch)
        metrics.update(aux_metrics)
        total = bc_loss + aux_loss
        metrics["loss"] = total.detach()
        if self.families and "family" in batch:
            metrics.update(self.family_metrics(latents, batch))
        return total, metrics

    @torch.no_grad()
    def family_metrics(self, latents: Dict[str, torch.Tensor],
                       batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Break the headline metrics out per task family.

        Pooled numbers are dominated by ``traverse`` at 65% of training steps — the
        family the policy already handles best — so a pooled move says little about
        where it came from. ``goal_vis`` is reported alongside ``exist_acc`` on
        purpose: it is the base rate, and it is 100% on ``kill`` and ``boss``, where
        an 100% ``exist_acc`` therefore means nothing at all.
        """
        out: Dict[str, torch.Tensor] = {}
        fam = batch["family"]
        mask, exist_t = batch["mask"], batch["exist"]
        correct = (latents["pi_logits"].argmax(-1) == batch["action"]).float()
        px = point_err_px(latents["point"], batch["point"])
        for i, name in enumerate(self.families):
            sel = (fam == i).float().unsqueeze(-1)
            m = mask * sel
            if m.sum() == 0:
                continue
            out[f"{name}/bc_acc"] = _masked_mean(correct, m)
            out[f"{name}/goal_vis"] = _masked_mean(exist_t, m)
            out[f"{name}/point_err_px"] = _masked_mean(px, m * exist_t)
        return out
