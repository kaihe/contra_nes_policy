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


@torch.no_grad()
def tail_ce_metrics(ce: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    modal_action: int) -> Dict[str, torch.Tensor]:
    """Cross-entropy restricted to valid steps whose *target* is not the modal action.

    Total validation CE is a frequency-weighted average, and 78% of rollout steps are
    ``R``, so it is dominated by the frames where any policy does well. It is not merely
    uninformative here — it is anti-correlated with play: the D8 checkpoint at the CE
    minimum (0.703 at step 3,000) scores 52.6% pooled on the 846 suite against 65.5% for
    the fully overfit final at CE 1.754. Survival depends on the rare frames — the jump
    timing, the dodge, the step where holding ``R`` kills you — and this restricts the
    average to them. See ``doc/0010-exp-dropout-regularization.md``.

    ``tail_n`` rides along because callers average over batches: the non-modal step count
    varies far more between batches than the valid-step count does, so an unweighted mean
    of per-batch means is not the dataset's tail CE. Weight by this.
    """
    tail = mask * (target != modal_action).float()
    return {"tail_ce": _masked_mean(ce, tail), "tail_n": tail.sum()}


class BehaviorCloneLoss(nn.Module):
    """Cross-entropy of the 21-way action head against the recorded action."""

    def __init__(self, weight: float = 1.0, label_smoothing: float = 0.0,
                 class_weights: torch.Tensor | None = None,
                 modal_action: int | None = None, diagnostics: bool = True):
        super().__init__()
        self.weight = weight
        self.label_smoothing = label_smoothing
        # The dataset's most common action. 68.2% of training steps are `R`, so a model
        # that ignores its input entirely scores bc_acc 0.68 — and action-prior collapse
        # is a failure this project has already hit (the prev_action run reached 37% on
        # a single action while boss completion fell to 1.8%). Given this index, the
        # metrics below can see the collapse that accuracy cannot.
        self.modal_action = modal_action
        self.diagnostics = diagnostics
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
        # Reported always, including under `diagnostics=False`: it is one extra masked
        # mean over the `ce` already computed, and it is the offline proxy 0010 exists to
        # build. It is computed under no_grad and never enters the returned loss, so a run
        # that logs it optimizes bit-identically to one that does not.
        tail = (tail_ce_metrics(ce, target, mask, int(self.modal_action))
                if self.modal_action is not None else {})
        if not self.diagnostics:
            return self.weight * loss, {"loss": loss.detach(), **tail}
        with torch.no_grad():
            pred = logits.argmax(-1)
            metrics = {"bc_loss": loss.detach(),
                       "bc_acc": _masked_mean((pred == target).float(), mask), **tail}
            metrics.update(prior_collapse_metrics(
                pred, target, mask, logits.shape[-1], self.modal_action))
        return self.weight * loss, metrics


@torch.no_grad()
def prior_collapse_metrics(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor, n_classes: int,
                           modal_action: int | None = None) -> Dict[str, torch.Tensor]:
    """The three numbers that see through action-prior collapse.

    ``bc_acc`` cannot: 68.2% of training steps are one action, so predicting it
    unconditionally scores 0.68 against a 0.76 gate — eight points of headroom for a
    model that never looks at the screen.

    ``pred_modal_frac``  share of *predictions* taken by the single most-predicted
                         action. Drifting toward the data's 0.68 is collapse, whatever
                         accuracy is doing.
    ``bc_bal_acc``       mean per-class recall over the classes actually present. A
                         constant predictor scores 1/21 ≈ 0.048, not 0.68.
    ``bc_nonmodal_acc``  accuracy on the ~32% of steps that are *not* the modal action,
                         which is where every real decision lives.
    """
    keep = mask.bool()
    p, t = pred[keep], target[keep]
    out: Dict[str, torch.Tensor] = {}
    if p.numel() == 0:
        return out

    counts = torch.bincount(p, minlength=n_classes).float()
    out["pred_modal_frac"] = counts.max() / counts.sum()

    total = torch.bincount(t, minlength=n_classes).float()
    hit = torch.bincount(t[p == t], minlength=n_classes).float()
    present = total > 0
    out["bc_bal_acc"] = (hit[present] / total[present]).mean()

    if modal_action is not None:
        off = t != int(modal_action)
        if bool(off.any()):
            out["bc_nonmodal_acc"] = (p[off] == t[off]).float().mean()
    return out


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
                 families: tuple[str, ...] = (), modal_action: int | None = None):
        super().__init__()
        self.bc = BehaviorCloneLoss(weight=bc_weight, label_smoothing=label_smoothing,
                                    class_weights=class_weights,
                                    modal_action=modal_action)
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
