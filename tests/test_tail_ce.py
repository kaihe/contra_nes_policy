"""Tail CE — the offline proxy from doc/0010.

Total validation CE is dominated by the modal action; these pin that `tail_ce` really is
the cross-entropy of the rest, that it aggregates over steps rather than over batches,
and that logging it does not perturb the objective.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from contra_policy.action_space import ACTION_NAMES
from contra_policy.loss import BehaviorCloneLoss, tail_ce_metrics
from contra_policy.train_bc import MODAL_ACTION, _weighted_tail


MODAL = ACTION_NAMES.index("R")


def test_modal_action_is_read_from_the_frozen_action_space():
    assert MODAL_ACTION == MODAL
    assert ACTION_NAMES[MODAL_ACTION] == "R"


def _ce(logits, target):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           target.reshape(-1), reduction="none").reshape(target.shape)


def test_tail_ce_is_the_cross_entropy_of_the_non_modal_steps_only():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, len(ACTION_NAMES))
    target = torch.tensor([[MODAL, 3, MODAL, 7, MODAL],
                           [12, MODAL, MODAL, MODAL, 4]])
    mask = torch.ones_like(target, dtype=torch.float32)

    ce = _ce(logits, target)
    out = tail_ce_metrics(ce, target, mask, MODAL)

    off = target != MODAL
    assert out["tail_n"].item() == float(off.sum())          # 4 of 10 steps
    assert math.isclose(out["tail_ce"].item(), ce[off].mean().item(), rel_tol=1e-6)
    # It must differ from the pooled figure, or it is not measuring anything new.
    assert not math.isclose(out["tail_ce"].item(), ce.mean().item(), rel_tol=1e-3)


def test_tail_ce_ignores_padded_steps():
    torch.manual_seed(1)
    logits = torch.randn(1, 4, len(ACTION_NAMES))
    target = torch.tensor([[3, 7, 9, 11]])                   # every step non-modal
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])              # last two are padding

    ce = _ce(logits, target)
    out = tail_ce_metrics(ce, target, mask, MODAL)

    assert out["tail_n"].item() == 2.0
    assert math.isclose(out["tail_ce"].item(), ce[0, :2].mean().item(), rel_tol=1e-6)


def test_all_modal_batch_is_safe_and_contributes_no_weight():
    logits = torch.randn(1, 3, len(ACTION_NAMES))
    target = torch.full((1, 3), MODAL)
    mask = torch.ones_like(target, dtype=torch.float32)

    out = tail_ce_metrics(_ce(logits, target), target, mask, MODAL)

    assert out["tail_n"].item() == 0.0
    assert torch.isfinite(out["tail_ce"])                    # no divide-by-zero


def test_weighted_tail_aggregates_over_steps_not_batches():
    # A batch with 1 non-modal step must not count as much as one with 99.
    rows = [{"tail_ce": 4.0, "tail_n": 1.0}, {"tail_ce": 2.0, "tail_n": 99.0}]
    out = _weighted_tail({"tail_ce": 3.0, "tail_n": 50.0}, rows)

    assert math.isclose(out["tail_ce"], (4.0 * 1 + 2.0 * 99) / 100)
    assert out["tail_n"] == 100.0                            # total, not a mean of counts
    assert not math.isclose(out["tail_ce"], 3.0)             # the unweighted mean is wrong


def test_weighted_tail_passes_through_when_nothing_is_non_modal():
    out = _weighted_tail({"loss": 1.5}, [{"tail_ce": 0.0, "tail_n": 0.0}])
    assert out == {"loss": 1.5}


def test_logging_tail_ce_does_not_change_the_objective():
    """0010's cells must be comparable against the existing dropout-0.0 control."""
    torch.manual_seed(2)
    logits = torch.randn(2, 6, len(ACTION_NAMES))
    batch = {"action": torch.randint(0, len(ACTION_NAMES), (2, 6)),
             "mask": torch.ones(2, 6)}
    latents = {"pi_logits": logits}

    plain, plain_metrics = BehaviorCloneLoss(diagnostics=False)(latents, batch)
    tailed, tail_metrics = BehaviorCloneLoss(
        diagnostics=False, modal_action=MODAL)(latents, batch)

    assert torch.equal(plain, tailed)
    assert "tail_ce" not in plain_metrics                    # opt-in, not always-on
    assert {"loss", "tail_ce", "tail_n"} == set(tail_metrics)
