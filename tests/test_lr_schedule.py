"""The LR schedule, and the property this repo picked it for.

WSD is not here because it converges better — it and cosine land in about the same place.
It is here because **cosine cannot be extended**: its LR is defined against `train.steps`
and reaches 0 exactly at the end, so a finished run is a dead end and every budget costs a
run from scratch. These tests pin the branchability that replaces that.
"""

from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

from contra_policy.train_bc import BCTrainer, wsd_onset


def _scale(steps, step, decay="wsd", warm=500, frac=0.1, floor=0.0):
    args = OmegaConf.create({"train": {"steps": steps, "warmup_steps": warm,
                                       "lr_decay": decay, "decay_frac": frac,
                                       "final_lr_frac": floor}})
    return BCTrainer._lr_scale(type("T", (), {"args": args})(), step)


def test_onset_is_the_last_decay_frac_of_the_budget():
    assert wsd_onset(20000, 0.1) == 18000
    assert wsd_onset(40000, 0.1) == 36000
    assert wsd_onset(80000, 0.25) == 60000


def test_warmup_then_flat_then_linear_to_zero():
    assert _scale(20000, 0) == pytest.approx(1 / 500)
    assert _scale(20000, 499) == pytest.approx(1.0)
    for step in (500, 5000, 12000, 17999):
        assert _scale(20000, step) == 1.0, "the stable phase must be exactly flat"
    assert _scale(20000, 18000) == pytest.approx(1.0)
    assert _scale(20000, 19000) == pytest.approx(0.5)
    assert _scale(20000, 20000) == pytest.approx(0.0)


def test_the_stable_phase_does_not_depend_on_the_total_budget():
    """The property that makes a checkpoint branchable: same step, any budget, same LR."""
    for step in (600, 5000, 15000):
        assert _scale(20000, step) == _scale(40000, step) == _scale(80000, step) == 1.0


def test_cosine_by_contrast_depends_on_the_total_everywhere():
    """Why a 40k cosine checkpoint cannot become an 80k run."""
    assert _scale(40000, 40000, decay="cosine") == pytest.approx(0.0, abs=1e-12)
    mid = _scale(80000, 40000, decay="cosine")
    assert mid == pytest.approx(0.5, abs=0.01)      # LR would jump 0 -> ~half of base


def test_final_lr_frac_sets_the_cooldown_floor():
    assert _scale(20000, 20000, floor=0.1) == pytest.approx(0.1)
    assert _scale(20000, 19000, floor=0.1) == pytest.approx(0.55)


def test_decay_frac_moves_the_onset_not_the_shape():
    assert _scale(20000, 10000, frac=0.5) == pytest.approx(1.0)
    assert _scale(20000, 15000, frac=0.5) == pytest.approx(0.5)


def test_repo_default_is_wsd():
    """`from now on, this repo only uses WSD` — pinned so a config edit cannot undo it."""
    from hydra import compose, initialize_config_module
    with initialize_config_module("contra_policy", version_base=None):
        for name in ("config_bc", "config_bc_scaling"):
            cfg = compose(config_name=name)
            assert cfg.train.lr_decay == "wsd", f"{name} must default to wsd"
            assert 0.0 < cfg.train.decay_frac < 1.0


def test_repo_default_keeps_only_the_wsd_trunk_and_final_checkpoint():
    from hydra import compose, initialize_config_module
    with initialize_config_module("contra_policy", version_base=None):
        for name in ("config_bc", "config_bc_scaling", "config_bc_scaling_lr",
                     "config_bc_scaling_40k", "config_bc_laser"):
            cfg = compose(config_name=name)
            assert list(cfg.train.save_steps) == []
            assert cfg.train.save_best is False
            assert cfg.train.save_wsd_trunk is True


def test_validation_tracks_best_without_saving_when_disabled():
    class Fake:
        args = OmegaConf.create({"train": {"save_best": False}})
        best = math.inf
        step = 10
        saved = False

        def validate(self, _batches):
            return {"loss": 0.75}

        def _accounting_metrics(self):
            return {}

        def _emit(self, _row, _tag):
            return None

        def save(self, **_kwargs):
            self.saved = True

    trainer = Fake()
    BCTrainer._run_val(trainer, 0)
    assert trainer.best == pytest.approx(0.75)
    assert trainer.saved is False


# ── extending a run: the point of the switch ─────────────────────────────────

class _FakeTrainer:
    """Just enough of BCTrainer to exercise `_resume`'s guards."""

    def __init__(self, cfg):
        self.args = OmegaConf.create(cfg)
        self.step, self.best = 0, math.inf
        self.policy = self.optimizer = self.scheduler = self.scaler = _Loadable()
        from collections import Counter
        self.draw_counts, self.token_counts = Counter(), Counter()


class _Loadable:
    def load_state_dict(self, *a, **k):
        return None


def _ckpt(tmp_path, steps, step, decay="wsd", frac=0.1):
    import torch
    cfg = {"seed": 0, "loader": {"family_draws": None},
           "boss_scaling": {"shard_count": 13},
           "train": {"steps": steps, "warmup_steps": 500, "lr_decay": decay,
                     "decay_frac": frac, "final_lr_frac": 0.0}}
    p = tmp_path / f"ck-{steps}-{step}-{decay}.pt"
    torch.save({"policy": {}, "optimizer": {}, "scheduler": {}, "scaler": {},
                "step": step, "train_config": cfg}, p)
    return str(p), cfg


def _resume(tmp_path, ckpt_steps, ckpt_step, new_steps, decay="wsd"):
    path, cfg = _ckpt(tmp_path, ckpt_steps, ckpt_step, decay)
    new = {**cfg, "train": {**cfg["train"], "steps": new_steps}}
    BCTrainer._resume(_FakeTrainer(new), path)


def test_a_wsd_trunk_extends_to_a_longer_budget(tmp_path):
    """20k trunk, stopped in the stable phase, continued as an 80k run."""
    _resume(tmp_path, 20000, 15000, 80000)


def test_a_cosine_run_still_refuses_to_be_extended(tmp_path):
    with pytest.raises(ValueError, match="cannot be extended"):
        _resume(tmp_path, 40000, 20000, 80000, decay="cosine")


def test_extending_from_inside_the_cooldown_is_refused(tmp_path):
    """Past the onset the weights are already annealed — not a trunk any more."""
    with pytest.raises(ValueError, match="began cooling down"):
        _resume(tmp_path, 20000, 19000, 80000)


def test_extending_backwards_is_refused(tmp_path):
    with pytest.raises(ValueError, match="before the checkpoint"):
        _resume(tmp_path, 40000, 30000, 20000)


def test_same_budget_resume_still_works_under_wsd(tmp_path):
    _resume(tmp_path, 20000, 19500, 20000)      # plain crash-resume, inside the cooldown
