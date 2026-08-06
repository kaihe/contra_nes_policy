"""GRPO resumption — the property a ten-hour run depends on (doc/0011 §5).

Both prior long runs died mid-flight, and until now a checkpoint carried only weights: a
restart reset the optimizer moments, the sampling stream and the difficulty tracker's
per-task history to update 0. Over a long run that history *is* the curriculum, so a warm
restart silently discards most of what the run had learned about which tasks carry
gradient. These pin that a resume continues rather than restarts, and that an old
checkpoint says so instead of pretending.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from contra_policy.rl.tasks import DifficultyTracker
from contra_policy.rl.trainer import GRPOTrainer


def _stub(tmp_path, update=0, elapsed=0.0, tracker=None):
    """A trainer with only what save/_resume touch, per tests/test_probe.py's pattern."""
    stub = types.SimpleNamespace()
    stub.run_dir = str(tmp_path)
    (tmp_path / "checkpoints").mkdir(exist_ok=True)
    stub.args = OmegaConf.create({})
    stub.update, stub.elapsed = update, elapsed
    stub.rng = np.random.default_rng(0)
    stub.groups = types.SimpleNamespace(difficulty=tracker)
    stub.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))], lr=1e-5)
    # Stand in for ContraPolicy.save, which writes weights alongside these extras.
    stub.policy = types.SimpleNamespace(
        save=lambda path, **extra: torch.save(dict(extra), path))
    stub.save = GRPOTrainer.save.__get__(stub)
    stub._resume = GRPOTrainer._resume.__get__(stub)
    return stub


def test_weights_only_checkpoint_is_rejected_rather_than_silently_restarting(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"policy": {}, "update": 40}, path)          # no optimizer: the old format
    stub = _stub(tmp_path)

    with pytest.raises(ValueError) as e:
        stub._resume(str(path))

    assert "optimizer" in str(e.value)
    assert "init_from" in str(e.value)                      # says what it *can* be used as
    assert stub.update == 0                                 # and changed nothing


def test_resume_restores_update_elapsed_and_curriculum(tmp_path):
    tracker = DifficultyTracker(group_size=8)
    tracker.observe("boss-7", "boss_level1", successes=3.0, attempts=8.0)
    before = tracker.p_hat("boss-7", "boss_level1")

    saved = _stub(tmp_path, update=137, elapsed=4.5 * 3600, tracker=tracker)
    path = saved.save()

    fresh_tracker = DifficultyTracker(group_size=8)
    fresh = _stub(tmp_path, tracker=fresh_tracker)
    assert fresh.update == 0 and fresh.elapsed == 0.0
    fresh._resume(path)

    assert fresh.update == 137
    assert fresh.elapsed == pytest.approx(4.5 * 3600)
    # The curriculum survived: a fresh tracker would read the Laplace prior, not 3/8.
    assert fresh_tracker.p_hat("boss-7", "boss_level1") == pytest.approx(before)


def test_elapsed_is_cumulative_so_max_hours_budgets_the_experiment(tmp_path):
    """A resumed run must not get its wall-clock budget back."""
    first = _stub(tmp_path, update=50, elapsed=6.0 * 3600,
                  tracker=DifficultyTracker(group_size=8))
    path = first.save()

    second = _stub(tmp_path, tracker=DifficultyTracker(group_size=8))
    second._resume(path)
    second.elapsed += 1800.0                                # half an hour more

    assert second.elapsed == pytest.approx(6.5 * 3600)


def test_save_records_the_sampling_stream(tmp_path):
    stub = _stub(tmp_path, update=3, tracker=DifficultyTracker(group_size=8))
    stub.rng.random(11)                                     # advance it off its seed
    expected = stub.rng.bit_generator.state

    ckpt = torch.load(stub.save(), map_location="cpu", weights_only=False)

    assert ckpt["sampler_rng"] == expected
    assert {"optimizer", "update", "difficulty", "elapsed_seconds"} <= set(ckpt)


def test_resume_of_a_run_without_difficulty_bias_is_not_an_error(tmp_path):
    saved = _stub(tmp_path, update=9, tracker=None)         # difficulty_bias.enabled=false
    path = saved.save()

    fresh = _stub(tmp_path, tracker=None)
    fresh._resume(path)

    assert fresh.update == 9
