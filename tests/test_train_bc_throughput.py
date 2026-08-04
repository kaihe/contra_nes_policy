"""Throughput accounting for whole-episode GPT behaviour cloning."""

import torch

from contra_policy.model import PREFIX
from contra_policy.train_bc import _model_tokens, _timed_train_iteration


def test_model_tokens_include_padding_and_the_two_token_prefix():
    batch = {"image": torch.empty(3, 7, 8, 8, 3)}

    assert PREFIX == 2
    assert _model_tokens(batch) == 3 * (7 + 2)


def test_step_timer_includes_batch_acquisition():
    events = []

    class Batches:
        def __iter__(self):
            return self

        def __next__(self):
            events.append("load")
            return {"image": torch.empty(1, 3, 1, 1, 3)}

    class Trainer:
        device = torch.device("cpu")

        def train_step(self, batch):
            events.append("train")
            return {"loss": 1.0}, _model_tokens(batch)

    times = iter((10.0, 10.25))

    row, tokens, elapsed, _ = _timed_train_iteration(
        Trainer(), Batches(), Batches(), clock=lambda: events.append("clock") or next(times))

    assert events == ["clock", "load", "train", "clock"]
    assert row == {"loss": 1.0}
    assert tokens == 5
    assert elapsed == 0.25
