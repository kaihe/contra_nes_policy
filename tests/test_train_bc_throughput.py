"""Throughput accounting for whole-episode GPT behaviour cloning."""

import torch

from contra_policy.model import PREFIX
from contra_policy.train_bc import (_model_tokens, _timed_train_iteration,
                                    _useful_model_tokens)


def test_compiled_core_does_not_change_checkpoint_keys(monkeypatch):
    from contra_policy.model import PolicyConfig, build_policy

    policy = build_policy(PolicyConfig(
        encoder={"hiddim": 32}, freeze_encoder=True, value_head=False, aux_size=0,
        core={"d_model": 32, "n_layer": 1, "n_head": 4, "n_kv_head": 4,
              "context": 16, "mlp_ratio": 2.0, "rope_theta": 10000.0,
              "dropout": 0.0, "norm_eps": 1e-5}))
    keys = set(policy.state_dict())
    called = []

    def fake_compile(module, dynamic):
        called.append((module, dynamic))
        return module

    monkeypatch.setattr(torch, "compile", fake_compile)
    policy.compile_core(dynamic=True)

    assert called[0] == (policy.core, True)
    assert called[1][0].__self__ is policy.core
    assert called[1][0].__func__ is policy.core.forward_varlen.__func__
    assert called[1][1] is True
    assert set(policy.state_dict()) == keys
    assert not any("_orig_mod" in key for key in policy.state_dict())
    eager = build_policy(policy.cfg)
    eager.load_state_dict(policy.state_dict(), strict=True)


def test_model_tokens_include_padding_and_the_two_token_prefix():
    batch = {"image": torch.empty(3, 7, 8, 8, 3)}

    assert PREFIX == 2
    assert _model_tokens(batch) == 3 * (7 + 2)


def test_useful_tokens_exclude_padding_but_keep_each_prefix():
    batch = {"mask": torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.float32)}

    assert _useful_model_tokens(batch) == 4 + 2 * PREFIX


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
