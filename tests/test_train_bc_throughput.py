"""Throughput accounting for whole-episode GPT behaviour cloning."""

import torch

from contra_policy.model import PREFIX
from contra_policy.train_bc import _model_tokens


def test_model_tokens_include_padding_and_the_two_token_prefix():
    batch = {"image": torch.empty(3, 7, 8, 8, 3)}

    assert PREFIX == 2
    assert _model_tokens(batch) == 3 * (7 + 2)
