"""Causal core tests: the properties whose violation is silent.

A transformer with a broken mask still trains and still produces plausible loss curves —
it just cheats. So the two things pinned hardest here are that information never flows
backwards in time, and never flows between packed episodes. Both would show up as a
suspiciously good BC number and nothing else.

The rest guards the reasons this replaced VPT's `ResidualRecurrentBlocks`: context
length is a config value (no learned positional parameter), and packing is free.
"""

from __future__ import annotations

import pytest
import torch

from contra_policy.causal import (Attention, CausalGPT, CausalGPTConfig, RMSNorm,
                                  apply_rope, build_rope_cache, causal_block_mask)

SMALL = dict(d_model=32, n_layer=2, n_head=4, n_kv_head=4, context=64)


def _gpt(**over):
    return CausalGPT(CausalGPTConfig(**{**SMALL, **over})).eval()


# ── causality ────────────────────────────────────────────────────────────────

def test_the_future_cannot_reach_the_past():
    """The failure that trains fine and invalidates every number afterwards."""
    m = _gpt()
    x = torch.randn(2, 40, 32)
    with torch.no_grad():
        y = m(x)
        x2 = x.clone()
        x2[:, 20:] = torch.randn_like(x2[:, 20:])       # rewrite the entire future
        y2 = m(x2)
    assert torch.allclose(y[:, :20], y2[:, :20], atol=1e-5), \
        "editing later tokens changed earlier outputs — the causal mask is not holding"
    # And the future *must* change, or the model is ignoring its input.
    assert not torch.allclose(y[:, 20:], y2[:, 20:], atol=1e-3)


def test_a_token_sees_itself():
    """Causal, not strictly causal: position t attends to t. `pi` reads frame t's own
    output to predict the action taken *from* that frame."""
    m = _gpt()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        y = m(x)
        x2 = x.clone()
        x2[:, 5] += 5.0
        y2 = m(x2)
    assert not torch.allclose(y[:, 5], y2[:, 5], atol=1e-4)


# ── packing ──────────────────────────────────────────────────────────────────

def test_packed_episodes_do_not_attend_to_each_other():
    """Two episodes in one row must give exactly what they give alone.

    This is what makes packing safe. Episodes average 100 frames against a 1024 context,
    so without it a batch is ~90% padding.
    """
    m = _gpt()
    a, b = torch.randn(1, 25, 32), torch.randn(1, 15, 32)
    with torch.no_grad():
        packed = m(torch.cat([a, b], 1), seq_lens=torch.tensor([[25, 15]]))
        alone_a, alone_b = m(a), m(b)
    assert torch.allclose(packed[:, :25], alone_a, atol=1e-5)
    assert torch.allclose(packed[:, 25:], alone_b, atol=1e-5)


def test_packing_needs_no_position_reset():
    """RoPE scores depend on relative distance, so the second episode does not need its
    positions restarted at zero — which is a step most packing implementations carry."""
    m = _gpt()
    a, b = torch.randn(1, 20, 32), torch.randn(1, 20, 32)
    with torch.no_grad():
        packed = m(torch.cat([a, b], 1), seq_lens=torch.tensor([[20, 20]]))
        assert torch.allclose(packed[:, 20:], m(b), atol=1e-5)


def test_block_mask_shape_and_blocking():
    total = 10
    mask = causal_block_mask(torch.tensor([[4, 3]]), total)
    assert mask.shape == (1, 1, total, total)
    m = mask[0, 0]
    assert bool(m[3, 0]) and bool(m[3, 3])          # within sequence 1, causal
    assert not bool(m[0, 3])                         # no peeking forward
    assert not bool(m[5, 2]) and not bool(m[2, 5])   # across the boundary, neither way
    assert bool(m[6, 4])                             # within sequence 2
    assert not bool(m[9, 9])                         # positions 7-9 are padding


def test_padding_positions_attend_to_nothing():
    """A row shorter than its buffer must not let real tokens see the pad tail."""
    mask = causal_block_mask(torch.tensor([[3]]), 6)[0, 0]
    assert not mask[:, 3:].any(), "real tokens can attend to padding"
    assert not mask[3:, :].any(), "padding attends to real tokens"


# ── context length is a config value ─────────────────────────────────────────

def test_no_learned_positional_parameter():
    """The reason context length is a config change here and surgery in VPT.

    `bandify`'s basis is a learned (10, mem_len x tokens) tensor whose *shape* depends
    on the horizon. RoPE is computed from positions, so nothing trained has to change.
    """
    m = _gpt()
    named = dict(m.named_parameters())
    assert not [n for n in named if "pos" in n or "rope" in n], \
        f"found a learned positional parameter: {[n for n in named if 'pos' in n or 'rope' in n]}"
    # The tables are buffers, and non-persistent, so they never enter a checkpoint.
    assert "rope_cos" not in m.state_dict() and "rope_sin" not in m.state_dict()


def test_varying_context_does_not_change_parameter_count():
    a = CausalGPT(CausalGPTConfig(**{**SMALL, "context": 64}))
    b = CausalGPT(CausalGPTConfig(**{**SMALL, "context": 4096}))
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_sequence_longer_than_context_is_refused():
    """Better a clear error than silently indexing past the RoPE table."""
    m = _gpt(context=16)
    with pytest.raises(ValueError, match="exceeds context"):
        m(torch.randn(1, 17, 32))


def test_shorter_than_context_is_fine():
    m = _gpt(context=64)
    assert m(torch.randn(1, 3, 32)).shape == (1, 3, 32)


# ── components ───────────────────────────────────────────────────────────────

def test_rmsnorm_computes_in_float32_under_bf16():
    """rsqrt of a mean of squares is where low precision bites, and this runs under
    bf16 autocast."""
    n = RMSNorm(64)
    x = torch.randn(4, 64, dtype=torch.bfloat16)
    out = n(x)
    assert out.dtype == torch.bfloat16                    # dtype preserved for the caller
    ref = n(x.float())
    assert (out.float() - ref).abs().max() < 0.05


def test_rope_is_a_rotation_so_it_preserves_norm():
    cos, sin = build_rope_cache(16, 8, 10000.0)
    x = torch.randn(2, 3, 16, 8)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), y.norm(dim=-1), atol=1e-4)


def test_grouped_query_attention_shrinks_kv_not_output():
    cfg = CausalGPTConfig(**{**SMALL, "n_head": 4, "n_kv_head": 2})
    a = Attention(cfg)
    assert a.k.out_features == 2 * a.head_dim          # 2 kv heads, not 4
    assert a.q.out_features == 4 * a.head_dim
    x = torch.randn(1, 6, cfg.d_model)
    cos, sin = build_rope_cache(6, a.head_dim, cfg.rope_theta)
    assert a(x, cos, sin, None).shape == x.shape


def test_config_rejects_indivisible_head_counts():
    with pytest.raises(ValueError, match="must divide"):
        CausalGPTConfig(d_model=30, n_head=4)
    with pytest.raises(ValueError, match="multiple of"):
        CausalGPTConfig(d_model=32, n_head=4, n_kv_head=3)


def test_gradients_reach_every_block():
    m = CausalGPT(CausalGPTConfig(**SMALL))
    m(torch.randn(2, 12, 32)).sum().backward()
    dead = [n for n, p in m.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"no gradient reached: {dead}"
