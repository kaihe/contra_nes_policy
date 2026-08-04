"""A small Llama-shaped causal transformer over *continuous* tokens.

Not a language model: there is no vocabulary, no embedding table and no LM head. It
consumes `(B, T, d_model)` — tokens produced by `contra_encoder.encode` plus an
interaction embedding — and returns `(B, T, d_model)`. Heads live in
:mod:`contra_policy.model`.

Replaces VPT's ``ResidualRecurrentBlocks``. That was a Transformer-XL: a learned
relative-position basis, a clipped-causal band mask, carried KV memory and per-chunk
``first`` bookkeeping — all of it scaffolding to simulate a long context from a 32-step
window. With one token per frame an episode *fits*, so the scaffolding goes rather than
gets ported. See ``doc/0002-gpt-policy.md``.

Shape: RMSNorm, RoPE, SwiGLU, grouped-query attention, pre-norm residual blocks — the
minimind / Llama starter arrangement, at the size the old core was (4 layers, 512 wide).

Two properties matter more here than in a text model:

**No learned positional parameter.** VPT's ``bandify`` basis is a learned
``(10, mem_len × tokens)`` tensor whose *shape* depends on the horizon, so lengthening
the context meant reshaping trained weights. RoPE is computed from positions, so context
length is a config value.

**Variable length is the common case, not the exception.** Episodes run from 24 to 1038
decisions with a mean of 100, so a fixed-length batch would be mostly padding. Every
entry point takes an optional ``seq_lens`` and builds a block-diagonal mask, which also
lets several episodes be *packed* into one row with no attention crossing between them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CausalGPTConfig:
    d_model: int = 512
    n_layer: int = 4
    n_head: int = 8
    #: Key/value heads. Equal to ``n_head`` is plain multi-head attention; fewer is
    #: grouped-query, which shrinks the KV cache at rollout. At this size the cache is
    #: not a constraint, so the default is MHA and GQA is available rather than assumed.
    n_kv_head: int = 8
    #: Longest sequence the model will ever see, prefix included. 1024 covers every task
    #: budget (max 1038 → 1024 with a 2-token prefix truncates nothing that matters);
    #: cost tracks *actual* length, so capacity is nearly free. See 0002 §3.
    context: int = 1024
    #: SwiGLU hidden width, before the standard 2/3 correction that keeps the parameter
    #: count equal to a ratio-4 ReLU MLP.
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    dropout: float = 0.0
    norm_eps: float = 1e-5

    def to_dict(self) -> Dict:
        return asdict(self)

    def __post_init__(self):
        if self.d_model % self.n_head:
            raise ValueError(f"d_model {self.d_model} must divide by n_head {self.n_head}")
        if self.n_head % self.n_kv_head:
            raise ValueError(
                f"n_head {self.n_head} must be a multiple of n_kv_head {self.n_kv_head}")


class RMSNorm(nn.Module):
    """Llama's norm: no mean subtraction, no bias — one scale per channel."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In float32: the reciprocal square root of a mean of squares is where low
        # precision bites, and this runs under bf16 autocast.
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(context: int, head_dim: int, theta: float,
                     device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(context, head_dim/2)`` cos and sin tables."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(context, device=device).float()
    freqs = torch.outer(pos, inv)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``(B, H, T, D)`` by position. ``cos``/``sin`` are ``(T, D/2)``."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


def apply_rope_varlen(x: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor) -> torch.Tensor:
    """Rotate packed ``(N, H, D)`` tokens; caches are indexed per-token already."""
    x1, x2 = x.float().chunk(2, dim=-1)
    c, s = cos[:, None], sin[:, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


def _varlen_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      cu_seqlens: torch.Tensor, max_seqlen: int,
                      dropout_p: float) -> torch.Tensor:
    """True varlen causal attention on CUDA, with an eager reference fallback.

    PyTorch 2.9 ships the autograd-enabled ATen FlashAttention varlen operator but not
    the public ``torch.nn.attention.varlen`` wrapper (added in 2.10). Keeping the call
    here makes that version-sensitive boundary explicit and easy to replace on upgrade.
    """
    if q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        return torch.ops.aten._flash_attention_forward(
            q, k, v, cu_seqlens, cu_seqlens, int(max_seqlen), int(max_seqlen),
            float(dropout_p), True, False)[0]

    # Correctness/reference path for CPU tests. It computes each segment independently,
    # so it has the same isolation semantics without pretending to be performant.
    outs = []
    bounds = cu_seqlens.detach().cpu().tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        qs = q[start:end].transpose(0, 1).unsqueeze(0)
        ks = k[start:end].transpose(0, 1).unsqueeze(0)
        vs = v[start:end].transpose(0, 1).unsqueeze(0)
        if qs.shape[1] != ks.shape[1]:
            rep = qs.shape[1] // ks.shape[1]
            ks = ks.repeat_interleave(rep, dim=1)
            vs = vs.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(
            qs, ks, vs, is_causal=True, dropout_p=dropout_p)
        outs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outs, dim=0)


class Attention(nn.Module):
    """Causal grouped-query attention, optionally block-diagonal over packed sequences."""

    def __init__(self, cfg: CausalGPTConfig):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.dropout = cfg.dropout
        self.q = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.o = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, t, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.n_rep > 1:                      # grouped-query: broadcast kv to q heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # `is_causal` and an explicit mask are mutually exclusive in SDPA: when a mask is
        # given it must already encode causality, which `causal_block_mask` does.
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=attn_mask is None,
            dropout_p=self.dropout if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(b, t, -1))

    def forward_varlen(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                       cu_seqlens: torch.Tensor, max_seqlen: int) -> torch.Tensor:
        n = x.shape[0]
        q = self.q(x).view(n, self.n_head, self.head_dim)
        k = self.k(x).view(n, self.n_kv_head, self.head_dim)
        v = self.v(x).view(n, self.n_kv_head, self.head_dim)
        q = apply_rope_varlen(q, cos, sin)
        k = apply_rope_varlen(k, cos, sin)
        out = _varlen_attention(
            q, k, v, cu_seqlens, max_seqlen,
            self.dropout if self.training else 0.0)
        return self.o(out.reshape(n, -1))


class SwiGLU(nn.Module):
    """Llama's MLP. The 2/3 factor keeps the parameter count at a ratio-4 ReLU MLP's."""

    def __init__(self, cfg: CausalGPTConfig):
        super().__init__()
        hidden = int(2 * cfg.mlp_ratio * cfg.d_model / 3)
        hidden = 64 * ((hidden + 63) // 64)          # round up for kernel friendliness
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class Block(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, cfg: CausalGPTConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.norm_attn(x), cos, sin, attn_mask)
        return x + self.mlp(self.norm_mlp(x))

    def forward_varlen(self, x, cos, sin, cu_seqlens, max_seqlen):
        x = x + self.attn.forward_varlen(
            self.norm_attn(x), cos, sin, cu_seqlens, max_seqlen)
        return x + self.mlp(self.norm_mlp(x))


def causal_block_mask(seq_lens: torch.Tensor, total: int,
                      device=None) -> torch.Tensor:
    """Causal *within* each packed sequence, blocked *between* them.

    ``seq_lens`` is ``(B, S)`` — the lengths of the sequences packed into each row, zero
    padded. Returns a ``(B, 1, total, total)`` boolean mask, ``True`` where attention is
    allowed, ready for ``scaled_dot_product_attention``.

    This is what lets several short episodes share one row with no attention crossing
    between them. Episodes average 100 frames against a 1024 context, so packing is the
    difference between 90% of a batch being padding and ~0%.
    """
    b = seq_lens.shape[0]
    # Segment id per position: which packed sequence does this token belong to.
    seg = torch.zeros(b, total, dtype=torch.long, device=device)
    for i in range(b):
        pos = 0
        for j, n in enumerate(seq_lens[i].tolist()):
            n = int(n)
            if n <= 0:
                continue
            seg[i, pos:pos + n] = j + 1          # 0 stays "padding"
            pos += n
    same = seg[:, :, None] == seg[:, None, :]
    real = seg > 0
    idx = torch.arange(total, device=device)
    causal = idx[None, :, None] >= idx[None, None, :]
    return (same & causal & real[:, :, None] & real[:, None, :]).unsqueeze(1)


class CausalGPT(nn.Module):
    """``(B, T, d_model)`` in, ``(B, T, d_model)`` out. No vocabulary, no LM head."""

    def __init__(self, cfg: CausalGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        cos, sin = build_rope_cache(cfg.context, cfg.d_model // cfg.n_head, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)
        # Scale the residual-path output projections by depth, as GPT-2 does: without it
        # the residual stream's variance grows with n_layer and deep stacks start unstable.
        for n, p in self.named_parameters():
            if n.endswith(("o.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                seq_lens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``seq_lens`` (B, S) builds a block-diagonal mask; ``attn_mask`` overrides it.

        With neither, attention is plain causal over the whole row — correct when each
        row holds exactly one episode and nothing is padded.
        """
        b, t, c = x.shape
        if t > self.cfg.context:
            raise ValueError(f"sequence of {t} exceeds context {self.cfg.context}")
        if attn_mask is None and seq_lens is not None:
            attn_mask = causal_block_mask(seq_lens, t, device=x.device)
        cos = self.rope_cos[:t].to(x.device)
        sin = self.rope_sin[:t].to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin, attn_mask)
        return self.norm(x)

    def forward_varlen(self, x: torch.Tensor, cu_seqlens: torch.Tensor,
                       max_seqlen: int) -> torch.Tensor:
        """Run concatenated sequences without padding or cross-sequence attention."""
        if int(max_seqlen) > self.cfg.context:
            raise ValueError(
                f"sequence of {int(max_seqlen)} exceeds context {self.cfg.context}")
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        starts = torch.repeat_interleave(cu_seqlens[:-1].long(), lengths.long())
        positions = torch.arange(x.shape[0], device=x.device) - starts
        cos = self.rope_cos.to(x.device)[positions]
        sin = self.rope_sin.to(x.device)[positions]
        for blk in self.blocks:
            x = blk.forward_varlen(x, cos, sin, cu_seqlens, int(max_seqlen))
        return self.norm(x)
