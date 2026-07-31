"""Decode occupancy heatmaps **from a single embedding vector**, and read them out.

The from-a-vector part is the whole point. If the heatmap were decoded from the conv
feature map, the frame token would be free to discard spatial structure and the
grounding loss would never touch it. Forcing the map through the token is what makes
"one token per frame" a claim about the token rather than about the encoder's internals.

That is not a new bet. ``contra_agent/dreamer/train_ae.py`` pretrains this same
``ConvEncoder`` with an ``EntityHead(embed_dim, n_classes=4, grid, depth)`` that decodes
four occupancy maps out of the embedding, precisely because — in its words — a
recon-only frozen encoder "goes entity-blind". This is that head, generalised over
class count so the 1-class goal target works now and the 4-class entity target works
once ``contra_nes_data`` exports per-frame RAM.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn


def _norm(channels: int) -> nn.Module:
    """GroupNorm with the largest group count <=32 that divides ``channels``.

    Matches ``contra_policy.encoder._norm`` so the two halves of the network
    normalise the same way.
    """
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class HeatmapHead(nn.Module):
    """``(B, dim)`` embedding → ``(B, n_classes, grid, grid)`` occupancy logits.

    A linear projection to a small spatial seed, then transposed-conv upsampling to
    ``grid``. ``base`` is chosen so the number of stride-2 steps is an integer; at
    ``grid=32`` that is a 4x4 seed and three upsamples.

    Logits, not probabilities: ``GoalHeatmapLoss`` applies
    ``binary_cross_entropy_with_logits`` and ``contra_nes_evaluation`` applies its own
    sigmoid to ``exist``.
    """

    def __init__(self, dim: int, grid: int = 32, n_classes: int = 1,
                 depth: int = 64, base: int = 4):
        super().__init__()
        if grid % base or not float(math.log2(grid / base)).is_integer():
            raise ValueError(f"grid {grid} must be base {base} times a power of two")
        n_up = int(round(math.log2(grid / base)))
        self.grid, self.n_classes, self.base = grid, n_classes, base
        # Widest at the seed and halving on the way up, mirroring the encoder's
        # doubling on the way down.
        ch = depth * 2 ** n_up
        self.seed = nn.Linear(dim, ch * base * base)
        self.seed_ch = ch

        layers: list[nn.Module] = []
        for _ in range(n_up):
            out = max(depth, ch // 2)
            layers += [nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1),
                       _norm(out), nn.SiLU()]
            ch = out
        self.ups = nn.Sequential(*layers)
        self.out = nn.Conv2d(ch, n_classes, 3, padding=1)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        b = token.shape[0]
        x = self.seed(token).view(b, self.seed_ch, self.base, self.base)
        x = self.ups(x)
        return self.out(x)


def heatmap_readout(heat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(..., A, A)`` logits → ``(point, exist)`` in the evaluator's conventions.

    Arithmetic copied from ``CrossViewContraRocket.heatmap_readout`` and it must stay
    identical, because ``point_err_px`` is pinned by ``contra_nes_evaluation`` and the
    whole gate on this rebuild is "boss point error no worse than the 5.3 px the
    current policy reports".

    In particular the soft-argmax has **no** +0.5 pixel-centre offset:
    ``goal.goal_mask`` places a blob at ``cx = x_norm * A`` exactly, so dividing the
    expected column by ``A`` inverts the target rendering. Adding half a cell would
    bias every prediction by ``0.5/A`` — about 3.7 screen px at A=32.

    ``exist`` is the max logit, "is any cell occupied", returned with a trailing
    singleton so it matches the policy's ``(..., 1)`` shape.
    """
    *lead, a, a2 = heat.shape
    if a != a2:
        raise ValueError(f"expected a square heatmap, got {a}x{a2}")
    flat = heat.reshape(*lead, a * a)
    p = torch.softmax(flat.float(), dim=-1).reshape(*lead, a, a)
    idx = torch.arange(a, device=heat.device, dtype=p.dtype)
    x = (p.sum(dim=-2) * idx).sum(-1) / a          # sum over rows → per-column mass
    y = (p.sum(dim=-1) * idx).sum(-1) / a          # sum over cols → per-row mass
    point = torch.stack([x, y], dim=-1).to(heat.dtype)
    exist = flat.max(dim=-1).values.unsqueeze(-1)
    return point, exist
