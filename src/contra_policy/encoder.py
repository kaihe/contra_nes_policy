"""Vision backbones — the Contra stand-in for ROCKET-2's frozen DINO ViT.

ROCKET-2 encodes both the agent view and the cross-view reference with a frozen
``timm`` ViT-B/16 (``features_only``), takes the last feature map, and treats its
patch grid as a token sequence. That pretrained-on-photographs prior is a poor fit
for 8-bit NES frames, where the task-critical entities are 2-4px sprites; the
``contra_agent`` repo already pretrained an encoder that *does* fit, in
``dreamer/train_ae.py`` — a DreamerV3-style ConvEncoder trained on level-1 traces
with a four-class entity-occupancy aux loss (player / player bullets / enemies /
enemy bullets) precisely so small sprites survive into the embedding.

So :class:`ConvEncoder` is borrowed from ``contra_agent/dreamer/models.py``
unchanged in architecture — same layer shapes, same names — which is what lets
``ae_pretrained.pt`` load straight into it. The one adaptation is
:meth:`ConvEncoder.forward_features`: ROCKET-2 needs a *spatial* token grid to fuse
agent view against cross view, while the Dreamer encoder ends in a flatten+Linear
to a single vector. Stopping before ``head`` gives a (B, C, minres, minres) map —
at the pretrained config (256px, depth 32) that is 1024 channels on a 4x4 grid, i.e.
16 tokens of width 1024, the direct analogue of ViT patch tokens.

The view backbone is frozen exactly as in ROCKET-2; the mask backbone is a small
from-scratch single-channel copy (ROCKET-2's ViT-tiny mask encoder is trainable too).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn


def _num_layers(size: int, minres: int = 4) -> int:
    """How many stride-2 halvings take `size` down to `minres` (power-of-two)."""
    n = int(round(math.log2(size / minres)))
    assert minres * 2 ** n == size, f"size {size} is not minres*2^k (minres={minres})"
    return n


def _norm(channels: int) -> nn.Module:
    # GroupNorm ~ DreamerV3's per-layer LayerNorm. Largest group count <=32 that
    # divides `channels`.
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ConvEncoder(nn.Module):
    """Frame (B, in_ch, size, size) in [0, 1] → embedding (B, embed_dim).

    Architecture is byte-compatible with ``contra_agent/dreamer/models.py`` so
    ``ae_pretrained.pt`` state dicts load without remapping. Use
    :meth:`forward_features` for the spatial token map the policy consumes.
    """

    def __init__(self, size: int = 128, in_ch: int = 3, depth: int = 32,
                 embed_dim: int = 1024, minres: int = 4, with_head: bool = True):
        super().__init__()
        n = _num_layers(size, minres)
        layers: list[nn.Module] = []
        ch = in_ch
        for i in range(n):
            out = depth * (2 ** i)
            layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), _norm(out), nn.SiLU()]
            ch = out
        self.convs = nn.Sequential(*layers)
        self.minres = minres
        self.conv_out_ch = ch                       # e.g. 1024 at size=256, depth=32
        self.flat_dim = ch * minres * minres
        self.embed_dim = embed_dim
        # The policy only ever calls forward_features, so `head` exists purely to let
        # a Dreamer `ae_pretrained.pt` load with strict=True. It is 16.8M parameters
        # at the pretrained config — never build it for a from-scratch encoder.
        self.head = nn.Linear(self.flat_dim, embed_dim) if with_head else None

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, size, size) → (B, conv_out_ch, minres, minres) spatial tokens."""
        return self.convs(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.head is None:
            raise RuntimeError("this ConvEncoder was built with with_head=False; "
                               "use forward_features()")
        return self.head(self.convs(x).flatten(1))


def load_pretrained_view_backbone(ckpt_path: str, freeze: bool = True) -> ConvEncoder:
    """Build a :class:`ConvEncoder` from an ``ae_pretrained.pt`` checkpoint.

    The checkpoint carries the config it was trained under (``size``, ``depth``,
    ``embed_dim``), so the module is sized from the file rather than from our config
    — a mismatch here would load silently wrong weights. ``minres`` is not stored;
    it is recovered from the number of conv blocks.
    """
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    state = ckpt["encoder"]
    n_blocks = sum(1 for k in state if k.startswith("convs.") and k.endswith(".weight")
                   and state[k].ndim == 4)
    minres = cfg["size"] // (2 ** n_blocks)
    enc = ConvEncoder(size=cfg["size"], in_ch=3, depth=cfg["depth"],
                      embed_dim=cfg["embed_dim"], minres=minres)
    enc.load_state_dict(state, strict=True)
    # `head` had to exist for the strict load, but the policy only calls
    # forward_features. Dropping it now saves 16.8M frozen params (67MB of VRAM, and
    # the same again in every saved checkpoint) that would otherwise just ride along.
    enc.head = None
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
    enc.pretrained_size = cfg["size"]
    return enc


def build_view_backbone(size: int, depth: int, minres: int,
                        pretrained_path: str | None = None,
                        freeze: bool = True) -> ConvEncoder:
    """The agent-view / cross-view RGB encoder, pretrained-and-frozen when available.

    Falls back to a from-scratch trainable encoder if no checkpoint is given, so the
    repo runs standalone; the frozen-pretrained path is the intended one and mirrors
    ROCKET-2's ``for param in self.view_backbone.parameters(): requires_grad = False``.
    """
    if pretrained_path:
        enc = load_pretrained_view_backbone(pretrained_path, freeze=freeze)
        if enc.pretrained_size != size:
            raise ValueError(
                f"checkpoint was trained at {enc.pretrained_size}px but config asks for "
                f"{size}px; set model.image_size={enc.pretrained_size} or drop the checkpoint")
        return enc
    return ConvEncoder(size=size, in_ch=3, depth=depth, minres=minres, with_head=False)


def build_mask_backbone(size: int, depth: int, minres: int) -> ConvEncoder:
    """The single-channel goal-mask encoder (ROCKET-2's ``in_chans=1`` ViT-tiny).

    Always trained from scratch and always small — its input is a Gaussian blob map,
    not a photograph.
    """
    return ConvEncoder(size=size, in_ch=1, depth=depth, minres=minres, with_head=False)
