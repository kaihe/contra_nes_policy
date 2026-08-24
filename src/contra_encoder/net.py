"""One token per image — the same function for an agent frame and for a goal frame.

    encode(image) -> token                       (B, hiddim)
    forward(image) -> token, entity_heatmap, [reconstruction]

**The encoder does not know what the goal is, and that is the design** (see
``doc/0002-symmetric-encoder.md``). It answers "what is in this image and where",
never "where is the target". Goal matching belongs to the policy's temporal
attention, which compares a frame token against a goal token — a strictly stronger
mechanism than the FiLM channel modulation this replaced, and the place
``contra_policy.model`` already computes goal grounding.

That symmetry is possible because ``goal.png`` needs no special handling: it is a real
episode frame with the target painted into the RGB as a saturated orange blob (sampled
at the goal points: (225, 110, 18) against an image mean of (56, 70, 14)). An image
with the answer drawn on it is still just an image.

Deleting the goal-specific path removed the mask trunk, a second projection and the
conditioning layer — 3.65M parameters that existed only to answer a question the policy
answers better.

Supervision is occupancy, decoded **from the token**, never from the conv map: that is
what forces spatial structure through the 512-d bottleneck rather than letting it live
in the feature map the head could read directly.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from contra_encoder.heads import HeatmapHead, ReconstructionHead
from contra_encoder.heads import _norm as _gn
from contra_policy.encoder import build_view_backbone


@dataclass
class EncoderConfig:
    """Everything that changes parameter shapes, so a checkpoint can rebuild itself."""

    image_size: int = 256
    hiddim: int = 512            # token width; must equal the policy's `hiddim`
    depth: int = 32              # ConvEncoder base width
    minres: int = 4              # conv trunk output grid
    # Channels after the 1x1 reduction, before flattening to a token. Flattening
    # 1024x4x4 straight into a Linear is 8.4M parameters — more than the trunk feeding
    # it; reducing first costs 0.26M and makes the projection 2.4M.
    proj_ch: int = 256
    aux_size: int = 32           # occupancy grid, A
    head_depth: int = 32         # HeatmapHead base width
    # player / player_bullets / enemies / enemy_bullets. 0 disables the head.
    entity_classes: int = 4
    # Reconstruction is an open question, not a default — see 0002 §4. A full
    # ConvDecoder at this config is 27.97M parameters, larger than everything else
    # combined, and the precedent only shows that a recon-*only* encoder goes
    # entity-blind, not that recon helps once an entity head exists. Settle by ablation.
    reconstruct: bool = False
    recon_depth: int = 16
    view_backbone_ckpt: Optional[str] = None
    freeze_view_backbone: bool = False   # a rebuild trains the trunk
    # Non-null only for the accepted 0019 native-resolution temporal encoder.
    input_height: Optional[int] = None
    input_width: Optional[int] = None
    n_layers: Optional[int] = None
    input_kind: str = "rgb"
    first_frame_delta: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class ContraFrameEncoder(nn.Module):
    """Image → one token, with occupancy (and optionally pixels) decoded back out."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        s, h, pc = cfg.image_size, cfg.hiddim, cfg.proj_ch
        temporal = cfg.input_kind == "rgb_signed_frame_difference"
        if temporal:
            if not cfg.input_height or not cfg.input_width or not cfg.n_layers:
                raise ValueError("temporal encoder requires input height, width, and layers")
            self.view_backbone = _TemporalBackbone(
                height=cfg.input_height, width=cfg.input_width,
                depth=cfg.depth, n_layers=cfg.n_layers)
        else:
            self.view_backbone = build_view_backbone(
                size=s, depth=cfg.depth, minres=cfg.minres,
                pretrained_path=cfg.view_backbone_ckpt,
                freeze=cfg.freeze_view_backbone)
        view_ch = self.view_backbone.conv_out_ch
        cells = (self.view_backbone.output_hw[0] * self.view_backbone.output_hw[1]
                 if temporal else cfg.minres * cfg.minres)

        self.reduce = nn.Sequential(nn.Conv2d(view_ch, pc, 1), _gn(pc), nn.SiLU())
        self.proj = nn.Sequential(
            nn.Linear(pc * cells, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h))
        self.token_ln = nn.LayerNorm(h)

        self.entity_head = (HeatmapHead(dim=h, grid=cfg.aux_size,
                                        n_classes=cfg.entity_classes,
                                        depth=cfg.head_depth)
                            if cfg.entity_classes > 0 else None)
        self.recon_head = (ReconstructionHead(dim=h, size=s, depth=cfg.recon_depth,
                                              base=cfg.minres)
                           if cfg.reconstruct else None)

    # -- the one operation -------------------------------------------------

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """``(B, S, S, 3)`` uint8 → ``(B, hiddim)``. Agent frame or goal frame alike."""
        if self.cfg.input_kind == "rgb_signed_frame_difference":
            return self.encode_pair(image, image)
        x = image.permute(0, 3, 1, 2).float() / 255.0
        z = self.reduce(self.view_backbone.forward_features(x))     # (B, pc, m, m)
        return self.token_ln(self.proj(z.flatten(1)))

    def encode_pair(self, current: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        """Encode a current/previous pair for the 0019 temporal checkpoint."""
        if self.cfg.input_kind != "rgb_signed_frame_difference":
            raise ValueError("encode_pair is available only on a temporal encoder")
        expected = (int(self.cfg.input_height), int(self.cfg.input_width), 3)
        if (current.dtype != torch.uint8 or previous.dtype != torch.uint8 or
                current.shape != previous.shape or tuple(current.shape[1:]) != expected):
            raise ValueError(f"temporal inputs must be equal uint8 B{expected}")
        cur = current.permute(0, 3, 1, 2).float().div(255)
        prev = previous.permute(0, 3, 1, 2).float().div(255)
        z = self.reduce(self.view_backbone.forward_features(
            torch.cat((cur, cur - prev), dim=1)))
        return self.token_ln(self.proj(z.flatten(1)))

    def encode_sequence(self, images: torch.Tensor) -> torch.Tensor:
        """Encode one episode sequence, assigning zero delta to its first frame."""
        if len(images) == 0:
            return torch.empty((0, self.cfg.hiddim), device=images.device)
        return self.encode_pair(images, torch.cat((images[:1], images[:-1]), dim=0))

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Token plus whatever heads are enabled.

        ``entity_heatmap`` is ``(B, C, A, A)`` logits; ``reconstruction`` is
        ``(B, S, S, 3)`` in [0, 1], matching the input's layout so a caller can compare
        it to ``image / 255`` without transposing.
        """
        token = self.encode(image)
        out = {"token": token}
        if self.entity_head is not None:
            out["entity_heatmap"] = self.entity_head(token)
        if self.recon_head is not None:
            out["reconstruction"] = self.recon_head(token).permute(0, 2, 3, 1)
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: str, **extra) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"encoder": self.state_dict(),
                    "config": self.cfg.to_dict(), **extra}, path)
        return path


def build_encoder(cfg: Optional[EncoderConfig] = None, **overrides) -> ContraFrameEncoder:
    """``EncoderConfig`` (or kwargs) → an initialised encoder."""
    if cfg is None:
        cfg = EncoderConfig(**overrides)
    elif overrides:
        cfg = EncoderConfig(**{**cfg.to_dict(), **overrides})
    return ContraFrameEncoder(cfg)


def load_pretrained_encoder(path: str, freeze: bool = False,
                            map_location: str = "cpu") -> ContraFrameEncoder:
    """Rebuild an encoder from its own checkpoint.

    The architecture comes out of the file, never from a caller's config — a mismatch
    there loads silently wrong weights, which is the failure
    ``contra_policy.encoder.load_pretrained_view_backbone`` documents and avoids the
    same way.
    """
    ckpt = torch.load(os.path.expanduser(path), map_location=map_location,
                      weights_only=False)
    enc = ContraFrameEncoder(EncoderConfig(**ckpt["config"]))
    enc.load_state_dict(ckpt["encoder"], strict=True)
    if freeze:
        for p in enc.parameters():
            p.requires_grad = False
        enc.eval()
    return enc


class _TemporalBackbone(nn.Module):
    """Rectangular six-channel backbone matching data experiment 0019."""

    def __init__(self, *, height: int, width: int, depth: int, n_layers: int):
        super().__init__()
        layers = []
        channels = 6
        for index in range(n_layers):
            out = depth * 2 ** index
            layers += [nn.Conv2d(channels, out, 4, stride=2, padding=1),
                       _gn(out), nn.SiLU()]
            channels = out
        self.convs = nn.Sequential(*layers)
        self.conv_out_ch = channels
        self.output_hw = (height // 2 ** n_layers, width // 2 ** n_layers)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.convs(image)
