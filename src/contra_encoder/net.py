"""One token per frame, one token per goal, and the grounding head that keeps them honest.

    encode_goal(goal_image, goal_mask)   -> (B, hiddim)                once per episode
    encode_frame(frame, goal_token)      -> (B, hiddim), (B, C, A, A)  once per decision

The frame encoder stays **goal-conditioned**, and that is not an oversight. A per-frame
heatmap answers "where is *the goal entity* in this frame", which is unanswerable
without the goal. What gets cheaper versus ``CrossViewContraRocket.encode_view_tokens``
is not that the goal leaves per-frame work — it is that conditioning happens against
**one token** (FiLM over channels) instead of spatial cross-attention over the goal's
whole ``minres x minres`` patch grid, and that the result is one token instead of four.

Conditioning is FiLM rather than concatenation because it is O(channels) per frame and
leaves the spatial resolution untouched, which the heatmap still needs.

The conv trunk is ``contra_policy.encoder.ConvEncoder``, reused unchanged so an existing
``ae_pretrained.pt`` can still initialise it. Unlike the policy's use of it, the trunk
here is **trainable by default** — the point of this package is to train the encoder,
and with a frozen trunk the grounding loss would only ever reach the projection layers.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from contra_encoder.heads import HeatmapHead, heatmap_readout
from contra_encoder.heads import _norm as _gn
from contra_policy.encoder import ConvEncoder, build_view_backbone
from contra_policy.goal import NUM_INTERACTIONS


@dataclass
class EncoderConfig:
    """Everything that changes the parameter shapes, so a checkpoint can rebuild itself."""

    image_size: int = 256
    hiddim: int = 512            # token width; matches the policy's `hiddim`
    depth: int = 32              # ConvEncoder base width
    minres: int = 4              # conv trunk output grid
    mask_depth: int = 8          # goal-mask trunk base width; its input is a blob map
    # Channels after the 1x1 reduction, before flattening to a token. The trunk emits
    # 1024 channels on a 4x4 grid at 256px, and `Linear(1024*16, 512)` alone is 8.4M
    # parameters — more than the trunk that feeds it. Reducing channels first makes the
    # projection 2.1M and costs 0.26M, while leaving the 4x4 layout intact for the
    # heatmap head to invert.
    proj_ch: int = 256
    aux_size: int = 32           # heatmap grid, A
    head_depth: int = 32         # HeatmapHead base width
    n_classes: int = 1           # channels of the *goal* head; 1 is the policy's contract
    # Channels of a second, goal-independent head supervised by the shards' per-class
    # entity occupancy: player, player_bullets, enemies, enemy_bullets. 0 = off.
    #
    # A separate head rather than extra channels on the goal head, because the two
    # answer different questions: the goal head is conditioned on *which* entity is the
    # target and feeds the pinned `point_err_px` gate, while entity occupancy is a
    # property of the frame alone. Sharing a head would entangle the gate metric with a
    # signal that has nothing to do with the task.
    entity_classes: int = 0
    view_backbone_ckpt: Optional[str] = None
    freeze_view_backbone: bool = False   # a rebuild trains the trunk; see module docstring

    def to_dict(self) -> Dict:
        return asdict(self)


class ContraFrameEncoder(nn.Module):
    """Frame + goal → one token per frame, plus occupancy logits decoded from that token."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        s, h = cfg.image_size, cfg.hiddim

        # -- trunks ---------------------------------------------------------
        # ONE RGB trunk, shared between the agent frame and the goal frame, exactly as
        # `CrossViewContraRocket` shares `view_backbone` between obs and cross-view.
        # Both inputs are Contra screens, so the features transfer and a second trunk
        # would be 11.2M duplicated parameters learning the same thing from less data.
        self.view_backbone = build_view_backbone(
            size=s, depth=cfg.depth, minres=cfg.minres,
            pretrained_path=cfg.view_backbone_ckpt,
            freeze=cfg.freeze_view_backbone)
        self.mask_backbone = ConvEncoder(size=s, in_ch=1, depth=cfg.mask_depth,
                                         minres=cfg.minres, with_head=False)

        view_ch = self.view_backbone.conv_out_ch
        goal_ch = view_ch + self.mask_backbone.conv_out_ch
        cells = cfg.minres * cfg.minres
        pc = cfg.proj_ch

        # -- 1x1 channel reduction, before any flattening -------------------
        self.frame_reduce = nn.Sequential(nn.Conv2d(view_ch, pc, 1), _gn(pc), nn.SiLU())
        self.goal_reduce = nn.Sequential(nn.Conv2d(goal_ch, pc, 1), _gn(pc), nn.SiLU())

        # -- goal token -----------------------------------------------------
        self.goal_proj = nn.Sequential(
            nn.Linear(pc * cells, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h))
        self.interaction = nn.Embedding(NUM_INTERACTIONS + 1, h)   # +1 for "no goal" (id -1)

        # -- FiLM conditioning, applied to the *reduced* frame map -----------
        # Zero-init so conditioning starts as identity (gamma=0 → scale 1, beta=0) and
        # the trunk trains from a clean signal on step 0 instead of through noise.
        self.film = nn.Linear(h, 2 * pc)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        # -- frame token ----------------------------------------------------
        self.frame_proj = nn.Sequential(
            nn.Linear(pc * cells, h), nn.LayerNorm(h), nn.SiLU(), nn.Linear(h, h))
        self.token_ln = nn.LayerNorm(h)

        # -- grounding, decoded from the token ------------------------------
        self.heatmap_head = HeatmapHead(dim=h, grid=cfg.aux_size,
                                        n_classes=cfg.n_classes, depth=cfg.head_depth)
        # Both heads read the same token, which is the point: the entity target's job is
        # to force sprite-level structure into that one vector, where the goal head
        # alone only ever demands one blob's worth.
        self.entity_head = (HeatmapHead(dim=h, grid=cfg.aux_size,
                                        n_classes=cfg.entity_classes,
                                        depth=cfg.head_depth)
                            if cfg.entity_classes > 0 else None)

    # -- goal ---------------------------------------------------------------

    def encode_goal(self, goal_image: torch.Tensor, goal_mask: torch.Tensor,
                    interaction: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(B, S, S, 3)`` uint8 goal frame + ``(B, S, S)`` uint8 blob → ``(B, hiddim)``.

        ``interaction`` is optional: when given, its embedding is added, which is how
        the single goal token carries "kill / pick / avoid / traverse / boss" as well as
        the pixels. The policy keeps a separate interaction token in the sequence; this
        one exists so the *encoder's* conditioning knows the task type too.
        """
        img = goal_image.permute(0, 3, 1, 2).float() / 255.0
        msk = goal_mask.unsqueeze(1).float() / 255.0
        z = torch.cat([self.view_backbone.forward_features(img),
                       self.mask_backbone.forward_features(msk)], dim=1)
        tok = self.goal_proj(self.goal_reduce(z).flatten(1))
        if interaction is not None:
            tok = tok + self.interaction(interaction + 1)
        return tok

    # -- frame --------------------------------------------------------------

    def encode_frame(self, frame: torch.Tensor, goal_token: torch.Tensor
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``(B, S, S, 3)`` uint8 frame + ``(B, hiddim)`` goal → ``(token, heat)``.

        ``heat`` is ``(B, n_classes, A, A)`` logits, decoded from ``token`` — so the
        token is the only path the grounding gradient has, which is what forces spatial
        structure to survive the compression.
        """
        x = frame.permute(0, 3, 1, 2).float() / 255.0
        z = self.frame_reduce(self.view_backbone.forward_features(x))   # (B, pc, m, m)
        gamma, beta = self.film(goal_token).chunk(2, dim=-1)
        z = z * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        token = self.token_ln(self.frame_proj(z.flatten(1)))
        return token, self.heatmap_head(token)

    # -- convenience --------------------------------------------------------

    def forward(self, frame: torch.Tensor, goal_image: torch.Tensor,
                goal_mask: torch.Tensor,
                interaction: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """One-shot path for pretraining and for tests.

        Returns the keys the policy's loss modules already expect — ``goal_heatmap``,
        ``point``, ``exist`` — so ``GoalHeatmapLoss`` can score this encoder unchanged.
        ``goal_heatmap`` drops the class axis when ``n_classes == 1``, matching the
        policy's ``(B, A, A)`` contract; multi-class keeps it.
        """
        goal_token = self.encode_goal(goal_image, goal_mask, interaction)
        token, heat = self.encode_frame(frame, goal_token)
        single = heat[:, 0] if self.cfg.n_classes == 1 else heat
        point, exist = heatmap_readout(heat[:, 0])
        out = {"token": token, "goal_token": goal_token,
               "goal_heatmap": single, "point": point, "exist": exist}
        if self.entity_head is not None:
            out["entity_heatmap"] = self.entity_head(token)
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
    there loads silently wrong weights, which is the failure mode
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
