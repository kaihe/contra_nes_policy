"""The policy: an episode is a sequence, and the action is read off each frame.

    [interaction, goal, frame_1, frame_2, …, frame_T]  ->  action_t at each frame

One `contra_encoder.encode` per image — agent frames and the goal frame alike — then a
causal transformer over the result. See ``doc/0002-gpt-policy.md``.

What is *not* here, and why, because their absence is the design:

* **No previous-action input.** Measured: feeding it collapses boss 8.8% → 1.8% and
  doubles grounding error 5.3 → 12.1 px. The sequence contains no action tokens at all,
  so this is not a Decision Transformer.
* **No recurrent memory, no chunking, no `first` flag.** An episode fits in the context,
  so nothing carries across a boundary that no longer exists.
* **No `index_bias`.** There is one token per frame, so heads read position *t*. The old
  arithmetic pointed at a token that causally *preceded* the interaction token, and the
  aux head consequently never knew which task it was looking at.

The goal token is a prefix, not a per-frame input: every frame attends back to it at
whatever distance RoPE gives, which is the comparison the encoder deliberately stopped
trying to make on its own.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn

from contra_encoder import load_pretrained_encoder
from contra_encoder.heads import heatmap_readout
from contra_encoder.net import EncoderConfig, build_encoder
from contra_policy.action_space import NUM_ACTIONS
from contra_policy.causal import CausalGPT, CausalGPTConfig
from contra_policy.goal import NUM_INTERACTIONS

#: Tokens before the first frame: `[interaction, goal]`.
PREFIX = 2


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@dataclass
class PolicyConfig:
    """Everything that changes parameter shapes, so a checkpoint rebuilds itself."""

    core: Dict = field(default_factory=lambda: CausalGPTConfig().to_dict())
    #: Path to a `contra_encoder` checkpoint. None builds one from scratch, which is
    #: only sensible for tests — stage A exists so this does not have to be learned twice.
    encoder_ckpt: Optional[str] = None
    encoder: Optional[Dict] = None          # used only when encoder_ckpt is None
    #: Frozen for the first BC run *on purpose*: it makes the causal core the only
    #: variable, so a bad validation loss is unambiguously the core's fault rather than a
    #: co-adaptation between two things that both changed.
    freeze_encoder: bool = True
    # Legacy checkpoints omit `value_head` and used aux_size=32, so these defaults must
    # preserve their exact state-dict shape. New action-only checkpoints explicitly set
    # value_head=false and aux_size=0.
    value_head: bool = True
    aux_size: int = 32                      # 0 disables the legacy goal-heatmap head
    #: Images per encoder forward. A whole-episode batch is batch x T frames — 4 x 321
    #: is 1,284 at 256px, which peaks near the 16 GB card. Chunking bounds the encoder's
    #: activation peak independently of how long the episodes in a batch happen to be,
    #: so a single long boss episode cannot OOM a run that was fine a step earlier.
    encode_chunk: int = 256

    def to_dict(self) -> Dict:
        return asdict(self)


class ContraPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = (load_pretrained_encoder(cfg.encoder_ckpt, freeze=cfg.freeze_encoder)
                        if cfg.encoder_ckpt
                        else build_encoder(EncoderConfig(**(cfg.encoder or {}))))
        if cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()

        d = self.encoder.cfg.hiddim
        core_cfg = CausalGPTConfig(**{**cfg.core, "d_model": d})
        self.core = CausalGPT(core_cfg)

        self.interaction = nn.Embedding(NUM_INTERACTIONS + 1, d)   # +1 for id -1
        self.pi_head = nn.Linear(d, NUM_ACTIONS)
        self.value_head = nn.Linear(d, 1) if cfg.value_head else None
        # Grounding lives here now, not in the encoder: predicting where the goal is
        # requires comparing this frame against the goal token, which is what attention
        # upstream has just done.
        self.aux_head = nn.Linear(d, cfg.aux_size ** 2) if cfg.aux_size > 0 else None

    @property
    def context(self) -> int:
        return self.core.cfg.context

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, T, S, S, 3)`` uint8 → ``(B, T, d)``. Frozen encoders skip the graph."""
        b, t = images.shape[:2]
        flat = images.reshape(b * t, *images.shape[2:])
        chunk = max(1, int(self.cfg.encode_chunk))
        ctx = torch.no_grad() if self.cfg.freeze_encoder else _null()
        with ctx:
            if flat.shape[0] <= chunk:
                tok = self.encoder.encode(flat)
            else:
                tok = torch.cat([self.encoder.encode(flat[i:i + chunk])
                                 for i in range(0, flat.shape[0], chunk)], dim=0)
        return tok.view(b, t, -1)

    def forward(self, images: torch.Tensor, goal_image: torch.Tensor,
                interaction: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """``images`` (B, T, S, S, 3), ``goal_image`` (B, S, S, 3), ``interaction`` (B,).

        Returns per-frame heads, each ``(B, T, …)`` — aligned with ``images``, with the
        prefix already stripped.
        """
        b, t = images.shape[:2]
        if t + PREFIX > self.context:
            raise ValueError(
                f"{t} frames + {PREFIX} prefix exceeds context {self.context}")

        frames = self.encode_images(images)                      # (B, T, d)
        goal = self.encode_images(goal_image.unsqueeze(1))       # (B, 1, d)
        inter = self.interaction(interaction + 1).unsqueeze(1)   # (B, 1, d)

        h = self.core(torch.cat([inter, goal, frames], dim=1), attn_mask=attn_mask)
        h = h[:, PREFIX:]                                        # frame positions only

        out = {"pi_logits": self.pi_head(h)}
        if self.value_head is not None:
            out["vpred"] = self.value_head(h).squeeze(-1)
        if self.aux_head is not None:
            heat = self.aux_head(h).view(b, t, self.cfg.aux_size, self.cfg.aux_size)
            point, exist = heatmap_readout(heat)
            out.update({"goal_heatmap": heat, "point": point, "exist": exist})
        return out

    # -- persistence --------------------------------------------------------

    def save(self, path: str, **extra) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"policy": self.state_dict(),
                    "config": self.cfg.to_dict(), **extra}, path)
        return path


def build_policy(cfg: Optional[PolicyConfig] = None, **overrides) -> ContraPolicy:
    if cfg is None:
        cfg = PolicyConfig(**overrides)
    elif overrides:
        cfg = PolicyConfig(**{**cfg.to_dict(), **overrides})
    return ContraPolicy(cfg)


def load_policy(path: str, map_location: str = "cpu",
                strict: bool = True) -> ContraPolicy:
    """Rebuild a policy from its own checkpoint.

    The architecture comes out of the file, never from a caller's config — a mismatch
    there loads silently wrong weights. The encoder is rebuilt from the *policy's*
    stored config too, so a moved or deleted stage-A checkpoint cannot change what
    loads.
    """
    ckpt = torch.load(os.path.expanduser(path), map_location=map_location,
                      weights_only=False)
    cfg = PolicyConfig(**ckpt["config"])
    # The encoder's weights are inside this checkpoint; do not re-read stage A.
    enc_cfg = cfg.encoder
    if cfg.encoder_ckpt and enc_cfg is None:
        enc_cfg = torch.load(os.path.expanduser(cfg.encoder_ckpt),
                             map_location="cpu", weights_only=False)["config"]
    model = ContraPolicy(PolicyConfig(**{**cfg.to_dict(),
                                         "encoder_ckpt": None, "encoder": enc_cfg}))
    model.load_state_dict(ckpt["policy"], strict=strict)
    return model
