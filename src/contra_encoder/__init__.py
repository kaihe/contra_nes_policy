"""The Contra frame encoder: one token per frame, grounding decoded from that token.

Split out of ``contra_policy`` because it has its own training process and its own
checkpoint format. The dependency runs one way — ``contra_policy.model`` imports
:func:`build_encoder`; nothing here imports ``contra_policy.model``.

The exception is deliberate: this package *does* import ``contra_policy.goal`` and
``point_err_px`` from ``contra_policy.loss``. Those are the repo's frozen shared
primitives — the same renderer that draws the cross-view prompt must draw the
supervision target, and ``point_err_px`` has exactly one definition because
``contra_nes_evaluation`` pins the number it reports. Re-implementing either here is
the drift they exist to prevent.

Why one token, when the policy today emits four view tokens plus interaction plus
previous action:

* a 32-decision window contains 2.1% of training tasks end to end, and boss — the
  family this work exists to fix — has a median budget of 281 decisions;
* 76% of a training update is GPU, so a longer window is not free;
* ``[interaction, goal, img x N]`` is 1,026 tokens for a 1,024-decision episode where
  the current layout needs 6,144, which is what makes whole-task context affordable.

See ``doc/0730-encoder-rebuild.md``.
"""

from __future__ import annotations

from contra_encoder.heads import HeatmapHead, heatmap_readout
from contra_encoder.net import (ContraFrameEncoder, EncoderConfig, build_encoder,
                                load_pretrained_encoder)

__all__ = [
    "ContraFrameEncoder",
    "EncoderConfig",
    "HeatmapHead",
    "build_encoder",
    "heatmap_readout",
    "load_pretrained_encoder",
]
