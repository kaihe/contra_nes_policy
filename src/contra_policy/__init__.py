"""Shared primitives and the data path for the Contra (NES) policy.

The policy itself is being rebuilt (``doc/0002-design-gpt-policy.md``); what lives here now is
the part that outlives any architecture:

* :mod:`~contra_policy.action_space` — the 21-action space. **Frozen**: shared verbatim
  with ``contra_nes_data`` and ``contra_nes_evaluation``.
* :mod:`~contra_policy.goal` — PPU coordinates, the goal-blob renderer, interaction ids.
  **Frozen**: ``goal_mask`` draws both the cross-view prompt and the training target, so
  a change here moves both at once.
* :mod:`~contra_policy.loss` — ``point_err_px`` is **frozen** (``contra_nes_evaluation``
  pins the number it reports); the rest is behaviour-cloning machinery.
* :mod:`~contra_policy.dataset` — shard tars → windowed samples, entity targets.
* :mod:`~contra_policy.encoder` — the ConvEncoder trunk, shared with
  :mod:`contra_encoder`.
* :mod:`~contra_policy.causal` — the Llama-shaped causal core the new policy is built on.

The previous policy (``CrossViewContraRocket``, the vendored VPT transformer-XL core,
the Lightning module and the PPO stack) was removed at commit `60a0c34`'s successor for
a clean rebuild. It is in git history, not gone.
"""

from contra_policy.action_space import ACTION_NAMES, NUM_ACTIONS
from contra_policy.goal import INTERACTIONS, NUM_INTERACTIONS

__all__ = ["ACTION_NAMES", "NUM_ACTIONS", "INTERACTIONS", "NUM_INTERACTIONS"]
