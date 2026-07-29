"""ROCKET-2's cross-view goal-conditioned policy, re-implemented for Contra (NES).

Mirrors the layout of the reference repo (``model.py`` / ``cross_view_dataset.py`` /
``train.py``), with the Minecraft- and MineStudio-specific parts replaced:

* :mod:`~contra_policy.model` — ``CrossViewContraRocket``, the policy
* :mod:`~contra_policy.dataset` — WebDataset shards → windowed training samples
* :mod:`~contra_policy.loss` — behaviour cloning + the cross-view grounding aux head
* :mod:`~contra_policy.lit` — LightningModule (replaces ``MineLightning``)
* :mod:`~contra_policy.encoder` — Contra-pretrained frozen vision backbone
* :mod:`~contra_policy.vpt` — vendored VPT transformer-XL recurrent core
"""

from contra_policy.action_space import ACTION_NAMES, NUM_ACTIONS
from contra_policy.goal import INTERACTIONS, NUM_INTERACTIONS

__all__ = ["ACTION_NAMES", "NUM_ACTIONS", "INTERACTIONS", "NUM_INTERACTIONS"]
