"""LightningModule wrapper — the stand-in for ``minestudio.offline.MineLightning``.

MineLightning's job in ROCKET-2's trainloop is small and entirely reproducible: run
the policy over a window, feed the latents to a list of objective callbacks, sum
them, log, and drive an AdamW + linear-warmup schedule. That is what this is, with
the Minecraft-specific parts (action-space plumbing, MineStudio's logging keys)
removed.

Warmup matters more here than the config might suggest: the recurrent core is
initialised at ``n_block ** -0.5`` residual scale and the resampler starts from
noise, while the view backbone arrives fully trained and frozen — without warmup the
first few hundred steps push the resampler hard enough to wash the pretrained
features out of the token stream.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import lightning as L
import torch

from contra_policy.loss import ContraObjective
from contra_policy.model import CrossViewContraRocket


def _to_model_input(batch: Dict[str, Any]) -> Dict[str, Any]:
    """The subset of the batch the policy consumes (the rest are loss targets)."""
    return {
        "image": batch["image"],
        "cross_view": batch["cross_view"],
        "prev_action": batch["prev_action"],
        "prev_action_dropout": batch["prev_action_dropout"],
        "first": batch["first"],
    }


class ContraRocketLightning(L.LightningModule):

    def __init__(self, policy: CrossViewContraRocket, learning_rate: float = 4e-5,
                 weight_decay: float = 0.001, warmup_steps: int = 2000,
                 bc_weight: float = 1.0, heatmap_weight: float = 1.0,
                 heatmap_pos_weight: float = 10.0,
                 label_smoothing: float = 0.0, log_freq: int = 20,
                 class_weights: Optional[torch.Tensor] = None,
                 families: tuple = (), carry_memory: bool = False,
                 hyperparameters: Optional[Dict] = None):
        super().__init__()
        self.policy = policy
        self.objective = ContraObjective(bc_weight, heatmap_weight, heatmap_pos_weight,
                                         label_smoothing, class_weights, families)
        self.carry_memory = carry_memory
        self._memory: Dict[str, Optional[list]] = {"train": None, "val": None}
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.log_freq = log_freq
        self.save_hyperparameters(hyperparameters or {}, ignore=["policy"])

    def _carried(self, stage: str, batch_size: int) -> Optional[list]:
        """The memory to hand this step, or None to start from the initial state.

        Dropped whenever the batch size changes, since lane *j* of the previous batch
        is only the same stream as lane *j* of this one at a fixed width.
        """
        if not self.carry_memory:
            return None
        mem = self._memory[stage]
        if mem is not None and mem[0].shape[0] != batch_size:
            return None
        return mem

    def _step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        memory = self._carried(stage, batch["action"].shape[0])
        latents, memory = self.policy(_to_model_input(batch), memory)
        if self.carry_memory:
            # Detached: the memory is context for the next window, not a path for
            # gradient to flow back into the previous one (truncated BPTT).
            self._memory[stage] = [m.detach() for m in memory]
        loss, metrics = self.objective(latents, batch)
        on_step = stage == "train"
        self.log_dict({f"{stage}/{k}": v for k, v in metrics.items()},
                      on_step=on_step, on_epoch=not on_step, prog_bar=False,
                      batch_size=batch["action"].shape[0], sync_dist=not on_step)
        self.log(f"{stage}/loss_bar", loss.detach(), prog_bar=True,
                 batch_size=batch["action"].shape[0])
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # Each pass restarts its sampler's lanes from the top, so memory carried in from
    # the previous pass belongs to different episodes. The per-window `first` flag
    # only resets lanes that happen to be at an episode boundary, so without these
    # the first batch of every validation pass would attend to a stale context from
    # ~1000 steps ago.
    def on_train_epoch_start(self) -> None:
        self._memory["train"] = None

    def on_validation_epoch_start(self) -> None:
        self._memory["val"] = None

    def configure_optimizers(self):
        # Frozen backbone params must be excluded or AdamW allocates (and DDP syncs)
        # optimizer state for 28M weights that never move.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate,
                                      weight_decay=self.weight_decay)

        def lr_lambda(step: int) -> float:
            if self.warmup_steps <= 0:
                return 1.0
            return min(1.0, (step + 1) / self.warmup_steps)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }
