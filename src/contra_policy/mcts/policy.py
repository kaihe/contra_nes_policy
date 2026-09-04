"""Neural-network adapter for policy/value MCTS evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np
import torch

from contra_policy.action_space import NUM_ACTIONS
from contra_policy.mcts.core import Evaluation


class TorchSearchPolicy:
    num_actions = NUM_ACTIONS

    def __init__(self, model, interaction: int, *, device: torch.device,
                 precision: str = "bf16"):
        if model.value_head is None:
            raise ValueError("MCTS requires a value head")
        if model.cfg.use_goal_image:
            raise ValueError("the fixed Laser loop requires a null-goal policy")
        self.model = model.to(device).eval()
        self.interaction, self.device = int(interaction), device
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[precision]
        self.autocast_dtype = dtype if device.type == "cuda" else None
        self.goal_token = model.null_goal.detach().to(device).unsqueeze(0)

    def _autocast(self):
        return (torch.autocast("cuda", dtype=self.autocast_dtype)
                if self.autocast_dtype is not None else nullcontext())

    @torch.no_grad()
    def encode(self, observation: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(np.ascontiguousarray(observation)).to(self.device).unsqueeze(0)
        with self._autocast():
            return self.model.encoder.encode(image)[0].float()

    @torch.no_grad()
    def evaluate(self, frame_tokens: Sequence[Any], previous_action: int) -> Evaluation:
        del previous_action
        if len(frame_tokens) + 2 > self.model.context:
            raise RuntimeError("MCTS history exceeds the GPT context")
        frames = torch.stack(list(frame_tokens)).float().unsqueeze(0)
        interaction = torch.tensor([self.interaction], device=self.device)
        with self._autocast():
            out = self.model.forward_tokens(frames, self.goal_token, interaction)
        priors = torch.softmax(out["pi_logits"][0, -1].float(), -1).cpu().numpy()
        value = torch.sigmoid(out["vpred"][0, -1].float())
        return Evaluation(priors, float(value.cpu()))
