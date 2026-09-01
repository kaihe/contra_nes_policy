"""Frozen Contra policy adapter used by MCTS."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Sequence

import numpy as np
import torch

from contra_policy.action_space import NUM_ACTIONS


class TorchSearchPolicy:
    """Encode frames once and evaluate arbitrary causal tree histories."""

    num_actions = NUM_ACTIONS

    def __init__(self, model, interaction: int, *, device: torch.device,
                 precision: str = "bf16"):
        self.model = model.to(device).eval()
        self.interaction = int(interaction)
        self.device = device
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[precision]
        if device.type != "cuda":
            self.autocast_dtype = None
        if model.cfg.use_goal_image:
            raise ValueError("the first Laser MCTS implementation requires a null-goal policy")
        if model.encoder.cfg.input_kind == "rgb_signed_frame_difference":
            raise ValueError("branch search needs paired-frame encoding for this encoder")
        with torch.no_grad():
            self.goal_token = model.null_goal.detach().to(device).unsqueeze(0)

    def _autocast(self):
        return (torch.autocast("cuda", dtype=self.autocast_dtype)
                if self.autocast_dtype is not None else nullcontext())

    @torch.no_grad()
    def encode(self, observation: np.ndarray) -> torch.Tensor:
        image = torch.from_numpy(np.ascontiguousarray(observation)).to(self.device).unsqueeze(0)
        with self._autocast():
            token = self.model.encoder.encode(image)
        return token[0].float()

    @torch.no_grad()
    def priors(self, frame_tokens: Sequence[Any]) -> np.ndarray:
        if len(frame_tokens) + 2 > self.model.context:
            raise RuntimeError(f"MCTS history of {len(frame_tokens)} frames exceeds policy "
                               f"context {self.model.context - 2}")
        frames = torch.stack(list(frame_tokens)).float().unsqueeze(0)
        interaction = torch.tensor([self.interaction], device=self.device)
        with self._autocast():
            logits = self.model.forward_tokens(
                frames, self.goal_token, interaction)["pi_logits"][0, -1]
        return torch.softmax(logits.float(), dim=-1).cpu().numpy()
