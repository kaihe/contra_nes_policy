"""CrossViewContraRocket — ROCKET-2's cross-view goal-aligned policy, for Contra.

A step-for-step re-implementation of ``ROCKET-2/model.py``'s ``CrossViewRocket``.
The five-stage structure is preserved exactly:

1. encode the agent view with a frozen view backbone → spatial tokens
2. encode the cross-view reference image with the *same* frozen backbone
3. encode the cross-view goal mask with a separate single-channel backbone
4. concat (2) and (3) channelwise, project both streams to ``hiddim``, then fuse
   agent-view and cross-view tokens *spatially* through a transformer resampler
   that reads out ``num_view_tokens`` learned CLS tokens
5. lay the per-step tokens out as one long sequence — ``[view..., interaction,
   prev_action]`` per timestep — and run a VPT transformer-XL over it with carried
   memory; read the aux head off the interaction token and the policy/value heads
   off the last token

What changes for Contra, and why:

* **Backbones.** ROCKET-2's frozen DINO ViT-B/16 is replaced by the Contra-pretrained
  ConvEncoder from ``contra_agent`` (see :mod:`contra_policy.encoder`). Same role,
  same freezing, tokens instead of patches.
* **Action head.** ROCKET-2 inherits MineStudio's hierarchical Minecraft head
  (button groups + camera bins). Contra's traces live entirely inside a 21-action
  discrete set, so ``pi_head`` is a single ``Linear(hiddim, 21)`` and the prev-action
  embedding is one ``nn.Embedding(21, hiddim)`` rather than a per-key sum.
* **Goal mask.** Regenerated from the shard's goal points rather than shipped as a
  segmentation mask (see :mod:`contra_policy.goal`).
* **Aux head.** ROCKET-2 regresses scalars — ``Linear(hiddim, 1+2+4)`` for
  exist/point/bbox. Here it is a dense ``aux_size**2`` goal-occupancy heatmap; see
  the head's own comment below for why the scalars could not work on this data.
  ``exist`` and ``point`` are still emitted, read out of the map.

Everything else — the resampler, the interaction embedding, the prev-action dropout
embedding, ``index_bias`` bookkeeping, the clipped-causal memory transformer, the
aux head's placement on the interaction token — is the reference design.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from contra_policy.action_space import NUM_ACTIONS
from contra_policy.encoder import build_mask_backbone, build_view_backbone
from contra_policy.goal import NUM_INTERACTIONS
from contra_policy.vpt.util import FanInInitReLULayer, ResidualRecurrentBlocks


class CrossViewContraRocket(nn.Module):

    def __init__(
        self,
        image_size: int = 256,
        view_depth: int = 32,
        mask_depth: int = 8,
        minres: int = 4,
        view_backbone_ckpt: Optional[str] = None,
        freeze_view_backbone: bool = True,
        hiddim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 4,
        timesteps: int = 32,
        mem_len: int = 32,
        use_prev_action: bool = True,
        num_view_tokens: int = 4,
        aux_size: int = 32,
        num_actions: int = NUM_ACTIONS,
        resampler_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.num_actions = num_actions
        self.timesteps = timesteps

        self.view_backbone = build_view_backbone(
            image_size, view_depth, minres, view_backbone_ckpt, freeze_view_backbone)
        self.mask_backbone = build_mask_backbone(image_size, mask_depth, minres)

        view_chs = self.view_backbone.conv_out_ch
        mask_chs = self.mask_backbone.conv_out_ch
        self.updim_obs = nn.Conv2d(view_chs, hiddim, kernel_size=1, bias=False)
        self.updim_cross = nn.Conv2d(view_chs + mask_chs, hiddim, kernel_size=1, bias=False)

        self.num_view_tokens = num_view_tokens
        self.view_cls_tokens = nn.Parameter(torch.randn(1, self.num_view_tokens, hiddim) * 1e-3)
        self.view_resampler = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hiddim,
                nhead=num_heads,
                dim_feedforward=hiddim * 2,
                dropout=resampler_dropout,
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        # +1 slot for "no goal" (interaction id -1), matching ROCKET-2's obj_id + 1.
        self.interaction = nn.Embedding(NUM_INTERACTIONS + 1, hiddim)
        self.num_step_tokens = self.num_view_tokens + 1

        # Negative index of the interaction token within a timestep's token block,
        # which is [view..., interaction] and then optionally prev_action. With no
        # prev-action token the interaction token is last, hence -1.
        #
        # This was -2 (so -3 with prev_action), which lands on the *last view token* —
        # and since attention is causal over the flattened sequence, that token
        # precedes the interaction token and can never attend to it. The aux head
        # therefore never knew whether the task was kill, pick, avoid, traverse or
        # boss. See doc/0728-policy-todo.md §1.1.
        self.index_bias = -1
        self.use_prev_action = use_prev_action
        if self.use_prev_action:
            self.action_embedding = nn.Embedding(num_actions, hiddim)
            self.num_step_tokens += 1
            self.index_bias -= 1

        self.dropout_embedding = nn.Parameter(torch.randn(1, 1, hiddim) * 1e-3)

        self.recurrent = ResidualRecurrentBlocks(
            hidsize=hiddim,
            timesteps=timesteps * self.num_step_tokens,
            recurrence_type="transformer",
            is_residual=True,
            use_pointwise_layer=True,
            pointwise_ratio=4,
            pointwise_use_activation=False,
            attention_mask_style="clipped_causal",
            attention_heads=num_heads,
            attention_memory_size=(mem_len + timesteps) * self.num_step_tokens,
            n_block=num_layers,
            inject_condition=False,
        )
        self.lastlayer = FanInInitReLULayer(hiddim, hiddim, layer_type="linear",
                                            batch_norm=False, layer_norm=True)
        self.final_ln = nn.LayerNorm(hiddim)

        # Aux grounding head: a dense goal-occupancy heatmap, replacing the scalar
        # exist + 2-D point + 4-D bbox regression. Every pixel outside the blob is a
        # negative example, so families whose goal is visible on 100% of frames
        # (kill, boss) still supply negatives — with a scalar `exist` they supplied
        # none at all, and the head could not learn to report absence. `point` and
        # `exist` are still emitted, derived below, because the evaluation harness
        # reads exactly those two keys plus `pi_logits`.
        self.aux_size = aux_size
        self.aux_vis_head = nn.Linear(hiddim, aux_size * aux_size)
        self.pi_head = nn.Linear(hiddim, num_actions)
        # Present for parity with ROCKET-2 (MinePolicy always carries one) and for a
        # later RL fine-tune. The behaviour-cloning objective never reads `vpred`, so
        # this head receives no gradient during offline training — which is why the
        # multi-GPU strategy is ddp_find_unused_parameters_true, as upstream.
        self.value_head = nn.Linear(hiddim, 1)

    # ── stages ────────────────────────────────────────────────────────────────

    def encode_view_tokens(self, agent_view: torch.Tensor, cross_view: Dict) -> torch.Tensor:
        """(B, T, H, W, 3) agent view + cross-view dict → (B, T, num_view_tokens, hiddim).

        The cross-view tensors may arrive with or without a time axis. A Contra task
        episode has exactly one goal, so the dataset ships it per *window* — encoding
        it once and expanding the resulting tokens over T is identical to encoding T
        copies, and roughly halves the frozen-backbone compute. ROCKET-2's per-timestep
        shape ((B, T, H, W, 3)) is still accepted, for a goal that varies within a
        window.
        """
        b, t = agent_view.shape[:2]

        # 1. encode observation with view_backbone
        obs_rgb = rearrange(agent_view, "b t h w c -> (b t) c h w").float() / 255.0
        x_obs = self.view_backbone.forward_features(obs_rgb)
        x_obs = self.updim_obs(x_obs)
        x_obs = rearrange(x_obs, "b c h w -> b (h w) c")

        cross_img, cross_msk = cross_view["cross_view_image"], cross_view["cross_view_obj_mask"]
        per_timestep = cross_img.ndim == 5
        if per_timestep:
            cross_img = rearrange(cross_img, "b t h w c -> (b t) h w c")
            cross_msk = rearrange(cross_msk, "b t h w -> (b t) h w")

        # 2. encode cross-view image with the same view_backbone
        cross_rgb = rearrange(cross_img, "n h w c -> n c h w").float() / 255.0
        x_cross_image = self.view_backbone.forward_features(cross_rgb)

        # 3. encode the cross-view goal mask with the single-channel mask_backbone
        cross_mask = rearrange(cross_msk, "n h w -> n 1 h w").float() / 255.0
        x_cross_mask = self.mask_backbone.forward_features(cross_mask)

        # 4. fuse cross image and cross mask in the feature dimension
        x_cross = torch.cat([x_cross_image, x_cross_mask], dim=1)
        x_cross = self.updim_cross(x_cross)
        x_cross = rearrange(x_cross, "n c h w -> n (h w) c")

        if not per_timestep:      # one goal per window → broadcast to every timestep
            x_cross = repeat(x_cross, "b s c -> (b t) s c", t=t)

        # 5. fuse obs and cross in the spatial dimension, read out the CLS tokens
        x_cls = self.view_cls_tokens.expand(x_obs.shape[0], -1, -1)
        x_view = torch.cat([x_cls, x_obs, x_cross], dim=1)
        x_view = self.view_resampler(x_view)[:, :self.num_view_tokens, :]
        return rearrange(x_view, "(b t) n c -> b t n c", b=b)

    def temporal_reason(self, x: torch.Tensor,
                        memory: Optional[List[torch.Tensor]] = None,
                        first_step: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """``x`` is (B, T*num_step_tokens, C); ``first_step`` is a (B,) episode flag.

        ``first`` is per *token*, not per timestep, and marking only the very first
        token of the window is what the memory mask needs: every token at or after it
        has the same cumulative-first count, so they attend to each other and none of
        them attends to the carried memory. Without an episode-stream loader every
        window is independent and this is all-False, which is the default.
        """
        b, t = x.shape[:2]
        first = torch.zeros((b, t), dtype=torch.bool, device=x.device)
        if first_step is not None:
            first[:, 0] = first_step.to(device=x.device, dtype=torch.bool)
        if memory is None:
            memory = [state.to(x.device) for state in self.recurrent.initial_state(b)]
        z, memory = self.recurrent(x, first, memory)
        z = F.relu(z, inplace=False)
        z = self.lastlayer(z)
        z = self.final_ln(z)
        return z, memory

    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None
                ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        b, t = input["image"].shape[:2]

        x = self.encode_view_tokens(input["image"], input["cross_view"])

        # interaction token — +1 so id -1 ("no goal") lands on embedding row 0
        x_cond = self.interaction(input["cross_view"]["cross_view_obj_id"] + 1)
        x = torch.cat([x, rearrange(x_cond, "b t c -> b t 1 c")], dim=-2)

        # prev-action token, randomly replaced by a learned "unknown" embedding so
        # the policy cannot degenerate into copying its own last action
        if self.use_prev_action:
            x_prev_a = self.action_embedding(input["prev_action"])
            if "prev_action_dropout" in input:
                keep = input["prev_action_dropout"][..., None]
                dropout_embedding = repeat(self.dropout_embedding, "1 1 c -> b t c", b=b, t=t)
                x_prev_a = x_prev_a * keep + dropout_embedding * (1 - keep)
            x = torch.cat([x, rearrange(x_prev_a, "b t c -> b t 1 c")], dim=-2)

        x = rearrange(x, "b t n c -> b (t n) c")
        first_step = input.get("first")
        if first_step is not None and first_step.ndim == 2:
            first_step = first_step[:, 0]        # (B, T) window flag → (B,) episode flag
        z, memory = self.temporal_reason(x, memory, first_step)
        z = rearrange(z, "b (t n) c -> b t n c", t=t)

        heat = self.aux_vis_head(z[:, :, self.index_bias, :])
        heat = rearrange(heat, "b t (h w) -> b t h w", h=self.aux_size)
        point, exist = self.heatmap_readout(heat)
        latents = {
            "pi_logits": self.pi_head(z[:, :, -1, :]),
            "vpred": self.value_head(z[:, :, -1, :]),
            "goal_heatmap": heat,          # logits — what the aux loss trains on
            "exist": exist,                # logit, as the evaluator expects
            "point": point,                # [0,1] screen space, as before
        }
        return latents, memory

    def heatmap_readout(self, heat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, T, A, A) logits → ``(point, exist)`` in the pre-existing conventions.

        ``point`` is a soft-argmax, deliberately without the usual +0.5 pixel-centre
        offset: ``goal.goal_mask`` places a blob at ``cx = x_norm * A`` exactly, so
        dividing the expected column by ``A`` inverts the target rendering. Adding
        half a cell would bias every prediction by ``0.5/A`` — about 3.7 screen px at
        A=32, a quarter of the current error.

        ``exist`` is the max logit: "is any cell occupied". The evaluator applies its
        own sigmoid, so a logit is what must come out.
        """
        b, t, a, _ = heat.shape
        flat = heat.reshape(b, t, a * a)
        p = torch.softmax(flat.float(), dim=-1).reshape(b, t, a, a)
        idx = torch.arange(a, device=heat.device, dtype=p.dtype)
        x = (p.sum(dim=-2) * idx).sum(-1) / a          # sum over rows → per-column mass
        y = (p.sum(dim=-1) * idx).sum(-1) / a          # sum over cols → per-row mass
        point = torch.stack([x, y], dim=-1).to(heat.dtype)
        exist = flat.max(dim=-1).values.unsqueeze(-1)
        return point, exist

    # ── rollout helpers ───────────────────────────────────────────────────────

    def initial_state(self, batch_size: Optional[int] = None) -> List[torch.Tensor]:
        device = next(self.parameters()).device
        if batch_size is None:
            return [s.squeeze(0).to(device) for s in self.recurrent.initial_state(1)]
        return [s.to(device) for s in self.recurrent.initial_state(batch_size)]

    @torch.no_grad()
    def act(self, input: Dict, memory: Optional[List[torch.Tensor]] = None,
            temperature: float = 1.0) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Sample one action index per batch element from a single-timestep input."""
        latents, memory = self.forward(input, memory)
        logits = latents["pi_logits"][:, -1, :] / max(temperature, 1e-6)
        return torch.distributions.Categorical(logits=logits).sample(), memory
