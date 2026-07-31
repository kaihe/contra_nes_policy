"""Checkpoints: one to resume the run exactly, one the evaluation harness can read.

Two files, because they answer two different questions.

``checkpoints/rl-<update>.pt`` is the resumable one: weights, optimizer, scheduler,
the rollout/update counters, and every RNG stream the run touches (Python, NumPy,
Torch, CUDA, the task sampler, the action sampler). Resuming continues the counters
rather than restarting them, so an interrupted run's learning-rate schedule, KL
early-stopping history and logged step axis all line up with the original.

``weights/weight-update=<n>.ckpt`` is the one ``contra_nes_evaluation`` loads. Its
shape is dictated by that harness's ``CheckpointPolicy``: a Lightning-style
``state_dict`` whose keys carry the ``policy.`` prefix, plus a ``hyper_parameters``
dict containing the ``model`` block it rebuilds the architecture from. That contract
is frozen — a weights file this repo can load but the evaluator cannot is a run whose
result cannot be measured.

The BC initialisation's own ``model`` block is read back out of its checkpoint rather
than retyped in ``config_rl.yaml``: the architecture must match the weights being
loaded, and a config file is exactly the wrong place for that invariant to live.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Optional

import numpy as np
import torch

WEIGHTS_PREFIX = "policy."


def model_config_from_checkpoint(ckpt_path: str) -> Dict:
    """The ``model`` block the checkpoint was trained under, ready to instantiate.

    ``view_backbone_ckpt`` is dropped: the frozen encoder's weights are already inside
    this checkpoint, and pointing at the original file again would make the RL run
    fail on any machine that lacks ``contra_agent``'s artefacts — the same reason the
    evaluation harness drops it.
    """
    ckpt = torch.load(os.path.expanduser(ckpt_path), map_location="cpu",
                      weights_only=False)
    cfg = ckpt.get("hyper_parameters")
    if not cfg or "model" not in cfg:
        raise ValueError(
            f"{ckpt_path} carries no hyper_parameters.model; it was not written by "
            f"contra_nes_policy and the architecture cannot be inferred from it")
    model_cfg = dict(cfg["model"])
    model_cfg["view_backbone_ckpt"] = None
    return model_cfg


def load_policy_weights(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load a BC (or weights-only RL) checkpoint into ``model``, strictly.

    Strict on purpose: a silently missing head is an architecture mismatch, and a
    randomly initialised ``pi_head`` would still collect plausible-looking — and
    entirely meaningless — episodes.
    """
    state = torch.load(os.path.expanduser(ckpt_path), map_location="cpu",
                       weights_only=False)
    state = state.get("state_dict", state)
    weights = {k[len(WEIGHTS_PREFIX):]: v for k, v in state.items()
               if k.startswith(WEIGHTS_PREFIX)}
    if not weights:                       # a bare state_dict, not a Lightning ckpt
        weights = state
    model.load_state_dict(weights, strict=True)


def save_weights_only(path: str, model: torch.nn.Module, config: Dict) -> str:
    """Write the evaluator-readable weights file. Returns the path written."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if "model" not in config:
        raise ValueError("the recorded config must carry a 'model' block or "
                         "contra_nes_evaluation cannot rebuild the architecture")
    payload = {
        "state_dict": {WEIGHTS_PREFIX + k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
        "hyper_parameters": config,
    }
    torch.save(payload, path)
    return path


# ── resumable state ───────────────────────────────────────────────────────────

def rng_state(sampler=None, actor=None) -> Dict:
    """Every RNG stream the run advances, in one dict."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if sampler is not None:
        state["task_sampler"] = sampler.state()
    if actor is not None:
        state["actor"] = actor.state()
    return state


def load_rng_state(state: Dict, sampler=None, actor=None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_cpu_bytes(state["torch"]))
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_cpu_bytes(s) for s in state["cuda"]])
    if sampler is not None and "task_sampler" in state:
        sampler.load_state(state["task_sampler"])
    if actor is not None and "actor" in state:
        actor.load_state(state["actor"])


def _cpu_bytes(t: torch.Tensor) -> torch.Tensor:
    """Torch RNG states must be CPU ByteTensors; a checkpoint round-trip can move them."""
    return t.cpu().to(torch.uint8)


def save_resumable(path: str, *, model: torch.nn.Module, optimizer, scheduler,
                   counters: Dict, config: Dict, sampler=None, actor=None) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": {WEIGHTS_PREFIX + k: v.detach().cpu()
                       for k, v in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "counters": dict(counters),
        "hyper_parameters": config,
        "rng": rng_state(sampler, actor),
    }, path)
    return path


def load_resumable(path: str, *, model: torch.nn.Module, optimizer=None, scheduler=None,
                   sampler=None, actor=None, map_location: str = "cpu") -> Dict:
    """Restore a run. Returns the saved counters so the loop continues, not restarts."""
    ckpt = torch.load(os.path.expanduser(path), map_location=map_location,
                      weights_only=False)
    load_policy_weights_from_state(model, ckpt["state_dict"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if "rng" in ckpt:
        load_rng_state(ckpt["rng"], sampler, actor)
    return dict(ckpt.get("counters", {}))


def load_policy_weights_from_state(model: torch.nn.Module, state: Dict) -> None:
    weights = {k[len(WEIGHTS_PREFIX):]: v for k, v in state.items()
               if k.startswith(WEIGHTS_PREFIX)}
    model.load_state_dict(weights or state, strict=True)


def latest_checkpoint(directory: str, pattern: str = "rl-") -> Optional[str]:
    """Newest resumable checkpoint in ``directory``, or None."""
    directory = os.path.expanduser(directory)
    if not os.path.isdir(directory):
        return None
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if f.startswith(pattern) and f.endswith(".pt")]
    return max(files, key=os.path.getmtime) if files else None
