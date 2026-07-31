"""Multi-process collection — one emulator per process, one shared copy of the weights.

``stable_retro`` allows exactly one emulator per process, so throughput past a single
env means more *processes*, not more envs. Each worker owns its own emulator, its own
task catalog and its own replica of the policy, and returns finished
:class:`~contra_policy.rl.trajectory.Episode` objects.

Weights move through **shared memory**, not through the queue. The policy is ~63 M
parameters (~250 MB of fp32), and pickling that to every worker before every rollout
would cost more than the rollout. The parent instead holds one shared-memory copy of
the state dict, writes the current weights into it, and bumps an integer version; a
worker reloads only when the version it holds is stale. Episodes come back through a
queue, which is the right way round — a rollout batch is a few hundred MB of uint8
frames produced once per update, while weights would move once per update *per
worker*.

Ordering is not deterministic across workers (whichever finishes first is appended
first), so the *set* of episodes in a batch is seeded and reproducible but their order
is not. Nothing downstream depends on that order: PPO shuffles episodes anyway, and
every statistic in :func:`~contra_policy.rl.trajectory.rollout_stats` is a mean.

Default is ``num_workers: 0`` — everything in the parent process, one emulator, fully
deterministic. Turn it up once a run is known to be correct.
"""

from __future__ import annotations

import math
import os
import signal
from typing import Dict, List, Optional, Sequence

import torch
import torch.multiprocessing as mp

from contra_policy.model import CrossViewContraRocket
from contra_policy.rl.rollout import EpisodeCollector
from contra_policy.rl.tasks import TaskCatalog, TaskSampler
from contra_policy.rl.trajectory import Episode


def shared_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """A CPU copy of the model's state dict living in shared memory."""
    return {k: v.detach().cpu().clone().share_memory_()
            for k, v in model.state_dict().items()}


def publish_weights(model: torch.nn.Module, shared: Dict[str, torch.Tensor]) -> None:
    """Copy the current weights into the shared buffers, in place."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            shared[k].copy_(v.detach().cpu())


def _build_collector(rank: int, spec: Dict, model: torch.nn.Module,
                     device: torch.device) -> EpisodeCollector:
    catalog = TaskCatalog(
        task_root=spec["task_root"], shard_dir=spec["shard_dir"],
        families=spec["families"], split="train", image_size=spec["image_size"],
        sigma_px=spec["sigma_px"], cache_dir=spec["cache_dir"],
        prompt_cache=spec["prompt_cache"], segment_cache=spec["segment_cache"],
        verbose=rank == 0)
    catalog.assert_split("train")
    sampler = TaskSampler(catalog, spec["natural_fraction"],
                          spec["balanced_family_fraction"],
                          spec.get("family_multiplier"),
                          seed=spec["seed"] + 7919 * (rank + 1))
    return EpisodeCollector(
        model, catalog, sampler, batch_size=spec["batch_size"],
        budget_mult=spec["budget_mult"], min_budget=spec["min_budget"],
        image_size=spec["image_size"], device=device,
        temperature=spec["temperature"], precision=spec["precision"],
        seed=spec["seed"] + 104729 * (rank + 1), reward=spec["reward"],
        max_episode_steps=spec["max_episode_steps"],
        collect_goal_points=spec["collect_goal_points"],
        owner=f"worker-{rank}")


def _die_with_parent(parent_pid: int) -> None:
    """Ask the kernel to SIGKILL this worker the moment the parent goes away.

    ``daemon=True`` is not enough. It is implemented by an ``atexit`` hook in the
    parent, so it covers a *clean* parent exit and nothing else — not SIGTERM (what
    ``timeout`` sends), not SIGKILL, not an OOM kill. Those are exactly the cases
    that matter: a worker holds a CUDA context, a 63 M-parameter model and an
    emulator, so a leaked set of them is several GB that never comes back, and the
    next run starts on top of it. ``PR_SET_PDEATHSIG`` moves the guarantee into the
    kernel, where no signal can skip it.
    """
    try:
        import ctypes

        PR_SET_PDEATHSIG = 1
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
    except Exception:                                        # pragma: no cover
        return                                               # not Linux; nothing to do
    # pdeathsig fires on the parent's death, so a parent that died between spawn and
    # this call would never deliver it. Re-check by hand to close that window.
    if os.getppid() != parent_pid:                           # pragma: no cover
        os._exit(1)


def _worker_main(rank: int, spec: Dict, model_cfg: Dict,
                 shared: Dict[str, torch.Tensor], cmd_q, out_q,
                 parent_pid: int) -> None:
    _die_with_parent(parent_pid)
    torch.set_num_threads(1)
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:                                        # pragma: no cover
        pass

    device = torch.device(spec["worker_device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    model = CrossViewContraRocket(**model_cfg).to(device).eval()
    collector = _build_collector(rank, spec, model, device)
    version = -1
    try:
        while True:
            cmd = cmd_q.get()
            if cmd is None:
                break
            min_steps, min_episodes, want_version = cmd
            if version != want_version:
                model.load_state_dict({k: v for k, v in shared.items()}, strict=True)
                model.to(device).eval()
                version = want_version
            episodes = collector.collect(min_steps, min_episodes)
            out_q.put((rank, episodes))
    except Exception as exc:                                 # pragma: no cover
        import traceback

        out_q.put((rank, RuntimeError(
            f"collector worker {rank} failed: {exc}\n{traceback.format_exc()}")))
    finally:
        collector.close()


class MultiProcessCollector:
    """``num_workers`` processes, each with one emulator, collecting into one batch."""

    def __init__(self, model: torch.nn.Module, model_cfg: Dict, spec: Dict,
                 num_workers: int):
        if num_workers < 1:
            raise ValueError("MultiProcessCollector needs at least one worker; use "
                             "EpisodeCollector directly for in-process collection")
        self.num_workers = num_workers
        self.model = model
        self.shared = shared_state_dict(model)
        self.version = 0
        ctx = mp.get_context("spawn")
        self.cmd_qs = [ctx.Queue() for _ in range(num_workers)]
        self.out_q = ctx.Queue()
        self.procs = [
            ctx.Process(target=_worker_main,
                        args=(rank, spec, model_cfg, self.shared,
                              self.cmd_qs[rank], self.out_q, os.getpid()),
                        daemon=True)
            for rank in range(num_workers)]
        for p in self.procs:
            p.start()

    def collect(self, min_steps: int, min_episodes: int) -> List[Episode]:
        publish_weights(self.model, self.shared)
        self.version += 1
        per_steps = int(math.ceil(min_steps / self.num_workers))
        per_episodes = int(math.ceil(min_episodes / self.num_workers))
        for q in self.cmd_qs:
            q.put((per_steps, per_episodes, self.version))
        out: List[Episode] = []
        for _ in range(self.num_workers):
            _rank, payload = self.out_q.get()
            if isinstance(payload, BaseException):
                self.close()
                raise payload
            out.extend(payload)
        return out

    def close(self) -> None:
        for q in self.cmd_qs:
            try:
                q.put(None)
            except Exception:                                # pragma: no cover
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():                                 # pragma: no cover
                p.terminate()
        self.procs = []
