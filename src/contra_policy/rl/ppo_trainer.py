"""Actor-critic PPO over complete emulator episodes (design 0027, experiment 0028)."""

from __future__ import annotations

import collections
import math
import os
import random
import signal
import time
from typing import Dict, List, Optional, Sequence

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from contra_policy.model import load_policy
from contra_policy.rl.buffer import (Episode, EpisodeOutcome, GroupBatch,
                                     iter_minibatches, iter_ppo_minibatches)
from contra_policy.rl.ppo import PPOConfig, explained_variance, gae, ppo_loss
from contra_policy.rl.rollout import EpisodeCollector
from contra_policy.rl.tasks import TaskCatalog, TaskSampler
from contra_policy.rl.trainer import CSVLogger, _null, _seed, _won, wilson


def load_actor_critic(path: str, map_location: str = "cpu"):
    """Load an actor checkpoint and add a neutral critic when it has none."""
    policy = load_policy(path, map_location=map_location)
    if policy.value_head is None:
        policy.value_head = nn.Linear(policy.core.cfg.d_model, 1)
        nn.init.zeros_(policy.value_head.weight)
        nn.init.zeros_(policy.value_head.bias)
        policy.cfg.value_head = True
    return policy


class PPOTrainer:
    def __init__(self, args: DictConfig, run_dir: str):
        self.args, self.run_dir = args, run_dir
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[args.precision]
        if self.device.type != "cuda":
            self.autocast_dtype = None

        self.policy = load_actor_critic(args.init_from, "cpu").to(self.device)
        self.policy.eval()
        self.ref = load_policy(args.init_from, map_location="cpu").to(self.device)
        for p in self.ref.parameters():
            p.requires_grad = False
        self.ref.eval()
        self.cfg = PPOConfig(**OmegaConf.to_container(args.ppo, resolve=True))
        self.cfg.temperature = float(args.rollout.temperature)

        self.actor_params = [p for n, p in self.policy.named_parameters()
                             if not n.startswith("value_head.") and p.requires_grad]
        self.critic_params = list(self.policy.value_head.parameters())
        self.optimizer = torch.optim.AdamW([
            {"params": self.actor_params, "lr": float(args.train.actor_learning_rate)},
            {"params": self.critic_params, "lr": float(args.train.critic_learning_rate)},
        ], weight_decay=float(args.train.weight_decay))

        self.catalog = TaskCatalog(
            task_root=args.task_root, shard_dir=args.shard_dir,
            families=list(args.families), split="train",
            image_size=int(args.image_size), sigma_px=float(args.sigma_px),
            cache_dir=args.cache_dir,
            task_filter=OmegaConf.to_container(args.get("task_filter", {}) or {},
                                               resolve=True),
            expected_tasks=int(args.get("expected_tasks", 0) or 0),
            require_prompt=bool(self.policy.cfg.use_goal_image))
        self.catalog.assert_split("train")
        self.sampler = TaskSampler(
            self.catalog, float(args.sampling.natural_fraction),
            float(args.sampling.balanced_family_fraction),
            OmegaConf.to_container(args.sampling.family_multiplier, resolve=True),
            seed=int(args.seed))
        self.collector = EpisodeCollector(
            self.policy, self.catalog, self.sampler,
            batch_size=int(args.rollout.batch_size),
            budget_mult=float(args.rollout.budget_mult),
            min_budget=int(args.rollout.min_budget), image_size=int(args.image_size),
            device=self.device, temperature=float(args.rollout.temperature),
            precision=args.precision, seed=int(args.seed),
            reward=OmegaConf.to_container(args.reward, resolve=True),
            max_episode_steps=int(args.rollout.max_episode_steps),
            collect_goal_points=False, owner="PPOTrainer")
        self.task = self.catalog.tasks[0]
        self.rng = np.random.default_rng(int(args.seed))
        self.logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.update = 0
        self.elapsed = 0.0
        self.warmup_metrics: Dict[str, float] = {}
        self._resume(str(args.resume_from) if args.get("resume_from") else None)
        print(f"[ppo] {sum(p.numel() for p in self.actor_params)/1e6:.2f}M actor · "
              f"{sum(p.numel() for p in self.critic_params)} critic params · "
              f"{int(args.rollout.episodes_per_update)} episodes/update", flush=True)

    def _ctx(self):
        return (torch.autocast("cuda", dtype=self.autocast_dtype)
                if self.autocast_dtype is not None else _null())

    def _forward(self, batch, model=None):
        model = model or self.policy
        with self._ctx():
            return model(batch.image, batch.goal_image, batch.interaction)

    def _collect(self, n: int) -> List[Episode]:
        return self.collector.collect_groups([[self.task] for _ in range(n)])

    def _check_memory(self) -> None:
        limit = float(self.args.train.get("memory_limit_gb", 0.0) or 0.0)
        if limit <= 0:
            return
        total = available = 0.0
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    total = float(line.split()[1]) / 1e6
                elif line.startswith("MemAvailable:"):
                    available = float(line.split()[1]) / 1e6
                    break
        used = total - available
        if used > limit:
            raise MemoryError(f"host memory {used:.2f} GB exceeds limit {limit:.1f} GB")

    # -- critic warmup --------------------------------------------------

    def _warmup_step(self, episodes: Sequence[Episode], optimizer) -> Dict[str, float]:
        losses, preds, targets = [], [], []
        for epoch in range(int(self.args.warmup.epochs)):
            # Report predictions only from the final pass; mixing pre- and post-update
            # values makes the warmup metric describe no actual critic checkpoint.
            if epoch == int(self.args.warmup.epochs) - 1:
                losses, preds, targets = [], [], []
            for batch in iter_minibatches(
                    episodes, np.zeros(len(episodes), np.float32),
                    int(self.args.train.minibatch_episodes), self.rng, self.device):
                out = self._forward(batch)
                target = batch.reward.unsqueeze(1).expand_as(out["vpred"])
                loss = F.binary_cross_entropy_with_logits(
                    out["vpred"].float(), target, reduction="none")
                loss = (loss * batch.mask).sum() / batch.mask.sum().clamp_min(1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_params, self.cfg.max_grad_norm)
                optimizer.step()
                if epoch == int(self.args.warmup.epochs) - 1:
                    real = batch.mask.bool()
                    losses.append(float(loss.detach()))
                    preds.extend(out["vpred"].detach().float().sigmoid()[real]
                                 .cpu().tolist())
                    targets.extend(target[real].cpu().tolist())
        p, y = np.asarray(preds), np.asarray(targets)
        return {"bce": float(np.mean(losses)), "brier": float(np.mean((p-y)**2)),
                "explained_variance": explained_variance(y, p)}

    @torch.no_grad()
    def _critic_metrics(self, episodes: Sequence[Episode]) -> Dict[str, float]:
        preds, targets = [], []
        for batch in iter_minibatches(
                episodes, np.zeros(len(episodes), np.float32),
                int(self.args.train.minibatch_episodes), self.rng, self.device):
            out = self._forward(batch)
            real = batch.mask.bool()
            preds.extend(out["vpred"].float().sigmoid()[real].cpu().tolist())
            targets.extend(batch.reward.unsqueeze(1).expand_as(out["vpred"])[real]
                           .cpu().tolist())
        p, y = np.asarray(preds), np.asarray(targets)
        return {"brier": float(np.mean((p-y)**2)),
                "explained_variance": explained_variance(y, p)}

    def warmup(self) -> None:
        if self.warmup_metrics:
            return
        started = time.time()
        for p in self.actor_params:
            p.requires_grad = False
        warm_opt = torch.optim.AdamW(self.critic_params,
                                     lr=float(self.args.train.critic_learning_rate))
        chunks = int(self.args.warmup.episodes) // int(self.args.warmup.chunk_episodes)
        train_success, last = [], {}
        for i in range(chunks):
            episodes = self._collect(int(self.args.warmup.chunk_episodes))
            train_success.extend(float(_won(e)) for e in episodes)
            last = self._warmup_step(episodes, warm_opt)
            print(f"[ppo] critic warmup {i+1}/{chunks}: success="
                  f"{np.mean(train_success):.3f} brier={last['brier']:.4f} "
                  f"EV={last['explained_variance']:.3f}", flush=True)
            del episodes
            self._check_memory()
        validation = self._collect(int(self.args.warmup.validation_episodes))
        val = self._critic_metrics(validation)
        p = float(np.mean(train_success))
        # Match the critic's timestep weighting: long episodes contribute more action
        # decisions to both PPO and Brier score than short episodes.
        y = np.concatenate([np.full(len(e), float(_won(e))) for e in validation])
        constant_brier = float(np.mean((p - y) ** 2))
        self.warmup_metrics = {
            "warmup/train_success": p,
            "warmup/train_brier": float(last["brier"]),
            "warmup/train_explained_variance": float(last["explained_variance"]),
            "warmup/val_brier": val["brier"],
            "warmup/val_explained_variance": val["explained_variance"],
            "warmup/constant_brier": constant_brier,
            "warmup/seconds": time.time() - started,
        }
        self.elapsed += self.warmup_metrics["warmup/seconds"]
        del validation
        for p_actor in self.actor_params:
            p_actor.requires_grad = True
        passed = val["brier"] < constant_brier and val["explained_variance"] > 0
        print(f"[ppo] critic gate: val_brier={val['brier']:.4f} < constant="
              f"{constant_brier:.4f}, EV={val['explained_variance']:.3f} -> "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
        self.save(tag="warmup")
        if not passed:
            raise RuntimeError("critic warmup failed its predeclared validation gate")

    # -- GAE and PPO update ---------------------------------------------

    @torch.no_grad()
    def _assign_values_and_gae(self, episodes: Sequence[Episode]) -> Dict[str, float]:
        all_adv = []
        for start in range(0, len(episodes), int(self.args.train.minibatch_episodes)):
            subset = episodes[start:start + int(self.args.train.minibatch_episodes)]
            batch = GroupBatch(subset, np.zeros(len(subset), np.float32), self.device)
            pred = self._forward(batch)["vpred"].float().sigmoid().cpu().numpy()
            for i, e in enumerate(subset):
                e.values = pred[i, :len(e)].astype(np.float32)
                raw, target = gae(e.reward, e.values, self.cfg.gamma,
                                  self.cfg.gae_lambda)
                e.advantages, e.value_targets = raw, target
                all_adv.append(raw)
        flat = np.concatenate(all_adv)
        mean, std = float(flat.mean()), float(flat.std())
        for e in episodes:
            e.advantages = ((e.advantages - mean) / max(std, 1e-6)).astype(np.float32)
        y = np.concatenate([np.full(len(e), e.reward, np.float32) for e in episodes])
        v = np.concatenate([e.values for e in episodes])
        return {"adv_mean_raw": mean, "adv_std_raw": std,
                "value_brier_pre": float(np.mean((v-y)**2)),
                "value_explained_variance_pre": explained_variance(y, v)}

    def train_on(self, episodes: Sequence[Episode]) -> Dict[str, float]:
        prep = self._assign_values_and_gae(episodes)
        agg: Dict[str, List[float]] = {}
        for _ in range(int(self.args.train.epochs)):
            for batch in iter_ppo_minibatches(
                    episodes, int(self.args.train.minibatch_episodes),
                    self.rng, self.device):
                out = self._forward(batch)
                with torch.no_grad():
                    ref_logits = self._forward(batch, self.ref)["pi_logits"]
                loss, metrics = ppo_loss(out["pi_logits"], out["vpred"], batch,
                                         self.cfg, ref_logits)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_gn = torch.sqrt(sum(
                    p.grad.detach().float().pow(2).sum() for p in self.actor_params
                    if p.grad is not None))
                critic_gn = torch.sqrt(sum(
                    p.grad.detach().float().pow(2).sum() for p in self.critic_params
                    if p.grad is not None))
                total_gn = torch.nn.utils.clip_grad_norm_(
                    self.actor_params + self.critic_params, self.cfg.max_grad_norm)
                self.optimizer.step()
                for k, v in metrics.items():
                    agg.setdefault(k, []).append(float(v))
                agg.setdefault("actor_grad_norm", []).append(float(actor_gn))
                agg.setdefault("critic_grad_norm", []).append(float(critic_gn))
                agg.setdefault("grad_norm", []).append(float(total_gn))
                if float(metrics["approx_kl"]) > self.cfg.target_kl:
                    agg.setdefault("early_stopped", []).append(1.0)
                    break
        return {**prep, **{k: float(np.mean(v)) for k, v in agg.items()}}

    # -- reporting and persistence -------------------------------------

    def outcome_stats(self, episodes) -> Dict[str, float]:
        wins = int(sum(_won(e) for e in episodes))
        lo, hi = wilson(wins, len(episodes))
        return {"success": wins / len(episodes), "episodes": float(len(episodes)),
                "ci_lo": lo, "ci_hi": hi,
                "mean_len": float(np.mean([len(e) for e in episodes]))}

    def probe(self) -> Dict[str, float]:
        episodes = self._collect(int(self.args.probe.repeats))
        wins = int(sum(_won(e) for e in episodes))
        lo, hi = wilson(wins, len(episodes))
        out = {"probe/success": wins / len(episodes),
               "probe/episodes": float(len(episodes)),
               "probe/ci_lo": lo, "probe/ci_hi": hi,
               "probe/damage_lost": float(np.mean(
                   [e.damage_frac for e in episodes if not _won(e)]))}
        del episodes
        return out

    def run(self) -> None:
        budget = float(self.args.train.max_hours) * 3600
        max_kl = float(self.args.train.max_kl_ref)
        kl_hist = collections.deque(maxlen=10)
        try:
            if self.update == 0:
                self.warmup()
            while self.update < int(self.args.train.updates):
                if budget and self.elapsed >= budget:
                    print("[ppo] wall-clock budget reached", flush=True)
                    break
                t0 = time.time()
                episodes = self._collect(int(self.args.rollout.episodes_per_update))
                outcomes = [EpisodeOutcome.of(e) for e in episodes]
                self._check_memory()
                metrics = self.train_on(episodes)
                self.update += 1
                row = {"update": self.update, **metrics,
                       **self.outcome_stats(outcomes)}
                if self.update == 1 or self.update % int(self.args.probe.every) == 0:
                    row.update(self.probe())
                row["seconds"] = time.time() - t0
                self.elapsed += row["seconds"]
                row["elapsed_hours"] = self.elapsed / 3600
                self.logger.log({**self.warmup_metrics, **row})
                print(f"[ppo {self.update}/{int(self.args.train.updates)}] "
                      f"success={row['success']:.3f} policy_loss="
                      f"{row['policy_loss']:.4f} value_loss={row['value_loss']:.4f} "
                      f"kl_ref={row['kl_ref']:.4f} EV="
                      f"{row['value_explained_variance_pre']:.3f} "
                      f"({row['seconds']:.0f}s)", flush=True)
                del episodes
                if self.update % int(self.args.train.save_every) == 0:
                    self.save()
                kl_hist.append(float(row.get("kl_ref", 0)))
                if max_kl and len(kl_hist) == 10 and np.mean(kl_hist) > max_kl:
                    print(f"[ppo] reference KL guard reached: {np.mean(kl_hist):.4f}",
                          flush=True)
                    break
        finally:
            self.save(final=True)
            self.collector.close()

    def save(self, final: bool = False, tag: Optional[str] = None) -> str:
        label = tag or ("final" if final else f"{self.update:06d}")
        path = os.path.join(self.run_dir, "checkpoints", f"ppo-{label}.pt")
        self.policy.save(
            path, update=self.update,
            train_config=OmegaConf.to_container(self.args, resolve=True),
            optimizer=self.optimizer.state_dict(), warmup_metrics=self.warmup_metrics,
            actor_rng=self.collector.actor.state(), trainer_rng=self.rng.bit_generator.state,
            elapsed_seconds=self.elapsed, python_rng=random.getstate(),
            numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        print(f"[ppo] saved {path}", flush=True)
        return path

    def _resume(self, path: Optional[str]) -> None:
        if not path:
            return
        ckpt = torch.load(os.path.expanduser(path), map_location="cpu", weights_only=False)
        required = {"policy", "optimizer", "update", "train_config", "warmup_metrics",
                    "actor_rng", "trainer_rng", "python_rng", "numpy_rng", "torch_rng",
                    "cuda_rng"}
        missing = required - set(ckpt)
        if missing:
            raise ValueError(f"incomplete PPO resume checkpoint: missing {sorted(missing)}")
        previous = os.path.abspath(os.path.expanduser(ckpt["train_config"]["init_from"]))
        current = os.path.abspath(os.path.expanduser(str(self.args.init_from)))
        if previous != current:
            raise ValueError("resume_from cannot change the frozen reference init_from")
        self.policy.load_state_dict(ckpt["policy"], strict=True)
        self.policy.eval()
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update = int(ckpt["update"])
        self.elapsed = float(ckpt.get("elapsed_seconds", 0))
        self.warmup_metrics = dict(ckpt["warmup_metrics"])
        self.collector.actor.load_state(ckpt["actor_rng"])
        self.rng.bit_generator.state = ckpt["trainer_rng"]
        random.setstate(ckpt["python_rng"])
        np.random.set_state(ckpt["numpy_rng"])
        torch.set_rng_state(ckpt["torch_rng"].cpu().to(torch.uint8))
        if ckpt["cuda_rng"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        print(f"[ppo] resumed {path} at update {self.update}", flush=True)


@hydra.main(config_path=".", config_name="laser_ppo", version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _install_signal_handlers()
    _seed(int(args.seed))
    run_dir = os.getcwd()
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as fh:
        fh.write(OmegaConf.to_yaml(args, resolve=True))
    PPOTrainer(args, run_dir).run()


def _install_signal_handlers() -> None:
    """Turn termination into an exception so the run's ``finally`` saves state."""
    def _exit(signum, _frame):
        print(f"\n[ppo] {signal.Signals(signum).name} — saving and stopping", flush=True)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


if __name__ == "__main__":
    main()
