"""The RL loop: warm the critic, collect, update, log, checkpoint, resume.

One *update* is: collect a batch of complete episodes, compute undiscounted returns
and advantages over each whole episode, then run ``ppo_epochs`` passes of clipped PPO
over that batch — shuffling **episodes**, never transitions.

Recurrent optimisation, and why there is no stored memory
---------------------------------------------------------
Each minibatch is a handful of whole episodes replayed **in order**, in chunks of
``seq_len`` steps, with the recurrent memory carried from chunk to chunk and detached
between them. The initial memory of every optimised sequence is therefore not an
approximation and not a stale copy saved during collection: chunk 0 starts from the
model's own ``initial_state`` (exactly what the collector reset to at the episode
boundary) and every later chunk's memory is *recomputed under the current parameters*,
which is strictly better than replaying the behaviour policy's saved state.

The detach between chunks is truncated BPTT: gradient flows across the ``seq_len``
steps inside a chunk but not backwards past its boundary. It is also the only place
memory is ever detached — never inside a chunk, and never in a way that clears it,
which would silently turn a 400-step boss episode into thirteen unrelated 32-step
ones.

The context each token sees is identical in both regimes. The recurrent core keeps
``mem_len`` timesteps of keys and values behind a clipped-causal mask, so a step at
rollout (``T=1`` with carried memory) and the same step during optimisation (inside a
``T=32`` chunk with carried memory) both attend to exactly the preceding 32 decisions.

``eval()`` throughout
---------------------
The model stays in eval mode during optimisation as well as collection. The resampler
carries dropout, and with it on, the log-probability recomputed for a stored action
would differ from the one the action was sampled under for a reason that has nothing
to do with the parameter update — every PPO ratio would then carry dropout noise.
Regularisation in this phase comes from the trust region, not from dropout.
"""

from __future__ import annotations

import collections
import csv
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from contra_policy.model import CrossViewContraRocket
from contra_policy.rl import checkpoint as ckpt_io
from contra_policy.rl.ppo import PPOConfig, PPOObjective, explained_variance
from contra_policy.rl.rollout import EpisodeCollector
from contra_policy.rl.tasks import TaskCatalog, TaskSampler
from contra_policy.rl.trajectory import (Episode, build_chunk, compute_returns,
                                         iter_chunks, iter_minibatches,
                                         normalize_advantages, rollout_stats)


class CSVMetricLogger:
    """Append-only CSV with a growing header — no schema to declare up front.

    Per-label keys appear and disappear with the sampling draw, so the header is
    rewritten (and earlier rows re-padded) whenever a new key shows up. That costs a
    file rewrite a handful of times at the start of a run and never again.
    """

    def __init__(self, path: str):
        self.path = path
        self.rows: List[Dict[str, float]] = []
        self.keys: List[str] = []

    def log(self, row: Dict[str, float]) -> None:
        self.rows.append(dict(row))
        new = [k for k in row if k not in self.keys]
        self.keys.extend(new)
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=self.keys, restval="")
            w.writeheader()
            w.writerows(self.rows)


class RLTrainer:

    def __init__(self, args, run_dir: str = "."):
        self.args = args
        self.run_dir = run_dir
        self.device = torch.device(args.device if torch.cuda.is_available()
                                   or not str(args.device).startswith("cuda") else "cpu")
        self.config = _resolved(args)

        # -- model ---------------------------------------------------------
        model_cfg = (ckpt_io.model_config_from_checkpoint(args.init_from)
                     if args.init_from else
                     dict(_resolved(args.model) if "model" in args else {}))
        if not model_cfg:
            raise ValueError("either init_from (a BC checkpoint) or a model block is "
                             "required to build the policy")
        self.model_cfg = model_cfg
        # The recorded config carries the architecture the weights actually have, so
        # the evaluator rebuilds exactly this model from a weights-only checkpoint.
        self.config["model"] = dict(model_cfg)

        self.model = CrossViewContraRocket(**model_cfg).to(self.device)
        if args.init_from:
            ckpt_io.load_policy_weights(self.model, args.init_from)
            print(f"[rl] initialised from {args.init_from}")
        self.model.eval()

        self.ref_model: Optional[CrossViewContraRocket] = None
        if float(args.ppo.bc_kl_coef) > 0.0:
            # A frozen copy of the initialisation, kept only to say what the BC policy
            # would have done. It never receives a gradient and never updates.
            self.ref_model = CrossViewContraRocket(**model_cfg).to(self.device).eval()
            self.ref_model.load_state_dict(self.model.state_dict())
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
            print("[rl] frozen BC reference policy enabled "
                  f"(bc_kl_coef={float(args.ppo.bc_kl_coef)})")

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in self.params)
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        print(f"[rl] policy: {trainable/1e6:.1f}M trainable + {frozen/1e6:.1f}M frozen · "
              f"{self.model.num_step_tokens} tokens/timestep")

        # -- optimisation --------------------------------------------------
        self.ppo_cfg = PPOConfig(**_resolved(args.ppo))
        self.objective = PPOObjective(self.ppo_cfg)
        self.optimizer = torch.optim.AdamW(self.params, lr=self.ppo_cfg.learning_rate,
                                           weight_decay=float(args.weight_decay))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_scale)
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[args.precision]
        if self.device.type != "cuda":
            self.autocast_dtype = None

        # -- tasks and collection -------------------------------------------
        self.catalog = TaskCatalog(
            task_root=args.task_root, shard_dir=args.shard_dir,
            families=list(args.families), split="train",
            image_size=int(model_cfg["image_size"]), sigma_px=float(args.sigma_px),
            cache_dir=args.cache_dir, prompt_cache=int(args.rollout.prompt_cache),
            segment_cache=int(args.rollout.segment_cache))
        self.catalog.assert_split("train")
        self.sampler = TaskSampler(self.catalog, float(args.sampling.natural_fraction),
                                   float(args.sampling.balanced_family_fraction),
                                   _resolved(args.sampling.family_multiplier),
                                   seed=int(args.seed))
        mix = self.sampler.expected_family_mix()
        print("[rl] episode-start mix: "
              + " · ".join(f"{f} {mix[f]:.1%}" for f in sorted(mix)))

        if float(args.rollout.temperature) != 1.0:
            # The PPO objective evaluates the untempered policy, so a tempered sampler
            # would make every ratio a ratio between two different distributions.
            # Explore with `ppo.entropy_coef`, not with temperature.
            raise ValueError(
                f"rollout.temperature must be 1.0 for PPO, got "
                f"{float(args.rollout.temperature)}; temperature is a replay knob "
                f"(0 = greedy) and belongs to evaluation, not to training")

        self.collector = None
        self.mp_collector = None
        self.rng = np.random.default_rng(int(args.seed))
        preflight_host_memory(args, int(model_cfg["image_size"]))
        self._build_collector()

        self.bc_loader = None
        self.bc_iter = None
        if int(args.bc_mix.batches_per_minibatch) > 0:
            self._build_bc_loader()

        # -- bookkeeping -----------------------------------------------------
        self.counters = {"update": 0, "episodes": 0, "steps": 0, "rollouts": 0}
        self.logger = CSVMetricLogger(os.path.join(run_dir, "metrics.csv"))
        os.makedirs(os.path.join(run_dir, "weights"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.wandb = None
        if args.logger == "wandb":
            import wandb

            self.wandb = wandb.init(project=args.project, config=self.config)

        if args.resume_from:
            self.resume(args.resume_from)

    # -- construction helpers ---------------------------------------------

    def _collector_spec(self) -> Dict:
        a = self.args
        return {
            "task_root": a.task_root, "shard_dir": a.shard_dir,
            "cache_dir": a.cache_dir, "families": list(a.families),
            "image_size": int(self.model_cfg["image_size"]), "sigma_px": float(a.sigma_px),
            "prompt_cache": int(a.rollout.prompt_cache),
            "segment_cache": int(a.rollout.segment_cache),
            "batch_size": int(a.rollout.batch_size),
            "budget_mult": float(a.rollout.budget_mult),
            "min_budget": int(a.rollout.min_budget),
            "temperature": float(a.rollout.temperature),
            "precision": a.precision, "seed": int(a.seed),
            "reward": _resolved(a.reward),
            "max_episode_steps": int(a.rollout.max_episode_steps),
            "collect_goal_points": bool(a.rollout.collect_goal_points),
            "natural_fraction": float(a.sampling.natural_fraction),
            "balanced_family_fraction": float(a.sampling.balanced_family_fraction),
            "family_multiplier": _resolved(a.sampling.family_multiplier),
            "worker_device": a.rollout.worker_device,
        }

    def _build_collector(self) -> None:
        a = self.args
        if int(a.rollout.num_workers) > 0:
            from contra_policy.rl.workers import MultiProcessCollector

            self.mp_collector = MultiProcessCollector(
                self.model, self.model_cfg, self._collector_spec(),
                int(a.rollout.num_workers))
            print(f"[rl] {int(a.rollout.num_workers)} collector worker(s), "
                  f"one emulator each")
            return
        self.collector = EpisodeCollector(
            self.model, self.catalog, self.sampler,
            batch_size=int(a.rollout.batch_size), budget_mult=float(a.rollout.budget_mult),
            min_budget=int(a.rollout.min_budget),
            image_size=int(self.model_cfg["image_size"]), device=self.device,
            temperature=float(a.rollout.temperature), precision=a.precision,
            seed=int(a.seed), reward=_resolved(a.reward),
            max_episode_steps=int(a.rollout.max_episode_steps),
            collect_goal_points=bool(a.rollout.collect_goal_points),
            owner="RLTrainer")

    def _build_bc_loader(self) -> None:
        """Optional BC minibatches mixed into PPO updates, from the existing datamodule."""
        from contra_policy.dataset import ContraDataModule

        a = self.args
        data = ContraDataModule(
            shard_dir=a.shard_dir, configs=list(a.families),
            win_len=int(self.ppo_cfg.seq_len), image_size=int(self.model_cfg["image_size"]),
            sigma_px=float(a.sigma_px), prev_action_keep_prob=0.0,
            aux_size=int(self.model_cfg["aux_size"]), batch_size=int(a.bc_mix.batch_size),
            num_workers=int(a.bc_mix.num_workers), cache_dir=a.cache_dir, seed=int(a.seed))
        self.bc_loader = data.train_dataloader()
        self.bc_iter = iter(self.bc_loader)
        from contra_policy.loss import ContraObjective

        self.bc_objective = ContraObjective(
            bc_weight=1.0, heatmap_weight=float(a.bc_mix.heatmap_weight),
            heatmap_pos_weight=self.ppo_cfg.heatmap_pos_weight).to(self.device)
        print(f"[rl] mixing {int(a.bc_mix.batches_per_minibatch)} BC batch(es) into "
              f"each PPO minibatch (weight {float(a.bc_mix.weight)})")

    def _lr_scale(self, step: int) -> float:
        warmup = int(self.args.train.lr_warmup_updates)
        scale = 1.0 if warmup <= 0 else min(1.0, (step + 1) / warmup)
        if self.args.train.lr_decay == "linear":
            total = max(1, int(self.args.train.updates))
            scale *= max(0.0, 1.0 - step / total)
        return scale

    # -- collection ---------------------------------------------------------

    def collect(self, min_steps: int, min_episodes: int) -> List[Episode]:
        t0 = time.time()
        if self.mp_collector is not None:
            episodes = self.mp_collector.collect(min_steps, min_episodes)
        else:
            episodes = self.collector.collect(min_steps, min_episodes)
        steps = sum(len(e) for e in episodes)
        dt = max(1e-9, time.time() - t0)
        self.counters["rollouts"] += 1
        self.counters["episodes"] += len(episodes)
        self.counters["steps"] += steps
        self._last_collect = {
            "collect_s": dt, "collect_steps_per_s": steps / dt,
            "collect_episodes_per_s": len(episodes) / dt,
            "rollout_mb": sum(e.nbytes() for e in episodes) / 1e6,
        }
        return episodes

    # -- optimisation -------------------------------------------------------

    @torch.no_grad()
    def refresh_behaviour_stats(self, episodes: Sequence[Episode]) -> float:
        """Recompute stored log-probs and values in the *optimisation* shape.

        Collection runs ``T=1`` forwards over 16 emulator slots; optimisation runs
        ``T=seq_len`` forwards over a handful of episodes. The parameters are identical
        and, because the clipped-causal memory gives both the same context window, so
        is the distribution — but not bit for bit under bf16, and a ratio that starts
        at 1.02 instead of 1.00 eats a fifth of a ``clip_ratio`` of 0.1 before the
        first gradient step.

        Rewriting the references here makes the first epoch's ratio exactly 1 and
        removes the mismatch entirely. Returns the mean absolute log-ratio it
        corrected, which is worth watching: if it is large, something more than
        numerics differs between the two paths.
        """
        drift, n = 0.0, 0
        for start in range(0, len(episodes), self.ppo_cfg.minibatch_episodes):
            mb = list(episodes[start:start + self.ppo_cfg.minibatch_episodes])
            memory = None
            for ci, (lo, hi) in enumerate(iter_chunks(mb, self.ppo_cfg.seq_len)):
                batch = build_chunk(mb, lo, hi, device=self.device, first=ci == 0)
                latents, memory = self._forward(self.model, batch["model"], memory)
                memory = [m.detach() for m in memory]
                logits = latents["pi_logits"].float()
                logprob = torch.log_softmax(logits, -1).gather(
                    -1, batch["action"].unsqueeze(-1)).squeeze(-1).cpu().numpy()
                value = latents["vpred"].squeeze(-1).float().cpu().numpy()
                for i, ep in enumerate(mb):
                    n_valid = max(0, min(hi, len(ep)) - lo)
                    if n_valid <= 0:
                        continue
                    sl = slice(lo, lo + n_valid)
                    drift += float(np.abs(ep.logprobs[sl] - logprob[i, :n_valid]).sum())
                    n += n_valid
                    ep.logprobs[sl] = logprob[i, :n_valid]
                    ep.values[sl] = value[i, :n_valid]
        return drift / max(1, n)

    def _forward(self, model, model_input: Dict, memory):
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null_context())
        with ctx:
            return model(model_input, memory)

    def update(self, episodes: List[Episode]) -> Dict[str, float]:
        cfg = self.ppo_cfg
        metrics: Dict[str, float] = {}

        if bool(self.args.train.recompute_old_logprobs):
            metrics["behaviour_logprob_drift"] = self.refresh_behaviour_stats(episodes)

        compute_returns(episodes, cfg.gamma, cfg.gae_lambda)
        metrics["explained_variance"] = explained_variance(
            np.concatenate([e.values for e in episodes]),
            np.concatenate([e.returns for e in episodes]))
        if cfg.normalize_advantages:
            mean, std = normalize_advantages(episodes)
            metrics["advantage_mean"], metrics["advantage_std"] = mean, std

        aux_size = int(self.model_cfg["aux_size"]) if cfg.heatmap_coef > 0 else 0
        acc: Dict[str, float] = collections.defaultdict(float)
        total_count = 0.0
        n_minibatches = 0
        stopped = False

        for epoch in range(cfg.ppo_epochs):
            for mb in iter_minibatches(episodes, cfg.minibatch_episodes, self.rng):
                valid = float(sum(len(e) for e in mb))
                if valid == 0:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                memory = ref_memory = None
                mb_acc: Dict[str, float] = collections.defaultdict(float)
                for ci, (lo, hi) in enumerate(iter_chunks(mb, cfg.seq_len)):
                    batch = build_chunk(mb, lo, hi, device=self.device,
                                        aux_size=aux_size, sigma_px=float(self.args.sigma_px),
                                        first=ci == 0)
                    latents, memory = self._forward(self.model, batch["model"], memory)
                    ref_logits = None
                    if self.ref_model is not None:
                        with torch.no_grad():
                            ref_latents, ref_memory = self._forward(
                                self.ref_model, batch["model"], ref_memory)
                        ref_logits = ref_latents["pi_logits"]
                        ref_memory = [m.detach() for m in ref_memory]
                    loss_sum, chunk_metrics, count = self.objective(
                        latents, batch, ref_logits)
                    if float(count) > 0:
                        (loss_sum / valid).backward()
                    # Truncated BPTT: the memory continues the episode, the graph does
                    # not. Detaching here — and only here — is what keeps a long
                    # episode one trajectory rather than a chain of unrelated windows.
                    memory = [m.detach() for m in memory]
                    for k, v in chunk_metrics.items():
                        mb_acc[k] += float(v)
                    mb_acc["_count"] += float(count)

                if self.bc_iter is not None:
                    mb_acc["bc_mix_loss"] += self._bc_mix_backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.optimizer.step()
                n_minibatches += 1

                c = max(1.0, mb_acc.pop("_count"))
                total_count += c
                for k, v in mb_acc.items():
                    acc[k] += v
                acc["grad_norm"] += float(grad_norm) * c
                acc["_count"] += c

                if cfg.target_kl > 0 and (mb_acc["approx_kl"] / c) > cfg.target_kl:
                    # Early termination of the *update*, not of training: the batch has
                    # already moved the policy as far as the trust region allows, and
                    # further epochs over the same data would move it further off the
                    # distribution the data was collected under.
                    stopped = True
                    break
            if stopped:
                break

        c = max(1.0, acc.pop("_count"))
        for k, v in acc.items():
            metrics[k] = v / c
        metrics["ppo_epochs_run"] = (epoch + (0 if stopped else 1))
        metrics["ppo_minibatches"] = float(n_minibatches)
        metrics["kl_early_stop"] = float(stopped)
        metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        self.scheduler.step()
        return metrics

    def _bc_mix_backward(self) -> float:
        """One or more BC minibatches, accumulated into the current PPO gradient."""
        from contra_policy.lit import _to_model_input

        total = 0.0
        weight = float(self.args.bc_mix.weight)
        for _ in range(int(self.args.bc_mix.batches_per_minibatch)):
            try:
                batch = next(self.bc_iter)
            except StopIteration:
                self.bc_iter = iter(self.bc_loader)
                batch = next(self.bc_iter)
            batch = {k: (v.to(self.device) if torch.is_tensor(v)
                         else {kk: vv.to(self.device) for kk, vv in v.items()})
                     for k, v in batch.items()}
            model_input = _to_model_input(batch)
            # prev_action_keep_prob is 0 for this loader, so the dropout mask it ships
            # is already all-zero — the same "unknown" branch the rollout uses.
            latents, _ = self._forward(self.model, model_input, None)
            loss, _m = self.bc_objective(latents, batch)
            (weight * loss).backward()
            total += float(loss.detach())
        return total

    # -- critic warmup ------------------------------------------------------

    def critic_warmup(self) -> None:
        """Fit the value head on fixed-policy episodes before any policy update.

        The value head received **no gradient during behaviour cloning** — the BC
        objective never reads ``vpred`` — so at update 0 the critic is an untrained
        linear readout of the trunk. Every advantage would then be
        ``return - noise``, and the first policy gradient would be driven by that
        noise. Warmup collects with the policy frozen and fits only the critic, so PPO
        starts from advantages that mean something.

        ``params: value_head`` (the default) trains the ``Linear(hiddim, 1)`` alone,
        which cannot damage the BC trunk it reads from. ``params: all`` lets the value
        loss reshape the trunk too, which is faster to fit and riskier.
        """
        rounds = int(self.args.critic_warmup.rounds)
        if rounds <= 0:
            return
        which = str(self.args.critic_warmup.params)
        params = (list(self.model.value_head.parameters()) if which == "value_head"
                  else self.params)
        if which not in ("value_head", "all"):
            raise ValueError(f"critic_warmup.params must be 'value_head' or 'all', "
                             f"got {which!r}")
        opt = torch.optim.AdamW(params, lr=float(self.args.critic_warmup.learning_rate))
        print(f"[rl] critic warmup: {rounds} round(s), optimising {which}")
        for r in range(rounds):
            episodes = self.collect(int(self.args.critic_warmup.steps),
                                    int(self.args.critic_warmup.episodes))
            compute_returns(episodes, self.ppo_cfg.gamma, self.ppo_cfg.gae_lambda)
            stats = rollout_stats(episodes)
            losses = []
            for _epoch in range(int(self.args.critic_warmup.epochs)):
                for mb in iter_minibatches(episodes, self.ppo_cfg.minibatch_episodes,
                                           self.rng):
                    valid = float(sum(len(e) for e in mb))
                    if valid == 0:
                        continue
                    opt.zero_grad(set_to_none=True)
                    memory = None
                    for ci, (lo, hi) in enumerate(iter_chunks(mb, self.ppo_cfg.seq_len)):
                        batch = build_chunk(mb, lo, hi, device=self.device, first=ci == 0)
                        latents, memory = self._forward(self.model, batch["model"], memory)
                        memory = [m.detach() for m in memory]
                        vpred = latents["vpred"].squeeze(-1).float()
                        loss = 0.5 * (((vpred - batch["returns"]) ** 2)
                                      * batch["mask"]).sum() / valid
                        loss.backward()
                        losses.append(float(loss.detach()))
                    torch.nn.utils.clip_grad_norm_(params, self.ppo_cfg.max_grad_norm)
                    opt.step()
            ev = explained_variance(np.concatenate([e.values for e in episodes]),
                                    np.concatenate([e.returns for e in episodes]))
            row = {"phase": "critic_warmup", "warmup_round": r,
                   "value_loss": float(np.mean(losses)) if losses else float("nan"),
                   "explained_variance_before": ev,
                   **{f"rollout/{k}": v for k, v in stats.items()},
                   **self._last_collect}
            self._emit(row, prefix=f"[warmup {r+1}/{rounds}]")

    # -- the loop -----------------------------------------------------------

    def run(self) -> None:
        try:
            if self.counters["update"] == 0:
                self.critic_warmup()
            total = int(self.args.train.updates)
            while self.counters["update"] < total:
                update = self.counters["update"]
                episodes = self.collect(int(self.args.rollout.steps),
                                        int(self.args.rollout.episodes))
                if not episodes:
                    raise RuntimeError("the collector returned no episodes")
                stats = rollout_stats(episodes)
                metrics = self.update(episodes)
                self.counters["update"] = update + 1

                row = {"phase": "ppo", "update": self.counters["update"],
                       "total_episodes": self.counters["episodes"],
                       "total_steps": self.counters["steps"],
                       **{f"rollout/{k}": v for k, v in stats.items()},
                       **{f"ppo/{k}": v for k, v in metrics.items()},
                       **self._last_collect, **_resource_usage()}
                self._emit(row, prefix=f"[update {self.counters['update']}/{total}]")
                self._check_finite(metrics)

                every = int(self.args.train.save_every)
                if every > 0 and self.counters["update"] % every == 0:
                    self.save()
        finally:
            self.save(final=True)
            self.close()

    def _check_finite(self, metrics: Dict[str, float]) -> None:
        bad = [k for k, v in metrics.items()
               if isinstance(v, float) and not np.isfinite(v) and k != "explained_variance"]
        if bad:
            raise FloatingPointError(
                f"non-finite PPO metrics at update {self.counters['update']}: {bad}")

    # -- output -------------------------------------------------------------

    def _emit(self, row: Dict, prefix: str) -> None:
        self.logger.log({k: v for k, v in row.items() if not isinstance(v, str)}
                        | {"phase": row.get("phase", "")})
        if self.wandb is not None:
            self.wandb.log({k: v for k, v in row.items() if not isinstance(v, str)})
        keys = ["rollout/completion", "rollout/macro_completion", "rollout/death",
                "rollout/timeout", "rollout/episodes", "rollout/mean_episode_len",
                "ppo/policy_loss", "ppo/value_loss", "ppo/entropy", "ppo/approx_kl",
                "ppo/clip_frac", "ppo/explained_variance", "value_loss",
                "collect_steps_per_s"]
        parts = [f"{k.split('/')[-1]}={row[k]:.4g}" for k in keys if k in row]
        print(f"{prefix} " + " ".join(parts), flush=True)
        # Completion only. The episode and step counts behind each rate stay in
        # metrics.csv, where `tools/rl_progress.py` pools them across updates — which
        # is the only shape they are readable in anyway, since one update gives boss
        # about four episodes.
        fam = [f"{f}({_trim(row[f'rollout/{f}/completion'])})"
               for f in ("kill", "item", "traverse", "boss")
               if f"rollout/{f}/completion" in row]
        if fam:
            print("    families: " + " ".join(fam), flush=True)

    def save(self, final: bool = False) -> Dict[str, str]:
        tag = "final" if final else f"{self.counters['update']:06d}"
        weights = ckpt_io.save_weights_only(
            os.path.join(self.run_dir, "weights", f"weight-update={tag}.ckpt"),
            self.model, self.config)
        resumable = ckpt_io.save_resumable(
            os.path.join(self.run_dir, "checkpoints", f"rl-{tag}.pt"),
            model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            counters=self.counters, config=self.config,
            sampler=self.sampler,
            actor=self.collector.actor if self.collector is not None else None)
        print(f"[rl] saved {weights} and {resumable}", flush=True)
        return {"weights": weights, "checkpoint": resumable}

    def resume(self, path: str) -> None:
        counters = ckpt_io.load_resumable(
            path, model=self.model, optimizer=self.optimizer, scheduler=self.scheduler,
            sampler=self.sampler,
            actor=self.collector.actor if self.collector is not None else None,
            map_location=str(self.device))
        self.counters.update(counters)
        self.model.to(self.device).eval()
        print(f"[rl] resumed {path} at update {self.counters['update']} "
              f"({self.counters['steps']} steps collected so far)")

    def close(self) -> None:
        if self.collector is not None:
            self.collector.close()
        if self.mp_collector is not None:
            self.mp_collector.close()
        self.catalog.close()
        if self.wandb is not None:
            self.wandb.finish()


# ── helpers ───────────────────────────────────────────────────────────────────

class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _resolved(node):
    from omegaconf import DictConfig, OmegaConf

    if isinstance(node, DictConfig):
        return OmegaConf.to_container(node, resolve=True)
    return dict(node) if node is not None else {}


def _trim(value: float) -> str:
    """Two decimals, minus a trailing zero: 0.7, 1.0, 0.89, 0.0.

    Keeping one decimal on the round numbers is what makes the family list read as a
    column of rates rather than a mix of integers and fractions.
    """
    s = f"{float(value):.2f}"
    return s[:-1] if s.endswith("0") else s


# ── host memory preflight ─────────────────────────────────────────────────────
# Calibrated on this box (WSL2, `memory=20GB`, 16 GB GPU) by running the two-update
# smoke under `tools/rss_guard.py` and reading peak group PSS:
#
#     num_workers=0, steps=2048, batch_size=16   ->  3.54 GB
#     num_workers=2, steps=2048, batch_size=16   ->  8.83 GB
#
# which gives a ~2.9 GB parent, ~2.4 GB per worker, and a frame buffer that scales
# with `rollout.steps`. Re-measure these if the model, the image size or the machine
# changes; they are a fit to two points, not a derivation.
_PARENT_BASE_GB = 2.9      # torch + CUDA context + policy + optimiser + task catalog
_WORKER_BASE_GB = 2.4      # torch + CUDA context + policy replica + emulator + caches
# Collection stops once BOTH `steps` and `episodes` targets are met and then drains
# what is in flight, so the batch overshoots its step target. Measured 3.1k steps
# against a 2048 target at batch_size=16.
_OVERSHOOT = 1.55


def estimate_peak_host_gb(args, image_size: int) -> float:
    """Projected peak host RAM for this configuration, in GB.

    The frame buffer is counted once in the parent and once more across the workers
    when there are any: a worker holds its episodes until it puts them on the queue,
    the queue pickles them, and the parent holds the assembled batch.
    """
    workers = int(args.rollout.num_workers)
    frame_bytes = image_size * image_size * 3
    steps = max(int(args.rollout.steps), int(args.critic_warmup.steps))
    frames_gb = _OVERSHOOT * steps * frame_bytes / 1e9
    est = _PARENT_BASE_GB + workers * _WORKER_BASE_GB
    est += frames_gb * (2.0 if workers > 0 else 1.0)
    if float(args.ppo.bc_kl_coef) > 0:
        est += 0.5             # a second frozen policy, host side
    return est


def preflight_host_memory(args, image_size: int) -> float:
    """Refuse to start a run that will not fit in host RAM.

    This exists because the failure it prevents is not a crash. A run that exceeds
    the VM's memory does not get OOM-killed — it swaps, and a swapping WSL2 VM takes
    the editor, the terminal and everything else down with it, with no traceback and
    nothing saved. Checking a multiplication up front is cheap; the alternative costs
    a reboot.
    """
    budget = float(args.host_ram_budget_gb)
    est = estimate_peak_host_gb(args, image_size)
    workers = int(args.rollout.num_workers)
    free = _system_available_gb()

    print(f"[rl] host-RAM preflight: ~{est:.1f} GB projected "
          f"({workers} worker(s), {int(args.rollout.steps)} steps/update) · "
          f"budget {budget:.1f} GB · {free:.1f} GB available now", flush=True)

    if budget <= 0:                                          # explicitly disabled
        return est
    if est > budget:
        raise MemoryError(
            f"this configuration projects ~{est:.1f} GB of host RAM, over the "
            f"{budget:.1f} GB in `host_ram_budget_gb`. Going over does not raise — it "
            f"swaps, and the VM stops responding. Lower `rollout.num_workers` "
            f"(~{_WORKER_BASE_GB} GB each) or `rollout.steps`, or raise the budget if "
            f"the machine really has the room.")
    # Reserve headroom on top of the projection rather than spending every free byte.
    # The other tenants of this VM are an editor server and an agent, and they grow
    # over a session — measured drifting from 3.4 GB to 8 GB inside a few hours. A run
    # that exactly fits at startup is a run that swaps by hour three.
    reserve = 2.0
    if free > 0 and est > free - reserve:
        raise MemoryError(
            f"this configuration projects ~{est:.1f} GB but only {free:.1f} GB is "
            f"available right now, and {reserve:.0f} GB is reserved for the rest of "
            f"the machine (an editor server grows over a long run). Free something "
            f"up, or lower `rollout.num_workers`/`rollout.steps`.")
    return est


def _system_available_gb() -> float:
    """``MemAvailable`` — what can be had without swapping. 0 if unknown."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1e6
    except OSError:                                          # pragma: no cover
        pass
    return 0.0


def _resource_usage() -> Dict[str, float]:
    """GPU peak allocation and host RSS — the two numbers that end a run early."""
    out: Dict[str, float] = {}
    if torch.cuda.is_available():
        out["gpu_alloc_gb"] = torch.cuda.memory_allocated() / 1e9
        out["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    out["host_rss_gb"] = float(line.split()[1]) / 1e6
                    break
    except OSError:                                          # pragma: no cover
        pass
    return out
