"""The GRPO loop: collect groups, drop the ones that agree, update, measure.

Owns neither generation nor the objective — `rollout` produces episodes, `buffer` turns
them into advantages, `grpo` scores them. This file is the schedule and the bookkeeping.

**Group filtering, and why it is on by default.** GRPO's advantage is
``(r_i − mean(r_group)) / std(r_group)``, so a group whose G rollouts all succeed *or*
all fail has zero advantage everywhere and moves the policy not at all. Keeping such
groups does not merely waste their rollouts — it rescales the update, because the mean
over a batch that is 70% zeros is 0.3x the mean over the survivors. The effective step
size would then track how hard the current task mix happens to be. Filtering makes it
mean the same thing every update.

``collect_filtered`` goes further and *oversamples*: it keeps drawing groups until
``groups_per_update`` have survived, so the batch size is constant rather than a
function of the day's difficulty. That is dynamic sampling, and it is capped — a task
mix where nothing survives should stop the run, not spin.
"""

from __future__ import annotations

import csv
import math
import os
import signal
import time
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from contra_policy.model import PolicyConfig, build_policy, load_policy
from contra_policy.rl.buffer import (filter_groups, group_advantages, iter_minibatches)
from contra_policy.rl.grpo import GRPOConfig, grpo_loss
from contra_policy.rl.rollout import EpisodeCollector
from contra_policy.rl.tasks import GroupSampler, TaskCatalog, TaskSampler

FAMILIES = ("kill", "item", "traverse", "boss")


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval — the one that behaves at 0/n and n/n, which per-family
    counts hit constantly (boss succeeds ~3.5% of the time)."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class CSVLogger:
    def __init__(self, path: str):
        self.path, self.keys = path, []

    def log(self, row: Dict[str, float]) -> None:
        new = [k for k in row if k not in self.keys]
        if new:
            old = self._read()
            self.keys += new
            with open(self.path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=self.keys, restval="")
                w.writeheader()
                w.writerows(old)
        with open(self.path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=self.keys, restval="").writerow(row)

    def _read(self) -> List[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return list(csv.DictReader(fh))


class GRPOTrainer:
    def __init__(self, args: DictConfig, run_dir: str):
        self.args, self.run_dir = args, run_dir
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.autocast_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                               "fp32": None}[args.precision]
        if self.device.type != "cuda":
            self.autocast_dtype = None

        # -- policy, and a frozen copy of where it started ---------------------
        self.policy = load_policy(args.init_from, map_location="cpu").to(self.device)
        self.cfg = GRPOConfig(**OmegaConf.to_container(args.grpo, resolve=True))
        self.ref = None
        if self.cfg.kl_coef > 0:
            # Not optional on measured grounds: the previous PPO run left this at zero
            # and `item` regressed 76.5% -> 71.1% while boss improved. RL on a verifiable
            # reward will trade away a family the reward does not mention.
            self.ref = load_policy(args.init_from, map_location="cpu").to(self.device)
            for p in self.ref.parameters():
                p.requires_grad = False
            self.ref.eval()

        n = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"[grpo] {n/1e6:.2f}M trainable · G={int(args.rollout.group_size)} · "
              f"reference KL {'on' if self.ref else 'OFF'} · "
              f"filtering {'on' if args.rollout.filter_groups else 'OFF'}", flush=True)

        # -- data ------------------------------------------------------------
        self.catalog = TaskCatalog(
            task_root=args.task_root, shard_dir=args.shard_dir,
            families=list(args.families), split="train",
            image_size=int(args.image_size), sigma_px=float(args.sigma_px),
            cache_dir=args.cache_dir)
        self.catalog.assert_split("train")
        sampler = TaskSampler(self.catalog, float(args.sampling.natural_fraction),
                              float(args.sampling.balanced_family_fraction),
                              OmegaConf.to_container(args.sampling.family_multiplier,
                                                     resolve=True),
                              seed=int(args.seed))
        self.groups = GroupSampler(sampler, int(args.rollout.group_size))
        self.collector = EpisodeCollector(
            self.policy, self.catalog, sampler,
            batch_size=int(args.rollout.batch_size),
            budget_mult=float(args.rollout.budget_mult),
            min_budget=int(args.rollout.min_budget),
            image_size=int(args.image_size), device=self.device,
            temperature=float(args.rollout.temperature), precision=args.precision,
            seed=int(args.seed),
            reward=OmegaConf.to_container(args.reward, resolve=True),
            max_episode_steps=int(args.rollout.max_episode_steps),
            collect_goal_points=False, owner="GRPOTrainer")

        self.optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=float(args.train.learning_rate),
            weight_decay=float(args.train.weight_decay))
        self.rng = np.random.default_rng(int(args.seed))
        self.logger = CSVLogger(os.path.join(run_dir, "metrics.csv"))
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        self.update = 0

    # -- collection ---------------------------------------------------------

    def collect_filtered(self):
        """Draw groups until ``groups_per_update`` survive filtering.

        Dynamic sampling: a constant number of *usable* groups per update, rather than a
        constant number of rollouts of unknown usefulness. Capped by
        ``max_oversample_factor`` — if a task mix cannot produce enough surviving groups,
        that is a finding to stop on, not a loop to spin in.
        """
        want = int(self.args.rollout.groups_per_update)
        batch_groups = int(self.args.rollout.collect_groups_at_once)
        cap = int(want * float(self.args.rollout.max_oversample_factor))

        kept, all_eps, drawn = [], [], 0
        stats: Dict[str, float] = {}
        while True:
            groups = self.groups.sample_groups(batch_groups)
            eps = self.collector.collect_groups(groups)
            drawn += batch_groups
            all_eps.extend(eps)
            if self.args.rollout.filter_groups:
                surv, st = filter_groups(eps)
            else:
                surv, st = list(eps), {"groups_collected": float(batch_groups),
                                       "groups_kept": float(batch_groups),
                                       "zero_variance_group_frac": 0.0}
            kept.extend(surv)
            n_kept = len({e.group_id for e in kept})
            if n_kept >= want or drawn >= cap:
                stats = {
                    "groups_drawn": float(drawn),
                    "groups_kept": float(n_kept),
                    "oversample_factor": drawn / max(1, want),
                    "zero_variance_group_frac": 1.0 - n_kept / max(1, drawn),
                    "episodes_rolled": float(len(all_eps)),
                    "episodes_used": float(len(kept)),
                }
                if drawn >= cap and n_kept < want:
                    print(f"[grpo] WARNING oversample cap hit: {n_kept}/{want} groups "
                          f"survived from {drawn} drawn. The task mix is saturated or "
                          f"impossible — see doc/0004 §5.", flush=True)
                break
        return kept, all_eps, stats

    # -- update -------------------------------------------------------------

    def _forward_logits(self, batch, model):
        ctx = (torch.autocast("cuda", dtype=self.autocast_dtype)
               if self.autocast_dtype is not None else _null())
        with ctx:
            return model(batch.image, batch.goal_image, batch.interaction)["pi_logits"]

    def train_on(self, episodes, advantages) -> Dict[str, float]:
        agg: Dict[str, List[float]] = {}
        for _ in range(int(self.args.train.epochs)):
            for batch in iter_minibatches(
                    episodes, advantages,
                    int(self.args.train.minibatch_episodes), self.rng, self.device):
                logits = self._forward_logits(batch, self.policy)
                ref_logits = None
                if self.ref is not None:
                    with torch.no_grad():
                        ref_logits = self._forward_logits(batch, self.ref)
                loss, m = grpo_loss(logits, batch, self.cfg, ref_logits)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.cfg.max_grad_norm)
                self.optimizer.step()

                for k, v in m.items():
                    agg.setdefault(k, []).append(float(v))
                agg.setdefault("grad_norm", []).append(float(gn))
                if float(m["approx_kl"]) > self.cfg.target_kl:
                    # Early stop on the *behaviour* KL: the batch has drifted far enough
                    # that its stored log-probs no longer describe this policy.
                    agg.setdefault("early_stopped", []).append(1.0)
                    return {k: float(np.mean(v)) for k, v in agg.items()}
        return {k: float(np.mean(v)) for k, v in agg.items()}

    # -- metrics ------------------------------------------------------------

    def outcome_stats(self, episodes) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if not episodes:
            return out
        r = np.array([e.reward for e in episodes])
        out["success"] = float(r.mean())
        out["episodes"] = float(len(episodes))
        out["mean_len"] = float(np.mean([len(e) for e in episodes]))
        for fam in FAMILIES:
            sel = [e for e in episodes if e.family == fam]
            if not sel:
                continue
            s = int(sum(e.reward > 0 for e in sel))
            lo, hi = wilson(s, len(sel))
            out[f"{fam}/success"] = s / len(sel)
            out[f"{fam}/episodes"] = float(len(sel))
            out[f"{fam}/ci_lo"], out[f"{fam}/ci_hi"] = lo, hi
        return out

    # -- the loop -----------------------------------------------------------

    def run(self) -> None:
        total = int(self.args.train.updates)
        try:
            while self.update < total:
                t0 = time.time()
                kept, all_eps, cstats = self.collect_filtered()
                if not kept:
                    raise RuntimeError(
                        "no groups survived filtering; nothing to learn from")
                adv, astats = group_advantages(
                    [e.reward for e in kept], [e.group_id for e in kept],
                    normalise=bool(self.args.rollout.normalise_advantages))
                m = self.train_on(kept, adv)
                self.update += 1

                row = {"update": self.update, **cstats, **astats, **m,
                       # Outcome stats over EVERYTHING rolled, not just what survived —
                       # filtering is an update-side decision and must not flatter the
                       # success rate it reports.
                       **self.outcome_stats(all_eps),
                       "seconds": time.time() - t0}
                self._emit(row)
                if int(self.args.train.save_every) and \
                        self.update % int(self.args.train.save_every) == 0:
                    self.save()
        finally:
            self.save(final=True)
            self.collector.close()

    def _emit(self, row: Dict[str, float]) -> None:
        self.logger.log(row)
        head = " ".join(f"{k}={row[k]:.4g}" for k in
                        ("success", "policy_loss", "kl_ref", "approx_kl", "entropy",
                         "zero_variance_group_frac", "oversample_factor")
                        if k in row)
        print(f"[update {self.update}/{int(self.args.train.updates)}] {head} "
              f"({row.get('seconds', 0):.0f}s)", flush=True)
        fam = [f"{f}={row[f'{f}/success']:.2f}[{row[f'{f}/ci_lo']:.2f},"
               f"{row[f'{f}/ci_hi']:.2f}]({int(row[f'{f}/episodes'])})"
               for f in FAMILIES if f"{f}/success" in row]
        if fam:
            print("    " + " ".join(fam), flush=True)

    def save(self, final: bool = False) -> str:
        tag = "final" if final else f"{self.update:06d}"
        path = os.path.join(self.run_dir, "checkpoints", f"grpo-{tag}.pt")
        self.policy.save(path, update=self.update,
                         train_config=OmegaConf.to_container(self.args, resolve=True))
        print(f"[grpo] saved {path}", flush=True)
        return path


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


@hydra.main(config_path=".", config_name="config_grpo", version_base=None)
def main(args: DictConfig) -> None:
    torch.set_float32_matmul_precision("high")
    _install_signal_handlers()
    _seed(int(args.seed))
    run_dir = os.getcwd()
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as fh:
        fh.write(OmegaConf.to_yaml(args, resolve=True))
    GRPOTrainer(args, run_dir=run_dir).run()


def _install_signal_handlers() -> None:
    """SIGTERM as a normal exit so `run`'s `finally` saves and closes the emulator.

    Python's default disposition skips both, which loses the checkpoint and leaves an
    emulator open in a worker that `PR_SET_PDEATHSIG` would otherwise have reaped.
    """
    def _exit(signum, _frame):
        print(f"\n[grpo] {signal.Signals(signum).name} — saving and stopping", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
