"""Recurrent PPO fine-tuning of the Contra cross-view policy.

A second entry point beside ``train.py``, not a mode of it: the behaviour-cloning
path is stable and produces the 72.8% checkpoint this run starts from, and threading
an on-policy loop through a Lightning ``fit`` would destabilise it for no benefit.
Nothing here imports ``contra_policy.lit``.

    python train_rl.py                                   # defaults, config_rl.yaml
    python train_rl.py train.updates=2 rollout.steps=64 rollout.episodes=2   # smoke
    python train_rl.py rollout.num_workers=4             # 4 emulators, 4 processes
    python train_rl.py resume_from=runs/rl/<date>/<time>/checkpoints/rl-000025.pt

Why RL at all: every BC demonstration is a *successful* trajectory, so no failure
state is ever demonstrated and no amount of further BC can supply recovery behaviour.
Closed-loop evaluation of the BC checkpoint says the grounding is already accurate
(5.3 px boss point error) while boss completion is 8.8% — the policy knows where the
target is and cannot execute or recover. That gap is what this optimises.

Runs land in ``runs/rl/<date>/<time>/`` with ``weights/`` (evaluator-readable),
``checkpoints/`` (resumable) and ``metrics.csv``.
"""

from __future__ import annotations

import os
import signal

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf


@hydra.main(config_path=".", config_name="config_rl", version_base=None)
def main(args):
    torch.set_float32_matmul_precision("high")
    _install_signal_handlers()
    _seed_everything(int(args.seed))

    run_dir = os.getcwd()          # hydra has already chdir'd into the run directory
    with open(os.path.join(run_dir, "resolved_config.yaml"), "w") as fh:
        fh.write(OmegaConf.to_yaml(args, resolve=True))

    from contra_policy.rl.trainer import RLTrainer

    RLTrainer(args, run_dir=run_dir).run()


def _install_signal_handlers() -> None:
    """Turn SIGTERM into a normal Python exit so ``RLTrainer.run``'s ``finally`` runs.

    Python's default SIGTERM disposition terminates the interpreter outright: no
    ``finally``, no ``atexit``. That loses the final checkpoint, leaves emulators
    open, and — before ``_die_with_parent`` — leaked one CUDA worker process per
    ``rollout.num_workers``. ``timeout``, ``kill`` and most job schedulers all send
    SIGTERM, so this is the ordinary way a long run ends, not an edge case.
    """
    def _exit(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\n[rl] {name} received — saving and shutting down", flush=True)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _exit)
    signal.signal(signal.SIGINT, _exit)


def _seed_everything(seed: int) -> None:
    """Python, NumPy and Torch. Worker task selection and action sampling are seeded
    separately, from this seed, so a worker's stream does not depend on how much the
    parent process happened to draw first."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
