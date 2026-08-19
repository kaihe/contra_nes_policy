"""Compare train and val CE across cells, on a shared *epoch* axis.

    python tools/plot_ce_cells.py doc/figures/0018-l-ce.png \
        --title "L (25.13M) — CE by exposure" \
        --cell "D10k C20k cos=runs/scaling/l-d13-s0" \
        --cell "D10k C40k=runs/scaling/L-D10k-C40k"

Cycles are not comparable across cells: 40,000 steps over 9,900 episodes and 160,000 over
38,024 are the same number of passes, and it is passes that drive the memorisation this
repo measures. So x is **epochs** = `steps x batch / train_episodes`, taken from each run's
`dataset.json` where it exists and from the release manifest's
`accepted_generated_episodes` otherwise. A cell whose episode count cannot be resolved is
skipped loudly rather than plotted on a guessed axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import yaml

TICKS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6, 2.2]


def train_episodes(run: str) -> int | None:
    meta = os.path.join(run, "dataset.json")
    if os.path.exists(meta):
        return int(json.load(open(meta))["train_episodes"])
    cfg = yaml.safe_load(open(os.path.join(run, "resolved_config.yaml")))
    path = (cfg.get("boss_scaling") or {}).get("manifest")
    if not path:
        return None
    path = os.path.expanduser(str(path)).replace("/releases/", "/releases-legacy/")
    if not os.path.exists(path):
        return None
    return int(json.load(open(path))["accepted_generated_episodes"])


def series(run: str, phase: str):
    with open(os.path.join(run, "metrics.csv")) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("phase") == phase]
    return (np.array([float(r["step"]) for r in rows]),
            np.array([float(r["loss"]) for r in rows]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cell", action="append", default=[], help="'label=run_dir', repeatable")
    ap.add_argument("--title", default="")
    ap.add_argument("--x", choices=["epochs", "cycles"], default="epochs",
                    help="cycles is only meaningful when every cell shares a data tier")
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--pair", action="store_true",
                    help="colour by pairs of cells, dashing the second of each — for an "
                         "A/B design where the pair, not the cell, is the unit")
    ap.add_argument("--group-size", type=int, default=1,
                    help="colour groups of N cells alike and vary line style within each "
                         "group; --pair is shorthand for 2")
    args = ap.parse_args()

    cells = []
    for spec in args.cell:
        label, _, run = spec.partition("=")
        n = train_episodes(run)
        if n is None:
            print(f"  SKIP {label}: cannot resolve train_episodes for {run}")
            continue
        cfg = yaml.safe_load(open(os.path.join(run, "resolved_config.yaml")))
        if "data" in cfg and "batch_size" in cfg["data"]:
            batch = int(cfg["data"]["batch_size"])
        else:
            batch = int((cfg.get("loader") or {}).get("batch_size", 16))
        cells.append((label, run, n, batch))

    # Categorical, not sequential: these cells are not an ordered series, and a viridis ramp
    # makes the two arms of a like-for-like pair nearly the same colour.
    palette = ["#4C72B0", "#C44E52", "#55A868", "#DD8452", "#8172B3", "#937860", "#DA8BC3"]
    group_size = 2 if args.pair else args.group_size
    if group_size > 1:
        line_styles = ["-", "--", "-.", ":"]
        if group_size > len(line_styles):
            ap.error(f"--group-size must be <= {len(line_styles)}")
        colors = [palette[(i // group_size) % len(palette)] for i in range(len(cells))]
        styles = [line_styles[i % group_size] for i in range(len(cells))]
    else:
        colors = [palette[i % len(palette)] for i in range(len(cells))]
        styles = ["-"] * len(cells)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=150, sharey=True)
    for (label, run, n, batch), color, ls in zip(cells, colors, styles):
        for ax, phase in zip(axes, ["train", "val"]):
            step, ce = series(run, phase)
            if not len(step):
                continue
            ep = step if args.x == "cycles" else step * batch / n
            if phase == "train":
                w = 20
                ax.plot(ep, ce, color=color, alpha=0.10, lw=0.6)
                ax.plot(ep[w - 1:], np.convolve(ce, np.ones(w) / w, mode="valid"),
                        color=color, lw=1.7, ls=ls,
                        label=label if group_size > 1 else f"{label}  ({n:,} eps)")
            else:
                ax.plot(ep, ce, color=color, lw=1.7, ls=ls)
                i = int(np.argmin(ce))
                ax.plot(ep[i], ce[i], marker="v", color=color, ms=6, mec="white", mew=0.8)
        print(f"  {label:24} {n:>6,} eps  {series(run,'train')[0][-1]*batch/n:5.1f} epochs")

    for ax, title in zip(axes, ["Train CE", "Val CE  (▾ = minimum)"]):
        if args.cooldown:
            ax.axvline(args.cooldown, color="0.45", ls=":", lw=1.2)
        ax.set_xlabel("training cycle" if args.x == "cycles"
                      else "epochs (passes over the training set)")
        ax.set_title(title, fontsize=11)
        ax.set_yscale("log")
        ax.set_yticks(TICKS)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.grid(alpha=0.22, which="both")
    axes[0].set_ylabel("cross-entropy")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    if args.title:
        fig.suptitle(args.title, fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
