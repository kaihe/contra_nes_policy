"""Put train CE, val CE and closed-loop win rate on one axis pair, per model size.

    python tools/plot_ce_vs_play.py runs/scaling40k doc/figures/0016-ce-vs-play.png \
        --eval ~/code/contra_nes_evaluation/runs \
        --probe 0817-d40k-c160k --probe 0818-d40k-mid --probe 0818-d40k-seeds \
        --cooldown 144000 --title "D40k-C160k ladder (0016)"

CE comes from this repo's `metrics.csv` (`loss` at `phase=train` / `phase=val`, the same
columns `tools/scaling_report.py` reads). Win rate comes from `contra_nes_evaluation`'s
`summary.json` files, read rather than transcribed, so the figure cannot drift from eval's
own numbers. Probe directories are named `<SIZE>-<role>[-s<seed>]-n100`.

The two quantities do not share units or direction — CE is a per-token loss on the left log
axis, win rate is a rate on the right linear axis — so this figure is a coincidence plot,
not a correlation. It exists because the interesting events (the cooldown, the val minimum,
the end-of-run jump) are located in *cycles*, and only a shared x-axis shows whether they
line up.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

ORDER = ["XS", "S", "M", "L", "XL", "XXL"]
CE_C, VAL_C, PLAY_C = "#2F5D8C", "#7FA6CC", "#C44E52"
TICKS = [0.15, 0.2, 0.3, 0.5, 0.8, 1.2, 1.6]


def ce_series(path: str, phase: str):
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("phase") == phase]
    return (np.array([float(r["step"]) for r in rows]),
            np.array([float(r["loss"]) for r in rows]))


def role_to_step(role: str, budget: int, best: int):
    """`40k` -> 40000, `final` -> the budget, `best` -> the val-CE minimum."""
    if role == "final":
        return budget
    if role == "best":
        return best
    m = re.fullmatch(r"(\d+)k", role)
    return int(m.group(1)) * 1000 if m else None


def collect(eval_root: str, probes, budget: int, best_step: dict):
    """{size: {step: [rate, ...]}} — one entry per seed that scored that step."""
    out = defaultdict(lambda: defaultdict(list))
    seen = set()
    for probe in probes:
        for path in sorted(glob.glob(os.path.join(eval_root, probe, "*", "summary.json"))):
            name = os.path.basename(os.path.dirname(path))
            m = re.fullmatch(r"([A-Za-z]+)-([0-9]+k|final|best)(?:-s(\d+))?-n\d+", name)
            if not m:
                continue
            size, role, seed = m.group(1).upper(), m.group(2), int(m.group(3) or 0)
            step = role_to_step(role, budget, best_step.get(size, 0))
            if step is None or (size, step, seed) in seen:
                continue          # a step scored in two probe dirs is one measurement
            seen.add((size, step, seed))
            out[size][step].append(float(json.load(open(path))["success_rate"]) * 100)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--eval", required=True, help="contra_nes_evaluation/runs")
    ap.add_argument("--probe", action="append", default=[], help="probe dir, repeatable")
    ap.add_argument("--cooldown", type=int, default=None)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    cells = []
    for name in sorted(os.listdir(args.root)):
        path = os.path.join(args.root, name, "metrics.csv")
        if os.path.exists(path):
            cells.append((re.split(r"-", name)[0].upper(), path))
    cells.sort(key=lambda c: ORDER.index(c[0]) if c[0] in ORDER else 99)

    budget, best_step = 0, {}
    for size, path in cells:
        vs, vy = ce_series(path, "val")
        ts, _ = ce_series(path, "train")
        budget = max(budget, int(ts[-1]))
        if len(vy):
            best_step[size] = int(vs[int(np.argmin(vy))])
    play = collect(os.path.expanduser(args.eval), args.probe, budget, best_step)

    n = len(cells)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(11.5, 7), dpi=150,
                             sharex=True, squeeze=False)
    flat = axes.ravel()
    for ax, (size, path) in zip(flat, cells):
        ts, ty = ce_series(path, "train")
        vs, vy = ce_series(path, "val")
        w = 20
        ax.plot(ts, ty, color=CE_C, alpha=0.10, lw=0.6)
        ax.plot(ts[w - 1:], np.convolve(ty, np.ones(w) / w, mode="valid"),
                color=CE_C, lw=1.6, label="train CE")
        ax.plot(vs, vy, color=VAL_C, lw=1.6, label="val CE")
        if len(vy):
            i = int(np.argmin(vy))
            ax.plot(vs[i], vy[i], marker="v", color=VAL_C, ms=7, mec="white", mew=0.8)
        ax.set_yscale("log")
        ax.set_yticks(TICKS)
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylim(0.12, 1.9)
        ax.grid(alpha=0.22, which="both")
        if args.cooldown:
            ax.axvline(args.cooldown, color="0.45", ls="--", lw=1)

        rx = ax.twinx()
        pts = sorted(play.get(size, {}).items())
        if pts:
            xs = np.array([p[0] for p in pts])
            mean = np.array([np.mean(p[1]) for p in pts])
            sd = np.array([np.std(p[1], ddof=1) if len(p[1]) > 1 else 0.0 for p in pts])
            reps = np.array([len(p[1]) > 1 for p in pts])
            rx.plot(xs, mean, color=PLAY_C, lw=1.5, marker="o", ms=4,
                    alpha=0.9, label="win rate")
            if reps.any():
                rx.errorbar(xs[reps], mean[reps], yerr=sd[reps], fmt="none",
                            ecolor=PLAY_C, elinewidth=1.4, capsize=3)
        rx.set_ylim(0, 100)
        rx.set_ylabel("win rate %", color=PLAY_C, fontsize=9)
        rx.tick_params(axis="y", colors=PLAY_C, labelsize=8)
        ax.set_title(size, fontsize=11, loc="left")
    for ax in flat[n:]:
        ax.axis("off")
    for ax in flat[:n]:
        ax.set_xticks(range(0, budget + 1, 40000))
        ax.set_xticklabels([f"{x // 1000}k" if x else "0" for x in
                            range(0, budget + 1, 40000)])
    for ax in axes[-1]:
        ax.set_xlabel("training cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("cross-entropy", color=CE_C, fontsize=9)
    h, l = flat[0].get_legend_handles_labels()
    fig.legend(h + [plt.Line2D([], [], color=PLAY_C, marker="o", ms=4, lw=1.5)],
               l + ["win rate (bars = SD over 5 seeds)"],
               loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.965))
    if args.title:
        fig.suptitle(args.title + " — CE and closed-loop play", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"wrote {args.out}")
    for size in [c[0] for c in cells]:
        got = {k: len(v) for k, v in sorted(play.get(size, {}).items())}
        print(f"  {size:4} steps/seeds: {got}")


if __name__ == "__main__":
    main()
