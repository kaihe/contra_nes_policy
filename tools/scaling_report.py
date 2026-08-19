#!/usr/bin/env python
"""Collate the scaling cells' cross-entropy curves — doc/0013 (sizes), doc/0014 (compute), doc/0015 (data).

    python tools/scaling_report.py runs/scaling

This repo's BC stage owns **CE only** — closed-loop success is `contra_nes_evaluation`'s
job, from the fixed-step checkpoints (doc/0013 §3). So this reports what CE can honestly
say about a scaling grid and refuses to imply the rest.

Two numbers per cell, and the gap between them is the point:

``train_ce``
    how well the cell fits 9,900 memorizable episodes. Falls with capacity, and is the
    **memorization-rate** signal the model axis is actually about.
``val_ce``
    held-out imitation loss. 0010 measured this *rising* while play improved, and the
    CE-optimal checkpoint playing 12.9 pp worse than the overfit final — so a lower
    ``val_ce`` here is **not** a better policy, and this tool will not rank cells by it.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np
import yaml

#: doc/0013 §2 — `(d_model, n_layer)` → (cell, trainable params in millions). `d_model: null`
#: means "take the encoder's width", which is the M cell, so both spellings of M appear.
LADDER = {(None, 4): ("M", 12.86), (512, 4): ("M", 12.86),
          (256, 2): ("XS", 1.74), (384, 3): ("S", 5.52), (640, 5): ("L", 25.13),
          (768, 6): ("XL", 42.89), (1024, 8): ("XXL", 101.76)}

#: Shard count of each release's *full* prefix. A run at the full prefix is named for the
#: release alone (`D20k`); anything shorter carries the prefix (`D20k.e8`).
FULL_PREFIX = {"10k": 13, "20k": 27}


def _config(run_dir: str) -> dict:
    path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.exists(path):
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _data_label(cfg: dict, run_dir: Optional[str] = None) -> str:
    """`D20k.e8` — how much data the run trained on, and from which release.

    The tier is named for **what the model saw**, never for the store the bytes came from:
    a datahouse run reading 13 of 53 shards is `D10k`, the same tier as the 10k release,
    because it trains on ~9.3k episodes. Naming it for the 40k store would put two very
    different runs under one label.

    Datahouse runs take the count from `dataset.json`, which `train_bc` writes with the
    episodes actually indexed; there is nothing in the config to derive it from. A run
    predating that file falls back to naming the slice.
    """
    dh = cfg.get("datahouse") or {}
    if dh.get("root"):
        meta = {}
        if run_dir:
            try:
                with open(os.path.join(run_dir, "dataset.json")) as fh:
                    meta = json.load(fh)
            except (OSError, ValueError):
                meta = {}
        w = dh.get("weapon", "?")
        weapons = list(w) if isinstance(w, (list, tuple)) else [str(w)]
        suffix = "" if weapons == ["spread"] else "(" + "+".join(s[:2] for s in weapons) + ")"
        n = meta.get("train_episodes")
        if n:
            # Nearest 10k tier, matching how the releases were named (9,900 -> 10k).
            return f"D{int(n / 10000 + 0.5) * 10}k{suffix}"
        return "DH:" + "+".join(weapons)
    scaling = cfg.get("boss_scaling") or {}
    manifest, shards = str(scaling.get("manifest", "")), scaling.get("shard_count")
    release = next((r for r in FULL_PREFIX if r in manifest), "mixed")
    return f"D{release}" if shards == FULL_PREFIX.get(release) else f"D{release}.e{shards}"


def _rows(path: str) -> List[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> Optional[float]:
    v = row.get(key)
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


def summarize(run_dir: str) -> Optional[Dict]:
    metrics = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(metrics):
        return None
    rows = _rows(metrics)
    train = [r for r in rows if r.get("phase") == "train"]
    val = [r for r in rows if r.get("phase", "").startswith("val")]
    if not train:
        return None

    # Tail-average the last 20 logged train rows: a single step's loss swings ~0.1 on a
    # batch of 16 episodes, which is larger than the differences between adjacent cells.
    tail = [x for x in (_f(r, "loss") for r in train[-20:]) if x is not None]
    val_by_step = {int(_f(r, "step") or 0): _f(r, "loss") for r in val}
    final_step = max(val_by_step) if val_by_step else None
    best_step = min(val_by_step, key=lambda s: val_by_step[s]) if val_by_step else None

    # Cell, data and cycles come from the *config and the metrics*, never from the
    # directory name. Runs are named `<model>-D<data>-C<cycles>`, but a name is a label
    # someone types: this table is read as evidence, so it reports what the run actually
    # resolved. Legacy names (`m-d13-s0`) therefore keep reporting with no special case.
    cfg = _config(run_dir)
    core = (cfg.get("policy") or {}).get("core") or {}
    d_model, n_layer = core.get("d_model"), core.get("n_layer")
    d = int(d_model) if d_model is not None else None
    label, params = LADDER.get((d, n_layer), (f"d{d}L{n_layer}", None))
    # The budget is the *config's*, not the last logged step: a run still training would
    # otherwise report its progress as its cell (`C42k` for an L run 42k into 160k) and
    # sort into the wrong place. `*` marks a run that has not reached its budget, so a
    # partial row is never mistaken for a finished cell.
    reached = int(_f(train[-1], "step") or 0)
    steps = int(((cfg.get("train") or {}).get("steps")) or reached)
    partial = "*" if reached < steps else ""
    # Schedule is reported because two of them now coexist: the retired cosine runs and
    # everything after. Same cell, same data, same cycles, different LR trajectory — without
    # this column those rows are indistinguishable in the table.
    sched = str(((cfg.get("train") or {}).get("lr_decay")) or "?")
    return {"cell": label, "params_m": params, "run": run_dir, "sched": sched,
            "data": _data_label(cfg, run_dir), "cycles": f"C{steps // 1000}k{partial}",
            "d_model": d_model, "n_layer": n_layer,
            "steps": steps,
            "train_ce": float(np.mean(tail)) if tail else None,
            "val_ce_final": val_by_step.get(final_step),
            "val_ce_best": val_by_step.get(best_step),
            "val_best_step": best_step,
            "step_ms": float(np.median([x for x in (_f(r, "step_ms") for r in train)
                                        if x is not None]) or 0)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="runs/scaling")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    cells = [c for c in (summarize(d) for d in sorted(glob.glob(f"{args.root}/*/")))
             if c and not os.path.basename(c["run"].rstrip("/")).startswith("smoke")]
    cells.sort(key=lambda c: (c["params_m"] is None, c["params_m"] or 0,
                             c["data"], c["steps"], c["sched"]))

    if args.json:
        print(json.dumps(cells, indent=2))
        return
    if not cells:
        print(f"no completed cells under {args.root}")
        return

    print(f"{'cell':5s} {'params':>8s} {'data':>15s} {'cycles':>7s} {'sched':>7s} "
          f"{'d':>5s} {'L':>3s} {'train_ce':>9s} {'val_ce':>8s} {'val_best':>9s} "
          f"{'@step':>7s} {'ms/step':>8s}")
    print("-" * 104)
    for c in cells:
        p = f"{c['params_m']:.2f}M" if c["params_m"] else "?"
        fmt = lambda v, w=8: (f"{v:{w}.4f}" if v is not None else " " * (w - 1) + "-")
        print(f"{c['cell']:5s} {p:>8s} {c['data']:>15s} {c['cycles']:>7s} "
              f"{c['sched']:>7s} {str(c['d_model'] or '-'):>5s} {str(c['n_layer'] or '-'):>3s} "
              f"{fmt(c['train_ce'], 9)} {fmt(c['val_ce_final'])} {fmt(c['val_ce_best'], 9)} "
              f"{str(c['val_best_step'] or '-'):>7s} {c['step_ms']:>8.1f}")
    print()
    print("train_ce: memorization rate — the signal the model axis is about.")
    print("val_ce  : NOT a ranking. 0010 measured it rising while play improved, and the")
    print("          CE-optimal checkpoint playing 12.9 pp worse than the overfit final.")
    print("Win rate is contra_nes_evaluation's, from the fixed-step checkpoints.")


if __name__ == "__main__":
    main()
