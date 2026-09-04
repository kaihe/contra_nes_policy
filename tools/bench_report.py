"""Tables for doc/0019 §3 from `tools/bench_step.py`'s JSONL.

    python tools/bench_report.py runs/bench/arms.jsonl --ladder 160000
    python tools/bench_report.py runs/bench/arms.jsonl --predict 64

Labels are `<size>-<arm>-r<repeat>`. Repeats are aggregated by **median of the per-run
medians**, and `spread` reports how far the repeats disagreed — on a power-capped mobile
4090 that is the number which says whether an arm's win is real. An arm whose repeats
disagree by more than a few percent has not been measured, it has been sampled once.

`--ladder N` prices a full ladder at each arm, at **matched episodes rather than matched
steps**: an arm at batch 32 reaches the same exposure in half the cycles, so charging it
N steps would credit it twice. That is the number doc/0019 is deciding on, since 20% off
XXL is worth more than halving M.

`--predict B` is the §2 gate that decides whether a larger batch is allowed to run at
all. It fits VRAM = fixed + slope x batch from the measured arms — `fixed_vram_gib` is
measured directly, before any activation exists — and refuses any batch whose prediction
breaks the 70%-of-16 GB VRAM budget or the 6 GB host headroom.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

SIZES = ["M", "L", "XL", "XXL"]
VRAM_BUDGET_GIB = 0.70 * 16.0        # §2: 70% of the 4090's 16 GB
HOST_GIB = 19.0                      # the WSL VM
RSS_HEADROOM_GIB = 6.0               # §2: leave this much of the host free


def load(path):
    runs = defaultdict(list)
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        # `<size>-<arm>-r<repeat>`
        parts = r["label"].split("-")
        size, arm = parts[0], parts[1]
        runs[(size, arm)].append(r)
    return runs


def agg(rs):
    """Median of per-run medians, plus how far the repeats disagreed."""
    med = np.array([r["step_ms_median"] for r in rs])
    out = dict(rs[0])
    out["step_ms"] = float(np.median(med))
    out["spread_pct"] = float((med.max() - med.min()) / np.median(med) * 100) if len(med) > 1 else 0.0
    out["n"] = len(rs)
    for k in ("peak_vram_gib", "peak_rss_gib", "fixed_vram_gib", "padding_waste",
              "padded_tokens_per_step", "real_tokens_per_step"):
        if k in rs[0]:
            out[k] = float(np.median([r[k] for r in rs]))
    out["tokens_per_sec"] = out["padded_tokens_per_step"] / (out["step_ms"] / 1000.0)
    out["episodes_per_sec"] = out["batch_size"] / (out["step_ms"] / 1000.0)
    return out


def table(runs, ladder):
    arms = sorted({a for _, a in runs}, key=lambda a: (len(a), a))
    hdr = (f"{'size':>4} {'arm':>4} {'bs':>3} {'wrk':>3} {'n':>2} {'ms/step':>8} "
           f"{'spread':>7} {'tok/s':>8} {'ep/s':>7} {'pad%':>5} {'VRAM':>6} "
           f"{'fixed':>6} {'RSS':>6}")
    if ladder:
        hdr += f" {'C'+str(ladder//1000)+'k':>8} {'vs A':>6}"
    print("\n" + hdr)
    for size in SIZES:
        base = runs.get((size, "A"))
        base_min = None
        if base and ladder:
            b = agg(base)
            base_min = ladder * 16 / b["batch_size"] * b["step_ms"] / 1000 / 60
        for arm in arms:
            rs = runs.get((size, arm))
            if not rs:
                continue
            r = agg(rs)
            line = (f"{size:>4} {arm:>4} {r['batch_size']:>3} {r['num_workers']:>3} "
                    f"{r['n']:>2} {r['step_ms']:>8.2f} {r['spread_pct']:>6.1f}% "
                    f"{r['tokens_per_sec']:>8.0f} {r['episodes_per_sec']:>7.1f} "
                    f"{r['padding_waste']*100:>5.1f} {r['peak_vram_gib']:>6.2f} "
                    f"{r.get('fixed_vram_gib', float('nan')):>6.2f} {r['peak_rss_gib']:>6.2f}")
            if ladder:
                mins = ladder * 16 / r["batch_size"] * r["step_ms"] / 1000 / 60
                line += f" {mins:>7.0f}m"
                if base_min:
                    line += f" {(mins/base_min - 1)*100:>+5.0f}%"
            print(line)
    if ladder:
        print(f"\nladder totals (C{ladder//1000}k, all four sizes):")
        for arm in arms:
            tot = 0.0
            have = []
            for size in SIZES:
                rs = runs.get((size, arm))
                if not rs:
                    continue
                r = agg(rs)
                tot += ladder * 16 / r["batch_size"] * r["step_ms"] / 1000 / 60
                have.append(size)
            if len(have) == len(SIZES):
                print(f"  arm {arm:>3}: {tot/60:5.2f} h")
            elif have:
                print(f"  arm {arm:>3}: {tot/60:5.2f} h (partial: {'+'.join(have)})")


def predict(runs, target):
    """Fit VRAM/RSS against batch and apply §2's gate."""
    print(f"\nbatch {target} prediction, from the measured arms (§2 gate):")
    print(f"  VRAM budget {VRAM_BUDGET_GIB:.1f} GiB · RSS ceiling "
          f"{HOST_GIB - RSS_HEADROOM_GIB:.1f} GiB")
    for size in SIZES:
        pts = []
        for (s, arm), rs in runs.items():
            if s != size:
                continue
            r = agg(rs)
            if int(r["num_workers"]) == 0:          # workers change RSS, not activations
                pts.append((r["batch_size"], r["peak_vram_gib"], r["peak_rss_gib"],
                            r.get("fixed_vram_gib", 0.0)))
        if not pts:
            continue
        pts.sort()
        if target in [p[0] for p in pts]:
            print(f"  {size:>4}: batch {target} already measured")
            continue
        if len(pts) >= 2:
            # Two measured batches pin fixed and slope exactly; no assumption needed.
            (b0, v0, r0, _), (b1, v1, r1, _) = pts[0], pts[-1]
            slope = (v1 - v0) / (b1 - b0)
            fixed = v0 - slope * b0
            vram = fixed + slope * target
            rslope = (r1 - r0) / (b1 - b0)
            rss = r0 + rslope * (target - b0)
            how = f"fit on batches {b0},{b1}"
        else:
            b0, v0, r0, fixed = pts[0]
            act = max(0.0, v0 - fixed)
            vram = fixed + act * target / b0
            rss = r0 * target / b0
            how = f"activation term of batch {b0} scaled"
        ok_v, ok_r = vram <= VRAM_BUDGET_GIB, rss <= HOST_GIB - RSS_HEADROOM_GIB
        print(f"  {size:>4}: VRAM {vram:5.2f} GiB {'ok ' if ok_v else 'OVER'} · "
              f"RSS {rss:5.2f} GiB {'ok ' if ok_r else 'OVER'} · {how} "
              f"-> {'RUN' if ok_v and ok_r else 'SKIP'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--ladder", type=int, default=0)
    ap.add_argument("--predict", type=int, default=0)
    a = ap.parse_args()
    runs = load(a.jsonl)
    table(runs, a.ladder)
    if a.predict:
        predict(runs, a.predict)


if __name__ == "__main__":
    main()
