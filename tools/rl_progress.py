"""Read a run's ``metrics.csv`` and say whether the four families are actually moving.

The per-update line the trainer prints is the right data in the wrong shape for
judging progress. One update collects ~36 episodes spread over four families, so
`boss` lands 4-9 of them: its completion is a fraction with a single-digit
denominator, taking values in {0, 1/4, 1/3, ...}. Printed once per update it looks
like a wildly oscillating signal, and reading a trend off it is reading noise.

So pool. ``completion x episodes`` is an exact integer success count, so summing both
over a window of updates recovers the true numerator and denominator with nothing
lost. This reports each family as ``successes/episodes`` with a Wilson 95% interval,
which — unlike successes/n +- 1.96*sqrt(p(1-p)/n) — stays sane at p=0 and p=1, the
two values `boss` spends most of its time at.

    python tools/rl_progress.py runs/rl/2026-07-29/16-20-00
    python tools/rl_progress.py <run_dir> --window 10      # updates pooled per bucket

What this is NOT: an evaluation number. These are *training* tasks sampled from a
deliberately reweighted mixture, played **stochastically** (``temperature=1.0``, which
PPO requires). The 72.8% / 8.8%-boss baseline is greedy play on the **held-out val
split**. The two are not comparable, and a rise here is evidence the objective is
moving, not that the policy got better on the metric the run is judged by. Use
``contra_nes_evaluation`` on a saved ``weights/`` checkpoint for that.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

FAMILIES = ("kill", "item", "traverse", "boss")


def wilson(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval — the one that behaves at 0/n and n/n."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def read_rows(run_dir: str) -> List[Dict[str, str]]:
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        sys.exit(f"no metrics.csv in {run_dir}")
    with open(path) as fh:
        return [r for r in csv.DictReader(fh) if r.get("phase") == "ppo"]


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pool(rows: List[Dict[str, str]], family: str) -> Tuple[int, int]:
    """``(successes, episodes)`` for one family over ``rows``.

    completion is successes/episodes, so the product is the integer count back. It is
    rounded rather than truncated because the CSV holds a decimal expansion of a
    ratio, and 0.6666666666666666 * 3 is 1.9999999999999998.
    """
    succ = eps = 0
    for r in rows:
        n = _f(r, f"rollout/{family}/episodes")
        c = _f(r, f"rollout/{family}/completion")
        if n is None or c is None:
            continue
        eps += int(round(n))
        succ += int(round(c * n))
    return succ, eps


def bar(lo: float, hi: float, p: float, width: int = 28) -> str:
    """The interval as a span, so a wide one is visibly wide."""
    cells = []
    for i in range(width):
        x = i / (width - 1)
        if abs(x - p) < 0.5 / width:
            cells.append("|")
        elif lo <= x <= hi:
            cells.append("-")
        else:
            cells.append(" ")
    return "".join(cells)


def report(run_dir: str, window: int, quiet: bool = False) -> int:
    rows = read_rows(run_dir)
    if not rows:
        sys.exit(f"{run_dir}/metrics.csv has no completed PPO updates yet")

    n_updates = len(rows)
    print(f"\n{os.path.abspath(run_dir)}")
    print(f"{n_updates} completed update(s), pooling {window} per bucket\n")

    # -- per-family, pooled over the whole run and over the last window ---------
    print(f"{'family':9} {'whole run':>18}   {'last ' + str(window) + ' updates':>18}"
          f"   95% CI (recent)")
    print("-" * 86)
    recent = rows[-window:]
    for fam in FAMILIES:
        s_all, n_all = pool(rows, fam)
        s_rec, n_rec = pool(recent, fam)
        if n_all == 0:
            print(f"{fam:9} {'no episodes':>18}")
            continue
        p_all = s_all / n_all
        if n_rec:
            p_rec = s_rec / n_rec
            lo, hi = wilson(s_rec, n_rec)
            print(f"{fam:9} {p_all:>10.1%} ({s_all:>3}/{n_all:<3}) "
                  f"  {p_rec:>10.1%} ({s_rec:>3}/{n_rec:<3})   "
                  f"[{lo:.2f},{hi:.2f}] {bar(lo, hi, p_rec)}")
        else:
            print(f"{fam:9} {p_all:>10.1%} ({s_all:>3}/{n_all:<3})   {'-':>18}")

    # -- trend: first bucket vs last, so a flat family is obvious ---------------
    if n_updates >= 2 * window:
        print(f"\ntrend (first {window} updates -> last {window}):")
        first = rows[:window]
        for fam in FAMILIES:
            s0, n0 = pool(first, fam)
            s1, n1 = pool(recent, fam)
            if not n0 or not n1:
                continue
            p0, p1 = s0 / n0, s1 / n1
            lo0, hi0 = wilson(s0, n0)
            lo1, hi1 = wilson(s1, n1)
            # Non-overlapping Wilson intervals is a deliberately conservative bar. It
            # is the honest one here: these samples are small and every family will
            # look like it is moving if you squint at point estimates.
            sep = "moved" if (lo1 > hi0 or lo0 > hi1) else "not separated"
            print(f"  {fam:9} {p0:>6.1%} (n={n0:<4}) -> {p1:>6.1%} (n={n1:<4})  "
                  f"{p1 - p0:+.1%}  {sep}")
    else:
        print(f"\ntrend needs >= {2 * window} updates; have {n_updates}.")

    # -- how long until a family can say anything ------------------------------
    print("\nresolution:")
    for fam in FAMILIES:
        s, n = pool(rows, fam)
        if not n:
            continue
        per_update = n / n_updates
        lo, hi = wilson(s, n)
        # Updates needed for a +-5pp interval at the current rate, as a planning number.
        p = max(min(s / n, 0.99), 0.01)
        need = 1.96 ** 2 * p * (1 - p) / (0.05 ** 2)
        print(f"  {fam:9} {per_update:4.1f} episodes/update · "
              f"whole-run CI [{lo:.2f},{hi:.2f}] · "
              f"~{math.ceil(need / per_update):>4} updates for a +-5pp read")

    if not quiet:
        print("\nThese are TRAINING tasks played stochastically (temperature=1.0), on a\n"
              "reweighted family mixture. They are not comparable to the 72.8% / 8.8%-boss\n"
              "baseline, which is greedy play on the held-out val split. Run\n"
              "contra_nes_evaluation against runs/.../weights/ for a number you can compare.")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a runs/rl/<date>/<time> directory")
    ap.add_argument("--window", type=int, default=10,
                    help="updates pooled per bucket (default 10)")
    ap.add_argument("--quiet", action="store_true", help="drop the closing caveat")
    args = ap.parse_args(argv)
    return report(args.run_dir, max(1, args.window), args.quiet)


if __name__ == "__main__":
    sys.exit(main())
