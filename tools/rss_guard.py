"""Run a command under a hard host-memory ceiling, and report what it actually used.

Why this exists: this box is a WSL2 VM with ``memory=20GB`` in ``.wslconfig`` and 8 GB
of swap. A rollout process is ~3.5 GB and every ``rollout.num_workers`` adds another
CUDA context, so a benchmark sweep can cross 20 GB, fall into swap, and take the whole
VM unresponsive — the terminal, the editor and the agent along with it. There is no
OOM killer response fast enough to save you, because nothing is technically out of
memory: it is just swapping.

So measure from outside and kill early. The child runs in its own process group; this
samples the group every ``--interval`` seconds and SIGKILLs the entire group the moment
the system crosses the ceiling.

    python tools/rss_guard.py --limit-gb 12 -- python train_rl.py rollout.num_workers=2

Two numbers are reported, and they answer different questions:

``group_pss``  Sum of PSS across the process group. Shared pages (the shared-memory
               state dict, the CUDA libraries mapped into every worker) are divided
               among the processes sharing them, so this is what the group genuinely
               costs — unlike summed RSS, which counts a shared page once per process
               and badly overstates a multi-worker run.
``sys_used``   System-wide ``MemTotal - MemAvailable``. This is the number that decides
               whether the VM swaps, and therefore the one the ceiling is applied to.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional, Tuple

PAGE_KB = 4


def _meminfo() -> Tuple[float, float]:
    """``(MemTotal, MemAvailable)`` in GB."""
    total = avail = 0.0
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) / 1e6
            elif line.startswith("MemAvailable:"):
                avail = float(line.split()[1]) / 1e6
                break
    return total, avail


def _group_pids(pgid: int) -> List[int]:
    out = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            if os.getpgid(pid) == pgid:
                out.append(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return out


def _pss_gb(pids: List[int]) -> float:
    """Proportional set size across ``pids``, in GB; falls back to RSS per process."""
    total_kb = 0.0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/smaps_rollup") as fh:
                for line in fh:
                    if line.startswith("Pss:"):
                        total_kb += float(line.split()[1])
                        break
        except (OSError, ValueError):
            try:                                     # smaps_rollup can be absent
                with open(f"/proc/{pid}/statm") as fh:
                    total_kb += float(fh.read().split()[1]) * PAGE_KB
            except (OSError, ValueError, IndexError):
                continue
    return total_kb / 1e6


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit-gb", type=float, default=14.0,
                    help="kill the process group once system used memory crosses this")
    ap.add_argument("--interval", type=float, default=0.5, help="sampling period, s")
    ap.add_argument("--label", default="", help="prefix for the summary line")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="the command to run, after a bare --")
    args = ap.parse_args(argv)

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given; put it after a bare --")

    total, avail0 = _meminfo()
    baseline = total - avail0
    print(f"[guard] {total:.1f} GB total, {baseline:.1f} GB already in use, "
          f"ceiling {args.limit_gb:.1f} GB", flush=True)

    proc = subprocess.Popen(cmd, start_new_session=True)
    pgid = os.getpgid(proc.pid)

    peak_pss = peak_sys = 0.0
    killed = False
    try:
        while proc.poll() is None:
            pids = _group_pids(pgid)
            pss = _pss_gb(pids)
            _, avail = _meminfo()
            used = total - avail
            peak_pss = max(peak_pss, pss)
            peak_sys = max(peak_sys, used)
            if used > args.limit_gb:
                print(f"\n[guard] system at {used:.2f} GB > {args.limit_gb:.1f} GB "
                      f"ceiling ({len(pids)} procs, group pss {pss:.2f} GB) — "
                      f"killing the group", flush=True)
                os.killpg(pgid, signal.SIGKILL)
                killed = True
                break
            time.sleep(args.interval)
        proc.wait()
    except KeyboardInterrupt:                                # pragma: no cover
        os.killpg(pgid, signal.SIGKILL)
        raise
    finally:
        # Whatever happened, do not leave the group behind — a leaked CUDA worker is
        # the exact failure this tool exists to prevent.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    label = f"{args.label} " if args.label else ""
    print(f"[guard] {label}peak group_pss={peak_pss:.2f} GB  "
          f"peak sys_used={peak_sys:.2f} GB  (baseline {baseline:.2f} GB, "
          f"headroom {total - peak_sys:.2f} GB)  exit={proc.returncode}"
          f"{'  KILLED-BY-GUARD' if killed else ''}", flush=True)
    return 137 if killed else (proc.returncode or 0)


if __name__ == "__main__":
    sys.exit(main())
