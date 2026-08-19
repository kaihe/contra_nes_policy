#!/bin/bash
# Sequential overnight job queue — one GPU, many runs, nobody watching.
#
#   bash tools/run_queue.sh [queue/jobs.txt]
#
# `jobs.txt` is TSV: `<name>\t<command>`, blank lines and `#` comments ignored. A job whose
# name already appears in `queue/state.tsv` is skipped, which is what makes the queue
# **appendable while it runs** and **resumable after a reboot** — add lines at midnight, or
# restart the runner, and only the unfinished jobs execute.
#
# Three things it does that a plain list of commands does not:
#
#   waits for the GPU   it will not start while another `train_bc` is alive, so it can be
#                       launched *before* the current run finishes and simply queues behind it
#   survives failures   a crashed job is recorded `failed` and the queue continues; one bad
#                       cell does not take the remaining ten hours down with it
#   refuses to fill /   checks free disk before each job. A 40k XXL run writes ~7.7 GB of
#                       checkpoints, and a disk that fills at 03:00 corrupts the run that
#                       happens to be writing
#
# Exits once nothing has been pending for IDLE_EXIT_MIN, so it does not linger forever.
# `touch queue/STOP` to stop it after the current job.
set -u

cd "$(dirname "$(dirname "$(readlink -f "$0")")")" || exit 1

QUEUE="${1:-queue/jobs.txt}"
STATE="queue/state.tsv"
LOGDIR="runs/queue"
POLL_SEC=60
IDLE_EXIT_MIN=30
MIN_FREE_GB=50

mkdir -p "$(dirname "$STATE")" "$LOGDIR"
[ -f "$STATE" ] || printf 'name\tstatus\tstarted\tended\tseconds\texit\n' > "$STATE"

log() { printf '[queue %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

free_gb() { df -BG --output=avail . | tail -1 | tr -dc '0-9'; }

idle_cycles=0
max_idle=$(( IDLE_EXIT_MIN * 60 / POLL_SEC ))

while true; do
    [ -f queue/STOP ] && { log "STOP file present — exiting"; exit 0; }

    # Someone else is on the GPU (e.g. a run started by hand). Wait, do not race it.
    if pgrep -f "contra_policy.train_bc" > /dev/null; then
        log "another train_bc is running — waiting"
        sleep "$POLL_SEC"; continue
    fi

    name=""; command=""
    while IFS=$'\t' read -r n c; do
        [ -z "${n:-}" ] && continue
        case "$n" in \#*) continue ;; esac
        cut -f1 "$STATE" | grep -qxF "$n" && continue
        name="$n"; command="$c"; break
    done < "$QUEUE"

    if [ -z "$name" ]; then
        idle_cycles=$(( idle_cycles + 1 ))
        if [ "$idle_cycles" -ge "$max_idle" ]; then
            log "no pending jobs for ${IDLE_EXIT_MIN}m — exiting"; exit 0
        fi
        sleep "$POLL_SEC"; continue
    fi
    idle_cycles=0

    avail=$(free_gb)
    if [ "$avail" -lt "$MIN_FREE_GB" ]; then
        log "BLOCKED $name — only ${avail}GB free, need ${MIN_FREE_GB}GB"
        printf '%s\tblocked\t%s\t\t\t\n' "$name" "$(date -Is)" >> "$STATE"
        continue
    fi

    started=$(date -Is); t0=$SECONDS
    log "START $name  (${avail}GB free)"
    eval "$command" > "$LOGDIR/$name.log" 2>&1
    rc=$?
    elapsed=$(( SECONDS - t0 ))
    status=$([ $rc -eq 0 ] && echo done || echo failed)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$status" "$started" "$(date -Is)" "$elapsed" "$rc" >> "$STATE"
    log "$status $name in $(( elapsed / 60 ))m (exit $rc) → $LOGDIR/$name.log"
done
