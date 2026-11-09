#!/usr/bin/env bash
# The trained 27B's systems on the 4B, all three arms in one detached driver (runs/sys4b_2026-11-08/preregistration.md).
# At the owner's word (2026-11-08) it first pauses R1, the thinking labeling pass, and waits for R1's driver to stop its
# server and exit. Then arm a, b and c in turn, each with its own dry run and gate (scripts/sys4b/run.sh).
#
#   setsid nohup bash scripts/sys4b/run_all.sh > /dev/null 2>&1 < /dev/null &    # progress: runs/sys4b_2026-11-08/all.txt
#   touch runs/sys4b_2026-11-08/PAUSE      # the arm in progress stops after its rows in flight; later arms hold
#   rm runs/sys4b_2026-11-08/PAUSE; setsid nohup bash scripts/sys4b/run_all.sh > /dev/null 2>&1 < /dev/null &   # resumes
#
# A finished arm is skipped on a relaunch, and inside an arm a finished step. A failed arm does not hold the next: the
# owner asked for all three. R1 is paused once and never again: if it has been resumed by a relaunch, the next arm
# waits for an idle card instead. R1 resumes only by hand (scripts/think_label/r1.sh's header).
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=${ROOT:-runs/sys4b_2026-11-08}
SELF=${SELF:-scripts/sys4b/run.sh}
R1=${R1:-runs/think_label_2026-11-01}
R1_DRIVER=${R1_DRIVER:-"^bash scripts/think_label/r1.sh"}     # its command line, anchored: no watcher matches it
ARMS=${ARMS:-"a b c"}
R1_POLL=${R1_POLL:-30}
mkdir -p "$ROOT"
say() { echo "$(date -Is) $*" | tee -a "$ROOT/all.txt"; }

if [ ! -f "$ROOT/done_r1_pause" ]; then
  if pgrep -f "$R1_DRIVER" > /dev/null; then
    touch "$R1/PAUSE"
    say "R1: paused at the owner's word; waiting for its driver to stop its server and exit"
    while pgrep -f "$R1_DRIVER" > /dev/null; do sleep "$R1_POLL"; done
    say "R1: its driver has exited ($(tail -n 1 "$R1/status.txt" 2>/dev/null | cut -c1-160))"
  else
    say "R1: its driver is not running"
  fi
  touch "$ROOT/done_r1_pause"
fi

rc=0
for arm in $ARMS; do
  mkdir -p "$ROOT/$arm"
  ln -sfn ../PAUSE "$ROOT/$arm/PAUSE"
  if [ -f "$ROOT/PAUSE" ]; then say "paused: arm $arm and the arms after it hold"; exit 2; fi
  if [ -f "$ROOT/$arm/done_arm" ]; then say "arm $arm: done earlier"; continue; fi
  say "arm $arm: start"
  ARM=$arm ROOT=$ROOT RUN=$ROOT/$arm bash "$SELF" >> "$ROOT/$arm/driver.log" 2>&1
  rc=$?
  say "arm $arm: finished, rc $rc"
  if [ "$rc" -eq 0 ]; then touch "$ROOT/$arm/done_arm"; fi
  if [ -f "$ROOT/PAUSE" ]; then say "paused: the arms after $arm hold"; exit 2; fi
done
say "all arms finished"
exit "$rc"
