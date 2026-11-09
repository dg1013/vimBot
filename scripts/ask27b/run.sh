#!/usr/bin/env bash
# The 5-of-8 ask rule on the trained 27B (scripts/ask27b/README.md; the rules are in
# runs/ask27b_2026-11-06/preregistration.md). One detached driver, so nothing waits on a session:
#
#   dry run   the whole chain on the first DRY_LIMIT rows of each split, in $RUN/dry
#   gate      did the turns happen on most of those rows? Stops here if not
#   views     CPU          the 32 samples per row on disk, through the box
#   question  27B          the question it asks, every row
#   placebo   27B          its answer after a reply with no information
#   simulate  9B, NF4      the audited simulated user replies (scripts/clarify/simulate.py, unchanged)
#   answer    27B          its answer after that reply
#   timing    27B          the round trip itself, on the 4B's 200 timing rows
#   judge     CPU          every answer after a reply, through the box
#   report    CPU          report_ask.py -> report.md
#
# Its waiting, retrying and yielding are scripts/serve27b/run.sh's, sourced below: nothing starts on a
# busy card, an endpoint that dies is restarted, a pace that falls threefold yields the card, a step
# already done is skipped, and endpoints are stopped at the end, whatever happened.
#
#   touch runs/ask27b_2026-11-06/PAUSE       # stops after the rows in flight and frees the card
#   rm runs/ask27b_2026-11-06/PAUSE; setsid nohup bash scripts/ask27b/run.sh > /dev/null 2>&1 &   # resumes
#
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
RUN=${RUN:-runs/ask27b_2026-11-06}
STEPS=${STEPS:-"views question placebo simulate answer timing judge report"}
DRY=${DRY:-1}                    # 1: the dry run and its gate first
DRY_LIMIT=${DRY_LIMIT:-12}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
source "$(dirname "$SELF")/../serve27b/run.sh"
ASK=scripts/ask27b/ask27b.py

# Only the 27B's endpoint here. The simulator is its own process, so its phase counts as not up: the
# driver then waits for an idle card first, and starting it is a no-op.
is_up() { case "$1" in 27b) port_up 8015 ;; sim) return 1 ;; *) return 0 ;; esac; }
start_ep() { case "$1" in 27b) OUT=$RUN bash scripts/serve27b/serve27b.sh start ;; esac; }
stop_ep() { case "$1" in 27b) OUT=$RUN bash scripts/serve27b/serve27b.sh stop ;; esac; }
cleanup() { stop_ep 27b >> "$RUN/driver.log" 2>&1; }

sim_step() {   # 0 done, 4 paused, else its rc. simulate.py exits 3 when paused
  local sp rc lim=()
  if [ "$LIMIT" -gt 0 ]; then lim=(--limit "$LIMIT"); fi
  for sp in val test; do
    $PY scripts/clarify/simulate.py --model qwen3.5-9b --cases "data/splits/$sp.jsonl" --turn1 "$RUN/question.jsonl" \
      --out "$RUN/replies.jsonl" --think-budget 1024 --batch 16 --stop-file "$RUN/PAUSE" "${lim[@]}"
    rc=$?
    if [ "$rc" -eq 3 ]; then return 4; fi
    if [ "$rc" -ne 0 ]; then return "$rc"; fi
  done
}

chain() {
  local r name
  status "start: steps $STEPS; limit $LIMIT"
  if has views; then step views none $PY $ASK views $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  for name in question placebo; do
    if has $name; then step $name 27b $PY $ASK $name $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  done
  stop_ep 27b >> "$RUN/driver.log" 2>&1
  if has simulate; then step simulate sim sim_step; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has answer; then step answer 27b $PY $ASK answer $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has timing; then
    step timing 27b $PY $ASK timing $ARGS $TIMING_ARGS; r=$?
    [ "$r" -eq 2 ] && return 2
    [ "$r" -ne 0 ] && status "timing failed; the cost table will lack it; going on"
  fi
  stop_ep 27b >> "$RUN/driver.log" 2>&1
  if has judge; then step judge none $PY $ASK judge $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has report; then step report none $PY scripts/ask27b/report_ask.py --run "$RUN"; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  return 0
}

dry_first() {   # the whole chain on a few rows, then the gate. A pause of the real run reaches it too
  local rc dry=$RUN/dry
  mkdir -p "$dry"
  ln -sfn ../PAUSE "$dry/PAUSE"
  if [ -f "$dry/done_gate" ]; then status "dry run: passed earlier"; return 0; fi
  status "dry run: the first $DRY_LIMIT rows of each split"
  DRY=0 RUN=$dry LIMIT=$DRY_LIMIT TIMING_ARGS="--timing-n 4" bash "$SELF"
  rc=$?
  if [ "$rc" -ne 0 ]; then status "dry run stopped, rc $rc; nothing else ran"; return "$rc"; fi
  if ! $PY $ASK gate --run "$dry" >> "$RUN/driver.log" 2>&1; then
    status "the dry run's gate failed ($(tail -1 "$RUN/driver.log" | cut -c1-120)); nothing else ran"
    return 3
  fi
  touch "$dry/done_gate"
  status "dry run: passed"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then      # sourced by the tests for its functions
  rc=0
  if [ "$DRY" = 1 ]; then dry_first; rc=$?; fi
  if [ "$rc" -eq 0 ]; then chain; rc=$?; fi
  status "finished, rc $rc"
  exit "$rc"
fi
