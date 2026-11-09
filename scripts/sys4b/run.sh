#!/usr/bin/env bash
# One arm of the trained 27B's systems on the 4B (the rules are in runs/sys4b_2026-11-08/preregistration.md): a dry run
# on the first DRY_LIMIT rows of each split, its gate, then the real run. One detached driver, so nothing waits on a
# session; run_all.sh runs the three arms in turn.
#
#   ARM=a  the served 4B, no examples: scripts/ask27b/run.sh's chain (prep and greedy first, since its samples are
#          the served pass's and its zero-shot answers are drawn here)
#   ARM=b  the served 4B, 8 BM25 examples: scripts/retr27b/run.sh's chain for the trained 27B
#   ARM=c  the base 4B, 8 BM25 examples: the same chain for the base 27B, on the base endpoint, with the verifier on
#          the served endpoint afterwards
#
#   prep      CPU      arm a: its 32 samples per row and their scores, from the served pass
#   plan      CPU      arms b, c: the plan files, BM25's own time
#   smoke     4B       arms b, c: same prompt, same model, long prompts under load. Stops the arm if not
#   greedy    4B       the §6 answer (arm c: with examples and zero-shot)
#   samples   4B       arms b, c: N samples per row, in requests of 8
#   views     CPU      every distinct sample through the box
#   question  4B       the question it asks, every row
#   placebo   4B       its answer after a reply with no information
#   simulate  9B, NF4  the audited simulated user (scripts/clarify/simulate.py, unchanged)
#   answer    4B       its answer after that reply
#   timing    4B       the round trip itself, one request at a time, on the 200 rows the 4B and the 27B were timed on
#   score     4B       arms b, c: the verifier on every distinct answer
#   judge     CPU      the greedy answers and every answer after a reply, through the box
# The reports are CPU and read the finished files; they are not part of the GPU chain.
#
# Its waiting, retrying and yielding are scripts/serve27b/run.sh's, sourced below: nothing starts on a busy card, an
# endpoint that dies is restarted, a pace that falls threefold yields the card, a step already done is skipped, and
# the endpoints are stopped at the end, whatever happened.
#
#   touch runs/sys4b_2026-11-08/PAUSE      # stops after the rows in flight and frees the card; every arm sees it
#
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
ROOT=${ROOT:-runs/sys4b_2026-11-08}
ARM=${ARM:-a}
RUN=${RUN:-$ROOT/$ARM}
case "$ARM" in
  a) STEPS=${STEPS:-"prep greedy views question placebo simulate answer timing judge"} ;;
  b|c) STEPS=${STEPS:-"plan smoke greedy samples views question placebo simulate answer timing score judge"} ;;
  *) echo "ARM must be a, b or c, not $ARM"; exit 2 ;;
esac
export ARM ROOT
DRY=${DRY:-1}                    # 1: the dry run and its gate first
DRY_LIMIT=${DRY_LIMIT:-12}
N=${N:-16}                       # samples per row, arms b and c
INFLIGHT=${INFLIGHT:-2}          # rows in flight in the samples stage: the 4B's prefill leaves room for two
FREE_MIB=${FREE_MIB:-4000}       # the 4B takes about 14 GB and the card pages past about 23,500 MiB
SELF=$(readlink -f "${BASH_SOURCE[0]}")
source "$(dirname "$SELF")/../serve27b/run.sh"
GO=scripts/sys4b/sys4b.py
ARGS="$ARGS --arm $ARM"
M=4b                             # the endpoint that writes the answers: the served 4B, or for arm c the base
if [ "$ARM" = c ]; then M=base4b; fi

# The simulator is its own process, so its phase counts as not up: the driver then waits for an idle card first, and
# starting it is a no-op. The verifier (v4) is a LoRA on the served endpoint.
is_up() { case "$1" in 4b|v4) port_up 8011 ;; base4b) port_up 8016 ;; sim) return 1 ;; *) return 0 ;; esac; }
start_ep() {
  case "$1" in
    4b|v4) OUT=$ROOT bash scripts/sys4b/serve4b.sh start served ;;
    base4b) OUT=$ROOT bash scripts/sys4b/serve4b.sh start base ;;
  esac
}
stop_ep() {
  case "$1" in
    4b|v4) OUT=$ROOT bash scripts/sys4b/serve4b.sh stop served ;;
    base4b) OUT=$ROOT bash scripts/sys4b/serve4b.sh stop base ;;
  esac
}
cleanup() { stop_ep 4b >> "$RUN/driver.log" 2>&1; stop_ep base4b >> "$RUN/driver.log" 2>&1; }

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
  status "start: arm $ARM; steps $STEPS; $N samples per row; limit $LIMIT"
  if has prep; then step prep none $PY $GO prep $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has plan; then step plan none $PY $GO plan $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  for name in smoke greedy; do
    if has $name; then step $name $M $PY $GO $name $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  done
  if has samples; then
    step samples $M $PY $GO samples $ARGS --n "$N" --inflight "$INFLIGHT" --budget-s "$BUDGET_S"; r=$?
    [ "$r" -ne 0 ] && return "$r"
  fi
  if has views; then step views none $PY $GO views $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  for name in question placebo; do
    if has $name; then step $name $M $PY $GO $name $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  done
  stop_ep $M >> "$RUN/driver.log" 2>&1
  if has simulate; then step simulate sim sim_step; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has answer; then step answer $M $PY $GO answer $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has timing; then
    step timing $M $PY $GO timing $ARGS $TIMING_ARGS; r=$?
    [ "$r" -eq 2 ] && return 2
    [ "$r" -ne 0 ] && status "timing failed; the cost table will lack it; going on"
  fi
  if [ "$M" != 4b ]; then stop_ep $M >> "$RUN/driver.log" 2>&1; fi
  if has score; then step score v4 $PY $GO score $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  stop_ep 4b >> "$RUN/driver.log" 2>&1
  if has judge; then step judge none $PY $GO judge $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
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
  if ! $PY $GO gate --run "$dry" --arm "$ARM" >> "$RUN/driver.log" 2>&1; then
    status "the dry run's gate failed ($(tail -1 "$RUN/driver.log" | cut -c1-120)); nothing else ran"
    return 3
  fi
  touch "$dry/done_gate"
  status "dry run: passed ($(grep 'the pace projects' "$RUN/driver.log" | tail -1 | cut -c1-80))"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then      # sourced by the tests for its functions
  rc=0
  if [ "$DRY" = 1 ]; then dry_first; rc=$?; fi
  if [ "$rc" -eq 0 ]; then chain; rc=$?; fi
  status "finished, rc $rc"
  exit "$rc"
fi
