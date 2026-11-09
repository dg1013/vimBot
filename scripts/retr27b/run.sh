#!/usr/bin/env bash
# A 27B with 8 BM25 examples, with and without the 5-of-8 rule, and asking less (the rules are in
# runs/retr27b_2026-11-06/preregistration.md). One detached driver, so nothing waits on a session. ARM=base is the
# untrained 27B, ARM=trained the trained one; run_all.sh runs the two in turn:
#
#   dry run   the whole chain on the first DRY_LIMIT rows of each split, in $RUN/dry
#   gate      did the turns happen, and does the pace fit under 12 h? Stops here if not
#   plan      CPU      the plan files, BM25's own time
#   smoke     27B      same prompt, same model, long prompts under load. Stops the night if not
#   greedy    27B      the §6 answer with examples and zero-shot, every row
#   samples   27B      N samples per row
#   views     CPU      every distinct sample through the box
#   question  27B      the question it asks, every row
#   placebo   27B      its answer after a reply with no information
#   simulate  9B, NF4  the audited simulated user replies (scripts/clarify/simulate.py, unchanged)
#   answer    27B      its answer after that reply
#   timing    27B      the round trip itself, on the 200 rows the 4B and the trained 27B were timed on
#   score     4B       the verifier on every distinct answer
#   judge     CPU      the greedy answers and every answer after a reply, through the box
#   report    CPU      report_retr.py and fewer.py
#
# Its waiting, retrying and yielding are scripts/serve27b/run.sh's, sourced below: nothing starts on a busy card,
# an endpoint that dies is restarted, a pace that falls threefold yields the card, a step already done is skipped,
# and endpoints are stopped at the end, whatever happened.
#
#   touch runs/retr27b_2026-11-06/PAUSE       # stops after the rows in flight and frees the card
#   rm runs/retr27b_2026-11-06/PAUSE; setsid nohup bash scripts/retr27b/run.sh > /dev/null 2>&1 &   # resumes
#
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
RUN=${RUN:-runs/retr27b_2026-11-06}
STEPS=${STEPS:-"plan smoke greedy samples views question placebo simulate answer timing score judge report"}
DRY=${DRY:-1}                    # 1: the dry run and its gate first
DRY_LIMIT=${DRY_LIMIT:-12}
N=${N:-16}                       # samples per row
ARM=${ARM:-base}                 # base: the untrained 27B; trained: the served adapter
export ARM
if [ "$ARM" = trained ]; then export ADAPTER=${ADAPTER:-runs/qwen3.5-27b-sft-distill/best}; fi
SELF=$(readlink -f "${BASH_SOURCE[0]}")
source "$(dirname "$SELF")/../serve27b/run.sh"
GO=scripts/retr27b/retr27b.py
ARGS="$ARGS --arm $ARM"

# The simulator is its own process, so its phase counts as not up: the driver then waits for an idle card first,
# and starting it is a no-op. The verifier needs only the pooled pick's 4B endpoint.
is_up() { case "$1" in 27b) port_up 8015 ;; v4) port_up 8011 ;; sim) return 1 ;; *) return 0 ;; esac; }
start_ep() {
  case "$1" in
    27b) OUT=$RUN bash scripts/retr27b/serve.sh start ;;
    v4) ONLY=4b OUT=$RUN bash scripts/serve_pooled/serve_pooled.sh start ;;
  esac
}
stop_ep() {
  case "$1" in
    27b) OUT=$RUN bash scripts/retr27b/serve.sh stop ;;
    v4) OUT=$RUN bash scripts/serve_pooled/serve_pooled.sh stop ;;
  esac
}
cleanup() { stop_ep 27b >> "$RUN/driver.log" 2>&1; stop_ep v4 >> "$RUN/driver.log" 2>&1; }

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

report_step() {
  local model=base
  if [ "$ARM" = trained ]; then model=trained-retr; fi
  $PY scripts/retr27b/report_retr.py --run "$RUN" --arm "$ARM" && $PY scripts/retr27b/fewer.py --run "$RUN" --model "$model"
}

chain() {
  local r name
  status "start: steps $STEPS; $N samples per row; limit $LIMIT"
  if has plan; then step plan none $PY $GO plan $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  for name in smoke greedy; do
    if has $name; then step $name 27b $PY $GO $name $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  done
  if has samples; then step samples 27b $PY $GO samples $ARGS --n "$N" --budget-s "$BUDGET_S"; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has views; then step views none $PY $GO views $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  for name in question placebo; do
    if has $name; then step $name 27b $PY $GO $name $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  done
  stop_ep 27b >> "$RUN/driver.log" 2>&1
  if has simulate; then step simulate sim sim_step; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has answer; then step answer 27b $PY $GO answer $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has timing; then
    step timing 27b $PY $GO timing $ARGS $TIMING_ARGS; r=$?
    [ "$r" -eq 2 ] && return 2
    [ "$r" -ne 0 ] && status "timing failed; the cost table will lack it; going on"
  fi
  stop_ep 27b >> "$RUN/driver.log" 2>&1
  if has score; then step score v4 $PY $GO score $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  stop_ep v4 >> "$RUN/driver.log" 2>&1
  if has judge; then step judge none $PY $GO judge $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has report; then
    step report none report_step; r=$?
    if [ "$r" -ne 0 ] && [ "$r" -ne 2 ] && [ "$LIMIT" -gt 0 ]; then
      status "the report failed on the dry rows; it is CPU and can be rerun, so the GPU chain goes on"    # never the night for this
    elif [ "$r" -ne 0 ]; then
      return "$r"
    fi
  fi
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
  if ! $PY $GO gate --run "$dry" >> "$RUN/driver.log" 2>&1; then
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
