#!/usr/bin/env bash
# The whole night: the trained 27B served, then in the pool (scripts/serve27b/README.md; the rules are
# in runs/serve27b_2026-11-05/preregistration.md). The 27B fills the card, so the pool's endpoints
# and the 27B's take turns:
#
#   pool_timing  4B + 9B endpoints   the two systems it is priced against, the same night
#   smoke        27B endpoint        the gate: is this the model that was measured? Stops the night if not
#   timing       27B endpoint        latency, throughput, k samples
#   greedy       27B endpoint        every val and test row
#   samples      27B endpoint        N samples per row
#   score        4B + 9B endpoints   the verifier on the 27B's new answers
#   judge        CPU                 every answer of all three models through the box
#   report       CPU                 pool27b.py -> report.md
#
# Nothing starts on a busy card. A step whose endpoints die waits for an idle card, restarts them and
# resumes, RETRIES times. Endpoints are stopped at the end, whatever happened. A step already done is
# skipped, so a relaunch resumes.
#
# The card is shared with the desktop. If the sampling pace falls threefold, something else wants the
# card: the driver stops its endpoints, waits for an idle card and resumes.
#
#   touch runs/serve27b_2026-11-05/PAUSE      # stops after the rows in flight and frees the card
#   rm runs/serve27b_2026-11-05/PAUSE; setsid nohup bash scripts/serve27b/run.sh > /dev/null 2>&1 &   # resumes
#
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN=${RUN:-runs/serve27b_2026-11-05}
PY=$HOME/miniconda3/envs/vimBot/bin/python
PASS=scripts/serve27b/pass27b.py
LIMIT=${LIMIT:-0}                # only the first LIMIT rows of each split (a dry run)
N=${N:-32}                       # the 27B's samples per row
STEPS=${STEPS:-"pool_timing smoke timing greedy samples score judge report"}
RETRIES=${RETRIES:-6}
# Idle: the 27B takes 20,350 MiB of its own under load and the card pages past about 23,500, so at
# most 3,150 MiB may be in use before it starts. The same bar for the pool, so that both are timed
# on the same quiet desktop. A game or a large document on the Windows side is more than that.
FREE_MIB=${FREE_MIB:-3150}
FREE_UTIL=${FREE_UTIL:-60}       # at most this utilization (the desktop alone reaches 40%)
MAX_WAIT=${MAX_WAIT:-86400}      # give up waiting for a card after 24 h
BUDGET_S=${BUDGET_S:-18000}      # no new row starts this long after the samples stage does
TIMING_ARGS=${TIMING_ARGS:-}
ARGS="--run $RUN --limit $LIMIT"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export KV9=805306368 CG="1 2 4 8"       # the settings that held for the pool on 2026-10-19 and 21
mkdir -p "$RUN"

card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
status() { echo "$(date -Is) [$(card)] $*" | tee -a "$RUN/status.txt"; }
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }

wait_idle() {   # empty: nothing on the card | ours: only our endpoints, idle; 12 checks 10 s apart
  local mode=$1 ok=0 waited=0 u m p
  while [ "$ok" -lt 12 ]; do
    if [ -f "$RUN/PAUSE" ]; then echo "$(date +%T) paused while waiting for a card"; return 1; fi
    IFS=, read -r u m <<< "$(card)"
    p=$(procs)
    if [ "$u" -le "$FREE_UTIL" ] && { [ "$mode" = ours ] || { [ "$p" -eq 0 ] && [ "$m" -le "$FREE_MIB" ]; }; }; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    sleep 10
    waited=$((waited + 10))
    if [ $((waited % 600)) -eq 0 ]; then echo "$(date +%T) waiting for an idle card: ${u}%, ${m} MiB, $p processes"; fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "$(date +%T) no idle card in $((MAX_WAIT / 3600)) h; giving up"; return 1; fi
  done
  echo "$(date +%T) card idle ($mode): ${u}%, ${m} MiB"
}

port_up() { curl -s -m 5 "http://127.0.0.1:$1/v1/models" > /dev/null; }
is_up() {
  case "$1" in
    pool) port_up 8011 && port_up 8012 ;;
    27b) port_up 8015 ;;
    *) return 0 ;;
  esac
}
start_ep() {
  case "$1" in
    pool) OUT=$RUN bash scripts/serve_pooled/serve_pooled.sh start ;;
    27b) OUT=$RUN bash scripts/serve27b/serve27b.sh start ;;
  esac
}
stop_ep() {
  case "$1" in
    pool) OUT=$RUN bash scripts/serve_pooled/serve_pooled.sh stop ;;
    27b) OUT=$RUN bash scripts/serve27b/serve27b.sh stop ;;
  esac
}
start_phase() {   # an idle card first, then the endpoints
  local attempt
  for attempt in 1 2 3; do
    wait_idle empty || return 1
    if start_ep "$1"; then return 0; fi
    stop_ep "$1"
  done
  return 1
}

step() {   # name phase command...: 0 done; 1 gave up; 2 paused; 3 a gate failed. The command's own
           # exit codes: 3 a gate, 4 paused, 5 the pace fell (stop, wait for an idle card, resume)
  local name=$1 phase=$2 log=$RUN/$1.log attempt rc=1
  shift 2
  if [ -f "$RUN/done_$name" ]; then status "$name: done earlier"; return 0; fi
  for attempt in $(seq 0 "$RETRIES"); do
    if [ -f "$RUN/PAUSE" ]; then status "$name: paused"; return 2; fi
    if [ "$attempt" -gt 0 ]; then
      status "$name: failed (rc $rc); restarting its endpoints (retry $attempt)"
      stop_ep "$phase" >> "$log" 2>&1
    fi
    if ! is_up "$phase"; then
      if ! start_phase "$phase" >> "$log" 2>&1; then status "$name: could not start the $phase endpoints"; rc=1; continue; fi
    fi
    case "$name" in pool_timing|timing) wait_idle ours >> "$log" 2>&1 ;; esac    # one-at-a-time latencies
    status "$name: start (attempt $attempt)"
    "$@" >> "$log" 2>&1
    rc=$?
    case "$rc" in
      0) touch "$RUN/done_$name"; status "$name: done ($(tail -1 "$log" | cut -c1-140))"; return 0 ;;
      3) status "$name: the gate failed ($(tail -1 "$log" | cut -c1-140))"; return 3 ;;
      4) status "$name: paused with rows left"; return 2 ;;
      5) status "$name: the pace fell, so something else wants the card; yielding it ($(tail -1 "$log" | cut -c1-100))" ;;
    esac
  done
  status "$name: gave up after $RETRIES retries (rc $rc; see $log)"
  return 1
}

cleanup() {
  stop_ep pool >> "$RUN/driver.log" 2>&1
  stop_ep 27b >> "$RUN/driver.log" 2>&1
}
trap cleanup EXIT
trap 'exit 143' TERM INT

main() {
  local r
  status "start: steps $STEPS; $N samples per row; limit $LIMIT"
  if has pool_timing; then
    step pool_timing pool $PY $PASS pool-timing $ARGS $TIMING_ARGS; r=$?
    [ "$r" -eq 2 ] && return 2
    [ "$r" -ne 0 ] && status "pool_timing failed; the cost table will lack it; going on"
  fi
  stop_ep pool >> "$RUN/driver.log" 2>&1
  if has smoke; then step smoke 27b $PY $PASS smoke $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has timing; then
    step timing 27b $PY $PASS timing $ARGS $TIMING_ARGS; r=$?
    [ "$r" -eq 2 ] && return 2
    [ "$r" -ne 0 ] && status "timing failed; the cost table will lack it; going on"
  fi
  if has greedy; then step greedy 27b $PY $PASS greedy $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has samples; then step samples 27b $PY $PASS samples $ARGS --n "$N" --budget-s "$BUDGET_S"; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  stop_ep 27b >> "$RUN/driver.log" 2>&1
  if has score; then step score pool $PY $PASS score $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  stop_ep pool >> "$RUN/driver.log" 2>&1
  if has judge; then step judge none $PY $PASS judge $ARGS; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  if has report; then step report none $PY scripts/serve27b/pool27b.py --run "$RUN"; r=$?; [ "$r" -ne 0 ] && return "$r"; fi
  return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then      # sourced by the tests for its functions
  main
  rc=$?
  status "finished, rc $rc"
  exit "$rc"
fi
