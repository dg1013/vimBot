#!/usr/bin/env bash
# The served pooled pick with up to 128 samples per model, in three steps (STEPS picks which):
#   smoke   3 val rows end to end into served_at_k_smoke/; stops here if it fails twice
#   timing  30 val rows, one request at a time, at TIMING_KS samples per model (pick_at_k.py
#           --timing; earlier timings of other k are kept). A failed timing does not stop the pass.
#   followups  followups.py: the other served systems' cost, the pooled pick at 1 and 4 under load,
#           the asking turns, asymmetric sampling (about 25 min). A failure does not stop the pass.
#   pass    every row of SPLITS at 128 samples per model (served_at_k.py), resumable
# The card is shared, so nothing starts on a busy one: the endpoints are started only once no
# other process is on the card and it has been idle for 2 minutes, and the timing waits for the
# endpoints to sit idle. A step whose endpoints die (stopped, or the 9B crashing as it did once at
# 23.6 of 24 GB, 2026-10-19) waits for an idle card again, restarts them with the same settings
# and resumes, RETRIES times. The endpoints are stopped at the end (STOP_AFTER=0 keeps them).
#
#   STEPS="smoke timing" bash scripts/serve_pooled/run_at_k.sh
#   STEPS="timing followups pass" TIMING_KS=64 RETRIES=4 bash scripts/serve_pooled/run_at_k.sh   # 2026-10-20 overnight
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=runs/serve_pooled
STEPS=${STEPS:-"smoke timing pass"}
SPLITS=${SPLITS:-val,test}               # the pass's splits
TIMING_KS=${TIMING_KS:-1,4,8,32,64,128}  # samples per model to time; earlier timings of other ks are kept
RETRIES=${RETRIES:-2}                    # restarts per step after a failure
FREE_MIB=${FREE_MIB:-4500}      # idle: at most the Windows desktop's share of memory in use,
FREE_UTIL=${FREE_UTIL:-10}      # at most this utilization (%), and no other process on the card
MAX_WAIT=${MAX_WAIT:-28800}     # give up waiting after 8 h
STOP_AFTER=${STOP_AFTER:-1}
export KV9=${KV9:-805306368} CG=${CG:-"1 2 4 8"}     # the settings that held on 2026-10-19
py() { conda run --no-capture-output -n vimBot python "$@"; }
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
up() { curl -s -m 5 http://127.0.0.1:8011/v1/models > /dev/null && curl -s -m 5 http://127.0.0.1:8012/v1/models > /dev/null; }

wait_idle() {   # empty: nothing on the card | ours: only our endpoints, idle; 12 checks 10 s apart
  local mode=$1 ok=0 waited=0 u m p
  while [ "$ok" -lt 12 ]; do
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

start() {   # an idle card first, then both endpoints
  for attempt in 1 2 3; do
    wait_idle empty || return 1
    if bash scripts/serve_pooled/serve_pooled.sh start; then return 0; fi
    bash scripts/serve_pooled/serve_pooled.sh stop
  done
  return 1
}

step() {   # name log command...: run it; if it fails, restart the endpoints politely and resume
  local name=$1 log=$2
  shift 2
  for attempt in $(seq 0 "$RETRIES"); do
    if [ "$attempt" -gt 0 ]; then
      echo "$(date +%T) $name failed; waiting to restart the endpoints (retry $attempt)" | tee -a "$log"
      bash scripts/serve_pooled/serve_pooled.sh stop >> "$log" 2>&1
      start >> "$log" 2>&1 || return 1
    fi
    up || start || return 1
    case "$name" in timing|followups) wait_idle ours || return 1 ;; esac    # one-at-a-time latencies
    echo "$(date +%T) $name"
    if "$@" >> "$log" 2>&1; then return 0; fi
  done
  return 1
}

rc=0
if has smoke; then
  step smoke $OUT/served_at_k_smoke.log py scripts/serve_pooled/served_at_k.py --splits val --limit 3 \
    --out $OUT/served_at_k_smoke || { echo "$(date +%T) smoke failed (see $OUT/served_at_k_smoke.log)"; rc=1; }
fi
if [ "$rc" -eq 0 ] && has timing; then
  step timing $OUT/pick_at_k_timing.log py scripts/serve_pooled/pick_at_k.py --out $OUT --timing \
    --timing-ks "$TIMING_KS" || echo "$(date +%T) timing failed (see $OUT/pick_at_k_timing.log); going on"
fi
if [ "$rc" -eq 0 ] && has followups; then
  step followups $OUT/followups.log py scripts/serve_pooled/followups.py --out $OUT/followups \
    || echo "$(date +%T) followups failed (see $OUT/followups.log); going on"
fi
if [ "$rc" -eq 0 ] && has pass; then
  step pass $OUT/served_at_k.log py scripts/serve_pooled/served_at_k.py --splits "$SPLITS" --out $OUT/served_at_k \
    || { echo "$(date +%T) pass failed (see $OUT/served_at_k.log)"; rc=1; }
fi
if [ "$STOP_AFTER" = 1 ]; then bash scripts/serve_pooled/serve_pooled.sh stop; fi
echo "$(date +%T) finished, rc $rc"
exit $rc
