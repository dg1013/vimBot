#!/usr/bin/env bash
# The 27B with thinking on the val rows it misses and the rows no model solves (scripts/think27b/
# README.md; the rules are in runs/think27b_2026-11-01/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   rows     CPU: think27b.py rows -> $RUN/rows.json (213 val rows)
#   run      GPU: think27b.py run, the four arms in order, until the deadline (HOURS after the
#                 stage first starts, kept in $RUN/deadline across retries)
#   report   CPU: report.py -> $RUN/report.md
# The run stage waits for an idle card. A log quiet for STALL_MIN minutes is killed, and the stage
# resumes from the rows on disk (MAX_ATTEMPTS). WAIT_PID: wait for that process first (the
# 2026-10-31 queue, so this starts when the GRPO chain ends).
#
#   WAIT_PID=293843 setsid nohup bash scripts/think27b/run.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/think27b_2026-11-01
STEPS=${STEPS:-"rows run report"}
WAIT_PID=${WAIT_PID:-}
HOURS=${HOURS:-10}                                     # pre-registered hard stop (§14b: under 12 h)
PY=$HOME/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
rows_done() { cat "$RUN"/rows_*.jsonl 2>/dev/null | wc -l; }

wait_idle() {   # 12 checks 10 s apart (run_at_k.sh's rule, its "empty" mode)
  local ok=0 waited=0 u m p
  while [ "$ok" -lt 12 ]; do
    IFS=, read -r u m <<< "$(card)"
    p=$(procs)
    if [ "$u" -le "$FREE_UTIL" ] && [ "$p" -eq 0 ] && [ "$m" -le "$FREE_MIB" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 10
    waited=$((waited + 10))
    if [ $((waited % 600)) -eq 0 ]; then echo "$(date +%T) waiting for an idle card: ${u}%, ${m} MiB, $p processes"; fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "$(date +%T) no idle card in $((MAX_WAIT / 3600)) h; giving up"; return 1; fi
  done
  echo "$(date +%T) card idle: ${u}%, ${m} MiB"
}

watched() {   # <log> <cmd...>: run_scaling.sh's watchdog -- kill a stage whose log goes quiet for STALL_MIN
  local log=$1; shift
  "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -f "$log" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
      if [ "$age" -gt $((STALL_MIN * 60)) ]; then
        status "WATCHDOG: no log activity for $((age / 60)) min, killing pid $pid"
        kill "$pid" 2>/dev/null; sleep 90; kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null; return 124
      fi
    fi
  done
  wait "$pid"; return $?
}

status "start: STEPS=\"$STEPS\" WAIT_PID=${WAIT_PID:-none}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  status "pid $WAIT_PID has exited"
fi

if has rows; then
  $PY scripts/think27b/think27b.py rows --run "$RUN" > "$RUN/rows.log" 2>&1 \
    && status "rows: $(tail -1 "$RUN/rows.log")" || { tail -3 "$RUN/rows.log"; status "rows FAILED"; exit 1; }
fi
if has run; then
  attempt=0
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    status "run: waiting for an idle card"
    wait_idle || { status "run: no idle card"; break; }
    if [ ! -f "$RUN/deadline" ]; then echo $(( $(date +%s) + HOURS * 3600 )) > "$RUN/deadline"; fi
    deadline=$(cat "$RUN/deadline")
    if [ "$(date +%s)" -ge "$deadline" ]; then status "run: the deadline has passed"; break; fi
    status "run attempt $attempt ($(rows_done) arm-rows on disk), stops starting batches at $(date -d @"$deadline" +%T)"
    watched "$RUN/run.log" $PY scripts/think27b/think27b.py run --run "$RUN" --deadline "$deadline"
    rc=$?
    status "run attempt $attempt rc $rc ($(rows_done) arm-rows on disk)"
    if [ $rc -eq 0 ]; then break; fi
  done
fi
if has report; then
  $PY scripts/think27b/report.py > "$RUN/report.log" 2>&1 \
    && status "report: $RUN/report.md ($(grep -o '\*\*[a-z]*\*\*' "$RUN/report.md" | tr -d '*' | paste -sd' '))" \
    || { tail -3 "$RUN/report.log"; status "report FAILED"; }
fi
status "done: $STEPS"
