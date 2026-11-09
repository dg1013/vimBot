#!/usr/bin/env bash
# The thinking 27B labels the 795 hard train rows (scripts/think_label/README.md; the rules are in
# runs/think_label_2026-11-01/preregistration.md).
#
# Waits for WAIT_PID (the KD driver), then runs only if the thinking run's TB reading is "helps".
# Stages, in this order, each only if named in STEPS:
#   trigger  CPU: label.py trigger -- exit 3 ends the driver: the pass does not run
#   plan     CPU: label.py plan -> $RUN/rows.json, $RUN/plan_retr8.jsonl
#   run      GPU: label.py run, the control then the thinking arm, until the deadline: HOURS after
#                 the stage first starts (§14b), and never past CUTOFF (the owner's Monday 08:00),
#                 kept in $RUN/deadline across retries
#   report   CPU: report.py -> $RUN/report.md
# The run stage waits for an idle card. A log quiet for STALL_MIN minutes is killed, and the stage
# resumes from the rows on disk (MAX_ATTEMPTS).
#
#   WAIT_PID=587040 setsid nohup bash scripts/think_label/run.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/think_label_2026-11-01
STEPS=${STEPS:-"trigger plan run report"}
WAIT_PID=${WAIT_PID:-}
HOURS=${HOURS:-12}                                     # pre-registered hard stop (§14b: at most 12 h)
CUTOFF=${CUTOFF:-"2026-11-02 08:00"}                   # the owner's: the card back by Monday 08:00
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

if has trigger; then
  $PY scripts/think_label/label.py trigger > "$RUN/trigger.log" 2>&1
  rc=$?
  status "trigger: $(tail -1 "$RUN/trigger.log")"
  if [ $rc -ne 0 ]; then status "done: the labeling pass did not run"; exit 0; fi
fi
if has plan; then
  $PY scripts/think_label/label.py plan --run "$RUN" > "$RUN/plan.log" 2>&1 \
    && status "plan: $(tail -1 "$RUN/plan.log")" || { tail -3 "$RUN/plan.log"; status "plan FAILED"; exit 1; }
fi
if has run; then
  attempt=0
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    status "run: waiting for an idle card"
    wait_idle || { status "run: no idle card"; break; }
    if [ ! -f "$RUN/deadline" ]; then
      deadline=$(( $(date +%s) + HOURS * 3600 ))
      cutoff=$(date -d "$CUTOFF" +%s)
      if [ "$cutoff" -lt "$deadline" ]; then deadline=$cutoff; fi
      echo "$deadline" > "$RUN/deadline"
    fi
    deadline=$(cat "$RUN/deadline")
    if [ "$(date +%s)" -ge "$deadline" ]; then status "run: the deadline has passed"; break; fi
    status "run attempt $attempt ($(rows_done) arm-rows on disk), stops starting batches at $(date -d @"$deadline" '+%a %T')"
    watched "$RUN/run.log" $PY scripts/think_label/label.py run --run "$RUN" --deadline "$deadline"
    rc=$?
    status "run attempt $attempt rc $rc ($(rows_done) arm-rows on disk)"
    if [ $rc -eq 0 ]; then break; fi
  done
fi
if has report; then
  $PY scripts/think_label/report.py > "$RUN/report.log" 2>&1 \
    && status "report: $RUN/report.md ($(grep -o '→ \*\*[a-z]*\*\*' "$RUN/report.md" | tr -d '*→ ' | head -1))" \
    || { tail -3 "$RUN/report.log"; status "report FAILED"; }
fi
status "done: $STEPS"
