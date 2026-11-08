#!/usr/bin/env bash
# The trained 27B's distributions on its own training rows (scripts/teacher_probs/README.md; the
# rules are in runs/teacher_probs_2026-10-31/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   record   GPU: record.py -> $RUN/probs.jsonl (resumable: rows on disk are skipped)
#   analyze  CPU: analyze.py -> $RUN/analysis.md
# The GPU stage waits for an idle card (run_at_k.sh's rule) and has the distill driver's watchdog:
# a log quiet for STALL_MIN minutes is killed, and the stage resumed once. WAIT_PID: wait for that
# process to exit first.
#
#   WAIT_PID=290307 bash scripts/teacher_probs/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/teacher_probs_2026-10-31
STEPS=${STEPS:-"record analyze"}
WAIT_PID=${WAIT_PID:-}
PY=$HOME/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STALL_MIN=${STALL_MIN:-20}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }

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

rows_done() { if [ -f "$RUN/probs.jsonl" ]; then wc -l < "$RUN/probs.jsonl"; else echo 0; fi; }

status "start: STEPS=\"$STEPS\" WAIT_PID=${WAIT_PID:-none}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  status "pid $WAIT_PID has exited"
fi

if has record; then
  total=$(wc -l < runs/distill_2026-10-31/train_distilled.jsonl)
  attempt=0
  while [ "$(rows_done)" -lt "$total" ] && [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    status "record: waiting for an idle card"
    wait_idle || break
    status "record attempt $attempt ($(rows_done) of $total rows on disk)"
    watched "$RUN/record.log" $PY scripts/teacher_probs/record.py
    status "record attempt $attempt rc $? ($(rows_done) rows)"
  done
  if [ "$(rows_done)" -lt "$total" ]; then status "record incomplete ($(rows_done) of $total)"; fi
fi

if has analyze; then
  if [ -f "$RUN/probs.jsonl" ]; then
    $PY scripts/teacher_probs/analyze.py > "$RUN/analyze.log" 2>&1 \
      && status "analyze: $RUN/analysis.md ($(grep -m1 -o 'Verdict, by the pre-registered rule: [a-z -]*' "$RUN/analysis.md"))" \
      || { tail -3 "$RUN/analyze.log"; status "analyze FAILED"; }
  else
    status "analyze: skipped, no probs.jsonl"
  fi
fi
status "done: $STEPS"
