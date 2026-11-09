#!/usr/bin/env bash
# R1: the labeling pass on R0's engine, night by night (scripts/think_label/README.md; the rules are in
# runs/think_label_2026-11-01/preregistration.md, addendum 2).
#
# Waits for WAIT_PID (OPD's driver), then, each only if named in STEPS:
#   trigger  CPU: serve.py trigger -- think_long's TB at 8,192 tokens must be helps, or the pass does not run
#   plan     CPU: label.py plan, unless $RUN/rows.json is on disk
#   prompts  CPU: serve.py prompts, unless $RUN/prompts_vllm.jsonl is on disk
#   run      GPU: vllm serve (R0's attempt 2, else 3) on an idle card, then serve.py run: the control,
#                 then the thinking arm, resuming from the rows on disk
#   report   CPU: report.py -> $RUN/report.md, once both arms have every row
# A pause: `touch $RUN/PAUSE`. No new row starts, the rows in flight get 10 minutes, then the server
# stops and the driver exits. To resume: rm $RUN/PAUSE, and start the driver again (STEPS="run report").
# A paging stop (exit 5), failed requests or a quiet log (STALL_MIN) retry, at most MAX_ATTEMPTS times.
# The server runs in its own session, so it is stopped on every exit, a kill of this driver included.
#
#   WAIT_PID=<OPD's driver> setsid nohup bash scripts/think_label/r1.sh > /dev/null 2>&1 &   # status: $RUN/status.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/think_label_2026-11-01
STEPS=${STEPS:-"trigger plan prompts run report"}
WAIT_PID=${WAIT_PID:-}
PY=$HOME/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
SERVE=scripts/think_label/serve.py
PORT=${PORT:-8014}
FREE_MIB=${FREE_MIB:-2600}       # the server takes 0.90 of the card, so little else may be on it
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
UP_MIN=${UP_MIN:-30}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
# R0's: FlashInfer's sampler needs nvcc; stats every 5 s for the paging guard; the cache holds the weights
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOG_STATS_INTERVAL=5 HF_HUB_OFFLINE=1
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
rows_done() { cat "$RUN"/rows_*.jsonl 2>/dev/null | wc -l; }

wait_idle() {   # 12 checks 10 s apart (scripts/kd27b/run.sh's rule)
  local ok=0 waited=0 u m p
  while [ "$ok" -lt 12 ]; do
    [ -f "$RUN/PAUSE" ] && { echo "$(date +%T) PAUSE is set"; return 1; }
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

stop_server() {   # scripts/engine27b/run.sh's
  if [ -f "$RUN/vllm.pid" ]; then
    kill -- -"$(cat "$RUN/vllm.pid")" 2>/dev/null || kill "$(cat "$RUN/vllm.pid")" 2>/dev/null
    sleep 15; kill -9 -- -"$(cat "$RUN/vllm.pid")" 2>/dev/null
    rm -f "$RUN/vllm.pid"
    status "server stopped"
  fi
}
trap stop_server EXIT
trap 'exit 143' TERM INT

start_server() {   # <attempt>: 0 once /v1/models answers, 1 if the server exits or UP_MIN passes (R0's)
  local a=$1 t0 args
  mapfile -t args < <($PY $SERVE serve-args --run "$RUN" --attempt "$a" --port "$PORT")
  t0=$(date +%s)
  setsid conda run --no-capture-output -n vimBot vllm serve "${args[@]}" >> "$RUN/vllm_$a.log" 2>&1 &
  echo $! > "$RUN/vllm.pid"
  status "server attempt $a: vllm serve ${args[*]}"
  while ! curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 10
    if ! kill -0 "$(cat "$RUN/vllm.pid")" 2>/dev/null; then
      status "server attempt $a: exited ($(grep -E 'Error|error' "$RUN/vllm_$a.log" | tail -1 | cut -c1-200))"
      stop_server; return 1
    fi
    if [ $(( $(date +%s) - t0 )) -ge $((UP_MIN * 60)) ]; then
      status "server attempt $a: not up in $UP_MIN min"; stop_server; return 1
    fi
  done
  status "server attempt $a: up in $(( $(date +%s) - t0 )) s; card memory $(card | cut -d, -f2) MiB"
}

watched() {   # <log> <cmd...>: kill a stage whose log goes quiet for STALL_MIN (scripts/kd27b/run.sh's watchdog)
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
if [ -f "$RUN/PAUSE" ]; then status "PAUSE is set: rm $RUN/PAUSE to resume"; exit 0; fi
if has trigger; then
  $PY $SERVE trigger > "$RUN/trigger.log" 2>&1
  rc=$?
  status "trigger: $(tail -1 "$RUN/trigger.log")"
  if [ $rc -ne 0 ]; then status "done: the labeling pass did not run"; exit 0; fi
fi
if has plan; then
  if [ -f "$RUN/rows.json" ] && [ -f "$RUN/plan_retr8.jsonl" ]; then status "plan: on disk already"
  else $PY scripts/think_label/label.py plan --run "$RUN" > "$RUN/plan.log" 2>&1 \
    && status "plan: $(tail -1 "$RUN/plan.log")" || { tail -3 "$RUN/plan.log"; status "plan FAILED"; exit 1; }
  fi
fi
if has prompts; then
  if [ -f "$RUN/prompts_vllm.jsonl" ]; then status "prompts: on disk already"
  else $PY $SERVE prompts --run "$RUN" > "$RUN/prompts.log" 2>&1 \
    && status "prompts: $(tail -1 "$RUN/prompts.log")" || { tail -3 "$RUN/prompts.log"; status "prompts FAILED"; exit 1; }
  fi
fi
complete=0
if has run; then
  attempt=0
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    status "run: waiting for an idle card ($(rows_done) arm-rows on disk)"
    wait_idle || { status "run: no idle card, or PAUSE is set"; break; }
    up=""
    for a in 2 3; do start_server "$a" && { up=$a; break; }; done
    if [ -z "$up" ]; then status "run attempt $attempt: no server came up"; continue; fi
    status "run attempt $attempt on server attempt $up ($(rows_done) arm-rows on disk)"
    watched "$RUN/run.log" $PY $SERVE run --run "$RUN" --port "$PORT" --attempt "$up" --server-log "$RUN/vllm_$up.log"
    rc=$?
    stop_server
    status "run attempt $attempt rc $rc ($(rows_done) arm-rows on disk): $(tail -1 "$RUN/run.log" | cut -c1-200)"
    if [ $rc -eq 0 ]; then complete=1; break; fi
    if [ $rc -eq 6 ]; then status "paused: rm $RUN/PAUSE, then STEPS=\"run report\" to resume"; break; fi
    if [ $rc -eq 7 ]; then status "run: stopped by the sanity check; not retrying"; break; fi
  done
fi
if has report; then
  if [ $complete -eq 1 ]; then
    $PY scripts/think_label/report.py > "$RUN/report.log" 2>&1 \
      && status "report: $RUN/report.md ($(grep -o '→ \*\*[a-z]*\*\*' "$RUN/report.md" | tr -d '*→ ' | head -1))" \
      || { tail -3 "$RUN/report.log"; status "report FAILED"; }
  else
    status "report: not yet, the pass has rows to go"
  fi
fi
status "done: $STEPS"
