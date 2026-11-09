#!/usr/bin/env bash
# R0, the engine probe (scripts/engine27b/README.md; the rules are in
# runs/engine27b_2026-11-02/preregistration.md).
#
#   prompts  CPU: probe.py prompts (the 16 rows, rendered as the base 27B arm renders them)
#   serve    GPU: vllm serve under attempt 1, 2 or 3 (the pre-registered fallbacks), the first that comes up
#   stages   A B C D1 D8 against the server; a stage the server dies under reruns under the next attempt
#   report   CPU: probe.py report -> probe.md
# Waits for an idle card first. A stage whose vLLM log is quiet for QUIET_MIN minutes is stopped, and
# no stage starts BUDGET_S seconds after the first one did.
#
#   setsid nohup bash scripts/engine27b/run.sh > /dev/null 2>&1 &   # status: runs/engine27b_2026-11-02/status.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/engine27b_2026-11-02
PY=$HOME/miniconda3/envs/vimBot/bin/python
PROBE=scripts/engine27b/probe.py
PORT=${PORT:-8013}
UP_MIN=${UP_MIN:-30}
QUIET_MIN=${QUIET_MIN:-10}
BUDGET_S=${BUDGET_S:-7200}
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STAGES="A B C D1 D8"
# serve.sh's reason (FlashInfer's sampler needs nvcc); stats every 5 s for the gate; the cache holds the weights
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOG_STATS_INTERVAL=5 HF_HUB_OFFLINE=1
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }

wait_idle() {   # 12 checks 10 s apart (scripts/kd27b/run.sh's rule)
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

attempt_line() {   # <attempt> <up> <load_s> <memory MiB> <t_start>
  echo "{\"attempt\": $1, \"up\": $2, \"load_s\": $3, \"memory_used_mib\": $4, \"t_start\": $5}" >> "$RUN/attempts.jsonl"
}

start_server() {   # <attempt>: 0 once /v1/models answers, 1 if the server exits or UP_MIN passes
  local a=$1 t0 args
  mapfile -t args < <($PY $PROBE serve-args --attempt "$a" --port "$PORT")
  t0=$(date +%s)
  setsid conda run --no-capture-output -n vimBot vllm serve "${args[@]}" > "$RUN/vllm_$a.log" 2>&1 &
  echo $! > "$RUN/vllm.pid"
  status "attempt $a: vllm serve ${args[*]}"
  while ! curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 10
    if ! kill -0 "$(cat "$RUN/vllm.pid")" 2>/dev/null; then
      status "attempt $a: the server exited ($(grep -m1 -E 'Error|error' "$RUN/vllm_$a.log" | cut -c1-200))"
      stop_server; attempt_line "$a" false null null "$t0"; return 1
    fi
    if [ $(( $(date +%s) - t0 )) -ge $((UP_MIN * 60)) ]; then
      status "attempt $a: not up in $UP_MIN min"; stop_server; attempt_line "$a" false null null "$t0"; return 1
    fi
  done
  local load=$(( $(date +%s) - t0 )) mem
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  status "attempt $a: up in ${load} s; card memory $mem MiB"
  attempt_line "$a" true "$load" "$mem" "$t0"
  return 0
}

stop_server() {
  if [ -f "$RUN/vllm.pid" ]; then
    kill -- -"$(cat "$RUN/vllm.pid")" 2>/dev/null || kill "$(cat "$RUN/vllm.pid")" 2>/dev/null
    sleep 15; kill -9 -- -"$(cat "$RUN/vllm.pid")" 2>/dev/null
    rm -f "$RUN/vllm.pid"
  fi
  sleep 5
}

run_stage() {   # <stage> <attempt>: 0 done, 1 failed, 124 the log went quiet
  local s=$1 a=$2 log=$RUN/vllm_$2.log pid age
  $PY $PROBE stage "$s" --port "$PORT" --attempt "$a" >> "$RUN/stages.log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
    age=$(( $(date +%s) - $(stat -c %Y "$log") ))
    if [ "$age" -gt $((QUIET_MIN * 60)) ]; then
      status "stage $s: vLLM's log quiet for $((age / 60)) min; stopping it"
      kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 124
    fi
  done
  wait "$pid"
}

status "start: R0, stages $STAGES"
$PY $PROBE prompts >> "$RUN/stages.log" 2>&1 || { status "prompts FAILED"; exit 1; }
status "prompts: $(tail -1 "$RUN/stages.log")"
status "waiting for an idle card"
wait_idle || { status "no idle card"; exit 1; }
first_stage=""
for a in 1 2 3; do
  start_server "$a" || continue
  server_ok=1
  for s in $STAGES; do
    [ -f "$RUN/stage_$s.json" ] && continue
    if [ -n "$first_stage" ] && [ $(( $(date +%s) - first_stage )) -ge "$BUDGET_S" ]; then
      status "stage $s: not started, the budget of $((BUDGET_S / 60)) min has passed"; continue
    fi
    [ -z "$first_stage" ] && first_stage=$(date +%s)
    status "stage $s: start (attempt $a)"
    run_stage "$s" "$a"; rc=$?
    if [ -f "$RUN/stage_$s.json" ]; then
      status "stage $s: done ($(tail -1 "$RUN/stages.log"))"
    elif ! curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then
      status "stage $s: the server died under attempt $a (rc $rc); the next attempt reruns it"; server_ok=0; break
    else
      status "stage $s: failed (rc $rc) with the server still up; going on"
    fi
  done
  stop_server
  [ "$server_ok" -eq 1 ] && break
done
$PY $PROBE report > "$RUN/report.log" 2>&1 \
  && status "report: $RUN/probe.md ($(grep -m1 -o 'Verdict, by the pre-registered rule: [a-z ]*' "$RUN/probe.md"))" \
  || { tail -3 "$RUN/report.log"; status "report FAILED"; }
status "done"
