#!/usr/bin/env bash
# GRPO on OPD's chosen adapters (scripts/grpo_opd/README.md; the rules are in
# runs/grpo_opd_2026-11-04/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   check   GPU: two gated GRPO steps on 64 prompts from seed 42's start, in a throwaway folder
#   s42     GPU: probe_difficulty.py + select_frontier.py, fuel.py's gate, train_grpo_gated.py;
#                CPU: report.py
#   s7      the same from seed 7's start
# Every GPU stage waits for an idle card, and a stage whose log is quiet for STALL_MIN minutes is
# killed: a round is resumed from its newest checkpoint, the probe from the rows on disk. A finished
# stage is skipped. A failed check stops everything; an arm that fails or stops at the fuel gate
# does not stop the other arm.
#
# To pause: touch runs/grpo_opd_2026-11-04/PAUSE. The stage in flight stops at its next step
# (rc 3 from the trainer or the probe) and the driver exits with 6. To resume: rm the file and run
# this again. The file stays until then, so a resume cannot be started by accident.
#
#   setsid nohup bash scripts/grpo_opd/run.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/grpo_opd_2026-11-04
STEPS=${STEPS:-"check s42 s7"}
MODEL=qwen3.5-4b
GOLDS=data/splits/train.jsonl
OLD_PROBE=runs/distill_followup_2026-10-31/difficulty_probe_qwen3.5-4b-distill.jsonl   # the check's gate only
PY=$HOME/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
PAUSED_RC=3      # train_grpo_gated.py's and probe_difficulty.py's own
FUEL_LOW_RC=5    # fuel.py's
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
pct() { $PY -c "import json; print(json.load(open('$1'))['metrics']['exec_match_pct'])" 2>/dev/null; }
rows_in() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }
arm_of() { echo "runs/qwen3.5-4b-opd27b-grpo-s$1"; }
start_of() { $PY -c "import json; print(json.load(open('runs/qwen3.5-4b-opd27b-s$1/chosen.json'))['adapter'])"; }

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

watched() {   # <log> <stop file, or -> <cmd...>: kill a stage whose log goes quiet; PAUSE reaches its stop file
  local log=$1 stop=$2; shift 2
  "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -f "$RUN/PAUSE" ] && [ "$stop" != "-" ]; then touch "$stop"; fi
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

check_stage() {   # the round's whole path, small: load, gate, roll out, reward, update, eval, full val
  local out=$RUN/check adapter rc
  if [ -f "$out/grpo_val.json" ]; then status "check: done already"; return 0; fi
  adapter=$(start_of 42) || return 1
  mkdir -p "$out"
  head -24 data/splits/val.jsonl > "$out/val24.jsonl"
  status "check: waiting for an idle card"
  wait_idle || return 1
  status "check: two steps from $adapter"
  watched "$out/train.log" "$out/STOP" $PY scripts/train_grpo_gated.py --model $MODEL --adapter "$adapter" \
      --run-name grpo-opd-check --out-dir "$out" --val "$out/val24.jsonl" --limit 64 --max-steps 2 \
      --eval-every 2 --eval-subset 8 --print-completions 0 --difficulty "$OLD_PROBE"
  rc=$?
  if [ -f "$out/grpo_val.json" ]; then status "check: passed"; return 0; fi
  if [ "$rc" -eq "$PAUSED_RC" ]; then return "$PAUSED_RC"; fi
  status "check FAILED (rc $rc); see $out/train.log"; return 1
}

probe_stage() {   # <seed> <adapter> <probe file>: resumable, complete when every train prompt has a row
  local s=$1 adapter=$2 probe=$3 stop=$RUN/probe_stop rows attempt=0 rc
  rows=$(wc -l < "$GOLDS")
  if [ "$(rows_in "$probe")" -ge "$rows" ]; then status "probe s$s: complete ($rows rows)"; return 0; fi
  while [ "$(rows_in "$probe")" -lt "$rows" ] && [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    status "probe s$s: waiting for an idle card"
    wait_idle || return 1
    status "probe s$s attempt $attempt ($(rows_in "$probe") of $rows rows so far) on $adapter"
    rm -f "$stop"
    watched "$RUN/probe_s$s.log" "$stop" $PY scripts/probe_difficulty.py --model $MODEL --adapter "$adapter" \
        --out "$probe" --seed "$s" --stop-file "$stop"
    rc=$?
    status "probe s$s attempt $attempt rc $rc ($(rows_in "$probe") rows)"
    if [ "$rc" -eq "$PAUSED_RC" ]; then return "$PAUSED_RC"; fi
  done
  if [ "$(rows_in "$probe")" -lt "$rows" ]; then status "probe s$s incomplete after $attempt attempts"; return 1; fi
  $PY scripts/select_frontier.py --difficulty "$probe" --summary "$RUN/difficulty_summary_s$s.md" \
      --cliff-out "$RUN/train_cliff_s$s.jsonl" >> "$RUN/probe_s$s.log" 2>&1
  status "probe s$s: done; shelves in $RUN/difficulty_summary_s$s.md"
}

grpo_round() {   # <seed> <adapter> <probe file>: resumed, never restarted
  local s=$1 adapter=$2 probe=$3 out attempt=0 rc
  out=$(arm_of "$s")
  if [ -f "$out/grpo_val.json" ]; then status "round s$s: done already ($out/grpo_val.json)"; return 0; fi
  if [ ! -f "$adapter/adapter_config.json" ]; then status "round s$s: no adapter at $adapter"; return 1; fi
  mkdir -p "$out"
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    local resume=()
    if ls -d "$out"/checkpoint-* > /dev/null 2>&1; then resume=(--resume); fi
    status "round s$s: waiting for an idle card"
    wait_idle || return 1
    status "round s$s attempt $attempt ${resume[*]:-(fresh)}: from $adapter -> $out"
    watched "$out/train.log" "$out/STOP" $PY scripts/train_grpo_gated.py --model $MODEL --adapter "$adapter" \
        --run-name "$(basename "$out")" --out-dir "$out" --wandb --print-completions 0 --seed "$s" \
        --difficulty "$probe" "${resume[@]}"
    rc=$?
    if [ -f "$out/grpo_val.json" ]; then status "round s$s: done; full val $(pct "$out/grpo_val.json")%"; return 0; fi
    if [ "$rc" -eq "$PAUSED_RC" ]; then status "round s$s: paused with a checkpoint"; return "$PAUSED_RC"; fi
    if grep -q "exceeds --max-hours" "$out/train.log" 2>/dev/null; then
      status "round s$s: stopped by its own 12 h projection; not resuming"; return 1
    fi
    status "round s$s attempt $attempt failed (rc $rc)"
  done
  return 1
}

arm() {   # <seed>: the probe, the fuel gate, the round
  local s=$1 adapter rc probe=$RUN/difficulty_probe_s$1.jsonl
  adapter=$(start_of "$s") || { status "s$s: no chosen adapter"; return 1; }
  probe_stage "$s" "$adapter" "$probe" || return $?
  $PY scripts/grpo_opd/fuel.py "$probe" --out "$RUN/fuel_s$s.json" | tee -a "$RUN/probe_s$s.log"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -eq "$FUEL_LOW_RC" ]; then status "s$s: under the fuel floor; the round does not run"; return "$rc"; fi
  if [ "$rc" -ne 0 ]; then status "s$s: fuel.py failed (rc $rc)"; return 1; fi
  grpo_round "$s" "$adapter" "$probe"
}

report() {
  $PY scripts/grpo_opd/report.py > "$RUN/report.log" 2>&1 \
    && status "report: $RUN/grpo_opd.md" || { tail -3 "$RUN/report.log"; status "report FAILED"; }
}

status "start: STEPS=\"$STEPS\""
if [ -f "$RUN/PAUSE" ]; then status "PAUSE is set; rm it to run"; exit 6; fi
if has check; then
  check_stage; rc=$?
  if [ "$rc" -eq "$PAUSED_RC" ]; then status "paused in the check"; exit 6; fi
  if [ "$rc" -ne 0 ]; then status "stopping: the check did not pass"; exit 1; fi
fi
for seed in 42 7; do
  if has "s$seed"; then
    arm "$seed"; rc=$?
    if [ "$rc" -eq "$PAUSED_RC" ]; then status "paused at seed $seed; rm $RUN/PAUSE and run this again to resume"; exit 6; fi
    if [ "$rc" -ne 0 ]; then status "s$seed: stopped (rc $rc)"; fi
    report
  fi
done
status "done: $STEPS"
