#!/usr/bin/env bash
# QLoRA SFT of the 27B on the distilled targets (scripts/qlora27b/README.md; the rules are in
# runs/qlora27b_2026-10-31/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   prepare  CPU: probe.py prepare -> $RUN/probe_train.jsonl (the 256 longest train rows) and
#            $RUN/probe_val.jsonl (16 val rows)
#   probe    GPU: 8 optimizer steps on the longest rows, one evaluation and the reload-and-evaluate
#            pass on the 16 val rows; probe.py verdict judges it. A failure at micro-batch 4 is
#            retried once at 2 (the same effective batch of 32). The micro-batch that passed is
#            written to $RUN/probe.ok, and without that file the train stage does not start.
#   train    GPU: the full run at that micro-batch -> runs/qwen3.5-27b-sft-distill/
#   report   CPU: report.py -> $RUN/report.md
# GPU stages wait for an idle card (run_at_k.sh's rule). Both GPU stages have the distill driver's
# watchdog: a log quiet for STALL_MIN minutes is killed. A failed train attempt is retried once with
# the partial run set aside, except when train_27b.py stopped itself because its projection passed
# 12 h: the pace would be the same the second time.
#
#   bash scripts/qlora27b/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/qlora27b_2026-10-31
OUT=runs/qwen3.5-27b-sft-distill
TRAIN=runs/distill_2026-10-31/train_distilled.jsonl
MODEL=qwen3.5-27b
STEPS=${STEPS:-"prepare probe train report"}
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STALL_MIN=${STALL_MIN:-30}
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

set_aside() {   # <dir>: move a previous run out of the way (never deleted)
  if [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; then
    local aside="$1.partial-$(date +%Y%m%d-%H%M%S)"
    mv "$1" "$aside"; status "a previous run was set aside as $(basename "$aside")"
  fi
}

probe_at() {   # <micro-batch>: one probe; returns probe.py's verdict (0 = pass)
  local mb=$1 d="$RUN/probe-mb$1" rc
  set_aside "$d"; mkdir -p "$d"
  status "probe at micro-batch $mb: waiting for an idle card"
  wait_idle || return 2
  status "probe at micro-batch $mb x $((32 / mb)): 8 steps on the 256 longest rows, eval and reload on 16 val rows"
  watched "$d/train.log" $PY scripts/qlora27b/train_27b.py --model $MODEL --load-in-4bit \
      --train "$RUN/probe_train.jsonl" --val "$RUN/probe_val.jsonl" --out-dir "$d" \
      --batch-size "$mb" --grad-accum $((32 / mb)) --max-steps 8 --eval-every 8 --eval-subset 0 \
      --no-eval-at-start --pace-every 1
  rc=$?
  $PY scripts/qlora27b/probe.py verdict "$d" $rc > "$d/verdict.log" 2>&1
  local v=$?
  status "$(tail -1 "$d/verdict.log")"
  return $v
}

train_stage() {
  local mb attempt=0 rc rows every
  if [ -f "$OUT/sft_val.json" ]; then status "train: done already ($OUT/sft_val.json)"; return 0; fi
  if [ ! -f "$RUN/probe.ok" ]; then status "train: no passed probe ($RUN/probe.ok), not starting"; return 1; fi
  mb=$(cat "$RUN/probe.ok")
  rows=$(wc -l < "$TRAIN"); every=$(( rows / 32 ))
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    set_aside "$OUT"; mkdir -p "$OUT"
    status "train: waiting for an idle card"
    wait_idle || return 1
    status "train attempt $attempt: $rows rows, micro-batch $mb x $((32 / mb)), eval every $every steps -> $OUT"
    watched "$OUT/train.log" $PY scripts/qlora27b/train_27b.py --model $MODEL --load-in-4bit \
        --train "$TRAIN" --out-dir "$OUT" --batch-size "$mb" --grad-accum $((32 / mb)) \
        --eval-every $every --no-eval-at-start --wandb
    rc=$?
    if [ -f "$OUT/sft_val.json" ]; then
      status "train: done; full val $($PY -c "import json; print(json.load(open('$OUT/sft_val.json'))['metrics']['exec_match_pct'])")%"
      return 0
    fi
    if grep -q "exceeds --max-hours" "$OUT/train.log"; then
      status "train: stopped by its own 12 h projection ($(grep -m1 'exceeds --max-hours' "$OUT/train.log")); not retrying"
      return 1
    fi
    status "train attempt $attempt failed (rc $rc)"
  done
  return 1
}

status "start: STEPS=\"$STEPS\" FREE_MIB=$FREE_MIB FREE_UTIL=$FREE_UTIL"

if has prepare; then
  [ -f "$TRAIN" ] || { status "prepare: $TRAIN missing"; exit 1; }
  [ -f runs/distill_2026-10-31/selfcheck.ok ] || { status "prepare: the distilled file has no passed self-check"; exit 1; }
  $PY scripts/qlora27b/probe.py prepare > "$RUN/prepare.log" 2>&1 \
    && status "prepare: $(head -1 "$RUN/prepare.log")" || { status "prepare FAILED: see $RUN/prepare.log"; exit 1; }
fi

if has probe; then
  if [ -f "$RUN/probe.ok" ]; then
    status "probe: passed already at micro-batch $(cat "$RUN/probe.ok")"
  elif probe_at 4; then
    echo 4 > "$RUN/probe.ok"
  elif probe_at 2; then
    echo 2 > "$RUN/probe.ok"
  else
    status "probe FAILED at micro-batch 4 and 2: no full run (see $RUN/probe-mb*/)"
  fi
fi

if has train; then
  train_stage || status "train: gave up"
fi

if has report; then
  if [ -f "$OUT/sft_val.json" ]; then
    $PY scripts/qlora27b/report.py > "$RUN/report.log" 2>&1 \
      && status "report: $RUN/report.md" || { tail -3 "$RUN/report.log"; status "report FAILED"; }
  else
    status "report: skipped, no finished run"
  fi
fi
status "done: $STEPS"
