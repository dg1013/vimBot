#!/usr/bin/env bash
# Approach 1: SFT on distilled targets (scripts/distill_targets/README.md; the rules are in
# runs/distill_2026-10-31/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   build      CPU: build_targets.py -> $RUN/train_distilled.jsonl (skipped when present)
#   selfcheck  CPU: src/evaluate.py on the distilled file, the gold self-check; anything under 100%
#              stops the driver before the SFT stages (skipped when $RUN/selfcheck.ok exists)
#   sft        GPU: train_sft.py on the distilled file -> runs/qwen3.5-4b-sft-distill/
#   rerun      GPU: the same command on the golds -> runs/qwen3.5-4b-sft-ds100-rerun/ (noise floor)
#   compare    CPU: compare.py -> $RUN/report.md
# Both SFT arms use the data-scaling run's exact command (scripts/data_scaling/run_scaling.sh: the
# round-1 recipe, --eval-every rows/32 --no-eval-at-start --wandb) and its watchdog: a stage whose
# log is quiet for STALL_MIN minutes is killed and retried, a partial run set aside first. A finished
# arm (its sft_val.json exists) is skipped. WAIT_PID: wait for that process to exit first. GPU stages
# wait for an idle card (run_at_k.sh's rule).
#
#   WAIT_PID=6411 bash scripts/distill_targets/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/distill_2026-10-31
STEPS=${STEPS:-"build selfcheck sft rerun compare"}
WAIT_PID=${WAIT_PID:-}
MODEL=qwen3.5-4b
PY=$HOME/miniconda3/envs/vimBot/bin/python    # as run_scaling.sh, so the watchdog's kill reaches it
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

sft_arm() {   # <name> <train file> <out dir>
  local name=$1 train=$2 out=$3 attempt=0 rc rows every
  if [ -f "$out/sft_val.json" ]; then status "$name: done already ($out/sft_val.json)"; return 0; fi
  rows=$(wc -l < "$train"); every=$(( rows / 32 ))
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
      local aside="$out.partial-$(date +%Y%m%d-%H%M%S)"
      mv "$out" "$aside"; status "$name: a partial run was set aside as $(basename "$aside")"
    fi
    mkdir -p "$out"
    status "$name: waiting for an idle card"
    wait_idle || return 1
    status "$name attempt $attempt: $rows rows, eval every $every steps -> $out"
    watched "$out/train.log" $PY scripts/train_sft.py --model $MODEL --train "$train" --out-dir "$out" \
        --eval-every $every --no-eval-at-start --wandb
    rc=$?
    if [ -f "$out/sft_val.json" ]; then
      status "$name: done; full val $($PY -c "import json; print(json.load(open('$out/sft_val.json'))['metrics']['exec_match_pct'])")%"
      return 0
    fi
    status "$name attempt $attempt failed (rc $rc)"
  done
  return 1
}

status "start: STEPS=\"$STEPS\" WAIT_PID=${WAIT_PID:-none}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  status "pid $WAIT_PID has exited"
fi

if has build; then
  if [ -f "$RUN/train_distilled.jsonl" ]; then
    status "build: done already"
  else
    status "build"
    $PY scripts/distill_targets/build_targets.py > "$RUN/build.log" 2>&1 \
      || { status "build FAILED: see $RUN/build.log"; exit 1; }
  fi
fi

if has selfcheck; then
  if [ -f "$RUN/selfcheck.ok" ]; then
    status "selfcheck: passed already"
  else
    status "selfcheck: the gold self-check of the distilled file"
    if $PY src/evaluate.py --cases "$RUN/train_distilled.jsonl" > "$RUN/selfcheck.log" 2>&1; then
      date -Is > "$RUN/selfcheck.ok"; status "selfcheck: 100%"
    else
      tail -3 "$RUN/selfcheck.log"; status "selfcheck FAILED: not every target reproduces its buffer; no SFT"; exit 1
    fi
  fi
fi

if has sft; then
  sft_arm sft "$RUN/train_distilled.jsonl" runs/qwen3.5-4b-sft-distill || status "sft: gave up"
fi
if has rerun; then
  sft_arm rerun data/splits/train.jsonl runs/qwen3.5-4b-sft-ds100-rerun || status "rerun: gave up"
fi
if has compare; then
  $PY scripts/distill_targets/compare.py > "$RUN/compare.log" 2>&1 \
    && status "compare: $RUN/report.md" || { tail -3 "$RUN/compare.log"; status "compare FAILED"; }
fi
status "done: $STEPS"
