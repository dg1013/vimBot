#!/usr/bin/env bash
# Soft-target KD from the trained 27B into the 4B (scripts/kd27b/README.md; the rules are in
# runs/kd27b_2026-10-31/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   check    GPU: train_kd.py --inspect 2 (builds the trainer, one real batch through the
#                 collator's alignment check, no training)
#   kd42     GPU: train_kd.py at --seed 42 -> runs/qwen3.5-4b-kd27b-s42
#   kd7      GPU: train_kd.py at --seed 7  -> runs/qwen3.5-4b-kd27b-s7
#   report   CPU: compare_kd.py -> $RUN/kd.md
# The KD arms run approach 1's command (scripts/distill_targets/run.sh, the distilled file) with the
# KD loss on top. Every GPU stage waits for an idle card, and a stage whose log is quiet for
# STALL_MIN minutes is killed and retried, with the partial run set aside. A finished arm is
# skipped. WAIT_PID: wait for that process first.
#
#   bash scripts/kd27b/run.sh            # status: runs/kd27b_2026-10-31/status.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/kd27b_2026-10-31
STEPS=${STEPS:-"check kd42 kd7 report"}
WAIT_PID=${WAIT_PID:-}
MODEL=qwen3.5-4b
DISTILLED=runs/distill_2026-10-31/train_distilled.jsonl
PROBS=runs/teacher_probs_2026-10-31/probs.jsonl
KD_ARGS=(--teacher-probs "$PROBS" --kd-alpha 0.5 --kd-temperature 2 --kd-topk 20)   # pre-registered
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
pct() { $PY -c "import json; print(json.load(open('$1'))['metrics']['exec_match_pct'])" 2>/dev/null; }

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

kd_arm() {   # <name> <seed> <out dir>: approach 1's sft_arm, through the KD trainer
  local name=$1 seed=$2 out=$3 attempt=0 rc rows every
  if [ -f "$out/sft_val.json" ]; then status "$name: done already ($out/sft_val.json)"; return 0; fi
  rows=$(wc -l < "$DISTILLED"); every=$(( rows / 32 ))
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
      local aside="$out.partial-$(date +%Y%m%d-%H%M%S)"
      mv "$out" "$aside"; status "$name: a partial run was set aside as $(basename "$aside")"
    fi
    mkdir -p "$out"
    status "$name: waiting for an idle card"
    wait_idle || return 1
    status "$name attempt $attempt: $rows rows, seed $seed, eval every $every steps -> $out"
    watched "$out/train.log" $PY scripts/kd27b/train_kd.py "${KD_ARGS[@]}" --model $MODEL --train "$DISTILLED" \
        --out-dir "$out" --eval-every $every --no-eval-at-start --seed "$seed" --wandb
    rc=$?
    if [ -f "$out/sft_val.json" ]; then status "$name: done; full val $(pct "$out/sft_val.json")%"; return 0; fi
    if grep -q "not the teacher's recorded ones" "$out/train.log" 2>/dev/null; then
      status "$name: the alignment check failed; not retrying"; return 1
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

if has check; then
  status "check: waiting for an idle card"
  if wait_idle; then
    $PY scripts/kd27b/train_kd.py "${KD_ARGS[@]}" --model $MODEL --train "$DISTILLED" \
        --out-dir "$RUN/check" --inspect 2 > "$RUN/check.log" 2>&1 \
      && status "check: passed ($RUN/check.log)" \
      || { tail -5 "$RUN/check.log"; status "check FAILED"; exit 1; }
  else
    status "check: no idle card"; exit 1
  fi
fi
if has kd42; then
  kd_arm kd-s42 42 runs/qwen3.5-4b-kd27b-s42 || status "kd-s42: gave up"
fi
if has kd7; then
  kd_arm kd-s7 7 runs/qwen3.5-4b-kd27b-s7 || status "kd-s7: gave up"
fi
if has report; then
  $PY scripts/kd27b/compare_kd.py > "$RUN/report.log" 2>&1 \
    && status "report: $RUN/kd.md ($(grep -m1 -o 'Verdict, by the pre-registered rule: [a-z ()-]*' "$RUN/kd.md"))" \
    || { tail -3 "$RUN/report.log"; status "report FAILED"; }
fi
status "done: $STEPS"
