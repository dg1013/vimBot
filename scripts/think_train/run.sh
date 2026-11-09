#!/usr/bin/env bash
# R2: the thinking teacher's labels (A1) and reasoning (A2) as training signals for the 4B
# (scripts/think_train/README.md; the rules are in runs/think_train_2026-11-08/preregistration.md).
#
# Waits for WAIT_PID (R1's driver), then, each only if named in STEPS:
#   gate     CPU: build_r2.py gate -- R1 complete on both arms, and its reading L is helps
#   data     CPU: build_r2.py build -> kept.jsonl, train_a1.jsonl, traces.jsonl, build.json
#   check    GPU: train_r2.py --check -- one optimizer step with the longest trace; A2 runs only if
#                 the peak reserved memory is <= A2_MAX_GIB and the projected R2 time is <= 12 h
#   a1-42    GPU: train_sft.py on train_a1.jsonl, seed 42 -> runs/qwen3.5-4b-r2-a1-s42
#   a2-42    GPU: train_r2.py (the reasoning task) on train_a1.jsonl, seed 42 -> runs/qwen3.5-4b-r2-a2-s42
#   a1-7     GPU: the same at seed 7 -> runs/qwen3.5-4b-r2-a1-s7
#   a2-7     GPU: the same at seed 7 -> runs/qwen3.5-4b-r2-a2-s7
#   report   CPU: compare_r2.py -> $RUN/r2.md
# The arms run approach 1's command (scripts/distill_targets/run.sh's sft_arm with the seed, as
# scripts/distill_followup/run.sh ran it), on A1's file. Every GPU stage waits for an idle card; a
# stage whose log is quiet for STALL_MIN minutes is killed and retried, with the partial run set
# aside. A finished arm is skipped. A pause: `touch $RUN/PAUSE` stops the driver before its next GPU
# stage (a running arm finishes); rm it and start the driver again to resume.
#
#   WAIT_PID=<R1's driver> setsid nohup bash scripts/think_train/run.sh > /dev/null 2>&1 &   # status: $RUN/status.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/think_train_2026-11-08
STEPS=${STEPS:-"gate data check a1-42 a2-42 a1-7 a2-7 report"}
WAIT_PID=${WAIT_PID:-}
MODEL=qwen3.5-4b
A1_TRAIN=$RUN/train_a1.jsonl
TRACES=$RUN/traces.jsonl
TRACE_ARGS=(--traces "$TRACES" --trace-weight 1)    # pre-registered
A2_MAX_GIB=${A2_MAX_GIB:-20.5}     # pre-registered: the 27B QLoRA probe's bound on this card
BUDGET_H=${BUDGET_H:-12}           # §14b
A1_H=${A1_H:-1.25}                 # one approach-1-recipe arm with its evaluations (KD's arms: ~1.2 h)
PY=$HOME/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1   # weights are cached
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
pct() { $PY -c "import json; print(json.load(open('$1'))['metrics']['exec_match_pct'])" 2>/dev/null; }
paused() { [ -f "$RUN/PAUSE" ] && { status "paused before $1: rm $RUN/PAUSE and start the driver again"; return 0; }; return 1; }

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

arm() {   # <name> <seed> <out dir> <trainer...>: approach 1's sft_arm with the seed, on A1's file
  local name=$1 seed=$2 out=$3 attempt=0 rc rows every
  shift 3
  if [ -f "$out/sft_val.json" ]; then status "$name: done already ($out/sft_val.json)"; return 0; fi
  rows=$(wc -l < "$A1_TRAIN"); every=$(( rows / 32 ))
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
    watched "$out/train.log" "$@" --model $MODEL --train "$A1_TRAIN" --out-dir "$out" \
        --eval-every $every --no-eval-at-start --seed "$seed" --wandb
    rc=$?
    if [ -f "$out/sft_val.json" ]; then status "$name: done; full val $(pct "$out/sft_val.json")%"; return 0; fi
    status "$name attempt $attempt failed (rc $rc)"
  done
  return 1
}

a2_gate() {   # prints "yes" or the reason A2 does not run, from the check and the data's trace lengths
  $PY - "$RUN/check/check.json" "$RUN/build.json" "$A1_TRAIN" "$A2_MAX_GIB" "$BUDGET_H" "$A1_H" <<'EOF'
import json, sys
check, build, a1, max_gib, budget, a1_h = sys.argv[1:7]
c, b = json.load(open(check)), json.load(open(build))
rows = sum(1 for line in open(a1) if line.strip())
steps = 3 * (rows // 32)
extra_h = steps * c["trace_seconds"] * b["trace_tokens"]["mean"] / c["tokens"] / 3600
total_h = 2 * float(a1_h) + 2 * (float(a1_h) + extra_h)
out = {"peak_reserved_gib": c["peak_reserved_gib"], "max_gib": float(max_gib), "steps": steps,
       "a2_extra_hours_per_seed": round(extra_h, 2), "projected_r2_hours": round(total_h, 2), "budget_h": float(budget)}
json.dump(out, open(check.replace("check.json", "a2_gate.json"), "w"), indent=1)
if c["peak_reserved_gib"] is None or c["peak_reserved_gib"] > float(max_gib):
    print("peak reserved %s GiB > %s" % (c["peak_reserved_gib"], max_gib))
elif total_h > float(budget):
    print("projected R2 %.1f h > %s h (A2 adds %.1f h per seed): needs the owner (§14b)" % (total_h, budget, extra_h))
else:
    print("yes")
EOF
}

status "start: STEPS=\"$STEPS\" WAIT_PID=${WAIT_PID:-none}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  status "pid $WAIT_PID has exited"
fi

if has gate; then
  why=$($PY scripts/think_train/build_r2.py gate 2>&1); rc=$?
  if [ $rc -ne 0 ]; then status "gate: R2 does not start: $why"; exit 3; fi
  status "gate: $why"
fi
if has data; then
  $PY scripts/think_train/build_r2.py build > "$RUN/build.log" 2>&1; rc=$?
  if [ $rc -ne 0 ]; then tail -3 "$RUN/build.log"; status "data: R2 does not start (rc $rc): $(tail -1 "$RUN/build.log")"; exit 3; fi
  status "data: $(tail -1 "$RUN/build.log")"
fi
A2_OK=no
if has check; then
  paused check && exit 0
  status "check: waiting for an idle card"
  if wait_idle; then
    $PY scripts/think_train/train_r2.py "${TRACE_ARGS[@]}" --check --model $MODEL --train "$A1_TRAIN" \
        --out-dir "$RUN/check" --eval-every 1000000 --no-eval-at-start --max-steps 1 > "$RUN/check.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$RUN/check/check.json" ]; then
      verdict=$(a2_gate)
      if [ "$verdict" = "yes" ]; then A2_OK=yes; status "check: passed, A2 runs ($(cat "$RUN/check/a2_gate.json" | tr -d '\n '))"
      else status "check: A2 does not run: $verdict; A1 still runs"; fi
    else
      tail -5 "$RUN/check.log"; status "check FAILED (rc $rc): A2 does not run; A1 still runs"
    fi
  else
    status "check: no idle card"; exit 1
  fi
elif [ -f "$RUN/check/a2_gate.json" ] && [ "$(a2_gate)" = "yes" ]; then
  A2_OK=yes                       # a resumed driver keeps the earlier check's verdict
fi
for spec in "a1-42 42 a1" "a2-42 42 a2" "a1-7 7 a1" "a2-7 7 a2"; do
  read -r step seed kind <<< "$spec"
  has "$step" || continue
  paused "$step" && exit 0
  out=runs/qwen3.5-4b-r2-$kind-s$seed
  if [ "$kind" = a1 ]; then
    arm "$step" "$seed" "$out" $PY scripts/train_sft.py || status "$step: gave up"
  elif [ "$A2_OK" = yes ]; then
    arm "$step" "$seed" "$out" $PY scripts/think_train/train_r2.py "${TRACE_ARGS[@]}" || status "$step: gave up"
  else
    status "$step: skipped (the check did not clear A2)"
  fi
done
if has report; then
  $PY scripts/think_train/compare_r2.py > "$RUN/report.log" 2>&1 \
    && status "report: $RUN/r2.md ($(grep -E '^\*\*P[12] ' "$RUN/r2.md" | sed 's/\*\*//g; s/ — .*//' | tr '\n' ';'))" \
    || { tail -3 "$RUN/report.log"; status "report FAILED"; }
fi
status "done: $STEPS"
