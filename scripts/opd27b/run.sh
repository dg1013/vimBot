#!/usr/bin/env bash
# On-policy distillation from the trained 27B into the 4B (scripts/opd27b/README.md; the rules are in
# runs/opd27b_2026-11-02/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   prompts   CPU: opd.py prompts (train prompt ids, identical for the 4B and the 27B)
#   check     GPU: one round on 64 prompts end to end, in $RUN/check (sample, judge, teacher, 2 train steps, eval on 8 rows)
#   s42 s7    GPU: an arm: round 0's subset eval, then per round sample -> judge (CPU, in the background)
#                  -> teacher -> train -> eval, then the choice and full val, then pass@8 for the start and the choice
#   report    CPU: report.py -> $RUN/opd.md
# The first GPU stage waits for an idle card (12 checks 10 s apart); later ones for 3 checks, since the
# card was just ours. A stage whose log is quiet for STALL_MIN minutes is killed and retried, at most
# MAX_ATTEMPTS times; every stage resumes from its outputs. WAIT_PID: wait for that process first.
#
#   WAIT_PID=<R0's driver> setsid nohup bash scripts/opd27b/run.sh > /dev/null 2>&1 &   # status: $RUN/status.txt
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/opd27b_2026-11-02
STEPS=${STEPS:-"prompts check s42 s7 report"}
WAIT_PID=${WAIT_PID:-}
PY=$HOME/miniconda3/envs/vimBot/bin/python    # direct, so the watchdog's kill reaches it
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-28800}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
ROUNDS=3
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }
FIRST_WAIT=12

wait_idle() {   # <checks>: that many checks 10 s apart (scripts/kd27b/run.sh's rule)
  local need=${1:-12} ok=0 waited=0 u m p
  while [ "$ok" -lt "$need" ]; do
    IFS=, read -r u m <<< "$(card)"
    p=$(procs)
    if [ "$u" -le "$FREE_UTIL" ] && [ "$p" -eq 0 ] && [ "$m" -le "$FREE_MIB" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 10
    waited=$((waited + 10))
    if [ $((waited % 600)) -eq 0 ]; then echo "$(date +%T) waiting for an idle card: ${u}%, ${m} MiB, $p processes"; fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "$(date +%T) no idle card in $((MAX_WAIT / 3600)) h; giving up"; return 1; fi
  done
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

gpu() {   # <name> <log> <cmd...>: an idle card, then the command, retried
  local name=$1 log=$2 attempt=0 rc; shift 2
  mkdir -p "$(dirname "$log")"
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    wait_idle "$FIRST_WAIT" || return 1
    FIRST_WAIT=3
    status "$name (attempt $attempt)"
    watched "$log" "$@"; rc=$?
    [ $rc -eq 0 ] && return 0
    status "$name attempt $attempt failed (rc $rc): $(tail -1 "$log" | cut -c1-200)"
    if grep -q "no matching teacher record\|tokenize differently\|changed the prompt\|without its stop token" "$log" 2>/dev/null; then
      status "$name: an alignment check failed; not retrying"; return 1
    fi
  done
  return 1
}

JUDGES=()
wait_judges() {   # the background judges, by pid: a bare `wait` also waits for the tee behind `exec > >(...)`, which never exits
  local p
  for p in "${JUDGES[@]}"; do wait "$p" || status "a background judge (pid $p) failed: see the rounds' judge.log"; done
  JUDGES=()
}

arm() {   # <seed> [root] [extra...]: one arm; the check passes a root and limits
  local seed=$1 root=${2:-runs/qwen3.5-4b-opd27b-s$1} r rd
  local RA=(--seed "$seed" --root "$root")
  gpu "s$seed round 0: subset eval of the start" "$root/round0/eval.log" $PY scripts/opd27b/opd.py eval "${RA[@]}" --round 0 ${EVAL_LIMIT:-} || return 1
  for r in $(seq 1 $ROUNDS); do
    rd=$root/round$r
    gpu "s$seed round $r: sample" "$rd/sample.log" $PY scripts/opd27b/sample.py train "${RA[@]}" --round "$r" ${SAMPLE_LIMIT:-} || return 1
    $PY scripts/opd27b/sample.py judge "${RA[@]}" --round "$r" >> "$rd/judge.log" 2>&1 &
    JUDGES+=($!)
    gpu "s$seed round $r: teacher" "$rd/teacher.log" $PY scripts/opd27b/teacher.py "${RA[@]}" --round "$r" || return 1
    gpu "s$seed round $r: train" "$rd/train.log" $PY scripts/opd27b/opd.py train "${RA[@]}" --round "$r" ${TRAIN_LIMIT:-} || return 1
    gpu "s$seed round $r: eval" "$rd/eval.log" $PY scripts/opd27b/opd.py eval "${RA[@]}" --round "$r" ${EVAL_LIMIT:-} || return 1
    status "s$seed round $r: $(tail -1 "$rd/eval.log")"
    [ -n "${ONE_ROUND:-}" ] && return 0
  done
  gpu "s$seed: choose and full val" "$root/final.log" $PY scripts/opd27b/opd.py final "${RA[@]}" || return 1
  status "s$seed: $(grep -E 'full val|no change' "$root/final.log" | tail -1)"
  local start chosen
  start=$($PY scripts/opd27b/opd.py adapter --seed "$seed" --round 0)
  chosen=$($PY -c "import json; print(json.load(open('$root/chosen.json'))['adapter'])")
  gpu "s$seed: pass@8 of the start" "$root/pass8.log" $PY scripts/opd27b/sample.py pass8 "${RA[@]}" --label start --adapter "$start" || return 1
  gpu "s$seed: pass@8 of the chosen adapter" "$root/pass8.log" $PY scripts/opd27b/sample.py pass8 "${RA[@]}" --label chosen --adapter "$chosen" || return 1
  wait_judges
  status "s$seed: done"
}

status "start: STEPS=\"$STEPS\" WAIT_PID=${WAIT_PID:-none}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  status "pid $WAIT_PID has exited"
fi
if has prompts; then
  if [ -f "$RUN/prompts.jsonl" ]; then status "prompts: on disk already"
  else $PY scripts/opd27b/opd.py prompts > "$RUN/prompts.log" 2>&1 && status "prompts: $(tail -1 "$RUN/prompts.log")" \
    || { tail -3 "$RUN/prompts.log"; status "prompts FAILED"; exit 1; }
  fi
fi
if has check; then
  if SAMPLE_LIMIT="--limit 64" TRAIN_LIMIT="--limit-steps 2" EVAL_LIMIT="--limit 8" ONE_ROUND=1 arm 42 "$RUN/check/s42" \
     && gpu "check: pass@8 on 8 rows with round 1's adapter" "$RUN/check/s42/pass8.log" $PY scripts/opd27b/sample.py pass8 \
          --seed 42 --root "$RUN/check/s42" --label check --adapter "$RUN/check/s42/round1/adapter" --limit 8; then
    wait_judges
    status "check: passed ($(grep -h -E 'samples in|matched|step 2/2' $RUN/check/s42/round1/*.log | tr '\n' ' ' | cut -c1-300))"
  else
    status "check FAILED"; exit 1
  fi
fi
for seed in 42 7; do
  if has "s$seed"; then arm "$seed" || { status "s$seed: gave up"; exit 1; }; fi
done
wait_judges
if has report; then
  $PY scripts/opd27b/report.py > "$RUN/report.log" 2>&1 \
    && status "report: $RUN/opd.md ($(grep -m1 -o 'Verdict, by the pre-registered rule: [a-z]*' "$RUN/opd.md" || echo 'partial'))" \
    || { tail -3 "$RUN/report.log"; status "report FAILED"; }
fi
status "done: $STEPS"
