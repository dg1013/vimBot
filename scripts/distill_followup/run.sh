#!/usr/bin/env bash
# Approach 1 at a second seed, then GRPO on the distilled SFT (scripts/distill_followup/README.md;
# the rules are in runs/distill_followup_2026-10-31/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   seed2     GPU: train_sft.py at --seed 7 on the golds and on the distilled file
#   seeds     CPU: compare_seeds.py -> $RUN/seeds.md
#   grpo1     GPU: train_grpo.py from the seed-42 distilled SFT -> runs/qwen3.5-4b-distill-grpo/
#   test1     GPU: baseline.py on test v2 with round 1's best (reporting only)
#   probe     GPU: probe_difficulty.py on round 1's best, then select_frontier.py's shelves
#   grpo2     GPU: train_grpo_gated.py from round 1's best, fed the probe
#                  -> runs/qwen3.5-4b-distill-grpo-gated/
#   test2     GPU: baseline.py on test v2 with round 2's best
#   report    CPU: report_grpo.py -> $RUN/grpo.md
# The SFT arms run the night's exact command (scripts/distill_targets/run.sh) plus the seed; the
# GRPO rounds and the probe run the served chain's (runs/grpo_2026-09-12/run_grpo.sh and
# runs/grpo_gated_2026-09-14/run_gated.sh) with the adapter and output named. Every GPU stage waits
# for an idle card, and a stage whose log is quiet for STALL_MIN minutes is killed: an SFT arm is
# retried with the partial run set aside, a GRPO round resumed from its newest checkpoint, the probe
# resumed from the rows on disk. A finished stage is skipped. WAIT_PID: wait for that process first.
#
#   WAIT_PID=289372 STEPS="seed2 seeds" bash scripts/distill_followup/run.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/distill_followup_2026-10-31
STEPS=${STEPS:-"seed2 seeds grpo1 test1 probe grpo2 test2 report"}
WAIT_PID=${WAIT_PID:-}
MODEL=qwen3.5-4b
SEED2=7
GOLDS=data/splits/train.jsonl
DISTILLED=runs/distill_2026-10-31/train_distilled.jsonl
START=runs/qwen3.5-4b-sft-distill/best                 # the pre-registered approach-1 arm (seed 42)
G1=runs/qwen3.5-4b-distill-grpo
G2=runs/qwen3.5-4b-distill-grpo-gated
PROBE=$RUN/difficulty_probe_qwen3.5-4b-distill.jsonl
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

sft_arm() {   # <name> <train file> <out dir>: the night's sft_arm, plus the seed
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
    status "$name attempt $attempt: $rows rows, seed $SEED2, eval every $every steps -> $out"
    watched "$out/train.log" $PY scripts/train_sft.py --model $MODEL --train "$train" --out-dir "$out" \
        --eval-every $every --no-eval-at-start --seed $SEED2 --wandb
    rc=$?
    if [ -f "$out/sft_val.json" ]; then status "$name: done; full val $(pct "$out/sft_val.json")%"; return 0; fi
    status "$name attempt $attempt failed (rc $rc)"
  done
  return 1
}

grpo_round() {   # <name> <trainer> <adapter in> <out dir> [trainer args...]: resumed, never restarted
  local name=$1 trainer=$2 adapter=$3 out=$4 attempt=0 rc
  shift 4
  if [ -f "$out/grpo_val.json" ]; then status "$name: done already ($out/grpo_val.json)"; return 0; fi
  if [ ! -f "$adapter/adapter_config.json" ]; then status "$name: no adapter at $adapter"; return 1; fi
  mkdir -p "$out"
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    local resume=()
    if ls -d "$out"/checkpoint-* > /dev/null 2>&1; then resume=(--resume); fi
    status "$name: waiting for an idle card"
    wait_idle || return 1
    status "$name attempt $attempt ${resume[*]:-(fresh)}: from $adapter -> $out"
    watched "$out/train.log" $PY "$trainer" --model $MODEL --adapter "$adapter" --run-name "$(basename "$out")" \
        --out-dir "$out" --wandb --print-completions 0 "${resume[@]}" "$@"
    rc=$?
    if [ -f "$out/grpo_val.json" ]; then status "$name: done; full val $(pct "$out/grpo_val.json")%"; return 0; fi
    if grep -q "exceeds --max-hours" "$out/train.log" 2>/dev/null; then
      status "$name: stopped by its own 12 h projection; not resuming"; return 1
    fi
    status "$name attempt $attempt failed (rc $rc)"
  done
  return 1
}

test_row() {   # <name> <adapter> <out dir> <condition>: the results-table path on test v2
  local name=$1 adapter=$2 out=$3 cond=$4 rc
  if [ -f "$out/baseline_test.json" ]; then status "$name: done already"; return 0; fi
  if [ ! -f "$adapter/adapter_config.json" ]; then status "$name: no adapter at $adapter"; return 1; fi
  status "$name: waiting for an idle card"
  wait_idle || return 1
  status "$name: baseline.py --adapter $adapter --condition $cond"
  watched "$out/test_eval.log" $PY scripts/baseline.py --model $MODEL --adapter "$adapter" --out-dir "$out" \
      --condition "$cond" --wandb
  rc=$?
  if [ -f "$out/baseline_test.json" ]; then status "$name: done; test v2 $(pct "$out/baseline_test.json")%"; return 0; fi
  status "$name failed (rc $rc)"; return 1
}

probe_stage() {   # run_gated.sh's stage 0: resumable, complete when every train prompt has a row
  local rows have attempt=0 rc
  rows=$(wc -l < "$GOLDS")
  have() { if [ -f "$PROBE" ]; then wc -l < "$PROBE"; else echo 0; fi; }
  if [ "$(have)" -ge "$rows" ]; then status "probe: complete ($(have) rows)"; return 0; fi
  if [ ! -f "$G1/best/adapter_config.json" ]; then status "probe: no round-1 adapter"; return 1; fi
  while [ "$(have)" -lt "$rows" ] && [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    status "probe: waiting for an idle card"
    wait_idle || return 1
    status "probe attempt $attempt ($(have) of $rows rows so far) on $G1/best"
    watched "$RUN/probe.log" $PY scripts/probe_difficulty.py --model $MODEL --adapter "$G1/best" --out "$PROBE"
    rc=$?
    status "probe attempt $attempt rc $rc ($(have) rows)"
  done
  if [ "$(have)" -lt "$rows" ]; then status "probe incomplete after $attempt attempts"; return 1; fi
  $PY scripts/select_frontier.py --difficulty "$PROBE" --summary "$RUN/difficulty_summary.md" \
      --cliff-out "$RUN/train_cliff.jsonl" >> "$RUN/probe.log" 2>&1
  status "probe: done; shelves in $RUN/difficulty_summary.md"
}

status "start: STEPS=\"$STEPS\" WAIT_PID=${WAIT_PID:-none}"
if [ -n "$WAIT_PID" ]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  status "pid $WAIT_PID has exited"
fi

if has seed2; then
  sft_arm golds-s7 "$GOLDS" runs/qwen3.5-4b-sft-ds100-s7 || status "golds-s7: gave up"
  sft_arm distill-s7 "$DISTILLED" runs/qwen3.5-4b-sft-distill-s7 || status "distill-s7: gave up"
fi
if has seeds; then
  $PY scripts/distill_followup/compare_seeds.py > "$RUN/seeds.log" 2>&1 \
    && status "seeds: $RUN/seeds.md" || { tail -3 "$RUN/seeds.log"; status "seeds FAILED"; }
fi
if has grpo1; then
  grpo_round grpo1 scripts/train_grpo.py "$START" "$G1" || status "grpo1: gave up"
fi
if has test1; then
  test_row test1 "$G1/best" "$G1" distill-grpo || true
fi
if has probe; then
  probe_stage || status "probe: gave up"
fi
if has grpo2; then
  if [ -f "$PROBE" ] && [ "$(wc -l < "$PROBE")" -ge "$(wc -l < "$GOLDS")" ]; then
    grpo_round grpo2 scripts/train_grpo_gated.py "$G1/best" "$G2" --difficulty "$PROBE" || status "grpo2: gave up"
  else
    status "grpo2: skipped, the probe is incomplete"
  fi
fi
if has test2; then
  test_row test2 "$G2/best" "$G2" distill-grpo-gated || true
fi
if has report; then
  $PY scripts/distill_followup/report_grpo.py > "$RUN/grpo.log" 2>&1 \
    && status "report: $RUN/grpo.md" || { tail -3 "$RUN/grpo.log"; status "report FAILED"; }
fi
status "done: $STEPS"
