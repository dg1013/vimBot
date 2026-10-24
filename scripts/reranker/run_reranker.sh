#!/bin/bash
# Reranker over the served adapter's 8 samples (plan item 3, 2026-10-16):
# train a yes/no verifier on executor-labelled train samples, select on val,
# compare a 4B and a 2B verifier (owner: "both, then compare"). Stages, each
# idempotent:
#   0  examples  train (leg-2 probe + cliff before-probe + the G = 32 re-check,
#                all of the served adapter) and val (the sample rule's 8 samples
#                + the greedy answer) -> $DIR/{train,val}_examples.jsonl
#   1  smoke     the 2B for 4 steps on 30 val prompts (wiring, scoring guard)
#   2  4B        verifier from the served 4B policy -> $DIR/verifier-4b/
#   3  2B        verifier from the 2B SFT policy    -> $DIR/verifier-2b/
#   4  report    -> $DIR/val_report.md (both verifiers, the baselines, beside the ask rule)
# It stops there: test v2 is scored only after the pre-registration is
# written and the owner says go (README "Test").
# Pause: touch $DIR/STOP_CHAIN (stops at the next stage boundary) and
# $DIR/verifier-*/STOP (a training stage stops after its step; the stage is
# restarted on the next run). Resume: run this script again.
REPO=/home/dgeorge28/dev/chatbot/vimBot
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python
DIR=${DIR:-$REPO/runs/reranker_2026-10-16}
RECHECK=${RECHECK:-$REPO/runs/cliff_recheck_2026-10-16/recheck_g32_qwen3.5-4b.jsonl}
ALLOW_NO_RECHECK=${ALLOW_NO_RECHECK:-0}
TRAIN_ARGS=${TRAIN_ARGS:-}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-2}
PAUSED_RC=3
STATUS=$DIR/status.txt
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DIR"
echo $$ > "$DIR/driver.pid"
log() { echo "$(date '+%m-%d %H:%M:%S') [reranker] $*" >> "$STATUS"; }
chain_paused() { [ -f "$DIR/STOP_CHAIN" ]; }
rows_in() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

watched() {   # <log> <cmd...> -- kill a stage whose log goes quiet for STALL_MIN
  local log=$1; shift
  "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -f "$log" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
      if [ "$age" -gt $((STALL_MIN * 60)) ]; then
        log "WATCHDOG: no log activity for $((age / 60)) min, killing pid $pid"
        kill "$pid" 2>/dev/null; sleep 90; kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null; return 124
      fi
    fi
  done
  wait "$pid"; return $?
}

train_stage() {   # <name> <out dir> <done file> <args...>
  local name=$1 out=$2 done=$3; shift 3
  local attempt=0 rc
  if [ -f "$done" ]; then log "$name: done already"; return 0; fi
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    if chain_paused; then log "pause requested before $name"; exit $PAUSED_RC; fi
    attempt=$((attempt + 1))
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
      local aside="$out.partial-$(date +%Y%m%d-%H%M%S)"
      mv "$out" "$aside"; log "$name: a partial run was set aside as $(basename "$aside")"
    fi
    mkdir -p "$out"
    log "$name attempt $attempt: train_verifier.py $*"
    watched "$out/train.log" $PY scripts/reranker/train_verifier.py --out-dir "$out" "$@"
    rc=$?
    log "$name attempt $attempt rc=$rc"
    if [ $rc -eq $PAUSED_RC ]; then log "$name PAUSED; run again to restart the stage"; exit $PAUSED_RC; fi
    if [ -f "$done" ]; then log "$name: done ($(cat "$done" | tr -d '\n '))"; return 0; fi
    sleep 30
  done
  log "$name: failed after $attempt attempts -- read $out/train.log"; exit 1
}

if chain_paused; then log "pause requested before the chain started; stopping"; exit $PAUSED_RC; fi
log "driver started (pid $$)"

# Stage 0: examples.
if [ -s "$DIR/train_examples.jsonl" ] && [ -s "$DIR/val_examples.jsonl" ]; then
  log "examples: exist ($(rows_in "$DIR/train_examples.jsonl") train, $(rows_in "$DIR/val_examples.jsonl") val)"
else
  extra=""
  if [ -s "$RECHECK" ]; then extra="--probe $RECHECK"
  elif [ "$ALLOW_NO_RECHECK" != "1" ]; then log "the G = 32 re-check ($RECHECK) is missing; run item 1 first or set ALLOW_NO_RECHECK=1"; exit 1
  fi
  $PY scripts/reranker/build_examples.py --cases data/splits/train.jsonl --train \
      --probe runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-leg2.jsonl \
      --probe runs/cliff_2026-09-16/probe_before_qwen3.5-4b.jsonl $extra \
      --keep-unanimous 0.25 --max-per-prompt 8 --out "$DIR/train_examples.jsonl" >> "$DIR/build.log" 2>&1 \
    && $PY scripts/reranker/build_examples.py --cases data/splits/val.jsonl \
      --probe runs/sample_rule_2026-10-09/probe_val.jsonl \
      --greedy runs/clarify_2026-10-06/val/verdicts_served.jsonl \
      --out "$DIR/val_examples.jsonl" >> "$DIR/build.log" 2>&1 \
    || { log "examples failed -- read $DIR/build.log"; exit 1; }
  log "examples: $(rows_in "$DIR/train_examples.jsonl") train, $(rows_in "$DIR/val_examples.jsonl") val"
fi

# Stage 1: smoke on the 2B (wiring, the answer-position check, the scoring guard).
train_stage smoke "$DIR/smoke-2b" "$DIR/smoke-2b/verifier_summary.json" \
    --model qwen3.5-2b --init-adapter runs/qwen3.5-2b-sft/best \
    --train "$DIR/train_examples.jsonl" --val "$DIR/val_examples.jsonl" \
    --max-steps 4 --eval-every 2 --eval-limit 30 --no-eval-at-start

# Stages 2-3: the two verifiers.
train_stage verifier-4b "$DIR/verifier-4b" "$DIR/verifier-4b/verifier_summary.json" \
    --model qwen3.5-4b --init-adapter runs/qwen3.5-4b-grpo-gated/best \
    --train "$DIR/train_examples.jsonl" --val "$DIR/val_examples.jsonl" --wandb $TRAIN_ARGS
train_stage verifier-2b "$DIR/verifier-2b" "$DIR/verifier-2b/verifier_summary.json" \
    --model qwen3.5-2b --init-adapter runs/qwen3.5-2b-sft/best \
    --train "$DIR/train_examples.jsonl" --val "$DIR/val_examples.jsonl" --wandb $TRAIN_ARGS

# Stage 4: the val report, from each verifier's best-step val scores (no GPU).
best4=$($PY -c "import json;print(json.load(open('$DIR/verifier-4b/best/selection.json'))['step'])")
best2=$($PY -c "import json;print(json.load(open('$DIR/verifier-2b/best/selection.json'))['step'])")
$PY scripts/reranker/rerank_eval.py --examples "$DIR/val_examples.jsonl" --split val \
    --scores 4b="$DIR/verifier-4b/val_scores_step$best4.jsonl" \
    --scores 2b="$DIR/verifier-2b/val_scores_step$best2.jsonl" \
    --ask-report runs/sample_rule_2026-10-09/val/report.json \
    --verdicts runs/clarify_2026-10-06/val/verdicts_served.jsonl \
    --out "$DIR/val_report.md" --json "$DIR/val_report.json" >> "$DIR/build.log" 2>&1 \
  && log "report: $DIR/val_report.md (4B best step $best4, 2B best step $best2)" \
  || { log "report failed -- read $DIR/build.log"; exit 1; }
log "driver done; test v2 waits for the pre-registration and the owner's go"
