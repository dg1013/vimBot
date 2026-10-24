#!/bin/bash
# Data-scaling SFT points for the 4B (plan item 6's sizing experiment,
# 2026-10-16). Stages, each idempotent:
#   0  build   nested whole-component subsets of train -> $DIR/train_25.jsonl, train_50.jsonl
#   1  sft 25  scripts/train_sft.py on 25%  -> runs/$MODEL-sft-ds25/
#   2  sft 50  ... 50%                      -> runs/$MODEL-sft-ds50/
#   3  sft 100 ... the full train split     -> runs/$MODEL-sft-ds100/  (re-run under
#              today's kernels and corpus, so all three points share one stack)
#   4  report  -> $DIR/report.md
# Every point: the round-1 recipe (scripts/train_sft.py defaults: LoRA r 16,
# alpha 32, 3 epochs, lr 1e-4 cosine, effective batch 32), one val-subset eval
# per epoch (--eval-every = steps per epoch), no step-0 eval (it is the same
# base for all three), the selected adapter scored on full val by the trainer.
# A stage is done when its sft_val.json exists. An interrupted SFT stage is set
# aside and restarted (a 15-80 minute stage; train_sft.py has no resume).
# Pause: touch $DIR/STOP_CHAIN -- the chain stops at the next stage boundary
# (an SFT stage in flight runs to its end). Resume: run this script again.
#
#   bash scripts/data_scaling/run_scaling.sh
REPO=/home/dgeorge28/dev/chatbot/vimBot
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python
MODEL=${MODEL:-qwen3.5-4b}
DIR=${DIR:-$REPO/runs/data_scaling_2026-10-16}
SFT_ARGS=${SFT_ARGS:-}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
PAUSED_RC=3
STATUS=$DIR/status.txt
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DIR"
echo $$ > "$DIR/driver.pid"
log() { echo "$(date '+%m-%d %H:%M:%S') [scaling $MODEL] $*" >> "$STATUS"; }
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

sft_point() {   # <pct> <train file>
  local pct=$1 train=$2 out=$REPO/runs/$MODEL-sft-ds$1 attempt=0 rc rows every
  if [ -f "$out/sft_val.json" ]; then log "sft $pct%: done already ($out/sft_val.json)"; return 0; fi
  rows=$(rows_in "$train"); every=$(( rows / 32 )); [ $every -lt 1 ] && every=1
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    if chain_paused; then log "pause requested before sft $pct%"; exit $PAUSED_RC; fi
    attempt=$((attempt + 1))
    if [ -d "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
      local aside="$out.partial-$(date +%Y%m%d-%H%M%S)"
      mv "$out" "$aside"; log "sft $pct%: a partial run was set aside as $(basename "$aside")"
    fi
    mkdir -p "$out"
    log "sft $pct% attempt $attempt: $rows rows, eval every $every steps (1 epoch) -> $out"
    watched "$out/train.log" $PY scripts/train_sft.py --model $MODEL --train "$train" --out-dir "$out" \
        --eval-every $every --no-eval-at-start --wandb $SFT_ARGS
    rc=$?
    log "sft $pct% attempt $attempt rc=$rc"
    if [ -f "$out/sft_val.json" ]; then
      log "sft $pct%: done; full val $($PY -c "import json;print(json.load(open('$out/sft_val.json'))['metrics']['exec_match_pct'])" 2>/dev/null)%"
      return 0
    fi
    sleep 30
  done
  log "sft $pct%: no sft_val.json after $attempt attempts -- read $out/train.log"; exit 1
}

if chain_paused; then log "pause requested before the chain started; stopping"; exit $PAUSED_RC; fi
log "driver started (pid $$), SFT_ARGS='$SFT_ARGS'"

if [ -s "$DIR/train_25.jsonl" ] && [ -s "$DIR/train_50.jsonl" ]; then
  log "build: subsets exist ($(rows_in "$DIR/train_25.jsonl") / $(rows_in "$DIR/train_50.jsonl") rows)"
else
  $PY scripts/data_scaling/data_scaling.py build --out-dir "$DIR" >> "$DIR/build.log" 2>&1 \
      || { log "build failed -- read $DIR/build.log"; exit 1; }
  log "build: $(rows_in "$DIR/train_25.jsonl") / $(rows_in "$DIR/train_50.jsonl") rows"
fi

sft_point 25 "$DIR/train_25.jsonl"
sft_point 50 "$DIR/train_50.jsonl"
sft_point 100 "$REPO/data/splits/train.jsonl"

$PY scripts/data_scaling/data_scaling.py report --out-dir "$DIR" \
    --point 25=runs/$MODEL-sft-ds25 --point 50=runs/$MODEL-sft-ds50 --point 100=runs/$MODEL-sft-ds100 \
    >> "$DIR/build.log" 2>&1 && log "report: $DIR/report.md" || { log "report failed -- read $DIR/build.log"; exit 1; }
log "driver done"
