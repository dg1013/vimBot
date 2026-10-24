#!/bin/bash
# SFT round 2 on the cliff shelf for ONE model (survey item #2, DECISIONS.md
# 2026-09-15): the bridge between two difficulty-gated GRPO legs. Modelled on
# runs/grpo_gated_2026-09-14/run_gated.sh (idempotent stages, the inactivity
# watchdog, a chain-level pause file checked at every stage boundary), with
# these stages:
#
#   0  candidates   pool the input adapter's probe + the gated run's live log
#                   -> cliff candidate ids, a 400-prompt known sample (the
#                   retention control), their union as the re-probe set
#   1  before-probe scripts/probe_difficulty.py on the INPUT adapter over the
#                   re-probe set (~12 min on the 4B; resumable, pausable)
#   2  build        the SFT train file: the cliff as the before-probe measures
#                   it + a replay slice; the audit page (consistent-wrong rows)
#   3  sft          scripts/cliff_sft/train_sft_cliff.py --adapter $ADAPTER,
#                   harness-selected on the val subset, full val at the end
#                   -> runs/$MODEL-sft-cliff/  (~25-35 min on the 4B)
#   4  after-probe  the same re-probe set on the NEW adapter (best/)
#   5  test row     baseline.py --condition sft-cliff
#   6  report       conversion / retention / rows -> cliff_report_$MODEL.md,
#                   and summary.md; prints the launch line of the next gated leg
#
# Pause: bash scripts/cliff_sft/pause.sh (writes $DIR/STOP_CHAIN, asks the
# running stage to stop). The probes save their rows and resume; the SFT stage
# stops after its step and is RESTARTED on relaunch (a ~30-minute stage; the
# partial run is set aside as runs/$MODEL-sft-cliff.paused-<stamp>/); a test
# row in flight is redone. Resume: bash scripts/cliff_sft/launch.sh.
#
#   MODEL=qwen3.5-4b ADAPTER=runs/qwen3.5-4b-grpo-gated/best bash scripts/cliff_sft/run_cliff.sh
#
# Knobs (environment): SFT_ARGS (default "--epochs 2 --lr 5e-5 --eval-every 25"),
# BUILD_ARGS (e.g. "--replay-frac 0.3 --hold-flagged"), PROBE_ARGS
# ("--prompts-per-call 8"), DIFFICULTY (the files stage 0 pools, earliest
# first), STALL_MIN, DIR (the run folder; default runs/cliff_<today>), OUT (the
# trainer's folder; default runs/$MODEL-sft-cliff), TEST_ARGS (a smoke's --limit).
REPO=/home/dgeorge28/dev/chatbot/vimBot
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python
MODEL=${MODEL:-qwen3.5-4b}
DIR=${DIR:-$REPO/runs/cliff_$(date +%Y-%m-%d)}
ADAPTER=${ADAPTER:-$REPO/runs/$MODEL-grpo-gated/best}
COMPARE_DIR=${COMPARE_DIR:-$(dirname "$ADAPTER")}
OUT=${OUT:-$REPO/runs/$MODEL-sft-cliff}
DIFFICULTY=${DIFFICULTY:-"$REPO/runs/grpo_gated_2026-09-14/difficulty_probe_$MODEL.jsonl $REPO/runs/$MODEL-grpo-gated/difficulty_log.jsonl"}
SFT_ARGS=${SFT_ARGS:-"--epochs 2 --lr 5e-5 --eval-every 25"}
BUILD_ARGS=${BUILD_ARGS:-}
PROBE_ARGS=${PROBE_ARGS:-"--prompts-per-call 8"}
TEST_ARGS=${TEST_ARGS:-}            # reaches baseline.py (a smoke passes --limit)
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
PAUSED_RC=3
STATUS=$DIR/status.txt
CLIFF_IDS=$DIR/cliff_candidates_$MODEL.txt
KNOWN_IDS=$DIR/known_sample_$MODEL.txt
REPROBE_IDS=$DIR/reprobe_ids_$MODEL.txt
BEFORE=$DIR/probe_before_$MODEL.jsonl
AFTER=$DIR/probe_after_$MODEL.jsonl
TRAIN=$DIR/train_cliff_set_$MODEL.jsonl
TAUGHT_IDS=$DIR/cliff_taught_$MODEL.txt
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1
mkdir -p "$DIR" "$OUT"
echo $$ > "$DIR/driver.pid"          # launch.sh and pause.sh read it
echo "$MODEL" > "$DIR/current_model"
echo "$OUT" > "$DIR/current_out"          # pause.sh reads it (OUT may be overridden)
stamp() { date "+%m-%d %H:%M:%S"; }
log() { echo "$(stamp) [$MODEL] $*" >> "$STATUS"; }
summarize() { $PY scripts/summarize_runs.py --out "$DIR/summary.md" > /dev/null 2>&1; }
chain_paused() { [ -f "$DIR/STOP_CHAIN" ]; }
rows_in() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

watched() {   # <log> <cmd...>  -- the gated driver's inactivity watchdog, unchanged
  local log=$1; shift
  "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -f "$log" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
      if [ "$age" -gt $((STALL_MIN * 60)) ]; then
        log "WATCHDOG: no log activity for $((age / 60)) min, killing pid $pid ($(basename "$log"))"
        kill "$pid" 2>/dev/null; sleep 90; kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null; return 124
      fi
    fi
  done
  wait "$pid"; return $?
}

probe_stage() {   # <adapter> <out-file> <log>  -- resumable by id, pausable via $DIR/STOP
  local adapter=$1 out=$2 plog=$3 want attempt=0 rc
  want=$(rows_in "$REPROBE_IDS")
  if [ "$(rows_in "$out")" -ge "$want" ]; then log "probe $(basename "$out"): complete ($want rows), skipping"; return 0; fi
  while [ "$(rows_in "$out")" -lt "$want" ] && [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    log "probe $(basename "$out") attempt $attempt on $adapter ($(rows_in "$out") of $want rows so far)"
    watched "$plog" $PY scripts/probe_difficulty.py --model $MODEL --adapter "$adapter" --ids "$REPROBE_IDS" \
        --out "$out" --stop-file "$DIR/STOP" $PROBE_ARGS
    rc=$?
    log "probe $(basename "$out") attempt $attempt rc=$rc ($(rows_in "$out") rows)"
    if [ $rc -eq $PAUSED_RC ]; then
      if chain_paused; then log "probe PAUSED; relaunch to continue"; return $PAUSED_RC; fi
      log "probe paused by the watchdog's signal, not by request; continuing"
    fi
  done
  if [ "$(rows_in "$out")" -lt "$want" ]; then log "probe incomplete after $attempt attempts -- read $plog"; return 1; fi
  return 0
}

if chain_paused; then log "pause requested before the round started; stopping"; exit $PAUSED_RC; fi
log "driver started (pid $$) from $ADAPTER; SFT_ARGS='$SFT_ARGS' BUILD_ARGS='$BUILD_ARGS' PROBE_ARGS='$PROBE_ARGS'"
if [ ! -f "$ADAPTER/adapter_config.json" ]; then log "no adapter at $ADAPTER; nothing to continue from"; exit 1; fi
rm -f "$DIR/STOP" "$DIR/PAUSED" "$OUT/STOP"      # a relaunch is a request to run

# Stage 0: candidates.
if [ -f "$REPROBE_IDS" ] && [ -f "$CLIFF_IDS" ] && [ -f "$KNOWN_IDS" ]; then
  log "candidates: exist ($(rows_in "$CLIFF_IDS") cliff, $(rows_in "$KNOWN_IDS") known sample), skipping"
else
  for f in $DIFFICULTY; do [ -f "$f" ] || { log "difficulty file missing: $f"; exit 1; }; done
  $PY scripts/cliff_sft/build_cliff_set.py candidates --difficulty $DIFFICULTY \
      --cliff-ids-out "$CLIFF_IDS" --known-ids-out "$KNOWN_IDS" --reprobe-ids-out "$REPROBE_IDS" \
      >> "$DIR/build_$MODEL.log" 2>&1 || { log "candidates failed -- read $DIR/build_$MODEL.log"; exit 1; }
  log "candidates: $(rows_in "$CLIFF_IDS") cliff, $(rows_in "$KNOWN_IDS") known sample, $(rows_in "$REPROBE_IDS") to re-probe"
fi
if chain_paused; then log "pause requested; stopping before the before-probe"; exit $PAUSED_RC; fi

# Stage 1: the before-probe on the input adapter.
probe_stage "$ADAPTER" "$BEFORE" "$DIR/probe_before_$MODEL.log"; rc=$?
[ $rc -ne 0 ] && exit $rc
if chain_paused; then log "pause requested; stopping before the build"; exit $PAUSED_RC; fi

# Stage 2: the train file (the before-probe last, so it wins under the pool).
if [ -f "$TRAIN" ]; then
  log "build: $TRAIN exists ($(rows_in "$TRAIN") rows), skipping"
else
  $PY scripts/cliff_sft/build_cliff_set.py build --difficulty $DIFFICULTY "$BEFORE" --out "$TRAIN" \
      --cliff-ids-out "$TAUGHT_IDS" --audit "$DIR/cliff_audit_$MODEL.md" --held-out "$DIR/held_for_review_$MODEL.jsonl" \
      $BUILD_ARGS >> "$DIR/build_$MODEL.log" 2>&1 || { log "build failed -- read $DIR/build_$MODEL.log"; exit 1; }
  log "build: $(rows_in "$TRAIN") rows ($(rows_in "$TAUGHT_IDS") cliff taught); audit in cliff_audit_$MODEL.md"
fi
if chain_paused; then log "pause requested; stopping before SFT"; exit $PAUSED_RC; fi

# Stage 3: SFT round 2, continuing the adapter. A paused partial run is set
# aside and the stage restarts; a finished run (sft_val.json) is kept.
if [ -f "$OUT/sft_val.json" ]; then
  log "sft: $OUT/sft_val.json exists, skipping training"
else
  if [ -f "$OUT/PAUSED" ] || [ -d "$OUT/best" ]; then
    aside="$OUT.paused-$(date +%Y%m%d-%H%M%S)"
    mv "$OUT" "$aside"; mkdir -p "$OUT"
    log "sft: a partial run was set aside as $(basename "$aside"); restarting the stage"
  fi
  log "sft: train_sft_cliff.py --adapter $ADAPTER --train $(basename "$TRAIN") $SFT_ARGS"
  watched "$OUT/train.log" $PY scripts/cliff_sft/train_sft_cliff.py --model $MODEL --adapter "$ADAPTER" \
      --train "$TRAIN" --out-dir "$OUT" --wandb $SFT_ARGS
  rc=$?
  log "sft rc=$rc"
  if [ $rc -eq $PAUSED_RC ]; then log "sft PAUSED after step $($PY -c "import json;print(json.load(open('$OUT/PAUSED'))['step'])" 2>/dev/null); relaunch restarts the stage"; exit $PAUSED_RC; fi
  if [ $rc -eq 124 ]; then log "sft hung and was killed; relaunch restarts the stage"; exit 1; fi
  if [ $rc -ne 0 ]; then log "sft failed (rc=$rc) -- read $OUT/train.log"; exit $rc; fi
fi
summarize
if chain_paused; then log "pause requested; stopping before the after-probe"; exit $PAUSED_RC; fi

# Stage 4: the after-probe on the new adapter.
probe_stage "$OUT/best" "$AFTER" "$DIR/probe_after_$MODEL.log"; rc=$?
[ $rc -ne 0 ] && exit $rc
if chain_paused; then log "pause requested; stopping before the test row"; exit $PAUSED_RC; fi

# Stage 5: the test row, labelled sft-cliff.
if [ -f "$OUT/baseline_test.json" ]; then
  log "test row: baseline_test.json exists, skipping"
else
  log "test row: baseline.py --adapter $OUT/best --condition sft-cliff"
  watched "$OUT/test_eval.log" $PY scripts/baseline.py --model $MODEL --adapter "$OUT/best" --condition sft-cliff --out-dir "$OUT" --wandb $TEST_ARGS
  rc=$?
  log "test row rc=$rc"
  if chain_paused; then log "pause requested during the test row; it will be redone on relaunch"; exit $PAUSED_RC; fi
fi

# Stage 6: the report.
$PY scripts/cliff_sft/build_cliff_set.py report --before "$BEFORE" --after "$AFTER" \
    --cliff-ids "$TAUGHT_IDS" --known-ids "$KNOWN_IDS" --run-dir "$OUT" --compare-dir "$COMPARE_DIR" \
    --out "$DIR/cliff_report_$MODEL.md" >> "$DIR/build_$MODEL.log" 2>&1 \
    && log "report: $DIR/cliff_report_$MODEL.md" || log "report failed -- read $DIR/build_$MODEL.log"
summarize
best_step=$($PY -c "import json;print(json.load(open('$OUT/best/selection.json'))['step'])" 2>/dev/null)
log "driver done (best step ${best_step:-?}). Next gated leg (continues $OUT/best): MODELS=$MODEL LEG=-leg2 ADAPTER_FROM=sft-cliff bash runs/grpo_gated_2026-09-14/launch.sh"
