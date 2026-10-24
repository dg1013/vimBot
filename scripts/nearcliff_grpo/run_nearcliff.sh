#!/bin/bash
# Follow-up #1's full run (runs/followups_2026-10-16/preregistration.md):
# a GRPO leg at G = 32 on the near-cliff prompts the re-check found, from the
# served adapter, then the full test. Stages, each idempotent:
#   0  leg       nearcliff.py build -> $DIR/leg_train.jsonl, nearcliff_ids.txt, max_steps.txt
#   1  grpo      scripts/train_grpo_gated.py (unchanged) --num-generations 32
#                --prompts-per-step 2 --difficulty <leg-2 probe> <re-check>
#                -> runs/$MODEL-grpo-nearcliff/   (resumes from its checkpoints)
#   2  test row  baseline.py --condition grpo-nearcliff (test v2)
#   3  after     the near-cliff prompts re-probed at G = 32 with best/
#   4  report    -> $DIR/report.md
# Pause: touch $DIR/STOP_CHAIN and runs/$MODEL-grpo-nearcliff/STOP (the trainer
# checkpoints and exits 3); run again to resume.
REPO=/home/dgeorge28/dev/chatbot/vimBot
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python
MODEL=${MODEL:-qwen3.5-4b}
ADAPTER=${ADAPTER:-$REPO/runs/$MODEL-grpo-gated/best}
DIR=${DIR:-$REPO/runs/nearcliff_2026-10-16}
PROBE=${PROBE:-$REPO/runs/grpo_gated_2026-09-14/difficulty_probe_$MODEL-leg2.jsonl}
RECHECK=${RECHECK:-$REPO/runs/cliff_recheck_2026-10-16/recheck_g32_$MODEL.jsonl}
OUT=${OUT:-$REPO/runs/$MODEL-grpo-nearcliff}
G=${G:-32}                  # group size; 16 x 4 prompts after the 2026-10-16 paging finding
PPS=${PPS:-2}               # prompts per step (G x PPS completions per step)
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-3}
PAUSED_RC=3
STATUS=$DIR/status.txt
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DIR" "$OUT"
echo $$ > "$DIR/driver.pid"
log() { echo "$(date '+%m-%d %H:%M:%S') [nearcliff $MODEL] $*" >> "$STATUS"; }
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

if chain_paused; then log "pause requested before the run started"; exit $PAUSED_RC; fi
log "driver started (pid $$) from $ADAPTER"
rm -f "$DIR/STOP" "$OUT/STOP"

# Stage 0: the leg's prompts.
if [ -s "$DIR/leg_train.jsonl" ]; then
  log "leg: exists ($(rows_in "$DIR/leg_train.jsonl") rows)"
else
  $PY scripts/nearcliff_grpo/nearcliff.py build --recheck "$RECHECK" --probe "$PROBE" --out-dir "$DIR" \
      >> "$DIR/build.log" 2>&1 || { log "leg build failed -- read $DIR/build.log"; exit 1; }
  log "leg: $(rows_in "$DIR/nearcliff_ids.txt") near-cliff + anchors, $(cat "$DIR/max_steps.txt") steps"
fi
steps=${STEPS:-$(cat "$DIR/max_steps.txt")}

# Stage 1: the GRPO leg.
if [ -f "$OUT/grpo_val.json" ]; then
  log "grpo: grpo_val.json exists, skipping"
else
  attempt=0
  while [ $attempt -lt $MAX_ATTEMPTS ]; do
    attempt=$((attempt + 1))
    resume=()
    if ls -d "$OUT"/checkpoint-* > /dev/null 2>&1; then resume=(--resume); fi
    log "grpo attempt $attempt ${resume[*]:-(fresh)}: G $G x $PPS prompts, $steps steps"
    watched "$OUT/train.log" $PY scripts/train_grpo_gated.py --model $MODEL --adapter "$ADAPTER" \
        --train "$DIR/leg_train.jsonl" --run-name "$MODEL-grpo-nearcliff" --out-dir "$OUT" \
        --difficulty "$PROBE" "$RECHECK" --num-generations $G --prompts-per-step $PPS \
        --epochs 100 --max-steps "$steps" --eval-every 50 --wandb --print-completions 0 ${resume[@]}
    rc=$?
    log "grpo attempt $attempt rc=$rc"
    [ $rc -eq 0 ] && [ -f "$OUT/grpo_val.json" ] && break
    if [ $rc -eq $PAUSED_RC ] && chain_paused; then log "grpo PAUSED; run again to resume"; exit $PAUSED_RC; fi
    sleep 60
  done
  [ -f "$OUT/grpo_val.json" ] || { log "grpo: no grpo_val.json after $attempt attempts -- read $OUT/train.log"; exit 1; }
fi
if chain_paused; then log "pause requested before the test row"; exit $PAUSED_RC; fi

# Stage 2: the test row (test v2).
if [ -f "$OUT/baseline_test.json" ]; then
  log "test row: exists"
else
  log "test row: baseline.py --adapter $OUT/best --condition grpo-nearcliff"
  watched "$OUT/test_eval.log" $PY scripts/baseline.py --model $MODEL --adapter "$OUT/best" \
      --condition grpo-nearcliff --out-dir "$OUT" --wandb
  log "test row rc=$?"
  [ -f "$OUT/baseline_test.json" ] || { log "test row failed -- read $OUT/test_eval.log"; exit 1; }
fi
if chain_paused; then log "pause requested before the after-probe"; exit $PAUSED_RC; fi

# Stage 3: the near-cliff prompts, after.
AFTER=$DIR/after_g32_$MODEL.jsonl
want=$(rows_in "$DIR/nearcliff_ids.txt"); attempt=0
while [ "$(rows_in "$AFTER")" -lt "$want" ] && [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt + 1))
  log "after-probe attempt $attempt ($(rows_in "$AFTER") of $want)"
  watched "$DIR/after_probe.log" $PY scripts/probe_difficulty.py --model $MODEL --adapter "$OUT/best" \
      --ids "$DIR/nearcliff_ids.txt" --num-generations 32 --prompts-per-call 2 --seed 42 \
      --out "$AFTER" --stop-file "$DIR/STOP"
  rc=$?
  if [ $rc -eq $PAUSED_RC ] && chain_paused; then log "after-probe PAUSED"; exit $PAUSED_RC; fi
done
[ "$(rows_in "$AFTER")" -ge "$want" ] || { log "after-probe incomplete -- read $DIR/after_probe.log"; exit 1; }

# Stage 4: the report.
$PY scripts/nearcliff_grpo/nearcliff.py report --run-dir "$OUT" --before "$RECHECK" --after "$AFTER" \
    --ids "$DIR/nearcliff_ids.txt" --out "$DIR/report.md" >> "$DIR/build.log" 2>&1 \
    && log "report: $DIR/report.md" || { log "report failed -- read $DIR/build.log"; exit 1; }
log "driver done"
