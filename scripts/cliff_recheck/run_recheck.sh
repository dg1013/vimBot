#!/bin/bash
# Re-check the never-solved train prompts at G = 32 (plan item 1, 2026-10-16).
# Stages, each idempotent:
#   0  ids     the train prompts the latest full probe of the served adapter
#              put at 0 of 8 -> $DIR/recheck_ids.txt
#   1  probe   scripts/probe_difficulty.py --num-generations 32 on them, from
#              the served adapter (resumable by id, pausable, inactivity watchdog,
#              retried after a crash such as the WSL2 "device not ready")
#   2  report  -> $DIR/report.md (+ report.json)
# Pause: touch $DIR/STOP_CHAIN and $DIR/STOP (the probe finishes its call and
# exits 3). Resume: run this script again.
#
#   bash scripts/cliff_recheck/run_recheck.sh
#
# Knobs: MODEL, ADAPTER, SOURCE (the 8-sample probe that defines the cliff),
# DIR, PROBE_ARGS (default 2 prompts per call = 64 sequences, ~12 GB on the
# 4B), STALL_MIN, MAX_ATTEMPTS.
REPO=/home/dgeorge28/dev/chatbot/vimBot
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python
MODEL=${MODEL:-qwen3.5-4b}
ADAPTER=${ADAPTER:-$REPO/runs/$MODEL-grpo-gated/best}
SOURCE=${SOURCE:-$REPO/runs/grpo_gated_2026-09-14/difficulty_probe_$MODEL-leg2.jsonl}
DIR=${DIR:-$REPO/runs/cliff_recheck_2026-10-16}
PROBE_ARGS=${PROBE_ARGS:-"--prompts-per-call 2 --seed 42"}
STALL_MIN=${STALL_MIN:-30}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-4}
PAUSED_RC=3
STATUS=$DIR/status.txt
IDS=$DIR/recheck_ids.txt
OUT=$DIR/recheck_g32_$MODEL.jsonl
PLOG=$DIR/probe_g32_$MODEL.log
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DIR"
echo $$ > "$DIR/driver.pid"
log() { echo "$(date '+%m-%d %H:%M:%S') [recheck $MODEL] $*" >> "$STATUS"; }
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

if chain_paused; then log "pause requested before the stage started; stopping"; exit $PAUSED_RC; fi
log "driver started (pid $$): adapter $ADAPTER, source $(basename "$SOURCE"), PROBE_ARGS='$PROBE_ARGS'"
[ -f "$ADAPTER/adapter_config.json" ] || { log "no adapter at $ADAPTER"; exit 1; }
rm -f "$DIR/STOP" "$DIR/PAUSED"

# Stage 0: the ids.
if [ -s "$IDS" ]; then
  log "ids: $(rows_in "$IDS") exist, skipping"
else
  $PY scripts/cliff_recheck/cliff_recheck.py ids --probe "$SOURCE" --out "$IDS" \
      --manifest "$DIR/recheck_manifest.json" >> "$DIR/build.log" 2>&1 \
      || { log "ids failed -- read $DIR/build.log"; exit 1; }
  log "ids: $(rows_in "$IDS") never-solved train prompts"
fi

# Stage 1: the 32-sample probe.
want=$(rows_in "$IDS"); attempt=0
while [ "$(rows_in "$OUT")" -lt "$want" ] && [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt + 1))
  log "probe attempt $attempt ($(rows_in "$OUT") of $want rows so far)"
  watched "$PLOG" $PY scripts/probe_difficulty.py --model $MODEL --adapter "$ADAPTER" --ids "$IDS" \
      --num-generations 32 --out "$OUT" --stop-file "$DIR/STOP" $PROBE_ARGS
  rc=$?
  log "probe attempt $attempt rc=$rc ($(rows_in "$OUT") rows)"
  if [ $rc -eq $PAUSED_RC ] && chain_paused; then log "probe PAUSED; run again to continue"; exit $PAUSED_RC; fi
  [ $rc -ne 0 ] && [ "$(rows_in "$OUT")" -lt "$want" ] && sleep 30
done
if [ "$(rows_in "$OUT")" -lt "$want" ]; then log "probe incomplete after $attempt attempts -- read $PLOG"; exit 1; fi
log "probe complete: $(rows_in "$OUT") rows"

# Stage 2: the report.
$PY scripts/cliff_recheck/cliff_recheck.py report --probe "$SOURCE" --ids "$IDS" --recheck "$OUT" \
    --out "$DIR/report.md" --json "$DIR/report.json" >> "$DIR/build.log" 2>&1 \
    && log "report: $DIR/report.md" || { log "report failed -- read $DIR/build.log"; exit 1; }
log "driver done"
