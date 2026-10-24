#!/usr/bin/env bash
# The difficulty probe (G = 8 on every train prompt) for the SFT adapters
# that have never had RL (scripts/fuel_probe/README.md).
#
#   bash scripts/fuel_probe/run.sh
#
# Phase 1, per model: the first SMOKE prompts, then the gate in compare.py
# (answers stop on their own and parse; the pace is logged). These rows are
# the start of the full run, not a throwaway. Phase 2: the rest of the train
# split, one model at a time, then each model's shelf summary and the
# cross-model report. Resumable (the probe skips rows already written);
# pausable (`touch $R/STOP`: the call in flight finishes, exit 3); a crash is
# retried from the rows on disk. Log of record: $R/status.txt.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=${R:-runs/fuel_probe_2026-09-26}
MODELS=${MODELS:-"qwen2.5-coder-7b seed-coder-8b granite-4.2-8b"}
SMOKE=${SMOKE:-100}
RETRIES=${RETRIES:-3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$R"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$R/status.txt"; }

probe() {   # model [extra probe args]; returns the probe's exit code
  local m=$1; shift
  local attempt=1 rc
  while :; do
    # </dev/null: nothing here should read the driver's stdin
    conda run --no-capture-output -n vimBot python scripts/probe_difficulty.py \
        --model "$m" --adapter "runs/$m-sft/best" --out "$R/difficulty_probe_$m.jsonl" \
        --stop-file "$R/STOP" "$@" >> "$R/probe_$m.log" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$attempt" -ge "$RETRIES" ]; then return "$rc"; fi
    log "[$m] crashed rc=$rc (attempt $attempt of $RETRIES); retrying in 60 s from the rows on disk"
    attempt=$((attempt + 1))
    sleep 60
  done
}

for m in $MODELS; do
  log "[$m] smoke: the first $SMOKE train prompts"
  probe "$m" --limit "$SMOKE"; rc=$?
  if [ "$rc" -eq 3 ]; then log "PAUSED during the smoke of $m -- relaunch to continue"; exit 3; fi
  if [ "$rc" -ne 0 ]; then log "[$m] smoke FAILED rc=$rc -- see $R/probe_$m.log"; exit "$rc"; fi
  if ! conda run -n vimBot python scripts/fuel_probe/compare.py --run "$R" --check "$m" \
        >> "$R/status.txt" 2>&1 < /dev/null; then
    log "[$m] stopped at the smoke gate -- see the line above"; exit 1
  fi
done

for m in $MODELS; do
  log "[$m] full probe"
  probe "$m"; rc=$?
  if [ "$rc" -eq 3 ]; then log "PAUSED during the full probe of $m -- relaunch to continue"; exit 3; fi
  if [ "$rc" -ne 0 ]; then log "[$m] FAILED rc=$rc -- see $R/probe_$m.log"; exit "$rc"; fi
  conda run -n vimBot python scripts/select_frontier.py \
      --difficulty "$R/difficulty_probe_$m.jsonl" --summary "$R/difficulty_summary_$m.md" \
      >> "$R/probe_$m.log" 2>&1 < /dev/null
  log "[$m] done: $(grep -E '^probe done' "$R/probe_$m.log" | tail -1)"
done

if conda run -n vimBot python scripts/fuel_probe/compare.py --run "$R" --models "$MODELS" \
      >> "$R/compare.log" 2>&1 < /dev/null; then
  log "report: $R/report.md"
else
  log "report FAILED -- see $R/compare.log"; exit 1
fi
