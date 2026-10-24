#!/usr/bin/env bash
# Regenerate greedy predictions on the 31 reworded test rows for every
# results-table run, one distinct (model, adapter, base) at a time, then build
# the test-v2 report (scripts/remeasure/README.md).
#
#   bash scripts/remeasure/run.sh
#
# Each regeneration is baseline.py on runs/remeasure_2026-09-25/cases/test.jsonl
# with its own --out-dir, so no recorded run is touched. Resumable: a
# regeneration whose baseline_test.json exists is skipped. A crash is retried
# (RETRIES, default 2). Pause: `touch $R/STOP` stops before the next one.
# Log of record: $R/status.txt.
set -uo pipefail
cd "$(dirname "$0")/../.."

R=${R:-runs/remeasure_2026-09-25}
RETRIES=${RETRIES:-2}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$R/regen"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$R/status.txt"; }

configs=$(conda run -n vimBot python scripts/remeasure/plan.py --configs) || {
  log "plan FAILED"; exit 1; }

while IFS=$'\t' read -r name model adapter base condition; do
  [ -z "$name" ] && continue
  out="$R/regen/$name"
  if [ -f "$out/baseline_test.json" ]; then continue; fi
  if [ -f "$R/STOP" ]; then rm -f "$R/STOP"; log "PAUSED before $name -- relaunch to continue"; exit 3; fi
  args=(--model "$model" --cases "$R/cases/test.jsonl" --out-dir "$out")
  if [ "$adapter" != "none" ]; then
    args+=(--adapter "$adapter" --base-dtype "$base" --condition "$condition")
  fi
  attempt=1
  while :; do
    log "[$name] regenerating (attempt $attempt): ${args[*]}"
    # </dev/null: the loop reads the plan from stdin, and conda run would
    # otherwise swallow the remaining lines -- the first launch ended after
    # one regeneration that way.
    conda run --no-capture-output -n vimBot python scripts/baseline.py "${args[@]}" \
        >> "$R/regen_$name.log" 2>&1 < /dev/null
    rc=$?
    if [ "$rc" -eq 0 ]; then
      log "[$name] $(grep -E 'exec_match|exec-match' "$R/regen_$name.log" | tail -1 | sed 's/^ *//')"
      break
    fi
    if [ "$attempt" -ge "$RETRIES" ]; then
      log "[$name] FAILED rc=$rc -- see $R/regen_$name.log"; exit "$rc"
    fi
    attempt=$((attempt + 1))
    sleep 30
  done
done <<< "$configs"

if conda run -n vimBot python scripts/remeasure/v2_report.py --run "$R" >> "$R/report.log" 2>&1; then
  log "report: $R/report.md"
else
  log "report FAILED -- see $R/report.log"; exit 1
fi
