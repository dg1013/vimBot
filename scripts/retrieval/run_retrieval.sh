#!/usr/bin/env bash
# The retrieval experiment end to end: plans (CPU) -> base arms -> adapter arms -> report.
# Resumable: finished arms are skipped by run_arms.py; plans are rebuilt only if missing.
#   bash scripts/retrieval/run_retrieval.sh            # full run, logs to $RUN/log.txt
#   LIMIT=4 RUN=runs/retrieval_smoke bash scripts/retrieval/run_retrieval.sh   # smoke
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/retrieval_2026-10-18}
MODEL=${MODEL:-qwen3.5-4b}
ADAPTER=${ADAPTER:-runs/qwen3.5-4b-grpo-gated/best}
PY=${PY:-"conda run --no-capture-output -n vimBot python"}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
mkdir -p "$RUN"
log() { echo "$(date '+%m-%d %H:%M:%S') retrieval: $*" | tee -a "$RUN/log.txt"; }

if [ ! -f "$RUN/plan_manifest.json" ]; then
  log "plans"
  $PY scripts/retrieval/retrieval.py plan --model "$MODEL" --out "$RUN" 2>&1 | tee -a "$RUN/log.txt"
fi
log "base arms"
$PY scripts/retrieval/run_arms.py --model "$MODEL" --run "$RUN" $LIMIT_ARG \
    --arms base-k0:plan_k0 base-random5:plan_random5 base-retr1:plan_retr1 base-retr3:plan_retr3 \
           base-retr5:plan_retr5 base-retr8:plan_retr8 2>&1 | grep -v "Loading weights\|Fetching" | tee -a "$RUN/log.txt"
log "adapter arms"
$PY scripts/retrieval/run_arms.py --model "$MODEL" --run "$RUN" $LIMIT_ARG --adapter "$ADAPTER" \
    --arms adapter-k0:plan_k0 adapter-retr5:plan_retr5 2>&1 | grep -v "Loading weights\|Fetching" | tee -a "$RUN/log.txt"
log "report"
$PY scripts/retrieval/retrieval.py report --run "$RUN" 2>&1 | tee -a "$RUN/log.txt" | tail -40
log "done"
