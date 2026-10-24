#!/usr/bin/env bash
# Pick after asking + a better trigger: features -> turn-2 samples -> verifier scores -> evaluation.
# Every stage is skipped when its output exists, and sampling resumes by id, so the same command
# continues after a pause or crash.
#   bash scripts/ask_pick/run.sh                                   # full run
#   LIMIT=4 RUN=/tmp/.../smoke bash scripts/ask_pick/run.sh        # smoke (no evaluation)
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/ask_pick_2026-10-18}
PY=${PY:-"conda run --no-capture-output -n vimBot python"}
LIMIT_ARG=${LIMIT:+--limit $LIMIT}
mkdir -p "$RUN"
log() { echo "$(date '+%m-%d %H:%M:%S') ask_pick: $*" | tee -a "$RUN/log.txt"; }
quiet() { grep -v "Loading weights\|Fetching\|Warning\|warn" || true; }

if [ ! -f "$RUN/probe_features.pt" ]; then
  log "hidden states (diag, val, test)"
  $PY scripts/ask_pick/probe_features.py --out "$RUN/probe_features.pt" $LIMIT_ARG 2>&1 | quiet | tee -a "$RUN/log.txt"
fi
for split in val test; do
  log "turn-2 samples: $split"
  $PY scripts/ask_pick/sample_turn2.py --split $split --out "$RUN/turn2_$split.jsonl" $LIMIT_ARG 2>&1 | quiet | tee -a "$RUN/log.txt"
  if [ ! -f "$RUN/turn2_scores_$split.jsonl" ]; then
    log "verifier scores: $split"
    $PY scripts/ask_pick/score_turn2.py --split $split --samples "$RUN/turn2_$split.jsonl" \
        --out "$RUN/turn2_scores_$split.jsonl" 2>&1 | quiet | tee -a "$RUN/log.txt"
  fi
done
if [ -z "${LIMIT:-}" ]; then
  log "evaluation"
  $PY scripts/ask_pick/ask_pick_eval.py --run "$RUN" 2>&1 | quiet | tee -a "$RUN/log.txt" | tail -5
fi
log "done"
