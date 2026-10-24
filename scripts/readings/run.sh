#!/usr/bin/env bash
# The readings-first experiment end to end (scripts/readings/README.md).
#
#   RUN=runs/readings_2026-10-09 bash scripts/readings/run.sh
#
# For each case set in SETS (diag = the noted rows of the 2026-10-06 run;
# val = data/splits/val.jsonl), in order:
#   readings  the served adapter lists the readings, the sandbox runs them,
#             and it writes a question where two or more results differ
#             (readings.py)
#   prepare   the asked rows, in the files the clarify harness reads
#   simulate  the audited simulated user answers each question
#             (scripts/clarify/simulate.py, unchanged)
#   turn 2    the answer after the reply, and after the placebo
#             (scripts/clarify/dialogue.py --phase turn2, unchanged)
#   score / report / transcripts   (post.py), joined to REF's verdicts
# Every stage resumes from the rows on disk, so relaunching the same command
# continues after a pause, crash or reboot. Pause: touch $RUN/STOP -- the
# stage in flight finishes its case, writes $RUN/PAUSED, and the driver exits
# 3. Log of record: $RUN/status.txt. LIMIT=N runs the first N cases of each
# set (smoke).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3.5-4b}
RUN=${RUN:-runs/readings_$(date +%F)}
SETS=${SETS:-"diag val"}
ELICIT=${ELICIT:-iterative}   # readings one at a time; "list" for the one-reply form
ADAPTER=${ADAPTER:-runs/$MODEL-grpo-gated/best}
LABEL=${LABEL:-served}
REF=${REF:-runs/clarify_2026-10-06}
SIMULATOR=${SIMULATOR:-qwen3.5-9b}
SIM_ARGS=${SIM_ARGS:-"--think-budget 1024 --batch 16"}
LIMIT=${LIMIT:-}
RETRIES=${RETRIES:-3}
RETRY_WAIT=${RETRY_WAIT:-60}
mkdir -p "$RUN"
rm -f "$RUN/PAUSED"
lim=(); [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$RUN/status.txt"; }
py() { conda run --no-capture-output -n vimBot python "$@"; }

# stage NAME LOGFILE CMD... : run with retries; exit the driver on pause/failure
stage() {
  local name=$1 logf=$2; shift 2
  local attempt=1 rc
  log "[$name] start"
  while :; do
    "$@" >> "$logf" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$attempt" -ge "$RETRIES" ]; then break; fi
    log "[$name] crashed rc=$rc (attempt $attempt of $RETRIES); retrying in ${RETRY_WAIT}s"
    attempt=$((attempt + 1))
    sleep "$RETRY_WAIT"
  done
  if [ "$rc" -eq 3 ]; then log "[$name] PAUSED -- relaunch the same command to continue"; exit 3; fi
  if [ "$rc" -ne 0 ]; then log "[$name] FAILED rc=$rc -- see $logf"; exit "$rc"; fi
  log "[$name] done"
}

py scripts/readings/post.py fingerprint --run "$RUN" | tee -a "$RUN/status.txt"

for set in $SETS; do
  D="$RUN/$(basename "$set" .jsonl)"
  mkdir -p "$D"
  case "$set" in
    val)  CASES=data/splits/val.jsonl ;;
    diag) CASES="$REF/diag_cases.jsonl" ;;
    *)    CASES=$set ;;
  esac
  name=$(basename "$D")
  ref="$REF/$name/verdicts_$LABEL.jsonl"
  [ -f "$ref" ] || { log "no reference verdicts at $ref"; exit 2; }

  stage "$name readings" "$D/readings.log" \
    py scripts/readings/readings.py --model "$MODEL" --adapter "$ADAPTER" --label "$LABEL" \
      --cases "$CASES" --out "$D/readings_$LABEL.jsonl" --elicit "$ELICIT" \
      --stop-file "$RUN/STOP" "${lim[@]}"
  stage "$name prepare" "$D/prepare.log" \
    py scripts/readings/post.py prepare --cases "$CASES" --dir "$D" --label "$LABEL"
  tail -n 1 "$D/prepare.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"

  if [ -s "$D/asked_cases.jsonl" ]; then
    # shellcheck disable=SC2086  # SIM_ARGS is a flag list
    stage "$name simulate" "$D/simulate.log" \
      py scripts/clarify/simulate.py --model "$SIMULATOR" --cases "$D/asked_cases.jsonl" \
        --turn1 "$D/questions_readings.jsonl" --out "$D/replies.jsonl" \
        --stop-file "$RUN/STOP" $SIM_ARGS
    stage "$name turn2" "$D/turn2.log" \
      py scripts/clarify/dialogue.py --phase turn2 --model "$MODEL" --adapter "$ADAPTER" \
        --label readings --cases "$D/asked_cases.jsonl" --turn1 "$D/questions_readings.jsonl" \
        --replies "$D/replies.jsonl" --out "$D/turn2_readings.jsonl" --stop-file "$RUN/STOP"
    stage "$name turn2 placebo" "$D/turn2_placebo.log" \
      py scripts/clarify/dialogue.py --phase turn2 --model "$MODEL" --adapter "$ADAPTER" \
        --label readings_placebo --cases "$D/asked_cases.jsonl" \
        --turn1 "$D/questions_readings.jsonl" --replies "$D/placebo_replies.jsonl" \
        --out "$D/turn2_readings_placebo.jsonl" --stop-file "$RUN/STOP"
  else
    log "[$name] no row asks: simulate and turn 2 skipped"
  fi

  stage "$name score" "$D/score.log" \
    py scripts/readings/post.py score --cases "$CASES" --dir "$D" --label "$LABEL" --ref "$ref"
  tail -n 1 "$D/score.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"
  stage "$name report" "$D/report.log" \
    py scripts/readings/post.py report --cases "$CASES" --dir "$D" --label "$LABEL" \
      --name "$name ($CASES)"
  stage "$name transcripts" "$D/transcripts.log" \
    py scripts/readings/post.py transcripts --cases "$CASES" --dir "$D" --label "$LABEL" \
      --model "$MODEL"
  tail -n 1 "$D/transcripts.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"
done
log "driver done"
