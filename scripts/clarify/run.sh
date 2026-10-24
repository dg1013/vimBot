#!/usr/bin/env bash
# The clarification experiment end to end (scripts/clarify/README.md).
#
#   RUN=runs/clarify_2026-10-06 bash scripts/clarify/run.sh
#
# For each case set in SETS (val = data/splits/val.jsonl; diag = the noted
# never-solved train rows, built by make_diag_cases.py), in order:
#   turn 1 for each checkpoint -> the simulated user (the 9B, NF4) ->
#   turn 2 for each checkpoint -> sandbox scoring -> report.md ->
#   transcripts_<label>.jsonl/.md (every prompt and action, for hand analysis)
# The launch records $RUN/prompt_fingerprint.json first: the transcripts
# rebuild every prompt from the code, and check it has not changed since.
# Every stage is resumable (rows on disk are skipped), so relaunching the
# same command continues after a pause, crash or reboot. Pause: touch
# $RUN/STOP -- the stage in flight finishes its case, writes $RUN/PAUSED and
# the driver exits 3. A crash is retried RETRIES times (the WSL2 driver
# hiccup of 2026-09-22). Log of record: $RUN/status.txt.
# LIMIT=N runs the first N cases of each set (smoke).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3.5-4b}
RUN=${RUN:-runs/clarify_$(date +%F)}
SETS=${SETS:-"val diag"}
CHECKPOINTS=${CHECKPOINTS:-"base=none served=runs/$MODEL-grpo-gated/best"}
SIMULATOR=${SIMULATOR:-qwen3.5-9b}
# The simulator's reasoning budget was picked by a hand audit of 50 replies
# (simulate.py's docstring, DECISIONS.md 2026-10-06).
SIM_ARGS=${SIM_ARGS:-"--think-budget 1024 --batch 16"}
LIMIT=${LIMIT:-}
RETRIES=${RETRIES:-3}
RETRY_WAIT=${RETRY_WAIT:-60}
mkdir -p "$RUN"
rm -f "$RUN/PAUSED"
lim=(); [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$RUN/status.txt"; }
reports=()

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

py() { conda run --no-capture-output -n vimBot python "$@"; }

# Once per run, before any generation: what the prompts are, as launched.
py scripts/clarify/transcripts.py --record-fingerprint "$RUN" | tee -a "$RUN/status.txt"

for set in $SETS; do
  # a named set writes to $RUN/<name>, a cases file to $RUN/<its basename>
  D="$RUN/$(basename "$set" .jsonl)"
  mkdir -p "$D"
  case "$set" in
    val)  CASES=data/splits/val.jsonl ;;
    diag) CASES="$RUN/diag_cases.jsonl"
          [ -f "$CASES" ] || py scripts/clarify/make_diag_cases.py --out "$CASES" \
                               | tee -a "$RUN/status.txt" ;;
    *)    CASES=$set ;;
  esac

  turn1_files=()
  for ck in $CHECKPOINTS; do
    label=${ck%%=*}; adapter=${ck#*=}
    stage "$set turn1 $label" "$D/turn1_$label.log" \
      py scripts/clarify/dialogue.py --phase turn1 --model "$MODEL" --adapter "$adapter" \
        --label "$label" --cases "$CASES" --out "$D/turn1_$label.jsonl" \
        --stop-file "$RUN/STOP" "${lim[@]}"
    turn1_files+=(--turn1 "$D/turn1_$label.jsonl")
  done

  # shellcheck disable=SC2086  # SIM_ARGS is a flag list
  stage "$set simulate" "$D/simulate.log" \
    py scripts/clarify/simulate.py --model "$SIMULATOR" --cases "$CASES" "${turn1_files[@]}" \
      --out "$D/replies.jsonl" --stop-file "$RUN/STOP" "${lim[@]}" $SIM_ARGS

  verdicts=()
  for ck in $CHECKPOINTS; do
    label=${ck%%=*}; adapter=${ck#*=}
    stage "$set turn2 $label" "$D/turn2_$label.log" \
      py scripts/clarify/dialogue.py --phase turn2 --model "$MODEL" --adapter "$adapter" \
        --label "$label" --cases "$CASES" --turn1 "$D/turn1_$label.jsonl" \
        --replies "$D/replies.jsonl" --out "$D/turn2_$label.jsonl" \
        --stop-file "$RUN/STOP" "${lim[@]}"
    stage "$set score $label" "$D/score_$label.log" \
      py scripts/clarify/score.py --cases "$CASES" --label "$label" \
        --turn1 "$D/turn1_$label.jsonl" --turn2 "$D/turn2_$label.jsonl" \
        --replies "$D/replies.jsonl" --out "$D/verdicts_$label.jsonl" "${lim[@]}"
    tail -n 6 "$D/score_$label.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"
    verdicts+=(--verdicts "$label=$D/verdicts_$label.jsonl")
  done

  extra=()
  [ "$set" = val ] && extra+=(--untouched "served=data/splits/val.jsonl")
  # a hand audit of the simulator (audit_sample.py), once one has been written
  [ -f "$D/simulator_audit.json" ] && extra+=(--audit "$D/simulator_audit.json")
  stage "$set report" "$D/report.log" \
    py scripts/clarify/report.py "${verdicts[@]}" "${extra[@]}" --cases-name "$set ($CASES)" \
      --out "$D/report.md"
  reports+=("$D/report.md")

  labels=()
  for ck in $CHECKPOINTS; do labels+=(--label "${ck%%=*}"); done
  stage "$set transcripts" "$D/transcripts.log" \
    py scripts/clarify/transcripts.py --cases "$CASES" --dir "$D" "${labels[@]}"
  tail -n "${#labels[@]}" "$D/transcripts.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"
done
log "driver done: ${reports[*]}"
