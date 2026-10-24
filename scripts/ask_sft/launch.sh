#!/usr/bin/env bash
# Stage 1 of training the ask decision, end to end (scripts/ask_sft/README.md).
#
#   RUN=runs/ask_sft_2026-10-07 bash scripts/ask_sft/launch.sh
#
#   1 split     30% of the noted diag rows held out (CPU)
#   2 sample    K more questions per train-split ask row, T = 1 (GPU, 4B)
#   3 simulate  the audited simulated user replies to each (GPU, 9B)
#   4 turn2     the served adapter answers after each reply (GPU, 4B)
#   5 verify    execute every answer; keep dialogues that exec-match (CPU)
#   6 build     verified ask dialogues + placebo turns + replay -> train file
#   7 sft       continue the served adapter (train_sft_ask.py) -> final/
#   8 eval      the clarify harness on the held-out rows and val, final/
#   9 gate      compare.py against the served adapter on the same rows
#
# Every stage is skipped when its output exists, and the long ones resume
# from the rows on disk, so relaunching the same command continues after a
# pause, crash or reboot. Pause: touch $RUN/STOP (the stage in flight
# finishes its case or step, then the driver exits 3). Log: $RUN/status.txt.
# LIMIT=N shrinks every stage to N rows and the SFT to 2 steps (smoke).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3.5-4b}
RUN=${RUN:-runs/ask_sft_$(date +%F)}
SERVED=${SERVED:-runs/$MODEL-grpo-gated/best}
DIAG=${DIAG:-runs/clarify_2026-10-06}          # the run whose diag/val rows are reused
OUT_ADAPTER=${OUT_ADAPTER:-runs/$MODEL-ask-sft}
K=${K:-3}
SFT_ARGS=${SFT_ARGS:-"--epochs 2 --lr 2e-5 --eval-every 10"}
BUILD_ARGS=${BUILD_ARGS:-}
LIMIT=${LIMIT:-}
RETRIES=${RETRIES:-3}
RETRY_WAIT=${RETRY_WAIT:-60}
mkdir -p "$RUN/sample"
rm -f "$RUN/PAUSED"
lim=(); [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")
smoke_sft=(); [ -n "$LIMIT" ] && smoke_sft=(--max-steps 2 --limit 40 --eval-subset 8 --no-eval-at-start)
labels=(); for i in $(seq 1 "$K"); do labels+=("q$i"); done

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$RUN/status.txt"; }
py() { conda run --no-capture-output -n vimBot python "$@"; }

# stage NAME DONE_FILE LOG CMD... : skip if DONE_FILE exists; retries; pause
stage() {
  local name=$1 done_file=$2 logf=$3; shift 3
  if [ -n "$done_file" ] && [ -e "$done_file" ]; then log "[$name] already done"; return 0; fi
  if [ -e "$RUN/STOP" ]; then rm -f "$RUN/STOP"; log "[$name] PAUSED before start"; exit 3; fi
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

S="$RUN/sample"
stage split "$RUN/train_ask_cases.jsonl" "$RUN/split.log" \
  py scripts/ask_sft/split.py --diag-cases "$DIAG/diag_cases.jsonl" --out-dir "$RUN"
tail -n 1 "$RUN/split.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"

stage sample "$S/questions_q$K.jsonl" "$S/sample.log" \
  py scripts/ask_sft/sample_questions.py --model "$MODEL" --adapter "$SERVED" \
    --cases "$RUN/train_ask_cases.jsonl" --greedy "$DIAG/diag/turn1_served.jsonl" \
    --k "$K" --out-dir "$S" "${lim[@]}"

turn1_files=(); for q in "${labels[@]}"; do turn1_files+=(--turn1 "$S/questions_$q.jsonl"); done
stage simulate "" "$S/simulate.log" \
  py scripts/clarify/simulate.py --cases "$RUN/train_ask_cases.jsonl" "${turn1_files[@]}" \
    --out "$S/replies.jsonl" --think-budget 1024 --batch 16 --stop-file "$RUN/STOP"

for q in "${labels[@]}"; do
  stage "turn2 $q" "" "$S/turn2_$q.log" \
    py scripts/clarify/dialogue.py --phase turn2 --model "$MODEL" --adapter "$SERVED" \
      --label "$q" --cases "$S/cases_$q.jsonl" --turn1 "$S/questions_$q.jsonl" \
      --replies "$S/replies.jsonl" --out "$S/turn2_$q.jsonl" --stop-file "$RUN/STOP"
done

stage verify "" "$RUN/verify.log" \
  py scripts/ask_sft/verify.py --cases "$RUN/train_ask_cases.jsonl" \
    --greedy-verdicts "$DIAG/diag/verdicts_served.jsonl" --sample-dir "$S" \
    --labels "${labels[@]}" --out "$RUN/dialogues.jsonl"
tail -n 1 "$RUN/verify.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"

# shellcheck disable=SC2086  # BUILD_ARGS is a flag list
stage build "" "$RUN/build.log" \
  py scripts/ask_sft/build_set.py --diag-train "$RUN/diag_train.jsonl" \
    --held-out "$RUN/heldout.jsonl" --dialogues "$RUN/dialogues.jsonl" \
    --greedy "$DIAG/diag/turn1_served.jsonl" --out "$RUN/train_set.jsonl" $BUILD_ARGS

# shellcheck disable=SC2086  # SFT_ARGS is a flag list
stage sft "$OUT_ADAPTER/final/adapter_config.json" "$RUN/sft.log" \
  py scripts/ask_sft/train_sft_ask.py --model "$MODEL" --adapter "$SERVED" \
    --train "$RUN/train_set.jsonl" --out-dir "$OUT_ADAPTER" $SFT_ARGS "${smoke_sft[@]}"

sets="$RUN/heldout.jsonl data/splits/val.jsonl"
[ -n "$LIMIT" ] && sets="$RUN/heldout.jsonl"
stage eval "" "$RUN/eval.log" \
  env RUN="$RUN/eval" SETS="$sets" CHECKPOINTS="asksft=$OUT_ADAPTER/final" LIMIT="$LIMIT" \
    bash scripts/clarify/run.sh

if [ -z "$LIMIT" ]; then
  stage gate "" "$RUN/gate.log" \
    py scripts/ask_sft/compare.py --held-new "$RUN/eval/heldout/verdicts_asksft.jsonl" \
      --held-ref "$DIAG/diag/verdicts_served.jsonl" \
      --val-new "$RUN/eval/val/verdicts_asksft.jsonl" --val-ref "$DIAG/val/verdicts_served.jsonl" \
      --out "$RUN/gate.md"
  head -n 7 "$RUN/gate.md" | tee -a "$RUN/status.txt"
fi
log "driver done"
