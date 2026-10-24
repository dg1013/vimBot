#!/usr/bin/env bash
# Stage 2 of training the ask decision: GRPO with a price on questions
# (scripts/ask_grpo/README.md).
#
#   COST=0.2 bash scripts/ask_grpo/launch.sh   (RUN defaults to runs/ask_grpo_<date>_c<COST>)
#
#   1 prompts  stage 1's train-split noted rows + fair replay (CPU)
#   2 grpo     continue stage 1's final/ with the two-turn rollout and the
#              ask-cost reward (train_grpo_ask.py); resumes from checkpoints
#   3 eval     the clarify harness on stage 1's held-out rows and val
#   4 gate     scripts/ask_sft/compare.py: against the served adapter, and
#              against stage 1 on the same rows; then scripts/ask_sft/analyze.py
#              (leakage, the P(ask) ranking, the val-picked threshold baseline)
#
# Stages are skipped when their output exists; GRPO resumes from its latest
# checkpoint; pause with touch $RUN/STOP (between stages) or the GRPO run's
# own STOP file in its out dir. Log: $RUN/status.txt. LIMIT=N: a smoke (16
# prompts, 2 steps, an N-row eval of the held-out rows, no gates).
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3.5-4b}
COST=${COST:-0.2}
# one RUN per cost: the eval resumes from rows on disk, so two costs sharing
# $RUN/eval would score the second cost on the first one's answers
RUN=${RUN:-runs/ask_grpo_$(date +%F)_c${COST}}
STAGE1_RUN=${STAGE1_RUN:-runs/ask_sft_2026-10-07}
ADAPTER=${ADAPTER:-runs/$MODEL-ask-sft/final}
DIAG=${DIAG:-runs/clarify_2026-10-06}
OUT=${OUT:-runs/$MODEL-ask-grpo-c${COST}}
GRPO_ARGS=${GRPO_ARGS:-"--epochs 2 --eval-every 25"}
LIMIT=${LIMIT:-}
RETRIES=${RETRIES:-3}
RETRY_WAIT=${RETRY_WAIT:-60}
mkdir -p "$RUN"
rm -f "$RUN/PAUSED"
# The smoke trains on 16 prompts, two full steps of --prompts-per-step 8 (TRL
# drops an incomplete step, so fewer prompts than a step would train on
# nothing), and logs every step, so both print the ask metrics. LIMIT sets
# the number of eval rows.
smoke=(); [ -n "$LIMIT" ] && smoke=(--limit 16 --max-steps 2 --logging-steps 1 --eval-subset 8
                                   --no-eval-at-start)

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$RUN/status.txt"; }
py() { conda run --no-capture-output -n vimBot python "$@"; }

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

[ -e "$ADAPTER/adapter_config.json" ] || { log "no stage-1 adapter at $ADAPTER"; exit 2; }

stage prompts "$RUN/rl_prompts.jsonl" "$RUN/prompts.log" \
  py scripts/ask_grpo/build_prompts.py --stage1-run "$STAGE1_RUN" --out "$RUN/rl_prompts.jsonl"
tail -n 1 "$RUN/prompts.log" | sed 's/^/    /' | tee -a "$RUN/status.txt"

resume=(); ls -d "$OUT"/checkpoint-* > /dev/null 2>&1 && resume=(--resume)
# shellcheck disable=SC2086  # GRPO_ARGS is a flag list
stage grpo "$OUT/final/adapter_config.json" "$RUN/grpo.log" \
  py scripts/ask_grpo/train_grpo_ask.py --model "$MODEL" --adapter "$ADAPTER" \
    --train "$RUN/rl_prompts.jsonl" --out-dir "$OUT" --run-name "$(basename "$OUT")" \
    --ask-cost "$COST" --bank-verdicts "$DIAG/diag/verdicts_served.jsonl" \
    --bank-dialogues "$STAGE1_RUN/dialogues.jsonl" $GRPO_ARGS "${resume[@]}" "${smoke[@]}"

sets="$STAGE1_RUN/heldout.jsonl data/splits/val.jsonl"
[ -n "$LIMIT" ] && sets="$STAGE1_RUN/heldout.jsonl"
stage eval "" "$RUN/eval.log" \
  env RUN="$RUN/eval" SETS="$sets" CHECKPOINTS="askgrpo=$OUT/final" LIMIT="$LIMIT" \
    bash scripts/clarify/run.sh

if [ -z "$LIMIT" ]; then
  stage "gate vs served" "" "$RUN/gate.log" \
    py scripts/ask_sft/compare.py --held-new "$RUN/eval/heldout/verdicts_askgrpo.jsonl" \
      --held-ref "$DIAG/diag/verdicts_served.jsonl" \
      --val-new "$RUN/eval/val/verdicts_askgrpo.jsonl" --val-ref "$DIAG/val/verdicts_served.jsonl" \
      --out "$RUN/gate_vs_served.md"
  stage "gate vs stage 1" "" "$RUN/gate.log" \
    py scripts/ask_sft/compare.py --held-new "$RUN/eval/heldout/verdicts_askgrpo.jsonl" \
      --held-ref "$STAGE1_RUN/eval/heldout/verdicts_asksft.jsonl" \
      --val-new "$RUN/eval/val/verdicts_askgrpo.jsonl" \
      --val-ref "$STAGE1_RUN/eval/val/verdicts_asksft.jsonl" --out "$RUN/gate_vs_stage1.md"
  # leakage, the ranking, the threshold baseline: rows either stage trained on count
  stage analysis "" "$RUN/gate.log" \
    py scripts/ask_sft/analyze.py --run "$STAGE1_RUN" --label askgrpo --eval-dir "$RUN/eval" \
      --trained "$STAGE1_RUN/train_set.jsonl" --trained "$RUN/rl_prompts.jsonl" \
      --out "$RUN/analysis.md"
  head -n 7 "$RUN/gate_vs_served.md" | tee -a "$RUN/status.txt"
fi
log "driver done"
