#!/usr/bin/env bash
# Sample each ceiling checkpoint in turn, then build the report
# (scripts/ceiling/README.md).
#
#   RUN=runs/ceiling_2026-09-22 CHECKPOINTS="base sft served" bash scripts/ceiling/run.sh
#
# CHECKPOINTS entries are LABEL or LABEL=ADAPTER_DIR. The known labels map to
# the 4B's rows: base = no adapter, sft = SFT round 1, served = the gated-GRPO
# adapter in the results table, cliffleg = the 2026-09-17 chain's stage C
# (cliff SFT then gated GRPO, 51.0% greedy). A label ending in -card samples
# at the model card's non-thinking settings (T 0.7, top-p 0.8, top-k 20; the
# card's presence penalty has no transformers equivalent and is left out)
# instead of the shared T = 1: base-card is the base at its best-case
# decoding, a sensitivity reading, not a point on the shared curves.
# Resumable: every stage skips the
# prompts already in its samples file, so relaunching the same command
# continues after a pause, crash or reboot. Pause: `touch $RUN/STOP` -- the
# stage in flight finishes its prompt, writes $RUN/PAUSED and the driver
# exits 3. Log of record: $RUN/status.txt.
set -uo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3.5-4b}
RUN=${RUN:-runs/ceiling_$(date +%F)}
N=${N:-128}
CASES=${CASES:-data/splits/test.jsonl}
CHECKPOINTS=${CHECKPOINTS:-"base sft served"}
EXTRA=${EXTRA:-}                  # more sample.py flags, e.g. "--temperature 0.7"
RETRIES=${RETRIES:-3}             # attempts per stage before it is logged FAILED
RETRY_WAIT=${RETRY_WAIT:-60}      # seconds between attempts
mkdir -p "$RUN"

log() { echo "$(date '+%m-%d %H:%M:%S') $*" | tee -a "$RUN/status.txt"; }

adapter_for() {
  case "$1" in
    *=*)                 echo "${1#*=}" ;;
    base|base-card)      echo none ;;
    sft|sft-card)        echo "runs/$MODEL-sft/best" ;;
    served|served-card)  echo "runs/$MODEL-grpo-gated/best" ;;
    cliffleg)            echo "runs/$MODEL-grpo-gated-cliffleg/best" ;;
    *)      echo "unknown checkpoint '$1' (give LABEL=ADAPTER_DIR)" >&2; return 2 ;;
  esac
}

flags_for() {
  case "$1" in
    *-card) echo "--temperature 0.7 --top-p 0.8 --top-k 20" ;;
    *)      echo "" ;;
  esac
}

greedy_for() {
  case "$1" in
    base|base-card)     echo "runs/$MODEL/exec_test.jsonl" ;;
    sft|sft-card)       echo "$RUN/greedy/$MODEL-sft_exec_test.jsonl" ;;
    served|served-card) echo "runs/$MODEL-grpo-gated/fla-kernels/exec_test.jsonl" ;;
    cliffleg)           echo "runs/$MODEL-grpo-gated-cliffleg/exec_test.jsonl" ;;
    *)                  echo "" ;;
  esac
}

labels=()
for ck in $CHECKPOINTS; do
  label=${ck%%=*}
  adapter=$(adapter_for "$ck") || exit 2
  flags="$(flags_for "$label") $EXTRA"
  labels+=("$label")
  log "[$label] sampling: adapter $adapter, n=$N, cases $CASES $flags"
  # A crash is retried: rows on disk are kept, so a retry resumes where the
  # crash left off. The case in mind is transient -- a WSL2 GPU driver error
  # ("CUDA driver error: device not ready") stopped the sft stage for 3.5 h
  # on 2026-09-22 -- and a real bug still fails after RETRIES attempts.
  attempt=1
  while :; do
    conda run --no-capture-output -n vimBot python scripts/ceiling/sample.py \
        --model "$MODEL" --adapter "$adapter" --label "$label" --cases "$CASES" --n "$N" \
        --out "$RUN/samples_$label.jsonl" --stop-file "$RUN/STOP" $flags \
        >> "$RUN/sample_$label.log" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ] || [ "$attempt" -ge "$RETRIES" ]; then break; fi
    log "[$label] crashed rc=$rc (attempt $attempt of $RETRIES); retrying in ${RETRY_WAIT}s from the rows on disk"
    attempt=$((attempt + 1))
    sleep "$RETRY_WAIT"
  done
  if [ "$rc" -eq 3 ]; then log "[$label] PAUSED -- relaunch the same command to continue"; exit 3; fi
  if [ "$rc" -ne 0 ]; then log "[$label] FAILED rc=$rc -- see $RUN/sample_$label.log"; exit "$rc"; fi
  log "[$label] $(grep -E '^  pass@1' "$RUN/sample_$label.log" | tail -1 | sed 's/^ *//')"
done

# The report: every samples file of this run, each paired with its greedy
# results-table row where one exists, and the unions of the greedy runs.
# PAIRS (A:B, space-separated) picks the paired differences; default: each
# checkpoint minus the one before it.
args=(--run "$RUN" --cases "$CASES")
for label in "${labels[@]}"; do
  args+=(--samples "$label=samples_$label.jsonl")
  g=$(greedy_for "$label")
  if [ -n "$g" ] && [ -f "$g" ]; then args+=(--greedy "$label=$g"); fi
done
for p in ${PAIRS:-}; do args+=(--pair "$p"); done
args+=(--pool "$MODEL=runs/$MODEL*/exec_test.jsonl" --pool "$MODEL=runs/$MODEL*/*/exec_test.jsonl"
       --pool "$MODEL=$RUN/greedy/$MODEL*_exec_test.jsonl"
       --pool "all=runs/*/exec_test.jsonl" --pool "all=runs/*/*/exec_test.jsonl"
       --pool "all=$RUN/greedy/*_exec_test.jsonl")
if conda run -n vimBot python scripts/ceiling/report.py "${args[@]}" >> "$RUN/report.log" 2>&1; then
  log "report: $RUN/report.md"
else
  log "report FAILED -- see $RUN/report.log"; exit 1
fi
