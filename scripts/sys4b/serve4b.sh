#!/usr/bin/env bash
# The 4B endpoints for scripts/sys4b (runs/sys4b_2026-11-08/preregistration.md), one on the card at a time:
#   served  :8011  the deployed 4B (runs/serve/qwen3.5-4b-served, merged bf16) with the verifier as a LoRA: the 4B line
#                  of scripts/serve_pooled/serve_pooled.sh, with room for 8 examples and both turns (--max-model-len
#                  2400, the 27B's) and a larger cache, since it has the card to itself
#   base    :8016  the registry's qwen3.5-4b at its pinned revision, text only (--language-model-only, as R0 served the
#                  base 27B), bf16, no adapter
#
#   bash scripts/sys4b/serve4b.sh start served      # returns once it answers
#   bash scripts/sys4b/serve4b.sh stop served
#
# Only the cache and the batch limits differ from serve_pooled.sh. They set how many sequences run at once, not what
# the model writes. The variables are this script's own (KV4B, CG4B, ...): scripts/serve27b/run.sh, which the driver
# sources, exports CG for the 27B.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/sys4b_2026-11-08}
WHICH=${2:-served}
U4B=${U4B:-0.60}
KV4B=${KV4B:-3221225472}         # 3 GiB: 8 attention layers of 32, 32 KiB a token
SEQS4B=${SEQS4B:-32}
MNBT4B=${MNBT4B:-4096}
CG4B=${CG4B:-"1 2 4 8 16 32"}
MAXLEN4B=${MAXLEN4B:-2400}
PY=$HOME/miniconda3/envs/vimBot/bin/python
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1
case "$WHICH" in
  served) PORT=8011 ;;
  base) PORT=8016 ;;
  *) echo "serve4b.sh: served or base, not $WHICH"; exit 2 ;;
esac
PID=$OUT/vllm_$WHICH.pid
LOG=$OUT/vllm_$WHICH.log

up() {   # seconds
  local t=0
  until curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 5; t=$((t + 5))
    if [ -f "$PID" ] && ! kill -0 "$(cat "$PID")" 2>/dev/null; then return 1; fi
    if [ "$t" -ge "$1" ]; then return 1; fi
  done
  return 0
}

registry() {   # "<hf_id> <revision>" of the registry's qwen3.5-4b: no repo id lives outside configs/models.yaml
  $PY -c 'import sys; sys.path.insert(0, "scripts"); from baseline import load_registry; r = load_registry("qwen3.5-4b"); print(r["hf_id"], r["revision"])'
}

case "${1:-start}" in
  start)
    common=(--dtype bfloat16 --max-model-len "$MAXLEN4B" --gpu-memory-utilization "$U4B" --max-num-seqs "$SEQS4B"
            --max-num-batched-tokens "$MNBT4B" --kv-cache-memory-bytes "$KV4B" --cudagraph-capture-sizes $CG4B
            --port "$PORT" --host 127.0.0.1 --generation-config vllm --seed 0)
    if [ "$WHICH" = served ]; then
      setsid conda run --no-capture-output -n vimBot vllm serve runs/serve/qwen3.5-4b-served \
        --served-model-name qwen3.5-4b-served "${common[@]}" --logprobs-mode processed_logits \
        --enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules verifier=runs/reranker_2026-10-16/verifier-4b/best \
        >> "$LOG" 2>&1 &
    else
      read -r HF REV <<< "$(registry)"
      if [ -z "${REV:-}" ]; then echo "no qwen3.5-4b in the registry"; exit 1; fi
      setsid conda run --no-capture-output -n vimBot vllm serve "$HF" --revision "$REV" --tokenizer-revision "$REV" \
        --served-model-name qwen3.5-4b-base --language-model-only "${common[@]}" \
        >> "$LOG" 2>&1 &
    fi
    echo $! > "$PID"
    up 900 || { echo "the $WHICH 4B endpoint did not come up (see $LOG)"; exit 1; }
    echo "$WHICH 4B up: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ;;
  stop)
    if [ -f "$PID" ]; then
      kill -- -"$(cat "$PID")" 2>/dev/null || kill "$(cat "$PID")" 2>/dev/null
      sleep 10
      kill -9 -- -"$(cat "$PID")" 2>/dev/null
      rm -f "$PID"
    fi
    sleep 5; echo "stopped $WHICH: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ;;
  *) echo "serve4b.sh start|stop served|base"; exit 2 ;;
esac
