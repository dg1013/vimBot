#!/usr/bin/env bash
# Serve the trained 27B on one RTX 3090: the 27B's text weights (runs/serve/qwen3.5-27b-text),
# quantized to 4-bit at load (bitsandbytes), with the trained adapter as a LoRA: the NF4 base plus
# adapter it was trained on, as serve_pooled.sh serves the 9B.
#
#   bash scripts/serve27b/serve27b.sh start     # up, waits until it answers
#   bash scripts/serve27b/serve27b.sh stop
#
# Memory: R0 measured 23.1 GB of the card at 90% utilization with thinking's cache. An answer here
# is short (prompts 117 tokens at the median, 404 at most), so the cache is set in bytes. Past about
# 23.5 GB, WSL2 pages GPU memory and everything slows tenfold.
#   KV    cache bytes
#   SEQS  sequences in one step
#   MNBT  tokens per scheduler step: bounds the activation peak vLLM reserves
#   CG    CUDA graph batch sizes; EAGER=1 runs without graphs (R0's attempt 2)
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/serve27b_2026-11-05}
PORT=${PORT:-8015}
U27=${U27:-0.92}
KV=${KV:-1073741824}
SEQS=${SEQS:-16}
MNBT=${MNBT:-2048}
CG=${CG:-"1 2 4 8"}
EAGER=${EAGER:-0}
BASE=${BASE:-runs/serve/qwen3.5-27b-text}
ADAPTER=${ADAPTER:-runs/qwen3.5-27b-sft-distill/best}
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1

up() {   # seconds
  local t=0
  until curl -s "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; do
    sleep 5; t=$((t + 5))
    if [ -f "$OUT/vllm_27b.pid" ] && ! kill -0 "$(cat "$OUT/vllm_27b.pid")" 2>/dev/null; then return 1; fi
    if [ "$t" -ge "$1" ]; then return 1; fi
  done
  return 0
}

case "${1:-start}" in
  start)
    if [ "$EAGER" = 1 ]; then graphs=(--enforce-eager); else graphs=(--cudagraph-capture-sizes $CG); fi
    setsid conda run --no-capture-output -n vimBot vllm serve "$BASE" \
      --served-model-name qwen3.5-27b-base --quantization bitsandbytes --dtype bfloat16 --max-model-len 2048 \
      --gpu-memory-utilization "$U27" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" \
      --kv-cache-memory-bytes "$KV" "${graphs[@]}" --port "$PORT" --host 127.0.0.1 \
      --generation-config vllm --seed 0 \
      --enable-lora --max-lora-rank 16 --max-loras 1 \
      --lora-modules served27b="$ADAPTER" \
      >> "$OUT/vllm_27b.log" 2>&1 &
    echo $! > "$OUT/vllm_27b.pid"
    up 900 || { echo "the 27B endpoint did not come up (see $OUT/vllm_27b.log)"; exit 1; }
    echo "27B up: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ;;
  stop)
    if [ -f "$OUT/vllm_27b.pid" ]; then
      kill -- -"$(cat "$OUT/vllm_27b.pid")" 2>/dev/null || kill "$(cat "$OUT/vllm_27b.pid")" 2>/dev/null
      sleep 10
      kill -9 -- -"$(cat "$OUT/vllm_27b.pid")" 2>/dev/null
      rm -f "$OUT/vllm_27b.pid"
    fi
    sleep 5; echo "stopped: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ;;
esac
