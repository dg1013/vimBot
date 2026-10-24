#!/usr/bin/env bash
# Serve the pooled pick on one RTX 3090: two vLLM endpoints side by side.
#   :8011  the served 4B (merged bf16, runs/serve/qwen3.5-4b-served) + the 4B verifier as a LoRA
#   :8012  the 9B base (runs/serve/qwen3.5-9b-text) quantized to 4-bit at load (bitsandbytes)
#          + the 9B's GRPO adapter as a LoRA (NF4 base + LoRA, the way it was trained)
#
#   bash scripts/serve_pooled/serve_pooled.sh start     # both up, waits until they answer
#   bash scripts/serve_pooled/serve_pooled.sh stop
#
# Memory: both instances must fit beside the Windows desktop's share (~3.5-4 GB on this machine);
# past ~24 GB, WSL2 pages GPU memory and everything slows ~10x. U4 / U9 are each instance's share of
# the card. --logprobs-mode processed_logits makes a scoring request return the logits after the
# yes/no restriction, i.e. exactly logit(yes) and logit(no) (the verifier's score is their difference).
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/serve_pooled}
U4=${U4:-0.37}
U9=${U9:-0.37}
SEQS=${SEQS:-16}
MNBT=${MNBT:-2048}          # max tokens per scheduler step: bounds the activation peak vLLM reserves
ONLY=${ONLY:-}              # "4b" or "9b": start one endpoint (to size it)
KV4=${KV4:-1073741824}       # explicit cache bytes per endpoint: vLLM's own estimate counts the other
KV9=${KV9:-1073741824}       # endpoint's memory under WSL2 (the 9B computed -10 GiB beside the 4B)
CG=${CG:-"1 2 4 8 16"}      # CUDA graph batch sizes (the 9B's defaults took 1.2 GiB)
mkdir -p "$OUT"
export VLLM_USE_FLASHINFER_SAMPLER=0

up() {   # port seconds
  local t=0
  until curl -s "http://127.0.0.1:$1/v1/models" > /dev/null 2>&1; do
    sleep 5; t=$((t + 5))
    if [ "$t" -ge "$2" ]; then return 1; fi
  done
  return 0
}

case "${1:-start}" in
  start)
    if [ "$ONLY" != "9b" ]; then
    setsid conda run --no-capture-output -n vimBot vllm serve runs/serve/qwen3.5-4b-served \
      --served-model-name qwen3.5-4b-served --dtype bfloat16 --max-model-len 2048 \
      --gpu-memory-utilization "$U4" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" --kv-cache-memory-bytes "$KV4" --cudagraph-capture-sizes $CG --port 8011 --host 127.0.0.1 \
      --generation-config vllm --seed 0 --logprobs-mode processed_logits \
      --enable-lora --max-lora-rank 16 --max-loras 1 \
      --lora-modules verifier=runs/reranker_2026-10-16/verifier-4b/best \
      > "$OUT/vllm_4b.log" 2>&1 &
    echo $! > "$OUT/vllm_4b.pid"
    up 8011 600 || { echo "the 4B endpoint did not come up (see $OUT/vllm_4b.log)"; exit 1; }
    echo "4B up: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    fi
    [ "$ONLY" = "4b" ] && exit 0
    setsid conda run --no-capture-output -n vimBot vllm serve runs/serve/qwen3.5-9b-text \
      --served-model-name qwen3.5-9b-base --quantization bitsandbytes --dtype bfloat16 --max-model-len 2048 \
      --gpu-memory-utilization "$U9" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" --kv-cache-memory-bytes "$KV9" --cudagraph-capture-sizes $CG --port 8012 --host 127.0.0.1 \
      --generation-config vllm --seed 0 \
      --enable-lora --max-lora-rank 16 --max-loras 1 \
      --lora-modules served9b=runs/qwen3.5-9b-grpo-gated/best \
      > "$OUT/vllm_9b.log" 2>&1 &
    echo $! > "$OUT/vllm_9b.pid"
    up 8012 900 || { echo "the 9B endpoint did not come up (see $OUT/vllm_9b.log)"; exit 1; }
    echo "both up: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ;;
  stop)
    for m in 4b 9b; do
      if [ -f "$OUT/vllm_$m.pid" ]; then
        kill -- -"$(cat "$OUT/vllm_$m.pid")" 2>/dev/null || kill "$(cat "$OUT/vllm_$m.pid")" 2>/dev/null
        rm -f "$OUT/vllm_$m.pid"
      fi
    done
    sleep 5; echo "stopped: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    ;;
esac
