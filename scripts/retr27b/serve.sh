#!/usr/bin/env bash
# The untrained 27B on one RTX 3090, for prompts that carry 8 examples. scripts/serve27b/serve27b.sh with three
# changes (runs/retr27b_2026-11-06/preregistration.md):
#
#   no adapter           nothing here needs the trained LoRA, and dropping it frees at least 445 MiB
#   --max-model-len 2400 a long prompt, the ask clause, the question, the reply and the 192-token cap overflow 2,048
#   KV 1.25 GiB          8,500 tokens against 6,826: prefix caching is off here, so 8 samples of an 830-token prompt
#                        hold about 6,800. The trained endpoint reached 23,731 MiB, at the paging line, which is why
#                        this stops short of the 30,000 tokens --enforce-eager gave R0
#
# ADAPTER=<path> serves that adapter as `served27b`, the way serve27b.sh does, for the trained 27B's run. The cache then
# goes back to serve27b.sh's 1 GiB, because the adapter is what paid for the extra 256 MiB.
#
#   bash scripts/retr27b/serve.sh start     # up, waits until it answers
#   bash scripts/retr27b/serve.sh stop
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-runs/retr27b_2026-11-06}
PORT=${PORT:-8015}
U27=${U27:-0.92}
ADAPTER=${ADAPTER:-}
if [ -n "$ADAPTER" ]; then KV=${KV:-1073741824}; else KV=${KV:-1342177280}; fi
SEQS=${SEQS:-16}
MNBT=${MNBT:-2048}
CG=${CG:-"1 2 4 8"}
MAXLEN=${MAXLEN:-2400}
BASE=${BASE:-runs/serve/qwen3.5-27b-text}
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
    lora=()
    if [ -n "$ADAPTER" ]; then lora=(--enable-lora --max-lora-rank 16 --max-loras 1 --lora-modules served27b="$ADAPTER"); fi
    setsid conda run --no-capture-output -n vimBot vllm serve "$BASE" \
      --served-model-name qwen3.5-27b-base --quantization bitsandbytes --dtype bfloat16 --max-model-len "$MAXLEN" \
      --gpu-memory-utilization "$U27" --max-num-seqs "$SEQS" --max-num-batched-tokens "$MNBT" \
      --kv-cache-memory-bytes "$KV" --cudagraph-capture-sizes $CG --port "$PORT" --host 127.0.0.1 \
      --generation-config vllm --seed 0 "${lora[@]}" \
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
