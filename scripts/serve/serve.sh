#!/usr/bin/env bash
# Serve the merged model on vLLM's OpenAI-compatible endpoint (§10).
#
#   bash scripts/serve/serve.sh        # runs/serve/qwen3.5-4b-served on :8011
#
# The weights are scripts/serve/merge_adapter.py's output. The registry
# supplies max_seq_len and any vllm_extra_args. `--generation-config vllm`
# keeps the model's saved generation config from changing request defaults:
# the round trip sets every sampling parameter itself. GPU_UTIL is the share
# of the card vLLM may take (the desktop holds ~2 GB on this machine).
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=${MODEL:-qwen3.5-4b}
WEIGHTS=${WEIGHTS:-runs/serve/$MODEL-served}
SERVED_NAME=${SERVED_NAME:-$MODEL-served}   # not NAME: this machine exports NAME (the hostname)
PORT=${PORT:-8011}
GPU_UTIL=${GPU_UTIL:-0.55}
[ -f "$WEIGHTS/merge.json" ] || { echo "no merged weights at $WEIGHTS (run merge_adapter.py)" >&2; exit 2; }

# conda run does not forward stdin, so the registry is read with -c
reg='import sys, yaml; m = yaml.safe_load(open("configs/models.yaml"))["models"][sys.argv[1]]; print(m.get("max_seq_len", 2048), " ".join(m.get("vllm_extra_args") or []))'
read -r MAXLEN EXTRA <<< "$(conda run -n vimBot python -c "$reg" "$MODEL")"

# FlashInfer's top-k/top-p sampling kernel is compiled at startup and needs
# nvcc, which this machine does not have (vLLM refused to start on
# 2026-10-13). PyTorch's own sampler draws from the same distribution, and
# the round trip samples at T = 1 with no top-k or top-p truncation.
export VLLM_USE_FLASHINFER_SAMPLER=0

# shellcheck disable=SC2086  # EXTRA is a flag list
exec conda run --no-capture-output -n vimBot vllm serve "$WEIGHTS" \
  --served-model-name "$SERVED_NAME" --dtype bfloat16 --max-model-len "$MAXLEN" \
  --gpu-memory-utilization "$GPU_UTIL" --port "$PORT" --host 127.0.0.1 \
  --generation-config vllm --seed 0 $EXTRA
