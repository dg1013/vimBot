#!/bin/bash
# Start or RESUME SFT round 2 on the cliff for one model, detached. Idempotent:
# run it again after a pause or any interruption and the round continues from
# the files on disk (probes by id; a paused SFT stage restarts).
#   bash scripts/cliff_sft/launch.sh                       # 4B from runs/qwen3.5-4b-grpo-gated/best
#   MODEL=qwen3.5-9b bash scripts/cliff_sft/launch.sh      # the 9B, once its gated round is done
#   ADAPTER=runs/qwen3.5-4b-grpo/best bash scripts/cliff_sft/launch.sh   # continue another adapter
#   SFT_ARGS="--epochs 1 --lr 3e-5 --eval-every 20" BUILD_ARGS="--hold-flagged" bash scripts/cliff_sft/launch.sh
REPO=/home/dgeorge28/dev/chatbot/vimBot
DIR=${DIR:-$REPO/runs/cliff_$(date +%Y-%m-%d)}
export DIR MODEL="${MODEL:-qwen3.5-4b}" ADAPTER OUT SFT_ARGS BUILD_ARGS PROBE_ARGS TEST_ARGS STALL_MIN DIFFICULTY COMPARE_DIR
cd "$REPO" || exit 1
mkdir -p "$DIR"
if [ -f "$DIR/driver.pid" ] && kill -0 "$(cat "$DIR/driver.pid")" 2>/dev/null; then
  echo "cliff round already running (pid $(cat "$DIR/driver.pid")); nothing to do"; exit 0
fi
for other in "$REPO"/runs/grpo*_*/driver.pid; do          # the GRPO chains (one GPU)
  [ -f "$other" ] && kill -0 "$(cat "$other")" 2>/dev/null && { echo "another GPU chain is running (pid $(cat "$other"), $other); pause it first"; exit 1; }
done
rm -f "$DIR/STOP_CHAIN"               # a relaunch is a request to run
rm -f "$DIR/driver.pid"
setsid nohup bash "$REPO/scripts/cliff_sft/run_cliff.sh" >> "$DIR/driver.log" 2>&1 < /dev/null &
sleep 2
echo "launched: cliff round pid $(cat "$DIR/driver.pid") for $MODEL"
echo "log of record: $DIR/status.txt"
echo "stage logs:    $DIR/probe_before_$MODEL.log, $DIR/build_$MODEL.log, $REPO/runs/$MODEL-sft-cliff/train.log, $DIR/probe_after_$MODEL.log, .../test_eval.log"
