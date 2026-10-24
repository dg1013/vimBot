#!/bin/bash
# PAUSE the cliff round wherever it is (under a minute): leaves $DIR/STOP_CHAIN
# so the driver stands down at the next stage boundary, and asks the running
# stage to stop -- a probe saves its rows and resumes later; the SFT stage
# stops after its step and is restarted on relaunch (best/ so far is kept
# aside); a running test row is terminated and redone. Resume with launch.sh.
#   bash scripts/cliff_sft/pause.sh
#   DIR=runs/cliff_2026-09-15 bash scripts/cliff_sft/pause.sh   # a round from another day
REPO=/home/dgeorge28/dev/chatbot/vimBot
DIR=${DIR:-$REPO/runs/cliff_$(date +%Y-%m-%d)}
MODEL=$(cat "$DIR/current_model" 2>/dev/null)
OUT=$(cat "$DIR/current_out" 2>/dev/null)
[ -z "$OUT" ] && OUT=$REPO/runs/$MODEL-sft-cliff
touch "$DIR/STOP_CHAIN"
pid=$(cat "$DIR/driver.pid" 2>/dev/null)
if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
  echo "no cliff round is running under $DIR; STOP_CHAIN left in place so a launch is needed to start again"; exit 0
fi
touch "$DIR/STOP"                                   # the probes
[ -n "$MODEL" ] && mkdir -p "$OUT" && touch "$OUT/STOP"   # the SFT stage
echo "pause requested for the cliff round (pid $pid), model ${MODEL:-?}. Watch for it:"
echo "  tail -f $DIR/status.txt"
pkill -TERM -f "baseline.py --model $MODEL --adapter $OUT/best" 2>/dev/null && echo "  (a test row was running; terminated, it will be redone on relaunch)"
while kill -0 "$pid" 2>/dev/null; do sleep 5; done
echo "cliff round stopped."
[ -f "$OUT/PAUSED" ] && cat "$OUT/PAUSED"
[ -f "$DIR/PAUSED" ] && cat "$DIR/PAUSED"
tail -3 "$DIR/status.txt"
