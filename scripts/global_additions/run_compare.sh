#!/bin/bash
# Follow-up #2's full run: SFT on train + the proposed global prompts against the
# 100% scaling point. Stages, each idempotent:
#   0  combine   data/splits/train.jsonl + data/proposed/global_2026-10-16/cases.jsonl
#                -> $DIR/train_plus_global.jsonl (the canonical splits are not touched)
#   1  sft       scripts/train_sft.py, the ds100 recipe -> runs/$MODEL-sft-plusglobal/
#   2  test rows baseline.py on the ds100 adapter and on the new one (test v2)
#   3  report    scripts/global_additions/compare.py -> $DIR/report.md
# Pause: touch $DIR/STOP_CHAIN (stops at the next boundary). Run again to resume.
REPO=/home/dgeorge28/dev/chatbot/vimBot
PY=/home/dgeorge28/miniconda3/envs/vimBot/bin/python
MODEL=${MODEL:-qwen3.5-4b}
DIR=${DIR:-$REPO/runs/global_additions_2026-10-16}
PROPOSED=$REPO/data/proposed/global_2026-10-16/cases.jsonl
BASE=$REPO/runs/$MODEL-sft-ds100
NEW=$REPO/runs/$MODEL-sft-plusglobal
STALL_MIN=${STALL_MIN:-30}
PAUSED_RC=3
STATUS=$DIR/status.txt
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$DIR"
echo $$ > "$DIR/driver.pid"
log() { echo "$(date '+%m-%d %H:%M:%S') [plusglobal $MODEL] $*" >> "$STATUS"; }
chain_paused() { [ -f "$DIR/STOP_CHAIN" ]; }
rows_in() { if [ -f "$1" ]; then wc -l < "$1"; else echo 0; fi; }

watched() {
  local log=$1; shift
  "$@" >> "$log" 2>&1 &
  local pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep 60
    if [ -f "$log" ]; then
      local age=$(( $(date +%s) - $(stat -c %Y "$log") ))
      if [ "$age" -gt $((STALL_MIN * 60)) ]; then
        log "WATCHDOG: no log activity for $((age / 60)) min, killing pid $pid"
        kill "$pid" 2>/dev/null; sleep 90; kill -9 "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null; return 124
      fi
    fi
  done
  wait "$pid"; return $?
}

if chain_paused; then log "pause requested before the run started"; exit $PAUSED_RC; fi
log "driver started (pid $$)"

# Stage 0: the combined train file.
if [ -s "$DIR/train_plus_global.jsonl" ]; then
  log "combine: exists ($(rows_in "$DIR/train_plus_global.jsonl") rows)"
else
  $PY - "$PROPOSED" "$DIR/train_plus_global.jsonl" >> "$DIR/build.log" 2>&1 <<'EOF' || { log "combine failed -- read $DIR/build.log"; exit 1; }
import json, sys
proposed, out = sys.argv[1], sys.argv[2]
train = [json.loads(l) for l in open("data/splits/train.jsonl", encoding="utf-8") if l.strip()]
add = [json.loads(l) for l in open(proposed, encoding="utf-8") if l.strip()]
add = [r for r in add if r.get("review_verdict") != "drop"]
ids = {r["id"] for r in train}
clash = [r["id"] for r in add if r["id"] in ids]
assert not clash, clash[:3]
with open(out, "w", encoding="utf-8") as fh:
    for r in train + add:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
print("train %d + proposed %d = %d" % (len(train), len(add), len(train) + len(add)))
EOF
  log "combine: $(rows_in "$DIR/train_plus_global.jsonl") rows"
fi

# Stage 1: SFT, the ds100 recipe on the combined file.
if [ -f "$NEW/sft_val.json" ]; then
  log "sft: done already"
else
  if [ -d "$NEW" ] && [ -n "$(ls -A "$NEW" 2>/dev/null)" ]; then
    aside="$NEW.partial-$(date +%Y%m%d-%H%M%S)"; mv "$NEW" "$aside"; log "sft: partial run set aside as $(basename "$aside")"
  fi
  mkdir -p "$NEW"
  rows=$(rows_in "$DIR/train_plus_global.jsonl"); every=$(( rows / 32 ))
  log "sft: $rows rows, eval every $every steps (1 epoch) -> $NEW"
  watched "$NEW/train.log" $PY scripts/train_sft.py --model $MODEL --train "$DIR/train_plus_global.jsonl" \
      --out-dir "$NEW" --eval-every $every --no-eval-at-start --wandb
  log "sft rc=$?"
  [ -f "$NEW/sft_val.json" ] || { log "sft: no sft_val.json -- read $NEW/train.log"; exit 1; }
fi
if chain_paused; then log "pause requested before the test rows"; exit $PAUSED_RC; fi

# Stage 2: test v2 rows for both adapters.
for spec in "$BASE sft-ds100" "$NEW sft-plusglobal"; do
  set -- $spec
  if [ -f "$1/exec_test.jsonl" ]; then log "test row $2: exists"; continue; fi
  log "test row $2: baseline.py --adapter $1/best"
  watched "$1/test_eval.log" $PY scripts/baseline.py --model $MODEL --adapter "$1/best" --condition "$2" \
      --out-dir "$1" --wandb
  [ -f "$1/exec_test.jsonl" ] || { log "test row $2 failed -- read $1/test_eval.log"; exit 1; }
  if chain_paused; then log "pause requested between test rows"; exit $PAUSED_RC; fi
done

# Stage 3: the report.
$PY scripts/global_additions/compare.py --base "$BASE" --new "$NEW" --out "$DIR/report.md" --json "$DIR/report.json" \
    >> "$DIR/build.log" 2>&1 && log "report: $DIR/report.md" || { log "report failed -- read $DIR/build.log"; exit 1; }
log "driver done"
