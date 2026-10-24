#!/usr/bin/env bash
# The 9B's capability ceiling on test v2 (owner, 2026-10-19: "check what the ceiling might be for
# 9B after everything is done"), measured the way the 4B's was (scripts/ceiling, 2026-09-22):
# 128 samples per row at T = 1, top-p 1, top-k off, seed 42; pass@k, maj@k, coverage and the
# uncovered core. The 4B served adapter's 2026-09-22 samples are reused for the test rows unchanged
# since, and the 31 rows reworded on 2026-09-25 are re-sampled, so both models are read on test v2.
#
# Waits for the methods chain (run_9b.sh) to exit, and starts only if it ended with "9b: done".
#   bash scripts/methods_9b/ceiling_9b.sh          # ~2 h of GPU after the chain
# Pause: touch $RUN/ceiling/STOP (the sampler finishes its prompt); relaunch to continue.
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/qwen9b_methods_2026-10-19}
C=$RUN/ceiling
mkdir -p "$C"
log() { echo "$(date '+%m-%d %H:%M:%S') ceiling: $*" | tee -a "$C/status.txt"; }
py() { conda run --no-capture-output -n vimBot python "$@"; }
quiet() { grep -v "Loading weights\|Fetching\|Warning\|warn" || true; }
die() { log "FAILED: $*"; exit 1; }

log "waiting for the methods chain to finish"
while pgrep -f "scripts/methods_9b/run_9b.sh" > /dev/null; do sleep 60; done
last=$(tail -1 "$RUN/status.txt")
case "$last" in
  *"9b: done"*) log "methods chain done; starting" ;;
  *) die "the methods chain did not end cleanly ($last); nothing started" ;;
esac

# 1. the ceilings already inside the methods run (CPU)
py scripts/methods_9b/free_ceilings.py --run "$RUN" --out "$C/free_ceilings.md" 2>&1 | quiet > "$C/free_ceilings.log" \
  || die "free ceilings"

# 2. the 4B on test v2: re-sample the rows changed since its 2026-09-22 ceiling run, keep the rest
V1=runs/ceiling_2026-09-22/samples_served.jsonl
py -c "
import json
v1 = {json.loads(l)['id']: json.loads(l) for l in open('runs/remeasure_2026-09-25/test_v1_445.jsonl')}
v2 = [json.loads(l) for l in open('data/splits/test.jsonl')]
keys = ('instruction', 'buffer_before', 'buffer_after', 'gold_command', 'mode', 'cursor')
changed = [c['id'] for c in v2 if c['id'] not in v1 or any(v1[c['id']].get(k) != c.get(k) for k in keys)]
open('$C/changed_ids.txt', 'w').write(''.join(i + '\n' for i in changed))
print('test v2 rows changed since 2026-09-22: %d' % len(changed))" | tee -a "$C/status.txt" || die "changed ids"
log "4B: re-sampling the changed rows"
py scripts/ceiling/sample.py --model qwen3.5-4b --adapter runs/qwen3.5-4b-grpo-gated/best --label served \
   --cases data/splits/test.jsonl --ids "$C/changed_ids.txt" --n 128 --chunk 64 --seed 42 \
   --out "$C/samples_served4b_changed.jsonl" --stop-file "$C/STOP" 2>&1 | quiet >> "$C/sample_4b.log" \
  || die "4B re-sampling"
py -c "
import json
changed = {l.strip() for l in open('$C/changed_ids.txt') if l.strip()}
v2 = [json.loads(l)['id'] for l in open('data/splits/test.jsonl')]
old = {json.loads(l)['id']: l for l in open('$V1')}
new = {json.loads(l)['id']: l for l in open('$C/samples_served4b_changed.jsonl')}
rows = [new[i] if i in changed else old[i] for i in v2]
open('$C/samples_served4b.jsonl', 'w').writelines(rows)
print('4B on test v2: %d rows (%d re-sampled)' % (len(rows), len(changed)))" | tee -a "$C/status.txt" || die "merge 4B samples"
cp "$V1.meta.json" "$C/samples_served4b.jsonl.meta.json"

# 3. the 9B: 128 samples per test v2 row
log "9B: 128 samples per row"
py scripts/ceiling/sample.py --model qwen3.5-9b --adapter runs/qwen3.5-9b-grpo-gated/best --label served9b \
   --cases data/splits/test.jsonl --n 128 --chunk 32 --seed 42 \
   --out "$C/samples_served9b.jsonl" --stop-file "$C/STOP" 2>&1 | quiet >> "$C/sample_9b.log" \
  || die "9B sampling"

# 4. the report: both curves, the paired difference, union coverage, the uncovered core
py scripts/ceiling/report.py --run "$C" --cases data/splits/test.jsonl \
   --samples served4b=samples_served4b.jsonl --samples served9b=samples_served9b.jsonl \
   --greedy served4b=runs/qwen3.5-4b-grpo-nearcliff/exec_test.jsonl \
   --greedy served9b=runs/remeasure_2026-09-25/v2/qwen3.5-9b-grpo-gated/exec_test.jsonl \
   --pair served9b:served4b 2>&1 | quiet | tee -a "$C/status.txt" || die "report"
log "done: $C/report.md, $C/free_ceilings.md"
