#!/usr/bin/env bash
# The top five methods on the 9B, end to end (runs/qwen9b_methods_2026-10-19/preregistration.md).
#
#   bash scripts/methods_9b/run_9b.sh                 # smoke first (4 rows, scratch), then the full chain
#   SKIP_SMOKE=1 bash scripts/methods_9b/run_9b.sh    # the full chain only (after a passing smoke)
#
# Stages, in order; each is skipped when its output exists, and the long ones resume by row:
#   1. 8-sample probes of the 9B on test and val                  (~30 min)
#   2. the 9B's dialogues on test and val (clarify harness)       (~3.5 h; the simulator is the slow part)
#   3. the sample rule's views on the 9B's probes                 (CPU)
#   4. the 9B's examples (+ its greedy answer), scored by V4      (~10 min)
#   5. V9: a verifier trained on the 9B's own train samples       (~1.5 h)
#   6. V9 scores of the 9B's and the 4B's candidates              (~10 min)
#   7. rerank_eval (V4 at λ = 0; V9 λ tuned on val, fixed on test) and compare_9b.py -> report.md
# Pause: touch $RUN/STOP (the clarify stages stop at the next case; relaunch to continue).
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/qwen9b_methods_2026-10-19}
POLICY=${POLICY:-runs/qwen3.5-9b-grpo-gated/best}
SERVED4B=runs/qwen3.5-4b-grpo-gated/best
V4=runs/reranker_2026-10-16/verifier-4b/best
PROBE9_TRAIN=runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-9b.jsonl
LIMIT=${LIMIT:-}
lim=(); [ -n "$LIMIT" ] && lim=(--limit "$LIMIT")
mkdir -p "$RUN"
log() { echo "$(date '+%m-%d %H:%M:%S') 9b: $*" | tee -a "$RUN/status.txt"; }
py() { conda run --no-capture-output -n vimBot python "$@"; }
quiet() { grep -v "Loading weights\|Fetching\|Warning\|warn" || true; }
die() { log "FAILED: $*"; exit 1; }

if [ -z "${LIMIT}" ] && [ -z "${SKIP_SMOKE:-}" ]; then
  SMOKE=$RUN/smoke
  log "smoke: every stage on 4 test rows in $SMOKE"
  LIMIT=4 RUN=$SMOKE SPLITS="test" V9_SMOKE=1 bash "$0" || die "smoke failed; nothing else ran"
  log "smoke passed"
fi
SPLITS=${SPLITS:-"test val"}
cases_of() { echo "data/splits/$1.jsonl"; }

# 1. probes
for s in $SPLITS; do
  out=$RUN/probe_$s.jsonl
  log "probe $s"
  py scripts/probe_difficulty.py --model qwen3.5-9b --adapter "$POLICY" --cases "$(cases_of $s)" \
     --out "$out" --num-generations 8 --prompts-per-call 4 --temperature 1.0 --seed 42 "${lim[@]}" \
     2>&1 | quiet >> "$RUN/probe_$s.log" || die "probe $s"
done

# 2. dialogues (the clarify harness; its own status.txt under $RUN/clarify)
sets=""; for s in $SPLITS; do sets="$sets $(cases_of $s)"; done
log "dialogues:$sets"
MODEL=qwen3.5-9b RUN="$RUN/clarify" SETS="$sets" CHECKPOINTS="served9b=$POLICY" LIMIT="$LIMIT" \
  bash scripts/clarify/run.sh || die "dialogues (see $RUN/clarify/status.txt)"

# 3. the sample rule's views
for s in $SPLITS; do
  py scripts/readings/sample_rule.py --probe "$RUN/probe_$s.jsonl" --cases "$(cases_of $s)" \
     --ref "$RUN/clarify/$s/verdicts_served9b.jsonl" --name "$s (9B)" --out "$RUN/sample_rule_$s/report.md" \
     2>&1 | quiet >> "$RUN/sample_rule.log" || die "sample rule $s"
done

# 4. examples and V4 scores
for s in $SPLITS; do
  [ -f "$RUN/examples_$s.jsonl" ] || py scripts/reranker/build_examples.py --cases "$(cases_of $s)" \
      --probe "$RUN/probe_$s.jsonl" --greedy "$RUN/clarify/$s/verdicts_served9b.jsonl" \
      --out "$RUN/examples_$s.jsonl" 2>&1 | quiet >> "$RUN/build.log" || die "examples $s"
  [ -f "$RUN/v4_scores_$s.jsonl" ] || py scripts/reranker/score.py --model qwen3.5-4b --init-adapter "$SERVED4B" \
      --verifier "$V4" --examples "$RUN/examples_$s.jsonl" --cases "$(cases_of $s)" \
      --out "$RUN/v4_scores_$s.jsonl" 2>&1 | quiet >> "$RUN/score.log" || die "V4 scores $s"
done

# 5. V9 (needs the 9B's val examples for its checkpoint selection)
V9DIR=$RUN/verifier-v9
if [ ! -f "$V9DIR/best/adapter_config.json" ]; then
  # the 2026-09-14 probe predates later curation: keep only ids still in train (3 dropped rows go)
  [ -f "$RUN/train_probe_9b.jsonl" ] || py -c "
import json, sys
ids = {json.loads(l)['id'] for l in open('data/splits/train.jsonl')}
rows = [l for l in open('$PROBE9_TRAIN') if json.loads(l)['id'] in ids]
open('$RUN/train_probe_9b.jsonl', 'w').writelines(rows)
print('train probe: kept %d rows still in train' % len(rows))" >> "$RUN/build.log" 2>&1 || die "filter the 9B train probe"
  [ -f "$RUN/train_examples_9b.jsonl" ] || py scripts/reranker/build_examples.py --cases data/splits/train.jsonl \
      --train --probe "$RUN/train_probe_9b.jsonl" --keep-unanimous 0.25 --max-per-prompt 8 \
      --out "$RUN/train_examples_9b.jsonl" 2>&1 | quiet >> "$RUN/build.log" || die "V9 train examples"
  VAL_EX=$RUN/examples_val.jsonl; VAL_CASES=data/splits/val.jsonl
  [ -f "$VAL_EX" ] || { VAL_EX=$RUN/examples_test.jsonl; VAL_CASES=data/splits/test.jsonl; }   # smoke: test only
  extra=(--wandb); [ -n "${V9_SMOKE:-}" ] && extra=(--max-steps 4 --eval-every 2 --eval-limit 30 --no-eval-at-start)
  log "V9 training"
  py scripts/reranker/train_verifier.py --model qwen3.5-4b --init-adapter "$SERVED4B" \
     --train "$RUN/train_examples_9b.jsonl" --val "$VAL_EX" --val-cases "$VAL_CASES" --out-dir "$V9DIR" "${extra[@]}" \
     2>&1 | quiet >> "$V9DIR.log" || die "V9 training (see $V9DIR.log)"
fi

# 6. V9 scores: the 9B's candidates and the 4B's (for the pooled pick)
for s in $SPLITS; do
  four=runs/reranker_2026-10-16/${s}_examples.jsonl
  [ -f "$RUN/v9_scores_$s.jsonl" ] || py scripts/reranker/score.py --model qwen3.5-4b --init-adapter "$SERVED4B" \
      --verifier "$V9DIR/best" --examples "$RUN/examples_$s.jsonl" --cases "$(cases_of $s)" \
      --out "$RUN/v9_scores_$s.jsonl" 2>&1 | quiet >> "$RUN/score.log" || die "V9 scores $s"
  [ -f "$RUN/v9_scores_4b_$s.jsonl" ] || py scripts/reranker/score.py --model qwen3.5-4b --init-adapter "$SERVED4B" \
      --verifier "$V9DIR/best" --examples "$four" --cases "$(cases_of $s)" \
      --out "$RUN/v9_scores_4b_$s.jsonl" 2>&1 | quiet >> "$RUN/score.log" || die "V9 scores of the 4B $s"
done

# 7. reports
if [ -z "$LIMIT" ]; then
  py scripts/reranker/rerank_eval.py --examples "$RUN/examples_val.jsonl" --split val \
     --scores v4="$RUN/v4_scores_val.jsonl" --scores v9="$RUN/v9_scores_val.jsonl" --lam v4=0 \
     --ask-report "$RUN/sample_rule_val/report.json" --verdicts "$RUN/clarify/val/verdicts_served9b.jsonl" \
     --out "$RUN/rerank_val.md" --json "$RUN/rerank_val.json" 2>&1 | quiet >> "$RUN/eval.log" || die "rerank val"
  LAM9=$(py -c "import json; print(json.load(open('$RUN/rerank_val.json'))['v9']['lam'])")
  py scripts/reranker/rerank_eval.py --examples "$RUN/examples_test.jsonl" --split test \
     --scores v4="$RUN/v4_scores_test.jsonl" --scores v9="$RUN/v9_scores_test.jsonl" --lam v4=0 --lam v9="$LAM9" \
     --ask-report "$RUN/sample_rule_test/report.json" --verdicts "$RUN/clarify/test/verdicts_served9b.jsonl" \
     --out "$RUN/rerank_test.md" --json "$RUN/rerank_test.json" 2>&1 | quiet >> "$RUN/eval.log" || die "rerank test"
  py scripts/methods_9b/compare_9b.py --run "$RUN" --lam-v9 "$LAM9" 2>&1 | quiet | tee -a "$RUN/eval.log" | tail -30
else
  py scripts/methods_9b/compare_9b.py --run "$RUN" --lam-v9 0 --smoke 2>&1 | quiet | tail -12 || die "compare (smoke)"
fi
log "done"
