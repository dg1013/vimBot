#!/usr/bin/env bash
# The prompted Qwen3.5-27B on val (scripts/teacher27b/README.md; the pre-registration is
# runs/teacher27b_2026-10-30/preregistration.md).
#
# Stages, in this order, each only if named in STEPS:
#   preflight  CPU: the weights cached at the pinned revision, the tokenizer equal to the 4B's,
#              the plans against the splits, the comparators present; copies the plans in.
#              Any problem stops the driver.
#   smoke      the k = 8 arm on 5 val rows into $RUN/smoke/ (one 27B load), then the median
#              seconds per row and a projection; slower than SMOKE_MAX_S per row stops the driver
#              (an over-full card under WSL2 pages instead of failing). The median, because the
#              first row also compiles kernels. With `samples` in STEPS, also 2 sampled rows.
#   greedy     27b-retr8 (k = 8, the decision arm), then 27b-k0 (zero-shot): 545 rows each
#   samples    opt-in: 8 samples per row at T = 1 on the k = 8 prompts, as probe rows;
#              resumable, and `touch $RUN/STOP` pauses it after the row in flight (exit 3)
#   report     CPU: $RUN/report.md and report.json
# The card is shared and the 27B needs about 19 of its 24 GB, so every GPU stage starts only on
# an idle card: no process listed, at most FREE_MIB in use and at most FREE_UTIL% utilization,
# for 2 minutes. Each GPU stage is its own process and frees the card when it exits.
#
#   bash scripts/teacher27b/run.sh                          # preflight smoke greedy report
#   STEPS="samples report" bash scripts/teacher27b/run.sh   # the opt-in sampled arm, afterwards
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=runs/teacher27b_2026-10-30
STEPS=${STEPS:-"preflight smoke greedy report"}
FREE_MIB=${FREE_MIB:-4500}      # about the Windows desktop's share; the 27B needs the rest
FREE_UTIL=${FREE_UTIL:-60}      # the desktop alone holds the card at 18-40% (2026-10-21)
MAX_WAIT=${MAX_WAIT:-28800}     # give up waiting for an idle card after 8 h
SMOKE_MAX_S=${SMOKE_MAX_S:-30}  # smoke seconds per k = 8 row above which the driver stops
SAMPLES_PER_CALL=${SAMPLES_PER_CALL:-8}
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
py() { conda run --no-capture-output -n vimBot python "$@"; }
has() { case " $STEPS " in *" $1 "*) return 0 ;; esac; return 1; }
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }

wait_idle() {   # 12 checks 10 s apart (run_at_k.sh's rule, its "empty" mode)
  local ok=0 waited=0 u m p
  while [ "$ok" -lt 12 ]; do
    IFS=, read -r u m <<< "$(card)"
    p=$(procs)
    if [ "$u" -le "$FREE_UTIL" ] && [ "$p" -eq 0 ] && [ "$m" -le "$FREE_MIB" ]; then
      ok=$((ok + 1))
    else
      ok=0
    fi
    sleep 10
    waited=$((waited + 10))
    if [ $((waited % 600)) -eq 0 ]; then echo "$(date +%T) waiting for an idle card: ${u}%, ${m} MiB, $p processes"; fi
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "$(date +%T) no idle card in $((MAX_WAIT / 3600)) h; giving up"; return 1; fi
  done
  echo "$(date +%T) card idle: ${u}%, ${m} MiB"
}

status "start: STEPS=\"$STEPS\" FREE_MIB=$FREE_MIB FREE_UTIL=$FREE_UTIL SAMPLES_PER_CALL=$SAMPLES_PER_CALL"

if has preflight; then
  status "preflight"
  py scripts/teacher27b/teacher27b.py preflight > "$RUN/preflight.log" 2>&1 \
    || { tail -3 "$RUN/preflight.log"; status "preflight FAILED: see $RUN/preflight.json"; exit 1; }
  status "preflight OK"
fi

if has smoke; then
  status "smoke: waiting for an idle card"
  wait_idle || exit 1
  rm -rf "$RUN/smoke"
  py scripts/teacher27b/teacher27b.py greedy --run "$RUN/smoke" --limit 5 --arms 27b-retr8:plan_retr8
  rc=$?
  [ $rc -eq 0 ] || { status "smoke FAILED (rc $rc)"; exit $rc; }
  s=$(python3 -c "import json,statistics; r=[json.loads(l) for l in open('$RUN/smoke/27b-retr8/completions_val.jsonl')]; print(round(statistics.median(x['latency_s'] for x in r), 1))")
  status "smoke OK: $s s per k = 8 row, about $(python3 -c "print(round($s * 545 / 60))") min for that arm's 545 rows"
  python3 -c "import sys; sys.exit(0 if $s <= $SMOKE_MAX_S else 1)" \
    || { status "smoke too slow ($s s per row, limit $SMOKE_MAX_S): the card is probably paging; stopping"; exit 5; }
  if has samples; then
    py scripts/teacher27b/teacher27b.py samples --out "$RUN/smoke/probe_val_27b-retr8.jsonl" --limit 2 \
      --samples-per-call "$SAMPLES_PER_CALL"
    rc=$?
    [ $rc -eq 0 ] || { status "samples smoke FAILED (rc $rc)"; exit $rc; }
    status "samples smoke OK"
  fi
fi

if has greedy; then
  status "greedy: waiting for an idle card"
  wait_idle || exit 1
  status "greedy: 27b-retr8, then 27b-k0"
  py scripts/teacher27b/teacher27b.py greedy
  rc=$?
  [ $rc -eq 0 ] || { status "greedy stopped (rc $rc); finished arms are kept, the same command resumes"; exit $rc; }
  status "greedy done"
fi

if has samples; then
  status "samples: waiting for an idle card"
  wait_idle || exit 1
  py scripts/teacher27b/teacher27b.py samples --samples-per-call "$SAMPLES_PER_CALL" --stop-file "$RUN/STOP"
  rc=$?
  if [ $rc -eq 3 ]; then status "samples paused: the same command resumes"; exit 3; fi
  [ $rc -eq 0 ] || { status "samples stopped (rc $rc); rows on disk are kept, the same command resumes"; exit $rc; }
  status "samples done"
fi

if has report; then
  status "report"
  py scripts/teacher27b/report.py > "$RUN/report.log" 2>&1 \
    || { tail -3 "$RUN/report.log"; status "report FAILED: see $RUN/report.log"; exit 1; }
  status "report: $RUN/report.md"
fi
status "done: $STEPS"
