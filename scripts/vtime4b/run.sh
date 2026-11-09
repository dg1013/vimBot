#!/usr/bin/env bash
# V timed as a system (runs/vtime4b_2026-11-09/plan.md), in one detached driver: wait for an idle card, serve the
# deployed 4B with its verifier (scripts/sys4b/serve4b.sh, unchanged), a 4-row smoke then the 200 timing rows for each
# arm, stop the endpoint, write the report.
#
#   setsid nohup bash scripts/vtime4b/run.sh > /dev/null 2>&1 < /dev/null &    # status: runs/vtime4b_2026-11-09/status.txt
#   touch runs/vtime4b_2026-11-09/PAUSE      # stops after the row in flight; the endpoint is stopped either way
#
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${RUN:-runs/vtime4b_2026-11-09}
ARMS=${ARMS:-"a b"}
PY=$HOME/miniconda3/envs/vimBot/bin/python
FREE_MIB=${FREE_MIB:-4500}
FREE_UTIL=${FREE_UTIL:-60}
MAX_WAIT=${MAX_WAIT:-7200}
STAGE_TIMEOUT=${STAGE_TIMEOUT:-3600}
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
mkdir -p "$RUN"
exec > >(tee -a "$RUN/driver.log") 2>&1
status() { echo "$(date -Is) $*" | tee -a "$RUN/status.txt"; }
card() { nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
procs() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -cE '^ *[0-9]+'; }

wait_idle() {   # 12 checks 10 s apart, as the other drivers
  local ok=0 waited=0 u m p
  while [ "$ok" -lt 12 ]; do
    IFS=, read -r u m <<< "$(card)"
    p=$(procs)
    if [ "$u" -le "$FREE_UTIL" ] && [ "$p" -eq 0 ] && [ "$m" -le "$FREE_MIB" ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 10
    waited=$((waited + 10))
    if [ "$waited" -ge "$MAX_WAIT" ]; then return 1; fi
  done
  echo "$(date +%T) card idle: ${u}%, ${m} MiB"
}

serving=no
stop_server() {
  if [ "$serving" = yes ]; then
    OUT=$RUN bash scripts/sys4b/serve4b.sh stop served >> "$RUN/serve.log" 2>&1
    serving=no
    status "endpoint stopped: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
  fi
}
trap stop_server EXIT

status "start: ARMS=\"$ARMS\""
wait_idle || { status "no idle card in $((MAX_WAIT / 60)) min; nothing ran"; exit 1; }
serving=yes
OUT=$RUN bash scripts/sys4b/serve4b.sh start served >> "$RUN/serve.log" 2>&1 \
  || { status "the endpoint did not come up (see $RUN/serve.log)"; exit 1; }
status "endpoint up: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

rc=0
for arm in $ARMS; do
  if [ -f "$RUN/PAUSE" ]; then status "paused before arm $arm"; exit 4; fi
  timeout "$STAGE_TIMEOUT" $PY scripts/vtime4b/vtime4b.py timing --arm "$arm" --n 4 --out "$RUN/smoke" >> "$RUN/smoke_$arm.log" 2>&1
  src=$?
  if [ "$src" -ne 0 ]; then status "arm $arm: smoke FAILED (rc $src): $(tail -1 "$RUN/smoke_$arm.log")"; rc=3; continue; fi
  status "arm $arm smoke: $(tail -1 "$RUN/smoke_$arm.log" | cut -c1-200)"
  timeout "$STAGE_TIMEOUT" $PY scripts/vtime4b/vtime4b.py timing --arm "$arm" >> "$RUN/$arm.log" 2>&1
  arc=$?
  if [ "$arc" -ne 0 ]; then status "arm $arm FAILED (rc $arc): $(tail -1 "$RUN/$arm.log")"; rc=3; continue; fi
  status "arm $arm: $(tail -1 "$RUN/$arm.log" | cut -c1-240)"
done
stop_server
if $PY scripts/vtime4b/vtime4b.py report > "$RUN/report.log" 2>&1; then status "report: $RUN/report.md"
else status "report FAILED: $(tail -1 "$RUN/report.log")"; rc=3; fi
status "done (rc $rc)"
exit "$rc"
