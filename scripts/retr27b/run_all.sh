#!/usr/bin/env bash
# Both BM25 runs in one detached driver: the untrained 27B (Part 2), then the trained one (Part 3). Each keeps its own dry run,
# gates, status file and report (runs/retr27b_2026-11-06/preregistration.md, addendum 2). The second starts whatever became of the
# first, since neither reads the other's rows to run, unless the first was paused: a pause holds both.
#
#   setsid nohup bash scripts/retr27b/run_all.sh > /dev/null 2>&1 < /dev/null &
#   touch runs/retr27b_2026-11-06/PAUSE       # stops after the rows in flight, frees the card and holds both parts
#
# Never edit this file while it runs: bash reads it as it goes.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
RUN=${RUN:-runs/retr27b_2026-11-06}
SELF=${SELF:-scripts/retr27b/run.sh}
mkdir -p "$RUN/trained"
ln -sfn ../PAUSE "$RUN/trained/PAUSE"
say() { echo "$(date -Is) $*" | tee -a "$RUN/all.txt"; }

say "part 2: the base 27B"
ARM=base RUN=$RUN bash "$SELF" >> "$RUN/driver.log" 2>&1
rc=$?
say "part 2: finished, rc $rc"
if [ -f "$RUN/PAUSE" ]; then say "paused: part 3 holds"; exit 2; fi
say "part 3: the trained 27B"
ARM=trained RUN=$RUN/trained bash "$SELF" >> "$RUN/trained/driver.log" 2>&1
rc3=$?
say "part 3: finished, rc $rc3"
exit "$rc3"
