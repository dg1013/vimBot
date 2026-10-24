#!/usr/bin/env python3
"""How many samples does the sample rule need? Accuracy against k, offline.

The rule was confirmed with k = 8 samples (ask when the most common result
holds at most 5 of 8; DECISIONS.md 2026-10-10). Fewer samples cost less
time and fewer tokens (serve/timing.py measures that half). This measures
the other half: what k samples keep of the gain.

From a probe file holding 8 samples per row, every distinct command runs
once in the sandbox (the validation path), so each sample has a result.
Then, for each k, R random k-subsets of the 8 are drawn per row, the rule
is applied at the same agreement share (ask when the top result holds at
most floor(5/8 * k) of the k), and the run is scored end to end with the
stored dialogues: the generic question's answer where it asks, the §6
answer elsewhere. Averaging over the draws gives the expected exec-match and
ask rate.

    python scripts/serve/k_sweep.py --probe runs/sample_rule_2026-10-09/probe_val.jsonl \
        --cases data/splits/val.jsonl --ref runs/clarify_2026-10-06/val/verdicts_served.jsonl
"""
import argparse
import collections
import json
import math
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", REPO / "scripts" / "readings"):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                       # noqa: E402
import readings_protocol as R                         # noqa: E402

KS = (2, 4, 6, 8)
SHARE = 5 / 8


def sample_keys(case, probe_row, entries):
    """The 8 samples as result keys (None for one that failed, changed
    nothing, or did not parse), expanded from the probe's counts."""
    keys = []
    for c, e in zip(probe_row["commands"], entries):
        ok = c.get("command") is not None and R.reading_status(case, e) == "ok"
        keys += [R.norm_key(e["buffer_after"]) if ok else None] * c["n"]
    return keys


def asks(keys):
    """Ask when the most common result holds at most floor(SHARE * k) of them."""
    counts = collections.Counter(k for k in keys if k is not None)
    top = max(counts.values()) if counts else 0
    return top <= math.floor(SHARE * len(keys) + 1e-9)


def sweep(rows, draws, seed):
    """{k: (mean ask rate, mean end-to-end exec-match)} over `draws` subsets per row."""
    rng = random.Random(seed)
    out = {}
    for k in KS:
        ask_sum = match_sum = 0.0
        for r in rows:
            keys, a_n = r["keys"], 0
            reps = 1 if k == len(keys) else draws
            for _ in range(reps):
                a_n += asks(rng.sample(keys, k))
            p = a_n / reps
            ask_sum += p
            match_sum += p * r["real"] + (1 - p) * r["direct"]
        out[k] = (ask_sum / len(rows), match_sum / len(rows))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ref", required=True, help="the clarify run's served verdicts for these cases")
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--name", default="")
    args = ap.parse_args()
    from score import execute
    probe = {r["id"]: r for r in load_jsonl(args.probe)}
    cases = [c for c in load_jsonl(args.cases) if c["id"] in probe]
    ref = {r["id"]: r for r in load_jsonl(args.ref)}
    answers = {c["id"]: {"c%d" % i: {"parse_ok": x.get("command") is not None, "mode": x.get("mode"),
                                     "command": x.get("command")}
                         for i, x in enumerate(probe[c["id"]]["commands"])} for c in cases}
    executed = execute(cases, answers)
    rows = []
    for c in cases:
        cmds = probe[c["id"]]["commands"]
        entries = [executed.get((c["id"], "c%d" % i)) for i in range(len(cmds))]
        rows.append({"keys": sample_keys(c, probe[c["id"]], entries),
                     "real": int(ref[c["id"]]["real"]["match"]),
                     "direct": int(ref[c["id"]]["direct"]["match"])})
    res = sweep(rows, args.draws, args.seed)
    direct = sum(r["direct"] for r in rows) / len(rows)
    print("%s: %d rows; §6 answer %.1f%%" % (args.name or args.cases, len(rows), 100 * direct))
    for k in KS:
        rate, match = res[k]
        print("  k = %d (ask when the top holds <= %d): asks %.1f%%, exec-match %.1f%% (%+.1f)"
              % (k, math.floor(SHARE * k + 1e-9), 100 * rate, 100 * match, 100 * (match - direct)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
