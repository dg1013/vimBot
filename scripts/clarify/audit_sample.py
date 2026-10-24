#!/usr/bin/env python3
"""A hand audit of the simulated user: a seeded sample, printed for judging.

The simulator's prompt and reasoning budget were tuned on 50 replies
(DECISIONS.md 2026-10-06), so their error rate is optimistic for the run.
This draws a fresh seeded sample from a run's replies.jsonl and prints, per
reply, what the simulator saw -- the request, the cursor, the exact changes
-- with the question, its check line and its reply. A judge writes verdicts
to a JSON file beside it, which report.py --audit reads:

    {"<label>|<id>": {"verdict": "ok" | "wrong" | "uninformative" | "leak"
                                 | "minor", "note": "..."}, ...}

"wrong" means the reply, followed literally, leads somewhere other than the
result; "uninformative" means it answers nothing (an echo of the question);
"leak" means it hands over the method (a command, a key, an option) the
rules forbid; "minor" a harmless how-answer it should have deflected.

    python scripts/clarify/audit_sample.py --cases data/splits/val.jsonl \
        --replies runs/clarify_2026-10-06/val/replies.jsonl --n 50
"""
import argparse
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl     # noqa: E402
import protocol as P                # noqa: E402

VERDICTS = ("ok", "wrong", "uninformative", "leak", "minor")


def sample(replies, n, seed):
    rows = sorted(replies, key=lambda r: (r["id"], r["label"]))
    return random.Random(seed).sample(rows, min(n, len(rows)))


def summarize(audit):
    """{verdict: count} over an audit file's entries, every verdict present."""
    counts = {v: 0 for v in VERDICTS}
    for entry in audit.values():
        counts[entry["verdict"]] += 1
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261007)
    args = ap.parse_args()
    cases = {c["id"]: c for c in load_jsonl(args.cases)}
    for i, r in enumerate(sample(load_jsonl(args.replies), args.n, args.seed)):
        c = cases[r["id"]]
        print("#%d %s|%s  (%s)" % (i, r["label"], r["id"], c.get("category")))
        print("  request: %s" % c["instruction"])
        print("  cursor : line %d col %d (%s)" % (*c.get("cursor", (1, 1)), P.cursor_char(c)))
        for d in P.visible_diff(c["buffer_before"], c["buffer_after"]).split("\n")[:12]:
            print("  | %s" % d)
        print("  Q    : %s" % r["question"])
        print("  check: %s" % r.get("check", ""))
        print("  reply: %s%s\n" % (r["reply"], "   FLAGS %s" % r["leak_flags"]
                                   if r["leak_flags"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
