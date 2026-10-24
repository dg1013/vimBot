#!/usr/bin/env python3
"""Combine chosen datasets into one case file for training, eval or splits.

The point of keeping v3 and v4 separate is being able to choose. Every
consumer in the project takes a single --cases path, so selection happens
here: pick datasets, get one file, with a `dataset` field on every row so
the mix stays auditable downstream.

    python scripts/select_datasets.py --datasets v3,v4 --out data/combined.jsonl
    python scripts/select_datasets.py --datasets v4 --out data/v4_only.jsonl

Rows are emitted in dataset order, ids unchanged — so when v4 is eventually
folded into v3, nothing downstream has to be re-keyed.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_corpus import DATASETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default="v3,v4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in names if d not in DATASETS]
    if unknown:
        sys.exit("unknown dataset(s): %s (known: %s)"
                 % (unknown, ", ".join(sorted(DATASETS))))

    rows, seen = [], set()
    for ds in names:
        path = REPO / "data" / ds / ("%s_cases.jsonl" % ds)
        if not path.exists():
            sys.exit("%s not built yet — run build_corpus.py --dataset %s" % (path, ds))
        n = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in seen:
                sys.exit("id %s appears in more than one dataset" % r["id"])
            seen.add(r["id"])
            r.setdefault("dataset", ds)
            rows.append(r)
            n += 1
        print("  %-6s %5d cases" % (ds, n))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n%d cases -> %s" % (len(rows), out))
    c = collections.Counter(r.get("category") for r in rows)
    print("  categories: " + " · ".join("%s %d" % kv for kv in c.most_common(6)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
