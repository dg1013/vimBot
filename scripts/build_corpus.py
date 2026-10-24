#!/usr/bin/env python3
"""Assemble a dataset from its per-source case files, with invariants.

One builder for every dataset generation. A dataset is a directory under
data/ plus an entry in DATASETS naming its source files; the invariants a
silent `cat` would skip are enforced here:

  * every row carries the §5 required keys (gold_command, not `command`)
  * ids unique within the dataset AND across every other dataset, so v3 and
    v4 can be loaded together without collision
  * (instruction, buffer_before) unique — the §7 dedup key — again across
    datasets, so v4 cannot silently restate a v3 case
  * every row carries source, source_corpus, source_license and
    instruction_provenance, so provenance survives the merge
  * rows whose instruction is still empty are held back and counted; the
    instruction is the model's whole input (§6), so a blank one trains
    nothing while still inflating corpus size

Datasets are deliberately disjoint: v3 is the mined and generated corpus,
v4 the hand-authored one. Select either or both for a run (see
scripts/select_datasets.py). When v3 grows enough to stand alone, v4's rows
merge into it by moving one file and deleting one DATASETS entry — the ids
never change, which is what makes the phase-out a non-event.

Usage:
    python scripts/build_corpus.py --dataset v4
    python scripts/build_corpus.py --dataset all
"""
import argparse
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

# The live datasets, and deliberately only these two (owner decision,
# 2026-08-25). v1 (data/archive/v1-hand-authored/) and v2 (the 15.5k
# deep-regex generation behind the stale data/splits/) are NOT selectable and
# must not be added here: v2 is superseded by v3's regex-derived rows, and
# anything worth reviving from either comes back through the v4 ingest gate
# as a hand-authored row — executed, deduped and provenance-stamped like any
# other. Adding an archive directory to this manifest would import rows that
# never passed the current gates.
DATASETS = {
    # mined from published corpora and test suites, plus regex-derived rows
    "v3": ["kb13_cases.jsonl", "sregex_cases.jsonl",
           "emulator_cases.jsonl", "candidate_cases.jsonl",
           # rows promoted out of v4 by an owner edit (see apply_curation.py);
           # this file IS the phase-out, happening one reviewed row at a time
           "handcrafted_cases.jsonl"],
    # authored by hand against the coverage gaps; folds into v3 later
    "v4": ["handcrafted_cases.jsonl"],
}
REQUIRED = ("id", "category", "instruction", "mode", "buffer_before",
            "gold_command", "buffer_after", "source", "source_corpus",
            "source_license", "instruction_provenance")


def read(dataset):
    rows, bad, pending = [], 0, collections.Counter()
    for name in DATASETS[dataset]:
        path = DATA / dataset / name
        if not path.exists():
            sys.exit("missing source file: %s" % path)
        n = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            missing = [k for k in REQUIRED if k not in r]
            if missing:
                print("INVARIANT: %s missing %s" % (r.get("id", "?"), missing))
                bad += 1
            if not (r.get("instruction") or "").strip():
                pending[name] += 1
                continue
            r["_dataset"] = dataset
            rows.append(r)
            n += 1
        held = "  (%d pending, held back)" % pending[name] if pending[name] else ""
        print("  %-28s %5d cases%s" % (dataset + "/" + name, n, held))
    return rows, bad, pending


def check(rows):
    """Cross-dataset invariants. Returns the failure count."""
    bad = 0
    ids = collections.Counter(r["id"] for r in rows)
    dups = [k for k, v in ids.items() if v > 1]
    if dups:
        print("INVARIANT: duplicate ids across datasets: %s" % dups[:5])
        bad += 1
    pairs = collections.Counter(
        (" ".join(r["instruction"].split()).lower(), r["buffer_before"])
        for r in rows)
    n = sum(v - 1 for v in pairs.values() if v > 1)
    if n:
        print("INVARIANT: %d duplicate (instruction, buffer_before) pairs" % n)
        bad += 1
    return bad


def write(dataset, rows):
    out = DATA / dataset / ("%s_cases.jsonl" % dataset)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            r = {k: v for k, v in r.items() if k != "_dataset"}
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n%s corpus: %d cases -> %s" % (dataset, len(rows), out))
    for key in ("source", "instruction_provenance", "tier"):
        c = collections.Counter(r.get(key) for r in rows)
        print("  %-24s " % key + " · ".join("%s %d" % (k, v)
                                            for k, v in c.most_common()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="all",
                    choices=sorted(DATASETS) + ["all"])
    args = ap.parse_args()

    targets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    built, everything, bad = {}, [], 0
    for ds in targets:
        rows, b, pending = read(ds)
        bad += b
        built[ds] = rows
        if pending:
            print("  held back: %d row(s) with no instruction" % sum(pending.values()))

    # cross-dataset checks always run over every dataset on disk, not just
    # the ones being rebuilt — a v4 row must not collide with a v3 row.
    for ds in DATASETS:
        everything += built[ds] if ds in built else read(ds)[0]
    bad += check(everything)
    if bad:
        sys.exit("%d invariant failure(s); nothing written" % bad)

    for ds in targets:
        write(ds, built[ds])
    if len(targets) > 1 or args.dataset == "all":
        print("\ncombined: %d cases across %s" % (len(everything), ", ".join(DATASETS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
