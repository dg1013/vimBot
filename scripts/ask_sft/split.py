#!/usr/bin/env python3
"""Stage 1's split: 30% of the noted diag rows held out to judge the decision.

From the 422 noted never-solved train rows of the 2026-10-06 run
(unstated / ambiguous / model), a seeded, class-stratified 30% is held out:
never sampled for dialogues, never in the train file (build_set.py asserts
it), and the evaluation set of the stage-1 gate. Writes `heldout.jsonl`,
`diag_train.jsonl`, and `train_ask_cases.jsonl` (the train-split rows that
are taught to ask: unstated and ambiguous).
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl                 # noqa: E402
from data import ASK_CLASSES, split_noted       # noqa: E402


def write(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-cases", default=str(REPO / "runs" / "clarify_2026-10-06" / "diag_cases.jsonl"))
    ap.add_argument("--frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, held = split_noted(load_jsonl(args.diag_cases), args.frac, args.seed)
    ask = [c for c in train if c["note_class"] in ASK_CLASSES]
    write(out / "heldout.jsonl", held)
    write(out / "diag_train.jsonl", train)
    write(out / "train_ask_cases.jsonl", ask)
    count = lambda rows: dict(collections.Counter(r["note_class"] for r in rows))   # noqa: E731
    print("held out %d %s; train %d %s; taught to ask %d" % (
        len(held), count(held), len(train), count(train), len(ask)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
