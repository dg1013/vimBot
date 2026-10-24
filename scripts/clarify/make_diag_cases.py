#!/usr/bin/env python3
"""The diagnostic case set: where a clarifying question *should* help.

Val carries no label for which instructions are ambiguous. The never-solved
train review does (runs/never_review_2026-09-25, scripts/never_review/notes.py):
each of the 1,553 train rows no full difficulty probe ever solved has a class.
This takes

* every ``ambiguous`` row -- the words admit another faithful reading;
* every ``unstated`` row -- the gold's exact form (spacing, names, values) is
  not derivable from the words or the buffer;
* a seeded sample of ``model`` rows -- fair rows the models get wrong -- as
  the control, where a question has nothing to resolve;

and writes them with the class and the note attached. These are TRAIN rows the
served adapter was trained on and never solved in eight samples, so their
absolute numbers are low by construction and say nothing about held-out
accuracy: the contrast *between classes* is the reading.
"""
import argparse
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "never_review"))
from evaluate import load_jsonl     # noqa: E402
from notes import load_notes        # noqa: E402


def build(train, notes, n_control, seed):
    by_id = {r["id"]: r for r in train}
    picked, missing = [], 0
    control = sorted(i for i, n in notes.items() if n["class"] == "model")
    random.Random(seed).shuffle(control)
    chosen = {i for i, n in notes.items() if n["class"] in ("ambiguous", "unstated")}
    chosen |= set(control[:n_control])
    for cid in sorted(chosen):
        if cid not in by_id:
            missing += 1                  # curated out of train since the notes
            continue
        n = notes[cid]
        picked.append(dict(by_id[cid], note_class=n["class"], note=n["comment"]))
    return picked, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--control", type=int, default=150, help="'model'-class rows to sample")
    ap.add_argument("--seed", type=int, default=20261006)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, missing = build(load_jsonl(args.train), load_notes(), args.control, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    counts = {}
    for r in rows:
        counts[r["note_class"]] = counts.get(r["note_class"], 0) + 1
    print("%d diagnostic cases -> %s %s; %d noted ids no longer in train" % (
        len(rows), out, counts, missing))
    return 0


if __name__ == "__main__":
    sys.exit(main())
