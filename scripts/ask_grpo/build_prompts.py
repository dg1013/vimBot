#!/usr/bin/env python3
"""Stage 2's prompts: where asking pays, where it does not, and replay.

* every train-split noted row of stage 1's split -- *unstated* and
  *ambiguous* (asking can pay: the bank holds real replies to real
  questions) and *model* (fair rows the model gets wrong: the bank holds the
  9B's replies, which add little);
* `--replay` fair rows from the rest of train (no note, or a *model* note
  outside the diag set), labelled `replay`: the bank holds nothing for them,
  so a question there earns the placebo reply and costs c.

The held-out rows of stage 1's split stay out (asserted): they judge stage 2
as they judge stage 1.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "never_review"))
from evaluate import load_jsonl     # noqa: E402


def build(diag_train, held_ids, train_rows, note_classes, n_replay, seed):
    """Case rows with `note_class` set; see the module docstring."""
    taken = {c["id"] for c in diag_train} | set(held_ids)
    pool = sorted((c for c in train_rows if c["id"] not in taken
                   and note_classes.get(c["id"]) in (None, "model")), key=lambda c: c["id"])
    replay = random.Random(seed).sample(pool, min(n_replay, len(pool)))
    rows = [dict(c) for c in diag_train] + [dict(c, note_class="replay") for c in replay]
    assert not {r["id"] for r in rows} & set(held_ids), "held-out rows in the prompt set"
    random.Random(seed + 1).shuffle(rows)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage1-run", required=True, help="stage 1's RUN (its split)")
    ap.add_argument("--train", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--replay", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from notes import load_notes
    s1 = pathlib.Path(args.stage1_run)
    rows = build(load_jsonl(str(s1 / "diag_train.jsonl")),
                 [c["id"] for c in load_jsonl(str(s1 / "heldout.jsonl"))],
                 load_jsonl(args.train), {i: n["class"] for i, n in load_notes().items()},
                 args.replay, args.seed)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d prompts -> %s %s" % (len(rows), args.out,
                                   dict(collections.Counter(r["note_class"] for r in rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
