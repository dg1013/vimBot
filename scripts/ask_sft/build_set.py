#!/usr/bin/env python3
"""Stage 1's train file: verified ask dialogues, placebo turns, and replay.

Three kinds of row (data.py has their exact shape):

* **ask** -- for each train-split *unstated* / *ambiguous* row with a
  verified dialogue (its answer exec-matched, its reply passed the leak
  screen): the question as a turn-1 target and the gold as the turn-2
  target, for up to `--max-dialogues` distinct questions. Rows with no
  verified dialogue contribute nothing: only what the verifier accepted is
  taught.
* **placebo** -- for every train-split diag row (all three classes), its
  greedy question followed by "Please just go with the instruction as
  written." and the gold. Only the answer carries loss, so this teaches the
  second turn not to churn, without teaching asking on the row.
* **answer** -- fair rows answered at once under the offer prompt: the
  train-split *model* rows of diag, a sample of the other *model* rows of
  the never-solved review (hard but fair), and a sample of ordinary train
  rows (replay, so the single-turn skill is not forgotten).

No held-out row appears in any of the three (asserted): the held-out rows
are the evaluation set for the decision.
"""
import argparse
import collections
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts" / "never_review", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                                     # noqa: E402
from data import (ASK_CLASSES, answer_row, ask_rows, pick_dialogues,  # noqa: E402
                  placebo_row)


def build(diag_train, held_ids, dialogues, greedy_questions, train_rows, note_classes,
          max_dialogues, n_model, n_other, seed):
    """The rows and a manifest; see the module docstring."""
    rng = random.Random(seed)
    rows, manifest = [], {"held_out": len(held_ids)}
    picked = pick_dialogues(dialogues, max_dialogues)
    asked = 0
    for c in diag_train:
        if c["note_class"] in ASK_CLASSES and c["id"] in picked:
            asked += 1
            for d in picked[c["id"]]:
                rows += ask_rows(c, d["question"], d["reply"])
    manifest["ask rows with a verified dialogue"] = asked
    manifest["ask rows without one (not taught)"] = sum(
        1 for c in diag_train if c["note_class"] in ASK_CLASSES and c["id"] not in picked)
    for c in diag_train:
        q = greedy_questions.get(c["id"])
        if q:
            rows.append(placebo_row(c, q))
    diag_ids = {c["id"] for c in diag_train} | set(held_ids)
    answer_cases = [c for c in diag_train if c["note_class"] == "model"]
    model_pool = sorted((c for c in train_rows if note_classes.get(c["id"]) == "model"
                         and c["id"] not in diag_ids), key=lambda c: c["id"])
    other_pool = sorted((c for c in train_rows if c["id"] not in note_classes
                         and c["id"] not in diag_ids), key=lambda c: c["id"])
    answer_cases += rng.sample(model_pool, min(n_model, len(model_pool)))
    answer_cases += rng.sample(other_pool, min(n_other, len(other_pool)))
    rows += [answer_row(c) for c in answer_cases]
    leaked = {r["id"] for r in rows} & set(held_ids)
    assert not leaked, "held-out rows in the train file: %s" % sorted(leaked)[:5]
    rng.shuffle(rows)
    manifest["rows by kind"] = dict(collections.Counter(r["kind"] for r in rows))
    manifest["rows"] = len(rows)
    return rows, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--diag-train", required=True, help="the train split of the noted diag rows")
    ap.add_argument("--held-out", required=True, help="the held-out split (only its ids are read)")
    ap.add_argument("--dialogues", required=True, help="verify.py --out")
    ap.add_argument("--greedy", required=True, help="the diag run's turn1_<label>.jsonl")
    ap.add_argument("--train", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--max-dialogues", type=int, default=2)
    ap.add_argument("--model-rows", type=int, default=300,
                    help="other never-solved 'model' rows answered at once (hard, fair)")
    ap.add_argument("--other-rows", type=int, default=400, help="ordinary replay rows")
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from notes import load_notes
    note_classes = {i: n["class"] for i, n in load_notes().items()}
    rows, manifest = build(
        load_jsonl(args.diag_train), [c["id"] for c in load_jsonl(args.held_out)],
        load_jsonl(args.dialogues),
        {r["id"]: r["question"]["text"] for r in load_jsonl(args.greedy)},
        load_jsonl(args.train), note_classes, args.max_dialogues, args.model_rows,
        args.other_rows, args.seed)
    out = pathlib.Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out.with_name("train_set_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
