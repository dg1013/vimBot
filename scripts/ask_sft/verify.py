#!/usr/bin/env python3
"""Stage 1 verification: which sampled dialogues end in a command that exec-matches.

The filter of rejection sampling (README.md). For every question asked of a
train-split ask row -- the greedy one from the diag run (`q0`) and the
sampled ones (`q1`..`qk`) -- this joins the question, the simulated user's
reply and the student's turn-2 answer, executes the answer through
src/evaluate.score (via scripts/clarify/score.execute, the same scoring every
clarify number uses), and screens the reply with the current leak screen.
Only a dialogue whose answer exec-matches and whose reply is clean is used
to train; the rest are kept on disk for the report.
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                       # noqa: E402
from protocol import leak_flags, quotes_result         # noqa: E402
import score as clarify_score                          # noqa: E402


def greedy_dialogues(verdicts, ids):
    """q0: the greedy question, its reply and the answer, from a diag verdict file."""
    out = []
    for v in verdicts:
        if v["id"] in ids:
            out.append({"id": v["id"], "label": "q0", "question": v["question"],
                        "reply": v["reply"], "mode": v["real"]["mode"],
                        "command": v["real"]["command"], "match": v["real"]["match"]})
    return out


def sampled_dialogues(cases, labels, sample_dir, timeout=5.0, workers=8):
    """q1..qk: join each label's question, reply and turn-2 answer; execute."""
    d = pathlib.Path(sample_dir)
    replies = {(r["label"], r["id"]): r["reply"] for r in load_jsonl(str(d / "replies.jsonl"))}
    rows, answers = [], {}
    for label in labels:
        qs = {r["id"]: r["question"]["text"] for r in load_jsonl(str(d / ("questions_%s.jsonl" % label)))}
        t2 = {r["id"]: r["answer"] for r in load_jsonl(str(d / ("turn2_%s.jsonl" % label)))}
        for cid, q in qs.items():
            if cid not in t2 or (label, cid) not in replies:
                continue                       # a stage still missing: not a dialogue yet
            answers.setdefault(cid, {})[label] = t2[cid]
            rows.append({"id": cid, "label": label, "question": q,
                         "reply": replies[(label, cid)], "mode": t2[cid]["mode"],
                         "command": t2[cid]["command"]})
    by_id = {c["id"]: c for c in cases}
    executed = clarify_score.execute([by_id[i] for i in answers], answers, timeout, workers)
    for r in rows:
        e = executed[(r["id"], r["label"])]
        r["match"] = bool(e and e["match"])
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True, help="the train-split ask rows")
    ap.add_argument("--greedy-verdicts", required=True,
                    help="the diag run's verdicts_<label>.jsonl (q0)")
    ap.add_argument("--sample-dir", required=True, help="sample_questions.py --out-dir")
    ap.add_argument("--labels", nargs="+", default=["q1", "q2", "q3"])
    ap.add_argument("--out", required=True, help="dialogues.jsonl")
    args = ap.parse_args()

    cases = load_jsonl(args.cases)
    by_id = {c["id"]: c for c in cases}
    rows = greedy_dialogues(load_jsonl(args.greedy_verdicts), set(by_id))
    rows += sampled_dialogues(cases, args.labels, args.sample_dir)
    for r in rows:
        c = by_id[r["id"]]
        r["note_class"] = c.get("note_class")
        r["leak_flags"] = leak_flags(r["reply"], c)
        r["quotes_result"] = quotes_result(r["reply"], c)
    rows.sort(key=lambda r: (r["id"], r["label"]))
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if r["match"] and not r["leak_flags"] and not r["quotes_result"]]
    print("%d dialogues over %d rows -> %s; verified %d, on %d rows (%s)" % (
        len(rows), len({r["id"] for r in rows}), args.out, len(ok), len({r["id"] for r in ok}),
        ", ".join("%s %d" % (k, len({r["id"] for r in ok if r["note_class"] == k}))
                  for k in ("unstated", "ambiguous"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
