#!/usr/bin/env python3
"""The trained 27B's systems on the 4B, one stage per call, each resumable.

By runs/sys4b_2026-11-08/preregistration.md. Every stage that touches a model is scripts/ask27b/ask27b.py's or
scripts/retr27b/retr27b.py's own. This file only points them at the 4B's endpoints and files, so the 4B is measured the
way the 27B was: the same rows, prompts, examples, rule, simulated user and box.

    --arm a   the served 4B, no examples: ask27b's design (the 5-of-8 rule, every turn on every row)
    --arm b   the served 4B with 8 BM25 examples: retr27b --arm trained (the 27B's Part 3)
    --arm c   the base 4B with 8 BM25 examples: retr27b --arm base (the 27B's Part 2)

Arm a's stages:
    prep      CPU  its 32 samples per row: the first 32 of the 128 the pooled pick's served pass drew from this same
                   endpoint (runs/serve_pooled/served_at_k), with the verifier's scores on them. The 27B's run read its
                   own served pass's samples the same way
    greedy    4B   the §6 answer, zero-shot, every val and test row, as served (also kept as predictions)
    views question placebo answer timing judge gate: ask27b's
Arms b and c: retr27b's plan smoke greedy samples views question placebo answer timing score judge gate. Arm c's smoke
judges its longest prompts by their agreement under load, as the trained arms' do (addendum 3, the owner's word).

Exit codes as scripts/serve27b/run.sh reads them: 3 a gate failed, 4 paused, 5 the pace fell.

    python scripts/sys4b/sys4b.py prep --arm a
"""
import argparse
import json
import os
import pathlib
import sys
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "scripts" / "retr27b"))
import retr27b as T                                               # noqa: E402

Q, A = T.Q, T.A
PLAIN_ENDPOINT = A.endpoint27                                     # pass27b's own: arm a's, with no examples

ROOT = REPO / "runs" / "sys4b_2026-11-08"
SERVED_DIR = ROOT / "a" / "served"                                # arm a's answers as served: samples, scores, greedy
SERVED_AT_K = REPO / "runs" / "serve_pooled" / "served_at_k"      # the pooled pick's pass: the 4B's 128 samples per row
MODEL = "qwen3.5-4b"                                              # the registry entry: tokenizer, template, stops
NAME_SERVED, NAME_BASE = "qwen3.5-4b-served", "qwen3.5-4b-base"   # serve4b.sh's names
URL_SERVED, URL_BASE = "http://127.0.0.1:8011", "http://127.0.0.1:8016"
RECORDED_BASE = REPO / "runs" / "retrieval_2026-10-18" / "base-retr8" / "preds_val.jsonl"   # the base 4B, these prompts, transformers
RECORDED_HF = REPO / "runs" / "qwen3.5-4b-grpo-gated" / "preds_val.jsonl"                 # the served adapter, transformers
N_A = 32                                                          # ask27b's samples per row
LABELS = {"a": "served4b", "b": "trained4b-retr8", "c": "base4b-retr8"}
RETR_ARM = {"b": "trained", "c": "base"}
OTHER_HOURS = 2.0                                                 # every stage but samples: the simulated user about 1.25 h


def use_4b(arm):
    """Point the 27B's stages at the 4B. They read these names at call time."""
    A.MODEL27, A.NAME27 = MODEL, NAME_SERVED
    A.URL27 = URL_BASE if arm == "c" else URL_SERVED
    A.URL4 = URL_SERVED                                           # the verifier: a LoRA on the served 4B
    Q.SERVE27 = SERVED_DIR
    Q.LABEL = LABELS[arm]
    T.BASE_NAME = NAME_BASE
    T.SERVED = {"base": NAME_BASE, "trained": NAME_SERVED}
    T.RECORDED = RECORDED_BASE
    zero = SERVED_DIR / "preds_val.jsonl"
    T.RECORDED_ZERO = zero if zero.exists() else RECORDED_HF     # arm a's, once it has run; transformers' before
    T.OTHER_HOURS = OTHER_HOURS
    if arm in RETR_ARM:
        T.ARM = RETR_ARM[arm]
        A.endpoint27 = T.endpoint_with_examples
    else:
        T.ARM = "trained"
        A.endpoint27 = PLAIN_ENDPOINT


def write_rows(path, rows):
    """All rows or none: a half-written file never stands."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def prep_rows(old, cases, n=N_A):
    """(sample rows, score rows) for arm a: the first n of a row's 4B samples, and the verifier's stored score on each
    distinct one. A row or a score the pass lacks raises: arm a reads every row, as the 27B's did."""
    samples, scores = [], []
    for sp, cs in cases.items():
        for c in cs:
            r = old[sp][c["id"]]
            got = r["samples"]["4b"][:n]
            if len(got) != n:
                raise SystemExit("%s holds %d 4B samples, not %d" % (c["id"], len(got), n))
            stored = {(x["mode"], x["command"]): x["score"] for x in r["cands"]}
            samples.append({"id": c["id"], "split": sp, "samples": got, "output_tokens": r["output_tokens"]["4b"][:n],
                            "prompt_tokens": r["prompt_tokens"]["4b"]})
            for mode, command in Q.distinct(got):
                if (mode, command) not in stored:
                    raise SystemExit("%s: no stored score for %s: %s" % (c["id"], mode, command))
                scores.append({"id": c["id"], "split": sp, "mode": mode, "command": command,
                               "score": stored[(mode, command)]})
    return samples, scores


def cmd_prep(args):
    """No model. Always writes the real run's files: a dry run reads them too, as the 27B's read its served pass."""
    SERVED_DIR.mkdir(parents=True, exist_ok=True)
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    samples, scores = prep_rows({sp: A.old_rows(sp) for sp in A.SPLITS}, cases)
    write_rows(SERVED_DIR / "samples.jsonl", samples)
    write_rows(SERVED_DIR / "scores27.jsonl", scores)
    print("prep: %d rows, %d samples each, %d distinct answers with their scores" % (len(samples), N_A, len(scores)), flush=True)
    return 0


def preds_rows(rows):
    """Prediction rows ({id, mode, command}) from a greedy file, in its order; an answer that did not parse has none."""
    got = T.greedy_preds(rows)
    return [dict(got[r["id"]], id=r["id"]) for r in rows if r["id"] in got]


def cmd_greedy_a(args):
    out = args.run / "served"
    out.mkdir(parents=True, exist_ok=True)
    rc = A.cmd_greedy(SimpleNamespace(run=out, limit=args.limit))
    if rc:
        return rc
    for sp in A.SPLITS:
        write_rows(out / ("preds_%s.jsonl" % sp), preds_rows(A.load_jsonl(out / ("greedy_%s.jsonl" % sp))))
    print("greedy: %s" % ", ".join("%s %d rows" % (sp, len(A.load_jsonl(out / ("greedy_%s.jsonl" % sp)))) for sp in A.SPLITS),
          flush=True)
    return 0


STAGES_A = {"prep": cmd_prep, "greedy": cmd_greedy_a, "views": Q.cmd_views, "question": Q.cmd_question,
            "placebo": Q.cmd_placebo, "answer": Q.cmd_answer, "timing": Q.cmd_timing, "judge": Q.cmd_judge,
            "gate": Q.cmd_gate}


AGREEING = T.agreeing                                             # retr27b's own: with the record, or with each other


class LongRows:
    """Addendum 3, at the owner's word (2026-11-08): arm c's smoke judges its longest prompts the way the trained arms'
    smokes do, by how many of the 8 answers under load agree with each other. Their match with the transformers record,
    which stopped the first dry run at 54 of 80, is counted beside it and reported. It no longer stops the run."""

    def __init__(self):
        self.recorded = self.answers = 0

    def __call__(self, samples, recorded=None):
        if recorded is not None:
            self.recorded += sum(A.same_answer(s, recorded) for s in samples)
            self.answers += len(samples)
        return AGREEING(samples, None)


def cmd_smoke_c(args):
    """retr27b's smoke on arm c, its longest prompts judged by LongRows. smoke.json also gets the record's count."""
    long_rows = LongRows()
    T.agreeing = long_rows
    try:
        rc = T.cmd_smoke(args)
    finally:
        T.agreeing = AGREEING
    path = args.run / "smoke.json"
    rep = json.loads(path.read_text())
    rep.update(longest_judged_by="agreement under load (addendum 3)", longest_identical_to_recorded=long_rows.recorded,
               longest_recorded_answers=long_rows.answers)
    A.write_json(path, rep)
    print("the longest prompts: %d of %d answers agree under load; %d of %d match the transformers record" % (
        rep["longest_identical"], rep["longest_answers"], long_rows.recorded, long_rows.answers), flush=True)
    return rc


def stages(arm):
    if arm == "a":
        return STAGES_A
    return dict(T.STAGES, smoke=cmd_smoke_c) if arm == "c" else T.STAGES


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage")
    ap.add_argument("--arm", choices=("a", "b", "c"), required=True)
    ap.add_argument("--run", default=None, help="default runs/sys4b_2026-11-08/<arm>")
    ap.add_argument("--limit", type=int, default=0, help="only the first rows of each split (a dry run)")
    ap.add_argument("--n", type=int, default=16, help="samples per row (arms b and c)")
    ap.add_argument("--inflight", type=int, default=1, help="rows in flight in the samples stage")
    ap.add_argument("--budget-s", type=float, default=18000.0, help="no new row starts this long into the samples stage")
    ap.add_argument("--smoke-n", type=int, default=40)
    ap.add_argument("--long-n", type=int, default=10)
    ap.add_argument("--timing-n", type=int, default=200)
    ap.add_argument("--rescore-n", type=int, default=100)
    args = ap.parse_args(argv)
    table = stages(args.arm)
    if args.stage not in table:
        ap.error("arm %s has no stage %r (it has %s)" % (args.arm, args.stage, ", ".join(sorted(table))))
    args.run = pathlib.Path(args.run) if args.run else ROOT / args.arm
    args.run.mkdir(parents=True, exist_ok=True)
    use_4b(args.arm)
    return table[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
