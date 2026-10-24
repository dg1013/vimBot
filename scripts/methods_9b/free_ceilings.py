#!/usr/bin/env python3
"""The ceilings already inside the 9B methods run (CPU): what each model, and
the two together, would score with a perfect chooser over what they produced.

Per model and split: the §6 answer; any of its 8 samples right; the answer
after the simulated user's reply; asking only where it helps; and anything it
produced (the §6 answer, any of the 8 samples, or the answer after the reply).
Then the same over both models' outputs together. These are bounds under an
oracle, never deployable numbers; the 128-sample pass@k (scripts/ceiling)
measures how far the distribution itself reaches.

    python scripts/methods_9b/free_ceilings.py --run runs/qwen9b_methods_2026-10-19 --out <md>
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def outcomes(probe_rows, verdicts, ids):
    """{name: {id: 0/1}} for one model."""
    probe = {r["id"]: r for r in probe_rows}
    out = {"§6 answer": {}, "any of 8 samples": {}, "after the reply": {}, "ask only where it helps": {},
           "anything it produced": {}}
    for i in ids:
        v = verdicts[i]
        d, r = int(bool(v["direct"]["match"])), int(bool(v["real"]["match"]))
        s = int(int(probe[i]["k"]) > 0)
        out["§6 answer"][i] = d
        out["any of 8 samples"][i] = s
        out["after the reply"][i] = r
        out["ask only where it helps"][i] = max(d, r)
        out["anything it produced"][i] = max(d, r, s)
    return out


def either(a, b):
    return {i: max(a[i], b[i]) for i in a}


def table(split, m4, m9, ids):
    pct = lambda o: 100 * sum(o[i] for i in ids) / len(ids)        # noqa: E731
    lines = ["## %s (%d rows)" % (split, len(ids)), "", "| ceiling | 4B | 9B | 4B or 9B |", "|---|---|---|---|"]
    for name in m4:
        lines.append("| %s | %.1f%% | %.1f%% | %.1f%% |" % (name, pct(m4[name]), pct(m9[name]), pct(either(m4[name], m9[name]))))
    only4 = sum(1 for i in ids if m4["anything it produced"][i] and not m9["anything it produced"][i])
    only9 = sum(1 for i in ids if m9["anything it produced"][i] and not m4["anything it produced"][i])
    neither = sum(1 for i in ids if not m4["anything it produced"][i] and not m9["anything it produced"][i])
    lines += ["", "Rows only the 4B reaches: %d; only the 9B: %d; neither, with everything both produced: %d (%.1f%%)." % (
        only4, only9, neither, 100 * neither / len(ids)), ""]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(RUNS / "qwen9b_methods_2026-10-19"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    lines = ["# Ceilings inside the 9B methods run", "",
             "A perfect chooser over what each model produced in this run (8 samples at T = 1, the §6 answer, "
             "the answer after the simulated user's reply). Bounds, not deployable numbers.", ""]
    four_verdicts = {"test": RUNS / "clarify_2026-10-07" / "test" / "verdicts_served.jsonl",
                     "val": RUNS / "clarify_2026-10-06" / "val" / "verdicts_served.jsonl"}
    for split in ("test", "val"):
        v9_path = run / "clarify" / split / "verdicts_served9b.jsonl"
        if not v9_path.exists():
            continue
        cases = [json.loads(l)["id"] for l in open(REPO / "data" / "splits" / ("%s.jsonl" % split))]
        v9 = {r["id"]: r for r in load_jsonl(v9_path)}
        v4 = {r["id"]: r for r in load_jsonl(four_verdicts[split])}
        p9 = load_jsonl(run / ("probe_%s.jsonl" % split))
        p4 = load_jsonl(RUNS / "sample_rule_2026-10-09" / ("probe_%s.jsonl" % split))
        ids = [i for i in cases if i in v9 and i in v4 and i in {r["id"] for r in p9}]
        lines += table(split, outcomes(p4, v4, ids), outcomes(p9, v9, ids), ids)
    text = "\n".join(lines) + "\n"
    pathlib.Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
