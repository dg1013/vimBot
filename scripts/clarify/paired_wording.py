#!/usr/bin/env python3
"""Can a question stand in for a better-written instruction? The paired test.

The 2026-09-25 review reworded 31 test rows whose instruction was an unfair
test of its gold -- something left unstated, or two faithful readings -- and
kept their buffers. 30 of them kept their gold too (hc-0381 was retargeted),
so each is a matched pair: the same edit under the vague v1 wording and the
fixed v2 wording. These are held-out rows no checkpoint trained on, of the
kind where asking paid on the diag set.

Per checkpoint, four cells per case: v1 direct, v1 asked (the simulated
user's reply), v2 direct, v2 asked. The readings:

    the reword's effect       v2 direct - v1 direct
    asking on vague words     v1 asked  - v1 direct
    recovery                  v1 asked  against v2 direct: does a question
                              get the vague wording to where the better
                              instruction gets it?
    asking on clear words     v2 asked  - v2 direct

Paired, on the same cases, with exact McNemar p. Writes a report with the
table and every case: both wordings, the question, the reply, four verdicts.
"""
import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl                       # noqa: E402
from policies import flips, mcnemar_p, mean          # noqa: E402

CELLS = (("v1", "direct"), ("v1", "real"), ("v2", "direct"), ("v2", "real"))
COMPARISONS = (
    ("the reword's effect (v2 direct − v1 direct)", ("v1", "direct"), ("v2", "direct")),
    ("asking on the vague words (v1 asked − v1 direct)", ("v1", "direct"), ("v1", "real")),
    ("recovery (v1 asked against v2 direct)", ("v2", "direct"), ("v1", "real")),
    ("asking on the clear words (v2 asked − v2 direct)", ("v2", "direct"), ("v2", "real")),
)


def pairs(v1_cases, v2_cases):
    """Ids whose v1 and v2 rows differ only in the instruction."""
    v2 = {c["id"]: c for c in v2_cases}
    keep, skipped = [], []
    for c in v1_cases:
        d = v2.get(c["id"])
        if d is None:
            continue
        same = all(c[k] == d[k] for k in ("buffer_before", "buffer_after", "gold_command"))
        (keep if same else skipped).append(c["id"])
    return keep, skipped


def table(ids, v1, v2):
    """{cell: [0/1 per id]} for the four cells."""
    src = {"v1": v1, "v2": v2}
    return {cell: [int(src[cell[0]][i][cell[1]]["match"]) for i in ids] for cell in CELLS}


def show(s, limit=140):
    s = (s or "").replace("|", "\\|").replace("\n", "⏎")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def report_for(label, ids, v1, v2, v1_cases):
    t = table(ids, v1, v2)
    n = len(ids)
    out = ["## %s (%d matched pairs)" % (label, n), "",
           "| | direct | asked (simulated reply) |", "|---|---|---|",
           "| v1 wording (vague) | %d (%.0f%%) | %d (%.0f%%) |" % (
               sum(t[("v1", "direct")]), 100 * mean(t[("v1", "direct")]),
               sum(t[("v1", "real")]), 100 * mean(t[("v1", "real")])),
           "| v2 wording (fixed) | %d (%.0f%%) | %d (%.0f%%) |" % (
               sum(t[("v2", "direct")]), 100 * mean(t[("v2", "direct")]),
               sum(t[("v2", "real")]), 100 * mean(t[("v2", "real")])),
           "", "| comparison | Δ cases | gained / lost | McNemar p |", "|---|---|---|---|"]
    for name, a, b in COMPARISONS:
        g, lo = flips(t[a], t[b])
        out.append("| %s | %+d | +%d / −%d | %.3f |" % (name, g - lo, g, lo, mcnemar_p(g, lo)))
    # Recovery case by case, not net: of the cases the reword turned right,
    # how many does asking on the vague wording also turn right? (A net
    # count would credit a fix on some other case.)
    gained = [k for k, i in enumerate(ids)
              if t[("v2", "direct")][k] and not t[("v1", "direct")][k]]
    recovered = [ids[k] for k in gained if t[("v1", "real")][k]]
    elsewhere = [i for k, i in enumerate(ids) if t[("v1", "real")][k]
                 and not t[("v1", "direct")][k] and k not in gained]
    if gained:
        out.append("\nOf the %d cases the reword turned right (%s), asking on the vague wording "
                   "also turns right %d%s." % (
                       len(gained), ", ".join(ids[k] for k in gained), len(recovered),
                       " (%s)" % ", ".join(recovered) if recovered else ""))
    if elsewhere:
        out.append("Asking on the vague wording also turns right %d case%s the reword did not: "
                   "%s." % (len(elsewhere), "" if len(elsewhere) == 1 else "s",
                            ", ".join(elsewhere)))
    by_id = {c["id"]: c for c in v1_cases}
    mark = lambda r, cond: "✓" if r[cond]["match"] else "✗"      # noqa: E731
    out += ["", "| id | v1 instruction → v2 instruction | question asked on v1 | reply | "
            "v1 direct | v1 asked | v2 direct | v2 asked |", "|---|---|---|---|---|---|---|---|"]
    for i in ids:
        a, b = v1[i], v2[i]
        out.append("| `%s` | %s → *%s* | %s | %s | %s | %s | %s | %s |" % (
            i, show(by_id[i]["instruction"], 90), show(by_id[i].get("v2_instruction"), 90),
            show(a["question"], 110), show(a["reply"], 110), mark(a, "direct"),
            mark(a, "real"), mark(b, "direct"), mark(b, "real")))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--v1-cases", required=True, help="the v1-wording case file")
    ap.add_argument("--v2-cases", default=str(REPO / "data" / "splits" / "test.jsonl"))
    ap.add_argument("--v1-dir", required=True, help="the run's set directory for the v1 file")
    ap.add_argument("--v2-dir", required=True, help="the run's set directory for test v2")
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    v1_cases = load_jsonl(args.v1_cases)
    ids, skipped = pairs(v1_cases, load_jsonl(args.v2_cases))
    parts = ["# Paired wording: can a question stand in for a better instruction?", "",
             "The %d test rows the 2026-09-25 review reworded with buffers and gold kept "
             "(v1 wording against v2), each measured direct and asked, by `paired_wording.py`. "
             "Left out, the gold changed too: %s." % (len(ids), ", ".join(skipped) or "none")]
    for label in args.label:
        def rows(d):
            p = pathlib.Path(d) / ("verdicts_%s.jsonl" % label)
            return {r["id"]: r for r in load_jsonl(str(p))}
        parts += ["", report_for(label, ids, rows(args.v1_dir), rows(args.v2_dir), v1_cases)]
    pathlib.Path(args.out).write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("wrote %s (%d pairs; skipped %s)" % (args.out, len(ids), skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
