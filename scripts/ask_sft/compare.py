#!/usr/bin/env python3
"""Stage 1's gate (TRAINING.md): did the adapter learn to ask in the right places?

Reads the clarify harness's verdicts for the stage-1 adapter and, on the same
rows, for the served adapter it continued (the 2026-10-06 run), and checks:

    G1  held-out noted rows: the adapter's OWN choice (it asks where its
        greedy first token is "ask", and the dialogue plays out) beats
        answering directly
    G2  held-out *model* rows (fair): it asks on at most 10% of them
    G3  val: its own choice does not fall below the served adapter's direct
        answer, the row the product serves today

with the numbers each check reads: exec-match of every condition, the ask
rate by class, and the second turn's churn (answers a reply carrying no
information flips). Paired differences carry the exact McNemar p.
"""
import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts" / "clarify"):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                       # noqa: E402
from policies import flips, mcnemar_p, mean           # noqa: E402

MAX_FAIR_ASK_RATE = 0.10


def own(rows):
    """0/1 per row: the dialogue's answer where greedy asks, else the unasked one."""
    return [int((r["real"] if r["chose_ask"] else r["noask"])["match"]) for r in rows]


def col(rows, cond):
    return [int(r[cond]["match"]) for r in rows]


def paired(a, b):
    g, l = flips(a, b)
    return "%+.1f (+%d/−%d, p %.3f)" % (100 * (mean(b) - mean(a)), g, l, mcnemar_p(g, l))


def section(name, new, ref):
    """One set: the conditions for both adapters, the ask rate by class."""
    by_id = {r["id"]: r for r in ref}
    new = [r for r in new if r["id"] in by_id]
    ref = [by_id[r["id"]] for r in new]
    ids = [r["id"] for r in new]
    lines = ["## %s (%d rows)" % (name, len(ids)), "",
             "| condition | served adapter | stage-1 adapter |", "|---|---|---|"]
    for label, f in (("direct (§6 prompt)", lambda rs: col(rs, "direct")),
                     ("offer prompt, may not ask", lambda rs: col(rs, "noask")),
                     ("**its own choice**", own),
                     ("always asks, simulated reply", lambda rs: col(rs, "real"))):
        lines.append("| %s | %.1f%% | %.1f%% |" % (label, 100 * mean(f(ref)), 100 * mean(f(new))))
    lines.append("| asks of its own accord | %.1f%% | %.1f%% |" % (
        100 * mean(int(r["chose_ask"]) for r in ref), 100 * mean(int(r["chose_ask"]) for r in new)))
    classes = sorted({r.get("note_class") for r in new if r.get("note_class")})
    for cls in classes:
        sub = [r for r in new if r.get("note_class") == cls]
        subr = [r for r in ref if r.get("note_class") == cls]
        lines.append("| asks on *%s* rows (%d) | %.1f%% | %.1f%% |" % (
            cls, len(sub), 100 * mean(int(r["chose_ask"]) for r in subr),
            100 * mean(int(r["chose_ask"]) for r in sub)))
    lines.append("| turn-2 churn: answers the placebo flips | %d | %d |" % (
        sum(flips(col(ref, "noask"), col(ref, "placebo"))),
        sum(flips(col(new, "noask"), col(new, "placebo")))))
    lines += ["", "- stage-1 own choice against its direct answer: %s" % paired(col(new, "direct"), own(new)),
              "- stage-1 own choice against the served adapter's direct answer: %s"
              % paired(col(ref, "direct"), own(new))]
    return lines, new, ref


def gate(held_new, held_ref, val_new, val_ref):
    """[(check, passed, evidence)] for G1-G3."""
    fair = [r for r in held_new if r.get("note_class") == "model"]
    fair_rate = mean(int(r["chose_ask"]) for r in fair) if fair else float("nan")
    g1 = mean(own(held_new)) > mean(col(held_new, "direct"))
    g3 = mean(own(val_new)) >= mean(col(val_ref, "direct"))
    return [("G1 held-out: own choice beats direct", g1, "%.1f%% vs %.1f%%" % (
                100 * mean(own(held_new)), 100 * mean(col(held_new, "direct")))),
            ("G2 held-out fair rows: asks on ≤ %d%%" % (100 * MAX_FAIR_ASK_RATE),
             fair_rate <= MAX_FAIR_ASK_RATE, "%.1f%% of %d" % (100 * fair_rate, len(fair))),
            ("G3 val: own choice not below the served direct answer", g3, "%.1f%% vs %.1f%%" % (
                100 * mean(own(val_new)), 100 * mean(col(val_ref, "direct"))))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--held-new", required=True, help="stage-1 verdicts on the held-out rows")
    ap.add_argument("--held-ref", required=True, help="served verdicts covering those rows (diag run)")
    ap.add_argument("--val-new", required=True)
    ap.add_argument("--val-ref", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    held_lines, held_new, held_ref = section("Held-out noted rows", load_jsonl(args.held_new),
                                             load_jsonl(args.held_ref))
    val_lines, val_new, val_ref = section("Val", load_jsonl(args.val_new), load_jsonl(args.val_ref))
    checks = gate(held_new, held_ref, val_new, val_ref)
    out = ["# Stage 1 gate: did the adapter learn to ask in the right places?", "",
           "| check | result | evidence |", "|---|---|---|"]
    out += ["| %s | %s | %s |" % (c, "**pass**" if ok else "**fail**", ev) for c, ok, ev in checks]
    out += [""] + held_lines + [""] + val_lines
    pathlib.Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:4 + len(checks)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
