#!/usr/bin/env python3
"""Ask when sampled answers differ: the rule scored offline, from a probe.

The owner (2026-10-09): "first check to see if there are multiple different
types of vim outputs for a question, and then decide to ask a clarification
question if there are." A difficulty probe already holds, per row, the
served adapter's 8 answers sampled at T = 1 (scripts/probe_difficulty.py:
each distinct command, with how many samples gave it). This scores the rule
on the rows that probe covers. The sandbox runs every distinct command once,
so "different" can be read three ways:

    text     two or more distinct commands
    result   two or more distinct buffers among the commands that ran and
             changed something (sandbox; research only)
    type     both an ex and a keys answer among the samples

It also reads agreement as a threshold: ask when the most common result
holds at most k of the samples.

Every rule asks with the 2026-10-06 generic question, so only *when*
changes. Asked rows get that dialogue's answer; the rest get the unasked
answer (the clarify policies' convention) or, end to end, the §6 answer.

    python scripts/readings/sample_rule.py \
        --probe runs/cliff_2026-09-16/probe_before_qwen3.5-4b.jsonl \
        --cases runs/clarify_2026-10-06/diag_cases.jsonl \
        --ref runs/clarify_2026-10-06/diag/verdicts_served.jsonl --out <report.md>
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                                       # noqa: E402
from policies import (at_rate, auroc, flips, mcnemar_p, mean, oracle,  # noqa: E402
                      random_at_rate, threshold_curve)
import readings_protocol as R                                         # noqa: E402

CLASSES = ("unstated", "ambiguous", "model")


def sample_view(case, probe_row, entries):
    """What the samples say for one row.

    `entries[i]` is the executor's entry for probe_row["commands"][i] (None
    when it has no command). Returns the sample count, distinct commands,
    whether both modes occur, the distinct results with their sample masses
    (largest first), and the mass of samples that failed or changed nothing."""
    cmds = probe_row["commands"]
    n = sum(c["n"] for c in cmds)
    masses, bad = collections.Counter(), 0
    for c, e in zip(cmds, entries):
        status = R.reading_status(case, e) if c.get("command") is not None else "unrun"
        if status == "ok":
            masses[R.norm_key(e["buffer_after"])] += c["n"]
        else:
            bad += c["n"]
    results = sorted(masses.values(), reverse=True)
    return {"n": n, "distinct_text": sum(1 for c in cmds if c.get("command") is not None),
            "both_types": len({c["mode"] for c in cmds if c.get("command") is not None}) > 1,
            "results": results, "distinct_results": len(results), "failed": bad,
            "top_share": (results[0] / n) if results else 0.0}


def rules(views):
    """[(name, {id: ask})]: the three readings of "different", and agreement thresholds."""
    out = [("text: two or more distinct commands", {i: v["distinct_text"] >= 2 for i, v in views.items()}),
           ("**result: two or more distinct results**", {i: v["distinct_results"] >= 2 for i, v in views.items()}),
           ("type: both ex and keys answers", {i: v["both_types"] for i, v in views.items()})]
    for k in (7, 6, 5, 4, 3, 2):
        out.append(("the top result holds at most %d of 8" % k,
                    {i: v["top_share"] * 8 <= k + 1e-9 for i, v in views.items()}))
    return out


def score(rows, ask):
    """(ask rate, exec-match with the unasked answer elsewhere, exec-match end
    to end with the §6 answer elsewhere, recall and precision of fixable rows)."""
    m = lambda r, c: int(r[c]["match"])                                     # noqa: E731
    a = [ask[r["id"]] for r in rows]
    pol = mean(m(r, "real") if x else m(r, "noask") for r, x in zip(rows, a))
    e2e = [m(r, "real") if x else m(r, "direct") for r, x in zip(rows, a)]
    fix = [int(r["real"]["match"] and not r["noask"]["match"]) for r in rows]
    hit = sum(x and f for x, f in zip(a, fix))
    return {"rate": mean(a), "policy": pol, "e2e": e2e,
            "recall": hit / sum(fix) if sum(fix) else float("nan"),
            "precision": hit / sum(a) if sum(a) else float("nan")}


def pct(x):
    return "—" if x != x else "%.1f%%" % (100 * x)


def report(rows, views, name):
    m = lambda r, c: int(r[c]["match"])                                     # noqa: E731
    direct = [m(r, "direct") for r in rows]
    fix_rate = mean(int(r["real"]["match"] and not r["noask"]["match"]) for r in rows)
    pr = [{"id": r["id"], "real": m(r, "real"), "noask": m(r, "noask")} for r in rows]
    o_rate, o_match = oracle(pr)
    nd = collections.Counter(v["distinct_text"] for v in views.values())
    rd = collections.Counter(v["distinct_results"] for v in views.values())
    out = ["# Ask when sampled answers differ: %s" % name, "",
           "Generated by `scripts/readings/sample_rule.py`. %d rows; the served adapter's 8 answers "
           "per row, sampled at T = 1 by the difficulty probe; every rule asks with the 2026-10-06 "
           "generic question. Exec-match *policy*: the unasked answer where it does not ask; *end to "
           "end*: the §6 answer there." % len(rows), "",
           "## The samples", "",
           "| | |", "|---|---|",
           "| distinct commands among 8 (1..8) | %s |" % " / ".join(str(nd[k]) for k in range(1, 9)),
           "| distinct results among 8 (0..8) | %s |" % " / ".join(str(rd[k]) for k in range(0, 9)),
           "| rows where a sample gives the expected result | %s |" % pct(mean(
               int(r["id"] in views and views[r["id"]]["n"] > 0 and any_match(r)) for r in rows)),
           "| the §6 answer right | %s |" % pct(mean(direct)), "",
           "## The rules", "",
           "| ask when | asks | policy | end to end (vs §6 %s) | fixable rows it asks on (recall) | its asks that are fixable (precision; random %s) |"
           % (pct(mean(direct)), pct(fix_rate)),
           "|---|---|---|---|---|---|"]
    for name_, ask in rules(views):
        s = score(rows, ask)
        g, l = flips(direct, s["e2e"])
        out.append("| %s | %s | %s | %s (+%d/−%d, p %.3f) | %s | %s |" % (
            name_, pct(s["rate"]), pct(s["policy"]), pct(mean(s["e2e"])), g, l, mcnemar_p(g, l),
            pct(s["recall"]), pct(s["precision"])))
    head = score(rows, dict(rules(views))["**result: two or more distinct results**"])
    out += ["| *random, at the headline rate* | %s | %s | — | — | %s |" % (
                pct(head["rate"]), pct(random_at_rate(pr, head["rate"])), pct(fix_rate)),
            "| *the top P(ask), at the headline rate* | %s | %s | — | — | — |" % (
                pct(head["rate"]), pct(at_rate(threshold_curve(
                    [dict(p, p=r["p_ask_norm"]) for p, r in zip(pr, rows)], key="p"), head["rate"])))]
    out += ["| *never ask* | 0.0%% | %s | %s | — | — |" % (pct(mean(m(r, "noask") for r in rows)), pct(mean(direct))),
            "| *always ask* | 100.0%% | %s | %s | 100%% | %s |" % (
                pct(mean(m(r, "real") for r in rows)), pct(mean(m(r, "real") for r in rows)), pct(fix_rate)),
            "| *oracle* | %s | %s | — | 100%% | 100%% |" % (pct(o_rate), pct(o_match)), ""]
    # ranking
    out += ["## As a ranking (AUROC, 0.5 = chance)", "",
            "| task | distinct results | 1 − top result's share | distinct commands | the served P(ask) |",
            "|---|---|---|---|---|"]
    tasks = [("unstated vs fair", lambda r: r.get("note_class") in ("unstated", "model"), lambda r: r.get("note_class") == "unstated"),
             ("unstated or ambiguous vs fair", lambda r: r.get("note_class") in CLASSES, lambda r: r.get("note_class") != "model"),
             ("ambiguous vs fair", lambda r: r.get("note_class") in ("ambiguous", "model"), lambda r: r.get("note_class") == "ambiguous"),
             ("fixable vs not (all rows)", lambda r: True, lambda r: bool(r["real"]["match"] and not r["noask"]["match"])),
             ("fixable among wrong answers", lambda r: not r["noask"]["match"], lambda r: bool(r["real"]["match"]))]
    for tname, keep, pos in tasks:
        sub = [r for r in rows if keep(r)]
        lab = [pos(r) for r in sub]
        cols = []
        for f in (lambda r: views[r["id"]]["distinct_results"], lambda r: 1 - views[r["id"]]["top_share"],
                  lambda r: views[r["id"]]["distinct_text"], lambda r: r["p_ask_norm"]):
            a = auroc([f(r) for r in sub], lab)
            cols.append("—" if a is None else "%.2f" % a)
        out.append("| %s (%d) | %s |" % (tname, len(sub), " | ".join(cols)))
    # by class for the headline rule
    if any(r.get("note_class") for r in rows):
        ask = dict(rules(views))["**result: two or more distinct results**"]
        out += ["", "## By class, for the headline rule (two or more distinct results)", "",
                "| class | rows | asks | end to end | §6 answer |", "|---|---|---|---|---|"]
        for c in CLASSES:
            sub = [r for r in rows if r.get("note_class") == c]
            if sub:
                s = score(sub, ask)
                out.append("| %s | %d | %s | %s | %s |" % (c, len(sub), pct(s["rate"]), pct(mean(s["e2e"])),
                                                           pct(mean(m(r, "direct") for r in sub))))
    return out


_MATCHED = {}


def any_match(r):
    return _MATCHED.get(r["id"], False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probe", required=True, help="a difficulty-probe JSONL (8 samples per row)")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--ref", required=True, help="the 2026-10-06 served verdicts for these cases")
    ap.add_argument("--out", required=True, help="report.md (a .json of the per-row views beside it)")
    ap.add_argument("--name", default="diag")
    args = ap.parse_args()
    from score import execute
    probe = {r["id"]: r for r in load_jsonl(args.probe)}
    cases = [c for c in load_jsonl(args.cases) if c["id"] in probe]
    ref = {r["id"]: r for r in load_jsonl(args.ref)}
    answers = {c["id"]: {"c%d" % i: {"parse_ok": x.get("command") is not None, "mode": x.get("mode"),
                                     "command": x.get("command")}
                         for i, x in enumerate(probe[c["id"]]["commands"])} for c in cases}
    executed = execute(cases, answers)
    views = {}
    for c in cases:
        cmds = probe[c["id"]]["commands"]
        entries = [executed.get((c["id"], "c%d" % i)) for i in range(len(cmds))]
        views[c["id"]] = sample_view(c, probe[c["id"]], entries)
        _MATCHED[c["id"]] = any(e is not None and e["match"] for e in entries)
    rows = [dict(ref[c["id"]], note_class=c.get("note_class")) for c in cases]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report(rows, views, args.name)) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(views, indent=0) + "\n")
    print("%d rows -> %s" % (len(rows), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
