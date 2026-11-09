#!/usr/bin/env python3
"""The thinking run's readings, by runs/think27b_2026-11-01/preregistration.md: each thinking arm
against its no-thinking control on the same rows (primary), the rescues of the trained 27B's
misses, the reasoning's length and format, and the rows no non-thinking source solves. CPU only;
works on partial results.

    python scripts/think27b/report.py
"""
import importlib.util
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load("distill_targets_compare", REPO / "scripts" / "distill_targets" / "compare.py")   # compare(), pct()
T = load("think27b_think27b", HERE / "think27b.py")

READINGS = (("TA", "nothink-trained", "think-trained"), ("TB", "nothink-retr8", "think-retr8"))
VERDICT = {"stronger": "helps", "weaker": "hurts", "tied": "tied"}
# Non-thinking references on the same rows, all from existing val runs.
GREEDY = {"trained 27B, greedy": "runs/qwen3.5-27b-sft-distill/exec_val.jsonl",
          "prompted 27B (8 examples), greedy": "runs/teacher27b_2026-10-30/27b-retr8/exec_val.jsonl",
          "9B (SFT + GRPO), greedy": "runs/qwen3.5-9b-grpo-gated/exec_val.jsonl",
          "served 4B, greedy": "runs/qwen3.5-4b-grpo-gated-leg2/exec_val.jsonl"}
SAMPLED = ("prompted 27B (8 examples), any of 8 samples at T = 1", "runs/teacher27b_2026-10-30/probe_val_27b-retr8.jsonl")
TIERS = ("T1", "T2", "T3")


def jsonl(path):
    p = REPO / path if not pathlib.Path(path).is_absolute() else pathlib.Path(path)
    if not p.exists():
        return None
    return [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]


def arm_rows(run, arm):
    rows = jsonl(run / ("rows_%s.jsonl" % arm)) or []
    return {r["id"]: r for r in rows}


def counts(match, rows, ids):
    sets = {"all": ids, "misses": [i for i in ids if rows[i]["in_fail"]], "never solved": [i for i in ids if rows[i]["in_hard"]]}
    return {k: {"n": len(v), "solved": sum(match.get(i, 0) for i in v)} for k, v in sets.items()}


def think_stats(rows):
    t = [r["think_tokens"] for r in rows.values()]
    closed = [r for r in rows.values() if r["closed"]]
    q = statistics.quantiles(t, n=10)[-1] if len(t) >= 10 else max(t)
    return {"rows": len(rows), "think_mean": round(statistics.fmean(t), 1), "think_median": statistics.median(t),
            "think_p90": round(q), "think_max": max(t), "unclosed": len(rows) - len(closed),
            "parse_fail_closed": sum(not r["parse_ok"] for r in closed),
            "valid_pct": C.pct(sum(r["valid"] for r in rows.values()), len(rows)),
            "s_per_row": round(statistics.fmean(r["batch_seconds"] / r["batch_size"] for r in rows.values()), 1)}


def build(run=T.RUN):
    meta = json.loads((run / "rows.json").read_text(encoding="utf-8"))
    flags = {r["id"]: r for r in meta["rows"]}
    ids = list(flags)
    val = {c["id"]: c for c in jsonl("data/splits/val.jsonl")}
    arms = {a: arm_rows(run, a) for a in T.ARMS}
    match = {a: {i: int(r["match"]) for i, r in rows.items()} for a, rows in arms.items()}
    rep = {"rows": {"n": meta["n"], "misses": meta["fail"], "never solved": meta["hard"]}, "arms": {}, "readings": {},
           "references": {}, "verdicts": {}}
    for a, rows in arms.items():
        if not rows:
            continue
        done = [i for i in ids if i in rows]
        rep["arms"][a] = {"done": len(done), "counts": counts(match[a], flags, done),
                          "tiers": {t: [sum(match[a][i] for i in done if val[i]["tier"] == t),
                                        sum(val[i]["tier"] == t for i in done)] for t in TIERS}}
        if T.ARMS[a][0] == T.THINK_MODEL:
            rep["arms"][a]["thinking"] = think_stats(rows)
    for name, control, think in READINGS:
        both = [i for i in ids if i in arms[control] and i in arms[think]]
        if both:
            r = C.compare(match[control], match[think], both)
            rep["readings"][name] = {"control": control, "think": think, **r}
            rep["verdicts"][name] = VERDICT[r["verdict"]]
    for label, path in GREEDY.items():
        rows = jsonl(path)
        if rows:
            m = {r["id"]: int(bool(r["match"])) for r in rows}
            rep["references"][label] = counts(m, flags, ids)
    rows = jsonl(SAMPLED[1])
    if rows:
        m = {r["id"]: int(r["k"] > 0) for r in rows}
        rep["references"][SAMPLED[0]] = counts(m, flags, ids)
    # Rows only thinking solves: no control, greedy reference or sample solved them.
    seen = set()
    for a in ("nothink-trained", "nothink-retr8"):
        seen |= {i for i, v in match[a].items() if v}
    for path in list(GREEDY.values()) + [SAMPLED[1]]:
        for r in jsonl(path) or []:
            if r["id"] in flags and (r.get("match") or r.get("k", 0) > 0):
                seen.add(r["id"])
    thought = set()
    for a in ("think-trained", "think-retr8"):
        thought |= {i for i, v in match[a].items() if v}
    rep["only_thinking"] = sorted(thought - seen)
    rep["any_thinking"] = len(thought)
    return rep


MEANING = ("If either thinking arm helps, thinking is a candidate teacher for hard rows: a train-side labeling run "
           "is the owner's call. If neither does, thinking stays off for this teacher.")


def render(rep):
    L = ["# The 27B with thinking, on its val misses and the never-solved rows", "",
         "By `preregistration.md` beside this file; built by `scripts/think27b/report.py`. Rows: %d (%d the trained 27B "
         "misses, %d no 4B or 9B sample solves)." % (rep["rows"]["n"], rep["rows"]["misses"], rep["rows"]["never solved"]), "",
         "| arm | rows done | solved, all | of the misses | of the never solved | T1 / T2 / T3 |", "|---|---|---|---|---|---|"]
    for a, r in rep["arms"].items():
        c = r["counts"]
        L.append("| `%s` | %d | %d | %d / %d | %d / %d | %s |" % (
            a, r["done"], c["all"]["solved"], c["misses"]["solved"], c["misses"]["n"], c["never solved"]["solved"],
            c["never solved"]["n"], " / ".join("%d of %d" % tuple(r["tiers"][t]) for t in TIERS)))
    L += ["", "## Primary: each thinking arm against its control (same rows, same sampling; exact McNemar)", "",
          "| reading | rows | thinking − control | + / − | p | verdict |", "|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s: `%s` vs `%s` | %d | %+.1f [%+.1f, %+.1f] | +%d / −%d | %.3g | **%s** |" % (
            name, r["think"], r["control"], r["n"], r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"],
            r["p"], rep["verdicts"][name]))
    L += ["", "## The reasoning", "", "| arm | rows | think tokens mean / median / p90 / max | unclosed (no answer) | "
          "parse failures after `</think>` | valid | s per row |", "|---|---|---|---|---|---|---|"]
    for a, r in rep["arms"].items():
        s = r.get("thinking")
        if s:
            L.append("| `%s` | %d | %s / %s / %s / %s | %d | %d | %s%% | %s |" % (
                a, s["rows"], s["think_mean"], s["think_median"], s["think_p90"], s["think_max"], s["unclosed"],
                s["parse_fail_closed"], s["valid_pct"], s["s_per_row"]))
    L += ["", "## Non-thinking references on the same rows", "", "| reference | solved, all | of the misses | of the never solved |",
          "|---|---|---|---|"]
    for label, c in rep["references"].items():
        L.append("| %s | %d | %d / %d | %d / %d |" % (label, c["all"]["solved"], c["misses"]["solved"], c["misses"]["n"],
                                                   c["never solved"]["solved"], c["never solved"]["n"]))
    L += ["", "Rows solved by a thinking arm: %d. Of them, solved by no control, greedy reference or sample: **%d**%s." % (
        rep["any_thinking"], len(rep["only_thinking"]),
        (" (" + ", ".join("`%s`" % i for i in rep["only_thinking"][:12]) + (", …" if len(rep["only_thinking"]) > 12 else "") + ")")
        if rep["only_thinking"] else ""), "", MEANING]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (T.RUN / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (T.RUN / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
