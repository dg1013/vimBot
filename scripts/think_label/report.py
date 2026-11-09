#!/usr/bin/env python3
"""The labeling pass's report (CPU): every matched answer checked on the perturbed buffers, the
labels, thinking against its control (L), and how the labels differ from the golds. By
runs/think_label_2026-11-01/preregistration.md; writes labels.jsonl, report.md and report.json into
the run folder. Uses what is on disk, so a pass cut by its deadline reports the rows it finished.

A matched answer is a label only if it is the gold itself, or approach 1's equivalence check
passes: on two longer copies of the buffer (scripts/distill_targets/build_targets.perturb) it gives
the gold's buffer. A match on the original buffer alone is coincidental.

    python scripts/think_label/report.py
"""
import collections
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


L = load("think_label_label", HERE / "label.py")
D = load("distill_targets_build_targets", REPO / "scripts" / "distill_targets" / "build_targets.py")
C = load("distill_targets_compare", REPO / "scripts" / "distill_targets" / "compare.py")

CONTROL, THINK = "nothink-retr8", "think-retr8"
VERDICT = {"stronger": "helps", "weaker": "hurts", "tied": "tied"}
LABELS = ("gold", "equivalent")
TIERS = ("T1", "T2", "T3")
MEANING = {
    "helps": "thinking labels rows its control cannot: the thinking labels are the candidate data for a use "
             "proposal (re-targeted SFT rows, or guidance on the RL cliff), which is the owner's call.",
    "tied": "thinking adds no labels its control does not find: any use should take the control's labels, at "
            "about a twentieth of the cost.",
    "hurts": "thinking finds fewer labels than its control: any use should take the control's labels.",
}


def check(case, mode, command, decide=D.decide):
    """"gold" (the gold command itself), or approach 1's verdict on the answer: "equivalent",
    "diverges" (coincidental), "undetermined" (the gold errs on the perturbed buffers),
    "no_match_now", or "gold_fails_now"."""
    k = D.key(mode, command)
    if k == D.key(case["mode"], case["gold_command"]):
        return "gold"
    d = decide(case, [(k, 1, ["labeler"])])
    return d["tried"][0]["verdict"] if d["tried"] else d["reason"]


def load_arm(path):
    """{id: row}; a row written twice (a resumed batch) keeps its last copy."""
    if not path.exists():
        return {}
    return {r["id"]: r for r in L.load_jsonl(str(path))}


def quantiles(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {"mean": round(statistics.fmean(xs), 1), "median": xs[len(xs) // 2],
            "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1]}


def share(rows, pred=lambda r: True):
    sub = [r for r in rows if pred(r)]
    return {"n": len(sub), "labels": sum(r["label"] for r in sub)}


def summarize(rows, thinking):
    rs = list(rows.values())
    checks = collections.Counter(r["check"] for r in rs if r["match"])
    secs = sum(r["batch_seconds"] / r["batch_size"] for r in rs)
    out = {"attempted": len(rs), "matched": sum(r["match"] for r in rs), "labels": sum(r["label"] for r in rs),
           "checks": dict(checks), "valid": sum(r["valid"] for r in rs),
           "parse_failures": sum(not r["parse_ok"] for r in rs if r["closed"]),
           "seconds_per_row": round(secs / len(rs), 1) if rs else None,
           "by_source": {"cliff": share(rs, lambda r: r["in_cliff"]), "never": share(rs, lambda r: r["in_never"])},
           "by_tier": {t: share(rs, lambda r, t=t: r["tier"] == t) for t in TIERS},
           "by_category": {c: share(rs, lambda r, c=c: r["category"] == c)
                           for c in sorted({r["category"] for r in rs})}}
    if thinking:
        out["thinking"] = {"tokens": quantiles([r["think_tokens"] for r in rs]),
                           "unclosed": sum(not r["closed"] for r in rs),
                           "hit_cap": sum(r["hit_cap"] for r in rs),
                           "forced": sum(bool(r.get("forced_close")) for r in rs),
                           "forced_labels": sum(bool(r.get("forced_close")) and r["label"] for r in rs)}
    return out


def forms(rows, train, examples=10):
    """How the arm's labels differ from the golds."""
    lab = [r for r in rows.values() if r["label"]]
    diff = [r for r in lab if r["check"] == "equivalent"]
    switch = collections.Counter("%s->%s" % (train[r["id"]]["mode"], r["mode"]) for r in diff
                                 if r["mode"] != train[r["id"]]["mode"])
    return {"labels": len(lab), "gold": len(lab) - len(diff), "different": len(diff), "mode_switches": dict(switch),
            "chars_gold_vs_label": [round(statistics.fmean(len(train[r["id"]]["gold_command"]) for r in diff), 1),
                                    round(statistics.fmean(len(r["command"]) for r in diff), 1)] if diff else None,
            "examples": [{"id": r["id"], "instruction": train[r["id"]]["instruction"],
                          "gold": [train[r["id"]]["mode"], train[r["id"]]["gold_command"]],
                          "label": [r["mode"], r["command"]]} for r in diff[:examples]]}


def build(run=L.RUN, decide=D.decide):
    run = pathlib.Path(run)
    meta = json.loads((run / "rows.json").read_text(encoding="utf-8"))
    train = {c["id"]: c for c in L.load_jsonl(str(L.TRAIN))}
    arms = {a: load_arm(run / ("rows_%s.jsonl" % a)) for a in L.ARMS}
    arms = {a: rows for a, rows in arms.items() if rows}
    for rows in arms.values():
        for i, r in rows.items():
            r["check"] = check(train[i], r["mode"], r["command"], decide) if r["match"] else None
            r["label"] = bool(r["match"]) and r["check"] in LABELS
    rep = {k: meta[k] for k in ("n", "cliff", "never", "both", "plan")}
    rep.update(arms={a: summarize(rows, a == THINK) for a, rows in arms.items()}, readings={}, verdict=None,
               forms={a: forms(rows, train) for a, rows in arms.items()})
    if CONTROL in arms and THINK in arms:
        a, b = arms[CONTROL], arms[THINK]
        ids = sorted(set(a) & set(b))
        if ids:
            rep["readings"]["L"] = C.compare({i: int(a[i]["label"]) for i in ids},
                                             {i: int(b[i]["label"]) for i in ids}, ids)
            rep["readings"]["matches"] = C.compare({i: int(a[i]["match"]) for i in ids},
                                                   {i: int(b[i]["match"]) for i in ids}, ids)
            rep["verdict"] = VERDICT[rep["readings"]["L"]["verdict"]]
    either = {i for rows in arms.values() for i, r in rows.items() if r["label"]}
    rep["labeled_by_either"] = len(either)
    with (run / "labels.jsonl").open("w", encoding="utf-8") as fh:
        for row in meta["rows"]:
            c = train[row["id"]]
            out = {"id": row["id"], "in_cliff": row["in_cliff"], "in_never": row["in_never"], "tier": c["tier"],
                   "category": c["category"], "gold": [c["mode"], c["gold_command"]]}
            for a, rows in arms.items():
                r = rows.get(row["id"])
                if r is not None:
                    out[a] = {k: r[k] for k in ("mode", "command", "match", "check", "label")}
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f], +%d / −%d of %d, p %.3g" % (
        r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["n"], r["p"])


def frac(s):
    return "%d / %d" % (s["labels"], s["n"])


def render(rep):
    L_ = ["# The thinking 27B labels the hard train rows", "",
          "Built by `scripts/think_label/report.py` from the rows on disk; the rules are in `preregistration.md` "
          "beside this file. %d rows: %d the 4B never solves, %d no trained model solves, %d both. Retrieved "
          "examples: effective k %.2f, prompt tokens mean %.0f." % (
              rep["n"], rep["cliff"], rep["never"], rep["both"], rep["plan"]["effective_k_mean"],
              rep["plan"]["prompt_tokens_mean"]), "",
          "| arm | attempted | matched | **labels** (gold / equivalent) | coincidental | cliff | never solved | "
          "T1 / T2 / T3 | s / row |", "|---|---|---|---|---|---|---|---|---|"]
    for a, s in rep["arms"].items():
        ch = s["checks"]
        L_.append("| `%s` | %d | %d | **%d** (%d / %d) | %d | %s | %s | %s | %s |" % (
            a, s["attempted"], s["matched"], s["labels"], ch.get("gold", 0), ch.get("equivalent", 0),
            ch.get("diverges", 0), frac(s["by_source"]["cliff"]), frac(s["by_source"]["never"]),
            " / ".join(frac(s["by_tier"][t]) for t in TIERS), s["seconds_per_row"]))
    if "L" in rep["readings"]:
        L_ += ["", "## L: thinking against its control (exact McNemar, paired bootstrap 95% interval)", "",
               "- **Labels:** %s → **%s**" % (fmt(rep["readings"]["L"]), rep["verdict"]),
               "- Matches before the check: %s" % fmt(rep["readings"]["matches"]),
               "- Rows labeled by either arm: %d of %d." % (rep["labeled_by_either"], rep["n"]), "",
               "**By the pre-registered rule:** %s" % MEANING[rep["verdict"]]]
    th = rep["arms"].get(THINK, {}).get("thinking")
    if th:
        L_ += ["", "## The reasoning", "",
               "- Thinking tokens: %s" % (", ".join("%s %s" % kv for kv in th["tokens"].items()) if th["tokens"] else "—"),
               "- Unclosed at the cap: %d (at the cap: %d); parse failures after `</think>`: %d" % (
                   th["unclosed"], th["hit_cap"], rep["arms"][THINK]["parse_failures"]),
               "- Finished by the loop breaker (`scripts/think_loop`): %d, of which %d are labels" % (
                   th.get("forced", 0), th.get("forced_labels", 0))]
    for a, f in rep["forms"].items():
        if not f["labels"]:
            continue
        L_ += ["", "## The labels' forms: `%s`" % a, "",
               "- %d labels: %d are the gold itself, %d differ from it and pass the check." % (
                   f["labels"], f["gold"], f["different"])]
        if f["different"]:
            L_.append("- Mode switches among those: %s; characters, gold vs label: %s vs %s." % (
                ", ".join("%s %d" % kv for kv in f["mode_switches"].items()) or "none",
                f["chars_gold_vs_label"][0], f["chars_gold_vs_label"][1]))
            L_ += ["", "| id | instruction | gold | label |", "|---|---|---|---|"]
            for e in f["examples"]:
                L_.append("| `%s` | %s | `%s: %s` | `%s: %s` |" % (
                    e["id"], e["instruction"].replace("|", "\\|")[:90], e["gold"][0],
                    e["gold"][1].replace("|", "\\|"), e["label"][0], e["label"][1].replace("|", "\\|")))
    return "\n".join(L_) + "\n"


def main():
    rep = build()
    (L.RUN / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    text = render(rep)
    (L.RUN / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
