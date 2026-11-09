#!/usr/bin/env python3
"""Every 27B method against the others and against the best 4B (CPU only; descriptive).

- Val, all 545 rows, with the 295 never used for checkpoint selection primary: the trained 27B
  (QLoRA SFT on the distilled targets), the untrained 27B with 8 retrieved examples (greedy, and the
  plurality of 8 samples at T = 1) and zero-shot, and the served 4B. The 2026-11-01 readings keep the
  served 4B as the best 4B: the KD and GRPO-on-distill 4Bs tie it on val.
- The hardest val rows: thinking ran only there, so each thinking arm is set beside every other
  method on the rows it reached at 8,192 tokens (`scripts/think_long`).
- Test v2 (429 rows): the trained 27B and the served 4B, the only two measured there.

The served 4B's val log predates the 2026-09-25 split rewrite. The prompted-27B report re-scored it
against the current val and no verdict moved; this script checks that audit before using the log.

    python scripts/compare27b/compare.py      # -> runs/compare27b_2026-11-01/report.{md,json}
"""
import importlib.util
import itertools
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = REPO / "runs" / "compare27b_2026-11-01"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# One copy of each: compare() (exact McNemar, paired bootstrap), untouched_ids(), pct(); the
# prompted 27B's plurality rule; the thinking run's arm orders, prefixes and chains.
C = load("distill_targets_compare", REPO / "scripts" / "distill_targets" / "compare.py")
P = load("teacher27b_report", REPO / "scripts" / "teacher27b" / "report.py")
L = load("think_long_report", REPO / "scripts" / "think_long" / "report.py")
E = L.E

TEACHER = "runs/teacher27b_2026-10-30"
GREEDY = {      # val, greedy, the results-table path (transformers generate, one request at a time)
    "27b-trained": ("runs/qwen3.5-27b-sft-distill", "27B, QLoRA SFT on the distilled targets"),
    "27b-retr8": (TEACHER + "/27b-retr8", "27B untrained, 8 retrieved examples"),
    "27b-k0": (TEACHER + "/27b-k0", "27B untrained, zero-shot"),
    "4b-served": ("runs/qwen3.5-4b-grpo-gated-leg2", "served 4B, SFT + GRPO"),
}
PROBES = {      # val, 8 samples at T = 1
    "27b-retr8": (TEACHER + "/probe_val_27b-retr8.jsonl", "27B untrained, 8 retrieved examples"),
    "4b-served": ("runs/sample_rule_2026-10-09/probe_val.jsonl", "served 4B"),
}
TEST = {        # test v2, greedy
    "27b-trained": "runs/qwen3.5-27b-sft-distill",
    "4b-served": "runs/remeasure_2026-09-25/v2/qwen3.5-4b-grpo-gated__fla-kernels",
}
MATRIX = ("27b-trained", "27b-retr8", "27b-retr8@8", "27b-k0", "4b-served")   # paired on val
SHORT = {"27b-trained": "27B trained", "27b-retr8": "27B + 8 ex.", "27b-retr8@8": "27B + 8 ex., vote of 8",
         "27b-k0": "27B zero-shot", "4b-served": "served 4B", "4b-served@8": "served 4B, vote of 8"}
TIERS = ("T1", "T2", "T3")


def jsonl(path):
    return C.load_jsonl(str(REPO / path))


def greedy_method(folder, split="val"):
    """id -> 0/1 match and valid from a run's exec log; tokens and seconds from its completions."""
    ex = {r["id"]: r for r in jsonl("%s/exec_%s.jsonl" % (folder, split))}
    path = REPO / folder / ("completions_%s.jsonl" % split)
    comp = {r["id"]: r for r in C.load_jsonl(str(path))} if path.exists() else {}
    return {"match": {i: int(bool(r["match"])) for i, r in ex.items()},
            "valid": {i: int(bool(r["valid"])) for i, r in ex.items()},
            "tokens": {i: r["output_tokens"] for i, r in comp.items() if "output_tokens" in r},
            "seconds": {i: r["latency_s"] for i, r in comp.items() if "latency_s" in r}}


def probe_method(path):
    """(the plurality of the G samples as a method, any of the G as a ceiling) from a val probe."""
    rows = {r["id"]: r for r in jsonl(path)}
    top = {i: max(P.commands(r), key=lambda c: c["n"]) if P.commands(r) else None for i, r in rows.items()}
    plurality = {"match": {i: P.plurality_right(r) for i, r in rows.items()},
                 "valid": {i: int(t is not None and t["state"] in ("match", "valid_wrong")) for i, t in top.items()},
                 "tokens": {i: r["G"] * r["tokens_mean"] for i, r in rows.items()}, "seconds": {}}
    return plurality, {i: int(r["k"] > 0) for i, r in rows.items()}


def summary(m, ids, tier):
    n, solved = len(ids), sum(m["match"][i] for i in ids)
    tokens = [m["tokens"][i] for i in ids if i in m.get("tokens", {})]
    secs = [m["seconds"][i] for i in ids if i in m.get("seconds", {})]
    s = {"n": n, "solved": solved, "pct": C.pct(solved, n),
         "valid_pct": C.pct(sum(m["valid"][i] for i in ids), n) if "valid" in m else None,
         "tokens_mean": round(statistics.fmean(tokens), 1) if len(tokens) == n else None,
         "seconds_mean": round(statistics.fmean(secs), 2) if len(secs) == n else None,
         "seconds_p50": round(statistics.median(secs), 2) if len(secs) == n else None,
         "tiers": {t: [sum(m["match"][i] for i in ids if tier[i] == t), sum(tier[i] == t for i in ids)]
                   for t in TIERS}}
    # GPU seconds per solved row on this card, one request at a time (§11's cost, before a $ rate).
    s["seconds_per_solved"] = round(sum(secs) / solved, 2) if len(secs) == n and solved else None
    return s


def cell(r):
    return {"diff_pts": r["diff_pts"], "ci95": r["ci95"], "gained": r["gained"], "lost": r["lost"],
            "p": r["p"], "verdict": r["verdict"]}


def val_section():
    audit = json.loads((REPO / TEACHER / "report.json").read_text(encoding="utf-8"))["audit"]["4b-served"]
    if audit["moved"] or audit["logged_match"] != audit["rescored_match"]:
        sys.exit("the served 4B's val log no longer matches its re-scoring; re-score it first")
    val = C.load_jsonl(str(C.VAL))
    ids, clean = [c["id"] for c in val], C.untouched_ids(val)
    tier = {c["id"]: c["tier"] for c in val}
    methods, labels, ceilings = {}, {}, {}
    for k, (folder, label) in GREEDY.items():
        methods[k], labels[k] = greedy_method(folder), label + ", greedy"
    for k, (path, label) in PROBES.items():
        methods[k + "@8"], ceilings[k + " any of 8"] = probe_method(path)
        labels[k + "@8"] = label + ", plurality of 8 samples (T = 1)"
    for k, m in methods.items():
        if set(m["match"]) != set(ids):
            sys.exit("%s does not cover val (%d of %d ids)" % (k, len(set(m["match"]) & set(ids)), len(ids)))
    rep = {"n": len(ids), "n_untouched": len(clean), "labels": labels, "methods": {}, "matrix": {}, "ceilings": {}}
    for k, m in methods.items():
        rep["methods"][k] = {"untouched": summary(m, clean, tier), "all": summary(m, ids, tier)}
    for part, rows in (("untouched", clean), ("all", ids)):
        rep["matrix"][part] = {"%s|%s" % (a, b): cell(C.compare(methods[b]["match"], methods[a]["match"], rows))
                               for a, b in itertools.permutations(MATRIX, 2)}
    # Ceilings: no rule picks these answers; they bound what a chooser over the methods could reach.
    greedy27 = {i: max(methods[k]["match"][i] for k in ("27b-trained", "27b-retr8", "27b-k0")) for i in ids}
    every27 = {i: max(greedy27[i], ceilings["27b-retr8 any of 8"][i]) for i in ids}
    served = methods["4b-served"]["match"]
    sets = {"27B any of 8 samples (8 examples, T = 1)": ceilings["27b-retr8 any of 8"],
            "served 4B any of 8 samples (T = 1)": ceilings["4b-served any of 8"],
            "the three 27B greedy methods, any": greedy27,
            "every 27B method, any (greedy and the 8 samples)": every27,
            "27B trained or served 4B, greedy": {i: max(methods["27b-trained"]["match"][i], served[i]) for i in ids}}
    for name, s in sets.items():
        rep["ceilings"][name] = {"untouched": C.pct(sum(s[i] for i in clean), len(clean)), "all": C.pct(sum(s[i] for i in ids), len(ids))}
    rep["served_only"] = sorted(i for i in ids if served[i] and not every27[i])
    rep["trained_27b_vs_served"] = {"both": sum(methods["27b-trained"]["match"][i] and served[i] for i in ids),
                                    "only_27b": sum(methods["27b-trained"]["match"][i] and not served[i] for i in ids),
                                    "only_4b": sum(served[i] and not methods["27b-trained"]["match"][i] for i in ids)}
    return rep, methods, ceilings


def thinking_section(methods, ceilings):
    """Each thinking arm at 8,192 tokens beside every other method, on the rows it reached."""
    byid = {arm: {r["id"]: r for r in rows} for arm, rows in E.first_rows(E.FIRST_RUN).items()}
    orders = E.arm_orders(E.FIRST_RUN)
    stages = {cap: {arm: E.stage_rows(E.RUN, cap, arm) for arm in E.THINK_ARMS} for cap in E.CAPS}
    controls = {a: {r["id"]: int(bool(r["match"])) for r in E.load_jsonl(str(E.FIRST_RUN / ("rows_%s.jsonl" % a)))}
                for a in L.CONTROL.values()}
    rep = {}
    for arm in E.THINK_ARMS:
        pairs = L.prefix(orders[arm], byid, stages, arm, E.CAPS[0])
        ids = [i for i, _ in pairs]
        think = {i: int(bool(o["match"])) for i, o in pairs}
        cols = {"no thinking, sampled at T 0.6 (its control)": controls[L.CONTROL[arm]]}
        for k in ("27b-trained", "27b-retr8", "27b-retr8@8", "27b-k0", "4b-served", "4b-served@8"):
            cols[k] = methods[k]["match"]
        cols["27B any of 8 samples (ceiling)"] = ceilings["27b-retr8 any of 8"]
        cols["served 4B any of 8 samples (ceiling)"] = ceilings["4b-served any of 8"]
        chains = {i: E.chain(i, byid, stages, arm, E.CAPS[1]) for i in ids}
        secs = [sum(r.get("seconds", r.get("batch_seconds", 0.0)) for r in chains[i]) for i in ids]
        toks = [sum(r["new_tokens"] for r in chains[i]) for i in ids]
        rep[arm] = {"rows": len(ids), "solved": sum(think.values()), "in_fail": sum(bool(o["in_fail"]) for _, o in pairs),
                    "in_hard": sum(bool(o["in_hard"]) for _, o in pairs),
                    "tokens_median": statistics.median(toks), "minutes_mean": round(statistics.fmean(secs) / 60, 1),
                    "minutes_per_solved": round(sum(secs) / 60 / max(1, sum(think.values())), 1),
                    "columns": {name: {"solved": sum(col[i] for i in ids),
                                       "thinking_vs": cell(C.compare(col, think, ids))} for name, col in cols.items()}}
    return rep


def test_section():
    test = C.load_jsonl(str(REPO / "data" / "splits" / "test.jsonl"))
    ids = [c["id"] for c in test]
    tier = {c["id"]: c["tier"] for c in test}
    m = {k: greedy_method(f, "test") for k, f in TEST.items()}
    for k, v in m.items():
        if set(v["match"]) != set(ids):
            sys.exit("%s does not cover test v2 (%d of %d ids)" % (k, len(set(v["match"]) & set(ids)), len(ids)))
    return {"n": len(ids), "methods": {k: summary(v, ids, tier) for k, v in m.items()},
            "reading": cell(C.compare(m["4b-served"]["match"], m["27b-trained"]["match"], ids))}


def build():
    val, methods, ceilings = val_section()
    return {"val": val, "thinking": thinking_section(methods, ceilings), "test": test_section()}


def fmt(r):
    return "%+.1f [%+.1f, %+.1f], +%d / −%d, p %.2g" % (r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["p"])


def num(x, unit=""):
    return "—" if x is None else "%s%s" % (x, unit)


def render(rep):
    v = rep["val"]
    L_ = ["# Every 27B method against the others and the best 4B", "",
          "Built by `scripts/compare27b/compare.py` from recorded runs; descriptive, no decision depends on it. "
          "Exact McNemar and paired bootstrap 95% intervals throughout.", "",
          "## Val (%d rows; the %d never used for checkpoint selection are primary)" % (v["n"], v["n_untouched"]), "",
          "| method | untouched | all val | valid | T1 / T2 / T3 (all val) | output tokens | s per answer (mean / p50) | GPU s per solved row |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in v["methods"].items():
        u, a = s["untouched"], s["all"]
        L_.append("| %s | **%s%%** | %s%% | %s%% | %s | %s | %s / %s | %s |" % (
            v["labels"][k], u["pct"], a["pct"], a["valid_pct"],
            " / ".join("%.0f%%" % (100.0 * x / n) for x, n in (a["tiers"][t] for t in TIERS)),
            num(a["tokens_mean"]), num(a["seconds_mean"]), num(a["seconds_p50"]), num(a["seconds_per_solved"])))
    L_ += ["", "Seconds are the results-table path's (transformers `generate`, one request, NF4 base for the 27B), "
           "not served latency. The vote-of-8 rows' tokens are all 8 samples; their time was not logged per row.", "",
           "### Paired, on the untouched rows: row minus column, points (p)", "",
           "| | " + " | ".join(SHORT[b] for b in MATRIX) + " |", "|---|" + "---|" * len(MATRIX)]
    for a in MATRIX:
        cells = []
        for b in MATRIX:
            if a == b:
                cells.append("")
                continue
            r = v["matrix"]["untouched"]["%s|%s" % (a, b)]
            cells.append(("**%+.1f** (%.2g)" if r["p"] < 0.05 else "%+.1f (%.2g)") % (r["diff_pts"], r["p"]))
        L_.append("| %s | %s |" % (SHORT[a], " | ".join(cells)))
    L_ += ["", "On all val: " + "; ".join("%s vs %s %s" % (SHORT[a], SHORT[b], fmt(v["matrix"]["all"]["%s|%s" % (a, b)]))
                                         for a, b in itertools.combinations(MATRIX, 2)) + ".", "",
           "### Ceilings (no rule picks these answers)", "", "| any of | untouched | all val |", "|---|---|---|"]
    for name, c in v["ceilings"].items():
        L_.append("| %s | %s%% | %s%% |" % (name, c["untouched"], c["all"]))
    t = v["trained_27b_vs_served"]
    L_ += ["", "The trained 27B and the served 4B, greedy, on all val: both right on %d rows, only the 27B on %d, only "
           "the 4B on %d. Rows the served 4B solves and no 27B method does (greedy or any of the 8 samples): %d." % (
               t["both"], t["only_27b"], t["only_4b"], len(v["served_only"])), ""]
    L_ += ["## Thinking, on the hardest val rows (8,192 tokens)", "",
           "Thinking ran only on 213 val rows: the 208 the trained 27B answers wrong greedily, and the 70 no 4B or 9B "
           "sample solved in 128 (65 in both). Each thinking arm reached the start of its own random order, so its "
           "rows differ from the other arm's. Every method is scored on the same rows as the arm; the trained 27B's "
           "greedy answer is wrong on almost all of them by construction.", ""]
    for arm, x in rep["thinking"].items():
        L_ += ["### %s (%s): %d rows" % (L.NAME[arm], arm, x["rows"]), "",
               "Thinking solves **%d of %d**: median %d generated tokens, %.1f minutes per row and %.1f per solved row "
               "on this card." % (x["solved"], x["rows"], x["tokens_median"], x["minutes_mean"], x["minutes_per_solved"]), "",
               "| method on the same rows | solves | thinking minus it |", "|---|---|---|"]
        for name, c in x["columns"].items():
            label = v["labels"].get(name, name)
            L_.append("| %s | %d | %s |" % (label, c["solved"], fmt(c["thinking_vs"])))
        L_.append("")
    te = rep["test"]
    L_ += ["## Test v2 (%d rows; reporting only)" % te["n"], "",
           "| method | exec-match | valid | T1 / T2 / T3 |", "|---|---|---|---|"]
    for k, s in te["methods"].items():
        L_.append("| %s | **%s%%** | %s%% | %s |" % (v["labels"][k], s["pct"], s["valid_pct"],
                                                 " / ".join("%.0f%%" % (100.0 * a / n) for a, n in (s["tiers"][t] for t in TIERS))))
    L_ += ["", "The trained 27B against the served 4B: %s. No other 27B method was run on test v2." % fmt(te["reading"])]
    return "\n".join(L_) + "\n"


def main():
    rep = build()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
