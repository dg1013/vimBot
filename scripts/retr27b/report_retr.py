#!/usr/bin/env python3
"""The report on a 27B with 8 BM25 examples, with and without the 5-of-8 rule (CPU).

By runs/retr27b_2026-11-06/preregistration.md, from what retr27b.py and simulate.py wrote. --arm base is Part 2, the
untrained 27B; --arm trained is Part 3 (addendum 2), whose zero-shot side is the trained 27B's own runs. A request gets the answer
after the reply where the rule asks, and its §6 answer elsewhere. The estimate is the mean over 20 random draws of 8 of
its 16 samples per row, plus one run from the first 8; the trained 27B's rule is read the same way from its own 32, by
scripts/ask27b/report_ask.py's functions. The fewer-questions arms are fewer.py's.

    python scripts/retr27b/report_retr.py --run runs/retr27b_2026-11-06
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "scripts" / "ask27b", REPO / "scripts" / "serve27b", REPO / "scripts" / "retrieval", HERE):
    sys.path.insert(0, str(p))
import fewer as F                                                 # noqa: E402
import posthoc_ask as H                                           # noqa: E402
import report_ask as T                                            # noqa: E402
import retr27b as G                                               # noqa: E402
import retrieval as B                                             # noqa: E402

A, C, P, Q = T.A, T.C, T.P, T.Q
SWEEP = (2, 4, 6, 8, 16)                                          # descriptive: the same share of agreement at other k
BASE_RUN = T.RUNS / "retr27b_2026-11-06"                          # Part 2, which Part 3's reading B stands beside
NAMES = {"base": "the base 27B", "trained": "the trained 27B"}


def zero_exec(run, arm):
    """{id: the zero-shot answer's verdict}: this run's pass, or for the trained 27B its served run's."""
    files = [Q.SERVE27 / ("exec_%s.jsonl" % sp) if arm == "trained" else run / ("exec_k0_%s.jsonl" % sp) for sp in A.SPLITS]
    return {r["id"]: r for f in files for r in A.load_jsonl(f)}


def load_rows(run, cases, arm="base"):
    """{id: row}, in report_ask's shape, plus what fewer.py reads: the samples, their scores and verdicts."""
    views = {r["id"]: r for r in A.load_jsonl(run / "views.jsonl")}
    samples = {r["id"]: r for r in A.load_jsonl(run / "samples.jsonl")}
    judged = {r["id"]: r for r in A.load_jsonl(run / "judged.jsonl")}
    served = {r["id"]: r for sp in A.SPLITS for r in A.load_jsonl(run / ("exec_retr8_%s.jsonl" % sp))}
    zero = zero_exec(run, arm)
    empty = {r["id"] for r in A.load_jsonl(run / "question.jsonl") if not r["question"]["text"]}
    scores, labels = {}, {}
    for r in A.load_jsonl(run / "scores.jsonl"):
        scores.setdefault(r["id"], {})[(r["mode"], r["command"])] = r["score"]
    for r in A.load_jsonl(run / "states.jsonl"):
        labels.setdefault(r["id"], {})[(r["mode"], r["command"])] = int(r["state"] == "match")
    shape = {r["id"]: r["same_shape_neighbor"] for sp in A.SPLITS for r in A.load_jsonl(G.PLANS[sp])}
    rows = {}
    for sp in A.SPLITS:
        for c in cases[sp]:
            i = c["id"]
            if i not in views or i not in served or i not in zero or (i not in judged and i not in empty):
                continue
            j = judged.get(i)
            d = int(served[i]["match"])
            rows[i] = {"id": i, "split": sp, "tier": c["tier"], "results": views[i]["results"], "samples": samples[i]["samples"],
                       "direct": d, "zero": int(zero[i]["match"]), "dialogue": j is not None,
                       "real": int(j["real"]["match"]) if j else d, "placebo": int(j["placebo"]["match"]) if j else d,
                       "judged": j, "score": scores.get(i, {}), "label": labels.get(i, {}), "same_shape": shape[i]}
    return rows


def request_gpu(t):
    """posthoc_ask.part, for a run whose timing rows may hold no question at all: a dry run has 4."""
    turn = (t["question_s"]["mean"] + t["turn2_s"]["mean"]) if t["question_s"] and t["turn2_s"] else 0.0
    return t["models_wall_s"]["mean"] + t["asks"] * turn


def costs(timing, bm25_ms, acc):
    """GPU-seconds per request, one request at a time, as report_ask.costs and posthoc_ask.part have them. BM25 runs on
    the CPU, so it adds to the wait and not to the GPU."""
    extra = bm25_ms / 1000
    s = timing["summary"]
    d = s["direct"]
    out = {"one answer": dict(P.per_solved(d["total_s"]["mean"], acc["direct"]), p50=round(d["total_s"]["p50"] + extra, 3),
                              p95=round(d["total_s"]["p95"] + extra, 3), tokens=round(d["tokens"], 1))}
    for k in Q.TIMING_KS:
        t = s["k%d" % k]
        out["round trip at %d" % k] = dict(P.per_solved(request_gpu(t), acc[k]), p50=round(t["first_response_s"]["p50"] + extra, 3),
                                           p95=round(t["first_response_s"]["p95"] + extra, 3), tokens=round(t["tokens"], 1), asks=t["asks"])
    return out


def trained_side(cases):
    rows = F.trained_rows(cases)
    d8 = T.draws(rows)
    one = T.one_run(rows)
    return {"rows": rows, "direct": {i: r["direct"] for i, r in rows.items()}, "rule": {i: v[0] for i, v in d8.items()},
            "one": {i: v[0] for i, v in one.items()}}


def base_side(run, cases):
    """Part 2's one answer and its rule, for Part 3's reading B and tables, or None if Part 2 has not finished."""
    if not (run / "report.json").exists():
        return None
    rows = load_rows(run, cases)
    d8 = T.draws(rows)
    return {"direct": {i: r["direct"] for i, r in rows.items()}, "rule": {i: v[0] for i, v in d8.items()}}


def build(run, trained=None, arm="base", base=None):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    rows = load_rows(run, cases, arm)
    ids = T.split_ids(rows, cases)
    parts = {"untouched": ids["untouched"], "test": ids["test"]}
    reading = parts["untouched"] + parts["test"]
    keep = set(rows)
    t = trained or trained_side(cases)
    t_direct, t_rule, t_one = t["direct"], t["rule"], t["one"]

    direct = {i: r["direct"] for i, r in rows.items()}
    zero = {i: r["zero"] for i, r in rows.items()}
    real = {i: r["real"] for i, r in rows.items()}
    placebo = {i: r["placebo"] for i, r in rows.items()}
    d8 = T.draws(rows)
    rule = {i: v[0] for i, v in d8.items()}
    rate = {i: v[1] for i, v in d8.items()}
    one = T.one_run(rows)
    one_rule = {i: v[0] for i, v in one.items()}
    mean_rate = statistics.fmean(rate.values())
    at_random = {i: mean_rate * r["real"] + (1 - mean_rate) * r["direct"] for i, r in rows.items()}
    oracle = {i: max(r["real"], r["direct"]) for i, r in rows.items()}
    four = T.deployed_4b(keep)

    def means(values):
        return {name: P.mean(values, ids[name]) for name in ("val", "untouched", "test")}

    def asks_of(values):
        return {name: round(100 * statistics.fmean(values[i] for i in ids[name]), 1) if ids[name] else None
                for name in ("val", "untouched", "test")}

    who = NAMES[arm]
    systems = {"%s, zero-shot" % who: means(zero), "%s, 8 examples, one answer" % who: means(direct),
               "%s, 8 examples, the rule (mean of draws)" % who: means(rule), "%s, 8 examples, the rule, one run" % who: means(one_rule),
               "always ask": means(real), "placebo reply on every row": means(placebo),
               "asking at random, at the rule's rate": means(at_random), "asking only where it helps": means(oracle)}
    if arm == "base":
        systems.update({"the trained 27B, one answer": means(t_direct), "the trained 27B, the rule": means(t_rule)})
        readings = {"C: 8 examples vs the trained 27B, one answer each": P.reading(t_direct, direct, parts, t_direct, direct),
                    "R: with the rule vs without it, on 8 examples": P.reading(direct, rule, parts, direct, one_rule),
                    "W: 8 examples vs the trained 27B, the rule on both": P.reading(t_rule, rule, parts, t_one, one_rule)}
    else:
        systems["the trained 27B, zero-shot, the rule"] = means(t_rule)
        readings = {"E: 8 examples vs none, on the trained 27B, one answer each": P.reading(t_direct, direct, parts, t_direct, direct),
                    "R: with the rule vs without it, on 8 examples": P.reading(direct, rule, parts, direct, one_rule),
                    "W: 8 examples vs none, the rule on both": P.reading(t_rule, rule, parts, t_one, one_rule)}
        if base and all(i in base["direct"] for i in rows):                 # B needs Part 2 on every row read here
            systems.update({"the base 27B, 8 examples, one answer": means(base["direct"]), "the base 27B, 8 examples, the rule": means(base["rule"])})
            readings["B: the trained vs the base 27B, 8 examples each, one answer each"] = P.reading(base["direct"], direct, parts)
    systems["the 4B round trip, as deployed"] = means(four)
    readings["I: after the reply vs after the placebo, every row"] = P.reading(placebo, real, parts, placebo, real)
    descriptive = {"the rule on 8 examples vs the trained 27B's one answer": P.reading(t_direct, rule, parts, t_direct, one_rule),
                   "8 examples, one answer, vs the 4B round trip": P.reading(four, direct, parts, four, direct),
                   "the rule on 8 examples vs the 4B round trip": P.reading(four, rule, parts, four, one_rule),
                   "the rule vs asking at random at its rate": P.reading(at_random, rule, parts),
                   "the rule vs always asking": P.reading(real, rule, parts, real, one_rule)}
    gap = None
    if arm == "base":
        descriptive = dict({"8 examples vs zero-shot": P.reading(zero, direct, parts, zero, direct)}, **descriptive)
        point, lo, hi = B.gap_closed(zero, direct, t_direct, reading)
        gap = {"point": None if point is None else round(100 * point, 1), "lo": None if lo is None else round(100 * lo, 1),
               "hi": None if hi is None else round(100 * hi, 1)}
    rep = {"arm": arm, "rows": {name: len(v) for name, v in ids.items()}, "dialogues": sum(r["dialogue"] for r in rows.values()),
           "empty_questions": sum(not r["dialogue"] for r in rows.values()), "systems": systems,
           "asks": {"mean of draws": asks_of(rate), "one run": asks_of({i: v[1] for i, v in one.items()})},
           "gap_closed": gap, "readings": readings, "descriptive": descriptive,
           "outcomes": T.question_outcomes({i: rows[i] for i in reading}, one)}

    rep["sweep"] = {}
    for k in SWEEP:
        d = T.draws(rows, k)
        rep["sweep"][k] = {"pick": means({i: v[0] for i, v in d.items()}), "asks": asks_of({i: v[1] for i, v in d.items()})}

    tier = {i: r["tier"] for i, r in rows.items()}
    rep["tiers"] = {}
    for name in T.TIERS:
        sub = [i for i in reading if tier[i] == name]
        rep["tiers"][name] = {"n": len(sub), "direct": P.mean(direct, sub), "rule": P.mean(rule, sub), "trained": P.mean(t_direct, sub),
                              "asks": round(100 * statistics.fmean(rate[i] for i in sub), 1) if sub else None}
    rep["shape"] = {}
    for name, flag in (("a same-shape example was retrieved", True), ("none was", False)):
        sub = [i for i in reading if rows[i]["same_shape"] == flag]
        rep["shape"][name] = {"n": len(sub), "zero": P.mean(zero, sub), "direct": P.mean(direct, sub), "rule": P.mean(rule, sub),
                              "trained": P.mean(t_direct, sub)}

    judged = {i: r["judged"] for i, r in rows.items() if r["judged"]}
    flagged = {i for i, j in judged.items() if j["leak_flags"] or j["quotes_result"]}
    clean_parts = {name: [i for i in v if i not in flagged] for name, v in parts.items()}
    rep["simulator"] = {"replies": len(judged), "flagged": len(flagged), "flagged_reading_rows": len(flagged & set(reading)),
                        "forced": H.simulator_record(run)["forced"], "R without flagged rows": P.reading(direct, rule, clean_parts)}
    answers = [j["real"] for j in judged.values()]
    rep["turn2"] = {"answers": len(answers), "unparsed": sum(not a["parse_ok"] for a in answers),
                    "second_ask": sum(a["second_ask"] for a in answers), "think": sum(a["think"] for a in answers),
                    "valid_pct": round(100 * statistics.fmean(a["valid"] for a in answers), 1) if answers else None,
                    "tokens": round(statistics.fmean(a["tokens"] for a in answers), 1) if answers else None}
    questions = [r["question"] for r in A.load_jsonl(run / "question.jsonl")]
    rep["questions"] = {"rows": len(questions), "tokens": round(statistics.fmean(q["output_tokens"] for q in questions), 1) if questions else None,
                        "capped": sum(q["output_tokens"] >= Q.QUESTION_MAX_TOKENS for q in questions)}

    valid = {r["id"]: bool(r["valid"]) for sp in A.SPLITS for r in A.load_jsonl(run / ("exec_retr8_%s.jsonl" % sp))}
    recorded = {r["id"]: r for r in A.load_jsonl(G.RECORDED)} if arm == "base" else None
    greedy = {r["id"]: r for r in A.load_jsonl(run / "greedy_retr8_val.jsonl")}
    prompts = [r["prompt_tokens"] for sp in A.SPLITS for r in A.load_jsonl(run / ("greedy_retr8_%s.jsonl" % sp))]
    rep["greedy"] = {"valid_test": P.mean({i: int(v) for i, v in valid.items()}, ids["test"]), "prompt_tokens_mean": round(statistics.fmean(prompts), 1),
                     "served_identical_val": sum(A.same_answer(r["samples"][0], recorded[i]) for i, r in greedy.items()) if recorded else None,
                     "val_rows": len(greedy),
                     "tokens_val": round(statistics.fmean(sum(r["output_tokens"]) for r in greedy.values()), 1)}
    rep["checks"] = {"the 4B round trip, val and test": [means(four)["val"], means(four)["test"]],
                     "the trained 27B's rule, untouched and test": [means(t_rule)["untouched"], means(t_rule)["test"]]}
    for name in ("smoke", "rescore", "bm25"):
        path = run / (name + ".json")
        rep[name] = json.loads(path.read_text()) if path.exists() else None

    if (run / "timing.json").exists():
        timing = json.loads((run / "timing.json").read_text())
        acc = {k: P.mean({i: v[0] for i, v in T.draws(rows, k).items()}, ids["test"]) for k in Q.TIMING_KS}
        acc["direct"] = P.mean(direct, ids["test"])
        rep["cost"] = {"base": costs(timing, (rep["bm25"] or {}).get("query_ms_mean", 0.0), acc), "published": None}
        trained_t = json.loads((F.ASK_RUN / "timing.json").read_text())
        if {r["id"] for r in timing["rows"]["k8"]} == {r["id"] for r in trained_t["rows"]["k8"]}:
            published = json.loads((F.ASK_RUN / "posthoc.json").read_text())
            rep["cost"]["published"] = published.get("same_rows")
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f]" % (r["diff_pts"], r["ci95"][0], r["ci95"][1])


def one_text(r):
    o = r.get("one_run")
    return "%+.1f, +%d / −%d, p %.3g" % (o["diff_pts"], o["gained"], o["lost"], o["p"]) if o else "—"


def s(v):
    return "—" if v is None else "%s%%" % v


def render(rep):
    n = rep["rows"]
    L = ["# %s with 8 BM25 examples, with and without the 5-of-8 rule" % NAMES[rep["arm"]].replace("the", "The", 1), "",
         "By `preregistration.md` beside this file; built by `scripts/retr27b/report_retr.py`. Rows: val %d, test %d, %d with a dialogue, "
         "%d where the model wrote no question and the §6 answer stands. Means of %d draws of 8 of the samples on disk, unless marked. "
         "The examples are in every turn." % (n["val"], n["test"], rep["dialogues"], rep["empty_questions"], T.DRAWS), "",
         "## The systems", "", "| system | val | untouched | test |", "|---|---|---|---|"]
    for name, v in rep["systems"].items():
        L.append("| %s | %s | %s | %s |" % (name, s(v["val"]), s(v["untouched"]), s(v["test"])))
    a, g = rep["asks"], rep["gap_closed"]
    L += ["", "The rule asks on %s of val and %s of test (one run: %s and %s)." % (
        s(a["mean of draws"]["val"]), s(a["mean of draws"]["test"]), s(a["one run"]["val"]), s(a["one run"]["test"])), ""]
    if g:
        L += ["The examples close %s of the gap from zero-shot to the trained 27B on the reading rows [%s, %s]." % (s(g["point"]), s(g["lo"]), s(g["hi"])), ""]
    L += ["## Readings", "",
        "On the %d untouched val rows plus the %d test rows. Paired bootstrap, conjunction rule; the one run is the first 8 samples, with "
        "exact McNemar." % (n["untouched"], n["test"]), "",
        "| reading | pooled rows | untouched | test | verdict | one run |", "|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | **%s** | %s | %s | **%s** | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), r["verdict"], one_text(r)))
    L += ["", "Descriptive, no verdict:", "", "| comparison | pooled rows | untouched | test | one run |", "|---|---|---|---|---|"]
    for name, r in rep["descriptive"].items():
        L.append("| %s | %s | %s | %s | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), one_text(r)))
    o = rep["outcomes"]
    L += ["", "## What its questions did (one run, the reading rows)", "", "| | rows | §6 answer right | right after the reply |", "|---|---|---|---|",
          "| it asked | %d | %d | %d |" % (o["asked"], o["right_before"], o["right_after"]), "| it answered | %d | %d | — |" % (o["left"], o["right_left"]),
          "", "Questions fixed %d answers and broke %d." % (o["fixed"], o["broke"]), "",
          "## Fewer or more samples (descriptive)", "", "| k | val | untouched | test | asks on test |", "|---|---|---|---|---|"]
    for k, v in rep["sweep"].items():
        L.append("| %s | %s | %s | %s | %s |" % (k, s(v["pick"]["val"]), s(v["pick"]["untouched"]), s(v["pick"]["test"]), s(v["asks"]["test"])))
    L += ["", "## By tier (the reading rows)", "", "| tier | n | one answer | the rule | the trained 27B, one answer | asks |", "|---|---|---|---|---|---|"]
    for name, v in rep["tiers"].items():
        L.append("| %s | %d | %s | %s | %s | %s |" % (name, v["n"], s(v["direct"]), s(v["rule"]), s(v["trained"]), s(v["asks"])))
    L += ["", "## By the nearest help (the reading rows)", "", "| | n | zero-shot | one answer | the rule | the trained 27B, one answer |", "|---|---|---|---|---|---|"]
    for name, v in rep["shape"].items():
        L.append("| %s | %d | %s | %s | %s | %s |" % (name, v["n"], s(v["zero"]), s(v["direct"]), s(v["rule"]), s(v["trained"])))
    if rep.get("cost"):
        L += ["", "## What it costs, one request at a time", "",
              "On the 200 seeded val rows, at test accuracy, with BM25 in the wait. GPU-seconds per request: the first response on the model, plus, "
              "on the ask rate, the question and the answer after the reply.", "",
              "| system | test | asks | first response p50 / p95 | tokens | GPU-s / request | GPU-s / solved | $ / 1,000 solved |", "|---|---|---|---|---|---|---|---|"]
        for name, c in rep["cost"]["base"].items():
            L.append("| 8 examples, %s | %s%% | %s | %.2f / %.2f s | %s | %.3f | %.3f | $%.4f |" % (
                name, c["exec_match"], "%.0f%%" % (100 * c["asks"]) if "asks" in c else "—", c["p50"], c["p95"], c["tokens"],
                c["gpu_s_per_request"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
        for name, c in (rep["cost"]["published"] or {}).items():
            L.append("| %s | %s%% | %s | %.2f / %.2f s | %s | %.3f | %.3f | $%.4f |" % (
                name.replace("27B", "the trained 27B").replace("4B,", "the deployed 4B,"), c["exec_match"], "%.0f%%" % (100 * c["asks"]) if "asks" in c else "—", c["p50"], c["p95"], c["tokens"],
                c["gpu_s_per_request"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
    q, t2, sim, sm, re_, c, gr, bm = rep["questions"], rep["turn2"], rep["simulator"], rep["smoke"], rep["rescore"], rep["checks"], rep["greedy"], rep["bm25"]
    L += ["", "## Checks", ""]
    if sm:
        L.append("- Smoke: %d of %d answers identical to the recorded ones; %d prompts off the plan's token count; the %d longest prompts (up to %d tokens), %d of %d "
                 "identical under load." % (sm["identical"], sm["rows"], len(sm["prompt_token_mismatches"]), sm["longest_rows"], sm["longest_prompt"],
                                            sm["longest_identical"], sm["longest_answers"]))
    if gr["served_identical_val"] is not None:
        L.append("- Served against recorded greedy answers on val: %d of %d identical." % (gr["served_identical_val"], gr["val_rows"]))
    L += ["- Prompts %s tokens on average; %s%% valid on test." % (gr["prompt_tokens_mean"], gr["valid_test"]),
          "- The 4B round trip rebuilt: val %s%%, test %s%% (published 55.4%%, 59.7%%)." % tuple(c["the 4B round trip, val and test"]),
          "- The trained 27B's rule rebuilt: untouched %s%%, test %s%% (published 65.2%%, 70.0%%)." % tuple(c["the trained 27B's rule, untouched and test"]),
          "- Questions: %d rows, %s tokens on average, %d at the cap." % (q["rows"], q["tokens"], q["capped"]),
          "- After the reply: %d answers, %d unparsed, %d second questions, %d with `<think>`, %s%% valid, %s tokens." % (
              t2["answers"], t2["unparsed"], t2["second_ask"], t2["think"], t2["valid_pct"], t2["tokens"]),
          "- The leak screen flagged %d of %d replies (%d on the reading rows); %d replies hit the simulator's budget. Without the flagged rows, R reads %s, %s." % (
              sim["flagged"], sim["replies"], sim["flagged_reading_rows"], sim["forced"], fmt(sim["R without flagged rows"]["pooled"]), sim["R without flagged rows"]["verdict"])]
    if re_:
        L.append("- The verifier again on %d stored answers: mean |diff| %s, max %s." % (re_["rows"], re_["mean_abs"], re_["max_abs"]))
    if bm:
        L.append("- BM25 over %d train rows: built in %s s, %s ms a query." % (bm["train_rows"], bm["build_s"], bm["query_ms_mean"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(T.RUNS / "retr27b_2026-11-06"))
    ap.add_argument("--arm", choices=sorted(NAMES), default="base")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    base = base_side(BASE_RUN, {sp: A.cases(sp) for sp in A.SPLITS}) if args.arm == "trained" else None
    rep = build(run, arm=args.arm, base=base)
    (run / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    (run / "report.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
