#!/usr/bin/env python3
"""The report on the 5-of-8 ask rule on the trained 27B (CPU).

By runs/ask27b_2026-11-06/preregistration.md, from what ask27b.py and simulate.py wrote.

A request gets the answer after the reply where the rule asks, and the §6 answer elsewhere. Which rows
it asks on depends on the samples, so the estimate is the mean over 20 random draws of 8 of the 32
samples on disk (a row's value is its share of right draws), plus one run from the first 8. The
question, the reply and the answer after it are one fixed dialogue per row. Differences are paired
over rows with a bootstrap and the conjunction rule, as in scripts/serve27b/pool27b.py, whose helpers
are used unchanged.

    python scripts/ask27b/report_ask.py --run runs/ask27b_2026-11-06
"""
import argparse
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts" / "serve27b"))
import ask27b as Q                                                # noqa: E402
import pool27b as P                                               # noqa: E402
from k_sweep import asks                                          # noqa: E402

A, C = P.A, P.C
RUNS = REPO / "runs"
K = 8                                  # the rule's samples
SWEEP = (2, 4, 6, 8, 16, 32)           # descriptive: the same share of agreement at other k
DRAWS = 20
TIERS = P.TIERS


def load_rows(run, cases):
    """{id: row}. direct, real, placebo are 0/1; results holds one result per sample. A row whose
    question came back empty has no dialogue, and real use answers it directly."""
    views = {r["id"]: r for r in A.load_jsonl(run / "views.jsonl")}
    judged = {r["id"]: r for r in A.load_jsonl(run / "judged.jsonl")}
    served = {r["id"]: r for sp in A.SPLITS for r in A.load_jsonl(Q.SERVE27 / ("exec_%s.jsonl" % sp))}
    empty = {r["id"] for r in A.load_jsonl(run / "question.jsonl") if not r["question"]["text"]}
    rows = {}
    for sp in A.SPLITS:
        for c in cases[sp]:
            i = c["id"]
            if i not in views or i not in served or (i not in judged and i not in empty):
                continue
            j = judged.get(i)
            d = int(served[i]["match"])
            rows[i] = {"id": i, "split": sp, "tier": c["tier"], "results": views[i]["results"], "direct": d,
                       "dialogue": j is not None, "real": int(j["real"]["match"]) if j else d,
                       "placebo": int(j["placebo"]["match"]) if j else d, "judged": j}
    return rows


def value(row, idxs):
    """One request with these samples -> (right, asked)."""
    ask = row["dialogue"] and asks([row["results"][n] for n in idxs])
    return (row["real"] if ask else row["direct"]), int(ask)


def one_run(rows, k=K):
    return {i: value(r, range(k)) for i, r in rows.items()}


def draws(rows, k=K, n=DRAWS):
    """{id: (share of right draws, share of draws that ask)}."""
    out = {}
    for i, r in rows.items():
        rng = random.Random("ask@%d|%s" % (k, i))
        got = [value(r, rng.sample(range(len(r["results"])), k)) for _ in range(n)]
        out[i] = (sum(g[0] for g in got) / n, sum(g[1] for g in got) / n)
    return out


def split_ids(rows, cases):
    untouched = [i for i in C.untouched_ids(cases["val"]) if i in rows]
    clean = set(untouched)
    return {"untouched": untouched, "test": [c["id"] for c in cases["test"] if c["id"] in rows],
            "val": [c["id"] for c in cases["val"] if c["id"] in rows],
            "choosing": [c["id"] for c in cases["val"] if c["id"] in rows and c["id"] not in clean]}


def deployed_4b(ids):
    """The 4B round trip as published (compare_9b.py's construction): one run, its 8 samples."""
    out = {}
    for sp, folder in (("test", "clarify_2026-10-07/test"), ("val", "clarify_2026-10-06/val")):
        view = json.loads((RUNS / "sample_rule_2026-10-09" / sp / "report.json").read_text(encoding="utf-8"))
        for r in A.load_jsonl(RUNS / folder / "verdicts_served.jsonl"):
            i = r["id"]
            if i in ids:
                ask = view[i]["top_share"] * view[i]["n"] <= 5 + 1e-9
                out[i] = int(bool(r["real"]["match"] if ask else r["direct"]["match"]))
    return out


def pooled_pick(cases, keep):
    """The pooled pick at 4 + 4, by the serve27b report's construction: (mean of draws, one run)."""
    base, one = {}, {}
    for sp in A.SPLITS:
        ids = [c["id"] for c in cases[sp]]
        rows = P.load_rows(Q.SERVE27, sp, ids)[0]
        base.update(P.base_shares(rows, list(rows)))
        one.update({i: P.first_k_arm(r, P.base_k())[0] for i, r in rows.items()})
    return {i: v for i, v in base.items() if i in keep}, {i: v for i, v in one.items() if i in keep}


def question_outcomes(rows, run_values):
    """On one run: what the rule's asks did to the §6 answer, and how it reads the rows it left."""
    asked = [r for i, r in rows.items() if run_values[i][1]]
    left = [r for i, r in rows.items() if not run_values[i][1]]
    return {"asked": len(asked), "left": len(left),
            "fixed": sum(r["real"] and not r["direct"] for r in asked), "broke": sum(r["direct"] and not r["real"] for r in asked),
            "right_before": sum(r["direct"] for r in asked), "right_after": sum(r["real"] for r in asked),
            "right_left": sum(r["direct"] for r in left)}


def costs(timing, acc, cost27):
    """GPU-seconds per request, one request at a time: what the first response took on the models,
    plus, on its ask rate, the question and the answer after the reply. The reply is not GPU time."""
    s = timing["summary"]
    rows = {}
    for k in Q.TIMING_KS:
        t = s["k%d" % k]
        turn = (t["question_s"]["mean"] + t["turn2_s"]["mean"]) if t["turn2_s"] else 0.0
        gpu = t["models_wall_s"]["mean"] + t["asks"] * turn
        rows[k] = dict(P.per_solved(gpu, acc[k]), asks=t["asks"], first_p50=t["first_response_s"]["p50"],
                       first_p95=t["first_response_s"]["p95"], answers_p50=t["when_it_answers_s"]["p50"] if t["when_it_answers_s"] else None,
                       asks_p50=t["when_it_asks_s"]["p50"] if t["when_it_asks_s"] else None, turn2_p50=t["turn2_s"]["p50"] if t["turn2_s"] else None,
                       sandbox_p50=t["sandbox_s"]["p50"], tokens=round(t["tokens"], 1), same_question=t["same_question"])
    d = s["direct"]
    rows["direct"] = dict(P.per_solved(d["total_s"]["mean"], acc["direct"]), first_p50=d["total_s"]["p50"], first_p95=d["total_s"]["p95"],
                          tokens=round(d["tokens"], 1))
    rows["published"] = cost27
    return rows


def build(run):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    rows = load_rows(run, cases)
    ids = split_ids(rows, cases)
    parts = {"untouched": ids["untouched"], "test": ids["test"]}
    reading_rows = parts["untouched"] + parts["test"]
    tier = {i: r["tier"] for i, r in rows.items()}

    direct = {i: r["direct"] for i, r in rows.items()}
    real = {i: r["real"] for i, r in rows.items()}
    placebo = {i: r["placebo"] for i, r in rows.items()}
    d8 = draws(rows)
    rule = {i: v[0] for i, v in d8.items()}
    rate = {i: v[1] for i, v in d8.items()}
    one = one_run(rows)
    one_rule = {i: v[0] for i, v in one.items()}
    mean_rate = statistics.fmean(rate.values())
    at_random = {i: mean_rate * r["real"] + (1 - mean_rate) * r["direct"] for i, r in rows.items()}
    oracle = {i: max(r["real"], r["direct"]) for i, r in rows.items()}
    four = deployed_4b(set(rows))
    base, one_base = pooled_pick(cases, set(rows))

    def means(values):
        return {name: P.mean(values, ids[name]) for name in ("val", "untouched", "test")}

    def asks_of(values):
        return {name: round(100 * statistics.fmean(values[i] for i in ids[name]), 1) if ids[name] else None
                for name in ("val", "untouched", "test")}

    rep = {"rows": {name: len(v) for name, v in ids.items()}, "dialogues": sum(r["dialogue"] for r in rows.values()),
           "empty_questions": sum(not r["dialogue"] for r in rows.values()),
           "systems": {"the 27B's §6 answer": means(direct), "the rule, 5 of 8 (mean of draws)": means(rule),
                       "the rule, one run": means(one_rule), "always ask": means(real), "placebo reply on every row": means(placebo),
                       "asking at random, at the rule's rate": means(at_random), "asking only where it helps": means(oracle),
                       "the 4B round trip, as deployed": means(four), "the pooled pick, 4 + 4": means(base)},
           "asks": {"mean of draws": asks_of(rate), "one run": asks_of({i: v[1] for i, v in one.items()})},
           "readings": {"S: the rule vs the 27B's §6 answer": P.reading(direct, rule, parts, direct, one_rule),
                        "I: after the reply vs after the placebo, every row": P.reading(placebo, real, parts, placebo, real),
                        "D: the rule vs the 4B round trip, as deployed": P.reading(four, rule, parts, four, one_rule)},
           "descriptive": {"the rule vs asking at random at its rate": P.reading(at_random, rule, parts),
                           "the rule vs always asking": P.reading(real, rule, parts, real, one_rule),
                           "the rule vs the pooled pick, 4 + 4": P.reading(base, rule, parts, one_base, one_rule)},
           "outcomes": {"all rows": question_outcomes(rows, one),
                        "reading rows": question_outcomes({i: rows[i] for i in reading_rows}, one)}}

    sweep = {}
    for k in SWEEP:
        d = draws(rows, k)
        sweep[k] = {"pick": means({i: v[0] for i, v in d.items()}), "asks": asks_of({i: v[1] for i, v in d.items()})}
    rep["sweep"] = sweep

    rep["tiers"] = {}
    for t in TIERS:
        sub = [i for i in reading_rows if tier[i] == t]
        rep["tiers"][t] = {"n": len(sub), "direct": P.mean(direct, sub), "rule": P.mean(rule, sub),
                           "asks": round(100 * statistics.fmean(rate[i] for i in sub), 1) if sub else None}

    judged = {i: r["judged"] for i, r in rows.items() if r["judged"]}
    flagged = {i for i, j in judged.items() if j["leak_flags"] or j["quotes_result"]}
    clean_parts = {name: [i for i in v if i not in flagged] for name, v in parts.items()}
    rep["simulator"] = {"replies": len(judged), "flagged": len(flagged), "flagged_reading_rows": len(flagged & set(reading_rows)),
                        "S without flagged rows": P.reading(direct, rule, clean_parts)}
    answers = [j["real"] for j in judged.values()]
    rep["turn2"] = {"answers": len(answers), "unparsed": sum(not a["parse_ok"] for a in answers),
                    "second_ask": sum(a["second_ask"] for a in answers), "think": sum(a["think"] for a in answers),
                    "valid_pct": round(100 * statistics.fmean(a["valid"] for a in answers), 1) if answers else None,
                    "tokens": round(statistics.fmean(a["tokens"] for a in answers), 1) if answers else None}
    questions = [r["question"] for r in A.load_jsonl(run / "question.jsonl")]
    rep["questions"] = {"rows": len(questions), "tokens": round(statistics.fmean(q["output_tokens"] for q in questions), 1) if questions else None,
                        "capped": sum(q["output_tokens"] >= Q.QUESTION_MAX_TOKENS for q in questions)}

    rep["check_4b"] = {name: means(four)[name] for name in ("val", "test")}
    if (run / "timing.json").exists():
        timing = json.loads((run / "timing.json").read_text())
        acc = {k: P.mean({i: v[0] for i, v in draws(rows, k).items()}, ids["test"]) for k in Q.TIMING_KS}
        acc["direct"] = P.mean(direct, ids["test"])
        served = json.loads((Q.SERVE27 / "report.json").read_text()).get("cost", {})
        rep["cost"] = costs(timing, acc, served)
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f]" % (r["diff_pts"], r["ci95"][0], r["ci95"][1])


def one_text(r):
    o = r.get("one_run")
    return "%+.1f, +%d / −%d, p %.3g" % (o["diff_pts"], o["gained"], o["lost"], o["p"]) if o else "—"


def s(v):
    return "—" if v is None else "%s%%" % v


def secs(v):
    return "—" if v is None else "%.2f s" % v


def render(rep):
    n = rep["rows"]
    L = ["# The 5-of-8 ask rule on the trained 27B", "",
         "By `preregistration.md` beside this file; built by `scripts/ask27b/report_ask.py`. Rows: val %d, test %d, %d with a "
         "dialogue, %d where the model wrote no question and real use answers directly. Means of %d draws of 8 of the 32 "
         "samples on disk, unless marked." % (n["val"], n["test"], rep["dialogues"], rep["empty_questions"], DRAWS), "",
         "## The systems", "", "| system | val | untouched | test |", "|---|---|---|---|"]
    for name, v in rep["systems"].items():
        L.append("| %s | %s | %s | %s |" % (name, s(v["val"]), s(v["untouched"]), s(v["test"])))
    a = rep["asks"]
    L += ["", "The rule asks on %s of val and %s of test (one run: %s and %s)." % (
        s(a["mean of draws"]["val"]), s(a["mean of draws"]["test"]), s(a["one run"]["val"]), s(a["one run"]["test"])), "",
        "## Readings", "",
        "On the %d untouched val rows plus the %d test rows, which chose nothing. Paired bootstrap, conjunction rule; the one "
        "run is the first 8 samples, with exact McNemar." % (n["untouched"], n["test"]), "",
        "| reading | pooled rows | untouched | test | verdict | one run |", "|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | **%s** | %s | %s | **%s** | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), r["verdict"], one_text(r)))
    L += ["", "Descriptive, no verdict:", "", "| comparison | pooled rows | untouched | test | one run |", "|---|---|---|---|---|"]
    for name, r in rep["descriptive"].items():
        L.append("| %s | %s | %s | %s | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), one_text(r)))
    o = rep["outcomes"]["reading rows"]
    L += ["", "## What its questions did (one run, the reading rows)", "",
          "| | rows | §6 answer right | right after the reply |", "|---|---|---|---|",
          "| it asked | %d | %d | %d |" % (o["asked"], o["right_before"], o["right_after"]),
          "| it answered | %d | %d | — |" % (o["left"], o["right_left"]), "",
          "Questions fixed %d answers and broke %d." % (o["fixed"], o["broke"]), "",
          "## Fewer or more samples (descriptive)", "",
          "The same share of agreement at other k: ask when the top result holds at most ⌊5/8 k⌋ of k.", "",
          "| k | val | untouched | test | asks on test |", "|---|---|---|---|---|"]
    for k, v in rep["sweep"].items():
        L.append("| %s | %s | %s | %s | %s |" % (k, s(v["pick"]["val"]), s(v["pick"]["untouched"]), s(v["pick"]["test"]), s(v["asks"]["test"])))
    if "cost" in rep:
        c = rep["cost"]
        L += ["", "## What it costs, one request at a time", "",
              "On the 200 seeded val rows the 4B's round trip was timed on, not the head of val. GPU-seconds per request: the "
              "first response on the models, plus, on the ask rate, the question and the answer after the reply. The first two "
              "lines cost at test accuracy, from this run's timing; pooled and 4B rows are not re-timed on these rows.", "",
              "| system | test | asks | first response p50 / p95 | when it answers / asks (p50) | turn 2 p50 | tokens | GPU-s / request | GPU-s / solved | $ / 1,000 solved |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        d = c["direct"]
        L.append("| the 27B's §6 answer | %s%% | — | %.2f / %.2f s | — | — | %.1f | %.3f | %.3f | $%.4f |" % (
            d["exec_match"], d["first_p50"], d["first_p95"], d["tokens"], d["gpu_s_per_request"], d["gpu_s_per_solved"], d["usd_per_1000_solved"]))
        for k in Q.TIMING_KS:
            r = c[k]
            L.append("| round trip, %d samples | %s%% | %.0f%% | %.2f / %.2f s | %s / %s | %s | %.0f | %.3f | %.3f | $%.4f |" % (
                k, r["exec_match"], 100 * r["asks"], r["first_p50"], r["first_p95"], secs(r["answers_p50"]), secs(r["asks_p50"]), secs(r["turn2_p50"]),
                r["tokens"], r["gpu_s_per_request"], r["gpu_s_per_solved"], r["usd_per_1000_solved"]))
    L += ["", "## By tier (the reading rows)", "", "| tier | n | §6 answer | the rule | asks |", "|---|---|---|---|---|"]
    for t, v in rep["tiers"].items():
        L.append("| %s | %d | %s | %s | %s |" % (t, v["n"], s(v["direct"]), s(v["rule"]), s(v["asks"])))
    sim, t2, q = rep["simulator"], rep["turn2"], rep["questions"]
    r = sim["S without flagged rows"]
    L += ["", "## Checks", "",
          "- The 4B round trip rebuilt: val %s, test %s (published 55.4%%, 59.7%%)." % (s(rep["check_4b"]["val"]), s(rep["check_4b"]["test"])),
          "- Questions: %d rows, %s tokens on average, %d at the cap." % (q["rows"], q["tokens"], q["capped"]),
          "- After the reply: %d answers, %d unparsed, %d second questions, %d with `<think>`, %s valid, %s tokens." % (
              t2["answers"], t2["unparsed"], t2["second_ask"], t2["think"], s(t2["valid_pct"]), t2["tokens"]),
          "- The leak screen flagged %d of %d replies (%d on the reading rows). Without those rows, S reads %s, %s." % (
              sim["flagged"], sim["replies"], sim["flagged_reading_rows"], fmt(r["pooled"]), r["verdict"])]
    if "cost" in rep:
        L.append("- Greedy questions in the timing pass identical to the stored ones: %.0f%% at k = 8." % (100 * rep["cost"][8]["same_question"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(REPO / "runs" / "ask27b_2026-11-06"))
    run = pathlib.Path(ap.parse_args().run)
    rep = build(run)
    (run / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    text = render(rep)
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
