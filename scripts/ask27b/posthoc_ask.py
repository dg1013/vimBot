#!/usr/bin/env python3
"""Not pre-registered: what carries the rule's gain on the 27B, and what it costs on one footing (CPU).

report.md read S helps and I tied: the rule beats the 27B's one answer, and on every row the reply is
worth no more than a placebo. The pre-registration took that to mean the gain is the second pass, not
the reply, but I is read over every row and the rule only asks on some. This asks it on those rows.

Two arms only, fixed before either was computed, on the same draws as the report:

    the rule, placebo     same rows, same question, the answer after the placebo. No user turn.
    the rule, top result  same rows, no second turn: the command behind the most common result of the
                          8 samples, the earliest on a tie, the §6 answer if none gives a result.

Also, because the report cannot show them: how many questions the rule puts per answer it gains; the
live round trip against this estimate on the 200 timing rows; and the 4B and the 27B on those same 200
rows, one request at a time, which no earlier table has.

Added after the two arms had been read, because it bears on what runs today: the same reply-against-
placebo comparison on the deployed 4B, from its published run; and the 27B's one answer against that
round trip, which the owner's choice rests on and the report does not pair. No other arm was computed.

Nothing here changes a reading in report.md, and a claim from it needs its own pre-registration. These
rows are reviewed to remove under-specified instructions, so they cannot show what a question gives
on those. Neither arm may run in real use as §1 stands: the answer shown must be the §6 answer or the
one after the user's reply, and picking among the samples by result is reserved for the owner
(DECISIONS.md 2026-10-13). They are analysis numbers.

    python scripts/ask27b/posthoc_ask.py --run runs/ask27b_2026-11-06
"""
import argparse
import collections
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ask27b as Q                                                # noqa: E402
import report_ask as T                                            # noqa: E402

A, C, P = T.A, T.C, T.P
RUN_4B = T.RUNS / "serve" / "timing_val.json"      # the 4B's round trip, 2026-10-13, the same 200 rows
ACC_4B = {"direct": 54.3, 4: 58.7, 8: 59.7}        # test v2, RESULTS.md, as published


def top_sample(results, idxs):
    """Index of the sample behind the most common result, the earliest on a tie. None if no result."""
    counts = collections.Counter(results[n] for n in idxs if results[n] is not None)
    if not counts:
        return None
    best = max(counts.values())
    return next(n for n in idxs if results[n] is not None and counts[results[n]] == best)


def arms(row, idxs):
    """One request with these samples -> what each way of handling an ask returns, 0 or 1."""
    if not (row["dialogue"] and T.asks([row["results"][n] for n in idxs])):
        return {"rule": row["direct"], "placebo": row["direct"], "top": row["direct"], "asked": 0}
    n = top_sample(row["results"], idxs)
    return {"rule": row["real"], "placebo": row["placebo"], "top": row["direct"] if n is None else row["right"][n], "asked": 1}


def draws(rows, k=T.K, n=T.DRAWS):
    """{id: {arm: share of right draws}}, on the report's draws: its streams, in its order."""
    out = {}
    for i, r in rows.items():
        rng = random.Random("ask@%d|%s" % (k, i))
        got = [arms(r, rng.sample(range(len(r["results"])), k)) for _ in range(n)]
        out[i] = {name: sum(g[name] for g in got) / n for name in got[0]}
    return out


def add_sample_verdicts(rows):
    """row["right"]: for each sample, 1 if it solves the row. Stored per distinct command."""
    state = {(r["id"], r["mode"], r["command"]): r["state"] for r in A.load_jsonl(Q.SERVE27 / "judged.jsonl")}
    for s in A.load_jsonl(Q.SERVE27 / "samples.jsonl"):
        if s["id"] in rows:
            rows[s["id"]]["right"] = [int(x is not None and state[(s["id"], x[0], x[1])] == "match") for x in s["samples"]]


def deployed_4b_arms(ids):
    """The 4B round trip as published, one run: ({id: the rule}, {id: the rule with its placebo}).
    Its own question and placebo answer, from the run report_ask.deployed_4b rebuilds."""
    rule, placebo = {}, {}
    for sp, folder in (("test", "clarify_2026-10-07/test"), ("val", "clarify_2026-10-06/val")):
        view = json.loads((T.RUNS / "sample_rule_2026-10-09" / sp / "report.json").read_text(encoding="utf-8"))
        for r in A.load_jsonl(T.RUNS / folder / "verdicts_served.jsonl"):
            i = r["id"]
            if i in ids:
                ask = view[i]["top_share"] * view[i]["n"] <= 5 + 1e-9
                rule[i] = int(bool(r["real"]["match"] if ask else r["direct"]["match"]))
                placebo[i] = int(bool(r["placebo"]["match"] if ask else r["direct"]["match"]))
    return rule, placebo


def simulator_record(run):
    """The instrument on this run: how many replies, and how many hit the budget (forced to answer)."""
    rows = A.load_jsonl(run / "replies.jsonl")
    return {"replies": len(rows), "forced": sum(bool(r.get("forced")) for r in rows)}


def outcomes(rows, one):
    """On one run, the rows it asks on: right before, and right after each way of answering."""
    asked = [r for i, r in rows.items() if one[i]["asked"]]
    left = [r for i, r in rows.items() if not one[i]["asked"]]
    arm = lambda name: sum(one[r["id"]][name] for r in asked)              # noqa: E731
    return {"asked": len(asked), "left": len(left), "right_before": sum(r["direct"] for r in asked),
            "after_reply": arm("rule"), "after_placebo": arm("placebo"), "top_result": arm("top"),
            "right_left": sum(r["direct"] for r in left),
            "reply_vs_placebo": {"both": sum(r["real"] and r["placebo"] for r in asked),
                                 "only_reply": sum(r["real"] and not r["placebo"] for r in asked),
                                 "only_placebo": sum(r["placebo"] and not r["real"] for r in asked),
                                 "neither": sum(not r["real"] and not r["placebo"] for r in asked)},
            "same_command": sum(r["judged"]["real"]["command"] == r["judged"]["placebo"]["command"] for r in asked)}


def burden(rows, ids, rate, rule):
    """Questions put per net right answer gained, over these rows, on the mean of draws."""
    asks = sum(rate[i] for i in ids)
    net = sum(rule[i] - rows[i]["direct"] for i in ids)
    return {"rows": len(ids), "questions": round(asks, 1), "net_right": round(net, 1),
            "questions_per_net_right": round(asks / net, 1) if net > 0 else None}


def live_vs_offline(timing, rate):
    """The round trip as it ran on the 200 rows against the stored samples' ask rate on them. Live
    and stored draws are independent, so they can agree only as often as two draws of a row do."""
    live = {r["id"]: r["type"] == "question" for r in timing["rows"]["k8"]}
    p = [rate[i] for i in live]
    return {"rows": len(live), "live_asks": round(100 * statistics.fmean(live.values()), 1),
            "stored_asks": round(100 * statistics.fmean(p), 1),
            "agreement": round(100 * statistics.fmean((rate[i] if a else 1 - rate[i]) for i, a in live.items()), 1),
            "agreement_of_two_draws": round(100 * statistics.fmean(x * x + (1 - x) * (1 - x) for x in p), 1)}


def part(t):
    """GPU-seconds per request: the models, plus, on its ask rate, the question and the turn after."""
    return t["models_wall_s"]["mean"] + t["asks"] * (t["question_s"]["mean"] + t["turn2_s"]["mean"])


def same_rows(timing, four, acc27):
    """Both models on the same 200 rows, one request at a time. {system: figures}."""
    ids4 = {r["id"] for r in four["rows"]["k8"]}
    if ids4 != {r["id"] for r in timing["rows"]["k8"]}:
        raise SystemExit("the 4B and the 27B were not timed on the same rows")
    out = {}
    for model, summary, acc in (("4B", four["summary"], ACC_4B), ("27B", timing["summary"], acc27)):
        d = summary["direct"]
        out["%s, one answer" % model] = dict(P.per_solved(d["total_s"]["mean"], acc["direct"]),
                                             p50=d["total_s"]["p50"], p95=d["total_s"]["p95"], tokens=round(d["tokens"], 1))
        for k in Q.TIMING_KS:
            t = summary["k%d" % k]
            out["%s, round trip at %d" % (model, k)] = dict(P.per_solved(part(t), acc[k]), p50=t["first_response_s"]["p50"],
                                                            p95=t["first_response_s"]["p95"], tokens=round(t["tokens"], 1), asks=t["asks"])
    return out


def valid_rate(rows, ids, rate):
    """Valid answers under the rule on these rows, on the mean of draws."""
    served = {r["id"]: r for sp in A.SPLITS for r in A.load_jsonl(Q.SERVE27 / ("exec_%s.jsonl" % sp))}
    return round(100 * statistics.fmean(rate[i] * bool(rows[i]["judged"]["real"]["valid"]) + (1 - rate[i]) * bool(served[i]["valid"])
                                        for i in ids), 1)


def build(run):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    rows = T.load_rows(run, cases)
    add_sample_verdicts(rows)
    ids = T.split_ids(rows, cases)
    parts = {"untouched": ids["untouched"], "test": ids["test"]}
    reading_rows = parts["untouched"] + parts["test"]
    direct = {i: r["direct"] for i, r in rows.items()}

    d8 = draws(rows)
    one = {i: arms(r, range(T.K)) for i, r in rows.items()}
    mean_of = {name: {i: v[name] for i, v in d8.items()} for name in ("rule", "placebo", "top", "asked")}
    one_of = {name: {i: v[name] for i, v in one.items()} for name in ("rule", "placebo", "top")}

    same = T.draws(rows)                                           # the report's own estimator
    if any(abs(same[i][0] - mean_of["rule"][i]) > 1e-9 for i in rows):
        raise SystemExit("the arms are not on the report's draws")

    def means(values):
        return {name: P.mean(values, ids[name]) for name in ("val", "untouched", "test")}

    rule4, placebo4 = deployed_4b_arms(set(rows))
    if rule4 != T.deployed_4b(set(rows)):
        raise SystemExit("the 4B here is not the 4B the report rebuilds")
    timing = json.loads((run / "timing.json").read_text())
    four = json.loads(RUN_4B.read_text())
    acc27 = {"direct": P.mean(direct, ids["test"])}
    acc27.update({k: P.mean({i: v[0] for i, v in T.draws(rows, k).items()}, ids["test"]) for k in Q.TIMING_KS})
    return {
        "rows": {name: len(v) for name, v in ids.items()},
        "systems": {"the 27B's §6 answer": means(direct), "the rule, the reply": means(mean_of["rule"]),
                    "the rule, the placebo": means(mean_of["placebo"]), "the rule, the top result": means(mean_of["top"])},
        "readings": {"the reply vs the placebo, on the rows it asks": P.reading(mean_of["placebo"], mean_of["rule"], parts, one_of["placebo"], one_of["rule"]),
                     "the rule with the placebo vs the §6 answer": P.reading(direct, mean_of["placebo"], parts, direct, one_of["placebo"]),
                     "the rule with the top result vs the §6 answer": P.reading(direct, mean_of["top"], parts, direct, one_of["top"]),
                     "the rule vs the rule with the top result": P.reading(mean_of["top"], mean_of["rule"], parts, one_of["top"], one_of["rule"])},
        "deployed_4b": {"systems": {"the rule, the reply": means(rule4), "the rule, the placebo": means(placebo4)},
                        "reply_vs_placebo": P.reading(placebo4, rule4, parts, placebo4, rule4),
                        "the 27B's one answer vs this round trip": P.reading(rule4, direct, parts, rule4, direct)},
        "simulator": simulator_record(run),
        "asked_rows": outcomes({i: rows[i] for i in reading_rows}, one),
        "burden": {"the reading rows": burden(rows, reading_rows, mean_of["asked"], mean_of["rule"]),
                   "test": burden(rows, ids["test"], mean_of["asked"], mean_of["rule"]),
                   "untouched": burden(rows, ids["untouched"], mean_of["asked"], mean_of["rule"])},
        "live": live_vs_offline(timing, mean_of["asked"]),
        "same_rows": same_rows(timing, four, acc27),
        "valid_test": valid_rate(rows, ids["test"], mean_of["asked"]),
    }


def render(rep):
    L = ["# Post hoc: what carries the rule's gain, and the cost on one footing", "",
         "**Not pre-registered.** Read after `report.md`. Nothing here changes a reading there, and a claim from it "
         "needs its own pre-registration. Built by `scripts/ask27b/posthoc_ask.py`; its two arms were fixed before either "
         "was computed. These rows were reviewed to remove under-specified instructions, so they cannot show what a "
         "question gives on those.", "",
         "## The rows the rule asks on: reply, placebo, or no second turn", "",
         "Same rows and draws as the report. Only what happens where it asks differs.", "",
         "| system | val | untouched | test |", "|---|---|---|---|"]
    for name, v in rep["systems"].items():
        L.append("| %s | %s%% | %s%% | %s%% |" % (name, v["val"], v["untouched"], v["test"]))
    L += ["", "| comparison | pooled rows | untouched | test | the rule's word | one run |", "|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | **%s** | %s | %s | %s | %s |" % (name, T.fmt(r["pooled"]), T.fmt(r["untouched"]), T.fmt(r["test"]), r["verdict"], T.one_text(r)))
    a, s = rep["asked_rows"], rep["asked_rows"]["reply_vs_placebo"]
    L += ["", "## What happens on the rows it asks (one run, the reading rows)", "",
          "| rows it asks | §6 answer right | after the reply | after the placebo | top result |", "|---|---|---|---|---|",
          "| %d | %d | %d | %d | %d |" % (a["asked"], a["right_before"], a["after_reply"], a["after_placebo"], a["top_result"]), "",
          "Reply and placebo give the identical command on %d of the %d. Both right %d, only the reply %d, only the placebo %d, "
          "neither %d. On the %d rows it does not ask, the §6 answer is right on %d." % (
              a["same_command"], a["asked"], s["both"], s["only_reply"], s["only_placebo"], s["neither"], a["left"], a["right_left"]),
          "", "## The same comparison on the deployed 4B (one run)", "",
          "Added after the arms above had been read. The 4B's own questions and placebo answers, on the rows its rule asks.", "",
          "| system | val | untouched | test |", "|---|---|---|---|"]
    for name, v in rep["deployed_4b"]["systems"].items():
        L.append("| %s | %s%% | %s%% | %s%% |" % (name, v["val"], v["untouched"], v["test"]))
    r = rep["deployed_4b"]["reply_vs_placebo"]
    L += ["", "The reply vs the placebo, on the rows it asks: **%s**; untouched %s, test %s; the rule's word: %s; %s." % (
        T.fmt(r["pooled"]), T.fmt(r["untouched"]), T.fmt(r["test"]), r["verdict"], T.one_text(r)),
          ""]
    r = rep["deployed_4b"]["the 27B's one answer vs this round trip"]
    L += ["The 27B's one answer vs this round trip: **%s**; untouched %s, test %s; the rule's word: %s; %s." % (
        T.fmt(r["pooled"]), T.fmt(r["untouched"]), T.fmt(r["test"]), r["verdict"], T.one_text(r)),
          "", "## Questions per answer gained (mean of draws)", "", "| rows | n | questions | net right answers | questions per net right |",
          "|---|---|---|---|---|"]
    for name, b in rep["burden"].items():
        L.append("| %s | %d | %s | %s | %s |" % (name, b["rows"], b["questions"], b["net_right"], b["questions_per_net_right"]))
    r = rep["live"]
    L += ["", "## The live round trip against this estimate", "",
          "On the %d timing rows the round trip itself asked on %s%%; the stored samples ask on %s%% of them. A live decision matches "
          "a stored draw %s%% of the time, where two draws of the same row would match %s%%." % (
              r["rows"], r["live_asks"], r["stored_asks"], r["agreement"], r["agreement_of_two_draws"]),
          "", "## The 4B and the 27B on the same 200 rows, one request at a time", "",
          "The 4B timed 2026-10-13, the 27B 2026-11-06, at test v2's accuracy. First response is what the user waits for. GPU-seconds "
          "add the question and the turn after the reply on the ask rate. The earlier tables took the head of val and, for the §6 "
          "answers, the best concurrency; these are not comparable to them.", "",
          "| system | test | first response p50 / p95 | tokens | GPU-s / request | GPU-s / solved | $ / 1,000 solved |", "|---|---|---|---|---|---|---|"]
    for name, c in rep["same_rows"].items():
        L.append("| %s | %s%% | %.2f / %.2f s | %.0f | %.3f | %.3f | $%.4f |" % (
            name, c["exec_match"], c["p50"], c["p95"], c["tokens"], c["gpu_s_per_request"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
    sim = rep["simulator"]
    L += ["", "Valid answers under the rule on test v2: %s%%. The simulated user hit its thinking budget on %d of %d replies." % (
        rep["valid_test"], sim["forced"], sim["replies"]), ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(T.REPO / "runs" / "ask27b_2026-11-06"))
    run = pathlib.Path(ap.parse_args().run)
    rep = build(run)
    (run / "posthoc.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    text = render(rep)
    (run / "posthoc.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
