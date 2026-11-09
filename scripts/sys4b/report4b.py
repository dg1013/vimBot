#!/usr/bin/env python3
"""The reports on the trained 27B's systems on the 4B (CPU).

By runs/sys4b_2026-11-08/preregistration.md, from what sys4b.py and simulate.py wrote. Every estimator is the 27B
reports' own: scripts/ask27b/report_ask.py's draws, one run and question outcomes, scripts/retr27b/fewer.py's T and V,
scripts/retr27b/report_retr.py's loader and costs, scripts/serve27b/pool27b.py's bootstrap and conjunction rule. Only the
folders and the names are the 4B's. The 27B's matching systems are read beside them, on the same rows, by the same code.

    python scripts/sys4b/report4b.py --arm a          # -> runs/sys4b_2026-11-08/a/report.md, report.json
    python scripts/sys4b/report4b.py --arm b          # needs arm a; reading B needs arm c
    python scripts/sys4b/report4b.py --arm c          # needs arm a
    python scripts/sys4b/report4b.py --arm summary    # the 4B beside the 27B, from the arms' report.json
"""
import argparse
import contextlib
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sys4b as S                                                 # noqa: E402  the 27B's stages, pointed at the 4B
for p in ("ask27b", "retr27b", "serve27b", "retrieval"):
    sys.path.insert(0, str(S.REPO / "scripts" / p))
import fewer as F                                                 # noqa: E402
import posthoc_ask as H                                           # noqa: E402
import report_ask as RA                                           # noqa: E402
import report_retr as RR                                          # noqa: E402

T, Q, A = S.T, S.Q, S.A
P, B = RA.P, RR.B
SWEEP = {"a": (2, 4, 6, 8, 16, 32), "b": (2, 4, 6, 8, 16), "c": (2, 4, 6, 8, 16)}
RETR27 = RR.BASE_RUN                                              # the base 27B's run; its trained/ is Part 3
NAMES = {"a": "the served 4B", "b": "the served 4B with 8 examples", "c": "the base 4B with 8 examples"}
REBOUND = [(A, "MODEL27"), (A, "NAME27"), (A, "URL27"), (A, "URL4"), (A, "endpoint27"), (Q, "SERVE27"), (Q, "LABEL"),
           (T, "BASE_NAME"), (T, "SERVED"), (T, "RECORDED"), (T, "RECORDED_ZERO"), (T, "OTHER_HOURS"), (T, "ARM")]
ORIGINAL = {(m, n): getattr(m, n) for m, n in REBOUND}           # the 27B's, before any use_4b


@contextlib.contextmanager
def the_27b():
    """The 27B's names back for a while: its own rows are read by its own loaders."""
    now = {(m, n): getattr(m, n) for m, n in REBOUND}
    for (m, n), v in ORIGINAL.items():
        setattr(m, n, v)
    try:
        yield
    finally:
        for (m, n), v in now.items():
            setattr(m, n, v)


# ------------------------------------------------------------ loading

def judge_a(run, cases):
    """Arm a's §6 answers and every distinct sample through the box as it stands today: exec logs beside its greedy
    answers, a verdict beside each sample. The 27B's served pass had both from its own judge stage."""
    from evaluate import score, write_exec_log
    from score import execute
    timeout = Q.settings_timeout()
    out = run / "served"
    for sp in A.SPLITS:
        path = out / ("exec_%s.jsonl" % sp)
        if not path.exists():
            greedy = A.load_jsonl(out / ("greedy_%s.jsonl" % sp))
            have = {r["id"] for r in greedy}
            rows = [c for c in cases[sp] if c["id"] in have]
            write_exec_log(str(path), rows, score(rows, T.greedy_preds(greedy), timeout=timeout))
    states = S.SERVED_DIR / "states.jsonl"
    if states.exists():
        return
    samples = {r["id"]: r for r in A.load_jsonl(S.SERVED_DIR / "samples.jsonl")}
    got = []
    for sp in A.SPLITS:
        cs = [c for c in cases[sp] if c["id"] in samples]
        answers = {c["id"]: {"c%d" % n: {"parse_ok": True, "mode": m, "command": cmd}
                             for n, (m, cmd) in enumerate(Q.distinct(samples[c["id"]]["samples"]))} for c in cs}
        executed = execute(cs, answers, timeout=timeout, workers=8)
        for c in cs:
            for name, a in answers[c["id"]].items():
                e = executed[(c["id"], name)]
                got.append({"id": c["id"], "split": sp, "mode": a["mode"], "command": a["command"],
                            "state": "match" if e and e["match"] else "no"})
    S.write_rows(states, got)


def rows_a(run, cases):
    """Arm a's rows in report_ask's shape, plus what fewer.py reads: the samples, their scores and verdicts."""
    S.use_4b("a")
    Q.SERVE27 = run / "served"                                    # its exec logs; a dry run has its own
    judge_a(run, cases)
    rows = RA.load_rows(run, cases)
    samples = {r["id"]: r["samples"] for r in A.load_jsonl(S.SERVED_DIR / "samples.jsonl")}
    scores, labels = {}, {}
    for r in A.load_jsonl(S.SERVED_DIR / "scores27.jsonl"):
        scores.setdefault(r["id"], {})[(r["mode"], r["command"])] = r["score"]
    for r in A.load_jsonl(S.SERVED_DIR / "states.jsonl"):
        labels.setdefault(r["id"], {})[(r["mode"], r["command"])] = int(r["state"] == "match")
    for i, r in rows.items():
        r.update(samples=samples[i], score=scores.get(i, {}), label=labels.get(i, {}))
    return rows


def rows_bc(run, cases, arm):
    """Arms b and c in report_retr's shape. Arm b's zero-shot side is arm a's served answers, arm c's its own pass."""
    S.use_4b(arm)
    Q.SERVE27 = S.SERVED_DIR
    return RR.load_rows(run, cases, S.RETR_ARM[arm])


def side(rows):
    """What other readings need of a system: its §6 answer, its rule (mean of draws) and one run."""
    d8, one = RA.draws(rows), RA.one_run(rows)
    return {"direct": {i: r["direct"] for i, r in rows.items()}, "rule": {i: v[0] for i, v in d8.items()},
            "rate": {i: v[1] for i, v in d8.items()}, "one": {i: v[0] for i, v in one.items()}}


def fewer_values(rows, ids):
    """fewer.py's V and its chosen T row by row, on its own draws, τ and choice of t: build() keeps only the summaries."""
    choosing = ids["choosing"] or list(rows)
    many = {i: [F.view(r, s) for s in F.draw_sets(r)] for i, r in rows.items()}
    first = {i: F.view(r, range(F.K)) for i, r in rows.items()}
    tau = F.tau_from([v[1] for i in choosing for v in many[i]])
    sweep = {t: F.Arm(rows, many, first, lambda r, v, t=t: F.at_threshold(r, v, t)) for t in F.TS}
    chosen = P.choose({t: F.pct_of(sweep[t].value, choosing) for t in F.TS}, list(F.TS))
    v = F.Arm(rows, many, first, lambda r, s: F.gated(r, s, tau, True))
    return {"V": v.value, "V_one": v.one, "T": sweep[chosen].value, "T_one": sweep[chosen].one, "t": chosen, "tau": tau}


def sides_27b(cases):
    """The trained 27B, the trained 27B with 8 examples and the base 27B with them, by their own loaders."""
    with the_27b():
        rows = {"27b": F.trained_rows(cases), "27b_ex": RR.load_rows(RETR27 / "trained", cases, "trained"),
                "base27b_ex": RR.load_rows(RETR27, cases, "base")}
        out = {name: dict(side(r), **fewer_values(r, RA.split_ids(r, cases))) for name, r in rows.items()}
    return out


def arm_side(arm, cases):
    """Another arm's side, or None if it has not finished its judge stage."""
    run = S.ROOT / arm
    need = run / "judged.jsonl"
    if arm == "a":
        if not need.exists() or not (run / "served" / "greedy_test.jsonl").exists():
            return None
        return side(rows_a(run, cases))
    if not need.exists() or not (run / "exec_retr8_test.jsonl").exists():
        return None
    return side(rows_bc(run, cases, arm))


# ------------------------------------------------------------ the report on one arm

def common_parts(parts, *values):
    """The reading rows every one of these systems has: a reading compares like with like."""
    keep = [set(v) for v in values]
    return {name: [i for i in ids if all(i in k for k in keep)] for name, ids in parts.items()}


def costs_a(timing, acc):
    """report_ask.costs without a published row: GPU-seconds per request, one request at a time."""
    s = timing["summary"]
    d = s["direct"]
    out = {"one answer": dict(P.per_solved(d["total_s"]["mean"], acc["direct"]), p50=d["total_s"]["p50"],
                              p95=d["total_s"]["p95"], tokens=round(d["tokens"], 1))}
    for k in Q.TIMING_KS:
        t = s["k%d" % k]
        out["round trip at %d" % k] = dict(P.per_solved(RR.request_gpu(t), acc[k]), p50=t["first_response_s"]["p50"],
                                           p95=t["first_response_s"]["p95"], tokens=round(t["tokens"], 1), asks=t["asks"])
    return out


def verifier_s():
    """The verifier's GPU-seconds per candidate, from the pooled pick's served pass (one batched request a row)."""
    tot_s = tot_c = 0.0
    for sp in A.SPLITS:
        for r in A.old_rows(sp).values():
            if r.get("score_s") and r.get("cands"):
                tot_s += r["score_s"]
                tot_c += len(r["cands"])
    return tot_s / tot_c if tot_c else None


def v_cost(rows, ids, timing, fewer, acc_v):
    """V on one card, composed from measured parts: the first response at 8, the verifier on the request's distinct
    samples, and the question and the turn after the reply on V's ask rate."""
    t = timing["summary"]["k8"]
    turn = (t["question_s"]["mean"] + t["turn2_s"]["mean"]) if t["question_s"] and t["turn2_s"] else 0.0
    per = verifier_s()
    cands = statistics.fmean(len({tuple(s) for s in rows[i]["samples"][:RA.K] if s is not None}) for i in ids)
    gpu = t["models_wall_s"]["mean"] + (per or 0.0) * cands + fewer["arms"]["V"]["asks_test"] / 100 * turn
    return dict(P.per_solved(gpu, acc_v), verifier_s_per_candidate=None if per is None else round(per, 4),
                candidates=round(cands, 2))


def build(arm, run, others, z):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    rows = rows_a(run, cases) if arm == "a" else rows_bc(run, cases, arm)
    ids = RA.split_ids(rows, cases)
    parts = {"untouched": ids["untouched"], "test": ids["test"]}
    reading = parts["untouched"] + parts["test"]
    me = side(rows)
    direct, rule, one_rule = me["direct"], me["rule"], me["one"]
    real = {i: r["real"] for i, r in rows.items()}
    placebo = {i: r["placebo"] for i, r in rows.items()}
    mean_rate = statistics.fmean(me["rate"].values())
    at_random = {i: mean_rate * r["real"] + (1 - mean_rate) * r["direct"] for i, r in rows.items()}
    oracle = {i: max(r["real"], r["direct"]) for i, r in rows.items()}
    four = RA.deployed_4b(set(rows))
    one = RA.one_run(rows)

    def means(values):
        return {name: P.mean(values, [i for i in ids[name] if i in values]) for name in ("val", "untouched", "test")}

    def asks_of(values):
        return {name: round(100 * statistics.fmean(values[i] for i in ids[name]), 1) if ids[name] else None
                for name in ("val", "untouched", "test")}

    def rd(a, b, one_a=None, one_b=None):
        sub = common_parts(parts, a, b)
        return P.reading(a, b, sub, one_a, one_b)

    who = NAMES[arm]
    systems, readings, descriptive = {}, {}, {}
    a = others.get("a")
    if arm == "a":
        systems.update({"%s, one answer" % who: means(direct), "the rule, 5 of 8 (mean of draws)": means(rule),
                        "the rule, one run": means(one_rule)})
        readings["S: the rule vs its §6 answer"] = rd(direct, rule, direct, one_rule)
        descriptive["the rule vs the 4B round trip as published (the same system, a parity check)"] = rd(four, rule, four, one_rule)
        twin = z["27b"]
        descriptive["the 4B's §6 answer vs the trained 27B's"] = rd(twin["direct"], direct, twin["direct"], direct)
        descriptive["the 4B's rule vs the trained 27B's"] = rd(twin["rule"], rule, twin["one"], one_rule)
    else:
        zero = {i: r["zero"] for i, r in rows.items()}
        systems.update({"%s, zero-shot" % NAMES[arm].replace(" with 8 examples", ""): means(zero),
                        "%s, one answer" % who: means(direct), "%s, the rule (mean of draws)" % who: means(rule),
                        "%s, the rule, one run" % who: means(one_rule)})
        if a:
            systems.update({"the served 4B, no examples, one answer": means(a["direct"]),
                            "the served 4B, no examples, the rule": means(a["rule"])})
        readings["R: with the rule vs without it, on 8 examples"] = rd(direct, rule, direct, one_rule)
        if arm == "b":
            if a:
                readings = dict({"E: 8 examples vs none, one answer each": rd(a["direct"], direct, a["direct"], direct)}, **readings)
                readings["W: 8 examples vs none, the rule on both"] = rd(a["rule"], rule, a["one"], one_rule)
            c = others.get("c")
            if c and all(i in c["direct"] for i in rows):
                systems.update({"the base 4B, 8 examples, one answer": means(c["direct"]),
                                "the base 4B, 8 examples, the rule": means(c["rule"])})
                readings["B: the trained vs the base 4B, 8 examples each, one answer each"] = rd(c["direct"], direct)
            twin = z["27b_ex"]
            descriptive["one answer vs the trained 27B's, 8 examples each"] = rd(twin["direct"], direct, twin["direct"], direct)
            descriptive["the rule vs the trained 27B's, 8 examples each"] = rd(twin["rule"], rule, twin["one"], one_rule)
        else:
            if a:
                readings = dict({"C: 8 examples vs the served 4B, one answer each": rd(a["direct"], direct, a["direct"], direct)}, **readings)
                readings["W: 8 examples vs the served 4B, the rule on both"] = rd(a["rule"], rule, a["one"], one_rule)
            descriptive["8 examples vs zero-shot, one answer each"] = rd(zero, direct, zero, direct)
            twin = z["base27b_ex"]
            descriptive["one answer vs the base 27B's, 8 examples each"] = rd(twin["direct"], direct, twin["direct"], direct)
            descriptive["the rule vs the base 27B's, 8 examples each"] = rd(twin["rule"], rule, twin["one"], one_rule)
    readings["I: after the reply vs after the placebo, every row"] = rd(placebo, real, placebo, real)
    systems.update({"always ask": means(real), "placebo reply on every row": means(placebo),
                    "asking at random, at the rule's rate": means(at_random), "asking only where it helps": means(oracle),
                    "the 4B round trip, as published": means(four)})
    descriptive.update({"the rule vs asking at random at its rate": rd(at_random, rule),
                        "the rule vs always asking": rd(real, rule, real, one_rule)})
    if arm != "a":
        descriptive["the rule vs the 4B round trip as published"] = rd(four, rule, four, one_rule)

    gap = None
    if arm == "c" and a:
        point, lo, hi = B.gap_closed(zero, direct, a["direct"], [i for i in reading if i in a["direct"]])
        gap = {k: None if v is None else round(100 * v, 1) for k, v in (("point", point), ("lo", lo), ("hi", hi))}

    rep = {"arm": arm, "rows": {name: len(v) for name, v in ids.items()}, "dialogues": sum(r["dialogue"] for r in rows.values()),
           "empty_questions": sum(not r["dialogue"] for r in rows.values()), "systems": systems,
           "asks": {"mean of draws": asks_of(me["rate"]), "one run": asks_of({i: v[1] for i, v in one.items()})},
           "gap_closed": gap, "readings": readings, "descriptive": descriptive,
           "outcomes": RA.question_outcomes({i: rows[i] for i in reading}, one)}

    rep["sweep"] = {}
    for k in SWEEP[arm]:
        d = RA.draws(rows, k)
        rep["sweep"][k] = {"pick": means({i: v[0] for i, v in d.items()}), "asks": asks_of({i: v[1] for i, v in d.items()})}
    tier = {i: r["tier"] for i, r in rows.items()}
    rep["tiers"] = {}
    for name in RA.TIERS:
        sub = [i for i in reading if tier[i] == name]
        rep["tiers"][name] = {"n": len(sub), "direct": P.mean(direct, sub), "rule": P.mean(rule, sub),
                              "asks": round(100 * statistics.fmean(me["rate"][i] for i in sub), 1) if sub else None}
    if arm != "a":
        rep["shape"] = {}
        for name, flag in (("a same-shape example was retrieved", True), ("none was", False)):
            sub = [i for i in reading if rows[i]["same_shape"] == flag]
            rep["shape"][name] = {"n": len(sub), "zero": P.mean(zero, sub), "direct": P.mean(direct, sub), "rule": P.mean(rule, sub)}

    judged = {i: r["judged"] for i, r in rows.items() if r["judged"]}
    flagged = {i for i, j in judged.items() if j["leak_flags"] or j["quotes_result"]}
    clean = {name: [i for i in v if i not in flagged] for name, v in parts.items()}
    head = "S" if arm == "a" else "R"
    rep["simulator"] = {"replies": len(judged), "flagged": len(flagged), "flagged_reading_rows": len(flagged & set(reading)),
                        "forced": H.simulator_record(run)["forced"], "head": head,
                        "head without flagged rows": P.reading(direct, rule, clean)}
    answers = [j["real"] for j in judged.values()]
    rep["turn2"] = {"answers": len(answers), "unparsed": sum(not x["parse_ok"] for x in answers),
                    "second_ask": sum(x["second_ask"] for x in answers), "think": sum(x["think"] for x in answers),
                    "valid_pct": round(100 * statistics.fmean(x["valid"] for x in answers), 1) if answers else None,
                    "tokens": round(statistics.fmean(x["tokens"] for x in answers), 1) if answers else None}
    questions = [r["question"] for r in A.load_jsonl(run / "question.jsonl")]
    rep["questions"] = {"rows": len(questions), "tokens": round(statistics.fmean(q["output_tokens"] for q in questions), 1) if questions else None,
                        "capped": sum(q["output_tokens"] >= Q.QUESTION_MAX_TOKENS for q in questions),
                        "empty": sum(not q["text"] for q in questions)}
    exec_files = [run / "served" / ("exec_%s.jsonl" % sp) for sp in A.SPLITS] if arm == "a" else \
                 [run / ("exec_retr8_%s.jsonl" % sp) for sp in A.SPLITS]
    valid = {r["id"]: int(bool(r["valid"])) for f in exec_files for r in A.load_jsonl(f)}
    rep["greedy"] = {"valid_test": P.mean(valid, [i for i in ids["test"] if i in valid])}
    rep["checks"] = {"the 4B round trip as published, val and test": [means(four)["val"], means(four)["test"]],
                     "the trained 27B's rule, untouched and test": [P.mean(z["27b"]["rule"], ids["untouched"]),
                                                                    P.mean(z["27b"]["rule"], ids["test"])]}
    for name in ("smoke", "rescore", "bm25"):
        path = run / (name + ".json")
        rep[name] = json.loads(path.read_text()) if path.exists() else None

    rep["fewer"] = F.build(rows, ids)
    mine = fewer_values(rows, ids)
    if mine["t"] != rep["fewer"]["chosen_t"] or mine["tau"] != rep["fewer"]["tau"] or \
            F.pct_of(mine["V"], ids["test"]) != rep["fewer"]["arms"]["V"]["test"]:
        raise SystemExit("fewer_values is not fewer.py's build")
    twin = z[{"a": "27b", "b": "27b_ex", "c": "base27b_ex"}[arm]]
    whose, each = {"a": ("the trained 27B's", ""), "b": ("the trained 27B's", ", 8 examples each"),
                   "c": ("the base 27B's", ", 8 examples each")}[arm]
    descriptive["V vs %s V%s" % (whose, each)] = rd(twin["V"], mine["V"], twin["V_one"], mine["V_one"])
    descriptive["T (t = %d) vs %s T (t = %d)%s" % (mine["t"], whose, twin["t"], each)] = rd(twin["T"], mine["T"], twin["T_one"], mine["T_one"])
    if arm == "a":                                                # the swap's question: the 4B's best here against the 27B asking nothing
        rep["posthoc"] = {"the 4B's V vs the trained 27B's one answer": rd(z["27b"]["direct"], mine["V"], z["27b"]["direct"], mine["V_one"])}
    if (run / "timing.json").exists():
        timing = json.loads((run / "timing.json").read_text())
        acc = {k: P.mean({i: v[0] for i, v in RA.draws(rows, k).items()}, ids["test"]) for k in Q.TIMING_KS}
        acc["direct"] = P.mean(direct, ids["test"])
        bm25 = (rep["bm25"] or {}).get("query_ms_mean", 0.0)
        rep["cost"] = costs_a(timing, acc) if arm == "a" else RR.costs(timing, bm25, acc)
        rep["cost"]["V on one card (composed)"] = v_cost(rows, ids["test"], timing, rep["fewer"], rep["fewer"]["arms"]["V"]["test"])
        rep["fewer_saves"] = {name: F.saved(rep["fewer"], timing, key)["saved_gpu_s"]
                              for name, key in (("T", "t = %d" % rep["fewer"]["chosen_t"]), ("V", "V"))}
    return rep


# ------------------------------------------------------------ rendering

def fmt(r):
    return "%+.1f [%+.1f, %+.1f]" % (r["diff_pts"], r["ci95"][0], r["ci95"][1])


def pct(v):
    return "—" if v is None else "%s%%" % v


def render(rep):
    n, arm = rep["rows"], rep["arm"]
    L = ["# %s: the trained 27B's systems on the 4B, arm %s" % (NAMES[arm][0].upper() + NAMES[arm][1:], arm), "",
         "By `../preregistration.md`; built by `scripts/sys4b/report4b.py` with the 27B reports' estimators. Rows: val %d, test %d, "
         "%d with a dialogue, %d where the model wrote no question and the §6 answer stands. Means of %d draws of 8 of the samples on "
         "disk, unless marked." % (n["val"], n["test"], rep["dialogues"], rep["empty_questions"], RA.DRAWS), "",
         "## The systems", "", "| system | val | untouched | test |", "|---|---|---|---|"]
    for name, v in rep["systems"].items():
        L.append("| %s | %s | %s | %s |" % (name, pct(v["val"]), pct(v["untouched"]), pct(v["test"])))
    a, g = rep["asks"], rep["gap_closed"]
    L += ["", "The rule asks on %s of val and %s of test (one run: %s and %s)." % (
        pct(a["mean of draws"]["val"]), pct(a["mean of draws"]["test"]), pct(a["one run"]["val"]), pct(a["one run"]["test"])), ""]
    if g:
        L += ["The examples close %s of the gap from the base 4B zero-shot to the served 4B on the reading rows [%s, %s]." % (
            pct(g["point"]), pct(g["lo"]), pct(g["hi"])), ""]
    L += ["## Readings", "", "On the %d untouched val rows plus the %d test rows. Paired bootstrap, conjunction rule; the one run is the "
          "first 8 samples, with exact McNemar." % (n["untouched"], n["test"]), "",
          "| reading | pooled rows | untouched | test | verdict | one run |", "|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | **%s** | %s | %s | **%s** | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), r["verdict"], RR.one_text(r)))
    fw = rep["fewer"]
    for name, r in fw["readings"].items():
        L.append("| %s | **%s** | %s | %s | **%s** | %s |" % (name.replace("V vs", "V: the verifier-gated ask vs").replace(
            "T vs", "T: at most %d of 8 vs" % fw["chosen_t"]), fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), r["verdict"], RR.one_text(r)))
    if "T vs the rule" not in fw["readings"]:
        L.append("")
        L.append("The choosing rows kept t = 5, so T has no reading.")
    L += ["", "t = %d and τ = %.3f, chosen on the %d choosing rows; T asks on %s of test, V on %s." % (
        fw["chosen_t"], fw["tau"], fw["rows"]["choosing"], pct(fw["arms"]["t = %d" % fw["chosen_t"]]["asks_test"]), pct(fw["arms"]["V"]["asks_test"])),
          "", "Descriptive, no verdict:", "", "| comparison | pooled rows | untouched | test | one run |", "|---|---|---|---|---|"]
    for name, r in rep["descriptive"].items():
        L.append("| %s | %s | %s | %s | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), RR.one_text(r)))
    if rep.get("posthoc"):
        L += ["", "Post hoc, not pre-registered, no verdict:", "", "| comparison | pooled rows | untouched | test | one run |", "|---|---|---|---|---|"]
        for name, r in rep["posthoc"].items():
            L.append("| %s | %s | %s | %s | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), RR.one_text(r)))
    L += ["", "## Asking fewer questions", "", "| arm | choosing | untouched | test | asks, reading rows | asks, test | questions per net right |",
          "|---|---|---|---|---|---|---|", "| the §6 answer | %s | %s | %s | 0 | 0 | — |" % (pct(fw["direct"]["choosing"]), pct(fw["direct"]["untouched"]), pct(fw["direct"]["test"]))]
    for name, x in fw["arms"].items():
        per = x["burden"]["per_net"]
        L.append("| %s | %s | %s | %s | %s | %s | %s |" % (F.name_of(name, fw["chosen_t"]), pct(x["choosing"]), pct(x["untouched"]), pct(x["test"]),
                                                           pct(x["asks_reading"]), pct(x["asks_test"]), "—" if per is None else per))
    extra = [(name, r) for name, r in fw["descriptive"].items() if name.startswith(("V vs", "Vg vs"))]
    if extra:
        L.append("")
    for name, r in extra:
        L.append("- %s: %s, %s." % (name, fmt(r["pooled"]), RR.one_text(r)))
    o = rep["outcomes"]
    L += ["", "## What its questions did (one run, the reading rows)", "", "| | rows | §6 answer right | right after the reply |", "|---|---|---|---|",
          "| it asked | %d | %d | %d |" % (o["asked"], o["right_before"], o["right_after"]), "| it answered | %d | %d | — |" % (o["left"], o["right_left"]),
          "", "Questions fixed %d answers and broke %d." % (o["fixed"], o["broke"]), "",
          "## Fewer or more samples (descriptive)", "", "| k | val | untouched | test | asks on test |", "|---|---|---|---|---|"]
    for k, v in rep["sweep"].items():
        L.append("| %s | %s | %s | %s | %s |" % (k, pct(v["pick"]["val"]), pct(v["pick"]["untouched"]), pct(v["pick"]["test"]), pct(v["asks"]["test"])))
    L += ["", "## By tier (the reading rows)", "", "| tier | n | one answer | the rule | asks |", "|---|---|---|---|---|"]
    for name, v in rep["tiers"].items():
        L.append("| %s | %d | %s | %s | %s |" % (name, v["n"], pct(v["direct"]), pct(v["rule"]), pct(v["asks"])))
    if rep.get("shape"):
        L += ["", "## By the nearest help (the reading rows)", "", "| | n | zero-shot | one answer | the rule |", "|---|---|---|---|---|"]
        for name, v in rep["shape"].items():
            L.append("| %s | %d | %s | %s | %s |" % (name, v["n"], pct(v["zero"]), pct(v["direct"]), pct(v["rule"])))
    if rep.get("cost"):
        L += ["", "## What it costs, one request at a time", "",
              "On the 200 seeded val rows, at test accuracy%s. GPU-seconds per request: the first response on the model, plus, on the ask rate, "
              "the question and the answer after the reply. V's row is composed: the round trip's first response, the verifier on the request's "
              "distinct samples at the pooled pick's measured time per candidate, and V's ask rate." % ("" if arm == "a" else ", with BM25 in the wait"), "",
              "| system | test | asks | first response p50 / p95 | tokens | GPU-s / request | GPU-s / solved | $ / 1,000 solved |", "|---|---|---|---|---|---|---|---|"]
        for name, c in rep["cost"].items():
            if name.startswith("V on"):
                L.append("| %s | %s%% | %s | — | — | %.3f | %.3f | $%.4f |" % (name, c["exec_match"], pct(rep["fewer"]["arms"]["V"]["asks_test"]),
                                                                             c["gpu_s_per_request"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
                continue
            L.append("| %s | %s%% | %s | %.2f / %.2f s | %s | %.3f | %.3f | $%.4f |" % (
                name, c["exec_match"], "%.0f%%" % (100 * c["asks"]) if "asks" in c else "—", c["p50"], c["p95"], c["tokens"],
                c["gpu_s_per_request"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
        sv = rep.get("fewer_saves") or {}
        L += ["", "Asking less saves only the question and the turn after it: T %.3f, V %.3f GPU-s per request." % (sv.get("T", 0.0), sv.get("V", 0.0))]
    q, t2, sim, c = rep["questions"], rep["turn2"], rep["simulator"], rep["checks"]
    L += ["", "## Checks", ""]
    sm, re_, bm = rep["smoke"], rep["rescore"], rep["bm25"]
    if sm:
        L.append("- Smoke: %d of %d answers identical to the recorded ones; %d prompts off the plan's token count; the %d longest prompts (up to %d tokens), "
                 "%d of %d identical under load." % (sm["identical"], sm["rows"], len(sm["prompt_token_mismatches"]), sm["longest_rows"],
                                                     sm["longest_prompt"], sm["longest_identical"], sm["longest_answers"]))
        if "longest_identical_to_recorded" in sm:                 # arm c, addendum 3: agreement under load is the gate
            L[-1] = L[-1][:-1] + (" (agreeing with each other, the gate since addendum 3); %d of %d match the transformers record." % (
                sm["longest_identical_to_recorded"], sm["longest_recorded_answers"]))
    L += ["- %s valid on test (the §6 answer)." % pct(rep["greedy"]["valid_test"]),
          "- The 4B round trip as published, rebuilt: val %s, test %s (published 55.4%%, 59.7%%)." % tuple(pct(x) for x in c["the 4B round trip as published, val and test"]),
          "- The trained 27B's rule, rebuilt: untouched %s, test %s (published 65.2%%, 70.0%%)." % tuple(pct(x) for x in c["the trained 27B's rule, untouched and test"]),
          "- Questions: %d rows, %s tokens on average, %d at the cap, %d empty." % (q["rows"], q["tokens"], q["capped"], q["empty"]),
          "- After the reply: %d answers, %d unparsed, %d second questions, %d with `<think>`, %s%% valid, %s tokens." % (
              t2["answers"], t2["unparsed"], t2["second_ask"], t2["think"], t2["valid_pct"], t2["tokens"]),
          "- The leak screen flagged %d of %d replies (%d on the reading rows); %d replies hit the simulator's budget. Without the flagged rows, %s reads %s, %s." % (
              sim["flagged"], sim["replies"], sim["flagged_reading_rows"], sim["forced"], sim["head"], fmt(sim["head without flagged rows"]["pooled"]),
              sim["head without flagged rows"]["verdict"])]
    if re_:
        L.append("- The verifier again on %d stored answers: mean |diff| %s, max %s." % (re_["rows"], re_["mean_abs"], re_["max_abs"]))
    if bm:
        L.append("- BM25 over %d train rows: built in %s s, %s ms a query." % (bm["train_rows"], bm["build_s"], bm["query_ms_mean"]))
    return "\n".join(L) + "\n"


# ------------------------------------------------------------ the 4B beside the 27B

SUMMARY_ROWS = (("a", "one answer", "the served 4B, one answer", "27b", "direct"),
                ("a", "the rule", "the rule, 5 of 8 (mean of draws)", "27b", "rule"),
                ("b", "8 examples, one answer", "the served 4B with 8 examples, one answer", "27b_ex", "direct"),
                ("b", "8 examples, the rule", "the served 4B with 8 examples, the rule (mean of draws)", "27b_ex", "rule"),
                ("c", "base, 8 examples, one answer", "the base 4B with 8 examples, one answer", "base27b_ex", "direct"),
                ("c", "base, 8 examples, the rule", "the base 4B with 8 examples, the rule (mean of draws)", "base27b_ex", "rule"))


def summary(z):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    reps = {arm: json.loads((S.ROOT / arm / "report.json").read_text()) for arm in "abc" if (S.ROOT / arm / "report.json").exists()}
    test = [c["id"] for c in cases["test"]]
    untouched = list(RA.C.untouched_ids(cases["val"]))
    L = ["# The trained 27B's systems on the 4B, beside the 27B", "",
         "From each arm's `report.json` and the 27B's own runs read by their own loaders; the same rows for both. Means of draws.", "",
         "| system | 4B untouched | 27B untouched | 4B test | 27B test |", "|---|---|---|---|---|"]
    for arm, label, key, twin, field in SUMMARY_ROWS:
        if arm not in reps:
            continue
        v = reps[arm]["systems"].get(key)
        if v is None:
            continue
        L.append("| %s | %s | %s | %s | %s |" % (label, pct(v["untouched"]), pct(P.mean(z[twin][field], untouched)), pct(v["test"]),
                                                pct(P.mean(z[twin][field], test))))
    L += ["", "| arm | reading | verdict | pooled rows |", "|---|---|---|---|"]
    for arm, rep in reps.items():
        for name, r in list(rep["readings"].items()) + list(rep["fewer"]["readings"].items()):
            L.append("| %s | %s | **%s** | %s |" % (arm, name, r["verdict"], fmt(r["pooled"])))
    if reps:
        L += ["", "Arms in: %s." % ", ".join(sorted(reps))]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("a", "b", "c", "summary"), required=True)
    ap.add_argument("--run", default=None, help="default runs/sys4b_2026-11-08/<arm>")
    args = ap.parse_args(argv)
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    z = sides_27b(cases)
    if args.arm == "summary":
        text = summary(z)
        (S.ROOT / "summary.md").write_text(text, encoding="utf-8")
        print(text)
        return 0
    run = pathlib.Path(args.run) if args.run else S.ROOT / args.arm
    needs = {"a": (), "b": ("a", "c"), "c": ("a",)}[args.arm]     # arm b's reading B reads arm c
    others = {other: arm_side(other, cases) for other in needs}
    rep = build(args.arm, run, others, z)
    (run / "report.json").write_text(json.dumps(rep, indent=1, default=str) + "\n", encoding="utf-8")
    (run / "report.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
