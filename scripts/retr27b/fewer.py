#!/usr/bin/env python3
"""Asking less: two ways to cut the 5-of-8 rule's questions, on a 27B's own samples (CPU).

By runs/retr27b_2026-11-06/preregistration.md. Every arm reads the rule's own draws, so an arm and the rule
differ only in which rows they ask on and what they answer where they do not:

    the rule   ask when the most common result of 8 samples holds at most 5 of them, else the §6 answer
    T          the rule at "at most t of 8"; the 250 choosing rows pick t: fewest questions first, the
               first within 1.0 point of the best
    V          the verifier-gated ask: ask when the 4B verifier's best score is at most τ, else its pick.
               τ is the 20th percentile over the choosing rows' draws. Text only, nothing executed
    Vg         V's asks, with the §6 answer in place of the pick. Descriptive

A sample that fails, changes nothing or does not parse counts against agreement, as in real use. No sample
parsed is the lowest score. Where no question exists, a row answers as if it had not asked.

    python scripts/retr27b/fewer.py --run runs/retr27b_2026-11-06 --model trained
    python scripts/retr27b/fewer.py --run runs/retr27b_2026-11-06 --model base
    python scripts/retr27b/fewer.py --run runs/retr27b_2026-11-06/trained --model trained-retr
"""
import argparse
import collections
import json
import math
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "scripts" / "ask27b", REPO / "scripts" / "serve27b", HERE):
    sys.path.insert(0, str(p))
import ask27b as Q                                                # noqa: E402
import report_ask as T                                            # noqa: E402

A, C, P = T.A, T.C, T.P
ASK_RUN = T.RUNS / "ask27b_2026-11-06"
GATE_RATE = 0.20
TS = (1, 2, 3, 4, 5)
K, DRAWS = T.K, T.DRAWS
RULE_T = 5


def draw_sets(row, k=K, n=DRAWS):
    """The report's draws, in its order, on its stream."""
    rng = random.Random("ask@%d|%s" % (k, row["id"]))
    return [rng.sample(range(len(row["results"])), k) for _ in range(n)]


def view(row, idxs):
    """What one request with these samples sees: (the top result's count, the best score, the pick).
    The pick is pooled.pick's: highest score, ties by count, then mode and text."""
    tally = collections.Counter(row["results"][n] for n in idxs if row["results"][n] is not None)
    cands = collections.Counter(tuple(row["samples"][n]) for n in idxs if row["samples"][n] is not None)
    top = max(tally.values(), default=0)
    if not cands:
        return top, -math.inf, None
    pick = max(cands, key=lambda c: (row["score"][c], cands[c], c))
    return top, row["score"][pick], pick


def at_threshold(row, seen, t):
    """(right, asked) for one request under "ask when the top result holds at most t"."""
    ask = row["dialogue"] and seen[0] <= t
    return (row["real"] if ask else row["direct"]), int(ask)


def gated(row, seen, tau, use_pick):
    """(right, asked) for one request under the verifier's gate."""
    if row["dialogue"] and seen[1] <= tau:
        return row["real"], 1
    if use_pick and seen[2] is not None:
        return row["label"][seen[2]], 0
    return row["direct"], 0


def tau_from(best, rate=GATE_RATE):
    """The smallest value with at least `rate` of them at or below it."""
    ordered = sorted(best)
    return ordered[max(0, math.ceil(rate * len(ordered)) - 1)]


class Arm:
    """One policy over the rows: means of draws and one run, each with its ask rate."""

    def __init__(self, rows, many, first, fn):
        got = {i: [fn(rows[i], v) for v in many[i]] for i in rows}
        self.value = {i: statistics.fmean(g[0] for g in got[i]) for i in rows}
        self.rate = {i: statistics.fmean(g[1] for g in got[i]) for i in rows}
        once = {i: fn(rows[i], first[i]) for i in rows}
        self.one = {i: once[i][0] for i in rows}
        self.one_rate = {i: once[i][1] for i in rows}


def pct_of(values, ids):
    return round(100 * statistics.fmean(values[i] for i in ids), 1) if ids else None


def questions_per_net(rows, ids, arm):
    """Questions put per net right answer gained on the §6 answer, on the mean of draws."""
    asked = sum(arm.rate[i] for i in ids)
    net = sum(arm.value[i] - rows[i]["direct"] for i in ids)
    return {"questions": round(asked, 1), "net_right": round(net, 1), "per_net": round(asked / net, 1) if net > 0 else None}


def build(rows, ids):
    parts = {"untouched": ids["untouched"], "test": ids["test"]}
    reading = parts["untouched"] + parts["test"]
    choosing = ids["choosing"] or list(rows)                      # a dry run's head of val may hold none
    many = {i: [view(r, s) for s in draw_sets(r)] for i, r in rows.items()}
    first = {i: view(r, range(K)) for i, r in rows.items()}
    direct = {i: r["direct"] for i, r in rows.items()}

    tau = tau_from([v[1] for i in choosing for v in many[i]])
    sweep = {t: Arm(rows, many, first, lambda r, v, t=t: at_threshold(r, v, t)) for t in TS}
    rule = sweep[RULE_T]
    v = Arm(rows, many, first, lambda r, s: gated(r, s, tau, True))
    vg = Arm(rows, many, first, lambda r, s: gated(r, s, tau, False))

    published = T.draws(rows)
    if any(abs(published[i][0] - rule.value[i]) > 1e-9 or abs(published[i][1] - rule.rate[i]) > 1e-9 for i in rows):
        raise SystemExit("the rule at 5 of 8 here is not the report's rule")

    chosen = P.choose({t: pct_of(sweep[t].value, choosing) for t in TS}, list(TS))

    def line(arm):
        return {"choosing": pct_of(arm.value, choosing), "untouched": pct_of(arm.value, parts["untouched"]),
                "test": pct_of(arm.value, parts["test"]), "reading": pct_of(arm.value, reading),
                "asks_choosing": pct_of(arm.rate, choosing), "asks_untouched": pct_of(arm.rate, parts["untouched"]),
                "asks_test": pct_of(arm.rate, parts["test"]), "asks_reading": pct_of(arm.rate, reading),
                "one_asks_reading": pct_of(arm.one_rate, reading), "burden": questions_per_net(rows, reading, arm)}

    rep = {"rows": {name: len(x) for name, x in ids.items()}, "tau": tau, "chosen_t": chosen,
           "t_on_choosing": {t: pct_of(sweep[t].value, choosing) for t in TS},
           "direct": {"choosing": pct_of(direct, choosing), "untouched": pct_of(direct, parts["untouched"]),
                      "test": pct_of(direct, parts["test"]), "reading": pct_of(direct, reading)},
           "arms": dict({"t = %d" % t: line(sweep[t]) for t in TS}, **{"V": line(v), "Vg": line(vg)})}

    rep["readings"] = {"V vs the rule": P.reading(rule.value, v.value, parts, rule.one, v.one)}
    if chosen != RULE_T:
        rep["readings"]["T vs the rule"] = P.reading(rule.value, sweep[chosen].value, parts, rule.one, sweep[chosen].one)
    rep["descriptive"] = {"V vs the §6 answer": P.reading(direct, v.value, parts, direct, v.one),
                          "Vg vs the rule": P.reading(rule.value, vg.value, parts, rule.one, vg.one),
                          "Vg vs the §6 answer": P.reading(direct, vg.value, parts, direct, vg.one),
                          "the rule vs the §6 answer": P.reading(direct, rule.value, parts, direct, rule.one)}
    for t in TS:
        if t != RULE_T:
            rep["descriptive"]["t = %d vs the rule" % t] = P.reading(rule.value, sweep[t].value, parts, rule.one, sweep[t].one)
            rep["descriptive"]["t = %d vs the §6 answer" % t] = P.reading(direct, sweep[t].value, parts, direct, sweep[t].one)
    rep["agreement"] = {"V's asks that the rule also asks": round(100 * sum(min(v.rate[i], rule.rate[i]) for i in reading)
                                                                 / max(1e-9, sum(v.rate[i] for i in reading)), 1)}
    return rep


def saved(rep, timing, arm):
    """What this arm saves on the round trip: only the question and the turn after it go, so the first
    response, with its 8 samples, stays. GPU-seconds per request, on test's ask rates."""
    t = timing["summary"]["k8"]
    turn = (t["question_s"]["mean"] + t["turn2_s"]["mean"]) if t["question_s"] and t["turn2_s"] else 0.0   # a dry run's 4 rows may hold none
    fewer = (rep["arms"]["t = %d" % RULE_T]["asks_test"] - rep["arms"][arm]["asks_test"]) / 100
    return {"turn_s": round(turn, 3), "saved_gpu_s": round(fewer * turn, 3), "first_response_gpu_s": round(t["models_wall_s"]["mean"], 3)}


def trained_rows(cases):
    """The trained 27B's rows: the ask run's turns, plus its 32 samples with their scores and verdicts."""
    rows = T.load_rows(ASK_RUN, cases)
    for sp in A.SPLITS:
        merged = P.load_rows(Q.SERVE27, sp, [c["id"] for c in cases[sp]])[0]
        for i, m in merged.items():
            if i in rows:
                score, label = P.S.row_maps(m)
                rows[i].update(samples=m["samples"]["27b"], score=score, label=label)
    return rows


def fmt(r):
    return "%+.1f [%+.1f, %+.1f]" % (r["diff_pts"], r["ci95"][0], r["ci95"][1])


def one_text(r):
    o = r.get("one_run")
    return "%+.1f, +%d / −%d, p %.3g" % (o["diff_pts"], o["gained"], o["lost"], o["p"]) if o else "—"


def name_of(arm, chosen):
    if arm == "t = %d" % RULE_T:
        return "the rule, 5 of 8"
    if arm == "t = %d" % chosen:
        return "**T: %s of 8**" % arm.split()[-1]
    if arm == "V":
        return "**V: the verifier-gated ask**"
    if arm == "Vg":
        return "Vg: V's asks, the §6 answer otherwise"
    return "at most %s of 8" % arm.split()[-1]


def render(rep, title, cost=None):
    n = rep["rows"]
    d = rep["direct"]
    L = ["# %s" % title, "",
         "By `preregistration.md` beside this file; built by `scripts/retr27b/fewer.py`. Every arm reads the rule's own 20 draws of 8 "
         "samples per row, so they differ only in where they ask and what they answer elsewhere. t and τ come from the %d choosing "
         "rows and are read on the %d untouched val rows plus the %d test rows." % (n["choosing"], n["untouched"], n["test"]), "",
         "**Chosen on the choosing rows:** t = %d; τ = %.3f." % (rep["chosen_t"], rep["tau"]), "",
         "| arm | choosing | untouched | test | asks, reading rows | asks, test | questions per net right |",
         "|---|---|---|---|---|---|---|",
         "| the §6 answer | %s%% | %s%% | %s%% | 0 | 0 | — |" % (d["choosing"], d["untouched"], d["test"])]
    for arm, a in rep["arms"].items():
        per = a["burden"]["per_net"]
        L.append("| %s | %s%% | %s%% | %s%% | %s%% | %s%% | %s |" % (
            name_of(arm, rep["chosen_t"]), a["choosing"], a["untouched"], a["test"], a["asks_reading"], a["asks_test"],
            "—" if per is None else per))
    L += ["", "## Readings", "", "Paired bootstrap, conjunction rule; the one run is the first 8 samples, with exact McNemar. "
          "\"No significant impact\" means *tied* or *helps*; the interval's lower end is how much it might lose.", "",
          "| reading | pooled rows | untouched | test | verdict | one run |", "|---|---|---|---|---|---|"]
    if not rep["readings"].get("T vs the rule"):
        L.insert(-2, "The choosing rows kept t = 5, so T has no reading.\n")
    for name, r in rep["readings"].items():
        L.append("| %s | **%s** | %s | %s | **%s** | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), r["verdict"], one_text(r)))
    L += ["", "Descriptive, no verdict:", "", "| comparison | pooled rows | untouched | test | one run |", "|---|---|---|---|---|"]
    for name, r in rep["descriptive"].items():
        L.append("| %s | %s | %s | %s | %s |" % (name, fmt(r["pooled"]), fmt(r["untouched"]), fmt(r["test"]), one_text(r)))
    for k, value in rep["agreement"].items():
        L += ["", "%s: %s%%." % (k, value)]
    if cost:
        L += ["", "## What it saves", "",
              "Only the question and the turn after it go: the first response keeps its 8 samples. One dialogue costs %.2f GPU-s, the first "
              "response %.2f." % (cost["turn_s"], cost["first_response_gpu_s"]), "",
              "| arm | GPU-s saved per request |", "|---|---|",
              "| T | %.3f |" % cost["T"], "| V or Vg (a second model is needed beside the 27B, which fills the card) | %.3f |" % cost["V"]]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(T.RUNS / "retr27b_2026-11-06"))
    ap.add_argument("--model", choices=("trained", "base", "trained-retr"), required=True)
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    if args.model == "trained":
        rows = trained_rows(cases)
        title, timing = "Fewer questions on the trained 27B", json.loads((ASK_RUN / "timing.json").read_text())
    else:
        import report_retr as B
        arm = "base" if args.model == "base" else "trained"
        rows = B.load_rows(run, cases, arm)
        title, timing = "Fewer questions on %s with 8 BM25 examples" % B.NAMES[arm], json.loads((run / "timing.json").read_text())
    rep = build(rows, T.split_ids(rows, cases))
    cost = {"T": saved(rep, timing, "t = %d" % rep["chosen_t"])["saved_gpu_s"], "V": saved(rep, timing, "V")["saved_gpu_s"],
            **{k: v for k, v in saved(rep, timing, "V").items() if k != "saved_gpu_s"}}
    rep["cost"] = cost
    (run / ("fewer_%s.json" % args.model)).write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    (run / ("fewer_%s.md" % args.model)).write_text(render(rep, title, cost), encoding="utf-8")
    print(render(rep, title, cost))
    return 0


if __name__ == "__main__":
    sys.exit(main())
