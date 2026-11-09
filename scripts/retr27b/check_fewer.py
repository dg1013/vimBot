#!/usr/bin/env python3
"""Recompute the fewer-questions readings from the raw files, sharing none of fewer.py's estimators (CPU).

Its own loading, draws (200 per row, on a fresh stream), τ, choice of t, interval and McNemar. Only the row
lists come from the project's code. After fewer_<model>.md: τ, t and the one-run counts must match exactly, and
the means and intervals to within the noise of the draws.

    python scripts/retr27b/check_fewer.py --model trained
    python scripts/retr27b/check_fewer.py --model base
    python scripts/retr27b/check_fewer.py --model trained-retr
"""
import argparse
import importlib.util
import json
import math
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"
ASK = RUNS / "ask27b_2026-11-06"
SERVE27 = RUNS / "serve27b_2026-11-05"
OLD = RUNS / "serve_pooled" / "served_at_k"
RETR = RUNS / "retr27b_2026-11-06"
SPLITS = ("val", "test")
K = 8
RULE_T = 5
DRAWS = 200
RATE = 0.20


def jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def row_lists():
    spec = importlib.util.spec_from_file_location("compare_ids", REPO / "scripts" / "distill_targets" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    val = jsonl(REPO / "data" / "splits" / "val.jsonl")
    test = jsonl(REPO / "data" / "splits" / "test.jsonl")
    untouched = mod.untouched_ids(val)
    clean = set(untouched)
    return untouched, [c["id"] for c in test], [c["id"] for c in val if c["id"] not in clean]


def load_trained():
    """{id: (results, samples, {answer: score}, {answer: right}, direct, real)}."""
    results = {r["id"]: r["results"] for r in jsonl(ASK / "views.jsonl")}
    samples = {r["id"]: r["samples"] for r in jsonl(SERVE27 / "samples.jsonl")}
    score, state = {}, {}
    for sp in SPLITS:
        seen = set()
        for r in jsonl(OLD / ("rows_%s.jsonl" % sp)):
            if r["id"] in seen:                                  # the first occurrence counts, as the report reads them
                continue
            seen.add(r["id"])
            for c in r["cands"]:
                score[(r["id"], c["mode"], c["command"])] = c["score"]
                state[(r["id"], c["mode"], c["command"])] = c["state"]
    for r in jsonl(SERVE27 / "scores27.jsonl"):
        score[(r["id"], r["mode"], r["command"])] = r["score"]
    for r in jsonl(SERVE27 / "judged.jsonl"):
        state[(r["id"], r["mode"], r["command"])] = r["state"]
    direct = {r["id"]: int(r["match"]) for sp in SPLITS for r in jsonl(SERVE27 / ("exec_%s.jsonl" % sp))}
    real = {r["id"]: int(r["real"]["match"]) for r in jsonl(ASK / "judged.jsonl")}
    return results, samples, score, state, direct, real


def load_retr(run):
    results = {r["id"]: r["results"] for r in jsonl(run / "views.jsonl")}
    samples = {r["id"]: r["samples"] for r in jsonl(run / "samples.jsonl")}
    score = {(r["id"], r["mode"], r["command"]): r["score"] for r in jsonl(run / "scores.jsonl")}
    state = {(r["id"], r["mode"], r["command"]): r["state"] for r in jsonl(run / "states.jsonl")}
    direct = {r["id"]: int(r["match"]) for sp in SPLITS for r in jsonl(run / ("exec_retr8_%s.jsonl" % sp))}
    real = {r["id"]: int(r["real"]["match"]) for r in jsonl(run / "judged.jsonl")}
    return results, samples, score, state, direct, real


def seen(i, idx, results, samples, score):
    """(the top result's count, the best score, the picked answer) for one set of sample indices."""
    tally = {}
    for n in idx:
        if results[i][n] is not None:
            tally[results[i][n]] = tally.get(results[i][n], 0) + 1
    cands = {}
    for n in idx:
        s = samples[i][n]
        if s is not None:
            cands[(s[0], s[1])] = cands.get((s[0], s[1]), 0) + 1
    best, pick = -math.inf, None
    for a, c in cands.items():
        rank = (score[(i, a[0], a[1])], c, a)
        if pick is None or rank > best_rank:
            best_rank, best, pick = rank, score[(i, a[0], a[1])], a
    return max(tally.values(), default=0), best, pick


def mcnemar(a, b, ids):
    up = sum(1 for i in ids if b[i] and not a[i])
    down = sum(1 for i in ids if a[i] and not b[i])
    n, k = up + down, min(up, down)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return up, down, p


def interval(a, b, ids, seed=11, boots=4000):
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    n = len(d)
    means = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(boots))
    return round(100 * sum(d) / n, 1), round(means[int(.025 * boots)], 1), round(means[int(.975 * boots) - 1], 1)


def pct(v, ids):
    return round(100 * statistics.fmean(v[i] for i in ids), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("trained", "base", "trained-retr"), required=True)
    args = ap.parse_args()
    run = RETR / "trained" if args.model == "trained-retr" else RETR
    results, samples, score, state, direct, real = load_trained() if args.model == "trained" else load_retr(run)
    untouched, test, choosing = row_lists()
    reading = untouched + test
    every = untouched + test + choosing
    print("rows: untouched %d, test %d, choosing %d" % (len(untouched), len(test), len(choosing)))

    rngs = {i: random.Random("fresh-fewer|%s" % i) for i in every}
    drawn = {i: [seen(i, rngs[i].sample(range(len(results[i])), K), results, samples, score) for _ in range(DRAWS)] for i in every}
    first = {i: seen(i, range(K), results, samples, score) for i in every}

    best = sorted(v[1] for i in choosing for v in drawn[i])
    tau = best[math.ceil(RATE * len(best)) - 1]
    print("τ on the choosing rows, by this stream: %.3f" % tau)

    def right(i, a):
        return int(state[(i, a[0], a[1])] == "match")

    def thresh(t):
        return ({i: statistics.fmean(real[i] if v[0] <= t else direct[i] for v in drawn[i]) for i in every},
                {i: statistics.fmean(v[0] <= t for v in drawn[i]) for i in every},
                {i: real[i] if first[i][0] <= t else direct[i] for i in every})

    def verifier(tau_):
        def one(i, v):
            return real[i] if v[1] <= tau_ else (right(i, v[2]) if v[2] is not None else direct[i])
        return ({i: statistics.fmean(one(i, v) for v in drawn[i]) for i in every},
                {i: statistics.fmean(v[1] <= tau_ for v in drawn[i]) for i in every},
                {i: one(i, first[i]) for i in every})

    arms = {t: thresh(t) for t in range(1, 6)}
    on_choosing = {t: pct(arms[t][0], choosing) for t in arms}
    print("choosing rows by t:", on_choosing)
    top = max(on_choosing.values())
    chosen = next(t for t in sorted(arms) if on_choosing[t] >= top - 1.0)
    print("t chosen by this stream: %d" % chosen)
    fewer = run / ("fewer_%s.json" % args.model)
    if fewer.exists():
        rep = json.loads(fewer.read_text())
        tau_used = rep["tau"]
        print("fewer.py's: τ %.3f, t %d" % (rep["tau"], rep["chosen_t"]))
        chosen = rep["chosen_t"]
    else:
        tau_used = tau
    rule = arms[RULE_T]
    v = verifier(tau_used)
    print("§6: untouched %.1f, test %.1f" % (pct(direct, untouched), pct(direct, test)))
    for name, arm in (("the rule", rule), ("T", arms[chosen]), ("V", v)):
        print("%s: untouched %.1f, test %.1f; asks untouched %.1f, test %.1f, reading %.1f" % (
            name, pct(arm[0], untouched), pct(arm[0], test), pct(arm[1], untouched), pct(arm[1], test), pct(arm[1], reading)))
    for name, arm in (("T vs the rule", arms[chosen]), ("V vs the rule", v)):
        print("%s, draws: %+.1f [%+.1f, %+.1f]" % ((name,) + interval(rule[0], arm[0], reading)))
        up, down, p = mcnemar(rule[2], arm[2], reading)
        print("%s, one run: %+.1f, +%d / -%d, p %.3g" % (name, 100 * (up - down) / len(reading), up, down, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
