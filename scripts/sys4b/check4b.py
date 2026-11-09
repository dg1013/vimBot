#!/usr/bin/env python3
"""Recompute an arm's headline readings from the raw files, sharing none of report4b.py's estimators (CPU).

scripts/retr27b/check_retr.py's and check_fewer.py's own loading, draws (200 per row, on fresh streams), interval,
conjunction rule, τ, choice of t and McNemar, pointed at the 4B's folders. The one run's ask decision does not come from
views.jsonl: the first 8 stored samples go back through the box through roundtrip.sample_evidence, the function real use
calls. After report.md: the one-run counts must match exactly, and the means and intervals to within the noise of the
draws.

    python scripts/sys4b/check4b.py --arm a
"""
import argparse
import json
import math
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "retr27b"))
import check_fewer as KF                                          # noqa: E402
import check_retr as K                                            # noqa: E402

ROOT = REPO / "runs" / "sys4b_2026-11-08"
SERVED = ROOT / "a" / "served"


def load(arm):
    """(side, samples, {(id, mode, command): score}, {(id, mode, command): state}) for an arm."""
    if arm == "a":
        side = K.side(SERVED, "exec_%s.jsonl", ROOT / "a", ROOT / "a")
        samples = {r["id"]: r["samples"] for r in K.jsonl(SERVED / "samples.jsonl")}
        score = {(r["id"], r["mode"], r["command"]): r["score"] for r in K.jsonl(SERVED / "scores27.jsonl")}
        state = {(r["id"], r["mode"], r["command"]): r["state"] for r in K.jsonl(SERVED / "states.jsonl")}
        return side, samples, score, state
    run = ROOT / arm
    side = K.side(run, "exec_retr8_%s.jsonl", run, run)
    samples = {r["id"]: r["samples"] for r in K.jsonl(run / "samples.jsonl")}
    score = {(r["id"], r["mode"], r["command"]): r["score"] for r in K.jsonl(run / "scores.jsonl")}
    state = {(r["id"], r["mode"], r["command"]): r["state"] for r in K.jsonl(run / "states.jsonl")}
    return side, samples, score, state


def show(name, a, b, one_a, one_b, untouched, test):
    reading = untouched + test
    pooled = K.interval(a, b, reading)
    parts = [K.interval(a, b, ids) for ids in (untouched, test)]
    up, down, p = K.mcnemar(one_a, one_b, reading)
    print("%s: %+.1f [%+.1f, %+.1f]; untouched %+.1f, test %+.1f; %s; one run %+.1f, +%d / -%d, p %.3g" % (
        (name,) + pooled + (parts[0][0], parts[1][0], K.judge(pooled, parts), 100 * (up - down) / len(reading), up, down, p)))


def fewer(arm, side, samples, score, state, untouched, test, choosing):
    """check_fewer.py's main, on this arm's loaded files."""
    results, direct, real = side["results"], side["direct"], side["real"]
    reading = untouched + test
    every = untouched + test + choosing
    rngs = {i: random.Random("fresh-fewer|%s" % i) for i in every}
    drawn = {i: [KF.seen(i, rngs[i].sample(range(len(results[i])), KF.K), results, samples, score) for _ in range(KF.DRAWS)] for i in every}
    first = {i: KF.seen(i, range(KF.K), results, samples, score) for i in every}
    best = sorted(v[1] for i in choosing for v in drawn[i])
    tau = best[math.ceil(KF.RATE * len(best)) - 1]
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
    on_choosing = {t: KF.pct(arms[t][0], choosing) for t in arms}
    print("choosing rows by t:", on_choosing)
    top = max(on_choosing.values())
    chosen = next(t for t in sorted(arms) if on_choosing[t] >= top - 1.0)
    print("t chosen by this stream: %d" % chosen)
    rep_path = ROOT / arm / "report.json"
    tau_used = tau
    if rep_path.exists():
        rep = json.loads(rep_path.read_text())["fewer"]
        print("report4b.py's: τ %.3f, t %d" % (rep["tau"], rep["chosen_t"]))
        tau_used, chosen = rep["tau"], rep["chosen_t"]
    rule = arms[KF.RULE_T]
    v = verifier(tau_used)
    for name, arm_ in (("the rule", rule), ("T", arms[chosen]), ("V", v)):
        print("%s: untouched %.1f, test %.1f; asks untouched %.1f, test %.1f, reading %.1f" % (
            name, KF.pct(arm_[0], untouched), KF.pct(arm_[0], test), KF.pct(arm_[1], untouched), KF.pct(arm_[1], test), KF.pct(arm_[1], reading)))
    for name, arm_ in (("T vs the rule", arms[chosen]), ("V vs the rule", v)):
        print("%s, draws: %+.1f [%+.1f, %+.1f]" % ((name,) + KF.interval(rule[0], arm_[0], reading)))
        up, down, p = KF.mcnemar(rule[2], arm_[2], reading)
        print("%s, one run: %+.1f, +%d / -%d, p %.3g" % (name, 100 * (up - down) / len(reading), up, down, p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("a", "b", "c"), required=True)
    args = ap.parse_args(argv)
    untouched, test, cases = K.row_lists()
    choosing = [i for i in cases if i not in set(untouched) | set(test)]
    every = list(cases)
    me, samples, score, state = load(args.arm)
    missing = [i for i in every if i not in me["direct"] or i not in me["results"] or i not in samples]
    if missing:
        print("rows missing from this arm's files: %d; they are left out" % len(missing))
        gone = set(missing)
        every, untouched, test, choosing = ([i for i in x if i not in gone] for x in (every, untouched, test, choosing))
    print("rows: untouched %d, test %d, choosing %d; dialogues %d of %d" % (
        len(untouched), len(test), len(choosing), sum(me["dialogue"][i] for i in every), len(every)))
    print("one answer: untouched %.1f, test %.1f" % (K.pct(me["direct"], untouched), K.pct(me["direct"], test)))
    stored = {i: K.asks_on(me["results"][i][:K.K]) for i in every}
    live = K.live_asks({i: cases[i] for i in every}, samples, every)
    print("one run's ask decision, real use's code vs the stored results: %d of %d rows differ" % (
        sum(stored[i] != live[i] for i in every), len(every)))
    me_one = {i: me["real"][i] if (live[i] and me["dialogue"][i]) else me["direct"][i] for i in every}
    me_rule, me_rate = K.rule_draws(me, every, "fresh-sys4b")
    print("the rule: untouched %.1f, test %.1f; asks untouched %.1f, test %.1f; one run asks test %.1f" % (
        K.pct(me_rule, untouched), K.pct(me_rule, test), K.pct(me_rate, untouched), K.pct(me_rate, test), K.pct(live, test)))
    if args.arm == "a":
        readings = [("S: the rule vs one answer", me["direct"], me_rule, me["direct"], me_one)]
    else:
        a = load("a")[0]
        a_one = {i: a["real"][i] if (K.asks_on(a["results"][i][:K.K]) and a["dialogue"][i]) else a["direct"][i] for i in every}
        a_rule, _ = K.rule_draws(a, every, "fresh-sys4b")
        first = "E: 8 examples vs none, one answer each" if args.arm == "b" else "C: 8 examples vs the served 4B, one answer each"
        readings = [(first, a["direct"], me["direct"], a["direct"], me["direct"]),
                    ("R: the rule vs one answer", me["direct"], me_rule, me["direct"], me_one),
                    ("W: the rule on both", a_rule, me_rule, a_one, me_one)]
        if args.arm == "b" and (ROOT / "c" / "exec_retr8_test.jsonl").exists():
            c = K.side(ROOT / "c", "exec_retr8_%s.jsonl")
            if all(i in c["direct"] for i in every):
                readings.append(("B: trained vs base, 8 examples each, one answer", c["direct"], me["direct"], c["direct"], me["direct"]))
    readings.append(("I: the reply vs the placebo, every row", me["placebo"], me["real"], me["placebo"], me["real"]))
    for r in readings:
        show(*r, untouched, test)
    fewer(args.arm, me, samples, score, state, untouched, test, choosing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
