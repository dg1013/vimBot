#!/usr/bin/env python3
"""Recompute report_retr.py's readings from the raw files, sharing none of its estimators (CPU).

Its own loading, rule, draws (200 per row, on a fresh random stream), interval, conjunction rule and McNemar. This
arm's ask decision on one run does not come from views.jsonl: the first 8 stored samples go back through the box
through roundtrip.sample_evidence, the function real use calls. The other side of each reading is read from its stored
files: the trained 27B with no examples from the ask run, whose own check (scripts/ask27b/check_ask.py) put its
samples through the box, and the base 27B with examples from Part 2. Only the row lists and that function come from
the project's code. After report.md: the one-run counts must match exactly, and the means and intervals to within the
noise of the draws.

    python scripts/retr27b/check_retr.py --arm base
    python scripts/retr27b/check_retr.py --arm trained
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
for p in ("src", "scripts", "scripts/serve", "scripts/clarify", "scripts/readings"):
    sys.path.insert(0, str(REPO / p))
RUNS = REPO / "runs"
RETR = RUNS / "retr27b_2026-11-06"
ASK = RUNS / "ask27b_2026-11-06"
SERVE27 = RUNS / "serve27b_2026-11-05"
SPLITS = ("val", "test")
DRAWS = 200
K = 8
MAX_AGREE = 5


def jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def row_lists():
    """(untouched val ids, test ids, {id: case})."""
    spec = importlib.util.spec_from_file_location("compare_ids", REPO / "scripts" / "distill_targets" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    val = jsonl(REPO / "data" / "splits" / "val.jsonl")
    test = jsonl(REPO / "data" / "splits" / "test.jsonl")
    return mod.untouched_ids(val), [c["id"] for c in test], {c["id"]: c for c in val + test}


def side(run, exec_name, views_run=None, judged_run=None):
    """{"direct", "results", "real", "placebo"} by id. A row with no dialogue answers directly, as real use does."""
    direct = {r["id"]: int(r["match"]) for sp in SPLITS for r in jsonl(run / (exec_name % sp))}
    out = {"direct": direct}
    if views_run is not None:
        out["results"] = {r["id"]: r["results"] for r in jsonl(views_run / "views.jsonl")}
        judged = {r["id"]: r for r in jsonl(judged_run / "judged.jsonl")}
        out["real"] = {i: int(judged[i]["real"]["match"]) if i in judged else d for i, d in direct.items()}
        out["placebo"] = {i: int(judged[i]["placebo"]["match"]) if i in judged else d for i, d in direct.items()}
        out["dialogue"] = {i: i in judged for i in direct}
    return out


def asks_on(results):
    tally = {}
    for r in results:
        if r is not None:
            tally[r] = tally.get(r, 0) + 1
    return max(tally.values(), default=0) <= MAX_AGREE


def rule_draws(s, ids, seed):
    """{id: share of right draws}, {id: share that ask}, on a stream no report used."""
    right, rate = {}, {}
    for i in ids:
        rng = random.Random("%s|%s" % (seed, i))
        res = s["results"][i]
        got = [asks_on(rng.sample(res, K)) and s["dialogue"][i] for _ in range(DRAWS)]
        rate[i] = statistics.fmean(got)
        right[i] = statistics.fmean(s["real"][i] if a else s["direct"][i] for a in got)
    return right, rate


def live_asks(cases, samples, ids):
    """{id: does real use's own evidence ask on the first 8 samples}, through the box."""
    from roundtrip import sample_evidence
    from score import execute
    from settings import load_settings
    timeout = load_settings(None)["executor"]["timeout"]
    out = {}
    for n, i in enumerate(ids):
        texts = ["no command" if s is None else "%s: %s" % (s[0], s[1]) for s in samples[i][:K]]
        out[i] = sample_evidence(cases[i], texts, execute, timeout)["top"] <= MAX_AGREE
        if (n + 1) % 200 == 0:
            print("  %d / %d rows through the box" % (n + 1, len(ids)), flush=True)
    return out


def mcnemar(a, b, ids):
    up = sum(1 for i in ids if b[i] and not a[i])
    down = sum(1 for i in ids if a[i] and not b[i])
    n, k = up + down, min(up, down)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return up, down, p


def interval(a, b, ids, seed=23, boots=4000):
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    n = len(d)
    means = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(boots))
    return 100 * sum(d) / n, means[int(.025 * boots)], means[int(.975 * boots) - 1]


def judge(pooled, parts):
    """Helps: the pooled interval above zero and both slices positive. Hurts: the mirror. Else tied."""
    _, lo, hi = pooled
    if lo > 0 and all(p[0] > 0 for p in parts):
        return "helps"
    if hi < 0 and all(p[0] < 0 for p in parts):
        return "hurts"
    return "tied"


def pct(v, ids):
    return 100 * statistics.fmean(v[i] for i in ids)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=("base", "trained"), required=True)
    args = ap.parse_args()
    run = RETR if args.arm == "base" else RETR / "trained"
    untouched, test, cases = row_lists()
    reading = untouched + test
    every = [i for i in cases]
    print("rows: untouched %d, test %d, reading %d, all %d" % (len(untouched), len(test), len(reading), len(every)))

    me = side(run, "exec_retr8_%s.jsonl", run, run)
    samples = {r["id"]: r["samples"] for r in jsonl(run / "samples.jsonl")}
    missing = [i for i in every if i not in me["direct"] or i not in me["results"] or i not in samples]
    if missing:
        print("rows missing from this arm's files: %d; they are left out" % len(missing))
        every = [i for i in every if i not in set(missing)]
        reading = [i for i in reading if i not in set(missing)]
        untouched = [i for i in untouched if i not in set(missing)]
        test = [i for i in test if i not in set(missing)]
    print("dialogues: %d of %d rows" % (sum(me["dialogue"][i] for i in every), len(every)))
    print("one answer: untouched %.1f, test %.1f" % (pct(me["direct"], untouched), pct(me["direct"], test)))

    trained = side(SERVE27, "exec_%s.jsonl", ASK, ASK)
    print("the trained 27B with no examples, one answer: untouched %.1f, test %.1f" % (
        pct(trained["direct"], untouched), pct(trained["direct"], test)))

    stored = {i: asks_on(me["results"][i][:K]) for i in every}
    live = live_asks(cases, samples, every)
    print("one run's ask decision, real use's code vs the stored results: %d of %d rows differ" % (
        sum(stored[i] != live[i] for i in every), len(every)))
    me_one = {i: me["real"][i] if (live[i] and me["dialogue"][i]) else me["direct"][i] for i in every}
    tr_one = {i: trained["real"][i] if (asks_on(trained["results"][i][:K]) and trained["dialogue"][i]) else trained["direct"][i]
              for i in every}
    me_rule, me_rate = rule_draws(me, every, "fresh-retr")
    tr_rule, _ = rule_draws(trained, every, "fresh-retr")
    print("the rule: untouched %.1f, test %.1f; asks untouched %.1f, test %.1f; one run asks test %.1f" % (
        pct(me_rule, untouched), pct(me_rule, test), pct(me_rate, untouched), pct(me_rate, test), pct(live, test)))
    print("the trained 27B's rule, no examples: untouched %.1f, test %.1f" % (pct(tr_rule, untouched), pct(tr_rule, test)))

    if args.arm == "base":
        readings = (("C: examples vs the trained 27B, one answer each", trained["direct"], me["direct"], trained["direct"], me["direct"]),
                    ("R: the rule vs one answer", me["direct"], me_rule, me["direct"], me_one),
                    ("W: examples vs the trained 27B, the rule on both", tr_rule, me_rule, tr_one, me_one),
                    ("I: the reply vs the placebo, every row", me["placebo"], me["real"], me["placebo"], me["real"]))
    else:
        base = side(RETR, "exec_retr8_%s.jsonl")
        readings = (("E: the trained 27B with examples vs without, one answer each", trained["direct"], me["direct"],
                     trained["direct"], me["direct"]),
                    ("R: the rule vs one answer", me["direct"], me_rule, me["direct"], me_one),
                    ("W: examples vs none, the rule on both", tr_rule, me_rule, tr_one, me_one),
                    ("B: trained vs base, 8 examples each, one answer", base["direct"], me["direct"], base["direct"], me["direct"]),
                    ("I: the reply vs the placebo, every row", me["placebo"], me["real"], me["placebo"], me["real"]))
    for name, a, b, one_a, one_b in readings:
        pooled = interval(a, b, reading)
        parts = [interval(a, b, ids) for ids in (untouched, test)]
        up, down, p = mcnemar(one_a, one_b, reading)
        print("%s: %+.1f [%+.1f, %+.1f]; untouched %+.1f, test %+.1f; %s; one run %+.1f, +%d / -%d, p %.3g" % (
            (name,) + pooled + (parts[0][0], parts[1][0], judge(pooled, parts), 100 * (up - down) / len(reading), up, down, p)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
