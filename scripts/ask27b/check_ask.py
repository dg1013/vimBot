#!/usr/bin/env python3
"""Recompute the report's headline numbers from the raw files, sharing none of its estimators (CPU).

Its own loading, rule, draws (200 per row, on a fresh random stream), interval and McNemar. The ask
decision on one run does not come from views.jsonl: the first 8 stored samples go back through the box
through roundtrip.sample_evidence, the function real use calls. Only the row lists and that function
come from the project's code. After report.md, the one-run counts and the 4B's rates must match
exactly, and the means and intervals to within the noise of the draws.

    python scripts/ask27b/check_ask.py --run runs/ask27b_2026-11-06
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
SERVE27 = REPO / "runs" / "serve27b_2026-11-05"
SPLITS = ("val", "test")
DRAWS = 200
K = 8
MAX_AGREE = 5


def jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def row_lists():
    """(untouched val, test), the report's reading rows, and the cases by id."""
    spec = importlib.util.spec_from_file_location("compare_ids", REPO / "scripts" / "distill_targets" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    val = jsonl(REPO / "data" / "splits" / "val.jsonl")
    test = jsonl(REPO / "data" / "splits" / "test.jsonl")
    return mod.untouched_ids(val), [c["id"] for c in test], {c["id"]: c for c in val + test}


def live_asks(cases, samples):
    """{id: does real use's own evidence ask on the first 8 samples}, through the box."""
    from roundtrip import sample_evidence
    from score import execute
    from settings import load_settings
    timeout = load_settings(None)["executor"]["timeout"]
    out = {}
    for n, (i, case) in enumerate(cases.items()):
        texts = ["no command" if s is None else "%s: %s" % (s[0], s[1]) for s in samples[i]["samples"][:K]]
        out[i] = sample_evidence(case, texts, execute, timeout)["top"] <= MAX_AGREE
        if (n + 1) % 200 == 0:
            print("  %d / %d rows through the box" % (n + 1, len(cases)), flush=True)
    return out


def asks_on(results):
    tally = {}
    for r in results:
        if r is not None:
            tally[r] = tally.get(r, 0) + 1
    return max(tally.values(), default=0) <= MAX_AGREE


def draws(rows, real, direct, seed="fresh-check"):
    """{id: (share of right draws, share that ask)}, on a stream the report never used."""
    out = {}
    for i, results in rows.items():
        rng = random.Random("%s|%s" % (seed, i))
        right = asked = 0
        for _ in range(DRAWS):
            a = asks_on(rng.sample(results, K))
            asked += a
            right += real[i] if a else direct[i]
        out[i] = (right / DRAWS, asked / DRAWS)
    return out


def deployed_4b(ids):
    out = {}
    for sp, folder in (("test", "clarify_2026-10-07/test"), ("val", "clarify_2026-10-06/val")):
        top = json.loads((REPO / "runs" / "sample_rule_2026-10-09" / sp / "report.json").read_text(encoding="utf-8"))
        for r in jsonl(REPO / "runs" / folder / "verdicts_served.jsonl"):
            if r["id"] in ids:
                ask = round(top[r["id"]]["top_share"] * top[r["id"]]["n"]) <= MAX_AGREE
                out[r["id"]] = int(bool((r["real"] if ask else r["direct"])["match"]))
    return out


def mcnemar(a, b, ids):
    up = sum(1 for i in ids if b[i] and not a[i])
    down = sum(1 for i in ids if a[i] and not b[i])
    n, k = up + down, min(up, down)
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(k + 1)) / 2 ** n) if n else 1.0
    return up, down, p


def interval(a, b, ids, seed=7, boots=4000):
    d = [b[i] - a[i] for i in ids]
    rng = random.Random(seed)
    n = len(d)
    means = sorted(100 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(boots))
    return round(100 * sum(d) / n, 1), round(means[int(.025 * boots)], 1), round(means[int(.975 * boots) - 1], 1)


def pct(values, ids):
    return round(100 * statistics.fmean(values[i] for i in ids), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(REPO / "runs" / "ask27b_2026-11-06"))
    run = pathlib.Path(ap.parse_args().run)
    untouched, test, cases = row_lists()
    reading = untouched + test
    print("rows: untouched %d, test %d, reading %d, all %d" % (len(untouched), len(test), len(reading), len(cases)))

    judged = {r["id"]: r for r in jsonl(run / "judged.jsonl")}
    views = {r["id"]: r["results"] for r in jsonl(run / "views.jsonl")}
    samples = {r["id"]: r for r in jsonl(SERVE27 / "samples.jsonl")}
    direct = {r["id"]: int(r["match"]) for sp in SPLITS for r in jsonl(SERVE27 / ("exec_%s.jsonl" % sp))}
    real = {i: int(r["real"]["match"]) for i, r in judged.items()}
    placebo = {i: int(r["placebo"]["match"]) for i, r in judged.items()}
    print("§6 answer: untouched %.1f, test %.1f" % (pct(direct, untouched), pct(direct, test)))
    print("always ask: untouched %.1f, test %.1f; placebo: untouched %.1f, test %.1f" % (
        pct(real, untouched), pct(real, test), pct(placebo, untouched), pct(placebo, test)))

    four = deployed_4b(set(cases))
    print("the 4B round trip: untouched %.1f, test %.1f" % (pct(four, untouched), pct(four, test)))

    stored = {i: asks_on(views[i][:K]) for i in cases}
    live = live_asks(cases, samples)
    print("one run's ask decision, real use's code vs the stored results: %d of %d rows differ" % (
        sum(stored[i] != live[i] for i in cases), len(cases)))
    one = {i: real[i] if live[i] else direct[i] for i in cases}
    print("one run asks: untouched %.1f, test %.1f" % (pct(live, untouched), pct(live, test)))
    for name, a, b in (("S", direct, one), ("I", placebo, real), ("D", four, one)):
        up, down, p = mcnemar(a, b, reading)
        print("one run %s: %+.1f, +%d / -%d, p %.3g" % (name, 100 * (up - down) / len(reading), up, down, p))

    d = draws({i: views[i] for i in cases}, real, direct)
    rule = {i: v[0] for i, v in d.items()}
    rate = {i: v[1] for i, v in d.items()}
    print("the rule: untouched %.1f, test %.1f; asks untouched %.1f, test %.1f" % (
        pct(rule, untouched), pct(rule, test), pct(rate, untouched), pct(rate, test)))
    for name, a, b in (("S", direct, rule), ("I", placebo, real), ("D", four, rule)):
        print("draws %s, 724 rows: %+.1f [%+.1f, %+.1f]" % ((name,) + interval(a, b, reading)))
    for sub, ids in (("untouched", untouched), ("test", test)):
        print("draws S, %s: %+.1f [%+.1f, %+.1f]" % ((sub,) + interval(direct, rule, ids)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
