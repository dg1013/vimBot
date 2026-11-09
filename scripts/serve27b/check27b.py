#!/usr/bin/env python3
"""Recompute the report's headline numbers from the raw files, sharing none of its estimators (CPU).

Its own loading, pick, draws (200 per row, on a fresh random stream), interval and McNemar. Only the
row lists come from the project's code. After report.md, the greedy rates and the one-run counts must
match exactly, and the means and intervals to within the noise of the draws.

    python scripts/serve27b/check27b.py --run runs/serve27b_2026-11-05
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
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
SERVED = REPO / "runs" / "serve_pooled" / "served_at_k"
SPLITS = ("val", "test")
DRAWS = 200
BASE = {"4b": 4, "9b": 4}
POOL = {"4b": 4, "9b": 4, "27g": 1}


def jsonl(path):
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def row_lists():
    """(untouched val, choosing val, test), the report's row lists."""
    spec = importlib.util.spec_from_file_location("compare_ids", REPO / "scripts" / "distill_targets" / "compare.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    val = jsonl(REPO / "data" / "splits" / "val.jsonl")
    untouched = mod.untouched_ids(val)
    clean = set(untouched)
    return untouched, [c["id"] for c in val if c["id"] not in clean], [c["id"] for c in jsonl(REPO / "data" / "splits" / "test.jsonl")]


class Rows:
    """Each model's answers for a row, and every answer's score and verdict, from the raw files."""

    def __init__(self, run):
        self.old = {r["id"]: r for sp in SPLITS for r in jsonl(SERVED / ("rows_%s.jsonl" % sp))}
        self.samples = {r["id"]: r for r in jsonl(run / "samples.jsonl")}
        self.greedy = {r["id"]: r for sp in SPLITS for r in jsonl(run / ("greedy_%s.jsonl" % sp))}
        self.score, self.state = {}, {}
        for r in self.old.values():
            for c in r["cands"]:
                self.score[(r["id"], c["mode"], c["command"])] = c["score"]
                self.state[(r["id"], c["mode"], c["command"])] = c["state"]
        for r in jsonl(run / "scores27.jsonl"):
            self.score[(r["id"], r["mode"], r["command"])] = r["score"]
        for r in jsonl(run / "judged.jsonl"):
            self.state[(r["id"], r["mode"], r["command"])] = r["state"]

    def answers(self, i, model):
        if model == "27g":
            return [self.greedy[i]["samples"][0]]
        return self.samples[i]["samples"] if model == "27b" else self.old[i]["samples"][model]

    def right(self, i, drawn):
        """Highest score, ties by count, then mode and text: 1 if that answer matches."""
        counts = {}
        for s in drawn:
            if s is not None:
                counts[tuple(s)] = counts.get(tuple(s), 0) + 1
        if not counts:
            return 0
        best = max(counts, key=lambda key: (self.score[(i, *key)], counts[key], key))
        return int(self.state[(i, *best)] == "match")

    def one_run(self, arm, ids):
        return {i: self.right(i, [s for m, k in arm.items() for s in self.answers(i, m)[:k]]) for i in ids}

    def draws(self, arm, ids, seed="fresh-check"):
        out = {}
        for i in ids:
            rng = random.Random("%s|%s" % (seed, i))
            hits = 0
            for _ in range(DRAWS):
                drawn = []
                for m, k in arm.items():
                    pop = self.answers(i, m)
                    drawn += rng.sample(pop, min(k, len(pop)))
                hits += self.right(i, drawn)
            out[i] = hits / DRAWS
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
    ap.add_argument("--run", default=str(REPO / "runs" / "serve27b_2026-11-05"))
    run = pathlib.Path(ap.parse_args().run)
    untouched, choosing, test = row_lists()
    reading = untouched + test
    every = reading + choosing
    print("rows: untouched %d, choosing %d, test %d" % (len(untouched), len(choosing), len(test)))

    matched = {r["id"]: int(r["match"]) for sp in SPLITS for r in jsonl(run / ("exec_%s.jsonl" % sp))}
    print("27B greedy: val %.1f, untouched %.1f, test %.1f" % (pct(matched, untouched + choosing), pct(matched, untouched), pct(matched, test)))

    rows = Rows(run)
    one_base, one_pool = rows.one_run(BASE, every), rows.one_run(POOL, every)
    for name, a, b in (("R", one_base, matched), ("A", matched, one_pool), ("B", one_base, one_pool)):
        up, down, p = mcnemar(a, b, reading)
        print("one run %s: %+.1f, +%d / -%d, p %.3g" % (name, 100 * (up - down) / len(reading), up, down, p))

    base, pool = rows.draws(BASE, every), rows.draws(POOL, every)
    for name, a, b in (("R", base, matched), ("A", matched, pool), ("B", base, pool)):
        print("draws %s, 724 rows: %+.1f [%+.1f, %+.1f]" % ((name,) + interval(a, b, reading)))
    print("pooled 4 + 4: untouched %.1f, test %.1f" % (pct(base, untouched), pct(base, test)))
    print("three-model pool: untouched %.1f, test %.1f" % (pct(pool, untouched), pct(pool, test)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
