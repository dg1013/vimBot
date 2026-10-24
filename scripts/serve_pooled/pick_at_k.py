#!/usr/bin/env python3
"""How the verifier's pick scales with samples: k = 1 ... 128 per model (test v2).

Uses the 128 samples per row each model already drew for the ceiling run
(runs/qwen9b_methods_2026-10-19/ceiling/samples_served{4b,9b}.jsonl). Every
distinct command is scored once by the verifier through the served endpoint
(serve_pooled.sh; the same scores as transformers within bf16 rounding).
Then, for each k, R random draws of k of a model's 128 samples (without
replacement) give that draw's candidates; the verifier picks the best-scored
one. Reported per k, for the 4B, the 9B and the two pooled (k each):
  pick@k   the verifier's pick is right (the deployable number)
  pass@k   some drawn sample is right (the ceiling the pick draws from)
  maj@k    the most common drawn command is right (voting, no verifier)
Each k's pick@k is also compared with the same model's pick@8, paired over
rows (a row's value is its share of right picks over the draws), with a 95%
bootstrap CI over rows.
With --timing, the served pooled pick is also timed (k = 1 / 4 / 8 / 32 / 64 /
128 by default; a new timing replaces only its own k's entry, earlier ones are
kept with when they were measured) and priced per solved task: one request already holds the card at k = 8
(measure.json: throughput flat from 1 to 32 concurrent), so its GPU-seconds
are its mean one-at-a-time latency, divided by the pooled pick@k above.

    python scripts/serve_pooled/pick_at_k.py --out runs/serve_pooled           # scores, then the curves
    python scripts/serve_pooled/pick_at_k.py --out runs/serve_pooled --timing  # also: served latency and cost
"""
import argparse
import json
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", HERE):
    sys.path.insert(0, str(p))
CEIL = REPO / "runs" / "qwen9b_methods_2026-10-19" / "ceiling"
KS = (1, 2, 4, 8, 16, 32, 64, 128)


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def expand(row):
    """A row's samples as a list of candidate indices (one per sample), plus the
    distinct candidates; unparsed samples are None (they can be drawn, never picked)."""
    cands = [{"mode": c["mode"], "command": c["command"], "label": int(c["state"] == "match")}
             for c in row["commands"] if c.get("mode") and c.get("command") is not None]
    pop = []
    for idx, c in enumerate(row["commands"]):
        if c.get("mode") and c.get("command") is not None:
            pop.extend([idx] * c["n"])
    pop.extend([None] * (row["n"] - len(pop)))
    return cands, pop


def draw_eval(draws, cands_by_model, scores, rng, k):
    """One draw of k samples per model -> (pick right, pass, maj right) for the pooled
    candidates of the drawn samples. draws: {model: population list}; k an int, or
    {model: k} for an asymmetric draw."""
    counts = {}
    for model, pop in draws.items():
        kk = k[model] if isinstance(k, dict) else k
        for idx in rng.sample(pop, min(kk, len(pop))):
            if idx is None:
                continue
            c = cands_by_model[model][idx]
            key = (c["mode"], c["command"])
            if key not in counts:
                counts[key] = [0, c["label"]]
            counts[key][0] += 1
    if not counts:
        return 0, 0, 0
    pick = max(counts, key=lambda key: (scores[key], counts[key][0], key))
    maj = max(counts, key=lambda key: (counts[key][0], key))
    return counts[pick][1], int(any(v[1] for v in counts.values())), counts[maj][1]


def paired_delta(a, b, rng, n_boot=2000):
    """Mean of b - a over rows, with a 95% percentile bootstrap CI over rows."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(d) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


CONFIGS = (("4B", ("4b",)), ("9B", ("9b",)), ("4B + 9B", ("4b", "9b")))


def curves_for(ids, rows, score_of, draws, seed="pick@k|"):
    """Summed (pick, pass, maj) per config and k over the rows, and each row's share of
    right picks. rows: {model: {id: row}}, each row's n samples in the ceiling format;
    score_of(id, mode, command) -> the verifier's score. Below k = 128, `draws` random
    draws of k samples per model; at 128 every sample is drawn, so one."""
    curves = {name: {k: [0.0, 0.0, 0.0] for k in KS} for name, _ in CONFIGS}
    per_row = {name: {k: [] for k in KS} for name, _ in CONFIGS}
    for i in ids:
        cb, pops = {}, {}
        for m in ("4b", "9b"):
            cb[m], pops[m] = expand(rows[m][i])
        scores = {(c["mode"], c["command"]): score_of(i, c["mode"], c["command"]) for m in cb for c in cb[m]}
        rng = random.Random(seed + i)
        for name, models in CONFIGS:
            for k in KS:
                reps = 1 if k == KS[-1] else draws
                acc = [0, 0, 0]
                for _ in range(reps):
                    r = draw_eval({m: pops[m] for m in models}, cb, scores, rng, k)
                    acc = [a + b for a, b in zip(acc, r)]
                curves[name][k] = [x + a / reps for x, a in zip(curves[name][k], acc)]
                per_row[name][k].append(acc[0] / reps)
    return curves, per_row


def load_timing(path):
    """{k: timing entry} from a pick_at_k.json, or {} (json keys are strings)."""
    if not pathlib.Path(path).exists():
        return {}
    return {int(k): v for k, v in json.loads(pathlib.Path(path).read_text()).get("timing", {}).items()}


def timing_lines(timing, res, gpu_hour):
    """The served latency and cost tables for every timed k. Cost per solved task divides the
    mean latency (one request holds the card at k = 8: measure.json) by the pooled pick@k in
    `res`; the entries gain the two cost fields."""
    if not timing:
        return []
    ks = sorted(timing)
    lines = ["", "## Served latency, one request at a time", "",
             "Each k is timed on the first %s val rows on an idle card; `measured` says when." %
             " / ".join(sorted({str(timing[k].get("rows", "?")) for k in ks})), "",
             "| k per model | p50 | p95 | sampling p50 | scoring p50 | distinct candidates | tokens generated | measured |",
             "|---|---|---|---|---|---|---|---|"]
    for k in ks:
        r = timing[k]
        lines.append("| %d | %.2f s | %.2f s | %.2f s | %.2f s | %.1f | %.0f | %s |" % (
            k, r["p50_s"], r["p95_s"], r["sample_p50_s"], r["score_p50_s"], r["candidates_mean"], r["tokens_mean"],
            r.get("measured", "")))
    lines += ["", "## Cost per solved task", "",
              "GPU-seconds per request = the mean latency above (one request holds the card); per solved task, "
              "divided by the pooled pick@k on test v2; dollars at $%.2f per GPU-hour. Latency is timed on val "
              "rows, accuracy read on test rows (the 128 samples exist only there)." % gpu_hour, "",
              "| k per model | GPU-s per request | pooled pick@k | GPU-s per solved | $ per 1,000 solved |",
              "|---|---|---|---|---|"]
    for k in ks:
        r = timing[k]
        acc = res["4B + 9B"][k]["pick"] / 100
        r["gpu_s_per_solved"] = round(r["mean_s"] / acc, 3)
        r["usd_per_1000_solved"] = round(r["mean_s"] / acc * 1000 / 3600 * gpu_hour, 4)
        lines.append("| %d | %.2f | %.1f%% | %.2f | $%.3f |" % (
            k, r["mean_s"], 100 * acc, r["gpu_s_per_solved"], r["usd_per_1000_solved"]))
    return lines


def write(out, lines, report):
    (out / "pick_at_k.md").write_text("\n".join(lines) + "\n")
    (out / "pick_at_k.json").write_text(json.dumps(report, indent=1) + "\n")


def curve_report(curves, per_row, n, title, intro, h="#", ks=KS):
    """(markdown lines, json dict): the pick / pass / maj table, then pick@k against
    the same config's pick@8, paired over rows with a 95% bootstrap CI; rows for `ks`."""
    lines = [h + " " + title, "", intro, "",
             "| k per model | 4B pick / pass / maj | 9B pick / pass / maj | 4B + 9B pick / pass / maj |", "|---|---|---|---|"]
    res = {}
    for k in ks:
        cells = []
        for name, _ in CONFIGS:
            p, a, mj = (100 * x / n for x in curves[name][k])
            res.setdefault(name, {})[k] = {"pick": round(p, 1), "pass": round(a, 1), "maj": round(mj, 1)}
            cells.append("**%.1f** / %.1f / %.1f" % (p, a, mj))
        lines.append("| %d | %s |" % (k, " | ".join(cells)))
    report = {"curves": res, "pick_vs_k8": {}}
    lines += ["", h + "# pick@k against pick@8, paired over rows", "",
              "Points over the same config's pick@8; 95%% bootstrap CI over the %d rows (2,000 resamples)." % n, "",
              "| k per model | 4B | 9B | 4B + 9B |", "|---|---|---|---|"]
    brng = random.Random("bootstrap")
    for k in ks:
        if k == 8:
            continue
        cells = []
        for name, _ in CONFIGS:
            d, lo, hi = (100 * x for x in paired_delta(per_row[name][8], per_row[name][k], brng))
            report["pick_vs_k8"].setdefault(name, {})[k] = {"delta": round(d, 1), "ci": [round(lo, 1), round(hi, 1)]}
            cells.append("%+.1f (%+.1f to %+.1f)" % (d, lo, hi))
        lines.append("| %d | %s |" % (k, " | ".join(cells)))
    return lines, report


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "serve_pooled"))
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--timing", action="store_true", help="also time served requests at --timing-ks")
    ap.add_argument("--timing-ks", default="1,4,8,32,64,128", help="samples per model to time (each must be in KS)")
    ap.add_argument("--timing-n", type=int, default=30)
    ap.add_argument("--gpu-hour", type=float, default=0.30, help="USD per GPU-hour for the dollar column (as measure.py)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    prev_timing = load_timing(out / "pick_at_k.json")
    import endpoint as E
    import pooled as P
    e4 = E.load("qwen3.5-4b", "http://127.0.0.1:8011", "qwen3.5-4b-served")
    e9 = E.load("qwen3.5-9b", "http://127.0.0.1:8012", "served9b")
    pp = P.Pooled(e4, e9)
    test = {c["id"]: c for c in load_jsonl(REPO / "data" / "splits" / "test.jsonl")}
    rows = {m: {r["id"]: r for r in load_jsonl(CEIL / ("samples_served%s.jsonl" % m))} for m in ("4b", "9b")}
    ids = [i for i in test if all(i in rows[m] for m in rows)]

    # 1. verifier scores for every distinct command of either model (resumable)
    score_path = out / "scores_128.jsonl"
    done = {}
    if score_path.exists():
        for r in load_jsonl(score_path):
            done[(r["id"], r["mode"], r["command"])] = r["score"]
    t0 = time.time()
    with score_path.open("a") as fh:
        for n_row, i in enumerate(ids):
            want, seen = [], set()
            for m in ("4b", "9b"):
                for c in rows[m][i]["commands"]:
                    key = (c.get("mode"), c.get("command"))
                    if key[0] and key[1] is not None and key not in seen and (i,) + key not in done:
                        seen.add(key)
                        want.append({"mode": key[0], "command": key[1]})
            for s in range(0, len(want), 64):
                chunk = want[s:s + 64]
                sc, _ = pp.scores(test[i], chunk)
                for c, v in zip(chunk, sc):
                    done[(i, c["mode"], c["command"])] = v
                    fh.write(json.dumps({"id": i, "mode": c["mode"], "command": c["command"], "score": v}) + "\n")
            if (n_row + 1) % 50 == 0:
                print("  scored %d / %d rows, %d candidates, %.0f s" % (n_row + 1, len(ids), len(done), time.time() - t0), flush=True)
    print("scores: %d candidates in %.0f s" % (len(done), time.time() - t0), flush=True)

    # 2. the curves
    curves, per_row = curves_for(ids, rows, lambda i, mode, cmd: done[(i, mode, cmd)], args.draws)
    lines, report = curve_report(
        curves, per_row, len(ids), "The verifier's pick as samples grow (test v2, %d rows)" % len(ids),
        "From the ceiling run's 128 samples per row per model; %d random draws per k below 128. "
        "pick@k: the verifier's choice is right; pass@k: some drawn sample is right; maj@k: the most common "
        "drawn command is right. For 4B + 9B, k samples from each model." % args.draws)
    report["timing"] = prev_timing          # earlier timings are kept; a new one replaces only its own k
    # written before any timing, so a stop there keeps the curves
    write(out, lines + timing_lines(report["timing"], report["curves"], args.gpu_hour), report)

    # 3. served latency and cost, one request at a time
    if args.timing:
        val = load_jsonl(REPO / "data" / "splits" / "val.jsonl")[:args.timing_n]
        for k in (int(x) for x in args.timing_ks.split(",")):
            pp.k = k
            rs = [pp.answer(c) for c in val]
            lat = sorted(r["latency_s"] for r in rs)
            row = {"mean_s": round(statistics.fmean(lat), 3), "p50_s": round(statistics.median(lat), 3),
                   "p95_s": round(lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))], 3),
                   "sample_p50_s": round(statistics.median(r["sample_s"] for r in rs), 3),
                   "score_p50_s": round(statistics.median(r["score_s"] for r in rs), 3),
                   "candidates_mean": round(statistics.fmean(len(r["candidates"]) for r in rs), 1),
                   "tokens_mean": round(statistics.fmean(sum(r["output_tokens"].values()) for r in rs), 0),
                   "rows": len(val), "measured": time.strftime("%Y-%m-%d %H:%M")}
            report["timing"][k] = row
            print("timing k=%d" % k, row, flush=True)
            write(out, lines + timing_lines(report["timing"], report["curves"], args.gpu_hour), report)
    lines += timing_lines(report["timing"], report["curves"], args.gpu_hour)
    write(out, lines, report)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
