#!/usr/bin/env python3
"""Where the verifier's pick gains and loses: pick@k by tier and category, and asymmetric
sampling (test v2, CPU only).

From the ceiling run's 128 samples per row per model and the saved verifier scores
(runs/serve_pooled/scores_128.jsonl); no model runs. 20 random draws per row and setting
(one at 128, where every sample is drawn).
  tiers / categories  the pooled pick / pass / maj and each model's pick at k = 1, 4, 8, 32, 128;
                      the verifier's gap (pass − pick); 128 vs 8 and 4 vs 8 paired per tier.
  asymmetric          k4 samples from the 4B and k9 from the 9B, each in {0, 1, 2, 4, 8, 16, 32}:
                      pooled pick and pass per cell, the generated tokens per request from each
                      model's mean tokens per sample, the settings no other beats on both, and
                      a few settings by tier.

    python scripts/serve_pooled/breakdown.py --out runs/serve_pooled/breakdown
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
import pick_at_k as K   # noqa: E402

KS = (1, 4, 8, 32, 128)
GRID = (0, 1, 2, 4, 8, 16, 32)
BY_TIER = ((8, 8), (8, 4), (8, 2), (8, 1), (4, 4), (4, 2), (4, 1), (4, 8), (2, 8), (1, 8))
TIERS = ("T1", "T2", "T3")


def load_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def mean_by(values, groups):
    """{group: (mean, n)} of per-id values, grouped by groups[id]."""
    out = {}
    for i, v in values.items():
        out.setdefault(groups[i], []).append(v)
    return {g: (statistics.fmean(v), len(v)) for g, v in out.items()}


def per_row(ids, rows, scores_of, draws):
    """Per id: draw means of (pick, pass, maj) for each config and k, and for each (k4, k9)."""
    res = {}
    for i in ids:
        cb, pops = {}, {}
        for m in ("4b", "9b"):
            cb[m], pops[m] = K.expand(rows[m][i])
        scores = {(c["mode"], c["command"]): scores_of(i, c["mode"], c["command"]) for m in cb for c in cb[m]}
        rng = random.Random("breakdown|" + i)
        sym, asym = {}, {}
        for name, models in K.CONFIGS:
            sym[name] = {}
            for k in KS:
                reps = 1 if k == 128 else draws
                tot = [0, 0, 0]
                for _ in range(reps):
                    tot = [a + b for a, b in zip(tot, K.draw_eval({m: pops[m] for m in models}, cb, scores, rng, k))]
                sym[name][k] = [t / reps for t in tot]
        for k4 in GRID:
            for k9 in GRID:
                if k4 == 0 and k9 == 0:
                    continue
                tot = [0, 0, 0]
                for _ in range(draws):
                    tot = [a + b for a, b in zip(tot, K.draw_eval(pops, cb, scores, rng, {"4b": k4, "9b": k9}))]
                asym[(k4, k9)] = [t / draws for t in tot]
        res[i] = {"sym": sym, "asym": asym}
    return res


def pct(x):
    return "%.1f" % (100 * x)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "serve_pooled" / "breakdown"))
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--served", help="a served_at_k rows file (rows_<split>.jsonl) instead of the ceiling samples")
    ap.add_argument("--split", default="test", help="the split the rows belong to (its tiers and categories)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    test = {c["id"]: c for c in load_jsonl(REPO / "data" / "splits" / ("%s.jsonl" % args.split))}
    if args.served:
        import served_at_k as A
        served = {}
        for r in load_jsonl(args.served):
            served.setdefault(r["id"], r)
        rows = {m: {i: A.ceiling_row(r, m) for i, r in served.items()} for m in ("4b", "9b")}
        done = {(i,) + key: v for i, r in served.items() for key, v in A.row_maps(r)[0].items()}
        for m in rows:                      # tokens per sample: the request's total over its samples
            for i, r in served.items():
                rows[m][i]["tokens_mean"] = sum(r["output_tokens"][m]) / max(1, len(r["output_tokens"][m]))
    else:
        rows = {m: {r["id"]: r for r in load_jsonl(K.CEIL / ("samples_served%s.jsonl" % m))} for m in ("4b", "9b")}
        done = {(r["id"], r["mode"], r["command"]): r["score"]
                for r in load_jsonl(REPO / "runs" / "serve_pooled" / "scores_128.jsonl")}
    ids = [i for i in test if all(i in rows[m] for m in rows)]
    R = per_row(ids, rows, lambda i, mode, cmd: done[(i, mode, cmd)], args.draws)
    tier = {i: test[i]["tier"] for i in ids}
    cat = {i: test[i]["category"] for i in ids}
    n = len(ids)
    tok = {m: statistics.fmean(rows[m][i]["tokens_mean"] for i in ids) for m in ("4b", "9b")}
    rep = {"rows": n, "draws": args.draws, "tokens_per_sample": {m: round(v, 1) for m, v in tok.items()}}
    src = ("the served pass's 128 samples per row per model (%s)" % args.served if args.served
           else "the ceiling run's 128 samples per row per model and the saved verifier scores")
    L = ["# Pick@k by tier and category, and asymmetric sampling (%s, %d rows)" % (args.split, n), "",
         "From %s; %d draws per setting and row. pick: the verifier's choice is right; pass: some drawn sample "
         "is right; maj: the most common command is right. Pooled = k samples from each model." % (src, args.draws), ""]

    # 1. by tier
    rep["by_tier"] = {}
    for t in TIERS:
        sub = [i for i in ids if tier[i] == t]
        rep["by_tier"][t] = {"n": len(sub)}
        L += ["## %s (%d rows)" % (t, len(sub)), "",
              "| k per model | 4B pick | 9B pick | pooled pick | pooled pass | pooled maj | verifier gap (pass − pick) |",
              "|---|---|---|---|---|---|---|"]
        for k in KS:
            v = {name: [statistics.fmean(R[i]["sym"][name][k][j] for i in sub) for j in range(3)] for name, _ in K.CONFIGS}
            rep["by_tier"][t][k] = {name: [round(100 * x, 1) for x in v[name]] for name in v}
            L.append("| %d | %s | %s | **%s** | %s | %s | %.1f |" % (
                k, pct(v["4B"][0]), pct(v["9B"][0]), pct(v["4B + 9B"][0]), pct(v["4B + 9B"][1]), pct(v["4B + 9B"][2]),
                100 * (v["4B + 9B"][1] - v["4B + 9B"][0])))
        L.append("")
    L += ["## Pooled pick against 8 per model, paired within each tier", "",
          "Points; 95% bootstrap CI over the tier's rows.", "",
          "| tier | n | 128 vs 8 | 4 vs 8 | 1 vs 8 |", "|---|---|---|---|---|"]
    brng = random.Random("breakdown-boot")
    rep["paired_by_tier"] = {}
    for t in TIERS + ("all",):
        sub = [i for i in ids if t == "all" or tier[i] == t]
        base = [R[i]["sym"]["4B + 9B"][8][0] for i in sub]
        cells = []
        rep["paired_by_tier"][t] = {}
        for k in (128, 4, 1):
            d, lo, hi = (100 * x for x in K.paired_delta(base, [R[i]["sym"]["4B + 9B"][k][0] for i in sub], brng))
            rep["paired_by_tier"][t][k] = [round(d, 1), round(lo, 1), round(hi, 1)]
            cells.append("%+.1f (%+.1f to %+.1f)" % (d, lo, hi))
        L.append("| %s | %d | %s |" % (t, len(sub), " | ".join(cells)))

    # 2. by category
    cats = sorted(set(cat.values()), key=lambda c: (-sum(1 for i in ids if cat[i] == c), c))
    L += ["", "## By category (pooled unless marked; sorted by size)", "",
          "| category | n | 4B pick@8 | 9B pick@8 | pooled pick@1 | pick@4 | pick@8 | pick@32 | pick@128 | pass@8 | pass@128 | gap@128 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rep["by_category"] = {}
    for c in cats:
        sub = [i for i in ids if cat[i] == c]
        g = lambda name, k, j: statistics.fmean(R[i]["sym"][name][k][j] for i in sub)   # noqa: E731
        row = {"n": len(sub), "4B_pick8": g("4B", 8, 0), "9B_pick8": g("9B", 8, 0),
               "pick": {k: g("4B + 9B", k, 0) for k in KS}, "pass8": g("4B + 9B", 8, 1), "pass128": g("4B + 9B", 128, 1)}
        rep["by_category"][c] = {k: (round(100 * v, 1) if isinstance(v, float) else
                                     ({kk: round(100 * vv, 1) for kk, vv in v.items()} if isinstance(v, dict) else v))
                                 for k, v in row.items()}
        L.append("| %s | %d | %s | %s | %s | %s | **%s** | %s | %s | %s | %s | %.1f |" % (
            c, len(sub), pct(row["4B_pick8"]), pct(row["9B_pick8"]), pct(row["pick"][1]), pct(row["pick"][4]),
            pct(row["pick"][8]), pct(row["pick"][32]), pct(row["pick"][128]), pct(row["pass8"]), pct(row["pass128"]),
            100 * (row["pass128"] - row["pick"][128])))

    # 3. asymmetric grid
    grid = {}
    for k4 in GRID:
        for k9 in GRID:
            if k4 == 0 and k9 == 0:
                continue
            v = [statistics.fmean(R[i]["asym"][(k4, k9)][j] for i in ids) for j in range(3)]
            grid[(k4, k9)] = {"pick": v[0], "pass": v[1], "maj": v[2], "tokens": k4 * tok["4b"] + k9 * tok["9b"]}
    rep["asym_grid"] = {"%d+%d" % kk: {a: round(100 * b, 1) if a != "tokens" else round(b) for a, b in v.items()}
                        for kk, v in grid.items()}
    for what in ("pick", "pass"):
        L += ["", "## Asymmetric sampling: pooled %s, rows = samples from the 4B, columns = samples from the 9B" % what, "",
              "| 4B \\ 9B | " + " | ".join(str(k) for k in GRID) + " |", "|---|" + "---|" * len(GRID)]
        for k4 in GRID:
            cells = []
            for k9 in GRID:
                if k4 == 0 and k9 == 0:
                    cells.append("—")
                else:
                    s = pct(grid[(k4, k9)][what])
                    cells.append("**%s**" % s if k4 == k9 == 8 else s)
            L.append("| %d | %s |" % (k4, " | ".join(cells)))
    L += ["", "Generated tokens per request = k4 × %.1f + k9 × %.1f (each model's mean tokens per sample over the "
          "test rows); the 9B's samples also cost about 3× the 4B's GPU time each (serving section, RESULTS.md)."
          % (tok["4b"], tok["9b"])]
    # settings no other setting beats on both pick and tokens
    front = []
    for kk, v in grid.items():
        if not any((o["pick"] >= v["pick"] and o["tokens"] < v["tokens"]) or (o["pick"] > v["pick"] and o["tokens"] <= v["tokens"])
                   for ok, o in grid.items() if ok != kk):
            front.append(kk)
    front.sort(key=lambda kk: grid[kk]["tokens"])
    rep["front"] = ["%d+%d" % kk for kk in front]
    L += ["", "**Settings no other setting beats on both pick and tokens** (4B + 9B samples → pick, tokens per request):", ""]
    L += ["- %d + %d → %s%%, %.0f tokens" % (kk[0], kk[1], pct(grid[kk]["pick"]), grid[kk]["tokens"]) for kk in front]

    # 4. a few asymmetric settings by tier
    L += ["", "## Asymmetric settings by tier (pooled pick)", "",
          "| 4B + 9B samples | tokens | all | " + " | ".join("%s (n = %d)" % (t, rep["by_tier"][t]["n"]) for t in TIERS) + " |",
          "|---|---|---|---|---|---|"]
    rep["asym_by_tier"] = {}
    for k4, k9 in BY_TIER:
        allv = grid[(k4, k9)]["pick"]
        per = mean_by({i: R[i]["asym"][(k4, k9)][0] for i in ids}, tier)
        rep["asym_by_tier"]["%d+%d" % (k4, k9)] = {"all": round(100 * allv, 1), **{t: round(100 * per[t][0], 1) for t in TIERS}}
        L.append("| %d + %d | %.0f | **%s** | %s |" % (k4, k9, grid[(k4, k9)]["tokens"], pct(allv),
                                                      " | ".join(pct(per[t][0]) for t in TIERS)))
    (out / "report.md").write_text("\n".join(L) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
