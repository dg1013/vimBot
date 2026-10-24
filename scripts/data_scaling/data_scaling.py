#!/usr/bin/env python3
"""Data-scaling SFT points: does more train data still pay? (plan item 6's
sizing experiment, 2026-10-16)

The served 4B is right by majority-of-8 on 78% of train prompts and greedy-right
on about 53-58% of held-out prompts at the same tier mix: the largest piece of
the gap is trained-on vs held-out. More distinct prompts would attack it, but
authoring them is expensive, so first measure the slope: SFT the 4B on nested
25% / 50% / 100% subsets of train with the round-1 recipe and read full val.

Subsets drop whole problem groups, not rows (owner decision 2026-10-16). A
"group" is a leakage component -- the union-find over identical instruction +
buffer, near-duplicate instruction, shared buffer, shared normalized command and
paraphrase links that `scripts/build_splits.py` uses to keep splits apart -- so
a 25% subset has a quarter of the distinct problems, not every problem with
fewer paraphrases. Components are dealt within their majority (category, tier)
stratum, in a seeded order, by the "closer" rule: take a component if the
stratum's count lands nearer its target with it than without. The 50% subset
continues the 25% subset's walk, so the subsets are nested.

    python scripts/data_scaling/data_scaling.py build --out-dir runs/data_scaling_2026-10-16
    python scripts/data_scaling/data_scaling.py report --out-dir runs/data_scaling_2026-10-16 \\
        --point 25=runs/qwen3.5-4b-sft-ds25 --point 50=runs/qwen3.5-4b-sft-ds50 \\
        --point 100=runs/qwen3.5-4b-sft-ds100
"""
import argparse
import collections
import json
import math
import pathlib
import random
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

DEFAULT_TRAIN = REPO / "data" / "splits" / "train.jsonl"
DEFAULT_VAL = REPO / "data" / "splits" / "val.jsonl"
FRACTIONS = (0.25, 0.5)
SEED = 20261016
BOOTSTRAP = 2000


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def stratum(case):
    return (case.get("category", "?"), case.get("tier") or "T?")


def components_of(cases):
    from build_splits import build_components
    return build_components(cases)


def deal(cases, comps, fractions=FRACTIONS, seed=SEED):
    """Nested whole-component subsets. Returns {fraction: sorted row indices}.

    Each component goes to its majority stratum; components within a stratum
    are walked in a seeded order. For the smallest fraction a component is
    taken when it moves the stratum's count closer to its target; each larger
    fraction keeps everything already taken and continues the same walk."""
    by_stratum = collections.defaultdict(list)
    for comp in comps:
        counts = collections.Counter(stratum(cases[i]) for i in comp)
        key = max(counts, key=lambda s: (counts[s], s))
        by_stratum[key].append(sorted(comp))
    rng = random.Random(seed)
    order = {}
    for key in sorted(by_stratum):
        lst = sorted(by_stratum[key], key=lambda c: min(cases[i]["id"] for i in c))
        rng.shuffle(lst)
        order[key] = lst
    taken = {key: set() for key in order}        # component positions taken so far
    count = {key: 0 for key in order}
    out = {}
    for frac in sorted(fractions):
        for key, lst in order.items():
            total = sum(len(c) for c in lst)
            target = frac * total
            for pos, comp in enumerate(lst):
                if pos in taken[key]:
                    continue
                if abs(count[key] + len(comp) - target) < abs(count[key] - target):
                    taken[key].add(pos)
                    count[key] += len(comp)
        out[frac] = sorted(i for key, lst in order.items() for pos in taken[key] for i in lst[pos])
    return out


def describe(cases, idx):
    rows = [cases[i] for i in idx]
    return {
        "rows": len(rows),
        "distinct_commands": len({(r.get("mode"), r.get("gold_command")) for r in rows}),
        "distinct_buffers": len({r.get("buffer_before") for r in rows}),
        "distinct_instructions": len({r.get("instruction") for r in rows}),
        "by_tier": dict(sorted(collections.Counter(r.get("tier") or "T?" for r in rows).items())),
        "by_category": dict(sorted(collections.Counter(r.get("category", "?") for r in rows).items())),
    }


def cmd_build(args):
    cases = load_jsonl(args.train)
    comps = components_of(cases)
    subsets = deal(cases, comps, seed=args.seed)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_of = {}
    for ci, comp in enumerate(comps):
        for i in comp:
            comp_of[i] = ci
    manifest = {"train": str(args.train), "seed": args.seed, "components": len(comps),
                "full": describe(cases, range(len(cases))), "subsets": {}}
    for frac, idx in subsets.items():
        name = "train_%d.jsonl" % round(100 * frac)
        with (out_dir / name).open("w", encoding="utf-8") as fh:
            for i in idx:
                fh.write(json.dumps(cases[i], ensure_ascii=False) + "\n")
        d = describe(cases, idx)
        d["file"] = name
        d["components"] = len({comp_of[i] for i in idx})
        d["row_fraction"] = len(idx) / len(cases)
        manifest["subsets"][str(frac)] = d
        print("%3d%%: %d rows (%.1f%%), %d components, %d distinct commands -> %s"
              % (round(100 * frac), len(idx), 100 * len(idx) / len(cases), d["components"],
                 d["distinct_commands"], out_dir / name))
    (out_dir / "subsets_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------ report

def fit_log2(xs, ys):
    """Least-squares y = a + b log2(x); returns (a, b)."""
    lx = [math.log2(x) for x in xs]
    mx, my = statistics.mean(lx), statistics.mean(ys)
    sxx = sum((u - mx) ** 2 for u in lx)
    b = sum((u - mx) * (v - my) for u, v in zip(lx, ys)) / sxx
    return my - b * mx, b


def bootstrap_slope(rows_by_point, xs, ids, n_boot=BOOTSTRAP, seed=SEED):
    """Paired bootstrap over val rows: resample ids, recompute each point's
    rate on the same resample, refit. Returns sorted slopes."""
    rng = random.Random(seed)
    slopes = []
    for _ in range(n_boot):
        draw = [ids[rng.randrange(len(ids))] for _ in ids]
        ys = [100 * sum(rows[i] for i in draw) / len(draw) for rows in rows_by_point]
        slopes.append(fit_log2(xs, ys)[1])
    return sorted(slopes)


def quantile(sorted_vals, q):
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def selection_subset_ids(val_cases):
    from train_sft import SUBSET_SEED, stratified_subset
    return {c["id"] for c in stratified_subset(val_cases, 250, seed=SUBSET_SEED)}


def load_point(run_dir):
    run_dir = pathlib.Path(run_dir)
    ex = {r["id"]: bool(r["match"]) for r in load_jsonl(run_dir / "exec_val.jsonl")}
    hist = load_jsonl(run_dir / "eval_history.jsonl") if (run_dir / "eval_history.jsonl").exists() else []
    sel = {}
    if (run_dir / "best" / "selection.json").exists():
        sel = json.loads((run_dir / "best" / "selection.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8")) \
        if (run_dir / "train_config.json").exists() else {}
    return ex, hist, sel, cfg


def rate(rows, ids):
    ids = [i for i in ids if i in rows]
    return 100 * sum(rows[i] for i in ids) / len(ids) if ids else float("nan")


def cmd_report(args):
    val = load_jsonl(args.val)
    val_ids = [c["id"] for c in val]
    by_id = {c["id"]: c for c in val}
    untouched = [i for i in val_ids if i not in selection_subset_ids(val)]
    out_dir = pathlib.Path(args.out_dir)
    manifest = json.loads((out_dir / "subsets_manifest.json").read_text(encoding="utf-8"))
    points = []
    for spec in args.point:
        pct_s, run_dir = spec.split("=", 1)
        pct_v = int(pct_s)
        ex, hist, sel, cfg = load_point(run_dir)
        if pct_v == 100:
            n_rows = manifest["full"]["rows"]
        else:
            n_rows = manifest["subsets"][str(pct_v / 100)]["rows"]
        points.append({"pct": pct_v, "dir": run_dir, "rows": n_rows, "exec": ex, "hist": hist,
                       "best_step": sel.get("step"), "cfg": cfg})
    points.sort(key=lambda p: p["rows"])
    xs = [p["rows"] for p in points]
    views = {"full val": val_ids, "val rows never used for selection": untouched}
    for t in ("T1", "T2", "T3"):
        views["val " + t] = [i for i in val_ids if by_id[i].get("tier") == t]
    L = ["# Data-scaling SFT points (qwen3.5-4b, round-1 recipe)", "",
         "Generated by `scripts/data_scaling/data_scaling.py report`. Nested subsets of train that "
         "keep or drop whole leakage components, stratified by category × tier "
         "(`subsets_manifest.json`). Same recipe at every point: LoRA r = 16, α = 32, "
         "3 epochs, eval once per epoch on the 250-row val subset, best adapter scored on full val.", "",
         "| point | train rows | components | distinct commands | best step | full val | untouched val (%d) |"
         % len(untouched),
         "|---|---|---|---|---|---|---|"]
    for p in points:
        if p["pct"] == 100:
            comps, dcmd = manifest["components"], manifest["full"]["distinct_commands"]
        else:
            s = manifest["subsets"][str(p["pct"] / 100)]
            comps, dcmd = s["components"], s["distinct_commands"]
        L.append("| %d%% | %d | %d | %d | %s | %.1f%% | %.1f%% |" % (
            p["pct"], p["rows"], comps, dcmd, p["best_step"],
            rate(p["exec"], val_ids), rate(p["exec"], untouched)))
    L += ["", "## The slope, per doubling of train", "",
          "Least squares on log2(train rows). The interval is a paired bootstrap over val rows "
          "(%d draws): every point is re-scored on the same resample." % args.bootstrap, "",
          "| view | rows | rates (small → full) | points per doubling | 95% CI | one more doubling | +50% rows |",
          "|---|---|---|---|---|---|---|"]
    fits = {}
    for name, ids in views.items():
        ys = [rate(p["exec"], ids) for p in points]
        a, b = fit_log2(xs, ys)
        boot = bootstrap_slope([p["exec"] for p in points], xs, ids, n_boot=args.bootstrap)
        lo, hi = quantile(boot, 0.025), quantile(boot, 0.975)
        fits[name] = {"rates": ys, "slope": b, "ci": [lo, hi], "n": len(ids)}
        L.append("| %s | %d | %s | %+.1f | %+.1f to %+.1f | %.1f%% | %.1f%% |" % (
            name, len(ids), " → ".join("%.1f" % y for y in ys), b, lo, hi,
            ys[-1] + b, ys[-1] + b * math.log2(1.5)))
    L += ["", "The last two columns extrapolate the fitted line past the full set: a straight line "
          "in log(rows) is the optimistic reading if the curve is bending, so compare the two "
          "halves of the curve before trusting them.", ""]
    if len(points) >= 3:
        y = fits["full val"]["rates"]
        L += ["Half-curve check (full val): %d%% → %d%% gave %+.1f, %d%% → %d%% gave %+.1f, per doubling."
              % (points[0]["pct"], points[1]["pct"], (y[1] - y[0]) / math.log2(xs[1] / xs[0]),
                 points[1]["pct"], points[2]["pct"], (y[2] - y[1]) / math.log2(xs[2] / xs[1])), ""]
    L += ["## Mid-training curves (val subset, the selection metric)", "",
          "| point | step: exec-match |", "|---|---|"]
    for p in points:
        curve = ", ".join("%s: %.1f" % (h.get("step"), h.get("exec_match_pct", float("nan")))
                          for h in p["hist"])
        L.append("| %d%% | %s |" % (p["pct"], curve or "—"))
    L.append("")
    L += ["## Per category (full val; categories under 25 rows are noise)", "",
          "| category | rows | " + " | ".join("%d%%" % p["pct"] for p in points) + " |",
          "|---|---|" + "---|" * len(points)]
    cats = collections.Counter(by_id[i].get("category", "?") for i in val_ids)
    for cat, n in cats.most_common():
        ids = [i for i in val_ids if by_id[i].get("category") == cat]
        L.append("| %s | %d | %s |" % (cat, n, " | ".join("%.1f" % rate(p["exec"], ids) for p in points)))
    L.append("")
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(
        {"points": [{k: v for k, v in p.items() if k not in ("exec", "hist", "cfg")} for p in points],
         "fits": fits}, indent=1) + "\n", encoding="utf-8")
    print("report -> %s" % (out_dir / "report.md"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the nested subsets")
    b.add_argument("--train", default=str(DEFAULT_TRAIN))
    b.add_argument("--out-dir", required=True)
    b.add_argument("--seed", type=int, default=SEED)
    r = sub.add_parser("report", help="read the three trained points")
    r.add_argument("--out-dir", required=True)
    r.add_argument("--val", default=str(DEFAULT_VAL))
    r.add_argument("--point", action="append", required=True, help="PCT=RUN_DIR, e.g. 25=runs/qwen3.5-4b-sft-ds25")
    r.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = ap.parse_args(argv)
    return cmd_build(args) if args.cmd == "build" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
