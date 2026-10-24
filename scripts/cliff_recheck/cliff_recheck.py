#!/usr/bin/env python3
"""Re-check the never-solved train prompts with 32 samples.

An 8-sample probe files a prompt on the cliff (never solved) when all 8
samples fail. A prompt the policy solves once in 30 tries lands there about
three times in four, so the cliff mixes prompts no sample solves with prompts
solved rarely. Measured on test (2026-10-15, the served adapter's 128-sample
ceiling run): about 28% of what an 8-sample check calls never solved is
solvable.

This module picks every train prompt the latest full probe of the served
adapter put at 0 of 8 (`ids`), and after `scripts/probe_difficulty.py` has
re-sampled them at G = 32, reports how much of the cliff is rare rather than
never (`report`). That is the size of "bigger groups for near-cliff prompts":
a prompt RL cannot learn from at G = 8 but can at G = 32. The report also
shows the shelves the gated sampler sees when the re-check is pooled with the
probe -- pass both files to `train_grpo_gated.py --difficulty`, probe first.

    python scripts/cliff_recheck/cliff_recheck.py ids \\
        --probe runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-leg2.jsonl \\
        --out runs/cliff_recheck_2026-10-16/recheck_ids.txt
    python scripts/cliff_recheck/cliff_recheck.py report \\
        --probe runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-leg2.jsonl \\
        --recheck runs/cliff_recheck_2026-10-16/recheck_g32_qwen3.5-4b.jsonl \\
        --out runs/cliff_recheck_2026-10-16/report.md

Read-only on everything but its --out files.
"""
import argparse
import collections
import json
import math
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from difficulty import DifficultyBook                      # noqa: E402

DEFAULT_PROBE = REPO / "runs" / "grpo_gated_2026-09-14" / "difficulty_probe_qwen3.5-4b-leg2.jsonl"
DEFAULT_CASES = REPO / "data" / "splits" / "train.jsonl"
K_BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 32))
CONSISTENT_SHARE = 0.75      # one runnable wrong command in >= 75% of samples
GROUP_SIZES = (8, 16, 32, 64)


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def never_solved(probe_rows, train_ids):
    """Ids at 0 successes in the probe, restricted to the current train split,
    in the probe's order. Returns (ids, skipped_not_in_train)."""
    out, skipped, seen = [], 0, set()
    for r in probe_rows:
        if int(r["k"]) != 0 or r["id"] in seen:
            continue
        seen.add(r["id"])
        if r["id"] not in train_ids:
            skipped += 1
            continue
        out.append(r["id"])
    return out, skipped


def wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion, in [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def k_bin(k):
    for lo, hi in K_BINS:
        if lo <= k <= hi:
            return "%d" % lo if lo == hi else "%d-%d" % (lo, hi)
    raise ValueError(k)


def consistent_wrong(row, share=CONSISTENT_SHARE):
    """The one runnable wrong command holding >= share of the samples, if any.
    These are the rows the 2026-09-16 cliff review found to be instruction or
    gold questions as often as model errors."""
    g = int(row["G"])
    best = None
    for c in row.get("commands") or []:
        if c.get("state") == "valid_wrong" and c.get("n", 0) >= share * g:
            if best is None or c["n"] > best["n"]:
                best = c
    return best


def per_visit(p, g):
    """Chance a group of g samples holds at least one success."""
    return 1 - (1 - p) ** g


def pct(x, digits=1):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else "%.*f%%" % (digits, 100 * x)


def cmd_ids(args):
    probe = load_jsonl(args.probe)
    train = {r["id"] for r in load_jsonl(args.cases)}
    ids, skipped = never_solved(probe, train)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(i + "\n" for i in ids), encoding="utf-8")
    by = {r["id"]: r for r in probe}
    manifest = {
        "probe": str(args.probe), "cases": str(args.cases), "ids": len(ids),
        "skipped_not_in_train": skipped,
        "probe_G": sorted({int(r["G"]) for r in probe}),
        "by_category": dict(collections.Counter(by[i].get("category", "?") for i in ids).most_common()),
        "by_tier": dict(sorted(collections.Counter(by[i].get("tier", "?") for i in ids).items())),
    }
    if args.manifest:
        pathlib.Path(args.manifest).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print("%d never-solved train prompts -> %s (%d probe ids not in the current train split)"
          % (len(ids), out, skipped))
    return 0


def summarize(recheck_rows, ids):
    """The numbers the report is built from, separate from formatting so the
    tests can pin them."""
    by = {}
    for r in recheck_rows:
        by[r["id"]] = r                      # a resumed probe never repeats an id; last wins
    rows = [by[i] for i in ids if i in by]
    n = len(rows)
    solved = [r for r in rows if int(r["k"]) > 0]
    p_solved = [int(r["k"]) / int(r["G"]) for r in solved]
    out = {
        "planned": len(ids), "rechecked": n, "missing": len(ids) - n,
        "G": sorted({int(r["G"]) for r in rows}),
        "solvable": len(solved),
        "solvable_share": len(solved) / n if n else float("nan"),
        "solvable_ci": wilson(len(solved), n),
        "k_hist": collections.Counter(k_bin(int(r["k"])) for r in rows),
        "median_p_solvable": statistics.median(p_solved) if p_solved else None,
        "per_visit": {g: (statistics.mean(per_visit(p, g) for p in p_solved) if p_solved else None)
                      for g in GROUP_SIZES},
        "refiled_at_8": (statistics.mean((1 - p) ** 8 for p in p_solved) if p_solved else None),
        "consistent_wrong": [],
    }
    for r in rows:
        if int(r["k"]) == 0:
            c = consistent_wrong(r)
            if c is not None:
                out["consistent_wrong"].append((r["id"], r.get("category", "?"), c["command"], c["n"], int(r["G"])))
    for key in ("category", "tier", "mode"):
        table = collections.defaultdict(lambda: [0, 0, []])
        for r in rows:
            t = table[r.get(key, "?")]
            t[0] += 1
            if int(r["k"]) > 0:
                t[1] += 1
                t[2].append(int(r["k"]) / int(r["G"]))
        out["by_" + key] = {v: (t[0], t[1], statistics.median(t[2]) if t[2] else None)
                            for v, t in table.items()}
    return out


def shelves(train_ids, files, pool="last:2"):
    book = DifficultyBook(train_ids, pool=pool)
    for f in files:
        book.load(f)
    return book.counts()


def cmd_report(args):
    probe = load_jsonl(args.probe)
    train_ids = [r["id"] for r in load_jsonl(args.cases)]
    ids = ([line.strip() for line in open(args.ids, encoding="utf-8") if line.strip()]
           if args.ids else never_solved(probe, set(train_ids))[0])
    recheck = load_jsonl(args.recheck)
    s = summarize(recheck, ids)
    before = shelves(train_ids, [args.probe])
    after = shelves(train_ids, [args.probe, args.recheck])
    n = s["rechecked"]
    lo, hi = s["solvable_ci"]
    g = "/".join(str(x) for x in s["G"]) or "?"
    L = ["# Never-solved train prompts, re-checked at G = %s" % g, "",
         "Generated by `scripts/cliff_recheck/cliff_recheck.py report`.",
         "",
         "- Selection: the %d train prompts at 0 of 8 in `%s`." % (s["planned"], pathlib.Path(args.probe).name),
         "- Re-check: `%s`, %d rows%s." % (pathlib.Path(args.recheck).name, n,
                                            "" if not s["missing"] else ", **%d still missing**" % s["missing"]),
         "",
         "## How much of the cliff is rare rather than never", "",
         "| | |", "|---|---|",
         "| re-checked | %d |" % n,
         "| solved at least once in %s | **%d (%s; 95%% CI %s–%s)** |" % (g, s["solvable"], pct(s["solvable_share"]), pct(lo), pct(hi)),
         "| median success rate of those | %s |" % pct(s["median_p_solvable"]),
         "| chance a fresh 8-sample probe files them on the cliff again | %s |" % pct(s["refiled_at_8"]),
         "",
         "Successes out of %s:" % g, "",
         "| k | " + " | ".join(k_bin(lo_) if lo_ == hi_ else "%d–%d" % (lo_, hi_) for lo_, hi_ in K_BINS) + " |",
         "|---|" + "---|" * len(K_BINS),
         "| prompts | " + " | ".join(str(s["k_hist"].get(k_bin(lo_), 0)) for lo_, _ in K_BINS) + " |",
         "",
         "## What it means for bigger groups", "",
         "Chance that one visit's group holds a success, for the prompts solved at least once here",
         "(each prompt's rate estimated from its own samples):", "",
         "| group size | " + " | ".join("G = %d" % x for x in GROUP_SIZES) + " |",
         "|---|" + "---|" * len(GROUP_SIZES),
         "| visits that carry a gradient | " + " | ".join(pct(s["per_visit"][x]) for x in GROUP_SIZES) + " |",
         "",
         "A prompt left at 0 here can still be solvable: 0 of %s rules out rates above about %s at 95%%." % (
             g, pct(1 - 0.05 ** (1 / s["G"][0])) if s["G"] else "—"),
         "",
         "## The gated sampler's shelves, before and after pooling the re-check", "",
         "The pool is `last:2`, so a re-checked prompt's shelf is decided by its 8 + %s samples." % g, "",
         "| shelf | probe alone | probe + re-check |", "|---|---|---|"]
    for sh in ("cliff", "frontier", "known", "unprobed"):
        L.append("| %s | %d | %d |" % (sh, before.get(sh, 0), after.get(sh, 0)))
    L += ["", "To use it: `train_grpo_gated.py --difficulty %s %s`." % (args.probe, args.recheck), ""]
    for key, title in (("tier", "tier"), ("category", "category"), ("mode", "gold mode")):
        L += ["## By %s" % title, "",
              "| %s | re-checked | solved at least once | share | median rate of those |" % title,
              "|---|---|---|---|---|"]
        rows_ = sorted(s["by_" + key].items(), key=lambda kv: (-kv[1][0], kv[0]))
        for v, (tot, sol, med) in rows_:
            L.append("| %s | %d | %d | %s | %s |" % (v, tot, sol, pct(sol / tot if tot else float("nan")), pct(med)))
        L.append("")
    cw = s["consistent_wrong"]
    L += ["## Still never solved, with one stable wrong answer", "",
          "%d prompts at 0 of %s give one runnable wrong command in at least %d%% of their samples. "
          "In the 2026-09-16 cliff review this shape was as often an instruction or gold question as a "
          "model error, so these belong to a review before any training targets them." % (
              len(cw), g, round(100 * CONSISTENT_SHARE)), ""]
    if cw:
        L += ["| id | category | the stable answer | samples |", "|---|---|---|---|"]
        for cid, cat, cmd, k, gg in sorted(cw, key=lambda t: (-t[3], t[0]))[:args.list]:
            L.append("| `%s` | %s | `%s` | %d of %d |" % (cid, cat, cmd.replace("|", "\\|"), k, gg))
        if len(cw) > args.list:
            L.append("| … %d more | | | |" % (len(cw) - args.list))
        L.append("")
    pathlib.Path(args.out).write_text("\n".join(L), encoding="utf-8")
    if args.json:
        dump = dict(s)
        dump["k_hist"] = dict(s["k_hist"])
        dump["shelves_before"], dump["shelves_after"] = before, after
        pathlib.Path(args.json).write_text(json.dumps(dump, indent=1, default=str) + "\n", encoding="utf-8")
    print("report -> %s: %d of %d re-checked prompts solved at least once (%s)"
          % (args.out, s["solvable"], n, pct(s["solvable_share"])))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ids", help="never-solved train ids from a probe")
    a.add_argument("--probe", default=str(DEFAULT_PROBE))
    a.add_argument("--cases", default=str(DEFAULT_CASES))
    a.add_argument("--out", required=True)
    a.add_argument("--manifest", default=None)
    b = sub.add_parser("report", help="what the re-check found")
    b.add_argument("--probe", default=str(DEFAULT_PROBE))
    b.add_argument("--cases", default=str(DEFAULT_CASES))
    b.add_argument("--ids", default=None, help="the ids file `ids` wrote (default: recompute)")
    b.add_argument("--recheck", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--json", default=None)
    b.add_argument("--list", type=int, default=40, help="rows of the stable-wrong table")
    args = ap.parse_args(argv)
    return cmd_ids(args) if args.cmd == "ids" else cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
