#!/usr/bin/env python3
"""Bigger groups for near-cliff prompts: the full run behind follow-up #1
(runs/followups_2026-10-16/preregistration.md).

The 32-sample re-check found train prompts that an 8-sample probe files as
never solved but that the served adapter solves now and then. In a group of 8
those prompts rarely show a success, so GRPO learns nothing from them; in a
group of 32 most visits do. This module builds the leg's prompt file and
judges the run:

    build    near-cliff prompts (k >= 1 of 32 in the re-check) + anchors from
             the known shelf -> a §5 train file for train_grpo_gated.py, the
             near-cliff id list, and the step count for ~3 visits per prompt
    decide   the pre-registered rule: does the re-check count as positive?
    report   test v2 and untouched val against the served adapter (paired),
             and the near-cliff prompts before vs after at G = 32

    python scripts/nearcliff_grpo/nearcliff.py decide --recheck-report runs/cliff_recheck_2026-10-16/report.json
    python scripts/nearcliff_grpo/nearcliff.py build --recheck runs/cliff_recheck_2026-10-16/recheck_g32_qwen3.5-4b.jsonl \\
        --out-dir runs/nearcliff_2026-10-16
    python scripts/nearcliff_grpo/nearcliff.py report --run-dir runs/qwen3.5-4b-grpo-nearcliff \\
        --before runs/cliff_recheck_2026-10-16/recheck_g32_qwen3.5-4b.jsonl \\
        --after runs/nearcliff_2026-10-16/after_g32_qwen3.5-4b.jsonl --out runs/nearcliff_2026-10-16/report.md
"""
import argparse
import collections
import json
import math
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

DEFAULT_PROBE = REPO / "runs" / "grpo_gated_2026-09-14" / "difficulty_probe_qwen3.5-4b-leg2.jsonl"
DEFAULT_TRAIN = REPO / "data" / "splits" / "train.jsonl"
SERVED_TEST = REPO / "runs" / "remeasure_2026-09-25" / "v2" / "qwen3.5-4b-grpo-gated__fla-kernels" / "exec_test.jsonl"
SERVED_VAL = REPO / "runs" / "clarify_2026-10-06" / "val" / "verdicts_served.jsonl"
POSITIVE_SHARE, POSITIVE_LOWER = 0.20, 0.15        # owner, 2026-10-16
PROMPTS_PER_STEP, FRONTIER_SHARE, VISITS, MAX_STEPS = 2, 0.80, 3, 600


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def decide(report):
    """The pre-registered rule for #1."""
    share = float(report["solvable_share"])
    lower = float(report["solvable_ci"][0])
    return share >= POSITIVE_SHARE and lower > POSITIVE_LOWER, share, lower


def leg_steps(n_near):
    """Optimizer steps for about VISITS visits per near-cliff prompt, when 80%
    of the 2 slots per step come from them."""
    return min(MAX_STEPS, math.ceil(VISITS * n_near / (PROMPTS_PER_STEP * FRONTIER_SHARE)))


def build_leg(train_rows, recheck_rows, probe_rows, anchor_ratio=0.5, seed=20261016):
    train = {r["id"]: r for r in train_rows}
    near = sorted({r["id"] for r in recheck_rows if int(r["k"]) > 0 and r["id"] in train})
    known = sorted({r["id"] for r in probe_rows
                    if int(r["k"]) == int(r["G"]) and r["id"] in train and r["id"] not in near})
    rng = random.Random(seed)
    anchors = sorted(rng.sample(known, min(len(known), round(anchor_ratio * len(near)))))
    rows = [dict(train[i], shelf="near-cliff") for i in near] + [dict(train[i], shelf="anchor") for i in anchors]
    return rows, near, anchors


def cmd_decide(args):
    report = json.loads(pathlib.Path(args.recheck_report).read_text(encoding="utf-8"))
    positive, share, lower = decide(report)
    print("%s: %.1f%% solvable (95%% lower bound %.1f%%); rule: >= %.0f%% and lower bound > %.0f%%"
          % ("POSITIVE" if positive else "NEGATIVE", 100 * share, 100 * lower,
             100 * POSITIVE_SHARE, 100 * POSITIVE_LOWER))
    return 0 if positive else 10


def cmd_build(args):
    rows, near, anchors = build_leg(load_jsonl(args.train), load_jsonl(args.recheck),
                                    load_jsonl(args.probe), args.anchor_ratio, args.seed)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "leg_train.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "nearcliff_ids.txt").write_text("".join(i + "\n" for i in near), encoding="utf-8")
    steps = leg_steps(len(near))
    manifest = {"near_cliff": len(near), "anchors": len(anchors), "max_steps": steps,
                "prompts_per_step": PROMPTS_PER_STEP, "num_generations": 32,
                "by_tier": dict(sorted(collections.Counter(r.get("tier") for r in rows if r["shelf"] == "near-cliff").items())),
                "by_category": dict(collections.Counter(r.get("category") for r in rows if r["shelf"] == "near-cliff").most_common())}
    (out / "leg_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (out / "max_steps.txt").write_text("%d\n" % steps, encoding="utf-8")
    print("leg: %d near-cliff + %d anchors, %d steps -> %s" % (len(near), len(anchors), steps, out / "leg_train.jsonl"))
    return 0


def mcnemar_p(g, l):
    n = g + l
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(g, l) + 1)) / 2 ** n)


def paired_row(name, base, new, ids):
    ids = [i for i in ids if i in base and i in new]
    g = sum(1 for i in ids if new[i] and not base[i])
    l = sum(1 for i in ids if base[i] and not new[i])
    b, n_ = sum(base[i] for i in ids), sum(new[i] for i in ids)
    return "| %s | %d | %.1f%% | %.1f%% | %+.1f | +%d / −%d | %.3f |" % (
        name, len(ids), 100 * b / len(ids), 100 * n_ / len(ids), 100 * (n_ - b) / len(ids), g, l, mcnemar_p(g, l))


def cmd_report(args):
    run = pathlib.Path(args.run_dir)
    test_cases = {r["id"]: r for r in load_jsonl(REPO / "data" / "splits" / "test.jsonl")}
    served_t = {r["id"]: bool(r["match"]) for r in load_jsonl(args.served_test)}
    new_t = {r["id"]: bool(r["match"]) for r in load_jsonl(run / "exec_test.jsonl")}
    L = ["# Bigger groups for near-cliff prompts: the full test", "",
         "Generated by `scripts/nearcliff_grpo/nearcliff.py report` for `%s`. Rules and expectations: "
         "`runs/followups_2026-10-16/preregistration.md`." % run.name, "",
         "## Test v2 and untouched val, against the served adapter", "",
         "| slice | rows | served | this run | Δ | gained / lost | exact McNemar p |", "|---|---|---|---|---|---|---|"]
    L.append(paired_row("test v2", served_t, new_t, list(test_cases)))
    for t in ("T1", "T2", "T3"):
        L.append(paired_row("test v2 %s" % t, served_t, new_t, [i for i, c in test_cases.items() if c.get("tier") == t]))
    if (run / "exec_val.jsonl").exists():
        from train_sft import SUBSET_SEED, stratified_subset
        val = load_jsonl(REPO / "data" / "splits" / "val.jsonl")
        sel = {c["id"] for c in stratified_subset(val, 250, seed=SUBSET_SEED)}
        served_v = {r["id"]: bool((r.get("direct") or {}).get("match")) for r in load_jsonl(args.served_val)}
        new_v = {r["id"]: bool(r["match"]) for r in load_jsonl(run / "exec_val.jsonl")}
        L.append(paired_row("val, never used for selection", served_v, new_v, [c["id"] for c in val if c["id"] not in sel]))
    L.append("")
    if args.before and args.after and pathlib.Path(args.after).exists():
        ids = [l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()] if args.ids else None
        before = {r["id"]: r for r in load_jsonl(args.before)}
        after = {r["id"]: r for r in load_jsonl(args.after)}
        ids = [i for i in (ids or sorted(after)) if i in before and i in after]

        def stats(rows):
            p = [int(r["k"]) / int(r["G"]) for r in rows]
            return (100 * sum(x > 0 for x in p) / len(p), 100 * sum(x > 0.5 for x in p) / len(p),
                    100 * sum(p) / len(p))
        b, a = stats([before[i] for i in ids]), stats([after[i] for i in ids])
        L += ["## The near-cliff prompts, before and after (G = 32 each)", "",
              "| | prompts | solved at least once | solved by a majority | mean p̂ |", "|---|---|---|---|---|",
              "| served adapter (the re-check) | %d | %.1f%% | %.1f%% | %.1f%% |" % ((len(ids),) + b),
              "| this run | %d | %.1f%% | %.1f%% | %.1f%% |" % ((len(ids),) + a), ""]
    pathlib.Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print("report -> %s" % args.out)
    return 0


def probe_stats(rows):
    p = [int(r["k"]) / int(r["G"]) for r in rows]
    return (len(p), 100 * sum(x > 0 for x in p) / len(p), 100 * sum(x > 0.5 for x in p) / len(p),
            100 * sum(p) / len(p))


def cmd_mechanism(args):
    """Did training convert the near-cliff prompts? Three G = 32 probes of the same 210:
    the re-check that selected them (biased upward: every one has k >= 1 by
    construction), a fresh probe of the served adapter, and a probe of the trained
    final adapter. The fair comparison is fresh served vs trained final."""
    ids = [l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()]
    sets = [("the re-check that selected them (biased: k ≥ 1 by construction)", args.selection),
            ("served adapter, fresh probe", args.served),
            ("trained adapter (final, step %s)" % args.final_step, args.final)]
    L = ["# The near-cliff prompts: did the leg convert them?", "",
         "Three 32-sample probes of the same %d prompts at T = 1. The selection run is biased upward "
         "(regression to the mean), so the comparison is the fresh served probe against the trained "
         "adapter. The pre-registered selection kept step 0 (val subset 53.6%% at step 0 against 48.0–49.2%% "
         "after), so the test row could not see what training did; this table can." % len(ids), "",
         "| adapter | prompts | solved at least once | solved by a majority | mean p̂ |", "|---|---|---|---|---|"]
    stats = {}
    for name, path in sets:
        rows = {r["id"]: r for r in load_jsonl(path)}
        got = [rows[i] for i in ids if i in rows]
        stats[name] = probe_stats(got)
        L.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% |" % ((name,) + stats[name]))
    L.append("")
    pathlib.Path(args.out).write_text("\n".join(L), encoding="utf-8")
    print("mechanism -> %s" % args.out)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mechanism")
    m.add_argument("--ids", required=True)
    m.add_argument("--selection", required=True, help="the re-check that selected the prompts")
    m.add_argument("--served", required=True, help="a fresh probe of the served adapter")
    m.add_argument("--final", required=True, help="a probe of the trained adapter")
    m.add_argument("--final-step", default="?")
    m.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--recheck-report", required=True)
    b = sub.add_parser("build")
    b.add_argument("--recheck", required=True)
    b.add_argument("--probe", default=str(DEFAULT_PROBE))
    b.add_argument("--train", default=str(DEFAULT_TRAIN))
    b.add_argument("--anchor-ratio", type=float, default=0.5)
    b.add_argument("--seed", type=int, default=20261016)
    b.add_argument("--out-dir", required=True)
    r = sub.add_parser("report")
    r.add_argument("--run-dir", required=True)
    r.add_argument("--served-test", default=str(SERVED_TEST))
    r.add_argument("--served-val", default=str(SERVED_VAL))
    r.add_argument("--before", default=None)
    r.add_argument("--after", default=None)
    r.add_argument("--ids", default=None)
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"decide": cmd_decide, "build": cmd_build, "report": cmd_report,
            "mechanism": cmd_mechanism}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
