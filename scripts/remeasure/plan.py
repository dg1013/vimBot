#!/usr/bin/env python3
"""The results-table runs to re-measure on test v2, and how to regenerate each.

Test v2 is the 429-row slice after the always-failing review (DECISIONS.md
2026-09-25): 16 rows dropped, 31 rows with a new instruction (one with a new
gold). A run's number on v2 is its saved greedy predictions re-scored on the
398 rows that did not change, plus fresh greedy predictions on the 31 that did.
This file names the runs, in the results table's order, and derives each one's
regeneration settings from the summary its own measurement wrote
(`runs/<run>/baseline_test.json`: model, adapter, base precision), so nothing
is restated by hand.

    python scripts/remeasure/plan.py            # the plan, one line per run
    python scripts/remeasure/plan.py --configs  # one line per distinct regeneration
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "runs"

# (results-table row, run directory under runs/), in the table's order.
TABLE = [
    ("qwen3.5-4b, zero-shot", "qwen3.5-4b"),
    ("qwen3.5-4b + SFT round 1", "qwen3.5-4b-sft"),
    ("qwen3.5-4b + SFT + GRPO", "qwen3.5-4b-grpo"),
    ("qwen3.5-4b + gated GRPO (served)", "qwen3.5-4b-grpo-gated"),
    ("the served adapter under the fla kernels", "qwen3.5-4b-grpo-gated/fla-kernels"),
    ("qwen3.5-4b chain stage C", "qwen3.5-4b-grpo-gated-cliffleg"),
    ("qwen3.5-0.8b, zero-shot", "qwen3.5-0.8b"),
    ("qwen3.5-0.8b + SFT", "qwen3.5-0.8b-sft"),
    ("qwen3.5-2b, zero-shot", "qwen3.5-2b"),
    ("qwen3.5-2b + SFT", "qwen3.5-2b-sft"),
    ("qwen3.5-9b, zero-shot", "qwen3.5-9b"),
    ("qwen3.5-9b + SFT (NF4)", "qwen3.5-9b-sft"),
    ("qwen3.5-9b + SFT, bf16 base", "qwen3.5-9b-sft/bf16-base"),
    ("qwen3.5-9b + SFT + GRPO (NF4)", "qwen3.5-9b-grpo"),
    ("qwen3.5-9b gated round, step 0 (NF4, fla)", "qwen3.5-9b-grpo-gated"),
    ("qwen3-4b-2507, zero-shot", "qwen3-4b-2507"),
    ("qwen3-4b-2507 + SFT", "qwen3-4b-2507-sft"),
    ("granite-4.2-8b, zero-shot", "granite-4.2-8b"),
    ("granite-4.2-8b + SFT (NF4)", "granite-4.2-8b-sft"),
    ("seed-coder-8b, zero-shot", "seed-coder-8b"),
    ("seed-coder-8b + SFT (NF4)", "seed-coder-8b-sft"),
    ("seed-coder-8b + SFT, bf16 base", "seed-coder-8b-sft/bf16-base"),
    ("qwen2.5-coder-7b, zero-shot", "qwen2.5-coder-7b"),
    ("qwen2.5-coder-7b + SFT (NF4)", "qwen2.5-coder-7b-sft"),
]


def settings(run_dir):
    """(model, adapter or None, base 'bf16'|'nf4', condition, date) from the
    run's own summary."""
    s = json.loads((RUNS / run_dir / "baseline_test.json").read_text(encoding="utf-8"))
    run = s.get("run") or {}
    adapter = run.get("adapter")
    if adapter:
        p = pathlib.Path(adapter)
        adapter = str(p.relative_to(REPO)) if p.is_absolute() else adapter
    return (s["model"], adapter, run.get("base") or "bf16",
            run.get("condition") or "zero-shot", run.get("date"))


def config_name(model, adapter, base):
    """A directory name for one regeneration: `<model>__zero-shot`, or the
    adapter's run directory plus the base it is loaded on (one adapter is
    measured on two bases for the 9B and Seed-Coder rows)."""
    if adapter is None:
        return model + "__zero-shot"
    return str(pathlib.Path(adapter).parent.relative_to("runs")).replace("/", "__") + "__" + base


def plan():
    rows, configs = [], {}
    for label, run_dir in TABLE:
        model, adapter, base, condition, date = settings(run_dir)
        key = (model, adapter, base)
        if key not in configs:
            configs[key] = {"name": config_name(model, adapter, base), "model": model,
                            "adapter": adapter, "base": base, "condition": condition}
        rows.append({"label": label, "run": run_dir, "model": model, "adapter": adapter,
                     "base": base, "date": date, "config": configs[key]["name"]})
    names = [c["name"] for c in configs.values()]
    if len(names) != len(set(names)):
        sys.exit("two regenerations share a directory name: %s" % names)
    return rows, list(configs.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--configs", action="store_true",
                    help="print one tab-separated line per distinct regeneration: "
                         "name, model, adapter (or none), base, condition")
    args = ap.parse_args()
    rows, configs = plan()
    if args.configs:
        for c in configs:
            print("\t".join([c["name"], c["model"], c["adapter"] or "none", c["base"],
                             c["condition"]]))
        return 0
    for r in rows:
        print("%-44s %-38s -> %s" % (r["label"], r["run"], r["config"]))
    print("%d runs, %d regenerations" % (len(rows), len(configs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
