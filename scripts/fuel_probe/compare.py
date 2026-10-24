#!/usr/bin/env python3
"""Compare difficulty probes across models: how much of the train split each
adapter solves sometimes (GRPO's fuel), never, or always -- and check a short
run before the full one is committed (README.md beside this file).

A probe row is one train prompt with `k` exec-matching samples of `G`
(scripts/probe_difficulty.py). GRPO learns only from prompts where the G
samples disagree: at k = 0 or k = G every group-relative advantage is zero.

    python scripts/fuel_probe/compare.py --run runs/fuel_probe_2026-09-26
    python scripts/fuel_probe/compare.py --run ... --check qwen2.5-coder-7b   # smoke gate
"""
import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
TRAIN_ROWS = 6548
MAX_TRUNCATED = 0.05     # a model whose answers run into the cap does not stop properly
MAX_UNPARSED = 0.10

# The Qwen3.5 probes already on disk, for reference (all of them after RL or a
# cliff SFT round -- the only SFT-round-1-like point is the GRPO start).
REFERENCE = [
    ("qwen3.5-4b after GRPO run 1", "runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b.jsonl"),
    ("qwen3.5-9b after GRPO (4-bit)", "runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-9b.jsonl"),
    ("qwen3.5-4b served (after the gated round)",
     "runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-leg2.jsonl"),
    ("qwen3.5-4b after the cliff SFT round",
     "runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-cliffleg.jsonl"),
]


def load(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                rows[r["id"]] = r
    return list(rows.values())


def stats(rows):
    """Shelf shares, mean fuel k(G-k)/G, and the answer states over all samples."""
    n = len(rows)
    shelves = collections.Counter("cliff" if r["k"] == 0 else "known" if r["k"] == r["G"]
                                  else "frontier" for r in rows)
    states = collections.Counter()
    for r in rows:
        states.update(r.get("states") or {})
    samples = sum(r["G"] for r in rows)
    return {"n": n, "never": shelves["cliff"] / n, "sometimes": shelves["frontier"] / n,
            "always": shelves["known"] / n,
            "fuel": sum(r["k"] * (r["G"] - r["k"]) / r["G"] for r in rows) / n,
            "samples": samples,
            "truncated": states.get("truncated", 0) / samples,
            "unparsed": states.get("parse", 0) / samples,
            "match": states.get("match", 0) / samples}


def pace(log_path):
    """(prompts, seconds) summed over every finished launch in a probe log."""
    prompts = seconds = 0
    if pathlib.Path(log_path).exists():
        text = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
        for m in re.finditer(r"probe done: (\d+) prompts in (\d+) s", text):
            prompts += int(m.group(1))
            seconds += int(m.group(2))
    return prompts, seconds


def check(run, model):
    """The smoke gate: answers must stop on their own and parse; prints the pace."""
    s = stats(load(run / ("difficulty_probe_%s.jsonl" % model)))
    p, secs = pace(run / ("probe_%s.log" % model))
    per = secs / p if p else float("nan")
    line = ("[%s] smoke: %d prompts, %.1f s/prompt -> about %.1f h for the train split; "
            "answers capped %.1f%%, unparsed %.1f%%, exec-match %.1f%% of samples; "
            "never %.0f%% / sometimes %.0f%% / always %.0f%%"
            % (model, s["n"], per, per * TRAIN_ROWS / 3600, 100 * s["truncated"],
               100 * s["unparsed"], 100 * s["match"], 100 * s["never"], 100 * s["sometimes"],
               100 * s["always"]))
    print(line)
    if s["truncated"] > MAX_TRUNCATED or s["unparsed"] > MAX_UNPARSED:
        print("[%s] smoke check FAILED: more than %d%% capped or %d%% unparsed answers"
              % (model, 100 * MAX_TRUNCATED, 100 * MAX_UNPARSED))
        return 1
    return 0


def report(run, models):
    lines = ["# Fuel probe: how much of the train split each SFT adapter sometimes solves", "",
             "G = 8 samples per train prompt at T = 1 (scripts/probe_difficulty.py). GRPO learns "
             "only from the *sometimes* shelf; *never* and *always* give zero advantage. Mean fuel "
             "is k(G−k)/G averaged over prompts (1.75 for a prompt at p = ½).", "",
             "| adapter | prompts | never | **sometimes** | always | mean fuel | exec-match of samples | capped | hours |",
             "|---|---|---|---|---|---|---|---|---|"]

    def row(label, rows, hours):
        s = stats(rows)
        lines.append("| %s | %d | %.0f%% | **%.0f%%** | %.0f%% | %.2f | %.1f%% | %.1f%% | %s |"
                     % (label, s["n"], 100 * s["never"], 100 * s["sometimes"], 100 * s["always"],
                        s["fuel"], 100 * s["match"], 100 * s["truncated"], hours))
    for m in models:
        path = run / ("difficulty_probe_%s.jsonl" % m)
        if path.exists():
            p, secs = pace(run / ("probe_%s.log" % m))
            row("%s + SFT (4-bit)" % m, load(path), "%.1f" % (secs / 3600) if p else "")
    lines.append("| *reference: earlier Qwen3.5 probes* | | | | | | | | |")
    for label, rel in REFERENCE:
        path = REPO / rel
        if path.exists():
            row(label, load(path), "")
    lines += ["", "For an SFT-round-1 adapter the Qwen3.5-4B's only measurement is its first GRPO "
              "run: about 65% of prompt groups had mixed rewards at the start (35% unanimous), "
              "56% unanimous by the end. Per-model breakdowns by category, tier and mode: "
              "`difficulty_summary_<model>.md`."]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--check", default=None, metavar="MODEL", help="the smoke gate for one model")
    ap.add_argument("--models", default="qwen2.5-coder-7b seed-coder-8b granite-4.2-8b")
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    if args.check:
        return check(run, args.check)
    report(run, args.models.split())
    return 0


if __name__ == "__main__":
    sys.exit(main())
