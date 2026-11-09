#!/usr/bin/env python3
"""The fuel gate of runs/grpo_opd_2026-11-04/preregistration.md: a round runs only if its start's
samples still disagree. Reads a probe_difficulty.py file, writes the mean k(G−k)/G and the shelves,
and exits FUEL_LOW_RC when the mean is under FLOOR. CPU only.

    python scripts/grpo_opd/fuel.py runs/grpo_opd_2026-11-04/difficulty_probe_s42.jsonl --out fuel.json
"""
import argparse
import collections
import json
import pathlib
import statistics
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from difficulty import fuel, shelf_of                             # noqa: E402

FLOOR = 0.35         # the 9B after GRPO, whose next round was null
FUEL_LOW_RC = 5


def summarize(rows):
    shelves = collections.Counter(shelf_of(r["k"], r["G"]) for r in rows)
    mean = statistics.fmean(fuel(r["k"], r["G"]) for r in rows)
    return {"rows": len(rows), "fuel": round(mean, 4), "floor": FLOOR, "go": mean >= FLOOR,
            "mean_p_pct": round(100 * statistics.fmean(r["k"] / r["G"] for r in rows), 1),
            "shelves_pct": {k: round(100 * shelves[k] / len(rows), 1) for k in ("cliff", "frontier", "known")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(line) for line in open(args.probe, encoding="utf-8") if line.strip()]
    s = summarize(rows)
    pathlib.Path(args.out).write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(s), flush=True)
    return 0 if s["go"] else FUEL_LOW_RC


if __name__ == "__main__":
    sys.exit(main())
