"""Differential validation: the unrestricted Python implementation vs the
stored vim-derived answers (all live cases) and vs the span-DP (dynamic programming) oracle.

Three independent implementations answer the same query:
  span-DP oracle (dsl_ast.matcher)   — definitional truth
  vim (sandboxed, converted pattern) — what the dataset actually executes
  python (py_impl, unrestricted re)  — boolean composition, no conversion

Any disagreement is a bug in one of the three — or a recorded intentional
divergence: rows whose `edit_note` marks an owner rename keep their
original `source_regex` as provenance, so the Python recomputation (which
reads source_regex) is expected to differ there and is reported separately.

Run from the repo (repository) root:  python scripts/deep_regex/compare_impls.py
Reference result (2026-08-14, after the TS-026 rename was reverted):
14,434/14,434 expressible cases agree; 0 oracle mismatches; 0 divergences.
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dsl_ast import dsl_match  # noqa: E402
from py_impl import apply_case, py_matcher  # noqa: E402


def main():
    stats = collections.Counter()
    disagreements, known = [], []
    for split in ("train", "val", "test"):
        for line in open(REPO / "data" / "splits" / f"{split}.jsonl",
                         encoding="utf-8"):
            r = json.loads(line)
            stats["cases"] += 1
            py = apply_case(r)
            if py is None:
                stats["inexpressible"] += 1
                continue
            stats["expressible"] += 1
            if py == r["buffer_after"]:
                stats["agree"] += 1
            elif "renamed" in r.get("edit_note", ""):
                stats["known-divergence"] += 1
                known.append(r["id"])
            else:
                stats["DISAGREE"] += 1
                disagreements.append((split, r["id"], r["source_regex"]))

    print("== python vs stored-vim, all live cases ==")
    for k in ("cases", "expressible", "inexpressible", "agree",
              "known-divergence", "DISAGREE"):
        print(f"  {k}: {stats[k]}")
    if known:
        print("  known (owner renames, provenance != content):", known)
    for d in disagreements[:12]:
        print("  DISAGREE:", *d)

    rng = random.Random(5)
    seen, checked, mism = set(), 0, 0
    for line in open(REPO / "data" / "splits" / "train.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        t = r.get("source_regex")
        if not t or t in seen:
            continue
        seen.add(t)
        f = py_matcher(t)
        if f is None:
            continue
        for ln in r["buffer_before"].split("\n"):
            checked += 1
            if f(ln) != dsl_match(t, ln):
                mism += 1
        if len(seen) > 1500:
            break
    print(f"== python vs span-oracle: {checked} line-checks, "
          f"{mism} mismatches ==")
    return 1 if disagreements or mism else 0


if __name__ == "__main__":
    sys.exit(main())
