"""Evaluation harness for vimBot.

Scores predicted vim commands by EXECUTION, not string match: a prediction
is correct iff running it in headless vim transforms buffer_before into
buffer_after. This automatically credits semantically equivalent commands
(`:%s/a/b/g` vs `ggVG` + substitution, `dd` vs `:1d`, ...), which exact-match
metrics get wrong.

Usage:
    # Self-check: run every case's gold command (should be ~100%)
    python3 src/evaluate.py --cases data/seed_cases.jsonl

    # Score model predictions
    python3 src/evaluate.py --cases data/seed_cases.jsonl \
                            --predictions runs/model_preds.jsonl

Case schema (JSONL, one object per line):
    id             str
    category       str   e.g. "substitute", "keys-textobj", "global"
    instruction    str   natural-language request (model input)
    mode           "ex" | "keys"
    buffer_before  str
    gold_command   str
    buffer_after   str   expected buffer after execution
    cursor         [line, col]  optional, default [1, 1]

Prediction schema (JSONL):
    id        str   must match a case id
    command   str
    mode      "ex" | "keys"   optional, defaults to the case's mode

Metrics:
    valid       command executed without error (ex mode: no vim error;
                keys mode: best-effort, process exit only)
    exec_match  buffer_after matches expected (the metric that matters)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict

from executor import run_vim


def _norm(text: str | None) -> list[str] | None:
    """Compare buffers as line lists, ignoring one trailing newline."""
    if text is None:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: bad JSON ({exc})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="data/seed_cases.jsonl")
    ap.add_argument("--predictions", default=None,
                    help="JSONL of model predictions; omit for gold self-check")
    ap.add_argument("--vim", default="vim", help="vim binary to use")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--show-failures", action="store_true", default=True)
    args = ap.parse_args()

    cases = load_jsonl(args.cases)
    preds: dict[str, dict] = {}
    if args.predictions:
        for p in load_jsonl(args.predictions):
            preds[p["id"]] = p

    n_valid = 0
    n_match = 0
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, valid, match
    failures: list[tuple[str, str, dict]] = []

    for case in cases:
        pred = preds.get(case["id"], {})
        command = pred.get("command", case["gold_command"])
        mode = pred.get("mode", case.get("mode", "ex"))
        cursor = tuple(case.get("cursor", [1, 1]))

        result = run_vim(
            case["buffer_before"], command,
            mode=mode, cursor=cursor,
            timeout=args.timeout, vim_bin=args.vim,
        )
        match = bool(
            result["buffer_after"] is not None
            and _norm(result["buffer_after"]) == _norm(case["buffer_after"])
        )
        # In keys mode `was_valid` is weak evidence; fold match into it for
        # reporting so the headline numbers stay honest.
        valid = result["was_valid"] or match

        n_valid += valid
        n_match += match
        cat = case.get("category", "misc")
        per_cat[cat][0] += 1
        per_cat[cat][1] += valid
        per_cat[cat][2] += match
        if not match:
            failures.append((case["id"], command, result))

    n = len(cases)
    src = "predictions" if args.predictions else "gold self-check"
    print(f"vimBot eval ({src}): {n} cases")
    print(f"  valid       {n_valid}/{n}")
    print(f"  exec_match  {n_match}/{n}")
    print()
    print(f"  {'category':<16} {'n':>3} {'valid':>6} {'match':>6}")
    for cat in sorted(per_cat):
        t, v, m = per_cat[cat]
        print(f"  {cat:<16} {t:>3} {v:>6} {m:>6}")

    if failures and args.show_failures:
        print("\nfailures:")
        for cid, command, result in failures:
            flag = "TIMEOUT" if result["timeout"] else f"rc={result['return_code']}"
            print(f"  {cid}: {command!r} [{flag}]")
            if result["buffer_after"] is not None:
                got = _norm(result["buffer_after"])
                print(f"    got: {got!r}")

    sys.exit(0 if n_match == n else 1)


if __name__ == "__main__":
    main()
