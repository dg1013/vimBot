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

    # ... and keep what vim actually did, one row per case
    python3 src/evaluate.py --cases data/splits/test.jsonl \
                            --predictions runs/qwen3.5-4b/preds_test.jsonl \
                            --exec-log runs/qwen3.5-4b/exec_test.jsonl

Case schema (JSONL (JSON Lines), one object per line):
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

Execution log (JSONL, one row per case, in case order) -- `--exec-log`, or
`write_exec_log` for callers that already hold a `score()` result:
    id             str   the case
    category       str   copied from the case, so the log slices alone
    command        str | None    what was run (None = no prediction)
    mode           "ex" | "keys"
    valid, match   bool  the two verdicts
    timeout        bool
    return_code    int | None    vim's exit status
    error          str   "E486"-style code, "TIMEOUT", or "" -- what to
                         cluster failures by
    stderr         str   vim's complaint, stripped
    buffer_after   str | None    what the command actually produced
    blocked        str   present only when src/scope.py refused the command

Scoring is the only place that knows these; without the log they are printed
and lost, and every consumer has to re-run the whole split through vim to get
them back.
"""

from __future__ import annotations

import argparse
import json
import re
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


# vim reports failures as "E486: Pattern not found"; the code is the stable
# part (the message is localized and carries the offending text), so it is
# what failure clustering groups by.
_VIM_ERROR = re.compile(r"\bE\d{1,4}\b")


def vim_error_code(stderr: str | None, timed_out: bool = False) -> str:
    """The E-code in vim's stderr, "TIMEOUT", or "" if it did not complain."""
    if timed_out:
        return "TIMEOUT"
    m = _VIM_ERROR.search(stderr or "")
    return m.group(0) if m else ""


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


def score(cases: list[dict], preds: dict[str, dict] | None = None,
          timeout: float = 5.0, vim_bin: str = "vim") -> dict:
    """Run every case through the sandbox and compare buffers.

    `preds` is None for the gold self-check (each case runs its own
    gold_command) or a {id: {command, mode?}} map for scoring a model. In
    predictions mode a case with NO prediction is a failure -- not valid,
    not matched -- never a silent fall-through to the gold command. That
    fall-through was a bug until 2026-09-05: a model that failed to parse on
    every row would have scored 100%.

    Returns n, n_valid, n_match, per_cat {category: [n, valid, match]},
    failures [(id, command, result)], and per_case {id: {...}} -- the last
    carrying the verdicts AND what vim did (return code, error code, stderr,
    the buffer it produced), so a caller can persist the run rather than
    re-executing the split to recover it. See `write_exec_log`.
    """
    n_valid = n_match = 0
    per_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # n, valid, match
    failures: list[tuple[str, str | None, dict]] = []
    per_case: dict[str, dict] = {}

    for case in cases:
        if preds is None:
            command, mode = case["gold_command"], case.get("mode", "ex")
        else:
            pred = preds.get(case["id"])
            command = pred.get("command") if pred else None
            mode = (pred or {}).get("mode", case.get("mode", "ex"))
        cursor = tuple(case.get("cursor", [1, 1]))

        if command is None:
            result = {"was_valid": False, "buffer_after": None,
                      "return_code": None, "timeout": False,
                      "stderr": "no prediction"}
        else:
            result = run_vim(case["buffer_before"], command, mode=mode,
                             cursor=cursor, timeout=timeout, vim_bin=vim_bin)
        match = bool(
            result["buffer_after"] is not None
            and _norm(result["buffer_after"]) == _norm(case["buffer_after"])
        )
        # In keys mode `was_valid` is weak evidence; fold match into it for
        # reporting so the headline numbers stay honest.
        valid = bool(result["was_valid"] or match)

        n_valid += valid
        n_match += match
        cat = case.get("category", "misc")
        per_cat[cat][0] += 1
        per_cat[cat][1] += valid
        per_cat[cat][2] += match
        entry = {"command": command, "mode": mode,
                 "valid": valid, "match": match,
                 "timeout": bool(result["timeout"]),
                 "return_code": result["return_code"],
                 "error": vim_error_code(result["stderr"], result["timeout"]),
                 "stderr": (result["stderr"] or "").strip(),
                 "buffer_after": result["buffer_after"]}
        if "blocked" in result:      # the scope screen refused it before vim
            entry["blocked"] = result["blocked"]
        per_case[case["id"]] = entry
        if not match:
            failures.append((case["id"], command, result))

    return {"n": len(cases), "n_valid": n_valid, "n_match": n_match,
            "per_cat": dict(per_cat), "failures": failures,
            "per_case": per_case}


def write_exec_log(path: str, cases: list[dict], scored: dict) -> int:
    """Write one execution record per case, in case order. Returns the count.

    Callers that already scored a split (scripts/baseline.py,
    scripts/train_sft.py) get the per-case audit trail for free -- the runs
    happened, this only keeps them. Rows join to `completions_<split>.jsonl`
    on `id`, and to the split file on `id`, so the three artifacts together
    describe a run without any vim re-execution.
    """
    per_case = scored["per_case"]
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for case in cases:
            entry = per_case.get(case["id"])
            if entry is None:        # case not scored; nothing to record
                continue
            rec = {"id": case["id"], "category": case.get("category", "misc")}
            rec.update(entry)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", default="data/seed_cases.jsonl")
    ap.add_argument("--predictions", default=None,
                    help="JSONL of model predictions; omit for gold self-check")
    ap.add_argument("--vim", default="vim", help="vim binary to use")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--exec-log", default=None,
                    help="JSONL to write one execution record per case into")
    ap.add_argument("--show-failures", action="store_true", default=True)
    args = ap.parse_args()

    cases = load_jsonl(args.cases)
    preds: dict[str, dict] | None = None
    if args.predictions:
        preds = {p["id"]: p for p in load_jsonl(args.predictions)}

    r = score(cases, preds, timeout=args.timeout, vim_bin=args.vim)
    n, n_valid, n_match = r["n"], r["n_valid"], r["n_match"]
    per_cat, failures = r["per_cat"], r["failures"]

    if args.exec_log:
        written = write_exec_log(args.exec_log, cases, r)
        print(f"exec log: {written} rows -> {args.exec_log}", file=sys.stderr)

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
            if command is None:
                flag = "NO PREDICTION"
            elif result["timeout"]:
                flag = "TIMEOUT"
            else:
                flag = f"rc={result['return_code']}"
            print(f"  {cid}: {command!r} [{flag}]")
            if result["buffer_after"] is not None:
                got = _norm(result["buffer_after"])
                print(f"    got: {got!r}")

    sys.exit(0 if n_match == n else 1)


if __name__ == "__main__":
    main()
