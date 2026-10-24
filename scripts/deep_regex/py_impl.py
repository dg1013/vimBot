"""Unrestricted pure-Python implementation of the global-edit task.

Third leg of the verifier triangle: the span-DP (dynamic programming) oracle defines truth, vim
executes the converted command in the sandbox, and this module answers the
same query in plain Python — no sandbox, no conversion tricks (`&`/`~`
become boolean composition over re, which vim cannot express directly).
Agreement of all three implementations is the strongest correctness
evidence the dataset can carry; any disagreement is a bug in one of them.

Coverage: py_oracle handles root-level op shapes; patterns it cannot
express are retried through their A3-verified normal form (equivalent by
the corpus-wide simplifier audit). Remaining gaps return None and are
reported, never guessed.

`apply_case(row)` recomputes a dataset row's buffer_after from
buffer_before + source_regex + the deterministic op assignment, entirely
in Python, for differential comparison against the stored vim result.
"""
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dsl2vim import py_oracle  # noqa: E402
from simplify_ast import normal_form  # noqa: E402


def py_matcher(pattern):
    """f(line)->bool via python-re, or None when inexpressible even
    through the verified normal form."""
    f = py_oracle(pattern)
    if f is not None:
        return f
    try:
        nf, _ = normal_form(pattern)
    except Exception:
        return None
    if nf != pattern:
        return py_oracle(nf)
    return None


def apply_case(row):
    """buffer_after computed in unrestricted Python, or None.

    Mirrors make_candidates' deterministic op assignment: the sha1 of the
    source pattern decides delete-matching vs keep-matching."""
    t = row.get("source_regex")
    if not t:
        return None
    f = py_matcher(t)
    if f is None:
        return None
    h = hashlib.sha1(t.encode()).hexdigest()
    keep_members = int(h, 16) % 2 != 0          # op == "keep"
    lines = row["buffer_before"].split("\n")
    survivors = [ln for ln in lines if f(ln) == keep_members]
    return "\n".join(survivors)
