#!/usr/bin/env python3
"""Validate Claude's per-row notes on the never-solved train rows and report
progress.

Notes live in runs/never_review_2026-09-25/comments/b*.json, each a list of
{id, class, comment, suggest?}. Checked here: every id is a row on the page
and appears once; the class is one of CLASSES; the comment is non-empty; and
a suggestion passes the curation guard -- an instruction must not contain the
gold command (apply_instructions.leaks), and a suggested command, buffer,
cursor or mode must run in the sandbox and change the buffer.

    python scripts/never_review/check_comments.py
"""
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from apply_instructions import leaks            # noqa: E402
from executor import run_vim                    # noqa: E402
from evaluate import _norm                      # noqa: E402
from notes import CLASSES, SUGGESTABLE, load_notes   # noqa: E402
from triage import load_cards                   # noqa: E402


def check(notes, cards):
    by_id = {c["id"]: c for c in cards}
    problems = []
    for rid, n in notes.items():
        c = by_id.get(rid)
        if c is None:
            problems.append("%s: not a row on the page" % rid)
            continue
        if n.get("class") not in CLASSES:
            problems.append("%s: unknown class %r" % (rid, n.get("class")))
        if not (n.get("comment") or "").strip():
            problems.append("%s: empty comment" % rid)
        s = n.get("suggest") or {}
        stray = sorted(set(s) - set(SUGGESTABLE))
        if stray:
            problems.append("%s: suggestion has fields apply_curation.py does not edit: %s" % (rid, stray))
        mode = s.get("mode", c["mode"])
        cmd = s.get("command", c["cmd"])
        guard = cmd.lstrip(":") if mode == "ex" else cmd
        ins = s.get("instruction", c["ins"])
        why = leaks(ins, guard, guard)
        if s.get("instruction") is not None and why:
            problems.append("%s: suggested instruction %s" % (rid, why))
        if any(k in s for k in ("command", "buffer_before", "cursor", "mode")):
            buf = s.get("buffer_before", c["bef"])
            res = run_vim(buf, guard, mode=mode, cursor=tuple(s.get("cursor", c["cur"])))
            if res.get("blocked") or res.get("timeout") or not res.get("was_valid"):
                problems.append("%s: suggested command does not run cleanly (%s)" % (rid, res.get("stderr", "")[:80]))
            elif _norm(res["buffer_after"]) == _norm(buf):
                problems.append("%s: suggested command does not change the buffer" % rid)
    return problems


def main():
    cards = load_cards()
    notes = load_notes()
    problems = check(notes, cards)
    order = {c["id"]: i for i, c in enumerate(cards)}
    done = sorted(order[i] for i in notes if i in order)
    first_gap = next((i for i in range(len(cards)) if i not in set(done)), len(cards))
    print("notes: %d of %d rows; next unnoted row index %d" % (len(notes), len(cards), first_gap))
    print("by class:", dict(collections.Counter(n["class"] for n in notes.values())))
    print("with a suggestion:", sum(1 for n in notes.values() if n.get("suggest")))
    for p in problems:
        print("PROBLEM", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
