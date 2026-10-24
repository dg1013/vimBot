#!/usr/bin/env python3
"""Restore rows from data/dropped_cases.jsonl, optionally with the case clause.

Drops are auditable and reversible (apply_curation.py): every dropped row is
kept whole with `dropped_from`, `dropped_on` and `dropped_reason`. This puts a
selection of them back where they came from, so a reviewer can re-verify them
by hand after a repair.

The one repair implemented is the owner's 2026-09-04 case rule for the KB13 (the Kushman and Barzilay 2013 corpus)
signature (the gold matches case-sensitively, the buffer holds the other case
on purpose, the instruction never says so). It is applied ONLY to rows the
review page's own classifier marks as case-silent:

    every quoted literal lowercase  ->  "'ba' in lowercase"
                                        (two: ", both in lowercase";
                                         more: ", all in lowercase")
    every quoted literal uppercase  ->  ", both/all in uppercase"
    otherwise                       ->  " (case-sensitive)"

A trailing clause is where this reads worst -- "the letter 'z' in lowercase
twice" strands "twice" -- so a repaired row is prose to be re-read, not a
finished instruction; 8 of the 43 rows repaired on 2026-09-04 were reworded
by hand afterwards to put the adjective before the noun ("the lowercase
letter 'z' twice"). See DECISIONS.md 2026-09-04.

Rows that are not case-silent are restored unchanged and marked
`restore_note: needs manual rewording`. Each restored row keeps its original
text in `instruction_before_restore` and gains `restored_on`,
`restore_edit` and `restored_from_drop`; the dropped_* fields go. Every row
is re-executed before it is written back. The restored ids are removed from
dropped_cases.jsonl and the whole event is written to --audit.

With --split, the rows are also appended to that split file, taking their
split-only fields (needs_review, theme, ...) from --split-backup when given,
so the frozen-by-id splits stay whole. Then run the usual chain:
build_corpus.py --dataset all, select_datasets.py, refresh_splits.py.

Usage:
    python scripts/restore_dropped.py --dropped-on 2026-09-04 --edit case-clause \\
        --split test --split-backup /path/to/pre-drop/test.jsonl \\
        --audit data/fixes/restore_dropped_2026-09-04.json [--dry-run]
"""
import argparse
import datetime
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import case_profile     # noqa: E402
from evaluate import _norm                     # noqa: E402
from executor import run_vim                   # noqa: E402

DROPPED = REPO / "data" / "dropped_cases.jsonl"
QUOTED = re.compile(r"'([^']+)'")
DROP_FIELDS = ("dropped_from", "dropped_on", "dropped_reason")


def is_case_silent(row):
    _, risk = case_profile(row, _norm(row["buffer_after"]))
    return risk == "unstated"


def case_clause(ins):
    """(edited instruction, rule name) for a case-silent instruction."""
    lits = [m for m in QUOTED.finditer(ins) if re.search(r"[A-Za-z]", m.group(1))]
    words = [m.group(1) for m in lits]
    stripped = ins.rstrip()
    tail = stripped[-1] if stripped.endswith((".", "!")) else ""
    body = stripped[:-1] if tail else stripped
    quantifier = "both" if len(lits) == 2 else "all"
    if words and all(w == w.lower() for w in words):
        if len(lits) == 1:
            m = lits[0]
            return ins[:m.end()] + " in lowercase" + ins[m.end():], "in lowercase"
        return (body + ", %s in lowercase" % quantifier + tail,
                "%s in lowercase" % quantifier)
    if words and all(w == w.upper() for w in words):
        return (body + ", %s in uppercase" % quantifier + tail,
                "%s in uppercase" % quantifier)
    return body + " (case-sensitive)" + tail, "(case-sensitive)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dropped-on", required=True, help="restore rows dropped on this date")
    ap.add_argument("--edit", choices=("none", "case-clause"), default="none")
    ap.add_argument("--split", help="also append the rows to data/splits/<split>.jsonl")
    ap.add_argument("--split-backup", help="pre-drop split file to take split-only fields from")
    ap.add_argument("--audit", required=True, help="JSON record of the restoration")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dropped = [json.loads(l) for l in DROPPED.open(encoding="utf-8") if l.strip()]
    chosen = [r for r in dropped if r.get("dropped_on") == args.dropped_on]
    if not chosen:
        sys.exit("no rows dropped on %s" % args.dropped_on)
    today = datetime.date.today().isoformat()

    restored, audit, by_file = [], [], {}
    for r in chosen:
        row = {k: v for k, v in r.items() if k not in DROP_FIELDS}
        original = row["instruction"]
        rule, note = "none", None
        if args.edit == "case-clause" and is_case_silent(row):
            row["instruction"], rule = case_clause(original)
        else:
            note = "needs manual rewording"
        res = run_vim(row["buffer_before"], row["gold_command"], mode=row.get("mode", "ex"),
                      cursor=tuple(row.get("cursor") or (1, 1)))
        if not res["was_valid"] or _norm(res["buffer_after"]) != _norm(row["buffer_after"]):
            sys.exit("%s no longer reproduces its buffer; not restoring" % row["id"])
        row["instruction_before_restore"] = original
        row["restored_on"] = today
        row["restore_edit"] = rule
        row["restored_from_drop"] = r["dropped_reason"]
        if rule != "none":
            # the text is no longer the annotator's verbatim sentence
            row["instruction_provenance"] = "owner-edited"
        if note:
            row["restore_note"] = note
        restored.append(row)
        by_file.setdefault(r["dropped_from"], []).append(row)
        audit.append({"id": row["id"], "dropped_from": r["dropped_from"],
                      "instruction_before": original, "instruction_after": row["instruction"],
                      "rule": rule, "note": note})

    rules = {}
    for a in audit:
        rules[a["rule"] if not a["note"] else a["note"]] = rules.get(a["rule"] if not a["note"] else a["note"], 0) + 1
    print("restoring %d row(s) dropped on %s:" % (len(restored), args.dropped_on))
    for k, n in sorted(rules.items(), key=lambda kv: -kv[1]):
        print("   %3d  %s" % (n, k))
    for path, rows in by_file.items():
        print("   -> %s (+%d)" % (path, len(rows)))
    if args.dry_run:
        print("--dry-run: nothing written")
        return

    for path, rows in by_file.items():
        with (REPO / "data" / path).open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    keep = [r for r in dropped if r.get("dropped_on") != args.dropped_on]
    with DROPPED.open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("   dropped_cases.jsonl: %d -> %d rows" % (len(dropped), len(keep)))

    if args.split:
        extra = {}
        if args.split_backup:
            corpus_keys = set(restored[0].keys())
            for l in open(args.split_backup, encoding="utf-8"):
                if l.strip():
                    b = json.loads(l)
                    extra[b["id"]] = {k: v for k, v in b.items() if k not in corpus_keys}
        split_path = REPO / "data" / "splits" / ("%s.jsonl" % args.split)
        with split_path.open("a", encoding="utf-8") as fh:
            for row in restored:
                fh.write(json.dumps({**extra.get(row["id"], {}), **row}, ensure_ascii=False) + "\n")
        print("   %s: +%d rows (%d with split-only fields recovered)"
              % (split_path.relative_to(REPO), len(restored), sum(1 for r in restored if r["id"] in extra)))

    pathlib.Path(args.audit).write_text(json.dumps(
        {"restored_on": today, "dropped_on": args.dropped_on, "edit": args.edit,
         "rows": audit}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("   audit -> %s" % args.audit)
    print("\nnow run: python scripts/build_corpus.py --dataset all\n"
          "then   : python scripts/select_datasets.py --datasets v3,v4 --out data/combined_cases.jsonl\n"
          "then   : python scripts/refresh_splits.py")


if __name__ == "__main__":
    main()
