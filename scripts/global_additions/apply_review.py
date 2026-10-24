#!/usr/bin/env python3
"""Apply the review page's change-set to the proposed global prompts.

The page exports {edits, deletes, review}, the shape apply_curation.py takes.
These rows are not in the corpus, so this applies the same change-set to
data/proposed/global_2026-10-16/cases.jsonl instead:
  deletes  rows move to dropped.jsonl, with the reviewer's reason;
  edits    fields change (instruction, gold_command, buffer_before, cursor, mode,
           category, tier); the gold is re-executed in the sandbox and buffer_after
           recomputed. An edit that makes the gold invalid or a no-op is refused.
  review   each row records review_verdict, review_note and reviewed_on.
Nothing is written with --dry-run. The previous file is kept as cases.<stamp>.jsonl.

    python scripts/global_additions/apply_review.py --changes changes.json --dry-run
    python scripts/global_additions/apply_review.py --changes changes.json
"""
import argparse
import datetime
import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from executor import run_vim                       # noqa: E402

DIR = REPO / "data" / "proposed" / "global_2026-10-16"
EDITABLE = {"instruction", "gold_command", "buffer_before", "cursor", "mode", "category", "tier"}


def load_jsonl(path):
    p = pathlib.Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def norm(b):
    return b[:-1] if b.endswith("\n") else b


def apply(rows, changes, today):
    """Returns (kept, dropped, report lines, refused)."""
    by_id = {r["id"]: dict(r) for r in rows}
    unknown = [i for part in ("edits", "deletes", "review") for i in changes.get(part, {}) if i not in by_id]
    if unknown:
        raise SystemExit("change-set names %d id(s) not in the proposed set, e.g. %s" % (len(unknown), unknown[:3]))
    both = set(changes.get("edits", {})) & set(changes.get("deletes", {}))
    if both:
        raise SystemExit("ids both edited and deleted: %s" % sorted(both)[:5])
    report, refused, dropped = [], [], []
    for cid, fields in changes.get("edits", {}).items():
        bad = set(fields) - EDITABLE
        if bad:
            raise SystemExit("%s: fields not editable here: %s" % (cid, sorted(bad)))
        row = dict(by_id[cid], **fields)
        res = run_vim(row["buffer_before"], row["gold_command"], mode=row.get("mode", "ex"),
                      cursor=tuple(row.get("cursor") or (1, 1)))
        after = res.get("buffer_after") or ""
        if res.get("blocked") or res.get("timeout") or not res.get("was_valid"):
            refused.append((cid, "the edited gold does not run"))
            continue
        if norm(after) == norm(row["buffer_before"]):
            refused.append((cid, "the edited gold is a no-op"))
            continue
        row["buffer_after"] = after
        row["edited_in_review"] = sorted(fields)
        by_id[cid] = row
        report.append("edit %s: %s" % (cid, ", ".join(sorted(fields))))
    for cid, reason in changes.get("deletes", {}).items():
        dropped.append(dict(by_id.pop(cid), drop_reason=reason, dropped_on=today))
        report.append("drop %s: %s" % (cid, reason))
    for cid, v in changes.get("review", {}).items():
        if cid in by_id:
            by_id[cid]["review_verdict"] = v.get("verdict")
            if v.get("note"):
                by_id[cid]["review_note"] = v["note"]
            by_id[cid]["reviewed_on"] = today
            if v.get("verdict") == "ok":
                by_id[cid]["needs_review"] = False
    kept = [by_id[r["id"]] for r in rows if r["id"] in by_id]
    return kept, dropped, report, refused


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--changes", required=True)
    ap.add_argument("--dir", default=str(DIR))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    d = pathlib.Path(args.dir)
    rows = load_jsonl(d / "cases.jsonl")
    changes = json.loads(pathlib.Path(args.changes).read_text(encoding="utf-8"))
    today = datetime.date.today().isoformat()
    kept, dropped, report, refused = apply(rows, changes, today)
    for line in report:
        print(line)
    for cid, why in refused:
        print("REFUSED %s: %s" % (cid, why))
    print("%d rows -> %d kept, %d dropped, %d edit(s) refused%s"
          % (len(rows), len(kept), len(dropped), len(refused), " (dry run, nothing written)" if args.dry_run else ""))
    if args.dry_run or refused:
        return 1 if refused else 0
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy(d / "cases.jsonl", d / ("cases.%s.jsonl" % stamp))
    with (d / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (d / "dropped.jsonl").open("a", encoding="utf-8") as fh:
        for r in dropped:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("written: cases.jsonl (previous kept as cases.%s.jsonl), dropped.jsonl" % stamp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
