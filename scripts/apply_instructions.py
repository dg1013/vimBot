#!/usr/bin/env python3
"""Write authored instructions into a v3 case file.

Instructions are authored separately from mining (see CLAUDE.md sec 6: the model's
input is the instruction, so a case without one cannot train anything). This
applies a {id: instruction} map to any v3 file, sets instruction_provenance,
and re-runs nothing else — the buffers and commands are already gate-verified.

Usage:
    python scripts/apply_instructions.py --cases data/v3/emulator_cases.jsonl \
                                         --instructions authored.json \
                                         [--provenance claude-authored] [--dry-run]

authored.json is {"emul-vsc-0004": "Move the first line below the second", ...}

Refuses an instruction that contains the gold command verbatim — that is the
answer-leak the upstream test titles carry, and the reason they are unusable.
"""

import argparse
import collections
import json
import pathlib
import re
import sys


def leaks(instruction, command, keys_raw):
    """Reject text that hands the model its own answer."""
    low = instruction.lower()
    forms = set()
    for form in filter(None, (command, keys_raw)):
        f = form.strip()
        forms.add(f)
        forms.add(re.sub(r"<[^>]*>", "", f))          # <Esc>:m2 -> :m2
        forms.add(re.sub(r"<[^>]*>", "", f).lstrip(":"))
    for f in forms:
        f = f.strip().lower()
        if len(f) >= 2 and f in low:
            return "contains the command %r verbatim" % f
    if re.search(r"\bpress\b|\bkeystroke|\btype the key", low):
        return "describes keystrokes rather than intent"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True)
    ap.add_argument("--instructions", required=True)
    ap.add_argument("--provenance", default="claude-authored")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.cases)
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    given = json.loads(pathlib.Path(args.instructions).read_text())

    by_id = {r["id"]: r for r in rows}
    applied, rejected, unknown = 0, [], []
    for rid, text in given.items():
        row = by_id.get(rid)
        if row is None:
            unknown.append(rid)
            continue
        text = (text or "").strip()
        if not text:
            continue
        why = leaks(text, row.get("gold_command", ""), row.get("keys_raw", ""))
        if why:
            rejected.append((rid, why, text))
            continue
        row["instruction"] = text
        row["instruction_provenance"] = args.provenance
        row["instruction_pending"] = False
        applied += 1

    print("cases in file      : %d" % len(rows))
    print("instructions given : %d" % len(given))
    print("applied            : %d" % applied)
    if rejected:
        print("REJECTED (answer leak):")
        for rid, why, text in rejected[:10]:
            print("   %-16s %s\n        %r" % (rid, why, text[:70]))
    if unknown:
        print("unknown ids        : %d  %s" % (len(unknown), unknown[:5]))
    still = sum(1 for r in rows if not (r.get("instruction") or "").strip())
    print("still pending      : %d of %d" % (still, len(rows)))
    print("provenance         : %s"
          % dict(collections.Counter(r.get("instruction_provenance") for r in rows)))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nwrote %s" % path)
    print("now run: python src/evaluate.py --cases %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
