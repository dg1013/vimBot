#!/usr/bin/env python3
"""Parse saved frontier-chat replies into a §5 prediction file.

Input: one or more text files, each a verbatim copy of a parallel-chat reply
(id-prefixed answer lines, one per case). Output: {id, command, mode} JSONL (JSON Lines)
for src/evaluate.py --predictions.

Lines that do not match `<known-id> ex:|keys: <command>` are counted and
reported — a malformed line is a format failure of the model under test, so
it is simply absent from the prediction file (evaluate.py scores absent
predictions as failures).
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prompt import parse_output  # noqa: E402

# `<id> <answer>`; the answer is then held to the §6 grammar by
# src/prompt.py, the single copy of that grammar.
ID_RE = re.compile(r"^(\S+)\s+(.*)$")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("responses", nargs="+", help="saved reply text files, in order")
    ap.add_argument("--cases", default="data/splits/test.jsonl")
    ap.add_argument("--out", required=True, help="prediction JSONL to write")
    args = ap.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        known = {json.loads(line)["id"] for line in f if line.strip()}

    preds, unparsed, duplicates = {}, [], []
    for path in args.responses:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                m = ID_RE.match(line)
                parsed = parse_output(m.group(2)) if m else None
                if not parsed or m.group(1) not in known:
                    unparsed.append((path, line))
                    continue
                case_id, (mode, command) = m.group(1), parsed
                if case_id in preds:
                    duplicates.append((path, case_id))
                    continue  # first answer wins
                preds[case_id] = {"id": case_id, "command": command, "mode": mode}

    with open(args.out, "w", encoding="utf-8") as f:
        for case_id in sorted(preds):
            f.write(json.dumps(preds[case_id]) + "\n")

    missing = sorted(known - preds.keys())
    print(f"parsed {len(preds)}/{len(known)} predictions -> {args.out}")
    if duplicates:
        print(f"{len(duplicates)} duplicate answers ignored (first kept)")
    if unparsed:
        print(f"{len(unparsed)} unparsed/unknown lines (format failures):")
        for path, line in unparsed[:10]:
            print(f"  {path}: {line[:80]}")
    if missing:
        print(f"{len(missing)} cases with no answer, e.g.: {', '.join(missing[:5])}")
    if not preds:
        sys.exit(1)


if __name__ == "__main__":
    main()
