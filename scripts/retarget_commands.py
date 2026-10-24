#!/usr/bin/env python3
"""Replace a mined row's gold command with an idiomatic equivalent, verified.

Rows mined from vim-implementation test suites carry the keystrokes the
upstream test happened to type. For most that is a fine command. For some it
is a fixture: a named-register juggle, an undo/redo pair, a text object that
deliberately fails to match. The buffer change is ordinary and describable,
but the *route* is one no plain request could motivate — so training on it
teaches the model to answer an ordinary request with a keystroke pile-up.

This retargets such a row: the mined buffers are kept exactly as verified,
and the command is replaced with one a person would actually write. Nothing
is taken on trust — every replacement is executed through src/executor.py and
must reproduce `buffer_after` under evaluate.py's line-list comparison, and
every instruction must survive the same leak guard apply_instructions.py
enforces, checked against the *new* command.

Provenance is recorded, not erased: the upstream keystrokes are preserved in
`source_command` and `command_provenance` marks the row as retargeted, so the
corpus never claims a hand-written command came from the upstream suite.

Input is {id: {"command": ..., "instruction": ...}}.

Usage (repo root):
    python scripts/retarget_commands.py --cases data/v3/emulator_cases.jsonl \\
                                        --map retarget.json [--dry-run]
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim            # noqa: E402
from evaluate import _norm              # noqa: E402
from apply_instructions import leaks    # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--provenance", default="claude-retargeted")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = pathlib.Path(args.cases)
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    by_id = {r["id"]: r for r in rows}
    given = json.loads(pathlib.Path(args.map).read_text(encoding="utf-8"))

    applied, failed, skipped = [], [], []
    for rid, spec in given.items():
        row = by_id.get(rid)
        if row is None:
            skipped.append((rid, "not in this file"))
            continue
        cmd, ins = spec["command"], spec["instruction"].strip()

        # 1. the replacement must actually reproduce the verified buffer
        res = run_vim(row["buffer_before"], cmd, mode=row.get("mode", "keys"),
                      cursor=tuple(row.get("cursor") or (1, 1)))
        if _norm(res.get("buffer_after")) != _norm(row["buffer_after"]):
            failed.append((rid, cmd, "does not reproduce buffer_after"))
            continue
        if not res.get("was_valid"):
            failed.append((rid, cmd, "vim reported the command invalid"))
            continue

        # 2. the instruction must not hand the model its own (new) answer
        why = leaks(ins, cmd, cmd)
        if why:
            failed.append((rid, cmd, "instruction leaks: " + why))
            continue

        row.setdefault("source_command", row.get("gold_command"))
        row["gold_command"] = cmd
        row["keys_raw"] = cmd
        row["command_provenance"] = args.provenance
        row["instruction"] = ins
        row["instruction_provenance"] = args.provenance
        row["instruction_pending"] = False
        applied.append(rid)

    print("cases in file      : %d" % len(rows))
    print("retargets given    : %d" % len(given))
    print("verified & applied : %d" % len(applied))
    if skipped:
        print("not in this file   : %d" % len(skipped))
    if failed:
        print("REFUSED            : %d" % len(failed))
        for rid, cmd, why in failed:
            print("   %-16s %-28r %s" % (rid, cmd, why))
    still = sum(1 for r in rows if not (r.get("instruction") or "").strip())
    print("still pending      : %d of %d" % (still, len(rows)))

    if failed:
        return 1
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\nwrote %s" % path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
