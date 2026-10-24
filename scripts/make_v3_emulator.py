#!/usr/bin/env python3
"""Build the v3 emulator slice from mined vim-emulator test suites.

Input:  data/external/emulator-tests/mined_records.json  -- every triple mined
        from the two suites, each already replayed through src/executor.py
        against real vim (see that directory's README for provenance).

Output: data/v3/emulator_cases.jsonl in the CLAUDE.md sec 5 schema, plus
        additive provenance fields so every row can be traced to the repo (repository),
        commit, file and test it came from.

Admission rules, in order:
  1. verdict == PASS      -- real vim reproduced the emulator's expectation
  2. no scope flag        -- inside the all-commands scope
                             (data/command_coverage.md)
  3. buffer must change   -- no-ops were already dropped upstream; re-checked
                             here so this script is safe to run on any input
  4. the SHIPPED screen must permit it -- src/scope.py still enforces frozen
                             scope v1 for named registers and macro recording.
                             Rows the current code refuses are written to
                             data/v3/emulator_pending_scope.jsonl instead of
                             the main file, so the main file always passes
                             `python src/evaluate.py --cases ...` 100%.

WHAT THIS SCRIPT DELIBERATELY DOES NOT DO: write an `instruction`. The
emulator test titles name the command ("Can handle 'dw'"), so using them
would leak the answer into the model's input. Every row ships with
instruction == "" and instruction_provenance == "pending", and the file is
NOT merged into data/v3/v3_cases.jsonl until instructions exist.

Stdlib only. Usage:
    python scripts/make_v3_emulator.py [--in PATH] [--out PATH]
"""

import argparse
import collections
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from scope import screen_dangerous  # noqa: E402
DEFAULT_IN = ROOT / "data/external/emulator-tests/mined_records.json"
DEFAULT_OUT = ROOT / "data/v3/emulator_cases.jsonl"

REPOS = {
    "vscodevim": {
        "corpus": "VSCodeVim",
        "repo": "github.com/VSCodeVim/Vim",
        "commit": "1118267006d0",
        "license": "MIT",
        "id_prefix": "emul-vsc",
    },
    "ideavim": {
        "corpus": "IdeaVim",
        "repo": "github.com/JetBrains/ideavim",
        "commit": "e66a3915a9b4",
        "license": "MIT",
        "id_prefix": "emul-idea",
    },
}

_COUNT_OR_RANGE = re.compile(r"\d")
_REGEX_CRAFT = re.compile(r"\\\(|\\\||\\%|\\z[se]|\\[dwsa]|\[\^?.+\]|\\=")
_FLAGS = re.compile(r"/[gice]+\s*$|!\s*$|\s+[niur]\b")


def guess_tier(mode, command):
    """Best-effort (category, tier) placement. Marked heuristic in the output --
    a curated tier assignment is authoring work, not something to infer."""
    if _REGEX_CRAFT.search(command):
        return "T3"
    ops = len(re.findall(r"[dcyxsprJ~<>]|g[uU~q?]|<C-[ax]>", command))
    if ops >= 3:
        return "T3"
    if _COUNT_OR_RANGE.search(command) or _FLAGS.search(command) or ops == 2:
        return "T2"
    return "T1"


def build(records):
    kept, pending, skipped = [], [], collections.Counter()
    counters = collections.Counter()

    for rec in records:
        if rec.get("verdict") != "PASS":
            skipped[rec.get("verdict", "?")] += 1
            continue
        if rec.get("scope_note"):
            skipped["out of scope: " + rec["scope_note"]] += 1
            continue
        before = rec["buffer_before"]
        after = rec["buffer_expected"]
        if before.rstrip("\n") == after.rstrip("\n"):
            skipped["no-op"] += 1
            continue

        meta = REPOS[rec["source"]]
        counters[rec["source"]] += 1
        row = {
            # --- CLAUDE.md sec 5 schema ---
            "id": "%s-%04d" % (meta["id_prefix"], counters[rec["source"]]),
            "category": rec["category"],
            "instruction": "",
            "mode": rec["mode"],
            "buffer_before": before,
            "gold_command": rec["command"],
            "buffer_after": after,
            "cursor": rec["cursor"],
            # --- additive: provenance ---
            "source": "emulator-" + rec["source"],
            "source_corpus": meta["corpus"],
            "source_repo": meta["repo"],
            "source_commit": meta["commit"],
            "source_license": meta["license"],
            "source_file": rec["file"],
            "source_test_title": rec.get("title", ""),
            "mining_round": rec.get("batch", ""),
            # --- additive: authoring + review state ---
            "instruction_provenance": "pending",
            "tier": guess_tier(rec["mode"], rec["command"]),
            "tier_provenance": "heuristic",
            "theme": rec.get("theme", ""),
            "keys_raw": rec.get("keys_raw", ""),
            "was_frozen_v1": rec.get("was_frozen_v1", "no"),
        }
        refused = screen_dangerous(row["gold_command"], row["mode"])
        if refused:
            row["pending_reason"] = refused
            pending.append(row)
            skipped["pending scope change: " + refused] += 1
            continue
        kept.append(row)

    # sec 7.3 cannot run without instructions; flag exact collisions so the
    # later dedup is mechanical rather than a re-derivation.
    groups = collections.defaultdict(list)
    for row in kept:
        groups[(row["mode"], row["gold_command"], row["buffer_before"])].append(row)
    dup_groups = 0
    for key, rows in groups.items():
        if len(rows) > 1:
            dup_groups += 1
            for row in rows:
                row["dup_group"] = "dup-%04d" % dup_groups
    return kept, pending, skipped, dup_groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.src).read_text())
    records = payload["cases"] if isinstance(payload, dict) else payload
    kept, pending, skipped, dup_groups = build(records)

    out = pathlib.Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if pending:
        pend_path = out.parent / "emulator_pending_scope.jsonl"
        with pend_path.open("w", encoding="utf-8") as fh:
            for row in pending:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("held     %d rows -> %s" % (len(pending), pend_path))
        print("           (admissible under data/command_coverage.md, but"
              " src/scope.py has not been updated yet)")
    print("read     %d mined records" % len(records))
    print("admitted %d -> %s" % (len(kept), out))
    for source, n in sorted(collections.Counter(r["source"] for r in kept).items()):
        print("           %-22s %d" % (source, n))
    print("skipped:")
    for reason, n in skipped.most_common():
        print("           %-22s %d" % (reason, n))
    print("mode:      %s" % dict(collections.Counter(r["mode"] for r in kept)))
    print("tier:      %s (heuristic)" % dict(sorted(collections.Counter(r["tier"] for r in kept).items())))
    print("category:  %s" % dict(sorted(collections.Counter(r["category"] for r in kept).items())))
    print("cursor:    %d positioned, %d at 1:1"
          % (sum(1 for r in kept if r["cursor"] != [1, 1]),
             sum(1 for r in kept if r["cursor"] == [1, 1])))
    print("dup groups: %d (exact command+buffer collisions, flagged not dropped)" % dup_groups)
    print("instructions pending: %d of %d" % (sum(1 for r in kept if not r["instruction"]), len(kept)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
