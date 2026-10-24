#!/usr/bin/env python3
"""Build the v3 candidate slice from CodeMirror, VsVim and toybox test suites.

Sibling of make_v3_emulator.py, for the sources mined after it. Same
discipline: the upstream expectation is never trusted, every row was replayed
through src/executor.py against real vim 8.2, and only exact agreement is
admitted.

Admission (the owner's rule, 2026-08-18):
  1. real vim reproduces the source's expected buffer with NO difference of
     any kind, whitespace included, under the sec 5 comparison (line lists
     ignoring one trailing newline)
  2. the buffer actually changes -- both as the source describes it and as
     vim produced it
  3. not already present in v3
  4. duplicate edits collapsed: where one edit is reachable by several
     keystroke spellings, one representative survives
  5. the shipped src/scope.py must also permit the command
  6. the command must not be an ACCUMULATION ARTIFACT. The CodeMirror miner
     emits one case per assertion with the keystrokes typed up to that point,
     so a test that increments repeatedly yields rows whose command is
     `<C-a>` sixteen times over. Those reproduce in vim and change the
     buffer, so every other rule admits them -- but nobody asks for them,
     which is why they resisted instruction authoring. Detected as a single
     key repeated five or more times, or a command over the configured
     [generate] max_command_len, and held rather than shipped.

Instructions are authored by this project, never taken from upstream: the
test titles name the command ("Can handle 'dw'"), so reusing them would leak
the answer into the model's input. Rows still awaiting one ship with
instruction == "" and instruction_pending == true.

Stdlib only. Usage:
    python scripts/make_v3_candidates.py [--in PATH] [--out PATH] [--edits PATH]

--edits merges an owner-exported instructions file from the review artifact,
overriding the authored text and marking those rows owner-edited.
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

DEFAULT_IN = ROOT / "data/external/emulator-tests/candidate_records.json"
DEFAULT_OUT = ROOT / "data/v3/candidate_cases.jsonl"

REPOS = {
    "codemirror": dict(corpus="CodeMirror", repo="github.com/replit/codemirror-vim",
                       commit="8640966b6977", license="MIT",
                       license_file="data/external/emulator-tests/LICENSE.CodeMirror",
                       prefix="cand-cm"),
    "vsvim": dict(corpus="VsVim", repo="github.com/VsVim/VsVim",
                  commit="2b0bc9b1943a", license="Apache-2.0",
                  license_file="data/external/emulator-tests/LICENSE.Apache-2.0",
                  prefix="cand-vsv"),
    "toybox": dict(corpus="toybox vi", repo="github.com/landley/toybox",
                   commit="b7ec52ac35e0", license="0BSD",
                   license_file="data/external/emulator-tests/LICENSE.toybox",
                   prefix="cand-tbx"),
}


MAX_COMMAND_LEN = 200          # [generate] max_command_len in src/settings.py
_TOKEN = re.compile(r"<[^>]+>|.")


def accumulation_artifact(command):
    """One key pressed over and over is a mining artifact, not a task."""
    if len(command) > MAX_COMMAND_LEN:
        return "command longer than max_command_len (%d)" % MAX_COMMAND_LEN
    # Count CONSECUTIVE runs, not total frequency. Counting frequency flagged
    # `<C-v>2jlllchello` because three of its `l`s are in the typed word
    # "hello" -- a real blockwise change, not an artifact. A pile-up is a run.
    tokens = _TOKEN.findall(command)
    run_token, run, best, best_token = None, 0, 0, None
    for tok in tokens:
        if tok == run_token:
            run += 1
        else:
            run_token, run = tok, 1
        if run > best:
            best, best_token = run, tok
    if best >= 5:
        return "key %r pressed %d times in a row" % (best_token, best)
    return None


def lines(text):
    text = text or ""
    if text.endswith("\n"):
        text = text[:-1]
    return text.split("\n")


def build(records, edits):
    kept, held, skipped = [], [], collections.Counter()
    counters = collections.Counter()

    for rec in records:
        if rec.get("admissible") != "yes":
            skipped[rec.get("excluded") or "not admissible"] += 1
            continue
        meta = REPOS.get(rec["source"])
        if meta is None:
            skipped["unknown source"] += 1
            continue

        command = rec["keys"].lstrip(":") if rec["mode"] == "ex" else rec["keys"]

        # the shipped screen is the final authority, exactly as in make_v3_emulator
        refused = screen_dangerous(command, rec["mode"])
        artifact = accumulation_artifact(command)

        instruction = rec.get("instruction") or ""
        provenance = "claude-authored" if instruction else "pending"
        if rec["uid"] in edits:
            instruction = edits[rec["uid"]]
            provenance = "owner-edited"

        counters[rec["source"]] += 1
        row = {
            # --- CLAUDE.md sec 5 schema ---
            "id": "%s-%04d" % (meta["prefix"], counters[rec["source"]]),
            "category": rec["category"],
            "instruction": instruction,
            "mode": rec["mode"],
            "buffer_before": rec["before"],
            "gold_command": command,
            "buffer_after": rec["expected"],
            "cursor": rec["cursor"],
            # --- provenance ---
            "source": "candidate-" + rec["source"],
            "source_corpus": meta["corpus"],
            "source_repo": meta["repo"],
            "source_commit": meta["commit"],
            "source_license": meta["license"],
            "source_license_file": meta["license_file"],
            "source_test": rec.get("name", ""),
            # --- authoring + review state ---
            "instruction_provenance": provenance,
            "instruction_pending": not bool(instruction),
            "keys_raw": rec["keys"],
            "tier": "",
            "tier_provenance": "unassigned",
            "whitespace_only_change": rec.get("wsonly") == "yes",
        }
        if artifact:
            row["pending_reason"] = artifact
            held.append(row)
            skipped["accumulation artifact"] += 1
            continue
        if refused:
            row["pending_reason"] = refused
            held.append(row)
            skipped["pending scope change: " + refused] += 1
            continue
        kept.append(row)
    return kept, held, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default=str(DEFAULT_IN))
    ap.add_argument("--out", dest="out", default=str(DEFAULT_OUT))
    ap.add_argument("--edits", default=None,
                    help="owner-exported instructions JSON from the review artifact")
    args = ap.parse_args()

    payload = json.loads(pathlib.Path(args.src).read_text())
    records = payload["cases"] if isinstance(payload, dict) else payload

    edits = {}
    if args.edits:
        for row in json.loads(pathlib.Path(args.edits).read_text()):
            if row.get("instruction_provenance") == "owner-edited" or row.get("edited"):
                edits[row["id" if "id" in row else "uid"]] = row["instruction"]

    kept, held, skipped = build(records, edits)

    out = pathlib.Path(args.out)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if held:
        hp = out.parent / "candidate_held.jsonl"
        with hp.open("w", encoding="utf-8") as fh:
            for row in held:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("held      %d rows -> %s" % (len(held), hp))

    print("read      %d mined records" % len(records))
    print("admitted  %d -> %s" % (len(kept), out))
    for src, n in sorted(collections.Counter(r["source"] for r in kept).items()):
        print("            %-24s %d" % (src, n))
    print("skipped:")
    for reason, n in skipped.most_common():
        print("            %-34s %d" % (reason[:34], n))
    prov = collections.Counter(r["instruction_provenance"] for r in kept)
    print("instructions: %s" % dict(prov))
    print("mode:         %s" % dict(collections.Counter(r["mode"] for r in kept)))
    print("categories:   %s" % dict(sorted(collections.Counter(r["category"] for r in kept).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
