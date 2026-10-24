#!/usr/bin/env python3
"""Give every stored buffer exactly one trailing newline, and re-derive the result.

Two spellings of one buffer have circulated since the first import: the
hand-authored rows end with a newline the way a POSIX text file does, and the
mined and generated corpora (kb13, sregex, emul, most cand) do not. 2,419 of
7,512 rows carried no final newline on 2026-09-06. That single invisible byte
cost real money:

  * Two buffer templates spanned train and test, because every guard compared
    raw strings while the allocator's `norm_buffer` ignores that byte. Fixed
    by hardening the guards -- but the guards only have to be clever because
    the data is not canonical.
  * The §6 prompt interpolates `buffer_before` verbatim between `<<<` and
    `>>>`, so 5,093 prompts carry a blank line before the closing delimiter
    and 2,419 do not. Nobody chose that; it is the source convention leaking
    into the model's input format.

Canonicalizing removes the class rather than guarding against it: byte
equality becomes semantic equality, and every prompt has the same shape.

The invariant is therefore in two halves, and both need vim to check:

    buffer_before == canonical(buffer_before)
    buffer_after  == what the sandbox produces from it

`buffer_after` is never patched by hand and never normalized -- whatever vim
writes is what gets stored, including `\n` for a line emptied in place and
`""` for a buffer whose last line was deleted, which are genuinely different
outputs. A row whose MEANING changes (not merely its spelling) is reported
and left alone rather than quietly rewritten.

Idempotent, and it costs one sandbox run per row to prove it: a second run
finds nothing to do.

Usage:
    python scripts/canonicalize_buffers.py --dry-run
    python scripts/canonicalize_buffers.py
"""
import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim               # noqa: E402
from build_corpus import DATASETS          # noqa: E402
from generate_data import norm_buffer      # noqa: E402


def canonical(text):
    """The buffer as a file on disk: every line ended, including the last.

    The empty buffer is the exception and is spelled "". Vim cannot hold zero
    lines -- an empty file opens as one empty line -- so "" and "\n" are the
    same buffer to it, verified in the sandbox: both produce the identical
    result for the same command. The executor emits "" when a command empties
    a buffer, so "" is the harness's own spelling and the one the corpus
    keeps; "\n" would otherwise be a second spelling of exactly the state
    this script exists to spell once.
    """
    if text is None:
        return None
    if text in ("", "\n"):
        return ""
    return text if text.endswith("\n") else text + "\n"


def rerun(row):
    """What the sandbox produces from the canonical buffer, or (None, why)."""
    mode = row.get("mode", "keys")
    cmd = row["gold_command"]
    if mode == "ex":
        cmd = cmd.lstrip(":")
    res = run_vim(canonical(row["buffer_before"]), cmd, mode=mode,
                  cursor=tuple(row.get("cursor") or (1, 1)))
    if res.get("blocked"):
        return None, "sandbox blocked: %s" % res["blocked"]
    if res.get("timeout"):
        return None, "timed out"
    if not res.get("was_valid"):
        return None, "vim reported the command invalid (rc=%s)" % res["return_code"]
    return res["buffer_after"], None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many rows (for a quick look)")
    args = ap.parse_args(argv)

    files, stats = collections.OrderedDict(), collections.Counter()
    changed_meaning, failed = [], []
    for ds, names in DATASETS.items():
        for name in names:
            path = REPO / "data" / ds / name
            if not path.exists():
                continue
            rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
            files[path] = rows

    seen = 0
    for path, rows in files.items():
        for row in rows:
            seen += 1
            if args.limit and seen > args.limit:
                break
            before_was = row["buffer_before"]
            after_was = row.get("buffer_after")
            produced, why = rerun(row)
            if why:
                failed.append((row["id"], why))
                continue
            if norm_buffer(produced) != norm_buffer(after_was or ""):
                # the canonical buffer makes the command mean something else
                changed_meaning.append((row["id"], row["gold_command"]))
                continue
            if before_was == canonical(before_was) and after_was == produced:
                stats["already canonical"] += 1
                continue
            row["buffer_before"] = canonical(before_was)
            row["buffer_after"] = produced
            stats["canonicalized"] += 1

    print("rows seen            : %d" % seen)
    print("already canonical    : %d" % stats["already canonical"])
    print("canonicalized        : %d" % stats["canonicalized"])
    if changed_meaning:
        print("MEANING CHANGED      : %d (left untouched)" % len(changed_meaning))
        for rid, cmd in changed_meaning[:20]:
            print("   %-16s %r" % (rid, cmd))
    if failed:
        print("COULD NOT RE-RUN     : %d (left untouched)" % len(failed))
        for rid, why in failed[:20]:
            print("   %-16s %s" % (rid, why))
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    if not stats["canonicalized"]:
        print("\nnothing to write")
        return 0
    for path, rows in files.items():
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("   rewrote %s (%d rows)" % (path.relative_to(REPO), len(rows)))
    print("\nnow run: python scripts/build_corpus.py --dataset all")
    print("then   : python scripts/select_datasets.py --datasets v3,v4 "
          "--out data/combined_cases.jsonl")
    print("then   : python scripts/refresh_splits.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
