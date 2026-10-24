#!/usr/bin/env python3
"""Refresh the split files by id after a curation pass, never reassigning.

DECISIONS.md 2026-09-04: `build_splits.py` is not stable under deletion — a
rebuild after the test-slice verification would have kept only 74 of the
364 verified test rows. The split files are therefore the frozen reference,
and this is the only way they change: rows no longer in the corpus are
dropped, rows whose case fields changed (an owner edit, a promotion) are
refreshed from the corpus, split-only fields such as `needs_review` are
kept, and every other line is left byte-identical. A full rebuild needs a
new verification pass and its own DECISIONS entry.

Newly authored rows are a separate problem: they are in the corpus but in
no split file, so they would never train. `--add-new <split>` appends them,
and because the split files are frozen it will only ever add to the split
you name — test and val hold verified rows and are not somewhere new,
unreviewed cases should appear. It refuses any row that would leak — a
buffer, a normalized instruction or a gold command that already appears in
either of the OTHER two splits (§7.3 keeps a command inside one split).

Both guards compare on the ALLOCATOR's keys, not on raw strings: two buffers
that differ only by a final newline are one template to `norm_buffer` and to
`evaluate.py`, and a gold command is normalized the way `build_splits` does it
(a leading `:` and outer whitespace dropped in ex mode, `<C-a>`/`<C-A>` folded
in keys mode). Comparing bytes here is how `1\n2\n3\n4\n5` in test and
`1\n2\n3\n4\n5\n` in train came to be one template on both sides of the
boundary (2026-09-06): the mined corpora store no final newline and the
hand-authored ones do, so the seam runs straight along the source boundary.

`--move <id>[,<id>...] --to <split>` moves rows between the frozen files.
It exists because `--add-new` guarded the command against test alone until
2026-09-06, so eight quota-fill rows reached train while six val rows held
their commands; repairing that means moving rows, and a frozen file should
be mutated by this tool or not at all. The move is refused if it would
leave a buffer or a command on both sides of a boundary — the same rules
the append obeys, checked against the state after the move.

Run it last in the curation chain, after `apply_curation.py`,
`build_corpus.py --dataset all` and `select_datasets.py`:

    python scripts/refresh_splits.py            # data/combined_cases.jsonl -> data/splits/
    python scripts/refresh_splits.py --dry-run  # report only
    python scripts/refresh_splits.py --add-new train   # place newly authored rows
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_splits import norm_command                    # noqa: E402
from generate_data import norm_buffer, norm_instruction  # noqa: E402


def buffer_key(row):
    """`build_splits.leakage_keys`' notion of one buffer template."""
    return norm_buffer(row["buffer_before"])


def command_key(row):
    """Ditto for a gold command. Mode is part of it: keys `dd` and ex `dd`
    are different commands, and the allocator keys them apart."""
    mode = row.get("mode", "ex")
    return (mode, norm_command(row["gold_command"], mode))


def show(key):
    """A buffer or command key, printable in a refusal message."""
    if isinstance(key, tuple) and key and key[0] in ("ex", "keys"):
        return "%s: %s" % key
    return "\n".join(key) if isinstance(key, tuple) else str(key)


def move_rows(args, ids, corpus):
    """Move named rows into args.to, refusing any move that would leak.

    The check is over the state AFTER the move, so it catches both halves of
    a leak: a buffer or gold command that the moved row would put on both
    sides of a boundary, and one it would leave behind.
    """
    splits = {}
    for name in ("train", "val", "test"):
        path = pathlib.Path(args.splits) / f"{name}.jsonl"
        with path.open(encoding="utf-8") as fh:
            splits[name] = [json.loads(l) for l in fh if l.strip()]
    have = {r["id"]: name for name in splits for r in splits[name]}
    missing = [i for i in ids if i not in have]
    if missing:
        raise SystemExit("not in any split: %s" % missing)
    already = [i for i in ids if have[i] == args.to]
    if already:
        raise SystemExit("already in %s: %s" % (args.to, already))

    moved = [r for name in splits for r in splits[name] if r["id"] in ids]
    after = {name: [r for r in rows if r["id"] not in ids]
             for name, rows in splits.items()}
    after[args.to] = after[args.to] + [corpus.get(r["id"], r) for r in moved]
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            for key, label in ((buffer_key, "buffer"),
                               (command_key, "gold command")):
                shared = ({key(r) for r in after[a]} & {key(r) for r in after[b]})
                if shared:
                    raise SystemExit(
                        "refused: the move would leave %d %s(s) in both %s and %s, "
                        "e.g. %r" % (len(shared), label, a, b,
                                     show(sorted(shared)[0])))

    for name, rows in after.items():
        print("%-5s %5d -> %5d" % (name, len(splits[name]), len(rows)))
    if args.dry_run:
        return
    for name, rows in after.items():
        path = pathlib.Path(args.splits) / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("   moved %d row(s) -> %s" % (len(moved), args.to))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--splits", default=str(REPO / "data" / "splits"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--add-new", choices=("train", "val", "test"), default=None,
                    help="append corpus rows that are in no split to this one")
    ap.add_argument("--move", default=None, metavar="IDS",
                    help="comma-separated ids to move between split files")
    ap.add_argument("--to", choices=("train", "val", "test"), default=None,
                    help="destination split for --move")
    args = ap.parse_args()

    corpus = {}
    with open(args.cases, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                corpus[row["id"]] = row

    for split in ("train", "val", "test"):
        path = pathlib.Path(args.splits) / f"{split}.jsonl"
        with path.open(encoding="utf-8") as fh:
            lines = [l for l in fh if l.strip()]
        out, dropped, refreshed = [], [], []
        for line in lines:
            row = json.loads(line)
            if row["id"] not in corpus:
                dropped.append(row["id"])
                continue
            merged = dict(row)
            merged.update(corpus[row["id"]])      # case fields from the corpus,
            if merged != row:                     # split-only fields kept
                refreshed.append(row["id"])
            out.append(json.dumps(merged, ensure_ascii=False) + "\n")
        print(f"{split:5s} {len(lines):5d} -> {len(out):5d}   dropped {len(dropped):3d}"
              f"   refreshed {len(refreshed):3d}"
              + (f"  {refreshed}" if 0 < len(refreshed) <= 8 else ""))
        if not args.dry_run:
            with path.open("w", encoding="utf-8") as fh:
                fh.writelines(out)
    if args.move:
        if not args.to:
            ap.error("--move needs --to")
        move_rows(args, [i for i in args.move.split(",") if i.strip()], corpus)

    if args.add_new:
        placed = set()
        for name in ("train", "val", "test"):
            path = pathlib.Path(args.splits) / f"{name}.jsonl"
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        placed.add(json.loads(line)["id"])
        # The guard is about the OTHER splits, not the target: a row may
        # share a command with the split it is joining -- that is what keeps a
        # command inside one split (§7.3) -- but never with a different one.
        # One dict per field: buffers and instructions used to share a
        # namespace, so a buffer could in principle be matched against an
        # instruction and reported as the wrong kind of collision.
        guard_bufs, guard_ins, guard_cmds = {}, {}, {}
        for name in ("train", "val", "test"):
            if name == args.add_new:
                continue
            path = pathlib.Path(args.splits) / f"{name}.jsonl"
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        guard_bufs[buffer_key(r)] = name
                        guard_ins[norm_instruction(r["instruction"])] = name
                        guard_cmds[command_key(r)] = name
        add, refused = [], []
        for rid, row in corpus.items():
            if rid in placed:
                continue
            why = None
            if buffer_key(row) in guard_bufs:
                why = "buffer already in " + guard_bufs[buffer_key(row)]
            elif norm_instruction(row["instruction"]) in guard_ins:
                why = ("instruction already in "
                       + guard_ins[norm_instruction(row["instruction"])])
            elif command_key(row) in guard_cmds:
                why = ("gold command also appears in %s (§7.3 keeps a command "
                       "in one split)" % guard_cmds[command_key(row)])
            (refused if why else add).append((rid, why))
        print(f"unplaced rows: {len(add) + len(refused)}   to {args.add_new}: {len(add)}"
              f"   refused: {len(refused)}")
        for rid, why in refused[:10]:
            print(f"   refused {rid}: {why}")
        if add and args.add_new == "test":
            print("   NOTE: these rows join the VERIFIED test slice and carry no "
                  "owner verdict yet; they need a review pass.")
        if add and not args.dry_run:
            path = pathlib.Path(args.splits) / f"{args.add_new}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                for rid, _ in add:
                    fh.write(json.dumps(corpus[rid], ensure_ascii=False) + "\n")
            try:
                shown = path.relative_to(REPO)
            except ValueError:                 # --splits outside the repo (repository)
                shown = path
            print(f"   appended {len(add)} rows -> {shown}")

    if args.dry_run:
        print("--dry-run: nothing written")


if __name__ == "__main__":
    main()
