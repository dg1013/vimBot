#!/usr/bin/env python3
"""Apply a curation change-set from the Corpus Curator artifact.

Two operations, both re-verified from scratch — the page proposes, vim decides:

  EDIT    a row's instruction, command, mode, buffer_before or cursor. Any
          change to the command or the buffer means `buffer_after` is stale,
          so it is ALWAYS recomputed by executing the (possibly new) command
          in the sandbox. The result must be valid, unblocked, in time, and
          must change the buffer; the instruction must survive the leak guard
          against the (possibly new) command; the cursor must exist.

  Both operations answer to §7.3 as well as to the sandbox: an edit that
  would put a buffer template, an instruction or a gold command on both
  sides of a split boundary is refused, keyed the way `build_splits` keys
  them. That check did not exist until 2026-09-06 -- giving a train row a
  test row's buffer was accepted, and only the standing split-file test
  caught it, after the fact.

  DELETE  a row, from whichever per-source file holds it. Deleted rows are
          appended to data/dropped_cases.jsonl with the reason and where they
          came from — the same courtesy the mined drops got, so a deletion
          stays auditable and reversible.

THE PROMOTION RULE (owner, 2026-08-25): **editing a v4 row promotes it to
v3.** An edit is the owner reviewing the row and standing behind it, which is
exactly the bar v3 asks for. The row moves from data/v4/handcrafted_cases.jsonl
to data/v3/handcrafted_cases.jsonl, keeps its id, and gains `promoted_from`
and `promoted_on`. This is the v4 phase-out running one reviewed row at a
time rather than as a big-bang migration; when v4 empties, delete its
DATASETS entry and it is over.

Deleting a v4 row does NOT promote it — deletion is a judgement that the row
should not exist, in either dataset.

Change-set shape (as exported by the artifact):
    {"edits":   {"<id>": {"instruction": ..., "command": ..., "mode": ...,
                          "buffer_before": ..., "cursor": [l, c]}},
     "deletes": {"<id>": "<reason>"}}
Only the fields actually changed need to appear in an edit.

Usage:
    python scripts/apply_curation.py --changes changes.json [--dry-run]
"""
import argparse
import collections
import datetime
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim            # noqa: E402
from apply_instructions import leaks    # noqa: E402
from build_corpus import DATASETS       # noqa: E402
from make_v3_emulator import guess_tier  # noqa: E402
from generate_data import norm_buffer, norm_instruction   # noqa: E402
from refresh_splits import buffer_key, command_key        # noqa: E402

DROPPED = REPO / "data" / "dropped_cases.jsonl"
SPLITS = REPO / "data" / "splits"
EDITABLE = ("instruction", "command", "mode", "buffer_before", "cursor")


def load_splits(splits_dir=SPLITS):
    """id -> split name, for every row in the frozen split files.

    Returns {} when the files are absent, which is the state a fresh clone or
    a unit test is in; the §7.3 check then has nothing to check against.
    """
    home = {}
    for name in ("train", "val", "test"):
        path = pathlib.Path(splits_dir) / ("%s.jsonl" % name)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    home[json.loads(line)["id"]] = name
    return home


def crosses_a_split(row, index, home):
    """Would this row put a buffer, instruction or gold command on both sides
    of a split boundary? Returns a reason or None.

    §7.3 is enforced when rows are PLACED (`refresh_splits.py --add-new`),
    but an edit to an already-placed row can create the same leak and nothing
    looked: giving a train row a test row's buffer was accepted until
    2026-09-06. The keys are the allocator's, so the two spellings of one
    template count as one (DECISIONS.md 2026-09-06).
    """
    mine = home.get(row["id"])
    if mine is None:
        return None                      # not placed yet; --add-new will guard it
    checks = [(buffer_key, "buffer template"),
              (lambda r: norm_instruction(r.get("instruction") or ""), "instruction"),
              (command_key, "gold command")]
    mine_keys = [(getter, label, getter(row)) for getter, label in checks]
    for rid, (_ds, _f, other) in index.items():
        if rid == row["id"]:
            continue
        theirs = home.get(rid)
        if theirs is None or theirs == mine:
            continue
        for getter, label, value in mine_keys:
            if getter(other) == value:
                return ("§7.3: this edit would put the %s in both %s (this row)"
                        " and %s (%s)" % (label, mine, theirs, rid))
    return None


# "The same buffer" and "the same instruction" are defined once, in
# generate_data, and every guard in the repo (repository) asks the same question of them.
# This file carried its own copy of norm_buffer until 2026-09-06 -- a third
# definition of a notion that only works if there is one (DECISIONS.md).
def norm(text):
    """`norm_buffer`, tolerating None the way this file's callers expect."""
    return None if text is None else norm_buffer(text)


norm_ins = norm_instruction


def load_all():
    """Every row, with the dataset and file that holds it."""
    index = {}
    files = collections.OrderedDict()
    for ds, names in DATASETS.items():
        for name in names:
            p = REPO / "data" / ds / name
            if not p.exists():
                continue
            rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
            files[(ds, name)] = rows
            for r in rows:
                index[r["id"]] = (ds, name, r)
    return index, files


def verify(row, index, triple_changed):
    """Re-derive buffer_after and re-run every gate. Returns (row, None|reason).

    triple_changed says whether this edit touched the command, buffer or
    cursor. The (buffer, cursor, command) dedup check only fires when it did:
    a pre-existing collision is not this edit's fault, and KB13 (the Kushman and Barzilay 2013 corpus) ships 195
    deliberate -a/-b paraphrase pairs that share exactly that triple. Blocking
    an instruction-only edit on one of those would make 195 rows uneditable.
    """
    mode = row.get("mode", "keys")
    if mode not in ("ex", "keys"):
        return None, "mode must be ex or keys, got %r" % mode
    cmd = row["gold_command"]
    if mode == "ex":
        cmd = cmd.lstrip(":")
    buf = row["buffer_before"]
    cur = row.get("cursor") or [1, 1]
    try:
        cur = [int(cur[0]), int(cur[1])]
    except (TypeError, ValueError, IndexError):
        return None, "cursor must be [line, col]"

    lines = norm(buf) or [""]
    if not (1 <= cur[0] <= len(lines)):
        return None, "cursor line %d outside the %d-line buffer" % (cur[0], len(lines))
    if not (1 <= cur[1] <= max(1, len(lines[cur[0] - 1]) + 1)):
        return None, ("cursor col %d outside line %d (%d chars)"
                      % (cur[1], cur[0], len(lines[cur[0] - 1])))

    ins = (row.get("instruction") or "").strip()
    if not ins:
        return None, "instruction is empty"
    why = leaks(ins, cmd, cmd)
    if why:
        return None, "instruction " + why

    res = run_vim(buf, cmd, mode=mode, cursor=tuple(cur))
    if res.get("blocked"):
        return None, "sandbox blocked: %s" % res["blocked"]
    if res.get("timeout"):
        return None, "timed out"
    if not res.get("was_valid"):
        return None, "vim reported the command invalid (rc=%s)" % res["return_code"]
    after = res["buffer_after"]
    if norm(after) == norm(buf):
        return None, "no-op: the buffer would not change (banned by §7)"

    # dedup against every other row in every dataset
    for rid, (_ds, _f, other) in index.items():
        if rid == row["id"]:
            continue
        if (norm_ins(other.get("instruction") or "") == norm_ins(ins)
                and other["buffer_before"] == buf):
            return None, "would duplicate (instruction, buffer) of %s" % rid
        if (triple_changed
                and other["buffer_before"] == buf
                and tuple(other.get("cursor") or (1, 1)) == tuple(cur)
                and other.get("gold_command") == cmd
                and not row.get("paraphrase_of")):
            return None, ("this edit would make it a duplicate of %s on "
                          "(buffer, cursor, command)" % rid)

    row["gold_command"] = cmd
    row["cursor"] = cur
    row["buffer_after"] = after
    row["instruction"] = ins
    if mode == "keys":
        row["keys_raw"] = cmd
    elif "keys_raw" in row:
        del row["keys_raw"]
    # An AUTHORED tier is a curated judgement and survives an edit; only a
    # heuristic one is re-derived. Overwriting authored tiers here silently
    # emptied three quota cells on 2026-09-06, because guess_tier disagrees
    # with the taxonomy on blockwise and composed commands (DECISIONS.md).
    if row.get("tier_provenance") != "authored":
        row["tier"] = guess_tier(mode, cmd)
        row["tier_provenance"] = "heuristic"
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--changes", required=True)
    ap.add_argument("--provenance", default="owner-edited")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(pathlib.Path(args.changes).read_text(encoding="utf-8"))
    edits = spec.get("edits") or {}
    deletes = spec.get("deletes") or {}
    index, files = load_all()
    home = load_splits()

    applied, promoted, removed, refused, retier = [], [], [], [], []
    today = datetime.date.today().isoformat()

    # ---- deletes first: an id cannot be both edited and deleted
    for rid, reason in deletes.items():
        if rid in edits:
            refused.append((rid, "listed as both an edit and a delete"))
            continue
        if rid not in index:
            refused.append((rid, "unknown id"))
            continue
        ds, name, row = index[rid]
        rec = dict(row)
        rec["dropped_from"] = "%s/%s" % (ds, name)
        rec["dropped_reason"] = reason or "removed via the curator"
        rec["dropped_on"] = today
        removed.append((rid, ds, name, rec))

    # ---- edits
    for rid, change in edits.items():
        if rid in deletes:
            continue          # already refused in the delete pass; do not edit too
        if rid not in index:
            refused.append((rid, "unknown id"))
            continue
        ds, name, row = index[rid]
        new = dict(row)
        touched = []
        for k in EDITABLE:
            if k not in change:
                continue
            target = "gold_command" if k == "command" else k
            if change[k] != new.get(target):
                new[target] = change[k]
                touched.append(k)
        if not touched:
            refused.append((rid, "no field actually changed"))
            continue
        triple_changed = any(k in touched for k in ("command", "buffer_before", "cursor"))
        checked, why = verify(new, index, triple_changed)
        if why:
            refused.append((rid, "%s (changed: %s)" % (why, ", ".join(touched))))
            continue
        if any(k in touched for k in ("buffer_before", "command", "instruction")):
            why = crosses_a_split(checked, index, home)
            if why:
                refused.append((rid, "%s (changed: %s)" % (why, ", ".join(touched))))
                continue
        checked["instruction_provenance"] = args.provenance
        checked["curated_on"] = today
        if "command" in touched and checked.get("tier_provenance") == "authored":
            # The tier survives an edit (see verify), which is right for a
            # reworded instruction and questionable for a new command: a
            # different command can deserve a different tier. Say so rather
            # than let the omission pass silently.
            retier.append((rid, row.get("gold_command"), checked["gold_command"],
                           checked.get("tier")))
        if ds == "v4":
            checked["promoted_from"] = "v4"
            checked["promoted_on"] = today
            checked["dataset"] = "v3"
            promoted.append((rid, touched))
        else:
            applied.append((rid, touched))
        index[rid] = (ds, name, checked)

    print("changes given : %d edit(s), %d delete(s)" % (len(edits), len(deletes)))
    print("edited in v3  : %d" % len(applied))
    print("promoted v4->v3: %d" % len(promoted))
    print("deleted       : %d" % len(removed))
    if refused:
        print("REFUSED       : %d" % len(refused))
        for rid, why in refused:
            print("   %-14s %s" % (rid, why))
    if retier:
        print("CHECK THE TIER: %d row(s) changed command under an AUTHORED tier;"
              % len(retier))
        print("                the tier was kept and may no longer fit.")
        for rid, was, now, tier in retier:
            print("   %-14s %s  %r -> %r" % (rid, tier, was, now))
    for rid, touched in promoted:
        print("   promoted %-12s (changed: %s)" % (rid, ", ".join(touched)))

    if refused:
        print("\nnothing written — fix the refusals and re-run")
        return 1
    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    # ---- rewrite each touched source file
    drop_ids = {rid for rid, _ds, _n, _r in removed}
    promo_ids = {rid for rid, _t in promoted}
    promoted_rows = []
    for (ds, name), rows in files.items():
        out, changed = [], False
        for r in rows:
            rid = r["id"]
            if rid in drop_ids:
                changed = True
                continue
            if rid in promo_ids and ds == "v4":
                promoted_rows.append(index[rid][2])
                changed = True
                continue
            if rid in index and index[rid][2] is not r:
                out.append(index[rid][2])
                changed = True
            else:
                out.append(r)
        if changed:
            p = REPO / "data" / ds / name
            with p.open("w", encoding="utf-8") as fh:
                for r in out:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print("  rewrote %s/%s (%d rows)" % (ds, name, len(out)))

    if promoted_rows:
        p = REPO / "data" / "v3" / "handcrafted_cases.jsonl"
        with p.open("a", encoding="utf-8") as fh:
            for r in promoted_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("  promoted %d row(s) -> v3/handcrafted_cases.jsonl" % len(promoted_rows))

    if removed:
        with DROPPED.open("a", encoding="utf-8") as fh:
            for _rid, _ds, _n, rec in removed:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("  recorded %d deletion(s) -> %s" % (len(removed), DROPPED))

    print("\nnow run: python scripts/build_corpus.py --dataset all")
    print("then   : python src/evaluate.py --cases data/v3/v3_cases.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
