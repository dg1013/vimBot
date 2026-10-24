#!/usr/bin/env python3
"""Ingest hand-authored case drafts: execute, gate, append to the corpus file.

Takes the JSON (JavaScript Object Notation) exported by the Case Forge artifact (a list of draft objects:
instruction, mode, command, buffer_before, cursor, category, theme,
optional expected_after) and turns each into a full §5 row by running the
command in the sandboxed executor. Nothing is taken from the draft on trust:

  * the command must execute valid, unblocked, within timeout
  * the buffer must change — no-ops are banned outright (§7, reward hacking)
  * if the draft states an expected_after, vim must agree with it
    (line-list comparison); a disagreement rejects the draft — that is the
    same agreement gate the emulator intake used
  * the instruction must survive apply_instructions.leaks against the command
  * the cursor must exist in buffer_before (a lesson from cand-cm-0024)
  * duplicates are rejected against every existing v3 source and within the
    batch, on both (instruction, buffer_before) and (buffer, cursor, command)

Valid drafts are appended to data/v3/handcrafted_cases.jsonl with sequential
hc-#### ids; rejects are reported with reasons. Exit 1 if anything was
rejected (the valid rows are still written).

Usage:
    python scripts/ingest_handcrafted.py --input drafts.json \\
        [--provenance owner-authored] [--dry-run]
"""
import argparse
import datetime
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim            # noqa: E402
from apply_instructions import leaks    # noqa: E402
from make_v3_emulator import guess_tier  # noqa: E402  (one heuristic, no drift)

sys.path.insert(0, str(REPO / "scripts"))
from build_corpus import DATASETS      # noqa: E402  (one manifest, no drift)


def norm(text):
    if text is None:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def vis(line):
    """Make the invisible visible: tabs and trailing spaces are exactly what
    expected-vs-vim mismatches are made of."""
    out = line.replace("\t", "-->\t")
    stripped = out.rstrip(" ")
    return stripped + "." * (len(out) - len(stripped))


def diff_block(stated, got):
    lines = ["stated expected_after vs what vim produced "
             "(--> is a tab, trailing dots are spaces):"]
    n = max(len(stated), len(got))
    for i in range(n):
        a = stated[i] if i < len(stated) else "<missing line>"
        b = got[i] if i < len(got) else "<missing line>"
        mark = " " if a == b else "!"
        lines.append("  %s line %-2d stated %r" % (mark, i + 1, vis(a)))
        if a != b:
            lines.append("    line %-2d vim    %r" % (i + 1, vis(b)))
    return "\n".join(lines)


def norm_ins(text):
    return " ".join(text.split()).lower()


def existing_keys():
    """pairs: (instruction, buffer). triples: (buffer, cursor, command) -> id,
    so a paraphrase can record which row it re-phrases."""
    pairs, triples, max_id = set(), {}, 0
    for ds, names in DATASETS.items():
        for name in names:
            p = REPO / "data" / ds / name
            if not p.exists():
                continue
            _scan(p, pairs, triples)
            max_id = max(max_id, _max_hc(p))
    return pairs, triples, max_id


def _max_hc(p):
    hi = 0
    for line in p.open(encoding="utf-8"):
        if line.strip():
            rid = json.loads(line)["id"]
            if rid.startswith("hc-"):
                hi = max(hi, int(rid[3:]))
    return hi


def _scan(p, pairs, triples):
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ins = r.get("instruction") or ""
        if ins.strip():
            pairs.add((norm_ins(ins), r["buffer_before"]))
        tk = (r["buffer_before"], tuple(r.get("cursor") or (1, 1)),
              r.get("gold_command"))
        triples.setdefault(tk, r["id"])



TIERS = ("T1", "T2", "T3")


def draft_tier(d):
    """(tier, provenance) for a draft.

    A draft may state its own tier. guess_tier is explicitly best-effort --
    its own docstring says "a curated tier assignment is authoring work, not
    something to infer" -- and for blockwise commands it is simply wrong: the
    `<` and `>` of `<C-v>` and `<Esc>` count as shift operators, so every
    `<C-v>...` command scores three-plus operations and lands in T3. That is
    why visual-block held 41 T3 rows and none in T1 or T2 (found 2026-09-06
    while filling the quota). An authored tier is recorded as such, so the
    curated and the inferred stay distinguishable in the data.
    """
    t = (d.get("tier") or "").strip().upper()
    if t in TIERS:
        return t, "authored"
    return guess_tier(d["mode"], d["command"]), "heuristic"


def gate(d, i, pairs, triples, allow_paraphrase=False):
    """Return (row, None) or (None, reason)."""
    for k in ("instruction", "mode", "command", "buffer_before", "category"):
        if not (d.get(k) or "").strip():
            return None, "missing field %r" % k
    if (d.get("tier") or "").strip() and (d["tier"] or "").strip().upper() not in TIERS:
        return None, "tier must be one of %s, got %r" % (", ".join(TIERS), d["tier"])
    mode = d["mode"]
    if mode not in ("ex", "keys"):
        return None, "mode must be ex or keys, got %r" % mode
    cmd = d["command"]
    if mode == "ex":
        cmd = cmd.lstrip(":")           # stored convention: no leading colon
    ins = d["instruction"].strip()
    buf = d["buffer_before"]
    cur = d.get("cursor") or [1, 1]
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
    after = res.get("buffer_after")
    if norm(after) == norm(buf):
        return None, "no-op: the buffer did not change (banned by §7)"

    exp = d.get("expected_after")
    if exp is not None and str(exp).strip() != "" and norm(exp) != norm(after):
        return None, "vim disagrees.\n" + diff_block(norm(exp), norm(after))

    pk = (norm_ins(ins), buf)
    tk = (buf, tuple(cur), cmd)
    if pk in pairs:
        return None, "duplicate (instruction, buffer) of an existing case"
    paraphrase_of = None
    if tk in triples:
        # Same edit, same buffer, different words. KB13 (the Kushman and Barzilay 2013 corpus) ships exactly this as
        # its -a/-b annotator pairs and sec 7 asks for paraphrase diversity —
        # but accidental redundancy is the retarget lesson, so it stays a
        # reject unless explicitly allowed, and the kinship is recorded so
        # split clustering can keep the pair together.
        if not allow_paraphrase:
            return None, ("duplicate (buffer, cursor, command) of %s — a "
                          "differently-worded instruction for the same edit "
                          "is a paraphrase; pass --allow-paraphrase to admit "
                          "it deliberately" % triples[tk])
        paraphrase_of = triples[tk]
    pairs.add(pk)
    triples.setdefault(tk, "batch[%d]" % i)

    row = {
        "id": None,                      # assigned after gating
        "category": d["category"],
        "instruction": ins,
        "mode": mode,
        "buffer_before": buf,
        "gold_command": cmd,
        "buffer_after": after,
        "cursor": cur,
        "source": "handcrafted",
        "source_corpus": "vimBot handcrafted",
        "source_license": "project-original",
        "tier": draft_tier(dict(d, mode=mode, command=cmd))[0],
        "tier_provenance": draft_tier(dict(d, mode=mode, command=cmd))[1],
        "theme": d.get("theme") or "",
        "authored_on": datetime.date.today().isoformat(),
    }
    if mode == "keys":
        row["keys_raw"] = cmd
    if paraphrase_of:
        row["paraphrase_of"] = paraphrase_of
    return row, None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--dataset", default="v4", choices=sorted(DATASETS),
                    help="which dataset the rows are appended to (default v4, "
                         "the hand-authored corpus)")
    ap.add_argument("--provenance", default="owner-authored")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-paraphrase", action="store_true",
                    help="admit a draft whose (buffer, cursor, command) matches "
                         "an existing case when its instruction differs; the row "
                         "records paraphrase_of")
    args = ap.parse_args()

    drafts = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(drafts, list):
        sys.exit("input must be a JSON list of draft objects")

    out_path = REPO / "data" / args.dataset / "handcrafted_cases.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)
    pairs, triples, max_id = existing_keys()
    good, rejects = [], []
    for i, d in enumerate(drafts):
        row, why = gate(d, i, pairs, triples, args.allow_paraphrase)
        if why:
            rejects.append((i, d.get("instruction", "")[:50], why))
        else:
            row["instruction_provenance"] = args.provenance
            row["dataset"] = args.dataset
            good.append(row)

    for k, row in enumerate(good, start=max_id + 1):
        row["id"] = "hc-%04d" % k

    print("drafts given   : %d" % len(drafts))
    print("passed the gate: %d" % len(good))
    if rejects:
        print("REJECTED       : %d" % len(rejects))
        for i, ins, why in rejects:
            if "\n" in why:
                print("   draft[%d] %r" % (i, ins))
                for line in why.split("\n"):
                    print("      " + line)
            else:
                print("   draft[%d] %-52r %s" % (i, ins, why))

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 1 if rejects else 0
    if good:
        with out_path.open("a", encoding="utf-8") as fh:
            for row in good:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("\nappended %d rows -> %s" % (len(good), out_path))
        print("now run: python scripts/build_corpus.py --dataset %s" % args.dataset)
        print("then   : python src/evaluate.py --cases data/%s/%s_cases.jsonl"
              % (args.dataset, args.dataset))
    return 1 if rejects else 0


if __name__ == "__main__":
    sys.exit(main())
