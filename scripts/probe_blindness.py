#!/usr/bin/env python3
"""Discriminative-power probe: can each case's buffer tell gold from a
sloppier command?

For every case, the gold command is mutated two ways — full-line anchor
stripped (containment variant) and word boundaries stripped — and executed
in the sandboxed vim. A case is *blind* on an axis when the mutated command
produces the identical buffer, i.e. exec-match could not penalize that
mistake. This is the Phase-A1 acceptance metric (v3 plan): legacy buffers
measured 66% anchor-blind / 78% word-blind on 2026-08-14; witness buffers
target <5%.

Usage (repo root):
    python scripts/probe_blindness.py [--cases FILE.jsonl ...] [--out out.json]
Default cases: all three split files. Runs ~8 vim workers (§9 pattern).
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from executor import run_vim  # noqa: E402

WRAP = re.compile(r"^([gv])/\^\\%\((.*)\\\)\$/d$")


def norm(buf):
    return buf[:-1].split("\n") if buf.endswith("\n") else buf.split("\n")


def probe(row):
    m = WRAP.match(row.get("gold_command") or row.get("command", ""))
    out = {"id": row.get("id", "?")}
    if not m:
        out["unparsed"] = True
        return out
    kind, inner = m.groups()
    if not (inner.startswith(".*") and inner.endswith(".*")):
        if "$" in inner:
            # lookahead-compiled conjunctions keep internal anchors, so
            # dropping the outer ^$ is a no-op mutation — probe invalid
            out["anchor_inherent"] = True
        else:
            res = run_vim(row["buffer_before"], f"{kind}/\\%({inner}\\)/d",
                          mode="ex")
            out["anchor_blind"] = bool(
                res["was_valid"] and norm(res["buffer_after"]) == norm(row["buffer_after"]))
    if "\\<" in inner or "\\>" in inner:
        stripped = inner.replace("\\<", "").replace("\\>", "")
        res = run_vim(row["buffer_before"], f"{kind}/^\\%({stripped}\\)$/d",
                      mode="ex")
        out["word_blind"] = bool(
            res["was_valid"] and norm(res["buffer_after"]) == norm(row["buffer_after"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    default = [str(REPO / "data" / "splits" / f"{s}.jsonl")
               for s in ("train", "val", "test")]
    ap.add_argument("--cases", nargs="+", default=default)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = []
    for path in args.cases:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                r["_src"] = Path(path).stem
                rows.append(r)
    print(f"probing {len(rows)} cases from {len(args.cases)} file(s)...",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(probe, rows))
    for r, row in zip(results, rows):
        r["src"] = row["_src"]

    for src in sorted({r["src"] for r in results}):
        rs = [r for r in results if r["src"] == src]
        an = [r for r in rs if "anchor_blind" in r]
        wn = [r for r in rs if "word_blind" in r]
        ab = sum(r["anchor_blind"] for r in an)
        wb = sum(r["word_blind"] for r in wn)
        print(f"{src}: anchored {len(an)}, blind {ab} "
              f"({100 * ab / max(1, len(an)):.0f}%) | "
              f"word-boundary {len(wn)}, blind {wb} "
              f"({100 * wb / max(1, len(wn)):.0f}%) | "
              f"unparsed {sum(1 for r in rs if r.get('unparsed'))}")
    if args.out:
        json.dump(results, open(args.out, "w"))
        print("written", args.out)


if __name__ == "__main__":
    main()
