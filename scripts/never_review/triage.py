#!/usr/bin/env python3
"""Print never-solved train rows compactly for a reader, a batch at a time,
in the review page's order (most adapters failing first).

Each row: the instruction, the input buffer, the gold edit as a line diff,
every probe's top answers, and where the most common runnable wrong answer's
buffer differs from the gold output -- flagged when the difference is only
whitespace or only letter case, the two shapes an unstated-form or ambiguous
row usually takes.

    python scripts/never_review/triage.py --start 0 --count 20
    python scripts/never_review/triage.py --ids hc-1234,kb13-0001a
"""
import argparse
import difflib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_never_review import collect          # noqa: E402

SHORT = {"valid_wrong": "wrong", "invalid": "ERR", "match": "OK", "parse": "unparsed",
         "truncated": "capped"}


def lines(text):
    ls = (text or "").split("\n")
    if ls and ls[-1] == "":
        ls.pop()
    return ls


def clip(s, n=110):
    return s if len(s) <= n else s[:n] + "…"


def show_buffer(text, limit=14):
    ls = lines(text)
    out = ["%2d|%s" % (i + 1, clip(l)) for i, l in enumerate(ls[:limit])]
    if len(ls) > limit:
        out.append("   … %d more lines" % (len(ls) - limit))
    return "\n".join("    " + l for l in out) if out else "    (empty)"


def diff(a, b, limit=10):
    """The changed lines of a -> b, numbered by their position in a / b."""
    la, lb = lines(a), lines(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=la, b=lb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for i in range(i1, i2):
            out.append("-%d|%s" % (i + 1, clip(la[i])))
        for j in range(j1, j2):
            out.append("+%d|%s" % (j + 1, clip(lb[j])))
    if len(out) > limit:
        out = out[:limit] + ["… %d more changed lines" % (len(out) - limit)]
    return "\n".join("    " + l for l in out) if out else "    (no change)"


def shape(gold, alt):
    g, a = lines(gold), lines(alt)
    if [l.rstrip() for l in g] == [l.rstrip() for l in a]:
        return "trailing whitespace only"
    if [" ".join(l.split()) for l in g] == [" ".join(l.split()) for l in a]:
        return "whitespace only"
    if [l.lower() for l in g] == [l.lower() for l in a]:
        return "letter case only"
    return None


def show(c):
    h = []
    probes = " | ".join("%s %s" % (p["key"], "-" if p["k"] is None else "%d/8" % p["k"]) for p in c["probes"])
    h.append("### [%d/4] %s  %s %s  gold %s: %s  cursor %d,%d  %s%s"
             % (c["never"], c["id"], c["cat"], c["tier"], c["mode"], c["cmd"], c["cur"][0], c["cur"][1],
                c["dataset"] or "", ("  <" + c["hist"] + ">") if c["hist"] else ""))
    h.append("  INS: " + c["ins"])
    h.append("  BEFORE:\n" + show_buffer(c["bef"]))
    h.append("  GOLD EDIT:\n" + diff(c["bef"], c["exp"]))
    h.append("  PROBES: " + probes)
    for p in c["probes"]:
        if p["top"]:
            h.append("    %-9s %s%s" % (p["key"], "  ".join(
                ("`%s: %s`" % (a["mode"], clip(a["cmd"], 70)) if a["cmd"] is not None else "(no command)")
                + " %d %s" % (a["n"], SHORT.get(a["state"], a["state"]))
                for a in p["top"][:3]), "  (probed before edit)" if p["stale"] else ""))
    alt = c["alt"]
    if alt:
        sh = shape(c["exp"], alt["buf"])
        h.append("  ALT `%s: %s` (%s)%s:\n%s" % (
            alt["mode"], clip(alt["cmd"], 80), ", ".join("%s %d" % kv for kv in alt["by"].items()),
            ("  [" + sh + "]") if sh else "", diff(c["exp"], alt["buf"], 8)))
    else:
        h.append("  ALT: no runnable wrong answer")
    return "\n".join(h)


CACHE = HERE.parents[1] / "runs" / "never_review_2026-09-25" / "cards.json"


def load_cards(refresh=False):
    """The collected rows, cached: collecting re-executes ~1,500 answers."""
    import json
    if CACHE.exists() and not refresh:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    cards, _cats = collect()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cards, ensure_ascii=False), encoding="utf-8")
    return cards


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--ids", default=None)
    ap.add_argument("--refresh", action="store_true", help="re-collect instead of using the cache")
    args = ap.parse_args()
    cards = load_cards(args.refresh)
    if args.ids:
        want = set(args.ids.split(","))
        chosen = [c for c in cards if c["id"] in want]
    else:
        chosen = cards[args.start:args.start + args.count]
    for c in chosen:
        print(show(c))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
