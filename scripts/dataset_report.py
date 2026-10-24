#!/usr/bin/env python3
"""What the corpus holds, per vim category and per VimGolf theme.

The two vocabularies are separate axes and are reported separately: the 15
taxonomy categories from data/taxonomy.yaml (with their quotas), and the 20
VimGolf theme concepts from data/vimgolf-themes.md. A case has exactly one
category and at most one theme, so the two tables count the same rows twice
by design.

    python scripts/dataset_report.py                  # both tables
    python scripts/dataset_report.py --group substitute   # drill into one
    python scripts/dataset_report.py --group T12 --samples 8
    python scripts/dataset_report.py --datasets v4    # one dataset only
"""
import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_corpus import DATASETS  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (present in the vimBot env)")

THEMES = [
    ("T01", "Case transformations"), ("T02", "Data-format conversion"),
    ("T03", "Code-to-code translation"), ("T04", "Micro-refactors"),
    ("T05", "Alignment & columns"), ("T06", "Sorting & dedup"),
    ("T07", "Numbering & counting"), ("T08", "Sequence & table generation"),
    ("T09", "Line reordering & transposition"), ("T10", "Wrapping, joining & splitting"),
    ("T11", "Quoting & bracket surgery"), ("T12", "Structured-config editing"),
    ("T13", "Whitespace & indentation repair"), ("T14", "Extraction & filtering"),
    ("T15", "Prose & typo repair"), ("T16", "Duplication-with-variation"),
    ("T17", "Regex-craft"), ("T18", "ASCII art & grids"),
    ("T19", "Encodings & ciphers"), ("T20", "Merge/diff repair"),
]


def load(datasets):
    rows = []
    for ds in datasets:
        p = REPO / "data" / ds / ("%s_cases.jsonl" % ds)
        if not p.exists():
            sys.exit("%s not built — run scripts/build_corpus.py --dataset %s" % (p, ds))
        rows += [dict(json.loads(l), _ds=ds)
                 for l in p.open(encoding="utf-8") if l.strip()]
    return rows


def bar(have, quota, width=18):
    if not quota:
        return ""
    filled = min(width, int(round(width * have / quota)))
    return "#" * filled + "." * (width - filled)


def table(title, keys, rows, keyfn, quotas, datasets, label=None):
    print("\n%s" % title)
    print("%-18s %-24s %6s %6s %6s   %5s %5s %5s %5s   %6s %6s  %s"
          % ("", "", *[d for d in datasets] + [""] * (3 - len(datasets)),
             "T1", "T2", "T3", "—", "total", "quota", "fill"))
    for k in keys:
        sub = [r for r in rows if keyfn(r) == k]
        per_ds = [sum(1 for r in sub if r["_ds"] == d) for d in datasets]
        tiers = collections.Counter(r.get("tier") or "—" for r in sub)
        q = quotas.get(k)
        name = (label or {}).get(k, "")
        cells = "".join("%6d" % v for v in per_ds) + "      " * (3 - len(per_ds))
        print("%-18s %-24s %s   %5d %5d %5d %5d   %6d %6s  %s"
              % (k, name[:24], cells,
                 tiers.get("T1", 0), tiers.get("T2", 0), tiers.get("T3", 0),
                 tiers.get("—", 0), len(sub), q or "—", bar(len(sub), q)))


def drill(group, rows, samples):
    is_theme = group.upper().startswith("T") and group[1:].isdigit()
    key = "theme" if is_theme else "category"
    want = group.upper() if is_theme else group
    sub = [r for r in rows if (r.get(key) or "") == want]
    if not sub:
        sys.exit("no rows for %s %r" % (key, group))
    print("\n=== %s %s — %d cases ===" % (key, want, len(sub)))
    for field in ("_ds", "mode", "tier", "source", "instruction_provenance"):
        c = collections.Counter(r.get(field) or "—" for r in sub)
        print("  %-22s %s" % (field.lstrip("_"),
                              " · ".join("%s %d" % kv for kv in c.most_common(6))))
    other = collections.Counter(
        (r.get("theme") if not is_theme else r.get("category")) or "—" for r in sub)
    print("  %-22s %s" % ("theme" if not is_theme else "category",
                          " · ".join("%s %d" % kv for kv in other.most_common(8))))
    cmds = collections.Counter(r["gold_command"] for r in sub)
    print("  %-22s %d distinct, most used: %s"
          % ("commands", len(cmds),
             " · ".join("%r x%d" % kv for kv in cmds.most_common(4))))
    print("\n  sample cases:")
    step = max(1, len(sub) // samples)
    for r in sub[::step][:samples]:
        print("    %-10s %-6s %-32r %s"
              % (r["id"], r.get("mode"), r["gold_command"][:32], r["instruction"][:58]))
        print("               %r" % (r["buffer_before"][:74],))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", default=",".join(sorted(DATASETS)))
    ap.add_argument("--group", help="one category or theme to drill into")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rows = load(datasets)
    tax = yaml.safe_load((REPO / "data/taxonomy.yaml").read_text(encoding="utf-8"))
    quotas = {c: sum(v["quotas"].values()) for c, v in tax["categories"].items()}

    if args.group:
        drill(args.group, rows, args.samples)
        return 0

    print("corpus: %d cases across %s" % (len(rows), ", ".join(datasets)))
    cats = sorted(set(quotas) | {r.get("category") for r in rows if r.get("category")},
                  key=lambda c: -sum(1 for r in rows if r.get("category") == c))
    table("VIM TAXONOMY CATEGORIES  (quota from data/taxonomy.yaml)",
          cats, rows, lambda r: r.get("category"), quotas, datasets)
    table("VIMGOLF THEMES  (concepts from data/vimgolf-themes.md; untagged rows excluded)",
          [t for t, _ in THEMES], rows, lambda r: r.get("theme"), {}, datasets,
          label=dict(THEMES))
    untagged = sum(1 for r in rows if not r.get("theme"))
    print("\n%d rows carry no theme tag (mostly the regex-derived v3 slice)." % untagged)
    print("Drill into one group:  python scripts/dataset_report.py --group substitute")
    return 0


if __name__ == "__main__":
    sys.exit(main())
