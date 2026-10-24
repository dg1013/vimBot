#!/usr/bin/env python3
"""Generate data/external/ATTRIBUTION.md from the dataset itself.

Every permissive licence we ingest carries a notice obligation, and Apache-2.0
carries one more: you must state that you changed the files. Reshaping a test
case into a dataset row IS a modification, so that duty applies to us.

This file is GENERATED rather than hand-maintained so it cannot drift from the
data. It reads the source_* provenance fields off the shipped case rows, groups
them, and emits the notices those rows actually require. Mine a new Apache-2.0
source and its section appears here on the next run.

Stdlib only. Usage:
    python scripts/build_attribution.py [--check]

--check exits 1 if the file on disk is stale, for CI (continuous integration).
"""

import argparse
import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
V3 = ROOT / "data/v3"
OUT = ROOT / "data/external/ATTRIBUTION.md"

CASE_FILES = ["kb13_cases.jsonl", "sregex_cases.jsonl", "emulator_cases.jsonl",
              "candidate_cases.jsonl", "handcrafted_cases.jsonl"]

# Copyright holders and vendored licence paths, keyed by the corpus name the
# rows carry. Adding a source means adding one entry here plus mining it.
REGISTRY = {
    "VSCodeVim": {
        "holder": "Copyright (c) 2015 VSCodeVim",
        "license": "MIT",
        "license_file": "data/external/emulator-tests/LICENSE.VSCodeVim",
        "upstream": "github.com/VSCodeVim/Vim",
    },
    "IdeaVim": {
        "holder": "Copyright (c) 2003-present The IdeaVim authors",
        "license": "MIT",
        "license_file": "data/external/emulator-tests/LICENSE.IdeaVim",
        "upstream": "github.com/JetBrains/ideavim",
        "carve_out": ("MIT except three files the project names under the Vim License "
                      "(ScrollViewHelper.kt, Tutor.kt, CodeWrapper.kt). All three are "
                      "implementation files; none is among the test files mined here."),
    },
    "KB13": {
        "holder": "Copyright (c) 2016 Nick Locascio",
        "license": "MIT",
        "license_file": "data/external/deep-regex/LICENSE",
        "upstream": "github.com/nicholaslocascio/deep-regex",
        "pinned_commit": "096490d",
        "cite": ("Locascio, Narasimhan, DeLeon, Kushman & Barzilay, EMNLP 2016. "
                 "KB13 sub-corpus originally Kushman & Barzilay, NAACL 2013."),
    },
    "StructuredRegex": {
        "holder": "Copyright (c) xiye17 and contributors",
        "license": "MIT",
        "license_file": None,
        "upstream": "github.com/xiye17/StructuredRegex",
        "preview": ("Not vendored. The 447 rows derive from 460 patterns recovered from a "
                    "prior artifact, not from a fetched checkout — a real import must vendor "
                    "the corpus under §14c and re-run."),
        "cite": "Ye, Chen, Dillig & Durrett, ACL 2020.",
    },
    "CodeMirror": {
        "holder": "Copyright (C) 2017 by Marijn Haverbeke and others",
        "license": "MIT",
        "license_file": "data/external/emulator-tests/LICENSE.CodeMirror",
        "upstream": "github.com/replit/codemirror-vim",
    },
    "toybox vi": {
        "holder": "Copyright (C) 2006, 2019 by Rob Landley",
        "license": "0BSD",
        "license_file": "data/external/emulator-tests/LICENSE.toybox",
        "upstream": "github.com/landley/toybox",
        "carve_out": ("0BSD removes even the attribution requirement; provenance is "
                      "recorded here as project discipline, not legal duty."),
    },
    "VsVim": {
        "holder": "Copyright 2012 Jared Parsons",
        "license": "Apache-2.0",
        "license_file": "data/external/emulator-tests/LICENSE.Apache-2.0",
        "notice_file": "data/external/emulator-tests/NOTICE.VsVim",
        "upstream": "github.com/VsVim/VsVim",
        "license_note": ("The project's own License.txt is the short-form Apache header, which "
                         "points at the licence rather than containing it. Apache-2.0 §4(a) "
                         "requires giving recipients a copy of the License, so the full text is "
                         "vendored separately and the project's notice kept alongside it."),
    },
}

# What we did to the material, stated once per licence family. Apache-2.0
# requires this; MIT (Massachusetts Institute of Technology) does not, but stating it for every source costs nothing
# and makes the provenance legible.
CHANGE_STATEMENT = (
    "The upstream files were not redistributed. Each project's **test cases were "
    "read and reshaped** into dataset rows: the starting buffer, the keystroke or "
    "ex command, and the resulting buffer were extracted, the cursor position was "
    "recovered from the source's own marker, and every triple was then **re-executed "
    "against real vim 8.2** through `src/executor.py`. Rows whose upstream expectation "
    "disagreed with vim were discarded rather than imported. Natural-language "
    "instructions are authored by this project and are not upstream content."
)


def norm(v):
    """Provenance fields are sometimes a one-element list; flatten to a string."""
    if isinstance(v, list):
        return v[0] if len(v) == 1 else "/".join(str(x) for x in v)
    return v


def collect():
    groups = collections.OrderedDict()
    gaps = collections.Counter()
    for name in CASE_FILES:
        path = V3 / name
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            corpus = norm(r.get("source_corpus")) or norm(r.get("source")) or "unknown"
            lic = norm(r.get("source_license")) or "unstated"
            commit = norm(r.get("source_commit"))
            key = (corpus, lic)
            g = groups.setdefault(key, {"rows": 0, "commits": set(), "files": set()})
            g["rows"] += 1
            if commit:
                g["commits"].add(commit)
            else:
                gaps[corpus] += 1
            g["files"].add(name)
    return groups, gaps


def render(groups, gaps):
    by_lic = collections.defaultdict(list)
    for (corpus, lic), g in groups.items():
        by_lic[lic].append((corpus, g))

    L = []
    L.append("# Attribution and licence notices")
    L.append("")
    L.append("<!-- GENERATED by scripts/build_attribution.py — do not edit by hand. -->")
    L.append("")
    L.append("Every source ingested into `data/v3/` is permissively licensed. This file "
             "discharges the notice obligations those licences carry, and is regenerated "
             "from the `source_*` fields on the case rows themselves so it cannot drift "
             "from the data.")
    L.append("")
    total = sum(g["rows"] for g in groups.values())
    L.append(f"**{total:,} rows** across **{len(groups)} sources**.")
    L.append("")
    L.append("## Licences we accept")
    L.append("")
    L.append("Posture A, permissive-only (DECISIONS.md 2026-08-17): MIT, BSD-2/3, ISC, "
             "Apache-2.0, 0BSD, Unlicense and public domain. Copyleft is excluded because "
             "§16 plans a public release and the mix cannot be un-made. Two of these carry "
             "obligations we actually have to do something about:")
    L.append("")
    L.append("| Licence | What we must do | Where that lands |")
    L.append("|---|---|---|")
    L.append("| **MIT** | Keep the copyright line and permission notice with the work. | "
             "Per-source notice below, licence text vendored under `data/external/`, and "
             "`source_license` on every row. |")
    L.append("| **Apache-2.0** | Everything MIT requires, **plus** give recipients a copy of "
             "the License, preserve any NOTICE file, and **state that the files were "
             "changed**. | Full licence text vendored (a project's own short-form header is "
             "not a copy of the License), upstream NOTICE kept beside it, and the "
             "*What was changed* statement below. |")
    L.append("")
    L.append("Apache-2.0 sections appear here automatically once an Apache source is mined; "
             "the licence text and notice are vendored ahead of that so the obligation is "
             "already discharged when the rows land.")
    L.append("")

    order = ["Apache-2.0", "MIT"] + [k for k in by_lic if k not in ("Apache-2.0", "MIT")]
    for lic in order:
        if lic not in by_lic:
            continue
        L.append(f"## {lic}")
        L.append("")
        if lic == "Apache-2.0":
            L.append("Apache-2.0 requires the licence, the notices, any NOTICE file, **and a "
                     "statement that the files were changed**. The statement is below and "
                     "applies to every Apache-2.0 source listed here.")
        elif lic == "MIT":
            L.append("MIT requires the copyright line and the permission notice to travel "
                     "with copies and substantial portions. The change statement is not "
                     "required here; it is stated anyway for legibility.")
        L.append("")
        for corpus, g in sorted(by_lic[lic]):
            meta = REGISTRY.get(corpus, {})
            L.append(f"### {corpus} — {g['rows']:,} rows")
            L.append("")
            if meta.get("holder"):
                L.append(f"> {meta['holder']}")
                L.append("")
            bits = []
            if meta.get("upstream"):
                bits.append(f"Upstream: `{meta['upstream']}`")
            if g["commits"]:
                bits.append("Commit: " + ", ".join(f"`{c}`" for c in sorted(g["commits"])))
            elif meta.get("pinned_commit"):
                bits.append(f"Commit: `{meta['pinned_commit']}` "
                            "(from the corpus register; the rows do not yet repeat it)")
            else:
                bits.append("Commit: **not pinned** — see Gaps below")
            if meta.get("license_file"):
                lf = ROOT / meta["license_file"]
                mark = "" if lf.exists() else "  ← **missing on disk**"
                bits.append(f"Licence text: `{meta['license_file']}`{mark}")
            else:
                bits.append("Licence text: **not vendored** — see Gaps below")
            L.append("- " + "\n- ".join(bits))
            if meta.get("notice_file"):
                bits.append(f"Upstream notice: `{meta['notice_file']}`")
            L[-1] = "- " + "\n- ".join(bits)
            if meta.get("license_note"):
                L.append(f"- Note: {meta['license_note']}")
            if meta.get("cite"):
                L.append(f"- Citation: {meta['cite']}")
            if meta.get("carve_out"):
                L.append(f"- Carve-out: {meta['carve_out']}")
            if meta.get("preview"):
                L.append(f"- Status: {meta['preview']}")
            L.append("")

    L.append("## What was changed")
    L.append("")
    L.append(CHANGE_STATEMENT)
    L.append("")

    L.append("## Gaps")
    L.append("")
    if gaps or any(
        not REGISTRY.get(c, {}).get("license_file")
        or not (ROOT / REGISTRY[c]["license_file"]).exists()
        for (c, _l) in groups if c in REGISTRY
    ):
        L.append("Recorded rather than hidden — §14c wants provenance pinned, and these are not yet:")
        L.append("")
        for corpus, n in gaps.most_common():
            pin = REGISTRY.get(corpus, {}).get("pinned_commit")
            if pin:
                L.append(f"- **{corpus}** — {n:,} rows carry no `source_commit`. The upstream "
                         f"revision is known (`{pin}`) and stated above; the maker script should "
                         f"write it onto the rows so the two cannot drift.")
            else:
                L.append(f"- **{corpus}** — {n:,} rows carry no `source_commit`, and no upstream "
                         f"revision is pinned anywhere yet.")
        for (corpus, _lic), _g in groups.items():
            meta = REGISTRY.get(corpus, {})
            lf = meta.get("license_file")
            if not lf:
                L.append(f"- **{corpus}** — licence text not vendored into `data/external/`.")
            elif not (ROOT / lf).exists():
                L.append(f"- **{corpus}** — `{lf}` is referenced but missing on disk.")
    else:
        L.append("None. Every source is pinned to a commit with its licence text vendored.")
    L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if ATTRIBUTION.md is stale")
    args = ap.parse_args()

    groups, gaps = collect()
    text = render(groups, gaps)

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print("ATTRIBUTION.md is stale — run: python scripts/build_attribution.py")
            return 1
        print("ATTRIBUTION.md is current")
        return 0

    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}")
    for (corpus, lic), g in groups.items():
        pin = sorted(g["commits"])[0] if g["commits"] else "UNPINNED"
        print(f"  {corpus:<18} {lic:<12} {g['rows']:>5} rows   {pin}")
    if gaps:
        print("gaps:")
        for corpus, n in gaps.most_common():
            print(f"  {corpus:<18} {n:>5} rows without source_commit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
