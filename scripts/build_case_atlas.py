#!/usr/bin/env python3
"""Build the case-sensitivity atlas: every row in v3 + v4 where case matters.

Three axes, computed by the same classifier the review pages use
(build_test_review.case_profile), so this page and those agree by
construction:

  TRANSFORM  the command changes letter case -- gU/gu/g~, \\U/\\L/\\u/\\l,
             or an explicit case flag (\\c, \\C, sort i, s///i).
  MATCH      the command MATCHES case-sensitively and it is load-bearing:
             re-reading the pattern case-insensitively produces a different
             buffer, which the page shows as the diverging command.
  WORDING    the instruction talks about case ("capitalize", "uppercase",
             "a capital A").

A row can carry several. The one that matters for data quality is MATCH
without WORDING -- the gold depends on case and the sentence never says so,
the defect that removed 49 KB13 (the Kushman and Barzilay 2013 corpus) rows from the test slice on 2026-09-03
(DECISIONS.md). Those rows are flagged and sorted first.

Usage:
    python scripts/build_case_atlas.py --out page.html
"""
import argparse
import collections
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import build, collect      # noqa: E402
from page_guard import check_page                 # noqa: E402

META = {
    "title": "Case Sensitivity Atlas",
    "lab": "vimBot &middot; v3 + v4 &middot; every row where letter case matters",
    "h1": "Where does case change the answer?",
    "total_label": "rows where case matters",
    "flag_label": "repaired 2026-09-05",
    "flag_button": "repaired 2026-09-05",
    "key": "vimbot-case-atlas",
}

MATCH_SIGNAL = "the pattern only matches ONE case"


def axes(case):
    """Which of the three axes this row carries."""
    out = []
    sig = case["case_signals"]
    if any(not s.startswith(("wording:", "match is", MATCH_SIGNAL)) for s in sig):
        out.append("transform")
    if any(s.startswith("match is") or s.startswith(MATCH_SIGNAL) for s in sig):
        out.append("match")
    if any(s.startswith("wording:") for s in sig):
        out.append("wording")
    return out


AUDIT = REPO / "data" / "fixes" / "case_clause_2026-09-05_audit.json"


def _reverified():
    path = REPO / "data" / "fixes" / "reverify_2026-09-05.json"
    if not path.exists():
        return set()
    import json
    rev = json.loads(path.read_text(encoding="utf-8")).get("review", {})
    return {i for i, v in rev.items() if v.get("verdict") == "ok"}


def repaired():
    """id -> the wording the row had before the case clause was added."""
    if not AUDIT.exists():
        return {}
    import json
    rows = json.loads(AUDIT.read_text(encoding="utf-8"))["rows"]
    return {r["id"]: r["before"] for r in rows}


def lede(cases, by_ds, counts, total, before):
    silent = [c for c in cases if c["case_risk"] == "unstated"]
    fixed = [c for c in cases if c["id"] in before]
    in_test = [c for c in fixed if c["hist"].split(" \u00b7 ")[1] == "test"]
    endings = set()
    for c in fixed:
        stem = before[c["id"]].rstrip(".!?")
        endings.add(c["ins"][len(stem):] if c["ins"].startswith(stem) else "(inline)")
    return """\
  <p class="lede">
    Every row in <b>v3 and v4</b> whose answer depends on letter case &mdash;
    <b>%d of %d cases</b>, %d from v3 and %d from v4. <b>%d of them were
    repaired on 2026-09-05</b>, and those are what this page is now for: each
    shows the wording it had under <span class="mono">was</span>, sorted first,
    and the priority filter isolates them. <b>%d rows are left where the answer
    depends on case and the instruction never says so</b> &mdash; the defect is
    closed.
  </p>
  <p class="lede" style="margin-top:10px">
    The clause varies on purpose. One stock suffix repeated %d times would be
    its own defect: &sect;7.2 asks for paraphrase diversity, and a corpus where
    every case-sensitive row ends the same way teaches the ending rather than
    the meaning. These carry <b>%d distinct phrasings</b> across three
    placements &mdash; after the literal, attributive in front of it
    (&ldquo;the lowercase letters 'ing'&rdquo;), or a closing sentence &mdash;
    and a range of register, from &ldquo;Uppercase only.&rdquo; to &ldquo;The
    match is case-sensitive.&rdquo; A trailing clause is used only where the
    literal ends the sentence, because otherwise it strands what follows.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>%d of the repaired rows are in the test slice</b> (%s) &mdash; approved
    on 2026-09-03, edited here, and %s. Every before and after is in
    <span class="mono">data/fixes/case_clause_2026-09-05_audit.json</span>, so
    the pass is reversible. The three axes still describe the page:
    <b>transform</b> (%d) the command changes case, <b>match</b> (%d) the
    pattern matches case-sensitively and it is load-bearing, <b>wording</b>
    (%d) the instruction talks about case.
  </p>
""" % (len(cases), total, by_ds.get("v3", 0), by_ds.get("v4", 0), len(fixed),
       len(silent), len(fixed), len(endings), len(in_test),
       ", ".join('<span class="mono">%s</span>' % c["id"] for c in in_test)
       or "none",
       ("re-read and approved again on 2026-09-05" if in_test and all(
           c["id"] in _reverified() for c in in_test)
        else "the ones to re-read first"),
       counts["transform"], counts["match"], counts["wording"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    split_of, dataset_of = {}, {}
    for name in ("train", "val", "test"):
        import json
        for line in open(REPO / "data" / "splits" / ("%s.jsonl" % name), encoding="utf-8"):
            if line.strip():
                split_of[json.loads(line)["id"]] = name
    import json
    for line in open(args.cases, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            dataset_of[r["id"]] = r.get("dataset", "?")

    before = repaired()
    everything = collect(
        args.cases,
        flag=lambda r, risk: r["id"] in before or risk == "unstated")
    cases = [c for c in everything if c["case_signals"]]
    for c in cases:
        c["was"] = before.get(c["id"])
        c["hist"] = "%s · %s · %s" % (
            dataset_of.get(c["id"], "?"),
            split_of.get(c["id"], "not in a split"),
            " + ".join(axes(c)) or "case wording only")
    order = {"unstated": 0, "ambiguous": 1, "relevant": 2, "": 3}
    cases.sort(key=lambda c: (0 if c["id"] in before else 1,
                              order.get(c["case_risk"], 3), c["cat"], c["id"]))

    counts = collections.Counter(a for c in cases for a in axes(c))
    by_ds = collections.Counter(dataset_of.get(c["id"], "?") for c in cases)
    page = build(cases, {**META, "lede": lede(cases, by_ds, counts,
                                                 len(everything), before)})
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="case atlas")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("case rows: %d of %d   (%s)   flagged unstated: %d   page: %d KB  -> %s"
          % (len(cases), len(everything),
             " · ".join("%s %d" % kv for kv in sorted(by_ds.items())),
             sum(1 for c in cases if c["flagged"]), len(page) // 1024, args.out))
    print("axes: " + " · ".join("%s %d" % (k, counts[k])
                                for k in ("transform", "match", "wording")))


if __name__ == "__main__":
    main()
