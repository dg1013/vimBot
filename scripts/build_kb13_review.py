#!/usr/bin/env python3
"""Build the hand-review page for the KB13 (the Kushman and Barzilay 2013 corpus) rows outside the test slice.

The 2026-09-03 test-slice verification dropped 49 of the 62 KB13 rows it
held, 47 of them with one signature: the gold matches case-sensitively, the
buffer deliberately holds the other case, and the instruction never says
case matters. The owner asked to review the remaining KB13 rows by hand
(DECISIONS.md 2026-09-04). This page is the test-slice review page over that
row set -- same re-execution, same case and language profiling, same
verdict/edit export in apply_curation.py's change-set shape -- with the
mechanical signature as the priority flag and sorted first.

Rows: every `kb13-` id in the combined corpus that is NOT in the test split
(those 13 were already reviewed and kept).

Usage:
    python scripts/build_kb13_review.py --out page.html
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import build, collect      # noqa: E402
from page_guard import check_page                 # noqa: E402

LEDE = """\
  <p class="lede">
    These are the <b>%d KB13 rows outside the test slice</b> (train and val).
    KB13 is the human-authored regex corpus (Kushman &amp; Barzilay 2013):
    each regex became a line-filtering edit &mdash; <span class="mono">g//d</span>
    deletes matching lines, <span class="mono">v//d</span> keeps them &mdash; over
    a witness buffer, and the annotator's description became the instruction.
    Every row passes the machine gate; the question is the same as on the
    test slice: <i>does the instruction describe this edit?</i>
  </p>
  <p class="lede" style="margin-top:10px">
    The test-slice review dropped 49 of 62 KB13 rows, 47 of them with one
    signature: the gold matches <b>case-sensitively</b>, the buffer holds the
    other case on purpose, and the instruction never says so. <b>%d rows here
    carry that signature</b>; they are flagged, sorted first, and the reason is
    spelled out on each card. Rows that name the case (&ldquo;a capital
    'A'&rdquo;) are not flagged. Verdicts and edits export as a change-set
    <span class="mono">apply_curation.py</span> runs directly; after applying,
    the split files are refreshed by id (DECISIONS.md 2026-09-04).
  </p>
"""

META = {
    "title": "KB13 Review",
    "lab": "vimBot &middot; KB13 corpus &middot; hand review of train + val rows",
    "h1": "Does the instruction describe this edit?",
    "total_label": "KB13 rows outside the test slice",
    "flag_label": "case-silent, case-sensitive",
    "flag_button": "case signature",
    "key": "vimbot-kb13",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--exclude", default=str(REPO / "data" / "splits" / "test.jsonl"),
                    help="JSONL whose ids are left out (default: the test split)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.exclude, encoding="utf-8") as fh:
        done = {json.loads(l)["id"] for l in fh if l.strip()}
    cases = collect(
        args.cases,
        select=lambda r: r["id"].startswith("kb13-") and r["id"] not in done,
        flag=lambda r, risk: risk == "unstated",
        sort_key=lambda c: (not c["flagged"], c["tier"], c["id"]),
    )
    if not cases:
        sys.exit("no KB13 rows outside %s in %s" % (args.exclude, args.cases))
    n, flagged = len(cases), sum(1 for c in cases if c["flagged"])
    page = build(cases, {**META, "lede": LEDE % (n, flagged)})
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="kb13 review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    bad = sum(1 for c in cases if not c["agrees"])
    print("kb13 rows: %d (%d flagged by the case signature)   vim disagreements: %d   "
          "page: %d KB  -> %s" % (n, flagged, bad, len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
