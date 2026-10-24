#!/usr/bin/env python3
"""Build the hand-review page for rows restored from the drop file.

On 2026-09-04 the owner restored every row dropped in the test-slice
verification (51: 49 KB13 (the Kushman and Barzilay 2013 corpus), 2 hand-authored) with scripts/restore_dropped.py,
applying the case rule to the rows the classifier marks case-silent and
leaving the rest unchanged for manual rewording. Those rows are back in the
test slice but are no longer verified: the owner reviews them here. Same
machinery as the test-slice page; each card shows the wording the row had
before restoration and, where the rule did not apply, a note.

Verdicts: ok = keep as restored; drop = drop again; reword/edit = change the
instruction. The export is a change-set apply_curation.py runs directly.

Usage:
    python scripts/build_restored_review.py --out page.html
"""
import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import build, collect      # noqa: E402
from page_guard import check_page                 # noqa: E402

META = {
    "title": "Restored Rows Review",
    "lab": "vimBot &middot; restored from the drop file &middot; hand review",
    "h1": "Does the repaired instruction describe this edit?",
    "total_label": "restored rows",
    "flag_label": "need manual rewording",
    "flag_button": "manual rewording",
    "key": "vimbot-restored",
}


def lede(cases):
    n = len(cases)
    rules = collections.Counter(c.get("rule") for c in cases)
    manual = sum(1 for c in cases if c["flagged"])
    return """\
  <p class="lede">
    <b>%d rows restored</b> on 2026-09-04 from
    <span class="mono">data/dropped_cases.jsonl</span>, all of them dropped in
    the test-slice verification the day before. They are back in the test
    slice but no longer verified &mdash; that is what this page is for.
    <b>%d</b> got the owner&rsquo;s case rule (every quoted literal lowercase
    &rarr; &ldquo;in lowercase&rdquo;, several &rarr; &ldquo;all in
    lowercase&rdquo;); <b>%d</b> were not case-silent by the classifier and
    came back unchanged with a note: they need a rewording by hand.
    Each card shows the wording the row had before, under <i>was</i>.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>ok</b> keeps the row as restored, <b>drop</b> drops it again,
    <b>reword</b> plus <b>Edit</b> changes the instruction. The export is a
    change-set <span class="mono">apply_curation.py</span> runs directly;
    after applying, the corpus is reassembled and the split files refreshed
    by id. The audit of the restoration is
    <span class="mono">data/fixes/restore_dropped_2026-09-04.json</span>.
  </p>
""" % (n, n - manual, manual)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = collect(
        args.cases,
        select=lambda r: bool(r.get("restored_on")),
        flag=lambda r, risk: r.get("restore_edit", "none") == "none",
        sort_key=lambda c: (not c["flagged"], c["id"]),
    )
    if not cases:
        sys.exit("no restored rows in %s" % args.cases)
    page = build(cases, {**META, "lede": lede(cases)})
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="restored review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("restored rows: %d (%d manual)   vim disagreements: %d   page: %d KB  -> %s"
          % (len(cases), sum(1 for c in cases if c["flagged"]),
             sum(1 for c in cases if not c["agrees"]), len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
