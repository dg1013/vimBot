#!/usr/bin/env python3
"""Build the review page for a day's newly authored cases.

The 2026-09-06 batch filled the taxonomy: 2,374 rows across all seventeen
categories, taking every one of the 51 category x tier cells to quota
(DECISIONS.md). Every one of those rows was executed against real vim before
it was ingested, so what needs a human eye is not whether the command works
but whether the INSTRUCTION describes the edit — the same question the
test-slice page asks, over rows that are new rather than inherited.

Same machinery as build_test_review, so verdicts and edits export as a
change-set apply_curation.py runs directly. Cards are grouped by category and
tier, and the priority filter isolates the rows that landed in the TEST split,
where an unreviewed row is not just a weak training example but an unverified
measurement.

Usage:
    python scripts/build_new_cases_review.py --on 2026-09-06 --out page.html
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
    "title": "New Cases 2026-09-06",
    "lab": "vimBot &middot; authored 2026-09-06 &middot; the quota-fill batch",
    "h1": "Does the instruction describe this edit?",
    "total_label": "cases authored",
    "flag_label": "in the test split",
    "flag_button": "test rows",
    # the localStorage key is the page's memory: keep it, so verdicts the
    # owner already recorded on the visual rows survive the re-scoping
    "key": "vimbot-new-visual",
}


def test_ids():
    """Ids in the test split — the rows whose review is not optional."""
    import json
    p = REPO / "data" / "splits" / "test.jsonl"
    if not p.exists():
        return set()
    return {json.loads(l)["id"] for l in p.open(encoding="utf-8") if l.strip()}


def lede(cases, on, disagree=0):
    cells = collections.Counter((c["cat"], c["tier"]) for c in cases)
    per_cat = collections.Counter(c["cat"] for c in cases)
    cmds = len({c["cmd"] for c in cases})
    rowsline = " &middot; ".join(
        "%s <b>%d</b>" % (cat, n) for cat, n in sorted(per_cat.items()))
    return """\
  <p class="lede">
    The <b>%d cases authored on %s</b> to fill the taxonomy, which they did:
    all 51 category &times; tier cells are now at or over quota, for the first
    time. %s. <b>%d distinct commands</b> over %d freshly invented buffers
    &mdash; SQL, config stanzas, CSV exports, dataframe code, logs, diffs,
    checklists, ASCII grids &mdash; drawn from the work the model is for: data
    analysis, cleaning and standardisation, software development, and fast
    repetitive edits.
  </p>
  <p class="lede" style="margin-top:10px">
    Every one already passes the machine gate: it was executed against real vim
    before ingest, the ingest ran it again and agreed, and this page ran all
    %d a third time &mdash; %s. So the question here is the one no gate can
    answer: does the sentence describe the edit, completely and without a
    second faithful reading? Vim overruled the author&rsquo;s expectation many
    times while these were written (counted yank paste position, search wrap,
    <span class="mono">:g</span> swallowing a bar, <span class="mono">&gt;}</span>
    including the cursor line), which is why each card shows the input, the
    expected output and what vim produced side by side.
  </p>
  <p class="lede" style="margin-top:10px">
    Most went to <b>train</b>, but <b>%d landed in test</b> &mdash; &sect;7.3
    keeps a gold command inside one split, so a new row whose command already
    lives in test follows it there. Those carry no verdict and the test slice
    is a measurement, so they are the priority filter. A <b>drop</b> here
    removes the row; <b>reword</b> plus <b>Edit</b> changes the instruction.
    The export is a change-set
    <span class="mono">apply_curation.py</span> runs directly.
  </p>
""" % (len(cases), on, rowsline, cmds, len({c["bef"] for c in cases}),
       len(cases),
       ("it disagreed with %d stored result(s)" % disagree) if disagree
       else "it disagreed with none of them",
       sum(1 for c in cases if c["flagged"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--on", default="2026-09-06", help="authored_on date to show")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases = collect(
        args.cases,
        select=lambda r: r.get("authored_on") == args.on,
        flag=lambda r, risk, ids=test_ids(): r["id"] in ids,
        sort_key=lambda c: (c["cat"], c["tier"], c["id"]),
    )
    if not cases:
        sys.exit("no cases authored on %s in %s" % (args.on, args.cases))
    disagree = sum(1 for c in cases if not c["agrees"])
    page = build(cases, {**META, "lede": lede(cases, args.on, disagree)})
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"], name="new cases")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("cases: %d (%d in test)   vim disagreements: %d   page: %d KB -> %s"
          % (len(cases), sum(1 for c in cases if c["flagged"]),
             sum(1 for c in cases if not c["agrees"]), len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
