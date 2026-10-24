#!/usr/bin/env python3
"""Build the hand-review page for the rows queued for rewording (open item #9).

Four rows whose instruction is ambiguous or underdetermined against its own
gold, collected during the test-slice verification and its follow-up
(DECISIONS.md 2026-09-04). The owner asked to reword them by hand. This page
is the test-slice review page over exactly those rows: same re-execution,
same profiling, same verdict/edit export in apply_curation.py's change-set
shape. The introduction spells out, per row, the issue, the language of the
buffer, and three buffers side by side -- the input, what the gold produces,
and what the OTHER reading the current wording allows would produce. The
alternative is executed through the sandbox at build time, never typed in,
so the page shows what vim does rather than what someone expected.

Usage:
    python scripts/build_reword_review.py --out page.html
"""
import argparse
import html
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import build, collect      # noqa: E402
from executor import run_vim                      # noqa: E402
from page_guard import check_page                 # noqa: E402

# id -> the issue, the buffer's language, and the other reading as a command.
# The notes quote the test rows they are about, so they are kept with the
# dataset, in data/ (not published), and read when the page is built.
TABLES = REPO / "data" / "review_tables" / "reword_review.json"


def reword_rows():
    if not TABLES.is_file():
        sys.exit("%s is missing: the notes quote test rows, so they are kept in "
                 "data/, which is not published" % TABLES)
    return json.loads(TABLES.read_text(encoding="utf-8"))["ROWS"]

_PRE = ('style="margin:0;padding:7px 9px;background:var(--sunk);'
        'border:1px solid var(--line);border-radius:5px;font-family:var(--mono);'
        'font-size:11.5px;line-height:1.45;white-space:pre;overflow-x:auto;'
        'color:var(--ink2)"')
_COL = 'style="flex:1 1 200px;min-width:0;display:flex;flex-direction:column;gap:4px"'
_CAP = ('style="font-family:var(--mono);font-size:10.5px;letter-spacing:.04em;'
        'text-transform:uppercase;color:var(--muted)"')


def _pre(text):
    return "<pre %s>%s</pre>" % (_PRE, html.escape(text.rstrip("\n")) or "&nbsp;")


def buffers_block(case, spec):
    """Three buffers for one row: input, the gold's result, the alternative's."""
    label, mode, alt_cmd = spec["alt"]
    res = run_vim(case["bef"], alt_cmd, mode=mode, cursor=tuple(case["cur"]))
    if not res["was_valid"] or res.get("blocked"):
        sys.exit("alternative for %s did not run: %r" % (case["id"], res))
    alt_after = res["buffer_after"] or ""
    if alt_after.rstrip("\n") == (case["vim"] or "").rstrip("\n"):
        sys.exit("alternative for %s produces the gold's buffer; it is not "
                 "an alternative" % case["id"])
    cols = [
        ("input buffer", case["bef"]),
        ("gold: %s: %s" % (case["mode"], case["cmd"]), case["vim"]),
        ("%s: %s: %s" % (label, mode, alt_cmd), alt_after),
    ]
    return '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:6px">' + "".join(
        "<div %s><span %s>%s</span>%s</div>" % (_COL, _CAP, html.escape(cap), _pre(text))
        for cap, text in cols) + "</div>"


def lede(cases, rows):
    blocks = []
    for c in cases:
        spec = rows[c["id"]]
        blocks.append(
            '<div style="margin-top:14px">'
            '<span class="mono">%s</span> &mdash; %s<br>'
            '<span %s>language</span> %s%s</div>'
            % (c["id"], spec["issue"], _CAP, spec["lang"], buffers_block(c, spec)))
    return """\
  <p class="lede">
    <b>Four rows queued for rewording.</b> Each passes the machine gate; the
    problem is that the instruction admits a reading the gold did not take, or
    leaves out a name the gold depends on. For each row: the issue, the
    language the buffer is written in, and three buffers &mdash; the input,
    what the gold produces, and what the other reading would produce, run
    through the sandbox at build time. <b>Edit</b> opens the instruction (and,
    if the fix is really in the command, the command); the export is a
    change-set <span class="mono">apply_curation.py</span> runs directly.
    Editing a v4 row promotes it to v3 under the 2026-08-25 rule.
  </p>
  <div class="lede" style="margin-top:4px;max-width:none">
""" + "\n".join(blocks) + """
  </div>
"""


META = {
    "title": "Rewording Review",
    "lab": "vimBot &middot; open item #9 &middot; four instructions to reword",
    "h1": "Which reading did the gold take?",
    "total_label": "rows to reword",
    "flag_label": "all queued",
    "flag_button": "queued",
    "key": "vimbot-reword",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = reword_rows()
    cases = collect(args.cases, select=lambda r: r["id"] in rows,
                    flag=lambda r, risk: True,
                    sort_key=lambda c: list(rows).index(c["id"]))
    missing = sorted(set(rows) - {c["id"] for c in cases})
    if missing:
        sys.exit("rows not in %s: %s" % (args.cases, ", ".join(missing)))
    page = build(cases, {**META, "lede": lede(cases, rows)})
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="reword review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("reword rows: %d   vim disagreements: %d   page: %d KB  -> %s"
          % (len(cases), sum(1 for c in cases if not c["agrees"]),
             len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
