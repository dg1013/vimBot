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
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import build, collect      # noqa: E402
from executor import run_vim                      # noqa: E402
from page_guard import check_page                 # noqa: E402

# id -> the issue, the buffer's language, and the other reading as a command
ROWS = {
    "hc-1024": {
        "issue": "says <i>capitalize</i>; the gold uppercases the whole "
                 "two-letter code, which “capitalize” would just as "
                 "well read as an initial capital.",
        "lang": "plain text — a city, a two-letter state code, a ZIP",
        "alt": ("initial capital only", "ex", r"%s/\<\a\{2}\>/\u&/g"),
    },
    "hc-1625": {
        "issue": "says <i>capitalise</i>; here the gold uppercases one "
                 "letter, the opposite reading of hc-1024 for the same word, "
                 "and the spelling is the British <span class=\"mono\">-ise</span> "
                 "against the corpus’s <span class=\"mono\">-ize</span>.",
        "lang": "Markdown note",
        "alt": ("the whole word in caps", "keys", "/paris<CR>gUiw"),
    },
    "hc-1908": {
        "issue": "already reworded on the test slice to say SQL, but the gold "
                 "inserts into a table called <span class=\"mono\">kv</span> "
                 "with a <span class=\"mono\">(name, value)</span> shape that "
                 "appears nowhere the model can see; any other table name or "
                 "column list is an equally faithful reading.",
        "lang": "SQL — a <span class=\"mono\">--</span> comment, then "
                "<span class=\"mono\">name=value</span> setting lines",
        "alt": ("a table name and column list the model might guess instead", "ex",
                r"v/^--/s/^\(\w\+\)=\(\d\+\)$/INSERT INTO settings (name, value) VALUES ('\1', \2);/"),
    },
    "hc-0803": {
        "issue": "your verdict on the test slice was <i>reword: should contain "
                 "language information</i>. <b>Reworded 2026-09-04</b> to name "
                 "the language: the buffer is <b>Ruby</b> "
                 "(<span class=\"mono\">class</span>, "
                 "<span class=\"mono\">def initialize</span>, "
                 "<span class=\"mono\">@rows</span>, "
                 "<span class=\"mono\">puts</span>, "
                 "<span class=\"mono\">end</span>). Still not in the wording: "
                 "the gold picks <span class=\"mono\">logger.debug</span>, and "
                 "the method and the level are unstated, so "
                 "<span class=\"mono\">logger.info</span> or "
                 "<span class=\"mono\">Rails.logger.debug</span> would be just "
                 "as faithful.",
        "lang": "Ruby",
        "alt": ("another level the wording allows", "keys",
                "/rows =/+1<CR>^cwlogger.info<Esc>"),
    },
}

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


def lede(cases):
    blocks = []
    for c in cases:
        spec = ROWS[c["id"]]
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
    cases = collect(args.cases, select=lambda r: r["id"] in ROWS,
                    flag=lambda r, risk: True,
                    sort_key=lambda c: list(ROWS).index(c["id"]))
    missing = sorted(set(ROWS) - {c["id"] for c in cases})
    if missing:
        sys.exit("rows not in %s: %s" % (args.cases, ", ".join(missing)))
    page = build(cases, {**META, "lede": lede(cases)})
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="reword review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("reword rows: %d   vim disagreements: %d   page: %d KB  -> %s"
          % (len(cases), sum(1 for c in cases if not c["agrees"]),
             len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
