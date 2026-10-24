#!/usr/bin/env python3
"""Build the coverage-report artifact: v3 vs taxonomy, VimGolf themes, vim itself.

Three ledgers, all computed from live data (data/v3/v3_cases.jsonl,
data/taxonomy.yaml, data/vimgolf-themes.md is summarised statically):

  1. category x tier counts against the taxonomy quotas
  2. the 20 VimGolf theme concepts, tagged rows counted per theme
  3. a hand-maintained capability atlas of vim's surface, each area classed
     by what the harness can verify (buffer-diff oracle, sandbox screens)

The page is static HTML (HyperText Markup Language) with no JavaScript, so there is nothing to break at
runtime. Rebuild after any dataset change:

    conda run -n vimBot python scripts/build_coverage_report.py --out page.html
Then publish over the existing coverage artifact URL (uniform resource locator).
"""
import argparse
import collections
import json
import re
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

try:
    import yaml
except ImportError:
    sys.exit("needs pyyaml (present in the vimBot env)")

# ---------------------------------------------------------------- themes
# Summary of data/vimgolf-themes.md (authored 2026-08-13, inspiration-only).
THEMES = [
    ("T01", "Case transformations", 20),
    ("T02", "Data-format conversion (CSV/JSON/YAML/XML)", 35),
    ("T03", "Code-to-code translation", 15),
    ("T04", "Micro-refactors", 30),
    ("T05", "Alignment & columns", 15),
    ("T06", "Sorting & dedup", 25),
    ("T07", "Numbering & counting", 20),
    ("T08", "Sequence & table generation", 30),
    ("T09", "Line reordering & transposition", 20),
    ("T10", "Wrapping, joining & splitting", 20),
    ("T11", "Quoting & bracket surgery", 15),
    ("T12", "Structured-config editing", 20),
    ("T13", "Whitespace & indentation repair", 15),
    ("T14", "Extraction & filtering", 20),
    ("T15", "Prose & typo repair", 15),
    ("T16", "Duplication-with-variation", 15),
    ("T17", "Regex-craft (captures, lookaround)", 10),
    ("T18", "ASCII art & grids", 30),
    ("T19", "Encodings & ciphers", 12),
    ("T20", "Merge/diff repair", 8),
]

# ------------------------------------------------------- capability atlas
# Classes: covered (rows in v3) / thin (workable, nearly empty) /
# frozen (works in the sandbox, excluded by taxonomy scope v1) /
# unverifiable (runs, but no buffer diff for the oracle) /
# blocked (sandbox or environment forbids it).
# Each entry: (area, examples, class, note). Counts for covered/thin/frozen
# areas are filled from the corpus via CAT_OF below where possible.
ATLAS = [
    ("Operators + motions", "d/c/y/g~/gu/gU/g?/gq + w e b f t / ? } G",
     "covered", "operator-motion {c:operator-motion} rows; dgn/cgn search operators included"),
    ("Text objects", 'iw aw iW i"/a" i(/a( it/at ip/as is',
     "covered", "text-objects {c:text-objects} rows, all major objects exercised"),
    (":global / :vglobal", ":g//d, :v//d, :g//m0, embedded :s",
     "covered", "global {c:global} rows — {p:global}% of the corpus, mostly delete-matching"),
    ("Counts & dot-repeat", "3dd 2daw d2} . 2.",
     "covered", "repeat-count {c:repeat-count} + counts throughout the mined rows"),
    ("Increment / decrement", "<C-a> <C-x> with counts; hex, binary, negatives",
     "covered", "increment {c:increment} rows incl. leading-zero and hex-case semantics"),
    ("Visual mode (v/V/<C-v>)", "vjd, Vjr1, <C-v>jlI",
     "covered", "visual {c:visual} + visual-block {c:visual-block} rows — admitted "
                "2026-08-18, taxonomy categories since 2026-09-03"),
    ("Marks & jumps", "m< m> gv g; g,",
     "covered", "admitted 2026-08-18; a handful of mined rows use marks (gv, '<)"),
    ("Named registers", '"ayy "ap "_d "1p',
     "blocked", "quarantined 2026-08-27 — a register NAME is not derivable from a "
                "request, so a model can only guess it; 2 rows removed, 0 remain"),
    (":sort / :join / :move / :copy / ranges", ":2,4sort n, :m0, :t.",
     "covered", "sort-uniq {c:sort-uniq}, join {c:join}, move-copy {c:move-copy}, "
                "line-ops {c:line-ops} — filled out by the v4 hand-authored batch"),
    (":substitute with captures", r":s/\(..\)/\1/, \u \U flags, \=submatch()",
     "thin", "{c:substitute} rows against the taxonomy's largest quota (460); "
             "the v4 batch moved this from 3, and the \\= expression allowlist "
             "exists for exactly this"),
    ("Search-anchored edits", "d/pat, cgn, :s//~/",
     "thin", "{c:search-edit} rows; quota is 300"),
    ("Whitespace & :retab", ":retab, :s/\\s\\+$//, :g/^$/d",
     "thin", "{c:whitespace} rows against a 320 quota"),
    ("Insert-mode edits", "i/a/o/O with counts, <C-t> <C-d> <C-y> <C-e> <C-o>",
     "covered", "insert-edit {c:insert-edit} rows, emulator suites plus the v4 batch"),
    ("Undo / redo as steps", "u <C-r> inside a sequence",
     "unverifiable", "executes fine, but a command whose value is the *history* "
                     "not the buffer can't be told apart from its net edit"),
    ("Pure motions / yank-only", "w, }, yy with no paste",
     "unverifiable", "no buffer diff — and no-ops are banned outright as "
                     "reward-hacking bait (§7)"),
    ("Folds, windows, tabs, scrolling", "zf, :split, <C-w>, zz",
     "unverifiable", "UI state, not buffer text; invisible to the oracle"),
    ("Macros", "q{reg} … @{reg}",
     "blocked", "@ is screened by src/scope.py in keys mode; taxonomy excludes "
                "recording"),
    ("Persistence & files", ":w :q ZZ :e other :saveas",
     "blocked", "the harness owns persistence; writes refused by screen + "
                "namespace (§5)"),
    ("Shell & filters", ":!sort, :%!fmt, system()",
     "blocked", "a VimGolf staple the sandbox will never admit — filter tasks "
                "must be expressed as :sort/:g equivalents"),
    (":normal / :execute / expression register", ":norm dw, <C-r>=",
     "blocked", "dynamic evaluation; <C-r>= screened explicitly, allowlisted "
                "\\= expressions in :s are the one sanctioned window"),
]

CLASS_META = {
    "covered":      ("in the corpus",        "pass"),
    "thin":         ("workable, nearly empty", "warn"),
    "frozen":       ("sandbox-capable, scope-frozen", "accent"),
    "unverifiable": ("runs, oracle-blind",   "muted"),
    "blocked":      ("sandbox forbids",      "stop"),
}


def load():
    rows = []
    for ds in ("v3", "v4"):
        p = REPO / "data" / ds / ("%s_cases.jsonl" % ds)
        if p.exists():
            # tag by the file the row came from — `source: handcrafted` is not
            # the same question once a row has been promoted from v4 into v3
            rows += [dict(json.loads(l), _ds=ds)
                     for l in p.open(encoding="utf-8") if l.strip()]
    tax = yaml.safe_load((REPO / "data/taxonomy.yaml").read_text(encoding="utf-8"))
    quotas = {c: v["quotas"] for c, v in tax["categories"].items()}
    return rows, quotas, tax


def esc(t):
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def bar(frac, cls="pass"):
    pct = max(0.0, min(1.0, frac)) * 100
    return ('<div class="meter"><i class="%s" style="width:%.1f%%"></i></div>'
            % (cls, pct))


def build(rows, quotas, tax):
    n = len(rows)
    ct = collections.Counter((r.get("category"), r.get("tier") or "—") for r in rows)
    cat_tot = collections.Counter(r.get("category") for r in rows)
    th = collections.Counter(r.get("theme") for r in rows if r.get("theme"))
    modes = collections.Counter(r.get("mode") for r in rows)
    quota_total = sum(sum(q.values()) for q in quotas.values())

    def fill(text):
        text = re.sub(r"\{c:([\w-]+)\}",
                      lambda m: "{:,}".format(cat_tot.get(m.group(1), 0)), text)
        return re.sub(r"\{p:([\w-]+)\}",
                      lambda m: "%.0f" % (100.0 * cat_tot.get(m.group(1), 0) / n), text)

    # ---- category table: union of taxonomy cats and observed cats
    all_cats = sorted(set(quotas) | set(cat_tot), key=lambda c: -cat_tot.get(c, 0))
    cat_rows = []
    for c in all_cats:
        q = quotas.get(c)
        qtot = sum(q.values()) if q else None
        have = cat_tot.get(c, 0)
        cells = "".join(
            '<td class="num">%s</td>' % (ct.get((c, t), 0) or "·")
            for t in ("T1", "T2", "T3", "—"))
        if qtot:
            frac = have / qtot
            cls = "pass" if frac >= 1 else ("warn" if frac >= .25 else "stop")
            quota_cell = ('<td class="num">%d</td><td>%s</td>'
                          % (qtot, bar(frac, cls)))
        else:
            quota_cell = ('<td class="num">—</td><td><span class="offplan">'
                          'outside taxonomy v1</span></td>')
        share = 100.0 * have / n
        cat_rows.append(
            '<tr><td class="mono">%s</td>%s<td class="num"><b>%d</b></td>'
            '<td class="num">%.1f%%</td>%s</tr>'
            % (c, cells, have, share, quota_cell))

    # ---- theme table
    tagged = sum(th.values())
    theme_rows = []
    for tid, name, approx in THEMES:
        have = th.get(tid, 0)
        note = ""
        if tid == "T14":
            note = (' <span class="offplan">+ ~1,182 untagged :g/:v rows are '
                    'this theme in spirit</span>')
        cls = "pass" if have >= 20 else ("warn" if have else "stop")
        theme_rows.append(
            '<tr><td class="mono">%s</td><td>%s%s</td>'
            '<td class="num">~%d</td><td class="num"><b>%s</b></td><td>%s</td></tr>'
            % (tid, name, note, approx, have or "0", bar(min(1.0, have / 25), cls)))

    # ---- atlas
    atlas_rows = []
    for area, ex, cls, note in ATLAS:
        label, tone = CLASS_META[cls]
        atlas_rows.append(
            '<tr class="a-%s"><td><b>%s</b><br><span class="mono ex">%s</span></td>'
            '<td><span class="pill p-%s">%s</span></td><td>%s</td></tr>'
            % (cls, esc(area), esc(ex), tone, label, esc(fill(note))))


    counts = collections.Counter(c for _, _, c, _ in ATLAS)
    thin = sorted((c for c in quotas
                   if cat_tot.get(c, 0) < 0.15 * sum(quotas[c].values())),
                  key=lambda c: cat_tot.get(c, 0))
    zero = [t for t, _, _ in THEMES if not th.get(t)]
    v4n = sum(1 for r in rows if r.get("_ds") == "v4")
    v3n = n - v4n
    skew = ("<b>The corpus is still one command family wearing a dataset's clothes.</b> "
            "%s of %s rows are <span class=\"mono\">:g//:v//d</span>-shape &mdash; the "
            "deep-regex inheritance, now %d%% of the corpus rather than the 60%% it was "
            "before v4. The taxonomy's largest planned category, "
            "<span class=\"mono\">substitute</span> (quota 460, the capture-group tier the "
            "model most needs), went <b>3 &rarr; %d</b> with the v4 batch. Still under 15%% "
            "of quota: %s."
            % ("{:,}".format(cat_tot.get("global", 0)), "{:,}".format(n),
               100.0 * cat_tot.get("global", 0) / n, cat_tot.get("substitute", 0),
               ", ".join('<span class="mono">%s</span>' % c for c in thin) or "none"))
    themenote = (("<b>Every theme is now represented.</b> The eight generative themes that "
                  "were at zero &mdash; T02 format conversion, T03 code translation, T05 "
                  "alignment, T08 sequence generation, T12 config editing, T15 prose repair, "
                  "T17 regex-craft, T20 merge repair &mdash; were the slice no mined test "
                  "suite covers, because test suites do not test realistic documents. They "
                  "are hand-authored in v4, which is exactly the slice that makes the plugin "
                  "demo look real.")
                 if not zero else
                 ("<b>%d themes still at zero:</b> %s. These are the generative themes no "
                  "mined test suite covers." % (len(zero), ", ".join(zero))))
    page = TEMPLATE
    for k, v in {
        "__N__": "{:,}".format(n),
        "__QUOTA__": "{:,}".format(quota_total),
        "__QPCT__": "%.0f" % (100.0 * n / quota_total),
        "__EX__": "{:,}".format(modes.get("ex", 0)),
        "__KEYS__": "{:,}".format(modes.get("keys", 0)),
        "__KPCT__": "%.0f" % (100.0 * modes.get("keys", 0) / n),
        "__GLOBAL__": "{:,}".format(cat_tot.get("global", 0)),
        "__GPCT__": "%.0f" % (100.0 * cat_tot.get("global", 0) / n),
        "__TAGGED__": str(tagged),
        "__ZEROTHEMES__": str(sum(1 for t, _, _ in THEMES if not th.get(t))),
        "__SKEWNOTE__": skew,
        "__THEMENOTE__": themenote,
        "__V3N__": "{:,}".format(v3n),
        "__V4N__": "{:,}".format(v4n),
        "__CATROWS__": "\n".join(cat_rows),
        "__THEMEROWS__": "\n".join(theme_rows),
        "__ATLASROWS__": "\n".join(atlas_rows),
        "__NCOV__": str(counts["covered"]),
        "__NTHIN__": str(counts["thin"]),
        "__NFROZEN__": str(counts["frozen"]),
        "__NBLIND__": str(counts["unverifiable"]),
        "__NBLOCK__": str(counts["blocked"]),
    }.items():
        page = page.replace(k, v)
    return page


TEMPLATE = r"""<title>Coverage Ledger</title>
<style>
:root{
  --ground:#F2F0EE; --surface:#FCFBFA; --sunk:#E9E5E2; --raise:#E4DFDB;
  --ink:#231E22; --ink2:#4A4046; --muted:#6E6369; --faint:#9A8F95;
  --line:#DAD3CF; --line2:#E7E1DD;
  --accent:#6B2D5C; --accent2:#8A3E77; --accent-soft:#EFE2EC;
  --pass:#2E6E4E; --pass-bg:#DFEDE5; --pass-line:#AACFBB;
  --warn:#8C5A15; --warn-bg:#F4E9D8; --warn-line:#DFC79A;
  --stop:#9C3535; --stop-bg:#F3E1E0; --stop-line:#DFB4B2;
  --mono:ui-monospace,"SF Mono","Cascadia Mono","DejaVu Sans Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --disp:"Charter","Iowan Old Style","Palatino Linotype",Palatino,Georgia,serif;
}
:root:not([data-theme="light"]){ color-scheme:light dark; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#151215; --surface:#1D191C; --sunk:#120F12; --raise:#282227;
    --ink:#E7E0E4; --ink2:#C0B6BC; --muted:#8D8288; --faint:#665D63;
    --line:#332C31; --line2:#282227;
    --accent:#C98FBA; --accent2:#DDA9CF; --accent-soft:#2B1F28;
    --pass:#6BC195; --pass-bg:#17281F; --pass-line:#2C4736;
    --warn:#D6A85C; --warn-bg:#2A2216; --warn-line:#493B20;
    --stop:#E08484; --stop-bg:#2C1B1B; --stop-line:#4A2A2A;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#151215; --surface:#1D191C; --sunk:#120F12; --raise:#282227;
  --ink:#E7E0E4; --ink2:#C0B6BC; --muted:#8D8288; --faint:#665D63;
  --line:#332C31; --line2:#282227;
  --accent:#C98FBA; --accent2:#DDA9CF; --accent-soft:#2B1F28;
  --pass:#6BC195; --pass-bg:#17281F; --pass-line:#2C4736;
  --warn:#D6A85C; --warn-bg:#2A2216; --warn-line:#493B20;
  --stop:#E08484; --stop-bg:#2C1B1B; --stop-line:#4A2A2A;
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; }
h1,h2{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.01em; text-wrap:balance; }
.mono{ font-family:var(--mono); }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }
.wrap{ max-width:1180px; margin:0 auto; padding:0 22px; }
header{ background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:26px 22px 20px; }
h1{ font-size:clamp(26px,4vw,38px); margin:8px 0 10px; }
.lede{ max-width:70ch; color:var(--ink2); font-size:15.5px; }
.lede b{ color:var(--ink); font-weight:600; }
.agg{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:7px; overflow:hidden; margin-top:18px; }
.agg div{ background:var(--surface); padding:11px 14px; }
.agg .v{ font-family:var(--mono); font-size:20px; font-weight:700; display:block;
  font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.agg .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin-top:2px; display:block; }
main{ padding:26px 0 80px; }
section{ margin-top:30px; }
h2{ font-size:21px; margin-bottom:4px; }
p.sub{ margin:0 0 12px; color:var(--muted); font-size:13.5px; max-width:80ch; }
p.sub b{ color:var(--ink2); }
.tbl{ width:100%; border-collapse:collapse; font-size:13.5px; background:var(--surface);
  border:1px solid var(--line); border-radius:7px; overflow:hidden; }
.tbl th,.tbl td{ text-align:left; padding:7px 11px; border-bottom:1px solid var(--line2);
  vertical-align:top; }
.tbl th{ font-family:var(--mono); font-size:9.5px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); font-weight:600; background:var(--surface); }
.tbl tr:last-child td{ border-bottom:0; }
.tbl td.num,.tbl th.num{ text-align:right; font-variant-numeric:tabular-nums;
  font-family:var(--mono); font-size:12.5px; white-space:nowrap; }
.scroll{ overflow-x:auto; border-radius:7px; }
.meter{ height:7px; min-width:90px; border-radius:3px; background:var(--raise); overflow:hidden; }
.meter i{ display:block; height:100%; }
.meter .pass{ background:var(--pass); } .meter .warn{ background:var(--warn); }
.meter .stop{ background:var(--stop); }
.offplan{ font-family:var(--mono); font-size:10.5px; color:var(--accent); }
.pill{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.05em;
  text-transform:uppercase; padding:2px 7px; border-radius:3px; white-space:nowrap; border:1px solid; }
.p-pass{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }
.p-warn{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.p-stop{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); }
.p-accent{ background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
.p-muted{ background:var(--raise); color:var(--muted); border-color:var(--line); }
.ex{ font-size:11px; color:var(--muted); overflow-wrap:anywhere; }
.note{ border-left:2px solid var(--accent); padding:4px 0 4px 12px; color:var(--ink2);
  font-size:13.5px; max-width:76ch; margin:14px 0 0; }
.note b{ color:var(--ink); }
.cards2{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px; margin-top:12px; }
.card2{ background:var(--surface); border:1px solid var(--line); border-radius:7px; padding:13px 15px; }
.card2 h3{ font-family:var(--disp); font-size:16.5px; margin:0 0 4px; font-weight:600; }
.card2 .tag{ font-family:var(--mono); font-size:9.5px; letter-spacing:.07em; text-transform:uppercase;
  font-weight:700; padding:1px 6px; border-radius:3px; margin-left:6px; vertical-align:2px; }
.t-ready{ background:var(--pass-bg); color:var(--pass); }
.t-build{ background:var(--warn-bg); color:var(--warn); }
.t-ask{ background:var(--stop-bg); color:var(--stop); }
.card2 p{ margin:4px 0 0; font-size:13px; color:var(--ink2); }
.card2 .yield{ font-family:var(--mono); font-size:11px; color:var(--muted); margin-top:7px; display:block; }
.foot{ margin-top:34px; padding-top:16px; border-top:1px solid var(--line); font-size:12.5px;
  color:var(--muted); max-width:80ch; }
</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; datasets v3 + v4 &middot; coverage</div>
  <h1>Coverage Ledger</h1>
  <p class="lede">
    Where the <b>__N__-case corpus</b> (v3 __V3N__ mined and generated, v4 __V4N__
    hand-authored) actually sits against the three yardsticks that matter:
    the taxonomy it was planned around, the 20 VimGolf theme concepts, and the slice of vim the
    harness can verify at all. Computed from the live corpus &mdash; rebuild
    <span class="mono">scripts/build_coverage_report.py</span> after any dataset change.
  </p>
  <div class="agg">
    <div><span class="v">__N__</span><span class="k">cases: v3 __V3N__ + v4 __V4N__</span></div>
    <div><span class="v">__QPCT__%</span><span class="k">of the __QUOTA__ taxonomy quota</span></div>
    <div><span class="v" style="color:var(--stop)">__GPCT__%</span><span class="k">is one category (:g / :v)</span></div>
    <div><span class="v">__KPCT__%</span><span class="k">keys mode (was 0% in v2)</span></div>
    <div><span class="v" style="color:var(--warn)">__ZEROTHEMES__/20</span><span class="k">vimgolf themes at zero</span></div>
  </div>
</div></header>

<main class="wrap">

<section>
  <h2>Category &times; tier, against the plan</h2>
  <p class="sub">Quota bars: <b>green</b> met, <b>amber</b> a quarter filled, <b>red</b> nearly
  empty. The dash column is rows with no tier (the mined candidate slice). Categories marked
  <span class="offplan">outside taxonomy v1</span> arrived through mining &mdash; the scope
  question they raise is below.</p>
  <div class="scroll"><table class="tbl">
    <thead><tr><th>category</th><th class="num">T1</th><th class="num">T2</th>
      <th class="num">T3</th><th class="num">&mdash;</th><th class="num">total</th>
      <th class="num">share</th><th class="num">quota</th><th>fill</th></tr></thead>
    <tbody>__CATROWS__</tbody>
  </table></div>
  <p class="note">__SKEWNOTE__</p>
  <p class="note"><b>Scope, settled:</b> the v1 freeze on visual mode and marks was lifted
  by owner decision on 2026-08-18 (<span class="mono">data/command_coverage.md</span>), and
  <span class="mono">visual</span> / <span class="mono">visual-block</span> became taxonomy
  categories on 2026-09-03 &mdash; this page had gone on labelling them &ldquo;outside
  taxonomy v1&rdquo; for two weeks after the decision. Their quotas are placeholders sized to
  the current corpus, for the owner to set. Named registers went the other way: quarantined
  2026-08-27, because a register <i>name</i> is never derivable from a request.</p>
</section>

<section>
  <h2>The twenty VimGolf themes</h2>
  <p class="sub">Theme concepts from <span class="mono">data/vimgolf-themes.md</span>
  (inspiration-only; ~620 challenge titles read, no content ingested). Only the mined rows carry
  a <span class="mono">theme</span> tag &mdash; __TAGGED__ rows &mdash; so this understates
  kinship, but the zeros are real zeros.</p>
  <div class="scroll"><table class="tbl">
    <thead><tr><th>theme</th><th>concept</th><th class="num">site&nbsp;~</th>
      <th class="num">rows</th><th>coverage</th></tr></thead>
    <tbody>__THEMEROWS__</tbody>
  </table></div>
  <p class="note">__THEMENOTE__</p>
</section>

<section>
  <h2>Capability atlas: vim vs the harness</h2>
  <p class="sub">__NCOV__ areas in the corpus &middot; __NTHIN__ workable but nearly empty
  &middot; __NFROZEN__ sandbox-capable but scope-frozen &middot; __NBLIND__ invisible to the
  buffer-diff oracle &middot; __NBLOCK__ blocked by the sandbox. The oracle's rule: a
  capability exists here only if it changes buffer text deterministically.</p>
  <div class="scroll"><table class="tbl">
    <thead><tr><th>surface area</th><th>status</th><th>evidence / consequence</th></tr></thead>
    <tbody>__ATLASROWS__</tbody>
  </table></div>
</section>

<section>
  <h2>Routes to more data</h2>
  <p class="sub">Ordered by leverage over the gaps above, not by novelty. Tags:
  <span class="tag t-ready">pipeline exists</span> <span class="tag t-build">new build</span>
  <span class="tag t-ask">owner decision first</span></p>
  <div class="cards2">
    <div class="card2"><h3>Vendor full StructuredRegex<span class="tag t-ready">pipeline exists</span></h3>
      <p>The 447 sregex rows come from a 460-pattern recovered sample; the corpus is ~3.5k
      patterns. Fetching and vendoring it (already a &sect;14c to-do) re-runs the existing
      converter for roughly <b>8&times; that slice</b> &mdash; but it deepens the
      <span class="mono">:g/:v</span> skew, so sequence it after the gap-fillers.</p>
      <span class="yield">est. +2,500&ndash;3,000 rows &middot; blocked on licensing audit only</span></div>
    <div class="card2"><h3>Theme-guided authoring for the eight zero themes<span class="tag t-ready">pipeline exists</span></h3>
      <p>The v1 pipeline already produced 44 vimgolf-inspired scenarios this way: invent buffer +
      instruction + command per theme, validate through the executor. In-session authoring (no
      API key needed) against T02/T12/T17 first &mdash; substitute-with-captures and realistic
      documents in one move. Every row lands in the emptiest quotas.</p>
      <span class="yield">est. +300&ndash;800 rows at ~50&ndash;100/session &middot; highest value per row</span></div>
    <div class="card2"><h3>Command-grammar sampling, back-translated<span class="tag t-build">new build</span></h3>
      <p>Grow data from the executor instead of from corpora: enumerate a grammar over
      (operator &times; motion/object &times; count/range/flag), execute against A1 witness
      buffers, keep the interesting non-no-op diffs, author the instruction from the diff.
      Execution-validated by construction; fills operator-motion/text-object T2&ndash;T3 cells
      to any depth wanted.</p>
      <span class="yield">est. +1,000s, tunable &middot; reuses witness machinery + the &sect;6 authoring rules</span></div>
    <div class="card2"><h3>Finish the emulator seams<span class="tag t-ready">pipeline exists</span></h3>
      <p>Measured and waiting: CodeMirror has 256 unread blocks, VsVim 396 skipped setups,
      toybox's 34 sed tests need a sed&rarr;ex translation, neovim's functional suite (Apache
      Lua slice) and atom-vim-mode-plus need parsers. Diminishing per-source yield &mdash; the
      cheap rows are already in.</p>
      <span class="yield">est. +150&ndash;400 rows &middot; same intake gate as before</span></div>
    <div class="card2"><h3>Mutation of verified cases<span class="tag t-build">new build</span></h3>
      <p>Perturb what already passes: vary counts, ranges and flags on a gold, re-execute, keep
      changed outcomes; swap buffers via witness sampling. Cheap difficulty ladders (T1&rarr;T2)
      with controlled novelty &mdash; but paraphrase-cluster the ids so splits keep mutants
      together, or leakage checks become fiction.</p>
      <span class="yield">est. 2&ndash;3&times; multiplier on keys-mode rows &middot; splits discipline required</span></div>
    <div class="card2"><h3>Quarantine &rarr; preference pairs<span class="tag t-ready">pipeline exists</span></h3>
      <p>Not new rows &mdash; new leverage: every execution-rejected candidate already carries a
      <span class="mono">reject_reason</span>, giving (instruction, good, bad) pairs for the
      Week-3 error-analysis round and any DPO-style pass, free.</p>
      <span class="yield">existing rejects, zero authoring cost</span></div>
    <div class="card2"><h3>vis-test as a candidate generator<span class="tag t-ask">owner decision first</span></h3>
      <p>Its 28 non-golf <span class="mono">.in/.keys</span> pairs lack expected outputs, so they
      were unusable as <i>agreement</i> candidates &mdash; but our gate is execution + authored
      instructions anyway. Using vim as sole oracle on someone else's inputs is a policy change
      worth an explicit yes/no. (The 6 golf-directory triples stay excluded regardless.)</p>
      <span class="yield">est. +20&ndash;25 rows &middot; ISC licence, small but keys-rich</span></div>
    <div class="card2"><h3>Standing red lines<span class="tag t-ask">owner decision first</span></h3>
      <p>Vim's own testdir (Vim licence), VimGolf content (no grant), CommitPackFT (instructions
      too abstract &mdash; measured, rejected), Stack Exchange (CC BY-SA share-alike &mdash;
      unresolved &sect;14c). All recorded in DECISIONS; none enter the dataset without a new
      licensing entry.</p>
      <span class="yield">excluded &middot; the audit trail is the asset</span></div>
  </div>
</section>

<p class="foot">Counts computed from <span class="mono">data/v3/v3_cases.jsonl</span>,
<span class="mono">data/v4/v4_cases.jsonl</span> and
<span class="mono">data/taxonomy.yaml</span> at build time &mdash; including the prose in the
callouts, which cites live figures rather than frozen ones. Theme approximations are challenge
counts read from vimgolf.com titles on 2026-08-13; no challenge content is used anywhere.
The capability atlas is hand-maintained in the builder &mdash; update it when the sandbox or
scope changes.</p>

</main>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows, quotas, tax = load()
    page = build(rows, quotas, tax)
    out = pathlib.Path(args.out)
    out.write_text(page, encoding="utf-8")
    print("cases: %d  page: %.0f KB  -> %s"
          % (len(rows), len(page.encode()) / 1024, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
