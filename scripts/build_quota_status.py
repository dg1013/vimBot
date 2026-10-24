#!/usr/bin/env python3
"""Build the quota status page: taxonomy cells and VimGolf theme coverage.

Two different questions, so two different charts:

  CATEGORIES have quotas (data/taxonomy.yaml, per category x tier), so the
  question is fill against a target. Each category is one bar of
  min(fill, 100)% with the real numbers beside it and its three tier cells
  broken out underneath; over-quota is stated as a surplus rather than drawn
  past the end, because the bar's scale IS the quota.

  THEMES (data/vimgolf-themes.md, T01-T20) have no quota — they were never a
  target, only a map of what VimGolf-style work looks like — so the question
  is coverage, and the chart is a plain count per theme on one shared scale.
  The rows with no theme at all (the mined and generated corpora) are stated
  rather than hidden.

Status colours are the six-checks-validated steps (DECISIONS.md 2026-09-05);
every state also carries a word, so colour is never the only channel.

Usage:
    python scripts/build_quota_status.py --out page.html
"""
import argparse
import collections
import html
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from page_guard import check_page          # noqa: E402

TIERS = ("T1", "T2", "T3")
THEME_LINE = re.compile(r"^## (T\d\d) (.+?)\s*\(", re.M)


def load():
    import yaml
    tax = yaml.safe_load((REPO / "data" / "taxonomy.yaml").read_text(encoding="utf-8"))
    cats = tax.get("categories", tax)
    quotas = {k: v["quotas"] for k, v in cats.items()
              if isinstance(v, dict) and "quotas" in v}
    modes = {k: (v.get("mode") or "") for k, v in cats.items()
             if isinstance(v, dict) and "quotas" in v}
    rows = [json.loads(l) for l in
            (REPO / "data" / "combined_cases.jsonl").open(encoding="utf-8")
            if l.strip()]
    themes = dict(THEME_LINE.findall(
        (REPO / "data" / "vimgolf-themes.md").read_text(encoding="utf-8")))
    return quotas, modes, rows, themes


def bar(pct, cls, label):
    return ('<div class="track" role="img" aria-label="%s"><i class="%s" style="width:%.1f%%"></i></div>'
            % (html.escape(label), cls, max(0.0, min(100.0, pct))))


def build(quotas, modes, rows, themes):
    have = collections.Counter((r["category"], r.get("tier") or "?") for r in rows)
    per_cat = collections.Counter(r["category"] for r in rows)
    tcount = collections.Counter(r.get("theme") or "" for r in rows)
    cells_met = sum(1 for c in quotas for t in TIERS if have[(c, t)] >= quotas[c][t])
    cells = len(quotas) * 3
    cats_met = sum(1 for c in quotas if per_cat[c] >= sum(quotas[c].values()))
    total_q = sum(sum(q.values()) for q in quotas.values())
    in_scope = sum(per_cat[c] for c in quotas)

    def fill(n, q):
        return 100.0 * n / q if q else 0.0

    order = sorted(quotas, key=lambda c: (fill(per_cat[c], sum(quotas[c].values())), c))
    crows = []
    for cat in order:
        q = sum(quotas[cat].values())
        n = per_cat[cat]
        pct = fill(n, q)
        met = n >= q
        surplus = ('<span class="chip over">+%d over</span>' % (n - q)) if met else \
                  ('<span class="chip short">%d short</span>' % (q - n))
        tiers = "".join(
            '<div class="tier"><span class="tl">%s</span>%s<span class="tn">%d/%d</span></div>'
            % (t, bar(fill(have[(cat, t)], quotas[cat][t]),
                      "ok" if have[(cat, t)] >= quotas[cat][t] else "short",
                      "%s %s: %d of %d" % (cat, t, have[(cat, t)], quotas[cat][t])),
               have[(cat, t)], quotas[cat][t])
            for t in TIERS)
        untiered = have[(cat, "?")]
        crows.append(
            '<div class="crow"><div class="chead"><span class="cname">%s</span>'
            '<span class="chip mode">%s</span>%s'
            '<span class="spacer"></span><span class="cnum">%d / %d &middot; %.0f%%</span></div>'
            '%s<div class="tiers">%s%s</div></div>'
            % (html.escape(cat), html.escape(modes.get(cat, "")), surplus, n, q, pct,
               bar(pct, "ok" if met else "short", "%s: %d of %d" % (cat, n, q)),
               tiers,
               ('<div class="tier"><span class="tl">&mdash;</span>'
                '<span class="tn" style="margin-left:8px">%d untiered</span></div>' % untiered)
               if untiered else ""))

    tmax = max(tcount[t] for t in themes) if themes else 1
    trows = "".join(
        '<div class="trow"><span class="tname">%s <b>%s</b></span>%s<span class="tn">%d</span></div>'
        % (t, html.escape(themes[t]), bar(100.0 * tcount[t] / tmax, "theme",
                                          "%s %s: %d cases" % (t, themes[t], tcount[t])),
           tcount[t])
        for t in sorted(themes))

    ctable = "".join("<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td>"
                     "<td>%d</td><td>%d</td><td>%.0f%%</td></tr>"
                     % (html.escape(c), have[(c, "T1")], quotas[c]["T1"],
                        have[(c, "T2")], quotas[c]["T2"], have[(c, "T3")], quotas[c]["T3"],
                        fill(per_cat[c], sum(quotas[c].values())))
                     for c in sorted(quotas))
    ttable = "".join("<tr><td>%s</td><td>%s</td><td>%d</td></tr>"
                     % (t, html.escape(themes[t]), tcount[t]) for t in sorted(themes))

    return """<title>Quota Status</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%(css)s</style>
<div id="tip" aria-hidden="true"></div>
<main>
<header class="wrap">
  <div class="lab">vimBot &middot; data/taxonomy.yaml &middot; data/vimgolf-themes.md</div>
  <h1>How full is every cell?</h1>
  <p class="lede">
    The corpus is <b>%(total)d cases</b>. <b>%(in_scope)d</b> sit in the
    %(ncat)d categories that carry a quota, against a target of
    <b>%(total_q)d</b>. <b>%(cells_met)d of %(cells)d</b> category&times;tier
    cells are at or over their quota, and <b>%(cats_met)d of %(ncat)d</b>
    categories are full overall. Sorted emptiest first, so the gaps read
    before the surpluses.
  </p>
  <div class="band">
    <div><span class="v">%(total)d</span><span class="k">cases in the corpus</span></div>
    <div><span class="v">%(cells_met)d<span class="of">/%(cells)d</span></span><span class="k">cells at quota</span></div>
    <div><span class="v">%(cats_met)d<span class="of">/%(ncat)d</span></span><span class="k">categories full</span></div>
    <div><span class="v">%(themed)d</span><span class="k">cases carrying a theme</span></div>
  </div>
</header>

<section class="wrap">
  <details open><summary><h2>Vim categories</h2><span class="hint">quota per tier from the taxonomy</span></summary>
  <div class="legend">
    <span><i class="sw ok"></i>at or over quota</span>
    <span><i class="sw short"></i>short of quota</span>
    <span class="note">A bar is filled against that cell&rsquo;s own quota, so 100%% is the target, not the maximum; a surplus is written out instead of drawn.</span>
  </div>
  <div class="cats">%(crows)s</div>
  <details class="tv"><summary>the same numbers as a table</summary>
  <table><tr><th>category</th><th>T1</th><th>quota</th><th>T2</th><th>quota</th><th>T3</th><th>quota</th><th>fill</th></tr>%(ctable)s</table>
  </details>
  </details>
</section>

<section class="wrap">
  <details open><summary><h2>VimGolf themes</h2><span class="hint">coverage, not a target</span></summary>
  <p class="lede small">
    The theme map has <b>no quota</b>: it was authored as a survey of what
    VimGolf-style tasks look like (inspiration only, no challenge content ever
    ingested), so these bars are counts on one shared scale rather than fill.
    <b>%(unthemed)d cases carry no theme at all</b> &mdash; the mined and
    generated corpora were never themed, which is a gap in the map rather than
    in the data.
  </p>
  <div class="themes">%(trows)s</div>
  <details class="tv"><summary>the same numbers as a table</summary>
  <table><tr><th>theme</th><th>name</th><th>cases</th></tr>%(ttable)s</table>
  </details>
  </details>
</section>
</main>
<script>%(js)s</script>
""" % {
        "css": CSS, "js": JS, "total": len(rows), "in_scope": in_scope,
        "ncat": len(quotas), "total_q": total_q, "cells_met": cells_met, "cells": cells,
        "cats_met": cats_met, "themed": sum(tcount[t] for t in themes),
        "unthemed": tcount[""], "crows": "".join(crows), "trows": trows,
        "ctable": ctable, "ttable": ttable,
    }


CSS = r"""
:root{
  --ground:#F2F4F6; --surface:#FBFCFD; --sunk:#E6EAEE; --raise:#DFE5EA;
  --ink:#161B22; --ink2:#39424D; --muted:#66717E; --faint:#98A2AE;
  --line:#D3DAE1; --line2:#E3E8ED;
  --accent:#0F5F66; --accent-soft:#DCEDEE;
  --ok:#006B41; --short:#AD721B; --theme:#0F5F66;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","DejaVu Sans Mono",Menlo,monospace;
  --body:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --disp:"Newsreader","Iowan Old Style",Georgia,serif;
}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
    --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
    --line:#2B333C; --line2:#222A32; --accent:#6FC3C8; --accent-soft:#123134;
    --ok:#24A96F; --short:#C48800; --theme:#6FC3C8;
  }
}
:root[data-theme="dark"]{
  --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
  --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
  --line:#2B333C; --line2:#222A32; --accent:#6FC3C8; --accent-soft:#123134;
  --ok:#24A96F; --short:#C48800; --theme:#6FC3C8;
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--body);
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }
main{ padding-bottom:56px; }
.wrap{ max-width:940px; margin:0 auto; padding:0 22px; }
h1,h2{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.012em; text-wrap:balance; }
h1{ font-size:clamp(27px,4vw,38px); margin:8px 0 12px; }
h2{ font-size:21px; display:inline; }
header{ background:var(--surface); border-bottom:1px solid var(--line); padding:24px 22px 20px; max-width:none; }
header > *{ max-width:940px; margin-left:auto; margin-right:auto; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.lede{ max-width:70ch; color:var(--ink2); font-size:15.5px; }
.lede b{ color:var(--ink); font-weight:600; }
.lede.small{ font-size:14px; margin:10px 0 16px; }
.band{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:9px; overflow:hidden; margin-top:18px; }
.band div{ background:var(--surface); padding:12px 15px; }
.band .v{ font-family:var(--mono); font-size:22px; font-weight:700; display:block; font-variant-numeric:tabular-nums; }
.band .of{ font-size:14px; color:var(--muted); font-weight:400; }
.band .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); margin-top:3px; display:block; }
section{ margin-top:30px; }
summary{ list-style:none; cursor:pointer; display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
  border-radius:6px; margin:0 -6px 8px; padding:4px 6px; user-select:none; }
summary::-webkit-details-marker{ display:none; }
summary:hover{ background:var(--sunk); }
summary::before{ content:"\25BE"; color:var(--muted); font-size:12px; }
details:not([open]) > summary::before{ content:"\25B8"; }
.hint{ font-family:var(--mono); font-size:10.5px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); }
.legend{ display:flex; flex-wrap:wrap; gap:8px 18px; align-items:center; font-family:var(--mono);
  font-size:11.5px; color:var(--ink2); margin:6px 0 14px; }
.legend .note{ font-family:var(--body); font-size:12.5px; color:var(--muted); max-width:60ch; }
.sw{ display:inline-block; width:11px; height:11px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.sw.ok{ background:var(--ok); } .sw.short{ background:var(--short); }
.cats{ display:flex; flex-direction:column; gap:12px; }
.crow{ background:var(--surface); border:1px solid var(--line); border-radius:9px; padding:11px 14px; }
.chead{ display:flex; align-items:baseline; gap:8px; flex-wrap:wrap; margin-bottom:7px; }
.cname{ font-family:var(--mono); font-size:13.5px; font-weight:700; }
.cnum{ font-family:var(--mono); font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; }
.spacer{ flex:1 1 auto; }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:4px; background:var(--raise); color:var(--ink2); }
.chip.over{ background:var(--accent-soft); color:var(--accent); font-weight:700; }
.chip.short{ color:var(--short); border:1px solid var(--short); background:transparent; font-weight:700; }
.track{ background:var(--sunk); border-radius:4px; height:11px; overflow:hidden; }
.track i{ display:block; height:100%; border-radius:4px 0 0 4px; }
.track i.ok{ background:var(--ok); } .track i.short{ background:var(--short); } .track i.theme{ background:var(--theme); }
.tiers{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:6px 16px; margin-top:9px; }
.tier{ display:flex; align-items:center; gap:8px; }
.tier .track{ flex:1 1 auto; height:7px; }
.tl{ font-family:var(--mono); font-size:10px; color:var(--muted); width:1.4em; }
.tn{ font-family:var(--mono); font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
.themes{ display:flex; flex-direction:column; gap:7px; }
.trow{ display:flex; align-items:center; gap:12px; }
.tname{ font-family:var(--mono); font-size:11.5px; color:var(--ink2); flex:0 0 250px; }
.tname b{ font-family:var(--body); font-weight:600; color:var(--ink); }
.trow .track{ flex:1 1 auto; height:12px; }
.trow .tn{ flex:0 0 3.5em; text-align:right; color:var(--ink2); }
details.tv{ margin-top:16px; }
details.tv > summary{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }
table{ border-collapse:collapse; font-family:var(--mono); font-size:11.5px; font-variant-numeric:tabular-nums; width:100%; }
th,td{ padding:4px 9px; border-bottom:1px solid var(--line2); text-align:right; }
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){ text-align:left; }
#tip{ position:fixed; pointer-events:none; background:var(--ink); color:var(--ground); font-family:var(--mono);
  font-size:11.5px; padding:5px 8px; border-radius:5px; opacity:0; transition:opacity .08s; z-index:50; white-space:nowrap; }
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; } }
"""

JS = r"""
const tip = document.getElementById('tip');
document.querySelectorAll('.track').forEach(function(t){
  t.addEventListener('mousemove', function(ev){
    tip.textContent = t.getAttribute('aria-label');
    tip.style.left = (ev.clientX + 12) + 'px';
    tip.style.top = (ev.clientY + 12) + 'px';
    tip.style.opacity = 1;
  });
  t.addEventListener('mouseleave', function(){ tip.style.opacity = 0; });
});
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    quotas, modes, rows, themes = load()
    page = build(quotas, modes, rows, themes)
    check_page(page, required_js=["querySelectorAll('.track')"], name="quota status")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    have = collections.Counter((r["category"], r.get("tier") or "?") for r in rows)
    met = sum(1 for c in quotas for t in TIERS if have[(c, t)] >= quotas[c][t])
    print("categories %d, cells at quota %d/%d, themes %d, page %d KB -> %s"
          % (len(quotas), met, len(quotas) * 3, len(themes), len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
