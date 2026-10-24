#!/usr/bin/env python3
"""Build the leakage-components page.

build_splits.py groups cases into components with a union-find over six keys
and deals whole components into train/val/test, so nothing near-duplicate can
straddle a split. The cost is that components can grow large -- the command
key merges whole command FAMILIES transitively through the buffer key (`dw`
links to another `dw` by command, that one links to `daw` by buffer). A
component bigger than any stratum's test budget is refused by test and val
and lands whole in train, dragging its categories' test share down with it.

Measured 2026-08-27: the largest component held 158 cases, a third of
`operator-motion`. That is as much a statement about corpus redundancy as
about splits, and this page exists so it can be inspected row by row rather
than reasoned about from a count.

Usage:
    python scripts/build_leakage_components.py --out page.html
"""
import argparse
import json
import pathlib
import sys
from collections import Counter

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_splits import build_components, leakage_keys   # noqa: E402
from page_guard import check_page                         # noqa: E402

KEY_NAMES = {"exact": "identical instruction + buffer",
             "near": "same instruction tokens (near-dup)",
             "buffer": "same buffer_before",
             "command": "same normalized command",
             "template": "same v2 template",
             "paraphrase": "paraphrase_of link"}


def collect(min_size):
    cases = [json.loads(l) for l in
             (REPO / "data" / "combined_cases.jsonl").open(encoding="utf-8") if l.strip()]
    where = {}
    for name in ("train", "val", "test"):
        p = REPO / "data" / "splits" / f"{name}.jsonl"
        if p.exists():
            for l in p.open(encoding="utf-8"):
                if l.strip():
                    where[json.loads(l)["id"]] = name

    comps = sorted(build_components(cases), key=len, reverse=True)
    out = []
    for ci, comp in enumerate(comps):
        if len(comp) < min_size:
            break
        members = [cases[i] for i in comp]
        # which keys actually bind this component: count how many members
        # share each key value with at least one other member
        seen = {}
        binding = Counter()
        for m in members:
            for k in leakage_keys(m):
                if k in seen:
                    binding[k[0]] += 1
                seen.setdefault(k, m["id"])
        splits = Counter(where.get(m["id"], "?") for m in members)
        out.append({
            "rank": ci + 1, "size": len(comp),
            "split": max(splits, key=splits.get),
            "splits": dict(splits),
            "cats": dict(Counter(m["category"] for m in members).most_common()),
            "cmds": Counter(m["gold_command"] for m in members).most_common(6),
            "bound_by": {KEY_NAMES.get(k, k): n for k, n in binding.most_common()},
            "rows": [{"id": m["id"], "cat": m["category"], "mode": m.get("mode", "ex"),
                      "cmd": m["gold_command"], "ins": m.get("instruction", ""),
                      "bef": m["buffer_before"], "split": where.get(m["id"], "?")}
                     for m in sorted(members, key=lambda r: (r["category"], r["id"]))],
        })
    sizes = [len(c) for c in comps]
    stats = {"cases": len(cases), "components": len(comps),
             "singletons": sum(1 for s in sizes if s == 1),
             "over_ten": sum(1 for s in sizes if s >= 10),
             "in_big": sum(s for s in sizes if s >= min_size)}
    return out, stats


CSS = r"""
:root{
  --ground:#F4F2EF; --surface:#FCFBF9; --sunk:#EBE7E2; --raise:#E3DED8;
  --ink:#201C19; --ink2:#4B443E; --muted:#72695F; --faint:#A1978C;
  --line:#DAD3CB; --line2:#E8E2DB;
  --accent:#8A4B1F; --accent2:#A65E2B; --accent-soft:#F1E3D6; --on-accent:#FCFBF9;
  --train:#4E5B8A; --train-bg:#E3E6F1; --val:#8A6A1E; --val-bg:#F3EAD3;
  --test:#2E6B4F; --test-bg:#DDEDE4;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
  --body:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --disp:"Fraunces","Iowan Old Style",Georgia,serif;
}
:root:not([data-theme="light"]){ color-scheme:light dark; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#16130F; --surface:#1E1A16; --sunk:#110F0C; --raise:#2A2520;
    --ink:#EDE7E0; --ink2:#C4BBB1; --muted:#8E847A; --faint:#615950;
    --line:#342E28; --line2:#2A2520;
    --accent:#D99A63; --accent2:#E7AE7C; --accent-soft:#33261A; --on-accent:#1E1A16;
    --train:#9AA7DA; --train-bg:#1E2235; --val:#D6B35E; --val-bg:#2C2415;
    --test:#6FC094; --test-bg:#16281E;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#16130F; --surface:#1E1A16; --sunk:#110F0C; --raise:#2A2520;
  --ink:#EDE7E0; --ink2:#C4BBB1; --muted:#8E847A; --faint:#615950;
  --line:#342E28; --line2:#2A2520;
  --accent:#D99A63; --accent2:#E7AE7C; --accent-soft:#33261A; --on-accent:#1E1A16;
  --train:#9AA7DA; --train-bg:#1E2235; --val:#D6B35E; --val-bg:#2C2415;
  --test:#6FC094; --test-bg:#16281E;
}
*{ box-sizing:border-box; }
html,body{ height:100%; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--body);
  font-size:15.5px; line-height:1.55; -webkit-font-smoothing:antialiased;
  display:flex; flex-direction:column; overflow:hidden; }
h1,h2,h3{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.01em;
  text-wrap:balance; }
.mono{ font-family:var(--mono); }
.wrap{ max-width:1120px; margin:0 auto; padding:0 22px; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
header{ flex:0 0 auto; background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:22px 22px 16px; }
h1{ font-size:clamp(28px,4.2vw,40px); margin:8px 0 10px; }
.lede{ max-width:68ch; color:var(--ink2); font-size:16px; }
.lede b{ color:var(--ink); font-weight:600; }
.band{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:9px;
  overflow:hidden; margin-top:16px; }
.band div{ background:var(--surface); padding:11px 14px; }
.band .v{ font-family:var(--mono); font-size:21px; font-weight:700; display:block;
  font-variant-numeric:tabular-nums; }
.band .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.07em;
  text-transform:uppercase; color:var(--muted); margin-top:3px; display:block; }
main{ flex:1 1 auto; min-height:0; overflow-y:auto; padding:20px 0 60px; }
.comp{ background:var(--surface); border:1px solid var(--line); border-radius:9px;
  margin-bottom:14px; overflow:hidden; border-left:4px solid var(--line); }
.comp.s-train{ border-left-color:var(--train); }
.comp.s-val{ border-left-color:var(--val); }
.comp.s-test{ border-left-color:var(--test); }
.chead{ display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center;
  padding:12px 16px; border-bottom:1px solid var(--line2); }
.rank{ font-family:var(--disp); font-size:22px; font-weight:600; color:var(--accent);
  min-width:2.4em; }
.size{ font-family:var(--mono); font-size:14px; font-weight:700;
  font-variant-numeric:tabular-nums; }
.pill{ font-family:var(--mono); font-size:10px; font-weight:700; letter-spacing:.05em;
  text-transform:uppercase; padding:2px 7px; border-radius:3px; }
.p-train{ background:var(--train-bg); color:var(--train); }
.p-val{ background:var(--val-bg); color:var(--val); }
.p-test{ background:var(--test-bg); color:var(--test); }
.spacer{ flex:1 1 auto; }
.meta{ padding:10px 16px; display:grid; gap:10px 20px;
  grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); font-size:13.5px;
  color:var(--ink2); border-bottom:1px solid var(--line2); }
.meta .lab{ display:block; margin-bottom:4px; }
.bar{ display:flex; height:9px; border-radius:4px; overflow:hidden; background:var(--raise);
  margin-top:5px; }
.bar i{ display:block; height:100%; }
.bar .train{ background:var(--train); } .bar .val{ background:var(--val); }
.bar .test{ background:var(--test); }
.cmds .mono{ font-size:12px; }
.cmds span{ display:inline-block; margin:2px 8px 2px 0; }
.cmds .n{ color:var(--muted); font-size:11px; }
details{ border-top:1px solid var(--line2); }
summary{ cursor:pointer; padding:9px 16px; font-family:var(--mono); font-size:12px;
  color:var(--accent); user-select:none; list-style:none; }
summary::-webkit-details-marker{ display:none; }
summary::before{ content:"\25B8  "; }
details[open] > summary::before{ content:"\25BE  "; }
summary:hover{ background:var(--sunk); }
table.rows{ width:100%; border-collapse:collapse; font-size:12.5px; }
table.rows th,table.rows td{ text-align:left; padding:6px 12px; vertical-align:top;
  border-top:1px solid var(--line2); }
table.rows th{ font-family:var(--mono); font-size:9.5px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:700; border-top:0; }
table.rows td.mono{ font-family:var(--mono); font-size:11.5px; white-space:nowrap; }
table.rows td.buf{ font-family:var(--mono); font-size:11px; color:var(--muted);
  white-space:pre; max-width:34ch; overflow:hidden; text-overflow:ellipsis; }
.note{ border-left:2px solid var(--accent); padding:5px 0 5px 14px; color:var(--ink2);
  font-size:14.5px; max-width:74ch; margin:14px 0 18px; }
.note b{ color:var(--ink); }
.scroll{ overflow-x:auto; }
"""

JS = r"""
document.querySelectorAll('.tt').forEach(function(t){
  t.addEventListener('click', function(){
    const s = t.getAttribute('data-split');
    document.querySelectorAll('.comp').forEach(function(c){
      c.style.display = (s === 'all' || c.classList.contains('s-' + s)) ? '' : 'none';
    });
    document.querySelectorAll('.tt').forEach(function(x){
      x.setAttribute('aria-pressed', String(x === t)); });
  });
});
"""


def esc(t):
    return (str(t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def render_comp(c):
    n = c["size"]
    bar = "".join(
        '<i class="%s" style="width:%.1f%%" title="%s %d"></i>'
        % (s, 100 * c["splits"].get(s, 0) / n, s, c["splits"].get(s, 0))
        for s in ("train", "val", "test"))
    cats = ", ".join("%s %d" % (esc(k), v) for k, v in c["cats"].items())
    cmds = "".join('<span><span class="mono">%s</span> <span class="n">&times;%d</span></span>'
                   % (esc(k[:30]), v) for k, v in c["cmds"])
    bound = ", ".join("%s (%d)" % (esc(k), v) for k, v in c["bound_by"].items())
    rows = "".join(
        '<tr><td class="mono">%s</td><td><span class="pill p-%s">%s</span></td>'
        '<td>%s</td><td class="mono">%s</td><td>%s</td><td class="buf">%s</td></tr>'
        % (esc(r["id"]), r["split"], r["split"], esc(r["cat"]), esc(r["cmd"][:36]),
           esc(r["ins"][:80]), esc(r["bef"][:60].replace("\n", "↵")))
        for r in c["rows"])
    return '''<div class="comp s-%s">
  <div class="chead"><span class="rank">#%d</span><span class="size">%d cases</span>
    <span class="pill p-%s">landed in %s</span><span class="spacer"></span>
    <span class="lab">%s</span></div>
  <div class="meta">
    <div><span class="lab">categories</span>%s
      <div class="bar">%s</div></div>
    <div><span class="lab">most repeated commands</span><div class="cmds">%s</div></div>
    <div><span class="lab">what binds it (shared keys)</span>%s</div>
  </div>
  <details><summary>show all %d rows</summary>
    <div class="scroll"><table class="rows"><thead><tr><th>id</th><th>split</th>
      <th>category</th><th>command</th><th>instruction</th><th>buffer</th></tr></thead>
      <tbody>%s</tbody></table></div></details>
</div>''' % (c["split"], c["rank"], n, c["split"], c["split"],
             "train %d &middot; val %d &middot; test %d" % (
                 c["splits"].get("train", 0), c["splits"].get("val", 0),
                 c["splits"].get("test", 0)),
             cats, bar, cmds, bound, n, rows)


def build(comps, stats, min_size):
    body = "".join(render_comp(c) for c in comps)
    return """<title>Leakage Components</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Fraunces:opsz,wght@9..144,600&family=Public+Sans:wght@400;600&\
family=JetBrains+Mono:wght@400;700&display=swap">
<style>%s</style>
<header><div class="wrap">
  <div class="lab">vimBot &middot; splits &middot; leakage components</div>
  <h1>Where near-duplicates cluster</h1>
  <p class="lede">
    Splits deal <b>whole components</b> &mdash; groups of cases joined by any of
    six leakage keys &mdash; so nothing near-duplicate can straddle train and
    test. The price is that the command key links whole command <i>families</i>
    through shared buffers, and a component bigger than any stratum's test
    budget is refused by test and val and lands entirely in train. Every
    component of %d or more rows is below, largest first, with every member.
  </p>
  <div class="band">
    <div><span class="v">%d</span><span class="k">cases</span></div>
    <div><span class="v">%d</span><span class="k">components</span></div>
    <div><span class="v">%d</span><span class="k">singletons</span></div>
    <div><span class="v">%d</span><span class="k">components of 10+</span></div>
    <div><span class="v">%d</span><span class="k">cases in the ones shown</span></div>
  </div>
  <div style="display:flex;gap:7px;margin-top:12px;flex-wrap:wrap;align-items:center">
    <span class="lab">landed in</span>
    <button class="pill tt" data-split="all" aria-pressed="true" style="cursor:pointer;border:1px solid var(--line);background:var(--surface);color:var(--ink2)">all</button>
    <button class="pill tt p-train" data-split="train" aria-pressed="false" style="cursor:pointer;border:0">train</button>
    <button class="pill tt p-val" data-split="val" aria-pressed="false" style="cursor:pointer;border:0">val</button>
    <button class="pill tt p-test" data-split="test" aria-pressed="false" style="cursor:pointer;border:0">test</button>
  </div>
</div></header>
<main><div class="wrap">
  <p class="note"><b>Reading the top of this list as a data finding:</b> a
  component that is a third of one category is that category being far weaker
  than its row count suggests. The rows are individually valid; collectively
  they teach one thing many times.</p>
  %s
</div></main>
<script>%s</script>
""" % (CSS, min_size, stats["cases"], stats["components"], stats["singletons"],
       stats["over_ten"], stats["in_big"], body, JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-size", type=int, default=6,
                    help="smallest component to list (default 6)")
    args = ap.parse_args()
    comps, stats = collect(args.min_size)
    page = build(comps, stats, args.min_size)
    check_page(page, required_js=["querySelectorAll('.tt')"], name="leakage")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("components listed: %d (>= %d rows)   largest: %s   page: %d KB -> %s"
          % (len(comps), args.min_size, [c["size"] for c in comps[:5]],
             len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
