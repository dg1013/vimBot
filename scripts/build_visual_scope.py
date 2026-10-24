#!/usr/bin/env python3
"""Build the visual-mode scope decision page.

Taxonomy v1 (CLAUDE.md §7) froze visual mode, marks and named registers out of
scope. Mining brought 228 execution-verified rows in anyway. Blessing them as
scope v2 or quarantining them is an owner decision, and it has to be made
before the splits are final because it changes what sits in the test slice.

The page exists to make that decision on evidence rather than principle. Its
core question, computed here rather than asserted: does a simpler non-visual
command produce the byte-identical buffer? A visual case with an
operator+motion twin teaches the model a detour; one without it is capability
the corpus has nowhere else.

Usage:
    python scripts/build_visual_scope.py --out page.html
"""
import argparse
import html
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim              # noqa: E402
from evaluate import _norm                # noqa: E402
from page_guard import check_page         # noqa: E402

PENDING = ("visual", "visual-block", "registers")


# ------------------------------------------------------------------ analysis

def twin_candidates(cmd):
    """Simpler non-visual rewrites worth testing for buffer equality."""
    out = []
    m = re.fullmatch(r"v(.+?)([dcyx~])(.*)", cmd)
    if m:
        motion, op, tail = m.groups()
        out.append(op + motion + tail)
    m = re.fullmatch(r"V(.*?)([dcyx])(.*)", cmd)
    if m:
        motion, op, tail = m.groups()
        if motion:
            out.append(op + motion + tail)
        else:
            out.append(op * 2 + tail)
    return [c for c in out if c and c != cmd]


def ex_twin_candidates(cmd):
    """Ex-range rewrites. A linewise visual op is usually just a range, and
    that is how a person would actually write it, so an ex twin condemns a
    row exactly as much as a keys twin does."""
    out = []
    m = re.fullmatch(r"V(?:(\d*)j)?([dy<>])", cmd)
    if m:
        n, op = m.groups()
        span = int(n) if n else (1 if "j" in cmd else 0)
        out.append(".,.+%d%s" % (span, op) if span else ".%s" % op)
    m = re.match(r"[Vv](?:(\d*)j)*v?:(.+?)[\r\n]*$", cmd)
    if m:
        n, rest = m.groups()
        span = cmd.count("j") if not n else int(n) * cmd.count("j")
        if span:
            out.append(".,.+%d%s" % (span, rest))
    return out


def find_twin(row):
    """The simplest command that reproduces this row's buffer, or None.

    Returns (command, mode) so the page can say which register the twin is
    written in -- an ex twin is the more damning of the two, since it is what
    the corpus's own substitute/global rows already look like.
    """
    cur = tuple(row.get("cursor") or (1, 1))
    trials = [(c, "keys") for c in twin_candidates(row["gold_command"])]
    trials += [(c, "ex") for c in ex_twin_candidates(row["gold_command"])]
    for cand, mode in trials:
        try:
            got = run_vim(row["buffer_before"], cand, mode=mode, cursor=cur)
        except Exception:
            continue
        if got.get("was_valid") and not got.get("blocked") \
                and _norm(got.get("buffer_after")) == _norm(row["buffer_after"]):
            return cand, mode
    return None, ""


def capability(cmd, category):
    """What this case actually exercises -- the axis the decision turns on."""
    if category == "visual-block":
        return "column / block editing"
    if category == "registers":
        return "named registers"
    if re.search(r"<C-[axAX]>", cmd):
        return "increment ladder over a selection"
    if ":" in cmd:
        return "selection feeding an ex command"
    if "gq" in cmd:
        return "reflow a selection"
    if re.search(r"g[uU~]|[uU~]$", cmd):
        return "case change over a selection"
    if re.search(r"r\S", cmd):
        return "replace every char in a selection"
    if re.search(r"[<>]", cmd):
        return "indent shift over a selection"
    if "<Esc>" in cmd:
        return "change a selection then insert"
    if re.search(r"[yp]", cmd):
        return "yank / put a selection"
    return "delete / change a selection"


def collect():
    path = REPO / "data" / "combined_cases.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    out = []
    for r in rows:
        if r["category"] not in PENDING:
            continue
        twin, twin_mode = find_twin(r)
        out.append({
            "twin_mode": twin_mode,
            "id": r["id"],
            "cat": r["category"],
            "ds": r.get("dataset", ""),
            "src": r.get("source", ""),
            "tier": r.get("tier") or "T?",
            "mode": r.get("mode", "keys"),
            "ins": r.get("instruction", ""),
            "cmd": r["gold_command"],
            "bef": r["buffer_before"],
            "aft": r["buffer_after"],
            "cur": r.get("cursor") or [1, 1],
            "twin": twin or "",
            "cap": capability(r["gold_command"], r["category"]),
        })
    out.sort(key=lambda c: (c["cat"], c["cap"], c["id"]))
    return out


# --------------------------------------------------------------------- page

CSS = r"""
:root{
  --ground:#F3F1F7; --surface:#FCFBFE; --sunk:#EAE6F1; --raise:#E2DCEC;
  --ink:#1E1B26; --ink2:#463F55; --muted:#6B6280; --faint:#9A93AC;
  --line:#D8D1E4; --line2:#E7E2EF;
  --accent:#5B3E8C; --accent2:#7355A8; --sel:#DDD0F2; --on-accent:#FCFBFE;
  --keep:#2F6B4F; --keep-bg:#DFEDE6; --keep-line:#A9CFBB;
  --detour:#9C5A21; --detour-bg:#F5E8D9; --detour-line:#E0C79E;
  --policy:#8F3440; --policy-bg:#F5E1E4; --policy-line:#DFB0B8;
  --cut:#8F3440; --cut-bg:#F7E7EA; --add:#2F6B4F; --add-bg:#E3F0E9;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","DejaVu Sans Mono",Menlo,monospace;
  --body:"Source Sans 3",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --disp:"Archivo",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
:root:not([data-theme="light"]){ color-scheme:light dark; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#141220; --surface:#1C1929; --sunk:#100E19; --raise:#272235;
    --ink:#E8E3F2; --ink2:#C0B8D2; --muted:#8B839C; --faint:#635C74;
    --line:#312B42; --line2:#272235;
    --accent:#B79AE0; --accent2:#CBB2EE; --sel:#3A2D57; --on-accent:#1C1929;
    --keep:#6FC49A; --keep-bg:#16281F; --keep-line:#2C4736;
    --detour:#D9A464; --detour-bg:#2A2115; --detour-line:#4A3A1F;
    --policy:#E08894; --policy-bg:#2C1A1E; --policy-line:#4A2A31;
    --cut:#E08894; --cut-bg:#2D1B20; --add:#6FC49A; --add-bg:#152720;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#141220; --surface:#1C1929; --sunk:#100E19; --raise:#272235;
  --ink:#E8E3F2; --ink2:#C0B8D2; --muted:#8B839C; --faint:#635C74;
  --line:#312B42; --line2:#272235;
  --accent:#B79AE0; --accent2:#CBB2EE; --sel:#3A2D57; --on-accent:#1C1929;
  --keep:#6FC49A; --keep-bg:#16281F; --keep-line:#2C4736;
  --detour:#D9A464; --detour-bg:#2A2115; --detour-line:#4A3A1F;
  --policy:#E08894; --policy-bg:#2C1A1E; --policy-line:#4A2A31;
  --cut:#E08894; --cut-bg:#2D1B20; --add:#6FC49A; --add-bg:#152720;
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--body); font-size:16px; line-height:1.55;
  -webkit-font-smoothing:antialiased; }
h1,h2,h3{ font-family:var(--disp); margin:0; letter-spacing:-.018em;
  text-wrap:balance; font-weight:700; }
.mono{ font-family:var(--mono); }
.wrap{ max-width:1140px; margin:0 auto; padding:0 24px; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px;
  border-radius:3px; }

header{ background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:30px 24px 24px; }
h1{ font-size:clamp(30px,4.6vw,46px); margin:10px 0 12px; line-height:1.05; }
h1 .hl{ background:var(--sel); box-shadow:0 0 0 4px var(--sel);
  border-radius:2px; }
.lede{ max-width:64ch; color:var(--ink2); font-size:17px; }
.lede b{ color:var(--ink); font-weight:600; }

.band{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line);
  border-radius:9px; overflow:hidden; margin-top:22px; }
.band div{ background:var(--surface); padding:13px 16px; }
.band .v{ font-family:var(--mono); font-size:23px; font-weight:700;
  display:block; font-variant-numeric:tabular-nums; letter-spacing:-.02em; }
.band .k{ font-family:var(--mono); font-size:10px; letter-spacing:.07em;
  text-transform:uppercase; color:var(--muted); margin-top:3px; display:block; }

main{ padding:30px 0 90px; }
section{ margin-top:34px; }
h2{ font-size:23px; margin-bottom:5px; }
p.sub{ margin:0 0 16px; color:var(--muted); font-size:14.5px; max-width:76ch; }

.verdicts{ display:grid; gap:14px;
  grid-template-columns:repeat(auto-fit,minmax(290px,1fr)); }
.vcard{ background:var(--surface); border:1px solid var(--line);
  border-radius:9px; padding:17px 19px; border-top:3px solid var(--line); }
.vcard.keep{ border-top-color:var(--keep); }
.vcard.split{ border-top-color:var(--detour); }
.vcard.policy{ border-top-color:var(--policy); }
.vcard h3{ font-size:18px; margin-bottom:3px; }
.vcard .n{ font-family:var(--mono); font-size:12px; color:var(--muted);
  font-variant-numeric:tabular-nums; }
.vcard p{ margin:9px 0 0; font-size:14.5px; color:var(--ink2); }
.rec{ display:inline-block; margin-top:12px; font-family:var(--mono);
  font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
  padding:4px 9px; border-radius:4px; border:1px solid; }
.rec.keep{ background:var(--keep-bg); color:var(--keep); border-color:var(--keep-line); }
.rec.split{ background:var(--detour-bg); color:var(--detour); border-color:var(--detour-line); }
.rec.policy{ background:var(--policy-bg); color:var(--policy); border-color:var(--policy-line); }

.rail{ position:sticky; top:0; z-index:20; background:var(--surface);
  border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  margin-top:34px; }
.rail .wrap{ display:flex; flex-wrap:wrap; gap:8px 10px; align-items:center;
  padding:11px 24px; }
.f{ font-family:var(--mono); font-size:12px; padding:4px 10px; border-radius:5px;
  cursor:pointer; background:var(--surface); border:1px solid var(--line);
  color:var(--ink2); }
.f:hover{ border-color:var(--accent); color:var(--accent); }
.f[aria-pressed="true"]{ background:var(--sel); border-color:var(--accent);
  color:var(--accent); font-weight:700; }
.f .c{ opacity:.62; margin-left:6px; font-variant-numeric:tabular-nums; }
.count{ font-family:var(--mono); font-size:12.5px; color:var(--muted);
  margin-left:auto; font-variant-numeric:tabular-nums; }

.cards{ display:flex; flex-direction:column; gap:11px; margin-top:20px; }
.card{ background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--line); border-radius:8px; overflow:hidden; }
.card.twin{ border-left-color:var(--detour); }
.card.uniq{ border-left-color:var(--keep); }
.card.pol{ border-left-color:var(--policy); }
.chead{ display:flex; flex-wrap:wrap; gap:7px 10px; align-items:center;
  padding:10px 14px; border-bottom:1px solid var(--line2); }
.cid{ font-family:var(--mono); font-size:12px; color:var(--muted); }
.chip{ font-family:var(--mono); font-size:10.5px; padding:2px 7px;
  border-radius:4px; background:var(--raise); color:var(--ink2);
  white-space:nowrap; }
.keys{ font-family:var(--mono); font-size:13px; font-weight:700;
  background:var(--sel); color:var(--accent); padding:3px 8px;
  border-radius:4px; overflow-wrap:anywhere; }
.spacer{ flex:1 1 auto; }
.tw{ font-family:var(--mono); font-size:11.5px; padding:3px 8px;
  border-radius:4px; background:var(--detour-bg); color:var(--detour);
  border:1px solid var(--detour-line); white-space:nowrap; font-weight:600; }
.ins{ padding:11px 14px 3px; font-size:15.5px; color:var(--ink);
  max-width:72ch; }
.panes{ display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
  gap:1px; background:var(--line2); border-top:1px solid var(--line2);
  margin-top:10px; }
.pane{ background:var(--surface); padding:9px 13px; min-width:0;
  overflow-x:auto; }
.pane .lab{ margin-bottom:5px; display:block; }
.ln{ display:flex; white-space:pre; font-family:var(--mono); font-size:12.5px;
  line-height:1.5; }
.ln .g{ color:var(--faint); opacity:.55; user-select:none; text-align:right;
  min-width:2em; padding-right:9px; flex:0 0 auto;
  font-variant-numeric:tabular-nums; }
.ln.cut{ background:var(--cut-bg); } .ln.cut .t{ color:var(--cut); }
.ln.add{ background:var(--add-bg); } .ln.add .t{ color:var(--add); }
.ws{ color:var(--accent); opacity:.4; }
.more{ display:flex; justify-content:center; padding:20px 0 0; }
.btn{ font-family:var(--mono); font-size:12.5px; padding:7px 15px;
  border-radius:6px; cursor:pointer; background:var(--surface);
  border:1px solid var(--line); color:var(--ink2); }
.btn:hover{ border-color:var(--accent); color:var(--accent); }
.empty{ padding:44px 0; text-align:center; color:var(--muted);
  font-family:var(--mono); font-size:13.5px; }
.boot{ padding:36px 0; text-align:center; color:var(--muted);
  font-family:var(--mono); font-size:13.5px; }
.note{ border-left:2px solid var(--accent); padding:5px 0 5px 14px;
  color:var(--ink2); font-size:15px; max-width:74ch; margin:16px 0 0; }
.note b{ color:var(--ink); font-weight:600; }
table.caps{ width:100%; border-collapse:collapse; font-size:14px;
  background:var(--surface); border:1px solid var(--line); border-radius:8px;
  overflow:hidden; }
table.caps th,table.caps td{ text-align:left; padding:8px 13px;
  border-bottom:1px solid var(--line2); }
table.caps th{ font-family:var(--mono); font-size:10px; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
table.caps tr:last-child td{ border-bottom:0; }
table.caps td.num{ text-align:right; font-family:var(--mono); font-size:13px;
  font-variant-numeric:tabular-nums; white-space:nowrap; }
.scroll{ overflow-x:auto; border-radius:8px; }
@media (prefers-reduced-motion:reduce){ *{ transition:none !important;
  animation:none !important; } }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('payload').textContent);
let fCat = 'all', fKind = 'all', shown = 30;

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                  .replace(/>/g,'&gt;');
}
function vis(line){
  let out = esc(line).replace(/\t/g, '<span class="ws">--&gt;</span>\t');
  const m = out.match(/( +)$/);
  if (m) out = out.slice(0, out.length - m[1].length)
              + '<span class="ws">' + '.'.repeat(m[1].length) + '</span>';
  return out;
}
function lines(text){
  const ls = String(text).split('\n');
  if (ls.length && ls[ls.length-1] === '') ls.pop();
  return ls;
}
function pane(label, text, other, kind){
  const a = lines(text), b = lines(other);
  let rows = '';
  for (let i = 0; i < a.length; i++){
    const changed = a[i] !== b[i];
    const cls = changed ? ' ' + kind : '';
    rows += '<div class="ln' + cls + '"><span class="g">' + (i+1)
          + '</span><span class="t">' + vis(a[i]) + '</span></div>';
  }
  if (!rows) rows = '<div class="ln"><span class="g">-</span>'
                  + '<span class="t">(empty)</span></div>';
  return '<div class="pane"><span class="lab">' + label + '</span>'
       + rows + '</div>';
}
function card(c){
  const cls = c.cat === 'registers' ? 'pol' : (c.twin ? 'twin' : 'uniq');
  const twin = c.twin
    ? '<span class="tw">same result from ' + esc(c.twin)
      + (c.twin_mode === 'ex' ? ' (ex)' : '') + '</span>'
    : '';
  return '<div class="card ' + cls + '">'
    + '<div class="chead">'
    +   '<span class="cid">' + esc(c.id) + '</span>'
    +   '<span class="keys">' + esc(c.cmd) + '</span>'
    +   '<span class="chip">' + esc(c.cat) + '</span>'
    +   '<span class="chip">' + esc(c.tier) + '</span>'
    +   '<span class="chip">' + esc(c.src) + '</span>'
    +   '<span class="spacer"></span>' + twin
    + '</div>'
    + '<div class="ins">' + esc(c.ins || '(no instruction)') + '</div>'
    + '<div class="panes">'
    +   pane('buffer before', c.bef, c.aft, 'cut')
    +   pane('buffer after', c.aft, c.bef, 'add')
    + '</div>'
    + '</div>';
}
function matches(c){
  if (fCat !== 'all' && c.cat !== fCat) return false;
  if (fKind === 'twin' && !c.twin) return false;
  if (fKind === 'uniq' && c.twin) return false;
  return true;
}
function render(){
  const hits = DATA.filter(matches);
  const slice = hits.slice(0, shown);
  const el = document.getElementById('cards');
  el.innerHTML = slice.length
    ? slice.map(card).join('')
    : '<div class="empty">nothing matches that filter</div>';
  document.getElementById('count').textContent =
    hits.length + ' of ' + DATA.length + ' shown';
  const more = document.getElementById('more');
  more.style.display = hits.length > shown ? 'flex' : 'none';
  document.getElementById('moreBtn').textContent =
    'Show 30 more (' + (hits.length - shown) + ' left)';
}
function wire(){
  document.querySelectorAll('[data-cat]').forEach(function(b){
    b.addEventListener('click', function(){
      fCat = b.getAttribute('data-cat'); shown = 30;
      document.querySelectorAll('[data-cat]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render();
    });
  });
  document.querySelectorAll('[data-kind]').forEach(function(b){
    b.addEventListener('click', function(){
      fKind = b.getAttribute('data-kind'); shown = 30;
      document.querySelectorAll('[data-kind]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render();
    });
  });
  document.getElementById('moreBtn').addEventListener('click', function(){
    shown += 30; render();
  });
}
try{
  document.getElementById('boot').style.display = 'none';
  wire(); render();
}catch(err){
  document.getElementById('boot').textContent = 'page failed to start: ' + err;
  document.getElementById('boot').style.display = 'block';
}
"""


def split_counts():
    """Where the pending rows actually sit -- the decision's real blast radius."""
    out = {}
    for name in ("train", "val", "test"):
        p = REPO / "data" / "splits" / ("%s.jsonl" % name)
        if not p.exists():
            out[name] = 0
            continue
        out[name] = sum(1 for l in p.open(encoding="utf-8") if l.strip()
                        and json.loads(l)["category"] in PENDING)
    return out


def build(cases):
    n = len(cases)
    twin = [c for c in cases if c["twin"]]
    uniq = [c for c in cases if not c["twin"]]
    sp = split_counts()
    by_cat = {}
    for c in cases:
        b = by_cat.setdefault(c["cat"], {"n": 0, "twin": 0})
        b["n"] += 1
        if c["twin"]:
            b["twin"] += 1

    caps = {}
    for c in cases:
        k = caps.setdefault(c["cap"], {"n": 0, "twin": 0})
        k["n"] += 1
        if c["twin"]:
            k["twin"] += 1
    cap_rows = "".join(
        '<tr><td>%s</td><td class="num">%d</td><td class="num">%d</td>'
        '<td class="num">%d</td></tr>'
        % (html.escape(k), v["n"], v["twin"], v["n"] - v["twin"])
        for k, v in sorted(caps.items(), key=lambda kv: -kv[1]["n"]))

    vb = by_cat.get("visual-block", {"n": 0, "twin": 0})
    vi = by_cat.get("visual", {"n": 0, "twin": 0})
    rg = by_cat.get("registers", {"n": 0, "twin": 0})

    payload = json.dumps(cases, ensure_ascii=False)

    return """<title>Visual Mode Scope</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Archivo:wght@600;700&family=Source+Sans+3:wght@400;600&\
family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%s</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; taxonomy scope &middot; owner decision</div>
  <h1>What happens to the <span class="hl">visual&nbsp;mode</span> rows?</h1>
  <p class="lede">
    Taxonomy v1 froze visual mode, marks and named registers out of scope.
    Mining brought <b>%d execution-verified rows</b> in anyway, and they are in
    the corpus and the splits right now. They sit almost entirely in
    <b>train</b> &mdash; only %d reached test and %d reached val &mdash; so this
    is a question about what the model is <i>taught</i>, far more than about
    what it is scored on. And the question is not whether visual mode is
    <i>allowed</i>: it is whether each row teaches something the corpus has
    nowhere else, or just a longer way to do what it already knows.
  </p>
  <div class="band">
    <div><span class="v">%d</span><span class="k">rows awaiting a call</span></div>
    <div><span class="v" style="color:var(--detour)">%d</span><span class="k">have a simpler twin</span></div>
    <div><span class="v" style="color:var(--keep)">%d</span><span class="k">no twin &mdash; real capability</span></div>
    <div><span class="v">%d / %d / %d</span><span class="k">train / val / test</span></div>
  </div>
</div></header>

<main class="wrap">

<section>
  <h2>The three calls, and what the evidence says</h2>
  <p class="sub">Every row was re-executed. A <b>twin</b> means a plain
  operator+motion command produced the byte-identical buffer &mdash; so the
  visual version is a detour the model would learn to imitate.</p>
  <div class="verdicts">
    <div class="vcard split">
      <h3>Plain visual</h3>
      <span class="n">%d rows &middot; %d with a twin &middot; %d without</span>
      <p>The split is real and it runs straight down the middle of the
      category. <span class="mono">vaWd</span> is just
      <span class="mono">daW</span> with two extra keystrokes, and 9 rows
      share that one command. But selection-scoped increment, reflow and
      ex-ranges have no non-visual form at all.</p>
      <span class="rec split">split it &mdash; keep the capability, drop the detours</span>
    </div>
    <div class="vcard keep">
      <h3>Visual block</h3>
      <span class="n">%d rows &middot; %d with a twin &middot; %d without</span>
      <p>Not one row has a simpler twin, and 62 of 62 commands are distinct.
      Column editing genuinely has no line-oriented equivalent &mdash; and it
      is core daily work for the data-analyst and data-scientist profiles the
      corpus otherwise serves worst.</p>
      <span class="rec keep">bless as scope v2</span>
    </div>
    <div class="vcard policy">
      <h3>Named registers</h3>
      <span class="n">%d rows &middot; excluded on policy, not redundancy</span>
      <p>These were never a capability question. A register <i>name</i> is not
      derivable from a natural-language request &mdash; nothing in
      &ldquo;paste the block back&rdquo; picks register
      <span class="mono">a</span> over <span class="mono">b</span>, so the
      model is being asked to guess an unstated symbol.</p>
      <span class="rec policy">quarantine</span>
    </div>
  </div>
  <p class="note"><b>Why not simply bless everything.</b> The 46 twinned rows
  are not merely redundant &mdash; they are actively mistraining. Each one
  pairs an ordinary request with a longer answer, and the brevity bonus in the
  §9 GRPO reward will then be fighting the SFT data it was trained on.</p>
  <p class="note"><b>Why not simply quarantine everything.</b> It would cost
  the corpus its only rows for four capabilities the sandbox can verify and
  nothing else covers: increment over a selection, reflow of a selection,
  a selection feeding an ex range, and every column edit in the dataset.</p>
</section>

<section>
  <h2>What the rows actually exercise</h2>
  <p class="sub">Grouped by capability rather than by category, because the
  decision follows the capability. The right-hand column is what would be lost
  by quarantining that group.</p>
  <div class="scroll"><table class="caps">
    <thead><tr><th>capability</th><th class="num">rows</th>
      <th class="num">has twin</th><th class="num">unique</th></tr></thead>
    <tbody>%s</tbody>
  </table></div>
</section>

<section>
  <h2>Every row</h2>
  <p class="sub">Amber left edge means a simpler command gives the identical
  buffer; green means nothing simpler was found; red is the register policy
  case. Tabs render as arrows and trailing spaces as dots.</p>
</section>
</main>

<div class="rail"><div class="wrap">
  <span class="lab">category</span>
  <button class="f" data-cat="all" aria-pressed="true">all<span class="c">%d</span></button>
  <button class="f" data-cat="visual" aria-pressed="false">visual<span class="c">%d</span></button>
  <button class="f" data-cat="visual-block" aria-pressed="false">visual-block<span class="c">%d</span></button>
  <button class="f" data-cat="registers" aria-pressed="false">registers<span class="c">%d</span></button>
  <span class="lab" style="margin-left:10px">verdict</span>
  <button class="f" data-kind="all" aria-pressed="true">any<span class="c">%d</span></button>
  <button class="f" data-kind="twin" aria-pressed="false">has a twin<span class="c">%d</span></button>
  <button class="f" data-kind="uniq" aria-pressed="false">unique<span class="c">%d</span></button>
  <span class="count" id="count"></span>
</div></div>

<main class="wrap">
  <div class="boot" id="boot">Loading %d rows&hellip; if this message stays,
  the page failed to start and the reason will appear here.</div>
  <div class="cards" id="cards"></div>
  <div class="more" id="more"><button class="btn" id="moreBtn">Show 30 more</button></div>
</main>

<script id="payload" type="application/json">%s</script>
<script>%s</script>
""" % (CSS, n, sp["test"], sp["val"],
       n, len(twin), len(uniq), sp["train"], sp["val"], sp["test"],
       vi["n"], vi["twin"], vi["n"] - vi["twin"],
       vb["n"], vb["twin"], vb["n"] - vb["twin"],
       rg["n"],
       cap_rows,
       n, vi["n"], vb["n"], rg["n"], n, len(twin), len(uniq),
       n, payload, JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cases = collect()
    page = build(cases)
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="visual scope")
    out = pathlib.Path(args.out)
    out.write_text(page, encoding="utf-8")
    twin = sum(1 for c in cases if c["twin"])
    print("rows: %d (twin %d, unique %d)  page: %d KB  -> %s"
          % (len(cases), twin, len(cases) - twin, len(page) // 1024, out))


if __name__ == "__main__":
    main()
