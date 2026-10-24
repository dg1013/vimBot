#!/usr/bin/env python3
"""Build the error-analysis page for a baseline run (CLAUDE.md §9).

Every test row with what the model answered and which of the four states it
landed in -- parse failure, invalid, valid-but-wrong, exec-match -- and the
three buffers side by side: the input, the expected output, and what the
model's command actually produced (or the vim error that stopped it). Filters
by state, category, gold mode and vim error code; a stacked bar per category
so the failure clusters are visible before scrolling; a table view of the
same numbers.

State comes from the run's exec_<split>.jsonl -- the execution records the
scoring pass already wrote -- and falls back to re-running a prediction
through the sandbox for any row the log does not cover (an older run, or a
prediction file edited since). Either way it is the same run_vim call
src/evaluate.py scores with, so the page and RESULTS.md cannot disagree.

Usage:
    python scripts/build_baseline_analysis.py --run runs/qwen3.5-4b --out page.html
"""
import argparse
import collections
import html
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import _norm, load_jsonl, vim_error_code    # noqa: E402
from executor import run_vim               # noqa: E402
from page_guard import check_page          # noqa: E402

STATES = ("match", "wrong", "invalid", "parse")
STATE_LABEL = {"match": "exec-match", "wrong": "valid, wrong buffer",
               "invalid": "invalid", "parse": "parse failure"}
STATE_GLYPH = {"match": "✓", "wrong": "≠", "invalid": "✗", "parse": "∅"}

def collect(run_dir, cases_path, timeout):
    cases = load_jsonl(cases_path)
    split = pathlib.Path(cases_path).stem
    recs = {r["id"]: r for r in load_jsonl(str(run_dir / ("completions_%s.jsonl" % split)))}
    summary = json.loads((run_dir / ("baseline_%s.json" % split)).read_text(encoding="utf-8"))
    exec_path = run_dir / ("exec_%s.jsonl" % split)
    logged = {r["id"]: r for r in load_jsonl(str(exec_path))} if exec_path.exists() else {}
    reran = from_log = 0
    out = []
    for c in cases:
        r = recs.get(c["id"])
        cur = tuple(c.get("cursor") or (1, 1))
        got, err, timed_out, code = None, "", False, ""
        if r and r["parse_ok"]:
            e = logged.get(c["id"])
            # Only trust a log row that is about this exact prediction, so an
            # edited preds file re-executes instead of rendering stale output.
            if e is not None and e["command"] == r["command"] and e["mode"] == r["mode"]:
                got, err, timed_out = e["buffer_after"], e["stderr"], e["timeout"]
                # `valid` in the log is folded with match (src/evaluate.py);
                # the state ladder below tests match first, so it is the same
                # answer here without re-running vim.
                valid, code = e["valid"], e["error"]
                from_log += 1
            else:
                res = run_vim(c["buffer_before"], r["command"], mode=r["mode"],
                              cursor=cur, timeout=timeout)
                reran += 1
                got = res.get("buffer_after")
                valid = bool(res.get("was_valid"))
                err = (res.get("stderr") or "").strip()
                timed_out = bool(res.get("timeout"))
                code = vim_error_code(err, timed_out)
        else:
            valid = False
        match = got is not None and _norm(got) == _norm(c["buffer_after"])
        if not r or not r["parse_ok"]:
            state = "parse"
        elif match:
            state = "match"
        elif valid:
            state = "wrong"
        else:
            state = "invalid"
        exp, gotn = _norm(c["buffer_after"]) or [], _norm(got) if got is not None else None
        out.append({
            "id": c["id"], "cat": c["category"], "tier": c.get("tier") or "T?",
            "gmode": c.get("mode", "ex"), "ins": c.get("instruction", ""),
            "gold": c["gold_command"], "bef": c["buffer_before"], "exp": c["buffer_after"],
            "mmode": r["mode"] if r else None, "cmd": r["command"] if r else None,
            "raw": r["completion"] if r else "", "tok": r["output_tokens"] if r else 0,
            "lat": r["latency_s"] if r else 0, "state": state, "err": code,
            "stderr": err[-400:], "got": got,
            "same": bool(r and r["command"] == c["gold_command"] and r["mode"] == c.get("mode", "ex")),
            "dlines": (sum(1 for a, b in zip(exp, gotn) if a != b) + abs(len(exp) - len(gotn))
                       if gotn is not None else None),
        })
    order = {"match": 0, "wrong": 1, "invalid": 2, "parse": 3}
    out.sort(key=lambda x: (order[x["state"]], x["cat"], x["id"]))
    print("execution state: %d rows from %s, %d re-run through vim"
          % (from_log, exec_path.name if logged else "(no exec log)", reran))
    return out, summary


CSS = r"""
/* Status colours are re-stepped from the family's tokens to pass the dataviz
   six checks on each surface (lightness band, chroma floor, CVD and
   normal-vision separation, contrast) — validated 2026-09-05 with a Python
   port of the skill's validator. Tints stay the family's. */
:root{
  --ground:#F2F4F6; --surface:#FBFCFD; --sunk:#E6EAEE; --raise:#DFE5EA;
  --ink:#161B22; --ink2:#39424D; --muted:#66717E; --faint:#98A2AE;
  --line:#D3DAE1; --line2:#E3E8ED;
  --accent:#0F5F66; --accent-soft:#DCEDEE;
  --ok:#006B41; --ok-bg:#DDEDE3; --warn:#AD721B; --warn-bg:#F4E9D5;
  --stop:#9F2234; --stop-bg:#F4E0E2; --none:#66717E; --none-bg:#E6EAEE;
  --cut:#98333C; --cut-bg:#F7E7E9; --add:#2C6A4A; --add-bg:#E1EFE7;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","DejaVu Sans Mono",Menlo,monospace;
  --body:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --disp:"Newsreader","Iowan Old Style",Georgia,serif;
}
:root:not([data-theme="light"]){ color-scheme:light dark; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
    --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
    --line:#2B333C; --line2:#222A32; --accent:#6FC3C8; --accent-soft:#123134;
    --ok:#24A96F; --ok-bg:#15271D; --warn:#C48800; --warn-bg:#2A2215;
    --stop:#DF5F72; --stop-bg:#2B1A1D; --none:#8492A0; --none-bg:#222A32;
    --cut:#DE8790; --cut-bg:#2C1B1E; --add:#6BBF92; --add-bg:#152720;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
  --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
  --line:#2B333C; --line2:#222A32; --accent:#6FC3C8; --accent-soft:#123134;
  --ok:#24A96F; --ok-bg:#15271D; --warn:#C48800; --warn-bg:#2A2215;
  --stop:#DF5F72; --stop-bg:#2B1A1D; --none:#8492A0; --none-bg:#222A32;
  --cut:#DE8790; --cut-bg:#2C1B1E; --add:#6BBF92; --add-bg:#152720;
}
*{ box-sizing:border-box; }
html,body{ height:100%; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--body);
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased;
  display:flex; flex-direction:column; overflow:hidden; }
header,.rail{ flex:0 0 auto; }
main{ flex:1 1 auto; min-height:0; overflow-y:auto; overflow-x:hidden; padding:20px 0 40px; }
h1,h2{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.012em; text-wrap:balance; }
.wrap{ max-width:1120px; margin:0 auto; padding:0 22px; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); font-weight:700; }
.mono{ font-family:var(--mono); }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
/* The header holds a chart with one row per category, so on a laptop-height
   frame it can outgrow the viewport; with the page pinned that would leave
   the card list no height at all (found 2026-09-05 -- "I don't see the
   rows"). Two guards: the explanation and chart collapse, and the header is
   capped and scrolls within itself, so the list always has >= 42vh. */
header{ background:var(--surface); border-bottom:1px solid var(--line);
  max-height:58vh; overflow-y:auto; }
header .wrap{ padding:16px 22px 12px; }
h1{ font-size:clamp(24px,3.4vw,32px); margin:6px 0 8px; display:inline; }
/* Each part of the top is its own section, collapsible on its own; the
   header also carries a native drag handle (resize) so it can be shrunk by
   hand. Section states are remembered per browser. */
header{ resize:vertical; min-height:64px; }
.topbar{ display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
.topbar .ctl{ margin-left:auto; display:flex; gap:6px; }
details.sec{ border-top:1px solid var(--line2); margin-top:8px; padding-top:6px; }
details.sec > summary{ list-style:none; cursor:pointer; user-select:none; display:flex; align-items:center; gap:8px;
  font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:700;
  padding:3px 6px; margin:0 -6px; border-radius:5px; }
details.sec > summary::-webkit-details-marker{ display:none; }
details.sec > summary:hover{ background:var(--sunk); color:var(--accent); }
details.sec > summary::before{ content:"\25BE"; width:1em; display:inline-block; }
details.sec:not([open]) > summary::before{ content:"\25B8"; }
details.sec > summary .n{ font-weight:400; letter-spacing:.04em; text-transform:none; opacity:.8; }
.grip{ text-align:center; font-family:var(--mono); font-size:9.5px; color:var(--faint); letter-spacing:.1em; padding:4px 0 0; user-select:none; }
.boot{ padding:34px 0; text-align:center; color:var(--muted); font-family:var(--mono); font-size:13px; }
.lede{ max-width:70ch; color:var(--ink2); font-size:15.5px; }
.lede b{ color:var(--ink); font-weight:600; }
.band{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:9px; overflow:hidden; margin-top:16px; }
.band div{ background:var(--surface); padding:11px 14px; }
.band .v{ font-family:var(--mono); font-size:21px; font-weight:700; display:block; font-variant-numeric:tabular-nums; }
.band .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); margin-top:3px; display:block; }
/* the chart: one row per category, one scale (0..n of that category), 2px surface gaps */
.chart{ margin-top:16px; display:grid; grid-template-columns:auto 1fr auto; gap:5px 10px; align-items:center; }
.chart .cl{ font-family:var(--mono); font-size:11.5px; color:var(--ink2); text-align:right; white-space:nowrap; }
.chart .cn{ font-family:var(--mono); font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap; }
.bar{ display:flex; height:14px; gap:2px; }
.seg{ height:100%; min-width:0; border-radius:0; cursor:default; }
.seg:first-child{ border-radius:3px 0 0 3px; } .seg:last-child{ border-radius:0 3px 3px 0; }
.seg.match{ background:var(--ok); } .seg.wrong{ background:var(--warn); }
.seg.invalid{ background:var(--stop); } .seg.parse{ background:var(--none); }
.legend{ display:flex; flex-wrap:wrap; gap:6px 16px; margin-top:10px; font-family:var(--mono); font-size:11.5px; color:var(--ink2); align-items:center; }
.legend i{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.btn{ font-family:var(--mono); font-size:11.5px; padding:4px 9px; border-radius:5px; cursor:pointer;
  background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.btn:hover{ border-color:var(--accent); color:var(--accent); }
table.tv{ border-collapse:collapse; margin-top:10px; font-family:var(--mono); font-size:11.5px; font-variant-numeric:tabular-nums; }
table.tv th,table.tv td{ padding:3px 10px; border-bottom:1px solid var(--line2); text-align:right; }
table.tv th:first-child,table.tv td:first-child{ text-align:left; }
#tip{ position:fixed; pointer-events:none; background:var(--ink); color:var(--ground); font-family:var(--mono);
  font-size:11.5px; padding:5px 8px; border-radius:5px; opacity:0; transition:opacity .08s; z-index:50; white-space:nowrap; }
.rail{ z-index:20; background:var(--surface); border-bottom:1px solid var(--line); }
.rail .wrap{ display:flex; flex-wrap:wrap; gap:6px 8px; align-items:center; padding:9px 22px; }
.f{ font-family:var(--mono); font-size:11.5px; padding:4px 9px; border-radius:5px; cursor:pointer;
  background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.f:hover{ border-color:var(--accent); color:var(--accent); }
.f[aria-pressed="true"]{ background:var(--accent-soft); border-color:var(--accent); color:var(--accent); font-weight:700; }
.f .c{ opacity:.6; margin-left:5px; }
.count{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto; font-variant-numeric:tabular-nums; }
.cards{ display:flex; flex-direction:column; gap:12px; }
.card{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--line); border-radius:8px; overflow:hidden; }
.card.s-match{ border-left-color:var(--ok); } .card.s-wrong{ border-left-color:var(--warn); }
.card.s-invalid{ border-left-color:var(--stop); } .card.s-parse{ border-left-color:var(--none); }
.chead{ display:flex; flex-wrap:wrap; gap:6px 9px; align-items:center; padding:9px 14px; border-bottom:1px solid var(--line2); }
.cid{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:4px; background:var(--raise); color:var(--ink2); white-space:nowrap; }
.chip.st{ font-weight:700; border:1px solid; }
.chip.st.match{ background:var(--ok-bg); color:var(--ok); border-color:var(--ok); }
.chip.st.wrong{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn); }
.chip.st.invalid{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop); }
.chip.st.parse{ background:var(--none-bg); color:var(--none); border-color:var(--none); }
.spacer{ flex:1 1 auto; }
.ins{ padding:12px 15px 6px; font-size:16px; color:var(--ink); max-width:72ch; font-family:var(--disp); line-height:1.45; }
.cmds{ padding:0 15px 10px; display:grid; grid-template-columns:auto 1fr; gap:3px 10px; font-family:var(--mono); font-size:12.5px; align-items:baseline; }
.cmds .k{ font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); }
.cmds .g{ color:var(--ink2); } .cmds .m{ color:var(--ink); font-weight:700; overflow-wrap:anywhere; }
.cmds .raw{ color:var(--muted); white-space:pre-wrap; overflow-wrap:anywhere; }
.same{ color:var(--ok); font-weight:700; }
.panes{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:1px; background:var(--line2); border-top:1px solid var(--line2); }
.pane{ background:var(--surface); padding:8px 12px; min-width:0; overflow-x:auto; }
.pane .lab{ margin-bottom:4px; display:block; }
.ln{ display:flex; white-space:pre; font-family:var(--mono); font-size:12px; line-height:1.5; }
.ln .g{ color:var(--faint); opacity:.55; user-select:none; text-align:right; min-width:2em; padding-right:8px; flex:0 0 auto; font-variant-numeric:tabular-nums; }
.ln.cut{ background:var(--cut-bg); } .ln.cut .t{ color:var(--cut); }
.ln.add{ background:var(--add-bg); } .ln.add .t{ color:var(--add); }
.errbox{ font-family:var(--mono); font-size:11.5px; color:var(--stop); background:var(--stop-bg); border:1px solid var(--stop); border-radius:5px; padding:6px 9px; white-space:pre-wrap; overflow-wrap:anywhere; }
.empty{ padding:40px 0; text-align:center; color:var(--muted); font-family:var(--mono); font-size:13px; }
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; } }
"""

JS = r"""
window.onerror = function(msg, src, line){
  const b = document.getElementById('boot');
  if (b) b.textContent = 'The page failed to start: ' + msg + (line ? ' (line ' + line + ')' : '');
};
const DATA = JSON.parse(document.getElementById('payload').textContent);
const LABEL = {match:'exec-match', wrong:'valid, wrong buffer', invalid:'invalid', parse:'parse failure'};
const GLYPH = {match:'✓', wrong:'≠', invalid:'✗', parse:'∅'};
let fState='all', fCat='all', fMode='all', fErr='all';
function esc(s){ return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function lines(t){ if(t==null) return null; const a=t.split('\n'); if(a.length && a[a.length-1]==='') a.pop(); return a; }
function pane(label, text, other, cls){
  const L=lines(text)||[], O=new Set(lines(other)||[]);
  let h='<div class="pane"><span class="lab">'+label+'</span>';
  if(!L.length) h+='<div class="ln"><span class="g">1</span><span class="t" style="color:var(--faint)">(empty buffer)</span></div>';
  L.forEach(function(l,i){ const hit=!O.has(l); h+='<div class="ln'+(hit?' '+cls:'')+'"><span class="g">'+(i+1)+'</span><span class="t">'+(esc(l)||' ')+'</span></div>'; });
  return h+'</div>';
}
function card(c){
  const st=c.state;
  let third;
  if(st==='parse') third='<div class="pane"><span class="lab">what the model produced</span><div class="errbox">not in the required form; nothing was run</div></div>';
  else if(st==='invalid') third='<div class="pane"><span class="lab">what vim said</span><div class="errbox">'+esc(c.stderr||(c.err||'error, no message'))+'</div></div>';
  else third=pane('what the model’s command produced', c.got, c.exp, 'add');
  return '<div class="card s-'+st+'" id="c-'+esc(c.id)+'">'
    +'<div class="chead"><span class="cid">'+esc(c.id)+'</span>'
    +'<span class="chip st '+st+'">'+GLYPH[st]+' '+LABEL[st]+'</span>'
    +'<span class="chip">'+esc(c.cat)+'</span><span class="chip">'+esc(c.tier)+'</span>'
    +'<span class="chip">gold '+esc(c.gmode)+(c.mmode?' → model '+esc(c.mmode):'')+'</span>'
    +(c.err?'<span class="chip">'+esc(c.err)+'</span>':'')
    +(c.dlines!=null&&st==='wrong'?'<span class="chip">'+c.dlines+' line'+(c.dlines===1?'':'s')+' off</span>':'')
    +'<span class="spacer"></span><span class="chip">'+c.tok+' tok · '+c.lat.toFixed(2)+' s</span></div>'
    +'<div class="ins">'+esc(c.ins)+'</div>'
    +'<div class="cmds"><span class="k">gold</span><span class="g">'+esc(c.gmode)+': '+esc(c.gold)+'</span>'
    +'<span class="k">model</span><span class="m">'+(c.cmd!=null?esc(c.mmode)+': '+esc(c.cmd):'(unparsed)')+(c.same?' <span class="same">= gold</span>':'')+'</span>'
    +((c.cmd==null||(c.raw||'').trim()!==(c.mmode+': '+c.cmd))?'<span class="k">raw</span><span class="raw">'+esc(c.raw)+'</span>':'')+'</div>'
    +'<div class="panes">'+pane('input buffer', c.bef, c.exp, 'cut')+pane('expected output', c.exp, c.got, 'cut')+third+'</div></div>';
}
function keep(c){
  return (fState==='all'||c.state===fState)&&(fCat==='all'||c.cat===fCat)&&(fMode==='all'||c.gmode===fMode)&&(fErr==='all'||c.err===fErr);
}
function render(){
  const rows=DATA.filter(keep);
  document.getElementById('cards').innerHTML = rows.length ? rows.map(card).join('') : '<div class="empty">nothing matches these filters</div>';
  document.getElementById('count').textContent = rows.length+' of '+DATA.length+' rows';
  document.getElementById('boot').style.display = 'none';
  document.getElementById('scroller').scrollTop=0;
}
document.querySelector('.rail').addEventListener('click', function(ev){
  const b=ev.target.closest('button.f'); if(!b) return;
  const grp=b.dataset.group, val=b.dataset.value;
  document.querySelectorAll('button.f[data-group="'+grp+'"]').forEach(function(x){ x.setAttribute('aria-pressed', x===b?'true':'false'); });
  if(grp==='state') fState=val; else if(grp==='cat') fCat=val; else if(grp==='mode') fMode=val; else fErr=val;
  render();
});
const tip=document.getElementById('tip');
document.querySelectorAll('.seg').forEach(function(s){
  s.addEventListener('mousemove', function(ev){ tip.textContent=s.dataset.tip; tip.style.left=(ev.clientX+12)+'px'; tip.style.top=(ev.clientY+12)+'px'; tip.style.opacity=1; });
  s.addEventListener('mouseleave', function(){ tip.style.opacity=0; });
  s.addEventListener('click', function(){
    const cat=s.dataset.cat, st=s.dataset.state;
    document.querySelector('button.f[data-group="cat"][data-value="'+cat+'"]').click();
    document.querySelector('button.f[data-group="state"][data-value="'+st+'"]').click();
  });
});
const SKEY='vimbot-analysis-sections-v1';
const secs=Array.from(document.querySelectorAll('details.sec'));
try{ const saved=JSON.parse(localStorage.getItem(SKEY)||'{}'); secs.forEach(function(d){ if(d.id in saved) d.open=!!saved[d.id]; }); }catch(e){}
function saveSecs(){ try{ const o={}; secs.forEach(function(d){ o[d.id]=d.open; }); localStorage.setItem(SKEY, JSON.stringify(o)); }catch(e){} }
secs.forEach(function(d){ d.addEventListener('toggle', saveSecs); });
document.getElementById('collapse-all').addEventListener('click', function(){ secs.forEach(function(d){ d.open=false; }); saveSecs(); });
document.getElementById('expand-all').addEventListener('click', function(){ secs.forEach(function(d){ d.open=true; }); saveSecs(); });
render();
"""


def build(cases, summary):
    n = len(cases)
    st = collections.Counter(c["state"] for c in cases)
    m = summary["metrics"]
    cats = sorted({c["cat"] for c in cases})
    bycat = {k: collections.Counter() for k in cats}
    for c in cases:
        bycat[c["cat"]][c["state"]] += 1
    errs = collections.Counter(c["err"] for c in cases if c["err"])
    modes = collections.Counter(c["mmode"] for c in cases if c["mmode"])
    gmodes = collections.Counter(c["gmode"] for c in cases)
    keys_match = sum(1 for c in cases if c["gmode"] == "keys" and c["state"] == "match")

    rows = []
    for cat in sorted(cats, key=lambda k: (-bycat[k]["match"] / max(1, sum(bycat[k].values())), k)):
        tot = sum(bycat[cat].values())
        segs = "".join(
            '<div class="seg %s" style="flex:%d 0 0" data-cat="%s" data-state="%s" data-tip="%s: %s %d of %d"></div>'
            % (s, bycat[cat][s], html.escape(cat), s, html.escape(cat), STATE_LABEL[s], bycat[cat][s], tot)
            for s in STATES if bycat[cat][s])
        rows.append('<div class="cl">%s</div><div class="bar">%s</div><div class="cn">%d✓ %d≠ %d✗ / %d</div>'
                    % (html.escape(cat), segs, bycat[cat]["match"], bycat[cat]["wrong"], bycat[cat]["invalid"], tot))
    table = "".join("<tr><td>%s</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%d</td><td>%.0f%%</td></tr>"
                    % (html.escape(cat), sum(bycat[cat].values()), bycat[cat]["match"], bycat[cat]["wrong"],
                       bycat[cat]["invalid"], bycat[cat]["parse"],
                       100 * bycat[cat]["match"] / max(1, sum(bycat[cat].values()))) for cat in cats)

    def fbtn(group, value, label, count=None, on=False):
        return ('<button class="f" data-group="%s" data-value="%s" aria-pressed="%s">%s%s</button>'
                % (group, html.escape(str(value)), "true" if on else "false", html.escape(label),
                   ('<span class="c">%d</span>' % count) if count is not None else ""))
    state_btns = fbtn("state", "all", "all", n, True) + "".join(
        fbtn("state", s, STATE_GLYPH[s] + " " + STATE_LABEL[s], st[s]) for s in STATES if st[s])
    cat_btns = fbtn("cat", "all", "all", None, True) + "".join(fbtn("cat", c, c, sum(bycat[c].values())) for c in cats)
    mode_btns = fbtn("mode", "all", "any", None, True) + "".join(fbtn("mode", k, "gold " + k, gmodes[k]) for k in ("ex", "keys"))
    err_btns = fbtn("err", "all", "any", None, True) + "".join(fbtn("err", e, e, k) for e, k in errs.most_common(8))

    return """<title>Zero-shot Error Analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%(css)s</style>
<div id="tip" aria-hidden="true"></div>
<header><div class="wrap">
  <div class="lab">vimBot &middot; %(model)s zero-shot &middot; %(split)s slice &middot; error analysis (&sect;9)</div>
  <div class="topbar"><h1>Where does the base model go wrong?</h1>
    <span class="ctl"><button class="btn" id="collapse-all">collapse all</button><button class="btn" id="expand-all">expand all</button></span></div>
  <details class="sec" id="s-reading" open><summary>the reading <span class="n">&middot; what the numbers mean</span></summary>
  <p class="lede">
    Every row of the %(n)d-case %(split)s slice with what <b>%(hf)s</b> answered
    (greedy, non-thinking) and which of four states it landed in. <b>Invalid</b>
    means vim refused the command; <b>valid, wrong buffer</b> means vim ran it and
    the file came out different; <b>exec-match</b> means the file came out right,
    whatever the route. Each card shows the input, the expected output, and what
    the model&rsquo;s command produced, with lines that differ marked; an invalid
    row shows vim&rsquo;s error instead. Click a bar segment to filter to it.
  </p>
  <p class="lede" style="margin-top:8px">
    The model chose ex mode on <b>%(mex)d of %(n)d</b> rows and keys mode on
    %(mkeys)d. That costs less than it might: %(keysmatch)d of the %(gkeys)d
    keys-gold rows were still solved through an ex equivalent, a better rate
    than the %(exmatch)d of %(gex)d ex-gold rows. The failure mass is pattern
    construction &mdash; <b>%(e486)d</b> invalid rows are E486 <i>pattern not
    found</i>, the model guessing text that is not in the buffer, and
    <b>%(e488)d</b> are E488 <i>trailing characters</i>, a malformed substitute
    &mdash; and it shows in the categories that live on regexes: global
    %(gmatch)d of %(gn)d, substitute %(smatch)d of %(sn)d.
  </p>
  </details>
  <details class="sec" id="s-numbers" open><summary>numbers <span class="n">&middot; %(n)d rows</span></summary>
  <div class="band" style="margin-top:4px">
    <div><span class="v">%(n)d</span><span class="k">rows</span></div>
    <div><span class="v" style="color:var(--ok)">%(match)d</span><span class="k">&#10003; exec-match &middot; %(matchpct).1f%%</span></div>
    <div><span class="v" style="color:var(--warn)">%(wrong)d</span><span class="k">&#8800; valid, wrong buffer</span></div>
    <div><span class="v" style="color:var(--stop)">%(invalid)d</span><span class="k">&#10007; invalid</span></div>
    <div><span class="v">%(parse)d</span><span class="k">&#8709; parse failure</span></div>
    <div><span class="v">%(tok).1f</span><span class="k">output tokens, mean</span></div>
    <div><span class="v">%(lat).2f s</span><span class="k">latency p50</span></div>
  </div>
  </details>
  <details class="sec" id="s-chart" open><summary>per category <span class="n">&middot; click a segment to filter</span></summary>
  <div class="chart" role="img" aria-label="Per category: exec-match, valid but wrong, invalid, parse failure">%(rows)s</div>
  <div class="legend">
    <span><i style="background:var(--ok)"></i>&#10003; exec-match</span>
    <span><i style="background:var(--warn)"></i>&#8800; valid, wrong buffer</span>
    <span><i style="background:var(--stop)"></i>&#10007; invalid</span>
    <span><i style="background:var(--none)"></i>&#8709; parse failure</span>
  </div>
  </details>
  <details class="sec" id="s-table"><summary>table <span class="n">&middot; the same numbers as text</span></summary>
  <table class="tv" id="tv"><tr><th>category</th><th>n</th><th>&#10003;</th><th>&#8800;</th><th>&#10007;</th><th>&#8709;</th><th>match %%</th></tr>%(table)s</table>
  </details>
  <div class="grip">&#8801; drag the bottom-right corner to shrink the top &middot; sections remember their state</div>
</div></header>

<div class="rail"><div class="wrap">
  <span class="lab">state</span>%(state_btns)s
  <span class="lab" style="margin-left:8px">category</span>%(cat_btns)s
  <span class="lab" style="margin-left:8px">gold mode</span>%(mode_btns)s
  <span class="lab" style="margin-left:8px">vim error</span>%(err_btns)s
  <span class="count" id="count"></span>
</div></div>

<main class="wrap" id="scroller">
  <div class="boot" id="boot">Loading %(n)d rows&hellip; if this message stays, the page failed to start and the reason will appear here.</div>
  <div class="cards" id="cards"></div>
</main>

<script id="payload" type="application/json">%(payload)s</script>
<script>%(js)s</script>
""" % {
        "css": CSS, "js": JS, "model": html.escape(summary["model"]), "hf": html.escape(summary["hf_id"]),
        "split": html.escape(summary["split"]), "n": n,
        "match": st["match"], "matchpct": 100 * st["match"] / n, "wrong": st["wrong"],
        "invalid": st["invalid"], "parse": st["parse"],
        "tok": m["output_tokens_mean"], "lat": m["latency_s_p50"],
        "mex": modes.get("ex", 0), "mkeys": modes.get("keys", 0), "gkeys": gmodes.get("keys", 0),
        "keysmatch": keys_match, "gex": gmodes.get("ex", 0),
        "exmatch": sum(1 for c in cases if c["gmode"] == "ex" and c["state"] == "match"),
        "e486": errs.get("E486", 0), "e488": errs.get("E488", 0),
        "gmatch": bycat.get("global", collections.Counter())["match"], "gn": sum(bycat.get("global", collections.Counter()).values()),
        "smatch": bycat.get("substitute", collections.Counter())["match"], "sn": sum(bycat.get("substitute", collections.Counter()).values()),
        "rows": "".join(rows), "table": table,
        "state_btns": state_btns, "cat_btns": cat_btns, "mode_btns": mode_btns, "err_btns": err_btns,
        "payload": json.dumps(cases, ensure_ascii=False),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(REPO / "runs" / "qwen3.5-4b"))
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "test.jsonl"))
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cases, summary = collect(pathlib.Path(args.run), args.cases, args.timeout)
    page = build(cases, summary)
    check_page(page, required_js=["function render(", "function card("], name="baseline analysis")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    st = collections.Counter(c["state"] for c in cases)
    print("rows %d   match %d  wrong %d  invalid %d  parse %d   page %d KB -> %s"
          % (len(cases), st["match"], st["wrong"], st["invalid"], st["parse"], len(page) // 1024, args.out))
    errs = collections.Counter(c["err"] for c in cases if c["err"])
    print("top vim errors:", ", ".join("%s %d" % kv for kv in errs.most_common(6)))


if __name__ == "__main__":
    main()
