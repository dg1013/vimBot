#!/usr/bin/env python3
"""Build the Corpus Curator artifact: view, edit and delete rows across v3+v4.

The browser cannot run vim, so the page and scripts/apply_curation.py split
the work: the page browses every row, records edits and deletions, and
enforces what is checkable statically (the §6 instruction rules, the leak
guard, cursor bounds); the apply script re-executes each edited row in the
sandbox, recomputes buffer_after, and is the only path that touches the data.

Editing a v4 row promotes it to v3 — the page says so on every v4 card,
because it is a consequential side effect and should never be a surprise.

    conda run -n vimBot python scripts/build_curator.py --out page.html
"""
import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from page_guard import check_page  # noqa: E402
from build_corpus import DATASETS  # noqa: E402


def collect():
    rows = []
    for ds in sorted(DATASETS):
        p = REPO / "data" / ds / ("%s_cases.jsonl" % ds)
        if not p.exists():
            continue
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            cmd = r.get("gold_command") or ""
            rows.append({
                "id": r["id"],
                "ds": ds,
                "cat": r.get("category") or "—",
                "tier": r.get("tier") or "—",
                "mode": r.get("mode") or "keys",
                "src": r.get("source") or "—",
                "cur": r.get("cursor") or [1, 1],
                "cmd": cmd,
                "ins": (r.get("instruction") or "").strip(),
                "bef": r.get("buffer_before", ""),
                "aft": r.get("buffer_after", ""),
                "prov": r.get("instruction_provenance") or "—",
                "theme": r.get("theme") or "",
                "was": r.get("source_command") or "",
                "para": r.get("paraphrase_of") or "",
                "promo": r.get("promoted_from") or "",
            })
    return rows


TEMPLATE = r"""<title>Corpus Curator</title>
<style>
:root{
  --ground:#F2F0EE; --surface:#FCFBFA; --sunk:#E9E5E2; --raise:#E4DFDB;
  --ink:#231E22; --ink2:#4A4046; --muted:#6E6369; --faint:#9A8F95;
  --line:#DAD3CF; --line2:#E7E1DD;
  --accent:#6B2D5C; --accent2:#8A3E77; --accent-soft:#EFE2EC; --on-accent:#FCFBFA;
  --pass:#2E6E4E; --pass-bg:#DFEDE5; --pass-line:#AACFBB;
  --warn:#8C5A15; --warn-bg:#F4E9D8; --warn-line:#DFC79A;
  --stop:#9C3535; --stop-bg:#F3E1E0; --stop-line:#DFB4B2;
  --cut:#9C3535; --cut-bg:#F6E6E5; --add:#2E6E4E; --add-bg:#E2EFE8;
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
    --accent:#C98FBA; --accent2:#DDA9CF; --accent-soft:#2B1F28; --on-accent:#1D191C;
    --pass:#6BC195; --pass-bg:#17281F; --pass-line:#2C4736;
    --warn:#D6A85C; --warn-bg:#2A2216; --warn-line:#493B20;
    --stop:#E08484; --stop-bg:#2C1B1B; --stop-line:#4A2A2A;
    --cut:#E08484; --cut-bg:#2E1C1C; --add:#6BC195; --add-bg:#152720;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#151215; --surface:#1D191C; --sunk:#120F12; --raise:#282227;
  --ink:#E7E0E4; --ink2:#C0B6BC; --muted:#8D8288; --faint:#665D63;
  --line:#332C31; --line2:#282227;
  --accent:#C98FBA; --accent2:#DDA9CF; --accent-soft:#2B1F28; --on-accent:#1D191C;
  --pass:#6BC195; --pass-bg:#17281F; --pass-line:#2C4736;
  --warn:#D6A85C; --warn-bg:#2A2216; --warn-line:#493B20;
  --stop:#E08484; --stop-bg:#2C1B1B; --stop-line:#4A2A2A;
  --cut:#E08484; --cut-bg:#2E1C1C; --add:#6BC195; --add-bg:#152720;
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--sans);
  font-size:15px; line-height:1.5; -webkit-font-smoothing:antialiased; }
h1,h2{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.01em; }
.mono{ font-family:var(--mono); }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
.wrap{ max-width:1200px; margin:0 auto; padding:0 20px; }
header{ background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:20px 20px 14px; }
h1{ font-size:clamp(24px,3.4vw,32px); margin:6px 0 8px; }
.lede{ max-width:74ch; color:var(--ink2); font-size:14.5px; }
.lede b{ color:var(--ink); }
.lede code{ font-family:var(--mono); font-size:.9em; background:var(--sunk);
  padding:1px 4px; border-radius:3px; }
.agg{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:7px;
  overflow:hidden; margin-top:14px; }
.agg div{ background:var(--surface); padding:9px 12px; }
.agg .v{ font-family:var(--mono); font-size:18px; font-weight:700; display:block;
  font-variant-numeric:tabular-nums; }
.agg .k{ font-family:var(--mono); font-size:9px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin-top:2px; display:block; }

.rail{ position:sticky; top:0; z-index:20; background:var(--surface);
  border-bottom:1px solid var(--line); }
.rail .wrap{ display:flex; flex-wrap:wrap; gap:7px 12px; align-items:center; padding:9px 20px; }
.f{ font-family:var(--mono); font-size:11.5px; padding:3px 9px; border-radius:4px; cursor:pointer;
  background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.f:hover{ border-color:var(--accent); color:var(--accent); }
.f[aria-pressed="true"]{ background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); font-weight:600; }
.f .c{ opacity:.6; margin-left:5px; font-variant-numeric:tabular-nums; }
select,input[type=search]{ font:inherit; font-size:12.5px; padding:4px 8px;
  border:1px solid var(--line); border-radius:4px; background:var(--ground);
  color:inherit; font-family:var(--mono); }
input[type=search]{ min-width:13rem; }
.count{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto;
  font-variant-numeric:tabular-nums; }
.btn{ font-family:var(--mono); font-size:11.5px; padding:4px 10px; border-radius:4px;
  cursor:pointer; background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.btn:hover{ border-color:var(--accent); color:var(--accent); }
.btn.pri{ background:var(--accent); border-color:var(--accent); color:var(--on-accent);
  font-weight:600; }
.btn.pri:hover{ background:var(--accent2); border-color:var(--accent2); }

main.wrap{ padding:18px 20px 150px; }
.cards{ display:flex; flex-direction:column; gap:10px; }
.card{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--line);
  border-radius:6px; overflow:hidden; }
.card.ed{ border-left-color:var(--accent); }
.card.del{ border-left-color:var(--stop); opacity:.62; }
.card.bad{ border-left-color:var(--stop); }
.chead{ display:flex; flex-wrap:wrap; gap:6px 9px; align-items:center; padding:8px 12px;
  border-bottom:1px solid var(--line2); }
.pill{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.05em;
  text-transform:uppercase; padding:2px 6px; border-radius:3px; white-space:nowrap;
  border:1px solid; }
.p-v3{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }
.p-v4{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.cid{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:3px;
  background:var(--raise); color:var(--ink2); white-space:nowrap; }
.spacer{ flex:1 1 auto; }
.keys{ font-family:var(--mono); font-size:12.5px; font-weight:600; background:var(--accent-soft);
  color:var(--accent); padding:2px 7px; border-radius:3px; overflow-wrap:anywhere; }
.body{ padding:9px 12px; }
.fieldrow{ display:flex; gap:8px; flex-wrap:wrap; }
.fieldrow>div{ flex:1 1 auto; min-width:0; }
.fieldrow>div.sm{ flex:0 0 84px; }
label.lab{ display:block; margin:7px 0 3px; }
textarea,input[type=text],input[type=number]{ width:100%; font:inherit; font-size:13px;
  padding:5px 8px; border:1px solid var(--line); border-radius:4px; background:var(--ground);
  color:var(--ink); }
textarea{ font-family:var(--mono); font-size:12.5px; resize:vertical; }
textarea.ins{ font-family:var(--sans); font-size:14px; font-weight:600; }
textarea:focus,input:focus{ outline:none; border-color:var(--accent);
  box-shadow:0 0 0 2px var(--accent-soft); }
.dirty{ border-color:var(--accent) !important; }
.flags{ display:flex; flex-wrap:wrap; gap:5px; margin-top:7px; }
.flag{ font-family:var(--mono); font-size:11px; padding:2px 7px; border-radius:3px;
  border:1px solid; }
.flag.err{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line);
  font-weight:600; }
.flag.warnf{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.flag.info{ background:var(--accent-soft); color:var(--accent); border-color:var(--accent); }
.promo{ margin-top:8px; padding:8px 10px; border-radius:5px; background:var(--warn-bg);
  border:1px solid var(--warn-line); color:var(--warn); font-size:12.5px; line-height:1.45; }
.promo b{ font-weight:700; }
.panes{ display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:1px;
  background:var(--line2); border-top:1px solid var(--line2); }
.pane{ background:var(--surface); padding:7px 11px; min-width:0; overflow-x:auto; }
.ln{ display:flex; white-space:pre; font-family:var(--mono); font-size:12px; line-height:1.5; }
.ln .g{ color:var(--faint); opacity:.6; user-select:none; text-align:right; min-width:2em;
  padding-right:8px; flex:0 0 auto; font-variant-numeric:tabular-nums; }
.ln.cut{ background:var(--cut-bg); } .ln.cut .t{ color:var(--cut); }
.ln.add{ background:var(--add-bg); } .ln.add .t{ color:var(--add); }
.ws{ color:var(--accent); opacity:.38; }
.stale{ font-family:var(--mono); font-size:11px; color:var(--warn); padding:6px 11px;
  background:var(--warn-bg); border-top:1px solid var(--warn-line); }
.cfoot{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; padding:7px 12px;
  border-top:1px solid var(--line2); }
.cfoot button{ font-family:var(--mono); font-size:11px; padding:3px 9px; border-radius:3px;
  cursor:pointer; background:none; border:1px solid var(--line); color:var(--muted); }
.cfoot button:hover{ border-color:var(--accent); color:var(--accent); }
.cfoot button.del[aria-pressed="true"]{ background:var(--stop-bg); color:var(--stop);
  border-color:var(--stop); font-weight:700; }
.prov{ font-family:var(--mono); font-size:10px; color:var(--faint); }
.more{ display:flex; justify-content:center; padding:16px 0 0; }
.empty{ padding:36px 0; text-align:center; color:var(--muted); font-family:var(--mono);
  font-size:13px; }
.dock{ position:fixed; left:0; right:0; bottom:0; z-index:30; background:var(--surface);
  border-top:1px solid var(--line); }
.dock .wrap{ display:flex; gap:10px; align-items:center; padding:9px 20px; flex-wrap:wrap; }
dialog{ max-height:92vh; overflow:auto; border:1px solid var(--line); border-radius:8px;
  background:var(--surface); color:var(--ink); max-width:min(900px,94vw); width:100%; padding:0; }
dialog::backdrop{ background:rgba(20,16,19,.55); }
dialog .dh{ padding:15px 18px 0; } dialog .db{ padding:12px 18px 16px;
  display:flex; flex-direction:column; gap:9px; }
dialog textarea{ min-height:240px; font-size:11.5px; }
dialog p.sub{ margin:0; color:var(--muted); font-size:12.5px; }
.cmdline{ font-family:var(--mono); font-size:11.5px; background:var(--sunk);
  border:1px solid var(--line); border-radius:4px; padding:7px 9px; color:var(--ink2);
  overflow-x:auto; white-space:pre; }
.boot{ padding:30px 0; text-align:center; color:var(--muted); font-family:var(--mono);
  font-size:13px; }
.boot.err{ color:var(--stop); text-align:left; white-space:pre-wrap; overflow-wrap:anywhere;
  background:var(--stop-bg); border:1px solid var(--stop-line); border-radius:6px; padding:14px; }
</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; datasets v3 + v4 &middot; curation</div>
  <h1>Corpus Curator</h1>
  <p class="lede">
    Every case in both datasets, editable. Change an instruction, a command, a buffer or a
    cursor; or mark a row for deletion. <b>Editing a v4 row promotes it to v3</b> &mdash; an
    edit is you standing behind the row, which is the bar v3 asks for, so the promotion is the
    phase-out running one reviewed row at a time. Nothing here writes to the corpus: export the
    change-set and <code>apply_curation.py</code> re-executes every edited row in the sandbox,
    recomputes its result buffer, and refuses anything that breaks.
  </p>
  <div class="agg" id="agg"></div>
</div></header>

<div class="rail"><div class="wrap" id="rail"></div></div>

<main class="wrap">
  <div class="boot" id="boot">Loading __TOTAL__ cases&hellip; if this message stays, the page
  failed to start and the reason will appear here.</div>
  <div class="cards" id="cards"></div>
  <div class="more"><button class="btn" id="more">Show 40 more</button></div>
</main>

<div class="dock"><div class="wrap">
  <button class="btn pri" id="export">Export change-set</button>
  <button class="btn" id="reset">Discard all changes</button>
  <span class="count" id="dockcount"></span>
</div></div>

<dialog id="dlg"><div class="dh"><h2 id="dlgtitle">Apply these changes</h2></div><div class="db">
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <button class="btn" id="copy">Copy JSON</button><span class="spacer"></span>
    <button class="btn" id="close">Close</button>
  </div>
  <textarea id="out" spellcheck="false" readonly></textarea>
  <div class="cmdline">conda run -n vimBot python scripts/apply_curation.py \
  --changes changes.json --dry-run</div>
  <p class="sub" id="dlgsub"></p>
</div></dialog>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const ROWS = JSON.parse(document.getElementById("payload").textContent);
const KEY = "vimbot.curator.v1";
const BY = new Map(ROWS.map(r => [r.id, r]));
let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
const st = id => (store[id] = store[id] || {});
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g,
  c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const FIELDS = ["ins", "cmd", "mode", "bef", "cur"];
function val(r, f) {
  const s = store[r.id];
  if (s && s[f] !== undefined) return s[f];
  return r[f];
}
function dirtyFields(r) {
  return FIELDS.filter(f => {
    const v = val(r, f);
    return f === "cur" ? JSON.stringify(v) !== JSON.stringify(r.cur) : v !== r[f];
  });
}
const isEdited = r => dirtyFields(r).length > 0;
const isDeleted = r => !!(store[r.id] && store[r.id].del);

/* ---- checks: mirrors of the gates apply_curation.py runs for real ---- */
function leakForms(cmd) {
  const out = new Set(); const f = (cmd || "").trim();
  if (f) { out.add(f); const s2 = f.replace(/<[^>]*>/g, ""); out.add(s2); out.add(s2.replace(/^:+/, "")); }
  return [...out].map(x => x.trim().toLowerCase()).filter(x => x.length >= 2);
}
function check(r) {
  const out = [];
  const ins = (val(r, "ins") || "").trim();
  const cmd = val(r, "cmd") || "";
  if (!ins) out.push(["err", "instruction is empty"]);
  else {
    const low = ins.toLowerCase();
    for (const f of leakForms(cmd))
      if (low.includes(f)) { out.push(["err", "leak: instruction contains " + JSON.stringify(f)]); break; }
    if (/\bpress\b|\bkeystroke|\btype the key/.test(low))
      out.push(["err", "leak: describes keystrokes, not intent"]);
    if (ins.endsWith(".")) out.push(["warnf", "trailing period"]);
    if (/\n/.test(ins)) out.push(["warnf", "more than one line"]);
    if (ins[0] !== ins[0].toUpperCase()) out.push(["warnf", "not sentence case"]);
  }
  if (!cmd.trim()) out.push(["err", "command is empty"]);
  const lines = (val(r, "bef") || "").split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop();
  const cur = val(r, "cur"), nl = Math.max(1, lines.length);
  if (cur[0] < 1 || cur[0] > nl) out.push(["err", "cursor line " + cur[0] + " outside the " + nl + "-line buffer"]);
  else {
    const len = (lines[cur[0] - 1] || "").length;
    if (cur[1] < 1 || cur[1] > len + 1)
      out.push(["err", "cursor col " + cur[1] + " outside line " + cur[0] + " (" + len + " chars)"]);
  }
  return out;
}
const hasErr = r => check(r).some(([k]) => k === "err");

/* ---- rendering ---- */
function vis(t) {
  return Array.from(t).map(ch => ch === " " ? '<span class="ws">&middot;</span>'
    : ch === "\t" ? '<span class="ws">&rarr;&nbsp;&nbsp;&nbsp;</span>' : esc(ch)).join("") || "&nbsp;";
}
function pane(text, cls, label, other) {
  const a = text.split("\n"); if (a.length && a[a.length - 1] === "") a.pop();
  const b = (other || "").split("\n"); if (b.length && b[b.length - 1] === "") b.pop();
  const oset = new Set(b);
  const body = a.map((l, i) => '<div class="ln' + (oset.has(l) ? "" : " " + cls) +
    '"><span class="g">' + (i + 1) + '</span><span class="t">' + vis(l) + "</span></div>").join("");
  return '<div class="pane"><span class="lab">' + label + "</span>" + (body || '<div class="ln"><span class="t">&nbsp;</span></div>') + "</div>";
}
function card(r) {
  const s = store[r.id] || {}, d = dirtyFields(r), del = isDeleted(r);
  const errs = check(r);
  const cls = del ? "del" : (errs.some(([k]) => k === "err") ? "bad" : (d.length ? "ed" : ""));
  const el = document.createElement("div");
  el.className = "card " + cls;
  el.dataset.case = r.id;
  const staleAfter = d.some(f => f === "cmd" || f === "bef" || f === "cur" || f === "mode");
  el.innerHTML =
    '<div class="chead">' +
      '<span class="pill p-' + r.ds + '">' + r.ds + "</span>" +
      '<span class="cid">' + esc(r.id) + "</span>" +
      '<span class="chip">' + esc(r.cat) + '</span><span class="chip">' + esc(r.tier) + "</span>" +
      '<span class="chip">' + esc(r.src) + "</span>" +
      (r.theme ? '<span class="chip">golf ' + esc(r.theme) + "</span>" : "") +
      (r.promo ? '<span class="chip">promoted from ' + esc(r.promo) + "</span>" : "") +
      (r.para ? '<span class="chip">paraphrase of ' + esc(r.para) + "</span>" : "") +
      '<span class="spacer"></span><span class="keys">' + esc(val(r, "cmd")) + "</span>" +
    "</div>" +
    '<div class="body">' +
      '<label class="lab">Instruction</label>' +
      '<textarea class="ins f-ins" rows="2">' + esc(val(r, "ins")) + "</textarea>" +
      '<div class="fieldrow">' +
        '<div class="sm"><label class="lab">Mode</label><select class="f-mode">' +
          ['ex', 'keys'].map(m => '<option' + (val(r, "mode") === m ? " selected" : "") + ">" + m + "</option>").join("") +
        "</select></div>" +
        '<div><label class="lab">Command</label>' +
          '<input type="text" class="f-cmd mono" value="' + esc(val(r, "cmd")) + '"></div>' +
        '<div class="sm"><label class="lab">Cur line</label>' +
          '<input type="number" class="f-cl" min="1" value="' + val(r, "cur")[0] + '"></div>' +
        '<div class="sm"><label class="lab">Cur col</label>' +
          '<input type="number" class="f-cc" min="1" value="' + val(r, "cur")[1] + '"></div>' +
      "</div>" +
      '<label class="lab">Buffer before</label>' +
      '<textarea class="f-bef" rows="3">' + esc(val(r, "bef")) + "</textarea>" +
      '<div class="flags">' + errs.map(([k, m]) => '<span class="flag ' + k + '">' + esc(m) + "</span>").join("") +
        (d.length ? '<span class="flag info">changed: ' + d.join(", ") + "</span>" : "") + "</div>" +
      (d.length && r.ds === "v4"
        ? '<div class="promo"><b>Editing this promotes it to v3.</b> On apply it moves out of ' +
          'v4 into v3, keeps the id <span class="mono">' + esc(r.id) + '</span>, and records ' +
          'promoted_from v4. Delete instead if the row should not exist at all.</div>' : "") +
    "</div>" +
    (staleAfter ? '<div class="stale">Result buffer below is the stored one and is now out of ' +
      'date &mdash; vim recomputes it at apply time.</div>' : "") +
    '<div class="panes">' + pane(val(r, "bef"), "cut", "buffer before", r.aft) +
      pane(r.aft, "add", "buffer after (stored)", val(r, "bef")) + "</div>" +
    '<div class="cfoot">' +
      '<button class="del" aria-pressed="' + del + '">' + (del ? "marked for deletion" : "delete") + "</button>" +
      '<button class="rst">revert</button>' +
      '<span class="spacer"></span><span class="prov">' + esc(r.prov) +
      (r.was ? " · mined as " + esc(r.was) : "") + "</span>" +
    "</div>";
  wire(el, r);
  return el;
}
function wire(el, r) {
  const redraw = () => { const f = card(r); el.replaceWith(f); tally(); };
  const bind = (sel, field, get) => {
    const node = el.querySelector(sel);
    if (!node) return;
    node.addEventListener("change", () => {
      const v = get(node);
      const base = field === "cur" ? JSON.stringify(r.cur) : r[field];
      const now = field === "cur" ? JSON.stringify(v) : v;
      if (now === base) { delete st(r.id)[field]; } else { st(r.id)[field] = v; }
      save(); redraw();
    });
    node.addEventListener("input", () => {
      const v = get(node);
      const base = field === "cur" ? JSON.stringify(r.cur) : r[field];
      const now = field === "cur" ? JSON.stringify(v) : v;
      if (now === base) delete st(r.id)[field]; else st(r.id)[field] = v;
      save(); node.classList.toggle("dirty", now !== base); tally();
    });
  };
  bind(".f-ins", "ins", n => n.value);
  bind(".f-cmd", "cmd", n => n.value);
  bind(".f-bef", "bef", n => n.value);
  bind(".f-mode", "mode", n => n.value);
  const cl = el.querySelector(".f-cl"), cc = el.querySelector(".f-cc");
  const curSet = () => {
    const v = [parseInt(cl.value || "1", 10), parseInt(cc.value || "1", 10)];
    if (JSON.stringify(v) === JSON.stringify(r.cur)) delete st(r.id).cur;
    else st(r.id).cur = v;
    save(); tally();
  };
  cl.addEventListener("input", curSet); cc.addEventListener("input", curSet);
  cl.addEventListener("change", () => { curSet(); redraw(); });
  cc.addEventListener("change", () => { curSet(); redraw(); });
  el.querySelector(".del").addEventListener("click", () => {
    const s = st(r.id);
    if (s.del) delete s.del;
    else s.del = window.prompt("Reason for deleting " + r.id + "?", "") || "removed via the curator";
    save(); redraw();
  });
  el.querySelector(".rst").addEventListener("click", () => {
    delete store[r.id]; save(); redraw();
  });
}

/* ---- filters ---- */
const CATS = [...new Set(ROWS.map(r => r.cat))].sort();
const SRCS = [...new Set(ROWS.map(r => r.src))].sort();
let filt = { ds: "all", state: "all", cat: "all", src: "all", q: "" };
function match(r) {
  if (filt.ds !== "all" && r.ds !== filt.ds) return false;
  if (filt.state === "edited" && !isEdited(r)) return false;
  if (filt.state === "deleted" && !isDeleted(r)) return false;
  if (filt.state === "problem" && !hasErr(r)) return false;
  if (filt.cat !== "all" && r.cat !== filt.cat) return false;
  if (filt.src !== "all" && r.src !== filt.src) return false;
  if (filt.q) {
    const hay = (r.id + " " + r.cmd + " " + val(r, "ins") + " " + r.bef).toLowerCase();
    if (!hay.includes(filt.q)) return false;
  }
  return true;
}
function buildRail() {
  const rail = $("rail");
  const dss = [["all", "both"], ["v3", "v3"], ["v4", "v4"]];
  const sts = [["all", "all"], ["edited", "edited"], ["deleted", "deleted"], ["problem", "problems"]];
  rail.innerHTML =
    '<span class="lab">dataset</span>' + dss.map(([k, l]) =>
      '<button class="f" data-ds="' + k + '">' + l + '<span class="c" data-cd="' + k + '"></span></button>').join("") +
    '<span class="lab">show</span>' + sts.map(([k, l]) =>
      '<button class="f" data-state="' + k + '">' + l + '<span class="c" data-cs="' + k + '"></span></button>').join("") +
    '<select id="fcat"><option value="all">all categories</option>' +
      CATS.map(c => '<option>' + esc(c) + "</option>").join("") + "</select>" +
    '<select id="fsrc"><option value="all">all sources</option>' +
      SRCS.map(c => '<option>' + esc(c) + "</option>").join("") + "</select>" +
    '<input type="search" id="q" placeholder="id, command, text…">' +
    '<span class="count" id="count"></span>';
  rail.querySelectorAll("[data-ds]").forEach(b => b.addEventListener("click", () => {
    filt.ds = b.dataset.ds; shown = PAGE; render(); }));
  rail.querySelectorAll("[data-state]").forEach(b => b.addEventListener("click", () => {
    filt.state = b.dataset.state; shown = PAGE; render(); }));
  $("fcat").addEventListener("change", e => { filt.cat = e.target.value; shown = PAGE; render(); });
  $("fsrc").addEventListener("change", e => { filt.src = e.target.value; shown = PAGE; render(); });
  $("q").addEventListener("input", e => {
    filt.q = e.target.value.toLowerCase().trim(); shown = PAGE; render(); });
}

/* ---- list ---- */
const PAGE = 40;
let shown = PAGE, view = [];
function render() {
  view = ROWS.filter(match);
  const host = $("cards");
  host.innerHTML = "";
  if (!view.length) { host.innerHTML = '<div class="empty">Nothing matches those filters.</div>'; }
  const frag = document.createDocumentFragment();
  view.slice(0, shown).forEach(r => frag.appendChild(card(r)));
  host.appendChild(frag);
  $("more").style.display = view.length > shown ? "" : "none";
  document.querySelectorAll("[data-ds]").forEach(b =>
    b.setAttribute("aria-pressed", filt.ds === b.dataset.ds));
  document.querySelectorAll("[data-state]").forEach(b =>
    b.setAttribute("aria-pressed", filt.state === b.dataset.state));
  tally();
}
$("more").addEventListener("click", () => { shown += PAGE; render(); });

function tally() {
  let ed = 0, del = 0, bad = 0, promo = 0;
  for (const r of ROWS) {
    const d = isEdited(r), x = isDeleted(r);
    if (x) del++; else if (d) { ed++; if (r.ds === "v4") promo++; }
    if (!x && hasErr(r)) bad++;
  }
  $("agg").innerHTML =
    '<div><span class="v">' + ROWS.length + '</span><span class="k">cases</span></div>' +
    '<div><span class="v">' + ROWS.filter(r => r.ds === "v3").length + '</span><span class="k">in v3</span></div>' +
    '<div><span class="v">' + ROWS.filter(r => r.ds === "v4").length + '</span><span class="k">in v4</span></div>' +
    '<div><span class="v" style="color:var(--accent)">' + ed + '</span><span class="k">edited</span></div>' +
    '<div><span class="v" style="color:var(--warn)">' + promo + '</span><span class="k">will promote to v3</span></div>' +
    '<div><span class="v" style="color:var(--stop)">' + del + '</span><span class="k">to delete</span></div>';
  const c = $("count"); if (c) c.textContent = view.length + " shown";
  $("dockcount").textContent = ed + " edited · " + promo + " promoting · " + del +
    " to delete" + (bad ? " · " + bad + " with problems" : "");
  const cd = { all: ROWS.length, v3: 0, v4: 0 };
  ROWS.forEach(r => cd[r.ds]++);
  document.querySelectorAll("[data-cd]").forEach(e => { e.textContent = cd[e.dataset.cd]; });
  const cs = { all: ROWS.length, edited: ed, deleted: del, problem: bad };
  document.querySelectorAll("[data-cs]").forEach(e => { e.textContent = cs[e.dataset.cs]; });
}

/* ---- export ---- */
function changeset() {
  const edits = {}, deletes = {};
  for (const r of ROWS) {
    if (isDeleted(r)) { deletes[r.id] = store[r.id].del; continue; }
    const d = dirtyFields(r);
    if (!d.length || hasErr(r)) continue;
    const e = {};
    d.forEach(f => { e[f === "ins" ? "instruction" : f === "cmd" ? "command"
      : f === "bef" ? "buffer_before" : f === "cur" ? "cursor" : "mode"] = val(r, f); });
    edits[r.id] = e;
  }
  return { edits, deletes };
}
$("export").addEventListener("click", () => {
  const cs = changeset();
  const ne = Object.keys(cs.edits).length, nd = Object.keys(cs.deletes).length;
  const promo = Object.keys(cs.edits).filter(id => BY.get(id).ds === "v4").length;
  const bad = ROWS.filter(r => !isDeleted(r) && isEdited(r) && hasErr(r)).length;
  $("dlgtitle").textContent = ne + " edit" + (ne === 1 ? "" : "s") + ", " + nd + " deletion" + (nd === 1 ? "" : "s");
  $("out").value = JSON.stringify(cs, null, 1);
  $("dlgsub").textContent =
    (promo ? promo + " of the edits are v4 rows and will be promoted into v3 (id unchanged). " : "") +
    (bad ? bad + " edited row(s) have errors and were left out of the export. " : "") +
    "Save as changes.json, dry-run, then drop --dry-run. Rebuild with build_corpus.py --dataset all afterwards.";
  $("dlg").showModal();
});
$("close").addEventListener("click", () => $("dlg").close());
$("copy").addEventListener("click", async e => {
  $("out").select();
  let ok = false;
  try { await navigator.clipboard.writeText($("out").value); ok = true; } catch (err) {}
  e.currentTarget.textContent = ok ? "Copied" : "Press Ctrl+C";
  setTimeout(() => { e.currentTarget.textContent = "Copy JSON"; }, 1800);
});
$("reset").addEventListener("click", () => {
  if (!window.confirm("Discard every unsaved edit and deletion mark?")) return;
  store = {}; save(); render();
});

function boot() { buildRail(); render(); const b = $("boot"); if (b) b.remove(); }
try { boot(); } catch (err) {
  const b = $("boot");
  if (b) { b.className = "boot err";
    b.textContent = "This page failed to start.\n\n" + (err && err.stack ? err.stack : String(err)); }
  throw err;
}
</script>
"""

REQUIRED_JS = [r"if (/\n/.test(ins))", r"/\bpress\b|\bkeystroke|\btype the key/"]
FORBIDDEN = {chr(c) for c in list(range(0, 9)) + [11, 12] + list(range(14, 32))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = collect()
    blob = json.dumps(rows, ensure_ascii=False,
                      separators=(",", ":")).replace("</", "<\\/")
    page = TEMPLATE.replace("__PAYLOAD__", blob).replace("__TOTAL__", str(len(rows)))
    print("  guard: " + check_page(page, REQUIRED_JS, "Corpus Curator"))

    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    c = collections.Counter(r["ds"] for r in rows)
    print("cases: %d (%s)  page: %.0f KB  -> %s"
          % (len(rows), " · ".join("%s %d" % kv for kv in sorted(c.items())),
             len(page.encode()) / 1024, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
