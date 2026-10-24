#!/usr/bin/env python3
"""Build the hand-review page for the never-solved train rows: every train
prompt that at least one probed adapter never solved in 8 samples, with a
full editor for every field of the row.

Four full difficulty probes exist (scripts/probe_difficulty.py, G = 8 at
T = 1 over the train split), all on Qwen3.5 adapters after RL or the cliff
SFT round (DECISIONS.md 2026-09-14 to 09-18). A row one of them never solves
is either beyond that adapter or a bad row, and a reader decides which. Each
card shows:

  * how each of the four adapters did on it (k of 8) and its most common
    answers, marking a probe that ran before the row's last edit;
  * the row's review history -- the 2026-09-16 cliff review kept, reworded or
    retargeted 282 of these rows;
  * the buffer the most common runnable wrong answer produces, lines that
    differ from the gold output marked;
  * an editor for the instruction, gold command, mode, cursor, buffer,
    category and tier, with the curation guard's checks run live;
  * Claude's read of the row, where one exists (runs/never_review_2026-09-25/
    comments, see notes.py): a class, the reason, and any suggested change,
    which a button copies into the editor.

Sorted by how many of the four adapters never solve the row (all four first).
Decisions are saved to the page's shared store (collection `decisions`, one
document per row id), where Claude reads them back; Export shows them as an
apply_curation.py change-set, with category and tier edits under `meta_edits`
(apply_curation.py does not edit those two fields).

Usage:
    python scripts/never_review/build_never_review.py --out page.html
"""
import argparse
import collections
import html
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from executor import run_vim                    # noqa: E402
from evaluate import _norm, load_jsonl          # noqa: E402
from build_test_review import CSS as BASE_CSS   # noqa: E402  (the review pages' look)
from page_guard import check_page               # noqa: E402
from notes import CLASSES as NOTE_CLASSES, load_notes   # noqa: E402

PROBE_DIR = REPO / "runs" / "grpo_gated_2026-09-14"
# (key, label, file, date the probe ran). The date decides staleness: a row
# edited after its probe ran was probed in its earlier wording.
PROBES = [
    ("grpo1", "4B after GRPO run 1", "difficulty_probe_qwen3.5-4b.jsonl", "2026-09-14"),
    ("9b", "9B after GRPO (4-bit)", "difficulty_probe_qwen3.5-9b.jsonl", "2026-09-15"),
    ("served", "4B served (gated GRPO)", "difficulty_probe_qwen3.5-4b-leg2.jsonl", "2026-09-17"),
    ("cliffsft", "4B cliff SFT", "difficulty_probe_qwen3.5-4b-cliffleg.jsonl", "2026-09-17"),
]
CLIFF_REVIEW = REPO / "data" / "fixes" / "cliff_disagreements_2026-09-16.json"
# The served adapter re-sampled 32 times on the rows its 8-sample probe never
# solved (scripts/cliff_recheck/, 2026-10-16). A row still at 0 of 32 whose one
# runnable wrong answer holds >= STABLE_SHARE of the samples is the shape the
# 2026-09-16 cliff review found to be a data question as often as a model
# error (83 of 292 changed), so the page can filter to those rows first.
RECHECK = REPO / "runs" / "cliff_recheck_2026-10-16" / "recheck_g32_qwen3.5-4b.jsonl"
STABLE_SHARE = 0.75
# The same day's whitespace repair rewrote 18 golds into token notation and,
# with them, restored four rows the cliff review had dropped.
WHITESPACE_FIX = REPO / "data" / "fixes" / "whitespace_golds_2026-09-16.json"
STATE_WORDS = {"valid_wrong": "wrong buffer", "invalid": "vim error", "match": "solves it",
               "parse": "not the §6 grammar", "truncated": "ran into the cap"}


def load_probe(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r
    return out


def never_sets(probes, train_ids):
    """{probe key: ids with k == 0}, restricted to rows still in train."""
    return {key: {i for i, r in rows.items() if r["k"] == 0 and i in train_ids}
            for key, rows in probes.items()}


def history(rid, cliff, ws=None):
    """The row's 2026-09-16 review history in one line, or None. `ws` is the
    whitespace repair's change-set: a row the cliff review dropped that is
    still in train came back through it."""
    parts = []
    if rid in cliff["deletes"]:
        parts.append("cliff review 2026-09-16: dropped")
    elif rid in cliff["edits"]:
        parts.append("cliff review 2026-09-16: " + ", ".join(
            "retargeted" if k == "command" else "reworded" if k == "instruction" else k
            for k in cliff["edits"][rid]))
    elif rid in cliff["keep"]:
        parts.append("cliff review 2026-09-16: kept as it was")
    if ws and rid in ws.get("edits", {}):
        parts.append("restored by the whitespace repair, gold in token notation" if rid in cliff["deletes"]
                     else "whitespace repair 2026-09-16: gold in token notation")
    return "; ".join(parts) or None


def pane_answer(rows_by_probe, rid):
    """The most common RUNNABLE wrong answer pooled over the probes, with the
    probes that gave it: {mode, cmd, by: {key: n}} or None."""
    pooled, by = collections.Counter(), collections.defaultdict(dict)
    for key, rows in rows_by_probe.items():
        r = rows.get(rid)
        for e in (r or {}).get("commands", []):
            if e["state"] == "valid_wrong":
                pooled[(e["mode"], e["command"])] += e["n"]
                by[(e["mode"], e["command"])][key] = e["n"]
    if not pooled:
        return None
    (mode, cmd), _ = max(pooled.items(), key=lambda kv: (kv[1], kv[0]))
    return {"mode": mode, "cmd": cmd, "by": by[(mode, cmd)]}


def collect():
    train = {c["id"]: c for c in load_jsonl(str(REPO / "data" / "splits" / "train.jsonl"))}
    rows = {key: load_probe(PROBE_DIR / f) for key, _l, f, _d in PROBES}
    never = never_sets(rows, set(train))
    union = set().union(*never.values())
    cliff = json.loads(CLIFF_REVIEW.read_text(encoding="utf-8"))
    cliff["keep"] = set(cliff.get("keep") or [])
    ws = json.loads(WHITESPACE_FIX.read_text(encoding="utf-8"))
    cards = []
    for rid in union:
        c = train[rid]
        cursor = list(c.get("cursor") or (1, 1))
        edited = c.get("curated_on")
        probes = []
        for key, label, _f, date in PROBES:
            r = rows[key].get(rid)
            probes.append({"key": key, "label": label, "date": date,
                           "k": None if r is None else r["k"], "G": None if r is None else r["G"],
                           "stale": bool(edited and r is not None and edited >= date),
                           "top": [] if r is None else [
                               {"mode": e["mode"], "cmd": e["command"], "n": e["n"], "state": e["state"]}
                               for e in r["commands"][:3]]})
        alt = pane_answer(rows, rid)
        if alt is not None:
            res = run_vim(c["buffer_before"], alt["cmd"], mode=alt["mode"], cursor=tuple(cursor))
            alt["buf"] = res.get("buffer_after")
            if alt["buf"] is not None and _norm(alt["buf"]) == _norm(c["buffer_after"]):
                alt = None         # the row changed since the probe and this answer now matches
        cmd = c["gold_command"].lstrip(":") if c.get("mode") == "ex" else c["gold_command"]
        cards.append({"id": rid, "cat": c["category"], "tier": c.get("tier") or "T?",
                      "mode": c.get("mode", "ex"), "cmd": c["gold_command"], "guardcmd": cmd,
                      "cur": cursor, "ins": c.get("instruction", ""), "bef": c["buffer_before"],
                      "exp": c["buffer_after"], "dataset": c.get("dataset"),
                      "prov": c.get("instruction_provenance"), "edited": edited,
                      "never": sum(rid in never[k] for k, *_ in PROBES),
                      "never_for": [k for k, *_ in PROBES if rid in never[k]],
                      "hist": history(rid, cliff, ws), "probes": probes, "alt": alt})
    attach_recheck(cards, train)
    cards.sort(key=lambda x: (-x["never"], x["cat"], x["tier"], x["id"]))
    cats = sorted({c["category"] for c in train.values()})
    return cards, cats


def attach_recheck(cards, train, path=RECHECK, share=STABLE_SHARE):
    """Mark the rows the 32-sample re-check still never solves with one stable
    runnable wrong answer: c["stable"] = {mode, cmd, n, G, buf}. Returns how many."""
    if not pathlib.Path(path).exists():
        return 0
    rec = {r["id"]: r for r in load_jsonl(str(path))}
    n_marked = 0
    for c in cards:
        r = rec.get(c["id"])
        if r is None or int(r["k"]) != 0:
            continue
        wrong = [e for e in r.get("commands", []) if e["state"] == "valid_wrong"]
        if not wrong:
            continue
        top = max(wrong, key=lambda e: (e["n"], e["command"]))
        if top["n"] < share * int(r["G"]):
            continue
        t = train[c["id"]]
        res = run_vim(t["buffer_before"], top["command"], mode=top["mode"], cursor=tuple(c["cur"]))
        c["stable"] = {"mode": top["mode"], "cmd": top["command"], "n": top["n"], "G": int(r["G"]),
                       "buf": res.get("buffer_after")}
        n_marked += 1
    return n_marked


def attach_notes(cards, notes):
    """Put Claude's note on each card it covers, as {cls, text, sugg, sbuf?}.
    A suggestion that changes the command, mode, cursor or buffer also gets
    the buffer it produces (`sbuf`), so the page can show the new output next
    to the gold's. Returns how many cards carry a note."""
    n_noted = 0
    for c in cards:
        n = notes.get(c["id"])
        if n is None:
            continue
        s = dict(n.get("suggest") or {})
        note = {"cls": n["class"], "text": n["comment"], "sugg": s}
        if any(k in s for k in ("command", "mode", "cursor", "buffer_before")):
            mode = s.get("mode", c["mode"])
            cmd = s.get("command", c["cmd"])
            res = run_vim(s.get("buffer_before", c["bef"]), cmd.lstrip(":") if mode == "ex" else cmd,
                          mode=mode, cursor=tuple(s.get("cursor", c["cur"])))
            note["sbuf"] = res.get("buffer_after")
        c["note"] = note
        n_noted += 1
    return n_noted


CSS = r"""
.card.v-ok{ border-left-color:var(--ok); }
.card.v-edited{ border-left-color:var(--warn); }
.card.v-drop{ border-left-color:var(--stop); }
.card.v-unsure{ border-left-color:var(--faint); }
.vb.ok[aria-pressed="true"]{ background:var(--ok-bg); color:var(--ok); border-color:var(--ok); font-weight:700; }
.vb.unsure[aria-pressed="true"]{ background:var(--raise); color:var(--ink); border-color:var(--ink2); font-weight:700; }
.chip.n4{ background:var(--stop-bg); color:var(--stop); border:1px solid var(--stop-line); font-weight:700; }
.chip.n3, .chip.n2{ background:var(--warn-bg); color:var(--warn); border:1px solid var(--warn-line); font-weight:700; }
.chip.n1{ background:var(--raise); color:var(--ink2); font-weight:700; }
.chip.hist{ background:var(--ok-bg); color:var(--ok); border:1px solid var(--ok-line); }
.chip.dirty{ background:var(--warn-bg); color:var(--warn); border:1px solid var(--warn-line); font-weight:700; }
.ev{ margin:0 15px 12px; border:1px solid var(--line2); border-radius:7px; overflow:hidden; }
.evh{ font-family:var(--mono); font-size:10px; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); font-weight:700; padding:7px 11px; background:var(--sunk);
  border-bottom:1px solid var(--line2); }
.evs{ overflow-x:auto; }
.evt{ border-collapse:collapse; width:100%; font-size:12.5px; }
.evt th{ font-family:var(--mono); font-size:11px; color:var(--ink2); text-align:left; vertical-align:top;
  padding:6px 11px; white-space:nowrap; font-weight:700; width:1%; }
.evt td{ padding:6px 11px 6px 0; vertical-align:top; }
.evt tr + tr th, .evt tr + tr td{ border-top:1px solid var(--line2); }
.kk{ font-family:var(--mono); font-size:11px; font-variant-numeric:tabular-nums; white-space:nowrap; }
.kk.zero{ color:var(--stop); font-weight:700; } .kk.some{ color:var(--ok); font-weight:700; }
.stale{ font-family:var(--mono); font-size:10px; color:var(--warn); margin-left:6px; }
.ans{ display:flex; flex-wrap:wrap; gap:4px 8px; align-items:baseline; padding:1px 0; }
.ans code{ font-family:var(--mono); font-size:12px; color:var(--ink); overflow-wrap:anywhere; }
.ans .n{ font-family:var(--mono); font-size:11px; color:var(--muted); white-space:nowrap; }
.st{ font-family:var(--mono); font-size:10px; padding:1px 6px; border-radius:4px; white-space:nowrap; border:1px solid; }
.st.valid_wrong{ color:var(--warn); background:var(--warn-bg); border-color:var(--warn-line); }
.st.invalid, .st.parse, .st.truncated{ color:var(--stop); background:var(--stop-bg); border-color:var(--stop-line); }
.st.match{ color:var(--ok); background:var(--ok-bg); border-color:var(--ok-line); }
.insed{ margin:10px 15px 4px; }
.insed textarea{ width:100%; font-family:var(--disp); font-size:16px; line-height:1.45; padding:7px 9px;
  border:1px solid var(--line); border-radius:6px; background:var(--surface); color:var(--ink);
  resize:vertical; min-height:52px; }
.insed textarea.dirty, .fields .dirty{ border-color:var(--warn); background:var(--warn-bg); }
details.more{ border-top:1px solid var(--line2); background:var(--sunk); }
details.more > summary{ cursor:pointer; padding:9px 14px; font-family:var(--mono); font-size:11.5px;
  color:var(--ink2); }
.fields{ display:flex; flex-direction:column; gap:9px; padding:4px 14px 12px; }
.frow{ display:flex; gap:9px; flex-wrap:wrap; align-items:flex-end; }
.frow > label{ display:flex; flex-direction:column; gap:4px; flex:1 1 200px; min-width:0; }
.frow > label.sm{ flex:0 0 108px; }
.fields label > span{ font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
.fields textarea, .fields input, .fields select{ font-family:var(--mono); font-size:12.5px; padding:6px 8px;
  border:1px solid var(--line); border-radius:5px; background:var(--surface); color:var(--ink); width:100%; }
.fields textarea{ resize:vertical; min-height:70px; white-space:pre; }
.checks{ font-family:var(--mono); font-size:11.5px; padding:0 15px 10px; }
.checks .bad{ color:var(--stop); } .checks .note{ color:var(--warn); } .checks .ok{ color:var(--ok); }
.pager{ display:flex; gap:8px; align-items:center; justify-content:center; padding:16px 0 4px;
  font-family:var(--mono); font-size:12px; color:var(--muted); flex-wrap:wrap; }
.save{ font-family:var(--mono); font-size:11px; padding:3px 9px; border-radius:5px; border:1px solid var(--line);
  color:var(--muted); background:var(--ground); }
.save.shared{ color:var(--ok); border-color:var(--ok-line); background:var(--ok-bg); }
.save.error{ color:var(--stop); border-color:var(--stop-line); background:var(--stop-bg); }
.copied{ font-family:var(--mono); font-size:11px; color:var(--ok); }
.fixrow{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
input.search{ font-family:var(--mono); font-size:12px; padding:4px 8px; border:1px solid var(--line);
  border-radius:5px; background:var(--ground); color:var(--ink); min-width:0; flex:0 1 180px; }
.read{ margin:0 15px 12px; border:1px solid var(--line2); border-left:3px solid var(--accent);
  border-radius:7px; background:var(--sunk); }
.readh{ display:flex; gap:8px; align-items:center; padding:8px 11px 0; }
.read p{ margin:6px 11px 9px; font-size:14px; line-height:1.5; color:var(--ink); }
.read .panes{ border-top:1px solid var(--line2); }
.sugg{ border-top:1px solid var(--line2); padding:8px 11px 10px; display:flex; flex-direction:column; gap:6px; }
.sf{ display:flex; gap:9px; align-items:baseline; flex-wrap:wrap; }
.sf .sk{ font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--muted); font-weight:700; flex:0 0 96px; }
.sf code{ font-family:var(--mono); font-size:12.5px; color:var(--ink); overflow-wrap:anywhere; flex:1 1 220px; }
.sf .prose{ font-family:var(--body); font-size:14.5px; }
.sgnote{ font-family:var(--mono); font-size:11px; color:var(--muted); }
.sgnote.used{ color:var(--ok); }
.chip.cl-defect{ background:var(--stop-bg); color:var(--stop); border:1px solid var(--stop-line); font-weight:700; }
.chip.cl-ambiguous, .chip.cl-unstated{ background:var(--warn-bg); color:var(--warn); border:1px solid var(--warn-line); font-weight:700; }
.chip.cl-hard{ background:var(--raise); color:var(--ink2); border:1px solid var(--line); font-weight:700; }
.chip.cl-model{ background:var(--ok-bg); color:var(--ok); border:1px solid var(--ok-line); font-weight:700; }
.legend{ display:grid; grid-template-columns:auto 1fr; gap:5px 10px; margin-top:10px; align-items:baseline;
  font-size:14px; color:var(--ink2); }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('payload').textContent);
const STATE_WORDS = JSON.parse(document.getElementById('states').textContent);
const CATS = JSON.parse(document.getElementById('cats').textContent);
const NOTE_CLASSES = JSON.parse(document.getElementById('classes').textContent);
const BY_ID = {};
DATA.forEach(function(c){ BY_ID[c.id] = c; });
const LKEY = 'vimbot-never-review-v1';
const PAGE = 30;
const FIELDS = ['instruction', 'command', 'mode', 'cursor', 'buffer_before', 'category', 'tier'];
const SUGG_LABELS = {instruction: 'instruction', command: 'gold command', mode: 'mode', cursor: 'cursor', buffer_before: 'buffer before'};
let store = {};                 // id -> {verdict, note, edits: {field: value}, updated}
let db = null;
const queue = {}, timers = {};
let page = 0, fNever = 'all', fCat = 'all', fHist = 'all', fState = 'all', fNote = 'all', fQuery = '';
let fStable = location.hash === '#stable' ? 'stable' : 'all';   // the 32-sample re-check's stable-wrong rows
let view = DATA.slice();
try { store = JSON.parse(localStorage.getItem(LKEY) || '{}'); } catch (e) { store = {}; }

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function vis(line){
  let out = esc(line).replace(/\t/g, '<span class="ws">--&gt;</span>\t');
  const m = out.match(/( +)$/);
  if (m) out = out.slice(0, out.length - m[1].length) + '<span class="ws">' + '.'.repeat(m[1].length) + '</span>';
  return out;
}
function lines(t){
  const l = String(t == null ? '' : t).split('\n');
  if (l.length && l[l.length-1] === '') l.pop();
  return l;
}
function pane(label, text, other, kind){
  const a = lines(text), b = lines(other);
  let rows = '';
  for (let i = 0; i < a.length; i++){
    const cls = a[i] !== b[i] ? ' ' + kind : '';
    rows += '<div class="ln' + cls + '"><span class="g">' + (i+1) + '</span><span class="t">' + vis(a[i]) + '</span></div>';
  }
  if (!rows) rows = '<div class="ln"><span class="g">-</span><span class="t">(empty buffer)</span></div>';
  return '<div class="pane"><span class="lab">' + label + '</span>' + rows + '</div>';
}
function ans(a){
  // A parse failure or a truncated sample has no command to show.
  const text = a.cmd == null ? '(no command: ' + (STATE_WORDS[a.state] || a.state) + ')' : a.mode + ': ' + a.cmd;
  return '<div class="ans"><code>' + esc(text) + '</code><span class="n">' + a.n
    + '/8</span><span class="st ' + esc(a.state) + '">' + esc(STATE_WORDS[a.state] || a.state) + '</span></div>';
}

// The curation guard's leak check (scripts/apply_instructions.py `leaks`).
function leakOf(ins, cmd){
  const low = ins.toLowerCase();
  const f0 = cmd.trim(), noTags = f0.replace(/<[^>]*>/g, '');
  const forms = [f0, noTags, noTags.replace(/^:+/, '')];
  for (let i = 0; i < forms.length; i++){
    const f = forms[i].trim().toLowerCase();
    if (f.length >= 2 && low.indexOf(f) >= 0) return 'the instruction contains the gold command “' + f + '” verbatim';
  }
  if (/\bpress\b|\bkeystroke|\btype the key/.test(low)) return 'the instruction describes keystrokes rather than intent';
  return null;
}
function orig(c, f){
  return f === 'instruction' ? c.ins : f === 'command' ? c.cmd : f === 'mode' ? c.mode
       : f === 'cursor' ? c.cur : f === 'buffer_before' ? c.bef : f === 'category' ? c.cat : c.tier;
}
function val(c, f){
  const e = (store[c.id] || {}).edits || {};
  return e[f] !== undefined ? e[f] : orig(c, f);
}
function isEdited(c){
  const e = (store[c.id] || {}).edits || {};
  return Object.keys(e).length > 0;
}
function problems(c){
  const out = [];
  const ins = String(val(c, 'instruction')).trim();
  if (!ins) out.push(['bad', 'the instruction is empty']);
  let cmd = String(val(c, 'command'));
  if (val(c, 'mode') === 'ex') cmd = cmd.replace(/^:+/, '');
  if (!cmd.trim()) out.push(['bad', 'the gold command is empty']);
  const leak = ins && cmd.trim() ? leakOf(ins, cmd) : null;
  if (leak) out.push(['bad', leak]);
  const ls = lines(val(c, 'buffer_before'));
  if (!ls.length) ls.push('');
  const cur = val(c, 'cursor');
  if (!(cur[0] >= 1 && cur[0] <= ls.length)) out.push(['bad', 'cursor line ' + cur[0] + ' is outside the ' + ls.length + '-line buffer']);
  else if (!(cur[1] >= 1 && cur[1] <= Math.max(1, ls[cur[0]-1].length + 1))) out.push(['bad', 'cursor column ' + cur[1] + ' is outside line ' + cur[0]]);
  const e = (store[c.id] || {}).edits || {};
  if (['command', 'buffer_before', 'cursor', 'mode'].some(function(f){ return e[f] !== undefined; }))
    out.push(['note', 'the gold output shown is now stale: apply_curation.py re-executes the row and recomputes it']);
  if (e.category !== undefined || e.tier !== undefined)
    out.push(['note', 'category and tier edits go in meta_edits: apply_curation.py does not change those fields']);
  return out;
}

function evidence(c){
  let rows = '';
  c.probes.forEach(function(p){
    const kk = p.k == null ? '<span class="kk">not probed</span>'
      : '<span class="kk ' + (p.k === 0 ? 'zero' : 'some') + '">' + p.k + '/' + p.G + '</span>';
    rows += '<tr><th>' + esc(p.label) + '</th><td>' + kk
      + (p.stale ? '<span class="stale">probed before this row’s ' + esc(c.edited) + ' edit</span>' : '')
      + p.top.map(ans).join('') + '</td></tr>';
  });
  if (c.stable){
    rows += '<tr><th>4B served, 32-sample re-check (2026-10-16)</th><td><span class="kk zero">0/' + c.stable.G + '</span>'
      + '<div class="ans"><code>' + esc(c.stable.mode + ': ' + c.stable.cmd) + '</code><span class="n">' + c.stable.n + '/' + c.stable.G
      + '</span><span class="st valid_wrong">' + esc(STATE_WORDS.valid_wrong || 'wrong buffer') + '</span></div></td></tr>';
  }
  return '<div class="ev"><div class="evh">How each probed adapter did (8 samples at T = 1' + (c.stable ? '; the re-check 32' : '') + ')</div>'
    + '<div class="evs"><table class="evt">' + rows + '</table></div></div>';
}

// Claude's note on the row: its class, why, and any suggested change, with
// the output a suggested command produces set against the gold output.
function hasSugg(c){ return !!(c.note && Object.keys(c.note.sugg || {}).length); }
function readHtml(c){
  const n = c.note;
  if (!n) return '';
  const keys = Object.keys(n.sugg || {});
  let sugg = '';
  if (keys.length){
    sugg = '<div class="sugg"><span class="lab">suggested change</span>'
      + keys.map(function(k){
          const v = k === 'cursor' ? n.sugg[k][0] + ',' + n.sugg[k][1] : String(n.sugg[k]);
          const cls = k === 'instruction' ? ' class="prose"' : k === 'buffer_before' ? ' style="white-space:pre-wrap"' : '';
          return '<div class="sf"><span class="sk">' + esc(SUGG_LABELS[k] || k) + '</span><code' + cls + '>' + esc(v) + '</code></div>';
        }).join('')
      + '<div class="fixrow"><button class="btn" data-sugg="' + esc(c.id) + '">Use suggestion</button>'
      + '<span class="sgnote" data-sgnote="' + esc(c.id) + '"></span></div></div>';
  }
  const out = n.sbuf === undefined ? '' : '<div class="panes">'
    + pane('output of the suggested change', n.sbuf, c.exp, 'add') + pane('gold output now', c.exp, n.sbuf, 'cut') + '</div>';
  return '<div class="read"><div class="readh"><span class="lab">Claude’s read</span>'
    + '<span class="chip cl-' + esc(n.cls) + '" title="' + esc(NOTE_CLASSES[n.cls] || '') + '">' + esc(n.cls) + '</span></div>'
    + '<p>' + esc(n.text) + '</p>' + sugg + out + '</div>';
}
function suggInUse(c){
  const s = c.note.sugg;
  return Object.keys(s).every(function(k){ return JSON.stringify(val(c, k)) === JSON.stringify(s[k]); });
}

function cardHtml(c){
  const id = esc(c.id);
  const opts = function(list, cur){
    return list.map(function(x){ return '<option value="' + esc(x) + '"' + (x === cur ? ' selected' : '') + '>' + esc(x) + '</option>'; }).join('');
  };
  const alt = c.alt
    ? pane('most common runnable wrong answer<br><code style="text-transform:none;letter-spacing:0">'
           + esc(c.alt.mode + ': ' + c.alt.cmd) + '</code>', c.alt.buf, c.exp, 'cut')
    : '<div class="pane"><span class="lab">no runnable wrong answer</span><div class="ln"><span class="t">'
      + 'every answer vim ran was an error</span></div></div>';
  const cur = val(c, 'cursor');
  return '<div class="card" id="c-' + id + '">'
    + '<div class="chead"><span class="cid">' + id + '</span>'
    +   '<span class="keys">' + esc(c.mode + ': ' + c.cmd) + '</span>'
    +   '<span class="chip">' + esc(c.cat) + '</span><span class="chip">' + esc(c.tier) + '</span>'
    +   '<span class="chip">cursor ' + c.cur[0] + ',' + c.cur[1] + '</span>'
    +   '<span class="chip">' + esc(c.dataset || '') + '</span>'
    +   (c.hist ? '<span class="chip hist">' + esc(c.hist) + '</span>' : '')
    +   (c.stable ? '<span class="chip n4" title="32-sample re-check, 2026-10-16: the same runnable wrong answer in '
    +     c.stable.n + ' of ' + c.stable.G + ' tries">stable wrong ' + c.stable.n + '/' + c.stable.G + '</span>' : '')
    +   '<span class="spacer"></span><span class="chip dirty" data-dirty="' + id + '" hidden>edited</span>'
    +   '<span class="chip n' + c.never + '">never for ' + c.never + ' of 4</span></div>'
    + '<div class="insed"><textarea data-id="' + id + '" data-field="instruction" id="f-instruction-' + id
    +   '" aria-label="instruction" spellcheck="true">' + esc(val(c, 'instruction')) + '</textarea></div>'
    + '<div class="checks" id="k-' + id + '"></div>'
    + readHtml(c)
    + evidence(c)
    + '<div class="panes">' + pane('input buffer', c.bef, c.exp, 'cut') + pane('gold output', c.exp, c.bef, 'add') + alt
    +   ((c.stable && (!c.alt || c.alt.cmd !== c.stable.cmd || c.alt.mode !== c.stable.mode))
        ? pane('the re-check’s stable answer<br><code style="text-transform:none;letter-spacing:0">'
               + esc(c.stable.mode + ': ' + c.stable.cmd) + '</code>', c.stable.buf, c.exp, 'cut') : '')
    + '</div>'
    + '<details class="more"><summary>Edit the gold command, mode, cursor, buffer, category or tier</summary>'
    + '<div class="fields">'
    +   '<div class="frow"><label><span>gold command</span><input data-id="' + id + '" data-field="command" id="f-command-' + id
    +     '" spellcheck="false" value="' + esc(val(c, 'command')) + '"></label>'
    +     (c.alt ? '<button class="btn" data-use="' + id + '">Use the models’ answer</button>' : '')
    +   '</div>'
    +   '<div class="frow">'
    +     '<label class="sm"><span>mode</span><select data-id="' + id + '" data-field="mode" id="f-mode-' + id + '">' + opts(['ex', 'keys'], val(c, 'mode')) + '</select></label>'
    +     '<label class="sm"><span>cursor line</span><input type="number" min="1" data-id="' + id + '" data-field="cursorline" id="f-cl-' + id + '" value="' + cur[0] + '"></label>'
    +     '<label class="sm"><span>cursor col</span><input type="number" min="1" data-id="' + id + '" data-field="cursorcol" id="f-cc-' + id + '" value="' + cur[1] + '"></label>'
    +     '<label class="sm"><span>category</span><select data-id="' + id + '" data-field="category" id="f-category-' + id + '">' + opts(CATS, val(c, 'category')) + '</select></label>'
    +     '<label class="sm"><span>tier</span><select data-id="' + id + '" data-field="tier" id="f-tier-' + id + '">' + opts(['T1', 'T2', 'T3'], val(c, 'tier')) + '</select></label>'
    +   '</div>'
    +   '<label><span>buffer before (the gold output is recomputed from it when applied)</span><textarea data-id="' + id
    +     '" data-field="buffer_before" id="f-buffer-' + id + '" spellcheck="false">' + esc(val(c, 'buffer_before')) + '</textarea></label>'
    +   '<div class="fixrow"><button class="btn" data-revert="' + id + '">Revert every edit on this row</button></div>'
    + '</div></details>'
    + '<div class="cfoot">'
    +   '<button class="vb ok" data-act="ok" data-id="' + id + '" aria-pressed="false">Row is fine</button>'
    +   '<button class="vb drop" data-act="drop" data-id="' + id + '" aria-pressed="false">Drop row</button>'
    +   '<button class="vb unsure" data-act="unsure" data-id="' + id + '" aria-pressed="false">Unsure</button>'
    +   '<input class="note" data-id="' + id + '" data-field="note" id="f-note-' + id + '" placeholder="note (optional)" aria-label="note">'
    + '</div></div>';
}

function paint(c){
  const el = document.getElementById('c-' + c.id);
  if (!el) return;
  const s = store[c.id] || {};
  const edited = isEdited(c);
  const kind = s.verdict === 'drop' ? 'drop' : s.verdict === 'unsure' ? 'unsure' : edited ? 'edited' : s.verdict === 'ok' ? 'ok' : '';
  el.className = 'card' + (kind ? ' v-' + kind : '');
  el.querySelectorAll('.vb[data-act]').forEach(function(b){
    b.setAttribute('aria-pressed', String(b.getAttribute('data-act') === s.verdict));
  });
  document.querySelector('[data-dirty="' + c.id + '"]').hidden = !edited;
  const e = s.edits || {};
  el.querySelectorAll('[data-field]').forEach(function(f){
    const field = f.getAttribute('data-field');
    const key = field === 'cursorline' || field === 'cursorcol' ? 'cursor' : field;
    if (key === 'note'){ if (document.activeElement !== f) f.value = s.note || ''; return; }
    if (document.activeElement !== f){
      const v = val(c, key);
      f.value = field === 'cursorline' ? v[0] : field === 'cursorcol' ? v[1] : v;
    }
    f.classList.toggle('dirty', e[key] !== undefined);
  });
  const sn = el.querySelector('[data-sgnote]');
  if (sn){
    const used = suggInUse(c);
    sn.classList.toggle('used', used);
    sn.textContent = used ? 'in the editor now; Revert under the card undoes it'
                          : 'copies it into the editor, where the checks run on it';
    el.querySelector('[data-sugg]').disabled = used;
  }
  const probs = problems(c);
  document.getElementById('k-' + c.id).innerHTML = probs.length
    ? probs.map(function(p){ return '<div class="' + p[0] + '">' + esc(p[1]) + '</div>'; }).join('')
    : (edited ? '<div class="ok">edited; passes the leak and cursor checks (apply_curation.py re-executes it on apply)</div>' : '');
}

function setField(c, key, value){
  const s = Object.assign({}, store[c.id] || {});
  const e = Object.assign({}, s.edits || {});
  const same = key === 'cursor' ? JSON.stringify(value) === JSON.stringify(orig(c, key)) : value === orig(c, key);
  if (same) delete e[key]; else e[key] = value;
  s.edits = e;
  store[c.id] = s;
}

function persist(id){
  const body = Object.assign({}, store[id] || {}, {updated: new Date().toISOString()});
  store[id] = body;
  try { localStorage.setItem(LKEY, JSON.stringify(store)); } catch (e) {}
  if (!db) return;
  queue[id] = (queue[id] || Promise.resolve()).then(function(){
    return db.collection('decisions').doc(id).set(body).catch(function(err){
      setStatus('error', 'Could not save ' + id + ' to the shared store (' + (err && err.code) + '). It is kept in this browser; Export hands it over.');
    });
  });
}
function persistSoon(id){ clearTimeout(timers[id]); timers[id] = setTimeout(function(){ persist(id); }, 700); }
function setStatus(kind, text){ const el = document.getElementById('save'); el.className = 'save ' + kind; el.textContent = text; }

function decided(c){ const s = store[c.id] || {}; return !!(s.verdict || isEdited(c)); }
function filter(){
  const q = fQuery.trim().toLowerCase();
  view = DATA.filter(function(c){
    if (fNever !== 'all' && String(c.never) !== fNever) return false;
    if (fStable === 'stable' && !c.stable) return false;
    if (fCat !== 'all' && c.cat !== fCat) return false;
    if (fHist === 'reviewed' && !c.hist) return false;
    if (fHist === 'fresh' && c.hist) return false;
    if (fState === 'todo' && decided(c)) return false;
    if (fState === 'done' && !decided(c)) return false;
    if (fNote === 'none' && c.note) return false;
    if (fNote === 'sugg' && !hasSugg(c)) return false;
    if (NOTE_CLASSES[fNote] && !(c.note && c.note.cls === fNote)) return false;
    if (q && [c.id, c.ins, c.cmd, c.note ? c.note.text : ''].join('\n').toLowerCase().indexOf(q) < 0) return false;
    return true;
  });
  page = Math.min(page, Math.max(0, Math.ceil(view.length / PAGE) - 1));
}
function render(){
  const start = page * PAGE, slice = view.slice(start, start + PAGE);
  document.getElementById('list').innerHTML = slice.length ? slice.map(cardHtml).join('')
    : '<div class="empty">No rows match these filters.</div>';
  slice.forEach(paint);
  const pages = Math.max(1, Math.ceil(view.length / PAGE));
  const pager = '<button class="btn" data-page="prev"' + (page === 0 ? ' disabled' : '') + '>Previous</button>'
    + '<span>page ' + (page + 1) + ' of ' + pages + ' &middot; rows ' + (view.length ? start + 1 : 0) + '–'
    + Math.min(view.length, start + PAGE) + ' of ' + view.length + '</span>'
    + '<button class="btn" data-page="next"' + (page >= pages - 1 ? ' disabled' : '') + '>Next</button>';
  document.getElementById('pager-top').innerHTML = pager;
  document.getElementById('pager').innerHTML = pager;
  progress();
}
function progress(){
  const done = DATA.filter(decided).length;
  document.getElementById('prog').textContent = done + ' / ' + DATA.length;
  document.getElementById('bar').style.width = (100 * done / DATA.length) + '%';
}

document.addEventListener('click', function(ev){
  const pg = ev.target.closest('[data-page]');
  if (pg){ page += pg.getAttribute('data-page') === 'next' ? 1 : -1; render();
           document.querySelector('main').scrollTop = 0; return; }
  const sg = ev.target.closest('[data-sugg]');
  if (sg){ const c = BY_ID[sg.getAttribute('data-sugg')], s = c.note.sugg;
           Object.keys(s).forEach(function(k){ setField(c, k, k === 'cursor' ? s[k].slice() : s[k]); });
           if (['command', 'mode', 'cursor', 'buffer_before'].some(function(k){ return k in s; })){
             const d = document.getElementById('c-' + c.id).querySelector('details.more');
             if (d) d.open = true;
           }
           paint(c); progress(); persist(c.id); return; }
  const use = ev.target.closest('[data-use]');
  if (use){ const c = BY_ID[use.getAttribute('data-use')];
            setField(c, 'command', c.alt.cmd); if (c.alt.mode !== c.mode) setField(c, 'mode', c.alt.mode);
            paint(c); progress(); persist(c.id); return; }
  const rv = ev.target.closest('[data-revert]');
  if (rv){ const c = BY_ID[rv.getAttribute('data-revert')];
           const s = Object.assign({}, store[c.id] || {}); s.edits = {}; store[c.id] = s;
           paint(c); progress(); persist(c.id); return; }
  const b = ev.target.closest('.vb[data-act]');
  if (b){ const c = BY_ID[b.getAttribute('data-id')], act = b.getAttribute('data-act');
          const s = Object.assign({}, store[c.id] || {}); s.verdict = s.verdict === act ? null : act; store[c.id] = s;
          paint(c); progress(); persist(c.id); }
});
function onEdit(ev){
  const t = ev.target, id = t.getAttribute && t.getAttribute('data-id'), field = t.getAttribute && t.getAttribute('data-field');
  if (!id || !field) return;
  const c = BY_ID[id];
  if (field === 'note'){ store[id] = Object.assign({}, store[id] || {}, {note: t.value}); persistSoon(id); return; }
  if (field === 'cursorline' || field === 'cursorcol'){
    const cur = val(c, 'cursor').slice();
    const n = parseInt(t.value, 10);
    cur[field === 'cursorline' ? 0 : 1] = isNaN(n) ? 0 : n;
    setField(c, 'cursor', cur);
  } else setField(c, field, t.value);
  paint(c); progress(); persistSoon(id);
}
document.getElementById('list').addEventListener('input', onEdit);
document.getElementById('list').addEventListener('change', onEdit);

document.querySelectorAll('[data-never]').forEach(function(b){
  b.addEventListener('click', function(){
    fNever = b.getAttribute('data-never');
    document.querySelectorAll('[data-never]').forEach(function(x){ x.setAttribute('aria-pressed', String(x === b)); });
    page = 0; filter(); render();
  });
});
document.getElementById('cat').addEventListener('change', function(ev){ fCat = ev.target.value; page = 0; filter(); render(); });
document.getElementById('hist').addEventListener('change', function(ev){ fHist = ev.target.value; page = 0; filter(); render(); });
document.getElementById('state').addEventListener('change', function(ev){ fState = ev.target.value; page = 0; filter(); render(); });
document.getElementById('cls').addEventListener('change', function(ev){ fNote = ev.target.value; page = 0; filter(); render(); });
document.getElementById('q').addEventListener('input', function(ev){ fQuery = ev.target.value; page = 0; filter(); render(); });
document.getElementById('stable').value = fStable;
document.getElementById('stable').addEventListener('change', function(ev){ fStable = ev.target.value; page = 0; filter(); render(); });

function changeSet(){
  const out = {edits: {}, deletes: {}, meta_edits: {}, review: {}, refused: []};
  DATA.forEach(function(c){
    const s = store[c.id];
    if (!s) return;
    const e = s.edits || {};
    if (s.verdict || s.note || Object.keys(e).length) out.review[c.id] = {verdict: s.verdict || (Object.keys(e).length ? 'edited' : null), note: s.note || ''};
    if (s.verdict === 'drop'){ out.deletes[c.id] = s.note || 'dropped in the never-solved train review: not a fair row'; return; }
    const bad = problems(c).filter(function(p){ return p[0] === 'bad'; });
    if (bad.length && Object.keys(e).length){ out.refused.push(c.id + ': ' + bad[0][1]); return; }
    const core = {};
    ['instruction', 'command', 'mode', 'cursor', 'buffer_before'].forEach(function(f){ if (e[f] !== undefined) core[f] = f === 'instruction' ? String(e[f]).trim() : e[f]; });
    if (Object.keys(core).length) out.edits[c.id] = core;
    const meta = {};
    ['category', 'tier'].forEach(function(f){ if (e[f] !== undefined) meta[f] = e[f]; });
    if (Object.keys(meta).length) out.meta_edits[c.id] = meta;
  });
  return out;
}
document.getElementById('export').addEventListener('click', function(){
  const out = changeSet();
  const refused = out.refused; delete out.refused;
  document.getElementById('out').value = JSON.stringify(out, null, 1);
  document.getElementById('dlgsub').textContent = Object.keys(out.review).length + ' reviewed, '
    + Object.keys(out.edits).length + ' edit(s), ' + Object.keys(out.deletes).length + ' drop(s), '
    + Object.keys(out.meta_edits).length + ' category/tier change(s)'
    + (refused.length ? '; left out, a check fails: ' + refused.join('; ') : '') + '.';
  document.getElementById('copied').textContent = '';
  document.getElementById('dlg').showModal();
});
document.getElementById('copy').addEventListener('click', function(){
  const out = document.getElementById('out');
  const done = function(){ document.getElementById('copied').textContent = 'Copied'; };
  try { navigator.clipboard.writeText(out.value).then(done, function(){ out.select(); }); } catch (e) { out.select(); }
});
document.getElementById('close').addEventListener('click', function(){ document.getElementById('dlg').close(); });

filter(); render();

(async function(){
  let d = null;
  try { d = window.claude && window.claude.use ? await window.claude.use('db') : null; } catch (e) { d = null; }
  if (!d){ setStatus('', 'Saving in this browser only. Export hands the decisions over.'); return; }
  db = d;
  setStatus('shared', 'Saving to this page’s shared store. Claude reads it back.');
  const local = Object.assign({}, store);
  let first = true;
  db.collection('decisions').onSnapshot(function(snap){
    const next = {};
    snap.docs.forEach(function(doc){ if (doc.exists) next[doc.id] = Object.assign({}, doc.data()); });
    if (first){
      first = false;
      Object.keys(local).forEach(function(id){ if (!next[id]){ next[id] = local[id]; store[id] = local[id]; persist(id); } });
    }
    const focusId = document.activeElement && document.activeElement.getAttribute && document.activeElement.getAttribute('data-id');
    Object.keys(next).forEach(function(id){ if (id !== focusId && !timers[id]) store[id] = next[id]; });
    const start = page * PAGE;
    view.slice(start, start + PAGE).forEach(paint);
    progress();
  }, function(err){
    setStatus('error', 'The shared store stopped answering (' + (err && err.code) + '). New decisions stay in this browser; Export hands them over.');
    db = null;
  });
})();
"""

LEDE = """\
  <p class="lede">
    These are the <b>%(n)d train rows</b> that at least one of the four probed adapters
    never solved in 8 samples. <b>%(all4)d</b> are never solved by any of the four. A row
    nothing solves is either beyond the model or a bad row, and only a reader can tell
    which. <b>%(reviewed)d</b> of these rows were already handled on 2026-09-16, in the
    cliff review or the whitespace repair; their green chip says what was done.
  </p>
  <p class="lede" style="margin-top:10px">
    Each card shows how each adapter did, its most common answers, and the buffer the
    most common runnable wrong answer produces. You can edit any part of a row: the
    instruction in place, and the gold command, mode, cursor, buffer, category and tier
    under the card. The expected output is never typed. It is whatever the gold command
    produces, recomputed when the change-set is applied. The curation guard&rsquo;s checks
    run as you type. These are train rows, so fixing them is how the next training
    round learns from them.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>Claude&rsquo;s read.</b> %(noted)d of the %(n)d rows carry a note from Claude on why it
    thinks the row fails, written hardest rows first; the rest are still being read. Each
    note has one of five classes:
  </p>
  <div class="legend">%(legend)s</div>
  <p class="lede" style="margin-top:10px">
    When a note proposes a fix, <b>Use suggestion</b> copies it into the editor, where the
    checks run on it. It becomes your edit only if you keep it; Revert under the card
    undoes it. A proposed command also shows the output it produces beside the
    gold&rsquo;s. The <i>Claude&rsquo;s read</i> filter picks out one class, or every row
    with a suggestion.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>Start with the %(stable)d rows marked <i>stable wrong</i>.</b> On 2026-10-16 the served
    adapter was re-sampled 32 times on the rows its 8-sample probe never solved. These are
    still never solved, and give the same runnable answer in at least 24 of 32 tries. A model
    that consistently disagrees with the gold is often right: in the 2026-09-16 cliff review,
    83 of 292 rows like this were reworded, retargeted or dropped. So these are the likeliest
    place for a bad instruction or gold. The <i>32-sample re-check</i> filter shows only them;
    each card shows the stable answer, how often it came up, and the buffer it produces.
  </p>
"""


def build(cards, cats):
    n = len(cards)
    counts = collections.Counter(c["never"] for c in cards)
    reviewed = sum(1 for c in cards if c["hist"])
    by_cls = collections.Counter(c["note"]["cls"] for c in cards if "note" in c)
    noted = sum(by_cls.values())
    n_sugg = sum(1 for c in cards if "note" in c and c["note"]["sugg"])
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    never_btns = "".join('<button class="f" data-never="%d" aria-pressed="false">%d of 4 '
                         '<span style="opacity:.6">%d</span></button>' % (k, k, counts[k])
                         for k in (4, 3, 2, 1))
    cat_opts = "".join('<option value="%s">%s</option>' % (c, c) for c in cats)
    cls_opts = "".join('<option value="%s">%s (%d)</option>' % (k, k, by_cls[k]) for k in NOTE_CLASSES) \
        + '<option value="sugg">with a suggestion (%d)</option>' % n_sugg \
        + '<option value="none">not read yet (%d)</option>' % (n - noted)
    legend = "".join('<span class="chip cl-%s">%s</span><span>%s</span>' % (k, k, html.escape(v))
                     for k, v in NOTE_CLASSES.items())
    n_stable = sum(1 for c in cards if c.get("stable"))
    lede = LEDE % {"n": n, "all4": counts[4], "reviewed": reviewed, "noted": noted, "legend": legend,
                   "stable": n_stable}
    return """<title>Never-Solved Train Rows</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&\
family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%s%s</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; difficulty probes &middot; hand review of the never-solved train rows</div>
  <details id="intro" open>
  <summary><h1>Why does no adapter solve these?</h1>
    <span class="hint" aria-hidden="true"></span></summary>
%s  </details>
  <div class="band">
    <div><span class="v">%d</span><span class="k">train rows never solved by some adapter</span></div>
    <div><span class="v" style="color:var(--stop)">%d</span><span class="k">never solved by all four</span></div>
    <div><span class="v" style="color:var(--ok)">%d</span><span class="k">handled on 2026-09-16</span></div>
    <div><span class="v" style="color:var(--accent)">%d</span><span class="k">read by Claude</span></div>
    <div><span class="v" id="prog">0 / %d</span><span class="k">decided here</span></div>
  </div>
</div></header>

<div class="dock"><div class="wrap">
  <span class="save" id="save">Connecting to the shared store&hellip;</span>
  <div class="bar" aria-hidden="true"><i id="bar" style="width:0%%"></i></div>
  <button class="btn pri" id="export">Export change-set</button>
</div></div>

<div class="rail"><div class="wrap">
  <span class="lab">never solved by</span>
  <button class="f" data-never="all" aria-pressed="true">any</button>%s
  <label class="lab" for="stable">32-sample re-check</label>
  <select id="stable"><option value="all">any</option><option value="stable">stable wrong answer (%d)</option></select>
  <label class="lab" for="cat">category</label>
  <select id="cat"><option value="all">all</option>%s</select>
  <label class="lab" for="hist">cliff review</label>
  <select id="hist"><option value="all">any</option><option value="fresh">not reviewed</option>
    <option value="reviewed">reviewed</option></select>
  <label class="lab" for="state">here</label>
  <select id="state"><option value="all">any</option><option value="todo">not decided</option>
    <option value="done">decided</option></select>
  <label class="lab" for="cls">Claude&rsquo;s read</label>
  <select id="cls"><option value="all">any</option>%s</select>
  <input class="search" id="q" type="search" placeholder="search id, text, command" aria-label="search">
</div></div>

<main><div class="wrap">
  <div class="pager" id="pager-top"></div>
  <div class="cards" id="list"></div>
  <div class="pager" id="pager"></div>
</div></main>

<dialog id="dlg" aria-labelledby="dlgh">
  <div class="dh"><h2 id="dlgh">Change-set for apply_curation.py</h2>
    <p class="lede" id="dlgsub" style="font-size:13.5px"></p></div>
  <div class="db">
    <textarea id="out" readonly aria-label="change-set JSON"></textarea>
    <div class="fixrow"><button class="btn pri" id="copy">Copy</button>
      <button class="btn" id="close">Close</button><span class="copied" id="copied"></span></div>
  </div>
</dialog>

<script type="application/json" id="payload">%s</script>
<script type="application/json" id="states">%s</script>
<script type="application/json" id="cats">%s</script>
<script type="application/json" id="classes">%s</script>
<script>%s</script>
""" % (BASE_CSS, CSS, lede,
       n, counts[4], reviewed, noted, n, never_btns, n_stable, cat_opts, cls_opts, payload,
       json.dumps(STATE_WORDS, ensure_ascii=False), json.dumps(cats),
       json.dumps(NOTE_CLASSES, ensure_ascii=False).replace("</", "<\\/"), JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="page to write (HTML)")
    args = ap.parse_args()
    cards, cats = collect()
    attach_notes(cards, load_notes())
    page = build(cards, cats)
    print(check_page(page, required_js=["split('\\n')", "replace(/\\t/g", "/<[^>]*>/g"],
                     name="never-solved review"))
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    by = collections.Counter(c["never"] for c in cards)
    print("rows: %d  by never-count %s  reviewed before: %d  read by Claude: %d  page: %d KB -> %s"
          % (len(cards), dict(sorted(by.items())), sum(1 for c in cards if c["hist"]),
             sum(1 for c in cards if "note" in c), len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
