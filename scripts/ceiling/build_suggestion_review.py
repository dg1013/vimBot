#!/usr/bin/env python3
"""Build the hand-review page for the suggested rewordings: the rewords of the
always-failing review that would still fail a correct answer, each with the
owner's wording, a suggested one, and why the suggestion was made.

The always-failing review (build_core_review.py, 2026-09-25) produced 28
rewords. Read against their buffers and golds, 13 of them either change the
meaning or leave the gold's exact form unstated, so a faithful answer written
another way still fails. The owner asked to see each suggestion with its
reason and decide by hand: take the suggestion, keep their own wording, or
edit either. A live copy of the curation guard's leak check (the instruction
must not contain the gold command) runs on every edit, so a wording the guard
would refuse is caught on the page rather than at apply time.

Not on this page, because they are settled: the 15 rewords that stand as
written, the 16 drops, and the three proposals the owner approved for
hc-0381 (retarget to $(BUILD)), hc-1550 and sregex-0102a.

Decisions are saved to the page's shared store (collection `choice`, one
document per row id), where Claude reads them back; Export shows them as an
apply_curation.py change-set fragment as well.

Usage:
    python scripts/ceiling/build_suggestion_review.py --out page.html
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl                 # noqa: E402
from build_test_review import CSS as BASE_CSS   # noqa: E402  (the review pages' look)
from page_guard import check_page               # noqa: E402
from apply_instructions import leaks            # noqa: E402

OWNER = REPO / "runs" / "ceiling_2026-09-22" / "review" / "owner_changeset_2026-09-25.json"

# id -> (kind, suggested instruction, why). kind: "meaning" -- the reword
# changes what is asked or adds an error; "form" -- the gold's exact form is
# still unstated, so a right answer written another way fails; "typo";
# "optional".
SUGGESTIONS = {
    "hc-1908": ("meaning",
        "Turn every setting line into an insert row for the kv table and leave the comment "
        "as it is for this SQL command",
        "The gold writes INSERT INTO kv, and the table name kv appears nowhere: not in the "
        "buffer (-- seed, alpha=1, beta=2) and not in either wording. “For the table "
        "settings” reads as a table named settings, so a faithful answer, INSERT INTO "
        "settings, fails. Naming the table is the only way a model can know it."),
    "sregex-0013a": ("meaning",
        "Keep only lines made only of letters that are at least 6 characters long, contain "
        "two capitals or two lowercase letters in a row, do not contain 'ecz', and do not "
        "end with a lowercase letter.",
        "The gold keeps a line when all five hold: letters only, at least 6 long, two "
        "capitals or two lowercase letters in a row, no 'ecz', and not ending in a lowercase "
        "letter. The wording keeps two errors from the original description: “numbers” "
        "(the gold allows letters only) and “lowercase underscore” (the gold means a "
        "lowercase letter). The models encoded a literal underscore. The suggestion states "
        "the five conditions directly."),
    "hc-3072": ("meaning",
        "Can we halve the sampling rate here? Leave the header where it is, then below it "
        "discard every other reading, starting with the one on line three.",
        "“Every duplicate reading” admits a second reading. The pressures come in pairs, "
        "so dropping every other line removes each pair’s second reading, which is what "
        "the gold does. But 09:06 (101.4) and 09:08 (101.3) also repeat earlier values "
        "(09:02, 09:00), and the gold keeps them. “Every other reading” says exactly "
        "what the gold does."),
    "hc-1369": ("meaning",
        "Sort the rows by region alphabetically, and within each region by the number, "
        "largest first",
        "The sentence lost its structure (“and, inside each region and eliminate ties by "
        "having…”), and “eliminate ties” can be read as removing rows. The gold "
        "sorts by region A to Z and, within a region, by the number from largest to "
        "smallest."),
    "kb13-0428a": ("meaning",
        "Keep only lines containing the whole word 'trade' in lowercase.",
        "The gold keeps 'trade' only as a whole word (\\<trade\\>): it deletes 'tradex' and "
        "'xtrade'. “No other variations” can be read as “no other capitalizations”, "
        "which “in lowercase” already says. “Whole word” names the rule the gold "
        "applies."),
    "hc-0281": ("form",
        "Add the missing separator row of dashes under the table header, each cell as wide "
        "as the header cell above it",
        "The gold’s separator is |------|-----|: each cell exactly as wide as the header "
        "cell above it (6 and 5 dashes). The common Markdown form |---|---| is a valid "
        "separator but a different buffer, so “made of dashes” still lets a right "
        "answer fail. Stating the width pins it down."),
    "hc-1804": ("form",
        "Fold these variables into a compose environment list for YAML, indented to sit "
        "under a service: the key at four spaces, each item at six",
        "The gold indents environment: by 4 spaces and each item by 6, as if under a compose "
        "service. Nothing in the buffer or the wording implies that, so an unindented list, "
        "which is valid YAML, fails. The alternative to stating it is to change the gold to "
        "an unindented list."),
    "hc-1539": ("form",
        "For the YAML snippet, turn the inline trigger list into a block list with each item "
        "indented four spaces",
        "The gold puts each list item at 4 spaces, but the file indents by 2 (build: sits two "
        "spaces under jobs:), so a model matching the file’s style indents the items by 2 and "
        "fails. "
        "Stating four spaces makes the gold the only faithful reading."),
    "hc-1752": ("form",
        "Collapse this cipher list back into a single line without the trailing comma or any "
        "space just inside the brackets",
        "Joining the lines leaves [ \"TLS…\" ] with a space just inside each bracket. The "
        "gold also removes those, giving [\"TLS…\"]. The wording does not ask for it, so a "
        "plain join-and-trim answer fails."),
    "hc-1381": ("form",
        "Reduce this diff output to just the replacement lines, without their > markers",
        "The buffer is plain diff output (2c2, <, ---, >), not a git diff, and the gold does "
        "not swap anything in: it reduces the output to the replacement lines and strips "
        "their “> ” markers. “Swap in … from this git diff” suggests applying the "
        "change to a file that is not there. The original wording was closer; the suggestion "
        "states the result."),
    "cand-cm-0089": ("typo",
        "Uppercase the first two letters on the first three lines",
        "A stray “a”: “Uppercase a the first two letters”. The rest is right: the gold "
        "uppercases the first two columns of lines 1 to 3."),
    "hc-0854": ("typo",
        "Turn this Python code into a plain YAML mapping with lowercase booleans",
        "“These Python cope” should read “this Python code” (the original said "
        "“toggles”). The meaning is otherwise unchanged."),
    "hc-2166": ("optional",
        "Resolve this conflict in favour of HEAD: keep its line and remove the incoming one "
        "along with the markers.",
        "The wording no longer says the marker lines go, and the gold also deletes the "
        "<<<<<<<, ======= and >>>>>>> lines. Resolving a conflict usually implies it, so this "
        "is optional: it removes the doubt rather than fixing an error."),
}
KIND_ORDER = {"meaning": 0, "form": 1, "typo": 2, "optional": 3}
KIND_LABEL = {"meaning": "changes the meaning", "form": "gold form unstated",
              "typo": "typo", "optional": "optional"}


def collect(owner_path=OWNER):
    cases = {c["id"]: c for c in load_jsonl(str(REPO / "data" / "splits" / "test.jsonl"))}
    owner = json.loads(pathlib.Path(owner_path).read_text(encoding="utf-8"))
    cards = []
    for cid, (kind, text, why) in SUGGESTIONS.items():
        c = cases[cid]
        mine = owner["edits"].get(cid, {}).get("instruction")
        if mine is None:
            sys.exit("%s has no owner reword in %s" % (cid, owner_path))
        cmd = c["gold_command"].lstrip(":") if c.get("mode") == "ex" else c["gold_command"]
        why_leak = leaks(text, cmd, cmd)
        if why_leak:
            sys.exit("%s: the suggestion itself fails the leak guard: %s" % (cid, why_leak))
        cards.append({"id": cid, "cat": c["category"], "tier": c.get("tier") or "T?",
                      "mode": c.get("mode", "ex"), "cmd": c["gold_command"],
                      "guardcmd": cmd, "cur": list(c.get("cursor") or (1, 1)),
                      "orig": c["instruction"], "mine": mine, "sugg": text, "why": why,
                      "kind": kind, "kindlabel": KIND_LABEL[kind],
                      "bef": c["buffer_before"], "exp": c["buffer_after"]})
    cards.sort(key=lambda x: (KIND_ORDER[x["kind"]], x["cat"], x["id"]))
    return cards


CSS = r"""
.card.done{ border-left-color:var(--ok); }
.chip.k-meaning{ background:var(--stop-bg); color:var(--stop); border:1px solid var(--stop-line);
  font-weight:700; }
.chip.k-form{ background:var(--warn-bg); color:var(--warn); border:1px solid var(--warn-line);
  font-weight:700; }
.chip.k-typo, .chip.k-optional{ background:var(--raise); color:var(--ink2); font-weight:700; }
.vers{ display:flex; flex-direction:column; gap:1px; background:var(--line2);
  border-top:1px solid var(--line2); border-bottom:1px solid var(--line2); }
.ver{ background:var(--surface); display:grid; grid-template-columns:118px 1fr auto;
  gap:6px 12px; align-items:start; padding:9px 14px; }
.ver .vl{ font-family:var(--mono); font-size:10px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); font-weight:700; padding-top:3px; }
.ver .vt{ font-family:var(--disp); font-size:15.5px; line-height:1.45; color:var(--ink);
  overflow-wrap:anywhere; }
.ver.orig .vt{ color:var(--muted); }
.ver.picked{ background:var(--ok-bg); }
.ver.picked .vl{ color:var(--ok); }
@media (max-width:620px){ .ver{ grid-template-columns:1fr; } }
.why{ margin:10px 15px; padding:9px 12px; border-radius:6px; font-size:13.5px;
  line-height:1.55; background:var(--accent-soft); color:var(--ink); max-width:80ch; }
.why b{ font-family:var(--mono); font-size:10px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--accent); margin-right:8px; }
.final{ display:flex; flex-direction:column; gap:6px; padding:10px 14px;
  border-top:1px solid var(--line2); background:var(--sunk); }
.final label{ font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
.final textarea{ font-family:var(--body); font-size:14.5px; padding:7px 9px; min-height:56px;
  border:1px solid var(--line); border-radius:5px; background:var(--surface); color:var(--ink);
  resize:vertical; width:100%; }
.guard{ font-family:var(--mono); font-size:11.5px; }
.guard.ok{ color:var(--ok); } .guard.bad{ color:var(--stop); } .guard.same{ color:var(--warn); }
.save{ font-family:var(--mono); font-size:11px; padding:3px 9px; border-radius:5px;
  border:1px solid var(--line); color:var(--muted); background:var(--ground); }
.save.shared{ color:var(--ok); border-color:var(--ok-line); background:var(--ok-bg); }
.save.error{ color:var(--stop); border-color:var(--stop-line); background:var(--stop-bg); }
.copied{ font-family:var(--mono); font-size:11px; color:var(--ok); }
.fixrow{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('payload').textContent);
const LKEY = 'vimbot-reword-suggestions-v1';
let store = {};                // id -> {choice: 'sugg'|'mine'|'edited', text, updated}
let db = null;
const queue = {}, timers = {};
try { store = JSON.parse(localStorage.getItem(LKEY) || '{}'); } catch (e) { store = {}; }

function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;');
}
function vis(line){
  let out = esc(line).replace(/\t/g, '<span class="ws">--&gt;</span>\t');
  const m = out.match(/( +)$/);
  if (m) out = out.slice(0, out.length - m[1].length)
              + '<span class="ws">' + '.'.repeat(m[1].length) + '</span>';
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
    rows += '<div class="ln' + cls + '"><span class="g">' + (i+1)
          + '</span><span class="t">' + vis(a[i]) + '</span></div>';
  }
  if (!rows) rows = '<div class="ln"><span class="g">-</span>'
                  + '<span class="t">(empty buffer)</span></div>';
  return '<div class="pane"><span class="lab">' + label + '</span>' + rows + '</div>';
}

// The curation guard's leak check (scripts/apply_instructions.py `leaks`),
// ported so a wording it would refuse is caught here rather than at apply.
function leakOf(ins, cmd){
  const low = ins.toLowerCase();
  const f0 = cmd.trim();
  const noTags = f0.replace(/<[^>]*>/g, '');
  const forms = [f0, noTags, noTags.replace(/^:+/, '')];
  for (let i = 0; i < forms.length; i++){
    const f = forms[i].trim().toLowerCase();
    if (f.length >= 2 && low.indexOf(f) >= 0) return 'contains the gold command “' + f + '” verbatim';
  }
  if (/\bpress\b|\bkeystroke|\btype the key/.test(low)) return 'describes keystrokes rather than intent';
  return null;
}
function norm(s){ return String(s).trim().replace(/\s+/g, ' '); }

function textFor(c){
  const s = store[c.id];
  if (!s || !s.choice) return c.sugg;
  if (s.choice === 'mine') return c.mine;
  if (s.choice === 'sugg') return c.sugg;
  return s.text != null ? s.text : c.sugg;
}

function cardHtml(c){
  function ver(cls, label, text, btn){
    return '<div class="ver ' + cls + '" data-ver="' + cls + '"><span class="vl">' + label
      + '</span><span class="vt">' + esc(text) + '</span>'
      + (btn ? '<button class="vb" data-pick="' + cls + '" data-id="' + esc(c.id) + '">'
             + btn + '</button>' : '<span></span>')
      + '</div>';
  }
  return '<div class="card" id="c-' + esc(c.id) + '">'
    + '<div class="chead"><span class="cid">' + esc(c.id) + '</span>'
    +   '<span class="keys">' + esc(c.mode + ': ' + c.cmd) + '</span>'
    +   '<span class="chip">' + esc(c.cat) + '</span>'
    +   '<span class="chip">' + esc(c.tier) + '</span>'
    +   '<span class="chip">cursor ' + c.cur[0] + ',' + c.cur[1] + '</span>'
    +   '<span class="spacer"></span>'
    +   '<span class="chip k-' + esc(c.kind) + '">' + esc(c.kindlabel) + '</span></div>'
    + '<div class="vers">'
    +   ver('orig', 'in the test slice', c.orig, '')
    +   ver('mine', 'your reword', c.mine, 'Keep mine')
    +   ver('sugg', 'suggested', c.sugg, 'Use suggestion')
    + '</div>'
    + '<div class="why"><b>why</b>' + esc(c.why) + '</div>'
    + '<div class="panes">'
    +   pane('input buffer', c.bef, c.exp, 'cut')
    +   pane('gold output', c.exp, c.bef, 'add')
    + '</div>'
    + '<div class="final"><label for="t-' + esc(c.id) + '">instruction that goes into the '
    +   'test slice (edit freely)</label>'
    +   '<textarea id="t-' + esc(c.id) + '" data-id="' + esc(c.id) + '" spellcheck="true"></textarea>'
    +   '<span class="guard" id="g-' + esc(c.id) + '"></span></div>'
    + '</div>';
}

function paint(id){
  const c = DATA.find(function(x){ return x.id === id; });
  const el = document.getElementById('c-' + id);
  const s = store[id] || {};
  el.className = 'card' + (s.choice ? ' done' : '');
  el.querySelectorAll('.ver').forEach(function(v){
    const which = v.getAttribute('data-ver');
    v.classList.toggle('picked', s.choice === which);
  });
  const ta = document.getElementById('t-' + id);
  if (document.activeElement !== ta) ta.value = textFor(c);
  guard(c, ta.value);
}
function guard(c, text){
  const g = document.getElementById('g-' + c.id);
  const s = store[c.id] || {};
  const why = !norm(text) ? 'the instruction is empty' : leakOf(text, c.guardcmd);
  if (why){ g.className = 'guard bad'; g.textContent = 'The curation guard would refuse this: ' + why + '.'; return; }
  if (norm(text) === norm(c.orig)){
    g.className = 'guard same';
    g.textContent = 'Same as the test slice already says: the row stays as it is.';
    return;
  }
  g.className = 'guard ok';
  g.textContent = s.choice ? ('Decided: ' + (s.choice === 'sugg' ? 'the suggestion'
                                             : s.choice === 'mine' ? 'your reword' : 'your edit')
                              + '. Passes the leak check.')
                           : 'Not decided yet: pick a version or edit the text. Passes the leak check.';
}

function persist(id){
  const body = Object.assign({}, store[id] || {}, {updated: new Date().toISOString()});
  store[id] = body;
  try { localStorage.setItem(LKEY, JSON.stringify(store)); } catch (e) {}
  if (!db) return;
  queue[id] = (queue[id] || Promise.resolve()).then(function(){
    return db.collection('choice').doc(id).set(body).catch(function(err){
      setStatus('error', 'Could not save ' + id + ' to the shared store (' + (err && err.code)
                + '). It is kept in this browser; Export hands it over.');
    });
  });
}
function persistSoon(id){
  clearTimeout(timers[id]);
  timers[id] = setTimeout(function(){ persist(id); }, 700);
}
function setStatus(kind, text){
  const el = document.getElementById('save');
  el.className = 'save ' + kind;
  el.textContent = text;
}
function progress(){
  const done = DATA.filter(function(c){ return (store[c.id] || {}).choice; }).length;
  document.getElementById('prog').textContent = done + ' / ' + DATA.length;
  document.getElementById('bar').style.width = (100 * done / DATA.length) + '%';
}

document.getElementById('list').innerHTML = DATA.map(cardHtml).join('');
DATA.forEach(function(c){ paint(c.id); });
progress();

document.getElementById('list').addEventListener('click', function(ev){
  const b = ev.target.closest('[data-pick]');
  if (!b) return;
  const id = b.getAttribute('data-id'), which = b.getAttribute('data-pick');
  const c = DATA.find(function(x){ return x.id === id; });
  store[id] = {choice: which, text: which === 'mine' ? c.mine : c.sugg};
  document.getElementById('t-' + id).value = store[id].text;
  paint(id); progress(); persist(id);
});
document.getElementById('list').addEventListener('input', function(ev){
  const t = ev.target;
  const id = t.getAttribute('data-id');
  if (!id || t.tagName !== 'TEXTAREA') return;
  const c = DATA.find(function(x){ return x.id === id; });
  const choice = t.value === c.sugg ? 'sugg' : t.value === c.mine ? 'mine' : 'edited';
  store[id] = {choice: choice, text: t.value};
  paint(id); progress(); persistSoon(id);
});

function changeSet(){
  const out = {edits: {}, undecided: [], refused: []};
  DATA.forEach(function(c){
    const s = store[c.id];
    if (!s || !s.choice){ out.undecided.push(c.id); return; }
    const text = s.choice === 'mine' ? c.mine : s.choice === 'sugg' ? c.sugg : (s.text || '');
    const why = !norm(text) ? 'empty' : leakOf(text, c.guardcmd);
    if (why){ out.refused.push(c.id + ': ' + why); return; }
    if (norm(text) !== norm(c.orig)) out.edits[c.id] = {instruction: text.trim()};
  });
  return out;
}
document.getElementById('export').addEventListener('click', function(){
  const out = changeSet();
  document.getElementById('out').value = JSON.stringify({edits: out.edits}, null, 1);
  document.getElementById('dlgsub').textContent = Object.keys(out.edits).length + ' edit(s)'
    + (out.undecided.length ? '; not decided yet: ' + out.undecided.join(', ') : '')
    + (out.refused.length ? '; left out, the guard would refuse: ' + out.refused.join('; ') : '')
    + '.';
  document.getElementById('copied').textContent = '';
  document.getElementById('dlg').showModal();
});
document.getElementById('copy').addEventListener('click', function(){
  const out = document.getElementById('out');
  const done = function(){ document.getElementById('copied').textContent = 'Copied'; };
  try { navigator.clipboard.writeText(out.value).then(done, function(){ out.select(); }); }
  catch (e) { out.select(); }
});
document.getElementById('close').addEventListener('click', function(){
  document.getElementById('dlg').close();
});

(async function(){
  let d = null;
  try { d = window.claude && window.claude.use ? await window.claude.use('db') : null; }
  catch (e) { d = null; }
  if (!d){ setStatus('', 'Saving in this browser only. Export hands the decisions over.'); return; }
  db = d;
  setStatus('shared', 'Saving to this page’s shared store. Claude reads it back.');
  const local = Object.assign({}, store);
  let first = true;
  db.collection('choice').onSnapshot(function(snap){
    const next = {};
    snap.docs.forEach(function(doc){ if (doc.exists) next[doc.id] = Object.assign({}, doc.data()); });
    if (first){
      first = false;
      Object.keys(local).forEach(function(id){
        if (!next[id] && local[id].choice){ next[id] = local[id]; store[id] = local[id]; persist(id); }
      });
    }
    Object.keys(next).forEach(function(id){
      const ta = document.getElementById('t-' + id);
      if (!(ta && document.activeElement === ta)) store[id] = next[id];
    });
    DATA.forEach(function(c){ paint(c.id); });
    progress();
  }, function(err){
    setStatus('error', 'The shared store stopped answering (' + (err && err.code)
              + '). New decisions stay in this browser; Export hands them over.');
    db = null;
  });
})();
"""

LEDE = """\
  <p class="lede">
    Your always-failing review reworded 28 test rows. Read against their buffers
    and golds, <b>%(n)d of them</b> either change what is asked or leave the gold&rsquo;s
    exact form unstated, so a model that follows the words can still be marked
    wrong. Each card shows the row&rsquo;s current instruction, your reword, a
    suggestion, and <b>why</b> the suggestion was made. Keep yours, take the
    suggestion, or edit either. The box at the bottom of each card is what goes
    into the test slice.
  </p>
  <p class="lede" style="margin-top:10px">
    The curation guard&rsquo;s leak check (the instruction may not contain the gold
    command) runs as you type, so nothing you settle on here is refused at apply
    time. Settled and not on this page: the other 15 rewords as you wrote them,
    the 16 drops, and the three proposals you approved (hc-0381, hc-1550,
    sregex-0102a). Once these %(n)d are decided, the whole change-set is applied in
    one pass and the results table is re-measured on the new slice.
  </p>
"""


def build(cards):
    n = len(cards)
    counts = {k: sum(1 for c in cards if c["kind"] == k) for k in KIND_ORDER}
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    return """<title>Always-Failing Rewords</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&\
family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%s%s</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; always-failing review &middot; suggested rewordings</div>
  <details id="intro" open>
  <summary><h1>Which wording goes into the test slice?</h1>
    <span class="hint" aria-hidden="true"></span></summary>
%s  </details>
  <div class="band">
    <div><span class="v">%d</span><span class="k">rewords with a suggestion</span></div>
    <div><span class="v" style="color:var(--stop)">%d</span><span class="k">change the meaning</span></div>
    <div><span class="v" style="color:var(--warn)">%d</span><span class="k">gold form unstated</span></div>
    <div><span class="v">%d</span><span class="k">typo or optional</span></div>
    <div><span class="v" id="prog">0 / %d</span><span class="k">decided</span></div>
  </div>
</div></header>

<div class="dock"><div class="wrap">
  <span class="save" id="save">Connecting to the shared store&hellip;</span>
  <div class="bar" aria-hidden="true"><i id="bar" style="width:0%%"></i></div>
  <button class="btn pri" id="export">Export change-set</button>
</div></div>

<main><div class="wrap"><div class="cards" id="list"></div></div></main>

<dialog id="dlg" aria-labelledby="dlgh">
  <div class="dh"><h2 id="dlgh">Rewording change-set for apply_curation.py</h2>
    <p class="lede" id="dlgsub" style="font-size:13.5px"></p></div>
  <div class="db">
    <textarea id="out" readonly aria-label="change-set JSON"></textarea>
    <div class="fixrow"><button class="btn pri" id="copy">Copy</button>
      <button class="btn" id="close">Close</button>
      <span class="copied" id="copied"></span></div>
  </div>
</dialog>

<script type="application/json" id="payload">%s</script>
<script>%s</script>
""" % (BASE_CSS, CSS, LEDE % {"n": n}, n, counts["meaning"], counts["form"],
       counts["typo"] + counts["optional"], n, payload, JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="page to write (HTML)")
    ap.add_argument("--owner", default=str(OWNER), help="the owner's exported change-set")
    args = ap.parse_args()
    cards = collect(args.owner)
    page = build(cards)
    print(check_page(page, required_js=["split('\\n')", "replace(/\\t/g", "/<[^>]*>/g"],
                     name="suggestion review"))
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("rows: %d  page: %d KB -> %s" % (len(cards), len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
