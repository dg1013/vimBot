#!/usr/bin/env python3
"""Build the hand-review page for the always-failing test rows: the prompts
no sample of any qwen3.5-4b checkpoint solves in the pass@k run
(`runs/ceiling_2026-09-22`, DECISIONS.md 2026-09-23).

A row nothing solves is either beyond the model or wrongly labelled, and only
a reader can tell which. Each card shows what the review pages always show --
instruction, input, gold output -- plus the evidence this row set adds:

  * what the models answer instead: each trained checkpoint's most common
    answers with their share of 128 samples and verdict, the greedy answers of
    the results-table rows, and any other model family's greedy run that
    solved the row (proof the gold is reachable from the words);
  * the buffer the most common RUNNABLE wrong answer produces, lines that
    differ from the gold output marked. The gold's own re-execution, the
    other review pages' third pane, is left out: the corpus self-check
    passes 7,538 / 7,538.

Priority, sorted first: the rows whose near-unanimous answer follows the
instruction as written while the gold does not (the two found in the pass@k
analysis, and one borderline), then every row where a trained checkpoint gives
one runnable wrong answer in at least half its samples -- the signature of a
label defect, though most such rows are plain model errors.

Verdicts are saved to the page's shared store (the artifact `db` capability),
where Claude reads them back; without it (a local file) they stay in the
browser, and Export shows them as an `apply_curation.py` change-set to hand
over. The page proposes; `apply_curation.py` re-executes every edited row.

Usage:
    python scripts/ceiling/build_core_review.py --out page.html
"""
import argparse
import collections
import glob
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim                    # noqa: E402
from evaluate import _norm, load_jsonl          # noqa: E402
from build_test_review import CSS as BASE_CSS   # noqa: E402  (the review pages' look)
from page_guard import check_page               # noqa: E402

RUN = REPO / "runs" / "ceiling_2026-09-22"
TRAINED = ("sft", "served", "cliffleg")
ALL_LABELS = ("base", "sft", "served", "cliffleg", "base-card")
CONFIDENT = 64          # one wrong answer in at least half of 128 samples
STATE_WORDS = {"valid_wrong": "wrong buffer", "invalid": "vim error",
               "match": "solves it", "parse": "not the §6 grammar"}   # shown on the page

# The pass@k analysis read these three before the review existed. Shown on
# the card as the analyst's reading, for the reviewer to confirm or overrule.
PRE_READ = {
    "kb13-0428a": ("suspected defect",
                   "The instruction asks for lines containing 'trade' in lowercase. "
                   "'tradex' and 'xtrade' contain it, yet the gold deletes them, because "
                   "it requires the whole word (\\<trade\\>). The models keep them: the "
                   "literal reading."),
    "sregex-0102a": ("suspected defect",
                     "The instruction says 2 to 3 lowercase letters; the gold's "
                     "[a-z]\\{3} allows exactly 3. 'XjirINGA' (Xj + ir + INGA) fits the "
                     "instruction, the gold deletes it, and the models keep it."),
    "hc-0809": ("borderline",
                "'br' is brotli's HTTP content-coding token, and the gold writes it; "
                "the instruction says 'brotli', which the models write. Whether the "
                "list wants the token depends on a convention the sentence does not "
                "state."),
}


def load_rows(label):
    with open(RUN / ("samples_%s.jsonl" % label), encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, fh) if r}


def greedy_logs():
    """{run name: {id: exec row}} for every greedy test run in the project."""
    out = {}
    patterns = [REPO / "runs" / "*" / "exec_test.jsonl",
                REPO / "runs" / "*" / "*" / "exec_test.jsonl",
                RUN / "greedy" / "*_exec_test.jsonl"]
    for pattern in patterns:
        for path in sorted(glob.glob(str(pattern))):
            p = pathlib.Path(path)
            if p.parent == RUN / "greedy":
                name = p.name[:-len("_exec_test.jsonl")]
            else:
                name = str(p.parent.relative_to(REPO / "runs"))
            out[name] = {r["id"]: r for r in load_jsonl(path)}
    return out


def top_answers(row, k=3):
    return [{"mode": e["mode"], "cmd": e["command"], "n": e["n"], "state": e["state"]}
            for e in row["commands"][:k]]


def pane_answer(rows_by_label, case_id):
    """The most common RUNNABLE wrong answer pooled over the trained
    checkpoints: {mode, cmd, by: {label: n}} or None. Runnable, because an
    answer vim rejects has no buffer to show."""
    pooled = collections.Counter()
    by = collections.defaultdict(dict)
    for label in TRAINED:
        for e in rows_by_label[label][case_id]["commands"]:
            if e["state"] == "valid_wrong":
                key = (e["mode"], e["command"])
                pooled[key] += e["n"]
                by[key][label] = e["n"]
    if not pooled:
        return None
    (mode, cmd), _ = max(pooled.items(), key=lambda kv: (kv[1], kv[0]))
    return {"mode": mode, "cmd": cmd, "by": by[(mode, cmd)]}


def priority(case_id, rows_by_label, solved_elsewhere):
    """(flag, rank): the chip on the card and its sort position."""
    if case_id in PRE_READ:
        return PRE_READ[case_id][0], 0 if PRE_READ[case_id][0] == "suspected defect" else 1
    for label in TRAINED:
        top = rows_by_label[label][case_id]["commands"][:1]
        if top and top[0]["state"] == "valid_wrong" and top[0]["n"] >= CONFIDENT:
            return "confident disagreement", 2
    if solved_elsewhere:
        return "solved by another model", 3
    return None, 4


def collect(core_ids=None):
    cases = {c["id"]: c for c in load_jsonl(str(REPO / "data" / "splits" / "test.jsonl"))}
    rows = {label: load_rows(label) for label in ALL_LABELS}
    if core_ids is None:
        core_ids = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))["uncovered_core"]
    logs = greedy_logs()
    cards = []
    for cid in core_ids:
        c = cases[cid]
        if any(rows[label][cid]["c"] for label in ALL_LABELS):
            sys.exit("%s is not always-failing: some sample solves it" % cid)
        cursor = tuple(c.get("cursor") or (1, 1))
        elsewhere = [{"run": name, "cmd": log[cid].get("command"), "mode": log[cid].get("mode")}
                     for name, log in logs.items() if cid in log and log[cid].get("match")]
        greedy = [{"run": name, "cmd": logs[name][cid].get("command"),
                   "mode": logs[name][cid].get("mode"),
                   "state": ("match" if logs[name][cid].get("match") else
                             "valid_wrong" if logs[name][cid].get("valid") else "invalid")}
                  for name in ("qwen3.5-4b-grpo-gated/fla-kernels", "qwen3.5-4b-sft")
                  if name in logs and cid in logs[name]]
        alt = pane_answer(rows, cid)
        if alt is not None:
            res = run_vim(c["buffer_before"], alt["cmd"], mode=alt["mode"], cursor=cursor)
            alt["buf"] = res.get("buffer_after")
            if _norm(alt["buf"]) == _norm(c["buffer_after"]):
                sys.exit("%s: the pane answer now matches the gold -- stale samples?" % cid)
        flag, rank = priority(cid, rows, elsewhere)
        cards.append({
            "id": cid, "cat": c["category"], "tier": c.get("tier") or "T?",
            "mode": c.get("mode", "ex"), "src": c.get("source_corpus") or c.get("source", ""),
            "ins": c.get("instruction", ""), "cmd": c["gold_command"],
            "bef": c["buffer_before"], "exp": c["buffer_after"], "cur": list(cursor),
            "flag": flag, "rank": rank,
            "pre": PRE_READ[cid][1] if cid in PRE_READ else None,
            "models": {label: {"top": top_answers(rows[label][cid]),
                               "distinct": rows[label][cid]["distinct"],
                               "states": rows[label][cid]["states"]} for label in TRAINED},
            "greedy": greedy, "elsewhere": elsewhere, "alt": alt,
        })
    cards.sort(key=lambda x: (x["rank"], x["cat"], x["tier"], x["id"]))
    return cards


CSS = r"""
.card.v-gold{ border-left-color:var(--ok); }
.card.v-reword, .card.v-retarget{ border-left-color:var(--warn); }
.card.v-drop{ border-left-color:var(--stop); }
.card.v-unsure{ border-left-color:var(--faint); }
.vb.gold[aria-pressed="true"]{ background:var(--ok-bg); color:var(--ok);
  border-color:var(--ok); font-weight:700; }
.vb.retarget[aria-pressed="true"]{ background:var(--warn-bg); color:var(--warn);
  border-color:var(--warn); font-weight:700; }
.vb.unsure[aria-pressed="true"]{ background:var(--raise); color:var(--ink);
  border-color:var(--ink2); font-weight:700; }
.chip.flag-suspected{ background:var(--stop-bg); color:var(--stop);
  border:1px solid var(--stop-line); font-weight:700; }
.chip.flag-borderline, .chip.flag-confident{ background:var(--warn-bg); color:var(--warn);
  border:1px solid var(--warn-line); font-weight:700; }
.chip.flag-elsewhere{ background:var(--ok-bg); color:var(--ok);
  border:1px solid var(--ok-line); font-weight:700; }
.pre{ margin:2px 15px 10px; padding:8px 11px; border-radius:6px; font-size:13px;
  line-height:1.5; background:var(--stop-bg); border:1px solid var(--stop-line);
  color:var(--ink); max-width:78ch; }
.pre.borderline{ background:var(--warn-bg); border-color:var(--warn-line); }
.pre b{ font-family:var(--mono); font-size:10px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); margin-right:8px; font-weight:700; }
.ev{ margin:0 15px 12px; border:1px solid var(--line2); border-radius:7px;
  overflow:hidden; }
.evh{ font-family:var(--mono); font-size:10px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); font-weight:700;
  padding:7px 11px; background:var(--sunk); border-bottom:1px solid var(--line2); }
.evs{ overflow-x:auto; }
.evt{ border-collapse:collapse; width:100%; font-size:12.5px; }
.evt th{ font-family:var(--mono); font-size:11px; color:var(--ink2); text-align:left;
  vertical-align:top; padding:6px 11px; white-space:nowrap; font-weight:700;
  width:1%; }
.evt td{ padding:6px 11px 6px 0; vertical-align:top; }
.evt tr + tr th, .evt tr + tr td{ border-top:1px solid var(--line2); }
.ans{ display:flex; flex-wrap:wrap; gap:4px 8px; align-items:baseline;
  padding:1px 0; }
.ans code{ font-family:var(--mono); font-size:12px; color:var(--ink);
  overflow-wrap:anywhere; }
.ans .n{ font-family:var(--mono); font-size:11px; color:var(--muted);
  font-variant-numeric:tabular-nums; white-space:nowrap; }
.st{ font-family:var(--mono); font-size:10px; padding:1px 6px; border-radius:4px;
  white-space:nowrap; border:1px solid; }
.st.valid_wrong{ color:var(--warn); background:var(--warn-bg); border-color:var(--warn-line); }
.st.invalid, .st.parse{ color:var(--stop); background:var(--stop-bg); border-color:var(--stop-line); }
.st.match{ color:var(--ok); background:var(--ok-bg); border-color:var(--ok-line); }
.fix{ display:flex; flex-direction:column; gap:6px; padding:10px 14px;
  border-top:1px solid var(--line2); background:var(--sunk); }
.fix label{ font-family:var(--mono); font-size:9.5px; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
.fix textarea, .fix input{ font-family:var(--mono); font-size:12.5px; padding:6px 8px;
  border:1px solid var(--line); border-radius:5px; background:var(--surface);
  color:var(--ink); width:100%; }
.fix textarea{ resize:vertical; min-height:52px; font-family:var(--body); font-size:14px; }
.fixrow{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.fixrow input{ flex:1 1 260px; min-width:0; }
.save{ font-family:var(--mono); font-size:11px; padding:3px 9px; border-radius:5px;
  border:1px solid var(--line); color:var(--muted); background:var(--ground); }
.save.shared{ color:var(--ok); border-color:var(--ok-line); background:var(--ok-bg); }
.save.error{ color:var(--stop); border-color:var(--stop-line); background:var(--stop-bg); }
.copied{ font-family:var(--mono); font-size:11px; color:var(--ok); }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('payload').textContent);
const LKEY = 'vimbot-core-review-v1';
const VERDICTS = [['gold', 'Gold is right'], ['reword', 'Reword instruction'],
                  ['retarget', 'Fix the gold'], ['drop', 'Drop row'], ['unsure', 'Unsure']];
let store = {};                 // id -> {verdict, note, instruction, command, updated}
let db = null;                  // the shared store, once the viewer grants it
const queue = {};               // id -> the write in flight, so writes to a doc never overlap
const timers = {};
let fFlag = 'all', fCat = 'all', fState = 'all';

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
function ans(a){
  return '<div class="ans"><code>' + esc(a.mode + ': ' + a.cmd) + '</code>'
    + (a.n != null ? '<span class="n">' + a.n + '/128</span>' : '')
    + '<span class="st ' + esc(a.state) + '">' + esc(STATE_WORDS[a.state] || a.state)
    + '</span></div>';
}
const STATE_WORDS = JSON.parse(document.getElementById('states').textContent);
const FLAG_CLASS = {'suspected defect': 'suspected', 'borderline': 'borderline',
                    'confident disagreement': 'confident',
                    'solved by another model': 'elsewhere'};

function evidence(c){
  let rows = '';
  ['sft', 'served', 'cliffleg'].forEach(function(label){
    const m = c.models[label];
    rows += '<tr><th>' + label + '</th><td>' + m.top.map(ans).join('')
         + '<div class="ans"><span class="n">' + m.distinct
         + ' distinct answers in 128 samples</span></div></td></tr>';
  });
  if (c.greedy.length){
    rows += '<tr><th>greedy</th><td>' + c.greedy.map(function(g){
      return '<div class="ans"><span class="n">' + esc(g.run) + '</span></div>'
           + ans({mode: g.mode || '?', cmd: g.cmd == null ? '(no parseable answer)' : g.cmd,
                  state: g.state});
    }).join('') + '</td></tr>';
  }
  if (c.elsewhere.length){
    rows += '<tr><th>solved by</th><td>' + c.elsewhere.map(function(g){
      return '<div class="ans"><span class="n">' + esc(g.run) + '</span></div>'
           + ans({mode: g.mode || '?', cmd: g.cmd, state: 'match'});
    }).join('') + '</td></tr>';
  }
  return '<div class="ev"><div class="evh">What the models answer instead &middot; '
    + '0 of 128 samples right in every checkpoint</div>'
    + '<div class="evs"><table class="evt">' + rows + '</table></div></div>';
}

function cardHtml(c){
  const flag = c.flag ? '<span class="chip flag-' + FLAG_CLASS[c.flag] + '">'
                      + esc(c.flag) + '</span>' : '';
  const alt = c.alt
    ? pane('most common runnable wrong answer &middot; '
           + Object.keys(c.alt.by).map(function(k){ return esc(k) + ' ' + c.alt.by[k] + '/128'; })
             .join(', ') + '<br><code style="text-transform:none;letter-spacing:0">'
           + esc(c.alt.mode + ': ' + c.alt.cmd) + '</code>', c.alt.buf, c.exp, 'cut')
    : '<div class="pane"><span class="lab">no runnable wrong answer</span>'
      + '<div class="ln"><span class="t">every answer vim ran was an error</span></div></div>';
  const btns = VERDICTS.map(function(v){
    return '<button class="vb ' + v[0] + '" data-act="' + v[0] + '" data-id="' + esc(c.id)
         + '" aria-pressed="false">' + v[1] + '</button>';
  }).join('');
  return '<div class="card" id="c-' + esc(c.id) + '" data-id="' + esc(c.id) + '">'
    + '<div class="chead"><span class="cid">' + esc(c.id) + '</span>'
    +   '<span class="keys">' + esc(c.mode + ': ' + c.cmd) + '</span>'
    +   '<span class="chip">' + esc(c.cat) + '</span>'
    +   '<span class="chip">' + esc(c.tier) + '</span>'
    +   '<span class="chip">cursor ' + c.cur[0] + ',' + c.cur[1] + '</span>'
    +   (c.src ? '<span class="chip">' + esc(c.src) + '</span>' : '')
    +   '<span class="spacer"></span>' + flag + '</div>'
    + '<div class="ins">' + esc(c.ins || '(no instruction)') + '</div>'
    + (c.pre ? '<div class="pre' + (c.flag === 'borderline' ? ' borderline' : '') + '">'
             + '<b>analyst’s read</b>' + esc(c.pre) + '</div>' : '')
    + '<div class="qn">Is the gold what this sentence asks for, and is the models’ '
    + 'reading also a faithful one?</div>'
    + evidence(c)
    + '<div class="panes">'
    +   pane('input buffer', c.bef, c.exp, 'cut')
    +   pane('gold output', c.exp, c.bef, 'add')
    +   alt
    + '</div>'
    + '<div class="fix" data-fix="' + esc(c.id) + '" hidden></div>'
    + '<div class="cfoot">' + btns
    +   '<input class="note" id="note-' + esc(c.id) + '" data-id="' + esc(c.id)
    +   '" placeholder="note (optional)" aria-label="note for ' + esc(c.id) + '">'
    + '</div></div>';
}

function fixHtml(c, v){
  const s = store[c.id] || {};
  if (v === 'reword'){
    return '<label for="ins-' + esc(c.id) + '">instruction as it should read</label>'
      + '<textarea id="ins-' + esc(c.id) + '" data-field="instruction" data-id="' + esc(c.id)
      + '" spellcheck="true">' + esc(s.instruction != null ? s.instruction : c.ins) + '</textarea>';
  }
  if (v === 'retarget'){
    return '<label for="cmd-' + esc(c.id) + '">gold command as it should be ('
      + esc(c.mode) + ' mode; apply_curation.py recomputes the expected output)</label>'
      + '<div class="fixrow"><input id="cmd-' + esc(c.id) + '" data-field="command" data-id="'
      + esc(c.id) + '" spellcheck="false" value="'
      + esc(s.command != null ? s.command : c.cmd) + '">'
      + (c.alt && c.alt.mode === c.mode
         ? '<button class="btn" data-use="' + esc(c.id) + '">Use the models’ answer</button>'
         : '')
      + '</div>';
  }
  return '';
}

function paint(id){
  const c = DATA.find(function(x){ return x.id === id; });
  const el = document.getElementById('c-' + id);
  if (!c || !el) return;
  const s = store[id] || {};
  el.className = 'card' + (s.verdict ? ' v-' + s.verdict : '');
  el.querySelectorAll('.vb[data-act]').forEach(function(b){
    b.setAttribute('aria-pressed', String(b.getAttribute('data-act') === s.verdict));
  });
  const note = document.getElementById('note-' + id);
  if (note && document.activeElement !== note) note.value = s.note || '';
  const fix = el.querySelector('[data-fix]');
  const want = s.verdict === 'reword' || s.verdict === 'retarget' ? s.verdict : '';
  if (fix.getAttribute('data-for') !== want){
    fix.innerHTML = want ? fixHtml(c, want) : '';
    fix.setAttribute('data-for', want);
  }
  fix.hidden = !want;
}

function persist(id){
  const body = Object.assign({}, store[id] || {}, {updated: new Date().toISOString()});
  store[id] = body;
  try { localStorage.setItem(LKEY, JSON.stringify(store)); } catch (e) {}
  if (!db) return;
  const run = function(){
    return db.collection('review').doc(id).set(body).catch(function(err){
      setStatus('error', 'Could not save ' + id + ' to the shared store (' + (err && err.code)
                + '). It is kept in this browser; use Export to hand it over.');
    });
  };
  queue[id] = (queue[id] || Promise.resolve()).then(run);
}
function persistSoon(id){
  clearTimeout(timers[id]);
  timers[id] = setTimeout(function(){ persist(id); }, 700);
}

function visible(c){
  const s = store[c.id] || {};
  if (fCat !== 'all' && c.cat !== fCat) return false;
  if (fFlag === 'flagged' && !c.flag) return false;
  if (fFlag === 'elsewhere' && !c.elsewhere.length) return false;
  if (fState === 'todo' && s.verdict) return false;
  if (fState === 'done' && !s.verdict) return false;
  return true;
}
function refresh(){
  let shown = 0, done = 0;
  DATA.forEach(function(c){
    const el = document.getElementById('c-' + c.id);
    const on = visible(c);
    el.hidden = !on;
    if (on) shown++;
    if ((store[c.id] || {}).verdict) done++;
  });
  document.getElementById('count').textContent = shown + ' of ' + DATA.length + ' shown';
  document.getElementById('prog').textContent = done + ' / ' + DATA.length;
  document.getElementById('bar').style.width = (100 * done / DATA.length) + '%';
}

function setStatus(kind, text){
  const el = document.getElementById('save');
  el.className = 'save ' + kind;
  el.textContent = text;
}

function changeSet(){
  const out = {edits: {}, deletes: {}, review: {}};
  DATA.forEach(function(c){
    const s = store[c.id];
    if (!s || (!s.verdict && !s.note)) return;
    out.review[c.id] = {verdict: s.verdict || null, note: s.note || ''};
    if (s.verdict === 'drop'){
      out.deletes[c.id] = s.note || 'dropped in the always-failing review: the row is not '
        + 'a fair test of the instruction';
    } else if (s.verdict === 'reword' && s.instruction != null && s.instruction !== c.ins){
      out.edits[c.id] = {instruction: s.instruction};
    } else if (s.verdict === 'retarget' && s.command != null && s.command !== c.cmd){
      out.edits[c.id] = {command: s.command};
    }
  });
  return out;
}

document.getElementById('list').innerHTML = DATA.map(cardHtml).join('');
DATA.forEach(function(c){ paint(c.id); });
refresh();

document.getElementById('list').addEventListener('click', function(ev){
  const use = ev.target.closest('[data-use]');
  if (use){
    const id = use.getAttribute('data-use');
    const c = DATA.find(function(x){ return x.id === id; });
    store[id] = Object.assign({}, store[id] || {}, {command: c.alt.cmd});
    const input = document.getElementById('cmd-' + id);
    if (input) input.value = c.alt.cmd;
    persist(id);
    return;
  }
  const b = ev.target.closest('.vb[data-act]');
  if (!b) return;
  const id = b.getAttribute('data-id'), act = b.getAttribute('data-act');
  const s = Object.assign({}, store[id] || {});
  s.verdict = s.verdict === act ? null : act;
  store[id] = s;
  paint(id);
  refresh();
  persist(id);
});
document.getElementById('list').addEventListener('input', function(ev){
  const t = ev.target;
  const id = t.getAttribute('data-id');
  if (!id) return;
  const field = t.classList.contains('note') ? 'note' : t.getAttribute('data-field');
  if (!field) return;
  store[id] = Object.assign({}, store[id] || {});
  store[id][field] = t.value;
  persistSoon(id);
});

document.querySelectorAll('[data-flag]').forEach(function(b){
  b.addEventListener('click', function(){
    fFlag = b.getAttribute('data-flag');
    document.querySelectorAll('[data-flag]').forEach(function(x){
      x.setAttribute('aria-pressed', String(x === b));
    });
    refresh();
  });
});
document.getElementById('cat').addEventListener('change', function(ev){
  fCat = ev.target.value; refresh();
});
document.getElementById('state').addEventListener('change', function(ev){
  fState = ev.target.value; refresh();
});
document.getElementById('export').addEventListener('click', function(){
  const out = changeSet();
  document.getElementById('out').value = JSON.stringify(out, null, 1);
  document.getElementById('dlgsub').textContent = Object.keys(out.review).length
    + ' reviewed, ' + Object.keys(out.edits).length + ' edit(s), '
    + Object.keys(out.deletes).length + ' delete(s). apply_curation.py re-executes every '
    + 'edited row and recomputes its expected output.';
  document.getElementById('copied').textContent = '';
  document.getElementById('dlg').showModal();
});
document.getElementById('copy').addEventListener('click', function(){
  const out = document.getElementById('out');
  const done = function(){ document.getElementById('copied').textContent = 'Copied'; };
  try {
    navigator.clipboard.writeText(out.value).then(done, function(){ out.select(); });
  } catch (e) { out.select(); }
});
document.getElementById('close').addEventListener('click', function(){
  document.getElementById('dlg').close();
});

// The shared store arrives after first paint; the page works without it.
(async function(){
  let d = null;
  try {
    d = window.claude && window.claude.use ? await window.claude.use('db') : null;
  } catch (e) { d = null; }
  if (!d){
    setStatus('', 'Saving in this browser only. Export hands the verdicts over.');
    return;
  }
  db = d;
  setStatus('shared', 'Saving to this page’s shared store. Claude reads it back.');
  const local = Object.assign({}, store);
  let first = true;
  db.collection('review').onSnapshot(function(snap){
    const next = {};
    snap.docs.forEach(function(doc){ if (doc.exists) next[doc.id] = Object.assign({}, doc.data()); });
    if (first){
      first = false;
      // Carry verdicts made in this browser before the store arrived.
      Object.keys(local).forEach(function(id){
        if (!next[id] && (local[id].verdict || local[id].note)){
          next[id] = local[id];
          store[id] = local[id];
          persist(id);
        }
      });
    }
    Object.keys(next).forEach(function(id){
      const mine = store[id] || {};
      if (!timers[id] || !document.activeElement
          || document.activeElement.getAttribute('data-id') !== id){
        store[id] = Object.assign({}, mine, next[id]);
      }
    });
    DATA.forEach(function(c){ paint(c.id); });
    refresh();
  }, function(err){
    setStatus('error', 'The shared store stopped answering (' + (err && err.code)
              + '). New verdicts stay in this browser; Export hands them over.');
    db = null;
  });
})();
"""

LEDE = """\
  <p class="lede">
    These <b>%(n)d test rows</b> were never solved by any of the 640 samples each
    received in the pass@k run: 128 each from the base 4B, SFT round 1, the served
    GRPO adapter, the cliff-leg adapter, and the base at the model card&rsquo;s
    settings. <b>%(elsewhere)d</b> of them some other model family&rsquo;s greedy run
    did solve. A row nothing solves is either beyond the model or wrongly labelled,
    and only a reader can tell which.
  </p>
  <p class="lede" style="margin-top:10px">
    Each card shows what the models answer instead, and the buffer their most common
    runnable wrong answer produces, with lines that differ from the gold output
    marked. <b>%(flagged)d rows are flagged</b> and sorted first: two where the
    near-unanimous answer follows the instruction as written and the gold does not,
    one borderline, and every row where a checkpoint gives one runnable wrong answer
    in at least half its samples. That is the signature of a label defect, though
    most such rows turn out to be plain model errors. These are test rows: judge
    them, but never author training data from them.
  </p>
"""


def build(cards):
    n = len(cards)
    flagged = sum(1 for c in cards if c["flag"] and c["flag"] != "solved by another model")
    elsewhere = sum(1 for c in cards if c["elsewhere"])
    cats = sorted({c["cat"] for c in cards})
    opts = "".join('<option value="%s">%s (%d)</option>'
                   % (c, c, sum(1 for x in cards if x["cat"] == c)) for c in cats)
    payload = json.dumps(cards, ensure_ascii=False).replace("</", "<\\/")
    states = json.dumps(STATE_WORDS, ensure_ascii=False)
    return """<title>Always-Failing Rows</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&\
family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%s%s</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; pass@k ceiling &middot; hand review of the uncovered core</div>
  <details id="intro" open>
  <summary><h1>Is the model wrong, or is the row?</h1>
    <span class="hint" aria-hidden="true"></span></summary>
%s  </details>
  <div class="band">
    <div><span class="v">%d</span><span class="k">rows no sample solves</span></div>
    <div><span class="v" style="color:var(--warn)">%d</span><span class="k">flagged, sorted first</span></div>
    <div><span class="v" style="color:var(--ok)">%d</span><span class="k">solved by another model</span></div>
    <div><span class="v" id="prog">0 / %d</span><span class="k">reviewed</span></div>
  </div>
</div></header>

<div class="dock"><div class="wrap">
  <span class="save" id="save">Connecting to the shared store&hellip;</span>
  <div class="bar" aria-hidden="true"><i id="bar" style="width:0%%"></i></div>
  <button class="btn pri" id="export">Export change-set</button>
</div></div>

<div class="rail"><div class="wrap">
  <span class="lab">show</span>
  <button class="f" data-flag="all" aria-pressed="true">all</button>
  <button class="f" data-flag="flagged" aria-pressed="false">flagged</button>
  <button class="f" data-flag="elsewhere" aria-pressed="false">solved by another model</button>
  <label class="lab" for="cat">category</label>
  <select id="cat"><option value="all">all</option>%s</select>
  <label class="lab" for="state">verdict</label>
  <select id="state"><option value="all">any</option><option value="todo">not yet</option>
    <option value="done">given</option></select>
  <span class="count" id="count"></span>
</div></div>

<main><div class="wrap"><div class="cards" id="list"></div></div></main>

<dialog id="dlg" aria-labelledby="dlgh">
  <div class="dh"><h2 id="dlgh">Change-set for apply_curation.py</h2>
    <p class="lede" id="dlgsub" style="font-size:13.5px"></p></div>
  <div class="db">
    <textarea id="out" readonly aria-label="change-set JSON"></textarea>
    <div class="fixrow"><button class="btn pri" id="copy">Copy</button>
      <button class="btn" id="close">Close</button>
      <span class="copied" id="copied"></span></div>
  </div>
</dialog>

<script type="application/json" id="payload">%s</script>
<script type="application/json" id="states">%s</script>
<script>%s</script>
""" % (BASE_CSS, CSS, LEDE % {"n": n, "elsewhere": elsewhere, "flagged": flagged},
       n, flagged, elsewhere, n, opts, payload, states, JS)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="page to write (HTML)")
    args = ap.parse_args()
    cards = collect()
    page = build(cards)
    print(check_page(page, required_js=["split('\\n')", "replace(/\\t/g", "'\\n'"],
                     name="always-failing review"))
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    by_flag = collections.Counter(c["flag"] or "unflagged" for c in cards)
    print("rows: %d  %s  page: %d KB -> %s"
          % (len(cards), dict(by_flag), len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
