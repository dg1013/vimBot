#!/usr/bin/env python3
"""Build the instruction-review artifact: the 635 mined cases, editable by hand.

Covers the rows that arrived from vim-implementation test suites as
(buffer_before, command, buffer_after) triples with no natural language —
600 in emulator_cases.jsonl, 35 in candidate_cases.jsonl. Their upstream test
titles name the command, so reusing them would leak the answer into the
model's own prompt (CLAUDE.md sec 6); the titles are shown on the page behind
a warning, as reviewer context only.

The page carries the same guard as scripts/apply_instructions.py: the leak
forms are computed here, in Python, by calling that module's own helper, and
shipped per row — so the browser cannot drift from the writer. Export emits a
{id: instruction} map per case file, ready for:

    python scripts/apply_instructions.py --cases data/v3/<file>.jsonl \
                                         --instructions <exported>.json

Usage (repo root):
    conda run -n vimBot python scripts/build_instruction_review.py --out page.html
"""
import argparse
import difflib
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from page_guard import check_page  # noqa: E402
from apply_instructions import leaks  # noqa: E402  (guard parity, see docstring)

FILES = ["emulator_cases.jsonl", "candidate_cases.jsonl"]

# The review cohort is the set of rows that arrived with no natural language.
# It cannot be recovered from instruction_provenance once they are authored —
# they read "claude-authored" exactly like the CodeMirror rows written in an
# earlier session — so the membership list is captured on disk instead.
COHORT = REPO / "data" / "v3" / "mined_cohort_2026-08-19.json"

# Why each held-back row was left pending. Every one of these has describable
# prose for its buffer change — that is the trap. The gold command's *route*
# is what no plain request could motivate, so writing the prose would teach
# the model to answer an ordinary request with a contrived keystroke pile-up.
# Grouped exactly as DECISIONS 2026-08-19 records them.
# Every mined row now carries an instruction; rows that could not be given one
# were dropped to data/v3/dropped_cases.jsonl (see DECISIONS 2026-08-19), so
# there is nothing left to justify here. Kept as the hook for the next round.
SKIP_REASONS = {}


def leak_forms(row):
    """The literal strings the guard refuses, computed the way the guard does.

    apply_instructions.leaks() builds these inline; rather than restate the
    regex (regular expression) in JavaScript and risk a dialect difference, probe the real function
    with each candidate form and keep the ones it actually rejects.
    """
    forms = set()
    for form in filter(None, (row.get("gold_command"), row.get("keys_raw"))):
        f = form.strip()
        forms.add(f)
        forms.add(re.sub(r"<[^>]*>", "", f))
        forms.add(re.sub(r"<[^>]*>", "", f).lstrip(":"))
    out = sorted({f.strip().lower() for f in forms if len(f.strip()) >= 2})
    # sanity: every shipped form must be one the writer would reject
    for f in out:
        assert leaks(f, row.get("gold_command", ""), row.get("keys_raw", "")), f
    return out


def difflines(before, after):
    """Per-line ops for the buffer panes, so 'what did the edit do' is visible.

    Returns (left, right) lists of [tag, lineno_or_None, text] where tag is
    ' ' kept, '-' cut, '+' added. Line numbers are the ones vim would show.
    """
    a = before.split("\n")
    b = after.split("\n")
    if a and a[-1] == "":
        a.pop()
    if b and b[-1] == "":
        b.pop()
    left, right = [], []
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i1, i2):
                left.append([" ", i1 + 1 + (k - i1), a[k]])
            for k in range(j1, j2):
                right.append([" ", j1 + 1 + (k - j1), b[k]])
        else:
            for k in range(i1, i2):
                left.append(["-", k + 1, a[k]])
            for k in range(j1, j2):
                right.append(["+", k + 1, b[k]])
    # renumber against each pane's own buffer
    for pane in (left, right):
        n = 0
        for e in pane:
            n += 1
            e[1] = n
    return left, right


def collect(cohort_path):
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    want = set(cohort["ids"])
    rows = []
    for fn in FILES:
        path = REPO / "data" / "v3" / fn
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] not in want:
                continue
            want.discard(r["id"])
            before, after = r["buffer_before"], r["buffer_after"]
            lft, rgt = difflines(before, after)
            rows.append({
                "id": r["id"],
                "file": fn,
                "cat": r.get("category") or "—",
                "tier": r.get("tier") or "—",
                "mode": r.get("mode") or "keys",
                "cur": r.get("cursor") or [1, 1],
                "cmd": (r.get("keys_raw") or r.get("gold_command") or ""),
                "ins": (r.get("instruction") or "").strip(),
                "prov": r.get("instruction_provenance"),
                "src": r.get("source") or "",
                "sfile": r.get("source_file") or "",
                "title": r.get("source_test_title") or "",
                "L": lft,
                "R": rgt,
                "forms": leak_forms(r),
                "why": SKIP_REASONS.get(r["id"], ("", "")),
                "retargeted": r.get("command_provenance") == "claude-retargeted",
                "was": r.get("source_command", ""),
            })
    dropped = sorted(want)   # cohort rows removed from the case files since
    blank = {r["id"] for r in rows if not r["ins"]}
    documented = set(SKIP_REASONS)
    if blank - documented:
        sys.exit("pending rows with no recorded reason: %s"
                 % sorted(blank - documented))
    if documented - blank:
        sys.exit("SKIP_REASONS names rows that are no longer pending: %s"
                 % sorted(documented - blank))
    return rows, dropped


PAGE = r"""<title>Instruction Bench</title>
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
body{ margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--sans); font-size:15px; line-height:1.55; -webkit-font-smoothing:antialiased; }
h1,h2{ font-family:var(--disp); font-weight:600; margin:0; text-wrap:balance; letter-spacing:-.01em; }
.mono{ font-family:var(--mono); }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase;
  color:var(--muted); font-weight:600; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px; border-radius:3px; }
.wrap{ max-width:1180px; margin:0 auto; padding:0 22px; }

header{ background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:24px 22px 18px; }
h1{ font-size:clamp(25px,4vw,36px); margin:7px 0 9px; }
.lede{ max-width:66ch; color:var(--ink2); font-size:15.5px; }
.lede b{ color:var(--ink); font-weight:600; }
.lede code{ font-family:var(--mono); font-size:.9em; background:var(--sunk);
  padding:1px 4px; border-radius:3px; }

.agg{ display:grid; grid-template-columns:repeat(auto-fit,minmax(122px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:7px; overflow:hidden; margin-top:18px; }
.agg div{ background:var(--surface); padding:10px 13px; }
.agg .v{ font-family:var(--mono); font-size:19px; font-weight:700; letter-spacing:-.02em;
  display:block; font-variant-numeric:tabular-nums; }
.agg .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin-top:2px; display:block; }
.agg .v.ok{ color:var(--pass); } .agg .v.no{ color:var(--warn); } .agg .v.ed{ color:var(--accent); }

.pbar{ height:6px; margin-top:14px; border-radius:3px; overflow:hidden; background:var(--raise); display:flex; }
.pbar i{ display:block; height:100%; }

.rail{ position:sticky; top:0; z-index:20; background:var(--surface); border-bottom:1px solid var(--line); }
.rail .wrap{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; padding:9px 22px; }
.grp{ display:flex; gap:3px; align-items:center; flex-wrap:wrap; }
.grp>.lab{ margin-right:3px; }
.f{ font-family:var(--mono); font-size:11.5px; padding:3px 9px; border-radius:4px; cursor:pointer;
  background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.f:hover{ border-color:var(--accent); color:var(--accent); }
.f[aria-pressed="true"]{ background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); font-weight:600; }
.f .c{ opacity:.6; margin-left:5px; font-variant-numeric:tabular-nums; }
select,input[type=search]{ font:inherit; font-size:12.5px; padding:4px 8px; border:1px solid var(--line);
  border-radius:4px; background:var(--ground); color:inherit; font-family:var(--mono); }
input[type=search]{ min-width:12rem; }
.count{ font-family:var(--mono); font-size:12px; color:var(--muted); margin-left:auto;
  font-variant-numeric:tabular-nums; }
.btn{ font-family:var(--mono); font-size:11.5px; padding:4px 10px; border-radius:4px; cursor:pointer;
  background:var(--surface); border:1px solid var(--line); color:var(--ink2); }
.btn:hover{ border-color:var(--accent); color:var(--accent); }
.btn.pri{ background:var(--accent); border-color:var(--accent); color:var(--on-accent); font-weight:600; }
.btn.pri:hover{ background:var(--accent2); border-color:var(--accent2); }

main.wrap{ padding:22px 22px 120px; }
.cards{ display:flex; flex-direction:column; gap:11px; }
.card{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--faint);
  border-radius:6px; overflow:hidden; scroll-margin-top:64px; }
.card.st-done{ border-left-color:var(--pass); }
.card.st-todo{ border-left-color:var(--warn); }
.card.st-bad{ border-left-color:var(--stop); }
.card.cur{ outline:2px solid var(--accent); outline-offset:1px; }
.chead{ display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; padding:8px 13px;
  border-bottom:1px solid var(--line2); }
.pill{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; padding:2px 6px; border-radius:3px; white-space:nowrap; border:1px solid; }
.p-done{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }
.p-todo{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.p-bad{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); }
.cid{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:3px;
  background:var(--raise); color:var(--ink2); white-space:nowrap; }
.keys{ font-family:var(--mono); font-size:12.5px; font-weight:600; background:var(--accent-soft);
  color:var(--accent); padding:2px 7px; border-radius:3px; overflow-wrap:anywhere; max-width:100%; }
.spacer{ flex:1 1 auto; }

.insrow{ padding:10px 13px 4px; }
.ins{ width:100%; font-family:var(--sans); font-size:15px; font-weight:600; color:var(--ink);
  line-height:1.45; border:1px solid var(--line); border-radius:5px; padding:7px 9px;
  background:var(--ground); resize:vertical; min-height:2.6em; }
.ins::placeholder{ color:var(--faint); font-weight:400; font-style:italic; }
.ins:focus{ outline:none; border-color:var(--accent); background:var(--surface);
  box-shadow:0 0 0 2px var(--accent-soft); }
.ins.edited{ border-color:var(--accent); }
.ins.bad{ border-color:var(--stop); box-shadow:0 0 0 2px var(--stop-bg); }
.rt{ display:flex; flex-direction:column; gap:5px; margin:2px 13px 0; padding:9px 11px;
  border-radius:5px; background:var(--accent-soft); border:1px solid var(--accent); }
.rt .whytag{ color:var(--accent); }
.rt b{ font-family:var(--mono); font-weight:600; }
.why{ display:flex; flex-direction:column; gap:5px; margin:2px 13px 0; padding:9px 11px;
  border-radius:5px; background:var(--warn-bg); border:1px solid var(--warn-line); }
.whytag{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:var(--warn); }
.whytxt{ font-size:13.5px; color:var(--ink2); line-height:1.5; max-width:88ch; }
.flags{ display:flex; flex-wrap:wrap; gap:6px; padding:5px 13px 9px; min-height:0; }
.flag{ font-family:var(--mono); font-size:11px; padding:2px 7px; border-radius:3px; border:1px solid; }
.flag.err{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); font-weight:600; }
.flag.warnf{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.flag.okf{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }

.panes{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:1px;
  background:var(--line2); border-top:1px solid var(--line2); }
.pane{ background:var(--surface); padding:8px 12px; min-width:0; overflow-x:auto; }
.pane .pl{ display:block; margin-bottom:5px; }
.pane.b .pl{ color:var(--warn); } .pane.a .pl{ color:var(--pass); }
.ln{ display:flex; gap:0; white-space:pre; font-family:var(--mono); font-size:12px; line-height:1.5; }
.ln .g{ color:var(--faint); opacity:.65; user-select:none; text-align:right; min-width:2.1em;
  font-variant-numeric:tabular-nums; flex:0 0 auto; padding-right:8px; }
.ln .t{ white-space:pre; }
.ln.cut{ background:var(--cut-bg); }
.ln.cut .t{ color:var(--cut); }
.ln.add{ background:var(--add-bg); }
.ln.add .t{ color:var(--add); }
.ws{ color:var(--accent); opacity:.38; }
.cx{ background:var(--accent); color:var(--on-accent); border-radius:2px; }
.nolines{ color:var(--faint); font-style:italic; font-family:var(--mono); font-size:12px; }

.cfoot{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding:7px 13px;
  border-top:1px solid var(--line2); }
.mk button{ font-family:var(--mono); font-size:11px; padding:3px 9px; border-radius:3px; cursor:pointer;
  background:none; border:1px solid var(--line); color:var(--muted); }
.mk button:hover{ border-color:var(--accent); color:var(--accent); }
.mk button[aria-pressed="true"]{ background:var(--pass-bg); color:var(--pass);
  border-color:var(--pass); font-weight:700; }
.mk button.skip[aria-pressed="true"]{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn); }
details.title{ font-family:var(--mono); font-size:11px; color:var(--muted); }
details.title summary{ cursor:pointer; color:var(--faint); }
details.title summary:hover{ color:var(--accent); }
details.title .body{ margin-top:5px; padding:6px 9px; border-radius:4px;
  background:var(--stop-bg); border:1px solid var(--stop-line); color:var(--stop);
  overflow-wrap:anywhere; }

.more{ display:flex; justify-content:center; padding:18px 0 0; }
.dock{ position:fixed; left:0; right:0; bottom:0; z-index:30; background:var(--surface);
  border-top:1px solid var(--line); }
.dock .wrap{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:center; padding:9px 22px; }
.dock .hint{ font-family:var(--mono); font-size:11px; color:var(--muted); display:flex;
  gap:11px; flex-wrap:wrap; }
.kbd{ font-family:var(--mono); font-size:10.5px; background:var(--raise); border:1px solid var(--line);
  border-bottom-width:2px; border-radius:3px; padding:0 4px; color:var(--ink2); }
dialog{ border:1px solid var(--line); border-radius:8px; background:var(--surface); color:var(--ink);
  max-width:min(860px,92vw); width:100%; padding:0; }
dialog::backdrop{ background:rgba(20,16,19,.55); }
dialog .dh{ padding:15px 18px 0; }
dialog h2{ font-size:19px; }
dialog p.sub{ margin:6px 0 0; color:var(--muted); font-size:13px; max-width:74ch; }
dialog .db{ padding:13px 18px 16px; display:flex; flex-direction:column; gap:9px; }
dialog textarea{ width:100%; min-height:250px; font-family:var(--mono); font-size:12px;
  border:1px solid var(--line); border-radius:5px; padding:9px; background:var(--ground);
  color:var(--ink); resize:vertical; }
dialog .row{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.cmdline{ font-family:var(--mono); font-size:11.5px; background:var(--sunk); border:1px solid var(--line);
  border-radius:4px; padding:7px 9px; color:var(--ink2); overflow-x:auto; white-space:pre; }
.foot{ margin-top:30px; padding-top:17px; border-top:1px solid var(--line); font-size:13px;
  color:var(--muted); max-width:78ch; }
.foot b{ color:var(--ink2); }
.boot{ padding:34px 0; text-align:center; color:var(--muted); font-family:var(--mono);
  font-size:13px; line-height:1.6; }
.boot.err{ color:var(--stop); text-align:left; white-space:pre-wrap; overflow-wrap:anywhere;
  background:var(--stop-bg); border:1px solid var(--stop-line); border-radius:6px; padding:14px 16px; }
.empty{ padding:40px 0; text-align:center; color:var(--muted); font-family:var(--mono); font-size:13px; }
@media (prefers-reduced-motion:no-preference){ .card{ transition:outline-color .12s ease; } }
</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; dataset v3 &middot; instruction review</div>
  <h1>Instruction Bench</h1>
  <p class="lede">
    The <b>__TOTAL__ cases mined from vim-implementation test suites</b>. Each one arrived as a
    buffer, a command and a resulting buffer, with <b>no natural language</b> — and each one still
    has to be told what the user wanted, because the instruction is the model's whole input.
    626 are written and merged into v3, <b>32 of them with the mined command replaced</b>
    by an idiomatic one that reproduces the same verified buffer. <b>The 9 below are held back</b> \u2014 each has
    prose that would describe its buffer change honestly, but a gold command whose route no
    plain request could motivate. The reason sits on every card: accept it, or write an
    instruction and the row joins v3 on the next apply. Switch to <b>all</b> to review the
    written ones. Edits are kept in this browser as you type.
  </p>
  <div class="agg" id="agg"></div>
  <div class="pbar" id="pbar"></div>
</div></header>

<div class="rail"><div class="wrap" id="rail"></div></div>

<main class="wrap">
  <div class="cards" id="cards"><div class="boot" id="boot">Loading 635 cases&hellip; if this message stays, the page failed to start and the reason will appear here.</div></div>
  <div class="more"><button class="btn" id="more">Show 50 more</button></div>
  <p class="foot">
    <b>Why the test titles are hidden.</b> Upstream titles read like
    <span class="mono">":move [address] will move the cursor line to below the line given by
    {address}"</span> — they name the command. Putting that in the instruction hands the model its
    own answer, so each title sits behind a disclosure and is marked as reviewer-only context.
    <br><br>
    <b>What the checks mean.</b> The <span class="mono">leak</span> flag is the same rule
    <span class="mono">apply_instructions.py</span> enforces: an instruction may not contain the
    gold command verbatim, before or after stripping <span class="mono">&lt;Esc&gt;</span>-style
    tokens and a leading colon. Those exact strings are computed by that module and shipped with
    each row, so what you see here is what the writer will accept. The remaining flags are the
    house style — sentence case, no trailing period, one line, describe intent rather than keys.
    <br><br>
    <b>Leaving one pending is a real answer.</b> Some rows resist honest phrasing: a register
    juggle or an undo/redo fixture whose gold command no plain request would ever produce. Marking
    such a row <span class="mono">skip</span> keeps it out of the export, which leaves it pending
    on disk — the intended outcome, not a gap.
  </p>
</main>

<div class="dock"><div class="wrap">
  <div class="hint">
    <span><span class="kbd">j</span><span class="kbd">k</span> move</span>
    <span><span class="kbd">Enter</span> edit</span>
    <span><span class="kbd">Esc</span> leave field</span>
    <span><span class="kbd">m</span> mark reviewed</span>
    <span><span class="kbd">s</span> skip</span>
    <span><span class="kbd">/</span> search</span>
  </div>
  <div class="spacer"></div>
  <span class="count" id="dockcount"></span>
  <button class="btn pri" id="export">Copy changes out</button>
</div></div>

<dialog id="dlg"><div class="dh">
    <h2>Apply these to v3</h2>
    <p class="sub" id="dlgsub"></p>
  </div><div class="db">
    <div class="row">
      <span class="lab">File</span>
      <span id="dlgtabs" class="grp"></span>
      <span class="spacer"></span>
      <button class="btn" id="copy">Copy JSON</button>
      <button class="btn" id="close">Close</button>
    </div>
    <textarea id="out" spellcheck="false" readonly></textarea>
    <div class="cmdline" id="cmd"></div>
    <p class="sub">Save the JSON to a file, then run the command above. The writer re-checks every
    line and refuses any that contains its own answer, so a bad edit fails loudly instead of
    landing. Re-run <span class="mono">src/evaluate.py</span> afterwards.</p>
  </div></dialog>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const ROWS = JSON.parse(document.getElementById("payload").textContent);
const KEY = "vimbot.instruction-bench.v1";
const BASE = new Map(ROWS.map(r => [r.id, r.ins]));

let store = {};
try { store = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { store = {}; }
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
const st = id => (store[id] = store[id] || {});
const textOf = id => { const s = store[id]; return s && s.t !== undefined ? s.t : BASE.get(id); };

/* ---- checks: leak parity with apply_instructions.py, then house style ---- */
function check(row) {
  const t = textOf(row.id) || "";
  const low = t.toLowerCase().trim();
  const out = [];
  if (!low) return [{ k: "warnf", m: "no instruction yet" }];
  for (const f of row.forms) {
    if (low.includes(f)) { out.push({ k: "err", m: "leak: contains " + JSON.stringify(f) }); break; }
  }
  if (/\bpress\b|\bkeystroke|\btype the key/.test(low))
    out.push({ k: "err", m: "leak: describes keystrokes, not intent" });
  if (t.trim().endsWith(".")) out.push({ k: "warnf", m: "trailing period" });
  if (/\n/.test(t)) out.push({ k: "warnf", m: "more than one line" });
  if (t.trim() && t.trim()[0] !== t.trim()[0].toUpperCase())
    out.push({ k: "warnf", m: "not sentence case" });
  if (t.trim().length < 12) out.push({ k: "warnf", m: "very short" });
  if (t !== t.trim()) out.push({ k: "warnf", m: "leading or trailing space" });
  return out;
}
const isErr = row => check(row).some(f => f.k === "err");
function status(row) {
  const s = store[row.id] || {};
  if (s.skip) return "todo";
  if (isErr(row)) return "bad";
  if (!(textOf(row.id) || "").trim()) return "todo";
  return s.done ? "done" : (textOf(row.id) !== BASE.get(row.id) ? "done" : "todo");
}

/* ---- rendering ---- */
const esc = s => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function vis(text, cursorCol) {
  // spaces and tabs made visible; the cursor cell highlighted where it applies
  const chars = Array.from(text);
  let out = "";
  chars.forEach((ch, i) => {
    let piece = ch === " " ? '<span class="ws">&middot;</span>'
      : ch === "\t" ? '<span class="ws">&rarr;&nbsp;&nbsp;&nbsp;</span>'
      : esc(ch);
    if (cursorCol && i === cursorCol - 1) piece = '<span class="cx">' + piece + "</span>";
    out += piece;
  });
  if (cursorCol && cursorCol > chars.length) out += '<span class="cx">&nbsp;</span>';
  return out || "";
}
function pane(lines, cls, label, cur) {
  if (!lines.length) return '<div class="pane ' + cls + '"><span class="lab pl">' + label +
    '</span><div class="nolines">empty buffer</div></div>';
  const body = lines.map(([tag, n, text]) => {
    const k = tag === "-" ? " cut" : tag === "+" ? " add" : "";
    const cc = (cur && cls === "b" && n === cur[0] && tag !== "+") ? cur[1] : 0;
    return '<div class="ln' + k + '"><span class="g">' + n + '</span><span class="t">' +
      (vis(text, cc) || "&nbsp;") + "</span></div>";
  }).join("");
  return '<div class="pane ' + cls + '"><span class="lab pl">' + label + "</span>" + body + "</div>";
}
function card(row) {
  const s = store[row.id] || {}, t = textOf(row.id) || "";
  const stt = status(row), edited = t !== BASE.get(row.id);
  const flags = check(row).map(f => '<span class="flag ' + f.k + '">' + esc(f.m) + "</span>").join("");
  const okf = (!flags && t.trim()) ? '<span class="flag okf">passes every check</span>' : "";
  const el = document.createElement("div");
  el.className = "card st-" + stt;
  el.dataset.case = row.id;
  el.innerHTML =
    '<div class="chead">' +
      '<span class="pill p-' + stt + '">' + (stt === "done" ? "written" : stt === "bad" ? "leak" : "to write") + "</span>" +
      '<span class="cid">' + esc(row.id) + "</span>" +
      '<span class="chip">' + esc(row.cat) + "</span>" +
      '<span class="chip">' + esc(row.tier) + "</span>" +
      '<span class="chip">' + esc(row.mode) + "</span>" +
      '<span class="chip">cursor ' + row.cur[0] + ":" + row.cur[1] + "</span>" +
      '<span class="spacer"></span>' +
      '<span class="keys">' + esc(row.cmd) + "</span>" +
    "</div>" +
    '<div class="insrow"><textarea class="ins' + (edited ? " edited" : "") + (stt === "bad" ? " bad" : "") +
      '" rows="1" spellcheck="true" placeholder="What did the user want done to this buffer?">' +
      esc(t) + "</textarea></div>" +
    (row.retargeted ? '<div class="rt"><span class="whytag">command retargeted</span>'
      + '<span class="whytxt">Mined as <b>' + esc(row.was) + '</b> \u2014 a fixture no plain '
      + 'request could motivate. Replaced with an idiomatic command that reproduces the same '
      + 'verified buffer. Upstream keystrokes kept in <span class="mono">source_command</span>.'
      + '</span></div>' : "") +
    (row.why[0] ? '<div class="why"><span class="whytag">held back &middot; ' +
      esc(row.why[0]) + '</span><span class="whytxt">' + esc(row.why[1]) + '</span></div>' : "") +
    '<div class="flags">' + flags + okf + "</div>" +
    '<div class="panes">' + pane(row.L, "b", "buffer before", row.cur) +
      pane(row.R, "a", "buffer after") + "</div>" +
    '<div class="cfoot"><span class="mk">' +
      '<button class="done" aria-pressed="' + (!!s.done) + '">reviewed</button> ' +
      '<button class="skip" aria-pressed="' + (!!s.skip) + '">skip &mdash; leave pending</button> ' +
      '<button class="rst">revert</button>' +
      "</span>" +
      '<span class="spacer"></span>' +
      '<details class="title"><summary>source &amp; upstream title</summary>' +
        '<div class="body"><b>Leaks the command &mdash; reviewer context only.</b><br>' +
        esc(row.src) + (row.sfile ? " &middot; " + esc(row.sfile) : "") +
        (row.title ? "<br>" + esc(row.title) : "") + "</div></details>" +
    "</div>";
  wire(el, row);
  return el;
}
function autosize(ta) { ta.style.height = "auto"; ta.style.height = (ta.scrollHeight + 2) + "px"; }
function wire(el, row) {
  const ta = el.querySelector(".ins");
  requestAnimationFrame(() => autosize(ta));
  ta.addEventListener("input", () => {
    st(row.id).t = ta.value; save(); autosize(ta); refreshCard(el, row); tally();
  });
  ta.addEventListener("focus", () => setCurrent(row.id, false));
  ta.addEventListener("keydown", e => { if (e.key === "Escape") { ta.blur(); e.stopPropagation(); } });
  el.querySelector(".done").addEventListener("click", e => {
    const s = st(row.id); s.done = !s.done; if (s.done) s.skip = false; save();
    e.currentTarget.setAttribute("aria-pressed", !!s.done); refreshCard(el, row); tally();
  });
  el.querySelector(".skip").addEventListener("click", e => {
    const s = st(row.id); s.skip = !s.skip; if (s.skip) s.done = false; save();
    e.currentTarget.setAttribute("aria-pressed", !!s.skip); refreshCard(el, row); tally();
  });
  el.querySelector(".rst").addEventListener("click", () => {
    delete store[row.id]; save(); const fresh = card(row);
    el.replaceWith(fresh); tally();
  });
  el.addEventListener("mousedown", () => setCurrent(row.id, false));
}
function refreshCard(el, row) {
  const stt = status(row), t = textOf(row.id) || "";
  el.className = "card st-" + stt + (el.classList.contains("cur") ? " cur" : "");
  const pill = el.querySelector(".pill");
  pill.className = "pill p-" + stt;
  pill.textContent = stt === "done" ? "written" : stt === "bad" ? "leak" : "to write";
  const ta = el.querySelector(".ins");
  ta.classList.toggle("edited", t !== BASE.get(row.id));
  ta.classList.toggle("bad", stt === "bad");
  const fl = check(row).map(f => '<span class="flag ' + f.k + '">' + esc(f.m) + "</span>").join("");
  el.querySelector(".flags").innerHTML = fl || (t.trim() ? '<span class="flag okf">passes every check</span>' : "");
}

/* ---- filters ---- */
const CATS = [...new Set(ROWS.map(r => r.cat))].sort();
const SRCS = [...new Set(ROWS.map(r => r.src))].sort();
let filt = { st: "all", cat: "all", src: "all", q: "" };
function match(r) {
  const s = store[r.id] || {}, stt = status(r);
  if (filt.st === "pending" && (textOf(r.id) || "").trim()) return false;
  if (filt.st === "retargeted" && !r.retargeted) return false;
  if (filt.st === "bad" && stt !== "bad") return false;
  if (filt.st === "edited" && textOf(r.id) === BASE.get(r.id)) return false;
  if (filt.st === "unreviewed" && s.done) return false;
  if (filt.st === "skip" && !s.skip) return false;
  if (filt.cat !== "all" && r.cat !== filt.cat) return false;
  if (filt.src !== "all" && r.src !== filt.src) return false;
  if (filt.q) {
    const hay = (r.id + " " + r.cmd + " " + (textOf(r.id) || "")).toLowerCase();
    if (!hay.includes(filt.q)) return false;
  }
  return true;
}
function buildRail() {
  const rail = document.getElementById("rail");
  const chips = [["all", "all"], ["retargeted", "retargeted"], ["bad", "leaks"],
                 ["edited", "edited"], ["unreviewed", "unreviewed"], ["skip", "skipped"]];
  rail.innerHTML =
    '<span class="grp"><span class="lab">show</span>' +
    chips.map(([k, l]) => '<button class="f" data-st="' + k + '" aria-pressed="' +
      (filt.st === k) + '">' + l + '<span class="c" data-c="' + k + '"></span></button>').join("") +
    "</span>" +
    '<span class="grp"><span class="lab">category</span><select id="fcat"><option value="all">all</option>' +
      CATS.map(c => '<option value="' + esc(c) + '">' + esc(c) + "</option>").join("") + "</select></span>" +
    '<span class="grp"><span class="lab">source</span><select id="fsrc"><option value="all">all</option>' +
      SRCS.map(c => '<option value="' + esc(c) + '">' + esc(c) + "</option>").join("") + "</select></span>" +
    '<input type="search" id="q" placeholder="id, command or text…">' +
    '<span class="count" id="count"></span>';
  rail.querySelectorAll("[data-st]").forEach(b => b.addEventListener("click", () => {
    filt.st = b.dataset.st; render();
  }));
  document.getElementById("fcat").addEventListener("change", e => { filt.cat = e.target.value; render(); });
  document.getElementById("fsrc").addEventListener("change", e => { filt.src = e.target.value; render(); });
  document.getElementById("q").addEventListener("input", e => {
    filt.q = e.target.value.toLowerCase().trim(); render();
  });
}

/* ---- list ---- */
let shown = 60, view = [], current = null;
function render() {
  view = ROWS.filter(match);
  shown = Math.min(Math.max(shown, 60), Math.max(view.length, 60));
  const host = document.getElementById("cards");
  host.innerHTML = "";
  if (!view.length) { host.innerHTML = '<div class="empty">Nothing matches those filters.</div>'; }
  const frag = document.createDocumentFragment();
  view.slice(0, shown).forEach(r => frag.appendChild(card(r)));
  host.appendChild(frag);
  document.getElementById("more").style.display = view.length > shown ? "" : "none";
  document.querySelectorAll("[data-st]").forEach(b =>
    b.setAttribute("aria-pressed", filt.st === b.dataset.st));
  tally();
  if (current && !view.some(r => r.id === current)) current = null;
  paintCurrent();
}
document.getElementById("more").addEventListener("click", () => { shown += 50; render(); });

function tally() {
  const n = ROWS.length;
  let written = 0, todo = 0, bad = 0, edited = 0, done = 0, skip = 0;
  for (const r of ROWS) {
    const s = store[r.id] || {}, stt = status(r);
    if (stt === "bad") bad++; else if (stt === "todo") todo++; else written++;
    if (textOf(r.id) !== BASE.get(r.id)) edited++;
    if (s.done) done++;
    if (s.skip) skip++;
  }
  document.getElementById("agg").innerHTML =
    '<div><span class="v">' + n + '</span><span class="k">mined cases</span></div>' +
    '<div><span class="v ok">' + written + '</span><span class="k">have an instruction</span></div>' +
    '<div><span class="v no">' + todo + '</span><span class="k">still to write</span></div>' +
    '<div><span class="v" style="color:var(--stop)">' + bad + '</span><span class="k">would be refused</span></div>' +
    '<div><span class="v ed">' + edited + '</span><span class="k">edited here</span></div>' +
    '<div><span class="v">' + done + '</span><span class="k">marked reviewed</span></div>';
  const pb = document.getElementById("pbar");
  const pc = x => (100 * x / n).toFixed(3) + "%";
  pb.innerHTML =
    '<i style="width:' + pc(done) + ';background:var(--pass)"></i>' +
    '<i style="width:' + pc(skip) + ';background:var(--warn)"></i>' +
    '<i style="width:' + pc(Math.max(0, n - done - skip)) + ';background:var(--raise)"></i>';
  const c = document.getElementById("count");
  if (c) c.textContent = view.length + " of " + n + " shown";
  document.getElementById("dockcount").textContent =
    done + " reviewed · " + skip + " skipped · " + edited + " edited";
  const cnt = { all: ROWS.length, pending: 0, retargeted: 0, bad: 0, edited: 0, unreviewed: 0, skip: 0 };
  for (const r of ROWS) {
    const s = store[r.id] || {}, stt = status(r);
    if (!(textOf(r.id) || "").trim()) cnt.pending++;
    if (r.retargeted) cnt.retargeted++;
    if (stt === "bad") cnt.bad++;
    if (textOf(r.id) !== BASE.get(r.id)) cnt.edited++;
    if (!s.done) cnt.unreviewed++;
    if (s.skip) cnt.skip++;
  }
  document.querySelectorAll("[data-c]").forEach(e => { e.textContent = cnt[e.dataset.c]; });
}

/* ---- keyboard ---- */
function setCurrent(id, scroll) {
  current = id; paintCurrent();
  if (scroll) {
    const el = document.querySelector('.card[data-case="' + CSS.escape(id) + '"]');
    if (el) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }
}
function paintCurrent() {
  document.querySelectorAll(".card").forEach(c =>
    c.classList.toggle("cur", c.dataset.case === current));
}
function step(d) {
  const list = view.slice(0, shown);
  if (!list.length) return;
  let i = list.findIndex(r => r.id === current);
  i = i < 0 ? (d > 0 ? 0 : list.length - 1) : i + d;
  if (i < 0) i = 0;
  if (i >= list.length) {
    if (view.length > shown) { shown += 50; render(); }
    i = Math.min(i, view.slice(0, shown).length - 1);
  }
  const nxt = view.slice(0, shown)[i];
  if (nxt) setCurrent(nxt.id, true);
}
document.addEventListener("keydown", e => {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = tag === "textarea" || tag === "input" || tag === "select";
  if (typing) return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "j") { e.preventDefault(); step(1); }
  else if (e.key === "k") { e.preventDefault(); step(-1); }
  else if (e.key === "/") { e.preventDefault(); document.getElementById("q").focus(); }
  else if (e.key === "Enter" && current) {
    e.preventDefault();
    const el = document.querySelector('.card[data-case="' + CSS.escape(current) + '"]');
    if (el) el.querySelector(".ins").focus();
  } else if ((e.key === "m" || e.key === "s") && current) {
    e.preventDefault();
    const el = document.querySelector('.card[data-case="' + CSS.escape(current) + '"]');
    if (el) el.querySelector(e.key === "m" ? ".done" : ".skip").click();
  }
});

/* ---- export ---- */
const FILES = [...new Set(ROWS.map(r => r.file))].sort();
let outFile = FILES[0];
function changed(file) {
  const m = {};
  for (const r of ROWS) {
    if (r.file !== file) continue;
    const s = store[r.id] || {};
    if (s.skip) continue;
    const t = (textOf(r.id) || "").trim();
    if (!t || t === (BASE.get(r.id) || "").trim()) continue;
    if (isErr(r)) continue;
    m[r.id] = t;
  }
  return m;
}
function paintDlg() {
  const m = changed(outFile), n = Object.keys(m).length;
  document.getElementById("dlgtabs").innerHTML = FILES.map(f =>
    '<button class="f" data-file="' + esc(f) + '" aria-pressed="' + (f === outFile) + '">' +
    esc(f) + '<span class="c">' + Object.keys(changed(f)).length + "</span></button>").join("");
  document.getElementById("dlgtabs").querySelectorAll("[data-file]").forEach(b =>
    b.addEventListener("click", () => { outFile = b.dataset.file; paintDlg(); }));
  document.getElementById("out").value = JSON.stringify(m, null, 1);
  const skipped = ROWS.filter(r => (store[r.id] || {}).skip).length;
  document.getElementById("dlgsub").textContent =
    n + " edited instruction" + (n === 1 ? "" : "s") + " for " + outFile +
    (skipped ? " · " + skipped + " row" + (skipped === 1 ? "" : "s") +
      " marked skip and deliberately left out" : "") +
    ". Rows that would be refused as leaks are excluded.";
  document.getElementById("cmd").textContent =
    "conda run -n vimBot python scripts/apply_instructions.py \\\n" +
    "  --cases data/v3/" + outFile + " \\\n" +
    "  --instructions authored.json --provenance owner-edited --dry-run";
}
document.getElementById("export").addEventListener("click", () => {
  paintDlg(); document.getElementById("dlg").showModal();
});
document.getElementById("close").addEventListener("click", () => document.getElementById("dlg").close());
document.getElementById("copy").addEventListener("click", async e => {
  const ta = document.getElementById("out");
  ta.select();
  let ok = false;
  try { await navigator.clipboard.writeText(ta.value); ok = true; } catch (err) { ok = false; }
  e.currentTarget.textContent = ok ? "Copied" : "Press ⌘C / Ctrl+C";
  setTimeout(() => { e.currentTarget.textContent = "Copy JSON"; }, 1900);
});

function boot() {
  buildRail();
  render();
  const b = document.getElementById("boot");
  if (b) b.remove();
}
try {
  boot();
} catch (err) {
  const b = document.getElementById("boot") || document.getElementById("cards");
  if (b) {
    b.className = "boot err";
    b.textContent = "This page failed to start.\n\n" +
      (err && err.stack ? err.stack : String(err)) +
      "\n\nCopy this text back to Claude Code \u2014 it names the line that broke.";
  }
  throw err;
}
</script>
"""


# JS escape sequences the template must emit untouched. If PAGE ever stops
# being a raw string these vanish, the script fails to parse, and the page
# renders blank with no error — the failure that produced this check.
REQUIRED_JS = [
    r"if (/\n/.test(t))",
    r"/\bpress\b|\bkeystroke|\btype the key/",
    r'"This page failed to start.\n\n"',
    r'py \\\n" +',
]
# Control characters that only appear when \b, \t, \f, \v or \0 were eaten.
FORBIDDEN = {chr(c) for c in list(range(0, 9)) + [11, 12] + list(range(14, 32))}


def check_escapes(page):
    for frag in REQUIRED_JS:
        if frag not in page:
            sys.exit("build guard: %r missing from the page — is PAGE still a "
                     "raw string? Python ate a JS escape." % frag)
    bad = sorted({c for c in page if c in FORBIDDEN})
    if bad:
        sys.exit("build guard: control characters in output: %r — a JS escape "
                 "was interpreted by Python." % bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cohort", default=str(COHORT))
    args = ap.parse_args()

    rows, dropped = collect(pathlib.Path(args.cohort))
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")  # cannot close the host <script>

    page = PAGE.replace("__PAYLOAD__", payload).replace("__TOTAL__", str(len(rows))).replace("__DROPPED__", str(len(dropped)))
    check_escapes(page)
    print("  guard: " + check_page(page, REQUIRED_JS, "Instruction Bench"))
    out = pathlib.Path(args.out)
    out.write_text(page, encoding="utf-8")

    todo = sum(1 for r in rows if not r["ins"])
    print("cases on page   : %d  (%d with an instruction, %d still pending)"
          % (len(rows), len(rows) - todo, todo))
    if dropped:
        print("dropped from v3 : %d  %s" % (len(dropped), ", ".join(dropped)))
    print("page size       : %.0f KB" % (len(page.encode("utf-8")) / 1024))
    print("wrote           : %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
