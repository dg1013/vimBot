#!/usr/bin/env python3
"""Build the Synth/Turk review artifact: the NL-RX cases whose commands passed.

Every case in dataset v2 derived from NL-RX-Synth or NL-RX-Turk (MIT (Massachusetts Institute of Technology), via
deep-regex) whose vim command was validated end-to-end — the command parses,
runs, and reproduces `buffer_after` in real vim. Those gates are why v2 looked
finished. The page exists to review the axis they cannot see: whether the
English attached to each valid command actually describes it.

Three things are surfaced per case that the case file alone does not show:

  * the static screen verdicts — `garble_screen.classify()` rules and
    `anchor_mismatch()`, the instruments behind the 25.4% / 26.7% approval
    figures in DECISIONS 2026-08-14;
  * the corpus sentences the instruction was seeded from, with a measured
    similarity, because the shipped instruction is usually the in-session
    teacher's rewrite rather than corpus text (median similarity 0.52);
  * per-line kept/cut marking, since every command here is `g//d` or `v//d`
    and `buffer_after` is therefore a line subsequence of `buffer_before`.

Usage (repo root):
    conda run -n vimBot python scripts/build_synth_turk_review.py --out page.html
"""
import argparse
import collections
import difflib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "deep_regex"))
from garble_screen import classify, anchor_mismatch  # noqa: E402

CASES = REPO / "data" / "generated_cases.jsonl"
EXT = REPO / "data" / "external" / "deep-regex"
PREFIXES = ("Delete ", "Keep only ", "Remove ")


def corpus_lines(name):
    p = EXT / name / "src.txt"
    return p.read_text(encoding="utf-8", errors="replace").splitlines() if p.exists() else []


def core(instruction):
    """The instruction minus the polarity wrapper generate_data.py adds."""
    for p in PREFIXES:
        if instruction.startswith(p):
            return instruction[len(p):].rstrip(".").strip()
    return instruction.rstrip(".").strip()


def kept_cut(before, after):
    """Mark each source line kept or cut. Both commands here are line filters,
    so buffer_after is a subsequence of buffer_before and the mapping is exact."""
    a = before.split("\n")
    b = after.split("\n")
    if a and a[-1] == "":
        a.pop()
    if b and b[-1] == "":
        b.pop()
    out, j = [], 0
    for line in a:
        if j < len(b) and b[j] == line:
            out.append([1, line])
            j += 1
        else:
            out.append([0, line])
    ok = j == len(b)          # every surviving line was accounted for
    return out, ok


def collect():
    src = {n: corpus_lines(n) for n in ("NL-RX-Synth", "NL-RX-Turk")}
    rows, unmapped = [], 0
    for line in CASES.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if not any(c.startswith("NL-RX") for c in (r.get("source_corpus") or [])):
            continue

        c = core(r["instruction"]).lower()
        seeds, best = [], 0.0
        for si in r.get("source_index") or []:
            name, idx = si.rsplit(":", 1)
            lines = src.get(name)
            if not lines or not (0 <= int(idx) < len(lines)):
                continue
            text = lines[int(idx)].strip()
            ratio = difflib.SequenceMatcher(None, c, text.lower()).ratio()
            seeds.append([name.replace("NL-RX-", ""), text, round(ratio, 2)])
            best = max(best, ratio)
        seeds.sort(key=lambda s: -s[2])

        regex = r.get("source_regex", "")
        rules = sorted(classify(regex, r["instruction"]))
        anchor = bool(anchor_mismatch(regex, r["instruction"]))
        marks, ok = kept_cut(r["buffer_before"], r["buffer_after"])
        if not ok:
            unmapped += 1

        rows.append({
            "i": r["id"],
            "t": r.get("tier", ""),
            "p": "g" if r["gold_command"].startswith("g/") else "v",
            "c": r["gold_command"],
            "x": regex,
            "n": r["instruction"],
            "s": seeds[:4],
            "r": round(best, 2),
            "g": rules,
            "a": anchor,
            "b": marks,
        })
    return rows, unmapped


PAGE = r"""<title>Synth and Turk Commands</title>
<style>
:root{
  --ground:#F2F0EE; --surface:#FCFBFA; --sunk:#E9E5E2; --raise:#E4DFDB;
  --ink:#231E22; --ink2:#4A4046; --muted:#6E6369; --faint:#9A8F95;
  --line:#DAD3CF; --line2:#E7E1DD;
  --accent:#6B2D5C; --accent2:#8A3E77; --accent-soft:#EFE2EC; --on-accent:#FCFBFA;
  --pass:#2E6E4E; --pass-bg:#DFEDE5; --pass-line:#AACFBB;
  --warn:#8C5A15; --warn-bg:#F4E9D8; --warn-line:#DFC79A;
  --stop:#9C3535; --stop-bg:#F3E1E0; --stop-line:#DFB4B2;
  --cut:#9C3535; --cut-bg:#F6E6E5; --keep:#2E6E4E; --keep-bg:#E2EFE8;
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
    --cut:#E08484; --cut-bg:#2E1C1C; --keep:#6BC195; --keep-bg:#152720;
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
  --cut:#E08484; --cut-bg:#2E1C1C; --keep:#6BC195; --keep-bg:#152720;
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
.lede{ max-width:70ch; color:var(--ink2); font-size:15.5px; }
.lede b{ color:var(--ink); font-weight:600; }
.lede code{ font-family:var(--mono); font-size:.9em; background:var(--sunk);
  padding:1px 4px; border-radius:3px; }

.agg{ display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:7px; overflow:hidden; margin-top:18px; }
.agg div{ background:var(--surface); padding:10px 13px; }
.agg .v{ font-family:var(--mono); font-size:19px; font-weight:700; letter-spacing:-.02em;
  display:block; font-variant-numeric:tabular-nums; }
.agg .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); margin-top:2px; display:block; line-height:1.35; }
.agg .v.ok{ color:var(--pass); } .agg .v.no{ color:var(--stop); } .agg .v.mid{ color:var(--warn); }

.note{ border-left:2px solid var(--accent); padding:4px 0 4px 13px; color:var(--ink2);
  font-size:13.5px; max-width:76ch; margin:16px 0 0; }
.note b{ color:var(--ink); }

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

main{ padding:22px 0 90px; }
.cards{ display:flex; flex-direction:column; gap:11px; }
.card{ background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--pass);
  border-radius:6px; overflow:hidden; }
.card.v-garble{ border-left-color:var(--stop); }
.card.v-anchor{ border-left-color:var(--warn); }
.card.v-both{ border-left-color:var(--stop); }
.chead{ display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; padding:8px 13px;
  border-bottom:1px solid var(--line2); }
.pill{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; padding:2px 6px; border-radius:3px; white-space:nowrap; border:1px solid; }
.p-clean{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }
.p-anchor{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.p-garble,.p-both{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); }
.cid{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:3px;
  background:var(--raise); color:var(--ink2); white-space:nowrap; }
.spacer{ flex:1 1 auto; }
.ok-run{ font-family:var(--mono); font-size:9.5px; letter-spacing:.06em; text-transform:uppercase;
  color:var(--pass); border:1px solid var(--pass-line); background:var(--pass-bg);
  padding:2px 6px; border-radius:3px; white-space:nowrap; }

.cmdrow{ padding:10px 13px 2px; display:flex; flex-direction:column; gap:6px; }
.cmd{ font-family:var(--mono); font-size:13px; font-weight:600; background:var(--accent-soft);
  color:var(--accent); padding:6px 9px; border-radius:4px; overflow-x:auto; white-space:pre; }
.pat{ font-family:var(--mono); font-size:11.5px; color:var(--muted); overflow-x:auto; white-space:pre; }
.pat b{ color:var(--ink2); font-weight:600; }
.instr{ padding:8px 13px 2px; font-size:15px; font-weight:600; color:var(--ink); line-height:1.45; }
.flags{ display:flex; flex-wrap:wrap; gap:6px; padding:7px 13px 9px; }
.flag{ font-family:var(--mono); font-size:11px; padding:2px 7px; border-radius:3px; border:1px solid; }
.flag.err{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); font-weight:600; }
.flag.warnf{ background:var(--warn-bg); color:var(--warn); border-color:var(--warn-line); }
.flag.okf{ background:var(--pass-bg); color:var(--pass); border-color:var(--pass-line); }

.seeds{ margin:0 13px 9px; border:1px solid var(--line2); border-radius:5px; overflow:hidden; }
.seeds .sh{ background:var(--sunk); padding:5px 10px; }
.seed{ display:flex; gap:9px; align-items:baseline; padding:5px 10px; border-top:1px solid var(--line2);
  font-size:13px; color:var(--ink2); }
.seed .who{ font-family:var(--mono); font-size:9.5px; font-weight:700; letter-spacing:.06em;
  text-transform:uppercase; color:var(--accent); flex:0 0 3.4rem; }
.seed .rt{ font-family:var(--mono); font-size:10.5px; color:var(--faint); flex:0 0 2.4rem;
  text-align:right; font-variant-numeric:tabular-nums; }

.panes{ border-top:1px solid var(--line2); padding:8px 12px; overflow-x:auto; }
.ln{ display:flex; gap:0; white-space:pre; font-family:var(--mono); font-size:12px; line-height:1.5; }
.ln .g{ color:var(--faint); opacity:.65; user-select:none; text-align:right; min-width:2.1em;
  flex:0 0 auto; padding-right:8px; font-variant-numeric:tabular-nums; }
.ln .m{ flex:0 0 auto; width:1.6em; font-weight:700; }
.ln.keep{ background:var(--keep-bg); } .ln.keep .m{ color:var(--keep); }
.ln.cut{ background:var(--cut-bg); } .ln.cut .m{ color:var(--cut); }
.ln.cut .t{ color:var(--cut); text-decoration:line-through; text-decoration-thickness:1px; }
.ws{ color:var(--accent); opacity:.38; }

.more{ display:flex; justify-content:center; padding:18px 0 0; }
.empty{ padding:40px 0; text-align:center; color:var(--muted); font-family:var(--mono); font-size:13px; }
.boot{ padding:34px 0; text-align:center; color:var(--muted); font-family:var(--mono);
  font-size:13px; line-height:1.6; }
.boot.err{ color:var(--stop); text-align:left; white-space:pre-wrap; overflow-wrap:anywhere;
  background:var(--stop-bg); border:1px solid var(--stop-line); border-radius:6px; padding:14px 16px; }
.foot{ margin-top:32px; padding-top:18px; border-top:1px solid var(--line); font-size:13px;
  color:var(--muted); max-width:78ch; }
.foot b{ color:var(--ink2); }
</style>

<header><div class="wrap">
  <div class="lab">vimBot &middot; dataset v2 &middot; NL-RX slice</div>
  <h1>Synth and Turk Commands</h1>
  <p class="lede">
    Every case derived from <b>NL-RX-Synth</b> or <b>NL-RX-Turk</b> whose vim command
    <b>passed</b> &mdash; it parses, runs, and reproduces its buffer in real vim. All __TOTAL__
    of them. These gates are why v2 looked finished; the page exists to review the one axis
    they cannot see. Each card pairs the validated command with the English attached to it,
    the corpus sentences that English was seeded from, and the static screen verdicts behind
    the <b>25.4% / 26.7%</b> approval figures.
  </p>
  <div class="agg" id="agg"></div>
  <p class="note" id="note"></p>
</div></header>

<div class="rail"><div class="wrap" id="rail"></div></div>

<main class="wrap">
  <div class="cards" id="cards"><div class="boot" id="boot">Loading __TOTAL__ cases&hellip;
    if this message stays, the page failed to start and the reason will appear here.</div></div>
  <div class="more"><button class="btn" id="more">Show 50 more</button></div>
  <p class="foot">
    <b>What &ldquo;passed&rdquo; means here.</b> Every command on this page was validated three
    ways: re-derived in real vim at ingest so any conversion error surfaced as a
    <span class="mono">mismatch</span> reject, re-audited witness-by-witness across all 9,495
    conversions, and cross-checked against an independent pure-Python implementation
    (14,433/14,434 agreeing). The commands are sound. That is precisely the point &mdash; no
    execution gate can see an instruction that describes something else.
    <br><br>
    <b>The two screens.</b> <span class="mono">garble</span> flags patterns whose DSL operators
    template into unusable English &mdash; token-level conjunction over class atoms, negated
    position expressions, vacuous group-stars, bare-dot alternation branches.
    <span class="mono">anchor</span> flags the A-rule: these regexes are <i>full-string</i>
    matches, but the instruction templates phrase <i>containment</i>, so any anchored pattern
    whose wording does not disclose the anchor contradicts its own command. Validated at
    precision 1.00 / recall 0.92 against a 382-case hand-labelled set.
    <br><br>
    <b>Read the seed panel carefully.</b> The shipped instruction is usually <i>not</i> corpus
    text &mdash; median similarity to its source sentences is 0.52, and only 248 of __TOTAL__ sit
    at 0.85 or above. The in-session teacher rewrote them, seeded by Synth/Turk prose. That
    makes this slice a test of the seed rather than of the typist, which is the finding that
    moved v3 to human-authored instructions.
  </p>
</main>

<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
"use strict";
const ROWS = JSON.parse(document.getElementById("payload").textContent);
const esc = s => String(s).replace(/[&<>"]/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const vis = s => esc(s).replace(/ /g, '<span class="ws">·</span>');

function verdict(r) {
  if (r.g.length && r.a) return "both";
  if (r.g.length) return "garble";
  if (r.a) return "anchor";
  return "clean";
}
const LABEL = { clean: "screens clean", garble: "garble", anchor: "anchor mismatch",
                both: "garble + anchor" };

function card(r) {
  const v = verdict(r);
  const el = document.createElement("div");
  el.className = "card v-" + v;
  const flags = [];
  if (r.a) flags.push('<span class="flag warnf">anchor mismatch — pattern is full-string, wording says containment</span>');
  r.g.forEach(g => flags.push('<span class="flag err">garble: ' + esc(g) + "</span>"));
  if (!flags.length) flags.push('<span class="flag okf">passes both static screens</span>');
  const seeds = r.s.length
    ? '<div class="seeds"><div class="sh"><span class="lab">corpus sentences this was seeded from</span></div>' +
      r.s.map(s => '<div class="seed"><span class="who">' + esc(s[0]) + '</span>' +
        '<span class="rt">' + s[2].toFixed(2) + "</span><span>" + esc(s[1]) + "</span></div>").join("") +
      "</div>"
    : "";
  const lines = r.b.map((m, i) =>
    '<div class="ln ' + (m[0] ? "keep" : "cut") + '"><span class="g">' + (i + 1) + "</span>" +
    '<span class="m">' + (m[0] ? "✓" : "✗") + '</span><span class="t">' +
    (vis(m[1]) || "&nbsp;") + "</span></div>").join("");
  el.innerHTML =
    '<div class="chead">' +
      '<span class="pill p-' + v + '">' + LABEL[v] + "</span>" +
      '<span class="cid">' + esc(r.i) + "</span>" +
      '<span class="chip">' + esc(r.t) + "</span>" +
      '<span class="chip">' + (r.p === "g" ? "delete matching" : "keep matching") + "</span>" +
      '<span class="chip">seed similarity ' + r.r.toFixed(2) + "</span>" +
      '<span class="spacer"></span>' +
      '<span class="ok-run">command verified in vim</span>' +
    "</div>" +
    '<div class="cmdrow"><div class="cmd">' + esc(r.c) + "</div>" +
      '<div class="pat"><b>source pattern</b>  ' + esc(r.x) + "</div></div>" +
    '<div class="instr">' + esc(r.n) + "</div>" +
    '<div class="flags">' + flags.join("") + "</div>" +
    seeds +
    '<div class="panes">' + lines + "</div>";
  return el;
}

let filt = { v: "all", t: "all", p: "all", sim: "all", q: "" };
function match(r) {
  const v = verdict(r);
  if (filt.v !== "all" && v !== filt.v) return false;
  if (filt.t !== "all" && r.t !== filt.t) return false;
  if (filt.p !== "all" && r.p !== filt.p) return false;
  if (filt.sim === "near" && r.r < 0.85) return false;
  if (filt.sim === "far" && r.r >= 0.5) return false;
  if (filt.q) {
    const hay = (r.i + " " + r.c + " " + r.n + " " + r.x).toLowerCase();
    if (!hay.includes(filt.q)) return false;
  }
  return true;
}

const TIERS = [...new Set(ROWS.map(r => r.t))].sort();
function buildRail() {
  const chips = [["all", "all"], ["clean", "screens clean"], ["anchor", "anchor"],
                 ["garble", "garble"], ["both", "both"]];
  const rail = document.getElementById("rail");
  rail.innerHTML =
    '<span class="grp"><span class="lab">verdict</span>' +
    chips.map(([k, l]) => '<button class="f" data-v="' + k + '" aria-pressed="' + (filt.v === k) +
      '">' + l + '<span class="c" data-c="' + k + '"></span></button>').join("") + "</span>" +
    '<span class="grp"><span class="lab">tier</span><select id="ft"><option value="all">all</option>' +
      TIERS.map(t => '<option value="' + esc(t) + '">' + esc(t) + "</option>").join("") + "</select></span>" +
    '<span class="grp"><span class="lab">polarity</span><select id="fp">' +
      '<option value="all">all</option><option value="g">delete matching</option>' +
      '<option value="v">keep matching</option></select></span>' +
    '<span class="grp"><span class="lab">seed</span><select id="fs">' +
      '<option value="all">any similarity</option><option value="near">near-verbatim (&ge;0.85)</option>' +
      '<option value="far">rewritten (&lt;0.50)</option></select></span>' +
    '<input type="search" id="q" placeholder="id, command, pattern, text…">' +
    '<span class="count" id="count"></span>';
  rail.querySelectorAll("[data-v]").forEach(b =>
    b.addEventListener("click", () => { filt.v = b.dataset.v; render(); }));
  document.getElementById("ft").addEventListener("change", e => { filt.t = e.target.value; render(); });
  document.getElementById("fp").addEventListener("change", e => { filt.p = e.target.value; render(); });
  document.getElementById("fs").addEventListener("change", e => { filt.sim = e.target.value; render(); });
  document.getElementById("q").addEventListener("input", e => {
    filt.q = e.target.value.toLowerCase().trim(); render();
  });
}

let shown = 50, view = [];
function render() {
  view = ROWS.filter(match);
  const host = document.getElementById("cards");
  host.innerHTML = "";
  if (!view.length) host.innerHTML = '<div class="empty">Nothing matches those filters.</div>';
  const frag = document.createDocumentFragment();
  view.slice(0, shown).forEach(r => frag.appendChild(card(r)));
  host.appendChild(frag);
  document.getElementById("more").style.display = view.length > shown ? "" : "none";
  document.querySelectorAll("[data-v]").forEach(b =>
    b.setAttribute("aria-pressed", filt.v === b.dataset.v));
  document.getElementById("count").textContent =
    view.length.toLocaleString() + " of " + ROWS.length.toLocaleString() + " shown";
}
document.getElementById("more").addEventListener("click", () => { shown += 50; render(); });

function tally() {
  const n = ROWS.length;
  const c = { clean: 0, garble: 0, anchor: 0, both: 0 };
  let near = 0, sim = 0;
  ROWS.forEach(r => { c[verdict(r)]++; if (r.r >= 0.85) near++; sim += r.r; });
  const flagged = n - c.clean;
  document.getElementById("agg").innerHTML =
    '<div><span class="v">' + n.toLocaleString() + '</span><span class="k">commands passed vim</span></div>' +
    '<div><span class="v ok">100%</span><span class="k">exec-match</span></div>' +
    '<div><span class="v no">' + Math.round(100 * flagged / n) + '%</span><span class="k">instruction flagged by a screen</span></div>' +
    '<div><span class="v mid">' + c.anchor.toLocaleString() + '</span><span class="k">anchor mismatch only</span></div>' +
    '<div><span class="v no">' + (c.garble + c.both).toLocaleString() + '</span><span class="k">garble rules fired</span></div>' +
    '<div><span class="v ok">' + c.clean.toLocaleString() + '</span><span class="k">clean on both screens</span></div>' +
    '<div><span class="v">' + (sim / n).toFixed(2) + '</span><span class="k">mean seed similarity</span></div>' +
    '<div><span class="v">' + near.toLocaleString() + '</span><span class="k">near-verbatim corpus text</span></div>';
  document.getElementById("note").innerHTML =
    "<b>Every command here is correct and every one of them ran.</b> " +
    "What separates the green cards from the red is not vim — it is whether the sentence " +
    "a model would be trained on describes the edit the command performs.";
  const cnt = { all: n, clean: c.clean, garble: c.garble, anchor: c.anchor, both: c.both };
  document.querySelectorAll("[data-c]").forEach(e => {
    e.textContent = (cnt[e.dataset.c] || 0).toLocaleString();
  });
}

function boot() {
  buildRail(); tally(); render();
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
      "\n\nCopy this text back to Claude Code — it names the line that broke.";
  }
  throw err;
}
</script>
"""

# JS escapes the template must emit untouched. A non-raw PAGE silently eats
# these: the page then fails to parse and renders blank with no error.
REQUIRED_JS = [
    r'.replace(/[&<>"]/g',
    r'.replace(/ /g, ',
    r'"This page failed to start.\n\n"',
    r"·",
]
FORBIDDEN = {chr(c) for c in list(range(0, 9)) + [11, 12] + list(range(14, 32))}


def check_escapes(page):
    for frag in REQUIRED_JS:
        if frag not in page:
            sys.exit("build guard: %r missing — is PAGE still a raw string?" % frag)
    bad = sorted({c for c in page if c in FORBIDDEN})
    if bad:
        sys.exit("build guard: control characters in output: %r" % bad)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows, unmapped = collect()
    if unmapped:
        print("WARNING: %d rows where buffer_after is not a line subsequence" % unmapped)

    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.replace("__PAYLOAD__", payload).replace("__TOTAL__", "{:,}".format(len(rows)))
    check_escapes(page)
    pathlib.Path(args.out).write_text(page, encoding="utf-8")

    v = collections.Counter(
        ("both" if r["g"] and r["a"] else "garble" if r["g"] else "anchor" if r["a"] else "clean")
        for r in rows)
    print("cases on page : %d" % len(rows))
    print("verdicts      : %s" % dict(v.most_common()))
    print("clean          : %d (%.1f%%)" % (v["clean"], 100 * v["clean"] / len(rows)))
    print("page size     : %.1f MB" % (len(page.encode("utf-8")) / 1024 / 1024))
    print("wrote         : %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
