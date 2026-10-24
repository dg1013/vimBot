#!/usr/bin/env python3
"""Build the review page for the 2026-09-06 audit of the quota-fill batch.

The audit re-checked the day's work as data rather than as claims: the
allocator's own leakage keys over the split files, the case-silence
classifier over the new rows, the pasteable rendering the way DECISIONS.md
measured it, tier against the taxonomy, and the two published pages against
their sources. Eight findings came out. Each is a card on this page with the
reason it needs attention, the rows behind it shown as vim sees them (input,
gold, and where it matters the rival reading or the pasted result), the
recommended fix, and a verdict control per item and per row.

Every proposed edit on the page was executed against the sandbox by this
builder before it was shown, so an approved edit is a verified one; the
export is a change-set `apply_curation.py` runs directly (`edits`,
`deletes`) plus the decisions that need a different tool (`items`, `tiers`,
`groups`). Verdicts live in the browser's localStorage per page; export the
JSON (JavaScript Object Notation) and hand it over.

Usage:
    python scripts/build_audit_review.py --out page.html
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
import executor                                  # noqa: E402
from paste import pasteable                      # noqa: E402
from build_splits import norm_command            # noqa: E402
from generate_data import norm_buffer            # noqa: E402
from page_guard import check_page                # noqa: E402

AUDIT = "2026-09-06"
FIELDS = ("id", "category", "tier", "tier_provenance", "mode", "instruction",
          "gold_command", "buffer_before", "buffer_after", "cursor", "authored_on")

# ---------------------------------------------------------------- proposals
# Verified by this builder against the sandbox before they reach the page.

LEAK_PAIRS = (("hc-3472", "cand-cm-0107"), ("hc-3523", "cand-cm-0107"),
              ("hc-5319", "cand-cm-0107"), ("hc-5481", "cand-cm-0107"),
              ("hc-4989", "cand-cm-0091"))
LEAK_BUFFERS = {
    "hc-3472": "region,total\nnorth,120\nsouth,98\nwest,77\neast,143\n",
    "hc-3523": "# config\nname = demo\nold_key = 1\nold_key2 = 2\nold_key3 = 3\n",
    "hc-5319": "import os\nimport re\nimport sys\nimport json\n# generated, do not edit\n",
    "hc-5481": "name: report\nowner: data\nformat: csv\nnotes: nightly\nrefresh at 02:00\n",
    "hc-4989": "ssh_key\napi_key\ndb_key\n",
}
CAP_REWORDS = {
    "hc-4731": "Make the acronym under the cursor all caps.",
    "hc-4740": "Uppercase the whole variable name here.",
    "hc-4749": "Put the environment name in all capitals.",
    "hc-4757": "Set the method name word in full caps.",
    "hc-4763": "Make the type name all caps.",
    "hc-4769": "Uppercase every letter of the second word on this line.",
    "hc-4776": "Write the format name in capitals throughout.",
}
PASTE_RETARGETS = {"hc-3884": ">2jjj<<", "hc-3584": "f\\x"}
CELL_IMPACT = [("join", "T2", 96, 100), ("operator-motion", "T2", 169, 170),
               ("text-objects", "T2", 117, 160), ("text-objects", "T3", 92, 100),
               ("case-ops", "T2", 91, 110), ("indent", "T2", 109, 110)]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# What became of each finding, stamped on the page so it stays a record and
# not a standing to-do list. Decided by the owner 2026-09-06 (DECISIONS.md).
APPLIED = {
    1: ("done", "Closed three ways. The two test rows moved to train, so each "
                "shared template sits inside one split (the replacement "
                "buffers below were NOT used). Then the guards were keyed to "
                "the allocator's normalizers, apply_curation gained the §7.3 "
                "check it never had, and on 2026-09-07 every buffer was "
                "canonicalized — one trailing newline, \"\" when empty — so "
                "byte equality and semantic equality now coincide and the "
                "class of defect is gone rather than guarded against."),
    2: ("done", "Six reworded in the owner's own words; hc-4731 kept as it "
                "stands. The cards show the wording now in the corpus."),
    3: ("done", "Both dropped rather than retargeted. That emptied indent T3 "
                "and search-edit T1, so two rows were authored to refill them, "
                "each verified to give the same buffer typed and pasted. "
                "Pasteability is now 3,362 of 3,369."),
    4: ("done", "All 26 verified and kept. Val now carries no unreviewed row."),
    5: ("done", "The stale sentence is computed from the data now, and both "
                "pages were republished to their own URLs."),
    6: ("done", "Capped at three uses. 146 rows rewritten: sort-uniq got a "
                "sentence naming what it orders; increment and repeat-count "
                "got a domain buffer as well, because their filler buffers "
                "(n = 4, stuvwxyz-tail18) gave a new sentence nothing to say. "
                "Distinct instructions in the batch: 2,187 to 2,333."),
    7: ("kept", "Owner: tier is not critical, it is there so the batch spans "
                "difficulties. The taxonomy's situational examples govern, "
                "nothing was re-tiered, and 51/51 stands."),
    8: ("done", "Counts refreshed, and apply_curation.py now prints CHECK THE "
                "TIER when an edit changes the command under an authored "
                "tier. Two tests pin it."),
}


def run(buf, cmd, mode, cur):
    return executor.run_vim(buf, cmd, mode=mode, cursor=tuple(cur or (1, 1)))


def run_pasted(buf, keys, cur):
    """What the rendered line does. The scope screen refuses :exe by design;
    it is bypassed in this process only, the way DECISIONS.md measured it."""
    line = pasteable("keys", keys)
    keep = executor.screen_dangerous
    executor.screen_dangerous = lambda *a, **k: None
    try:
        res = executor.run_vim(buf, line.lstrip(":"), mode="ex", cursor=tuple(cur or (1, 1)))
    finally:
        executor.screen_dangerous = keep
    return line, res.get("buffer_after")


def gather():
    corpus = load(REPO / "data" / "combined_cases.jsonl")
    by_id = {r["id"]: r for r in corpus}
    split_of = {}
    for s in ("train", "val", "test"):
        for r in load(REPO / "data" / "splits" / f"{s}.jsonl"):
            split_of[r["id"]] = s
    today = [r for r in corpus if r.get("authored_on") == AUDIT]
    have = {norm_buffer(r["buffer_before"]) for r in corpus}

    dropped = {}
    dpath = REPO / "data" / "dropped_cases.jsonl"
    if dpath.exists():
        for r in load(dpath):
            dropped[r["id"]] = r

    def row(i):
        r = by_id.get(i) or dropped[i]
        d = {k: r.get(k) for k in FIELDS}
        d["split"] = split_of.get(i) or (
            "dropped %s" % r.get("dropped_on") if i in dropped else None)
        d["dropped"] = i in dropped and i not in by_id
        return d

    leak = []
    for t, x in LEAK_PAIRS:
        tr = row(t)
        buf = LEAK_BUFFERS[t]
        res = run(buf, tr["gold_command"], tr["mode"], tr["cursor"])
        assert res.get("was_valid") and norm_buffer(buf) not in have, t
        tr["proposed_buffer"], tr["proposed_after"] = buf, res["buffer_after"]
        tr["superseded"] = True     # the owner consolidated the split instead
        leak.append({"train": tr, "test": row(x)})

    cap = []
    for i, reword in CAP_REWORDS.items():
        r = row(i)
        r["alt_after"] = run(r["buffer_before"], "gUl", "keys", r["cursor"]).get("buffer_after")
        r["proposed"] = reword
        cap.append(r)

    paste = []
    for i, proposed in PASTE_RETARGETS.items():
        r = row(i)
        r["rendered"], r["pasted_after"] = run_pasted(r["buffer_before"], r["gold_command"], r["cursor"])
        prop = run(r["buffer_before"], proposed, "keys", r["cursor"])
        r["proposed"], r["proposed_after"] = proposed, prop.get("buffer_after")
        r["proposed_rendered"], r["proposed_pasted_after"] = run_pasted(r["buffer_before"], proposed, r["cursor"])
        paste.append(r)

    val = [row(r["id"]) for r in today if split_of.get(r["id"]) == "val"]

    byins = collections.defaultdict(list)
    for r in today:
        byins[r["instruction"]].append(r)
    repeat = []
    for ins, rs in sorted(byins.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(rs) > 1:
            repeat.append({
                "instruction": ins, "n": len(rs),
                "category": sorted({r["category"] for r in rs}),
                "commands": sorted({r["gold_command"] for r in rs}),
                "rows": [{"id": r["id"], "buffer_before": r["buffer_before"],
                          "gold_command": r["gold_command"], "cursor": r["cursor"],
                          "split": split_of.get(r["id"])} for r in rs]})

    grp = collections.defaultdict(list)
    for r in today:
        grp[(r["category"], r["mode"], norm_command(r["gold_command"], r["mode"]))].append(r)
    tiers = []
    for k, v in grp.items():
        c = collections.Counter(x["tier"] for x in v)
        if len(c) > 1:
            modal = c.most_common(1)[0][0]
            for x in v:
                if x["tier"] != modal:
                    d = row(x["id"])
                    d["modal"], d["spread"] = modal, dict(sorted(c.items()))
                    tiers.append(d)
    tiers.sort(key=lambda d: (d["category"], d["gold_command"], d["id"]))

    import yaml
    tax = yaml.safe_load((REPO / "data" / "taxonomy.yaml").read_text(encoding="utf-8"))
    cats = tax["categories"]
    return {
        "audit": AUDIT,
        "stats": {"corpus": len(corpus), "today": len(today),
                  "keys_rows": sum(1 for r in corpus if r["mode"] == "keys"),
                  "val": len(val), "repeat_rows": sum(g["n"] for g in repeat),
                  "repeat_groups": len(repeat), "tiers": len(tiers)},
        "applied": {str(k): {"state": v[0], "text": v[1]} for k, v in APPLIED.items()},
        "leak": leak, "capitalize": cap, "paste": paste, "val": val,
        "repeat": repeat, "tiers": tiers,
        "tier_defs": tax["tiers"],
        "tier_examples": {c: cats[c].get("tier_examples") for c in
                          ("text-objects", "case-ops", "operator-motion", "search-edit", "join")},
        "cell_impact": CELL_IMPACT,
    }


# ------------------------------------------------------------------- page

HEAD = r"""<title>Quota Fill Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>
:root{
  --ground:#F2F4F6; --surface:#FBFCFD; --sunk:#E6EAEE; --raise:#DFE5EA;
  --ink:#161B22; --ink2:#39424D; --muted:#66717E; --faint:#98A2AE;
  --line:#D3DAE1; --line2:#E3E8ED;
  --accent:#0F5F66; --accent-soft:#DCEDEE; --accent-ink:#0B4A50;
  --ok:#006B41; --warn:#AD721B; --bad:#9F2234;
  --ok-soft:#DCEFE5; --warn-soft:#F6E9D3; --bad-soft:#F5DEE1;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","DejaVu Sans Mono",Menlo,monospace;
  --body:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --disp:"Newsreader","Iowan Old Style",Georgia,serif;
  color-scheme: light;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
    --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
    --line:#2B333C; --line2:#222A32; --accent:#6FC3C8; --accent-soft:#123134; --accent-ink:#9BDDE0;
    --ok:#24A96F; --warn:#C48800; --bad:#DF5F72;
    --ok-soft:#12332A; --warn-soft:#332A10; --bad-soft:#3A1A20;
    color-scheme: dark;
  }
}
:root[data-theme="dark"]{
  --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
  --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
  --line:#2B333C; --line2:#222A32; --accent:#6FC3C8; --accent-soft:#123134; --accent-ink:#9BDDE0;
  --ok:#24A96F; --warn:#C48800; --bad:#DF5F72;
  --ok-soft:#12332A; --warn-soft:#332A10; --bad-soft:#3A1A20;
  color-scheme: dark;
}
*{ box-sizing:border-box; }
body{ margin:0; background:var(--ground); color:var(--ink); font-family:var(--body); font-size:14px; line-height:1.5; }
a{ color:var(--accent-ink); }
h1,h2,h3{ font-family:var(--disp); font-weight:600; margin:0; letter-spacing:-.012em; text-wrap:balance; }
h1{ font-size:34px; line-height:1.1; }
h2{ font-size:22px; line-height:1.2; }
h3{ font-size:16px; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.wrap{ max-width:1180px; margin:0 auto; padding:26px 22px 120px; }
header{ display:grid; grid-template-columns: minmax(0,1fr) auto; gap:18px 32px; align-items:end; padding-bottom:18px; border-bottom:1px solid var(--line); }
header p.lede{ max-width:68ch; margin:10px 0 0; color:var(--ink2); }
.band{ display:flex; gap:22px; flex-wrap:wrap; }
.band .v{ font-family:var(--mono); font-size:22px; font-weight:700; display:block; font-variant-numeric:tabular-nums; }
.band .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.07em; text-transform:uppercase; color:var(--muted); margin-top:3px; display:block; }
nav.toc{ position:sticky; top:0; z-index:5; background:var(--ground); border-bottom:1px solid var(--line); padding:8px 0; margin:0 0 6px; display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; }
nav.toc a{ text-decoration:none; font-family:var(--mono); font-size:11px; color:var(--ink2); padding:4px 8px; border:1px solid var(--line); border-radius:4px; background:var(--surface); display:inline-flex; gap:7px; align-items:center; }
nav.toc a:hover{ border-color:var(--accent); }
nav.toc .prog{ margin-left:auto; font-family:var(--mono); font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums; }
.dot{ width:8px; height:8px; border-radius:50%; display:inline-block; }
.dot.bad{ background:var(--bad); } .dot.warn{ background:var(--warn); } .dot.ok{ background:var(--ok); } .dot.info{ background:var(--accent); }
.sev{ font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; font-weight:700; padding:2px 7px; border-radius:3px; display:inline-flex; gap:6px; align-items:center; }
.sev.bad{ color:var(--bad); background:var(--bad-soft); } .sev.warn{ color:var(--warn); background:var(--warn-soft); } .sev.info{ color:var(--accent-ink); background:var(--accent-soft); }
.sev.done{ color:var(--ok); background:var(--ok-soft); }
.sev.kept{ color:var(--muted); background:var(--raise); }
.outcome{ margin:12px 0 0 58px; padding:10px 14px; border-left:3px solid var(--ok);
  background:var(--surface); border-radius:0 6px 6px 0; max-width:74ch; color:var(--ink2); }
.outcome.kept{ border-left-color:var(--muted); }
.outcome b{ color:var(--ink); }
.banner{ margin:16px 0 0; padding:12px 16px; border:1px solid var(--ok);
  border-radius:6px; background:var(--ok-soft); color:var(--ink); max-width:80ch; }
.card.superseded{ opacity:.62; }
section.item{ margin-top:34px; padding-top:22px; border-top:1px solid var(--line); scroll-margin-top:52px; }
section.item .head{ display:grid; grid-template-columns: 44px minmax(0,1fr); gap:14px; align-items:start; }
section.item .num{ font-family:var(--disp); font-size:30px; color:var(--faint); line-height:1; padding-top:2px; font-variant-numeric:tabular-nums; }
.meta{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:8px; }
.count{ font-family:var(--mono); font-size:11px; color:var(--muted); }
.why{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:14px 26px; margin:14px 0 0 58px; }
.why > div{ max-width:62ch; }
.why p{ margin:4px 0 0; color:var(--ink2); }
.why code, .mono{ font-family:var(--mono); font-size:12.5px; }
code{ font-family:var(--mono); font-size:12.5px; background:var(--sunk); padding:0 4px; border-radius:3px; }
.decide{ margin:16px 0 0 58px; display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; padding:12px 14px; background:var(--surface); border:1px solid var(--line); border-radius:6px; }
.decide .q{ font-weight:600; flex:1 1 320px; }
.seg{ display:inline-flex; border:1px solid var(--line); border-radius:5px; overflow:hidden; }
.seg button{ font:600 11.5px var(--mono); padding:6px 10px; border:0; background:var(--surface); color:var(--ink2); cursor:pointer; border-right:1px solid var(--line); }
.seg button:last-child{ border-right:0; }
.seg button:hover{ background:var(--raise); }
.seg button.on{ background:var(--accent); color:#fff; }
.seg button.on.v-drop, .seg button.on.v-reject{ background:var(--bad); }
.seg button.on.v-keep, .seg button.on.v-defer{ background:var(--muted); }
.seg button:focus-visible, textarea:focus-visible, input:focus-visible{ outline:2px solid var(--accent); outline-offset:1px; }
input.note{ font:12.5px var(--body); padding:6px 8px; border:1px solid var(--line); border-radius:4px; background:var(--surface); color:var(--ink); min-width:220px; flex:1 1 220px; }
.cards{ margin:14px 0 0 58px; display:grid; gap:12px; }
.card{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:12px 14px; }
.card.done{ border-color:var(--ok); }
.card .top{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:baseline; }
.card .id{ font-family:var(--mono); font-weight:700; font-size:12.5px; }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:4px; background:var(--raise); color:var(--ink2); }
.chip.split-test{ color:var(--bad); border:1px solid var(--bad); background:transparent; font-weight:700; }
.chip.split-val{ color:var(--warn); border:1px solid var(--warn); background:transparent; font-weight:700; }
.ins{ margin:8px 0 0; font-size:14.5px; }
.cmd{ font-family:var(--mono); font-size:12.5px; color:var(--accent-ink); }
.bufs{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:10px; margin-top:10px; }
.buf .lab{ margin-bottom:4px; display:block; }
.buf.rival .lab{ color:var(--bad); }
pre{ margin:0; font-family:var(--mono); font-size:12px; line-height:1.45; background:var(--sunk); border:1px solid var(--line2); border-radius:4px; padding:8px 10px; overflow-x:auto; white-space:pre; }
pre .tab{ color:var(--warn); font-weight:700; }
pre .eol{ color:var(--faint); }
pre .cur{ background:var(--accent-soft); outline:1px solid var(--accent); }
.fix{ margin-top:10px; display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:10px; }
.fix textarea, .fix input{ width:100%; font-family:var(--mono); font-size:12px; padding:7px 9px; border:1px solid var(--line); border-radius:4px; background:var(--surface); color:var(--ink); resize:vertical; }
.fix textarea.dirty, .fix input.dirty{ border-color:var(--warn); }
.fix .lab{ display:block; margin-bottom:4px; }
.vrow{ display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; margin-top:10px; }
.vrow .note{ min-width:180px; }
table{ border-collapse:collapse; font-family:var(--mono); font-size:12px; margin-top:8px; }
th,td{ text-align:left; padding:4px 10px 4px 0; border-bottom:1px solid var(--line2); font-variant-numeric:tabular-nums; }
th{ font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
td.short{ color:var(--warn); font-weight:700; }
details.rows{ margin-top:8px; }
details.rows summary{ cursor:pointer; font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.rowlist{ display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap:8px; margin-top:8px; }
.rowlist .mini{ font-family:var(--mono); font-size:11px; }
.rowlist .mini pre{ font-size:11px; max-height:110px; overflow:auto; }
.good{ margin:36px 0 0; padding-top:18px; border-top:1px solid var(--line); }
.good ul{ columns:2; column-gap:30px; margin:8px 0 0; padding-left:18px; color:var(--ink2); }
.good li{ break-inside:avoid; margin-bottom:4px; }
.export{ position:fixed; left:0; right:0; bottom:0; background:var(--surface); border-top:1px solid var(--line); padding:10px 22px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:6; }
.export button{ font:600 12px var(--mono); padding:8px 12px; border:1px solid var(--accent); border-radius:5px; background:var(--accent); color:#fff; cursor:pointer; }
.export button.quiet{ background:transparent; color:var(--accent-ink); }
.export .sum{ font-family:var(--mono); font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
.export textarea{ width:100%; height:160px; font-family:var(--mono); font-size:11px; border:1px solid var(--line); border-radius:4px; background:var(--sunk); color:var(--ink); }
.export textarea[hidden]{ display:none; }
#boot{ font-family:var(--mono); font-size:12px; color:var(--bad); }
@media (max-width: 700px){ .why, .decide, .cards{ margin-left:0; } .good ul{ columns:1; } .fix{ grid-template-columns:1fr; } }
@media (prefers-reduced-motion: no-preference){ .card{ transition: border-color .15s; } }
</style>
"""

BODY = r"""
<div class="wrap">
<header>
  <div>
    <span class="lab">vimBot &middot; audit of the 2026-09-06 quota fill</span>
    <h1>Eight findings, each with its rows</h1>
    <p class="lede">The day's work re-checked as data, not as claims: the allocator's own leakage keys over the
    split files, the case-silence classifier over the new rows, the pasteable rendering measured the way
    DECISIONS.md measured it, tier against the taxonomy, and the two published pages against their sources.
    Every proposed edit below was run through the sandbox before it was shown, so approving it approves a
    verified change. Nothing has been changed yet.</p>
  </div>
  <div class="band" id="band"></div>
  <p class="banner" style="grid-column:1/-1"><b>All eight were decided and applied on
  2026-09-06.</b> This page is now the record of what was done, not a list of open
  work. Each finding carries its outcome below its title, and the cards show the
  rows as they stand after the change. Verdict controls still work and still
  export, for anything you want to revisit.</p>
</header>
<nav class="toc" id="toc"></nav>
<div id="boot">Loading the findings&hellip;</div>
<div id="items"></div>
<div class="good" id="good"></div>
</div>
<div class="export">
  <button id="exportBtn">Export decisions</button>
  <button id="copyBtn" class="quiet">Copy JSON</button>
  <button id="resetBtn" class="quiet">Clear all verdicts</button>
  <span class="sum" id="sum"></span>
  <textarea id="out" hidden aria-label="exported decisions"></textarea>
</div>
<script>
window.onerror = function (m) { var b = document.getElementById('boot'); if (b) b.textContent = 'This page hit a script error: ' + m; };
const DATA = @@DATA@@;
const KEY = 'vimbot-audit-0906', EKEY = KEY + '-edits';
let verdicts = {}, edits = {};
try { verdicts = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { verdicts = {}; }
try { edits = JSON.parse(localStorage.getItem(EKEY) || '{}'); } catch (e) { edits = {}; }
function save() {
  try { localStorage.setItem(KEY, JSON.stringify(verdicts)); localStorage.setItem(EKEY, JSON.stringify(edits)); } catch (e) {}
}
function esc(s) { return String(s === null || s === undefined ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function bufHtml(text, opts) {
  opts = opts || {};
  if (text === null || text === undefined) return '<pre><span class="eol">(no result)</span></pre>';
  const parts = String(text).split('\n');
  const endsNl = parts.length > 1 && parts[parts.length - 1] === '';
  if (endsNl) parts.pop();
  const cur = opts.cursor || null;
  const lines = parts.map(function (line, i) {
    let body = esc(line).replace(/\t/g, '<span class="tab">&#x21E5;   </span>');
    if (cur && cur[0] === i + 1) body = '<span class="cur">' + body + '</span>';
    const mark = opts.eol ? ((i < parts.length - 1 || endsNl) ? '<span class="eol">&para;</span>' : '<span class="eol"> &#x2205; no final newline</span>') : '';
    return body + mark;
  });
  if (!lines.length) lines.push('<span class="eol">(empty)</span>');
  return '<pre>' + lines.join('\n') + '</pre>';
}
function buf(label, text, opts, cls) {
  return '<div class="buf ' + (cls || '') + '"><span class="lab">' + label + '</span>' + bufHtml(text, opts) + '</div>';
}
function chip(t, cls) { return '<span class="chip ' + (cls || '') + '">' + esc(t) + '</span>'; }
function splitChip(s) { return chip(s || 'unplaced', s === 'test' ? 'split-test' : (s === 'val' ? 'split-val' : '')); }
function seg(key, options, current) {
  return '<span class="seg" data-key="' + esc(key) + '">' + options.map(function (o) {
    return '<button type="button" class="' + (current === o ? 'on ' : '') + 'v-' + esc(o.replace(/[^a-z0-9]+/gi, '-')) + '" data-v="' + esc(o) + '">' + esc(o) + '</button>';
  }).join('') + '</span>';
}
function noteBox(key, ph) {
  const v = verdicts[key] || {};
  return '<input class="note" data-note="' + esc(key) + '" placeholder="' + esc(ph || 'note') + '" value="' + esc(v.note || '') + '">';
}
function vget(key) { return (verdicts[key] || {}).verdict; }
function field(id, key, orig, label, proposal, multiline) {
  const cur = (edits[id] && edits[id][key] !== undefined) ? edits[id][key] : (proposal !== undefined ? proposal : orig);
  const dirty = cur !== orig;
  const attrs = ' data-id="' + esc(id) + '" data-field="' + esc(key) + '" data-orig="' + esc(orig) + '" class="' + (dirty ? 'dirty' : '') + '"';
  const ctl = multiline ? '<textarea rows="' + (Math.max(3, String(cur).split('\n').length + 1)) + '"' + attrs + '>' + esc(cur) + '</textarea>'
                        : '<input' + attrs + ' value="' + esc(cur) + '">';
  return '<div><span class="lab">' + label + '</span>' + ctl + '</div>';
}
// -------------------------------------------------------------- items
function rowTop(r, extra) {
  return '<div class="top"><span class="id">' + esc(r.id) + '</span>' + chip(r.category) + chip(r.tier || 'no tier') + chip(r.mode) +
    (r.dropped ? chip('dropped from the corpus', 'split-test') : splitChip(r.split)) + (extra || '') + '</div>';
}
function leakCard(e) {
  const t = e.train, x = e.test, k = t.id;
  const v = vget(k);
  return '<div class="card superseded' + (v ? ' done' : '') + '" id="row-' + esc(k) + '">' + rowTop(t, chip('buffer kept; the test row moved to train')) +
    '<p class="ins">' + esc(t.instruction) + ' &nbsp;<span class="cmd">' + esc(t.mode) + ': ' + esc(t.gold_command) + '</span></p>' +
    '<div class="bufs">' + buf('train row, buffer as stored', t.buffer_before, { eol: true, cursor: t.cursor }) +
    buf('test row ' + esc(x.id) + ' (' + esc(x.category) + ', ' + esc(x.mode) + ': ' + esc(x.gold_command) + ')', x.buffer_before, { eol: true, cursor: x.cursor }, 'rival') +
    '</div><div class="fix">' + field(k, 'buffer_before', t.buffer_before, 'replacement buffer (verified: valid, no collision)', t.proposed_buffer, true) +
    buf('what the gold command does to it', t.proposed_after, { cursor: null }) + '</div>' +
    '<div class="vrow">' + seg(k, ['new buffer', 'keep as is', 'drop'], v) + noteBox(k) + '</div></div>';
}
function capCard(r) {
  const k = r.id, v = vget(k);
  return '<div class="card' + (v ? ' done' : '') + '" id="row-' + esc(k) + '">' + rowTop(r) +
    '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">keys: ' + esc(r.gold_command) + '</span></p>' +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('gold: gUiw, the whole word', r.buffer_after) +
    buf('the other reading: gUl, an initial capital', r.alt_after, {}, 'rival') + '</div>' +
    '<div class="fix">' + field(k, 'instruction', r.instruction, 'reworded instruction', r.proposed, false) + '<div></div></div>' +
    '<div class="vrow">' + seg(k, ['reword', 'keep', 'drop'], v) + noteBox(k) + '</div></div>';
}
function pasteCard(r) {
  const k = r.id, v = vget(k);
  return '<div class="card' + (v ? ' done' : '') + '" id="row-' + esc(k) + '">' + rowTop(r) +
    '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">keys: ' + esc(r.gold_command) + '</span></p>' +
    '<p class="mono" style="margin:6px 0 0;color:var(--muted)">rendered as ' + esc(r.rendered) + '</p>' +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('gold (typed keys)', r.buffer_after) +
    buf('what the pasted line produces', r.pasted_after, {}, 'rival') + '</div>' +
    '<div class="fix">' + field(k, 'command', r.gold_command, 'replacement command (verified: same gold, typed and pasted)', r.proposed, false) +
    buf('the replacement, typed and pasted alike', r.proposed_after) + '</div>' +
    '<div class="vrow">' + seg(k, ['retarget', 'keep', 'drop'], v) + noteBox(k) + '</div></div>';
}
function valCard(r) {
  const k = r.id, v = vget(k);
  return '<div class="card' + (v ? ' done' : '') + '" id="row-' + esc(k) + '">' + rowTop(r) +
    '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></p>' +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('after', r.buffer_after) + '</div>' +
    '<div class="fix">' + field(k, 'instruction', r.instruction, 'instruction (edit if rewording)', undefined, false) + '<div></div></div>' +
    '<div class="vrow">' + seg(k, ['ok', 'reword', 'drop'], v) + noteBox(k) + '</div></div>';
}
function repeatCard(g, i) {
  const k = 'group:' + i, v = vget(k);
  return '<div class="card' + (v ? ' done' : '') + '" id="row-' + esc(k) + '"><div class="top"><span class="id">&times;' + g.n + '</span>' +
    g.category.map(function (c) { return chip(c); }).join('') + '<span class="cmd">' + g.commands.map(esc).join(' &middot; ') + '</span></div>' +
    '<p class="ins">&ldquo;' + esc(g.instruction) + '&rdquo;</p>' +
    '<details class="rows"><summary>the ' + g.n + ' rows carrying it</summary><div class="rowlist">' +
    g.rows.map(function (r) { return '<div class="mini">' + esc(r.id) + ' ' + splitChip(r.split) + ' <span class="cmd">' + esc(r.gold_command) + '</span>' + bufHtml(r.buffer_before, { cursor: r.cursor }) + '</div>'; }).join('') +
    '</div></details>' +
    '<div class="vrow">' + seg(k, ['keep', 'rephrase'], v) + noteBox(k, 'how many to rephrase, or a phrasing to use') + '</div></div>';
}
function tierCard(r) {
  const k = r.id, v = vget(k);
  const spread = Object.keys(r.spread).map(function (t) { return t + ' ' + r.spread[t]; }).join(' / ');
  return '<div class="card' + (v ? ' done' : '') + '" id="row-' + esc(k) + '">' + rowTop(r, chip('authored as ' + r.tier) + chip('others: ' + spread)) +
    '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></p>' +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('after', r.buffer_after) + '</div>' +
    '<div class="vrow"><span class="lab">tier</span>' + seg(k, ['keep ' + r.tier, 'set ' + r.modal, 'T1', 'T2', 'T3'], v) + noteBox(k) + '</div></div>';
}
function decide(key, question, options, ph) {
  return '<div class="decide"><span class="q">' + question + '</span>' + seg(key, options, vget(key)) + noteBox(key, ph) + '</div>';
}
function why(blocks) {
  return '<div class="why">' + blocks.map(function (b) { return '<div><span class="lab">' + b[0] + '</span><p>' + b[1] + '</p></div>'; }).join('') + '</div>';
}
function itemHtml(it) {
  const ap = DATA.applied[String(it.n)] || null;
  const stamp = ap ? '<span class="sev ' + ap.state + '">' + (ap.state === 'done' ? '&#x2713; applied' : '&#x25CB; decided, no change') + '</span>' : '';
  const outcome = ap ? '<div class="outcome' + (ap.state === 'kept' ? ' kept' : '') + '"><span class="lab">what was done</span>' + esc(ap.text) + '</div>' : '';
  return '<section class="item" id="item-' + it.n + '"><div class="head"><div class="num">' + it.n + '</div><div>' +
    '<h2>' + it.title + '</h2><div class="meta">' + stamp + '<span class="sev ' + it.sev + '">' + it.sevWord + '</span><span class="count">' + it.count + '</span></div></div></div>' +
    outcome + why(it.why) + (it.decisions || []).map(function (d) { return decide(d[0], d[1], d[2], d[3]); }).join('') +
    (it.extra || '') + (it.cards ? '<div class="cards">' + it.cards + '</div>' : '') + '</section>';
}
const S = DATA.stats;
S.leak_rows = DATA.leak.length;
const ITEMS = [
  { n: 1, sev: 'bad', sevWord: '&#x25CF; leak', title: 'Two buffer templates still span train and test',
    count: S.leak_rows + ' train rows authored today share a template with 2 test rows',
    why: [
      ['why it needs fixing', 'CLAUDE.md &sect;7.3 keeps buffer templates disjoint across splits, and the allocator&rsquo;s own key (<code>norm_buffer</code>) ignores one trailing newline. The test rows store <code>1\\n2\\n3\\n4\\n5</code> and <code>aa\\nbb\\ncc</code> without a final newline; today&rsquo;s train rows store the same lines with one. The <code>--add-new</code> guard, the new <code>--move</code> tool and <code>tests/test_split_files.py</code> all compare raw strings, so every one of them passes while the allocator&rsquo;s key fails. RESULTS.md line 88 states that no buffer template spans two splits. Test is the measured slice, so this is the one finding that touches a reported number.'],
      ['recommended', 'Give the five train rows fresh buffers from the domain through <code>apply_curation.py</code>, keeping the test rows untouched (the prefilled replacements below are already verified: valid, they change the buffer, and no normalized collision anywhere in the corpus). Separately, make the guard and the test compare on <code>norm_buffer</code> and <code>norm_command</code>, so the check is the allocator&rsquo;s check. The <code>&para;</code> marks show where each buffer&rsquo;s final newline is.']
    ],
    decisions: [['item:1-code', 'Also change the append guard, the move tool and the split test to the normalized keys?', ['approve', 'reject', 'defer'], 'note']],
    cards: DATA.leak.map(leakCard).join('') },
  { n: 2, sev: 'warn', sevWord: '&#x25B2; ambiguous', title: '&ldquo;Capitalize&rdquo; means two different edits',
    count: DATA.capitalize.length + ' case-ops rows authored today',
    why: [
      ['why it needs review', 'Each of these says &ldquo;Capitalize the &hellip;&rdquo; and its gold is <code>gUiw</code>, which uppercases the whole word. In ordinary English &ldquo;capitalize&rdquo; means an initial capital, and that reading (<code>gUl</code>) produces a different buffer, shown beside the gold. The case-atlas classifier flags exactly this as <em>ambiguous</em>; the corpus already held 19 such rows, which you reviewed on 2026-09-04 and kept where the context made the reading clear. These seven are new and were not part of that review. An ambiguous instruction punishes a model that answers correctly.'],
      ['recommended', 'Reword each so the whole-word reading is the only one. The prefilled sentences vary the phrasing (all caps, uppercase the whole, every letter, in capitals throughout) so the seven do not become one template; edit freely. None contains the command, so the leak guard is satisfied.']
    ],
    cards: DATA.capitalize.map(capCard).join('') },
  { n: 3, sev: 'warn', sevWord: '&#x25B2; stale figure', title: 'Two new keys rows fail the pasteable rendering, and the documented figure is stale',
    count: '3,362 of ' + S.keys_rows.toLocaleString() + ' keys rows now reproduce identically; the docs said 1,812 of 1,819',
    why: [
      ['why it needs fixing', 'DECISIONS.md 2026-09-04 measured every keys row through <code>:exe "normal! &hellip;"</code> and found 7 that diverge. Re-measured the same way over today&rsquo;s corpus: the same 7 plus these two. <code>hc-3884</code> only works when typed, because the space in <code>&gt;2jj j&lt;&lt;</code> beeps at the end of a line and typed keys continue while <code>:normal!</code> aborts; DECISIONS.md already calls that class &ldquo;a questionable gold regardless of rendering&rdquo;. In <code>hc-3584</code> a <code>/</code> search run through <code>:normal!</code> never lands on the backslash while <code>:call search()</code> does; the escaping is not the cause. RESULTS.md, HANDOFF.md and DECISIONS.md all still carry 1,812 / 1,819.'],
      ['what was done instead', 'Both rows were dropped rather than retargeted. The replacement command shown on each card was verified both ways but not used. Dropping them emptied indent T3 and search-edit T1, so two rows were authored to refill those cells, each checked to behave identically typed and pasted. The three documents now record 3,362 of 3,369, the seven remaining being the ones already listed on 2026-09-04.']
    ],
    decisions: [['item:3-docs', 'Update the pasteability figure in RESULTS.md, HANDOFF.md and DECISIONS.md to 3,360 / 3,369 with the nine ids?', ['approve', 'reject', 'defer'], 'note']],
    cards: DATA.paste.map(pasteCard).join('') },
  { n: 4, sev: 'warn', sevWord: '&#x25B2; unreviewed', title: 'Twenty-six unreviewed rows sit in val',
    count: S.val + ' rows authored today, placed in val by the command rule',
    why: [
      ['why it needs review', 'Val selects checkpoints. These rows were authored today to fill quota cells and reached val only because &sect;7.3 keeps a gold command inside one split and val already held their commands. They carry no verdict, no document mentions them, and the 12:32 DECISIONS entry says the six-row move &ldquo;keeps unreviewed rows out of val&rdquo;, which was already untrue when it was written. The question for each is the one no gate answers: does the sentence describe the edit, completely and without a second faithful reading?'],
      ['what was done', 'All 26 were read and kept as they stand. Val now carries no unreviewed row, and DECISIONS.md, HANDOFF.md and RESULTS.md say so. Test still holds 39, which is the remaining review.']
    ],
    decisions: [['item:4-docs', 'Record the 26 val rows in the documents and add them to the New Cases priority filter?', ['approve', 'reject', 'defer'], 'note']],
    cards: DATA.val.map(valCard).join('') },
  { n: 5, sev: 'warn', sevWord: '&#x25B2; stale page', title: 'Two published pages are wrong or stale',
    count: 'New Cases 2026-09-06 and Test Slice Verification',
    why: [
      ['why it needs fixing', 'The New Cases page was re-scoped from the 195-row visual batch to all 2,374 rows, but one sentence from the old lede survived: &ldquo;Vim corrected six of the author&rsquo;s own expectations along the way.&rdquo; That was true of the visual batch; across the three batches vim corrected far more (counted yank paste position, <code>n</code> wrap, unescaped <code>/</code>, <code>:g</code> swallowing <code>|</code>, a bare <code>&amp;</code> in a replacement, trailing whitespace under <code>a"</code> and <code>aw</code>, <code>&gt;}</code> including the cursor line, empty lines never indented, <code>J</code>&rsquo;s double space, <code>&lt;BS&gt;</code> inert under <code>-u NONE</code>). The Test Slice Verification page was published when test held 422 rows; rebuilt today it reads &ldquo;409 of these 448 rows carry a human verdict&rdquo;, and HANDOFF still describes it as &ldquo;all 409 rows&rdquo;.'],
      ['recommended', 'Replace the sentence in <code>build_new_cases_review.py</code> with a count that is computed, republish that page to its URL, and republish the Test Slice page to its URL. Your verdicts on both pages live in that origin&rsquo;s localStorage and survive a republish.']
    ],
    decisions: [['item:5-newcases', 'Fix the sentence and republish New Cases 2026-09-06?', ['approve', 'reject', 'defer'], 'note'],
                ['item:5-testslice', 'Republish Test Slice Verification at 448 rows?', ['approve', 'reject', 'defer'], 'note']] },
  { n: 6, sev: 'info', sevWord: '&#x25C6; your call', title: 'Instruction diversity is thinner than the row count suggests',
    count: 'now ' + S.repeat_rows + ' of ' + S.today.toLocaleString() + ' rows share a sentence, none of them more than three times, in ' + S.repeat_groups + ' groups',
    why: [
      ['why it needed review', 'The brief for this batch was a good representation of the spoken and written English for each command. Most of it delivered that, but three categories leaned on repeated sentences across different buffers: increment said &ldquo;Bump the number on this line by one.&rdquo; 35 times and &ldquo;Knock one off this counter.&rdquo; 30 times, sort-uniq used seven phrasings over ten buffers each, and repeat-count said &ldquo;Delete N characters from the start of this token.&rdquo; for six counts. 212 rows in 25 groups. A model that sees one sentence many times learns the buffer&rsquo;s role well and the language&rsquo;s role less.'],
      ['what the cards show now', 'The 25 sentences as they stand after the cap, each on at most three rows. The 146 rows that carried a fourth-or-later use were rewritten: sort-uniq by sentence alone, increment and repeat-count by buffer and sentence together, since a filler buffer gives a new sentence nothing to name. Every rewrite was executed before it was applied.']
    ],
    decisions: [['item:6-scope', 'Overall: rephrase the repeated sentences?', ['all groups marked', 'none', 'defer'], 'note']],
    cards: DATA.repeat.map(repeatCard).join('') },
  { n: 7, sev: 'info', sevWord: '&#x25C6; definition', title: 'Tier is not a function of the command',
    count: S.tiers + ' rows authored today carry a tier different from other rows with the same category, mode and command',
    why: [
      ['why it needs review', 'The taxonomy defines T1 as &ldquo;' + esc(DATA.tier_defs.T1) + '&rdquo;, T2 as &ldquo;' + esc(DATA.tier_defs.T2) + '&rdquo; and T3 as &ldquo;' + esc(DATA.tier_defs.T3) + '&rdquo;, which reads as a property of the command. But its own examples make some categories situational: text-objects T2 includes &ldquo;ci( mid-line&rdquo; and &ldquo;dap on a chosen paragraph&rdquo;, case-ops T2 includes &ldquo;guiw on a target word&rdquo;. Today&rsquo;s batch followed the examples, so <code>di"</code> appears at T1, T2 and T3 depending on the buffer. Corpus-wide the same pattern covers 277 rows over 143 command keys, mostly older heuristic drift (<code>&lt;C-a&gt;</code> T1 50 / T2 40, <code>sort</code> T2 8, <code>dd</code> T2 5). The quota table depends on the answer: if bare primitives in these categories were T1, the cells below would fall short.'],
      ['recommended', 'Decide the definition first. Command-only is easier to audit and to fill honestly, and a re-tier pass would follow; situational matches the taxonomy&rsquo;s examples and leaves the 51/51 standing. Per row, keep or set the tier; &ldquo;set&rdquo; adopts the modal tier of the rows sharing that command.']
    ],
    decisions: [['item:7-def', 'What should a tier be a function of?', ['command only', 'situational, as the examples say', 'defer'], 'note']],
    extra: '<div class="cards"><div class="card"><span class="lab">cells that fall short if bare primitives were T1</span><table><tr><th>category</th><th>tier</th><th>would hold</th><th>quota</th></tr>' +
      DATA.cell_impact.map(function (c) { return '<tr><td>' + esc(c[0]) + '</td><td>' + esc(c[1]) + '</td><td class="short">' + c[2] + '</td><td>' + c[3] + '</td></tr>'; }).join('') + '</table>' +
      '<div style="margin-top:10px"><span class="lab">the taxonomy&rsquo;s own examples</span><table><tr><th>category</th><th>T1</th><th>T2</th><th>T3</th></tr>' +
      Object.keys(DATA.tier_examples).map(function (c) { const e = DATA.tier_examples[c] || {}; return '<tr><td>' + esc(c) + '</td><td>' + esc((e.T1 || []).join(' · ')) + '</td><td>' + esc((e.T2 || []).join(' · ')) + '</td><td>' + esc((e.T3 || []).join(' · ')) + '</td></tr>'; }).join('') + '</table></div></div></div>',
    cards: DATA.tiers.map(tierCard).join('') },
  { n: 8, sev: 'info', sevWord: '&#x25C6; minor', title: 'Two small things',
    count: 'a stale count and a tool that could warn',
    why: [
      ['why', 'HANDOFF.md line 40 says &ldquo;10 tracked files modified, 51 untracked entries&rdquo;; it is 11 and 77 today. And <code>apply_curation.py</code> now keeps an authored tier through any edit, including one that changes the command itself; a command edit can change the tier a row deserves, and the tool stays silent about it.'],
      ['recommended', 'Refresh the counts. Have the tool print a warning naming the row whenever an edit changes the gold command under an authored tier, so the re-tier is a visible decision rather than a silent omission.']
    ],
    decisions: [['item:8-handoff', 'Refresh the HANDOFF counts?', ['approve', 'reject', 'defer'], 'note'],
                ['item:8-warn', 'Add the warning to apply_curation.py?', ['approve', 'reject', 'defer'], 'note']] },
];
const GOOD = [
  'Corpus 7,512 = v3 2,011 + v4 5,501; every id unique.',
  'Every corpus row placed exactly once; split copies match the corpus apart from the split-only review flag.',
  'No duplicate (instruction, buffer) pair; the 197 shared (buffer, cursor, command) triples are the deliberate a/b paraphrase pairs.',
  'Leak guard over the whole corpus: only cand-cm-0018 and cand-cm-0025, the documented "de" inside "Delete".',
  'Gold self-check 7,512 / 7,512; 502 tests green including the new split-file and tier tests.',
  'Quotas 51 / 51 as tiered; the three tiers I restored match the drafts and the taxonomy examples.',
  'The :>> and :<< rows that looked mis-tiered are ex-mode double shifts with correct instructions.',
  'No case-silent row among the 2,374 authored today (0 unstated).',
  'The two gq rows set their own textwidth, so they do not inherit the terminal width.',
  'The 39 never-reviewed test rows are all from today; RESULTS.md counts (5,532 commands, 6,554 buffers, 55.2% ex) verified.',
  'No tracked file changed today; the 12:32 session\'s six-row move is consistent with its fix file and its tests pass.',
  'Pre-existing oddities, not from today: cand-cm-0105 and cand-cm-0106 hold an empty buffer with the cursor on line 2 (vim clamps it).'
];
// --------------------------------------------------------------- render
function render() {
  document.getElementById('band').innerHTML = [
    [S.today.toLocaleString(), 'rows authored'], ['1', 'leak'], ['4', 'defects or stale'], ['3', 'decisions'], [rowKeys().length, 'rows to review']
  ].map(function (b) { return '<div><span class="v">' + b[0] + '</span><span class="k">' + b[1] + '</span></div>'; }).join('');
  document.getElementById('toc').innerHTML = ITEMS.map(function (it) {
    return '<a href="#item-' + it.n + '"><span class="dot ' + it.sev + '"></span>' + it.n + ' ' + it.title.replace(/&[a-z]+;|<[^>]+>/g, '').slice(0, 44) + '</a>';
  }).join('') + '<span class="prog" id="prog"></span>';
  document.getElementById('items').innerHTML = ITEMS.map(itemHtml).join('');
  document.getElementById('good').innerHTML = '<details><summary class="lab" style="cursor:pointer">what was checked and passed (' + GOOD.length + ')</summary><ul>' + GOOD.map(function (g) { return '<li>' + esc(g) + '</li>'; }).join('') + '</ul></details>';
  document.getElementById('boot').remove();
  progress();
}
function rowKeys() {
  const keys = [];
  DATA.leak.forEach(function (e) { keys.push(e.train.id); });
  DATA.capitalize.forEach(function (r) { keys.push(r.id); });
  DATA.paste.forEach(function (r) { keys.push(r.id); });
  DATA.val.forEach(function (r) { keys.push(r.id); });
  DATA.repeat.forEach(function (g, i) { keys.push('group:' + i); });
  DATA.tiers.forEach(function (r) { keys.push(r.id); });
  return keys;
}
function progress() {
  const keys = rowKeys();
  const done = keys.filter(function (k) { return vget(k); }).length;
  const items = ITEMS.reduce(function (n, it) { return n + (it.decisions || []).length; }, 0);
  const idone = ITEMS.reduce(function (n, it) { return n + (it.decisions || []).filter(function (d) { return vget(d[0]); }).length; }, 0);
  const t = 'rows ' + done + '/' + keys.length + ' · decisions ' + idone + '/' + items;
  document.getElementById('prog').textContent = t;
  document.getElementById('sum').textContent = t;
}
document.addEventListener('click', function (ev) {
  const b = ev.target.closest('.seg button');
  if (!b) return;
  const key = b.parentNode.getAttribute('data-key'), v = b.getAttribute('data-v');
  verdicts[key] = verdicts[key] || {};
  if (verdicts[key].verdict === v) delete verdicts[key].verdict; else verdicts[key].verdict = v;
  if (!verdicts[key].verdict && !verdicts[key].note) delete verdicts[key];
  save();
  Array.prototype.forEach.call(b.parentNode.querySelectorAll('button'), function (x) { x.classList.toggle('on', x === b && !!(verdicts[key] && verdicts[key].verdict)); });
  const card = b.closest('.card');
  if (card) card.classList.toggle('done', !!(verdicts[key] && verdicts[key].verdict));
  progress();
});
document.addEventListener('input', function (ev) {
  const t = ev.target;
  if (t.hasAttribute('data-note')) {
    const key = t.getAttribute('data-note');
    verdicts[key] = verdicts[key] || {};
    if (t.value) verdicts[key].note = t.value; else delete verdicts[key].note;
    if (!verdicts[key].verdict && !verdicts[key].note) delete verdicts[key];
    save(); return;
  }
  if (t.hasAttribute('data-field')) {
    const id = t.getAttribute('data-id'), f = t.getAttribute('data-field'), orig = t.getAttribute('data-orig');
    edits[id] = edits[id] || {};
    if (t.value === orig) delete edits[id][f]; else edits[id][f] = t.value;
    if (!Object.keys(edits[id]).length) delete edits[id];
    t.classList.toggle('dirty', t.value !== orig);
    save();
  }
});
function exportJson() {
  const out = { audit: DATA.audit, items: {}, review: {}, edits: {}, deletes: {}, tiers: {}, groups: {} };
  ITEMS.forEach(function (it) { (it.decisions || []).forEach(function (d) { if (verdicts[d[0]]) out.items[d[0]] = verdicts[d[0]]; }); });
  function edit(id, field, value) { out.edits[id] = out.edits[id] || {}; out.edits[id][field] = value; }
  DATA.leak.forEach(function (e) {
    const id = e.train.id, v = verdicts[id] || {};
    if (v.verdict === 'new buffer') edit(id, 'buffer_before', (edits[id] && edits[id].buffer_before !== undefined) ? edits[id].buffer_before : e.train.proposed_buffer);
    if (v.verdict === 'drop') out.deletes[id] = v.note || 'audit 2026-09-06: buffer template shared with test';
    if (v.verdict || v.note) out.review[id] = v;
  });
  DATA.capitalize.forEach(function (r) {
    const id = r.id, v = verdicts[id] || {};
    if (v.verdict === 'reword') edit(id, 'instruction', (edits[id] && edits[id].instruction !== undefined) ? edits[id].instruction : r.proposed);
    if (v.verdict === 'drop') out.deletes[id] = v.note || 'audit 2026-09-06: ambiguous case wording';
    if (v.verdict || v.note) out.review[id] = v;
  });
  DATA.paste.forEach(function (r) {
    const id = r.id, v = verdicts[id] || {};
    if (v.verdict === 'retarget') edit(id, 'command', (edits[id] && edits[id].command !== undefined) ? edits[id].command : r.proposed);
    if (v.verdict === 'drop') out.deletes[id] = v.note || 'audit 2026-09-06: gold not reproducible through the pasteable rendering';
    if (v.verdict || v.note) out.review[id] = v;
  });
  DATA.val.forEach(function (r) {
    const id = r.id, v = verdicts[id] || {};
    if (v.verdict === 'reword' && edits[id] && edits[id].instruction !== undefined) edit(id, 'instruction', edits[id].instruction);
    if (v.verdict === 'drop') out.deletes[id] = v.note || 'audit 2026-09-06: val review';
    if (v.verdict || v.note) out.review[id] = v;
  });
  DATA.repeat.forEach(function (g, i) {
    const v = verdicts['group:' + i];
    if (v) out.groups[g.instruction] = { verdict: v.verdict, note: v.note, rows: g.rows.map(function (r) { return r.id; }) };
  });
  DATA.tiers.forEach(function (r) {
    const v = verdicts[r.id] || {};
    let t = null;
    if (v.verdict && v.verdict.indexOf('set ') === 0) t = r.modal;
    else if (v.verdict === 'T1' || v.verdict === 'T2' || v.verdict === 'T3') t = v.verdict;
    if (t && t !== r.tier) out.tiers[r.id] = t;
    if (v.verdict || v.note) out.review[r.id] = v;
  });
  const ta = document.getElementById('out');
  ta.value = JSON.stringify(out, null, 1);
  ta.hidden = false;
  document.getElementById('sum').textContent = Object.keys(out.edits).length + ' edits · ' + Object.keys(out.deletes).length + ' deletes · ' + Object.keys(out.tiers).length + ' tier changes · ' + Object.keys(out.items).length + ' decisions';
  return ta.value;
}
document.getElementById('exportBtn').addEventListener('click', exportJson);
document.getElementById('copyBtn').addEventListener('click', function () {
  const text = exportJson();
  if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(function () { document.getElementById('sum').textContent += ' · copied'; }, function () {});
});
document.getElementById('resetBtn').addEventListener('click', function () {
  if (!confirm('Clear every verdict, note and edit on this page?')) return;
  verdicts = {}; edits = {}; save(); render();
});
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = gather()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = HEAD + BODY.replace("@@DATA@@", payload)
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"], name="audit review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    s = data["stats"]
    print("findings 8 · leak %d · capitalize %d · paste %d · val %d · repeat groups %d (%d rows) · tiers %d · page %d KB -> %s"
          % (len(data["leak"]), len(data["capitalize"]), len(data["paste"]), s["val"],
             s["repeat_groups"], s["repeat_rows"], s["tiers"], len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
