#!/usr/bin/env python3
"""Build the review page for the open corpus decisions of 2026-09-08.

Eight items from the handoff's open list, each with the rows behind it, what
is at stake, and a verdict control: the 39 unreviewed test rows, the empty
buffer as a template, the twinned visual rows, the mined near-duplicates, the
tierless mined rows, the leakage component, case-insensitive matching (with
sandbox-verified drafts to approve), and the British spellings in test.

Everything on the page is computed from the corpus when the page is built;
the drafts are executed before they are shown. The export is one JSON (JavaScript Object Notation):
`edits` and `deletes` run through apply_curation.py directly; `retarget`
is the map retarget_commands.py takes; `tiers`, `hubs`, `drafts` and `items`
carry the decisions that need another tool or a conversation.

Usage:
    python scripts/build_decisions_review.py --out page.html
"""
import argparse
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim                              # noqa: E402
from apply_instructions import leaks                      # noqa: E402
from build_splits import build_components                 # noqa: E402
from build_test_review import review_history, collect     # noqa: E402
from build_visual_scope import find_twin                  # noqa: E402
from generate_data import norm_buffer                     # noqa: E402
from make_v3_emulator import guess_tier                   # noqa: E402
from page_guard import check_page                         # noqa: E402

TODAY = "2026-09-08"
FIELDS = ("id", "category", "tier", "tier_provenance", "mode", "instruction",
          "gold_command", "buffer_before", "buffer_after", "cursor", "authored_on")
STRICT_TWIN = re.compile(r"^v[ia].[dcyx~]")      # select a text object, then operate
DUP_PAIRS = (("emul-idea-0001", "cand-cm-0158"), ("emul-idea-0003", "cand-cm-0057"),
             ("emul-idea-0006", "cand-vsv-0046"))
BRITISH = {"hc-0438": ("capitalised", "capitalized"), "hc-1359": ("Alphabetise", "Alphabetize"),
           "hc-1360": ("Alphabetise", "Alphabetize"), "hc-1446": ("Capitalise", "Capitalize"),
           "hc-1625": ("Capitalise", "Capitalize")}
EMPTY_MOVED = "cand-vsv-0006"

# Sandbox-verified by this builder before they reach the page.
CI_DRAFTS = [
    {"category": "substitute", "tier": "T2", "mode": "ex", "command": "%s/error/failure/gi",
     "instruction": "Replace every occurrence of error with failure, upper or lower case alike.",
     "buffer_before": "Error: disk full\nerror: retry scheduled\nERROR: fatal\n", "cursor": [1, 1],
     "expected_after": "failure: disk full\nfailure: retry scheduled\nfailure: fatal\n"},
    {"category": "substitute", "tier": "T2", "mode": "ex", "command": "%s/\\ctodo/DONE/g",
     "instruction": "Mark every todo as DONE, no matter how it is cased.",
     "buffer_before": "todo: write tests\nTODO: ship it\nTodo: update docs\n", "cursor": [1, 1],
     "expected_after": "DONE: write tests\nDONE: ship it\nDONE: update docs\n"},
    {"category": "substitute", "tier": "T2", "mode": "ex", "command": "2,$s/null/NA/gi",
     "instruction": "In the data rows, turn every null marker into NA whether it is written in capitals or not.",
     "buffer_before": "id,val\n1,NULL\n2,null\n3,Null\n", "cursor": [1, 1],
     "expected_after": "id,val\n1,NA\n2,NA\n3,NA\n"},
    {"category": "global", "tier": "T2", "mode": "ex", "command": "g/\\cdebug/d",
     "instruction": "Drop the debug lines, ignoring case.",
     "buffer_before": "info start\nDEBUG probe\ndebug tick\ninfo stop\nDebug end\n", "cursor": [1, 1],
     "expected_after": "info start\ninfo stop\n"},
    {"category": "global", "tier": "T2", "mode": "ex", "command": "v/\\ckeep/d",
     "instruction": "Keep only the lines that mention keep in any letter case, and delete the rest.",
     "buffer_before": "KEEP a\ndrop b\nkeep c\nKeep d\nremove e\n", "cursor": [1, 1],
     "expected_after": "KEEP a\nkeep c\nKeep d\n"},
    {"category": "search-edit", "tier": "T2", "mode": "keys", "command": "/\\cwarning<CR>dd",
     "instruction": "Jump to the first warning, matched without regard to case, and delete that line.",
     "buffer_before": "ok\nWARNING low disk\nok again\n", "cursor": [1, 1],
     "expected_after": "ok\nok again\n"},
    {"category": "sort-uniq", "tier": "T2", "mode": "ex", "command": "sort i",
     "instruction": "Order these fruit names alphabetically, treating upper and lower case as the same letter.",
     "buffer_before": "banana\nApple\ncherry\nBanana\napple\n", "cursor": [1, 1],
     "expected_after": "Apple\napple\nbanana\nBanana\ncherry\n"},
    {"category": "sort-uniq", "tier": "T2", "mode": "ex", "command": "sort iu",
     "instruction": "Alphabetize these ignoring case, and keep just one of each name that differs only by capitals.",
     "buffer_before": "banana\nApple\ncherry\nBanana\napple\n", "cursor": [1, 1],
     "expected_after": "Apple\nbanana\ncherry\n"},
]


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


# What the owner decided and what was applied, 2026-09-08. Stamped on the page
# so it reads as a record rather than a standing to-do list.
APPLIED = {
    2: ("done", "All 39 read and approved as they stand. The test slice is "
                "fully verified again, 445 of 445, and review_history() reads "
                "the pass so the cards below say so."),
    6: ("kept", "Owner: keep the rule. The empty buffer counts as a template, "
                "cand-vsv-0006 stays in train, and the guard keeps no special "
                "case. The coverage cost is accepted: because empty is a "
                "singleton template, the measured slice can never test typing "
                "into an empty file."),
    8: ("done", "Capped at three per gold command: 11 rows dropped from the "
                "four commands that were over (vaWd 9, vawd 6, vi'c 4, v2aWd "
                "4), keeping the three lowest ids of each. emul-vsc-0505 moved "
                "out of val. Twins are now 60, all in train, none over three. "
                "No quota cell fell short."),
    11: ("done", "Both rows of each pair kept — fewer than three duplicates in "
                 "train is acceptable and all six are in train."),
    12: ("done", "All filed by the heuristic and marked heuristic: T1 33, T2 "
                 "45, T3 69 in the final pass, 59 more tiered along the way by "
                 "the buffer batch. The corpus has no tierless row left."),
    13: ("done", "441 rows — the ones on the 72 most-shared buffers — were "
                 "given distinct domain buffers, each executed against its own "
                 "gold command first. The component fell from 883 rows to 182, "
                 "and operator-motion left it entirely."),
    14: ("done", "40 rows authored across substitute, global, search-edit and "
                 "sort-uniq, in 40 distinct phrasings. Corpus rows exercising "
                 "case-insensitive matching: 19 to 59."),
    15: ("done", "All five Americanized; they remain verified test rows."),
}


def gather():
    corpus = load(REPO / "data" / "combined_cases.jsonl")
    by_id = {r["id"]: r for r in corpus}
    split_of = {}
    for s in ("train", "val", "test"):
        for r in load(REPO / "data" / "splits" / f"{s}.jsonl"):
            split_of[r["id"]] = s

    def row(i):
        r = by_id[i]
        d = {k: r.get(k) for k in FIELDS}
        d["split"] = split_of.get(i)
        d["source"] = i.rsplit("-", 1)[0]
        return d

    # 2 -- the unreviewed test rows, with the case signals the test page computes
    line = review_history()
    never = {r["id"] for r in load(REPO / "data" / "splits" / "test.jsonl")
             if "NEVER REVIEWED" in line(r)}
    signals = {c["id"]: c for c in collect(str(REPO / "data" / "combined_cases.jsonl"),
                                           select=lambda r: r["id"] in never,
                                           flag=lambda r, risk: risk == "ambiguous")}
    unreviewed = []
    for i in sorted(never):
        d = row(i)
        c = signals.get(i, {})
        d["case_risk"], d["case_signals"] = c.get("case_risk", ""), c.get("case_signals", [])
        unreviewed.append(d)

    # 6 -- every row that starts from an empty buffer
    empty = [row(r["id"]) for r in corpus if r["buffer_before"] == ""]

    # 8 -- visual rows a plainer command reproduces
    twins = []
    for r in corpus:
        if r["category"] in ("visual", "visual-block") and r["mode"] == "keys":
            twin, mode = find_twin(r)
            if twin:
                d = row(r["id"])
                d["twin"], d["twin_mode"] = twin, mode
                d["strict"] = bool(STRICT_TWIN.match(r["gold_command"]))
                d["mentions_selection"] = bool(re.search(
                    r"\b(select|selection|highlight|visual)", r["instruction"], re.I))
                twins.append(d)
    twins.sort(key=lambda d: (not d["strict"], d["source"], d["id"]))

    # 11 -- the mined near-duplicates the normalized dedup surfaced
    dups = [[row(a), row(b)] for a, b in DUP_PAIRS]

    # 12 -- tierless rows, with what the heuristic would say
    tierless = []
    for r in corpus:
        if r.get("tier") not in ("T1", "T2", "T3"):
            d = row(r["id"])
            d["suggested"] = guess_tier(r["mode"], r["gold_command"])
            tierless.append(d)
    tierless.sort(key=lambda d: (d["category"], d["id"]))

    # 13 -- the largest leakage component and its hub buffers
    comps = sorted(build_components(corpus), key=len, reverse=True)
    big = [corpus[i] for i in comps[0]]
    bybuf = collections.defaultdict(list)
    for r in big:
        bybuf[r["buffer_before"]].append(r)
    hubs = []
    for buf, rs in sorted(bybuf.items(), key=lambda kv: -len(kv[1]))[:30]:
        hubs.append({"buffer_before": buf, "n": len(rs),
                     "categories": dict(collections.Counter(r["category"] for r in rs).most_common()),
                     "sources": dict(collections.Counter(r["id"].rsplit("-", 1)[0] for r in rs).most_common()),
                     "rows": [{"id": r["id"], "gold_command": r["gold_command"], "mode": r["mode"],
                               "instruction": r["instruction"], "category": r["category"]} for r in rs]})
    op = [r for r in corpus if r["category"] == "operator-motion"]
    component = {
        "n": len(big), "distinct_commands": len({r["gold_command"] for r in big}),
        "distinct_buffers": len(bybuf),
        "by_category": dict(collections.Counter(r["category"] for r in big).most_common()),
        "by_source": dict(collections.Counter(r["id"].rsplit("-", 1)[0] for r in big).most_common()),
        "by_split": dict(collections.Counter(split_of.get(r["id"]) for r in big)),
        "operator_motion": [sum(1 for r in big if r["category"] == "operator-motion"), len(op)],
        "next_sizes": [len(c) for c in comps[1:6]],
        "hub_rows": sum(h["n"] for h in hubs), "hubs": hubs,
    }

    # 14 -- case-insensitive matching. The authored batch is the source of
    # truth once it is ingested; CI_DRAFTS is the fallback for a run before
    # that. Each row is re-executed here, so the page shows what vim does now.
    have = {norm_buffer(r["buffer_before"]): r["id"] for r in corpus}
    batch = REPO / "data" / "fixes" / "ignore_case_drafts_2026-09-08.json"
    source = json.loads(batch.read_text(encoding="utf-8")) if batch.exists() else CI_DRAFTS
    drafts = []
    for d in source:
        cmd = d["command"].lstrip(":") if d["mode"] == "ex" else d["command"]
        res = run_vim(d["buffer_before"], cmd, mode=d["mode"], cursor=tuple(d["cursor"]))
        assert res.get("was_valid") and res["buffer_after"] == d["expected_after"], d["command"]
        assert not leaks(d["instruction"], cmd, cmd), d["command"]
        drafts.append(dict(d, vim_after=res["buffer_after"],
                           ingested=have.get(norm_buffer(d["buffer_before"]))))
    ci_now = sum(1 for r in corpus if re.search(r"\\c|/[a-z]*i[a-z]*$|^sort.*\bi", r["gold_command"]))

    # 15 -- British spellings in test rows
    british = []
    for i, (word, us) in BRITISH.items():
        d = row(i)
        d["word"], d["proposed"] = word, d["instruction"].replace(word, us)
        british.append(d)

    return {
        "today": TODAY,
        "stats": {"corpus": len(corpus), "test": sum(1 for v in split_of.values() if v == "test"),
                  "unreviewed": len(unreviewed), "twins": len(twins),
                  "strict": sum(1 for t in twins if t["strict"]), "tierless": len(tierless),
                  "ci_rows_now": ci_now},
        "unreviewed": unreviewed, "empty": empty, "twins": twins, "dups": dups,
        "tierless": tierless,
        "tierless_stats": {"by_category": dict(collections.Counter(d["category"] for d in tierless).most_common()),
                           "suggested": dict(collections.Counter(d["suggested"] for d in tierless))},
        "component": component, "drafts": drafts, "british": british,
        "applied": {str(k): {"state": v[0], "text": v[1]} for k, v in APPLIED.items()},
    }


# ------------------------------------------------------------------- page

HEAD = r"""<title>Corpus Decisions 2026-09-08</title>
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
h3{ font-size:15px; font-family:var(--body); font-weight:600; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); font-weight:700; }
.wrap{ max-width:1180px; margin:0 auto; padding:26px 22px 130px; }
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
.dot.bad{ background:var(--bad); } .dot.warn{ background:var(--warn); } .dot.info{ background:var(--accent); }
.sev{ font-family:var(--mono); font-size:10px; letter-spacing:.1em; text-transform:uppercase; font-weight:700; padding:2px 7px; border-radius:3px; display:inline-flex; gap:6px; align-items:center; }
.sev.bad{ color:var(--bad); background:var(--bad-soft); } .sev.warn{ color:var(--warn); background:var(--warn-soft); } .sev.info{ color:var(--accent-ink); background:var(--accent-soft); }
.sev.done{ color:var(--ok); background:var(--ok-soft); }
.sev.kept{ color:var(--muted); background:var(--raise); }
.outcome.kept{ border-left-color:var(--muted); }
.outcome{ margin:12px 0 0 66px; padding:10px 14px; border-left:3px solid var(--ok);
  background:var(--surface); border-radius:0 6px 6px 0; max-width:74ch; color:var(--ink2); }
.outcome.open{ border-left-color:var(--warn); }
.banner{ margin:16px 0 0; padding:12px 16px; border:1px solid var(--ok); border-radius:6px;
  background:var(--ok-soft); color:var(--ink); max-width:82ch; }
section.item{ margin-top:34px; padding-top:22px; border-top:1px solid var(--line); scroll-margin-top:52px; }
section.item .head{ display:grid; grid-template-columns: 52px minmax(0,1fr); gap:14px; align-items:start; }
section.item .num{ font-family:var(--disp); font-size:30px; color:var(--faint); line-height:1; padding-top:2px; font-variant-numeric:tabular-nums; }
.meta{ display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin-top:8px; }
.count{ font-family:var(--mono); font-size:11px; color:var(--muted); }
.why{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:14px 26px; margin:14px 0 0 66px; }
.why > div{ max-width:62ch; }
.why p{ margin:4px 0 0; color:var(--ink2); }
code{ font-family:var(--mono); font-size:12.5px; background:var(--sunk); padding:0 4px; border-radius:3px; }
.decide{ margin:16px 0 0 66px; display:flex; flex-wrap:wrap; gap:10px 14px; align-items:center; padding:12px 14px; background:var(--surface); border:1px solid var(--line); border-radius:6px; }
.decide .q{ font-weight:600; flex:1 1 320px; }
.seg{ display:inline-flex; border:1px solid var(--line); border-radius:5px; overflow:hidden; }
.seg button{ font:600 11.5px var(--mono); padding:6px 10px; border:0; background:var(--surface); color:var(--ink2); cursor:pointer; border-right:1px solid var(--line); }
.seg button:last-child{ border-right:0; }
.seg button:hover{ background:var(--raise); }
.seg button.on{ background:var(--accent); color:#fff; }
.seg button.on.v-drop, .seg button.on.v-reject, .seg button.on.v-drop-left, .seg button.on.v-drop-right{ background:var(--bad); }
.seg button.on.v-keep, .seg button.on.v-defer, .seg button.on.v-keep-both, .seg button.on.v-leave-tierless{ background:var(--muted); }
.seg button:focus-visible, textarea:focus-visible, input:focus-visible{ outline:2px solid var(--accent); outline-offset:1px; }
input.note{ font:12.5px var(--body); padding:6px 8px; border:1px solid var(--line); border-radius:4px; background:var(--surface); color:var(--ink); min-width:220px; flex:1 1 220px; }
.cards{ margin:14px 0 0 66px; display:grid; gap:12px; }
.card{ background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:12px 14px; }
.card.done{ border-color:var(--ok); }
.card.compact{ padding:8px 12px; }
.card .top{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:baseline; }
.card .id{ font-family:var(--mono); font-weight:700; font-size:12.5px; }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px; border-radius:4px; background:var(--raise); color:var(--ink2); }
.chip.split-test{ color:var(--bad); border:1px solid var(--bad); background:transparent; font-weight:700; }
.chip.split-val{ color:var(--warn); border:1px solid var(--warn); background:transparent; font-weight:700; }
.chip.strict{ color:var(--accent-ink); background:var(--accent-soft); font-weight:700; }
.chip.amb{ color:var(--bad); background:var(--bad-soft); font-weight:700; }
.chip.sug{ color:var(--warn); background:var(--warn-soft); }
.ins{ margin:8px 0 0; font-size:14.5px; }
.ins mark{ background:var(--warn-soft); color:var(--ink); padding:0 2px; border-radius:2px; }
.cmd{ font-family:var(--mono); font-size:12.5px; color:var(--accent-ink); }
.arrow{ color:var(--muted); padding:0 6px; }
.bufs{ display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:10px; margin-top:10px; }
.buf .lab{ margin-bottom:4px; display:block; }
pre{ margin:0; font-family:var(--mono); font-size:12px; line-height:1.45; background:var(--sunk); border:1px solid var(--line2); border-radius:4px; padding:8px 10px; overflow-x:auto; white-space:pre; }
pre .tab{ color:var(--warn); font-weight:700; }
pre .eol{ color:var(--faint); }
pre .cur{ background:var(--accent-soft); outline:1px solid var(--accent); }
.fix{ margin-top:10px; display:grid; grid-template-columns: minmax(0,1fr) minmax(0,1fr); gap:10px; }
.fix textarea, .fix input{ width:100%; font-family:var(--mono); font-size:12px; padding:7px 9px; border:1px solid var(--line); border-radius:4px; background:var(--surface); color:var(--ink); resize:vertical; }
.fix textarea.dirty, .fix input.dirty{ border-color:var(--warn); }
.fix .lab{ display:block; margin-bottom:4px; }
.vrow{ display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; margin-top:10px; }
.pair{ display:grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap:12px; }
.pair > div{ border:1px solid var(--line2); border-radius:5px; padding:8px 10px; background:var(--ground); }
table{ border-collapse:collapse; font-family:var(--mono); font-size:12px; margin-top:8px; }
th,td{ text-align:left; padding:4px 12px 4px 0; border-bottom:1px solid var(--line2); font-variant-numeric:tabular-nums; vertical-align:top; }
th{ font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--muted); }
details.rows{ margin-top:8px; }
details.rows summary{ cursor:pointer; font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.rowlist{ display:grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap:6px 14px; margin-top:8px; font-family:var(--mono); font-size:11.5px; }
.group{ margin:16px 0 0 66px; }
.group h3{ margin:0 0 6px; }
.tl{ display:grid; grid-template-columns: 120px 150px minmax(0,1fr) auto auto; gap:8px 12px; align-items:center; padding:6px 0; border-bottom:1px solid var(--line2); font-size:13px; }
.tl .id{ font-family:var(--mono); font-size:11.5px; }
.tl .c{ font-family:var(--mono); font-size:11.5px; color:var(--accent-ink); overflow-wrap:anywhere; }
.tl .i{ color:var(--ink2); }
.export{ position:fixed; left:0; right:0; bottom:0; background:var(--surface); border-top:1px solid var(--line); padding:10px 22px; display:flex; gap:12px; align-items:center; flex-wrap:wrap; z-index:6; }
.export button{ font:600 12px var(--mono); padding:8px 12px; border:1px solid var(--accent); border-radius:5px; background:var(--accent); color:#fff; cursor:pointer; }
.export button.quiet{ background:transparent; color:var(--accent-ink); }
.export .sum{ font-family:var(--mono); font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
.export textarea{ width:100%; height:160px; font-family:var(--mono); font-size:11px; border:1px solid var(--line); border-radius:4px; background:var(--sunk); color:var(--ink); }
.export textarea[hidden]{ display:none; }
#boot{ font-family:var(--mono); font-size:12px; color:var(--bad); }
@media (max-width: 700px){ .why, .decide, .cards, .group{ margin-left:0; } .fix{ grid-template-columns:1fr; } .tl{ grid-template-columns:1fr; } }
@media (prefers-reduced-motion: no-preference){ .card{ transition: border-color .15s; } }
</style>
"""

BODY = r"""
<div class="wrap">
<header>
  <div>
    <span class="lab">vimBot &middot; open corpus decisions &middot; 2026-09-08</span>
    <h1>Eight decisions, each with its rows</h1>
    <p class="lede">The items still open after the quota fill and its audit, numbered as in the
    list they came from. Each one shows the rows it is about as they stand in the corpus today,
    says what is at stake, and takes a verdict. The case-insensitive drafts were executed in the
    sandbox before they were shown. Nothing has been changed.</p>
  </div>
  <div class="band" id="band"></div>
  <p class="banner" style="grid-column:1/-1"><b>All eight were decided on 2026-09-08.</b> Seven
  changed the corpus; item 6 kept the rule as it stands. The cards show the corpus after those
  changes, so the counts here are the new ones &mdash; 60 twins rather than 74, no tierless row,
  a 182-row component rather than 883. Verdict controls still work and still export.</p>
</header>
<nav class="toc" id="toc"></nav>
<div id="boot">Loading the items&hellip;</div>
<div id="items"></div>
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
const KEY = 'vimbot-decisions-0908', EKEY = KEY + '-edits';
let verdicts = {}, edits = {};
try { verdicts = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { verdicts = {}; }
try { edits = JSON.parse(localStorage.getItem(EKEY) || '{}'); } catch (e) { edits = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(verdicts)); localStorage.setItem(EKEY, JSON.stringify(edits)); } catch (e) {} }
function esc(s) { return String(s === null || s === undefined ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function bufHtml(text, opts) {
  opts = opts || {};
  if (text === null || text === undefined) return '<pre><span class="eol">(no result)</span></pre>';
  const parts = String(text).split('\n');
  if (parts.length > 1 && parts[parts.length - 1] === '') parts.pop();
  const cur = opts.cursor || null;
  const lines = parts.map(function (line, i) {
    let body = esc(line).replace(/\t/g, '<span class="tab">&#x21E5;   </span>');
    if (cur && cur[0] === i + 1) body = '<span class="cur">' + body + '</span>';
    return body;
  });
  if (!lines.length || (lines.length === 1 && lines[0] === '')) return '<pre><span class="eol">(empty buffer)</span></pre>';
  return '<pre>' + lines.join('\n') + '</pre>';
}
function buf(label, text, opts) { return '<div class="buf"><span class="lab">' + label + '</span>' + bufHtml(text, opts) + '</div>'; }
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
function rowTop(r, extra) {
  return '<div class="top"><span class="id">' + esc(r.id) + '</span>' + chip(r.category) + chip(r.tier || 'no tier') + chip(r.mode) + splitChip(r.split) + (extra || '') + '</div>';
}
function cardOpen(k, cls) { return '<div class="card ' + (cls || '') + (vget(k) ? ' done' : '') + '" id="row-' + esc(k) + '">'; }
// ------------------------------------------------------------------ cards
function unreviewedCard(r) {
  const k = r.id;
  const caseChip = r.case_risk === 'ambiguous' ? chip('case: ambiguous', 'amb') : (r.case_risk === 'relevant' ? chip('case matters') : '');
  return cardOpen(k) + rowTop(r, caseChip) +
    '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></p>' +
    (r.case_signals && r.case_signals.length ? '<p class="count" style="margin:4px 0 0">' + esc(r.case_signals.join(' · ')) + '</p>' : '') +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('after', r.buffer_after) + '</div>' +
    '<div class="fix">' + field(k, 'instruction', r.instruction, 'instruction (edit if rewording)', undefined, false) + '<div></div></div>' +
    '<div class="vrow">' + seg(k, ['ok', 'reword', 'drop'], vget(k)) + noteBox(k) + '</div></div>';
}
function emptyCard(r) {
  const moved = r.id === DATA.empty_moved ? chip('moved out of test 2026-09-07 · also on the unreproducible-gold list', 'amb') : '';
  return '<div class="card compact" id="row-' + esc(r.id) + '">' + rowTop(r, moved) +
    '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></p>' +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('after', r.buffer_after) + '</div></div>';
}
function twinCard(r) {
  const k = r.id;
  return cardOpen(k) + rowTop(r, (r.strict ? chip('strict: text object then operate', 'strict') : chip('broad')) + chip(r.source) +
      (r.mentions_selection ? chip('instruction speaks of selecting') : '')) +
    '<p class="ins">' + esc(r.instruction) + '</p>' +
    '<p class="cmd" style="margin:6px 0 0">' + esc(r.gold_command) + '<span class="arrow">&rarr;</span>' + esc(r.twin) + (r.twin_mode === 'ex' ? ' <span class="count">(ex)</span>' : '') + ' <span class="count">same buffer, verified</span></p>' +
    '<div class="bufs">' + buf('before', r.buffer_before, { cursor: r.cursor }) + buf('after', r.buffer_after) + '</div>' +
    '<div class="vrow">' + seg(k, ['retarget', 'keep', 'drop'], vget(k)) + noteBox(k) + '</div></div>';
}
function dupCard(pair, i) {
  const k = 'dup:' + i, a = pair[0], b = pair[1];
  function side(r) {
    return '<div><div class="top"><span class="id">' + esc(r.id) + '</span>' + chip(r.category) + splitChip(r.split) + '</div>' +
      '<p class="ins">' + esc(r.instruction) + ' &nbsp;<span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></p>' +
      bufHtml(r.buffer_before, { cursor: r.cursor }) + '</div>';
  }
  return cardOpen(k) + '<div class="pair">' + side(a) + side(b) + '</div>' +
    '<div class="vrow">' + seg(k, ['keep both', 'drop left', 'drop right'], vget(k)) + noteBox(k) + '</div></div>';
}
function tierlessRow(r) {
  const k = r.id;
  return '<div class="tl" id="row-' + esc(k) + '"><span class="id">' + esc(r.id) + '</span><span>' + chip(r.source) + splitChip(r.split) + '</span>' +
    '<span class="i">' + esc(r.instruction) + ' <span class="c">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></span>' +
    '<span>' + chip('heuristic says ' + r.suggested, 'sug') + '</span><span>' + seg(k, ['T1', 'T2', 'T3'], vget(k)) + '</span></div>';
}
function hubCard(h, i) {
  const k = 'hub:' + i;
  const cats = Object.keys(h.categories).map(function (c) { return c + ' ' + h.categories[c]; }).join(' · ');
  const srcs = Object.keys(h.sources).map(function (c) { return c + ' ' + h.sources[c]; }).join(' · ');
  return cardOpen(k) + '<div class="top"><span class="id">' + h.n + ' rows share this buffer</span>' + chip(cats) + chip(srcs) + '</div>' +
    '<div class="bufs" style="grid-template-columns:minmax(0,420px)">' + buf('buffer', h.buffer_before) + '</div>' +
    '<details class="rows"><summary>the rows and their commands</summary><div class="rowlist">' +
    h.rows.map(function (r) { return '<div>' + esc(r.id) + ' <span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span><br><span class="count">' + esc(r.instruction) + '</span></div>'; }).join('') +
    '</div></details>' +
    '<div class="vrow">' + seg(k, ['re-author these buffers', 'keep'], vget(k)) + noteBox(k, 'how many to re-author, or a domain to use') + '</div></div>';
}
function draftCard(d, i) {
  const k = 'draft:' + i;
  return cardOpen(k) + '<div class="top"><span class="id">' + (d.ingested ? esc(d.ingested) : 'draft ' + (i + 1)) + '</span>' + chip(d.category) + chip(d.tier) + chip(d.mode) + chip(d.ingested ? 'in the corpus, in train' : 'sandbox-verified', 'strict') + '</div>' +
    '<p class="ins">' + esc(d.instruction) + ' &nbsp;<span class="cmd">' + esc(d.mode) + ': ' + esc(d.command) + '</span></p>' +
    '<div class="bufs">' + buf('before', d.buffer_before, { cursor: d.cursor }) + buf('what vim produced', d.vim_after) + '</div>' +
    '<div class="fix">' + field(k, 'instruction', d.instruction, 'instruction (edit if rewording)', undefined, false) + '<div></div></div>' +
    '<div class="vrow">' + seg(k, ['approve', 'reword', 'reject'], vget(k)) + noteBox(k) + '</div></div>';
}
function britishCard(r) {
  const k = r.id;
  const shown = esc(r.instruction).replace(esc(r.word), '<mark>' + esc(r.word) + '</mark>');
  return cardOpen(k) + rowTop(r) +
    '<p class="ins">' + shown + ' &nbsp;<span class="cmd">' + esc(r.mode) + ': ' + esc(r.gold_command) + '</span></p>' +
    '<div class="fix">' + field(k, 'instruction', r.instruction, 'American spelling (prefilled)', r.proposed, false) + '<div></div></div>' +
    '<div class="vrow">' + seg(k, ['americanize', 'keep', 'drop'], vget(k)) + noteBox(k) + '</div></div>';
}
function decide(key, question, options, ph) {
  return '<div class="decide"><span class="q">' + question + '</span>' + seg(key, options, vget(key)) + noteBox(key, ph) + '</div>';
}
function why(blocks) {
  return '<div class="why">' + blocks.map(function (b) { return '<div><span class="lab">' + b[0] + '</span><p>' + b[1] + '</p></div>'; }).join('') + '</div>';
}
function itemHtml(it) {
  const ap = DATA.applied[String(it.n)] || null;
  const label = { done: '&#x2713; applied', kept: '&#x25CB; decided, no change' }[ap && ap.state] || '&#x25CB; still open';
  const stamp = ap ? '<span class="sev ' + (ap.state === 'done' ? 'done' : (ap.state === 'kept' ? 'kept' : 'warn')) + '">' + label + '</span>' : '';
  const outcome = ap ? '<div class="outcome' + (ap.state === 'done' ? '' : ' ' + ap.state) + '"><span class="lab">what was decided</span>' + esc(ap.text) + '</div>' : '';
  return '<section class="item" id="item-' + it.n + '"><div class="head"><div class="num">' + it.n + '</div><div>' +
    '<h2>' + it.title + '</h2><div class="meta">' + stamp + '<span class="sev ' + it.sev + '">' + it.sevWord + '</span><span class="count">' + it.count + '</span></div></div></div>' +
    outcome + why(it.why) + (it.decisions || []).map(function (d) { return decide(d[0], d[1], d[2], d[3]); }).join('') +
    (it.extra || '') + (it.cards ? '<div class="cards">' + it.cards + '</div>' : '') + '</section>';
}
const S = DATA.stats, C = DATA.component;
function tierlessGroups() {
  const by = {};
  DATA.tierless.forEach(function (r) { (by[r.category] = by[r.category] || []).push(r); });
  return Object.keys(by).sort(function (a, b) { return by[b].length - by[a].length; }).map(function (c) {
    return '<div class="group"><h3>' + esc(c) + ' <span class="count">' + by[c].length + '</span></h3>' + by[c].map(tierlessRow).join('') + '</div>';
  }).join('');
}
const ITEMS = [
  { n: 2, sev: 'bad', sevWord: '&#x25CF; measurement', title: 'The 39 test rows nobody has read',
    count: S.unreviewed + ' of ' + S.test + ' test rows carry no verdict',
    why: [
      ['what is at stake', 'These rows were authored on 2026-09-06 to fill quota cells and landed in test only because &sect;7.3 keeps a gold command inside one split. Every other test row has been read by you; these have not. The test slice is the measurement, so an unread row is not a weak example but an unverified instrument. The baseline reruns once these are settled.'],
      ['how to read them', 'The question is the one no gate answers: does the sentence describe the edit, completely and without a second faithful reading? <b>ok</b> keeps the row, <b>reword</b> plus an edited sentence changes it (the export is a change-set apply_curation.py runs), <b>drop</b> removes it. Rows the case classifier flags carry a chip.']
    ],
    cards: DATA.unreviewed.map(unreviewedCard).join('') },
  { n: 6, sev: 'info', sevWord: '&#x25C6; definition', title: 'Does the empty buffer count as a template?',
    count: DATA.empty.length + ' rows start from an empty buffer; all in train since 2026-09-07',
    why: [
      ['what happened', 'Canonicalizing buffers merged two spellings of empty (<code>""</code> and <code>"\\n"</code>) into one, and that exposed 13 train rows and one test row all starting from nothing. On the consolidation precedent, <code>' + esc(DATA.empty_moved) + '</code> moved to train: test 446 &rarr; 445, verified 407 &rarr; 406.'],
      ['the two readings', '&sect;7.3 says buffer templates are disjoint across splits, and read literally the empty buffer is a template like any other. But the rule exists so the model cannot memorize content, and the empty buffer has none: a train row starting from nothing tells the model nothing about a test row starting from nothing. Exempting it is intellectually clean and costs one special case in the guard &mdash; and a special case in the guard is exactly what hid the last defect. Note that the moved row is also one of the seven golds the pasteable rendering cannot reproduce, so it may leave test on that ground regardless.']
    ],
    decisions: [['item:6', 'Exempt the empty buffer from the template rule and move ' + esc(DATA.empty_moved) + ' back to test?', ['exempt, move it back', 'keep the rule', 'defer'], 'note']],
    cards: DATA.empty.map(emptyCard).join('') },
  { n: 8, sev: 'warn', sevWord: '&#x25B2; detours', title: 'Visual rows a plainer command reproduces',
    count: S.twins + ' of 421 visual rows have a twin, ' + S.strict + ' of them the strict select-a-text-object-then-operate kind; was 47 / 26 before yesterday\'s batch',
    why: [
      ['what is at stake', 'A twin means a shorter command gives the identical buffer: <code>vawd</code> is <code>daw</code>. Rows mined from editor test suites took the visual route because a fixture did, and training on them teaches the model to answer an ordinary request with a detour. Rows authored yesterday are different: the taxonomy admits visual as a category, and a sentence that says &ldquo;select the word and delete it&rdquo; legitimately earns <code>vawd</code>. The source chip and the &ldquo;speaks of selecting&rdquo; chip are there to tell the two apart.'],
      ['what each verdict does', '<b>retarget</b> replaces the gold with the twin through retarget_commands.py, which keeps the upstream keystrokes as provenance; a row retargeted to a non-visual command will need re-filing out of the visual category, which is a direct edit. <b>keep</b> accepts the visual route as intended. <b>drop</b> removes the row. The earlier recommendation was retarget about 35 and keep about 10, over the 47 that existed then.']
    ],
    decisions: [['item:8', 'Default for the mined rows (emul-vsc, cand-*) that do not speak of selecting?', ['retarget them', 'keep them', 'defer'], 'note']],
    cards: DATA.twins.map(twinCard).join('') },
  { n: 11, sev: 'info', sevWord: '&#x25C6; redundancy', title: 'Three mined pairs that are one task in two spellings',
    count: '6 rows, all in train',
    why: [
      ['what is at stake', 'The dedup keyed on the shared normalizers finds these three pairs the byte comparison missed: same lines, same cursor, same command, held apart only by a final newline that canonicalization has since removed. They are exact duplicates of the mechanical task now. Two spellings of one sentence over one buffer is the paraphrase design the corpus uses deliberately (the -a/-b pairs), but these are not marked as paraphrases and come from different sources.'],
      ['what each verdict does', '<b>keep both</b> leaves them; <b>drop left</b> / <b>drop right</b> removes that row through apply_curation.py.']
    ],
    cards: DATA.dups.map(dupCard).join('') },
  { n: 12, sev: 'info', sevWord: '&#x25C6; bookkeeping', title: 'The 206 mined rows with no tier',
    count: S.tierless + ' rows; the heuristic would file ' + Object.keys(DATA.tierless_stats.suggested).sort().map(function (t) { return t + ' ' + DATA.tierless_stats.suggested[t]; }).join(', '),
    why: [
      ['what is at stake', 'Every row mined from the candidate corpora (cand-cm, cand-vsv, toybox) carries an empty tier. The split builder stratifies on tier, but the splits are frozen now, so tier only matters for the quota table &mdash; and you said tier is not critical, just a spread of difficulty. That makes this bookkeeping rather than a defect. The heuristic (<code>guess_tier</code>) is shown beside each row as a suggestion; it is best-effort and wrong on blockwise commands, so it is not applied by default.'],
      ['what each option does', '<b>assign the heuristic tier</b> fills all 206 in one pass with <code>tier_provenance: heuristic</code>. <b>leave tierless</b> records that as the decision. <b>hand-tier here</b> means you pick per row below; only rows you touch are exported.']
    ],
    decisions: [['item:12', 'How should the 206 be tiered?', ['assign the heuristic tier', 'leave tierless', 'hand-tier here'], 'note']],
    extra: tierlessGroups() },
  { n: 13, sev: 'warn', sevWord: '&#x25B2; structure', title: 'The leakage component is now 883 rows',
    count: C.n + ' rows chained by shared buffers and commands, ' + C.distinct_buffers + ' buffers, ' + C.distinct_commands + ' commands; the next components are ' + C.next_sizes.join(', ') + ' rows',
    why: [
      ['what it is', 'The allocator groups rows that share a buffer template or a gold command, because §7.3 must keep each group inside one split. Groups chain: row A shares a buffer with B, B shares a command with C, and so on. The handoff described a 158-row component that was a third of operator-motion. It has grown to <b>' + C.n + '</b> rows &mdash; ' + Object.keys(C.by_source).map(function (s) { return s + ' ' + C.by_source[s]; }).join(', ') + ' &mdash; because yesterday\'s batch reused small buffers (<code>1\\n2\\n3\\n4\\n5</code>, <code>a\\nb\\nc</code>) and common commands, and every reuse welded more rows on. All of it sits in train, so it is not a leak. It is a constraint: none of these ' + C.n + ' rows can ever be sampled into val or test, which is why operator-motion (' + C.operator_motion[0] + ' of ' + C.operator_motion[1] + ' rows are in here) and visual are thin in test.'],
      ['the lever', 'Commands are shared legitimately and cannot be the lever. Buffers can: the 30 most-shared buffers below are the hubs, and giving their rows distinct domain buffers cuts the chains. The audit\'s own rewrite of 97 filler buffers yesterday is the same operation. <b>re-author these buffers</b> marks a hub for that; the note can say how many of its rows, or a domain to use.']
    ],
    decisions: [['item:13', 'Overall approach to the component?', ['diversify the hub buffers', 'accept it', 'defer'], 'note']],
    cards: C.hubs.map(hubCard).join('') },
  { n: 14, sev: 'info', sevWord: '&#x25C6; coverage', title: 'Case-insensitive matching is untaught',
    count: S.ci_rows_now + ' rows now use \\c, the i flag or sort i, from 19 before this batch; all ' + DATA.drafts.length + ' are below',
    why: [
      ['what is at stake', 'The corpus teaches case-sensitive matching thoroughly &mdash; 460 rows were repaired to state it &mdash; and the opposite not at all. A user who says &ldquo;ignoring case&rdquo; is asking for <code>\\c</code>, the <code>i</code> flag on <code>:s</code>, or <code>:sort i</code>, and the model has never seen any of them. This is a coverage gap in exactly the kind of instruction the case-clause work made the corpus careful about.'],
      ['what was authored', 'Forty rows across substitute 14, global 10, search-edit 8 and sort-uniq 8, each stating the case-insensitivity in a different phrasing, each executed in the sandbox with the result shown here. Four drafts were repaired against vim before ingest: <span class="mono">:normal</span> is refused inside <span class="mono">:g</span>, <span class="mono">:g/…/d|sort i</span> loses its sort because <span class="mono">:g</span> swallows the bar, <span class="mono">sort iu</span> with a pattern does not dedupe on the compared key, and descending case-folded order puts kiwi before Mango.']
    ],
    decisions: [['item:14', 'Author a case-insensitive batch in this style?', ['yes, about 40 rows', 'yes, these 8 only', 'no'], 'size or categories']],
    cards: DATA.drafts.map(draftCard).join('') },
  { n: 15, sev: 'info', sevWord: '&#x25C6; consistency', title: 'Five British spellings in test rows',
    count: DATA.british.length + ' verified test rows; the rest of the corpus writes -ize',
    why: [
      ['what is at stake', 'Small, but test is the measured slice and the model is trained on -ize. A spelling the model has rarely seen is a needless obstacle in an evaluation row. These are verified rows, so the edit is a reword and the row stays verified in substance.'],
      ['what each verdict does', '<b>americanize</b> applies the prefilled spelling through apply_curation.py; <b>keep</b> leaves it; <b>drop</b> removes the row.']
    ],
    cards: DATA.british.map(britishCard).join('') }
];
function rowKeys() {
  const keys = [];
  DATA.unreviewed.forEach(function (r) { keys.push(r.id); });
  DATA.twins.forEach(function (r) { keys.push(r.id); });
  DATA.dups.forEach(function (p, i) { keys.push('dup:' + i); });
  C.hubs.forEach(function (h, i) { keys.push('hub:' + i); });
  DATA.drafts.forEach(function (d, i) { keys.push('draft:' + i); });
  DATA.british.forEach(function (r) { keys.push(r.id); });
  return keys;
}
function render() {
  document.getElementById('band').innerHTML = [
    [ITEMS.length, 'decisions'], [S.unreviewed, 'test rows to read'], [S.twins, 'twinned visual rows'], [C.n.toLocaleString(), 'rows in the component'], [rowKeys().length, 'cards']
  ].map(function (b) { return '<div><span class="v">' + b[0] + '</span><span class="k">' + b[1] + '</span></div>'; }).join('');
  document.getElementById('toc').innerHTML = ITEMS.map(function (it) {
    return '<a href="#item-' + it.n + '"><span class="dot ' + it.sev + '"></span>' + it.n + ' ' + it.title.replace(/&[a-z#0-9]+;|<[^>]+>/g, '').slice(0, 40) + '</a>';
  }).join('') + '<span class="prog" id="prog"></span>';
  document.getElementById('items').innerHTML = ITEMS.map(itemHtml).join('');
  document.getElementById('boot').remove();
  progress();
}
function progress() {
  const keys = rowKeys();
  const done = keys.filter(function (k) { return vget(k); }).length;
  const tiers = DATA.tierless.filter(function (r) { return vget(r.id); }).length;
  const items = ITEMS.reduce(function (n, it) { return n + (it.decisions || []).length; }, 0);
  const idone = ITEMS.reduce(function (n, it) { return n + (it.decisions || []).filter(function (d) { return vget(d[0]); }).length; }, 0);
  const t = 'cards ' + done + '/' + keys.length + ' · tiers ' + tiers + '/' + DATA.tierless.length + ' · decisions ' + idone + '/' + items;
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
  const out = { audit: DATA.today, items: {}, review: {}, edits: {}, deletes: {}, retarget: {}, tiers: {}, hubs: {}, drafts: [] };
  ITEMS.forEach(function (it) { (it.decisions || []).forEach(function (d) { if (verdicts[d[0]]) out.items[d[0]] = verdicts[d[0]]; }); });
  function edit(id, f, value) { out.edits[id] = out.edits[id] || {}; out.edits[id][f] = value; }
  DATA.unreviewed.forEach(function (r) {
    const v = verdicts[r.id] || {};
    if (v.verdict === 'reword' && edits[r.id] && edits[r.id].instruction !== undefined) edit(r.id, 'instruction', edits[r.id].instruction);
    if (v.verdict === 'drop') out.deletes[r.id] = v.note || 'decisions 2026-09-08: test-row review';
    if (v.verdict || v.note) out.review[r.id] = v;
  });
  DATA.twins.forEach(function (r) {
    const v = verdicts[r.id] || {};
    if (v.verdict === 'retarget') out.retarget[r.id] = { command: r.twin, mode: r.twin_mode, was: r.gold_command };
    if (v.verdict === 'drop') out.deletes[r.id] = v.note || 'decisions 2026-09-08: visual detour';
    if (v.verdict || v.note) out.review[r.id] = v;
  });
  DATA.dups.forEach(function (p, i) {
    const v = verdicts['dup:' + i] || {};
    if (v.verdict === 'drop left') out.deletes[p[0].id] = v.note || 'decisions 2026-09-08: duplicate of ' + p[1].id;
    if (v.verdict === 'drop right') out.deletes[p[1].id] = v.note || 'decisions 2026-09-08: duplicate of ' + p[0].id;
    if (v.verdict || v.note) out.review['dup:' + i] = v;
  });
  DATA.tierless.forEach(function (r) { const v = vget(r.id); if (v) out.tiers[r.id] = v; });
  C.hubs.forEach(function (h, i) {
    const v = verdicts['hub:' + i];
    if (v) out.hubs[h.buffer_before] = { verdict: v.verdict, note: v.note, rows: h.rows.map(function (r) { return r.id; }) };
  });
  DATA.drafts.forEach(function (d, i) {
    const k = 'draft:' + i, v = verdicts[k] || {};
    if (v.verdict === 'approve' || v.verdict === 'reword') {
      const ins = (edits[k] && edits[k].instruction !== undefined) ? edits[k].instruction : d.instruction;
      out.drafts.push({ category: d.category, tier: d.tier, mode: d.mode, command: d.command, instruction: ins,
                        buffer_before: d.buffer_before, cursor: d.cursor, expected_after: d.expected_after, verdict: v.verdict, note: v.note });
    } else if (v.verdict || v.note) { out.drafts.push({ command: d.command, verdict: v.verdict, note: v.note }); }
  });
  DATA.british.forEach(function (r) {
    const v = verdicts[r.id] || {};
    if (v.verdict === 'americanize') edit(r.id, 'instruction', (edits[r.id] && edits[r.id].instruction !== undefined) ? edits[r.id].instruction : r.proposed);
    if (v.verdict === 'drop') out.deletes[r.id] = v.note || 'decisions 2026-09-08: spelling';
    if (v.verdict || v.note) out.review[r.id] = v;
  });
  const ta = document.getElementById('out');
  ta.value = JSON.stringify(out, null, 1);
  ta.hidden = false;
  document.getElementById('sum').textContent = Object.keys(out.edits).length + ' edits · ' + Object.keys(out.deletes).length + ' deletes · ' + Object.keys(out.retarget).length + ' retargets · ' + Object.keys(out.tiers).length + ' tiers · ' + Object.keys(out.hubs).length + ' hubs · ' + out.drafts.length + ' drafts · ' + Object.keys(out.items).length + ' decisions';
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
    data["empty_moved"] = EMPTY_MOVED
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = HEAD + BODY.replace("@@DATA@@", payload)
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"], name="decisions review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    s = data["stats"]
    print("unreviewed %d · empty %d · twins %d (strict %d) · dups %d · tierless %d · component %d rows / %d hubs · drafts %d · british %d · page %d KB -> %s"
          % (s["unreviewed"], len(data["empty"]), s["twins"], s["strict"], len(data["dups"]), s["tierless"],
             data["component"]["n"], len(data["component"]["hubs"]), len(data["drafts"]), len(data["british"]),
             len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
