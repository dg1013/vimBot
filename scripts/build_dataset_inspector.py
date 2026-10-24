#!/usr/bin/env python3
"""Build the dataset-inspector artifact: every corpus in the project, browsable.

Covers the material the review-queue artifact does not: dataset v3, the v1
hand-authored pool, the A3 regeneration tiers, the quarantine, the seed
cases, and the import-candidate samples. The live v2 splits appear as a
stratified sample only — the review-queue page browses those in full,
grouped by error class, and duplicating 15k cases here would just make a
slower copy of it.

Cards encode the review task directly: where `buffer_after` is a
subsequence of `buffer_before` (every `g//d` and `v//d` case), each source
line is marked kept or cut, so "did the right lines go?" is legible without
reading both panes.

Usage (repo root):
    conda run -n vimBot python scripts/build_dataset_inspector.py --out page.html
Then publish over the existing dataset-inspector artifact URL (uniform resource locator).
"""
import argparse
import base64
import collections
import html
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "deep_regex"))

V2_SAMPLE = 700
A3_PER_TIER = 120
SEED = 20260817


def read_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in Path(path).open(encoding="utf-8") if l.strip()]


def line_state(before, after):
    """Mark each before-line kept/cut when after is a subsequence of before.

    Returns None when it is not — keys-mode edits and substitutions rewrite
    lines rather than filter them, and a fake diff there would mislead.
    """
    b = before.split("\n")
    a = after.split("\n")
    out, i = [], 0
    for line in b:
        if i < len(a) and a[i] == line:
            out.append(1)
            i += 1
        else:
            out.append(0)
    return out if i == len(a) else None


def gold_of(row):
    """§5 key is `gold_command` (v3's kb13 file wrote `command` until the
    2026-08-17 schema fix). Tolerance kept for stale copies of that file."""
    return row.get("gold_command") or row.get("command") or ""


def card(ref, tab, group, row, meta, badges=None, facets=None):
    before = row.get("buffer_before", "")
    after = row.get("buffer_after", "")
    ins = row.get("instruction", "")
    return {
        "ref": ref, "tab": tab, "group": group,
        "badges": badges or [],
        "ins": [ins] if isinstance(ins, str) else list(ins),
        "cmd": gold_of(row),
        "before": before, "after": after,
        "keep": line_state(before, after),
        "meta": meta,
        "f": facets or {},
    }


# ---------------------------------------------------------------- corpora

def kb13_cards():
    rows = read_jsonl(REPO / "data" / "v3" / "kb13_cases.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        w = r.get("buffer_witnesses") or []
        meta = (f"normal form: {r.get('normal_form', '—')} · "
                f"witnesses: {', '.join(w) if w else 'none'} · "
                f"{r.get('source_index', [''])[0]} · MIT (KB13, human-authored)")
        out.append(card(f"KB13-{n:04d}", "kb13", r.get("tier", "?"), r, meta,
                        badges=[r.get("tier", ""),
                                "delete matching" if gold_of(r).startswith("g/")
                                else "keep matching"],
                        facets={"tier": r.get("tier", ""), "mode": r.get("mode", ""),
                                "category": r.get("category", ""),
                                "witness": w or ["none"],
                                "license": "MIT", "authorship": "human"}))
    return out


def v1_cards():
    rows = read_jsonl(REPO / "data" / "archive" / "v1-hand-authored" / "generated_cases.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        meta = (f"category: {r.get('category')} · domain: {r.get('domain', '—')} · "
                f"template: {r.get('template_id', '—')} · "
                f"hand-authored (our content, no third-party licence)")
        out.append(card(f"V1-{n:03d}", "v1", r.get("category", "?"), r, meta,
                        badges=[r.get("tier", ""), r.get("mode", ""), r.get("category", "")],
                        facets={"tier": r.get("tier", ""), "mode": r.get("mode", ""),
                                "category": r.get("category", ""),
                                "license": "ours", "authorship": "human"}))
    return out


def sregex_v3_cards():
    """The 447 StructuredRegex cases actually shipped in v3.

    Distinct from the `sregex` tab, which holds import *candidates* recovered
    from a prior artifact. These have been through the pipeline: converted,
    executed, and compared against real vim.
    """
    rows = read_jsonl(REPO / "data" / "v3" / "sregex_cases.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        prov = r.get("instruction_provenance", "?")
        w = r.get("buffer_witnesses") or []
        meta = (f"normal form: {r.get('normal_form', '—')} · "
                f"witnesses: {', '.join(w) if w else 'none'} · "
                f"instruction: {prov} · MIT (StructuredRegex, ACL 2020)")
        note = ("provenance_note: preview only — these 447 derive from 460 patterns "
                "recovered from a prior artifact, not a vendored checkout. A real "
                "import must vendor the corpus under §14c and re-run."
                if r.get("provenance_note") else "")
        c = card(f"SRX3-{n:04d}", "sregexv3", r.get("tier", "?"), r, meta,
                 badges=[r.get("tier", ""), prov,
                         "delete matching" if gold_of(r).startswith("g/") else "keep matching"],
                 facets={"tier": r.get("tier", ""), "mode": r.get("mode", ""),
                         "category": r.get("category", ""),
                         "instruction": prov, "domain": r.get("domain", ""),
                         "license": "MIT", "authorship": prov})
        if note:
            c["note"] = note
        out.append(c)
    return out


def emulator_cards():
    """All 600 execution-validated emulator cases now shipping in v3.

    Every row reproduced in real vim 8.2 before admission. Instructions are
    still pending by design — the emulator test titles name the command, so
    using them would leak the answer into the model's input.
    """
    rows = read_jsonl(REPO / "data" / "v3" / "emulator_cases.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        corpus = r.get("source_corpus", "?")
        commit = (r.get("source_commit") or "")[:12]
        pending = not (r.get("instruction") or "").strip()
        meta = (f"{corpus} @ {commit} · {r.get('source_file', '—')} · "
                f"test: {r.get('source_test_title') or '—'} · "
                f"{r.get('mining_round', '—')} · MIT")
        badges = [r.get("tier", ""), r.get("mode", ""), r.get("category", "")]
        if pending:
            badges.append("instruction pending")
        if r.get("was_frozen_v1") == "yes":
            badges.append("admitted by scope change")
        if r.get("dup_group"):
            badges.append(r["dup_group"])
        c = card(f"EMU-{n:04d}", "emulator", r.get("category", "?"), r, meta,
                 badges=[b for b in badges if b],
                 facets={"tier": r.get("tier", ""), "mode": r.get("mode", ""),
                         "category": r.get("category", ""),
                         "source": corpus, "round": r.get("mining_round", ""),
                         "instruction": "pending" if pending else "authored",
                         "unfrozen": r.get("was_frozen_v1", "no"),
                         "license": "MIT", "authorship": "test-derived"})
        c["cmdlbl"] = ("keystrokes" if r.get("mode") == "keys" else "ex command")
        if pending:
            c["note"] = ("No instruction yet. The upstream test title names the command "
                         "(\"Can handle 'dw'\"), so reusing it would leak the answer — the "
                         "natural language has to be authored. Everything else here is "
                         "verified: this exact command produced this exact buffer in vim 8.2.")
        if r.get("tier_provenance") == "heuristic":
            c["meta"] += " · tier assigned heuristically, not curated"
        out.append(c)
    return out


def candidate_cards():
    """The 211 cases mined from CodeMirror, VsVim and toybox after the emulator
    import — the first Apache-2.0 and 0BSD material in the project."""
    rows = read_jsonl(REPO / "data" / "v3" / "candidate_cases.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        prov = r.get("instruction_provenance", "?")
        pending = r.get("instruction_pending")
        meta = (f"{r.get('source_corpus')} @ {(r.get('source_commit') or '')[:12]} · "
                f"test: {r.get('source_test') or '—'} · {r.get('source_license')}")
        badges = [r.get("mode", ""), r.get("category", ""), r.get("source_license", "")]
        if pending:
            badges.append("instruction pending")
        if prov == "owner-edited":
            badges.append("owner-edited")
        if r.get("whitespace_only_change"):
            badges.append("whitespace-only")
        c = card(f"CAND-{n:04d}", "cand", r.get("category", "?"), r, meta,
                 badges=[b for b in badges if b],
                 facets={"mode": r.get("mode", ""), "category": r.get("category", ""),
                         "source": r.get("source_corpus", ""),
                         "license": r.get("source_license", ""),
                         "instruction": prov,
                         "authorship": "test-derived"})
        c["cmdlbl"] = "keystrokes" if r.get("mode") == "keys" else "ex command"
        if pending:
            c["note"] = ("No instruction yet. Held back deliberately rather than filled with "
                         "prose fitted to a keystroke sequence.")
        elif prov == "owner-edited":
            c["note"] = "Instruction written or rewritten by the owner in the review artifact."
        out.append(c)
    return out


def v2_sample_cards(rng):
    """Stratified sample of the live splits — full browse lives on the
    review-queue artifact, so this is a representative slice, labelled."""
    buckets = collections.defaultdict(list)
    for split in ("train", "val", "test"):
        for r in read_jsonl(REPO / "data" / "splits" / f"{split}.jsonl"):
            r["_split"] = split
            buckets[(split, r.get("tier"))].append(r)
    total = sum(len(v) for v in buckets.values())
    picked = []
    for key, rows in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        n = max(1, round(V2_SAMPLE * len(rows) / total))
        picked += rng.sample(rows, min(n, len(rows)))
    out = []
    for n, r in enumerate(sorted(picked, key=lambda x: x["id"]), 1):
        corp = "/".join(r.get("source_corpus") or ["?"])
        auth = "human" if "KB13" in corp else "machine"
        meta = (f"split: {r['_split']} · source pattern: {r.get('source_regex', '—')} · "
                f"corpus: {corp} · MIT (deep-regex)")
        out.append(card(f"V2-{n:04d}", "v2", r["_split"], r, meta,
                        badges=[r.get("tier", ""), r["_split"], corp],
                        facets={"tier": r.get("tier", ""), "split": r["_split"],
                                "corpus": corp, "authorship": auth,
                                "mode": r.get("mode", ""), "license": "MIT"}))
    return out


def a3_cards(rng):
    """Regenerated English at each readability tier, against the live case
    the pattern currently rides on."""
    census = REPO / "data" / "v3" / "a3_census.json"
    if not census.exists():
        return []
    rows = json.loads(census.read_text())
    by_rx = {}
    for split in ("train", "val", "test"):
        for r in read_jsonl(REPO / "data" / "splits" / f"{split}.jsonl"):
            by_rx.setdefault(r.get("source_regex"), r)
    out, n = [], 0
    for tier in ("clean", "borderline", "stilted"):
        pool = [r for r in rows if r.get("tier") == tier and r["regex"] in by_rx]
        for r in rng.sample(pool, min(A3_PER_TIER, len(pool))):
            n += 1
            live = dict(by_rx[r["regex"]])
            flags = ", ".join(r["flags"]) if r["flags"] else "none"
            meta = (f"pattern: {r['regex']} · {r['words']} words · flags: {flags} · "
                    f"{r['cases']} live case(s) · CURRENT: “{live.get('instruction','')}”")
            live["instruction"] = " ⁄ ".join(r["variants"])
            out.append(card(f"A3-{n:04d}", "a3", tier, live, meta,
                            badges=[tier, f"{r['words']}w", live.get("tier", "")],
                            facets={"tier": tier, "flags": r["flags"] or ["none"],
                                    "authorship": "generated", "mode": "ex"}))
    return out


def quarantine_cards():
    rows = read_jsonl(REPO / "data" / "quarantine_review.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        reason = r.get("reject_reason", "—")
        round_ = reason.split(":")[0] if ":" in reason else "removed"
        meta = f"reason: {reason} · source pattern: {r.get('source_regex', '—')}"
        out.append(card(f"QR-{n:03d}", "quar", round_, r, meta,
                        badges=[r.get("tier", ""), "removed"],
                        facets={"tier": r.get("tier", ""), "round": round_,
                                "mode": r.get("mode", "")}))
    return out


def seed_cards():
    rows = read_jsonl(REPO / "data" / "seed_cases.jsonl")
    out = []
    for n, r in enumerate(rows, 1):
        meta = f"category: {r.get('category')} · gold self-check corpus (14/14)"
        out.append(card(f"SD-{n:02d}", "seed", r.get("category", "?"), r, meta,
                        badges=[r.get("mode", ""), r.get("category", "")],
                        facets={"mode": r.get("mode", ""),
                                "category": r.get("category", ""),
                                "license": "ours", "authorship": "human"}))
    return out


def _strip_head(block):
    """'POSITIVE examples:\\nfoo bar' -> ['foo','bar'] (the export packs the
    strings space-separated on one line under a heading)."""
    body = block.split("\n", 1)[1] if "\n" in block else ""
    return [s for s in body.split() if s]


def sregex_panes(c):
    """pos/neg example strings ARE the ground-truth partition, so render the
    partition — this is the shape an imported case would take."""
    pos, neg = _strip_head(c.get("before", "")), _strip_head(c.get("after", ""))
    if not pos and not neg:
        return None
    lines = pos + neg
    return {"lbl": "example strings — match / non-match",
            "text": "\n".join(lines),
            "keep": [1] * len(pos) + [0] * len(neg),
            "note": (f"{len(pos)} positive · {len(neg)} negative · ground truth "
                     f"from the corpus, not executed — no vim command exists "
                     f"until the DSL adapter is built")}


def rdjson_panes(c):
    """Run the pattern over its own test text where Python can compile it."""
    before = c.get("before", "")
    if not before.startswith("test text:"):
        return None
    lines = [l for l in before.split("\n")[1:] if l.strip() not in ("", "-")]
    if not lines:
        return None      # heading with a placeholder body: no test text at all
    import re as _re
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rx = _re.compile(c.get("command", ""))
            hits = [1 if rx.search(l) else 0 for l in lines]
    except Exception as exc:
        return {"lbl": "test text — pattern did not compile", "text": "\n".join(lines),
                "keep": None,
                "note": f"python re rejected the pattern: {type(exc).__name__}. "
                        f"No result derivable."}
    if not any(hits) or all(hits):
        return {"lbl": "test text — pattern does not split it", "text": "\n".join(lines),
                "keep": None,
                "note": ("pattern matches every line or none, so there is no "
                         "contrastive edit here — unusable as a case as it stands")}
    return {"lbl": "test text — match / non-match", "text": "\n".join(lines),
            "keep": hits,
            "note": (f"{sum(hits)} of {len(hits)} lines match · derived with python "
                     f"re, NOT through the vim executor — indicative only")}


def import_cards():
    """Import-candidate samples, recovered from the previously published
    artifact into data/inspector_imports.json (they have no other source).

    These are candidates, not cases: none has been through the conversion
    pipeline, so none has a vim command or an executed result. Where the
    source carries enough to derive the match/non-match partition we show
    it and say how it was derived; where it does not, we say so rather
    than printing a placeholder dash.
    """
    path = REPO / "data" / "inspector_imports.json"
    if not path.exists():
        return []
    out = []
    for c in json.loads(path.read_text()):
        tab = c["tab"]
        lic = ("MIT" if tab == "sregex" else
               "CC BY-SA (excluded)" if tab == "regelso" else "MIT label (disputed)")
        panes = sregex_panes(c) if tab == "sregex" else \
            rdjson_panes(c) if tab == "rdjson" else None
        cmd = c.get("command", "")
        if cmd in ("null", "TBD", "-", ""):
            cmd = ""
        note = panes["note"] if panes else (
            "no example strings and no executable form in the source — only the "
            "question text survives, so nothing can be derived here")
        out.append({
            "ref": c["ref"], "tab": tab, "group": c.get("group", "—"),
            "badges": c.get("badges", []), "ins": c.get("instructions", []),
            "cmd": cmd,
            "cmdlbl": "source pattern (not a vim command)" if cmd else "",
            "before": panes["text"] if panes else "",
            "lbl": panes["lbl"] if panes else "",
            "after": "", "keep": panes["keep"] if panes else None,
            "note": note,
            "meta": c.get("meta", ""),
            "f": {"license": lic,
                  "derived": ("match partition" if panes and panes["keep"]
                              else "nothing derivable"),
                  "authorship": "human" if tab in ("sregex", "regelso") else "unknown"},
        })
    return out


# ------------------------------------------------------------------ page

TABS = [
    ("kb13", "v3 KB13 core", "733 cases · human-authored · the original v3 core"),
    ("sregexv3", "v3 StructuredRegex", "447 cases shipped in v3 · MIT · preview-provenance, not yet vendored"),
    ("emulator", "v3 emulator", "600 cases · MIT · every one reproduced in real vim 8.2 · instructions still pending"),
    ("cand", "v3 candidates", "211 cases from CodeMirror, VsVim and toybox · MIT / Apache-2.0 / 0BSD · 176 instructions written"),
    ("v1", "v1 hand-authored", "354 cases across all 15 categories, ~50% keys-mode — the only breadth material in the project"),
    ("v2", "v2 live splits", f"stratified sample of {V2_SAMPLE} from 15,373 — the review-queue artifact browses all of them by error class"),
    ("a3", "A3 regeneration", "deterministic English at each readability tier, against the instruction it would replace"),
    ("quar", "quarantine", "186 cases removed across four review rounds, each with its reason"),
    ("seed", "seed cases", "14 hand-written cases behind the gold self-check"),
    ("sregex", "StructuredRegex", "MIT · Turker describe-the-pattern · import candidate"),
    ("regelso", "regel / SO", "CC BY-SA taint · presumed excluded · shown for judgment"),
    ("rdjson", "regex_dataset", "MIT label, anonymous provenance · recommend dropping"),
]

# Per-corpus lineage: where the material came from and what made it a case.
# Kept beside the data so the page explains itself while it is browsed.
PROV = {
  "emulator": {
    "origin": "Test suites of two MIT-licensed vim emulators — VSCodeVim "
              "(@1118267006d0) and IdeaVim (@e66a3915a9b4). Their cases are already "
              "our schema: a starting buffer, a keystroke or ex command, and the "
              "buffer that should result, with the cursor marked in the source "
              "itself (`|` in VSCodeVim, `${c}` in IdeaVim).",
    "steps": [
      "Mine 61 test files across two rounds; round 2 became eligible only after "
      "the frozen-scope-v1 exclusions were replaced by data/command_coverage.md.",
      "Parse three test shapes: VSCodeVim's declarative newTest({...}), IdeaVim's "
      "doTest(keys, before, after), and typeTextInFile(...) + assertState(...).",
      "Recover the cursor from the source's own marker — 397 of the admitted rows "
      "carry a position other than 1:1, where all of v3 sits at 1:1.",
      "Replay every triple through src/executor.py against real vim 8.2 and "
      "compare with evaluate.py line-list semantics. The emulator's expectation "
      "is never trusted.",
      "Drop no-ops: 201 rows where buffer_after equals buffer_before are "
      "undecidable, since a passing verdict cannot distinguish a command that "
      "correctly did nothing from one that silently did nothing.",
      "Admit only rows the shipped src/scope.py also permits, so the file always "
      "passes the project's own gold self-check.",
    ],
    "gate": "600 of 919 mined rows admitted; 600/600 on evaluate.py. Rejected: 233 "
            "diverged from real vim, 59 multicursor (not a vim feature), 26 blocked "
            "by the scope screen, 1 invalid.",
    "caveat": "Instructions are PENDING on all 600 — the upstream test titles name "
              "the command, so reusing them would leak the answer. Tiers are "
              "heuristic, not curated, and the categories lean on keystroke shape, "
              "so visual operators mostly landed in operator-motion.",
  },
  "cand": {
    "origin": "Test suites of three more vim implementations, mined after the "
              "emulator import: CodeMirror's vim mode (MIT, @8640966b6977), VsVim "
              "(Apache-2.0, @2b0bc9b1943a — all 136 files of its VimCoreTest) and "
              "toybox's vi (0BSD, @b7ec52ac35e0). The project's first non-MIT "
              "material.",
    "steps": [
      "Mine 1,078 candidate triples; the CodeMirror miner pairs each keystroke "
      "batch with the assertion that follows it rather than concatenating a whole test.",
      "Replay every one through src/executor.py against real vim 8.2.",
      "Admit only exact agreement — no difference of any kind, whitespace included — "
      "under the §5 comparison, and require the buffer to change both as the source "
      "describes it and as vim produced it.",
      "Collapse duplicate edits: where one edit is reachable by several keystroke "
      "spellings, keep one representative.",
      "Hold accumulation artifacts — a key pressed five or more times in a row, or a "
      "command over max_command_len — which reproduce faithfully but are nobody's request.",
      "Author instructions by hand; the owner reviewed and rewrote 24 of them.",
    ],
    "gate": "211 of 1,078 admitted; 211/211 on evaluate.py. Rejected: 723 disagreed "
            "with vim, 82 accumulation artifacts, 44 changed nothing, 18 duplicate edits.",
    "caveat": "35 instructions still pending, and tier is deliberately unassigned rather "
              "than guessed. mini.nvim was mined too and yielded nothing: every literal-"
              "argument row that passes is a no-op.",
  },
  "sregexv3": {
    "origin": "StructuredRegex (Ye, Chen, Dillig & Durrett, ACL 2020), MIT. "
              "Turker-annotated in the describe-the-pattern methodology — genuine "
              "authorship at KB13 grade, and it ships positive and negative example "
              "strings that become the buffer directly.",
    "steps": [
      "Adapt the corpus DSL to our AST (sregex_adapter.py); all 14 operators map "
      "onto existing nodes, and 460/460 patterns agree with the corpus's own "
      "positive/negative strings.",
      "Rewrite the whole-string prose into participle instruction frames that read "
      "the same in singular and plural, so one rewrite serves delete and keep.",
      "Convert to a vim :g//:v// command through the same audited table as KB13.",
      "Use the corpus's own example strings as the buffer — human-authored "
      "near-misses rather than synthesised gibberish.",
      "Execute and require the buffer to match.",
    ],
    "gate": "Full-triangle verified; the file scores clean on evaluate.py.",
    "caveat": "PREVIEW PROVENANCE. These derive from 460 patterns recovered from a "
              "prior artifact, not from a vendored checkout — the corpus is not in "
              "data/external/ and no commit is pinned. A real import must vendor it "
              "under §14c and re-run. 207 of the instructions are a3-generated "
              "rather than human-rewritten.",
  },
  "kb13": {
    "origin": "KB13 slice of the deep-regex corpora (Kushman & Barzilay, NAACL 2013; "
              "redistributed MIT by Locascio et al., EMNLP 2016). 823 lines of "
              "(English description, regex) pairs, written by people describing a "
              "pattern they were shown — genuine authorship, not paraphrase.",
    "steps": [
      "Group the 823 lines by regex, collecting every distinct English description.",
      "Normalise each pattern with the A3 simplifier and merge patterns that "
      "collapse to the same normal form — one scenario per equivalence class.",
      "Template each human description into an instruction: “Delete {nl}” or "
      "“Keep only {nl}”, with a deterministic per-class coin flip choosing which.",
      "Screen every instruction through the garble and anchor-mismatch rules; "
      "drop any that fail. 9 were screened out, costing 8 whole scenarios.",
      "Compile the pattern to a vim command via the A2-audited conversion table, "
      "choosing :g or :v so the command matches the requested operation.",
      "Build an adversarial witness buffer — near-miss lines sampled from mutated "
      "ASTs so a sloppier command would give a visibly different result.",
      "Run it in sandboxed vim and require the buffer to match the span oracle; "
      "cross-check against the unrestricted Python implementation where expressible.",
      "Emit up to 3 paraphrase cases per scenario.",
    ],
    "gate": "Full verifier triangle per case — 590 three-way, 70 two-leg, "
            "0 mismatches shipped. 56 scenarios dropped for unbuildable buffers.",
    "caveat": "One category (global), one mode (ex), synthetic probe buffers.",
  },
  "v1": {
    "origin": "Written from scratch inside Claude Code sessions against the 15-category "
              "taxonomy — no third-party corpus. Sources live in data/candidates/. "
              "Includes 44 scenarios authored from the VimGolf theme taxonomy "
              "(theme concepts only, in our own words — no challenge content).",
    "steps": [
      "Author candidate tuples by hand per taxonomy cell: instruction, buffer, "
      "command, expected result, in both ex and keys mode.",
      "Feed each through generate_data.py, which screens the command against the "
      "scope policy (no writes, shell, scripting, macros, registers, visual mode).",
      "Execute in sandboxed headless vim and keep only tuples whose command "
      "reproduces the claimed buffer AND actually changes it.",
    ],
    "gate": "Execution gate. It caught two real authoring bugs before they landed — "
            "an unescaped \\| in a :s replacement and a dot-repeat miscount.",
    "caveat": "Archived when v2 replaced it. Restorable by one pipeline run; the "
              "full-pool rebuild yields 442 cases from 216 scenarios.",
  },
  "v2": {
    "origin": "All three deep-regex corpora merged — KB13 (823 human-authored) plus "
              "NL-RX-Turk (9,999 human-paraphrased machine text) and NL-RX-Synth "
              "(9,999 machine-generated). 10,367 unique regexes.",
    "steps": [
      "Merge the corpora by regex, so Synth and Turk descriptions of the same "
      "pattern become paraphrases of a single scenario.",
      "Template each source description: “Delete {nl}” / “Keep only {nl}”, with a "
      "deterministic hash splitting scenarios between the two operations.",
      "Build a probe buffer of up to 3 matching and 3 non-matching lines, "
      "classified by the span oracle on the original pattern.",
      "Predict the result oracle-side, then re-derive it in real vim — any "
      "disagreement surfaces as a `mismatch` reject.",
      "Stratify into train/val/test on (source, category, tier, feature bucket), "
      "with buffer templates and paraphrase clusters never straddling splits.",
    ],
    "gate": "Execution gate only — it validates command against buffer, never "
            "English against command. That gap is what four review rounds found.",
    "caveat": "Instruction quality tracks the seed corpus: KB13 99.1% screen "
              "approval, Turk 26.7%, Synth 25.4%. 44% of the test slice was removed.",
  },
  "a3": {
    "origin": "Not a corpus — generated. Same patterns as v2, but the English is "
              "derived from the pattern's syntax tree instead of taken from the "
              "corpus, because corpus English was the defective half.",
    "steps": [
      "Parse the pattern to an AST and normalise it to its true span semantics.",
      "Render English from the normalised tree with anchor-, adjacency- and "
      "count-honest phrasing, plus seeded surface variants.",
      "Refuse to render patterns whose honest phrasing does not exist — those "
      "route to degenerate-quarantine rather than getting invented text.",
      "Sort the renderings into clean / borderline / stilted by readability.",
    ],
    "gate": "Honesty is structural: the English is a function of the same tree the "
            "command compiles from, so it cannot contradict the command.",
    "caveat": "That is also the weakness — instruction and command are both "
              "deterministic functions of one tree, so training and testing on A3 "
              "text measures template inversion. Renders 71.6% of live cases.",
  },
  "quar": {
    "origin": "Cases removed from the live dataset across four owner review rounds "
              "plus the automated audits, kept rather than deleted so the "
              "defect taxonomy stays inspectable and can seed preference pairs.",
    "steps": [
      "Round 1 — 31 cases the owner condemned from a 200-case flagged slice.",
      "Round 2 — 20 more from the flagged half.",
      "Round 3 — all 71 unflagged-sweep cases; this crossed the 300-case test "
      "floor and triggered a pre-approved leakage-safe val→test backfill.",
      "Round 4 — 54 audit-condemned test cases and 10 degenerate train duplicates.",
    ],
    "gate": "Every row carries a `reject_reason` naming the round and the defect.",
    "caveat": "186 cases total. Cumulatively 44% of the original 400-case test slice.",
  },
  "seed": {
    "origin": "Fourteen cases written by hand at the very start of the project, "
              "before any corpus existed, to pin the executor's behaviour.",
    "steps": [
      "Hand-write instruction, buffer, command and expected result across "
      "6 categories in both modes.",
      "Run as the gold self-check: src/evaluate.py with no predictions file "
      "executes each gold command and requires an exact buffer match.",
    ],
    "gate": "Self-check is 14/14 and is part of the standing health check.",
    "caveat": "Not training material — a fixture for the harness itself.",
  },
  "sregex": {
    "origin": "StructuredRegex (Ye et al., ACL 2020; MIT). Turker-annotated in the "
              "describe-the-pattern methodology — the same authorship regime as "
              "KB13, and ships positive/negative example strings per regex.",
    "steps": ["Sampled for hand review only. Not imported: its custom DSL needs an "
              "adapter to our AST and oracle before any case can be built."],
    "gate": "None yet — these are candidate cards, not cases.",
    "caveat": "Read the register the instructions are written in before weighting "
              "this as KB13-grade; much of it is block-capital spec prose. It also "
              "deepens `global`, already ~95% of the corpus.",
  },
  "regelso": {
    "origin": "regel (utopia-group). The repo is MIT, but its distinctive benchmark "
              "text is StackOverflow-derived.",
    "steps": ["Sampled for judgment only."],
    "gate": "None — presumed excluded.",
    "caveat": "CC BY-SA share-alike attaches to the part worth having, which fails "
              "the permissive-only posture. The remainder duplicates deep-regex.",
  },
  "rdjson": {
    "origin": "innovatorved/regex_dataset — a GitHub repo carrying an MIT label and "
              "the self-description “just a regex dataset”.",
    "steps": ["Sampled for content inspection."],
    "gate": "None.",
    "caveat": "Anonymous provenance on a saved-pattern export: the licence may not "
              "be the uploader's to give, and the descriptions are mostly not "
              "natural language. Recommendation is to drop it.",
  },
}


def census(cards):
    out = {}
    for tab in {c["tab"] for c in cards}:
        rows = [c for c in cards if c["tab"] == tab]
        stat = {"n": len(rows)}
        for key in ("mode", "category", "tier", "authorship"):
            vals = collections.Counter()
            for c in rows:
                v = c["f"].get(key)
                if isinstance(v, str) and v:
                    vals[v] += 1
            if vals:
                stat[key] = vals.most_common(6)
        out[tab] = stat
    return out


def build(cards, out_path):
    payload = base64.b64encode(json.dumps(cards, separators=(",", ":"))
                               .encode()).decode()
    cen = json.dumps(census(cards), separators=(",", ":"))
    tabs_html = "".join(
        f'<button class="tab" data-tab="{t}" role="tab" aria-selected="false">'
        f'<span class="tl">{html.escape(label)}</span>'
        f'<span class="tn" data-n="{t}"></span></button>'
        for t, label, _ in TABS)
    blurbs = json.dumps({t: b for t, _, b in TABS}, separators=(",", ":"))

    page = PAGE.replace("__TABS__", tabs_html).replace("__DATA__", payload)
    page = page.replace("__CENSUS__", cen).replace("__BLURBS__", blurbs)
    page = page.replace("__PROV__", json.dumps(PROV, separators=(",", ":")))
    Path(out_path).write_text(page, encoding="utf-8")
    size = len(page.encode())
    assert size < 15_000_000, f"page too large: {size}"
    print(f"{len(cards)} cards · page {size/1e6:.2f} MB -> {out_path}")
    for t, label, _ in TABS:
        print(f"   {label:<22} {sum(1 for c in cards if c['tab']==t):>5}")


PAGE = r"""<title>vimBot Corpus Room</title>
<style>
:root{
  --ground:#f6f6f3; --surface:#ffffff; --raise:#fbfbf9;
  --ink:#191b1f; --dim:#676c75; --line:#dcdcd5; --line2:#eaeae4;
  --accent:#775510; --accent-soft:#f1e8d4;
  --keep:#2c6a4a; --keep-bg:#e8f2ec; --cut:#97362f; --cut-bg:#f8e9e7;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#131519; --surface:#1b1e23; --raise:#22262c;
  --ink:#e6e7e4; --dim:#8d939d; --line:#2b2f36; --line2:#23272d;
  --accent:#d9ad57; --accent-soft:#3a3222;
  --keep:#63b48c; --keep-bg:#1a2a22; --cut:#d97a70; --cut-bg:#2c1f1e;
}}
:root[data-theme="dark"]{
  --ground:#131519; --surface:#1b1e23; --raise:#22262c;
  --ink:#e6e7e4; --dim:#8d939d; --line:#2b2f36; --line2:#23272d;
  --accent:#d9ad57; --accent-soft:#3a3222;
  --keep:#63b48c; --keep-bg:#1a2a22; --cut:#d97a70; --cut-bg:#2c1f1e;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
.wrap{max-width:78rem;margin:0 auto;padding:0 1.5rem}
.lab{font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--dim);font-weight:600}

header{border-bottom:1px solid var(--line);background:var(--surface)}
header .wrap{padding-top:1.6rem;padding-bottom:1.1rem}
h1{font-size:1.5rem;font-weight:700;letter-spacing:-.015em;margin:.15rem 0 .3rem}
.lede{color:var(--dim);max-width:60ch;margin:0;font-size:.94rem}

.bar{position:sticky;top:0;z-index:9;background:var(--surface);
  border-bottom:1px solid var(--line)}
.bar .wrap{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;
  padding-top:.55rem;padding-bottom:.55rem}
.tabs{display:flex;gap:.25rem;flex-wrap:wrap;flex:1 1 auto}
.tab{display:inline-flex;gap:.4rem;align-items:baseline;cursor:pointer;
  border:1px solid transparent;background:none;color:var(--dim);
  font:inherit;font-size:.86rem;padding:.3rem .6rem;border-radius:.4rem}
.tab:hover{background:var(--raise);color:var(--ink)}
.tab[aria-selected="true"]{background:var(--accent-soft);color:var(--accent);
  border-color:var(--accent);font-weight:650}
.tab .tn{font-variant-numeric:tabular-nums;font-size:.76rem;opacity:.8}
.tab:focus-visible,input:focus-visible,.chip:focus-visible,.pg:focus-visible
  {outline:2px solid var(--accent);outline-offset:2px}
input[type=search]{font:inherit;font-size:.86rem;padding:.32rem .6rem;
  border:1px solid var(--line);border-radius:.4rem;background:var(--ground);
  color:inherit;min-width:15rem}

.panel{background:var(--surface);border:1px solid var(--line);
  border-radius:.6rem;padding:1rem 1.1rem;margin:1.1rem 0}
.blurb{color:var(--dim);font-size:.9rem;margin:.35rem 0 0;max-width:70ch}
.stats{display:flex;gap:2rem;flex-wrap:wrap;margin-top:.9rem}
.stat{min-width:11rem}
.stat .v{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;
  letter-spacing:-.01em}
.comp{margin-top:.5rem;display:flex;height:.5rem;border-radius:.25rem;
  overflow:hidden;background:var(--line2)}
.comp i{display:block;height:100%}
.legend{display:flex;gap:.9rem;flex-wrap:wrap;margin-top:.4rem;
  font-size:.78rem;color:var(--dim)}
.legend b{font-weight:600;color:var(--ink)}
.legend s{display:inline-block;width:.55rem;height:.55rem;border-radius:.15rem;
  margin-right:.3rem;text-decoration:none}

details.prov{margin-top:1rem;border-top:1px solid var(--line2);padding-top:.7rem}
details.prov summary{cursor:pointer;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--accent);font-weight:600;
  list-style:none;display:inline-flex;gap:.4rem;align-items:center}
details.prov summary::-webkit-details-marker{display:none}
details.prov summary::after{content:"▸";transition:transform .15s}
details.prov[open] summary::after{transform:rotate(90deg)}
@media (prefers-reduced-motion:reduce){details.prov summary::after{transition:none}}
details.prov summary:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.provbody{display:grid;grid-template-columns:8.5rem 1fr;gap:.55rem 1.1rem;
  margin-top:.8rem;font-size:.88rem;max-width:76ch}
@media(max-width:40rem){.provbody{grid-template-columns:1fr;gap:.2rem .5rem}}
.provbody dt{color:var(--dim);font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;font-weight:600;padding-top:.18rem}
.provbody dd{margin:0;line-height:1.6}
.provbody ol{margin:0;padding-left:1.15rem}
.provbody ol li{margin-bottom:.3rem;line-height:1.55}
.provbody ol li::marker{color:var(--accent);font-variant-numeric:tabular-nums}
.provbody .warn{color:var(--cut)}
.chips{display:flex;gap:.3rem;flex-wrap:wrap;margin:.9rem 0 .2rem}
.chip{cursor:pointer;font:inherit;font-size:.78rem;padding:.16rem .55rem;
  border:1px solid var(--line);border-radius:1rem;background:var(--surface);
  color:var(--dim)}
.chip:hover{color:var(--ink)}
.chip[data-on="1"]{border-color:var(--accent);color:var(--accent);
  background:var(--accent-soft);font-weight:600}
.tally{color:var(--dim);font-size:.84rem;margin:.7rem 0}

.card{background:var(--surface);border:1px solid var(--line);
  border-radius:.55rem;padding:.75rem .9rem;margin:.55rem 0}
.chead{display:flex;gap:.45rem;align-items:baseline;flex-wrap:wrap}
.ref{font:600 .78rem/1 var(--mono);color:var(--accent);letter-spacing:.02em}
.bdg{font-size:.72rem;color:var(--dim);border:1px solid var(--line);
  border-radius:.9rem;padding:.05rem .45rem}
.ins{margin:.45rem 0 .1rem;font-weight:600;line-height:1.45}
.ins+.ins{font-weight:400;color:var(--dim);font-size:.9rem}
.cmd{font:.82rem/1.5 var(--mono);background:var(--raise);color:var(--ink);
  border:1px solid var(--line2);border-radius:.35rem;padding:.35rem .55rem;
  margin:.45rem 0;overflow-x:auto;white-space:pre}
.panes{display:grid;grid-template-columns:1fr 1fr;gap:.6rem;margin:.5rem 0}
.panes.one{grid-template-columns:1fr}
@media(max-width:44rem){.panes{grid-template-columns:1fr}}
.note{margin:.45rem 0 0;font-size:.79rem;line-height:1.5;color:var(--dim);
  border-left:2px solid var(--accent);padding-left:.6rem}
.pane{min-width:0}
.pane pre{margin:.2rem 0 0;font:.8rem/1.5 var(--mono);background:var(--raise);
  border:1px solid var(--line2);border-radius:.35rem;padding:.4rem .1rem;
  overflow-x:auto}
.pane pre em{display:block;padding:0 .5rem;font-style:normal;white-space:pre}
.pane pre em.cut{background:var(--cut-bg);color:var(--cut);
  text-decoration:line-through;text-decoration-thickness:1px}
.pane pre em.kp{background:var(--keep-bg);color:var(--keep)}
.meta{color:var(--dim);font-size:.79rem;margin-top:.45rem;line-height:1.5;
  word-break:break-word}
.pager{display:flex;gap:.4rem;align-items:center;margin:1rem 0 3rem}
.pg{font:inherit;font-size:.82rem;padding:.25rem .7rem;border-radius:.4rem;
  border:1px solid var(--line);background:var(--surface);color:var(--ink);
  cursor:pointer}
.pg[disabled]{opacity:.4;cursor:default}
.pgi{color:var(--dim);font-size:.82rem;font-variant-numeric:tabular-nums}
.empty{color:var(--dim);padding:2rem 0}
</style>

<header><div class="wrap">
  <p class="lab">vimBot dataset</p>
  <h1>Corpus Room</h1>
  <p class="lede">Every corpus in the project, side by side — what ships, what
  was cut, what could be imported. Each card carries its licence and
  provenance; where a command filters lines rather than rewriting them, the
  source buffer marks which lines the gold command keeps and which it cuts.</p>
</div></header>

<div class="bar"><div class="wrap">
  <div class="tabs" role="tablist">__TABS__</div>
  <input id="q" type="search" placeholder="search instructions, patterns, commands…"
         aria-label="Search cards">
</div></div>

<div class="wrap">
  <div class="panel">
    <p class="lab" id="cName"></p>
    <p class="blurb" id="cBlurb"></p>
    <div class="stats" id="cStats"></div>
    <details class="prov" id="cProv"><summary>Where it came from</summary>
      <dl class="provbody" id="cProvBody"></dl></details>
  </div>
  <div class="chips" id="chips"></div>
  <p class="tally" id="tally"></p>
  <div id="cards"></div>
  <div class="pager" id="pager"></div>
</div>

<script>
const DATA=JSON.parse(new TextDecoder().decode(
  Uint8Array.from(atob("__DATA__"),c=>c.charCodeAt(0))));
const CENSUS=__CENSUS__, BLURBS=__BLURBS__, PROV=__PROV__;
const PER=40;
let tab=DATA.length?DATA[0].tab:"kb13", page=0, q="", on={};

const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
const el=id=>document.getElementById(id);

for(const b of document.querySelectorAll(".tab")){
  const n=DATA.filter(c=>c.tab===b.dataset.tab).length;
  b.querySelector(".tn").textContent=n;
  b.onclick=()=>{tab=b.dataset.tab;page=0;on={};render();};
}
el("q").oninput=e=>{q=e.target.value.toLowerCase();page=0;render();};

function facetsFor(rows){
  const f={};
  for(const r of rows) for(const k in r.f){
    const v=r.f[k]; (f[k]=f[k]||new Map());
    for(const x of (Array.isArray(v)?v:[v])) if(x) f[k].set(x,(f[k].get(x)||0)+1);
  }
  return f;
}
function passes(r){
  for(const k in on){
    if(!on[k].size) continue;
    const v=r.f[k], vs=Array.isArray(v)?v:[v];
    if(!vs.some(x=>on[k].has(x))) return false;
  }
  if(q){
    const hay=(r.ins.join(" ")+" "+r.cmd+" "+r.meta+" "+r.ref).toLowerCase();
    if(!hay.includes(q)) return false;
  }
  return true;
}

const COMPCOL=["var(--accent)","var(--keep)","var(--dim)","var(--cut)",
               "var(--line)","var(--accent-soft)"];

function stats(){
  const s=CENSUS[tab]||{n:0};
  el("cName").textContent=document.querySelector(`.tab[data-tab="${tab}"] .tl`).textContent;
  el("cBlurb").textContent=BLURBS[tab]||"";
  let h=`<div class="stat"><p class="lab">cases</p><p class="v">${s.n}</p></div>`;
  for(const key of ["category","mode","tier","authorship"]){
    if(!s[key]) continue;
    const tot=s[key].reduce((a,[,n])=>a+n,0);
    h+=`<div class="stat"><p class="lab">${key}</p>`+
       `<p class="v">${s[key][0][0]} ${Math.round(100*s[key][0][1]/tot)}%</p>`+
       `<div class="comp">`+s[key].map(([,n],i)=>
         `<i style="width:${100*n/tot}%;background:${COMPCOL[i%6]}"></i>`).join("")+
       `</div><div class="legend">`+s[key].map(([k,n],i)=>
         `<span><s style="background:${COMPCOL[i%6]}"></s><b>${esc(k)}</b> ${n}</span>`).join("")+
       `</div></div>`;
  }
  el("cStats").innerHTML=h;

  const p=PROV[tab];
  el("cProv").style.display=p?"":"none";
  if(p) el("cProvBody").innerHTML=
    `<dt>Origin</dt><dd>${esc(p.origin)}</dd>`+
    `<dt>How a case was made</dt><dd><ol>`+
      p.steps.map(s=>`<li>${esc(s)}</li>`).join("")+`</ol></dd>`+
    `<dt>What gated it</dt><dd>${esc(p.gate)}</dd>`+
    `<dt>Known limits</dt><dd class="warn">${esc(p.caveat)}</dd>`;
}

function chips(rows){
  const f=facetsFor(rows); let h="";
  for(const k of Object.keys(f).sort()){
    const vals=[...f[k].entries()].sort((a,b)=>b[1]-a[1]);
    if(vals.length<2) continue;
    h+=`<span class="lab" style="align-self:center;margin-right:.2rem">${k}</span>`;
    for(const [v,n] of vals.slice(0,14))
      h+=`<button class="chip" data-k="${esc(k)}" data-v="${esc(v)}" `+
         `data-on="${on[k]&&on[k].has(v)?1:0}">${esc(v)} ${n}</button>`;
  }
  el("chips").innerHTML=h;
  for(const c of el("chips").querySelectorAll(".chip")) c.onclick=()=>{
    const k=c.dataset.k,v=c.dataset.v;
    on[k]=on[k]||new Set();
    on[k].has(v)?on[k].delete(v):on[k].add(v);
    page=0;render();
  };
}

function pane(label,text,keep){
  let body;
  if(keep) body=text.split("\n").map((l,i)=>
    `<em class="${keep[i]?"kp":"cut"}">${esc(l)||"&nbsp;"}</em>`).join("");
  else body=text.split("\n").map(l=>`<em>${esc(l)||"&nbsp;"}</em>`).join("");
  return `<div class="pane"><p class="lab">${label}</p><pre>${body}</pre></div>`;
}

function render(){
  for(const b of document.querySelectorAll(".tab"))
    b.setAttribute("aria-selected",b.dataset.tab===tab);
  const all=DATA.filter(c=>c.tab===tab);
  stats(); chips(all);
  const rows=all.filter(passes);
  el("tally").textContent=`${rows.length} of ${all.length} shown`;
  const pages=Math.max(1,Math.ceil(rows.length/PER));
  page=Math.min(page,pages-1);
  const slice=rows.slice(page*PER,(page+1)*PER);
  el("cards").innerHTML = slice.length ? slice.map(c=>`
    <article class="card">
      <div class="chead"><span class="ref">${esc(c.ref)}</span>
        ${c.badges.filter(Boolean).map(b=>`<span class="bdg">${esc(b)}</span>`).join("")}
      </div>
      ${c.ins.map(i=>`<p class="ins">${esc(i)}</p>`).join("")}
      ${c.cmdlbl?`<p class="lab">${esc(c.cmdlbl)}</p>`:""}
      ${c.cmd?`<div class="cmd">${esc(c.cmd)}</div>`:""}
      ${c.before?`<div class="panes${c.after?"":" one"}">
        ${pane(c.lbl||(c.keep?"buffer — kept / cut":"buffer before"),c.before,c.keep)}
        ${c.after?pane("result",c.after,null):""}</div>`:""}
      ${c.note?`<p class="note">${esc(c.note)}</p>`:""}
      <p class="meta">${esc(c.meta)}</p>
    </article>`).join("")
    : `<p class="empty">No cards match these filters.</p>`;
  el("pager").innerHTML =
    `<button class="pg" id="prev" ${page?"":"disabled"}>Previous</button>`+
    `<button class="pg" id="next" ${page<pages-1?"":"disabled"}>Next</button>`+
    `<span class="pgi">page ${page+1} of ${pages}</span>`;
  const p=el("prev"),n=el("next");
  if(p)p.onclick=()=>{page--;render();window.scrollTo({top:0});};
  if(n)n.onclick=()=>{page++;render();window.scrollTo({top:0});};
}
render();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    cards = (kb13_cards() + sregex_v3_cards() + emulator_cards() + candidate_cards() + v1_cards()
             + v2_sample_cards(rng) + a3_cards(rng)
             + quarantine_cards() + seed_cards() + import_cards())
    build(cards, args.out)


if __name__ == "__main__":
    main()
