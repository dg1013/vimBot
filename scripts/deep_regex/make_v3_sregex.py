"""StructuredRegex -> v3 cases: conversion prototype.

Runs the recovered StructuredRegex sample through the full existing
pipeline — adapter -> AST (abstract syntax tree) -> vim conversion -> buffer -> sandboxed vim ->
verifier triangle — and writes §5 cases. Nothing here is new machinery
except the adapter (`sregex_adapter.py`) and the instruction rewriter
below; the simplifier, converter, oracle and executor are used unchanged.

PROVENANCE WARNING. StructuredRegex is **not vendored in this repo (repository)**. The
input is `data/inspector_imports.json` — 460 distinct patterns recovered
from a previously published artifact, with refs running contiguously
SR-0001..SR-0718, so it is not a verified random sample of the ~3.5k
corpus. Everything this script produces is a *preview of what an import
would look like*, not an import. A real import fetches and vendors the
corpus under §14c first.

Two things make this corpus fit unusually well:

  * Its patterns are whole-string matchers and its annotators were
    describing whole strings, so the anchor defect that invalidated 57% of
    v2 train cannot arise by construction.
  * It ships positive and negative example strings per pattern — human-
    authored near-misses. Those become the buffer directly, so the buffer
    is discriminative *and* readable, unlike our synthetic probe gibberish.

Instructions are rewritten to participle frames ("beginning with…",
"consisting of…") because those read the same in singular and plural, so
one rewrite serves both the delete and keep phrasings without conjugation.
Any instruction whose frame is not recognised is dropped rather than
force-fitted; coverage is reported, not assumed.

Usage:
    conda run -n vimBot python scripts/deep_regex/make_v3_sregex.py
    conda run -n vimBot python scripts/deep_regex/make_v3_sregex.py --json out.jsonl
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from executor import run_vim  # noqa: E402
from ast2nl import render_variants  # noqa: E402
from build_conversions import bound_risk, probes  # noqa: E402
from dsl2vim import generate  # noqa: E402
from embedded_conj import render_embedded  # noqa: E402
from dsl_ast import matcher, serialize  # noqa: E402
from garble_screen import anchor_mismatch, classify  # noqa: E402
from make_candidates import tier_for  # noqa: E402
from py_impl import py_matcher  # noqa: E402
from witnesses import build_buffer  # noqa: E402
from sregex_adapter import examples, parse_sr  # noqa: E402

OVERRIDES_PATH = REPO / "data" / "v3" / "sregex_overrides.json"


def esc_slash(pattern):
    """Escape '/' for a /-delimited :g//:v command — but only OUTSIDE
    collections. Vim's delimiter scan tracks []-collections, where a raw
    '/' is literal and '\\/' corrupts the class (the backslash becomes a
    collection member / range endpoint — measured 2026-08-17: the blanket
    escape turned [!-/…] into [!-\\…], swallowing digits and uppercase).
    """
    out, in_class, i = [], False, 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            out.append(pattern[i:i + 2])
            i += 2
            continue
        if not in_class and c == "[":
            in_class = True
            j = i + 1
            if j < len(pattern) and pattern[j] == "^":
                j += 1
            # a ']' immediately after '[' or '[^' is a literal member
            out.append(pattern[i:j])
            i = j
            if i < len(pattern) and pattern[i] == "]":
                out.append("]")
                i += 1
            continue
        if in_class and c == "]":
            in_class = False
        elif not in_class and c == "/":
            out.append("\\/")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)

MAX_LINE = 60          # keep buffers readable; SR (StructuredRegex) has some very long strings
MAX_LINES = 8

# Frames we can turn into a number-neutral participle. Order matters: the
# longer, more specific patterns must be tried first.
FRAMES = [
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+that\s+"
     r"(?:should\s+|must\s+|will\s+)?(?:begins?|starts?)\s+with\s+", "beginning with "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+that\s+"
     r"(?:should\s+|must\s+|will\s+)?ends?\s+with\s+", "ending with "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+that\s+"
     r"(?:should\s+|must\s+|will\s+)?(?:consists?\s+of|is\s+made\s+of)\s+",
     "consisting of "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+that\s+"
     r"(?:should\s+|must\s+|will\s+)?(?:contains?|has|have|includes?)\s+",
     "containing "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+"
     r"(?:beginning|starting)\s+with\s+", "beginning with "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+ending\s+with\s+",
     "ending with "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+"
     r"(?:consisting|comprised|composed|made\s+up)\s+of\s+", "consisting of "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+containing\s+",
     "containing "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+"
     r"(?:should\s+|must\s+|will\s+)?(?:begins?|starts?)\s+with\s+",
     "beginning with "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+"
     r"(?:should\s+|must\s+|will\s+)?ends?\s+with\s+", "ending with "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+"
     r"(?:should\s+|must\s+|will\s+)?(?:consists?\s+of)\s+", "consisting of "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+"
     r"(?:should\s+|must\s+|will\s+)?(?:contains?|has|have|includes?)\s+",
     "containing "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+of\s+",
     "consisting of "),
    (r"^(?:a|the|this|each|every)\s+(?:single\s+)?strings?\s+with\s+",
     "containing "),
]

# Registers we refuse outright: multi-part format specs, meta-commentary.
REJECT = re.compile(
    r"^(?:a\s+list\s+of|there\s+(?:is|are)|this\s+is\s+a|this\s+pattern|"
    r"the\s+(?:regex|pattern|expression))", re.I)


def to_line_instruction(nl):
    """SR's whole-string prose -> a participle clause about lines, or None.

    Returns the clause without the operation verb, so one rewrite serves
    both "Delete every line <clause>" and "Keep only lines <clause>".
    """
    s = " ".join(nl.split()).rstrip(".")
    if REJECT.match(s):
        return None
    for rx, frame in FRAMES:
        m = re.match(rx, s, re.I)
        if m:
            rest = s[m.end():].strip()
            if not rest:
                return None
            return frame + rest
    return None


def phrase(clause, op):
    return (f"Delete every line {clause}." if op == "delete"
            else f"Keep only lines {clause}.")


def instr_of(cl, op):
    """cl is a clause string, or an ('a3', 'lines …') full noun phrase."""
    if isinstance(cl, tuple):
        p = cl[1]
        return f"Delete {p}." if op == "delete" else f"Keep only {p}."
    return phrase(cl, op)


def usable_lines(pos, neg):
    """SR example strings -> buffer lines, keeping the partition balanced."""
    def clean(xs):
        out = []
        for s in xs:
            if not s or len(s) > MAX_LINE or "\n" in s:
                continue
            if s not in out:
                out.append(s)
        return out
    return clean(pos), clean(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "data" / "v3" / "sregex_cases.jsonl"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cards = [c for c in json.loads(
        (REPO / "data" / "inspector_imports.json").read_text())
        if c["tab"] == "sregex"]

    overrides = {}
    if OVERRIDES_PATH.exists():
        overrides = {k: v for k, v in
                     json.loads(OVERRIDES_PATH.read_text()).items()
                     if not k.startswith("_")}

    by_pattern = collections.OrderedDict()
    for c in cards:
        e = by_pattern.setdefault(c["command"], {"nls": [], "card": c,
                                                 "refs": []})
        e["refs"].append(c["ref"])
        nl = (c.get("instructions") or [""])[0]
        if nl and nl not in e["nls"]:
            e["nls"].append(nl)

    rows, skip, stat = [], collections.Counter(), collections.Counter()
    rejects = []
    n_scen = 0

    def convert_preview(ast, nf, pos, neg, h):
        """Best-available vim command for a rejected pattern, validated in
        the sandbox against the example strings — the claim 'this command
        works' is executed, never asserted. '/' is escaped only outside
        collections (esc_slash)."""
        g = generate(nf, exact_boundary=bound_risk(nf))
        exec_, pol = g.get("exec"), g.get("pol", "g")
        if not exec_:
            emb = render_embedded(nf, exact_boundary=bound_risk(nf))
            if emb:
                exec_, pol = emb["exec"], emb["pol"]
        if not exec_:
            return ("", "no single vim pattern — embedded and/~ under a "
                    "repeat/alternation, or variable-length conjuncts with a "
                    "variable tail (beyond suffix-pinned lookaheads too)",
                    None, None, None, None)
        shown = esc_slash(exec_)
        note = " · '/' escaped outside collections" if shown != exec_ else ""
        cmd = f"g/{shown}/d" if pol == "g" else f"v/{shown}/d"
        lines = (pos + neg)[:MAX_LINES]
        if not lines:
            return (cmd, "renders; no example strings to validate against",
                    None, None, None, None)
        random.Random(h).shuffle(lines)
        truth = {ln: matcher(ast, ln) for ln in lines}
        expected = [ln for ln in lines if not truth[ln]]
        res = run_vim("\n".join(lines), cmd, mode="ex")
        got = res["buffer_after"].split("\n")
        while got and got[-1] == "":
            got.pop()
        ok = res["was_valid"] and got == expected
        status = (f"validated in sandboxed vim · shown in delete form{note}"
                  if ok else f"renders but FAILED vim validation{note}")
        return (cmd, status, lines, [1 if truth[ln] else 0 for ln in lines],
                expected, got if res["was_valid"] else ["<vim error>"])

    def reject(bucket, note, ast, nf, src, e, h, rules=None):
        pos, neg = usable_lines(*examples(e["card"]))
        cmd, status, lines, part, expected, got = \
            convert_preview(ast, nf, pos, neg, h)
        rejects.append({
            "bucket": bucket, "note": note,
            "source_dsl": src, "normal_form": nf,
            "prose": e["nls"][:3],
            "vim_command": cmd, "vim_status": status,
            "lines": lines or [], "match": part or [],
            "expected": expected, "vim_output": got,
            "rules": sorted(rules) if rules else [],
        })

    for src, e in by_pattern.items():
        if args.limit and n_scen >= args.limit:
            break
        try:
            ast = parse_sr(src)
        except Exception:                               # noqa: BLE001
            skip["adapter-parse"] += 1
            continue
        nf = serialize(ast)
        h = hashlib.sha1(nf.encode()).hexdigest()

        # instruction sources, in trust order: owner-approved override >
        # corpus prose through the frame rewriter > A3 deterministic
        # rendering from the adapted AST. Which one fed a case is recorded
        # in instruction_provenance.
        clauses, prov = [], None
        if src in overrides:
            clauses = list(overrides[src]["clauses"])
            prov = "owner-authored"
        if not clauses:
            for nl in e["nls"]:
                cl = to_line_instruction(nl)
                if cl is None:
                    stat["instruction-frame-unrecognised"] += 1
                    continue
                clauses.append(cl)
            if clauses:
                prov = "human-rewritten"
        if not clauses:
            try:
                # SR conjunctions are denser than deep-regex shapes; double
                # the phrase budget — screens still gate the output
                phrases = render_variants(nf, random.Random(h), k=2, units=16)
            except Exception:                           # noqa: BLE001
                phrases = None
            if phrases:
                clauses = [("a3", p) for p in phrases]
                prov = "a3-generated"
                stat["a3-fallback-used"] += 1
        if not clauses:
            skip["no-usable-instruction"] += 1
            nl0 = " ".join(e["nls"][0].split()).lower() if e["nls"] else ""
            kind = ("'a list of N strings' format spec — genuinely not a "
                    "line-edit instruction" if nl0.startswith("a list") else
                    "'there is/are …' framing" if nl0.startswith("there") else
                    "meta-prose about the pattern itself"
                    if nl0.startswith(("this is", "this pattern",
                                       "the regex", "the pattern")) else
                    "frame outside the rewriter's vocabulary AND the A3 "
                    "renderer routes it degenerate/unverbalizable")
            reject("prose-refused", kind, ast, nf, src, e, h)
            continue

        pos, neg = usable_lines(*examples(e["card"]))

        g = generate(nf, exact_boundary=bound_risk(nf))
        if not g.get("exec"):
            emb = render_embedded(nf, exact_boundary=bound_risk(nf))
            if emb:
                g = {"exec": emb["exec"], "pol": emb["pol"]}
                stat["embedded-conj-rendered"] += 1
        if not g.get("exec"):
            skip["no-vim-conversion"] += 1
            reject("unconvertible",
                   "embedded and/~ beyond even suffix-pinned lookaheads — "
                   "under a repeat/alternation, or variable-length conjuncts "
                   "with a variable-length tail",
                   ast, nf, src, e, h)
            continue
        # '/' no longer refuses: esc_slash escapes it only outside
        # collections (measured 2026-08-17 — raw '/' inside [] is what vim
        # wants; the old blanket escape corrupted the class)
        exec_vim = esc_slash(g["exec"])
        if exec_vim != g["exec"]:
            stat["slash-escaped-outside-collections"] += 1
        pol = g.get("pol", "g")

        rng = random.Random(h)
        op = "delete" if int(h, 16) % 2 == 0 else "keep"

        witnesses_used = False
        if len(pos) >= 2 and len(neg) >= 2:
            lines = (pos + neg)[:MAX_LINES]
            rng.shuffle(lines)
            truth = {ln: matcher(ast, ln) for ln in lines}
            # the corpus's own labels must agree with our oracle
            if any(truth[s] is not (s in pos) for s in lines):
                skip["oracle-vs-corpus-label"] += 1
                continue
        else:
            # not enough corpus strings for a contrastive buffer: fall back
            # to the A1 witness sampler (same machinery as the KB13 (the Kushman and Barzilay 2013 corpus) build)
            buf = build_buffer(nf, rng, probe_lines=probes(nf),
                               pol=g.get("pol", "g"))
            if buf is None:
                skip["too-few-examples"] += 1
                reject("too-few-examples",
                       f"{len(pos)} positive / {len(neg)} negative usable "
                       f"strings AND the witness sampler found no buffer",
                       ast, nf, src, e, h)
                continue
            lines, truth = buf["lines"], buf["truth"]
            wlabels = sorted({lab for lab in buf["labels"].values()
                              if lab not in ("probe", "member", "generic")})
            witnesses_used = True
            stat["witness-buffer-fallback"] += 1

        keep_members = (op == "keep")
        if op == "delete":
            cmd = f"g/{exec_vim}/d" if pol == "g" else f"v/{exec_vim}/d"
        else:
            cmd = f"v/{exec_vim}/d" if pol == "g" else f"g/{exec_vim}/d"
        expected = [ln for ln in lines if truth[ln] == keep_members]
        if not expected or len(expected) == len(lines):
            skip["degenerate-result"] += 1
            continue

        res = run_vim("\n".join(lines), cmd, mode="ex")
        got = res["buffer_after"].split("\n")
        while got and got[-1] == "":
            got.pop()
        if not res["was_valid"] or got != expected:
            skip["vim-mismatch"] += 1
            continue

        f = py_matcher(nf)
        if f is not None:
            if [ln for ln in lines if f(ln) == keep_members] != expected:
                skip["python-mismatch"] += 1
                continue
            stat["triangle-full"] += 1
        else:
            stat["triangle-two-leg"] += 1

        kept = []
        for cl in clauses:
            instr = instr_of(cl, op)
            if classify(nf, instr) or anchor_mismatch(nf, instr):
                stat["instruction-screened-out"] += 1
                continue
            kept.append(instr)
        if not kept:
            skip["all-instructions-screened"] += 1
            fired = set()
            for cl in clauses:
                instr = instr_of(cl, op)
                fired |= set(classify(nf, instr) or ())
                if anchor_mismatch(nf, instr):
                    fired.add("anchor-mismatch")
            reject("screened",
                   "the pattern converts and validates, but every rewritten "
                   "phrasing tripped a garble/anchor screen",
                   ast, nf, src, e, h, rules=fired)
            continue

        n_scen += 1
        for j, instr in enumerate(kept[:3]):
            rows.append({
                "id": f"sregex-{n_scen:04d}{chr(ord('a') + j)}",
                "category": "global", "tier": tier_for(nf), "mode": "ex",
                "domain": "probe" if witnesses_used else "example-strings",
                "buffer_before": "\n".join(lines), "cursor": [1, 1],
                "gold_command": cmd,   # §5 contract key
                "buffer_after": "\n".join(expected),
                "instruction": instr,
                "source": "structured-regex", "source_corpus": ["StructuredRegex"],
                "source_index": sorted(set(e["refs"])),
                "source_nl": e["nls"][:3],
                "source_dsl": src, "normal_form": nf,
                "source_license": "MIT", "instruction_provenance": prov,
                "buffer_witnesses": (wlabels if witnesses_used
                                     else ["corpus-negative-examples"]),
                "provenance_note": ("recovered 460-pattern sample, corpus not "
                                    "vendored — preview only"),
            })

    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    rej_path = out.parent / "sregex_rejects.jsonl"
    with rej_path.open("w", encoding="utf-8") as fh:
        for r in rejects:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    buckets = collections.Counter(r["bucket"] for r in rejects)
    stats = {"patterns_in": len(by_pattern), "cards_in": len(cards),
             "scenarios": n_scen, "cases": len(rows),
             "verifier": dict(stat), "skips": dict(skip),
             "reject_buckets": dict(buckets)}
    (out.parent / "sregex_stats.json").write_text(json.dumps(stats, indent=1))

    print(f"StructuredRegex -> v3 conversion preview")
    print(f"  input            {len(by_pattern)} distinct patterns "
          f"({len(cards)} cards)")
    print(f"  scenarios built  {n_scen}")
    print(f"  cases written    {len(rows)}  -> {out}")
    print(f"  verifier:        {dict(stat)}")
    print(f"  skips:           {dict(skip)}")
    print(f"  rejects written  {len(rejects)}  -> {rej_path}")
    print(f"  reject buckets:  {dict(buckets)}")


if __name__ == "__main__":
    main()
