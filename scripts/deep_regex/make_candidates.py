"""Turn validated conversions into §5 candidate scenarios.

Per unique usable regex (never-match/trivial/stuck excluded), merged across
the three corpora:

  operation    deterministic hash split — half "delete matching lines",
               half "keep only matching lines". The conversion's (exec,
               pol) selects :g or :v so the command does the right thing
               either way.
  instruction  template over each source NL (natural language): "Delete {nl}" / "Keep only
               {nl}", first letter lowercased. Synth+Turk descriptions of
               the same regex become paraphrases of one scenario.
  buffer       up to 3 matching and 3 non-matching non-empty probe lines,
               classified by the span oracle on the ORIGINAL DSL (domain-specific language) pattern
               and deterministically shuffled. Scenarios lacking either
               side are skipped (reported).
  buffer_after oracle-side prediction (surviving lines). generate_data.py
               then re-derives it in real vim — any disagreement between
               oracle and converted command is caught there as `mismatch`.
  tier         T1 literal-only pattern; T3 alternation / counted repeat /
               lookahead / 3+ feature families; T2 otherwise.

Output: data/candidates_deepregex/deepregex_NNN.jsonl (1,000 scenarios per
chunk) with source_* provenance fields on every candidate.
"""
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from build_conversions import CORPORA, CORPUS, probes  # noqa: E402
from witnesses import build_buffer  # noqa: E402

OUTDIR = REPO / "data" / "candidates_deepregex"
CHUNK = 1000


def load_pairs():
    """regex -> {"nls": [...], "sources": [(corpus, line_no), ...]}"""
    merged = defaultdict(lambda: {"nls": [], "sources": []})
    for name in CORPORA:
        srcs = (CORPUS / name / "src.txt").read_text(
            encoding="utf-8").splitlines()
        targs = (CORPUS / name / "targ.txt").read_text(
            encoding="utf-8").splitlines()
        for i, (s, t) in enumerate(zip(srcs, targs), 1):
            s, t = s.strip(), t.strip()
            if not t:
                continue
            entry = merged[t]
            if s and s not in entry["nls"]:
                entry["nls"].append(s)
            entry["sources"].append((name, i))
    return merged


def tier_for(pattern):
    feats = 0
    if re.search(r"(?<!\\)\[", pattern):
        feats += 1
    if re.search(r"[*+]", pattern):
        feats += 1
    if "\\b" in pattern:
        feats += 1
    if re.search(r"\{\d", pattern) or "|" in pattern:
        return "T3"
    if "&" in pattern or "~" in pattern:
        feats += 1
    if feats == 0:
        return "T1"
    return "T3" if feats >= 3 else "T2"


def instruction_for(nl, op):
    body = nl[0].lower() + nl[1:] if nl and nl[1:2].islower() else nl
    return f"Delete {body}" if op == "delete" else f"Keep only {body}"


def main():
    conv = json.loads((CORPUS / "conversions.json").read_text(
        encoding="utf-8"))
    pairs = load_pairs()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for old in OUTDIR.glob("deepregex_*.jsonl"):
        old.unlink()

    made = skipped_sides = skipped_conv = 0
    chunk_rows = []
    chunk_no = 0

    def flush():
        nonlocal chunk_rows, chunk_no
        if not chunk_rows:
            return
        path = OUTDIR / f"deepregex_{chunk_no:03d}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in chunk_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        chunk_rows = []
        chunk_no += 1

    for t in sorted(pairs):
        c = conv.get(t)
        if not c or c["k"] == 2 or c["n"] in ("degenerate", "trivial") \
                or not c.get("exec"):
            skipped_conv += 1
            continue

        h = hashlib.sha1(t.encode()).hexdigest()
        rng = random.Random(h)

        buf = build_buffer(t, rng, probe_lines=probes(t), pol=c["pol"])
        if buf is None:
            skipped_sides += 1
            continue
        lines, truth = buf["lines"], buf["truth"]
        witness_kinds = sorted({lab for lab in buf["labels"].values()
                                if lab not in ("probe", "member", "generic")})

        op = "delete" if int(h, 16) % 2 == 0 else "keep"
        pol = c["pol"]
        if op == "delete":
            cmd = f"g/{c['exec']}/d" if pol == "g" else f"v/{c['exec']}/d"
            survivors = [ln for ln in lines if not truth[ln]]
        else:
            cmd = f"v/{c['exec']}/d" if pol == "g" else f"g/{c['exec']}/d"
            survivors = [ln for ln in lines if truth[ln]]

        entry = pairs[t]
        chunk_rows.append({
            "category": "global",
            "tier": tier_for(t),
            "mode": "ex",
            "domain": "probe",
            "buffer_before": "\n".join(lines),
            "cursor": [1, 1],
            "command": cmd,
            "buffer_after": "\n".join(survivors),
            "instructions": [instruction_for(nl, op)
                             for nl in entry["nls"][:3]],
            "source": "deep-regex",
            "source_corpus": sorted({s[0] for s in entry["sources"]}),
            "source_index": [f"{name}:{i}"
                             for name, i in entry["sources"][:6]],
            "source_regex": t,
            "source_license": "MIT",
            "buffer_witnesses": witness_kinds,
        })
        made += 1
        if len(chunk_rows) >= CHUNK:
            flush()
    flush()

    print(f"scenarios written {made} across {chunk_no} chunk(s); "
          f"skipped: no-conversion/degenerate {skipped_conv}, "
          f"one-sided probes {skipped_sides}")


if __name__ == "__main__":
    main()
