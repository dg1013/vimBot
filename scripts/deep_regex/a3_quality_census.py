"""A3 quality census: how much of the machine-seeded corpus can the
deterministic renderer verbalize, and how *readable* is the result?

The A3 pilot (149 cards) established that regenerated English is
anchor/adjacency-honest. It did not establish that the English is
idiomatic. Spot-reading the pilot shows two regimes:

  good  "Delete lines containing at least 4 occurrences of 'c'"
  bad   "Delete lines that neither are exactly a letter nor are exactly
         an uppercase letter nor are exactly 'dog'"

Both are honest; only the first is something a person would type. The
renderer's `TooComplex` budget gates *verbalizability*, not readability,
so coverage alone cannot decide the A3 verdict.

This script measures both axes over the live corpus so a complexity cap
can be chosen with numbers:

  * coverage      — rendered vs degenerate (renderer returns None)
  * stiltedness   — named surface constructions that mark a rendering as
                    a parse tree read aloud rather than a request
  * tier          — clean / borderline / stilted, case-weighted

Usage:
    conda run -n vimBot python scripts/deep_regex/a3_quality_census.py
    conda run -n vimBot python scripts/deep_regex/a3_quality_census.py --json out.json
    conda run -n vimBot python scripts/deep_regex/a3_quality_census.py \
        --manifest data/fixes/phase_a3_admission.json --per-tier 25

`--manifest` writes review-artifact cards (the Fixes-tab schema) so the
tier boundaries can be approved by looking at real renderings rather than
at thresholds — the owner's standing rule that every fix reaches the
artifact before or with application.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from ast2nl import render_variants  # noqa: E402

SPLITS = ("train", "val", "test")

# Surface markers of a rendering that has stopped being English. Each is a
# construction the renderer emits when it has to spell out AST (abstract syntax tree) structure
# it cannot compress into an idiom.
STILT_RULES = (
    # nested alternation under repetition: "parts, each either X or Y"
    ("parts_each", re.compile(r"\bparts, each\b")),
    # multi-way negation: "neither ... nor ... nor ..."
    ("neither_nor", re.compile(r"\bneither\b.*\bnor\b.*\bnor\b")),
    # colon-list disjunction: "starting with any of: 'dog', a lowercase..."
    ("any_of_list", re.compile(r"\bany of:")),
    # stacked positional slack: two or more "then anywhere later"
    ("multi_later", re.compile(r"(anywhere later.*){2,}")),
    # "any number of" plus a nested body — the vacuous-star tell
    ("any_number_nested", re.compile(r"\bany number of\b.*\b(either|each)\b")),
    # triple-or-longer conjunction chains
    ("long_chain", re.compile(r"(\b(and|or)\b.*){3,}")),
)

CLEAN_MAX_WORDS = 12
BORDERLINE_MAX_WORDS = 18


def stilt_flags(phrase):
    return [name for name, rx in STILT_RULES if rx.search(phrase)]


def tier_of(phrase):
    """clean | borderline | stilted — the proposed A3 admission ladder."""
    flags = stilt_flags(phrase)
    words = len(phrase.split())
    if flags:
        return "stilted", flags
    if words <= CLEAN_MAX_WORDS:
        return "clean", flags
    if words <= BORDERLINE_MAX_WORDS:
        return "borderline", flags
    return "stilted", ["too_long"]


def load_population(repo):
    """Unique source_regex over the live splits, with case counts, the
    corpus tier each pattern came from (KB13 (the Kushman and Barzilay 2013 corpus) = human-authored), and one
    representative live case for artifact cards."""
    pop = {}
    for split in SPLITS:
        path = repo / "data" / "splits" / f"{split}.jsonl"
        if not path.exists():
            continue
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rx = row.get("source_regex")
            if not rx:
                continue
            entry = pop.setdefault(rx, {"cases": 0, "corpora": set(),
                                        "tiers": collections.Counter(),
                                        "example": row})
            entry["cases"] += 1
            entry["corpora"].update(row.get("source_corpus") or [])
            entry["tiers"][row.get("tier")] += 1
    return pop


def write_manifest(path, rows, per_tier, seed):
    """Fixes-tab cards, per the phase_a3_pilot.json convention: the
    instruction goes in `command` and a labelled note in `buffer_after`."""
    rng = random.Random(seed)
    cards = []
    for tier in ("clean", "borderline", "stilted"):
        pool = [r for r in rows if r.get("tier") == tier]
        for r in rng.sample(pool, min(per_tier, len(pool))):
            ex = r["example"]
            flags = ", ".join(r["flags"]) if r["flags"] else "none"
            cards.append({
                "id": f"a3adm/{r['regex']}",
                "phase": f"phaseA3-adm-{tier}",
                "kind": "regen-admission",
                "summary": (f"{tier}; {r['words']} words; flags: {flags}; "
                            f"{r['cases']} live case(s) on this pattern"),
                "old": {
                    "command": ex["instruction"],
                    "buffer_after": ("current corpus instruction — "
                                     f"source: {'/'.join(ex.get('source_corpus') or ['?'])}"),
                },
                "new": {
                    "command": " | ".join(r["variants"]),
                    "buffer_after": (f"A3 rendering · tier {tier} · "
                                     f"flags: {flags} · words: {r['words']}"),
                },
                "case": {
                    "instruction": ex["instruction"],
                    "gold_command": ex["gold_command"],
                    "buffer_before": ex["buffer_before"],
                    "buffer_after": ex["buffer_after"],
                    "tier": ex.get("tier", ""),
                },
            })
    Path(path).write_text(json.dumps(cards, indent=1))
    print(f"\nwrote {len(cards)} admission cards -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the full census to this path")
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--samples", type=int, default=6,
                    help="example renderings to print per tier")
    ap.add_argument("--manifest", help="write review-artifact cards here")
    ap.add_argument("--per-tier", type=int, default=25,
                    help="cards per tier for --manifest")
    args = ap.parse_args()

    pop = load_population(REPO)
    if not pop:
        sys.exit("no live splits found under data/splits/")

    rows = []
    for rx, meta in sorted(pop.items()):
        rng = random.Random(args.seed ^ (hash(rx) & 0xFFFFFFFF))
        try:
            variants = render_variants(rx, rng, k=2)
        except Exception as exc:                      # renderer crash
            rows.append({"regex": rx, "cases": meta["cases"],
                         "outcome": "error", "detail": repr(exc)[:120],
                         "human": "KB13" in meta["corpora"]})
            continue
        if not variants:
            rows.append({"regex": rx, "cases": meta["cases"],
                         "outcome": "degenerate",
                         "human": "KB13" in meta["corpora"]})
            continue
        # judge on the primary variant; the ladder must hold for the text
        # that would actually ship as the instruction
        tier, flags = tier_of(variants[0])
        rows.append({"regex": rx, "cases": meta["cases"], "outcome": "rendered",
                     "tier": tier, "flags": flags, "text": variants[0],
                     "variants": variants, "words": len(variants[0].split()),
                     "human": "KB13" in meta["corpora"],
                     "example": meta["example"]})

    total_pat = len(rows)
    total_case = sum(r["cases"] for r in rows)

    def agg(pred):
        pats = [r for r in rows if pred(r)]
        return len(pats), sum(r["cases"] for r in pats)

    print(f"A3 quality census — {total_pat} unique patterns / "
          f"{total_case} live cases\n")

    print("coverage")
    for outcome in ("rendered", "degenerate", "error"):
        p, c = agg(lambda r, o=outcome: r["outcome"] == o)
        if p:
            print(f"  {outcome:<12} {p:>6} patterns ({p/total_pat:5.1%})  "
                  f"{c:>6} cases ({c/total_case:5.1%})")

    print("\nreadability tier (rendered only)")
    for tier in ("clean", "borderline", "stilted"):
        p, c = agg(lambda r, t=tier: r.get("tier") == t)
        print(f"  {tier:<12} {p:>6} patterns ({p/total_pat:5.1%})  "
              f"{c:>6} cases ({c/total_case:5.1%})")

    print("\nstiltedness flags (patterns may carry several)")
    flag_counts = collections.Counter()
    flag_cases = collections.Counter()
    for r in rows:
        for f in r.get("flags", []):
            flag_counts[f] += 1
            flag_cases[f] += r["cases"]
    for f, n in flag_counts.most_common():
        print(f"  {f:<20} {n:>6} patterns  {flag_cases[f]:>6} cases")

    print("\nadmission scenarios (case-weighted, machine-seeded only)")
    machine = [r for r in rows if not r["human"]]
    mtotal = sum(r["cases"] for r in machine)
    for label, keep in (
            ("clean only", lambda r: r.get("tier") == "clean"),
            ("clean+borderline", lambda r: r.get("tier") in ("clean", "borderline")),
            ("all rendered (72% path)", lambda r: r["outcome"] == "rendered")):
        c = sum(r["cases"] for r in machine if keep(r))
        print(f"  {label:<26} {c:>6} cases  ({c/mtotal:5.1%} of machine-seeded)")

    rng = random.Random(args.seed)
    for tier in ("clean", "borderline", "stilted"):
        pool = [r for r in rows if r.get("tier") == tier]
        if not pool:
            continue
        print(f"\n--- {tier} samples ---")
        for r in rng.sample(pool, min(args.samples, len(pool))):
            flags = f"  [{','.join(r['flags'])}]" if r["flags"] else ""
            print(f"  {r['regex'][:52]:<54} {r['text'][:104]}{flags}")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        slim = [{k: v for k, v in r.items() if k != "example"} for r in rows]
        out.write_text(json.dumps(slim, indent=1))
        print(f"\nwrote {out}")

    if args.manifest:
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        write_manifest(args.manifest, rows, args.per_tier, args.seed)


if __name__ == "__main__":
    main()
