"""StructuredRegex DSL (domain-specific language) -> our AST (abstract syntax tree): feasibility prototype and validator.

StructuredRegex (Ye et al., ACL (Association for Computational Linguistics) 2020; MIT (Massachusetts Institute of Technology)) writes patterns as nested
function calls — `concat(repeatatleast(notcc(<num>),1),...)` — where
deep-regex writes regex-ish text. Different surface syntax, but the census
over 718 samples found only 14 operators, and every one of them maps onto a
node the deep-regex AST already has:

    concat -> seq        or -> alt          and -> and      not -> not
    repeat/repeatatleast/repeatrange/star/optional -> rep (m,n)
    const/<c> -> char    <num>/<let>/... -> class     <any> -> dot
    contain(A)   -> .*A.*        startwith(A) -> A.*
    endwith(A)   -> .*A          notcc(<X>)   -> complement class

`and` and `not` are the operators that make a regex (regular expression) DSL hard to evaluate,
and our span-DP (dynamic programming) oracle already implements exactly those (same-span
conjunction and negation) because deep-regex uses `&` and `~`. So this is a
parser, not a semantics project — nothing downstream needs to change.

The corpus ships positive and negative example strings per pattern, which
makes the adapter self-validating: our matcher must accept every positive
and reject every negative. Disagreement means the adapter is wrong (or the
DSL is not what we think), and the rate is reported rather than assumed.

Usage:
    conda run -n vimBot python scripts/deep_regex/sregex_adapter.py
    conda run -n vimBot python scripts/deep_regex/sregex_adapter.py --verbose
"""
import argparse
import collections
import json
import re
import string
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from dsl_ast import N, matcher, serialize  # noqa: E402
from garble_screen import anchor_mismatch, classify  # noqa: E402
from make_candidates import instruction_for, tier_for  # noqa: E402

PRINTABLE = frozenset(chr(c) for c in range(0x20, 0x7f))
CLASSES = {
    "<num>": frozenset(string.digits),
    "<let>": frozenset(string.ascii_letters),
    "<low>": frozenset(string.ascii_lowercase),
    "<cap>": frozenset(string.ascii_uppercase),
    "<spec>": frozenset(c for c in PRINTABLE
                        if not c.isalnum() and c != " "),
}


class SRParseError(Exception):
    pass


def _tokenize(text):
    """`<...>` is atomic — literals include '<,>' and '<.>', so commas and
    parens inside angle brackets must not split arguments."""
    toks, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif c == "<":
            j = text.find(">", i + 1)
            if j < 0:
                raise SRParseError("unterminated <")
            toks.append(("tok", text[i:j + 1]))
            i = j + 1
        elif c in "(),":
            toks.append((c, c))
            i += 1
        elif c.isdigit():
            j = i
            while j < n and text[j].isdigit():
                j += 1
            toks.append(("int", int(text[i:j])))
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (text[j].isalpha() or text[j] == "_"):
                j += 1
            toks.append(("id", text[i:j]))
            i = j
        else:
            raise SRParseError(f"unexpected {c!r} at {i}")
    return toks


def _literal(tok):
    """<LEj> -> seq of chars; <-> -> one char."""
    body = tok[1:-1]
    if body == "":
        raise SRParseError("empty token")
    if len(body) == 1:
        return N("char", val=body)
    return N("seq", [N("char", val=ch) for ch in body])


def _terminal(tok):
    if tok == "<any>":
        return N("dot")
    if tok in CLASSES:
        return N("class", val=CLASSES[tok])
    return _literal(tok)


STAR = lambda: N("rep", [N("dot")], m=0, n=None)          # noqa: E731


def parse_sr(text):
    """StructuredRegex DSL text -> deep-regex AST node."""
    toks = _tokenize(text)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None)

    def eat(kind):
        nonlocal pos
        if peek()[0] != kind:
            raise SRParseError(f"expected {kind}, got {peek()} at {pos}")
        pos += 1
        return toks[pos - 1][1]

    def expr():
        nonlocal pos
        kind, val = peek()
        if kind == "tok":
            pos += 1
            return _terminal(val)
        if kind != "id":
            raise SRParseError(f"expected operator, got {peek()}")
        op = eat("id")
        eat("(")
        args = [expr()] if peek()[0] != ")" else []
        while peek()[0] == ",":
            eat(",")
            if peek()[0] == "int":
                args.append(eat("int"))
            else:
                args.append(expr())
        eat(")")
        return build(op, args)

    def build(op, a):
        if op == "concat":
            return N("seq", list(a))
        if op == "or":
            return N("alt", list(a))
        if op == "and":
            return N("and", list(a))
        if op == "not":
            return N("not", [a[0]])
        if op == "const":
            return a[0]
        if op == "star":
            return N("rep", [a[0]], m=0, n=None)
        if op == "optional":
            return N("rep", [a[0]], m=0, n=1)
        if op == "repeat":
            return N("rep", [a[0]], m=a[1], n=a[1])
        if op == "repeatatleast":
            return N("rep", [a[0]], m=a[1], n=None)
        if op == "repeatrange":
            return N("rep", [a[0]], m=a[1], n=a[2])
        if op == "contain":
            return N("seq", [STAR(), a[0], STAR()])
        if op == "startwith":
            return N("seq", [a[0], STAR()])
        if op == "endwith":
            return N("seq", [STAR(), a[0]])
        if op == "notcc":
            node = a[0]
            if node.kind == "class":
                return N("class", val=frozenset(PRINTABLE - node.val))
            if node.kind == "char":
                return N("class", val=frozenset(PRINTABLE - {node.val}))
            raise SRParseError("notcc over a non-class operand")
        raise SRParseError(f"unknown operator {op!r}")

    root = expr()
    if pos != len(toks):
        raise SRParseError(f"trailing tokens at {pos}")
    return root


# ------------------------------------------------------------------ census

def examples(card):
    """Positive and negative example strings.

    A bare '-' is the export's empty-slot placeholder, not a string: it
    occurs as a negative on 10 cards and as a positive on none, and on the
    one pattern where it changes the verdict the corpus's own positives
    contain a hyphen. Dropping it is a fix to our reading of the export,
    not a correction of the corpus.
    """
    def body(block):
        toks = block.split("\n", 1)[1].split() if "\n" in block else []
        return [t for t in toks if t != "-"]
    return body(card.get("before", "")), body(card.get("after", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true",
                    help="print each disagreeing pattern")
    args = ap.parse_args()

    cards = [c for c in json.loads(
        (REPO / "data" / "inspector_imports.json").read_text())
        if c["tab"] == "sregex"]

    stat = collections.Counter()
    fail_ops = collections.Counter()
    parse_errs = collections.Counter()
    disagree = []

    # The export carries several annotator descriptions per pattern, so
    # validating card-by-card counts the same pattern up to three times.
    # Independent checks = distinct patterns.
    by_pattern = {}
    for c in cards:
        by_pattern.setdefault(c["command"], c)
    for c in by_pattern.values():
        src = c["command"]
        try:
            ast = parse_sr(src)
        except (SRParseError, Exception) as exc:      # noqa: BLE001
            stat["parse-fail"] += 1
            parse_errs[f"{type(exc).__name__}: {exc}"[:60]] += 1
            continue
        stat["parsed"] += 1
        pos, neg = examples(c)
        if not pos and not neg:
            stat["no-examples"] += 1
            continue
        bad = [s for s in pos if not matcher(ast, s)]
        bad += [s for s in neg if matcher(ast, s)]
        if bad:
            stat["disagrees"] += 1
            for m in re.finditer(r"([a-zA-Z_]+)\(", src):
                fail_ops[m.group(1)] += 1
            disagree.append((c["ref"], src, bad[:3]))
        else:
            stat["agrees"] += 1

    n = len(by_pattern)
    print(f"StructuredRegex adapter — {n} DISTINCT patterns "
          f"({len(cards)} cards; the export carries up to 3 annotator "
          f"descriptions per pattern)")
    print("  NOTE: recovered sample only — StructuredRegex is not vendored in "
          "this repo,\n        and refs run contiguously SR-0001..SR-0718, so "
          "this is not a\n        verified random sample of the corpus. Do not "
          "extrapolate from it.\n")
    print(f"  parsed                {stat['parsed']:>4} / {n}"
          f"  ({stat['parsed']/n:.1%})")
    print(f"  parse failures        {stat['parse-fail']:>4}")
    checked = stat["agrees"] + stat["disagrees"]
    if checked:
        print(f"\n  validated against the corpus's own pos/neg strings:")
        print(f"    agree               {stat['agrees']:>4} / {checked}"
              f"  ({stat['agrees']/checked:.1%})")
        print(f"    disagree            {stat['disagrees']:>4}")
    if parse_errs:
        print("\n  parse failure reasons:")
        for k, v in parse_errs.most_common(6):
            print(f"    {v:>4}  {k}")
    if fail_ops:
        print("\n  operators present in disagreeing patterns:")
        for k, v in fail_ops.most_common(8):
            print(f"    {k:<16} {v}")
    # ---- instruction quality, measured exactly as the provenance analysis
    # measured KB13 (99.1%) / NL-RX-Turk (26.7%) / NL-RX-Synth (25.4%)
    screened = collections.Counter()
    rules = collections.Counter()
    tiers = collections.Counter()
    for c in cards:
        try:
            nf = serialize(parse_sr(c["command"]))
        except Exception:                              # noqa: BLE001
            continue
        tiers[tier_for(nf)] += 1
        nl = (c.get("instructions") or [""])[0]
        if not nl:
            continue
        cand = instruction_for(nl, "delete")
        fired = classify(nf, cand)
        anch = anchor_mismatch(nf, cand)
        if fired or anch:
            screened["rejected"] += 1
            for r in (fired or []):
                rules[r] += 1
            if anch:
                rules["anchor-mismatch"] += 1
        else:
            screened["approved"] += 1
    tot = screened["approved"] + screened["rejected"]
    if tot:
        print(f"\n  instruction screen (same rules as the provenance analysis):")
        print(f"    approved            {screened['approved']:>4} / {tot}"
              f"  ({screened['approved']/tot:.1%})")
        print(f"    for reference — KB13 99.1% · NL-RX-Turk 26.7% · "
              f"NL-RX-Synth 25.4%")
        if rules:
            print("\n  rules fired on rejected instructions:")
            for k, v in rules.most_common(8):
                print(f"    {k:<24} {v}")
    if tiers:
        print(f"\n  tier distribution: "
              + " · ".join(f"{k} {v}" for k, v in sorted(tiers.items())))

    # ---- register: the screen tests honesty, not readability. The A3
    # census showed those come apart, so characterise the prose directly.
    reg = collections.Counter()
    words = []
    for c in cards:
        nl = (c.get("instructions") or [""])[0]
        if not nl:
            continue
        words.append(len(nl.split()))
        letters = [ch for ch in nl if ch.isalpha()]
        if letters and sum(ch.isupper() for ch in letters) / len(letters) > .8:
            reg["shouted (>80% caps)"] += 1
        if re.search(r"[a-z]\.[A-Z]", nl):
            reg["run-on (missing space after '.')"] += 1
        if len(re.findall(r"\.\s|\.$", nl)) > 1:
            reg["multi-sentence"] += 1
        if re.search(r"\b(?:first|second|third|fourth|fifth)\b", nl, re.I):
            reg["positional spec prose"] += 1
    if words:
        words.sort()
        print(f"\n  instruction register ({len(words)} instructions):")
        print(f"    length              median {words[len(words)//2]} words, "
              f"p90 {words[int(.9*len(words))]}, max {words[-1]}")
        for k, v in reg.most_common():
            print(f"    {k:<32} {v:>4}  ({v/len(words):.0%})")

    if args.verbose:
        for ref, src, bad in disagree[:15]:
            print(f"\n  {ref}: {src[:110]}")
            print(f"     mismatched strings: {bad}")


if __name__ == "__main__":
    main()
