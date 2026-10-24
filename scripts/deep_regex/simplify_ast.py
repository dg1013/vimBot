"""AST (abstract syntax tree) normalization for honest instruction generation (Phase A3).

The A2 adjudications showed that many corpus patterns' TRUE span semantics
are simpler than their syntax: span-negations beside slack are absorbable
(~(X) admits the empty span whenever ε ∉ L(X)), match-empty repetitions
beside slack contribute nothing, `&`-operands equal to .* are no-ops, and a
bare-dot alternation branch under an unbounded repeat reduces the repeat to
a length count. Verbalizing the RAW syntax would produce honest absurdity;
this module rewrites each AST to a normal form whose English is the task's
real meaning.

Every rule is language-preserving by argument, and `verify_simplify()`
checks each pattern empirically: membership of original vs simplified is
compared by the span oracle on members + witnesses of BOTH sides. A pattern
whose simplification fails verification keeps its original AST.

The serialized normal form is also the semantic-dedup key (owner decision
2026-08-14: one scenario per equivalence class).
"""
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dsl_ast import N, matcher, parse, serialize  # noqa: E402


def nullable(nd):
    """ε ∈ L(nd), by the oracle itself."""
    return matcher(nd, "")


def is_slack(nd):
    """rep over dot with unbounded max and min 0 — the .* wildcard."""
    return (nd.kind == "rep" and nd.m == 0 and nd.n is None
            and nd.kids and nd.kids[0].kind == "dot")


def _slacklike(nd):
    """Language equals Σ* (structurally): .* or (.*) groups thereof."""
    return is_slack(nd)


def _literal_chars(nd):
    """The character set of a purely-literal node (char / seq of chars),
    else None."""
    if nd.kind == "char":
        return frozenset(nd.val)
    if nd.kind == "seq" and all(c.kind == "char" for c in nd.kids):
        return frozenset(c.val for c in nd.kids)
    return None


def simplify(nd):
    """Bottom-up normalization; returns a NEW node tree."""
    kids = [simplify(c) for c in nd.kids]
    k = nd.kind

    if k == "seq":
        # flatten nested seqs
        flat = []
        for c in kids:
            flat.extend(c.kids if c.kind == "seq" else [c])
        # absorption to fixpoint: a nullable element adjacent to slack is
        # covered by the slack (seq(.*, N) ≡ .* for nullable N, so
        # seq(.*, N, M) ≡ seq(.*, M)); merge adjacent slacks
        changed = True
        while changed:
            changed = False
            out = []
            i = 0
            while i < len(flat):
                c = flat[i]
                if is_slack(c) and out and is_slack(out[-1]):
                    changed = True          # .*.* -> .*
                    i += 1
                    continue
                if (not is_slack(c) and nullable(c)
                        and ((out and is_slack(out[-1]))
                             or (i + 1 < len(flat) and is_slack(flat[i + 1])))):
                    changed = True          # nullable beside slack: absorbed
                    i += 1
                    continue
                out.append(c)
                i += 1
            flat = out
        if not flat:
            return N("rep", [N("dot")], m=0, n=None)   # everything absorbed
        return flat[0] if len(flat) == 1 else N("seq", flat)

    if k == "alt":
        seen, out = set(), []
        for c in kids:
            s = serialize(c)
            if s not in seen:
                seen.add(s)
                out.append(c)
        return out[0] if len(out) == 1 else N("alt", out)

    if k == "and":
        # an operand whose language is Σ* constrains nothing
        out = [c for c in kids if not _slacklike(c)]
        # a starred/plussed class operand is implied by a purely-literal
        # operand whose characters all lie in the class (ea & [A-Za-z]* ≡ ea)
        lits = [c for c in out if _literal_chars(c) is not None]
        if lits:
            def implied(c):
                if (c.kind == "rep" and c.n is None and c.m <= 1
                        and c.kids[0].kind == "class"):
                    cs = c.kids[0].val
                    return any(_literal_chars(l) <= cs for l in lits)
                return False
            out = [c for c in out if not implied(c)]
        if not out:
            return N("rep", [N("dot")], m=0, n=None)
        return out[0] if len(out) == 1 else N("and", out)

    if k == "rep":
        child = kids[0]
        # bare-dot branch under an unbounded repeat: every other branch
        # consumes ≥1 char, so the language is exactly "m or more chars"
        if (nd.n is None and child.kind == "alt"
                and any(c.kind == "dot" for c in child.kids)
                and all(not nullable(c) for c in child.kids)):
            return N("rep", [N("dot")], m=nd.m, n=None)
        return N("rep", [child], m=nd.m, n=nd.n)

    return N(k, kids, val=nd.val, m=nd.m, n=nd.n)


def normal_form(text):
    """Parse, simplify, serialize. Returns (simplified_text, ast) or
    (text, original_ast) when parsing round-trips fail."""
    ast = parse(text)
    simp = simplify(ast)
    out = serialize(simp)
    # serialization must re-parse to the same language (cheap guard)
    try:
        parse(out)
    except Exception:
        return text, ast
    return out, simp


def verify_simplify(text, samples=60, seed=0):
    """Empirical equivalence check original vs simplified on members and
    witnesses of both sides. True = verified (or no-op)."""
    from witnesses import build_buffer, sample_members
    out, _ = normal_form(text)
    if out == text:
        return True
    rng = random.Random(f"{seed}:{text}")
    orig, simp = parse(text), parse(out)
    pool = set()
    for t in (text, out):
        pool.update(sample_members(t, rng, want=6))
        buf = build_buffer(t, rng)
        if buf:
            pool.update(buf["lines"])
    pool.update(["", "a", "!", "dog", "the Dog ran", "  "])
    return all(matcher(orig, s) == matcher(simp, s)
               for s in pool if "\n" not in s and len(s) <= 40)
