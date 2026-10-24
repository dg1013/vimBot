"""Pins for adjudicated DSL (domain-specific language) span semantics (Phase A2, 2026-08-14).

The full-corpus audit suspected two converter bugs; direct adjudication
proved both conversions FAITHFUL to the source DSL's actual semantics.
These pins protect that finding: a future "fix" that changes either
behavior would silently split the converter from the span oracle.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "scripts" / "deep_regex"))

from dsl_ast import dsl_match  # noqa: E402


def test_span_negation_admits_empty():
    """~(X) is span negation: the EMPTY span is not in L([a-z]), so
    ~([a-z]) matches it — a leading (~([a-z])) is absorbable and
    (~([a-z])).*X.* is exactly .*X.* (the 'negation elision' rewrite is
    an equivalence, not a bug)."""
    assert dsl_match("~([a-z])", "")
    t = "(~([a-z])).*((dog)+).*"
    assert dsl_match(t, "zz dog")       # lowercase before dog: still matches
    assert dsl_match(t, "dog")
    assert not dsl_match(t, "DOG")
    t2 = "(([A-Za-z])+).*(~([A-Z])).*"
    assert dsl_match(t2, "truckdog")
    assert dsl_match(t2, "AB")          # trailing ~([A-Z]) absorbable too


def test_exact_boundary_rendering_under_repetition():
    """A2 regression: DSL \\b is direction-agnostic, so a boundary inside a
    repeated group must render as \\%(\\<\\|\\>\\) (union), not the
    heuristic's one-sided \\< or \\>. Counterexamples are the 2026-08-14
    audit's mismatch strings — exact rendering must agree with the oracle."""
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from dsl2vim import generate
    from executor import run_vim

    cases = {
        "(\\b(.*)(dog)\\b){2,}": ["dog dogT9dogz dog", "dog!Tdogdog", "za"],
        "([0-9]).*(\\b.\\b).*": ["7!!9", "6Tq!!x", "ab"],
        "(\\b(.)|(dog)\\b){6,}": ["T qdog!zdog", "dog 9 dog z"],
        # boundary-dense languages that member-sampling cannot reach —
        # the static bound_risk rule must force exact rendering for these
        "(\\b.\\b){5,}": ["9 T a", "T x!x", "axzzz"],
        "(\\b[AEIOUaeiou].*[0-9].*\\b){4,}": ["ixz39u 1 eT2ix1 u5!Iaq8"],
    }
    for t, strings in cases.items():
        g = generate(t, exact_boundary=True)
        assert g["exec"], t
        for s in strings:
            res = run_vim(s, f"{g['pol']}/{g['exec']}/s/^/#/", mode="ex")
            marked = res["buffer_after"].rstrip("\n").startswith("#")
            assert marked == dsl_match(t, s), (t, s)


def test_mixed_boundary_alternation_precedence():
    """Concat binds tighter than | in the DSL, so \\b(A)|(B)\\b is
    (\\b A)|(B \\b) — the vim rendering \\<\\(A\\)\\|\\(B\\)\\> has the
    SAME precedence and is faithful. The corpus authored it this way;
    the English describing it is what lies (scenario 2329 / LU-024)."""
    t = "(.*)(\\b([AEIOUaeiou])|(dog)\\b)"
    assert dsl_match(t, "xdog")         # dog branch needs only a RIGHT boundary
    assert dsl_match(t, "dog")
    assert dsl_match(t, "z a")          # vowel branch with a left boundary
    assert not dsl_match(t, "xdogx")    # no right boundary after dog
    assert not dsl_match(t, "za")       # no boundary before the vowel
