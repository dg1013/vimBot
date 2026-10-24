"""Phase A1 witness sampler: every witness class produces oracle-verified
lines of the right kind, and build_buffer assembles discriminating buffers.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "scripts" / "deep_regex"))

from dsl_ast import dsl_match  # noqa: E402
import witnesses as W  # noqa: E402


def rng():
    return random.Random(20260814)


# --------------------------------------------------------------- classes

def test_anchor_witness_breaks_full_match():
    t = "[a-z][0-9]"                      # anchored two-char language
    members = W.sample_members(t, rng(), want=3)
    assert members, "sampler must find members"
    out = W.anchor_witnesses(t, members, rng())
    assert out, "anchored language must yield anchor witnesses"
    for s in out:
        assert not dsl_match(t, s)
        assert any(m in s for m in members), "witness wraps a member"


def test_anchor_witness_absent_for_slack_wrapped():
    t = ".*dog.*"
    members = W.sample_members(t, rng(), want=2)
    assert W.anchor_witnesses(t, members, rng()) == []


def test_boundary_witness_glues_word_chars():
    t = ".*\\bdog\\b.*"
    members = W.sample_members(t, rng(), want=3) or ["z dog z"]
    out = W.boundary_witnesses(t, members, rng())
    assert out
    for s in out:
        assert "dog" in s and not dsl_match(t, s)


def test_case_witness_flips_case():
    t = "dog.*"
    out = W.case_witnesses(t, ["dogxy"])
    assert out and out[0] == "DOGXY"
    assert not dsl_match(t, out[0])


def test_count_witness_one_below_minimum():
    t = "([a-z]){3,}"
    out = W.count_witnesses(t, rng())
    assert out
    for s in out:
        assert not dsl_match(t, s)
        assert len(s) == 2                # exactly m-1 repetitions


def test_count_witness_for_plus():
    t = "x([0-9])+"
    out = W.count_witnesses(t, rng())
    assert out
    for s in out:
        assert not dsl_match(t, s)


def test_adjacency_members_both_readings():
    t = ".*a.*[0-9].*"
    out = W.adjacency_members(t, rng())
    assert len(out) >= 2
    for s in out:
        assert dsl_match(t, s), "adjacency witnesses are members"
    joined = {len(s) for s in out}
    assert len(joined) > 1, "adjacent and gapped variants differ"


def test_negation_witness_satisfies_unnegated():
    t = "~(.*dog.*)"
    out = W.negation_witnesses(t, rng())
    assert out
    for s in out:
        assert "dog" in s
        assert not dsl_match(t, s)


def test_anchor_witness_under_inverted_polarity():
    # ~(X) compiles with pol 'v': the command full-matches X, so the anchor
    # witness must wrap an X-member and lands on the t-MEMBER side
    t = "~(([AEIOUaeiou]).*([0-9]).*)"
    out = W.anchor_witnesses(t, [], rng(), pol="v")
    assert out
    for s in out:
        assert dsl_match(t, s), "inverted-polarity anchor witness is a t-member"
        inner = W.inner_membership(t, "v")
        assert not inner(s)


def test_anchor_witness_nonword_pads_for_boundary_inner():
    # \b-edged inner: a word-char pad would kill the contained match, so
    # pads must be non-word for the witness to discriminate
    t = "\\b(dog.*[0-9].*){6,}\\b"
    members = W.sample_members(t, rng(), want=2)
    assert members
    out = W.anchor_witnesses(t, members, rng())
    assert out
    for s in out:
        assert not dsl_match(t, s)
        assert s[0] not in W.WORD or s[-1] not in W.WORD


# ----------------------------------------------------------- build_buffer

REAL_PATTERNS = [
    ".*\\bFacebook\\b.*",                # word-boundary containment
    "[AEIOUaeiou]*",                     # anchored star (only-vowels)
    "(.*)([A-Za-z].*[a-z].*)",           # trailing-anchored sequence
    ".*[a-z].*[A-Za-z].*",               # VA-0094 class-sequence
    "(([AEIOUaeiou])|([a-z])){3,}",      # anchored counted alternation
]


def test_build_buffer_properties():
    for t in REAL_PATTERNS:
        buf = W.build_buffer(t, rng())
        assert buf is not None, t
        lines = buf["lines"]
        assert 6 <= len(lines) <= 10, t
        assert len(set(lines)) == len(lines)
        for s in lines:
            assert buf["truth"][s] == dsl_match(t, s), (t, s)
        members = [s for s in lines if buf["truth"][s]]
        nonmembers = [s for s in lines if not buf["truth"][s]]
        assert len(members) >= 2 and len(nonmembers) >= 2, t
        # at least one non-generic witness on the non-member side
        assert any(buf["labels"][s] not in ("generic", "probe")
                   for s in nonmembers), t


def test_build_buffer_deterministic():
    t = REAL_PATTERNS[2]
    a = W.build_buffer(t, random.Random(7))
    b = W.build_buffer(t, random.Random(7))
    assert a["lines"] == b["lines"]


def test_anchored_pattern_gets_anchor_witness():
    t = "(([AEIOUaeiou])|([a-z])){3,}"   # the TB-019 shape
    buf = W.build_buffer(t, rng())
    labels = set(buf["labels"].values())
    assert "anchor" in labels or "count" in labels, labels
