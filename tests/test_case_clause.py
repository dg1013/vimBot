"""Tests for scripts/case_clause.py — the varied case clause (§7.2 diversity).

Pins what the 2026-09-04 wording pass learned the hard way: the clause must
read as English in the sentence it lands in, must never strand a trailing
modifier, and must always say something the case classifier recognises.
"""

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from build_test_review import _SAYS_CASE, _case_is_shown  # noqa: E402
from case_clause import add_case_clause, shape, variants  # noqa: E402

SENTENCES = [
    "Delete lines that contain words starting with 'ba'.",
    "Keep only lines which contain the letter 'z' twice.",
    "Delete lines using 'art' or 'led'",
    "Delete lines that ends with 'ABC' and contain two numbers.",
    "Keep only lines containing 'GHF', but not 'JFK'",
    "Keep only lines starting with a 'T'.",
    "Keep only lines containing words that end with 'P'",
    "Delete lines with the word 'Goal' in them.",
    "Keep only lines that begin with 'The' or 'For'.",
    "Delete lines that contain 'Beaker' but do not contain 'Bunsen'.",
    "Keep only lines that contain 'Freddie Mercury' and contain 'AIDS'.",
    "Delete lines where the letter c appears more than one time",
]


def explicit(text):
    return bool(_SAYS_CASE.search(text)) or _case_is_shown(text)


@pytest.mark.parametrize("ins", SENTENCES)
def test_every_variant_states_the_case(ins):
    opts = variants(ins)
    assert opts, ins
    for v in opts:
        assert explicit(v), (ins, v)


@pytest.mark.parametrize("ins", SENTENCES)
def test_no_variant_is_a_no_op(ins):
    assert all(v != ins for v in variants(ins))


@pytest.mark.parametrize("ins", SENTENCES)
def test_chosen_clause_is_stable_and_explicit(ins):
    first = add_case_clause(ins, "row-1")
    assert first == add_case_clause(ins, "row-1")     # same row, same wording
    assert explicit(first)


def test_the_same_sentence_varies_across_rows():
    ins = "Delete lines that contain words starting with 'ba'."
    got = {add_case_clause(ins, "kb13-%04d" % n) for n in range(60)}
    assert len(got) >= 4, got


def test_singular_in_capital_is_never_produced():
    # "the letter 'G' in capital" is not English; the trailing form has to be
    # plural, and the adjective form goes in front of the noun.
    for ins in SENTENCES:
        for v in variants(ins):
            assert not re.search(r"\bin capital\b(?!s)", v), v


def test_capital_letters_are_named_without_repeats():
    # 'Beaker' and 'Bunsen' share a B: "a capital B and a capital B" is wrong.
    ins = "Delete lines that contain 'Beaker' but do not contain 'Bunsen'."
    for v in variants(ins):
        assert "a capital B and a capital B" not in v, v


def test_a_trailing_clause_is_only_used_when_the_literal_ends_the_clause():
    # "the letter 'z' in lowercase twice" strands "twice" -- that shape must
    # not appear; the fix is an attributive form or a closing sentence.
    ins = "Keep only lines which contain the letter 'z' twice."
    for v in variants(ins):
        assert "'z' in lowercase twice" not in v, v


def test_shape_detection():
    assert shape("words starting with 'ba'") == "lower"
    assert shape("ends with 'ABC'") == "upper"
    assert shape("starting with a 'T'") == "capletter"
    assert shape("the word 'Goal'") == "initial"
    assert shape("'Freddie Mercury' and 'AIDS'") == "mixed"
    assert shape("the letter c appears twice") == "none"
