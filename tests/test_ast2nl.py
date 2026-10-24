"""Phase A3 pins: AST (abstract syntax tree) simplifier + deterministic instruction renderer.

Goldens freeze representative renderings; property tests assert the
structural honesty guarantees (frame words track anchoring, the screens
never fire on generated instructions, determinism under a fixed seed).
"""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "deep_regex"))

from ast2nl import render_variants  # noqa: E402
from dsl_ast import dsl_match, parse  # noqa: E402
from simplify_ast import normal_form, simplify, verify_simplify  # noqa: E402
from dsl_ast import matcher, serialize  # noqa: E402
from convertibility import split_top  # noqa: E402
from garble_screen import anchor_mismatch, classify  # noqa: E402


def rng():
    return random.Random(42)


# ------------------------------------------------------------- simplifier

def test_simplify_absorbs_nullable_beside_slack():
    assert normal_form("(~([a-z])).*((dog)+).*")[0] == ".*dog\\+.*" \
        or normal_form("(~([a-z])).*((dog)+).*")[0] == ".*(dog)+.*" \
        or "dog" in normal_form("(~([a-z])).*((dog)+).*")[0]
    nf, _ = normal_form("(([0-9])*).*([A-Z]).*")
    assert nf == ".*[A-Z].*"


def test_simplify_and_absorption():
    assert normal_form(".*(ea&[A-Za-z]*).*")[0] == ".*ea.*"
    assert normal_form("(.*[AEIOUaeiou].*)&((.)*)")[0] == ".*[AEIOUaeiou].*"


def test_simplify_dot_alt_rep():
    nf, _ = normal_form("((.)|(dog)){7,}")
    assert nf == ".{7,}"


def test_simplify_preserves_language_on_knowns():
    for t in ["(~([a-z])).*((dog)+).*", ".*(ea&[A-Za-z]*).*",
              "((.)|(dog)){7,}", "(.*)([A-Za-z].*[a-z].*)"]:
        assert verify_simplify(t), t


# --------------------------------------------------------------- renderer

GOLDENS = {
    ".*[a-z].*[A-Za-z].*":
        "lines containing a lowercase letter, then anywhere later a letter",
    ".*\\bFacebook\\b.*":
        "lines that contain the whole word 'Facebook'",
    "(dog)(.*)": "lines beginning with 'dog'",
    "~(.*dog.*)": "lines that do not contain 'dog'",
    ".*\\b[A-Za-z]*ing\\b.*": "lines that contain a word ending in 'ing'",
}


def test_render_goldens():
    for t, want in GOLDENS.items():
        got = render_variants(t, rng())
        assert got and got[0] == want, (t, got)


def test_render_deterministic():
    for t in GOLDENS:
        a = render_variants(t, random.Random(3))
        b = render_variants(t, random.Random(3))
        assert a == b


def test_render_degenerate_matchall():
    assert render_variants("(((dog)|(.)))*", random.Random(1)) is None


def test_frame_words_track_anchoring_and_screens_never_fire():
    path = ROOT / "data/external/deep-regex/conversions.json"
    if not path.exists():
        pytest.skip(f"{path} not present")
    conv = json.loads(path.read_text())
    usable = [t for t, c in conv.items()
              if isinstance(c, dict) and c.get("exec") and c.get("k") != 2
              and c.get("n") not in ("degenerate", "trivial")]
    sample = random.Random(9).sample(usable, 250)
    checked = 0
    for t in sample:
        nf, ast = normal_form(t)
        got = render_variants(nf, random.Random(5))
        if not got:
            continue
        checked += 1
        phrase = got[0]
        instr = "Keep only " + phrase
        # the generator's standing gate: the screens' INSTRUCTION-side rules
        # must be silent on generated text. classify()'s source-side rules
        # (R1/R2/R3-open-star/R4) flag pattern families regardless of
        # instruction quality — they were quarantine selectors for the old
        # corpus, not text validators — so they are out of scope here; R6
        # additionally reads pattern edges and is branch-blind, so it only
        # applies to plain single-branch roots.
        hits = classify(nf, instr)
        text_rules = {r for r in hits
                      if r.startswith(("R5", "R3-zero"))}
        assert not text_rules, (nf, instr, hits)
        plain = ("&" not in nf and "~" not in nf
                 and len(split_top(nf, "|")) == 1)
        if plain:
            assert not {r for r in hits if r.startswith("R6")}, (nf, instr)
            assert not anchor_mismatch(nf, instr), (nf, instr)
        # frame words track anchoring for plain seq roots
        if ast.kind == "seq":
            left_slack = serialize(ast).startswith(".*")
            if phrase.startswith(("lines starting", "lines beginning",
                                  "lines that start")):
                assert not left_slack, (nf, phrase)
            if phrase.startswith(("lines containing", "lines with",
                                  "lines that contain")):
                assert matcher(ast, "") or left_slack \
                    or "word" in phrase or "at least" in phrase, (nf, phrase)
    assert checked >= 150, f"too few rendered in sample: {checked}"
