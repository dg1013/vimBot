"""Every gold command must survive the model's own output grammar.

`src/prompt.py` renders a training target as one line, `ex: <command>` or
`keys: <keystrokes>`, and `parse_output` reads a model answer back by taking
the FIRST line and stripping its trailing whitespace. A gold that ends in a
space, a tab or a raw newline, or that holds a raw newline in the middle,
therefore cannot be matched by anything the model says: SFT teaches the
string and the parser cuts it. Found 2026-09-16 (DECISIONS.md): 19 golds
ended in whitespace and 42 held a newline mid-command, all in train; on one
of them the trained policy answered `4r` six times of eight, the gold `4r `
minus its space. The §6 notation already spells those keystrokes as `<CR>`,
`<Space>` and `<Tab>`, and the executor reads them, so the repair was
spelling; this test keeps the corpus that way.

The check is the round trip itself, not a list of bad characters: whatever
`parse_output` would do to a gold, a gold it changes is a gold no answer can
equal. The two named rules below give a readable reason when it fails.
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from prompt import parse_output, render_target   # noqa: E402

FILES = [REPO / "data" / "combined_cases.jsonl"] + [
    REPO / "data" / "splits" / f"{name}.jsonl" for name in ("train", "val", "test")]


def _rows(path):
    if not path.exists():
        pytest.skip(f"{path} not present")
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _survives(mode, command):
    """The command as the grammar would hand it back, or None if unparsable."""
    parsed = parse_output(render_target(mode, command))
    return parsed[1] if parsed else None


# ---------------------------------------------------------------- the rule

def test_the_parser_strips_trailing_whitespace_and_cuts_at_a_newline():
    """The behaviour the corpus rule exists for, pinned so the reason for the
    rule stays in the suite even if the parser is ever changed."""
    assert _survives("keys", "4r ") == "4r"
    assert _survives("keys", "r\t") == "r"
    assert _survives("keys", "/hello\nggdgn") == "/hello"
    assert _survives("keys", "d/four\n") == "d/four"
    # the token spellings round-trip
    for cmd in ("4r<Space>", "r<Tab>", "/hello<CR>ggdgn", "d/four<CR>", "/\t<CR>r<Space>"):
        assert _survives("keys", cmd) == cmd
    assert _survives("ex", "%s/a/b/g") == "%s/a/b/g"


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_every_gold_survives_the_output_grammar(path):
    """render_target then parse_output returns the gold unchanged, for every
    row of the corpus and of each frozen split file."""
    bad = []
    for r in _rows(path):
        cmd = r["gold_command"]
        got = _survives(r["mode"], cmd)
        if got != cmd:
            bad.append((r["id"], cmd, got))
    assert not bad, "golds the output grammar would alter (id, gold, what survives): %r" % bad[:10]


@pytest.mark.parametrize("path", FILES, ids=[p.name for p in FILES])
def test_no_gold_ends_in_whitespace_or_holds_a_raw_newline(path):
    """The two concrete shapes the round trip has caught, named so a failure
    reads as the defect and not as a parser mismatch."""
    trailing = [r["id"] for r in _rows(path) if r["gold_command"] != r["gold_command"].rstrip()]
    newline = [r["id"] for r in _rows(path) if "\n" in r["gold_command"]]
    assert not trailing, "golds ending in whitespace (spell the key as <Space>/<Tab>): %r" % trailing[:10]
    assert not newline, "golds holding a raw newline (spell it as <CR>): %r" % newline[:10]
