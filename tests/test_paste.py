"""Tests for the pasteable rendering in src/paste.py (§10, decided 2026-09-04).

Pins: the escaping rules, that every notation token convert_keys() knows is
rendered as a key code, fidelity through real vim against the executor's
keys mode, the parse_output integration, the CLI (command-line interface), and the boundary: the
executor refuses the rendered form, so it can never come from the model.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from executor import _KEYMAP, run_vim  # noqa: E402
from paste import escape_keys, pasteable  # noqa: E402
from prompt import parse_output  # noqa: E402

PASTE_PY = REPO / "src" / "paste.py"


# ---------------------------------------------------------------- escaping

@pytest.mark.parametrize("keys, body", [
    ('ci"foo<Esc>', 'ci\\"foo\\<Esc>'),
    ("dd", "dd"),
    (">ipgv<C-a>", ">ipgv\\<C-a>"),
    ("/paris<CR>gUl", "/paris\\<CR>gUl"),
    (":set tw=72<CR>gqip", ":set tw=72\\<CR>gqip"),
    ("A |<Esc>", "A \\<Bar>\\<Esc>"),               # bar stays out of the ex parser
    ("A<Bar><Esc>", "A\\<Bar>\\<Esc>"),
    ("i\\n<Esc>", "i\\\\n\\<Esc>"),                 # backslash doubled
    ("i<div><Esc>", "i<div>\\<Esc>"),               # unknown token is literal text
    ("i<lt>b<gt><Esc>", "i<b>\\<Esc>"),             # <lt>/<gt> are plain characters
    ("a<Space>x<Esc>", "a x\\<Esc>"),
    ("<esc>", "\\<esc>"),                           # spelling is kept
    (" x", "1 x"),                                  # :normal would skip a leading space
    ("<Space>x", "1 x"),
])
def test_escape_keys(keys, body):
    assert escape_keys(keys) == body


def test_every_notation_token_is_rendered_as_a_key_code():
    for name in _KEYMAP:
        if name in ("<lt>", "<gt>", "<space>"):
            continue
        assert escape_keys(name) == "\\" + name, name
    assert escape_keys("<C-v>") == "\\<C-v>"
    assert escape_keys("<c-q>") == "\\<c-q>"


def test_pasteable_forms():
    assert pasteable("keys", 'ci"foo<Esc>') == ':exe "normal! ci\\"foo\\<Esc>"'
    assert pasteable("ex", "%s/a/b/g") == ":%s/a/b/g"
    assert pasteable("ex", ":g/^$/d") == ":g/^$/d"          # one colon, never two
    with pytest.raises(ValueError):
        pasteable("normal", "dd")


def test_renders_a_parsed_completion():
    assert pasteable(*parse_output('keys: ci"x<Esc>')) == ':exe "normal! ci\\"x\\<Esc>"'
    assert pasteable(*parse_output("ex: g/^$/d")) == ":g/^$/d"


# ---------------------------------------------------------------- fidelity

def _vim_after_paste(buffer_before: str, line: str) -> list:
    """What the user's own vim holds after pasting `line`: a bare vim in Ex
    silent mode with the sandbox's flags, writing the buffer to a scratch
    file that is read back. Independent of the executor on purpose. Not
    :print -- in silent mode that *displays* lines (tabs as spaces, an empty
    line as one space, beeps into stdout) rather than dumping them."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "buf.txt"
        path.write_text(buffer_before, encoding="utf-8")
        subprocess.run(
            ["vim", "-Z", "-u", "NONE", "-i", "NONE", "-n", "-es",
             "-c", "set nrformats-=octal", "-c", "call cursor(1,1)",
             "-c", line.lstrip(":"), "-c", "w! out.txt", "-c", "q!", str(path)],
            capture_output=True, text=True, timeout=10, cwd=d)
        after = (Path(d) / "out.txt").read_text(encoding="utf-8")
    return after.rstrip("\n").split("\n")


@pytest.mark.skipif(shutil.which("vim") is None, reason="needs vim")
@pytest.mark.parametrize("keys, before", [
    ("dd", "a\nb\n"),
    ('ci"foo<Esc>', 'say "hi" now\n'),
    ("A |<Esc>", "x\n"),
    ("3J", "a\nb\nc\nd\n"),
    (":s/a/b/<CR>", "a\n"),
    ("<C-a>", "7\n"),
    ("i<div><Esc>", "x\n"),
    ("i\\n<Esc>", "x\n"),
    ("Vj>gv<C-a>", "1\n2\n\n3\n"),             # shift, reselect, increment both
    ("vipU", "a\nb\n\nc\n"),
    ("Vjygv>", "a\nb\n"),
    ("<C-v>jjd", "ab\ncd\nef\n"),
])
def test_pasted_line_reproduces_the_sandbox_result(keys, before):
    ref = run_vim(before, keys, mode="keys")
    assert ref["was_valid"], ref
    assert _vim_after_paste(before, pasteable("keys", keys)) == \
        ref["buffer_after"].rstrip("\n").split("\n")


# ---------------------------------------------------------------- boundary

def test_executor_refuses_the_rendered_form():
    # The rendering is for people. If it ever came from the model, the
    # screen must refuse it before vim runs -- this is the §7 :normal/:execute gate.
    line = pasteable("keys", "dd")
    r = run_vim("a\nb\n", line.lstrip(":"), mode="ex")
    assert "blocked" in r and r["was_valid"] is False


# --------------------------------------------------------------------- CLI

def test_cli_renders_completions_and_pairs():
    ok = subprocess.run([sys.executable, str(PASTE_PY), "--completion",
                         'keys: ci"x<Esc>'], capture_output=True, text=True)
    assert ok.returncode == 0 and ok.stdout == ':exe "normal! ci\\"x\\<Esc>"\n'
    pair = subprocess.run([sys.executable, str(PASTE_PY), "--mode", "ex",
                           "--command", "g/^$/d"], capture_output=True, text=True)
    assert pair.returncode == 0 and pair.stdout == ":g/^$/d\n"
    bad = subprocess.run([sys.executable, str(PASTE_PY), "--completion",
                          "just delete it"], capture_output=True, text=True)
    assert bad.returncode == 1 and "parse failure" in bad.stderr
