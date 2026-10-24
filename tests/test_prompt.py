"""Tests for the §6 model I/O contract in src/prompt.py.

Pins: the system text and user layout against CLAUDE.md §6 itself (so the
contract document and the code cannot drift apart silently), the k-shot
message shape, the strict output grammar with its three fixed details, and
that the frontier batch renderer uses the module rather than a copy.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import prompt  # noqa: E402
from prompt import (  # noqa: E402
    SYSTEM_TEXT,
    build_messages,
    messages_for_case,
    parse_output,
    render_target,
    render_user,
)

PROMPT_PY = REPO / "src" / "prompt.py"
PYTHON = sys.executable

ROW = {
    "id": "t-1",
    "category": "substitute",
    "instruction": "Replace every foo with bar",
    "mode": "ex",
    "buffer_before": "foo one\nfoo two\n",
    "gold_command": "%s/foo/bar/g",
    "buffer_after": "bar one\nbar two\n",
    "cursor": [2, 3],
}


# ------------------------------------------------------- contract document

def _section_6_blocks():
    """The system and user blocks exactly as CLAUDE.md §6 draws them."""
    path = REPO / "CLAUDE.md"
    if not path.exists():
        pytest.skip(f"{path} not present")
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("## 6."))
    sys_i = next(i for i in range(start, len(lines))
                 if lines[i].startswith("system: "))
    system = []
    for l in lines[sys_i:]:
        if not l.strip():
            break
        system.append(l[len("system: "):])
    user_i = next(i for i in range(sys_i, len(lines))
                  if lines[i].startswith("user:   "))
    user = []
    for l in lines[user_i:]:
        if l.startswith("```"):
            break
        user.append(l[len("user:   "):])
    return "\n".join(system), "\n".join(user)


def test_system_text_is_claude_md_section_6_verbatim():
    system, _ = _section_6_blocks()
    assert SYSTEM_TEXT == system


def test_user_layout_is_claude_md_section_6_verbatim():
    _, user = _section_6_blocks()
    # the document uses placeholders; render them literally and compare
    assert render_user("<instruction>", "<buffer_before>", ("<L>", "<C>")) == user


# -------------------------------------------------------------- rendering

def test_render_user_layout():
    assert render_user("Delete line 2", "a\nb\nc", (2, 1)) == (
        "Instruction: Delete line 2\n"
        "Cursor: line 2, column 1\n"
        "Buffer:\n<<<\na\nb\nc\n>>>"
    )


def test_render_user_default_cursor_is_1_1():
    assert "Cursor: line 1, column 1\n" in render_user("x", "y")


def test_render_target_and_mode_check():
    assert render_target("ex", "%s/a/b/") == "ex: %s/a/b/"
    assert render_target("keys", "ciwfoo<Esc>") == "keys: ciwfoo<Esc>"
    with pytest.raises(ValueError):
        render_target("normal", "dd")


def test_zero_shot_messages():
    msgs = build_messages("Delete line 2", "a\nb\nc", (2, 1))
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SYSTEM_TEXT
    assert msgs[1]["content"] == render_user("Delete line 2", "a\nb\nc", (2, 1))


def test_k_shot_examples_are_alternating_pairs_in_target_form():
    ex1 = dict(ROW, id="e1")
    ex2 = dict(ROW, id="e2", mode="keys", gold_command="dd", cursor=[1, 1])
    msgs = build_messages("q", "buf", (1, 1), examples=[ex1, ex2])
    assert [m["role"] for m in msgs] == [
        "system", "user", "assistant", "user", "assistant", "user"]
    assert msgs[2]["content"] == "ex: %s/foo/bar/g"
    assert msgs[4]["content"] == "keys: dd"
    assert msgs[1]["content"] == render_user(ex1["instruction"],
                                             ex1["buffer_before"], [2, 3])
    assert msgs[-1]["content"] == render_user("q", "buf")


def test_messages_for_case_reads_the_row_cursor():
    msgs = messages_for_case(ROW)
    assert "Cursor: line 2, column 3\n" in msgs[-1]["content"]
    assert "Cursor: line 1, column 1\n" in messages_for_case(
        {k: v for k, v in ROW.items() if k != "cursor"})[-1]["content"]


# ---------------------------------------------------------- output grammar

@pytest.mark.parametrize("text, expected", [
    ("ex: %s/a/b/g", ("ex", "%s/a/b/g")),
    ("keys: ciwfoo<Esc>", ("keys", "ciwfoo<Esc>")),
    ("keys:ciwfoo<Esc>", ("keys", "ciwfoo<Esc>")),       # the space is optional
    ("ex: g/^$/d\n", ("ex", "g/^$/d")),
    ("ex: g/^$/d\r\n", ("ex", "g/^$/d")),
    ("\n\n  ex: g/^$/d", ("ex", "g/^$/d")),               # leading whitespace skipped
    ("ex: dd   ", ("ex", "dd")),                          # trailing whitespace dropped
    ("ex: dd\nThis deletes the line.", ("ex", "dd")),     # first line decides
    ("ex:  two spaces", ("ex", " two spaces")),           # only one space consumed
    ("ex: s/a: b/c/", ("ex", "s/a: b/c/")),               # colons inside the command
])
def test_parse_output_accepts(text, expected):
    assert parse_output(text) == expected


@pytest.mark.parametrize("text", [
    "",
    "\n",
    "dd",
    "ex:",
    "ex: ",
    "ex:   ",
    "EX: dd",                                  # case-sensitive
    "Ex: dd",
    "normal: dd",
    "ex - dd",
    "Sure! ex: dd",                            # text before the answer
    "Answer:\nex: dd",
    "<think>\n\n</think>\nex: dd",             # any think content fails (§4)
    "ex: dd\n</think>",
    "ex: dd <think>later</think>",
    None,
])
def test_parse_output_rejects(text):
    assert parse_output(text) is None


@pytest.mark.parametrize("mode, command", [
    ("ex", "%s/foo/bar/g"),
    ("ex", "g/^\\s*$/d"),
    ("keys", "ci\"hello<Esc>"),
    ("keys", ">ipgv<C-a>"),
    ("ex", "s/a: b/c: d/"),
])
def test_render_then_parse_round_trips(mode, command):
    assert parse_output(render_target(mode, command)) == (mode, command)


# -------------------------------------------------------------------- CLI (command-line interface)

def test_cli_prints_the_system_text_exactly():
    out = subprocess.run([PYTHON, str(PROMPT_PY), "--system"],
                         capture_output=True, text=True, check=True)
    assert out.stdout == SYSTEM_TEXT + "\n"


def test_cli_renders_a_case_and_its_messages(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    out = subprocess.run([PYTHON, str(PROMPT_PY), "--case", str(cases),
                          "--id", "t-1"], capture_output=True, text=True,
                         check=True)
    assert out.stdout == render_user(ROW["instruction"], ROW["buffer_before"],
                                     ROW["cursor"]) + "\n"
    out = subprocess.run([PYTHON, str(PROMPT_PY), "--case", str(cases),
                          "--messages"], capture_output=True, text=True,
                         check=True)
    assert json.loads(out.stdout) == messages_for_case(ROW)
    missing = subprocess.run([PYTHON, str(PROMPT_PY), "--case", str(cases),
                              "--id", "nope"], capture_output=True, text=True)
    assert missing.returncode == 1 and "nope" in missing.stderr


# ------------------------------------------------- consumers use the module

def test_frontier_batch_renderer_uses_the_module_not_a_copy():
    import make_frontier_batches as mfb
    assert mfb.SYSTEM_TEXT is prompt.SYSTEM_TEXT
    assert mfb.render_case(ROW) == "### t-1\n" + render_user(
        ROW["instruction"], ROW["buffer_before"], ROW["cursor"])


def test_frontier_ingester_holds_replies_to_the_same_grammar(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps({"id": i}) + "\n"
                             for i in ("a-1", "a-2", "a-3", "a-4")),
                     encoding="utf-8")
    reply = tmp_path / "reply.txt"
    reply.write_text(
        "a-1 ex: %s/foo/bar/g\n"
        "  a-2 keys:ciwfoo<Esc>   \n"          # tolerated whitespace
        "a-3 EX: dd\n"                          # format failure: case
        "a-4 the answer is ex: dd\n"            # format failure: text first
        "zzz ex: dd\n"                          # unknown id
        "a-1 ex: something else\n",             # duplicate: first wins
        encoding="utf-8")
    out = tmp_path / "preds.jsonl"
    run = subprocess.run(
        [PYTHON, str(REPO / "scripts" / "ingest_frontier_preds.py"),
         str(reply), "--cases", str(cases), "--out", str(out)],
        capture_output=True, text=True, check=True)
    preds = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert preds == [
        {"id": "a-1", "command": "%s/foo/bar/g", "mode": "ex"},
        {"id": "a-2", "command": "ciwfoo<Esc>", "mode": "keys"},
    ]
    assert "3 unparsed/unknown lines" in run.stdout
    assert "1 duplicate" in run.stdout
    assert "2 cases with no answer" in run.stdout
