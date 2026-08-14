"""Test suite for the vimBot executor and dataset.

Two layers:
  1. Gold self-check — every seed case's own gold command must reproduce
     its expected buffer (validates dataset and executor against each other).
  2. Adversarial battery — the sandbox and failure-detection guarantees.
     These encode real bugs found during development (-Z does not block
     :w to arbitrary paths; the Esc-recovery suffix can mask hangs), so
     they must stay green after any executor change.

Run:  python3 -m pytest tests/ -v
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from executor import convert_keys, run_vim, run_vim_on_file

REPO = Path(__file__).resolve().parents[1]
CASES = [
    json.loads(line)
    for line in (REPO / "data" / "seed_cases.jsonl").read_text().splitlines()
    if line.strip()
]


def _norm(text):
    if text is None:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


# ---------------------------------------------------------------- gold cases

@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_gold_case_executes_to_expected_buffer(case):
    result = run_vim(
        case["buffer_before"],
        case["gold_command"],
        mode=case.get("mode", "ex"),
        cursor=tuple(case.get("cursor", [1, 1])),
    )
    assert result["buffer_after"] is not None
    assert _norm(result["buffer_after"]) == _norm(case["buffer_after"]), (
        f"{case['id']}: gold command did not reproduce expected buffer"
    )


# ---------------------------------------------------------- adversarial battery

BUF = "the cat sat\na cat and a cat\nno animals here"


def test_wrong_but_valid_command_yields_mismatch():
    # Missing /g: executes cleanly but produces the wrong buffer.
    r = run_vim(BUF, "%s/cat/dog/", mode="ex")
    assert r["was_valid"] is True
    assert "a dog and a cat" in r["buffer_after"]  # second 'cat' untouched


def test_invalid_syntax_is_marked_invalid():
    r = run_vim(BUF, "%s/unclosed", mode="ex")
    assert r["was_valid"] is False


def test_shell_escape_is_blocked():
    r = run_vim(BUF, "!touch /tmp/vimBot_pwned", mode="ex")
    assert r["was_valid"] is False
    assert not os.path.exists("/tmp/vimBot_pwned")


def test_write_to_arbitrary_path_is_blocked_ex_mode():
    r = run_vim(BUF, "w! /tmp/vimBot_exfil.txt", mode="ex")
    assert r["was_valid"] is False
    assert not os.path.exists("/tmp/vimBot_exfil.txt")


def test_write_to_arbitrary_path_is_blocked_keys_mode():
    run_vim(BUF, ":w! /tmp/vimBot_exfil2.txt<CR>", mode="keys")
    assert not os.path.exists("/tmp/vimBot_exfil2.txt")


def test_infinite_loop_hits_timeout():
    t0 = time.time()
    r = run_vim(BUF, "while 1 | endwhile", mode="ex", timeout=3)
    assert r["timeout"] is True
    assert time.time() - t0 >= 2.5


def test_buffer_is_recoverable_after_partial_keys():
    # Lone ':' would await input; the Esc-recovery suffix must rescue it
    # and leave the buffer unchanged rather than hanging.
    r = run_vim(BUF, ":", mode="keys", timeout=3)
    assert r["timeout"] is False
    assert _norm(r["buffer_after"]) == _norm(BUF)


def test_executor_runs_a_real_vim_process():
    # v:version is vim's own internal version variable: a reimplementation
    # could not produce it. This pins "we shell out to real vim" in the suite.
    r = run_vim("marker\n", "put =v:version", mode="ex")
    assert r["was_valid"] is True
    lines = _norm(r["buffer_after"])
    assert lines[0] == "marker"
    assert lines[1].isdigit() and int(lines[1]) >= 700


# ------------------------------------------------------ pinned option baseline
#
# -u NONE skips vim's own defaults.vim, whose line 60 is
# `set nrformats-=octal`. Left alone, the sandbox reads 007 as octal
# (<C-a> -> 010) while every editor a user actually runs -- vim loading
# defaults.vim, and neovim -- gives 008. That is a silent train/serve
# mismatch: the GRPO reward oracle, the MCP server and the neovim plugin
# would disagree about a single keystroke. executor._OPTION_BASELINE pins the
# behaviour the real editors share; these two tests pin it in both modes.

def test_increment_does_not_read_leading_zero_as_octal():
    r = run_vim("code 007", "<C-a>", mode="keys")
    assert _norm(r["buffer_after"]) == ["code 008"], (
        "leading zero was read as octal: 'set nrformats-=octal' is not "
        "reaching keys mode"
    )


def test_option_baseline_reaches_ex_mode_too():
    # The baseline lives in one constant precisely so the two code paths
    # cannot drift. This is the test that fails when an option is added to
    # one of them and forgotten in the other.
    r = run_vim("marker", "put =&nrformats", mode="ex")
    assert r["was_valid"] is True
    nrformats = _norm(r["buffer_after"])[1]
    assert "octal" not in nrformats, f"octal still enabled in ex mode: {nrformats}"
    assert "hex" in nrformats, f"baseline clobbered nrformats entirely: {nrformats}"


# ---------------------------------------------------- sandbox-escape screening
#
# Probed empirically 2026-08-11: 'nowrite' alone does NOT hold. A command
# containing 'set write' re-enables writing and the following :w! succeeds --
# in BOTH modes, and in ex mode it even returned rc=0 (reported valid). These
# pin the screen that now runs before vim is ever spawned.

ESCAPES = [
    ("ex", "set write | w! {t}"),
    ("ex", "set write|w! {t}"),
    ("ex", "se write|w! {t}"),
    ("ex", "setlocal write|w! {t}"),
    ("ex", "let &write=1|w! {t}"),
    ("ex", "w! {t}"),
    ("ex", "g/cat/w! {t}"),
    ("keys", ":w! {t}<CR>"),
    ("keys", ":set write<CR>:w! {t}<CR>"),
    ("keys", "dd:set write<CR>:w! {t}<CR>"),   # not at the start -- was missed
    ("keys", ":set write|w! {t}<CR>"),
]


@pytest.mark.parametrize("mode,template", ESCAPES,
                         ids=[f"{m}:{t}" for m, t in ESCAPES])
def test_write_escape_is_blocked_before_vim_runs(mode, template, tmp_path):
    target = tmp_path / "escaped.txt"
    r = run_vim(BUF, template.format(t=target), mode=mode)
    assert r["was_valid"] is False
    assert r.get("blocked"), "should be refused by the screen, not by vim"
    assert not target.exists(), "command escaped the sandbox and wrote a file"


@pytest.mark.parametrize("mode,command", [
    ("ex", "%s/cat/dog/g"),
    ("ex", "sort"),
    ("ex", "g/^the/d"),
    ("ex", "%s/write/read/g"),      # 'write' as substitute text, not an option
    ("ex", "%s/set write/x/"),      # ditto, must not false-positive
    ("keys", "dd"),
    ("keys", 'ci"x<Esc>'),
    ("keys", "/cat<CR>dw"),
    ("keys", "oNew line<Esc>"),
])
def test_legitimate_commands_are_not_blocked(mode, command):
    r = run_vim(BUF, command, mode=mode)
    assert r.get("blocked") is None, f"false positive on {command!r}"


def test_reading_another_file_is_blocked():
    # exfiltration vector: pull a file into the buffer, which the caller reads
    r = run_vim(BUF, "r /etc/passwd", mode="ex")
    assert r.get("blocked")
    assert r["buffer_after"] is None


# ------------------------------------------------------------------ file input

def test_file_input_applies_command(tmp_path):
    src = tmp_path / "buf.txt"
    src.write_text("the cat sat\na cat too\n")
    r = run_vim_on_file(src, "%s/cat/dog/g", mode="ex")
    assert r["was_valid"] is True
    assert _norm(r["buffer_after"]) == ["the dog sat", "a dog too"]
    assert r["source_path"] == str(src)
    assert r["written_to"] is None


def test_file_input_never_modifies_the_source(tmp_path):
    src = tmp_path / "buf.txt"
    original = "keep me\ndelete me\nkeep me too\n"
    src.write_text(original)
    r = run_vim_on_file(src, "2d", mode="ex")
    assert _norm(r["buffer_after"]) == ["keep me", "keep me too"]
    assert src.read_text() == original  # the whole point


def test_file_input_keys_mode(tmp_path):
    src = tmp_path / "buf.txt"
    src.write_text("alpha\nbravo\ncharlie\n")
    r = run_vim_on_file(src, "dd", mode="keys", cursor=(2, 1))
    assert _norm(r["buffer_after"]) == ["alpha", "charlie"]


def test_file_input_accepts_str_and_path(tmp_path):
    src = tmp_path / "buf.txt"
    src.write_text("x\n")
    from_path = run_vim_on_file(src, "%s/x/y/", mode="ex")
    src.write_text("x\n")
    from_str = run_vim_on_file(str(src), "%s/x/y/", mode="ex")
    assert _norm(from_path["buffer_after"]) == _norm(from_str["buffer_after"]) == ["y"]


def test_file_input_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_vim_on_file(tmp_path / "nope.txt", "%s/a/b/", mode="ex")


def test_file_input_sandbox_still_blocks_writes(tmp_path):
    src = tmp_path / "buf.txt"
    src.write_text("data\n")
    target = tmp_path / "exfil.txt"
    r = run_vim_on_file(src, f"w! {target}", mode="ex")
    assert r["was_valid"] is False
    assert not target.exists()
    assert src.read_text() == "data\n"


def test_file_input_output_path_writes_result(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\ntwo\n")
    out = tmp_path / "out.txt"
    r = run_vim_on_file(src, "%s/two/three/", mode="ex", output_path=out)
    assert r["written_to"] == str(out)
    assert out.read_text() == "one\nthree\n"
    assert src.read_text() == "one\ntwo\n"


def test_file_input_output_path_may_not_be_the_source(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    with pytest.raises(ValueError):
        run_vim_on_file(src, "%s/one/two/", mode="ex", output_path=src)
    assert src.read_text() == "one\n"


def test_output_refuses_an_existing_file_by_default(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    out = tmp_path / "out.txt"
    out.write_text("PRECIOUS\n")
    with pytest.raises(FileExistsError):
        run_vim_on_file(src, "%s/one/two/", mode="ex", output_path=out)
    assert out.read_text() == "PRECIOUS\n"   # untouched


def test_output_overwrite_mode_replaces(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    out = tmp_path / "out.txt"
    out.write_text("old\n")
    r = run_vim_on_file(src, "%s/one/two/", mode="ex", output_path=out,
                        on_exists="overwrite")
    assert r["written_to"] == str(out)
    assert out.read_text() == "two\n"


def test_output_suffix_mode_never_overwrites_and_counts_up(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    out = tmp_path / "out.txt"

    # free path: uses it as given
    r1 = run_vim_on_file(src, "%s/one/A/", mode="ex", output_path=out,
                         on_exists="suffix")
    assert r1["written_to"] == str(out)

    # taken: next free number, extension preserved
    r2 = run_vim_on_file(src, "%s/one/B/", mode="ex", output_path=out,
                         on_exists="suffix")
    assert r2["written_to"] == str(tmp_path / "out_1.txt")

    # taken again: counts up rather than clobbering out_1.txt
    r3 = run_vim_on_file(src, "%s/one/C/", mode="ex", output_path=out,
                         on_exists="suffix")
    assert r3["written_to"] == str(tmp_path / "out_2.txt")

    # every earlier result survives -- this is the property _temp lacked
    assert out.read_text() == "A\n"
    assert (tmp_path / "out_1.txt").read_text() == "B\n"
    assert (tmp_path / "out_2.txt").read_text() == "C\n"


def test_output_suffix_goes_before_the_extension(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    out = tmp_path / "report.txt"
    out.write_text("existing\n")
    r = run_vim_on_file(src, "%s/one/two/", mode="ex", output_path=out,
                        on_exists="suffix")
    # report_1.txt, NOT report.txt_1 -- globs and file-type detection survive
    assert r["written_to"].endswith("report_1.txt")


def test_unknown_on_exists_is_rejected(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    with pytest.raises(ValueError):
        run_vim_on_file(src, "%s/one/two/", mode="ex",
                        output_path=tmp_path / "out.txt", on_exists="clobber")


def test_file_input_writes_nothing_when_command_fails(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    out = tmp_path / "out.txt"
    r = run_vim_on_file(src, "%s/unclosed", mode="ex", output_path=out)
    assert r["was_valid"] is False
    assert r["written_to"] is None
    assert not out.exists()


# ------------------------------------------------------------------------ CLI

EXECUTOR = REPO / "src" / "executor.py"


def test_cli_reads_a_file_and_prints_result(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("the cat sat\n")
    proc = subprocess.run(
        [sys.executable, str(EXECUTOR), "--file", str(src),
         "--command", "%s/cat/dog/g"],
        capture_output=True, text=True,
    )
    assert proc.returncode== 0
    assert proc.stdout == "the dog sat\n"
    assert src.read_text() == "the cat sat\n"  # source untouched


def test_cli_output_flag_and_failure_exit_code(tmp_path):
    src = tmp_path / "in.txt"
    src.write_text("a\n")
    out = tmp_path / "out.txt"
    ok = subprocess.run(
        [sys.executable, str(EXECUTOR), "--file", str(src),
         "--command", "%s/a/b/", "--output", str(out)],
        capture_output=True, text=True,
    )
    assert ok.returncode == 0 and out.read_text() == "b\n"

    bad = subprocess.run(
        [sys.executable, str(EXECUTOR), "--text", "a\n",
         "--command", "%s/unclosed"],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1


def _cli(*extra, tmp_path):
    src = tmp_path / "in.txt"
    if not src.exists():
        src.write_text("one\n")
    return subprocess.run(
        [sys.executable, str(EXECUTOR), "--file", str(src),
         "--command", "%s/one/two/", *extra],
        capture_output=True, text=True,
    )


def test_cli_refuses_existing_output_by_default(tmp_path):
    out = tmp_path / "out.txt"
    out.write_text("PRECIOUS\n")
    p = _cli("--output", str(out), tmp_path=tmp_path)
    assert p.returncode == 2, "distinct exit code: command ok, output not written"
    assert "already exists" in p.stderr
    assert out.read_text() == "PRECIOUS\n"


def test_cli_on_exists_suffix_and_overwrite(tmp_path):
    out = tmp_path / "out.txt"
    out.write_text("PRECIOUS\n")

    suf = _cli("--output", str(out), "--on-exists", "suffix", tmp_path=tmp_path)
    assert suf.returncode == 0
    assert (tmp_path / "out_1.txt").read_text() == "two\n"
    assert out.read_text() == "PRECIOUS\n"      # still not clobbered

    ovr = _cli("--output", str(out), "--on-exists", "overwrite", tmp_path=tmp_path)
    assert ovr.returncode == 0
    assert out.read_text() == "two\n"


# ------------------------------------------------------------- key translation

def test_translate_named_keys():
    assert convert_keys("ci\"error<Esc>") == 'ci"error\x1b'
    assert convert_keys("<C-a>") == "\x01"
    assert convert_keys("a<lt>b") == "a<b"
