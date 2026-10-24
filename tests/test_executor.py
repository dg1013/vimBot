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
SEED = REPO / "data" / "seed_cases.jsonl"
CASES = [
    json.loads(line)
    for line in SEED.read_text().splitlines()
    if line.strip()
] if SEED.exists() else []


def _norm(text):
    if text is None:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


# ---------------------------------------------------------------- gold cases

@pytest.mark.skipif(not SEED.exists(), reason=f"{SEED} not present")
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
    # `while 1 | endwhile` used to be the hang here; :while became a screened
    # expression context on 2026-08-18, so it no longer reaches vim. The
    # invariant under test is the hard wall clock, not the hanging command --
    # :sleep hangs just as well and is still in scope.
    t0 = time.time()
    r = run_vim(BUF, "sleep 30", mode="ex", timeout=3)
    assert r["timeout"] is True
    assert time.time() - t0 >= 2.5


def test_vimscript_loop_is_screened_before_it_can_hang():
    r = run_vim(BUF, "while 1 | endwhile", mode="ex", timeout=3)
    assert r["blocked"] is not None
    assert r["was_valid"] is False


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
# mismatch: the GRPO (Group Relative Policy Optimization) reward oracle, the MCP (Model Context Protocol) server and the neovim plugin
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


# A MEASURED LIMIT, not a pin (2026-08-26). With textwidth at its default 0,
# `gq` wraps at min(columns, 79), so reflow gold depends on the COLUMNS
# environment variable rather than on any vim setting. Pinning textwidth in
# _OPTION_BASELINE was tried and rejected -- it also drives insert-mode
# auto-wrap and broke hc-0663's `cis`. The convention is that a reflow case
# states its own width. These two tests pin both halves: the hazard, so nobody
# rediscovers it, and the fix that dataset rows are expected to use.

_LONG = ("The importer reads the whole file into memory before validating any "
         "of it, which is fine for the sample data but falls over in prod.")


def _reflow_at(columns, keys):
    import os
    before = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = columns
    try:
        return _norm(run_vim(_LONG, keys, mode="keys")["buffer_after"])
    finally:
        if before is None:
            os.environ.pop("COLUMNS", None)
        else:
            os.environ["COLUMNS"] = before


def test_bare_gq_wrap_depends_on_terminal_width():
    """Known limit: a gq gold with no explicit textwidth is not portable."""
    narrow = _reflow_at("40", "gqq")
    wide = _reflow_at("100", "gqq")
    assert narrow != wide, (
        "bare gq no longer varies with COLUMNS -- if textwidth was pinned in "
        "_OPTION_BASELINE, check hc-0663 and the insert-edit category first, "
        "then delete this test and its note in executor.py"
    )
    assert all(len(line) <= 39 for line in narrow), narrow


def test_case_supplied_textwidth_makes_reflow_portable():
    """The convention reflow rows follow: state the width in the command."""
    at40 = _reflow_at("40", ":set tw=60\rgqq")
    at100 = _reflow_at("100", ":set tw=60\rgqq")
    assert at40 == at100, (at40, at100)
    assert all(len(line) <= 60 for line in at40), at40
    assert len(at40) > 1, at40


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


def test_executor_never_leaves_a_file_behind(tmp_path):
    # Owner decision 2026-09-04: the executor writes no files. A valid run
    # and a failing run against a file leave the directory holding exactly
    # the source, unchanged; the result exists only in the returned dict.
    src = tmp_path / "in.txt"
    src.write_text("one\ntwo\n")
    ok = run_vim_on_file(src, "%s/two/three/", mode="ex")
    assert ok["was_valid"] is True
    assert _norm(ok["buffer_after"]) == ["one", "three"]
    bad = run_vim_on_file(src, "%s/unclosed", mode="ex")
    assert bad["was_valid"] is False
    assert sorted(p.name for p in tmp_path.iterdir()) == ["in.txt"]
    assert src.read_text() == "one\ntwo\n"
    assert "written_to" not in ok and "written_to" not in bad


# ------------------------------------------------------------------------ CLI (command-line interface)

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


def test_cli_invalid_command_exits_1(tmp_path):
    bad = subprocess.run(
        [sys.executable, str(EXECUTOR), "--text", "a\n",
         "--command", "%s/unclosed"],
        capture_output=True, text=True,
    )
    assert bad.returncode == 1


def test_cli_has_no_write_flags(tmp_path):
    # The write path was removed 2026-09-04; the flags must not quietly
    # come back as no-ops. argparse rejects them with its usage exit code.
    src = tmp_path / "in.txt"
    src.write_text("one\n")
    for flags in (["--output", str(tmp_path / "out.txt")],
                  ["--on-exists", "overwrite"]):
        p = subprocess.run(
            [sys.executable, str(EXECUTOR), "--file", str(src),
             "--command", "%s/one/two/", *flags],
            capture_output=True, text=True,
        )
        assert p.returncode == 2 and "unrecognized arguments" in p.stderr
    assert sorted(q.name for q in tmp_path.iterdir()) == ["in.txt"]


# ------------------------------------------------------------- key translation

def test_translate_named_keys():
    assert convert_keys("ci\"error<Esc>") == 'ci"error\x1b'
    assert convert_keys("<C-a>") == "\x01"
    assert convert_keys("a<lt>b") == "a<b"


# --------------------------------------------------- diagnostics (2026-09-05)

def test_invalid_ex_command_reports_vims_exception():
    # The try/catch wrapper used to swallow the message; the §9 error analysis
    # needs the E-code. The exception is written inside the sandbox dir (the
    # one writable place) and read back -- no new write path anywhere else.
    r = run_vim("foo\n", "s/nothere/x/", mode="ex")
    assert r["was_valid"] is False
    assert "E486" in r["stderr"] and "nothere" in r["stderr"]
    r = run_vim("foo\n", "notacommand", mode="ex")
    assert r["was_valid"] is False and "E492" in r["stderr"]


def test_valid_ex_command_reports_no_exception():
    r = run_vim("foo\n", "s/foo/bar/", mode="ex")
    assert r["was_valid"] is True and "vim:" not in r["stderr"]


# ----------------------------------------- NUL byte in an ex command (2026-09-22)

@pytest.mark.parametrize("command", [
    "2\x00a",                  # the base model's sample at T = 1 that crashed a run
    "%s/cat/dog/g\x00",        # a right answer plus a stray NUL: must not become a match
    "%s/c\x00at/dog/",         # inside the pattern
    ":\x00%s/cat/dog/",        # after the colon; strip() does not remove a NUL
])
def test_ex_command_with_nul_is_invalid_not_a_crash(command):
    # Ex commands reach vim in argv, which cannot carry a NUL: subprocess
    # raised ValueError out of run_vim and aborted the caller's whole run.
    r = run_vim(BUF, command, mode="ex")
    assert r["was_valid"] is False and r["buffer_after"] is None
    assert r["timeout"] is False and "NUL" in r["stderr"]
    assert "blocked" not in r            # unrunnable, not dangerous


def test_nul_does_not_mask_the_scope_screen():
    # The screen answers first, so a dangerous command still reports blocked.
    r = run_vim(BUF, "w! /tmp/vimBot_nul_exfil.txt\x00", mode="ex")
    assert r["blocked"] is not None
    assert not os.path.exists("/tmp/vimBot_nul_exfil.txt")


def test_nul_in_keys_mode_is_still_a_key():
    # Keys travel through a file, where NUL is insert-mode CTRL-@: insert the
    # last inserted text and leave insert mode.
    r = run_vim("foo bar\nfoo\n", "ibar<Esc>jo\x00", mode="keys")
    assert _norm(r["buffer_after"]) == ["barfoo bar", "foo", "bar"]


def test_other_control_bytes_still_reach_vim_in_ex_mode():
    # Only NUL is special to argv; a literal Ctrl-A in a replacement runs.
    r = run_vim("foo\n", "s/foo/\x01/", mode="ex")
    assert r["was_valid"] is True and _norm(r["buffer_after"]) == ["\x01"]


def test_file_input_inherits_the_nul_verdict(tmp_path):
    p = tmp_path / "in.txt"
    p.write_text("foo\n")
    r = run_vim_on_file(p, "s/foo/\x00/")
    assert r["was_valid"] is False and "NUL" in r["stderr"]
    assert p.read_text() == "foo\n"
