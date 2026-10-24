"""Sandbox escape battery — filesystem writes from an in-scope vim command.

Found 2026-08-18 while auditing whether named registers were safe to admit.
They were; these were not. Sixteen commands, none of them using `:w`, none
using a screened scripting word, and none blocked by `-Z`, wrote to an
arbitrary absolute path outside the sandbox. Four of them (`:redir`,
`:profile`, `:hardcopy`, `:helptags`) needed no expression at all — they were
simply command words nobody had enumerated. The rest reach `writefile()`
through a vimscript expression context: `:put =`, `:echo`, `:s//\\=/`,
control-flow blocks, and the insert-mode expression register `<C-r>=`.

Each test asserts on a CANARY DIRECTORY outside the sandbox: after the
command runs, nothing may appear or change there. That is the property that
matters -- whether the screen refuses the command or the OS (operating system) boundary stops
it, the observable outcome must be the same.

Load-bearing, like the rest of the adversarial battery: these encode a real
breach, not a hypothetical one.
"""

import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from executor import run_vim          # noqa: E402
from scope import screen_dangerous    # noqa: E402

BUFFER = "hello\nworld"


@pytest.fixture()
def canary(tmp_path):
    """A directory outside the sandbox that nothing may touch."""
    d = tmp_path / "canary"
    d.mkdir()
    (d / "existing.txt").write_text("ORIGINAL")
    return d


def _writefile(path):
    return "writefile(['pwned'], '%s')" % path


def _snapshot(d):
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


# (label, mode, builder) -- builder(canary) -> command string
ESCAPES = [
    ("put-expression",        "ex",   lambda c: "put =" + _writefile(c / "a.txt")),
    ("put-abbreviated",       "ex",   lambda c: "pu=" + _writefile(c / "b.txt")),
    ("put-with-range",        "ex",   lambda c: "2pu=" + _writefile(c / "c.txt")),
    ("substitute-expression", "ex",   lambda c: "s#hello#\\=" + _writefile(c / "d.txt") + "#"),
    ("global-drives-put",     "ex",   lambda c: "g/hello/pu=" + _writefile(c / "e.txt")),
    ("echo",                  "ex",   lambda c: "echo " + _writefile(c / "f.txt")),
    ("echomsg",               "ex",   lambda c: "echom " + _writefile(c / "g.txt")),
    ("if-block",              "ex",   lambda c: "if 1 | pu=" + _writefile(c / "h.txt") + " | endif"),
    ("for-loop",              "ex",   lambda c: "for i in [1] | pu=" + _writefile(c / "i.txt") + " | endfor"),
    ("redir",                 "ex",   lambda c: "redir > %s" % (c / "j.txt")),
    ("profile",               "ex",   lambda c: "profile start %s" % (c / "k.txt")),
    ("hardcopy",              "ex",   lambda c: "hardcopy > %s" % (c / "l.ps")),
    ("helptags",              "ex",   lambda c: "helptags %s" % c),
    ("expression-register",   "keys", lambda c: "i<C-r>=" + _writefile(c / "m.txt") + "<CR><Esc>"),
    ("expression-register-2", "keys", lambda c: "a<C-r><C-r>=" + _writefile(c / "n.txt") + "<CR><Esc>"),
    ("openline-expression",   "keys", lambda c: "o<C-r>=" + _writefile(c / "o.txt") + "<CR><Esc>"),
]


@pytest.mark.parametrize("label,mode,build", ESCAPES, ids=[e[0] for e in ESCAPES])
def test_command_cannot_write_outside_the_sandbox(label, mode, build, canary):
    """The observable property: the filesystem outside the sandbox is untouched."""
    before = _snapshot(canary)
    run_vim(BUFFER, build(canary), mode=mode, cursor=(1, 1), timeout=5.0)
    assert _snapshot(canary) == before, "%s escaped the sandbox and wrote to disk" % label


@pytest.mark.parametrize("label,mode,build", ESCAPES, ids=[e[0] for e in ESCAPES])
def test_escape_is_refused_by_the_screen(label, mode, build, canary):
    """Defense in depth: the screen should also refuse, before vim is spawned."""
    assert screen_dangerous(build(canary), mode) is not None, \
        "%s passed the static screen" % label


# --------------------------------------------------------- regressions

@pytest.mark.parametrize("command", [
    "call writefile(['x'], '/tmp/x')",
    "let x = 1",
    'execute "!touch /tmp/x"',
    "normal ihello",
    "source /etc/passwd",
    "w! /tmp/x",
    "set write | w! /tmp/x",
    "r !id",
])
def test_previously_screened_commands_stay_screened(command):
    assert screen_dangerous(command, "ex") is not None


def test_shell_is_still_blocked_inside_an_expression(canary):
    """-Z holds the shell; this must not regress when expressions are gated."""
    before = _snapshot(canary)
    run_vim(BUFFER, "s#hello#\\=system('touch %s/shell.txt')#" % canary, mode="ex")
    assert _snapshot(canary) == before


# --------------------------------------------------------- what stays allowed

@pytest.mark.parametrize("mode,command", [
    ("ex", r"%s/\d\+/\=submatch(0)+3/"),
    ("ex", r"1,2s/\%>1c./\=toupper(submatch(0))/g"),
    ("ex", "%s/hello/goodbye/"),
    ("ex", "2,3d"),
    ("ex", "put"),
    ("keys", "dw"),
    ("keys", '"ayy"ap'),
    ("keys", "qaddq"),
])
def test_in_scope_commands_are_not_screened(mode, command):
    """Pure expressions and the newly admitted registers/macro recording stay usable.

    The `\\=` allowlist exists so the increment and case categories survive; a
    blanket ban on expressions would take them out.
    """
    assert screen_dangerous(command, mode) is None, "%r was wrongly screened" % command


# --------------------------------------------------------- concurrency

def test_isolation_probe_is_not_defeated_by_concurrency():
    """A thread pool must not race the probe into reporting 'unavailable'.

    Regression for a bug shipped 2026-08-18: isolation_available() cached
    False *before* running its ~200 ms (milliseconds) probe, so under the sec 9 GRPO (Group Relative Policy Optimization) reward
    pool (~8 workers) every thread but the first read that False and ran vim
    with no OS boundary at all.
    """
    import importlib
    from concurrent.futures import ThreadPoolExecutor

    import executor as ex_mod
    importlib.reload(ex_mod)  # clear the cached probe

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: ex_mod.isolation_available(), range(8)))

    assert len(set(results)) == 1, (
        "isolation_available() disagreed across threads: %r" % (results,)
    )


# ------------------------------------------- screen-parser gaps (2026-08-26)
#
# Found while assessing whether the mined visual-mode rows were safe to admit
# to scope v2. Three separate gaps let damage-capable commands past
# src/scope.py. None of them ESCAPED -- 'nowrite' refused every write and -Z
# refused every shell -- so these are hardening regressions, not a breach
# like the 2026-08-18 sixteen above. They are load-bearing anyway, because
# CLAUDE.md §5 makes the screen the layer the GRPO reward loop, the MCP (Model Context Protocol)
# server and the neovim plugin inherit, and those are exactly the places
# where a MODEL's output, not a curated row, reaches the executor.
#
#   1. _EX_HEAD's address part consumed "'" as a bare range character, so the
#      mark's second character became the command word: "'a,'bw! /path"
#      screened as command 'a' and "'<,'>w! /path" as command '<' (because
#      [<>]+ is a command-word branch). Verified at the time:
#      ":1mark a|2mark b|'a,'bd" executed with the screen seeing nothing.
#      "'<,'>" is what vim inserts the instant ':' is pressed in visual mode.
#   2. ex_fragments_from_keys() keys on a literal ':', but the filter operator
#      '!' opens the command line without one -- "Vj!cmd<CR>" was unscreened.
#   3. The Ex-mode pattern was anchored (r"^g?Q"), so only a command STARTING
#      with Q/gQ was screened; "VjgQ..." and "ddgQ..." walked through.

MARK_RANGE_BYPASSES = [
    ("visual-range-write",     "ex",   lambda c: "'<,'>w! %s" % (c / "vr.txt")),
    ("visual-range-redir",     "ex",   lambda c: "'<,'>redir > %s" % (c / "vd.txt")),
    ("mark-range-write",       "ex",   lambda c: "1mark a|2mark b|'a,'bw! %s" % (c / "mw.txt")),
    ("mark-range-k-write",     "ex",   lambda c: "1ka|2kb|'a,'bw! %s" % (c / "kw.txt")),
    ("mark-range-put-expr",    "ex",   lambda c: "1mark a|2mark b|'a,'bput =" + _writefile(c / "mp.txt")),
    ("mark-range-normal",      "ex",   lambda c: "'a,'bnormal ihello"),
    ("single-mark-write",      "ex",   lambda c: "'aw! %s" % (c / "sm.txt")),
    ("visual-filter",          "keys", lambda c: "Vj!tee %s<CR>" % (c / "vf.txt")),
    ("normal-filter",          "keys", lambda c: "!!tee %s<CR>" % (c / "nf.txt")),
    ("visual-block-filter",    "keys", lambda c: "<C-v>j!tee %s<CR>" % (c / "bf.txt")),
    ("ex-mode-after-visual",   "keys", lambda c: "VjgQw! %s<CR>vi<CR>" % (c / "eq.txt")),
    ("ex-mode-after-delete",   "keys", lambda c: "ddgQw! %s<CR>vi<CR>" % (c / "ed.txt")),
    ("ex-mode-bare-Q-suffix",  "keys", lambda c: "VjQw! %s<CR>vi<CR>" % (c / "bq.txt")),
]


@pytest.mark.parametrize("label,mode,build", MARK_RANGE_BYPASSES,
                         ids=[e[0] for e in MARK_RANGE_BYPASSES])
def test_screen_parser_gap_is_refused_by_the_screen(label, mode, build, canary):
    """The screen must see the command word through a mark range or a filter."""
    assert screen_dangerous(build(canary), mode) is not None, \
        "%s passed the static screen" % label


@pytest.mark.parametrize("label,mode,build", MARK_RANGE_BYPASSES,
                         ids=[e[0] for e in MARK_RANGE_BYPASSES])
def test_screen_parser_gap_writes_nothing(label, mode, build, canary):
    """And the observable property holds regardless of which layer refuses."""
    before = _snapshot(canary)
    run_vim(BUFFER, build(canary), mode=mode, cursor=(1, 1), timeout=5.0)
    assert _snapshot(canary) == before, "%s wrote outside the sandbox" % label


@pytest.mark.parametrize("command,word", [
    ("w! /tmp/x", "w"),
    ("1,2w! /tmp/x", "w"),
    ("'<,'>w! /tmp/x", "w"),
    ("'a,'bw! /tmp/x", "w"),
    ("'<,'>!echo hi", "!"),
    ("''d", "d"),
    ("%s/a/b/", "s"),
    (".,+3d", "d"),
    ("'a,'bnormal dd", "normal"),
])
def test_ex_head_finds_the_real_command_word(command, word):
    """The parser-level property the screen depends on.

    A mark must be consumed as an ADDRESS (two characters), never split so its
    letter becomes the command. This is the assertion that fails first if
    someone puts "'" back into the plain character class.
    """
    import scope
    m = scope._EX_HEAD.match(command)
    assert m is not None, "unparseable: %r" % command
    assert m.group(1) == word, \
        "%r parsed as command %r, expected %r" % (command, m.group(1), word)


# ------------------------------------- the mode view must not over-screen
#
# Screening '!' and 'Q' anywhere would refuse ordinary rows: 12 corpus cases
# type a literal '!' inside insert mode and 4 type a literal 'Q'. These are
# real gold commands from data/combined_cases.jsonl -- if this test fails,
# the screen has started rejecting validated corpus rows.

@pytest.mark.parametrize("command", [
    "i!",                                        # emul-vsc-0031
    "I!",                                        # emul-vsc-0032
    "a!",                                        # emul-vsc-0033
    "/color:<CR>$i !important<Esc>",             # hc-1612
    "ci[the FAQ<Esc>",                           # hc-1409
    "oQ2<CR>Q3<CR>Q4<Esc>",                      # hc-1483
    "2l<C-Q>2j$d",                               # cand-vsv-0062 -- Q inside a token
    "O        guard !text.isEmpty else { return }<Esc>",   # hc-2635
    "cwQ<Esc>",                                  # change-to-insert then a literal Q
    "s!<Esc>",                                   # substitute-char enters insert
])
def test_insert_mode_literals_are_not_screened(command):
    assert screen_dangerous(command, "keys") is None, \
        "%r is insert-mode text, not a normal-mode command" % command


@pytest.mark.parametrize("command", [
    "Vj!sort<CR>",          # visual filter
    "!!sort<CR>",           # normal filter, whole line
    "!japply<CR>",          # normal filter over a motion
    "<C-v>j!sort<CR>",      # visual-block filter
    "dd!!sort<CR>",         # filter after an edit
    "<Esc>!!sort<CR>",      # filter after leaving insert
])
def test_filter_operator_is_screened_in_normal_and_visual(command):
    assert screen_dangerous(command, "keys") == "shell filter (!{motion})"


# ------------------------------- mode-tracker exits (re-audit, 2026-08-26)
#
# The mode view BLANKS text, so every insert-mode exit it does not know about
# is a place the screen stopped looking. The first version of
# _normal_mode_view() knew only <Esc> and <C-c>, and a re-audit immediately
# found three ways to smuggle a macro past a screen that had caught it before
# the mode view existed at all:
#
#   ihi<C-[>@a    <C-[> IS <Esc> -- same key, different spelling
#   ihi<C-o>@a    <C-o> runs one normal command from insert, then returns
#   ihi<C-o>ZZ    ditto, with save/quit
#
# plus the raw control bytes, which reach the executor through the same path
# as their <...> spellings. Each entry below is a keystroke sequence where vim
# genuinely leaves insert mode and executes what follows.

@pytest.mark.parametrize("command", [
    "ihi<Esc>@a",
    "ihi<C-[>@a",
    "ihi<C-c>@a",
    "ihi<C-\\><C-n>@a",
    "ihi\x1b@a",
    "ihi\x03@a",
    "ihi<C-o>@a",
    "ihi\x0f@a",
    "ihi<C-o>ZZ",
    "ihi<C-[>!!sort<CR>",
    "Rxx<Esc>@a",
    "ahi<C-[>ZQ",
])
def test_insert_mode_exits_are_all_tracked(command):
    """If any of these stops being screened, the walker is stuck in insert."""
    assert screen_dangerous(command, "keys") is not None, \
        "%r left insert mode but the screen kept blanking" % command


def test_unterminated_insert_is_still_literal_text():
    """The other direction: with no exit, the tail really is typed text."""
    assert screen_dangerous("ihello@a", "keys") is None
    assert screen_dangerous("ci[the FAQ", "keys") is None
