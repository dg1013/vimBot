"""Command modifiers and the scope screen (found 2026-11-02, fixed 2026-11-04).

A modifier runs the command after it, and the screen stopped at the first word,
so `keepjumps w! /path` screened as `keepjumps` and passed. Each case asserts both
that the screen refuses it and that nothing outside the sandbox changed.
"""

import pytest

from executor import run_vim
from scope import screen_dangerous

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


# Together with the old wrapper's writable submounts, `keepjumps pu=writefile(...)`
# reached /mnt/c, so these are a real escape and not a hypothetical one.

MODIFIERS = [
    ("aboveleft", 3), ("belowright", 3), ("botright", 2), ("browse", 3), ("confirm", 4),
    ("hide", 3), ("keepalt", 3), ("keepjumps", 3), ("keepmarks", 3), ("keeppatterns", 3),
    ("leftabove", 5), ("legacy", 3), ("lockmarks", 3), ("noautocmd", 3), ("noswapfile", 3),
    ("rightbelow", 6), ("tab", 3), ("topleft", 2), ("unsilent", 3), ("verbose", 4),
    ("vertical", 4), ("vim9cmd", 4),
]
MODIFIER_FORMS = [word[:n] for word, shortest in MODIFIERS for n in range(shortest, len(word) + 1)]

MODIFIER_BYPASSES = [
    ("write",              "ex",   lambda c: "keepjumps w! %s" % (c / "mw.txt")),
    ("write-abbreviated",  "ex",   lambda c: "keepj w! %s" % (c / "mx.txt")),
    ("write-shortest",     "ex",   lambda c: "to w! %s" % (c / "my.txt")),
    ("redir",              "ex",   lambda c: "noautocmd redir > %s" % (c / "mr.txt")),
    ("put-expression",     "ex",   lambda c: "keepjumps pu=" + _writefile(c / "mp.txt")),
    ("counted",            "ex",   lambda c: "3verbose w! %s" % (c / "mc.txt")),
    ("chained",            "ex",   lambda c: "tab keepalt vert w! %s" % (c / "mh.txt")),
    ("vim9cmd",            "ex",   lambda c: "vim9cmd w! %s" % (c / "m9.txt")),
    ("after-a-bar",        "ex",   lambda c: "s/hello/x/ | keepj w! %s" % (c / "mb.txt")),
    ("after-global",       "ex",   lambda c: "g/hello/keepj w! %s" % (c / "mg.txt")),
    ("from-keys",          "keys", lambda c: ":keepj w! %s<CR>" % (c / "mk.txt")),
    ("read-another-file",  "ex",   lambda c: "keepjumps r /etc/hostname"),
    ("normal",             "ex",   lambda c: "keepjumps normal ihello"),
    ("silent-abbreviated", "ex",   lambda c: "sile w! %s" % (c / "ms.txt")),
    ("sandbox-abbreviated", "ex",  lambda c: "san w! %s" % (c / "mn.txt")),
]


@pytest.mark.parametrize("label,mode,build", MODIFIER_BYPASSES,
                         ids=[e[0] for e in MODIFIER_BYPASSES])
def test_modifier_is_refused_by_the_screen(label, mode, build, canary):
    assert screen_dangerous(build(canary), mode) is not None, \
        "%s passed the static screen" % label


@pytest.mark.parametrize("label,mode,build", MODIFIER_BYPASSES,
                         ids=[e[0] for e in MODIFIER_BYPASSES])
def test_modifier_writes_nothing(label, mode, build, canary):
    before = _snapshot(canary)
    run_vim(BUFFER, build(canary), mode=mode, cursor=(1, 1), timeout=5.0)
    assert _snapshot(canary) == before, "%s wrote outside the sandbox" % label


@pytest.mark.parametrize("form", MODIFIER_FORMS)
def test_every_abbreviation_of_a_modifier_is_seen_through(form):
    assert screen_dangerous(form + " normal ix", "ex") is not None


@pytest.mark.parametrize("form", MODIFIER_FORMS)
def test_a_modifier_on_a_harmless_command_stays_usable(form):
    # Refusing modifiers outright would lose `keeppatterns %s/a/b/`.
    assert screen_dangerous(form + " s/a/b/", "ex") is None


def test_a_bare_modifier_word_is_not_an_error():
    assert screen_dangerous("hide", "ex") is None


def test_a_long_modifier_chain_cannot_raise():
    # A recursion here would abort the caller's whole run, like the NUL byte did.
    assert screen_dangerous("tab " * 5000 + "w! /tmp/x", "ex") is not None
    assert screen_dangerous("tab " * 5000 + "s/a/b/", "ex") is None
