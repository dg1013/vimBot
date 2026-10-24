r"""Embedded-conjunction rendering: differential pins against the oracle.

Every pattern the module renders is executed in the sandboxed vim over a
brute-forced buffer (all strings up to length 4 over a small adversarial
alphabet, plus targeted longer probes) and must delete exactly the lines
the span-DP (dynamic programming) oracle says are members. A rendering that disagrees with the
oracle on ANY string is a soundness bug, not a coverage gap.

Also pinned: the shapes the module must refuse (ops under a repeat,
variable tail with variable conjuncts), and that dsl2vim.generate is
untouched — this feature is opt-in fallback only.
"""
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "deep_regex"))
sys.path.insert(0, str(ROOT / "src"))

from dsl_ast import parse, matcher  # noqa: E402
from dsl2vim import generate  # noqa: E402
from embedded_conj import render_embedded, lenrange  # noqa: E402
from executor import run_vim  # noqa: E402

ALPHA = "a5!B"      # lower / digit / special / upper — one of each family


def universe(extra=()):
    out = [""]
    for n in (1, 2, 3, 4):
        out += ["".join(t) for t in itertools.product(ALPHA, repeat=n)]
    out += list(extra)
    return out


def check(pattern, extra=()):
    """Render; if renderable, vim's survivors must equal the oracle's."""
    r = render_embedded(pattern)
    if r is None:
        return None
    ast = parse(pattern)
    lines = [s for s in universe(extra) if s]      # vim buffers: no "" line
    member = {s: matcher(ast, s) for s in lines}
    cmd = (f"g/{r['exec']}/d" if r["pol"] == "g" else f"v/{r['exec']}/d")
    res = run_vim("\n".join(lines), cmd, mode="ex")
    assert res["was_valid"], f"{pattern}: vim rejected {cmd}"
    got = res["buffer_after"].split("\n")
    while got and got[-1] == "":
        got.pop()
    expected = [s for s in lines if not member[s]]
    assert got == expected, (
        f"{pattern}\n  cmd {cmd}\n  vim  {got}\n  oracle {expected}")
    return r


# ------------------------------------------------- shapes that MUST render

def test_conj_at_end_k0():
    # PRE (A&B) $  — conjunction final, pin is bare $
    r = check("a(([a-z0-9]+)&(.*5.*))")
    assert r is not None


def test_conj_mid_fixed_suffix():
    # (A&B) then a fixed-length tail — pin is .\{k}$
    r = check("(([a-z0-9]+)&(.*5.*))!B")
    assert r is not None and "\\{2}$" in r["exec"]


def test_negated_conjunct():
    r = check("a((.+)&(~(.*!.*)))")
    assert r is not None and "\\@!" in r["exec"]


def test_pure_not_element_fixed_tail():
    # ~X mid-sequence with fixed suffix; ~ spans variable length
    r = check("(~(a.*))5B")
    assert r is not None


def test_root_not_polarity():
    r = check("~(a(([a-z]+)&(.*5.*)))")
    assert r is not None and r["pol"] == "v"


def test_equal_fixed_lengths_variable_tail():
    # R2: conjuncts fixed equal length, tail unbounded
    r = check("(([a-zB])&([aB5]))[a5!B]*")
    assert r is not None


def test_all_negative_conjuncts():
    r = check("a((~([a-z]))&(~(5)))B")
    assert r is not None


def test_sr_shape_charset_and_start():
    # the PR-106 family: (class)+ & startwith(...) at root
    r = check("(([B5!])+)&(([!B]).*)")
    assert r is not None


# ------------------------------------------------- shapes that MUST refuse

def test_refuses_op_under_repeat():
    assert render_embedded("(([a-z]+)&(.*5.*))*") is None


def test_refuses_variable_tail_variable_conjuncts():
    assert render_embedded("(([a-z]+)&(.*5.*))[a-z]*") is None


def test_refuses_op_under_alt():
    assert render_embedded("a|(([a-z]+)&(.*5.*))b*") is None


# ------------------------------------------------------------- guarantees

def test_generate_untouched_fallback_only():
    # the corpus-wide converter still goes stuck on embedded ops; the new
    # module is a separate, opt-in fallback
    stuck = generate("a(([a-z0-9]+)&(.*5.*))")
    assert stuck["exec"] is None
    assert render_embedded("a(([a-z0-9]+)&(.*5.*))") is not None


def test_lenrange_basics():
    assert lenrange(parse("[a-z]{2,4}")) == (2, 4)
    assert lenrange(parse("a[0-9]*")) == (1, None)
    assert lenrange(parse("(ab|xyz)")) == (2, 3)
    assert lenrange(parse("(([a-z]{3})&(.*))")) == (3, 3)
