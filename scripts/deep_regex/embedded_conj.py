r"""Embedded and/~ -> vim: suffix-pinned lookahead rendering.

`dsl2vim.generate` renders &/~ only at the root, where both operands can be
pinned to the whole line (`^…$`) — same-span conjunction falls out of the
shared anchors. Below the root it goes stuck: `\%(A\)\@=B` alone requires A
to match a *prefix* of B's span, not the same span, and distributing the
continuation into the lookahead is unsound in general
(L(A·T) ∩ L(B·T) ⊋ L((A&B)·T)).

But every command we emit is whole-line anchored, which restores an anchor
mid-sequence in two situations, and only these are rendered:

  R1 — the material after the conjunction has FIXED total length k.
       Then the conjunct span must end exactly at end−k, so each extra
       conjunct becomes a lookahead pinned there:
           PRE (A&B) SUF$   ->   PRE \%(A.\{k}$\)\@=B SUF$      (k>0)
           PRE (A&B) $      ->   PRE \%(A$\)\@=B$               (k=0)
       ~X conjuncts use \@! with the same pin. A pure ~X element consumes
       `.\{-}`; the fixed suffix forces where that span ends.

  R2 — every conjunct (including negated inners) has the SAME fixed
       length m. Same start + equal fixed length = same span, no pin
       needed: \%(A\)\@=B. (All-negative: consume `.\{m}`.)

Anything else — ops under a repeat, under an alternation, variable-length
suffix with variable-length conjuncts — returns None rather than a command
that vim would interpret differently from the span oracle.

Soundness is enforced twice over: `tests/test_embedded_conj.py` brute-forces
every rendering against the span-DP (dynamic programming) oracle over a small alphabet in real
vim, and every pipeline use re-validates through the executor + triangle.

Opt-in by design: nothing in dsl2vim changes, so the A2 corpus-wide audit
of the existing conversions is untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dsl_ast import parse, serialize  # noqa: E402
from dsl2vim import esc_portable, group_style  # noqa: E402


# ------------------------------------------------------------- length facts

def lenrange(nd):
    """(min, max) span lengths of a node; max None = unbounded."""
    k = nd.kind
    if k in ("char", "class", "dot"):
        return 1, 1
    if k == "bound":
        return 0, 0
    if k == "seq":
        lo = hi = 0
        for c in nd.kids:
            a, b = lenrange(c)
            lo += a
            hi = None if (hi is None or b is None) else hi + b
        return lo, hi
    if k == "alt":
        rs = [lenrange(c) for c in nd.kids]
        lo = min(a for a, _ in rs)
        hs = [b for _, b in rs]
        return lo, (None if any(b is None for b in hs) else max(hs))
    if k == "rep":
        a, b = lenrange(nd.kids[0])
        lo = nd.m * a
        hi = None if (nd.n is None or b is None) else nd.n * b
        return lo, hi
    if k == "and":
        rs = [lenrange(c) for c in nd.kids]
        lo = max(a for a, _ in rs)
        hs = [b for _, b in rs]
        fin = [b for b in hs if b is not None]
        return lo, (min(fin) if fin else None)
    if k == "not":
        return 0, None
    raise ValueError(k)


def fixed_len(nd):
    a, b = lenrange(nd)
    return a if a == b else None


def is_plain(nd):
    """No and/not anywhere in the subtree — esc_portable can render it."""
    if nd.kind in ("and", "not"):
        return False
    return all(is_plain(c) for c in nd.kids)


def _vim(nd, exact_boundary=False):
    txt = serialize(nd)
    if nd.kind == "alt":
        txt = f"({txt})"
    return esc_portable(txt, exact_boundary)


def _conjuncts(nd):
    """Flatten nested and-nodes into (negated, operand) pairs."""
    out = []
    for c in nd.kids:
        if c.kind == "and":
            out += _conjuncts(c)
        elif c.kind == "not":
            out.append((True, c.kids[0]))
        else:
            out.append((False, c))
    return out


# --------------------------------------------------------------- rendering

def _pin(k):
    return "$" if k == 0 else f".\\{{{k}}}$"


def _render_conj(nd, k, xb):
    """An and/not node at a sequence position; k = fixed tail length or
    None. Returns the vim fragment or None."""
    conjs = ([(True, nd.kids[0])] if nd.kind == "not" else _conjuncts(nd))
    if not all(is_plain(op) for _, op in conjs):
        return None

    if k is not None:                                          # R1
        looks, consumer = [], None
        for neg, op in conjs:
            v = _vim(op, xb)
            if neg:
                looks.append(f"\\%({v}{_pin(k)}\\)\\@!")
            elif consumer is None:
                consumer = f"\\%({v}\\)"
            else:
                looks.append(f"\\%({v}{_pin(k)}\\)\\@=")
        return "".join(looks) + (consumer if consumer else ".\\{-}")

    lens = [fixed_len(op) for _, op in conjs]                  # R2
    if None in lens or len(set(lens)) != 1:
        return None
    m = lens[0]
    looks, consumer = [], None
    for neg, op in conjs:
        v = _vim(op, xb)
        if neg:
            looks.append(f"\\%({v}\\)\\@!")
        elif consumer is None:
            consumer = f"\\%({v}\\)"
        else:
            looks.append(f"\\%({v}\\)\\@=")
    return "".join(looks) + (consumer if consumer else f".\\{{{m}}}")


def _render_seq(kids, tail, xb):
    """tail = fixed length from the end of this element list to $, or None."""
    out = []
    for i, child in enumerate(kids):
        rest = kids[i + 1:]
        k = tail
        for r in rest:
            f = fixed_len(r)
            k = None if (k is None or f is None) else k + f
        if is_plain(child):
            out.append(_vim(child, xb))
        elif child.kind in ("and", "not"):
            piece = _render_conj(child, k, xb)
            if piece is None:
                return None
            out.append(piece)
        elif child.kind == "seq":
            piece = _render_seq(child.kids, k, xb)
            if piece is None:
                return None
            out.append(piece)
        else:                       # ops under rep/alt: not renderable here
            return None
    return "".join(out)


def render_embedded(text, exact_boundary=False):
    """DSL (domain-specific language) text -> {'exec','pol'} using suffix-pinned lookaheads, or None.

    Meant as a fallback after dsl2vim.generate returns no exec; callers
    must still validate the command through the executor.
    """
    try:
        ast = parse(text)
    except Exception:                                   # noqa: BLE001
        return None
    pol = "g"
    if ast.kind == "not":
        pol = "v"
        ast = ast.kids[0]
    kids = ast.kids if ast.kind == "seq" else [ast]
    body = _render_seq(kids, 0, exact_boundary)
    if body is None:
        return None
    return {"exec": group_style(f"^\\%({body}\\)$"), "pol": pol}
