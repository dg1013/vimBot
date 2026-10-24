r"""Textual DSL (domain-specific language) -> vim conversion: escaping and the structural templates.

Tags: "p" portable (escape only), "c" convertible (template), "s" stuck
(handed on to the semantic rewriter / manual table by build_conversions).

Validated conclusions this file encodes (2026-08-12/13 session):
  - vim magic mode INVERTS several escapes: DSL \? is a literal question
    mark, vim \? is a quantifier — a DSL escape must be re-derived per
    char, never passed through.
  - a ROOT ~ must span the whole expression: `~(dog).*x.*` negates only
    (dog); treating it as a root complement silently changes the language.
  - empty class intersections must NOT be declared "never matches" here:
    (∅)* is ε and ∅ drops out of alternations — only the AST (abstract syntax tree) empty-language
    algebra (rewrite_stuck) may issue that verdict.
"""
import re

from convertibility import strip_outer, split_top, has_ops


# ---------------------------------------------------------------- escaping

def esc_portable(e, exact_boundary=False):
    r"""DSL (no &/~) -> vim magic-mode pattern.

    exact_boundary=True renders \b as \%(\<\|\>\) — the exact union of vim's
    two one-sided assertions, needed when the \<//\> position heuristic is
    wrong (e.g. \b inside a quantified group). Validation decides per
    pattern.
    """
    out, i, n = [], 0, len(e)
    in_class = False
    while i < n:
        c = e[i]
        if in_class:
            out.append(c)
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "\\" and i + 1 < n:
            nxt = e[i + 1]
            if nxt == "b":
                out.append(r"\%(\<\|\>\)" if exact_boundary
                           else _boundary(e, i))
            elif nxt in ".*[]^$~\\/":
                out.append("\\" + nxt)      # backslash keeps these literal
            else:
                out.append(nxt)             # bare form IS the vim literal
            i += 2
            continue
        if c == "[":
            in_class = True
            out.append(c)
        elif c == "|":
            out.append(r"\|")
        elif c == "+":
            out.append(r"\+")
        elif c == "?":
            out.append(r"\=")
        elif c == "{":
            out.append(r"\{")
        elif c == "(":
            out.append(r"\(")
        elif c == ")":
            out.append(r"\)")
        else:
            out.append(c)
        i += 1
    return "".join(out)


_WORDISH = re.compile(r"[A-Za-z0-9_]")


def _boundary(e, i):
    r"""Decide \b -> \< or \> from context (corpus wraps words in \b...\b).

    A following class counts as wordish only when it is NOT negated:
    `\b[^A-Za-z]*` is a boundary before junk — that is a word END (\>).
    The 2026-08-13 ingest gate caught `\<` being emitted there.
    """
    j = i + 2
    while j < len(e) and e[j] in "()":
        j += 1
    nxt = e[j] if j < len(e) else ""
    nxt2 = e[j + 1] if j + 1 < len(e) else ""
    if _WORDISH.match(nxt or " ") or (nxt == "[" and nxt2 != "^"):
        return r"\<"
    k = i - 1
    while k >= 0 and e[k] in "()":
        k -= 1
    prv = e[k] if k >= 0 else ""
    if prv and (prv.isalnum() or prv in "]*+}?"):
        return r"\>"
    return r"\<"


def group_style(vim_pattern):
    r"""Vim errors past 9 capture groups; fall back to \%( for heavy ones."""
    if vim_pattern.count(r"\(") > 9:
        return vim_pattern.replace(r"\(", r"\%(")
    return vim_pattern


# ------------------------------------------------------- class intersection

def charset(atom):
    """Character set of a single-char atom ([class] or bare char)."""
    if not atom.startswith("["):
        return set(atom)
    inner, out, i = atom[1:-1], set(), 0
    while i < len(inner):
        if i + 2 < len(inner) and inner[i + 1] == "-":
            out.update(chr(c)
                       for c in range(ord(inner[i]), ord(inner[i + 2]) + 1))
            i += 3
        else:
            out.add(inner[i])
            i += 1
    return out


def set_to_class(s):
    """Render a char set as a compact vim/python class."""
    if not s:
        return None
    chars = sorted(s)
    parts, i = [], 0
    while i < len(chars):
        j = i
        while j + 1 < len(chars) and ord(chars[j + 1]) == ord(chars[j]) + 1:
            j += 1
        if j - i >= 2:
            parts.append(f"{chars[i]}-{chars[j]}")
        else:
            parts.extend(chars[i:j + 1])
        i = j + 1
    body = "".join(parts)
    return body if len(s) == 1 and body not in r".*+?[](){}|&~\^$" \
        else f"[{body}]"


def reduce_class_and(e):
    """Statically intersect (c1)&(c2)[&(c3)] chains of single-char atoms.

    The guards are load-bearing: operands are atomic only when nothing
    extends them — the preceding char must be one of ( | & ~ (or start) and
    the following one of ) | & (or end). Without them the reducer once
    grabbed `([A-Z])&([a-z])` out of `.*([A-Z])&([a-z]).*`, whose real
    operands are `.*([A-Z])` and `([a-z]).*` by precedence.
    """
    degenerate = False

    def repl(m):
        nonlocal degenerate
        atoms = [a for a in m.groups() if a]
        s = charset(atoms[0])
        for a in atoms[1:]:
            s &= charset(a)
        if not s:
            degenerate = True
            return "[^\\x00-\\xff]"   # matches nothing, keeps structure
        return set_to_class(s)

    pat = re.compile(
        r"(?<![^(|&~])"
        r"\((\[[^\]]*\]|[A-Za-z0-9])\)&\((\[[^\]]*\]|[A-Za-z0-9])\)"
        r"(?:&\((\[[^\]]*\]|[A-Za-z0-9])\))?(?![^)|&])")
    prev = None
    while prev != e:
        prev = e
        e = pat.sub(repl, e)
        e = re.sub(r"\((\[[^\]]*\])\)", r"\1", e)  # unwrap ([x]) -> [x]
    return e, degenerate


def not_spans_whole(e):
    r"""For e starting with '~': the negated group must cover ALL of e."""
    if not e.startswith("~"):
        return False
    rest = e[1:]
    if len(rest) == 1 or (rest.startswith("[") and rest.endswith("]")
                          and "]" not in rest[1:-1]):
        return True
    if not rest.startswith("("):
        return False
    depth, i = 0, 0
    while i < len(rest):
        c = rest[i]
        if c == "\\":
            i += 2
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i == len(rest) - 1
        i += 1
    return False


# ---------------------------------------------------------------- generator

def generate(expr, exact_boundary=False):
    """-> dict(tag, display, exec, pol, note)."""
    e = strip_outer(expr)

    if not has_ops(e):
        vim = group_style(esc_portable(e, exact_boundary))
        return {"tag": "p", "display": vim if vim != expr else None,
                "exec": f"^\\%({vim}\\)$", "pol": "g", "note": None}

    # CLASS-& first: may clear all ops (works at any depth)
    reduced, degenerate = reduce_class_and(e)
    if not has_ops(reduced):
        if degenerate:
            # ∅ needs the AST empty-language algebra — fall through
            return {"tag": "s", "display": None, "exec": None, "pol": "g",
                    "note": None}
        vim = group_style(esc_portable(reduced, exact_boundary))
        return {"tag": "c", "display": vim, "exec": f"^\\%({vim}\\)$",
                "pol": "g", "note": "class-and"}

    # ROOT-~ : whole expression is ~X -> :g/:v polarity flip
    if e.startswith("~") and not_spans_whole(e):
        inner = strip_outer(e[1:])
        if len(split_top(e, "&")) == 1 and len(split_top(e, "|")) == 1:
            innr, deg = reduce_class_and(inner)
            if not has_ops(innr):
                if deg:
                    return {"tag": "s", "display": None, "exec": None,
                            "pol": "g", "note": None}
                vim = group_style(esc_portable(innr, exact_boundary))
                return {"tag": "c",
                        "display": f":v/^\\%({vim}\\)$/  (polarity flip)",
                        "exec": f"^\\%({vim}\\)$", "pol": "v",
                        "note": "root-not"}
            sub = generate(innr, exact_boundary)
            if sub["tag"] == "c" and sub["note"] == "root-and" and sub["exec"]:
                return {"tag": "c",
                        "display": f":v/{sub['exec']}/  (polarity flip)",
                        "exec": sub["exec"], "pol": "v", "note": "root-not"}
        return {"tag": "s", "display": None, "exec": None, "pol": "g",
                "note": None}

    # ROOT-& : depth-0 intersection -> chained anchored lookaheads
    parts = split_top(e, "&")
    if len(parts) > 1 and len(split_top(e, "|")) == 1:
        looks, consumer = [], None
        ok = True
        for p in parts:
            p = strip_outer(p)
            neg = False
            if p.startswith("~"):
                if not not_spans_whole(p):
                    ok = False
                    break
                neg, p = True, strip_outer(p[1:])
            p, deg = reduce_class_and(p)
            if deg or has_ops(p):
                ok = False
                break
            vim = esc_portable(p, exact_boundary)
            # the operand gets its own group BEFORE the $ anchor: without
            # it, in an operand like a\|b the $ binds only to the last branch
            if neg:
                looks.append(f"\\%(\\%({vim}\\)$\\)\\@!")
            elif consumer is None:
                consumer = vim          # first positive operand consumes
            else:
                looks.append(f"\\%(\\%({vim}\\)$\\)\\@=")
        if ok:
            body = "".join(looks) + (f"\\%({consumer}\\)$" if consumer
                                     else ".*$")
            vim = group_style("^" + body)
            return {"tag": "c", "display": vim, "exec": vim, "pol": "g",
                    "note": "root-and"}

    return {"tag": "s", "display": None, "exec": None, "pol": "g",
            "note": None}


# ------------------------------------------------- python ground truth (val)

def py_oracle(expr):
    """f(str)->bool via python-re for root-level shapes, else None.

    Independent of the vim templates; the span-DP (dynamic programming) oracle (dsl_ast) covers
    everything this returns None for.
    """
    e = strip_outer(expr)
    if not has_ops(e):
        try:
            rx = re.compile(f"(?:{e})\\Z")
        except re.error:
            return None
        return lambda s: bool(rx.match(s))
    reduced, deg = reduce_class_and(e)
    if not has_ops(reduced):
        if deg:
            return None
        try:
            rx = re.compile(f"(?:{reduced})\\Z")
        except re.error:
            return None
        return lambda s: bool(rx.match(s))
    if e.startswith("~") and not_spans_whole(e) \
            and len(split_top(e, "&")) == 1 \
            and len(split_top(e, "|")) == 1:
        inner = py_oracle(strip_outer(e[1:]))
        return (lambda s: not inner(s)) if inner else None
    parts = split_top(e, "&")
    if len(parts) > 1 and len(split_top(e, "|")) == 1:
        preds = []
        for p in parts:
            p = strip_outer(p)
            if p.startswith("~"):
                if not not_spans_whole(p):
                    return None
                sub = py_oracle(strip_outer(p[1:]))
                if not sub:
                    return None
                preds.append(lambda s, f=sub: not f(s))
            else:
                sub = py_oracle(p)
                if not sub:
                    return None
                preds.append(sub)
        return lambda s: all(f(s) for f in preds)
    return None
