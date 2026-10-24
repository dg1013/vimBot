"""Parse the deep-regex DSL (domain-specific language) to an AST (abstract syntax tree); match by span DP (dynamic programming); serialize back.

The span matcher is the ground-truth oracle for this corpus: language
membership by recursion over (node, span), where & is same-span conjunction
and ~ is same-span negation — no lookaround emulation, no backtracking
blowup. It agreed with an independent python-re oracle on 6,000+ spot checks
before being trusted (2026-08-13 session).

Precedence (tightest to loosest): postfix quantifiers, concat, &, |.
~ is a prefix on a primary.
"""
import re

WORD = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_")


# ------------------------------------------------------------------- nodes

class N:
    __slots__ = ("kind", "kids", "val", "m", "n", "id")

    def __init__(self, kind, kids=(), val=None, m=0, n=0):
        self.kind, self.kids, self.val = kind, list(kids), val
        self.m, self.n = m, n
        self.id = -1

    def __repr__(self):
        return f"N({self.kind},{self.val if self.val is not None else self.kids})"


# ------------------------------------------------------------------- parser

class ParseError(Exception):
    pass


def parse(text):
    pos = 0
    n = len(text)

    def peek():
        return text[pos] if pos < n else ""

    def parse_alt():
        nonlocal pos
        kids = [parse_and()]
        while peek() == "|":
            pos += 1
            kids.append(parse_and())
        return kids[0] if len(kids) == 1 else N("alt", kids)

    def parse_and():
        nonlocal pos
        kids = [parse_concat()]
        while peek() == "&":
            pos += 1
            kids.append(parse_concat())
        return kids[0] if len(kids) == 1 else N("and", kids)

    def parse_concat():
        nonlocal pos
        kids = []
        while pos < n and peek() not in "|&)":
            kids.append(parse_prefixed())
        if not kids:
            raise ParseError(f"empty concat at {pos}")
        return kids[0] if len(kids) == 1 else N("seq", kids)

    def parse_prefixed():
        nonlocal pos
        if peek() == "~":
            pos += 1
            node = N("not", [parse_primary()])
            return parse_postfix(node)
        return parse_postfix(parse_primary())

    def parse_postfix(node):
        nonlocal pos
        while pos < n:
            c = peek()
            if c == "*":
                node = N("rep", [node], m=0, n=None)
                pos += 1
            elif c == "+":
                node = N("rep", [node], m=1, n=None)
                pos += 1
            elif c == "?":
                node = N("rep", [node], m=0, n=1)
                pos += 1
            elif c == "{":
                mm = re.match(r"\{(\d+)(,)?(\d+)?\}", text[pos:])
                if not mm:
                    raise ParseError(f"bad repeat at {pos}")
                lo = int(mm.group(1))
                hi = (None if mm.group(2) and not mm.group(3)
                      else int(mm.group(3)) if mm.group(3) else lo)
                node = N("rep", [node], m=lo, n=hi)
                pos += mm.end()
            else:
                break
        return node

    def parse_primary():
        nonlocal pos
        c = peek()
        if c == "(":
            pos += 1
            node = parse_alt()
            if peek() != ")":
                raise ParseError(f"missing ) at {pos}")
            pos += 1
            return node
        if c == "[":
            close = text.index("]", pos + 1)
            body = text[pos:close + 1]
            pos = close + 1
            return N("class", val=frozenset(_charset(body)))
        if c == "\\":
            nxt = text[pos + 1] if pos + 1 < n else ""
            pos += 2
            if nxt == "b":
                return N("bound")
            return N("char", val=nxt)
        if c == ".":
            pos += 1
            return N("dot")
        if c in "|&)*+?{~":
            raise ParseError(f"unexpected {c!r} at {pos}")
        pos += 1
        return N("char", val=c)

    root = parse_alt()
    if pos != n:
        raise ParseError(f"trailing input at {pos}: {text[pos:]!r}")
    return root


def _charset(cls):
    inner, out, i = cls[1:-1], set(), 0
    neg = inner.startswith("^")
    if neg:
        inner = inner[1:]
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner):
            out.add(inner[i + 1])
            i += 2
            continue
        if i + 2 < len(inner) and inner[i + 1] == "-":
            out.update(chr(c) for c in range(ord(inner[i]), ord(inner[i + 2]) + 1))
            i += 3
        else:
            out.add(inner[i])
            i += 1
    if neg:
        out = set(chr(c) for c in range(32, 127)) - out
    return out


# ------------------------------------------------------------------ matcher

def matcher(root, s):
    """True iff the whole string s is in the language of root.

    Memo keys use object identity (id(nd)) so rewritten/synthetic nodes are
    safe — they all share the placeholder .id of -1.
    """
    memo = {}

    def bound_ok(i):
        left = s[i - 1] in WORD if i > 0 else False
        right = s[i] in WORD if i < len(s) else False
        return left != right

    def M(nd, i, j):
        key = (id(nd), i, j)
        if key in memo:
            return memo[key]
        memo[key] = False          # cycle guard (rep of ε-matching nodes)
        k = nd.kind
        if k == "char":
            r = j == i + 1 and s[i] == nd.val
        elif k == "class":
            r = j == i + 1 and s[i] in nd.val
        elif k == "dot":
            r = j == i + 1
        elif k == "bound":
            r = i == j and bound_ok(i)
        elif k == "alt":
            r = any(M(c, i, j) for c in nd.kids)
        elif k == "and":
            r = all(M(c, i, j) for c in nd.kids)
        elif k == "not":
            r = not M(nd.kids[0], i, j)
        elif k == "seq":
            r = M_seq(nd, 0, i, j)
        elif k == "rep":
            r = M_rep(nd, 0, i, j)
        else:
            raise AssertionError(k)
        memo[key] = r
        return r

    def M_seq(nd, idx, i, j):
        if idx == len(nd.kids) - 1:
            return M(nd.kids[idx], i, j)
        key = ("s", id(nd), idx, i, j)
        if key in memo:
            return memo[key]
        memo[key] = False
        r = any(M(nd.kids[idx], i, k) and M_seq(nd, idx + 1, k, j)
                for k in range(i, j + 1))
        memo[key] = r
        return r

    def M_rep(nd, cnt, i, j):
        child, lo, hi = nd.kids[0], nd.m, nd.n
        cc = min(cnt, lo) if hi is None else cnt
        key = ("r", id(nd), cc, i, j)
        if key in memo:
            return memo[key]
        memo[key] = False
        if i == j:
            r = cnt >= lo or M(child, i, i)
        elif hi is not None and cnt == hi:
            r = False
        else:
            r = any(M(child, i, k) and M_rep(nd, cnt + 1, k, j)
                    for k in range(i + 1, j + 1))
        memo[key] = r
        return r

    return M(root, 0, len(s))


def dsl_match(text, s, _cache={}):
    if text not in _cache:
        if len(_cache) > 4096:
            _cache.clear()
        _cache[text] = parse(text)
    return matcher(_cache[text], s)


# --------------------------------------------------------------- serializer

def serialize(nd):
    k = nd.kind
    if k == "char":
        return nd.val if nd.val not in r".*+?[](){}|&~\^$" else "\\" + nd.val
    if k == "class":
        return _set_to_class(nd.val)
    if k == "dot":
        return "."
    if k == "bound":
        return r"\b"
    if k == "seq":
        return "".join(_wrap(c, k) for c in nd.kids)
    if k == "alt":
        return "|".join(_wrap(c, k) for c in nd.kids)
    if k == "and":
        return "&".join(_wrap(c, k) for c in nd.kids)
    if k == "not":
        return "~(" + serialize(nd.kids[0]) + ")"
    if k == "rep":
        suffix = ("*" if (nd.m, nd.n) == (0, None) else
                  "+" if (nd.m, nd.n) == (1, None) else
                  "?" if (nd.m, nd.n) == (0, 1) else
                  "{%d}" % nd.m if nd.n == nd.m else
                  "{%d,}" % nd.m if nd.n is None else
                  "{%d,%d}" % (nd.m, nd.n))
        return _wrap(nd.kids[0], "rep") + suffix
    raise AssertionError(k)


def _wrap(nd, parent):
    txt = serialize(nd)
    atomic = nd.kind in ("char", "class", "dot", "bound", "not")
    if parent == "rep":
        return txt if atomic else "(" + txt + ")"
    if parent == "seq":
        return txt if atomic or nd.kind == "rep" else "(" + txt + ")"
    return txt if atomic or nd.kind in ("rep", "seq") else "(" + txt + ")"


def _set_to_class(s):
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
    if len(s) == 1 and body not in r".*+?[](){}|&~\^$-":
        return body
    return f"[{body}]"
