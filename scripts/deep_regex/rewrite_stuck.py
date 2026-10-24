r"""Semantic rewrites: compute closed forms for mid-pattern set operators.

Every rule here is validation-gated downstream — build_conversions executes
each output in vim against the span oracle before it may ship, and
never-match verdicts additionally require zero matching probes.

  EMPTY ALGEBRA  ∅ propagates exactly: through concat/&, drops out of |,
                 (∅)* = ε, ~∅ = .* — "never matches" only when ∅ reaches
                 the root.
  FINITE-&       operands with small finite languages (optionally \b-edged,
                 optionally under ~) intersect exactly; single chars inside
                 \b..\b are word-restricted.
  FINITE-FILTER  one finite operand + arbitrary partners: enumerate its
                 strings, keep those every other operand accepts (matcher).
  WORD-&         (\b[A-Za-z]+\b)&(P1)&(P2)…: starts / ends / ordered
                 contains-chains / not-contains (class subtraction) /
                 not-starts / not-ends / length / k-of, with fixed-length
                 overlap templates so single-char words survive.
  STARTS/ENDS-&  (X.*)&(.*Y) -> X.*Y | overlap templates; same-side merges.
  CONTAINS-&     order alternation (≤3 cores).
  ~-ABSORB       ε-admitting factor beside .* is vacuous; \b F \b collapses
                 to \b beside .*; ~(C*) beside .* becomes C*[^C] (side-
                 padded); ~(X*) for finite X of min length ≥2 becomes `.`;
                 ~((.)*) is ∅.
  REP-1          (.*X.*){1,} = .*X.*
"""
from itertools import permutations

from dsl_ast import N, WORD, matcher, parse, serialize

PRINTABLE = frozenset(chr(c) for c in range(32, 127))
FINITE_CAP = 128

EMPTY = object()          # the empty language ∅


def dotstar():
    return N("rep", [N("dot")], m=0, n=None)


def eps_node():
    return N("rep", [N("dot")], m=0, n=0)


def is_dotstar(nd):
    return nd.kind == "rep" and (nd.m, nd.n) == (0, None) \
        and nd.kids[0].kind == "dot"


def is_eps(nd):
    return nd.kind == "rep" and nd.n == 0


def cls_rep(cls, m=0):
    return N("rep", [N("class", val=frozenset(cls))], m=m, n=None)


def atom_set(nd):
    if nd.kind == "char":
        return frozenset(nd.val)
    if nd.kind == "class":
        return nd.val
    if nd.kind == "dot":
        return PRINTABLE
    if nd.kind == "alt":
        acc = frozenset()
        for c in nd.kids:
            s = atom_set(c)
            if s is None:
                return None
            acc |= s
        return acc
    return None


def finite_lang(nd):
    if nd.kind in ("char", "class", "dot"):
        return set(atom_set(nd))
    if nd.kind == "alt" and all(len(c.kids) == 0 for c in nd.kids) \
            and atom_set(nd) is not None:
        return set(atom_set(nd))
    if nd.kind == "seq":
        acc = {""}
        for c in nd.kids:
            sub = finite_lang(c)
            if sub is None:
                return None
            acc = {a + b for a in acc for b in sub}
            if len(acc) > FINITE_CAP:
                return None
        return acc
    if nd.kind == "alt":
        acc = set()
        for c in nd.kids:
            sub = finite_lang(c)
            if sub is None:
                return None
            acc |= sub
            if len(acc) > FINITE_CAP:
                return None
        return acc
    return None


def bounded_finite(nd):
    """-> (pre_bound, lang, post_bound, negated) or None."""
    kids = list(nd.kids) if nd.kind == "seq" else [nd]
    pre = post = False
    while kids and kids[0].kind == "bound":
        pre = True
        kids.pop(0)
    while kids and kids[-1].kind == "bound":
        post = True
        kids.pop()
    if not kids:
        return None
    core = kids[0] if len(kids) == 1 else N("seq", kids)
    neg = False
    if core.kind == "not":
        neg, core = True, core.kids[0]
    lang = finite_lang(core)
    if lang is None:
        return None
    return pre, lang, post, neg


def lang_to_node(strings):
    if all(len(s) == 1 for s in strings):
        if len(strings) == 1:
            return N("char", val=next(iter(strings)))
        return N("class", val=frozenset(strings))
    if len(strings) == 1:
        return N("seq", [N("char", val=c) for c in next(iter(strings))])
    return N("alt", [N("seq", [N("char", val=c) for c in s])
                     for s in sorted(strings)])


def eps_may(nd):
    """Over-approximate ε-membership (False = surely does not match "")."""
    k = nd.kind
    if k in ("char", "class", "dot"):
        return False
    if k in ("bound", "not"):
        return True
    if k == "seq":
        return all(eps_may(c) for c in nd.kids)
    if k == "alt":
        return any(eps_may(c) for c in nd.kids)
    if k == "rep":
        return nd.m == 0 or eps_may(nd.kids[0])
    return True


def eps_must(nd):
    """Under-approximate ε-membership (True = surely matches "")."""
    k = nd.kind
    if k in ("char", "class", "dot", "bound"):
        return False
    if k == "not":
        return not eps_may(nd.kids[0])
    if k == "seq":
        return all(eps_must(c) for c in nd.kids)
    if k == "alt":
        return any(eps_must(c) for c in nd.kids)
    if k == "rep":
        return nd.m == 0 or eps_must(nd.kids[0])
    return False


def word_core(nd):
    r"""\b? [C]+ / [C]* / [C][C]* \b? -> (pre, cls, post, min_len)."""
    kids = list(nd.kids) if nd.kind == "seq" else [nd]
    pre, post = [], []
    while kids and kids[0].kind == "bound":
        pre.append(kids.pop(0))
    while kids and kids[-1].kind == "bound":
        post.insert(0, kids.pop())
    core = m0 = None
    if len(kids) == 1:
        c = kids[0]
        if c.kind == "rep" and c.n is None and c.m in (0, 1) \
                and c.kids[0].kind == "class":
            core, m0 = c.kids[0].val, c.m
    elif len(kids) == 2:
        a, b = kids
        if a.kind == "class" and b.kind == "rep" \
                and (b.m, b.n) == (0, None) \
                and b.kids[0].kind == "class" and b.kids[0].val == a.val:
            core, m0 = a.val, 1
    if core is not None and core <= frozenset(WORD):
        return pre, core, post, m0
    return None


# ------------------------------------------------ word constraint folding

def _strip_bounds(kids):
    pre, post = [], []
    while kids and kids[0].kind == "bound":
        pre.append(kids.pop(0))
    while kids and kids[-1].kind == "bound":
        post.insert(0, kids.pop())
    return pre, post


def _atom_sets(atoms):
    out = []
    for a in atoms:
        s = atom_set(a)
        if s is None:
            return None
        out.append(s)
    return out


def _classify_partner(nd):
    """-> (kind, payload) or None.
    kinds: starts/ends/exact (payload includes a had-bounds flag), chain,
    len, k-of, not-contains, not-starts, not-ends."""
    kids = list(nd.kids) if nd.kind == "seq" else [nd]
    kids = list(kids)
    ppre, ppost = _strip_bounds(kids)        # partner \b's carry over
    if not kids:
        return None

    if len(kids) == 1 and kids[0].kind == "not":
        ik = list(kids[0].kids[0].kids) \
            if kids[0].kids[0].kind == "seq" else [kids[0].kids[0]]
        _strip_bounds(ik)
        lead = len(ik) > 1 and is_dotstar(ik[0])
        trail = len(ik) > 1 and is_dotstar(ik[-1])
        atoms = ik[1 if lead else 0: -1 if trail else len(ik)]
        sets = _atom_sets(atoms)
        if sets is None or not sets:
            return None
        if lead and trail and len(sets) == 1:
            return ("not-contains", sets[0])
        if trail and not lead and len(sets) == 1:
            return ("not-starts", sets[0])
        if lead and not trail and len(sets) == 1:
            return ("not-ends", sets[0])
        return None

    if len(kids) == 1 and kids[0].kind == "rep" \
            and kids[0].kids[0].kind in ("class", "char", "dot"):
        return ("len", (atom_set(kids[0].kids[0]), kids[0].m, kids[0].n))

    if len(kids) == 1 and kids[0].kind == "rep" and kids[0].m >= 1:
        inner = kids[0].kids[0]
        ik = list(inner.kids) if inner.kind == "seq" else [inner]
        if len(ik) >= 3 and is_dotstar(ik[0]) and is_dotstar(ik[-1]):
            sets = _atom_sets(ik[1:-1])
            if sets:
                return ("k-of", (sets, kids[0].m))
        return None

    lead = len(kids) > 1 and is_dotstar(kids[0])
    trail = len(kids) > 1 and is_dotstar(kids[-1])
    body = kids[1 if lead else 0: -1 if trail else len(kids)]
    if not body:
        return ("chain", [])                 # bare .* — no constraint
    if lead and trail:
        runs, run = [], []
        for a in body:
            if is_dotstar(a):
                if run:
                    runs.append(run)
                run = []
                continue
            s = atom_set(a)
            if s is None:
                return None
            run.append(s)
        if run:
            runs.append(run)
        return ("chain", runs) if runs else None
    sets = _atom_sets(body)
    if sets is None:
        return None
    if trail:
        return ("starts", (sets, bool(ppre)))
    if lead:
        return ("ends", (sets, bool(ppost)))
    return ("exact", (sets, bool(ppre) or bool(ppost)))


def _sets_to_atoms(sets):
    return [N("class", val=frozenset(s)) if len(s) > 1
            else N("char", val=min(s)) for s in sets]


def fold_word(W, pre, post, m0, partners):
    """Compose multiple word constraints into one closed-form node."""
    starts = ends = exact = length = kof = None
    chains, not_contains, not_starts, not_ends = [], set(), set(), set()
    for p in partners:
        c = _classify_partner(p)
        if c is None:
            return None
        kind, payload = c
        if kind == "starts":
            if starts:
                return None
            starts, sb = payload
            if sb:
                pre = pre or [N("bound")]
        elif kind == "ends":
            if ends:
                return None
            ends, eb = payload
            if eb:
                post = post or [N("bound")]
        elif kind == "exact":
            if exact:
                return None
            exact, xb = payload
            if xb:
                pre = pre or [N("bound")]
                post = post or [N("bound")]
        elif kind == "len":
            if length:
                return None
            length = payload
        elif kind == "k-of":
            if kof:
                return None
            kof = payload
        elif kind == "chain":
            if payload:
                chains.append(payload)
        elif kind == "not-contains":
            not_contains |= payload
        elif kind == "not-starts":
            not_starts |= payload
        elif kind == "not-ends":
            not_ends |= payload

    Wp = W - not_contains
    if not Wp and (m0 or starts or ends or exact or chains or kof):
        return EMPTY

    def fix(sets):
        out = []
        for s in sets:
            ss = s & Wp
            if not ss:
                return EMPTY
            out.append(ss)
        return out

    if exact is not None:
        if starts or ends or chains or length or kof or not_starts \
                or not_ends:
            return None
        sets = fix(exact)
        if sets is EMPTY:
            return EMPTY
        return N("seq", pre + _sets_to_atoms(sets) + post)

    if kof is not None:
        if starts or ends or chains or length or not_starts or not_ends:
            return None
        sets, k = kof
        sets = fix(sets)
        if sets is EMPTY:
            return EMPTY
        unit = N("seq", [cls_rep(Wp)] + _sets_to_atoms(sets))
        return N("seq", pre + [N("rep", [unit], m=k, n=None), cls_rep(Wp)]
                 + post)

    if length is not None:
        if starts or ends or chains or not_starts or not_ends:
            return None
        lcls, lm, ln = length
        sub = lcls & Wp
        if not sub:
            return EMPTY
        return N("seq", pre + [N("rep", [N("class", val=frozenset(sub))],
                                m=max(lm, m0), n=ln)] + post)

    if len(chains) > 1:
        if starts or ends or not_starts or not_ends:
            return None
        runs = [r for ch in chains for r in ch]
        if len(runs) > 3:
            return None
        branches = []
        for perm in permutations(runs):
            kids = [cls_rep(Wp)]
            for run in perm:
                sets = fix(run)
                if sets is EMPTY:
                    return EMPTY
                kids.extend(_sets_to_atoms(sets))
                kids.append(cls_rep(Wp))
            branches.append(N("seq", pre + kids + post))
        return N("alt", branches)

    if chains:
        if starts or ends or not_starts or not_ends:
            return None
        kids = [cls_rep(Wp)]
        for run in chains[0]:
            sets = fix(run)
            if sets is EMPTY:
                return EMPTY
            kids.extend(_sets_to_atoms(sets))
            kids.append(cls_rep(Wp))
        if m0 and all(c.kind == "rep" and c.m == 0 for c in kids):
            kids[0] = N("rep", kids[0].kids, m=1, n=kids[0].n)
        return N("seq", pre + kids + post)

    # starts/ends with first/last-char subtractions: one long branch plus
    # fixed-length overlap templates (a single-vowel word both starts and
    # ends with its vowel — length 1 must survive)
    P = fix(starts) if starts is not None else []
    S = fix(ends) if ends is not None else []
    if P is EMPTY or S is EMPTY:
        return EMPTY
    ls, le = len(P), len(S)
    if ls + le > 6:
        return None

    branches = []
    first = list(P) if P else ([Wp - not_starts] if not_starts else [])
    last = list(S) if S else ([Wp - not_ends] if not_ends else [])

    def edgefix(sets, at0, atL):
        sets = [set(s) for s in sets]
        if at0 and sets:
            sets[0] -= not_starts
        if atL and sets:
            sets[-1] -= not_ends
        return None if any(not s for s in sets) else sets

    fa = edgefix(first, True, False) if first else []
    fb = edgefix(last, False, True) if last else []
    if fa is not None and fb is not None:
        branches.append(N("seq", _sets_to_atoms(fa or []) + [cls_rep(Wp)]
                         + _sets_to_atoms(fb or [])))
    for L in range(max(ls, le, m0, 1),
                   max(ls + le, len(first) + len(last))):
        tmpl = []
        for i in range(L):
            s = set(Wp)
            if i < ls:
                s &= P[i]
            if i >= L - le:
                s &= S[i - (L - le)]
            if i == 0:
                s -= not_starts
            if i == L - 1:
                s -= not_ends
            if not s:
                tmpl = None
                break
            tmpl.append(s)
        if tmpl:
            branches.append(N("seq", _sets_to_atoms(tmpl)))
    if not branches:
        return EMPTY
    if len(branches) == 1:
        kids = list(branches[0].kids)
        if m0 and all(c.kind == "rep" and c.m == 0 for c in kids):
            kids[0] = N("rep", kids[0].kids, m=1, n=kids[0].n)
        return N("seq", pre + kids + post)
    return N("seq", pre + [N("alt", branches)] + post) if pre or post \
        else N("alt", branches)


# ------------------------------------------------- other and-node helpers

def _has_bound(nd):
    if nd.kind == "bound":
        return True
    return any(_has_bound(c) for c in nd.kids)


def _edge_shape(nd):
    kids = list(nd.kids) if nd.kind == "seq" else [nd]
    if any(k.kind == "bound" for k in kids):
        return None
    lead = len(kids) > 1 and is_dotstar(kids[0])
    trail = len(kids) > 1 and is_dotstar(kids[-1])
    atoms = kids[1 if lead else 0: -1 if trail else len(kids)]
    if not atoms or not all(atom_set(a) is not None for a in atoms):
        return None
    sets = [atom_set(a) for a in atoms]
    if lead and trail:
        return ("contains", sets)
    if trail:
        return ("starts", sets)
    if lead:
        return ("ends", sets)
    return ("exact", sets)


def _starts_ends(P, S):
    branches = [N("seq", _sets_to_atoms(P) + [dotstar()]
                  + _sets_to_atoms(S))]
    for L in range(max(len(P), len(S)), len(P) + len(S)):
        tmpl, ok = [], True
        for i in range(L):
            s = PRINTABLE
            if i < len(P):
                s = s & P[i]
            if i >= L - len(S):
                s = s & S[i - (L - len(S))]
            if not s:
                ok = False
                break
            tmpl.append(s)
        if ok:
            branches.append(N("seq", _sets_to_atoms(tmpl)))
    return N("alt", branches) if len(branches) > 1 else branches[0]


def _merge_prefix(A, B):
    merged = []
    for i in range(max(len(A), len(B))):
        s = PRINTABLE
        if i < len(A):
            s = s & A[i]
        if i < len(B):
            s = s & B[i]
        if not s:
            return EMPTY
        merged.append(s)
    return N("seq", _sets_to_atoms(merged) + [dotstar()])


def _norm_seq(kids):
    flat = []
    for c in kids:
        if c.kind == "seq":
            flat.extend(_norm_seq(c.kids))
        elif not is_eps(c):
            flat.append(c)
    out = []
    for c in flat:
        if (out and c.kind == "rep" and (c.m, c.n) == (0, None)
                and c.kids[0].kind == "class" and out[-1].kind == "class"
                and out[-1].val == c.kids[0].val):
            out[-1] = N("rep", [out[-1]], m=1, n=None)
        else:
            out.append(c)
    return out


# ----------------------------------------------------------------- rewrite

def rewrite(nd):
    """Bottom-up rewrite. Returns a node or EMPTY."""
    new_kids = []
    for c in nd.kids:
        r = rewrite(c)
        if r is EMPTY:
            if nd.kind in ("seq", "and"):
                return EMPTY
            if nd.kind == "not":
                return dotstar()                      # ~∅ = everything
            if nd.kind == "rep":
                return eps_node() if nd.m == 0 else EMPTY
            if nd.kind == "alt":
                continue                              # drop ∅ branch
            return EMPTY
        new_kids.append(r)
    if nd.kind == "alt" and not new_kids:
        return EMPTY
    if nd.kind == "alt" and len(new_kids) == 1:
        return new_kids[0]
    nd = N(nd.kind, new_kids, nd.val, nd.m, nd.n)

    # REP-1: (.*X.*){1,} == .*X.*
    if nd.kind == "rep" and (nd.m, nd.n) == (1, None):
        c = nd.kids[0]
        ck = c.kids if c.kind == "seq" else [c]
        if len(ck) >= 2 and is_dotstar(ck[0]) and is_dotstar(ck[-1]):
            return c

    if nd.kind == "seq":
        kids = _norm_seq(nd.kids)
        # \b F \b -> \b beside a dotstar, for ε-admitting F
        out, i = [], 0
        while i < len(kids):
            c = kids[i]
            if (c.kind == "bound" and i + 2 < len(kids)
                    and kids[i + 2].kind == "bound"
                    and eps_must(kids[i + 1])
                    and ((out and is_dotstar(out[-1]))
                         or (i + 3 < len(kids)
                             and is_dotstar(kids[i + 3])))):
                out.append(c)
                i += 3
                continue
            out.append(c)
            i += 1
        kids = out
        # ~(X*) beside .*
        out = []
        for i, c in enumerate(kids):
            near_ds = (out and is_dotstar(out[-1])) \
                or (i + 1 < len(kids) and is_dotstar(kids[i + 1]))
            if (near_ds and c.kind == "not"
                    and c.kids[0].kind == "rep"
                    and c.kids[0].m == 0 and c.kids[0].n is None):
                inner = c.kids[0].kids[0]
                if inner.kind in ("class", "char", "dot"):
                    cset = atom_set(inner)
                    comp = PRINTABLE - cset
                    if not comp:
                        return EMPTY          # ~((.)*) — nothing left
                    neg = N("class", val=frozenset(comp))
                    right_ds = i + 1 < len(kids) and is_dotstar(kids[i + 1])
                    # one-sided: the non-C char sits anywhere in the factor,
                    # so pad the closed side with C*
                    if right_ds:
                        out.extend([cls_rep(cset), neg])
                    else:
                        out.extend([neg, cls_rep(cset)])
                    continue
                flang = finite_lang(inner)
                if flang and all(len(s) >= 2 for s in flang):
                    out.append(N("dot"))      # any 1 char breaks X*
                    continue
            out.append(c)
        kids = out
        # ε-admitting factor directly beside .* is vacuous
        out = []
        for i, c in enumerate(kids):
            if eps_must(c) and not is_eps(c):
                left = out and is_dotstar(out[-1])
                right = i + 1 < len(kids) and is_dotstar(kids[i + 1])
                if left or right:
                    continue
            out.append(c)
        if not out:
            return eps_node()
        return out[0] if len(out) == 1 else N("seq", out)

    if nd.kind != "and":
        return nd

    # dedupe identical operands
    ops, seen = [], set()
    for c in nd.kids:
        key = serialize(c)
        if key not in seen:
            seen.add(key)
            ops.append(c)
    if len(ops) == 1:
        return ops[0]

    # FINITE-& with \b edges and negated operands
    bf = [bounded_finite(c) for c in ops]
    pos_ix = [i for i, b in enumerate(bf) if b is not None and not b[3]]
    neg_ix = [i for i, b in enumerate(bf) if b is not None and b[3]]
    if len(pos_ix) >= 2 or (len(pos_ix) >= 1 and neg_ix):
        inter = None
        pre = post = False
        for i in pos_ix:
            inter = bf[i][1] if inter is None else inter & bf[i][1]
            pre |= bf[i][0]
            post |= bf[i][2]
        for i in neg_ix:
            inter = inter - bf[i][1]
            pre |= bf[i][0]
            post |= bf[i][2]
        if pre or post:
            # a single char inside \b..\b must be a word char — drop the
            # never-matchable rest (also keeps '/' out of vim classes)
            inter = {s for s in inter
                     if len(s) > 1 or s in frozenset(WORD)}
        if not inter:
            return EMPTY
        core = lang_to_node(inter)
        merged = ([N("bound")] if pre else []) + [core] \
            + ([N("bound")] if post else [])
        merged = merged[0] if len(merged) == 1 else N("seq", merged)
        rest = [c for i, c in enumerate(ops) if i not in pos_ix + neg_ix]
        if not rest:
            return merged
        ops = [merged] + rest
        if len(ops) == 1:
            return ops[0]

    # FINITE-FILTER: enumerate a finite operand, keep the strings every
    # other operand accepts (the matcher speaks every construct); \b-bearing
    # partners are excluded — their sub-span context is unknowable
    langs = [(i, finite_lang(c)) for i, c in enumerate(ops)]
    langs = [(i, l) for i, l in langs
             if l is not None and len(l) <= FINITE_CAP]
    if langs:
        fi, F = min(langs, key=lambda x: len(x[1]))
        others = [c for i, c in enumerate(ops) if i != fi]
        if all(not _has_bound(c) for c in others):
            keep = {s for s in F if all(matcher(c, s) for c in others)}
            if not keep:
                return EMPTY
            return lang_to_node(keep)

    # WORD-& multi-constraint folding
    for wi, c in enumerate(ops):
        wc = word_core(c)
        if not wc:
            continue
        pre, W, post, m0 = wc
        folded = fold_word(W, pre, post, m0, ops[:wi] + ops[wi + 1:])
        if folded is EMPTY:
            return EMPTY
        if folded is not None:
            if folded.kind == "seq":
                flat = _norm_seq(folded.kids)
                return flat[0] if len(flat) == 1 else N("seq", flat)
            return folded

    # STARTS / ENDS / CONTAINS combinations
    shapes = [_edge_shape(c) for c in ops]
    if all(s is not None for s in shapes):
        kinds = sorted(s[0] for s in shapes)
        if len(ops) == 2:
            a, b = shapes
            if {a[0], b[0]} == {"starts", "ends"}:
                P = a[1] if a[0] == "starts" else b[1]
                S = a[1] if a[0] == "ends" else b[1]
                return _starts_ends(P, S)
            if kinds == ["starts", "starts"]:
                return _merge_prefix(a[1], b[1])
            if kinds == ["ends", "ends"]:
                m = _merge_prefix(list(reversed(a[1])),
                                  list(reversed(b[1])))
                if m is EMPTY:
                    return EMPTY
                atoms = m.kids[:-1]
                return N("seq", [dotstar()] + list(reversed(atoms)))
        if 2 <= len(ops) <= 3 and all(k == "contains" for k in kinds):
            branches = []
            for perm in permutations([s[1] for s in shapes]):
                kids = [dotstar()]
                for sets in perm:
                    kids.extend(_sets_to_atoms(sets))
                    kids.append(dotstar())
                branches.append(N("seq", kids))
            return N("alt", branches)

    return nd


def semantic_rewrite(text):
    """-> (status, new_text, new_ast); status: rewritten | empty | stuck."""
    try:
        ast = parse(text)
    except Exception:
        return "stuck", None, None
    new = rewrite(ast)
    if new is EMPTY:
        return "empty", None, None
    new_text = serialize(new)
    if new_text == text:
        return "stuck", None, new
    return "rewritten", new_text, new
