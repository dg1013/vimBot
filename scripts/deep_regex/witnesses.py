"""Adversarial witness sampling for probe buffers (dataset v3, Phase A1).

The 2026-08-14 audit measured that 66% of cases could not distinguish the
gold command from a sloppier variant (anchor dropped, word boundary
stripped): the old buffers used 3 members + 3 lines from a static junk pool,
so nothing near the decision boundary ever appeared. This module builds
buffers whose non-members are *near-misses* — strings sampled from mutated
versions of the pattern's own AST (abstract syntax tree) and then verified as non-members by the
span-DP (dynamic programming) oracle — so a slightly-wrong command visibly changes the result.

Witness classes (emitted only when applicable to the pattern):
  anchor    a member wrapped in extra chars — separates full-line match
            from containment
  boundary  the matched word glued to word chars — separates \\b from none
  case      a case-flipped member — separates case-specific classes
  count     sampled at one repetition below the minimum — separates {n}
            from {n-1} and + from nothing
  adjacency two *members*: slack forced empty (XY) and forced wide
            (X..gap..Y) — both readings of "followed by" are witnessed
  negation  a member of the un-negated pattern — separates ~(X) from X

Every emitted line is oracle-verified against the ORIGINAL pattern; a
witness generator that cannot produce a verified line for a pattern simply
contributes nothing. Buffers carry 2–5 members and 3–5 non-members
(6–10 lines), deterministically from the caller's rng.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convertibility import has_ops, split_top, strip_outer  # noqa: E402
from dsl2vim import reduce_class_and  # noqa: E402
from dsl_ast import WORD, dsl_match, parse  # noqa: E402
from rewrite_stuck import semantic_rewrite  # noqa: E402

MAX_LEN = 40
_PADS = ["x", "9", "!", "Z", " q"]
_PADS_NONWORD = ["!", " ", "?", ". "]   # for \b-edged inners: a word-char pad
                                        # would kill the contained match
_GENERIC_POOL = ["@@@@", "####", "zqzqzq", "12321", "   ", "!?!?", "T",
                 "unrelated words here", "9 9 9", "xyzzy plugh"]


# ------------------------------------------------------------- sampling

def sample_ast(nd, rng, force=None):
    """Member string of an op-free AST; force maps id(rep-node) -> count."""
    force = force or {}
    k = nd.kind
    if k == "char":
        return nd.val
    if k == "class":
        return rng.choice(sorted(nd.val))
    if k == "dot":
        return rng.choice("axz9 !Tq")
    if k == "bound":
        return ""
    if k == "seq":
        return "".join(sample_ast(c, rng, force) for c in nd.kids)
    if k == "alt":
        return sample_ast(rng.choice(nd.kids), rng, force)
    if k == "rep":
        if id(nd) in force:
            count = force[id(nd)]
        else:
            lo = nd.m
            hi = nd.n if nd.n is not None else lo + 2
            count = rng.randint(lo, max(lo, min(hi, lo + 2)))
        return "".join(sample_ast(nd.kids[0], rng, force) for _ in range(count))
    raise ValueError(k)


def op_free_sources(t):
    """DSL (domain-specific language) texts to sample from, best-effort, for the original pattern t."""
    out = []
    if not has_ops(t):
        out.append(t)
    else:
        reduced, deg = reduce_class_and(t)
        if not deg and not has_ops(reduced):
            out.append(reduced)
        status, new_text, _ = semantic_rewrite(t)
        if status == "rewritten" and new_text and not has_ops(new_text):
            out.append(new_text)
        e = strip_outer(t)
        parts = split_top(e, "&")
        if len(parts) > 1:
            for p in parts:
                p = strip_outer(p)
                if not has_ops(p):
                    out.append(p)
    return out


def _asts(t):
    out = []
    for src in op_free_sources(t):
        try:
            out.append(parse(src))
        except Exception:
            pass
    return out


def _ok(s, taken):
    return 0 < len(s) <= MAX_LEN and "\n" not in s and s not in taken


def sample_members(t, rng, want=3, attempts=60, taken=()):
    """Oracle-verified member lines for t."""
    found, asts = [], _asts(t)
    for _ in range(attempts):
        if len(found) >= want or not asts:
            break
        try:
            s = sample_ast(rng.choice(asts), rng)
        except ValueError:
            continue
        if _ok(s, found) and s not in taken and dsl_match(t, s):
            found.append(s)
    return found


# ------------------------------------------------------------- witnesses

def _walk(nd):
    yield nd
    for c in nd.kids:
        yield from _walk(c)


def _slack_reps(root):
    """rep nodes over dot with unbounded max — the .* / .+ slack."""
    return [n for n in _walk(root)
            if n.kind == "rep" and n.n is None and n.kids
            and n.kids[0].kind == "dot"]


def inner_membership(t, pol):
    """Membership test for the pattern the COMMAND full-matches.

    The converter compiles ~(X) by inverting command polarity (pol 'v'), so
    the command's inner language is the complement of L(t) there. Witness
    geometry (what a sloppier command would treat differently) follows the
    inner language, not the source."""
    if pol == "v":
        return lambda s: not dsl_match(t, s)
    return lambda s: dsl_match(t, s)


def inner_member_samples(t, pol, rng, want=3):
    """Member lines of the command's inner language."""
    if pol != "v":
        return sample_members(t, rng, want=want)
    # complement: sample the un-negated source (exact check via oracle)
    variant = t[2:-1] if t.startswith("~(") and t.endswith(")") \
        else t.replace("~(", "(")
    is_inner = inner_membership(t, pol)
    out, asts = [], _asts(variant)
    for _ in range(40):
        if len(out) >= want or not asts:
            break
        try:
            s = sample_ast(rng.choice(asts), rng)
        except ValueError:
            continue
        if _ok(s, out) and is_inner(s):
            out.append(s)
    return out


def anchor_witnesses(t, members, rng, cap=2, pol="g"):
    """Inner-members wrapped in pad chars so full-match breaks while the
    contained match survives; none for effectively slack-wrapped inners."""
    is_inner = inner_membership(t, pol)
    base = [m for m in members if is_inner(m)] \
        or inner_member_samples(t, pol, rng)
    pads = _PADS_NONWORD if "\\b" in t else _PADS
    out = []
    for m in base:
        for pad in pads:
            for cand in (pad + m, m + pad, pad + m + pad):
                if len(out) >= cap:
                    return out
                if _ok(cand, out) and cand not in base \
                        and not is_inner(cand):
                    out.append(cand)
    return out


def boundary_witnesses(t, members, rng, cap=2, pol="g"):
    """Glue word chars against word-run edges so \\b stops holding."""
    if "\\b" not in t:
        return []
    is_inner = inner_membership(t, pol)
    base = [m for m in members if is_inner(m)] \
        or inner_member_samples(t, pol, rng)
    out = []
    for m in base:
        edits = []
        for i in range(len(m) + 1):
            left = m[i - 1] in WORD if i > 0 else False
            right = m[i] in WORD if i < len(m) else False
            if left != right:                      # a word boundary in m
                edits.append(m[:i] + "x" + m[i:])
        rng.shuffle(edits)
        for cand in edits:
            if len(out) >= cap:
                return out
            if _ok(cand, out) and not is_inner(cand):
                out.append(cand)
    return out


def case_witnesses(t, members, cap=1):
    out = []
    for m in members:
        for cand in (m.swapcase(),):
            if len(out) >= cap:
                return out
            if cand != m and _ok(cand, out) and not dsl_match(t, cand):
                out.append(cand)
    return out


def count_witnesses(t, rng, cap=2, attempts=12):
    """Sample with one counted rep forced below its minimum."""
    out = []
    for ast in _asts(t):
        targets = [n for n in _walk(ast) if n.kind == "rep" and n.m >= 1]
        for nd in targets:
            for _ in range(attempts):
                if len(out) >= cap:
                    return out
                try:
                    s = sample_ast(ast, rng, force={id(nd): nd.m - 1})
                except ValueError:
                    break
                if _ok(s, out) and not dsl_match(t, s):
                    out.append(s)
                    break
    return out


def adjacency_members(t, rng, cap=2, attempts=12):
    """Member pairs: slack forced empty (adjacent) and forced wide (gap).
    Both verified members — they witness either reading of 'followed by'."""
    out = []
    for ast in _asts(t):
        slacks = [n for n in _slack_reps(ast) if n.m == 0]
        if not slacks:
            continue
        inner = [n for n in slacks]
        for width in (0, 3):
            for _ in range(attempts):
                try:
                    s = sample_ast(ast, rng,
                                   force={id(n): width for n in inner})
                except ValueError:
                    break
                if _ok(s, out) and dsl_match(t, s):
                    out.append(s)
                    break
        if len(out) >= cap:
            return out[:cap]
    return out


def negation_witnesses(t, rng, cap=2, attempts=20):
    """Members of the pattern with its negations removed -> non-members."""
    if "~" not in t:
        return []
    variant = t.replace("~(", "(")
    out = []
    asts = _asts(variant)
    for _ in range(attempts):
        if len(out) >= cap or not asts:
            break
        try:
            s = sample_ast(rng.choice(asts), rng)
        except ValueError:
            continue
        if _ok(s, out) and not dsl_match(t, s):
            out.append(s)
    return out


# ---------------------------------------------------------- buffer build

def build_buffer(t, rng, probe_lines=(), pol="g"):
    """Assemble a witness-bearing buffer for pattern t (command polarity
    pol — see inner_membership).

    Returns {"lines": [...], "truth": {line: bool}, "labels": {line: cls}}
    or None when either side cannot be populated."""
    truth, labels = {}, {}

    def add(s, is_member, label):
        if s and s not in truth and len(s) <= MAX_LEN and "\n" not in s:
            truth[s] = is_member
            labels[s] = label

    for s in probe_lines:
        if s and len(s) <= MAX_LEN and "\n" not in s:
            add(s, dsl_match(t, s), "probe")

    members = [s for s, ok in truth.items() if ok]
    for s in adjacency_members(t, rng):
        add(s, True, "adjacency")
    members = [s for s, ok in truth.items() if ok]
    if len(members) < 2:
        for s in sample_members(t, rng, want=3 - len(members), taken=truth):
            add(s, True, "member")
    members = [s for s, ok in truth.items() if ok]
    if len(members) < 2:
        return None

    for label, gen in (("anchor", anchor_witnesses(t, members, rng, pol=pol)),
                       ("boundary", boundary_witnesses(t, members, rng, pol=pol)),
                       ("case", case_witnesses(t, members)),
                       ("count", count_witnesses(t, rng)),
                       ("negation", negation_witnesses(t, rng))):
        for s in gen:
            # a witness's buffer side follows the ORACLE, not the class:
            # under inverted polarity an anchor witness is a t-member
            add(s, dsl_match(t, s), label)

    nonmembers = [s for s, ok in truth.items() if not ok]
    fillers = [s for s in _GENERIC_POOL if s not in truth
               and not dsl_match(t, s)]
    rng.shuffle(fillers)
    while len(nonmembers) < 2 and fillers:
        s = fillers.pop()
        add(s, False, "generic")
        nonmembers = [s for s, ok in truth.items() if not ok]
    if len(nonmembers) < 2:
        return None

    # trim to 6-10 lines: witnesses outrank probe/generic filler (a probe
    # line is junk from the legacy pool; a witness is the whole point)
    WITNESS = ("anchor", "boundary", "case", "count", "negation")

    def n_prio(s):
        return 0 if labels[s] in WITNESS else 1

    def m_prio(s):
        return 0 if labels[s] in WITNESS or labels[s] == "adjacency" else 1

    members = [s for s, ok in truth.items() if ok]
    nonmembers = [s for s, ok in truth.items() if not ok]
    keep_m = sorted(members, key=m_prio)[:5]
    wit_n = [s for s in nonmembers if labels[s] in WITNESS]
    junk_n = [s for s in nonmembers if labels[s] not in WITNESS]
    # junk (probe/generic) non-members are a fallback, not filler: once two
    # real witnesses exist they add nothing and dilute the buffer
    # (owner review of pilot card FX-297, 2026-08-14)
    keep_n = (wit_n[:5] if len(wit_n) >= 2
              else (wit_n + junk_n)[:max(2, len(wit_n))])
    lines = keep_m + keep_n
    if len(lines) < 6:
        extra = sample_members(t, rng, want=6 - len(lines), taken=truth)
        for s in extra:
            add(s, True, "member")
            lines.append(s)
    if len(lines) < 6:
        return None
    rng.shuffle(lines)
    return {"lines": lines,
            "truth": {s: truth[s] for s in lines},
            "labels": {s: labels[s] for s in lines}}
