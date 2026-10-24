"""Deterministic English from normalized DSL (domain-specific language) ASTs (Phase A3).

The instruction frame comes from the pattern's anchoring, so the A-anchor
defect class cannot be generated: slack on both edges -> a containment
frame, one edge -> start/end frames, none -> an exact frame with a
disclosing word. Sequence gaps are worded explicitly ("… then, anywhere
later, …") versus adjacency ("immediately followed by"), which settles the
SEQ ambiguity by disclosure. Counts are explicit ("exactly three" / "three
or more"). Whatever cannot be verbalized inside the complexity budget
returns None and is routed to degenerate quarantine by the caller.

render_variants(text, rng, k) returns k seeded surface variants of the
noun phrase ("lines containing …"); make_candidates' instruction_for()
wraps them with the operation verb exactly as before.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dsl_ast import N, parse, serialize  # noqa: E402
from simplify_ast import is_slack, simplify  # noqa: E402

MAX_UNITS = 8      # leaf phrases before we call the English dishonestly long

_LOW = frozenset("abcdefghijklmnopqrstuvwxyz")
_UP = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIG = frozenset("0123456789")
_VOW = frozenset("AEIOUaeiou")
_CLASS_NAMES = {
    _DIG: ("a digit", "digits"),
    _LOW: ("a lowercase letter", "lowercase letters"),
    _UP: ("an uppercase letter", "uppercase letters"),
    _LOW | _UP: ("a letter", "letters"),
    _VOW: ("a vowel", "vowels"),
    frozenset("aeiou"): ("a lowercase vowel", "lowercase vowels"),
    frozenset("AEIOU"): ("an uppercase vowel", "uppercase vowels"),
    _LOW | _UP | _DIG: ("a letter or digit", "letters or digits"),
}

# StructuredRegex vocabulary (2026-08-17): its <spec> class and the
# printable-complement classes its notcc() operator produces. Additive
# keys only — deep-regex patterns never form these sets, so existing
# renderings (and their goldens) are untouched.
_ALL = frozenset(chr(c) for c in range(32, 127))
_SPEC = _ALL - _LOW - _UP - _DIG - frozenset(" ")
_CLASS_NAMES.update({
    _SPEC: ("a special character", "special characters"),
    _ALL - _DIG: ("a character other than a digit",
                  "characters other than digits"),
    _ALL - _LOW: ("a character other than a lowercase letter",
                  "characters other than lowercase letters"),
    _ALL - _UP: ("a character other than an uppercase letter",
                 "characters other than uppercase letters"),
    _ALL - (_LOW | _UP): ("a character other than a letter",
                          "characters other than letters"),
})

FRAMES = {
    "contains": ["lines containing {b}", "lines with {b}",
                 "lines that contain {b}"],
    "starts": ["lines starting with {b}", "lines that start with {b}",
               "lines beginning with {b}"],
    "ends": ["lines ending with {b}", "lines that end with {b}"],
    "exact": ["lines that are exactly {b}",
              "lines consisting entirely of {b}",
              "lines made up of nothing but {b}"],
}
GAPS = [", then anywhere later {y}", ", with {y} somewhere after it"]
ADJ = [" immediately followed by {y}"]


class TooComplex(Exception):
    pass


class _Budget:
    def __init__(self, n):
        self.left = n

    def spend(self, k=1):
        self.left -= k
        if self.left < 0:
            raise TooComplex()


def _class_phrase(cs, plural=False):
    if cs in _CLASS_NAMES:
        return _CLASS_NAMES[cs][1 if plural else 0]
    full = frozenset(chr(c) for c in range(32, 127))
    comp = full - cs
    if len(comp) <= 3 and comp:
        inner = "', '".join(sorted(comp))
        base = f"any character except '{inner}'"
        return base + ("s" if plural and False else "")
    if len(cs) <= 5:
        inner = "', '".join(sorted(cs))
        return ("one of the characters " if not plural
                else "characters from ") + f"'{inner}'"
    body = serialize(N("class", val=cs))
    return ("a character matching " if not plural
            else "characters matching ") + body


def _merge_literals(kids):
    """Consecutive char nodes -> ('lit', 'dog') pseudo-elements."""
    out, run = [], []
    for c in kids:
        if not isinstance(c, tuple) and c.kind == "char":
            run.append(c.val)
        else:
            if run:
                out.append(("lit", "".join(run)))
                run = []
            out.append(c)
    if run:
        out.append(("lit", "".join(run)))
    return out


def _is_wordish(el):
    if isinstance(el, tuple):
        return True
    if el.kind == "char":
        return el.val in (_LOW | _UP | _DIG)
    if el.kind == "class":
        return el.val <= (_LOW | _UP | _DIG)
    if el.kind == "rep" and el.kids:
        return _is_wordish(el.kids[0]) if not isinstance(el.kids[0], tuple) \
            else True
    if el.kind in ("alt", "seq"):
        return all(_is_wordish(c) for c in el.kids)
    return False


_NONWORD = frozenset(chr(c) for c in range(32, 127)) - (_LOW | _UP | _DIG | {"_"})


def _is_nonword_filler(el):
    """Starred class with no letters: the separator junk between words
    (the corpus filler [ -@[-`{-~]* contains digits — words here means
    letter-words, which is what the phrase says)."""
    return (not isinstance(el, tuple) and el.kind == "rep" and el.m == 0
            and el.kids and el.kids[0].kind == "class"
            and not (el.kids[0].val & (_LOW | _UP)))


def _word_content_phrase(content, budget):
    """Render the letters between two bounds as a word phrase, or None."""
    def letters_star(el):
        return (not isinstance(el, tuple) and el.kind == "rep"
                and el.n is None and el.kids[0].kind == "class"
                and el.kids[0].val == (_LOW | _UP))

    lits = [c for c in content if isinstance(c, tuple)]
    reps = [c for c in content if not isinstance(c, tuple)]
    budget.spend()

    def char_alt(el):
        """alt of chars / short literals -> "'y' or 'i'" (affix choices)."""
        if isinstance(el, tuple) or el.kind != "alt":
            return None
        opts = []
        for b in el.kids:
            if b.kind == "char":
                opts.append(b.val)
            elif b.kind == "seq" and all(c.kind == "char" for c in b.kids):
                opts.append("".join(c.val for c in b.kids))
            else:
                return None
        return " or ".join(f"'{o}'" for o in opts)

    if len(content) == 1 and lits:
        return f"the whole word '{lits[0][1]}'"
    if len(content) == 1 and not lits:
        ca = char_alt(content[0])
        if ca:
            return f"the whole word {ca}"
    if len(content) == 2 and len(lits) == 0:
        ca = char_alt(content[1])
        if ca and reps and content[0] is reps[0] \
                and reps[0].kind == "rep" and reps[0].n is None \
                and reps[0].kids[0].kind == "class" \
                and reps[0].kids[0].val == (_LOW | _UP):
            return f"a word ending in {ca}"
    if len(content) == 1 and reps and letters_star(reps[0]):
        return "a word" if reps[0].m else "a word"
    if len(content) == 2 and len(lits) == 1 and len(reps) == 1 \
            and letters_star(reps[0]):
        if isinstance(content[0], tuple):
            return f"a word starting with '{lits[0][1]}'"
        return f"a word ending in '{lits[0][1]}'"
    if (len(content) == 3 and len(lits) == 1 and len(reps) == 2
            and all(letters_star(r) for r in reps)
            and isinstance(content[1], tuple)):
        return f"a word containing '{lits[0][1]}'"
    if (len(content) == 1 and reps and reps[0].kind == "rep"
            and reps[0].kids[0].kind == "class"
            and reps[0].kids[0].val <= (_LOW | _UP)):
        r = reps[0]
        if r.n == r.m:
            return f"a word of exactly {r.m} letters"
        if r.n is None and r.m > 1:
            return f"a word of {r.m} or more letters"
    # alternating star/literal: a word containing 'c' and later 'e'
    def letters_star2(el):
        return (not isinstance(el, tuple) and el.kind == "rep"
                and el.n is None and el.kids[0].kind == "class"
                and el.kids[0].val == (_LOW | _UP))
    inner_lits = []
    ok = True
    for idx, c in enumerate(content):
        if idx % 2 == 0:
            ok = ok and letters_star2(c)
        else:
            ok = ok and isinstance(c, tuple)
            if isinstance(c, tuple):
                inner_lits.append(c[1])
    if ok and len(content) % 2 == 1 and len(inner_lits) >= 2:
        seq = ", then ".join(f"'{l}'" for l in inner_lits)
        return f"a word containing {seq} in that order"
    return None


def _word_shape(elems, i, rng, budget):
    """bound … bound  ->  'a word that …' phrase; returns (phrase, next_i)
    or None when the delimited content isn't wordish."""
    j = i + 1
    content = []
    while j < len(elems):
        el = elems[j]
        if not isinstance(el, tuple) and el.kind == "bound":
            break
        content.append(el)
        j += 1
    else:
        return None
    if not content:
        return None
    if len(content) == 1 and not isinstance(content[0], tuple) \
            and content[0].kind == "and":
        # \b( X & Y & ... )\b — a word satisfying several constraints
        wa = _word_and(content[0], rng, budget, implicit_word=True)
        if wa:
            return (wa, j + 1)
        return None
    if not all(_is_wordish(c) for c in content):
        return None
    phrase = _word_content_phrase(content, budget)
    if phrase is not None:
        return (phrase, j + 1)
    inner = _render_seq(content, rng, budget, allow_word=False)
    return (f"a whole word made of {inner}", j + 1)


def _word_count_shape(ast, rng, budget):
    """Counting repetitions.

    Two corpus idioms, with DIFFERENT count semantics:
    - filler form  rep(seq(nonletter*, \\b X \\b, nonletter*), m, n):
      the whole line is fillers+words, so counts are EXACT;
    - contains form  rep(seq(.*, X, .*), m, n): slack absorbs anything,
      so {m} means AT LEAST m occurrences.
    Returns the count phrase or None."""
    if ast.kind != "rep" or not ast.kids or ast.kids[0].kind != "seq":
        return None
    kids = _merge_literals(ast.kids[0].kids)
    contains_form = any(not isinstance(c, tuple) and is_slack(c)
                        for c in kids)
    core = [c for c in kids
            if not _is_nonword_filler(c)
            and not (not isinstance(c, tuple) and is_slack(c))]
    if not core or len(core) > 7:
        return None
    has_bounds = (not isinstance(core[0], tuple) and core[0].kind == "bound"
                  and not isinstance(core[-1], tuple)
                  and core[-1].kind == "bound")
    if has_bounds:
        content = core[1:-1]
        if not content or not all(_is_wordish(c) for c in content):
            return None
        inner = _word_content_phrase(content, budget)
        if inner is None and contains_form and len(content) == 1 \
                and not isinstance(content[0], tuple) \
                and content[0].kind == "class" \
                and content[0].val <= (_LOW | _UP | _DIG):
            budget.spend()
            inner = f"a word starting with {_class_phrase(content[0].val)}"
        if inner is None:
            return None                # never invent "a word": dishonest
        noun = inner.replace("a word", "words").replace("the whole word",
                                                        "the word")
    elif contains_form and len(core) == 1 and not isinstance(core[0], tuple) \
            and core[0].kind == "class":
        budget.spend()
        inner = _class_phrase(core[0].val)
        noun = _class_phrase(core[0].val, plural=True)
    elif contains_form and len(core) == 1 and isinstance(core[0], tuple):
        budget.spend()
        inner = f"'{core[0][1]}'"
        noun = f"occurrences of '{core[0][1]}'"
    else:
        return None
    m, n = ast.m, ast.n
    if contains_form:
        # stacked contains: only a lower bound is real
        if m == 0:
            return None
        if n is not None and n != m:
            return None                    # bounded contains-count: not exact
        bare = inner
        for pre in ("an ", "a "):
            if bare.startswith(pre):
                bare = bare[len(pre):]
        phrase = (f"at least {m} {noun}" if m != 1
                  else f"at least one {bare}")
        return ("contains", phrase)
    if n == m:
        return ("exact", f"exactly {m} {noun}" if m != 1
                else "exactly one " + (inner[2:] if inner.startswith("a ")
                                       else inner))
    if n is None:
        if m == 0:
            return None                       # unconstrained: frame decides
        return ("exact", f"{m} or more {noun}")
    if m == 0:
        return ("exact", f"at most {n} {noun}")
    return ("exact", f"between {m} and {n} {noun}")


def _atom(el, rng, budget, plural=False):
    budget.spend()
    if isinstance(el, tuple):
        return f"'{el[1]}'"
    k = el.kind
    if k == "class":
        return _class_phrase(el.val, plural)
    if k == "dot":
        return "characters" if plural else "any single character"
    if k == "char":
        return f"'{el.val}'"
    if k == "rep":
        return _rep(el, rng, budget)
    if k == "alt":
        parts = [_render_node(c, rng, budget) for c in el.kids[:4]]
        if len(el.kids) > 4:
            raise TooComplex()
        return "either " + " or ".join(parts) if len(parts) == 2 \
            else "any of: " + ", ".join(parts)
    if k == "seq":
        return _render_seq(_merge_literals(el.kids), rng, budget)
    if k == "bound":
        raise TooComplex()          # stray boundary: not honestly wordable
    if k == "and":
        wa = _word_and(el, rng, budget)
        if wa:
            return wa
        raise TooComplex()
    if k == "not":
        raise TooComplex()          # nested ~ residue: route degenerate
    raise TooComplex()


def _is_word_atom(el):
    """seq(\\b, [A-Za-z]+ / [A-Za-z]*lit..., \\b) — 'a word' operand."""
    if isinstance(el, tuple) or el.kind != "seq":
        return None
    kids = _merge_literals(el.kids)
    if len(kids) < 3:
        return None
    first, last = kids[0], kids[-1]
    if isinstance(first, tuple) or first.kind != "bound" \
            or isinstance(last, tuple) or last.kind != "bound":
        return None
    return kids[1:-1]


def _word_and(el, rng, budget, implicit_word=False):
    """(\\b letters \\b) & constraints -> 'a word that C1 and C2'.
    implicit_word: the bounds wrap the whole &, so there is no word
    operand to find — every operand is a constraint on the word."""
    word_content = [] if implicit_word else None
    constraints = []
    all_letters = False
    for op in el.kids:
        wc = None if implicit_word else _is_word_atom(op)
        if wc is not None and word_content is None:
            word_content = wc
            continue
        if (not isinstance(op, tuple) and op.kind == "rep" and op.m >= 1
                and op.n is None and op.kids[0].kind == "class"
                and op.kids[0].val == (_LOW | _UP)):
            all_letters = True          # [A-Za-z]+ operand: affix pattern
            continue
        if (not isinstance(op, tuple) and op.kind == "rep" and op.m == 0
                and op.n is None and op.kids[0].kind == "class"):
            missing = frozenset(chr(c) for c in range(32, 127)) - op.kids[0].val
            if 0 < len(missing) <= 3:
                inner = "', '".join(sorted(missing))
                constraints.append(f"contains no '{inner}'")
                budget.spend()
                continue
        neg_c = False
        if not isinstance(op, tuple) and op.kind == "not":
            neg_c, op = True, op.kids[0]
        if not isinstance(op, tuple) and op.kind == "rep":
            wcp = _word_count_shape(op, rng, budget)
            if wcp and wcp[0] == "contains":
                constraints.append(("does not contain " if neg_c
                                    else "contains ") + wcp[1])
                continue
        frame, body = _frame_of(op) if op.kind == "seq" else ("exact", [op])
        if not body:
            return None
        b = _render_seq(body, rng, budget)
        verb = {"contains": "contains {b}", "starts": "starts with {b}",
                "ends": "ends in {b}", "exact": "is exactly {b}"}[frame]
        if neg_c:
            verb = {"contains": "does not contain {b}",
                    "starts": "does not start with {b}",
                    "ends": "does not end in {b}",
                    "exact": "is not exactly {b}"}[frame]
        constraints.append(verb.format(b=b))
    if word_content is None:
        return None
    if implicit_word:
        if not constraints:
            return None
        budget.spend()
        return "a word that " + " and ".join(constraints)
    if all_letters:
        # \b <affix shape with slack> \b  &  [A-Za-z]+ : the slack is
        # constrained to letters, so this is a word with an affix
        lits = [c for c in word_content if isinstance(c, tuple)]
        slacks = [c for c in word_content
                  if not isinstance(c, tuple) and is_slack(c)]
        if len(lits) == 1 and len(lits) + len(slacks) == len(word_content):
            budget.spend()
            lit = lits[0][1]
            if isinstance(word_content[0], tuple) and len(word_content) == 2:
                base = f"a word starting with '{lit}'"
            elif isinstance(word_content[-1], tuple) \
                    and len(word_content) == 2:
                base = f"a word ending in '{lit}'"
            elif len(word_content) == 3:
                base = f"a word containing '{lit}'"
            elif len(word_content) == 1:
                base = f"the whole word '{lit}'"
            else:
                return None
            if constraints:
                return f"{base} that " + " and ".join(constraints)
            return base
        return None
    if not constraints:
        return None
    if not all(_is_wordish(c) for c in word_content):
        return None
    base = _word_content_phrase(word_content, budget) or "a word"
    if not base.startswith("a word"):
        base = "a word"
    return f"{base} that " + " and ".join(constraints)


def _rep(el, rng, budget):
    child = el.kids[0]
    m, n = el.m, el.n
    if is_slack(el):
        return "anything"
    if child.kind == "dot":
        if n is None:
            return f"at least {m} characters" if m != 1 else "at least one character"
        if m == n:
            return f"exactly {m} character{'s' if m != 1 else ''}"
        return f"between {m} and {n} characters"
    def _open_chunk(nd):
        return nd.kind == "seq" and any(is_slack(c) for c in nd.kids)

    if _open_chunk(child) or (child.kind == "alt"
                              and any(_open_chunk(b) for b in child.kids)):
        # counted repetition over an open-ended chunk: the condemned
        # anchored-repeat family — no honest natural phrasing exists
        # (the benign contains-form is handled by _word_count_shape)
        raise TooComplex()
    inner_pl = _atom(child, rng, budget, plural=True)
    if child.kind == "char" or (child.kind == "seq"
                                and all(c.kind == "char" for c in child.kids)):
        lit = serialize(child).replace("(", "").replace(")", "")
        inner_pl = f"'{lit}' repeated"
        if n is None:
            if m == 0:
                return f"'{lit}' repeated any number of times"
            return f"{inner_pl} {m} or more times"
        if m == n:
            return f"{inner_pl} exactly {m} times"
        return f"{inner_pl} between {m} and {n} times"
    if child.kind == "alt":
        # pluralizing an alternation is hopeless English; count parts
        each = _atom(child, rng, budget)
        if n is None:
            lead = ("any number of" if m == 0
                    else "one or more" if m == 1 else f"{m} or more")
            return f"{lead} parts, each {each}"
        if m == n:
            return f"exactly {m} parts, each {each}"
        return f"between {m} and {n} parts, each {each}"
    if n is None:
        if m == 0:
            return f"any number of {inner_pl}"
        if m == 1:
            return f"one or more {inner_pl}"
        return f"{m} or more {inner_pl}"
    if m == n:
        return f"exactly {m} {inner_pl}"
    return f"between {m} and {n} {inner_pl}"


def _render_node(el, rng, budget):
    if not isinstance(el, tuple) and el.kind == "seq":
        return _render_seq(_merge_literals(el.kids), rng, budget)
    return _atom(el, rng, budget)


def _render_seq(elems, rng, budget, allow_word=True):
    if not isinstance(elems, list):
        elems = elems.kids if elems.kind == "seq" else [elems]
    elems = _merge_literals(elems)
    parts, i = [], 0
    pending_gap = False
    while i < len(elems):
        el = elems[i]
        if not isinstance(el, tuple) and is_slack(el):
            pending_gap = True
            i += 1
            continue
        if not isinstance(el, tuple) and el.kind == "bound" and allow_word:
            ws = _word_shape(elems, i, rng, budget)
            if ws:
                phrase, i = ws
                parts.append((pending_gap, phrase))
                pending_gap = False
                continue
            raise TooComplex()
        phrase = _atom(el, rng, budget)
        parts.append((pending_gap, phrase))
        pending_gap = False
        i += 1
    if not parts:
        return "anything"
    out = parts[0][1]
    for gap, phrase in parts[1:]:
        if gap:
            out += rng.choice(GAPS).format(y=phrase)
        else:
            out += rng.choice(ADJ).format(y=phrase)
    return out


def _frame_of(ast):
    """(frame, body-elements) from edge slack of a seq root."""
    if ast.kind != "seq":
        return ("exact", [ast])
    kids = list(ast.kids)
    lead = kids and is_slack(kids[0])
    trail = len(kids) > 1 and is_slack(kids[-1])
    if lead and trail:
        return ("contains", kids[1:-1])
    if lead:
        return ("ends", kids[1:])
    if trail:
        return ("starts", kids[:-1])
    return ("exact", kids)


CLAUSES = {
    "contains": "contain {b}",
    "starts": "start with {b}",
    "ends": "end with {b}",
    "exact": "are exactly {b}",
}
NEG_CLAUSES = {
    "contains": "do not contain {b}",
    "starts": "do not start with {b}",
    "ends": "do not end with {b}",
    "exact": "are not exactly {b}",
}


def _clause(ast, rng, budget, neg=False):
    """Verb-phrase form ('contain a digit') for and/alt/not joins."""
    if ast.kind == "not":
        return _clause(ast.kids[0], rng, budget, neg=not neg)
    wc = _word_count_shape(ast, rng, budget)
    if wc:
        form, phrase = wc
        if form == "contains":
            return ("do not contain " if neg else "contain ") + phrase
        return ("are not " if neg else "are ") + phrase
    if ast.kind == "and":
        if len(ast.kids) > 3:
            raise TooComplex()
        inner = " and ".join(_clause(b, rng, budget) for b in ast.kids)
        return f"do not ({inner})" if neg else inner
    if ast.kind == "alt":
        if len(ast.kids) > 3:
            raise TooComplex()
        inner = " or ".join(_clause(b, rng, budget) for b in ast.kids)
        return f"neither {inner}".replace(" or ", " nor ") if neg \
            else f"either {inner}"
    frame, body = _frame_of(ast)
    if not body:
        raise TooComplex()
    b = _render_seq(body, rng, budget)
    return (NEG_CLAUSES if neg else CLAUSES)[frame].format(b=b)


def _render_root(ast, rng, budget):
    if ast.kind in ("alt", "and", "not"):
        return "lines that " + _clause(ast, rng, budget)
    wc = _word_count_shape(ast, rng, budget)
    if wc:
        form, phrase = wc
        return rng.choice(FRAMES[form if form == "contains"
                                 else "exact"]).format(b=phrase)
    frame, body = _frame_of(ast)
    if not body:
        raise TooComplex()          # pure .* : degenerate, not wordable
    # an edge boundary beside a literal: line start/end already acts as a
    # word edge, so `once\b` at line start is the whole word 'once'
    merged = _merge_literals(body)
    if (frame == "starts" and len(merged) == 2
            and isinstance(merged[0], tuple)
            and not isinstance(merged[1], tuple) and merged[1].kind == "bound"):
        b = f"the whole word '{merged[0][1]}'"
        budget.spend()
    elif (frame == "ends" and len(merged) == 2
            and isinstance(merged[1], tuple)
            and not isinstance(merged[0], tuple) and merged[0].kind == "bound"):
        b = f"the whole word '{merged[1][1]}'"
        budget.spend()
    else:
        b = _render_seq(body, rng, budget)
    pool = FRAMES[frame]
    if frame == "exact" and any(w in b for w in
                                ("or more", "at least", "at most", "between",
                                 "any number")):
        # "are exactly {count-phrase}" contradicts itself; use the
        # composed-of variants for count bodies
        pool = [f for f in pool if "exactly" not in f]
    return rng.choice(pool).format(b=b)


def render_variants(text, rng, k=2, units=MAX_UNITS):
    """k surface variants of the noun phrase, or None (degenerate /
    unverbalizable). Deterministic for a given rng state.

    `units` is the complexity budget; the default keeps corpus-wide
    behaviour (census, pilot) unchanged. Callers rendering denser DSLs
    (StructuredRegex conjunctions, 2026-08-17) may raise it — the garble
    and anchor screens still gate whatever comes out.
    """
    ast = simplify(parse(text))
    nf = serialize(ast)
    if nf in (".*", ".+") or (ast.kind == "rep" and is_slack(ast)):
        return None
    out = []
    for _ in range(k * 2):
        if len(out) >= k:
            break
        try:
            phrase = _render_root(ast, rng, _Budget(units))
        except TooComplex:
            return None
        if phrase not in out:
            out.append(phrase)
    return out or None
