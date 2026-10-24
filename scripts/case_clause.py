#!/usr/bin/env python3
"""Say the case out loud, in more than one voice.

A case-dependent instruction that never mentions case is a defect (see
DECISIONS.md 2026-09-05 for the standard). The repair is to state it — but
stating it the same way 460 times would be its own defect: the corpus exists
to teach a model to read requests as people actually write them, and §7.2 asks
for paraphrase diversity on purpose. A single stock suffix would also hand the
model a spurious cue, since every case-sensitive row would end in the same
five characters.

So the clause varies along three axes:

  PHRASING   "in lowercase" · "lowercase only" · "(lowercase)" ·
             "Case matters: it is lowercase." · "in capitals" ·
             "with a capital G" · "capitalized" · "case-sensitive"
  PLACEMENT  after the quoted literal · attributive, before it
             ("the lowercase letters 'ing'") · a separate closing sentence
  REGISTER   terse and spoken ("Lowercase only.") through to written and
             formal ("The match is case-sensitive.")

Placement is not free: an appended clause strands whatever the sentence says
after the literal — "the letter 'z' in lowercase twice" was one of eight rows
reworded by hand on 2026-09-04. So a trailing clause is only offered when the
literal actually ends the sentence; otherwise the choice is an attributive
form or a closing sentence, both of which leave the original wording intact.

Every variant contains a word the case classifier recognises, which
`test_case_clause.py` asserts for all of them.

The variant is chosen by a stable hash of the row id, so the same row always
gets the same wording and a rerun changes nothing.
"""

import hashlib
import re

_QUOTED = re.compile(r"'([^']+)'")
_DETERMINER = re.compile(r"\b(the|a|an|each|every|any|one|two|both)\s*$", re.I)


def literals(ins):
    return [m for m in _QUOTED.finditer(ins) if re.search(r"[A-Za-z]", m.group(1))]


def shape(ins):
    words = [m.group(1) for m in literals(ins)]
    if not words:
        return "none"
    if all(w == w.lower() for w in words):
        return "lower"
    if all(w == w.upper() and len(w) == 1 for w in words):
        return "capletter"
    if all(w == w.upper() for w in words):
        return "upper"
    if all(w[:1].isupper() and w[1:] == w[1:].lower() for w in words):
        return "initial"
    return "mixed"


def _split_tail(ins):
    s = ins.rstrip()
    return (s[:-1], s[-1]) if s.endswith((".", "!", "?")) else (s, "")


def _ends_with_literal(ins):
    lits = literals(ins)
    if not lits:
        return False
    rest = ins[lits[-1].end():].strip()
    return rest in ("", ".", "!", "?", ",")


def _after_literal(ins, clause):
    """Insert `clause` right after the last quoted literal."""
    m = literals(ins)[-1]
    return ins[:m.end()] + " " + clause + ins[m.end():]


def _attributive(ins, adjective):
    """'the letters ...' -> 'the lowercase letters ...', when a determiner
    governs the literal; otherwise None, because the adjective would land in
    front of a bare noun and read wrong."""
    m = literals(ins)[0]
    head = ins[:m.start()].rstrip()
    noun = re.search(r"\b(letters?|words?|strings?|sequences?|characters?|"
                     r"names?|phrases?|instances?)\s*$", head, re.I)
    if not noun:
        return None
    before = head[:noun.start()]
    if not _DETERMINER.search(before.rstrip()):
        return None
    return (ins[:noun.start()] + adjective + " " + ins[noun.start():])


def _sentence(ins, text):
    body, tail = _split_tail(ins)
    return body + (tail or ".") + " " + text


def variants(ins):
    """Every rewrite that reads well for this sentence, best-first."""
    sh, out = shape(ins), []
    body, tail = _split_tail(ins)
    ends = _ends_with_literal(ins)
    n = len(literals(ins))

    if sh == "lower":
        if ends and n == 1:
            out += [_after_literal(ins, "in lowercase"),
                    _after_literal(ins, "in lower case"),
                    body + ", in lowercase" + tail,
                    body + " (lowercase)" + tail,
                    body + ", lowercase only" + tail,
                    body + ", spelled in lowercase" + tail]
        if ends and n > 1:
            quant = "both" if n == 2 else "all"
            out += [body + ", %s in lowercase" % quant + tail,
                    body + ", %s lowercase" % quant + tail,
                    body + " — %s spelled in lowercase" % quant + tail]
        attr = _attributive(ins, "lowercase")
        if attr:
            out.append(attr)
        out += [_sentence(ins, "Lowercase only."),
                _sentence(ins, "Case matters: the match is lowercase."),
                _sentence(ins, "Match the lowercase spelling only."),
                _sentence(ins, "The letters are lowercase."),
                _sentence(ins, "Only the lowercase form counts.")]

    elif sh == "upper":
        if ends and n == 1:
            out += [_after_literal(ins, "in uppercase"),
                    _after_literal(ins, "in capitals"),
                    body + " (uppercase)" + tail]
        if ends and n > 1:
            quant = "both" if n == 2 else "all"
            out += [body + ", %s in uppercase" % quant + tail,
                    body + ", %s in capitals" % quant + tail]
        attr = _attributive(ins, "uppercase")
        if attr:
            out.append(attr)
        out += [_sentence(ins, "Uppercase only."),
                _sentence(ins, "Case matters: the match is in capitals.")]

    elif sh == "capletter":
        # A lone capital letter needs the adjective in front of a noun -- "a
        # capital 'T'", "the capital letter 'G'". The trailing form has to be
        # PLURAL ("in capitals"), because "in capital" is not English, and it
        # is only offered when the literal ends the clause; otherwise the
        # clause would strand whatever follows.
        m = literals(ins)[0]
        head = ins[:m.start()].rstrip()
        art = re.search(r"\b(a|an)\s*$", head, re.I)
        if art:
            base = head[:art.start()]
            out.append(base + "a capital " + ins[m.start():])
            out.append(base + "an uppercase " + ins[m.start():])
        for adjective in ("capital", "uppercase"):
            attr = _attributive(ins, adjective)
            if attr:
                out.append(attr)
        if ends:
            out += [_after_literal(ins, "in capitals"),
                    _after_literal(ins, "in uppercase")]
        names = " and ".join("'%s'" % x.group(1) for x in literals(ins))
        out += [_sentence(ins, "The %s is a capital." % names
                          if len(literals(ins)) == 1
                          else "The %s are capitals." % names),
                _sentence(ins, "Uppercase only.")]

    elif sh == "initial":
        # Name the capitals, but only the DISTINCT ones: 'Beaker' and
        # 'Bunsen' share a B, and "with a capital B and a capital B" is
        # nonsense. Past three the enumeration stops being readable, so only
        # the "capitalized as shown" forms are offered there.
        seen_letters = list(dict.fromkeys(m.group(1)[0] for m in literals(ins)))
        parts = ["a capital %s" % c for c in seen_letters]
        if len(parts) == 1:
            caps = parts[0]
        elif len(parts) <= 3:
            caps = ", ".join(parts[:-1]) + " and " + parts[-1]
        else:
            caps = None
        if caps:
            out += [body + ", with " + caps + tail,
                    _sentence(ins, "Written with %s." % caps)]
        out.append(body + ", capitalized as shown" + tail)
        attr = _attributive(ins, "capitalized")
        if attr:
            out.append(attr)

    else:                                     # mixed spellings, or no literal
        # every variant must carry a word the classifier knows -- "matching
        # case exactly" says it in plain English but registers as silent, so
        # it appears only alongside the explicit term
        out += [body + " (case-sensitive)" + tail,
                body + ", case-sensitive" + tail,
                _sentence(ins, "The match is case-sensitive."),
                _sentence(ins, "Case-sensitive."),
                _sentence(ins, "Match the case exactly; the search is "
                               "case-sensitive.")]

    seen, unique = set(), []
    for v in out:
        if v and v != ins and v not in seen:
            seen.add(v)
            unique.append(v)
    return unique


def add_case_clause(ins, key=""):
    """One rewrite, chosen stably from `variants` by a hash of `key`."""
    opts = variants(ins)
    if not opts:
        return ins
    h = int(hashlib.sha256(("%s|%s" % (key, ins)).encode()).hexdigest(), 16)
    return opts[h % len(opts)]
