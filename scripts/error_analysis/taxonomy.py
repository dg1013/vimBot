"""Pure helpers for the error analysis (stdlib only, no vim).

- `classify(before, expected, got)`: what kind of wrong a produced buffer is,
  judged at line granularity against the gold buffer (CLAUDE.md §5 compares
  line lists ignoring one trailing newline; this module keeps that rule).
- `features(case)`: the case-side properties the report stratifies by.
- `mutations(command, mode)`: one-edit variants of a wrong command, for the
  "one mechanical edit away" probe (the caller runs them in the sandbox).
- `buffer_hash(buf)`: the ceiling files' 16-hex result key, so gold and
  no-op results can be recognised without re-executing anything.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re

# ----------------------------------------------------------------- buffers

def norm(text):
    """Buffer as a line list, ignoring one trailing newline (src/evaluate._norm)."""
    if text is None:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def buffer_hash(buf):
    """scripts/ceiling/sample.py's key: equal exactly when exec-match would call two buffers equal."""
    lines = norm(buf)
    if lines is None:
        return None
    return hashlib.sha1(json.dumps(lines).encode("utf-8")).hexdigest()[:16]


def touched(before, after):
    """Where an edit acted on `before`: ('line', i) for a before-line changed or
    deleted, ('ins', i) for lines inserted in front of before-line i."""
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    out = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            out.update(("line", i) for i in range(i1, i2))
        elif tag == "insert":
            out.add(("ins", i1))
    return out


def line_diff_size(expected, got):
    """Lines that would have to change to turn `got` into `expected`."""
    sm = difflib.SequenceMatcher(None, expected, got, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")


_WS = re.compile(r"\s+")

KINDS = ("match", "noop", "ws_only", "blank_lines_only", "case_only", "same_lines_wrong_text",
         "scope_under", "scope_over", "scope_partial", "scope_disjoint", "no_buffer")

KIND_TEXT = {
    "match": "exec-match",
    "noop": "no-op: the buffer came back unchanged",
    "ws_only": "right text, whitespace differs (indent, tabs, trailing spaces)",
    "blank_lines_only": "right text, an empty line more or fewer",
    "case_only": "right text, letter case differs",
    "same_lines_wrong_text": "right lines touched, wrong text on them",
    "scope_under": "touched a strict subset of the gold's lines (range or pattern too narrow)",
    "scope_over": "touched the gold's lines and more (range or pattern too wide)",
    "scope_partial": "touched some of the gold's lines and some others",
    "scope_disjoint": "touched none of the gold's lines",
    "no_buffer": "no buffer (timeout, blocked, or no prediction)",
}


def classify(before, expected, got):
    """Kind of wrong, with the line-diff size and a character similarity.

    Precedence: match, noop, ws_only, blank_lines_only, case_only, then the
    scope classes (by which before-lines the two edits touched). The scope
    classes are at line granularity: a substitute missing its `g` flag changes
    the right lines with the wrong text, so it is `same_lines_wrong_text`; the
    mutation probe is what detects that case.
    """
    b, e, g = norm(before), norm(expected), norm(got)
    if g is None:
        return {"kind": "no_buffer", "diff_lines": None, "d_lines": None, "similarity": None}
    out = {"d_lines": len(g) - len(e),
           "diff_lines": line_diff_size(e, g),
           "similarity": round(difflib.SequenceMatcher(None, "\n".join(e), "\n".join(g), autojunk=False).ratio(), 3)}
    if g == e:
        kind = "match"
    elif g == b:
        kind = "noop"
    elif [_WS.sub("", l) for l in g] == [_WS.sub("", l) for l in e]:
        kind = "ws_only"
    elif [l for l in g if l.strip()] == [l for l in e if l.strip()]:
        kind = "blank_lines_only"
    elif [l.lower() for l in g] == [l.lower() for l in e]:
        kind = "case_only"
    else:
        tg, te = touched(b, g), touched(b, e)
        if tg == te:
            kind = "same_lines_wrong_text"
        elif tg < te:
            kind = "scope_under"
        elif tg > te:
            kind = "scope_over"
        elif tg & te:
            kind = "scope_partial"
        else:
            kind = "scope_disjoint"
    out["kind"] = kind
    out["within"] = within_line(b, e, g) if kind == "same_lines_wrong_text" else None
    return out


WITHIN = ("under-applied", "over-applied", "wrong replacement", "elsewhere in the line", "mixed", "unalignable")
WITHIN_TEXT = {
    "under-applied": "changed a strict subset of the characters the gold changes (first occurrence only, a shorter match)",
    "over-applied": "changed the gold's characters and more (a greedier match)",
    "wrong replacement": "changed exactly the gold's characters, to other text",
    "elsewhere in the line": "changed other characters of the same line",
    "mixed": "different verdicts on different lines",
    "unalignable": "lines could not be paired one to one",
}


def _char_touched(before, after):
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    out = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            out.update(range(i1, i2))
        elif tag == "insert":
            out.add(("ins", i1))
    return out


def _paired(before, after):
    """{before-line index: after line} for replace blocks that pair lines one to one."""
    out = {}
    sm = difflib.SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                out[i1 + k] = after[j1 + k]
    return out


def within_line(b, e, g):
    """What an answer that touched the gold's lines did inside them (one verdict per
    changed line, from the before-line characters each edit touched; summarised)."""
    pe, pg = _paired(b, e), _paired(b, g)
    verdicts = set()
    for i in pe:
        if i not in pg:
            continue
        te, tg = _char_touched(b[i], pe[i]), _char_touched(b[i], pg[i])
        if tg == te:
            verdicts.add("wrong replacement")
        elif tg < te:
            verdicts.add("under-applied")
        elif tg > te:
            verdicts.add("over-applied")
        else:
            verdicts.add("elsewhere in the line")
    if not verdicts:
        return "unalignable"
    return verdicts.pop() if len(verdicts) == 1 else "mixed"


# ---------------------------------------------------------------- features

_SCOPE_WORDS = re.compile(r"\b(all|every|each|entire|whole|everywhere|throughout|globally)\b", re.I)
_NEGATION = re.compile(r"\b(except|but not|only|without|keep|leave|don't|do not|not the|other than|unless)\b", re.I)
_QUOTE = re.compile(r"[\"'`“”‘’]")
_LINE_REF = re.compile(r"\b(line|lines)\s+\d|\b\d+(st|nd|rd|th)\b|\bfirst\b|\blast\b|\bsecond\b|\bthird\b", re.I)


def bucket(x, edges, labels):
    """x into the first bucket whose upper edge it does not exceed; edges ascending, len(labels) == len(edges) + 1."""
    for e, lab in zip(edges, labels):
        if x <= e:
            return lab
    return labels[-1]


def unescaped_bar(cmd):
    """An ex command chains several commands when it holds a `|` that is not escaped."""
    return re.search(r"(?<!\\)\|", cmd) is not None


def provenance_group(p):
    p = p or ""
    if p.startswith("claude"):
        return "claude-authored"
    if p in ("owner-edited", "owner-authored", "human", "human-rewritten"):
        return "owner/human-written"
    if "review" in p or "rephrased" in p or "retargeted" in p:
        return "edited in a review"
    return p or "unknown"


def features(case):
    """Case-side properties (gold, instruction, buffer); every value is a short label."""
    gold, mode = case["gold_command"], case["mode"]
    before, after = norm(case["buffer_before"]), norm(case["buffer_after"])
    instr = case["instruction"]
    cursor = tuple(case.get("cursor") or (1, 1))
    t = touched(before, after)
    f = {
        "tier": case.get("tier") or "T?",
        "gold mode": mode,
        "source": case.get("source") or "?",
        "instruction provenance": provenance_group(case.get("instruction_provenance")),
        "cursor": "(1,1)" if cursor == (1, 1) else "elsewhere",
        "buffer lines": bucket(len(before), (3, 6, 9), ("1-3", "4-6", "7-9", "10+")),
        "instruction words": bucket(len(instr.split()), (8, 12, 18), ("<=8", "9-12", "13-18", "19+")),
        "gold length (chars)": bucket(len(gold), (10, 20, 40), ("<=10", "11-20", "21-40", "41+")),
        "lines the gold touches": bucket(len(t), (1, 3), ("1", "2-3", "4+")),
        "gold changes the line count": "yes" if len(before) != len(after) else "no",
        "instruction quotes text": "yes" if _QUOTE.search(instr) else "no",
        "instruction has a number": "yes" if re.search(r"\d", instr) else "no",
        "instruction names a line": "yes" if _LINE_REF.search(instr) else "no",
        "instruction has a scope word": "yes" if _SCOPE_WORDS.search(instr) else "no",
        "instruction has a restriction word": "yes" if _NEGATION.search(instr) else "no",
    }
    if mode == "ex":
        f["ex: chained with |"] = "yes" if unescaped_bar(gold) else "no"
        f["ex: capture group \\("] = "yes" if "\\(" in gold else "no"
        f["ex: expression \\="] = "yes" if "\\=" in gold else "no"
        f["ex: lookaround or \\%("] = "yes" if ("\\@" in gold or "\\%(" in gold) else "no"
        f["ex: edits line breaks (\\r or \\n)"] = "yes" if ("\\r" in gold or "\\n" in gold) else "no"
        f["ex: range given"] = "yes" if re.match(r"^\s*(%|\d|\.|\$|'|/|\?|,)", gold) else "no"
    else:
        f["keys: inserts text (<Esc>)"] = "yes" if "<Esc>" in gold else "no"
        f["keys: has a count"] = "yes" if re.search(r"(^|[^<\w])\d", gold) else "no"
        f["keys: text object"] = "yes" if re.search(r"[dcyvg<>=][ia][wWspb(){}\[\]<>t\"'`]", gold) else "no"
    return f


# --------------------------------------------------------------- mutations

_DELIM_OK = re.compile(r"[^\w\s\\|\"]")
_RANGE = r"(?:%|\d+|\.|\$|'[\w<>\[\]]|/(?:\\.|[^/])*/|\?(?:\\.|[^?])*\?|[,;+\-]|\s)*"
_SUB_HEAD = re.compile(r"^(" + _RANGE + r")s(?:ubstitute)?(?=[^\w\s|\"])")
_GV_HEAD = re.compile(r"^(" + _RANGE + r")(g(?:lobal)?|v(?:global)?)(?=[^\w])")


def _split_bar(cmd):
    """Segments of an ex command at unescaped bars (positions kept so the command can be rebuilt)."""
    parts, cur, i = [], "", 0
    while i < len(cmd):
        ch = cmd[i]
        if ch == "\\" and i + 1 < len(cmd):
            cur += cmd[i:i + 2]
            i += 2
            continue
        if ch == "|":
            parts.append(cur)
            cur = ""
        else:
            cur += ch
        i += 1
    parts.append(cur)
    return parts


def add_g(cmd):
    """The command with a `g` flag on every substitute that lacks one; None if nothing to add."""
    changed = False
    out = []
    for seg in _split_bar(cmd):
        m = _SUB_HEAD.match(seg)
        if not m:
            out.append(seg)
            continue
        i = m.end()
        delim = seg[i]
        if not _DELIM_OK.match(delim):
            out.append(seg)
            continue
        j, fields = i + 1, 0
        while j < len(seg) and fields < 2:
            if seg[j] == "\\":
                j += 2
                continue
            if seg[j] == delim:
                fields += 1
            j += 1
        if fields < 2:                       # closing delimiter omitted: s/a/b
            out.append(seg + delim + "g")
            changed = True
            continue
        flags = re.match(r"[&cegiInp#lr]*", seg[j:]).group(0)
        if "g" in flags:
            out.append(seg)
            continue
        out.append(seg[:j] + "g" + seg[j:])
        changed = True
    return "|".join(out) if changed else None


def toggle_gv(cmd):
    """`g` <-> `v` on a leading global command; None if it is not one."""
    m = _GV_HEAD.match(cmd)
    if not m:
        return None
    word = m.group(2)
    new = "v" if word.startswith("g") else "g"
    return cmd[:m.start(2)] + new + cmd[m.end(2):]


def mutations(command, mode):
    """One-edit variants worth trying on a wrong answer: [(rule, command)]."""
    if not command:
        return []
    out = []
    if mode == "ex":
        g = add_g(command)
        if g:
            out.append(("add g flag", g))
        if not re.match(r"^\s*(%|\d|\.|\$|'|/|\?|,|\+|-)", command):
            out.append(("prefix %", "%" + command))
        if re.match(r"^\s*%", command):
            out.append(("drop %", re.sub(r"^\s*%", "", command, count=1)))
        t = toggle_gv(command)
        if t:
            out.append(("g <-> v", t))
    else:
        out.append(("prefix gg", "gg" + command))
        out.append(("prefix 0", "0" + command))
    return out
