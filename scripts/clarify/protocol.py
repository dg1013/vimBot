"""The clarification protocol: the experiment's prompt, grammar and turns.

An experiment *beside* the §6 contract, not a change to it. CLAUDE.md §6 fixes
the prompt and the one-line output, and src/prompt.py stays the single copy of
both: this module imports the system text, the user-turn layout and the parser
from there and only *adds* to them, so the experiment cannot drift from the
contract it is compared against.

What it adds:

* ``OFFER_SYSTEM`` -- the §6 system text verbatim, plus one clause offering a
  third output form, ``ask: <question>``, when the instruction can be read
  more than one way or leaves out something the edit needs.
* The dialogue: turn 1 is the §6 user turn under ``OFFER_SYSTEM``; if the
  model asks, the user's reply is appended and turn 2 must be a command in
  the §6 grammar. One question at most: a second ``ask:`` is a parse failure.
* The **user simulator**'s prompt. The simulator plays the person who wrote
  the instruction: it sees the buffer before and the buffer they want after,
  never the gold command, and answers in words about the text, the way a
  user who knows the result but not the keystrokes would.
* ``PLACEBO_REPLY`` -- a reply that carries no information. Comparing the
  answer after it with the answer after the real reply separates what the
  *information* in a clarification is worth from what merely *asking* is
  worth (a second turn, a question the model wrote for itself).
* A leak screen for simulator replies: a simulated user that knows the
  answer can hand it over, which would inflate every gain this experiment
  measures. Flagged replies are reported, never silently used or dropped.

Stdlib only; the GPU stages import it.
"""
import difflib
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from prompt import SYSTEM_TEXT, parse_output, render_user   # noqa: E402

ASK_CLAUSE = (
    'If the instruction can be read in more than one way that would leave\n'
    'the buffer different, or leaves out something the edit needs, you may\n'
    'instead output one line "ask: <question>" with a single short\n'
    'clarifying question. The user will answer it, and then you must give\n'
    'the command.'
)
OFFER_SYSTEM = SYSTEM_TEXT + "\n" + ASK_CLAUSE

ASK_PREFIX = "ask:"
ASK_RE = re.compile(r"^ask: ?(.+)$")
_THINK_RE = re.compile(r"</?think>")

PLACEBO_REPLY = "Please just go with the instruction as written."

# How a request reads after the model's forced "ask:" prefix: the question is
# the first line of what follows, so the one-line rule holds for questions too.
QUESTION_MAX_TOKENS = 64


def turn1_messages(case: dict) -> list:
    """System (§6 text plus the ask clause) and the §6 user turn, unchanged."""
    return [{"role": "system", "content": OFFER_SYSTEM},
            {"role": "user", "content": render_user(
                case["instruction"], case["buffer_before"],
                case.get("cursor", (1, 1)))}]


def turn2_messages(case: dict, question: str, reply: str) -> list:
    """Turn 1, the model's question, and the user's reply to it."""
    return turn1_messages(case) + [
        {"role": "assistant", "content": "%s %s" % (ASK_PREFIX, question)},
        {"role": "user", "content": reply}]


def parse_turn1(text):
    """``("ask", question)``, ``(mode, command)`` or None.

    Same rules as §6's ``parse_output`` -- leading whitespace skipped, the
    first line read, any think tag a failure -- with ``ask:`` as a third form.
    """
    if text is None or _THINK_RE.search(text):
        return None
    first = text.lstrip().split("\n", 1)[0].rstrip()
    m = ASK_RE.match(first)
    if m:
        return "ask", m.group(1).strip()
    return parse_output(text)


def parse_turn2(text):
    """Turn 2 must be a command: the §6 grammar, so a second question fails."""
    return parse_output(text)


def cut_question(generated: str) -> str:
    """The question the model wrote after a forced ``ask:``: its first line."""
    return generated.lstrip(" ").split("\n", 1)[0].strip()


# ---------------------------------------------------------------------------
# The user simulator

SIM_SYSTEM = (
    "You are playing the user of an editing assistant, to help evaluate it.\n"
    "You asked for an edit to a text buffer, and you know exactly the result\n"
    "you want: you are shown the buffer before and after the edit you have\n"
    "in mind. The assistant has asked you one clarifying question. Reply as\n"
    "that user would, in one or two short sentences.\n"
    "\n"
    "Rules:\n"
    "- Answer only the question that was asked, so that your answer matches\n"
    "  the result you want. If it offers options, choose the matching one.\n"
    "  Answer it; never repeat the question back.\n"
    "  For where and how much, go by the exact changes listed, not by the\n"
    "  wording of your request.\n"
    "- Do not agree just because the assistant suggests something. If its\n"
    "  proposal is not what the result shows, say no and say what you want.\n"
    "- If the question is about which lines a rule should catch, check your\n"
    "  answer against every line that is removed or changed and every line\n"
    "  that is left alone.\n"
    "- Talk about the text, never about how to edit it: do not mention vim,\n"
    "  commands, keys, keystrokes, modes, registers or regular expressions.\n"
    "- Quote the buffer only when the question asks for text, and only the\n"
    "  part it asks about. Never paste the whole result; if you are asked for\n"
    "  it, describe the change in words.\n"
    "- If the question is not about your edit, or the result does not settle\n"
    "  it, say that it doesn't matter.\n"
    "\n"
    "Write exactly two lines. First \"Check: \" and one sentence saying what\n"
    "the result does on the point the assistant asks about, read from the\n"
    "exact changes and the lines left alone. Then \"Reply: \" and your answer\n"
    "to the assistant. Only the reply is shown to the assistant."
)


def _q(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def _lines(buf: str) -> list:
    lines = buf.split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _span(a: int, b: int) -> str:
    return "column %d" % (a + 1) if b - a == 1 else "columns %d-%d" % (a + 1, b)


def char_changes(old: str, new: str) -> list:
    """What happens inside one line, in columns (1-based characters)."""
    out = []
    sm = difflib.SequenceMatcher(None, old, new, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "insert":
            out.append("inserted %s before column %d" % (_q(new[j1:j2]), i1 + 1))
        elif op == "delete":
            out.append("deleted %s (%s)" % (_q(old[i1:i2]), _span(i1, i2)))
        elif op == "replace":
            out.append("replaced %s (%s) with %s" % (_q(old[i1:i2]), _span(i1, i2),
                                                     _q(new[j1:j2])))
    return out


def visible_diff(before: str, after: str) -> str:
    """The edit, line by line and then character by character.

    Changed lines are shown in JSON quotes so spaces at the ends show. Where a
    line is changed in place, the change is also spelled out in columns
    ("inserted "aa" before column 1"): questions about *where* and *how
    much* are answered from that, which whole lines leave to the reader. The
    2026-10-06 smoke showed the simulator misjudging both from lines alone.
    """
    a, b = _lines(before), _lines(after)
    out = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        if op == "replace" and i2 - i1 == j2 - j1:
            for k in range(i2 - i1):
                out.append("line %d:" % (i1 + k + 1))
                out.append("  before %s" % _q(a[i1 + k]))
                out.append("  after  %s" % _q(b[j1 + k]))
                for c in char_changes(a[i1 + k], b[j1 + k]):
                    out.append("  change: %s" % c)
            continue
        where = "line%s %s" % ("s" if i2 - i1 > 1 else "",
                               "%d-%d" % (i1 + 1, i2) if i2 - i1 > 1 else i1 + 1)
        if op == "replace":
            # "line 4 replaced", not "removed" plus "new lines after line 3":
            # the 2026-10-06 audit had the simulator place a comma on line 3
            # from the second wording.
            out.append("%s replaced:" % where)
        elif op == "delete":
            out.append("%s removed:" % where)
        else:
            # Name the neighbours: "after line 1" alone had the simulator put
            # blank lines before the wrong sections (2026-10-06 audit).
            spots = []
            if i1:
                spots.append("after line %d %s" % (i1, _q(a[i1 - 1])))
            if i1 < len(a):
                spots.append("before line %d %s" % (i1 + 1, _q(a[i1])))
            out.append("new line%s %s:" % ("s" if j2 - j1 > 1 else "",
                                          " and ".join(spots) or "in the empty buffer"))
        out.extend("  before %s" % _q(x) for x in a[i1:i2])
        out.extend("  after  %s" % _q(x) for x in b[j1:j2])
    return "\n".join(out) if out else "(no line changes)"


def cursor_char(case: dict) -> str:
    """The character under the cursor, as the simulator is told it."""
    line, col = case.get("cursor", (1, 1))
    lines = case["buffer_before"].split("\n")
    if 1 <= line <= len(lines) and 1 <= col <= len(lines[line - 1]):
        return "on %s" % _q(lines[line - 1][col - 1])
    return "past the end of the line" if 1 <= line <= len(lines) else "outside the buffer"


def sim_user_turn(case: dict, question: str) -> str:
    line, col = case.get("cursor", (1, 1))
    return (
        "Your request: %s\n"
        "The cursor is on line %d, column %d (%s).\n\n"
        "Buffer before:\n<<<\n%s\n>>>\n\n"
        "Buffer after (the result you want):\n<<<\n%s\n>>>\n\n"
        "Exactly what changes (lines in quotes, so spaces at the ends are visible; "
        "columns count characters from 1):\n%s\n\n"
        "The assistant asks: %s\n\n"
        "Your reply:" % (case["instruction"], line, col, cursor_char(case),
                         case["buffer_before"], case["buffer_after"],
                         visible_diff(case["buffer_before"], case["buffer_after"]),
                         question))


def sim_messages(case: dict, question: str) -> list:
    return [{"role": "system", "content": SIM_SYSTEM},
            {"role": "user", "content": sim_user_turn(case, question)}]


def clean_reply(text: str, thinking: bool = False) -> str:
    """The simulator's reply as the user turn: reasoning out, first paragraph.

    With thinking on, the chat template opens the think block in the prompt,
    so the completion holds only its end: everything up to the last
    ``</think>`` is reasoning. A completion that never closes the block (it
    ran out of tokens mid-thought) has no reply and comes back empty -- never
    the reasoning itself, which would hand the student the simulator's
    working-out.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif thinking or "<think>" in text:
        return ""
    text = _THINK_RE.sub("", text).strip()
    return split_check(text)[1]


_CHECK_RE = re.compile(r"^\s*check:\s*(.*)$", re.I | re.M)
_REPLY_RE = re.compile(r"^\s*reply:\s*", re.I | re.M)


def split_check(text: str):
    """(check, reply) from the simulator's two-line answer.

    The check is the simulator's own note on what the result does -- it saw
    the result, so the check never reaches the assistant. Without a
    "Reply:" line, the reply is what is left after any "Check:" line.
    """
    m = _CHECK_RE.search(text)
    check = m.group(1).strip() if m else ""
    r = _REPLY_RE.search(text)
    if r:
        reply = text[r.end():]
    else:
        reply = _CHECK_RE.sub("", text) if m else text
    return check, reply.strip().split("\n\n", 1)[0].strip()


# Vim talk a user who does not know vim would not produce. A hit is a flag for
# the audit, not proof of a leak: "the second word" is fine, `dw` is not.
_VIM_TALK = [
    (re.compile(r"<(?:Esc|CR|BS|Tab|Space|C-[A-Za-z\[\]]|S-\w+|leader)>", re.I), "key notation"),
    (re.compile(r"(?:^|[\s(`]):[%\d.,$'<>]*(?:[sgv]/|norm(?:al)?\b|sort\b|[mt]\s*\d|co\s*\d)"),
     "ex command"),
    # Any ":word" after a space, bracket or backtick: the 2026-10-06 audit
    # found "Use `:retab`" (the gold was %retab 4) passing the pattern above.
    (re.compile(r"(?:^|[\s(`])[:%][a-z]{2,}\b"), "ex command"),
    (re.compile(r"\b(?:tabstop|shiftwidth|softtabstop|expandtab|textwidth|retab)\b", re.I),
     "vim option"),
    (re.compile(r"(?:^|[\s(`])s/[^/\s]*/"), "substitute pattern"),
    (re.compile(r"\\[(){}<>|vVmMz]"), "regex escape"),
    (re.compile(r"\b(?:vim|keystrokes?|regex(?:es)?|regular expressions?|register|"
                r"normal mode|visual mode|insert mode|ex mode)\b", re.I), "editor vocabulary"),
]


def leak_flags(reply: str, case: dict) -> list:
    """Why a reply might be handing over the command (empty = clean).

    The gold command verbatim counts only from four characters up: shorter
    ones (``dd``, ``J``, ``x``) occur inside ordinary words.
    """
    flags = [why for rx, why in _VIM_TALK if rx.search(reply)]
    gold = case.get("gold_command", "")
    if len(gold) >= 4 and gold in reply:
        flags.append("gold command")
    return flags


def quotes_result(reply: str, case: dict, min_len: int = 20) -> bool:
    """True if the reply pastes a whole changed line of the wanted result.

    A user can name the text an edit needs ("the seed is 42"); pasting
    whole result lines turns the clarification into the answer. Long lines
    only, so a short quoted word is not counted.
    """
    before = set(case["buffer_before"].split("\n"))
    for line in case["buffer_after"].split("\n"):
        s = line.strip()
        if len(s) >= min_len and line not in before and s in reply:
            return True
    return False
