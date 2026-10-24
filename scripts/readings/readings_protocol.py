"""The readings-first protocol: list the readings, check them, then ask.

An experiment beside the §6 contract (owner, 2026-10-09), built on
scripts/clarify/. Before deciding whether to ask, the model gives the
distinct ways the instruction could be read that would leave the buffer
different, one command per reading: one at a time by default (its §6
answer, then `ALT_SYSTEM` for each further reading given those found so
far), or as one list (`READINGS_SYSTEM`, which the 2026-10-09 smoke showed
this model does not follow). The sandbox runs every reading:
readings that fail or change nothing drop, and readings that leave the same
buffer merge. When two or more distinct results remain, the instruction is
ambiguous *by construction*. The model is then shown those results and
writes one question that decides between them. From there the dialogue is
the clarify protocol's own (`protocol.turn2_messages`), so against the
2026-10-06 experiment only two things change: when it asks, and what.

Research only: real use never executes a command (CLAUDE.md §1, the
execution boundary), so a shipped version would have to decide without the
sandbox.

Stdlib only.
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts" / "clarify"):
    sys.path.insert(0, str(p))
from evaluate import _norm                         # noqa: E402  the scorer's comparison
from prompt import parse_output, render_user      # noqa: E402
from protocol import visible_diff                  # noqa: E402

MAX_READINGS = 4
READINGS_MAX_TOKENS = 256

READINGS_SYSTEM = (
    "You translate natural-language editing instructions into vim commands.\n"
    "First list the different ways the instruction could be read that would\n"
    "leave the buffer different. If it leaves out something the edit needs,\n"
    "such as the exact text, give one reading per plausible choice. Write one\n"
    "command per reading, each on its own line, in the form \"ex: <command>\"\n"
    "or \"keys: <keystrokes>\", at most %d lines. If the instruction can only\n"
    "be read one way, write one line. No explanation." % MAX_READINGS
)

# The readings, one at a time (the default since the 2026-10-09 smoke).
# READINGS_SYSTEM asks for a list in one reply, and the served adapter --
# trained to answer with exactly one command -- wrote one line on 170 of
# 176 diag rows, and where it wrote more they were the *steps* of a compound
# edit, not other readings. So reading 1 is its own §6 answer, and each
# further reading is asked for alone, in the one-line form it was trained
# on, given the readings found so far.
ALT_SYSTEM = (
    "You translate natural-language editing instructions into vim commands.\n"
    "An instruction can often be read more than one way. Below the request\n"
    "are commands for the readings of it found so far. If the instruction can\n"
    "be read another way that would leave the buffer different from every one\n"
    "of them, including another choice of anything it leaves out, such as the\n"
    "exact text, output one line with the command for that reading, in the\n"
    "form \"ex: <command>\" or \"keys: <keystrokes>\". The command must make the\n"
    "whole edit by itself. If there is no other reading, output \"none\"."
)

QUESTION_SYSTEM = (
    "You translate natural-language editing instructions into vim commands.\n"
    "The instruction below can be read in more than one way: the readings\n"
    "listed after it leave the buffer different. Write one short question for\n"
    "the user that decides between them, as one line \"ask: <question>\". Ask\n"
    "about the result they want, in plain words, not about vim."
)

# A list marker the model may put before a reading: "1.", "2)", "-", "*".
_MARKER_RE = re.compile(r"^\s*(?:\d+[.):]|[-*•])\s*")
_THINK_RE = re.compile(r"</?think>")

STATUSES = ("ok", "no_change", "invalid", "unrun")


def readings_messages(case: dict) -> list:
    """READINGS_SYSTEM and the §6 user turn, unchanged."""
    return [{"role": "system", "content": READINGS_SYSTEM},
            {"role": "user", "content": render_user(
                case["instruction"], case["buffer_before"], case.get("cursor", (1, 1)))}]


def parse_readings(text, max_readings=MAX_READINGS):
    """([(mode, command)], tally) read off the model's lines.

    Each non-empty line is one reading in the §6 grammar. A leading list
    marker ("1.", "-") is tolerated and counted. Lines past `max_readings`
    are counted, not used. A think tag fails the whole output, as in §6.
    """
    tally = {"lines": 0, "parsed": 0, "unparsed": 0, "marked": 0, "extra": 0,
             "think": False}
    if text is None:
        return [], tally
    if _THINK_RE.search(text):
        tally["think"] = True
        return [], tally
    out = []
    for line in text.split("\n"):
        if not line.strip():
            continue
        tally["lines"] += 1
        marked = _MARKER_RE.match(line)
        parsed = parse_output(line[marked.end():] if marked else line)
        if parsed is None:
            tally["unparsed"] += 1
            continue
        if len(out) >= max_readings:
            tally["extra"] += 1
            continue
        tally["marked"] += bool(marked)
        tally["parsed"] += 1
        out.append(parsed)
    return out, tally


_NONE_RE = re.compile(r"^\s*\"?none\b", re.I)


def alt_user_turn(case: dict, readings: list) -> str:
    """The §6 user turn, then the readings found so far, numbered."""
    parts = [render_user(case["instruction"], case["buffer_before"],
                         case.get("cursor", (1, 1))),
             "", "Readings so far:"]
    parts += ["%d. %s: %s" % (k, mode, command) for k, (mode, command) in enumerate(readings, 1)]
    return "\n".join(parts)


def alt_messages(case: dict, readings: list) -> list:
    return [{"role": "system", "content": ALT_SYSTEM},
            {"role": "user", "content": alt_user_turn(case, readings)}]


def parse_alt(text):
    """("none", None), ("reading", (mode, command)) or ("fail", None), read off
    the first line; a list marker is tolerated, a think tag fails."""
    if text is None or _THINK_RE.search(text):
        return "fail", None
    first = text.lstrip().split("\n", 1)[0]
    if _NONE_RE.match(first):
        return "none", None
    marked = _MARKER_RE.match(first)
    parsed = parse_output(first[marked.end():] if marked else first)
    return ("reading", parsed) if parsed else ("fail", None)


def norm_key(buffer: str):
    """The scorer's own comparison (evaluate._norm) as a hashable key."""
    return tuple(_norm(buffer))


def reading_status(case: dict, entry) -> str:
    """ok (ran and changed the buffer), no_change, invalid (vim error, timeout,
    blocked by the scope screen) or unrun (no executor entry)."""
    if entry is None:
        return "unrun"
    if entry.get("blocked") or entry.get("timeout") or not entry.get("valid"):
        return "invalid"
    if norm_key(entry["buffer_after"]) == norm_key(case["buffer_before"]):
        return "no_change"
    return "ok"


def group_readings(case: dict, entries: list):
    """(groups, statuses): the readings that ran and changed the buffer,
    merged by the buffer they leave, in order of first appearance.

    `entries[i]` is the executor's per-case entry for reading i. A group is
    {"members": [i, ...], "buffer_after": str, "match": bool}; `match` says
    whether its buffer is the case's expected result."""
    statuses = [reading_status(case, e) for e in entries]
    groups, at = [], {}
    for i, (e, s) in enumerate(zip(entries, statuses)):
        if s != "ok":
            continue
        k = norm_key(e["buffer_after"])
        if k not in at:
            at[k] = len(groups)
            groups.append({"members": [], "buffer_after": e["buffer_after"],
                           "match": bool(e.get("match"))})
        groups[at[k]]["members"].append(i)
    return groups, statuses


def question_user_turn(case: dict, readings: list, groups: list) -> str:
    """The §6 user turn, then each distinct reading and what it changes.

    The change is `protocol.visible_diff`: the same view of an edit the
    simulated user gets for the result it wants."""
    parts = [render_user(case["instruction"], case["buffer_before"],
                         case.get("cursor", (1, 1))),
             "", "Readings that leave the buffer different:"]
    for k, g in enumerate(groups, 1):
        mode, command = readings[g["members"][0]]
        parts.append("Reading %d: %s: %s" % (k, mode, command))
        parts.append(visible_diff(case["buffer_before"], g["buffer_after"]))
    return "\n".join(parts)


def question_messages(case: dict, readings: list, groups: list) -> list:
    return [{"role": "system", "content": QUESTION_SYSTEM},
            {"role": "user", "content": question_user_turn(case, readings, groups)}]
