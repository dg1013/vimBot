"""Stage 1's data (README.md): the held-out split and the three kinds of row.

Every SFT row is a TRL prompt-completion pair in message form -- exactly what
scripts/train_sft.py's `build_dataset` makes -- whose `prompt` is the message
list the student is given at inference by scripts/clarify/dialogue.py, and
whose `completion` is the one assistant turn that carries the loss:

    ask-turn1     offer prompt                         -> "ask: <question>"
    ask-turn2     offer prompt, the question, a reply  -> "<mode>: <gold>"
    placebo-turn2 offer prompt, the question, PLACEBO  -> "<mode>: <gold>"
    answer        offer prompt                         -> "<mode>: <gold>"

So a dialogue is trained one turn at a time, each turn on the prompt it is
generated from (the template renders an earlier assistant turn without the
empty think block; the turn being generated gets it -- a single rendering of
the whole dialogue would train turn 1 on a prompt never served). A placebo
row teaches only the second turn: its question sits in the prompt, so it
carries no loss and does not teach asking on the row.

Stdlib only; the GPU stages import it.
"""
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "clarify"))
from prompt import render_target                       # noqa: E402
from protocol import (ASK_PREFIX, PLACEBO_REPLY,      # noqa: E402
                      turn1_messages, turn2_messages)

ASK_CLASSES = ("unstated", "ambiguous")     # the noted classes taught to ask


def split_noted(cases, frac, seed):
    """(train, held_out): `frac` of each note class held out, seeded.

    Stratified by class so the held-out rows keep the mix the gate reads
    (unstated / ambiguous / model), and order-independent: rows are sorted
    by id before the shuffle.
    """
    by_class = {}
    for c in cases:
        by_class.setdefault(c["note_class"], []).append(c)
    rng = random.Random(seed)
    train, held = [], []
    for cls in sorted(by_class):
        rows = sorted(by_class[cls], key=lambda c: c["id"])
        rng.shuffle(rows)
        k = round(len(rows) * frac)
        held += rows[:k]
        train += rows[k:]
    key = lambda c: c["id"]                                            # noqa: E731
    return sorted(train, key=key), sorted(held, key=key)


def norm_question(q):
    return " ".join(q.split()).lower()


def unique_questions(questions, seen=()):
    """The questions in order, dropping empties and repeats (also of `seen`)."""
    out, have = [], {norm_question(q) for q in seen}
    for q in questions:
        n = norm_question(q)
        if n and n not in have:
            have.add(n)
            out.append(q.strip())
    return out


def _assistant(text):
    return [{"role": "assistant", "content": text}]


def _gold(case):
    return _assistant(render_target(case["mode"], case["gold_command"]))


def ask_rows(case, question, reply):
    """A verified dialogue as two rows: the question, then the answer."""
    return [{"id": case["id"], "kind": "ask-turn1", "prompt": turn1_messages(case),
             "completion": _assistant("%s %s" % (ASK_PREFIX, question))},
            {"id": case["id"], "kind": "ask-turn2",
             "prompt": turn2_messages(case, question, reply), "completion": _gold(case)}]


def placebo_row(case, question):
    """After a reply that says nothing, the answer the instruction asks for."""
    return {"id": case["id"], "kind": "placebo-turn2",
            "prompt": turn2_messages(case, question, PLACEBO_REPLY), "completion": _gold(case)}


def answer_row(case):
    """A fair instruction under the offer prompt: answer, do not ask."""
    return {"id": case["id"], "kind": "answer", "prompt": turn1_messages(case),
            "completion": _gold(case)}


def pick_dialogues(dialogues, max_per_row):
    """{id: [verified dialogues]}, distinct questions, at most max_per_row each.

    A dialogue is usable only if its final command exec-matched and its reply
    passed the leak screen: the rejection-sampling filter.
    """
    out = {}
    for d in dialogues:
        if not d["match"] or d["leak_flags"] or d.get("quotes_result"):
            continue
        rows = out.setdefault(d["id"], [])
        if len(rows) < max_per_row and norm_question(d["question"]) not in {
                norm_question(r["question"]) for r in rows}:
            rows.append(d)
    return out
