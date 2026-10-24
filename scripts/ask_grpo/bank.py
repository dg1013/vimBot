"""Stage 2's simulated user, kept out of the training loop (README.md).

The audited 9B cannot sit next to GRPO on the card: the 4B's GRPO peaks at
13.8 GB, the 9B in NF4 adds ~7 GB next to the desktop's 2.1, and its
reasoning (~5 s a reply) would multiply a 14 s step several times. So the
user's replies come from a bank of what it has already said:

* every (question, reply) pair the 9B produced for a row -- the greedy
  question of the 2026-10-06 diag run, for every noted row, and stage 1's
  sampled questions for the rows taught to ask -- leak-screened, whatever
  the dialogue's outcome (a reply is the user's answer to a question; the
  student's turn 2 after it is the student's business);
* a question the policy asks is answered with the reply to the most similar
  banked question for that row, by word-set Jaccard similarity;
* a row with nothing banked -- a fair replay row -- gets the placebo reply.
  A complete instruction leaves the user nothing to add, so asking there can
  only cost, which is exactly what the policy has to learn.

Stdlib only.
"""
import re

_WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return frozenset(_WORD.findall((text or "").lower()))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class ReplyBank:
    """{row id: [(question, reply, question words)]}, and the lookup."""

    def __init__(self, pairs, placebo):
        self.placebo = placebo
        self.by_id = {}
        for p in pairs:
            if p.get("leak_flags") or p.get("quotes_result") or not p.get("reply"):
                continue
            entries = self.by_id.setdefault(p["id"], [])
            if all(e[0] != p["question"] for e in entries):
                entries.append((p["question"], p["reply"], words(p["question"])))

    def reply(self, cid, question):
        """(reply, similarity or None, source): the nearest banked question's
        reply ("bank"; ties go to the first banked), or the placebo."""
        entries = self.by_id.get(cid)
        if not entries:
            return self.placebo, None, "placebo"
        q = words(question)
        best = max(entries, key=lambda e: jaccard(q, e[2]))
        return best[1], jaccard(q, best[2]), "bank"

    def __len__(self):
        return sum(len(v) for v in self.by_id.values())


def bank_pairs(diag_verdicts, dialogues, ids):
    """The pairs for the rows in `ids`: the diag run's greedy question (its
    verdict rows carry question, reply and the scoring-time leak screen) and
    stage 1's dialogues (q0 is the same greedy question again; it is kept
    once by the bank)."""
    pairs = [{"id": v["id"], "question": v["question"], "reply": v["reply"],
              "leak_flags": v.get("leak_flags"), "quotes_result": v.get("quotes_result")}
             for v in diag_verdicts if v["id"] in ids]
    pairs += [d for d in dialogues if d["id"] in ids]
    return pairs
