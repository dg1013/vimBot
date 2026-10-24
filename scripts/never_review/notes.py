"""Claude's per-row notes on the never-solved train rows: where they live, the
five classes a note can take, and the loader that check_comments.py and the
review page builder share.

A note is {id, class, comment, suggest?}; `suggest` holds any of instruction,
command, mode, cursor, buffer_before -- the fields apply_curation.py edits.
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
COMMENTS = REPO / "runs" / "never_review_2026-09-25" / "comments"
# Most to least actionable; the page lists them in this order.
CLASSES = {
    "defect": "the gold does something the instruction does not say, or the row cannot be solved from its words",
    "ambiguous": "the words admit another faithful reading, which the models take",
    "unstated": "the gold's exact form (spacing, names, values) is not derivable from the words or buffer",
    "hard": "a fair row that demands a long or subtle composition",
    "model": "a fair row; the models make an error",
}
SUGGESTABLE = ("instruction", "command", "mode", "cursor", "buffer_before")


def load_notes(folder=COMMENTS):
    """{row id: note} over every b*.json in `folder`; a row noted twice is an error."""
    notes, where = {}, {}
    for p in sorted(pathlib.Path(folder).glob("b*.json")):
        for n in json.loads(p.read_text(encoding="utf-8")):
            if n["id"] in notes:
                sys.exit("%s: %s already noted in %s" % (p.name, n["id"], where[n["id"]]))
            notes[n["id"]] = n
            where[n["id"]] = p.name
    return notes
