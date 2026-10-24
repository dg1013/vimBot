"""Tests for scripts/never_review/ (the never-solved train review page): which
rows count as never solved, the review-history line, and the answer shown in
the third buffer pane.

Run:  python3 -m pytest scripts/never_review        (not collected by `pytest tests/`)
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (REPO / "scripts", REPO / "src", HERE):
    sys.path.insert(0, str(p))

import build_never_review as bnr               # noqa: E402


def test_never_sets_count_zero_of_g_and_skip_rows_no_longer_in_train():
    probes = {"a": {"x": {"k": 0}, "y": {"k": 3}, "gone": {"k": 0}},
              "b": {"x": {"k": 0}, "y": {"k": 0}}}
    assert bnr.never_sets(probes, {"x", "y"}) == {"a": {"x"}, "b": {"x", "y"}}


def test_history_names_what_the_cliff_review_did():
    cliff = {"edits": {"r1": {"instruction": "t"}, "r2": {"command": "dd"}},
             "deletes": {"r3": "why"}, "keep": {"r4"}}
    assert bnr.history("r1", cliff) == "cliff review 2026-09-16: reworded"
    assert bnr.history("r2", cliff) == "cliff review 2026-09-16: retargeted"
    assert bnr.history("r3", cliff) == "cliff review 2026-09-16: dropped"
    assert bnr.history("r4", cliff) == "cliff review 2026-09-16: kept as it was"
    assert bnr.history("r5", cliff) is None


def test_history_says_when_the_whitespace_repair_brought_a_dropped_row_back():
    cliff = {"edits": {"r1": {"instruction": "t"}}, "deletes": {"r3": "why"}, "keep": set()}
    ws = {"edits": {"r3": {"command": "r<Space>"}, "r1": {"command": "x"}, "r6": {"command": "y"}}}
    assert bnr.history("r3", cliff, ws) == \
        "cliff review 2026-09-16: dropped; restored by the whitespace repair, gold in token notation"
    assert bnr.history("r1", cliff, ws) == \
        "cliff review 2026-09-16: reworded; whitespace repair 2026-09-16: gold in token notation"
    assert bnr.history("r6", cliff, ws) == "whitespace repair 2026-09-16: gold in token notation"


def test_attach_notes_runs_a_suggested_command_and_leaves_unread_rows_bare():
    card = {"id": "x", "mode": "keys", "cmd": "dd", "cur": [1, 1], "bef": "a b\nc\n"}
    bare = {"id": "y", "mode": "ex", "cmd": "1d", "cur": [1, 1], "bef": "a\n"}
    notes = {"x": {"id": "x", "class": "defect", "comment": "why",
                   "suggest": {"instruction": "t", "command": "dw"}}}
    assert bnr.attach_notes([card, bare], notes) == 1
    assert card["note"]["cls"] == "defect" and card["note"]["sugg"]["command"] == "dw"
    assert card["note"]["sbuf"].split("\n")[:2] == ["b", "c"]
    assert "note" not in bare


def test_attach_notes_skips_the_run_when_only_the_wording_changes():
    card = {"id": "x", "mode": "ex", "cmd": "1d", "cur": [1, 1], "bef": "a\n"}
    bnr.attach_notes([card], {"x": {"id": "x", "class": "unstated", "comment": "c",
                                    "suggest": {"instruction": "Delete line one"}}})
    assert "sbuf" not in card["note"]


def test_load_notes_refuses_a_row_noted_twice(tmp_path):
    import json
    import pytest
    import notes
    (tmp_path / "b000.json").write_text(json.dumps([{"id": "x", "class": "model", "comment": "a"}]))
    (tmp_path / "b001.json").write_text(json.dumps([{"id": "y", "class": "model", "comment": "b"}]))
    assert set(notes.load_notes(tmp_path)) == {"x", "y"}
    (tmp_path / "b002.json").write_text(json.dumps([{"id": "x", "class": "hard", "comment": "c"}]))
    with pytest.raises(SystemExit):
        notes.load_notes(tmp_path)


def test_pane_answer_pools_runnable_wrong_answers_across_probes():
    def row(*cmds):
        return {"commands": [{"mode": "ex", "command": c, "n": n, "state": s} for c, n, s in cmds]}
    rows = {"a": {"x": row(("p", 5, "valid_wrong"), ("bad", 8, "invalid"))},
            "b": {"x": row(("q", 6, "valid_wrong"), ("p", 2, "valid_wrong"))},
            "c": {}}
    alt = bnr.pane_answer(rows, "x")
    assert (alt["cmd"], alt["by"]) == ("p", {"a": 5, "b": 2})       # 7 pooled beats 6
    assert bnr.pane_answer({"a": {"x": row(("bad", 8, "invalid"))}}, "x") is None
