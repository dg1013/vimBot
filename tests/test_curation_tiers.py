"""A curated tier is authoring work, and no tool may quietly re-derive it.

`guess_tier` is best-effort by its own docstring and is wrong on whole
families of commands: the `<` and `>` inside `<C-v>` and `<Esc>` read as
shift operators, so every blockwise command scores three-plus operations and
lands in T3. Two tools touch the tier field, and both must defer to an
authored one — ingest when a draft states its tier, and apply_curation when
it rewrites a row. The second lesson cost three quota cells on 2026-09-06:
editing sixteen rows' buffers silently retiered them and emptied join T1,
move-copy T3 and operator-motion T3.
"""

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from apply_curation import verify                    # noqa: E402
from ingest_handcrafted import draft_tier            # noqa: E402
from make_v3_emulator import guess_tier              # noqa: E402

DRAFT = {"category": "visual-block", "mode": "keys",
         "command": "<C-v>3jI# <Esc>", "instruction": "x",
         "buffer_before": "a\nb\nc\nd\n"}


def _row(**over):
    row = {"id": "t-1", "category": "operator-motion", "mode": "keys",
           "instruction": "Delete the second word on this line.",
           "gold_command": "wdw", "buffer_before": "alpha beta gamma\n",
           "cursor": [1, 1], "tier": "T3", "tier_provenance": "authored"}
    row.update(over)
    return row


# ------------------------------------------------------------------ ingest

def test_draft_tier_honours_an_authored_tier():
    assert draft_tier(dict(DRAFT, tier="T2")) == ("T2", "authored")
    assert draft_tier(dict(DRAFT, tier=" t1 ")) == ("T1", "authored")


def test_draft_tier_falls_back_to_the_heuristic():
    tier, prov = draft_tier(dict(DRAFT))
    assert (tier, prov) == (guess_tier("keys", DRAFT["command"]), "heuristic")


def test_the_heuristic_is_the_reason_authored_tiers_exist():
    # the blockwise miscount this test file's docstring describes
    assert guess_tier("keys", "<C-v>3jI# <Esc>") == "T3"


# ---------------------------------------------------------- apply_curation

def test_editing_a_row_keeps_its_authored_tier():
    row, why = verify(_row(), {}, True)
    assert why is None
    assert (row["tier"], row["tier_provenance"]) == ("T3", "authored")


def test_editing_a_row_re_derives_a_heuristic_tier():
    row, why = verify(_row(tier="T3", tier_provenance="heuristic"), {}, True)
    assert why is None
    assert (row["tier"], row["tier_provenance"]) == (
        guess_tier("keys", "wdw"), "heuristic")


def test_a_row_with_no_tier_still_gets_one():
    row = _row()
    del row["tier"], row["tier_provenance"]
    row, why = verify(row, {}, True)
    assert why is None
    assert row["tier_provenance"] == "heuristic" and row["tier"] in ("T1", "T2", "T3")


def test_a_command_edit_under_an_authored_tier_is_reported(tmp_path, capsys, monkeypatch):
    """Keeping the tier is right for a reworded instruction and questionable
    for a new command, so the tool names the rows whose tier it kept."""
    import json

    import apply_curation

    row = _row(id="hc-9001", dataset="v3")
    changes = tmp_path / "c.json"
    changes.write_text(json.dumps({"edits": {"hc-9001": {"command": "dw"}}}),
                       encoding="utf-8")
    monkeypatch.setattr(apply_curation, "load_all", lambda: (
        {"hc-9001": ("v3", "handcrafted_cases.jsonl", dict(row))},
        {("v3", "handcrafted_cases.jsonl"): [dict(row)]}))
    monkeypatch.setattr(sys, "argv",
                        ["apply_curation.py", "--changes", str(changes), "--dry-run"])
    assert apply_curation.main() == 0
    out = capsys.readouterr().out
    assert "CHECK THE TIER" in out and "hc-9001" in out
    assert "'wdw' -> 'dw'" in out


def test_an_instruction_edit_alone_is_not_reported(tmp_path, capsys, monkeypatch):
    import json

    import apply_curation

    row = _row(id="hc-9002", dataset="v3")
    changes = tmp_path / "c.json"
    changes.write_text(json.dumps(
        {"edits": {"hc-9002": {"instruction": "Remove the first word on this line."}}}),
        encoding="utf-8")
    monkeypatch.setattr(apply_curation, "load_all", lambda: (
        {"hc-9002": ("v3", "handcrafted_cases.jsonl", dict(row))},
        {("v3", "handcrafted_cases.jsonl"): [dict(row)]}))
    monkeypatch.setattr(sys, "argv",
                        ["apply_curation.py", "--changes", str(changes), "--dry-run"])
    assert apply_curation.main() == 0
    assert "CHECK THE TIER" not in capsys.readouterr().out


# --------------------------------------------------------------- the corpus

@pytest.mark.parametrize("path", ["data/v3/handcrafted_cases.jsonl",
                                  "data/v4/handcrafted_cases.jsonl"])
def test_every_handcrafted_row_states_where_its_tier_came_from(path):
    import json
    p = REPO / path
    if not p.exists():
        pytest.skip("%s not present" % path)
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    bad = [r["id"] for r in rows
           if r.get("tier") and r.get("tier_provenance") not in
           ("authored", "heuristic")]
    assert not bad, "tier with no provenance: %s" % bad[:5]
