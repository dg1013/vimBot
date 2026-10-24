"""Tests for the proposed global prompts: `pytest scripts/global_additions`."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import apply_review as A       # noqa: E402
import compare as C            # noqa: E402
import generate as G           # noqa: E402
import validate as V           # noqa: E402


def test_generate_is_seeded_and_complete():
    a, b = G.generate(24, 5), G.generate(24, 5)
    assert a == b and a
    for r in a:
        assert r["category"] == "global" and r["mode"] == "ex" and r["skill"] in G.WHY
        assert r["buffer_before"].endswith("\n") and r["instruction"] and r["gold_command"]
    assert G.generate(24, 6) != a


def test_every_domain_yields_candidates():
    rows = G.generate(len(G.DOMAINS) * 4, 11)
    assert {r["domain"] for r in rows} == set(G.DOMAINS)


def test_case_blind_marks_every_global_pattern_and_nothing_else():
    assert V.case_blind("g/x/d") == "g/\\cx/d"
    assert V.case_blind("2,$v/a/d") == "2,$v/\\ca/d"
    assert V.case_blind("%s/^<\\{7}.*\\n//|g/^=\\{7}/.,/^>\\{7}/d") == "%s/^<\\{7}.*\\n//|g/\\c^=\\{7}/.,/^>\\{7}/d"
    assert V.case_blind("/^\\[db\\]$/,$g/^#/d") == "/^\\[db\\]$/,$g/\\c^#/d"


def row(cmd, buf="a\nB\nc\n", ins="Edit the lines.", **kw):
    r = {"cand": 1, "instruction": ins, "buffer_before": buf, "gold_command": cmd, "mode": "ex",
         "cursor": [1, 1], "tier": "T3", "category": "global", "skill": "move", "domain": "csv", "why": "w"}
    r.update(kw)
    return r


def test_gate_rejects_no_ops_invalid_and_case_dependent_rows():
    assert V.gate(row("g/zzz/d"), set(), set(), set()) == "no-op"
    assert V.gate(row("g/a/qqqq"), set(), set(), set()).startswith("vim rejected it")
    # 'b' matches only the lowercase... the buffer has 'B': case decides the result
    assert V.gate(row("g/b/d", buf="b\nB\nc\n"), set(), set(), set()) == "case decides the result"
    ok = row("g/c/d")
    assert V.gate(ok, set(), set(), set()) is None and ok["buffer_after"] == "a\nB\n"


def test_gate_refuses_a_leakage_key_shared_with_held_out_rows():
    from build_splits import leakage_keys
    r = row("g/c/d")
    held = {k for k in leakage_keys(dict(r, id="x")) if k[0] == "command"}
    assert V.gate(r, held, set(), set()).startswith("leakage key shared with val/test")


def test_select_caps_commands_and_sentences():
    rows = [row("g/c/d", ins="same", cand=i, domain="d%d" % i) for i in range(6)]
    assert len(V.select(rows, 10, 1)) == V.PER_COMMAND
    rows = [row("g/c%d/d" % i, ins="same words", cand=i, domain="d%d" % i) for i in range(6)]
    assert len(V.select(rows, 10, 1)) == V.PER_SENTENCE


def proposed(tmp_path):
    rows = [{"id": "glb16-0001", "instruction": "Delete c.", "buffer_before": "a\nb\nc\n", "gold_command": "g/c/d",
             "buffer_after": "a\nb\n", "mode": "ex", "cursor": [1, 1], "needs_review": True},
            {"id": "glb16-0002", "instruction": "Delete b.", "buffer_before": "a\nb\nc\n", "gold_command": "g/b/d",
             "buffer_after": "a\nc\n", "mode": "ex", "cursor": [1, 1], "needs_review": True}]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def test_apply_review_drops_edits_and_records_verdicts(tmp_path):
    rows = proposed(tmp_path)
    changes = {"edits": {"glb16-0002": {"gold_command": "g/a/d", "instruction": "Delete a."}},
               "deletes": {"glb16-0001": "wording does not match"},
               "review": {"glb16-0002": {"verdict": "ok"}}}
    kept, dropped, report, refused = A.apply(rows, changes, "2026-10-17")
    assert [r["id"] for r in kept] == ["glb16-0002"] and not refused
    assert kept[0]["buffer_after"] == "b\nc\n" and kept[0]["needs_review"] is False
    assert dropped[0]["drop_reason"] == "wording does not match"


def test_apply_review_refuses_an_edit_that_breaks_the_gold(tmp_path):
    rows = proposed(tmp_path)
    kept, dropped, report, refused = A.apply(rows, {"edits": {"glb16-0001": {"gold_command": "g/zzz/d"}}}, "x")
    assert refused == [("glb16-0001", "the edited gold is a no-op")]


def test_apply_review_cli_dry_run_writes_nothing(tmp_path):
    proposed(tmp_path)
    ch = tmp_path / "c.json"
    ch.write_text(json.dumps({"deletes": {"glb16-0001": "x"}}))
    before = (tmp_path / "cases.jsonl").read_text()
    assert A.main(["--changes", str(ch), "--dir", str(tmp_path), "--dry-run"]) == 0
    assert (tmp_path / "cases.jsonl").read_text() == before
    assert A.main(["--changes", str(ch), "--dir", str(tmp_path)]) == 0
    assert len((tmp_path / "cases.jsonl").read_text().splitlines()) == 1
    assert (tmp_path / "dropped.jsonl").exists()


def test_compare_paired_counts_and_mcnemar():
    r = C.paired({"a": 0, "b": 1, "c": 1}, {"a": 1, "b": 1, "c": 0}, ["a", "b", "c", "zz"])
    assert (r["n"], r["gained"], r["lost"]) == (3, 1, 1) and r["p"] == 1.0
    assert abs(C.mcnemar_p(6, 0) - 0.03125) < 1e-12
