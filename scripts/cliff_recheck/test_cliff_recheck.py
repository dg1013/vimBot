"""Tests for the cliff re-check: run with `pytest scripts/cliff_recheck`."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cliff_recheck as C        # noqa: E402


def row(cid, k, g=8, **kw):
    r = {"id": cid, "k": k, "G": g, "category": kw.get("category", "substitute"),
         "tier": kw.get("tier", "T3"), "mode": kw.get("mode", "ex"),
         "commands": kw.get("commands", [])}
    return r


def test_never_solved_keeps_only_zero_k_train_ids_in_probe_order():
    probe = [row("b", 0), row("a", 3), row("c", 0), row("z", 0), row("b", 0)]
    ids, skipped = C.never_solved(probe, {"a", "b", "c"})
    assert ids == ["b", "c"]
    assert skipped == 1                           # z is not in the train split


def test_wilson_interval_contains_the_estimate_and_stays_in_bounds():
    lo, hi = C.wilson(3, 10)
    assert 0 <= lo < 0.3 < hi <= 1
    assert C.wilson(0, 10)[0] == 0.0
    assert C.wilson(10, 10)[1] == 1.0


def test_k_bins_cover_every_count_once():
    labels = [C.k_bin(k) for k in range(33)]
    assert labels[0] == "0" and labels[1] == "1" and labels[32] == "32"
    assert labels[2] == labels[3] == "2-3" and labels[31] == "16-31"


def test_consistent_wrong_needs_a_runnable_wrong_majority():
    cmds = [{"command": "x", "state": "valid_wrong", "n": 24},
            {"command": "y", "state": "invalid", "n": 8}]
    assert C.consistent_wrong(row("a", 0, 32, commands=cmds))["command"] == "x"
    cmds[0]["n"] = 23                                  # below 75% of 32
    assert C.consistent_wrong(row("a", 0, 32, commands=cmds)) is None
    cmds = [{"command": "y", "state": "invalid", "n": 32}]
    assert C.consistent_wrong(row("a", 0, 32, commands=cmds)) is None


def test_summarize_counts_solvable_and_per_visit_odds():
    ids = ["a", "b", "c", "d"]
    recheck = [row("a", 0, 32), row("b", 1, 32), row("c", 16, 32, tier="T1"), row("e", 5, 32)]
    s = C.summarize(recheck, ids)
    assert s["rechecked"] == 3 and s["missing"] == 1        # d not yet probed, e not planned
    assert s["solvable"] == 2
    assert abs(s["solvable_share"] - 2 / 3) < 1e-9
    assert s["by_tier"]["T1"][:2] == (1, 1)
    # a prompt at p = 1/32 is seen in a group of 32 more often than in a group of 8
    assert s["per_visit"][32] > s["per_visit"][8]
    assert 0 < s["refiled_at_8"] < 1


def test_shelves_pool_the_recheck_over_the_probe(tmp_path):
    probe = tmp_path / "p.jsonl"
    recheck = tmp_path / "r.jsonl"
    probe.write_text("\n".join(json.dumps(r) for r in [row("a", 0), row("b", 0), row("c", 8)]) + "\n")
    recheck.write_text("\n".join(json.dumps(r) for r in [row("a", 2, 32), row("b", 0, 32)]) + "\n")
    before = C.shelves(["a", "b", "c"], [probe])
    after = C.shelves(["a", "b", "c"], [probe, recheck])
    assert before["cliff"] == 2 and before["known"] == 1
    assert after["cliff"] == 1 and after["frontier"] == 1 and after["known"] == 1


def test_report_end_to_end(tmp_path):
    cases = tmp_path / "train.jsonl"
    cases.write_text("".join(json.dumps({"id": i}) + "\n" for i in "abc"))
    probe = tmp_path / "p.jsonl"
    probe.write_text("".join(json.dumps(r) + "\n" for r in [row("a", 0), row("b", 0), row("c", 8)]))
    ids = tmp_path / "ids.txt"
    assert C.main(["ids", "--probe", str(probe), "--cases", str(cases), "--out", str(ids)]) == 0
    assert ids.read_text().split() == ["a", "b"]
    recheck = tmp_path / "r.jsonl"
    recheck.write_text("".join(json.dumps(r) + "\n" for r in [
        row("a", 3, 32), row("b", 0, 32, commands=[{"command": "dd", "state": "valid_wrong", "n": 30}])]))
    out = tmp_path / "report.md"
    assert C.main(["report", "--probe", str(probe), "--cases", str(cases), "--ids", str(ids),
                   "--recheck", str(recheck), "--out", str(out), "--json", str(tmp_path / "r.json")]) == 0
    text = out.read_text()
    assert "solved at least once in 32 | **1 (50.0%" in text
    assert "`dd`" in text                               # the stable wrong answer is listed
    dump = json.loads((tmp_path / "r.json").read_text())
    assert dump["shelves_after"]["frontier"] == 1
