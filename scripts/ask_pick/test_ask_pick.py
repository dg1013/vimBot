"""Tests for the ask-and-pick experiment (CPU only): `pytest scripts/ask_pick`."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ask_pick_lib as L        # noqa: E402

CASE = {"id": "a", "instruction": "Insert hello at the start", "buffer_before": "x\n", "cursor": [1, 1],
        "buffer_after": "SENTINEL-AFTER\n", "gold_command": "SENTINEL-GOLD"}


def test_clarified_case_appends_the_dialogue_and_hides_nothing_new():
    c = L.clarified_case(CASE, "With a space?", "No space.")
    assert c["instruction"].startswith("Insert hello at the start\nClarifying question: With a space?")
    assert c["instruction"].endswith("User's answer: No space.")
    assert c["buffer_before"] == CASE["buffer_before"] and CASE["instruction"] == "Insert hello at the start"
    sys.path.insert(0, str(HERE.parents[0] / "reranker"))
    from verifier_lib import verifier_messages
    text = json.dumps(verifier_messages(c, "ex", "s/^/hello/"))
    assert "SENTINEL" not in text and "No space." in text


def test_candidates_skip_parse_failures_and_label_matches():
    row = {"id": "a", "k": 1, "commands": [{"mode": "ex", "command": "s/^/hello/", "state": "match", "n": 3},
                                           {"mode": "ex", "command": "s/^/hello /", "state": "valid_wrong", "n": 4},
                                           {"mode": None, "command": None, "state": "parse", "n": 1}]}
    c = L.candidates(row)
    assert [x["label"] for x in c] == [1, 0] and sum(x["n"] for x in c) == 7


def test_pick_majority_and_best_score():
    cands = [{"cand_mode": "ex", "cand_command": "a", "n": 1, "label": 1},
             {"cand_mode": "ex", "cand_command": "b", "n": 5, "label": 0}]
    scores = {("x", "ex", "a"): 2.0, ("x", "ex", "b"): -1.0}
    assert L.pick_label(cands, scores, "x") == 1 and L.majority_label(cands) == 0
    assert L.best_score(cands, scores, "x") == 2.0 and L.pick_label([], scores, "x") == 0


def test_policy_asks_the_top_share_and_answers_elsewhere():
    ids = ["a", "b", "c", "d", "e"]
    trig = {"a": 5, "b": 4, "c": 3, "d": 2, "e": 1}
    after = {i: 1 for i in ids}
    notasked = {i: 0 for i in ids}
    out, asked = L.policy(ids, trig, 0.4, after, notasked)
    assert asked == {"a", "b"} and out == {"a": 1, "b": 1, "c": 0, "d": 0, "e": 0}


def test_ranks_share_ties_and_combine_averages():
    assert L.ranks([10, 20, 20, 5]) == [1.0, 2.5, 2.5, 0.0]
    comb = L.combine(["a", "b", "c"], {"a": 3, "b": 2, "c": 1}, {"a": 1, "b": 2, "c": 3})
    assert comb["a"] == comb["b"] == comb["c"] == 1.0


def test_paired_ci_and_mcnemar():
    d, lo, hi = L.paired_ci(28, 21, 429)
    assert round(d, 1) == 1.6 and round(lo, 1) == -1.6 and round(hi, 1) == 4.8
    assert round(L.mcnemar_p(28, 21), 3) == 0.392 and round(L.mcnemar_p(11, 5), 2) == 0.21
    assert L.auroc([3, 4], [1, 2]) == 1.0 and L.auroc([1], [1]) == 0.5
