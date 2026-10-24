"""Tests for the 9B methods comparison (CPU only): `pytest scripts/methods_9b`."""
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_9b as C        # noqa: E402


def cand(cid, cmd, n, label, greedy=False, mode="ex"):
    return {"id": cid, "cand_mode": mode, "cand_command": cmd, "n": n, "label": label, "greedy": greedy}


def test_pick_follows_score_then_count_then_greedy():
    cands = [cand("a", "x", 1, 1), cand("a", "y", 5, 0), cand("a", "z", 0, 1, greedy=True)]
    scores = {("a", "ex", "x"): 2.0, ("a", "ex", "y"): -1.0, ("a", "ex", "z"): 9.0}
    assert C.pick(cands, scores)[0] == 1                       # z is greedy-only (n = 0): never ranked
    assert C.pick(cands, scores, lam=5.0)[0] == 0               # a strong count prior flips it to y
    tie = [cand("b", "p", 2, 0), cand("b", "q", 2, 1, greedy=True)]
    assert C.pick(tie, {("b", "ex", "p"): 1.0, ("b", "ex", "q"): 1.0})[0] == 1
    assert C.pick([], {}) == (0, float("-inf"))


def test_pool_merges_duplicates_sums_counts_and_keeps_labels():
    c4 = [cand("a", "x", 3, 1), cand("a", "y", 5, 0)]
    c9 = [cand("a", "x", 4, 1, greedy=True), cand("a", "w", 4, 0)]
    pool = C.pooled(c4, c9)
    by = {c["cand_command"]: c for c in pool}
    assert by["x"]["n"] == 7 and by["x"]["sources"] == {"4b", "9b"} and by["x"]["greedy"]
    assert set(by) == {"x", "y", "w"}
    with pytest.raises(ValueError):
        C.pooled([cand("a", "x", 1, 1)], [cand("a", "x", 1, 0)])


def test_pooled_scores_take_either_file():
    pool = C.pooled([cand("a", "x", 1, 1)], [cand("a", "w", 1, 0)])
    s = C.pooled_scores(pool, {("a", "ex", "x"): 1.0}, {("a", "ex", "w"): 2.0}, "a")
    assert s == {("a", "ex", "x"): 1.0, ("a", "ex", "w"): 2.0}


def test_model_methods_route_asked_rows_to_the_dialogue():
    ids = ["a", "b", "c", "d", "e"]
    ex = []
    scores = {}
    for k, i in enumerate(ids):                    # row a has the lowest best score
        ex.append(cand(i, "r", 4, int(k % 2 == 0)))
        ex.append(cand(i, "s", 4, int(k % 2 == 1)))
        scores[(i, "ex", "r")] = float(k)
        scores[(i, "ex", "s")] = float(k) - 1
    verd = {i: {"real": {"match": True}, "direct": {"match": False}, "placebo": {"match": False}} for i in ids}
    asks = {i: i == "e" for i in ids}
    out, info = C.model_methods(ex, scores, 0.0, verd, asks, ids)
    assert out["verifier pick"] == {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1}
    assert info["gated"] == {"a"}                 # 20% of 5 rows: the least confident one
    assert out["verifier-gated ask"]["a"] == 1 and out["verifier-gated ask"]["b"] == 0
    assert out["sample rule"] == {"a": 0, "b": 0, "c": 0, "d": 0, "e": 1}
    assert out["rule + verifier"]["e"] == 1 and out["rule + verifier"]["b"] == 0
    assert out["ask only where it helps"] == {i: 1 for i in ids}


def test_statistics():
    assert round(C.mcnemar_p(35, 12), 3) == 0.001 and C.mcnemar_p(0, 0) == 1.0
    d, lo, hi = C.paired_ci(28, 21, 429)
    assert round(d, 1) == 1.6 and lo < 0 < hi


def test_free_ceilings_take_the_best_of_what_each_model_produced():
    import free_ceilings as F
    probe = [{"id": "a", "k": 0}, {"id": "b", "k": 3}, {"id": "c", "k": 0}]
    verd = {"a": {"direct": {"match": True}, "real": {"match": False}},
            "b": {"direct": {"match": False}, "real": {"match": False}},
            "c": {"direct": {"match": False}, "real": {"match": True}}}
    out = F.outcomes(probe, verd, ["a", "b", "c"])
    assert out["any of 8 samples"] == {"a": 0, "b": 1, "c": 0}
    assert out["ask only where it helps"] == {"a": 1, "b": 0, "c": 1}
    assert out["anything it produced"] == {"a": 1, "b": 1, "c": 1}
    assert F.either({"a": 0, "b": 1}, {"a": 1, "b": 0}) == {"a": 1, "b": 1}
