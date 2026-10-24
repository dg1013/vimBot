"""Tests for the served pooled pick (CPU only, no endpoints): `pytest scripts/serve_pooled`."""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import measure as M     # noqa: E402
import pooled as P      # noqa: E402


def test_merge_counts_each_model_and_dedupes_commands():
    cands = P.merge([("4b", "ex", "dd"), ("9b", "ex", "dd"), ("9b", "ex", "dd"), ("9b", "keys", "x")])
    by = {(c["mode"], c["command"]): c for c in cands}
    assert by[("ex", "dd")]["n"] == 3 and by[("ex", "dd")]["from"] == {"4b": 1, "9b": 2}
    assert by[("keys", "x")]["from"] == {"9b": 1}


def test_pick_takes_the_highest_score_then_the_count():
    cands = [{"mode": "ex", "command": "a", "n": 1, "score": 2.0}, {"mode": "ex", "command": "b", "n": 5, "score": 2.0},
             {"mode": "ex", "command": "c", "n": 9, "score": -1.0}]
    assert P.pick(cands)["command"] == "b" and P.pick([]) is None


def test_single_model_pick_uses_only_that_models_candidates():
    cands = [{"mode": "ex", "command": "a", "n": 2, "from": {"4b": 2}, "score": 3.0},
             {"mode": "ex", "command": "b", "n": 3, "from": {"9b": 3}, "score": 1.0}]
    assert M.single_pick(cands, "9b")["command"] == "b" and M.single_pick(cands, "4b")["command"] == "a"
    assert M.single_pick(cands, "2b") is None


def test_percentile_is_an_order_statistic():
    assert M.pct([3, 1, 2], 50) == 2 and M.pct([], 50) is None and M.pct([5], 95) == 5


def test_draws_pick_the_best_scored_drawn_candidate():
    import random
    import pick_at_k as K
    row = {"n": 4, "commands": [{"mode": "ex", "command": "good", "n": 1, "state": "match"},
                                {"mode": "ex", "command": "bad", "n": 3, "state": "valid_wrong"}]}
    cands, pop = K.expand(row)
    assert sorted(pop) == [0, 1, 1, 1] and [c["label"] for c in cands] == [1, 0]
    scores = {("ex", "good"): 5.0, ("ex", "bad"): 1.0}
    # all four drawn: the verifier picks the right one, voting the wrong one
    assert K.draw_eval({"4b": pop}, {"4b": cands}, scores, random.Random(0), 4) == (1, 1, 0)
    row2 = {"n": 3, "commands": [{"mode": "ex", "command": "bad", "n": 2, "state": "valid_wrong"}]}
    c2, p2 = K.expand(row2)
    assert p2 == [0, 0, None]                     # an unparsed sample can be drawn, never picked


def test_paired_delta_is_the_mean_row_difference():
    import random
    import pick_at_k as K
    a = [0, 1, 0.5, 1]
    assert K.paired_delta(a, a, random.Random(0)) == (0, 0, 0)
    d, lo, hi = K.paired_delta(a, [x + 0.25 for x in a], random.Random(0))
    assert d == lo == hi == 0.25


ROW = {"id": "r1",
       "samples": {"4b": [["ex", "a"], ["ex", "b"], None, ["ex", "b"]],
                   "9b": [["ex", "c"], ["ex", "a"], ["ex", "a"], ["ex", "a"]]},
       "output_tokens": {"4b": [5, 6, 7, 8], "9b": [1, 2, 3, 4]},
       "cands": [{"mode": "ex", "command": "a", "score": 1.0, "state": "valid_wrong"},
                 {"mode": "ex", "command": "b", "score": 3.0, "state": "match"},
                 {"mode": "ex", "command": "c", "score": 2.0, "state": "match"}]}


def test_one_run_uses_each_models_first_k_samples():
    import served_at_k as A
    assert A.first_k(ROW, ("4b",), 1) == (0, 0, 0, 1)            # the 4B's first sample is the wrong a
    assert A.first_k(ROW, ("4b", "9b"), 1) == (1, 1, 1, 2)       # c (the 9B's first) outscores a
    assert A.first_k(ROW, ("4b", "9b"), 4) == (1, 1, 0, 3)       # the verifier takes b; the vote takes a (4 of 7)


def test_draws_at_every_sample_agree_with_the_one_run():
    import served_at_k as A
    import pick_at_k as K
    rows = {m: {"r1": A.ceiling_row(ROW, m)} for m in ("4b", "9b")}
    sc = A.row_maps(ROW)[0]
    curves, _ = K.curves_for(["r1"], rows, lambda i, mode, cmd: sc[(mode, cmd)], 3)
    assert curves["4B + 9B"][128] == list(A.first_k(ROW, ("4b", "9b"), 128)[:3]) == [1, 1, 0]
    assert curves["4B"][128] == [1, 1, 1]


def test_report_from_saved_rows(tmp_path):
    import argparse
    import json
    import served_at_k as A
    rows = [dict(ROW, split="val"), dict(ROW, id="r2", split="val",
                                         cands=[dict(c, state="valid_wrong") for c in ROW["cands"]])]
    (tmp_path / "rows_val.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "passes.jsonl").write_text(json.dumps({"split": "val", "rows": 2, "wall_s": 30.0, "inflight": 2}) + "\n")
    timing = tmp_path / "pick_at_k.json"
    timing.write_text(json.dumps({"timing": {"8": {"mean_s": 1.5}, "64": {"mean_s": 6.0}}}))
    A.report(argparse.Namespace(draws=3, timing_json=str(timing), gpu_hour=0.36), tmp_path, ["val"],
             {"val": [{"id": "r1"}, {"id": "r2"}]})
    rep = json.loads((tmp_path / "report.json").read_text())["splits"]["val"]
    assert rep["one_run"]["4B + 9B"][str(128)]["pick"] == 50.0 and rep["throughput"]["s_per_row"] == 15.0
    assert rep["one_run_vs_8"]["4B + 9B"][str(128)]["gained"] == 0
    assert sorted(rep["one_run"]["4B"], key=int) == ["1", "4", "8", "32", "64", "128"]
    # priced only where timed, by this split's pooled pick: 1.5 s / 0.5 solved = 3 GPU-s; $0.30 per 1,000 at $0.36/h
    assert sorted(rep["cost"]) == ["64", "8"] and rep["cost"]["8"]["gpu_s_per_solved"] == 3.0
    assert rep["cost"]["8"]["usd_per_1000_solved"] == 0.3


def test_saved_timings_load_with_integer_keys_and_keep_when_measured(tmp_path):
    import json
    import pick_at_k as K
    p = tmp_path / "pick_at_k.json"
    assert K.load_timing(p) == {}
    p.write_text(json.dumps({"timing": {"128": {"mean_s": 14.0}, "8": {"mean_s": 1.2}}}))
    t = K.load_timing(p)
    assert sorted(t) == [8, 128]
    entry = dict(t[8], p50_s=1, p95_s=2, sample_p50_s=1, score_p50_s=0, candidates_mean=5, tokens_mean=100,
                 measured="then")
    lines = K.timing_lines({8: entry}, {"4B + 9B": {8: {"pick": 60.0}}}, 0.30)
    assert any(l.startswith("| 8 | 1.00 s") and l.endswith("| then |") for l in lines)
    assert any(l.startswith("| 8 | 1.20 | 60.0% | 2.00 |") for l in lines)


def test_asymmetric_draws_take_k_per_model():
    import random
    import pick_at_k as K
    c4, p4 = K.expand({"n": 2, "commands": [{"mode": "ex", "command": "bad", "n": 2, "state": "valid_wrong"}]})
    c9, p9 = K.expand({"n": 2, "commands": [{"mode": "ex", "command": "good", "n": 2, "state": "match"}]})
    scores = {("ex", "bad"): 1.0, ("ex", "good"): 2.0}
    draws, cands = {"4b": p4, "9b": p9}, {"4b": c4, "9b": c9}
    assert K.draw_eval(draws, cands, scores, random.Random(0), {"4b": 2, "9b": 0}) == (0, 0, 0)
    assert K.draw_eval(draws, cands, scores, random.Random(0), {"4b": 2, "9b": 1}) == (1, 1, 0)   # pick right, vote wrong


def test_pooled_samples_k_per_model():
    import types
    e4, e9 = types.SimpleNamespace(name="a", url="u"), types.SimpleNamespace(name="b", url="v")
    pp = P.Pooled(e4, e9, k=8)
    assert pp.k_for("4b") == 8 and pp.k_for("9b") == 8
    pp.k9 = 2
    assert pp.k_for("4b") == 8 and pp.k_for("9b") == 2


def test_composed_cost_and_cost_rows():
    import followups as F
    assert F.composed(0.4, 0.1, 0.2, 0.2) == 0.4 + 0.2 * 0.3
    r = F.cost_row(1.0, 50.0, 0.36)
    assert r["gpu_s_per_solved"] == 2.0 and r["usd_per_1000_solved"] == 0.2
    assert F.cost_row(1.0, 0.0, 0.36)["gpu_s_per_solved"] is None
    assert F.best_gpu_s([{"req_per_s": 2.0, "concurrency": 1}, {"req_per_s": 4.0, "concurrency": 8}]) == (0.25, 8)


def test_mean_by_groups_per_id_values():
    import breakdown as B
    assert B.mean_by({"a": 1.0, "b": 0.0, "c": 1.0}, {"a": "T1", "b": "T1", "c": "T2"}) == {"T1": (0.5, 2), "T2": (1.0, 1)}
