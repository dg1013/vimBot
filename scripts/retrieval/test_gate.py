"""Tests for the similarity gate (CPU only): `pytest scripts/retrieval`."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gate as G            # noqa: E402


def test_on_off_and_variable_k_choose_by_the_right_score():
    s = [30.0, 20.0, 12.0, 9.0, 8.0, 7.0, 6.0, 5.0]       # nearest first
    assert G.choose_on_off(s, 25.0, 5) == 5 and G.choose_on_off(s, 31.0, 5) == 0 and G.choose_on_off([], 0.0, 5) == 0
    assert G.choose_variable(s, 5.0) == 8                  # the 8th scores 5.0 >= 5.0
    assert G.choose_variable(s, 8.5) == 3                  # 3rd 12.0 ok, 5th 8.0 not
    assert G.choose_variable(s, 25.0) == 1
    assert G.choose_variable(s, 31.0) == 0


def _arms(ids, right):
    """Arms where the retrieved arms are right on `right[k]` and cost more with k."""
    arms = {}
    for k, name in G.K_ARMS.items():
        arms[name] = {i: {"match": int(i in right[k]), "tokens": 100 + 100 * k, "prompt": 90 + 100 * k} for i in ids}
    return arms


def test_evaluate_and_curve_are_monotone_in_cost_and_pick_keeps_the_gain():
    ids = list(range(10))
    scores = {i: [float(30 - 3 * i)] + [1.0] * 7 for i in ids}      # row 0 most similar
    right = {0: {9}, 1: {0, 9}, 3: {0, 1, 9}, 5: {0, 1, 2, 3, 9}, 8: {0, 1, 2, 3, 4, 9}}
    arms = _arms(ids, right)
    e_all = G.evaluate("on-off", 0.0, ids, scores, arms, 5)
    e_none = G.evaluate("on-off", 100.0, ids, scores, arms, 5)
    assert e_all["exec_match"] == 5 and e_all["gated_on_pct"] == 100 and e_none["exec_match"] == 1
    cv = G.curve("on-off", ids, scores, arms, 5)
    assert cv[0]["gated_on_pct"] == 0 and cv[-1]["gated_on_pct"] == 100
    assert all(a["prompt_tokens_mean"] <= b["prompt_tokens_mean"] for a, b in zip(cv, cv[1:]))
    tau = G.pick_tau(cv, 10.0, 50.0, keep=0.9)              # need >= 46%: rows 0-3 on (score >= 21)
    e = G.evaluate("on-off", tau, ids, scores, arms, 5)
    assert e["exec_match"] == 5 and e["gated_on_pct"] == 40.0
    assert G.pick_tau(cv, 10.0, 50.0, keep=0.0) == float("inf")   # nothing to keep: never retrieve


def test_summarize_curve_gives_one_point_per_fraction():
    cv = [{"tau": float(t), "gated_on_pct": p, "exec_match_pct": 0, "mean_k": 0, "prompt_tokens_mean": 0,
           "tokens_per_solved": 0} for t, p in ((9, 0), (8, 20), (7, 40), (6, 60), (5, 80), (4, 100))]
    pts = G.summarize_curve(cv, fractions=(0, 25, 50, 100))
    assert [p["gated_on_pct"] for p in pts] == [0, 40, 60, 100]


def test_main_writes_a_report_from_arm_folders(tmp_path):
    ids = ["a", "b", "c", "d"]
    right = {0: {"d"}, 1: {"a", "d"}, 3: {"a", "b", "d"}, 5: {"a", "b", "c", "d"}, 8: {"a", "b", "c", "d"}}
    for split, plans in (("test", tmp_path), ("val", tmp_path / "val_plans")):
        plans.mkdir(exist_ok=True)
        (plans / "plan_retr8.jsonl").write_text("".join(json.dumps(
            {"id": i, "scores": [1.0] * 7 + [float(40 - 10 * n)]}) + "\n" for n, i in enumerate(ids)))
        for k, name in G.K_ARMS.items():
            d = tmp_path / name; d.mkdir(exist_ok=True)
            (d / ("exec_%s.jsonl" % split)).write_text("".join(json.dumps({"id": i, "match": i in right[k]}) + "\n" for i in ids))
            (d / ("completions_%s.jsonl" % split)).write_text("".join(json.dumps(
                {"id": i, "prompt_tokens": 100 + 100 * k, "output_tokens": 10}) + "\n" for i in ids))
    rc = G.main(["--run", str(tmp_path), "--pick-split", "val", "--pick-plans", str(tmp_path / "val_plans"), "--read-split", "test"])
    assert rc == 0
    rep = json.loads((tmp_path / "gate_report.json").read_text())
    pk = rep["policies"]["on-off k=5"]["picked"]
    assert pk["read_split"]["exec_match_pct"] >= 75.0 and pk["read_split"]["gated_on_pct"] < 100
    assert (tmp_path / "gate_report.md").exists()
