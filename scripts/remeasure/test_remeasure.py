"""Tests for scripts/remeasure/ (test v2, DECISIONS.md 2026-09-25): the plan
covers the results table with one regeneration per distinct (model, adapter,
base), and the report merges a run's re-scored unchanged rows with its
regenerated reworded rows.

Run:  python3 -m pytest scripts/remeasure            (not collected by `pytest tests/`)
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (REPO / "scripts", REPO / "src", HERE):
    sys.path.insert(0, str(p))

import plan                                    # noqa: E402
import v2_report as report                       # noqa: E402


def test_config_names():
    assert plan.config_name("qwen3.5-4b", None, "bf16") == "qwen3.5-4b__zero-shot"
    assert plan.config_name("qwen3.5-9b", "runs/qwen3.5-9b-sft/best", "nf4") == "qwen3.5-9b-sft__nf4"
    assert plan.config_name("qwen3.5-9b", "runs/qwen3.5-9b-sft/best", "bf16") == "qwen3.5-9b-sft__bf16"


@pytest.mark.skipif(not all((plan.RUNS / d / "baseline_test.json").exists() for _, d in plan.TABLE),
                    reason="recorded runs not on this machine")
def test_plan_covers_the_results_table_once_per_regeneration():
    rows, configs = plan.plan()
    assert len(rows) == len(plan.TABLE)
    names = {c["name"] for c in configs}
    assert len(names) == len(configs)
    assert all(r["config"] in names for r in rows)
    # the served adapter's two rows (pure-torch and fused kernels) share one regeneration
    served = [r["config"] for r in rows if r["run"].startswith("qwen3.5-4b-grpo-gated")
              and "cliffleg" not in r["run"]]
    assert len(served) == 2 and len(set(served)) == 1


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _case(i, before, after, ins="x"):
    return {"id": i, "category": "substitute", "instruction": ins, "mode": "ex",
            "buffer_before": before, "gold_command": "", "buffer_after": after, "cursor": [1, 1]}


def test_measure_merges_reused_and_regenerated_rows(tmp_path, monkeypatch):
    v1 = {"a": _case("a", "foo\n", "bar\n"), "b": _case("b", "one\ntwo\n", "two\n"),
          "r": _case("r", "x\n", "y\n", "old"), "d": _case("d", "p\n", "q\n")}
    v2 = [v1["a"], v1["b"], _case("r", "x\n", "y\n", "new")]
    run = tmp_path / "runs" / "m"
    (run).mkdir(parents=True)
    (run / "baseline_test.json").write_text(json.dumps(
        {"n": 4, "metrics": {"exec_match_pct": 50.0, "valid_pct": 75.0, "output_tokens_mean": 5}}))
    _write(run / "preds_test.jsonl", [{"id": "a", "command": "s/foo/bar/", "mode": "ex"},
                                      {"id": "b", "command": "s/one/uno/", "mode": "ex"},
                                      {"id": "r", "command": "notacommand", "mode": "ex"},
                                      {"id": "d", "command": "s/p/q/", "mode": "ex"}])
    _write(run / "completions_test.jsonl", [{"id": i, "output_tokens": 4} for i in "abrd"])
    regen = tmp_path / "regen"
    _write(regen / "exec_test.jsonl", [{"id": "r", "category": "substitute", "command": "s/x/y/",
                                        "mode": "ex", "valid": True, "match": True}])
    _write(regen / "completions_test.jsonl", [{"id": "r", "output_tokens": 10}])
    monkeypatch.setattr(report, "RUNS", tmp_path / "runs")
    row = {"label": "m", "run": "m", "config": "m", "model": "qwen3.5-4b", "date": "2026-09-20"}
    res = report.measure(row, v1, v2, {"r"}, {"d"}, regen)
    assert res["v2"]["exec_match"] == 2            # a from the saved prediction, r regenerated
    assert res["v2"]["exec_match_pct"] == round(100 * 2 / 3, 1)
    assert res["v2"]["tokens_mean"] == round((4 + 4 + 10) / 3, 1)
    assert (res["reworded_solved_before"], res["reworded_solved_after"]) == (0, 1)
    assert res["dropped_solved"] == 1
    assert res["mixed_kernels"] is False           # measured after the kernels arrived
    assert set(res["_merged"]) == {"a", "b", "r"}
