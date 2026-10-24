"""Tests for scripts/fuel_probe/: the shelf arithmetic and the smoke gate that
stops a model whose answers do not end on their own.

Run:  python3 -m pytest scripts/fuel_probe            (not collected by `pytest tests/`)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare                                 # noqa: E402


def _row(i, k, states, g=8):
    return {"id": i, "k": k, "G": g, "states": states}


def test_stats_shelves_and_fuel():
    rows = [_row("a", 0, {"valid_wrong": 8}), _row("b", 8, {"match": 8}),
            _row("c", 4, {"match": 4, "invalid": 4}), _row("d", 2, {"match": 2, "truncated": 6})]
    s = compare.stats(rows)
    assert (s["never"], s["sometimes"], s["always"]) == (0.25, 0.5, 0.25)
    assert s["fuel"] == (0 + 0 + 4 * 4 / 8 + 2 * 6 / 8) / 4
    assert s["truncated"] == 6 / 32 and s["match"] == 14 / 32


def _write(run, model, rows, log):
    (run / ("difficulty_probe_%s.jsonl" % model)).write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / ("probe_%s.log" % model)).write_text(log)


def test_smoke_gate_passes_clean_answers_and_stops_capped_ones(tmp_path, capsys):
    ok = [_row("r%d" % i, 3, {"match": 3, "valid_wrong": 5}) for i in range(10)]
    _write(tmp_path, "good", ok, "probe done: 10 prompts in 30 s -> x\n")
    assert compare.check(tmp_path, "good") == 0
    assert "3.0 s/prompt" in capsys.readouterr().out
    capped = [_row("r%d" % i, 0, {"truncated": 8}) for i in range(10)]
    _write(tmp_path, "rambles", capped, "probe done: 10 prompts in 90 s -> x\n")
    assert compare.check(tmp_path, "rambles") == 1


def test_pace_sums_every_finished_launch(tmp_path):
    log = tmp_path / "p.log"
    log.write_text("probe done: 100 prompts in 250 s -> a\r  5 / 10\nprobe done: 900 prompts in 2000 s -> a\n")
    assert compare.pace(log) == (1000, 2250)
