"""Tests for src/evaluate.py's scoring and its execution log.

Pins the 2026-09-05 fix: in predictions mode a case with no prediction is a
failure, never a silent fall-through to the gold command (which would have let
a model that fails to parse on every row score 100%).

Also pins the execution log: scoring is the only place that sees what vim
actually did, so those details have to leave `score()` and reach disk, or
every consumer has to re-run the split through vim to get them back.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluate import score, vim_error_code, write_exec_log  # noqa: E402

EVALUATE = REPO / "src" / "evaluate.py"
CASES = [
    {"id": "a", "category": "substitute", "instruction": "x", "mode": "ex",
     "buffer_before": "foo\nfoo\n", "gold_command": "%s/foo/bar/",
     "buffer_after": "bar\nbar\n"},
    {"id": "b", "category": "line-ops", "instruction": "y", "mode": "ex",
     "buffer_before": "one\ntwo\n", "gold_command": "1d",
     "buffer_after": "two\n"},
]


def test_gold_self_check_scores_everything():
    r = score(CASES)
    assert (r["n"], r["n_valid"], r["n_match"]) == (2, 2, 2)
    assert r["per_cat"] == {"substitute": [1, 1, 1], "line-ops": [1, 1, 1]}
    assert r["failures"] == []


def verdict(entry):
    """The judgement half of a per_case row, without the execution detail."""
    return {k: entry[k] for k in ("command", "mode", "valid", "match", "timeout")}


def test_wrong_prediction_is_valid_but_not_a_match():
    r = score(CASES, {"a": {"command": "%s/foo/baz/"},
                      "b": {"command": "1d"}})
    assert r["n_match"] == 1 and r["n_valid"] == 2
    assert verdict(r["per_case"]["a"]) == {"command": "%s/foo/baz/", "mode": "ex",
                                           "valid": True, "match": False,
                                           "timeout": False}


def test_absent_prediction_is_a_failure_not_the_gold():
    r = score(CASES, {"b": {"command": "1d"}})   # nothing for "a"
    assert r["n_match"] == 1 and r["n_valid"] == 1
    assert r["per_case"]["a"]["command"] is None
    assert r["per_case"]["a"]["match"] is False
    assert r["failures"][0][0] == "a" and r["failures"][0][1] is None


def test_invalid_prediction_is_neither():
    r = score(CASES, {"a": {"command": "%s/unclosed"},
                      "b": {"command": "1d"}})
    assert verdict(r["per_case"]["a"]) == {"command": "%s/unclosed", "mode": "ex",
                                           "valid": False, "match": False,
                                           "timeout": False}


def test_prediction_mode_overrides_the_case_mode():
    r = score(CASES, {"b": {"command": "dd", "mode": "keys"},
                      "a": {"command": "%s/foo/bar/"}})
    assert r["per_case"]["b"]["mode"] == "keys" and r["per_case"]["b"]["match"]


def test_cli_exit_code_follows_exec_match(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text("".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
    ok = subprocess.run([sys.executable, str(EVALUATE), "--cases", str(cases)],
                        capture_output=True, text=True)
    assert ok.returncode == 0 and "exec_match  2/2" in ok.stdout
    preds = tmp_path / "p.jsonl"
    preds.write_text(json.dumps({"id": "b", "command": "1d"}) + "\n", encoding="utf-8")
    bad = subprocess.run([sys.executable, str(EVALUATE), "--cases", str(cases),
                          "--predictions", str(preds)], capture_output=True, text=True)
    assert bad.returncode == 1
    assert "exec_match  1/2" in bad.stdout and "NO PREDICTION" in bad.stdout


# --- execution log --------------------------------------------------------

def test_score_keeps_what_vim_did_not_just_the_verdict():
    """Without this the error-analysis page has to re-run every prediction
    through the sandbox to recover the buffer and the error code."""
    r = score(CASES, {"a": {"command": "%s/foo/baz/"}, "b": {"command": "1d"}})
    a = r["per_case"]["a"]
    assert a["buffer_after"] is not None
    assert a["buffer_after"].splitlines() == ["baz", "baz"]
    assert a["return_code"] == 0 and a["error"] == "" and a["stderr"] == ""


def test_a_vim_error_reaches_the_row_as_a_code():
    r = score(CASES, {"a": {"command": "%s/nomatch/x/"}, "b": {"command": "1d"}})
    a = r["per_case"]["a"]
    assert a["valid"] is False and a["error"] == "E486"
    assert "E486" in a["stderr"]


def test_a_screened_command_says_so_instead_of_looking_like_a_vim_error():
    r = score(CASES, {"a": {"command": "w! /tmp/vimbot-should-not-exist"}})
    a = r["per_case"]["a"]
    assert a["blocked"] and a["valid"] is False and a["return_code"] is None


def test_a_nul_in_one_prediction_does_not_abort_the_others():
    # 2026-09-22: an ex command holding a NUL made run_vim raise (argv cannot
    # carry one), and the whole score() call died with it. It is now one
    # invalid row among scored ones -- and never a match.
    r = score(CASES, {"a": {"command": "%s/foo/bar/\x00"}, "b": {"command": "1d"}})
    a = r["per_case"]["a"]
    assert a["valid"] is False and a["match"] is False and "NUL" in a["stderr"]
    assert r["n"] == 2 and r["n_match"] == 1 and r["per_case"]["b"]["match"] is True


def test_cli_scores_a_predictions_file_holding_a_nul(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text("".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
    preds = tmp_path / "p.jsonl"
    preds.write_text(json.dumps({"id": "a", "command": "%s/foo/bar/\x00"}) + "\n"
                     + json.dumps({"id": "b", "command": "1d"}) + "\n", encoding="utf-8")
    p = subprocess.run([sys.executable, str(EVALUATE), "--cases", str(cases),
                        "--predictions", str(preds)], capture_output=True, text=True)
    assert "Traceback" not in p.stderr
    assert p.returncode == 1 and "exec_match  1/2" in p.stdout


def test_vim_error_code_reads_the_code_not_the_message():
    assert vim_error_code("E486: Pattern not found: nomatch") == "E486"
    assert vim_error_code("E486: Pattern not found", timed_out=True) == "TIMEOUT"
    assert vim_error_code("") == "" and vim_error_code(None) == ""
    assert vim_error_code("Error detected while processing") == ""


def test_exec_log_is_one_row_per_case_in_case_order(tmp_path):
    r = score(CASES, {"a": {"command": "%s/foo/baz/"}, "b": {"command": "1d"}})
    out = tmp_path / "exec_test.jsonl"
    assert write_exec_log(str(out), CASES, r) == 2
    rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert [x["id"] for x in rows] == [c["id"] for c in CASES]
    # category rides along so the log slices by cell without a join
    assert [x["category"] for x in rows] == ["substitute", "line-ops"]
    assert rows[0]["match"] is False and rows[1]["match"] is True


def test_exec_log_records_the_missing_prediction_too(tmp_path):
    """A row the model never answered is a failure with a reason, not a gap."""
    r = score(CASES, {"b": {"command": "1d"}})
    out = tmp_path / "exec.jsonl"
    assert write_exec_log(str(out), CASES, r) == 2
    rows = {json.loads(x)["id"]: json.loads(x)
            for x in out.read_text(encoding="utf-8").splitlines()}
    assert rows["a"]["command"] is None and rows["a"]["buffer_after"] is None
    assert rows["a"]["stderr"] == "no prediction"


def test_cli_writes_the_exec_log_beside_the_scores(tmp_path):
    cases = tmp_path / "c.jsonl"
    cases.write_text("".join(json.dumps(c) + "\n" for c in CASES), encoding="utf-8")
    preds = tmp_path / "p.jsonl"
    preds.write_text(json.dumps({"id": "a", "command": "%s/foo/bar/"}) + "\n"
                     + json.dumps({"id": "b", "command": "1d"}) + "\n", encoding="utf-8")
    log = tmp_path / "exec.jsonl"
    p = subprocess.run([sys.executable, str(EVALUATE), "--cases", str(cases),
                        "--predictions", str(preds), "--exec-log", str(log)],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "exec_match  2/2" in p.stdout          # the log stays off stdout
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(x["match"] for x in rows)
