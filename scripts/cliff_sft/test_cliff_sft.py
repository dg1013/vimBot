"""Tests for scripts/cliff_sft/ (SFT round 2 on the cliff, DECISIONS.md
2026-09-15): the guard that train_sft_cliff.py has not drifted from
scripts/train_sft.py in anything the round did not mean to change, the pause
callback, and the set builder's shelving, replay, hold-out and report.

Run:  python3 -m pytest scripts/cliff_sft            (not collected by `pytest tests/`)
"""

import inspect
import json
import signal
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (REPO / "scripts", REPO / "src", HERE):
    sys.path.insert(0, str(p))

import train_sft                               # noqa: E402
import train_sft_cliff as cliff                # noqa: E402
import train_grpo                              # noqa: E402
import build_cliff_set as bcs                  # noqa: E402
from difficulty import DifficultyBook          # noqa: E402

ORIGINAL = REPO / "scripts" / "train_sft.py"
COPY = HERE / "train_sft_cliff.py"


# ------------------------------------------------------------ drift guard

def _unmarked(text):
    """The copy's lines minus every [cliff] addition: lines carrying the
    marker and lines inside a '# [cliff] begin' ... '# [cliff] end' block."""
    out, inside = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# [cliff] begin"):
            inside = True
            continue
        if s.startswith("# [cliff] end"):
            inside = False
            continue
        if inside or "[cliff]" in line:
            continue
        out.append(line)
    return out


def test_every_addition_is_marked_and_the_rest_is_the_original():
    orig = ORIGINAL.read_text(encoding="utf-8").splitlines()
    copy = _unmarked(COPY.read_text(encoding="utf-8"))
    # Lines of the original that the copy REPLACED (each replacement carries
    # the marker, so it was stripped above). This list is the whole diff.
    replaced = {
        "REPO = pathlib.Path(__file__).resolve().parents[1]",
        "                         processing_class=tok, peft_config=lora)",
        '        run_name=args.run_name, condition="sft", wandb_state=wandb_state,',
    }
    missing = [l for l in orig if l not in set(copy)]
    assert set(missing) == replaced, missing
    extra = [l for l in copy if l not in set(orig)]
    assert extra == [], extra


def test_the_copy_declares_itself():
    src = COPY.read_text(encoding="utf-8")
    assert "A copy of\n`scripts/train_sft.py`" in src
    assert "The original's docstring follows unchanged." in src
    assert src.count("[cliff]") >= 14
    assert "Qwen/" not in src and "hf_id: " not in src and 'reg["hf_id"]' in src
    assert "enable_thinking\": True" not in src and "enable_thinking=True" not in src


def test_shared_functions_are_the_originals():
    for name in ("build_dataset", "stratified_subset", "inference_mode", "make_callbacks",
                 "resolved_targets", "stack_versions", "full_val_eval", "show_masked_batch"):
        assert inspect.getsource(getattr(cliff, name)) == inspect.getsource(getattr(train_sft, name)), name


def test_shared_cli_defaults_are_the_originals():
    orig = {a.dest: a.default for a in train_sft.build_arg_parser()._actions}
    copy = {a.dest: a.default for a in cliff.build_arg_parser()._actions}
    assert set(orig) <= set(copy)
    for dest, default in orig.items():
        assert copy[dest] == default, dest
    assert set(copy) - set(orig) == {"adapter", "base_dtype"}


def test_restore_and_pause_constants_match_the_grpo_trainer():
    assert inspect.getsource(cliff.restore_adapter_weights) == inspect.getsource(train_grpo.restore_adapter_weights)
    for name in ("STOP_FILE", "PAUSED_MARKER", "PAUSED_RC"):
        assert getattr(cliff, name) == getattr(train_grpo, name), name


def test_help_needs_no_torch_and_names_the_adapter_flag():
    out = subprocess.run([sys.executable, str(COPY), "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0
    assert "--adapter" in out.stdout and "--base-dtype" in out.stdout


# ---------------------------------------------------------- pause callback

class _Control:
    should_training_stop = False
    should_save = False


class _State:
    def __init__(self, step):
        self.global_step = step


def test_stop_file_stops_after_the_step_without_a_checkpoint(tmp_path):
    seen = []
    cb = cliff.make_pause_callback(object, tmp_path, on_pause=lambda: seen.append("paused"))
    control = _Control()
    cb.on_step_end(None, _State(3), control)
    assert control.should_training_stop is False and cb.paused_at is None
    (tmp_path / cliff.STOP_FILE).write_text("")
    cb.on_step_end(None, _State(4), control)
    assert control.should_training_stop is True and control.should_save is False
    assert cb.paused_at == 4 and cb.requested == "STOP file" and seen == ["paused"]
    assert not (tmp_path / cliff.STOP_FILE).exists()          # consumed
    cb.on_step_end(None, _State(5), control)                   # idempotent
    assert cb.paused_at == 4 and seen == ["paused"]


def test_one_signal_pauses_and_a_second_aborts(tmp_path):
    cb = cliff.make_pause_callback(object, tmp_path)
    cb.install_signal_handlers()
    try:
        handler = signal.getsignal(signal.SIGTERM)
        handler(signal.SIGTERM, None)
        assert cb.requested == "SIGTERM"
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)
    finally:
        for sig, prev in cb._previous.items():
            signal.signal(sig, prev)


# --------------------------------------------------------------- builder

def _cases(n=12):
    cats = ["substitute", "global", "join"]
    return [{"id": "c%02d" % i, "category": cats[i % 3], "tier": "T%d" % (i % 3 + 1),
             "mode": "ex" if i % 2 else "keys", "instruction": "do %d" % i,
             "buffer_before": "a\n", "gold_command": "%%s/a/b%d/" % i, "buffer_after": "b\n"}
            for i in range(n)]


def _difficulty(tmp_path, name, rows):
    p = tmp_path / name
    with p.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return str(p)


def _obs(cid, k, g=8, **extra):
    row = {"id": cid, "k": k, "G": g, "step": 0, "source": "probe"}
    row.update(extra)
    return row


def test_build_rows_teaches_the_cliff_and_replays_a_slice_of_the_solved(tmp_path):
    cases = _cases()
    book = DifficultyBook([c["id"] for c in cases], pool="last:2")
    # c00-c03 cliff, c04-c07 frontier, c08-c11 known
    book.load(_difficulty(tmp_path, "probe.jsonl",
                          [_obs(c["id"], 0 if i < 4 else (3 if i < 8 else 8)) for i, c in enumerate(cases)]))
    cliff_rows, replay = bcs.build_rows(cases, book, replay_frac=0.5, seed=1)
    assert [r["id"] for r in cliff_rows] == ["c00", "c01", "c02", "c03"]
    assert all(r["shelf"] == "cliff" and r["k_hat"] == 0 and r["g_hat"] == 8 for r in cliff_rows)
    assert len(replay) == 2 and all(r["shelf"] == "replay" and r["k_hat"] > 0 for r in replay)
    assert {r["id"] for r in replay} <= {"c%02d" % i for i in range(4, 12)}
    # the gold fields are untouched: the trainer reads a §5 row
    assert cliff_rows[0]["gold_command"] == "%s/a/b0/"
    # deterministic in the seed
    assert bcs.build_rows(cases, book, 0.5, seed=1)[1] == replay
    assert bcs.build_rows(cases, book, 0.0, seed=1)[1] == []


def test_a_later_probe_wins_under_the_pool_so_a_reprobe_moves_a_prompt_off_the_cliff(tmp_path):
    cases = _cases(4)
    book = DifficultyBook([c["id"] for c in cases], pool="last:2")
    book.load(_difficulty(tmp_path, "probe.jsonl", [_obs(c["id"], 0) for c in cases]))
    book.load(_difficulty(tmp_path, "before.jsonl", [_obs("c00", 2), _obs("c01", 0)]))
    cliff_rows, _ = bcs.build_rows(cases, book, 0.0, seed=1)
    assert [r["id"] for r in cliff_rows] == ["c01", "c02", "c03"]          # c00 re-probed at 2 of 8


def test_consistent_wrong_needs_one_valid_command_in_six_of_eight():
    row = {"G": 8, "commands": [{"mode": "keys", "command": "x", "state": "valid_wrong", "n": 6},
                                {"mode": "keys", "command": "dw", "state": "valid_wrong", "n": 2}]}
    assert bcs.consistent_wrong(row)["command"] == "x"
    row["commands"][0]["n"], row["commands"][1]["n"] = 5, 3
    assert bcs.consistent_wrong(row) is None                       # a mixture
    row["commands"] = [{"mode": "ex", "command": "%s/x", "state": "invalid", "n": 8}]
    assert bcs.consistent_wrong(row) is None                       # rejected by vim: not a stable answer


def test_build_cli_holds_flagged_rows_out_and_writes_the_audit(tmp_path):
    cases = _cases(6)
    cases_path = tmp_path / "cases.jsonl"
    with cases_path.open("w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")
    probe = _difficulty(tmp_path, "probe.jsonl", [
        _obs("c00", 0, commands=[{"mode": "keys", "command": "x", "state": "valid_wrong", "n": 8}]),
        _obs("c01", 0, commands=[{"mode": "keys", "command": "x", "state": "valid_wrong", "n": 3},
                                 {"mode": "keys", "command": "dw", "state": "invalid", "n": 5}]),
        _obs("c02", 5), _obs("c03", 8), _obs("c04", 8), _obs("c05", 8)])
    out, audit, held = tmp_path / "train.jsonl", tmp_path / "audit.md", tmp_path / "held.jsonl"
    cmd = [sys.executable, str(HERE / "build_cliff_set.py"), "build", "--cases", str(cases_path),
           "--difficulty", probe, "--out", str(out), "--audit", str(audit), "--replay-frac", "0.5",
           "--hold-flagged", "--held-out", str(held)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert {x["id"] for x in rows if x["shelf"] == "cliff"} == {"c01"}       # c00 held out
    assert len([x for x in rows if x["shelf"] == "replay"]) == 1              # round(0.5 * 2 cliff)
    assert [json.loads(l)["id"] for l in held.read_text().splitlines()] == ["c00"]
    text = audit.read_text()
    assert "Held out pending a label review" in text and "| c00 |" in text and "keys: x" in text
    assert "**Train file: 2 rows = 1 cliff + 1 replay.**" in text


def test_report_pairs_before_and_after_by_id(tmp_path):
    cases = _cases(8)
    cases_path = tmp_path / "cases.jsonl"
    with cases_path.open("w") as fh:
        for c in cases:
            fh.write(json.dumps(c) + "\n")
    cliff_ids, known_ids = ["c00", "c01", "c02"], ["c05", "c06", "c07"]
    before = _difficulty(tmp_path, "before.jsonl",
                         [_obs("c00", 0), _obs("c01", 0), _obs("c02", 0), _obs("c05", 8), _obs("c06", 8), _obs("c07", 8)])
    after = _difficulty(tmp_path, "after.jsonl",
                        [_obs("c00", 3), _obs("c01", 8), _obs("c02", 0), _obs("c05", 8), _obs("c06", 7), _obs("c07", 0)])
    (tmp_path / "cliff.txt").write_text("".join(i + "\n" for i in cliff_ids))
    (tmp_path / "known.txt").write_text("".join(i + "\n" for i in known_ids))
    run_dir = tmp_path / "run"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "sft_val.json").write_text(json.dumps({"metrics": {"exec_match_pct": 50.0, "valid_pct": 90.0,
                                                                  "output_tokens_mean": 16.0}}))
    (run_dir / "eval_history.jsonl").write_text(json.dumps({"step": 0, "exec_match_pct": 54.0}) + "\n"
                                                + json.dumps({"step": 25, "exec_match_pct": 52.0}) + "\n")
    (run_dir / "best" / "selection.json").write_text(json.dumps({"step": 0}))
    out = tmp_path / "report.md"
    cmd = [sys.executable, str(HERE / "build_cliff_set.py"), "report", "--cases", str(cases_path),
           "--before", before, "--after", after, "--cliff-ids", str(tmp_path / "cliff.txt"),
           "--known-ids", str(tmp_path / "known.txt"), "--run-dir", str(run_dir), "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert "**2 of 3 cliff prompts (66.7%) now have at least one success**: 1 on the frontier, 1 solved 8 of 8; 1 still on the cliff." in text
    assert "**1 of 3 still solved 8 of 8** (33.3%); 1 slipped to the frontier, 1 to the cliff." in text
    assert "50.0% (valid 90.0%, 16.0 tok)" in text
    assert "**Selection kept step 0**" in text


def test_conversion_and_retention_arithmetic():
    pairs = [("a", 0, 0, 8), ("b", 0, 4, 8), ("c", 0, 8, 8), ("d", 2, 8, 8)]
    assert bcs.conversion(pairs) == {"still_cliff": 1, "now_frontier": 1, "now_known": 1, "not_on_cliff_before": 1}
    ret = bcs.retention([("a", 8, 8, 8), ("b", 8, 6, 8), ("c", 8, 0, 8), ("d", 7, 8, 8)])
    assert (ret["still_known"], ret["now_frontier"], ret["now_cliff"], ret["not_known_before"]) == (1, 1, 1, 1)
    assert ret["p_hat_before"] == pytest.approx((8 + 8 + 8 + 7) / 32)
    assert bcs.mean_fuel([("a", 0, 4, 8)], "after") == pytest.approx(2.0)
