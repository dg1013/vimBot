"""CPU tests for scripts/grpo_opd (GRPO on OPD's adapters): `pytest scripts/grpo_opd`."""
import importlib.util
import json
import pathlib
import re
import subprocess

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL = (REPO / "data" / "splits" / "train.jsonl").exists()
local_only = pytest.mark.skipif(not LOCAL, reason="the splits are local-only")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FUEL = load("grpo_opd_fuel_t", HERE / "fuel.py")
DRIVER = (HERE / "run.sh").read_text(encoding="utf-8")


def function(name, src=DRIVER):
    """One function of run.sh, as text."""
    return re.search(r"^%s\(\) \{.*?^\}$" % name, src, re.S | re.M).group(0)


def test_fuel_by_hand():
    rows = [{"k": k, "G": 8} for k in (0, 2, 4, 8)]        # 0, 1.5, 2, 0
    s = FUEL.summarize(rows)
    assert s["fuel"] == pytest.approx(0.875) and s["mean_p_pct"] == pytest.approx(43.8, abs=0.05)
    assert s["shelves_pct"] == {"cliff": 25.0, "frontier": 50.0, "known": 25.0} and s["go"]


def test_the_gate_stops_under_the_floor_and_not_at_it(tmp_path):
    probe, out = tmp_path / "probe.jsonl", tmp_path / "fuel.json"

    def gate(ks):
        probe.write_text("".join(json.dumps({"k": k, "G": 8}) + "\n" for k in ks))
        return FUEL.main([str(probe), "--out", str(out)])
    assert gate([7, 7, 8, 8, 8]) == 0                         # 1.75 / 5 is the floor itself
    assert gate([7, 8, 8]) == FUEL.FUEL_LOW_RC                # 0.875 / 3
    assert not json.loads(out.read_text())["go"]


def test_the_driver_runs_the_preregistered_commands():
    assert "scripts/probe_difficulty.py --model $MODEL --adapter \"$adapter\"" in DRIVER
    assert '--out "$probe" --seed "$s"' in DRIVER
    round_fn = function("grpo_round")
    assert "scripts/train_grpo_gated.py --model $MODEL --adapter \"$adapter\"" in round_fn
    assert '--difficulty "$probe"' in round_fn and '--seed "$s"' in round_fn and "--wandb" in round_fn
    for flag in ("--lr", "--num-generations", "--prompts-per-step", "--epochs", "--anchor-frac", "--cliff-frac", "--pool"):
        assert flag not in round_fn, flag                     # the trainer's defaults are the recipe
    assert "chosen.json" in DRIVER and "['adapter']" in DRIVER   # the starts are OPD's recorded choices
    assert 'STEPS:-"check s42 s7"' in DRIVER


def test_the_driver_never_uses_a_bare_wait():
    assert not [ln for ln in DRIVER.splitlines() if re.match(r"\s*wait\s*(#.*)?$", ln)]


def harness(tmp_path, *names):
    """The named functions of run.sh, with the card and the clock stubbed."""
    fns = "\n".join(function(n).replace("sleep 60", "sleep 0.05") for n in names)
    return "\n".join(["set -uo pipefail", "cd %s" % tmp_path, "RUN=run", "mkdir -p run", "STALL_MIN=30", "MAX_ATTEMPTS=3",
                      "PAUSED_RC=3", "MODEL=m", "PY=true", "status() { echo \"$*\" >> status.txt; }",
                      "wait_idle() { :; }", "arm_of() { echo arm; }", "pct() { echo 50; }", fns])


def test_pause_reaches_the_stop_file_and_the_stage_ends_with_its_rc(tmp_path):
    fake = ("while [ ! -f stop ]; do sleep 0.02; done; exit 3")
    script = harness(tmp_path, "watched") + "\ntouch run/PAUSE\nwatched log.txt stop bash -c '%s'\necho $? > rc\n" % fake
    subprocess.run(["bash", "-c", script], timeout=20, check=True)
    assert (tmp_path / "rc").read_text() == "3\n"


def test_a_paused_round_stops_and_a_failed_one_resumes(tmp_path):
    script = harness(tmp_path, "grpo_round") + """
mkdir -p adapter arm/checkpoint-50; touch adapter/adapter_config.json
watched() { echo "$*" >> calls.txt; return ${NEXT_RC}; }
NEXT_RC=3 grpo_round 42 adapter probe; echo $? > paused_rc
wc -l < calls.txt > paused_calls
rm calls.txt
NEXT_RC=1 grpo_round 42 adapter probe; echo $? > failed_rc
"""
    subprocess.run(["bash", "-c", script], timeout=20, check=True)
    assert (tmp_path / "paused_rc").read_text() == "3\n" and (tmp_path / "paused_calls").read_text().strip() == "1"
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    assert (tmp_path / "failed_rc").read_text() == "1\n" and len(calls) == 3 and all("--resume" in c for c in calls)


def test_no_model_ids_or_thinking_in_code():
    for f in ("fuel.py", "report.py", "run.sh", "test_grpo_opd.py"):
        text = (HERE / f).read_text(encoding="utf-8")
        assert "Qwen" + "/" not in text, f
        assert "enable_thinking" + "=True" not in text and "enable_thinking" + ": true" not in text, f


@local_only
def test_the_report_verdicts_wait_for_both_arms():
    pytest.importorskip("torch")
    R = load("grpo_opd_report_t", HERE / "report.py")
    rep = R.build()
    both = all("grpo-s%d" % s in rep["arms"] for s in R.SEEDS)
    assert rep["n_untouched"] == 295 and rep["n"] == 545
    assert (rep["verdict"]["G"] in ("helps", "tied", "hurts")) if both else (rep["verdict"]["G"] is None)
    assert "# GRPO on OPD's adapters" in R.render(rep)


def test_meaning_by_verdict():
    pytest.importorskip("torch")
    R = load("grpo_opd_report_m", HERE / "report.py")
    assert R.meaning(None, None) is None
    assert "candidate" in R.meaning("helps", "helps")
    assert "up to the plateau" in R.meaning("helps", "tied") and "up to the plateau" in R.meaning("helps", "hurts")
    assert "no more RL" in R.meaning("tied", "tied") and "no more RL" in R.meaning("hurts", "helps")
