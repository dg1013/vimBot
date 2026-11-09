"""Tests for the trained 27B's systems on the 4B (CPU, no endpoints): `pytest scripts/sys4b`."""
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import sys4b as S                 # noqa: E402

T, Q, A = S.T, S.Q, S.A
LOCAL = (REPO / "data" / "splits" / "val.jsonl").exists() and T.PLANS["test"].exists()


@pytest.fixture(autouse=True)
def restore():
    """use_4b rebinds module names; put the 27B's back after each test."""
    names = [(A, "MODEL27"), (A, "NAME27"), (A, "URL27"), (A, "URL4"), (A, "endpoint27"), (Q, "SERVE27"), (Q, "LABEL"),
             (T, "BASE_NAME"), (T, "SERVED"), (T, "RECORDED"), (T, "RECORDED_ZERO"), (T, "OTHER_HOURS"), (T, "ARM")]
    saved = [(m, n, getattr(m, n)) for m, n in names]
    yield
    for m, n, v in saved:
        setattr(m, n, v)


class Fake:
    def __init__(self, url, name):
        self.url, self.name = url, name

    def render(self, messages, prefill=""):
        return "|".join(m["content"] for m in messages) + prefill


@pytest.fixture
def loads(monkeypatch):
    import endpoint as E
    seen = []
    monkeypatch.setattr(E, "load", lambda model, url, name: seen.append((model, url, name)) or Fake(url, name))
    monkeypatch.setattr(T, "load_plan", lambda: {})
    return seen


# ------------------------------------------------------------ pointing the 27B's stages at the 4B

def test_each_arm_asks_its_own_endpoint_for_its_own_model(loads):
    for arm, url, name, examples in (("a", S.URL_SERVED, S.NAME_SERVED, False), ("b", S.URL_SERVED, S.NAME_SERVED, True),
                                     ("c", S.URL_BASE, S.NAME_BASE, True)):
        S.use_4b(arm)
        e = A.endpoint27()
        assert loads[-1] == ("qwen3.5-4b", url, name)
        assert isinstance(e, T.WithExamples) is examples


def test_arm_a_after_arm_c_in_one_process_has_no_examples(loads):
    S.use_4b("c")
    S.use_4b("a")
    e = A.endpoint27()
    assert not isinstance(e, T.WithExamples) and loads[-1] == ("qwen3.5-4b", S.URL_SERVED, S.NAME_SERVED)


def test_the_verifier_is_the_served_endpoints_lora_for_every_arm():
    for arm in "abc":
        S.use_4b(arm)
        assert A.URL4 == S.URL_SERVED == "http://127.0.0.1:8011"


def test_each_arm_writes_its_turns_under_its_own_label():
    got = set()
    for arm in "abc":
        S.use_4b(arm)
        got.add(Q.LABEL)
    assert len(got) == 3 and not any("27b" in x for x in got)


def test_arms_b_and_c_are_retr27bs_trained_and_base():
    S.use_4b("b")
    assert T.ARM == "trained" and T.SERVED["trained"] == S.NAME_SERVED and T.ARMS["trained"] == ("retr8",)
    S.use_4b("c")
    assert T.ARM == "base" and T.SERVED["base"] == S.NAME_BASE and T.ARMS["base"] == ("retr8", "k0")
    assert T.RECORDED == S.RECORDED_BASE


def test_arm_bs_smoke_reads_arm_as_answers_once_they_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "SERVED_DIR", tmp_path)
    S.use_4b("b")
    assert T.RECORDED_ZERO == S.RECORDED_HF
    (tmp_path / "preds_val.jsonl").write_text("")
    S.use_4b("b")
    assert T.RECORDED_ZERO == tmp_path / "preds_val.jsonl"


def test_arm_a_reads_its_samples_from_the_served_folder():
    S.use_4b("a")
    assert Q.SERVE27 == S.SERVED_DIR == S.ROOT / "a" / "served"


def test_the_stage_tables():
    assert set(S.stages("a")) == {"prep", "greedy", "views", "question", "placebo", "answer", "timing", "judge", "gate"}
    assert S.stages("b") is T.STAGES
    assert S.stages("c") == dict(T.STAGES, smoke=S.cmd_smoke_c)    # addendum 3: only arm c's smoke differs
    assert S.STAGES_A["views"] is Q.cmd_views and S.STAGES_A["gate"] is Q.cmd_gate


def test_arm_cs_longest_prompts_are_judged_by_agreement_and_the_record_is_only_counted():
    x, y = ("ex", "g/x/d"), ("ex", "g/y/d")
    record = {"mode": "ex", "command": "g/x/d"}
    long_rows = S.LongRows()
    assert long_rows([x] * 5 + [y] * 3, record) == 5             # the largest group that agrees
    assert long_rows([y] * 8, record) == 8                         # all agree, none match the record
    assert long_rows([None] * 2 + [y] * 6, record) == 6            # an unparsed answer agrees with nothing it is not
    assert (long_rows.recorded, long_rows.answers) == (5, 24)
    assert S.AGREEING([y] * 8, record) == 0                        # retr27b's own counts against the record
    assert S.AGREEING([y] * 8) == 8                                # and, without one, as LongRows does


def test_arm_cs_smoke_reports_the_records_count_and_puts_retr27bs_agreeing_back(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def smoke(args):                                               # retr27b's smoke, its long rows only
        n = T.agreeing([("ex", "a")] * 8, {"mode": "ex", "command": "b"})
        A.write_json(args.run / "smoke.json", {"longest_identical": n, "longest_answers": 8})
        return 0 if n >= T.PARITY_GATE * 8 else 3

    monkeypatch.setattr(T, "cmd_smoke", smoke)
    assert S.cmd_smoke_c(SimpleNamespace(run=tmp_path)) == 0
    rep = json.loads((tmp_path / "smoke.json").read_text())
    assert (rep["longest_identical"], rep["longest_identical_to_recorded"], rep["longest_recorded_answers"]) == (8, 0, 8)
    assert T.agreeing is S.AGREEING

    def broken(args):
        raise RuntimeError("the endpoint died")

    monkeypatch.setattr(T, "cmd_smoke", broken)
    with pytest.raises(RuntimeError):
        S.cmd_smoke_c(SimpleNamespace(run=tmp_path))
    assert T.agreeing is S.AGREEING


def test_an_arm_refuses_a_stage_it_does_not_have():
    with pytest.raises(SystemExit):
        S.main(["samples", "--arm", "a"])
    with pytest.raises(SystemExit):
        S.main(["prep", "--arm", "b"])


# ------------------------------------------------------------ arm a's files

def old_row(i, samples, cands):
    return {"id": i, "samples": {"4b": samples, "9b": []}, "output_tokens": {"4b": list(range(len(samples)))},
            "prompt_tokens": {"4b": 90}, "cands": cands}


def test_prep_takes_the_first_32_samples_and_the_stored_score_of_each_distinct_one():
    samples = [["ex", "a"]] * 20 + [["ex", "b"]] * 12 + [["ex", "c"]] * 96
    cands = [{"mode": "ex", "command": c, "score": s, "state": "x"} for c, s in (("a", 0.5), ("b", -1.0), ("c", 2.0))]
    rows, scores = S.prep_rows({"val": {"x": old_row("x", samples, cands)}}, {"val": [{"id": "x"}]})
    assert rows[0]["samples"] == samples[:32] and rows[0]["output_tokens"] == list(range(32)) and rows[0]["split"] == "val"
    assert [(s["command"], s["score"]) for s in scores] == [("a", 0.5), ("b", -1.0)]


def test_prep_stops_on_a_missing_score_or_a_short_row():
    cands = [{"mode": "ex", "command": "a", "score": 0.5, "state": "x"}]
    with pytest.raises(SystemExit):
        S.prep_rows({"val": {"x": old_row("x", [["ex", "a"]] * 31 + [["ex", "z"]], cands)}}, {"val": [{"id": "x"}]})
    with pytest.raises(SystemExit):
        S.prep_rows({"val": {"x": old_row("x", [["ex", "a"]] * 31, cands)}}, {"val": [{"id": "x"}]})


def test_an_answer_that_did_not_parse_has_no_prediction_and_order_is_kept():
    rows = [{"id": "p", "samples": [["ex", "x"]]}, {"id": "q", "samples": [None]}, {"id": "r", "samples": [["keys", "dd"]]}]
    assert S.preds_rows(rows) == [{"id": "p", "mode": "ex", "command": "x"}, {"id": "r", "mode": "keys", "command": "dd"}]


def test_a_written_file_replaces_the_old_one_whole(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text("old\n")
    S.write_rows(p, [{"a": 1}, {"a": 2}])
    assert p.read_text() == '{"a": 1}\n{"a": 2}\n' and not (tmp_path / "f.jsonl.tmp").exists()


@pytest.mark.skipif(not (S.SERVED_AT_K / "rows_val.jsonl").exists() or not LOCAL, reason="local-only files")
def test_the_served_pass_covers_every_row_and_every_distinct_sample():
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    rows, scores = S.prep_rows({sp: A.old_rows(sp) for sp in A.SPLITS}, cases)
    assert len(rows) == sum(len(c) for c in cases.values()) and all(len(r["samples"]) == 32 for r in rows)


@pytest.mark.skipif(not LOCAL, reason="local-only files")
def test_the_4bs_prompts_with_examples_have_the_plans_token_counts():
    import random
    from prompt import messages_for_case
    S.use_4b("b")
    e = A.endpoint27()
    plan = {r["id"]: r["prompt_tokens"] for sp in A.SPLITS for r in A.load_jsonl(T.PLANS[sp])}
    rows = [c for sp in A.SPLITS for c in A.cases(sp)]
    random.Random(1).shuffle(rows)
    for c in rows[:60]:
        assert len(e.ep.tok(e.render(messages_for_case(c)), add_special_tokens=False)["input_ids"]) == plan[c["id"]]


# ------------------------------------------------------------ the driver

def driver(body, arm):
    """Sources run.sh with its heavy parts stubbed, runs `body`, returns its stdout's lines."""
    script = """
    set -u
    export RUN=$(mktemp -d)
    export DRY=0 ARM=%s
    source %s
    PY=stubpy
    status() { echo "status $*"; }
    %s
    """ % (arm, HERE / "run.sh", body)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30).stdout.split("\n")


CHAIN = """
step() { echo "step $1 $2" >> "$RUN/calls"; }
stop_ep() { echo "stop $1" >> "$RUN/calls"; }
chain
echo "rc=$?" >> "$RUN/calls"
cat "$RUN/calls"
"""


def calls(arm):
    return [line for line in driver(CHAIN, arm) if line.startswith(("step", "stop", "rc"))]


def test_arm_a_runs_ask27bs_chain_after_its_prep_and_greedy():
    assert calls("a") == ["step prep none", "step greedy 4b", "step views none", "step question 4b", "step placebo 4b", "stop 4b",
                          "step simulate sim", "step answer 4b", "step timing 4b", "stop 4b", "step judge none", "rc=0"]


def test_arm_b_runs_retr27bs_chain_on_the_served_endpoint_and_keeps_it_for_the_verifier():
    assert calls("b") == ["step plan none", "step smoke 4b", "step greedy 4b", "step samples 4b", "step views none",
                          "step question 4b", "step placebo 4b", "stop 4b", "step simulate sim", "step answer 4b",
                          "step timing 4b", "step score v4", "stop 4b", "step judge none", "rc=0"]


def test_arm_c_answers_on_the_base_endpoint_and_scores_on_the_served_one():
    assert calls("c") == ["step plan none", "step smoke base4b", "step greedy base4b", "step samples base4b", "step views none",
                          "step question base4b", "step placebo base4b", "stop base4b", "step simulate sim",
                          "step answer base4b", "step timing base4b", "stop base4b", "step score v4", "stop 4b",
                          "step judge none", "rc=0"]


def test_the_phases_reach_the_right_ports_and_the_simulator_is_never_up():
    out = driver("""
    port_up() { echo "port $1"; return 0; }
    for p in 4b v4 base4b sim none; do is_up $p; echo "$p rc=$?"; done
    """, "a")
    assert [x for x in out if x] == ["port 8011", "4b rc=0", "port 8011", "v4 rc=0", "port 8016", "base4b rc=0", "sim rc=1",
                                     "none rc=0"]


def test_the_samples_go_in_requests_of_eight_with_two_rows_in_flight():
    out = driver("""
    step() { [ "$1" = samples ] && echo "$*"; return 0; }
    stop_ep() { :; }
    chain
    """, "b")
    got = [x for x in out if x.startswith("samples")]
    assert len(got) == 1 and "--n 16 --inflight 2" in got[0] and "--arm b" in got[0]


def test_a_failed_step_stops_the_arm():
    out = driver("""
    step() { [ "$1" = question ] && return 1; echo "step $1"; return 0; }
    stop_ep() { :; }
    chain; echo "rc=$?"
    """, "a")
    assert "rc=1" in out and "step answer" not in out


def test_the_endpoints_start_from_this_folders_script():
    out = driver("""
    bash() { echo "bash $*"; }
    start_ep 4b; start_ep base4b; stop_ep v4; stop_ep base4b
    """, "a")
    assert [x for x in out if x.startswith("bash")] == [
        "bash scripts/sys4b/serve4b.sh start served", "bash scripts/sys4b/serve4b.sh start base",
        "bash scripts/sys4b/serve4b.sh stop served", "bash scripts/sys4b/serve4b.sh stop base"]


def test_an_unknown_arm_is_refused():
    done = subprocess.run(["bash", str(HERE / "run.sh")], capture_output=True, text=True, timeout=30,
                          env={"PATH": "/usr/bin:/bin", "ARM": "d"})
    assert done.returncode == 2


def test_the_serve_script_refuses_an_unknown_endpoint():
    done = subprocess.run(["bash", str(HERE / "serve4b.sh"), "start", "other"], capture_output=True, text=True, timeout=30)
    assert done.returncode == 2


@pytest.mark.skipif(not (REPO / "configs" / "models.yaml").exists(), reason="no registry")
def test_the_base_endpoint_takes_its_repo_and_revision_from_the_registry():
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    r = load_registry("qwen3.5-4b")
    done = subprocess.run(["bash", "-c", "source <(sed -n '/^registry()/,/^}/p' %s); PY=%s; cd %s; registry" % (
        HERE / "serve4b.sh", sys.executable, REPO)], capture_output=True, text=True, timeout=60)
    assert done.stdout.split() == [r["hf_id"], r["revision"]]


# ------------------------------------------------------------ all three arms

def run_all(tmp_path, rcs=None, pause_after=None, r1=None, arms="a b c"):
    """run_all.sh with its arms stubbed: each records itself and exits as told."""
    rcs = rcs or {}
    root = tmp_path / "root"
    root.mkdir(exist_ok=True)
    log = tmp_path / "calls.txt"
    stub = tmp_path / "arm.sh"
    lines = ['#!/bin/bash', 'echo "$ARM $(basename $RUN)" >> %s' % log]
    for arm, rc in rcs.items():
        lines.append('[ "$ARM" = %s ] && exit %d' % (arm, rc))
    if pause_after:
        lines.append('[ "$ARM" = %s ] && touch %s/PAUSE' % (pause_after, root))
    lines.append("exit 0")
    stub.write_text("\n".join(lines) + "\n")
    r1dir = tmp_path / "r1"
    r1dir.mkdir(exist_ok=True)
    (r1dir / "status.txt").write_text("done\n")
    env = {"PATH": "/usr/bin:/bin", "ROOT": str(root), "SELF": str(stub), "R1": str(r1dir), "R1_POLL": "1", "ARMS": arms,
           "R1_DRIVER": r1 or "^no-such-process-anywhere"}
    done = subprocess.run(["bash", str(HERE / "run_all.sh")], capture_output=True, text=True, timeout=60, env=env)
    got = log.read_text().split("\n")[:-1] if log.exists() else []
    return done.returncode, got, root, r1dir


def test_the_three_arms_run_in_turn(tmp_path):
    rc, got, root, _ = run_all(tmp_path)
    assert rc == 0 and got == ["a a", "b b", "c c"]
    assert all((root / arm / "PAUSE").is_symlink() and (root / arm / "done_arm").exists() for arm in "abc")


def test_a_failed_arm_does_not_hold_the_next(tmp_path):
    rc, got, root, _ = run_all(tmp_path, rcs={"a": 1})
    assert got == ["a a", "b b", "c c"] and rc == 0
    assert not (root / "a" / "done_arm").exists() and (root / "b" / "done_arm").exists()


def test_a_pause_holds_the_arms_after_it_and_a_relaunch_resumes_there(tmp_path):
    rc, got, root, _ = run_all(tmp_path, pause_after="b")
    assert rc == 2 and got == ["a a", "b b"]
    (root / "PAUSE").unlink()
    rc, got, _, _ = run_all(tmp_path)
    assert rc == 0 and got == ["a a", "b b", "c c"]       # the log holds both launches: c ran only on the second


def test_the_pause_link_reaches_an_arms_dry_run(tmp_path):
    _, _, root, _ = run_all(tmp_path)
    dry = root / "c" / "dry"
    dry.mkdir()
    (dry / "PAUSE").symlink_to("../PAUSE")
    assert not (dry / "PAUSE").exists()
    (root / "PAUSE").write_text("")
    assert (dry / "PAUSE").exists()


def test_r1_is_paused_first_and_waited_for(tmp_path):
    marker = "sleep 2.7182818"
    proc = subprocess.Popen(marker.split())
    try:
        time.sleep(0.2)
        t0 = time.time()
        rc, got, root, r1dir = run_all(tmp_path, r1="^" + marker.replace(".", "\\."))
        assert (r1dir / "PAUSE").exists() and time.time() - t0 >= 1.5 and got == ["a a", "b b", "c c"]
        assert "R1: paused" in (root / "all.txt").read_text()
    finally:
        proc.kill()


def test_r1_is_paused_once_and_never_on_a_relaunch(tmp_path):
    run_all(tmp_path, arms="a")
    root, r1dir = tmp_path / "root", tmp_path / "r1"
    assert (root / "done_r1_pause").exists() and not (r1dir / "PAUSE").exists()
    marker = "sleep 3.1415926"
    proc = subprocess.Popen(marker.split())
    try:
        time.sleep(0.2)
        run_all(tmp_path, arms="a", r1="^" + marker.replace(".", "\\."))
        assert not (r1dir / "PAUSE").exists()
    finally:
        proc.kill()


# ------------------------------------------------------------ the reports

def test_the_report_reads_the_27b_with_the_27bs_names_and_puts_the_4bs_back():
    import report4b as R
    S.use_4b("b")
    with R.the_27b():
        assert A.MODEL27 == "qwen3.5-27b" and Q.SERVE27.name == "serve27b_2026-11-05" and T.ARM == "base"
    assert A.MODEL27 == S.MODEL and Q.SERVE27 == S.SERVED_DIR and T.ARM == "trained"


def test_a_reading_compares_only_rows_both_systems_have():
    import report4b as R
    parts = {"untouched": ["u1", "u2"], "test": ["t1", "t2", "t3"]}
    got = R.common_parts(parts, {"u1": 1, "u2": 0, "t1": 1, "t3": 0}, {"u1": 0, "t1": 1, "t2": 1, "t3": 1})
    assert got == {"untouched": ["u1"], "test": ["t1", "t3"]}


def test_arm_a_costs_add_the_dialogue_only_on_the_ask_rate():
    import report4b as R
    st = lambda m, p50=0.1, p95=0.2: {"mean": m, "p50": p50, "p95": p95}             # noqa: E731
    k = {"asks": 0.5, "models_wall_s": st(1.0), "question_s": st(0.2), "turn2_s": st(0.3), "first_response_s": st(1.0), "tokens": 10.0}
    timing = {"summary": {"direct": {"total_s": st(0.25), "tokens": 5.0}, "k4": k, "k8": k}}
    got = R.costs_a(timing, {"direct": 50.0, 4: 60.0, 8: 60.0})
    assert got["one answer"]["gpu_s_per_solved"] == 0.5
    assert got["round trip at 8"]["gpu_s_per_request"] == 1.25 and got["round trip at 8"]["asks"] == 0.5


@pytest.mark.skipif(not LOCAL or not (S.REPO / "runs" / "retr27b_2026-11-06" / "trained" / "judged.jsonl").exists(),
                    reason="local-only files")
def test_the_27bs_published_numbers_rebuild_through_the_report():
    import report4b as R
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    z = R.sides_27b(cases)
    unt = list(R.RA.C.untouched_ids(cases["val"]))
    test = [c["id"] for c in cases["test"]]
    got = {k: (R.P.mean(z[k]["direct"], unt), R.P.mean(z[k]["direct"], test), R.P.mean(z[k]["rule"], unt), R.P.mean(z[k]["rule"], test))
           for k in z}
    assert got == {"27b": (60.7, 67.8, 65.2, 70.0), "27b_ex": (63.7, 70.4, 69.7, 74.1), "base27b_ex": (54.2, 57.3, 55.6, 60.8)}
    run = S.REPO / "runs" / "retr27b_2026-11-06"
    for k, path in (("27b", run / "fewer_trained.json"), ("27b_ex", run / "trained" / "fewer_trained-retr.json"),
                    ("base27b_ex", run / "fewer_base.json")):
        fw = json.loads(path.read_text())                         # fewer.py's own output for this 27B
        assert (z[k]["t"], z[k]["tau"]) == (fw["chosen_t"], fw["tau"])
        assert [R.F.pct_of(z[k]["V"], ids) for ids in (unt, test)] == [fw["arms"]["V"]["untouched"], fw["arms"]["V"]["test"]]
        t = "t = %d" % fw["chosen_t"]
        assert [R.F.pct_of(z[k]["T"], ids) for ids in (unt, test)] == [fw["arms"][t]["untouched"], fw["arms"][t]["test"]]


def test_the_check_reads_the_same_folders_as_the_report():
    import check4b as K4
    import report4b as R
    assert K4.ROOT == S.ROOT and K4.SERVED == S.SERVED_DIR and R.S is S
