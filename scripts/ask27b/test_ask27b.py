"""Tests for the 5-of-8 ask rule on the 27B (CPU, no endpoints): `pytest scripts/ask27b`."""
import collections
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ask27b as Q          # noqa: E402
import posthoc_ask as H      # noqa: E402
import report_ask as T      # noqa: E402
from roundtrip import ANSWER_MAX_TOKENS, sample_evidence     # noqa: E402

CASE = {"id": "x", "instruction": "do it", "buffer_before": "a\n", "buffer_after": "b\n", "cursor": [1, 1]}


def entry(buffer, valid=True):
    return {"valid": valid, "timeout": False, "blocked": None, "buffer_after": buffer}


def test_a_sample_that_fails_or_changes_nothing_counts_against_agreement():
    assert Q.result_key(CASE, entry("b\n")) is not None
    assert Q.result_key(CASE, entry("a\n")) is None                         # changed nothing
    assert Q.result_key(CASE, entry("b\n", valid=False)) is None            # vim said no
    assert Q.result_key(CASE, None) is None                                 # never ran


def test_two_commands_that_leave_the_same_buffer_are_one_result():
    by_key = {("ex", "s/a/b/"): entry("b\n"), ("keys", "rb"): entry("b\n"), ("ex", "bad"): entry("x\n", valid=False)}
    keys = Q.sample_results(CASE, [["ex", "s/a/b/"], ["keys", "rb"], ["ex", "bad"], None], by_key)
    assert keys[0] == keys[1] and keys[0] is not None and keys[2] is None and keys[3] is None


def test_agreement_is_what_the_round_trip_counts():
    """roundtrip.sample_evidence is what real use runs: the same samples must give the same top."""
    results = {"a": entry("b\n"), "b": entry("b\n"), "c": entry("c\n"), "d": entry("a\n"), "e": entry("e\n", valid=False)}

    def execute(cases, answers, timeout, workers):
        return {(cases[0]["id"], name): results[a["command"]] for name, a in answers[cases[0]["id"]].items()}

    for commands in (["a"] * 3 + ["b"] * 2 + ["c"] * 3, ["a"] * 6 + ["c"] * 2, ["d"] * 8, ["e"] * 4 + ["c"] * 4,
                     ["a", "b", "c", "c", "d", "e", "a", "c"]):
        evidence = sample_evidence(CASE, ["ex: %s" % c for c in commands], execute, 5.0)
        keys = Q.sample_results(CASE, [["ex", c] for c in commands], {("ex", c): results[c] for c in commands})
        counts = collections.Counter(k for k in keys if k is not None)
        assert evidence["top"] == (max(counts.values()) if counts else 0), commands
        assert T.asks(keys) == (evidence["top"] <= 5), commands


def test_the_rule_asks_at_five_of_eight_and_not_at_six():
    assert T.asks(["a"] * 5 + ["b"] * 3) and T.asks(["a"] * 5 + [None] * 3)
    assert not T.asks(["a"] * 6 + ["b"] * 2) and not T.asks(["a"] * 6 + [None] * 2)
    assert T.asks([None] * 8)


def row(results, direct=0, real=1, dialogue=True):
    return {"results": results, "direct": direct, "real": real, "dialogue": dialogue}


def test_a_request_takes_the_answer_after_the_reply_only_where_it_asks():
    split = ["a"] * 4 + ["b"] * 4
    assert T.value(row(split), range(8)) == (1, 1)
    assert T.value(row(["a"] * 8), range(8)) == (0, 0)
    assert T.value(row(split, dialogue=False), range(8)) == (0, 0)          # no question written: answers directly


def test_one_run_is_the_first_eight_and_draws_repeat():
    rows = {"x": row(["a"] * 8 + ["b"] * 24), "y": row(["a"] * 32)}
    one = T.one_run(rows)
    assert one["x"] == (0, 0) and one["y"] == (0, 0)
    assert T.draws(rows) == T.draws(rows) and T.draws(rows)["y"] == (0.0, 0.0)
    assert T.draws({"x": rows["x"]}, k=32)["x"] == (0.0, 0.0)               # 24 of 32 agree, floor(5/8 k) = 20


def test_draws_ask_at_the_rate_the_samples_imply():
    """8 drawn from 8 + 24 alike: the top holds at most 5 of 8 when 3 to 5 of the 24 come up, 30.8%."""
    rows = {"r%d" % n: row(["a"] * 8 + ["b"] * 24) for n in range(400)}
    d = T.draws(rows)
    rate = sum(v[1] for v in d.values()) / len(d)
    assert 0.27 < rate < 0.35
    assert all(v[0] == v[1] for v in d.values())                            # real = 1, direct = 0: right exactly when it asks


class Stub:
    """An endpoint that records what it was asked."""

    def __init__(self):
        self.calls = []

    def render(self, messages, prefill=""):
        return (messages, prefill)

    def complete(self, text, max_tokens, **kwargs):
        self.calls.append((text, max_tokens, kwargs))
        return {"texts": [" which line?\nmore"], "output_tokens": [4], "prompt_tokens": 9, "latency_s": 0.1}


def test_the_turns_are_the_prompts_real_use_sends():
    e = Stub()
    r = Q.question_row(e, "val", CASE)
    (messages, prefill), cap, kw = e.calls[0]
    assert messages == Q.turn1_messages(CASE) and prefill == Q.ASK_PREFIX and cap == Q.QUESTION_MAX_TOKENS and kw == {"stop": ["\n"]}
    assert r["question"]["text"] == "which line?" and r["label"] == "served27b"
    Q.answer_row(e, "val", CASE, "which line?", "the first")
    (messages, prefill), cap, _ = e.calls[1]
    assert messages == Q.turn2_messages(CASE, "which line?", "the first") and prefill == "" and cap == ANSWER_MAX_TOKENS == Q.ANSWER_MAX_TOKENS


def test_a_second_question_is_a_parse_failure_and_is_counted():
    a = Q.answer_fields("ask: which one?")
    assert not a["parse_ok"] and a["second_ask"] and a["command"] is None
    b = Q.answer_fields("ex: d")
    assert b["parse_ok"] and not b["second_ask"] and (b["mode"], b["command"]) == ("ex", "d")
    assert Q.answer_fields("<think>x</think>ex: d")["think"]


def pass_rows(tmp_path, ids, fell=lambda seconds: False, monkeypatch=None):
    if monkeypatch:
        monkeypatch.setattr(Q.A, "pace_fell", fell)
    def work(sp, c):
        time.sleep(0.02)                                                    # slower than the check that stops the queue
        return {"id": c["id"]}
    return Q.run_rows(tmp_path, "t", {i: ("val", {"id": i}) for i in ids}, work, workers=1)


def test_a_pass_resumes_and_writes_each_row_once(tmp_path):
    assert pass_rows(tmp_path, ["a", "b"]) == 0
    assert pass_rows(tmp_path, ["a", "b", "c"]) == 0
    ids = [json.loads(line)["id"] for line in (tmp_path / "t.jsonl").open()]
    assert sorted(ids) == ["a", "b", "c"]


def test_a_pass_stops_for_a_pause_and_yields_when_the_pace_falls(tmp_path, monkeypatch):
    (tmp_path / "PAUSE").touch()
    assert pass_rows(tmp_path, ["a", "b"]) == 4
    (tmp_path / "PAUSE").unlink()
    assert pass_rows(tmp_path, ["a", "b", "c", "d"], fell=lambda seconds: len(seconds) >= 1, monkeypatch=monkeypatch) == 5


def test_timing_uses_the_rows_the_4b_was_timed_on():
    assert "default=%d" % Q.TIMING_SEED in (HERE.parent / "serve" / "timing.py").read_text()
    rows = list(range(545))
    assert Q.timing_rows(rows, 200) == Q.timing_rows(rows, 200) and len(Q.timing_rows(rows, 200)) == 200


def test_the_gate_needs_the_turns_on_most_rows():
    ok = [{"parse_ok": True}] * 20
    assert Q.dry_fit(24, list("q" * 20), ["a reply"] * 20, ok) is None
    assert "question" in Q.dry_fit(24, list("q" * 10), ["a reply"] * 20, ok)
    assert "reply" in Q.dry_fit(24, list("q" * 20), [" "] * 20, ok)
    assert "parses" in Q.dry_fit(24, list("q" * 20), ["a reply"] * 20, [{"parse_ok": False}] * 24)
    assert Q.dry_fit(0, [], [], []) == "no rows"


def test_outcomes_count_what_the_questions_did():
    rows = {"fix": {"real": 1, "direct": 0}, "break": {"real": 0, "direct": 1}, "same": {"real": 1, "direct": 1}, "left": {"real": 0, "direct": 1}}
    one = {"fix": (1, 1), "break": (0, 1), "same": (1, 1), "left": (1, 0)}
    o = T.question_outcomes(rows, one)
    assert (o["asked"], o["fixed"], o["broke"], o["right_before"], o["right_after"], o["left"], o["right_left"]) == (3, 1, 1, 2, 2, 1, 1)


def timing(asks=0.5):
    part = {"mean": 1.0, "p50": 1.0, "p95": 2.0}
    t = {"asks": asks, "first_response_s": part, "when_it_answers_s": part, "when_it_asks_s": part, "models_wall_s": {"mean": 2.0},
         "sandbox_s": {"p50": 0.01}, "question_s": {"mean": 0.5}, "turn2_s": {"mean": 0.5, "p50": 0.5}, "tokens": 100.0, "same_question": 1.0}
    return {"summary": {"direct": {"total_s": {"mean": 0.5, "p50": 0.4, "p95": 1.0}, "tokens": 14.0}, "k4": t, "k8": t}}


def test_the_cost_of_a_round_trip_adds_its_question_and_answer_on_its_ask_rate():
    c = T.costs(timing(0.5), {4: 50.0, 8: 50.0, "direct": 50.0}, {})
    assert c[8]["gpu_s_per_request"] == 2.5 and c[8]["gpu_s_per_solved"] == 5.0
    assert c["direct"]["gpu_s_per_request"] == 0.5


def driver(body):
    """Sources run.sh with its heavy parts stubbed, runs `body`, returns its stdout's lines."""
    script = """
    set -u
    export RUN=$(mktemp -d)
    export DRY=0
    source %s
    PY=stubpy
    status() { echo "status $*"; }
    %s
    """ % (HERE / "run.sh", body)
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30).stdout.split("\n")


def test_the_chain_swaps_the_card_in_the_right_order():
    out = driver("""
    step() { echo "step $1 $2" >> "$RUN/calls"; }
    stop_ep() { echo "stop $1" >> "$RUN/calls"; }
    chain
    echo "rc=$?" >> "$RUN/calls"
    cat "$RUN/calls"
    """)
    got = [line for line in out if line.startswith(("step", "stop", "rc"))]
    assert got == ["step views none", "step question 27b", "step placebo 27b", "stop 27b", "step simulate sim", "step answer 27b",
                   "step timing 27b", "stop 27b", "step judge none", "step report none", "rc=0"]


def test_the_simulator_counts_as_down_so_the_driver_waits_for_a_card_first():
    out = driver("""
    is_up sim; echo "sim=$?"
    is_up none; echo "none=$?"
    """)
    assert "sim=1" in out and "none=0" in out


def test_a_paused_simulator_is_a_pause_and_a_crash_is_a_crash():
    out = driver("""
    stubpy() { return 3; }; sim_step; echo "paused=$?"
    stubpy() { return 7; }; sim_step; echo "crash=$?"
    stubpy() { return 0; }; sim_step; echo "fine=$?"
    """)
    assert "paused=4" in out and "crash=7" in out and "fine=0" in out


def test_the_real_run_starts_only_after_a_passing_dry_run():
    body = """
    bash() { echo "child DRY=$DRY RUN=${RUN##*/} LIMIT=$LIMIT"; return %d; }
    stubpy() { return %d; }
    dry_first; echo "rc=$?"
    dry_first; echo "again=$?"
    """
    passed = driver(body % (0, 0))
    assert "rc=0" in passed and "again=0" in passed
    assert sum(line.startswith("child") for line in passed) == 1 and "child DRY=0 RUN=dry LIMIT=12" in passed
    assert any("passed earlier" in line for line in passed)
    assert "rc=3" in driver(body % (0, 1))               # the gate failed
    assert "rc=5" in driver(body % (5, 0))               # the dry run itself stopped


def test_a_pause_of_the_real_run_reaches_the_dry_run():
    out = driver("""
    bash() { [ -f "$RUN/PAUSE" ] && echo paused-seen || echo no-pause; return 2; }     # $RUN is the dry folder here
    dry_first
    touch "$RUN/PAUSE"
    dry_first
    """)
    assert out.index("no-pause") < out.index("paused-seen")


def posthoc_row(results, direct=0, real=1, placebo=0, right=None, dialogue=True):
    return {"id": "r", "results": results, "direct": direct, "real": real, "placebo": placebo, "dialogue": dialogue,
            "right": right if right is not None else [0] * len(results)}


def test_the_top_result_is_the_most_common_and_the_earliest_on_a_tie():
    assert H.top_sample(["a", "b", "b", "a", "c"], range(5)) == 0
    assert H.top_sample([None, "b", "a", "a"], range(4)) == 2
    assert H.top_sample(["a", "b", "b"], [0, 1, 2]) == 1
    assert H.top_sample([None, None], range(2)) is None


def test_the_arms_differ_only_where_the_rule_asks():
    agreed = posthoc_row(["a"] * 6 + ["b", "c"], direct=1, real=0, placebo=0)
    assert H.arms(agreed, range(8)) == {"rule": 1, "placebo": 1, "top": 1, "asked": 0}
    split = posthoc_row(["a"] * 3 + ["b"] * 2 + ["c"] * 3, direct=0, real=1, placebo=0, right=[0, 0, 0, 0, 0, 1, 1, 1])
    assert H.arms(split, range(8)) == {"rule": 1, "placebo": 0, "top": 0, "asked": 1}


def test_a_top_result_arm_with_no_result_keeps_the_section_6_answer():
    row = posthoc_row([None] * 8, direct=1, real=0, placebo=0)
    assert H.arms(row, range(8)) == {"rule": 0, "placebo": 0, "top": 1, "asked": 1}


def test_a_row_with_no_question_is_answered_directly_by_every_arm():
    row = posthoc_row(["a", "b", "c", "d", "e", "f", "g", "h"], direct=1, real=0, placebo=0, dialogue=False)
    assert H.arms(row, range(8)) == {"rule": 1, "placebo": 1, "top": 1, "asked": 0}


def test_the_posthoc_arms_sit_on_the_reports_draws():
    results = [("a" if n % 3 else "b") for n in range(32)]
    rows = {"x%d" % n: dict(posthoc_row(list(results), direct=n % 2, real=(n + 1) % 2), id="x%d" % n) for n in range(40)}
    mine, report = H.draws(rows), T.draws(rows)
    assert all(abs(mine[i]["rule"] - report[i][0]) < 1e-12 and abs(mine[i]["asked"] - report[i][1]) < 1e-12 for i in rows)


def test_questions_per_net_right_needs_a_gain():
    rows = {"a": {"direct": 0}, "b": {"direct": 0}, "c": {"direct": 1}}
    got = H.burden(rows, ["a", "b", "c"], {"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1, "b": 1, "c": 1})
    assert got["questions"] == 3.0 and got["net_right"] == 2.0 and got["questions_per_net_right"] == 1.5
    assert H.burden(rows, ["c"], {"c": 1.0}, {"c": 1})["questions_per_net_right"] is None


def test_two_draws_of_a_row_agree_as_their_rates_say():
    timing = {"rows": {"k8": [{"id": "a", "type": "question"}, {"id": "b", "type": "command"}]}}
    got = H.live_vs_offline(timing, {"a": 1.0, "b": 0.5})
    assert got["live_asks"] == 50.0 and got["stored_asks"] == 75.0
    assert got["agreement"] == 75.0 and got["agreement_of_two_draws"] == 75.0


def test_the_cost_comparison_refuses_rows_that_are_not_the_same():
    t = {"rows": {"k8": [{"id": "a"}]}}
    four = {"rows": {"k8": [{"id": "b"}]}}
    try:
        H.same_rows(t, four, {})
    except SystemExit as err:
        assert "same rows" in str(err)
    else:
        raise AssertionError("mixed rows were costed")
