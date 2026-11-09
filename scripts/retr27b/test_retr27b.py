"""Tests for the base 27B with BM25 examples and for asking less (CPU, no endpoints): `pytest scripts/retr27b`."""
import json
import math
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import fewer as F                 # noqa: E402
import report_retr as B           # noqa: E402
import retr27b as G               # noqa: E402

Q, A, T, P = G.Q, G.A, F.T, F.P
from prompt import SYSTEM_TEXT, messages_for_case         # noqa: E402
from protocol import OFFER_SYSTEM, turn1_messages, turn2_messages   # noqa: E402
from roundtrip import RoundTrip                            # noqa: E402

CASE = {"id": "x", "instruction": "do it", "buffer_before": "a\n", "buffer_after": "b\n", "cursor": [1, 1], "category": "c"}
EXAMPLE = {"instruction": "EXAMPLE-ASK", "buffer_before": "e\n", "cursor": [1, 1], "mode": "ex", "gold_command": "s/e/f/"}
LOCAL = (REPO / "data" / "splits" / "val.jsonl").exists() and G.PLANS["test"].exists()


# ------------------------------------------------------------ the examples in every turn

def test_the_examples_go_where_section_6_puts_them():
    assert G.insert_examples(messages_for_case(CASE), [EXAMPLE]) == messages_for_case(CASE, [EXAMPLE])


def test_no_examples_change_nothing():
    for m in (messages_for_case(CASE), turn1_messages(CASE), turn2_messages(CASE, "which?", "that one")):
        assert G.insert_examples(m, []) == m


def test_each_turn_keeps_its_own_system_text_and_order():
    pair = messages_for_case(CASE, [EXAMPLE])[1:3]
    first = G.insert_examples(turn1_messages(CASE), [EXAMPLE])
    assert first[0]["content"] == OFFER_SYSTEM and first[1:3] == pair and first[3:] == turn1_messages(CASE)[1:]
    second = G.insert_examples(turn2_messages(CASE, "which?", "that one"), [EXAMPLE])
    assert second[1:3] == pair and [m["role"] for m in second[3:]] == ["user", "assistant", "user"]
    assert G.insert_examples(messages_for_case(CASE), [EXAMPLE])[0]["content"] == SYSTEM_TEXT


class Fake:
    """An endpoint that records what it was asked to send."""

    def __init__(self, sampled=("ex: a",)):
        self.sent, self.sampled, self.timeout = [], sampled, 0

    def render(self, messages, prefill=""):
        return json.dumps(messages) + prefill

    def complete(self, text, max_tokens, temperature=0.0, n=1, stop=None, seed=None):
        self.sent.append(text)
        texts = list(self.sampled)[:n] if temperature else ["ex: a"]
        return {"texts": texts, "output_tokens": [3] * len(texts), "prompt_tokens": 10, "latency_s": 0.01}


def test_a_row_keeps_its_examples_through_the_dialogue():
    fake = Fake()
    e = G.WithExamples(fake, {G.user_turn(CASE): [EXAMPLE]})
    Q.question_row(e, "val", CASE)
    Q.answer_row(e, "val", CASE, "which?", "that one")
    A.request(e, CASE, 1, 0.0)
    assert len(fake.sent) == 3 and all("EXAMPLE-ASK" in s for s in fake.sent)


def test_an_unknown_row_raises_and_never_goes_out_zero_shot():
    e = G.WithExamples(Fake(), {})
    with pytest.raises(KeyError):
        e.render(messages_for_case(CASE))


def test_the_round_trip_runs_unchanged_through_it():
    fake = Fake(sampled=("ex: a",) * 4 + ("ex: b",) * 4)
    rt = RoundTrip(G.WithExamples(fake, {G.user_turn(CASE): [EXAMPLE]}), samples=8, max_agree=5, timeout=1.0)

    def execute(cases, answers, timeout, workers):
        cid = cases[0]["id"]
        return {(cid, name): {"valid": True, "timeout": False, "blocked": None, "buffer_after": a["command"] + "\n"}
                for name, a in answers[cid].items()}

    rt.execute = execute
    out = rt.translate(CASE["instruction"], CASE["buffer_before"], CASE["cursor"])
    assert out["type"] == "question"
    done = rt.reply(out["dialogue_id"], "that one")
    assert done["type"] == "command" and len(fake.sent) == 4 and all("EXAMPLE-ASK" in s for s in fake.sent)


@pytest.mark.parametrize("order", ["pass", "import measure; from prompt import messages_for_case; from roundtrip import RoundTrip; from timing import stats"])
def test_each_stage_that_executes_gets_the_clarify_scorer_not_the_rerankers(order):
    """scripts/reranker has a score.py too, and importing pooled puts it first, so this needs each stage's own order in a
    fresh process: views and judge (nothing first) and timing (RoundTrip after measure and timing). Only score imports pooled."""
    code = "import sys; sys.path.insert(0, %r); import retr27b; %s; import score; print(score.__file__, hasattr(score, 'execute'))" % (str(HERE), order)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert "/clarify/score.py True" in out.stdout


def test_the_score_stage_keeps_pooled_to_itself():
    src = (HERE / "retr27b.py").read_text(encoding="utf-8")
    assert src.count("import pooled") == 1 and "import pooled" in src[src.index("def cmd_score"):src.index("def greedy_preds")]


def test_turn2_fits_the_endpoint_on_the_longest_prompt():
    longest, ask_clause, turn, cap = 1851, 60, 148, Q.ANSWER_MAX_TOKENS       # 1851: the longest plan; 148: the trained run's longest turn
    limit = int(next(line.split(":-")[1].rstrip("}") for line in (HERE / "serve.sh").read_text().split("\n") if line.startswith("MAXLEN=")))
    assert longest + ask_clause + turn + cap <= limit


# ------------------------------------------------------------ the stages

def test_samples_come_in_requests_of_eight_and_stay_in_order(monkeypatch):
    calls = []

    def request(e, case, n, temperature):
        calls.append((n, temperature))
        base = len(calls) * 100
        return {"samples": [["ex", str(base + s)] for s in range(n)], "output_tokens": [1] * n, "prompt_tokens": 7}

    monkeypatch.setattr(A, "request", request)
    row = G.sample_row(None, "val", CASE, 16)
    assert calls == [(8, 1.0), (8, 1.0)] and len(row["samples"]) == 16 and row["samples"][8] == ["ex", "200"] and row["prompt_tokens"] == 7
    calls.clear()
    assert len(G.sample_row(None, "val", CASE, 11)["samples"]) == 11 and calls == [(8, 1.0), (3, 1.0)]


def test_a_budget_stops_the_samples_and_goes_on_with_what_it_has(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "endpoint27", lambda: None)
    monkeypatch.setattr(A, "cases", lambda sp, limit=0: [dict(CASE, id="%s%d" % (sp, n)) for n in range(2)])
    args = type("Args", (), {"run": tmp_path, "n": 2, "limit": 0, "budget_s": -1, "inflight": 1})
    assert G.cmd_samples(args) == 0


def test_a_sample_file_of_another_size_stops_the_stage(tmp_path, monkeypatch):
    (tmp_path / "samples.jsonl").write_text(json.dumps({"id": "a", "samples": [None]}) + "\n")
    monkeypatch.setattr(A, "endpoint27", lambda: None)
    with pytest.raises(SystemExit):
        G.cmd_samples(type("Args", (), {"run": tmp_path, "n": 2, "limit": 0, "budget_s": 9, "inflight": 1}))


def test_an_answer_that_did_not_parse_has_no_prediction():
    got = G.greedy_preds([{"id": "a", "samples": [["ex", "x"]]}, {"id": "b", "samples": [None]}])
    assert got == {"a": {"command": "x", "mode": "ex"}}


def test_the_pace_projects_by_prompt_tokens_not_by_the_light_head_rows():
    tokens = {"a": 100, "b": 100, "c": 800}
    hours = G.projected_hours([{"id": "a", "row_s": 360.0}, {"id": "b", "row_s": 360.0}], tokens)
    assert hours == pytest.approx(1.0 + G.OTHER_HOURS)            # 720 s over 200 tokens, 1,000 tokens in all: 1 h


def test_the_gate_stops_a_run_that_would_pass_twelve_hours(tmp_path, monkeypatch):
    run = tmp_path
    (run / "question.jsonl").write_text(json.dumps({"id": "a", "question": {"text": "q"}}) + "\n")
    (run / "replies.jsonl").write_text(json.dumps({"id": "a", "reply": "r"}) + "\n")
    (run / "answer.jsonl").write_text(json.dumps({"id": "a", "parse_ok": True}) + "\n")
    plan = tmp_path / "plan.jsonl"
    plan.write_text(json.dumps({"id": "a", "prompt_tokens": 100}) + "\n")
    monkeypatch.setattr(G, "PLANS", {"val": plan, "test": plan})
    args = type("Args", (), {"run": run})
    (run / "samples.jsonl").write_text(json.dumps({"id": "a", "row_s": 60.0}) + "\n")
    assert G.cmd_gate(args) == 0
    (run / "samples.jsonl").write_text(json.dumps({"id": "a", "row_s": 60000.0}) + "\n")
    assert G.cmd_gate(args) == 3


@pytest.mark.skipif(not LOCAL, reason="the splits and plans stay off GitHub")
def test_the_measured_plans_cover_both_splits_and_stay_in_budget():
    train = {r["id"] for r in A.load_jsonl(REPO / "data" / "splits" / "train.jsonl")}
    for sp in A.SPLITS:
        rows = {r["id"]: r for r in A.load_jsonl(G.PLANS[sp])}
        assert set(rows) == {c["id"] for c in A.cases(sp)}
        assert all(e in train for r in rows.values() for e in r["examples"])
        assert max(r["prompt_tokens"] for r in rows.values()) <= 2048 - Q.ANSWER_MAX_TOKENS


# ------------------------------------------------------------ asking less

def row(results, samples=None, score=None, label=None, direct=0, real=1, dialogue=True, rid="r"):
    samples = samples if samples is not None else [["ex", str(r)] if r is not None else None for r in results]
    score = score if score is not None else {("ex", str(r)): 0.0 for r in results if r is not None}
    label = label if label is not None else {k: 0 for k in score}
    return {"id": rid, "results": results, "samples": samples, "score": score, "label": label, "direct": direct, "real": real,
            "placebo": 0, "dialogue": dialogue}


def test_a_sample_with_no_result_counts_against_agreement_and_no_parse_is_the_lowest_score():
    assert F.view(row(["a"] * 3 + [None] * 5), range(8))[0] == 3
    top, best, pick = F.view(row([None] * 8, samples=[None] * 8, score={}), range(8))
    assert (top, best, pick) == (0, -math.inf, None)


def test_the_pick_is_the_top_score_then_the_count_then_the_text():
    samples = [["ex", "low"]] * 5 + [["ex", "high"]] + [["keys", "tie"]] * 2
    r = row(["a"] * 8, samples=samples, score={("ex", "low"): -1.0, ("ex", "high"): 2.0, ("keys", "tie"): 2.0})
    assert F.view(r, range(8))[1:] == (2.0, ("keys", "tie"))
    r["score"][("ex", "high")] = 3.0
    assert F.view(r, range(8))[1:] == (3.0, ("ex", "high"))
    assert F.view(r, [0, 1])[1:] == (-1.0, ("ex", "low"))              # only the samples drawn


def test_a_stricter_threshold_asks_less_and_keeps_the_section_6_answer():
    seen = (4, 0.0, None)
    assert F.at_threshold(row([]), seen, 5) == (1, 1)
    assert F.at_threshold(row([]), seen, 4) == (1, 1)
    assert F.at_threshold(row([]), seen, 3) == (0, 0)
    assert F.at_threshold(row([], dialogue=False), seen, 5) == (0, 0)


def test_the_gate_asks_at_or_below_tau_and_answers_with_its_pick_or_the_section_6_answer():
    r = row([], direct=0, real=1, score={}, label={("ex", "good"): 1})
    assert F.gated(r, (0, -1.0, ("ex", "good")), -0.5, True) == (1, 1)            # doubts: asks
    assert F.gated(r, (0, -0.5, ("ex", "good")), -0.5, True) == (1, 1)            # at τ: asks
    assert F.gated(r, (0, 0.0, ("ex", "good")), -0.5, True) == (1, 0)             # sure: its pick
    assert F.gated(r, (0, 0.0, ("ex", "good")), -0.5, False) == (0, 0)            # Vg: the §6 answer
    r["dialogue"] = False
    assert F.gated(r, (0, -1.0, ("ex", "good")), -0.5, True) == (1, 0)            # no question exists
    assert F.gated(r, (0, -math.inf, None), -0.5, True) == (0, 0)                 # no pick either


def test_tau_is_the_smallest_value_with_the_rate_at_or_below_it():
    assert F.tau_from([5, 1, 3, 2, 4], 0.2) == 1
    assert F.tau_from([5, 1, 3, 2, 4], 0.21) == 2
    assert F.tau_from([1, 1, 1, 9, 9], 0.2) == 1


def fixture_rows(n=40):
    """Half the rows agree and are right; half split 4 / 4, and a question fixes them."""
    rows = {}
    for x in range(n):
        i = "r%02d" % x
        split = x % 2 == 0
        results = (["a"] * 4 + ["b"] * 4) if split else ["a"] * 8          # 8 samples: every draw sees all of them
        samples = [["ex", r] for r in results]
        rows[i] = row(results, samples=samples, score={("ex", "a"): -1.0 if split else 2.0, ("ex", "b"): -2.0},
                      label={("ex", "a"): 0 if split else 1, ("ex", "b"): 0}, direct=0 if split else 1, real=1, rid=i)
        rows[i]["placebo"] = 0
    return rows


def fixture_ids(rows):
    ids = sorted(rows)
    return {"untouched": ids[:10], "test": ids[10:30], "val": ids[:10] + ids[30:], "choosing": ids[30:]}


def test_every_arm_reads_the_reports_draws_and_a_mismatch_stops_the_build():
    rows = fixture_rows()
    assert F.draw_sets(rows["r00"]) == F.draw_sets(rows["r00"])
    rep = F.build(rows, fixture_ids(rows))
    assert rep["arms"]["t = 5"]["test"] == P.mean({i: T.draws(rows)[i][0] for i in rows}, fixture_ids(rows)["test"])


def test_the_build_refuses_a_rule_that_is_not_the_reports(monkeypatch):
    rows = fixture_rows()
    monkeypatch.setattr(T, "draws", lambda r: {i: (0.0, 0.0) for i in r})
    with pytest.raises(SystemExit):
        F.build(rows, fixture_ids(rows))


def test_t_is_the_fewest_questions_within_a_point_of_the_best(monkeypatch):
    rows = fixture_rows()
    rep = F.build(rows, fixture_ids(rows))
    assert rep["chosen_t"] == 4 and "T vs the rule" in rep["readings"]                  # at 4 of 8 the split rows still ask
    assert P.choose({1: 50.0, 2: 59.0, 3: 59.5, 4: 60.0, 5: 60.0}, [1, 2, 3, 4, 5]) == 2
    assert P.choose({1: 50.0, 2: 52.0, 3: 55.0, 4: 58.0, 5: 60.0}, [1, 2, 3, 4, 5]) == 5


def test_a_rule_kept_at_five_has_no_reading_and_says_so():
    rows = fixture_rows()
    for i, r in rows.items():                                 # only "at most 5 of 8" reaches the split rows
        if r["direct"] == 0:
            r["results"] = ["a"] * 5 + ["b"] * 3
            r["samples"] = [["ex", x] for x in r["results"]]
    rep = F.build(rows, fixture_ids(rows))
    assert rep["chosen_t"] == 5 and "T vs the rule" not in rep["readings"]
    assert "kept t = 5" in F.render(rep, "x")


def test_what_asking_less_saves_is_only_the_dialogue():
    rep = {"arms": {"t = 5": {"asks_test": 40.0}, "t = 4": {"asks_test": 30.0}}}
    timing = {"summary": {"k8": {"question_s": {"mean": 1.0}, "turn2_s": {"mean": 1.0}, "models_wall_s": {"mean": 4.0}}}}
    got = F.saved(rep, timing, "t = 4")
    assert got == {"turn_s": 2.0, "saved_gpu_s": 0.2, "first_response_gpu_s": 4.0}


# ------------------------------------------------------------ the report, end to end on a small run

def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def small_run(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    cases = {"val": [dict(CASE, id="v%d" % n, tier="T1") for n in range(8)], "test": [dict(CASE, id="t%d" % n, tier="T2") for n in range(8)]}
    every = cases["val"] + cases["test"]
    results = {c["id"]: ["r%d" % (n % 2) for n in range(16)] if n % 2 else ["r0"] * 16 for n, c in enumerate(every)}
    write(run / "views.jsonl", [{"id": i, "results": r} for i, r in results.items()])
    write(run / "samples.jsonl", [{"id": i, "samples": [["ex", x] for x in r]} for i, r in results.items()])
    write(run / "scores.jsonl", [{"id": i, "mode": "ex", "command": x, "score": 1.0} for i in results for x in ("r0", "r1")])
    write(run / "states.jsonl", [{"id": i, "mode": "ex", "command": x, "state": "match" if x == "r0" else "no"} for i in results for x in ("r0", "r1")])
    write(run / "question.jsonl", [{"id": c["id"], "question": {"text": "q", "output_tokens": 5}} for c in every])
    write(run / "replies.jsonl", [{"id": c["id"], "reply": "r"} for c in every])
    verdict = {"match": True, "valid": True, "parse_ok": True, "second_ask": False, "think": False, "tokens": 4, "mode": "ex", "command": "r0"}
    write(run / "judged.jsonl", [{"id": c["id"], "split": "val", "real": verdict, "placebo": dict(verdict, match=False), "leak_flags": [],
                                  "quotes_result": False} for c in every])
    for sp in A.SPLITS:
        write(run / ("exec_retr8_%s.jsonl" % sp), [{"id": c["id"], "match": n % 2 == 0, "valid": True} for n, c in enumerate(cases[sp])])
        write(run / ("exec_k0_%s.jsonl" % sp), [{"id": c["id"], "match": False, "valid": True} for c in cases[sp]])
        write(run / ("greedy_retr8_%s.jsonl" % sp), [{"id": c["id"], "samples": [["ex", "r0"]], "output_tokens": [4], "prompt_tokens": 800} for c in cases[sp]])
    plan = run / "plan.jsonl"
    write(plan, [{"id": c["id"], "same_shape_neighbor": False} for c in every])
    recorded = run / "recorded.jsonl"
    write(recorded, [{"id": c["id"], "mode": "ex", "command": "r0"} for c in every])
    monkeypatch.setattr(G, "PLANS", {"val": plan, "test": plan})
    monkeypatch.setattr(G, "RECORDED", recorded)
    monkeypatch.setattr(A, "cases", lambda sp, limit=0: cases[sp])
    monkeypatch.setattr(B.C, "untouched_ids", lambda val: [c["id"] for c in val[:6]])
    monkeypatch.setattr(T, "deployed_4b", lambda ids: {i: 0 for i in ids})
    ids = [c["id"] for c in every]
    trained = {"rows": {}, "direct": {i: 1 for i in ids}, "rule": {i: 1 for i in ids}, "one": {i: 1 for i in ids}}
    return run, trained


def test_the_report_reads_a_small_run_end_to_end(tmp_path, monkeypatch):
    run, trained = small_run(tmp_path, monkeypatch)
    rep = B.build(run, trained)
    assert rep["rows"]["test"] == 8 and rep["dialogues"] == 16
    assert set(rep["readings"]) == {"C: 8 examples vs the trained 27B, one answer each", "R: with the rule vs without it, on 8 examples",
                                    "W: 8 examples vs the trained 27B, the rule on both", "I: after the reply vs after the placebo, every row"}
    assert rep["readings"]["C: 8 examples vs the trained 27B, one answer each"]["pooled"]["diff_pts"] < 0
    text = B.render(rep)
    assert "the 4B round trip, as deployed" in text and "## Checks" in text


def test_fewer_reads_the_same_small_run(tmp_path, monkeypatch):
    run, _ = small_run(tmp_path, monkeypatch)
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    rows = B.load_rows(run, cases)
    rep = F.build(rows, T.split_ids(rows, cases))
    assert rep["arms"]["t = 5"]["asks_test"] > 0 and "V vs the rule" in rep["readings"]


def test_a_row_with_no_question_stands_on_its_section_6_answer(tmp_path, monkeypatch):
    run, trained = small_run(tmp_path, monkeypatch)
    rows = A.load_jsonl(run / "question.jsonl")
    rows[1]["question"]["text"] = ""
    write(run / "question.jsonl", rows)
    write(run / "judged.jsonl", [r for r in A.load_jsonl(run / "judged.jsonl") if r["id"] != rows[1]["id"]])
    got = B.load_rows(run, {sp: A.cases(sp) for sp in A.SPLITS})[rows[1]["id"]]
    assert got["dialogue"] is False and got["real"] == got["direct"] == got["placebo"]


def test_timing_rows_with_no_question_cost_only_the_first_response():
    t = {"models_wall_s": {"mean": 4.0}, "asks": 0.0, "question_s": None, "turn2_s": None, "first_response_s": {"p50": 4.0, "p95": 9.0}, "tokens": 90.0}
    timing = {"summary": {"direct": {"total_s": {"mean": 1.0, "p50": 1.0, "p95": 2.0}, "tokens": 16.0}, "k4": t, "k8": t}}
    assert B.costs(timing, 0.0, {"direct": 50.0, 4: 50.0, 8: 50.0})["round trip at 8"]["gpu_s_per_request"] == 4.0
    rep = {"arms": {"t = 5": {"asks_test": 40.0}, "t = 4": {"asks_test": 30.0}}}
    assert F.saved(rep, timing, "t = 4")["saved_gpu_s"] == 0.0


def test_the_cost_keeps_bm25_out_of_the_gpu_and_in_the_wait():
    t = {"models_wall_s": {"mean": 4.0}, "asks": 0.5, "question_s": {"mean": 1.0}, "turn2_s": {"mean": 1.0},
         "first_response_s": {"p50": 4.0, "p95": 9.0}, "tokens": 100.0}
    timing = {"summary": {"direct": {"total_s": {"mean": 1.0, "p50": 1.0, "p95": 2.0}, "tokens": 16.0}, "k4": t, "k8": t}}
    got = B.costs(timing, 10.0, {"direct": 50.0, 4: 50.0, 8: 50.0})
    assert got["one answer"]["gpu_s_per_request"] == 1.0 and got["one answer"]["p50"] == 1.01
    assert got["round trip at 8"]["gpu_s_per_request"] == 5.0 and got["round trip at 8"]["p50"] == 4.01


# ------------------------------------------------------------ the trained 27B (addendum 2)

def test_each_arm_asks_the_endpoint_for_its_own_model(monkeypatch):
    import endpoint as E
    seen = []
    monkeypatch.setattr(E, "load", lambda model, url, name: seen.append(name) or Fake())
    for arm, name in (("base", G.BASE_NAME), ("trained", "served27b")):
        monkeypatch.setattr(G, "ARM", arm)
        G.plain_endpoint()
        assert seen[-1] == name


def test_the_trained_27b_runs_no_zero_shot_pass():
    assert G.ARMS["base"] == ("retr8", "k0") and G.ARMS["trained"] == ("retr8",)


def test_a_row_agrees_with_the_record_or_where_there_is_none_with_itself():
    assert G.agreeing([["ex", "a"], ["ex", "a"], ["ex", "b"]], {"mode": "ex", "command": "a"}) == 2
    assert G.agreeing([["ex", "a"], ["ex", "a"], ["ex", "b"], None]) == 2
    assert G.agreeing([None, None, ["ex", "b"]]) == 2
    assert G.agreeing([["ex", "a"]], G.NO_ANSWER) == 0


def trained_run(tmp_path, monkeypatch):
    run, trained = small_run(tmp_path, monkeypatch)
    served = tmp_path / "served"
    served.mkdir()
    for sp in A.SPLITS:
        write(served / ("exec_%s.jsonl" % sp), [{"id": c["id"], "match": True, "valid": True} for c in A.cases(sp)])
    monkeypatch.setattr(Q, "SERVE27", served)
    return run, trained


def test_the_trained_arm_reads_its_zero_shot_side_from_the_served_run(tmp_path, monkeypatch):
    run, _ = trained_run(tmp_path, monkeypatch)
    rows = B.load_rows(run, {sp: A.cases(sp) for sp in A.SPLITS}, "trained")
    assert all(r["zero"] == 1 for r in rows.values())                       # the fixture's k0 pass reads 0 everywhere
    assert all(r["zero"] == 0 for r in B.load_rows(run, {sp: A.cases(sp) for sp in A.SPLITS}).values())


def test_the_trained_report_has_its_own_readings_and_no_gap(tmp_path, monkeypatch):
    run, trained = trained_run(tmp_path, monkeypatch)
    ids = list(trained["direct"])
    base = {"direct": {i: 0 for i in ids}, "rule": {i: 0 for i in ids}}
    rep = B.build(run, trained, arm="trained", base=base)
    assert set(rep["readings"]) == {"E: 8 examples vs none, on the trained 27B, one answer each", "R: with the rule vs without it, on 8 examples",
                                    "W: 8 examples vs none, the rule on both", "B: the trained vs the base 27B, 8 examples each, one answer each",
                                    "I: after the reply vs after the placebo, every row"}
    assert rep["gap_closed"] is None and rep["greedy"]["served_identical_val"] is None
    assert "8 examples vs zero-shot" not in rep["descriptive"]
    text = B.render(rep)
    assert text.startswith("# The trained 27B with 8 BM25 examples") and "the gap from zero-shot" not in text


def test_reading_b_waits_for_part_2(tmp_path, monkeypatch):
    run, trained = trained_run(tmp_path, monkeypatch)
    assert not any(k.startswith("B:") for k in B.build(run, trained, arm="trained")["readings"])
    partial = {"direct": {"v0": 0}, "rule": {"v0": 0}}
    assert not any(k.startswith("B:") for k in B.build(run, trained, arm="trained", base=partial)["readings"])


def test_base_side_is_none_until_part_2_has_reported(tmp_path, monkeypatch):
    run, _ = small_run(tmp_path, monkeypatch)
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    assert B.base_side(run, cases) is None
    (run / "report.json").write_text("{}")
    assert set(B.base_side(run, cases)) == {"direct", "rule"}


def stubbed_serve(tmp_path, adapter):
    """Runs serve.sh start with conda, curl and nvidia-smi stubbed, and returns the vLLM flags it passed."""
    stubs = tmp_path / "bin"
    stubs.mkdir()
    args = tmp_path / "args.txt"
    (stubs / "conda").write_text('#!/bin/bash\nprintf "%%s\\n" "$@" > %s\n' % args)
    (stubs / "curl").write_text("#!/bin/bash\nexit 0\n")
    (stubs / "nvidia-smi").write_text("#!/bin/bash\necho 1\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    env = {"PATH": "%s:/usr/bin:/bin" % stubs, "OUT": str(tmp_path / "out"), "HOME": str(tmp_path)}
    if adapter:
        env["ADAPTER"] = "some/adapter"
    done = subprocess.run(["bash", str(HERE / "serve.sh"), "start"], capture_output=True, text=True, env=env, timeout=30)
    assert done.returncode == 0, done.stderr
    for _ in range(200):
        if args.exists() and args.read_text():
            break
        import time
        time.sleep(0.01)
    return args.read_text().split("\n")


def test_the_base_endpoint_has_no_adapter_and_the_larger_cache(tmp_path):
    flags = stubbed_serve(tmp_path, adapter=False)
    assert "--enable-lora" not in flags and "--lora-modules" not in flags
    assert flags[flags.index("--kv-cache-memory-bytes") + 1] == "1342177280"
    assert flags[flags.index("--max-model-len") + 1] == "2400"
    assert flags[flags.index("--served-model-name") + 1] == G.BASE_NAME


def test_the_trained_endpoint_serves_the_adapter_at_the_trained_runs_cache(tmp_path):
    flags = stubbed_serve(tmp_path, adapter=True)
    assert flags[flags.index("--lora-modules") + 1] == "served27b=some/adapter"
    assert flags[flags.index("--kv-cache-memory-bytes") + 1] == "1073741824"
    assert flags[flags.index("--max-model-len") + 1] == "2400"


def run_all(tmp_path, rc_first, pause=False):
    """run_all.sh with its parts stubbed: each part records its environment and exits as told."""
    stub = tmp_path / "part.sh"
    calls = tmp_path / "calls.txt"
    stub.write_text('#!/bin/bash\necho "$ARM $(basename $RUN)" >> %s\n[ "$ARM" = base ] && exit %d\nexit 0\n' % (calls, rc_first))
    run = tmp_path / "run"
    run.mkdir()
    if pause:
        (run / "PAUSE").write_text("")
    done = subprocess.run(["bash", str(HERE / "run_all.sh")], capture_output=True, text=True, timeout=30,
                          env={"PATH": "/usr/bin:/bin", "RUN": str(run), "SELF": str(stub)})
    return done.returncode, calls.read_text().split("\n")[:-1], run


def test_both_parts_run_in_turn(tmp_path):
    rc, calls, run = run_all(tmp_path, 0)
    assert rc == 0 and calls == ["base run", "trained trained"]
    assert (run / "trained" / "PAUSE").is_symlink()


def test_a_failed_first_part_does_not_hold_the_second(tmp_path):
    rc, calls, _ = run_all(tmp_path, 1)
    assert calls == ["base run", "trained trained"] and rc == 0


def test_a_pause_holds_the_second_part(tmp_path):
    rc, calls, _ = run_all(tmp_path, 2, pause=True)
    assert rc == 2 and calls == ["base run"]


def test_the_pause_link_reaches_the_second_parts_dry_run(tmp_path):
    _, _, run = run_all(tmp_path, 0)
    dry = run / "trained" / "dry"
    dry.mkdir()
    (dry / "PAUSE").symlink_to("../PAUSE")
    assert not (dry / "PAUSE").exists()
    (run / "PAUSE").write_text("")
    assert (dry / "PAUSE").exists()


# ------------------------------------------------------------ the driver

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
    assert got == ["step plan none", "step smoke 27b", "step greedy 27b", "step samples 27b", "step views none", "step question 27b",
                   "step placebo 27b", "stop 27b", "step simulate sim", "step answer 27b", "step timing 27b", "stop 27b", "step score v4",
                   "stop v4", "step judge none", "step report none", "rc=0"]


def test_a_report_that_fails_on_the_dry_rows_does_not_stop_the_gpu_chain():
    body = """
    LIMIT=%d
    step() { [ "$1" = report ] && return 1; return 0; }
    stop_ep() { :; }
    chain
    echo "rc=$?"
    """
    assert "rc=0" in driver(body % 12)
    assert "rc=1" in driver(body % 0)                  # on the real rows it is reported as what it is


def test_a_pause_during_the_dry_report_still_pauses():
    out = driver("""
    LIMIT=12
    step() { [ "$1" = report ] && return 2; return 0; }
    stop_ep() { :; }
    chain
    echo "rc=$?"
    """)
    assert "rc=2" in out


def test_the_smoke_is_as_sure_in_a_dry_run_as_in_the_real_one():
    src = (HERE / "retr27b.py").read_text(encoding="utf-8")
    body = src[src.index("def cmd_smoke"):src.index("def cmd_greedy")]
    assert "args.limit" not in body


def test_a_failed_smoke_stops_the_night_before_the_card_is_spent():
    out = driver("""
    step() { echo "step $1" >> "$RUN/calls"; [ "$1" = smoke ] && return 3; return 0; }
    stop_ep() { :; }
    chain
    echo "rc=$?" >> "$RUN/calls"
    cat "$RUN/calls"
    """)
    assert [x for x in out if x.startswith(("step", "rc"))] == ["step plan", "step smoke", "rc=3"]


def test_the_simulator_counts_as_down_and_the_verifier_uses_the_4b_port():
    out = driver("""
    is_up sim; echo "sim=$?"
    is_up none; echo "none=$?"
    port_up() { echo "port $1"; }
    is_up v4; is_up 27b
    """)
    assert "sim=1" in out and "none=0" in out and "port 8011" in out and "port 8015" in out


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
    assert "rc=3" in driver(body % (0, 1))
    assert "rc=5" in driver(body % (5, 0))


def test_the_trained_arm_serves_the_adapter_and_reports_as_itself():
    out = driver("""
    ARM=trained
    stubpy() { echo "py $*"; return 0; }
    report_step
    """)
    lines = [x for x in out if x.startswith("py")]
    assert "--arm trained" in lines[0] and lines[1].endswith("--model trained-retr")


def test_arm_trained_sets_the_adapter_and_passes_the_arm_on():
    script = """
    set -u
    export RUN=$(mktemp -d) DRY=0 ARM=trained
    source %s
    echo "adapter=$ADAPTER"
    echo "args=$ARGS"
    """ % (HERE / "run.sh")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30).stdout.split("\n")
    assert "adapter=runs/qwen3.5-27b-sft-distill/best" in out
    assert any(x.startswith("args=") and x.endswith("--arm trained") for x in out)


def test_the_base_arm_serves_no_adapter():
    script = """
    set -u
    unset ADAPTER
    export RUN=$(mktemp -d) DRY=0
    source %s
    echo "adapter=${ADAPTER:-none}"
    echo "args=$ARGS"
    """ % (HERE / "run.sh")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30).stdout.split("\n")
    assert "adapter=none" in out and any(x.endswith("--arm base") for x in out)


def test_the_report_step_runs_both_reports_and_stops_on_the_first_failure():
    out = driver("""
    stubpy() { echo "py $*"; return 0; }
    report_step
    """)
    lines = [x for x in out if x.startswith("py")]
    assert len(lines) == 2 and "report_retr.py" in lines[0] and "fewer.py" in lines[1] and lines[1].endswith("--model base")
    failed = driver("""
    stubpy() { echo "py $*"; return 1; }
    report_step
    """)
    assert sum(line.startswith("py") for line in failed) == 1


# ------------------------------------------------------------ the independent check of the readings

def load_check():
    """check_retr.py, loaded without leaving its sys.path order behind for the other tests."""
    import importlib.util
    saved = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("check_retr", HERE / "check_retr.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


CHECK = load_check()


def test_the_check_counts_a_sample_with_no_result_against_agreement():
    assert CHECK.asks_on(["a"] * 5 + [None] * 3)
    assert not CHECK.asks_on(["a"] * 6 + ["b", None])
    assert CHECK.asks_on([None] * 8)


@pytest.mark.parametrize("pooled,parts,want", [
    ((2.0, 0.5, 4.0), [(1.0,), (2.0,)], "helps"),
    ((2.0, 0.5, 4.0), [(3.0,), (-0.2,)], "tied"),
    ((2.0, -1.0, 4.0), [(1.0,), (2.0,)], "tied"),
    ((-2.0, -4.0, -0.5), [(-1.0,), (-2.0,)], "hurts"),
    ((-2.0, -4.0, -0.5), [(-1.0,), (0.0,)], "tied"),
])
def test_the_checks_conjunction_rule_matches_the_reports(pooled, parts, want):
    assert CHECK.judge(pooled, parts) == want


def test_the_checks_mcnemar_is_exact():
    a = {i: 1 for i in "abcde"}
    b = {i: 0 for i in "abcde"}
    assert CHECK.mcnemar(a, b, list(a)) == (0, 5, 0.0625)
    assert CHECK.mcnemar(a, a, list(a)) == (0, 0, 1.0)


def test_a_row_with_no_dialogue_never_asks_in_the_checks_draws():
    s = {"results": {"x": [None] * 16}, "dialogue": {"x": False}, "real": {"x": 0}, "direct": {"x": 1}}
    right, rate = CHECK.rule_draws(s, ["x"], "t")
    assert right["x"] == 1 and rate["x"] == 0


def test_the_checks_interval_is_b_minus_a_in_points():
    a = {"x": 0, "y": 1, "z": 0, "w": 0}
    b = {"x": 1, "y": 1, "z": 1, "w": 0}
    diff, lo, hi = CHECK.interval(a, b, list(a), boots=500)
    assert diff == 50.0 and lo <= 50.0 <= hi
