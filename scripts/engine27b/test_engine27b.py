"""CPU tests for scripts/engine27b (R0, the engine probe)."""
import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("engine27b_probe", HERE / "probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

LOCAL = (REPO / "runs" / "think_long_2026-11-01" / "rows_8192_think-retr8.jsonl").exists()
local_only = pytest.mark.skipif(not LOCAL, reason="the thinking runs' rows are local-only")

LINE = ("(APIServer pid=4242) INFO 11-02 10:23:45 [loggers.py:264] Engine 000: Avg prompt throughput: 12.5 tokens/s, "
        "Avg generation throughput: %s tokens/s, Running: %d reqs, Waiting: 4 reqs,%s GPU KV cache usage: 71.0%%, "
        "Prefix cache hit rate: 0.0%%")


def test_stats_parser_reads_vllms_line_with_and_without_preemptions():
    a = P.parse_stats(LINE % ("61.3", 8, " Preemptions: 2,"), 2026)
    b = P.parse_stats(LINE % ("30.0", 1, " Deferred: 1 reqs,"), 2026)
    assert [(s["gen"], s["running"], s["waiting"], s["preemptions"], s["kv"]) for s in a] == [(61.3, 8, 4, 2, 71.0)]
    assert [(s["gen"], s["running"], s["preemptions"]) for s in b] == [(30.0, 1, 0)]
    assert a[0]["t"] == b[0]["t"]


def test_by_running_keeps_only_intervals_inside_the_windows():
    stats = [{"t": t, "gen": g, "running": n} for t, g, n in
             [(100, 10.0, 1), (103, 99.0, 1), (110, 12.0, 1), (200, 50.0, 8), (300, 77.0, 8)]]
    runs = P.by_running(stats, [(99, 120), (190, 250)])
    # a line at t covers the 5 s before it: 100 and 103 began before the window opened, 300 is after it
    assert runs == {1: {"intervals": 1, "median": 12.0}, 8: {"intervals": 1, "median": 50.0}}


def test_gate_rule():
    def runs(**kw):
        return {int(k[1:]): {"intervals": v[0], "median": v[1]} for k, v in kw.items()}
    assert P.gate(runs(n8=(3, 130.0))) == (130.0, 8, "fits one window")
    assert P.gate(runs(n8=(5, 119.9))) == (119.9, 8, "two windows")
    assert P.gate(runs(n8=(9, 59.0))) == (59.0, 8, "not on this card")
    assert P.gate(runs(n8=(2, 500.0), n4=(6, 70.0))) == (70.0, 4, "two windows")    # too few at 8: the highest with 3
    assert P.gate({}) == (None, None, "not on this card")


def test_stage_plan_covers_each_row_once_at_first_draw():
    firsts = [k for s in P.THINK_STAGES for k, d in P.STAGES[s][2] if d == 0]
    assert sorted(firsts) == list(range(P.N_ROWS))
    assert [(s, P.STAGES[s][0], len(P.STAGES[s][2])) for s in P.STAGES] == [
        ("A", 1, 2), ("B", 4, 6), ("C", 8, 12), ("D1", 1, 16), ("D8", 8, 16)]
    seeds = {P.seed_for("r%d" % k, d, t) for k in range(16) for d in (0, 1) for t in (True, False)}
    assert len(seeds) == 64


def test_serve_args_come_from_the_registry_and_the_attempts():
    sys_path_reg = P.load("baseline_for_test", REPO / "scripts" / "baseline.py").load_registry(P.T.THINK_MODEL)
    meta = {"max_model_len": 9984}
    one = P.serve_args(1, 8013, meta)
    three = P.serve_args(3, 8013, meta)
    assert one[0] == sys_path_reg["hf_id"] and one[one.index("--revision") + 1] == sys_path_reg["revision"]
    for flag in ("--language-model-only", "--quantization", "bitsandbytes", "--generation-config"):
        assert flag in one
    assert one[one.index("--max-num-seqs") + 1] == "8" and "--enforce-eager" not in one
    assert three[three.index("--max-num-seqs") + 1] == "4" and "--enforce-eager" in three
    assert one[one.index("--max-model-len") + 1] == "9984"
    for f in ("probe.py", "run.sh"):
        assert "Qwen/" not in (HERE / f).read_text(encoding="utf-8"), f


def test_request_body_and_record():
    meta = {"sampling": dict(P.T.PREREGISTERED_SAMPLING), "stop_ids": [248044, 248046], "think_end_id": 7}
    p = {"k": 3, "id": "hc-1", "think_ids": [1, 2, 3], "plain_ids": [1]}
    body = P.request_body(p, 1, True, meta)
    assert (body["max_tokens"], body["temperature"], body["top_p"], body["top_k"]) == (8192, 0.6, 0.95, 20)
    assert body["prompt"] == [1, 2, 3] and body["stop_token_ids"] == [248044, 248046] and not body["skip_special_tokens"]
    assert P.request_body(p, 0, False, meta)["max_tokens"] == 192 and P.request_body(p, 0, False, meta)["prompt"] == [1]
    resp = {"choices": [{"text": "let me see</think>\n\nex: %s/a/b/", "token_ids": [5, 6, 7, 8, 9], "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5}}
    r = P.completion_record("C", p, 1, True, meta, resp, 10.0, 12.5)
    assert (r["think_tokens"], r["closed"], r["answer"], r["seconds"]) == (2, True, "\n\nex: %s/a/b/", 2.5)
    cut = dict(resp, choices=[{"text": "still thinking", "token_ids": [5, 6], "finish_reason": "length"}])
    r = P.completion_record("C", p, 0, True, meta, cut, 0.0, 1.0)
    assert (r["closed"], r["answer"], r["think_tokens"]) == (False, None, 2)


@local_only
def test_rows_are_the_start_of_the_arms_order():
    rows = P.select_rows()
    order = P.E.arm_orders(P.E.FIRST_RUN)[P.ARM]
    assert [i for i, _ in rows] == order[:P.N_ROWS]


@local_only
def test_prompts_file_matches_the_thinking_runs_rendering():
    path = P.RUN / "prompts.jsonl"
    if not path.exists():
        pytest.skip("prompts.jsonl not built yet")
    from evaluate import load_jsonl
    tok, info, _ = P.engine_info()
    val = {c["id"]: c for c in load_jsonl(str(P.T.VAL))}
    train = {c["id"]: c for c in load_jsonl(str(P.T.TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(P.T.PLAN))}
    p = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    text = P.E.prompt_text(tok, val[p["id"]], P.ARM, train, plan, info)
    assert tok.decode(p["think_ids"]) == text
    assert p["think_ids"] == tok(text, add_special_tokens=False)["input_ids"]     # nothing added in front


@local_only
def test_the_judge_reproduces_a_recorded_outcome():
    from evaluate import load_jsonl
    val = {c["id"]: c for c in load_jsonl(str(P.T.VAL))}
    rec = next(r for r in load_jsonl(str(REPO / "runs" / "think_long_2026-11-01" / "rows_8192_think-retr8.jsonl"))
               if r.get("fresh") and r["closed"])
    parsed, per = P.T.judge([val[rec["id"]]], [rec["answer"]], 5.0)
    assert per[rec["id"]]["match"] == rec["match"]
