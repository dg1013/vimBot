"""Tests for the served 27B and the three-model pool (CPU, no endpoints): `pytest scripts/serve27b`."""
import pathlib
import random
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_text_27b as T   # noqa: E402
import pass27b as A         # noqa: E402
import pool27b as R         # noqa: E402
import posthoc27b as H      # noqa: E402


def row(samples, cands, tokens=None):
    """A merged row: samples {model: [[mode, command] | None]}, cands [(command, score, state)]."""
    return {"id": "x", "samples": samples, "output_tokens": tokens or {},
            "cands": [{"mode": "ex", "command": c, "score": s, "state": st} for c, s, st in cands]}


def test_text_names_keep_the_text_model_and_only_the_text_model():
    assert T.text_name("model.language_model.layers.3.mlp.up_proj.weight") == "model.layers.3.mlp.up_proj.weight"
    assert T.text_name("model.language_model.embed_tokens.weight") == "model.embed_tokens.weight"
    assert T.text_name("lm_head.weight") == "lm_head.weight"
    assert T.text_name("model.visual.blocks.0.attn.proj.weight") is None
    assert T.text_name("mtp.fc.weight") is None


def test_text_config_is_the_text_block_with_the_causal_class():
    full = {"tie_word_embeddings": False, "text_config": {"hidden_size": 5120, "rope_parameters": {"partial_rotary_factor": 0.25}}}
    cfg = T.text_config(full)
    assert cfg["architectures"] == ["Qwen3_5ForCausalLM"] and cfg["hidden_size"] == 5120
    assert cfg["partial_rotary_factor"] == 0.25 and cfg["tie_word_embeddings"] is False
    assert cfg["bos_token_id"] is None and cfg["pad_token_id"] is None
    assert "architectures" not in full["text_config"]


def test_the_row_order_mixes_splits_and_does_not_depend_on_input_order():
    ids = ["a-%d" % i for i in range(50)] + ["b-%d" % i for i in range(50)]
    assert A.mix(ids) == A.mix(list(reversed(ids)))
    first = A.mix(ids)[:30]
    assert 5 < sum(i.startswith("a") for i in first) < 25


def test_the_pace_check_needs_a_real_fall_and_enough_rows():
    fast = [10.0] * 25
    assert not A.pace_fell(fast + [100.0] * 9)                              # too few rows to call
    assert A.pace_fell(fast + [100.0] * 10)
    assert not A.pace_fell(fast + [25.0] * 10)                              # slower, not paging
    assert not A.pace_fell(fast + [100.0, 10.0] * 5 + [10.0])               # a few long rows are not a fall


def driver(codes, retries=3, setup=""):
    """Runs run.sh's step() on a stub command that exits with each code in turn, stopping and
    starting endpoints as stubs too -> (step's return code, the stub's calls, the status lines)."""
    script = """
    set -u
    export RUN=$(mktemp -d)
    source %s
    RETRIES=%d
    card() { echo 0,0; }; procs() { echo 0; }; wait_idle() { echo waited >> "$RUN/calls"; }
    stop_ep() { echo stop >> "$RUN/calls"; }; start_ep() { echo start >> "$RUN/calls"; }
    is_up() { grep -q start "$RUN/calls" 2>/dev/null && [ "$(tail -1 "$RUN/calls")" != stop ]; }
    codes=(%s); n=0
    stub() { echo run >> "$RUN/calls"; local c=${codes[$(cat "$RUN/n" 2>/dev/null || echo 0)]}
             echo $(( $(cat "$RUN/n" 2>/dev/null || echo 0) + 1 )) > "$RUN/n"; return "$c"; }
    %s
    step samples 27b stub > /dev/null
    echo "rc=$?"; cat "$RUN/calls"; echo ---; cut -d' ' -f3- "$RUN/status.txt"
    """ % (HERE / "run.sh", retries, " ".join(map(str, codes)), setup)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30).stdout
    head, _, status = out.partition("---\n")
    lines = head.split()
    return int(lines[0][3:]), lines[1:], status


def test_the_driver_yields_the_card_when_the_pace_falls_and_resumes():
    rc, calls, status = driver([5, 0])
    assert rc == 0
    assert calls == ["waited", "start", "run", "stop", "waited", "start", "run"]
    assert "yielding" in status and "samples: done" in status


def test_a_gate_stops_the_driver_at_once_and_a_pause_is_not_a_failure():
    assert driver([3, 0])[0] == 3 and driver([3, 0])[1].count("run") == 1
    assert driver([4, 0])[0] == 2 and driver([4, 0])[1].count("run") == 1


def test_the_driver_gives_up_after_its_retries_and_a_done_step_is_skipped():
    rc, calls, status = driver([1, 1, 1], retries=2)
    assert rc == 1 and calls.count("run") == 3 and "gave up" in status
    rc, calls, _ = driver([1], setup='touch "$RUN/done_samples"')
    assert rc == 0 and "run" not in calls


def test_a_pause_file_stops_the_driver_before_it_touches_the_card():
    rc, calls, _ = driver([0], setup='touch "$RUN/PAUSE"')
    assert rc == 2 and calls == []


def test_a_pause_stops_the_wait_for_a_card_at_once_and_the_card_stays_untouched():
    script = """
    export RUN=$(mktemp -d)
    source %s
    card() { echo 99,20000; }; procs() { echo 0; }; start_ep() { echo STARTED; }; stop_ep() { :; }; is_up() { return 1; }
    ( sleep 1; touch "$RUN/PAUSE" ) &
    SECONDS=0
    step samples 27b true
    echo "rc=$? seconds=$SECONDS"
    """ % (HERE / "run.sh")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=40).stdout
    assert "STARTED" not in out and "rc=2" in out
    assert int(out.split("seconds=")[1].split()[0]) <= 15                   # within one 10-second check


def test_same_answer_needs_mode_and_command():
    want = {"command": "dd", "mode": "keys"}
    assert A.same_answer(("keys", "dd"), want) and A.same_answer(["keys", "dd"], want)
    assert not A.same_answer(("ex", "dd"), want) and not A.same_answer(None, want)


def test_distinct_answers_put_greedy_first_and_drop_the_unparsed():
    g, s = {"samples": [["ex", "a"]]}, {"samples": [["ex", "b"], None, ["ex", "a"], ["keys", "b"]]}
    assert A.distinct_answers(g, s) == [("ex", "a"), ("ex", "b"), ("keys", "b")]
    assert A.distinct_answers(None, None) == []
    assert A.distinct_answers({"samples": [None]}, None) == []


def test_only_unknown_answers_go_to_the_verifier():
    assert A.to_score([("ex", "a"), ("ex", "b")], {("ex", "a")}) == [("ex", "b")]


def test_the_family_runs_cheapest_first_and_never_past_the_samples_drawn():
    names = [c for c, _ in R.family(8)]
    assert names == ["g", "1", "2", "4", "8"]
    assert dict(R.family(32))["g"] == {"4b": 4, "9b": 4, "27g": 1}
    assert dict(R.family(32))["32"] == {"4b": 4, "9b": 4, "27b": 32}


def test_the_rule_takes_the_cheapest_setting_within_a_point_of_the_best():
    order = ["g", "1", "4", "8"]
    assert R.choose({"g": 60.0, "1": 61.5, "4": 62.0, "8": 62.4}, order) == "1"
    assert R.choose({"g": 62.0, "1": 61.5, "4": 62.0, "8": 62.4}, order) == "g"
    assert R.choose({"g": 60.0, "1": 60.0, "4": 60.0, "8": 64.0}, order) == "8"
    assert R.choose({"g": 60.0, "1": 62.0, "4": 63.0, "8": 63.0}, order) == "1"    # exactly a point down still counts


def test_paired_differences_are_b_minus_a_in_points():
    a = {"x": 0, "y": 1, "z": 0, "w": 0}
    b = {"x": 1, "y": 1, "z": 1, "w": 0}
    r = R.paired(a, b, list(a), n_boot=500)
    assert r["diff_pts"] == 50.0 and r["n"] == 4 and r["ci95"][0] <= 50.0 <= r["ci95"][1]


@pytest.mark.parametrize("ci,parts,want", [
    ([0.5, 4.0], [1.0, 2.0], "helps"),
    ([0.5, 4.0], [3.0, -0.2], "tied"),        # the interval clears zero, one slice disagrees
    ([-1.0, 4.0], [1.0, 2.0], "tied"),
    ([-4.0, -0.5], [-1.0, -2.0], "hurts"),
    ([-4.0, -0.5], [-1.0, 0.0], "tied"),
])
def test_the_conjunction_rule(ci, parts, want):
    assert R.verdict({"ci95": ci}, [{"diff_pts": p} for p in parts]) == want


def old_row():
    return {"id": "x", "samples": {"4b": [["ex", "a"]], "9b": [["ex", "b"]]}, "output_tokens": {"4b": [3], "9b": [4]},
            "cands": [{"mode": "ex", "command": "a", "score": 1.0, "state": "match"},
                      {"mode": "ex", "command": "b", "score": 0.0, "state": "valid_wrong"}]}


def test_a_merged_row_adds_the_27b_and_overlays_todays_verdicts():
    m = R.merged_row(old_row(), {"samples": [["ex", "c"], None], "output_tokens": [5, 6]}, {"samples": [["ex", "a"]], "output_tokens": [3]},
                     {("ex", "c"): 2.0}, {("ex", "c"): "match", ("ex", "a"): "invalid"})
    by = {c["command"]: c for c in m["cands"]}
    assert by["c"]["score"] == 2.0 and by["c"]["state"] == "match"
    assert by["a"]["state"] == "invalid" and by["a"]["score"] == 1.0       # the verdict moves, the score does not
    assert by["b"]["state"] == "valid_wrong"                                # no new verdict: the old one stands
    assert m["samples"]["27g"] == [["ex", "a"]] and m["output_tokens"]["27b"] == [5, 6]


def test_a_missing_score_or_verdict_stops_the_report():
    sample, greedy = {"samples": [["ex", "c"]], "output_tokens": [5]}, {"samples": [None], "output_tokens": [1]}
    with pytest.raises(KeyError):
        R.merged_row(old_row(), sample, greedy, {}, {})                     # no score for "c"
    with pytest.raises(KeyError):
        R.merged_row(old_row(), sample, greedy, {("ex", "c"): 1.0}, {})     # no verdict for "c"


def test_one_run_takes_each_models_first_k_and_the_verifiers_choice():
    r = row({"4b": [["ex", "a"], ["ex", "b"]], "27g": [["ex", "c"]]},
            [("a", 1.0, "valid_wrong"), ("b", 9.0, "match"), ("c", 2.0, "match")])
    assert R.first_k_arm(r, {"4b": 1})[:3] == (0, 0, 0)                    # "b" is the second sample: not drawn
    assert R.first_k_arm(r, {"4b": 2})[3] == ("ex", "b")
    pick_right, passed, maj_right, key, n = R.first_k_arm(r, {"4b": 1, "27g": 1})
    assert (pick_right, passed, key, n) == (1, 1, ("ex", "c"), 2) and maj_right in (0, 1)


def test_an_unparsed_greedy_answer_is_no_candidate_and_ties_go_to_the_count():
    r = row({"27g": [None], "4b": [["ex", "a"], ["ex", "b"], ["ex", "b"]]}, [("a", 1.0, "match"), ("b", 1.0, "valid_wrong")])
    assert R.first_k_arm(r, {"27g": 1})[:3] == (0, 0, 0)
    assert R.first_k_arm(r, {"27g": 1, "4b": 3})[3] == ("ex", "b")


def test_draws_always_hold_the_greedy_answer_and_sample_the_rest():
    r = row({"27g": [["ex", "g"]], "4b": [["ex", "a"], ["ex", "b"], ["ex", "b"], ["ex", "b"]]},
            [("g", 3.0, "match"), ("a", 9.0, "valid_wrong"), ("b", 1.0, "valid_wrong")])
    rows = {"x": r}
    share = R.arm_shares(rows, ["x"], {"27g": 1}, draws=5)
    assert share["x"] == (1.0, 1.0, 1.0)
    pick, passed, _ = R.arm_shares(rows, ["x"], {"27g": 1, "4b": 1}, draws=200)["x"]
    assert passed == 1.0                     # the right answer is in every draw
    assert 0.6 < pick < 0.9                  # "a" outscores it in the quarter of draws that hold "a"


def test_the_arm_estimate_does_not_depend_on_row_order_or_on_other_arms():
    r = {"x": row({"4b": [["ex", "a"], ["ex", "b"]] * 4}, [("a", 1.0, "match"), ("b", 2.0, "valid_wrong")])}
    random.seed(1)
    one = R.arm_shares(r, ["x"], {"4b": 1})
    R.arm_shares(r, ["x"], {"4b": 4})
    assert R.arm_shares(r, ["x"], {"4b": 1}) == one


def test_cost_divides_by_the_solved_share():
    r = R.per_solved(1.0, 50.0)
    assert r["gpu_s_per_solved"] == 2.0 and r["usd_per_1000_solved"] == round(2.0 * 1000 / 3600 * 0.30, 4) and r["exec_match"] == 50.0


def timing_stub():
    return ({"greedy": {"p50_s": 0.5, "p95_s": 1.0, "mean_s": 0.6, "tokens_mean": 10.0},
             "throughput": [{"req_per_s": 1.5}, {"req_per_s": 2.5}, {"req_per_s": 2.0}],
             "samples": {"4": {"p50_s": 1.4, "mean_s": 1.5, "tokens_mean": 40.0}}},
            {"greedy4b": {"p50_s": 0.2, "p95_s": 0.4, "tokens_mean": 16.0}, "throughput4b": [{"req_per_s": 19.0}, {"req_per_s": 25.0}],
             "pooled44": {"mean_s": 0.9, "p50_s": 0.8, "p95_s": 1.5, "tokens_mean": 104.0, "sample_p50_s": 0.5, "score_p50_s": 0.3}})


def test_costs_use_the_best_rate_for_an_answer_and_the_mean_for_a_full_card_system():
    timing, pool = timing_stub()
    c = R.costs(timing, pool, {"4b": 54.3, "base": 66.0, "27b": 67.6})
    assert c["trained 27B, one answer"]["gpu_s_per_request"] == 0.4
    assert c["served 4B, one answer"]["gpu_s_per_request"] == 0.04
    assert c["pooled pick, 4 + 4"]["gpu_s_per_request"] == 0.9


def test_the_three_model_pool_adds_gpu_time_and_is_marked_as_composed():
    timing, pool = timing_stub()
    r = R.composed(timing, pool, "4", 70.0)
    assert r["gpu_s_per_request"] == 2.4 and r["measured"] is False
    assert r["p50_s"] == round(max(0.5, 1.4) + 0.3, 3)                     # two cards overlap the sampling
    assert R.composed(timing, pool, "g", 70.0)["gpu_s_per_request"] == 1.5


def test_the_composed_latency_is_never_below_the_pool_it_contains():
    timing, pool = timing_stub()
    pool["pooled44"].update(p50_s=1.0, sample_p50_s=0.84, score_p50_s=0.14)      # summed medians read 0.98
    assert R.composed(timing, pool, "g", 70.0)["p50_s"] == 1.0


def test_the_post_hoc_family_holds_only_arms_the_report_showed():
    shown = dict(R.descriptive(32))
    assert all(name in shown for name in H.FAMILY)
    assert all(set(shown[name]) <= {"27b", "27g"} for name in H.FAMILY)


def test_the_post_hoc_cost_adds_the_27b_and_one_verifier_pass():
    timing, pool = timing_stub()
    pool["pooled44"]["score_mean_s"] = 0.2
    assert H.composed({"27b": 4}, timing, pool, 50.0)["gpu_s_per_request"] == 1.7
    assert H.composed({"27g": 1, "27b": 4}, timing, pool, 50.0)["gpu_s_per_request"] == 2.3


def test_overrides_counts_where_the_pick_left_the_27bs_answer():
    def case(left, right):
        return row({"4b": [["ex", "b"]], "27g": [["ex", "a"]]}, [("a", 1.0, right), ("b", 2.0, left)])
    rows = {"pick": case("match", "wrong"), "own": case("wrong", "match"), "both": case("match", "match"), "none": case("wrong", "wrong"),
            "same": row({"4b": [["ex", "a"]], "27g": [["ex", "a"]]}, [("a", 1.0, "match")])}
    assert H.overrides(rows, {"4b": 1, "27g": 1}) == {"both": 1, "pick_only": 1, "27b_only": 1, "neither": 1, "agree": 1}


def test_the_driver_never_waits_bare():
    text = (HERE / "run.sh").read_text()
    assert "exec > >(" not in text                                          # bash 5.1 waited on the tee (OPD, 2026-11-02)
    assert not [line for line in text.splitlines() if line.strip() == "wait"]
