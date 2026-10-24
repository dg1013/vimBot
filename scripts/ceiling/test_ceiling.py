"""Tests for scripts/ceiling/ (the pass@k ceiling probe): the estimators'
arithmetic against brute-force enumeration, the sampler's per-prompt
bookkeeping and its verdicts through the real sandbox, and the report on a
synthetic run.

Run:  python3 -m pytest scripts/ceiling            (not collected by `pytest tests/`)
"""

import itertools
import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in (REPO / "scripts", REPO / "src", HERE):
    sys.path.insert(0, str(p))

import passk                                   # noqa: E402
import report                                  # noqa: E402
import sample                                  # noqa: E402


# ------------------------------------------------------------------ pass@k

def _pass_brute(n, c, k):
    """Share of k-subsets of n samples (c correct) holding a correct one."""
    items = [True] * c + [False] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(items[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n,c,k", [(6, 0, 3), (6, 1, 1), (6, 2, 3), (7, 3, 4), (8, 5, 2), (5, 5, 5),
                                   (9, 1, 9), (9, 8, 1)])
def test_pass_at_k_is_the_subset_probability(n, c, k):
    assert passk.pass_at_k(n, c, k) == pytest.approx(_pass_brute(n, c, k), abs=1e-12)


def test_pass_at_k_edges():
    assert passk.pass_at_k(128, 0, 128) == 0.0
    assert passk.pass_at_k(128, 128, 1) == 1.0
    assert passk.pass_at_k(128, 37, 1) == pytest.approx(37 / 128)
    assert passk.pass_at_k(10, 3, 8) == 1.0          # n - c < k: every 8-subset has a hit
    for bad in ((4, 5, 1), (4, -1, 1), (4, 2, 0), (4, 2, 5)):
        with pytest.raises(ValueError):
            passk.pass_at_k(*bad)


def test_pass_at_k_is_unbiased_where_the_plug_in_is_not():
    """Averaged over c ~ Binomial(n, p) the estimator equals 1 - (1-p)^k
    exactly; the plug-in 1 - (1 - c/n)^k falls short of it."""
    n, k, p = 8, 4, 0.3
    pmf = [math.comb(n, c) * p ** c * (1 - p) ** (n - c) for c in range(n + 1)]
    unbiased = sum(q * passk.pass_at_k(n, c, k) for c, q in enumerate(pmf))
    plug_in = sum(q * (1 - (1 - c / n) ** k) for c, q in enumerate(pmf))
    truth = 1 - (1 - p) ** k
    assert unbiased == pytest.approx(truth, abs=1e-12)
    assert plug_in < truth - 0.01


def test_k_grid():
    assert passk.k_grid(1) == [1]
    assert passk.k_grid(128) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert passk.k_grid(100) == [1, 2, 4, 8, 16, 32, 64, 100]


# ------------------------------------------------------------------- maj@k

def test_vote_value_ties_and_abstentions():
    assert passk.vote_value([]) == 0.0
    assert passk.vote_value([(None, False)] * 3) == 0.0
    assert passk.vote_value([("a", True), ("a", True), ("b", False)]) == 1.0
    assert passk.vote_value([("a", True), ("b", False)]) == 0.5          # a tie is its expectation
    assert passk.vote_value([("a", True), ("b", False), ("c", False)]) == pytest.approx(1 / 3)
    # abstentions do not vote, however many there are
    assert passk.vote_value([(None, False)] * 5 + [("a", True)]) == 1.0


def _maj_brute(pop, k):
    subsets = list(itertools.combinations(pop, k))
    return sum(passk.vote_value(s) for s in subsets) / len(subsets)


def test_maj_at_k_exact_ends_and_monte_carlo_middle():
    pop = [("a", True)] * 3 + [("b", False)] * 4 + [(None, False)] * 2
    n = len(pop)
    assert passk.maj_at_k(pop, 1) == pytest.approx(3 / n)                # = pass@1
    assert passk.maj_at_k(pop, n) == 0.0                                 # b wins 4 to 3
    for k in (2, 3, 5):
        est = passk.maj_at_k(pop, k, draws=4000, rng=random.Random(1))
        assert est == pytest.approx(_maj_brute(pop, k), abs=0.03)


def test_mean_se_and_paired_diff():
    m, se = passk.mean_se([1.0, 0.0, 1.0, 0.0])
    assert m == 0.5 and se == pytest.approx(math.sqrt((4 * 0.25 / 3) / 4))
    d, se = passk.paired_diff([1, 1, 0, 1], [0, 1, 0, 1])
    assert d == 0.25 and se > 0
    with pytest.raises(ValueError):
        passk.paired_diff([1], [1, 0])


# ------------------------------------------------------------------ sampler

CASE = {"id": "t-1", "category": "substitute", "tier": "T2", "mode": "ex",
        "buffer_before": "foo bar\nfoo\n", "buffer_after": "baz bar\nbaz\n", "cursor": [1, 1],
        "gold_command": "%s/foo/baz/"}


def test_cut_stops_pads_and_caps():
    assert sample.cut([5, 6, 99, 0, 0], stops=[99], pad_id=0) == ([5, 6], True)
    assert sample.cut([5, 6, 7], stops=[99], pad_id=0) == ([5, 6, 7], False)
    assert sample.cut([99, 0], stops=[99], pad_id=0) == ([], True)


def test_buffer_hash_follows_exec_match_normalization():
    assert sample.buffer_hash("a\nb\n") == sample.buffer_hash("a\nb")      # one trailing newline
    assert sample.buffer_hash("") != sample.buffer_hash("\n")             # [] vs [""]
    assert sample.buffer_hash(None) is None


def test_execute_through_the_real_sandbox():
    state, buf = sample.execute(CASE, "ex", CASE["gold_command"], timeout=5.0)
    assert state == "match" and buf == sample.buffer_hash(CASE["buffer_after"])
    state, buf = sample.execute(CASE, "ex", "%s/bar/qux/", timeout=5.0)
    assert state == "valid_wrong" and buf == sample.buffer_hash("foo qux\nfoo\n")
    assert sample.execute(CASE, "ex", "notacommand", timeout=5.0) == ("invalid", None)


def test_an_ex_command_with_a_nul_is_invalid_and_never_reaches_vim():
    """The first base launch crashed here: run_vim puts ex commands in argv,
    where a NUL raises ValueError. Keys mode goes through a file and runs."""
    def boom(*a, **k):
        raise AssertionError("the scorer must not be called")
    assert sample.execute(CASE, "ex", "%s/foo/\x00/", 5.0, scorer=boom) == ("invalid", None)
    assert sample.unrunnable("ex", "a\x00b") and not sample.unrunnable("keys", "a\x00b")
    assert not sample.unrunnable("ex", "%s/foo/baz/")
    assert sample.execute(CASE, "keys", "x\x00", 5.0)[0] in ("valid_wrong", "invalid", "match")


def test_execute_agrees_with_the_trainer_states():
    import train_grpo
    for mode, command in (("ex", CASE["gold_command"]), ("ex", "%s/bar/qux/"), ("ex", "notacommand"),
                          ("keys", "cwbaz<Esc>jcwbaz<Esc>"), ("keys", "dd")):
        assert sample.execute(CASE, mode, command, 5.0)[0] == \
            train_grpo.execution_state(CASE, mode, command, 5.0)


def test_observe_counts_and_groups():
    samples = [("ex: %s/foo/baz/", 7, True),
               ("ex: %s/foo/baz/", 7, True),
               ("ex: %s/foo/baz/\nand then some rambling", 192, False),   # capped, first line right
               ("keys: dd", 3, True),
               ("I would use :s", 5, True),                                # not the §6 grammar
               ("<think>hmm</think>\nex: %s/foo/baz/", 12, True)]          # think content: parse failure
    calls = []

    def run(keys):
        calls.append(keys)
        table = {("ex", "%s/foo/baz/"): ("match", "h1"), ("keys", "dd"): ("valid_wrong", "h2")}
        return {k: table[k] for k in keys}

    row = sample.observe(CASE, samples, run)
    assert calls == [[("ex", "%s/foo/baz/"), ("keys", "dd")]]        # each distinct answer runs once
    assert (row["n"], row["c"], row["c_strict"]) == (6, 3, 2)
    assert (row["unparsed"], row["think"], row["truncated"]) == (2, 1, 1)
    assert row["states"] == {"match": 3, "valid_wrong": 1, "parse": 2}
    assert row["distinct"] == 2 and row["distinct_buffers"] == 2
    assert row["commands"][0] == {"mode": "ex", "command": "%s/foo/baz/", "n": 3, "truncated": 1,
                                  "state": "match", "buf": "h1"}
    assert row["tokens_mean"] == pytest.approx((7 + 7 + 192 + 3 + 5 + 12) / 6, abs=0.01)
    assert row["tokens_max"] == 192
    assert row["unrunnable"] == 0
    nul = sample.observe(CASE, [("ex: s/a/\x00/", 4, True)], lambda keys: {k: ("invalid", None) for k in keys})
    assert nul["unrunnable"] == 1 and nul["states"] == {"invalid": 1}


def test_prompt_seed_is_stable_and_distinct():
    assert sample.prompt_seed(42, "a") == sample.prompt_seed(42, "a")
    assert sample.prompt_seed(42, "a") != sample.prompt_seed(42, "b")
    assert sample.prompt_seed(42, "a") != sample.prompt_seed(43, "a")


def test_check_meta_refuses_a_different_distribution(tmp_path):
    cfg = {k: None for k in sample.SAMPLING_KEYS}
    cfg.update(model="m", n=128, temperature=1.0)
    meta = tmp_path / "s.jsonl.meta.json"
    assert sample.check_meta(meta, cfg) == {"sampling": cfg, "launches": []}
    meta.write_text(json.dumps({"sampling": cfg, "launches": [{}]}))
    assert sample.check_meta(meta, dict(cfg))["launches"] == [{}]
    with pytest.raises(SystemExit):
        sample.check_meta(meta, dict(cfg, temperature=0.7))


# ------------------------------------------------------------------- report

def _row(i, n, c, commands, unparsed=0):
    return {"id": i, "category": "substitute", "tier": "T2", "mode": "ex", "n": n, "c": c,
            "c_strict": c, "unparsed": unparsed, "think": 0, "truncated": 0, "tokens_mean": 10.0,
            "states": {}, "distinct": len(commands),
            "distinct_buffers": len({e["buf"] for e in commands if e["buf"]}), "commands": commands}


def _cmd(command, n, state, buf):
    return {"mode": "ex", "command": command, "n": n, "truncated": 0, "state": state, "buf": buf}


def test_population_by_buffer_merges_spellings_and_drops_invalid():
    row = _row("x", 4, 2, [_cmd("a", 1, "match", "h"), _cmd("b", 1, "match", "h"),
                           _cmd("c", 1, "invalid", None)], unparsed=1)
    by_cmd = report.population(row, "command")
    by_buf = report.population(row, "buffer")
    assert len(by_cmd) == len(by_buf) == 4
    assert passk.vote_value(by_cmd) == pytest.approx(2 / 3)     # a, b, c tie one vote each
    assert passk.vote_value(by_buf) == 1.0                       # a and b are one buffer


def test_report_on_a_synthetic_run(tmp_path):
    cases = [dict(CASE, id="p%d" % i) for i in range(3)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    a = [_row("p0", 4, 4, [_cmd("g", 4, "match", "h")]),
         _row("p1", 4, 1, [_cmd("g", 1, "match", "h"), _cmd("w", 3, "valid_wrong", "v")]),
         _row("p2", 4, 0, [_cmd("w", 4, "valid_wrong", "v")])]
    b = [_row("p0", 4, 2, [_cmd("g", 2, "match", "h"), _cmd("w", 2, "valid_wrong", "v")]),
         _row("p1", 4, 0, [_cmd("w", 4, "valid_wrong", "v")])]                 # p2 not yet sampled
    for name, rows in (("a", a), ("b", b)):
        (tmp_path / ("samples_%s.jsonl" % name)).write_text("".join(json.dumps(r) + "\n" for r in rows))
    greedy = tmp_path / "exec_a.jsonl"
    greedy.write_text("".join(json.dumps({"id": "p%d" % i, "match": i == 0}) + "\n" for i in range(3)))
    assert report.main(["--run", str(tmp_path), "--cases", str(tmp_path / "cases.jsonl"),
                        "--samples", "a=samples_a.jsonl", "--samples", "b=samples_b.jsonl",
                        "--greedy", "a=%s" % greedy, "--pool", "g=%s" % greedy]) == 0
    s = json.loads((tmp_path / "summary.json").read_text())
    assert s["prompts"] == 2                       # only prompts present in every file
    assert s["k"] == [1, 2, 4]
    assert s["pass_at_k"]["a"]["1"][0] == pytest.approx((1 + 0.25) / 2)
    assert s["pass_at_k"]["a"]["4"][0] == 1.0
    assert s["pass_at_k"]["b"]["4"][0] == 0.5
    assert s["covered"] == {"a": 2, "b": 1} and s["union_covered"] == 2
    assert s["greedy"]["a"] == 0.5 and s["pools"]["g"] == {"runs": 1, "solved": 1}
    assert s["uncovered_core"] == []
    md = (tmp_path / "report.md").read_text()
    assert "pass@k" in md and "(of 3 in the split: a run is still in progress)" in md
    assert "| b − a |" in md                       # default pair: each checkpoint minus the one before
    report.main(["--run", str(tmp_path), "--cases", str(tmp_path / "cases.jsonl"),
                 "--samples", "a=samples_a.jsonl", "--samples", "b=samples_b.jsonl", "--pair", "a:b"])
    assert "| a − b |" in (tmp_path / "report.md").read_text()
    with pytest.raises(SystemExit):
        report.main(["--run", str(tmp_path), "--cases", str(tmp_path / "cases.jsonl"),
                     "--samples", "a=samples_a.jsonl", "--pair", "a:zzz"])


def _rows_for(cid, **per_label):
    """{label: {cid: row}} with each label's commands given as tuples."""
    out = {}
    for label in ("sft", "served", "cliffleg"):
        cmds = [_cmd(c, n, st, "h-" + c if st != "invalid" else None)
                for c, n, st in per_label.get(label, [])]
        out[label] = {cid: _row(cid, 128, 0, cmds)}
    return out


def test_pane_answer_pools_runnable_answers_over_the_trained_checkpoints():
    import build_core_review as bcr
    rows = _rows_for("x", sft=[("a", 60, "valid_wrong"), ("bad", 68, "invalid")],
                     served=[("b", 70, "valid_wrong")], cliffleg=[("a", 20, "valid_wrong")])
    alt = bcr.pane_answer(rows, "x")
    assert (alt["cmd"], alt["by"]) == ("a", {"sft": 60, "cliffleg": 20})   # 80 pooled beats 70
    assert bcr.pane_answer(_rows_for("y", sft=[("bad", 128, "invalid")]), "y") is None


def test_priority_orders_pre_read_then_confident_then_elsewhere():
    import build_core_review as bcr
    confident = _rows_for("x", served=[("a", 64, "valid_wrong")])
    diffuse = _rows_for("x", served=[("a", 63, "valid_wrong")])
    assert bcr.priority("kb13-0428a", confident, []) == ("suspected defect", 0)
    assert bcr.priority("hc-0809", confident, []) == ("borderline", 1)
    assert bcr.priority("x", confident, [{"run": "r"}]) == ("confident disagreement", 2)
    assert bcr.priority("x", diffuse, [{"run": "r"}]) == ("solved by another model", 3)
    assert bcr.priority("x", diffuse, []) == (None, 4)


def test_show_makes_any_command_table_safe():
    assert report.show("%s/a|b/c/") == "%s/a\\|b/c/"
    assert report.show("x\x00y\x1b") == "x\\x00y\\x1b"
    assert report.show("`cmd`") == "'cmd'"
