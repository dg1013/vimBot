"""Tests for the retrieval experiment (CPU only, no model): `pytest scripts/retrieval`."""
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
import retrieval as R            # noqa: E402


def row(i, instruction, buffer="alpha beta\n", gold="s/alpha/omega/", mode="ex", **extra):
    r = {"id": "r%d" % i, "category": "substitute", "tier": "T1", "mode": mode, "instruction": instruction,
         "buffer_before": buffer, "buffer_after": "omega beta\n", "gold_command": gold, "cursor": [1, 1]}
    r.update(extra)
    return r


TRAIN = [row(1, "Replace alpha with omega on the first line"),
         row(2, "Delete the second word", buffer="one two three\n", gold="dw", mode="keys"),
         row(3, "Uppercase every vowel in the buffer", buffer="quiet\n", gold="s/[aeiou]/\\u&/g"),
         row(4, "Join the next two lines", buffer="a\nb\nc\n", gold="j", mode="keys"),
         row(5, "Delete 3 characters from the start of this token", buffer="abcdef\n", gold="3x", mode="keys")]


def count_chars(messages):
    """A stand-in token counter: characters of all message contents."""
    return sum(len(m["content"]) for m in messages)


# ------------------------------------------------------------ BM25

def test_tokens_are_lowercase_alnum_runs():
    assert R.tokens("Delete 2 characters from 'the' START.") == ["delete", "2", "characters", "from", "the", "start"]


def test_bm25_ranks_the_identical_instruction_first_and_ignores_disjoint_docs():
    idx = R.BM25([R.tokens(r["instruction"]) for r in TRAIN])
    top = idx.top(R.tokens("Delete the second word"), 3)
    assert top[0][1] == 1                          # r2
    assert all(i != 3 for _, i in top) or top[-1][0] < top[0][0]   # "Join the next two lines" shares only "the"
    assert idx.top(R.tokens("zzz qqq"), 3) == []   # nothing shared: nothing returned


def test_bm25_ties_break_by_index_for_determinism():
    idx = R.BM25([["a", "b"], ["a", "b"], ["c"]])
    assert [i for _, i in idx.top(["a"], 2)] == [0, 1]


# ------------------------------------------------------------ the guard and the flag

def test_guard_excludes_rows_sharing_a_buffer_or_a_gold_command():
    query = row(99, "Swap alpha for omega", buffer="alpha beta\n", gold="s/alpha/omega/")
    assert R.shared_leakage_keys(query, TRAIN[0])            # same buffer and same command
    assert R.shared_leakage_keys(query, row(7, "Anything", buffer="alpha beta\n", gold="dd"))   # buffer only
    assert R.shared_leakage_keys(query, row(8, "Anything", buffer="zzz\n", gold=":s/alpha/omega/"))  # command only
    assert R.shared_leakage_keys(query, TRAIN[1]) == []       # unrelated


def test_command_shape_masks_literals_but_keeps_keys_operators_apart():
    assert R.command_shape("ex", "g/foo/d") == R.command_shape("ex", "g/bar/d")
    assert R.command_shape("keys", "2x") == R.command_shape("keys", "3x")
    assert R.command_shape("keys", "dw") != R.command_shape("keys", "yw")
    assert R.command_shape("ex", "%s/a/b/") != R.command_shape("keys", "%s/a/b/")


# ------------------------------------------------------------ planning

def test_retrieved_plan_orders_nearest_last_and_flags_same_shape():
    query = row(50, "Delete 2 characters from the start of this token", buffer="xyzuvw\n", gold="2x", mode="keys")
    idx = R.BM25([R.tokens(r["instruction"]) for r in TRAIN])
    p = R.plan_case(query, TRAIN, "retrieved", 3, count_chars, 10 ** 6, idx)
    assert p["examples"][-1] == "r5"                  # the nearest sits next to the query
    assert p["scores"] == sorted(p["scores"])         # least similar first
    assert p["effective_k"] == 3 and p["dropped_for_budget"] == 0 and p["guard_excluded"] == 0
    assert p["same_shape_neighbor"] is True           # 2x ~ 3x
    assert p["nearest_score"] == p["scores"][-1]


def test_guard_hits_are_skipped_and_counted():
    query = row(51, "Replace alpha with omega on the first line", buffer="alpha beta\n", gold="s/alpha/omega/")
    idx = R.BM25([R.tokens(r["instruction"]) for r in TRAIN])
    p = R.plan_case(query, TRAIN, "retrieved", 2, count_chars, 10 ** 6, idx)
    assert "r1" not in p["examples"] and p["guard_excluded"] == 1
    assert p["effective_k"] == 2


def test_budget_drops_the_least_similar_first_and_records_it():
    query = row(52, "Delete the second word", buffer="one two three\n", gold="dw", mode="keys")
    idx = R.BM25([R.tokens(r["instruction"]) for r in TRAIN])
    full = R.plan_case(query, TRAIN, "retrieved", 3, count_chars, 10 ** 6, idx)
    tight = R.plan_case(query, TRAIN, "retrieved", 3, count_chars, full["prompt_tokens"] - 1, idx)
    assert tight["effective_k"] < 3 and tight["dropped_for_budget"] >= 1
    assert tight["examples"] == full["examples"][-tight["effective_k"]:]   # the nearest survive
    assert tight["prompt_tokens"] <= full["prompt_tokens"] - 1


def test_random_plan_is_seeded_per_row_and_differs_between_rows():
    a = row(60, "Do something", buffer="q\n", gold="x", mode="keys")
    b = row(61, "Do something", buffer="q\n", gold="x", mode="keys")
    pa = R.plan_case(a, TRAIN, "random", 2, count_chars, 10 ** 6)
    pa2 = R.plan_case(a, TRAIN, "random", 2, count_chars, 10 ** 6)
    pb = R.plan_case(b, TRAIN, "random", 2, count_chars, 10 ** 6)
    assert pa["examples"] == pa2["examples"] and len(pa["examples"]) == 2
    assert pa["examples"] != pb["examples"] or pa["id"] != pb["id"]
    assert pa["nearest_score"] is None and pa["scores"] == [0.0, 0.0]


def test_zero_shot_plan_has_no_examples():
    p = R.plan_case(TRAIN[0], TRAIN, "retrieved", 0, count_chars, 10 ** 6, None)
    assert p["examples"] == [] and p["effective_k"] == 0 and p["arm"] == "zero-shot"


def test_plan_split_covers_every_case_in_order():
    cases = [row(70, "Uppercase the vowels", buffer="z\n"), row(71, "Join two lines", buffer="y\n")]
    rows = R.plan_split(cases, TRAIN, "retrieved", 2, count_chars, 10 ** 6)
    assert [r["id"] for r in rows] == ["r70", "r71"]


# ------------------------------------------------------------ the generation hook

class _Enc(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        self.messages.append(msgs)
        return "\n".join(m["content"] for m in msgs)

    def __call__(self, text, return_tensors="pt"):
        import torch
        return _Enc(input_ids=torch.full((1, 3 + len(text) % 5), 7, dtype=torch.long))

    def decode(self, ids, skip_special_tokens=True):
        return "ex: %dd" % len(ids)


class FakeModel:
    def parameters(self):
        import torch
        yield torch.zeros(1)

    def generate(self, input_ids=None, max_new_tokens=0, do_sample=False, eos_token_id=None, pad_token_id=None, **kw):
        import torch
        return torch.cat([input_ids, torch.tensor([[9, 9, 1]])], dim=1)


def test_generate_predictions_prepends_the_examples_only_when_asked(monkeypatch):
    pytest.importorskip("torch")
    import baseline
    monkeypatch.setattr(baseline, "stop_token_ids", lambda tok, stops: [1])
    reg = {"stop": [], "chat_template_kwargs": {}}
    case = row(80, "Delete the second word", buffer="one two\n", gold="dw", mode="keys")
    tok = FakeTok()
    records, preds = baseline.generate_predictions(FakeModel(), tok, [case], reg, 8)
    assert [m["role"] for m in tok.messages[0]] == ["system", "user"]
    assert records[0]["k_examples"] == 0 and preds["r80"]["mode"] == "ex"
    tok = FakeTok()
    records, _ = baseline.generate_predictions(FakeModel(), tok, [case], reg, 8,
                                               examples_for=lambda c: [TRAIN[1], TRAIN[3]])
    roles = [m["role"] for m in tok.messages[0]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert tok.messages[0][2]["content"] == "keys: dw"           # the example's target, in the §6 form
    assert "Delete the second word" in tok.messages[0][-1]["content"]   # the query stays last
    assert records[0]["k_examples"] == 2


# ------------------------------------------------------------ statistics

def test_mcnemar_matches_the_reranker_helper_and_gap_closed_is_bounded():
    sys.path.insert(0, str(REPO / "scripts" / "reranker"))
    from verifier_lib import mcnemar_p as ref
    for g, l in ((10, 2), (0, 0), (5, 5), (37, 13)):
        assert R.mcnemar_p(g, l) == ref(g, l)
    ids = list(range(100))
    zero = {i: int(i < 10) for i in ids}            # 10%
    tuned = {i: int(i < 60) for i in ids}           # 60%
    retr = {i: int(i < 35) for i in ids}            # 35%: half the gap
    pt, lo, hi = R.gap_closed(zero, retr, tuned, ids, n_boot=500)
    assert abs(pt - 0.5) < 1e-9 and lo <= pt <= hi
    assert R.paired(zero, retr) == (25, 0, 100)


def test_report_runs_on_a_tiny_synthetic_run(tmp_path):
    cases = [row(i, "Instruction %d" % i, buffer="b%d\n" % i, gold="c%d" % i) for i in range(6)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    arms = {"base-k0": [0, 0, 0, 1, 0, 0], "base-random5": [0, 0, 1, 1, 0, 0], "base-retr5": [1, 0, 1, 1, 0, 0],
            "adapter-k0": [1, 1, 1, 1, 0, 0], "adapter-retr5": [1, 1, 1, 1, 1, 0]}
    plan = [{"id": c["id"], "arm": "retrieved", "k": 5, "examples": [], "scores": [], "effective_k": 5,
             "prompt_tokens": 400, "dropped_for_budget": 0, "guard_excluded": 0,
             "same_shape_neighbor": i % 2 == 0, "nearest_score": 1.0} for i, c in enumerate(cases)]
    (tmp_path / "plan_retr5.jsonl").write_text("".join(json.dumps(p) + "\n" for p in plan))
    for arm, hits in arms.items():
        d = tmp_path / arm; d.mkdir()
        (d / "exec_test.jsonl").write_text("".join(json.dumps({"id": c["id"], "match": bool(h), "valid": True}) + "\n"
                                                   for c, h in zip(cases, hits)))
        (d / "completions_test.jsonl").write_text("".join(json.dumps({"id": c["id"], "prompt_tokens": 300, "output_tokens": 10,
                                                                         "k_examples": 5, "latency_s": 0.5}) + "\n" for c in cases))
        (d / "baseline_test.json").write_text(json.dumps({"metrics": {"parse_failures": 0, "prompt_tokens_mean": 300.0,
                                                                        "output_tokens_mean": 10.0, "latency_s_p50": 0.5}}))
    rc = R.main(["report", "--run", str(tmp_path), "--cases", str(tmp_path / "cases.jsonl"),
                 "--arms", *arms, "--breakdown-arms", "base-k0", "base-retr5", "adapter-k0"])
    assert rc == 0
    rep = json.loads((tmp_path / "report.json").read_text())
    assert rep["table"]["base-retr5"]["exec_match"] == 3
    assert rep["tests"]["primary: retrieval vs random, k = 5"]["gained"] == 1
    assert abs(rep["gap_closed"]["base-retr5"]["point"] - 2 / 3) < 1e-9       # (3-1)/(4-1)
    assert "same-shape neighbour retrieved" in rep["same_shape"]
    assert (tmp_path / "report.md").exists()
