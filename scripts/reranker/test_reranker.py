"""Tests for the reranker (CPU only, no model): `pytest scripts/reranker`."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_examples as B        # noqa: E402
import rerank_eval as E           # noqa: E402
import verifier_lib as V          # noqa: E402

CASE = {"id": "a", "category": "substitute", "tier": "T3", "mode": "ex",
        "instruction": "Swap the words", "buffer_before": "one two\n", "cursor": [1, 1],
        "buffer_after": "SENTINEL-AFTER\n", "gold_command": "SENTINEL-GOLD"}


def cmd(command, state, n, mode="ex"):
    return {"mode": mode, "command": command, "state": state, "n": n}


# ------------------------------------------------------------ the prompt

def test_prompt_never_shows_the_answer():
    msgs = V.verifier_messages(CASE, "ex", "s/one/two/")
    text = json.dumps(msgs)
    assert "SENTINEL-AFTER" not in text and "SENTINEL-GOLD" not in text
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert msgs[1]["content"].endswith("\nCandidate: ex: s/one/two/")
    assert "Instruction: Swap the words" in msgs[1]["content"]
    assert "<<<\none two\n\n>>>" in msgs[1]["content"]          # the §6 buffer block


class WordTok:
    """A chat template over whole words: prompt + answer extends the generation
    prompt by the answer word, as Qwen's template does by the answer tokens."""

    def __init__(self):
        self.vocab = {}

    def _id(self, w):
        return self.vocab.setdefault(w, len(self.vocab) + 1)

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False, **kw):
        body = []
        for m in msgs[:2]:
            body += ["<s>", m["role"]] + m["content"].split() + ["</s>"]
        body += ["<s>", "assistant"]
        if msgs[-1]["role"] == "assistant":
            body += [msgs[-1]["content"], "</s>"]
        return [self._id(w) for w in body]


def test_answer_token_ids_finds_distinct_first_tokens():
    tok = WordTok()
    y, n = V.answer_token_ids(tok, {})
    assert y != n and y == tok.vocab["yes"] and n == tok.vocab["no"]


# ------------------------------------------------------------ building examples

def test_build_labels_merges_probes_and_drops_unrankable():
    probes = [{"id": "a", "G": 8, "commands": [cmd("s/x/y/", "match", 5), cmd("dd", "valid_wrong", 2),
                                               {"mode": None, "command": None, "state": "parse", "n": 1}]},
              {"id": "a", "G": 32, "commands": [cmd("s/x/y/", "match", 20), cmd("x", "invalid", 12)]}]
    rows, stats = B.build([CASE], probes)
    by = {r["cand_command"]: r for r in rows}
    assert set(by) == {"s/x/y/", "dd", "x"}                   # the parse failure is not ranked
    assert by["s/x/y/"]["label"] == 1 and by["s/x/y/"]["n"] == 25
    assert by["dd"]["label"] == 0 and by["x"]["label"] == 0
    assert rows[0]["G"] == 40 and stats["positives"] == 1


def test_state_conflict_ties_go_to_match_and_are_counted():
    probes = [{"id": "a", "G": 8, "commands": [cmd("s/x/y/", "match", 1)]},
              {"id": "a", "G": 8, "commands": [cmd("s/x/y/", "invalid", 1)]}]
    rows, stats = B.build([CASE], probes)
    assert rows[0]["label"] == 1 and stats["state_conflicts"] == 1


def test_greedy_answer_is_flagged_or_added():
    probes = [{"id": "a", "G": 8, "commands": [cmd("dd", "valid_wrong", 8)]}]
    greedy = [{"id": "a", "direct": {"parse_ok": True, "mode": "ex", "command": "s/x/y/", "match": True}}]
    rows, _ = B.build([CASE], probes, greedy)
    added = [r for r in rows if r["greedy"]]
    assert len(added) == 1 and added[0]["n"] == 0 and added[0]["label"] == 1
    greedy[0]["direct"]["command"] = "dd"
    greedy[0]["direct"]["match"] = False
    rows, _ = B.build([CASE], probes, greedy)
    assert len(rows) == 1 and rows[0]["greedy"] and rows[0]["n"] == 8


def test_unanimous_prompts_are_downsampled_and_caps_keep_positives():
    cases = [dict(CASE, id="p%d" % i) for i in range(200)]
    probes = [{"id": "p%d" % i, "G": 8, "commands": [cmd("s/x/y/", "match", 8)]} for i in range(200)]
    rows, stats = B.build(cases, probes, keep_unanimous=0.25, seed=1)
    assert 25 <= len(rows) <= 75 and stats["unanimous_dropped"] == 200 - len(rows)
    many = [{"id": "a", "G": 32, "commands": [cmd("ok", "match", 1)] +
             [cmd("w%d" % k, "valid_wrong", k + 1) for k in range(12)]}]
    rows, _ = B.build([CASE], many, max_per_prompt=4)
    assert len(rows) == 4 and sum(r["label"] for r in rows) == 1
    assert {r["cand_command"] for r in rows if not r["label"]} == {"w11", "w10", "w9"}


def test_split_hygiene_is_enforced():
    probes = [{"id": "zz", "G": 8, "commands": [cmd("dd", "match", 8)]}]
    try:
        B.build([CASE], probes)
        assert False, "an id outside the cases must be refused"
    except ValueError:
        pass
    probes = [{"id": "a", "G": 8, "commands": [cmd("dd", "match", 8)]}]
    try:
        B.build([CASE], probes, forbid_ids={"a"})
        assert False, "a held-out id must be refused"
    except ValueError:
        pass


# ------------------------------------------------------------ rules and statistics

def row(cid, command, label, n, greedy=False):
    return {"id": cid, "cand_mode": "ex", "cand_command": command, "label": label, "n": n, "greedy": greedy}


def test_rules_greedy_majority_verifier_oracle():
    rows = [row("a", "right", 1, 3), row("a", "wrong", 0, 5, greedy=True),
            row("b", "r1", 1, 4), row("b", "w1", 0, 4)]
    groups = V.group(rows)
    base = V.rule_outcomes(groups)
    assert base["a"]["greedy"] == 0 and base["a"]["majority"] == 0 and base["a"]["oracle_samples"] == 1
    assert base["b"]["majority"] == 0.5                      # a tie is split, not broken
    scores = {("a", "ex", "right"): 2.0, ("a", "ex", "wrong"): -1.0,
              ("b", "ex", "r1"): 0.1, ("b", "ex", "w1"): 0.0}
    out = V.rule_outcomes(groups, scores)
    assert out["a"]["verifier_samples"] == 1 and out["b"]["verifier_samples"] == 1
    # a strong frequency prior flips prompt a back to the frequent wrong answer
    assert V.rule_outcomes(groups, scores, lam=10.0)["a"]["verifier_samples"] == 0


def test_auroc_and_mcnemar():
    assert V.auroc([3, 4], [1, 2]) == 1.0
    assert V.auroc([1], [1]) == 0.5
    assert abs(V.auroc([1, 3], [2]) - 0.5) < 1e-12
    assert V.mcnemar_p(0, 0) == 1.0
    assert abs(V.mcnemar_p(5, 0) - 0.0625) < 1e-12
    assert V.mcnemar_p(3, 7) == V.mcnemar_p(7, 3)
    assert V.paired({"a": 0, "b": 1, "c": None}, {"a": 1, "b": 1, "c": 1}) == (1, 0, 2)


def test_rerank_eval_end_to_end(tmp_path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text("".join(json.dumps(dict(CASE, id=i)) + "\n" for i in ("a", "b")))
    ex = tmp_path / "ex.jsonl"
    ex.write_text("".join(json.dumps(r) + "\n" for r in [
        row("a", "right", 1, 3), row("a", "wrong", 0, 5, greedy=True),
        row("b", "r1", 1, 7, greedy=True), row("b", "w1", 0, 1)]))
    sc = tmp_path / "s.jsonl"
    sc.write_text("".join(json.dumps({"id": i, "cand_mode": "ex", "cand_command": c, "score": s}) + "\n"
                          for i, c, s in [("a", "right", 1.0), ("a", "wrong", 0.0), ("b", "r1", 2.0), ("b", "w1", 0.0)]))
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"a": {"top_share": 0.625, "n": 8}, "b": {"top_share": 0.875, "n": 8}}))
    ver = tmp_path / "ver.jsonl"
    ver.write_text("".join(json.dumps({"id": i, "real": {"match": m}}) + "\n" for i, m in (("a", False), ("b", True))))
    out = tmp_path / "r.md"
    assert E.main(["--examples", str(ex), "--scores", "v=%s" % sc, "--split", "test", "--lam", "v=0",
                   "--cases", str(cases), "--ask-report", str(rep), "--verdicts", str(ver),
                   "--out", str(out), "--json", str(tmp_path / "r.json")]) == 0
    text = out.read_text()
    assert "| greedy §6 answer (served adapter) | 50.0% |" in text
    assert "**100.0%**" in text                                   # the verifier fixes prompt a
    assert json.loads((tmp_path / "r.json").read_text())["v"]["all"] == 100.0
