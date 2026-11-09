"""CPU tests for scripts/vtime4b: the live V decides as fewer.py's V does, and its timings add up. No model, no card."""
import math
import pathlib
import random
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vtime4b as V                                               # noqa: E402

REPO = V.REPO
sys.path.insert(0, str(REPO / "scripts" / "retr27b"))
sys.path.insert(0, str(REPO / "scripts" / "serve27b"))
import fewer as F                                                 # noqa: E402

CMDS = [("ex", "%s/a/b/g"), ("ex", "g/^$/d"), ("keys", "dd"), ("keys", "ciwx<Esc>"), ("ex", "1,2d")]


def random_row(rng, k=8):
    samples = [list(rng.choice(CMDS)) if rng.random() > 0.15 else None for _ in range(k)]
    distinct = {tuple(s) for s in samples if s is not None}
    # scores on the verifier's grid (multiples of 0.125), so ties happen
    score = {c: rng.randint(-24, 8) / 8 for c in distinct}
    return samples, score


@pytest.mark.parametrize("seed", range(5))
def test_gate_decides_as_fewer_view(seed):
    """For the same samples and scores, the live gate asks and picks exactly as fewer.view and fewer.gated read V."""
    rng = random.Random(seed)
    for _ in range(400):
        samples, score = random_row(rng)
        row = {"samples": samples, "results": [tuple(s) if s else None for s in samples], "score": score}
        _, best, pick = F.view(row, list(range(len(samples))))
        cands = V.PL.merge([("4b",) + tuple(s) for s in samples if s is not None])
        for c in cands:
            c["score"] = score[(c["mode"], c["command"])]
        tau = rng.choice([-0.875, -1.125, -0.5, 0.0])
        ask, got = V.gate(cands, tau)
        assert ask == (best <= tau)
        assert (None if got is None else (got["mode"], got["command"])) == pick


def test_no_parsed_sample_asks():
    assert V.gate([], -0.875) == (True, None)


class FakeEndpoint:
    """render() returns the messages' last user text; complete() answers by what was asked for."""

    def __init__(self, samples, question="Which line?", direct="ex: 1d", turn2="keys: dd"):
        self.samples, self.question, self.direct, self.turn2 = samples, question, direct, turn2
        self.calls = []

    def render(self, messages, prefill=""):
        return ("TURN2:" if len(messages) > 2 else "") + ("ASK:" if prefill else "") + messages[-1]["content"][:20]

    def complete(self, text, max_tokens, temperature=0.0, n=1, stop=None, seed=None):
        self.calls.append((text[:6], n, temperature))
        if text.startswith("ASK:"):
            texts = [self.question]
        elif text.startswith("TURN2:"):
            texts = [self.turn2]
        elif n > 1:
            texts = list(self.samples)
        else:
            texts = [self.direct]
        return {"texts": texts, "output_tokens": [3] * len(texts), "prompt_tokens": 10, "latency_s": 0.01}


class FakeVerifier:
    def __init__(self, table):
        self.table, self.seen = table, []

    def scores(self, case, cands):
        self.seen.append([(c["mode"], c["command"]) for c in cands])
        return [self.table[(c["mode"], c["command"])] for c in cands], 0.001


CASE = {"id": "x", "instruction": "delete line 1", "buffer_before": "a\nb\n", "cursor": [1, 1]}


def live(samples, table, tau=-0.875, **kw):
    ep = FakeEndpoint(samples, **kw)
    return V.LiveV(ep, FakeVerifier(table), tau), ep


def test_confident_pick_answers_with_the_highest_score_not_the_majority():
    v, _ = live(["ex: 1d"] * 5 + ["keys: dd"] * 3, {("ex", "1d"): -0.5, ("keys", "dd"): 0.25})
    res = v.translate(CASE)
    assert res["type"] == "command" and (res["mode"], res["command"]) == ("keys", "dd")
    assert not res["ask"] and res["best"] == 0.25
    assert set(res["timings"]) == {"direct_s", "samples_s", "models_wall_s", "verifier_s", "total_s"}


def test_low_best_score_asks_and_the_reply_turn_answers():
    v, ep = live(["ex: 1d"] * 8, {("ex", "1d"): -0.875})
    res = v.translate(CASE)
    assert res["type"] == "question" and res["question"] == "Which line?" and "question_s" in res["timings"]
    done = v.reply(CASE, res["question"], "the first")
    assert (done["mode"], done["command"]) == ("keys", "dd")
    # one direct, one request of 8 samples, one question, one turn 2
    assert sorted(n for _, n, _ in ep.calls) == [1, 1, 1, 8]


def test_empty_question_answers_with_the_pick():
    v, _ = live(["ex: 1d"] * 8, {("ex", "1d"): -2.0}, question="   ")
    res = v.translate(CASE)
    assert res["type"] == "command" and (res["mode"], res["command"]) == ("ex", "1d") and res["fallback"]


def test_empty_question_and_no_parsed_sample_gives_the_direct_answer():
    v, _ = live(["no grammar here"] * 8, {}, question="", direct="ex: 2d")
    res = v.translate(CASE)
    assert res["ask"] and res["type"] == "command" and (res["mode"], res["command"]) == ("ex", "2d")


def test_the_verifier_sees_each_distinct_parsed_sample_once():
    v, _ = live(["ex: 1d", "ex: 1d", "keys: dd", "junk"] + ["ex: 1d"] * 4, {("ex", "1d"): 0.0, ("keys", "dd"): -1.0})
    v.translate(CASE)
    assert v.pp.seen == [[("ex", "1d"), ("keys", "dd")]]


def test_gpu_seconds_add_the_parts_and_leave_the_sandbox_out():
    t = {"models_wall_s": 0.5, "verifier_s": 0.2, "question_s": 0.3, "turn2_s": 0.4, "sandbox_s": 9.0, "total_s": 1.0}
    assert math.isclose(V.gpu_s(t), 1.4)
    assert math.isclose(V.gpu_s({"models_wall_s": 0.5, "verifier_s": 0.1}), 0.6)


def test_v_row_records_the_reply_turn_in_the_gpu_seconds():
    v, _ = live(["ex: 1d"] * 8, {("ex", "1d"): -3.0})
    replies = {"x": {"question": "Which line?", "reply": "the first"}}
    row = V.v_row(v, CASE, replies)
    assert row["type"] == "question" and row["same_question"] and row["answer"] == {"mode": "keys", "command": "dd"}
    assert math.isclose(row["gpu_s"], V.gpu_s(row["timings"])) and "turn2_s" in row["timings"]


def test_per_solved_is_pool27bs():
    import pool27b
    for gpu, acc in ((0.834, 61.6), (1.596, 61.6), (0.738, 58.6)):
        assert V.per_solved(gpu, acc) == pool27b.per_solved(gpu, acc)
    assert V.GPU_HOUR == pool27b.GPU_HOUR


def test_rescore_compares_only_scored_pairs():
    rows = [{"id": "x", "cands": [{"mode": "ex", "command": "1d", "score": -0.5},
                                  {"mode": "keys", "command": "dd", "score": 0.0}]}]
    got = V.rescore(rows, {("x", "ex", "1d"): -0.375})
    assert got == {"pairs": 1, "identical": 0, "mean_abs": 0.125, "max_abs": 0.125}


def test_driver_parses():
    assert subprocess.run(["bash", "-n", str(HERE / "run.sh")]).returncode == 0
    text = (HERE / "run.sh").read_text()
    for flag in ("--cpus-per-task", "--time", "sbatch", "srun"):
        assert flag not in text
