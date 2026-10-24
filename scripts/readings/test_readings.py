"""Tests for scripts/readings/ (the readings-first clarification experiment).

Run:  python3 -m pytest scripts/readings          (not collected by `pytest tests/`)

The reading parser; the prompts against the §6 user turn; grouping readings
by the buffer they leave, through the real sandbox; the question prompt; the
files handed to the unchanged clarify harness; and the report's arithmetic.
"""
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))

import post                                            # noqa: E402
import protocol as P                                   # noqa: E402
import readings_protocol as R                          # noqa: E402

CASE = {"id": "k01", "category": "substitute", "note_class": "ambiguous",
        "instruction": "Rename the first word", "mode": "ex",
        "buffer_before": "alpha beta\n", "gold_command": "s/alpha/x/",
        "buffer_after": "x beta\n", "cursor": [1, 1]}


def test_parse_readings_reads_one_command_per_line():
    text = "1. ex: s/a/b/\n\n- keys: dw\nthis is prose\nex: %d\nex: 2d\nex: 3d"
    readings, tally = R.parse_readings(text)
    assert readings == [("ex", "s/a/b/"), ("keys", "dw"), ("ex", "%d"), ("ex", "2d")]
    assert tally == {"lines": 6, "parsed": 4, "unparsed": 1, "marked": 2, "extra": 1,
                     "think": False}
    assert R.parse_readings("<think>\n</think>\nex: d")[0] == []
    assert R.parse_readings("<think>x</think>")[1]["think"] is True
    assert R.parse_readings(None) == ([], {"lines": 0, "parsed": 0, "unparsed": 0,
                                           "marked": 0, "extra": 0, "think": False})


def test_the_prompts_keep_the_section6_user_turn():
    msgs = R.readings_messages(CASE)
    assert msgs[0]["content"] == R.READINGS_SYSTEM
    assert msgs[1] == P.turn1_messages(CASE)[1]          # the §6 user turn, unchanged
    assert "at most %d lines" % R.MAX_READINGS in R.READINGS_SYSTEM.replace("\n", " ")


def _run(readings):
    from score import execute
    answers = {"r%d" % i: {"parse_ok": True, "mode": m, "command": c}
               for i, (m, c) in enumerate(readings)}
    ex = execute([CASE], {CASE["id"]: answers}, workers=2)
    return [ex[(CASE["id"], "r%d" % i)] for i in range(len(readings))]


def test_readings_merge_by_result_and_failures_drop():
    readings = [("ex", "s/alpha/x/"), ("keys", "cwx<Esc>"), ("ex", "s/beta/y/"),
                ("ex", "s/zzz/q/"), ("ex", "s/alpha/alpha/")]
    groups, statuses = R.group_readings(CASE, _run(readings))
    assert statuses == ["ok", "ok", "ok", "invalid", "no_change"]
    assert [g["members"] for g in groups] == [[0, 1], [2]]
    assert [g["match"] for g in groups] == [True, False]
    assert R.reading_status(CASE, None) == "unrun"


def test_the_question_prompt_shows_each_distinct_result():
    readings = [("ex", "s/alpha/x/"), ("ex", "s/beta/y/")]
    groups, _ = R.group_readings(CASE, _run(readings))
    msgs = R.question_messages(CASE, readings, groups)
    assert msgs[0]["content"] == R.QUESTION_SYSTEM
    user = msgs[1]["content"]
    assert user.startswith(P.turn1_messages(CASE)[1]["content"])
    assert "Reading 1: ex: s/alpha/x/" in user and "Reading 2: ex: s/beta/y/" in user
    assert 'after  "x beta"' in user and 'after  "alpha y"' in user


def test_prepare_hands_the_clarify_harness_what_it_reads():
    rows = [{"id": "k01", "asked": True, "question": {"text": "Which word?", "raw": " Which word?"}},
            {"id": "k02", "asked": False, "question": None}]
    cases = [CASE, dict(CASE, id="k02"), dict(CASE, id="k03")]
    questions, asked, placebo = post.prepare(rows, cases)
    assert [c["id"] for c in asked] == ["k01"]
    # simulate.py reads (label, id, question.text); dialogue.py turn2 reads question.text
    assert questions == [{"id": "k01", "label": "readings", "question": rows[0]["question"]}]
    assert placebo[0]["label"] == "readings_placebo" and placebo[0]["reply"] == P.PLACEBO_REPLY


def _v(i, cls, asked, direct, noask, real, placebo, rr=None, rp=None, p=0.1, distinct=None):
    c = lambda x: {"match": bool(x), "tokens": 5, "completion": "ex: x", "outcome": "m"}  # noqa: E731
    return {"id": "v%02d" % i, "note_class": cls, "instruction": "do", "category": "x",
            "asked": asked, "distinct": distinct if distinct is not None else (2 if asked else 1),
            "n_readings": 2, "statuses": ["ok", "ok"], "gold_in_readings": bool(real),
            "readings_tokens": 20, "readings_tally": {"lines": 2, "parsed": 2},
            "question": "q?" if asked else None, "question_tokens": 6 if asked else 0,
            "reply": "r" if asked else None, "reply_leak_flags": [] if asked else None,
            "reply_quotes_result": False if asked else None,
            "readings_real": c(rr) if asked else None,
            "readings_placebo": c(rp) if asked else None,
            "direct": c(direct), "noask": c(noask), "real": c(real), "placebo": c(placebo),
            "p_ask_norm": p, "chose_ask": False, "generic_question": "g?",
            "generic_question_tokens": 6, "generic_reply": "gr"}


def test_the_report_scores_who_to_ask_and_the_deployed_policy():
    rows = [_v(0, "unstated", True, 0, 0, 1, 0, rr=1, rp=0, p=0.2),
            _v(1, "unstated", True, 0, 0, 0, 0, rr=1, rp=0, p=0.3),
            _v(2, "model", False, 1, 1, 0, 1, p=0.9),
            _v(3, "model", False, 0, 0, 1, 0, p=0.05)]
    pol = {lab.strip("*"): (rate, match) for lab, rate, match in post.policy_rows(rows)}
    assert pol["ask iff the readings differ"] == (0.5, 0.5)           # rows 0 (fixed), 2 (noask right)
    assert pol["oracle: ask only where it turns wrong to right"] == (0.5, 0.75)
    assert [post.e2e(r) for r in rows] == [1, 1, 1, 0]
    assert post.e2e_tokens(rows[0]) == 20 + 5 and post.e2e_tokens(rows[2]) == 20 + 5
    text = "\n".join(post.report_lines(rows, "test", "x.jsonl"))
    assert "| **readings first** | 75.0% | 50.0% |" in text
    assert "| unstated vs fair | 1.00 |" in text                         # distinct 2 vs 1
    s = post.summary(rows)
    assert s["readings_first"] == 0.75 and s["ask_rate_unstated"] == 1.0


def test_the_fingerprint_covers_both_prompts(monkeypatch):
    pytest.importorskip("hashlib")
    a = post.prompt_fingerprint()
    assert len(a) == 64 and a == post.prompt_fingerprint()
    monkeypatch.setattr(R, "QUESTION_SYSTEM", R.QUESTION_SYSTEM + " ")
    assert post.prompt_fingerprint() != a


# --- the readings one at a time (the default elicitation) ------------------------------

def test_parse_alt_reads_one_line_or_none():
    assert R.parse_alt("none") == ("none", None)
    assert R.parse_alt(' "None." ') == ("none", None)
    assert R.parse_alt("ex: 2d\nmore") == ("reading", ("ex", "2d"))
    assert R.parse_alt("1. keys: dw") == ("reading", ("keys", "dw"))
    assert R.parse_alt("there is no other way") == ("fail", None)
    assert R.parse_alt("<think></think>ex: 2d") == ("fail", None)


def test_the_alternative_prompt_lists_the_readings_so_far():
    msgs = R.alt_messages(CASE, [("ex", "s/alpha/x/"), ("keys", "dw")])
    assert msgs[0]["content"] == R.ALT_SYSTEM
    user = msgs[1]["content"]
    assert user.startswith(P.turn1_messages(CASE)[1]["content"])
    assert user.endswith("Readings so far:\n1. ex: s/alpha/x/\n2. keys: dw")


class ScriptedDecoder:
    """Replies in order from a script; records which prompt each call was given."""

    class tok:
        @staticmethod
        def encode(s, add_special_tokens=False):
            return list(s)

    def __init__(self, script):
        self.script, self.prompts = list(script), []

    def render(self, messages, prefill=""):
        return messages[0]["content"][:40] + "|" + messages[-1]["content"] + "|" + prefill

    def complete(self, text, max_new_tokens=192, ban_first=(), stop_at_newline=False):
        self.prompts.append(text)
        out = self.script.pop(0)
        return {"completion": out, "output_tokens": len(out.split()), "prompt_tokens": 10,
                "latency_s": 0.01, "first_id": None}


def _iterative(script):
    import readings as RD
    from score import execute
    dec = ScriptedDecoder(script)
    return RD.run_case(dec, CASE, execute, elicit="iterative"), dec


@pytest.mark.parametrize("script, stop, distinct, n_calls", [
    (["ex: s/alpha/x/", "ex: s/beta/y/", "none", " Which word?"], "none", 2, 4),
    (["ex: s/alpha/x/", "ex: s/alpha/x/"], "repeat", 1, 2),
    (["ex: s/alpha/x/", "keys: cwx<Esc>"], "same_result", 1, 2),
    (["ex: s/alpha/x/", "ex: s/zzz/q/"], "invalid", 1, 2),
    (["ex: s/alpha/x/", "I am not sure"], "unparsed", 1, 2),
    (["no command here"], "unparsed", 0, 1),
    (["ex: s/alpha/x/", "ex: s/beta/y/", "ex: s/alpha/z/", "ex: s/beta/w/", " Which?"], "cap", 4, 5),
])
def test_the_search_for_readings_stops_where_it_should(script, stop, distinct, n_calls):
    row, dec = _iterative(script)
    meta = row["readings"]
    assert meta["elicit"] == "iterative" and meta["stop"] == stop
    assert row["distinct"] == distinct and row["asked"] == (distinct >= 2)
    assert len(dec.prompts) == n_calls                       # incl. the question call when it asks
    assert dec.prompts[0].startswith(P.SYSTEM_TEXT[:40])     # reading 1 is the §6 answer
    assert all(p.startswith(R.ALT_SYSTEM[:40]) for p in dec.prompts[1:len(meta["calls"])])
    if row["asked"]:
        assert dec.prompts[-1].startswith(R.QUESTION_SYSTEM[:40]) and dec.prompts[-1].endswith("|ask:")
        assert row["question"]["text"] in ("Which word?", "Which?")


def test_transcripts_rebuild_each_reading_call():
    row, _ = _iterative(["ex: s/alpha/x/", "ex: s/beta/y/", "none", " Which word?"])
    calls = post.readings_calls(CASE, row)
    from prompt import messages_for_case
    assert calls[0]["messages"] == messages_for_case(CASE)
    assert calls[1]["messages"] == R.alt_messages(CASE, [("ex", "s/alpha/x/")])
    assert calls[2]["messages"] == R.alt_messages(CASE, [("ex", "s/alpha/x/"), ("ex", "s/beta/y/")])
    assert [c["completion"] for c in calls] == ["ex: s/alpha/x/", "ex: s/beta/y/", "none"]


# --- ask when sampled answers differ (sample_rule.py) ---------------------------------

def test_sample_view_reads_text_results_types_and_agreement():
    import sample_rule as S
    from score import execute
    probe_row = {"commands": [{"mode": "ex", "command": "s/alpha/x/", "n": 5},
                              {"mode": "keys", "command": "cwx<Esc>", "n": 1},     # same result
                              {"mode": "ex", "command": "s/beta/y/", "n": 1},
                              {"mode": "ex", "command": "s/zzz/q/", "n": 1}]}       # fails
    answers = {"c%d" % i: {"parse_ok": True, "mode": c["mode"], "command": c["command"]}
               for i, c in enumerate(probe_row["commands"])}
    ex = execute([CASE], {CASE["id"]: answers}, workers=2)
    v = S.sample_view(CASE, probe_row, [ex[(CASE["id"], "c%d" % i)] for i in range(4)])
    assert (v["n"], v["distinct_text"], v["both_types"]) == (8, 4, True)
    assert v["results"] == [6, 1] and v["distinct_results"] == 2 and v["failed"] == 1
    assert v["top_share"] == 6 / 8
    asks = dict(S.rules({"k01": v}))
    assert asks["**result: two or more distinct results**"]["k01"] is True
    assert asks["the top result holds at most 6 of 8"]["k01"] is True
    assert asks["the top result holds at most 5 of 8"]["k01"] is False
