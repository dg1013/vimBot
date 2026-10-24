"""Tests for the clarification experiment: `pytest scripts/clarify`.

No GPU: the protocol, the simulator's view, the leak screen, the policy
arithmetic (against brute force), scoring through the real sandbox, and the
report on a synthetic run.
"""
import itertools
import json
import math
import pathlib
import random
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))

import policies                                                    # noqa: E402
import protocol as P                                               # noqa: E402
from prompt import SYSTEM_TEXT, messages_for_case                  # noqa: E402

CASE = {"id": "t-1", "category": "operator-motion", "instruction": "Delete the second word",
        "mode": "keys", "buffer_before": "alpha beta gamma\n", "gold_command": "wdaw",
        "buffer_after": "alpha gamma\n", "cursor": [1, 1]}


# --- the protocol stays an addition to §6 -----------------------------------

def test_offer_prompt_is_the_section6_text_plus_the_clause():
    assert P.OFFER_SYSTEM.startswith(SYSTEM_TEXT)
    assert P.OFFER_SYSTEM == SYSTEM_TEXT + "\n" + P.ASK_CLAUSE


def test_turn1_user_turn_is_byte_identical_to_section6():
    assert P.turn1_messages(CASE)[1] == messages_for_case(CASE)[1]
    assert [m["role"] for m in P.turn1_messages(CASE)] == ["system", "user"]


def test_turn2_appends_the_question_and_the_reply():
    msgs = P.turn2_messages(CASE, "Which word?", "The second one.")
    assert msgs[:2] == P.turn1_messages(CASE)
    assert msgs[2] == {"role": "assistant", "content": "ask: Which word?"}
    assert msgs[3] == {"role": "user", "content": "The second one."}


@pytest.mark.parametrize("text, expected", [
    ("ask: Which word?", ("ask", "Which word?")),
    ("\n ask:Which word?", ("ask", "Which word?")),
    ("ask: one?\nex: d", ("ask", "one?")),
    ("ex: %s/a/b/", ("ex", "%s/a/b/")),
    ("keys: wdaw", ("keys", "wdaw")),
    ("Ask: Which word?", None),          # strict, like ex/keys
    ("ask:", None),
    ("<think>x</think>ask: q", None),
    ("Sure! ask: q", None),
])
def test_parse_turn1(text, expected):
    assert P.parse_turn1(text) == expected


def test_turn2_must_be_a_command():
    assert P.parse_turn2("ask: again?") is None
    assert P.parse_turn2("keys: wdaw") == ("keys", "wdaw")


def test_cut_question_keeps_the_first_line():
    assert P.cut_question(" Which word?\nex: dw") == "Which word?"
    assert P.cut_question("\nex: dw") == ""


# --- the first-token ban ----------------------------------------------------

def test_ban_at_first_never_writes_into_the_logits_it_is_given():
    # transformers keeps the tensor it hands a processor as the step's raw
    # logits; an in-place ban made log P(ask) = -inf on every case (2026-10-06)
    torch = pytest.importorskip("torch")
    import dialogue as D
    proc = D.ban_at_first((1, 3), n_prompt=4)
    raw = torch.zeros(1, 6)
    out = proc(torch.zeros(1, 4, dtype=torch.long), raw)
    assert torch.equal(raw, torch.zeros(1, 6))                  # untouched
    assert out[0, 1] == float("-inf") and out[0, 3] == float("-inf")
    assert out[0, 0] == 0 and out[0, 2] == 0
    later = torch.zeros(1, 6)                                    # second position: no ban
    assert proc(torch.zeros(1, 5, dtype=torch.long), later) is later


# --- the simulator's view ---------------------------------------------------

def test_the_simulator_never_sees_the_gold_command():
    case = dict(CASE, gold_command="wdaw")
    text = P.sim_user_turn(case, "Which word?")
    assert "wdaw" not in text
    assert "Delete the second word" in text and "Which word?" in text
    assert "alpha beta gamma" in text and "alpha gamma" in text
    assert all("gold" not in m["content"].lower() for m in P.sim_messages(case, "q"))


def test_visible_diff_quotes_lines_so_end_spaces_show():
    d = P.visible_diff("a\nb \nc\n", "a\nb\nc\n")
    assert '"b "' in d and '"b"' in d and "line 2:" in d
    assert 'change: deleted " " (column 2)' in d
    assert P.visible_diff("same\n", "same\n") == "(no line changes)"


def test_visible_diff_spells_out_where_and_how_much():
    d = P.visible_diff("12\n  xxxx\n", "12\naa  xxxx\n")
    assert 'change: inserted "aa" before column 1' in d
    d = P.visible_diff("wOrld\nabcde\n", "woRLd\naBCDe\n")
    assert 'replaced "Orl" (columns 2-4) with "oRL"' in d
    assert 'replaced "bcd" (columns 2-4) with "BCD"' in d


def test_visible_diff_whole_lines_removed_and_added():
    d = P.visible_diff("a\nb\nc\n", "a\nc\nd\ne\n")
    assert 'line 2 removed:\n  before "b"' in d
    assert 'new lines after line 3 "c":\n  after  "d"\n  after  "e"' in d
    assert ('new line before line 1 "a":\n  after  "z"'
            in P.visible_diff("a\n", "z\na\n"))
    assert ('new line after line 1 "version: 3" and before line 2 "services:":\n  after  ""'
            in P.visible_diff("version: 3\nservices:\n", "version: 3\n\nservices:\n"))
    assert "in the empty buffer" in P.visible_diff("", "x\n")


def test_visible_diff_names_the_replaced_line_not_its_neighbour():
    d = P.visible_diff("SELECT\n    o.id,\n    o.placed_at\nFROM o\n",
                       "SELECT\n    o.id,\n    o.placed_at,\n    o.total\nFROM o\n")
    assert d.startswith("line 3 replaced:")
    assert '  before "    o.placed_at"\n  after  "    o.placed_at,"\n  after  "    o.total"' in d
    assert "after line 2" not in d


def test_cursor_char():
    assert P.cursor_char(dict(CASE, cursor=[1, 7])) == 'on "b"'
    assert P.cursor_char(dict(CASE, cursor=[1, 99])) == "past the end of the line"
    assert P.cursor_char(dict(CASE, cursor=[9, 1])) == "outside the buffer"


def test_clean_reply_drops_think_and_later_paragraphs():
    assert P.clean_reply("<think>hmm</think>\nKeep it.\n\nExtra.") == "Keep it."


def test_the_check_line_never_reaches_the_assistant():
    text = "Check: the result keeps retries: 5.\nReply: Keep the incoming version only."
    assert P.split_check(text) == ("the result keeps retries: 5.",
                                   "Keep the incoming version only.")
    assert P.clean_reply(text) == "Keep the incoming version only."
    assert P.clean_reply("check: x\nNo, keep it.") == "No, keep it."
    assert P.clean_reply("No, keep it.") == "No, keep it."
    assert P.split_check("Reply: Yes.\n\nCheck: later") == ("later", "Yes.")


def test_clean_reply_with_thinking_never_passes_reasoning_through():
    # the template opens the block in the prompt: the completion holds its end
    assert P.clean_reply("line 2 is removed, so...</think>\n\nNo, keep it.",
                         thinking=True) == "No, keep it."
    # ran out of tokens mid-thought: no reply, never the reasoning
    assert P.clean_reply("line 2 is removed, so the rule is", thinking=True) == ""
    assert P.clean_reply("Keep it.", thinking=False) == "Keep it."


@pytest.mark.parametrize("reply, flagged", [
    ("Yes, keep the space after it.", False),
    ("Just the second word, please.", False),
    ("Only the first one; the rest stays.", False),
    ("Note: delete only the first.", False),
    ("Press <Esc> after typing it.", True),
    ("Do :%s/a/b/g on it.", True),
    ("Something like s/foo/bar/ would do.", True),
    ("Use a regex for that.", True),
    ("In vim I would do that.", True),
    ("Match \\(foo\\) there.", True),
    ("Use `:retab` to explicitly set the tabstop width.", True),     # the audit's miss
    ("Run :sort on it.", True),
    ("Keep the tab stops at four columns.", False),
    ("The time is 10:30, keep it.", False),
    ("Yes: keep the header.", False),
])
def test_leak_screen(reply, flagged):
    assert bool(P.leak_flags(reply, CASE)) == flagged


def test_leak_screen_gold_verbatim_from_four_characters():
    assert "gold command" in P.leak_flags("type wdaw", CASE)
    short = dict(CASE, gold_command="dd")
    assert P.leak_flags("add the line", short) == []


def test_quotes_result_counts_whole_long_result_lines_only():
    case = dict(CASE, buffer_before="x\n", buffer_after="the seed value is forty-two\n")
    assert P.quotes_result('It should say "the seed value is forty-two".', case)
    assert not P.quotes_result("Forty-two.", case)
    short = dict(CASE, buffer_before="x\n", buffer_after="y = 42\n")
    assert not P.quotes_result("y = 42", short)


# --- the policy arithmetic --------------------------------------------------

def _rows(spec):
    return [{"id": "r%d" % i, "noask": a, "real": b, "p_ask_norm": p}
            for i, (a, b, p) in enumerate(spec)]


def test_policy_match_and_oracle():
    rows = _rows([(0, 1, .9), (1, 0, .8), (0, 0, .1), (1, 1, .2)])
    assert policies.policy_match(rows, set()) == 0.5
    assert policies.policy_match(rows, {"r0"}) == 0.75
    assert policies.policy_match(rows, {"r0", "r1", "r2", "r3"}) == 0.5
    assert policies.oracle(rows) == (0.25, 0.75)


def test_threshold_curve_takes_ties_together_and_interpolates():
    rows = _rows([(0, 1, .5), (1, 0, .5), (0, 1, .2), (0, 0, .1)])
    curve = policies.threshold_curve(rows)
    assert curve == [(0.0, 0.25), (0.5, 0.25), (0.75, 0.5), (1.0, 0.5)]
    # inside the tied group: the expectation of a random tie-break
    assert policies.at_rate(curve, 0.25) == pytest.approx(0.25)
    assert policies.at_rate(curve, 0.625) == pytest.approx(0.375)


def test_random_at_rate_is_the_straight_line():
    rows = _rows([(0, 1, .9), (0, 1, .8), (1, 1, .1), (0, 0, .2)])
    assert policies.random_at_rate(rows, 0) == 0.25
    assert policies.random_at_rate(rows, 1) == 0.75
    assert policies.random_at_rate(rows, 0.5) == pytest.approx(0.5)
    # brute force: the mean over every subset of size 2
    subsets = list(itertools.combinations([r["id"] for r in rows], 2))
    brute = sum(policies.policy_match(rows, set(s)) for s in subsets) / len(subsets)
    assert policies.random_at_rate(rows, 0.5) == pytest.approx(brute)


def test_auroc_against_pairwise_brute_force():
    rng = random.Random(3)
    for _ in range(50):
        scores = [rng.choice([0.1, 0.2, 0.3, 0.5, 0.9]) for _ in range(12)]
        labels = [rng.random() < 0.4 for _ in range(12)]
        pos = [s for s, y in zip(scores, labels) if y]
        neg = [s for s, y in zip(scores, labels) if not y]
        got = policies.auroc(scores, labels)
        if not pos or not neg:
            assert got is None
            continue
        brute = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))
        assert got == pytest.approx(brute)


def test_mcnemar_exact_values():
    assert policies.mcnemar_p(0, 0) == 1.0
    assert policies.mcnemar_p(10, 0) == pytest.approx(2 / 1024)
    assert policies.mcnemar_p(3, 3) == 1.0
    assert policies.mcnemar_p(1, 5) == policies.mcnemar_p(5, 1)


def test_flips_and_paired_se():
    a, b = [1, 0, 0, 1], [1, 1, 0, 0]
    assert policies.flips(a, b) == (1, 1)
    d, se = policies.paired_se(a, b)
    assert d == 0 and se == pytest.approx((2 / 3 / 4) ** 0.5)


# --- scoring through the real sandbox ---------------------------------------

def _answer(command, mode="keys", tokens=3):
    return {"completion": "%s: %s" % (mode, command) if command else "junk",
            "output_tokens": tokens, "parse_ok": command is not None,
            "mode": mode if command else None, "command": command}


def test_execute_scores_through_evaluate_and_runs_shared_answers_once():
    import score as S
    answers = {"t-1": {"direct": _answer("wdaw"), "noask": _answer("wdaw"),
                       "placebo": _answer("dw"), "real": _answer(None)}}
    ex = S.execute([CASE], answers, timeout=5.0, workers=2)
    assert ex[("t-1", "direct")]["match"] and ex[("t-1", "noask")]["match"]
    assert not ex[("t-1", "placebo")]["match"]      # dw from line 1 col 1 deletes "alpha "
    assert ex[("t-1", "real")] is None


def test_outcome_classes_through_the_real_sandbox():
    import score as S
    answers = {"t-1": {"direct": _answer("wdaw"),               # the gold
                       "noask": _answer("l"),                   # moves the cursor only
                       "placebo": _answer("frobnicate", "ex"),  # not an ex command
                       "real": _answer("w /tmp/clarify-x", "ex")}}   # a write: screened
    ex = S.execute([CASE], answers, workers=2)
    got = {c: S.outcome_class(CASE, ex[("t-1", c)], answers["t-1"][c]) for c in answers["t-1"]}
    assert got == {"direct": "match", "noask": "no_change", "placebo": "vim_error",
                   "real": "blocked"}
    assert S.outcome_class(CASE, None, _answer(None)) == "parse_fail"
    assert S.outcome_class(CASE, None, dict(_answer(None), second_ask=True)) == "second_ask"
    wrong = dict(ex[("t-1", "direct")], match=False, buffer_after="alpha\n")
    assert S.outcome_class(CASE, wrong, _answer("x")) == "wrong_buffer"
    assert S.outcome_class(CASE, dict(wrong, timeout=True), _answer("x")) == "timeout"


def _t1(question="Which word?", chose=False, free=None):
    offer = {"p_ask": 0.2, "p_ask_norm": 0.25, "ask_margin": math.log(1 / 3),
             "chose_ask": chose, "top": "ask" if chose else "keys"}
    if free is not None:
        offer["free_completion"] = free
    return {"direct": _answer("wdaw"), "noask": _answer("dw"),
            "question": {"text": question, "output_tokens": 6},
            "placebo": dict(_answer("dw"), second_ask=False), "offer": offer}


def test_free_consistency():
    import score as S
    assert S.free_consistent(_t1()) is None
    assert S.free_consistent(_t1(chose=True, free="ask:  Which  word?\nmore")) == "same"
    assert S.free_consistent(_t1(chose=True, free="ask: Another?")) == "diverged"
    # the forced question stopped at the cap: the free one runs on past it
    capped = _t1(chose=True, free="ask: Which word? Or the line?")
    capped["question"]["output_tokens"] = P.QUESTION_MAX_TOKENS + S.ASK_PREFIX_TOKENS
    assert S.question_capped(capped) and S.free_consistent(capped) == "cut"
    # a prefix that ended at its own newline is a different question, not a cut one
    short = _t1(chose=True, free="ask: Which word? Or the line?")
    assert not S.question_capped(short) and S.free_consistent(short) == "diverged"


def test_verdict_rows_charge_the_question_to_the_asked_answers():
    import score as S
    t1 = _t1()
    t2 = {"answer": dict(_answer("wdaw", tokens=4), second_ask=False)}
    rep = {"reply": "The second one.", "leak_flags": [], "quotes_result": False}
    answers = {"t-1": S.answers_for(t1, t2)}
    ex = S.execute([CASE], answers, workers=1)
    row = S.verdict_rows([CASE], "base", {"t-1": t1}, {"t-1": t2}, {"t-1": rep}, ex)[0]
    assert row["direct"]["tokens"] == 3 and row["real"]["tokens"] == 6 + 4
    assert row["direct"]["match"] and row["real"]["match"] and not row["noask"]["match"]
    # what vim did is kept for error analysis: the buffer the wrong answer left
    assert row["noask"]["outcome"] == "wrong_buffer"
    assert row["noask"]["buffer_after"] == "beta gamma\n"         # dw at col 1 took "alpha "
    assert row["noask"]["completion"] == "keys: dw" and row["noask"]["stderr"] is not None


# --- the report on a synthetic run ------------------------------------------

def _verdict(i, direct, noask, placebo, real, p, chose, cat="c1", flags=()):
    cond = lambda m, t: {"match": bool(m), "valid": True, "parse_ok": True,   # noqa: E731
                         "mode": "ex", "command": "x", "error": "", "tokens": t,
                         "second_ask": False}
    return {"id": "v%02d" % i, "label": "base", "category": cat, "tier": "T1",
            "gold_mode": "ex", "instruction": "do it", "p_ask": p, "p_ask_norm": p,
            "ask_margin": math.log(p / (1 - p)),
            "chose_ask": chose, "top": "ask" if chose else "ex", "free_consistent": None,
            "question": "Which?", "question_tokens": 5, "reply": "That one.",
            "leak_flags": list(flags), "quotes_result": False,
            "direct": cond(direct, 4), "noask": cond(noask, 4),
            "placebo": cond(placebo, 9), "real": cond(real, 9)}


def test_report_end_to_end(tmp_path):
    import report as R
    rows = ([_verdict(i, 1, 1, 1, 1, 0.1, False) for i in range(4)]
            + [_verdict(4 + i, 0, 0, 0, 1, 0.9, True, "c2") for i in range(4)]
            + [_verdict(8 + i, 0, 0, 0, 0, 0.2, False, flags=["key notation"]) for i in range(2)])
    path = tmp_path / "verdicts_base.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    h = {x["name"]: x for x in R.headline(rows)}
    assert h["direct"]["match"] == pytest.approx(0.4)
    assert h["always ask, simulated reply"]["match"] == pytest.approx(0.8)
    assert h["offer, the model's choice"]["match"] == pytest.approx(0.8)
    assert h["offer, the model's choice"]["ask_rate"] == pytest.approx(0.4)
    assert h["offer, the model's choice"]["gained"] == 4
    assert h["always ask, placebo reply"]["match"] == pytest.approx(0.4)
    assert h["oracle timing"]["ask_rate"] == pytest.approx(0.4)
    out = tmp_path / "report.md"
    sys.argv = ["report.py", "--verdicts", "base=%s" % path, "--out", str(out)]
    assert R.main() == 0
    text = out.read_text()
    assert "Does the model know when to ask?" in text and "AUROC" in text
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["base"]["n"] == 10
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"base|v00": {"verdict": "ok"},
                                 "base|v04": {"verdict": "wrong", "note": "x"},
                                 "served|v00": {"verdict": "uninformative"}}))
    sys.argv = ["report.py", "--verdicts", "base=%s" % path, "--audit", str(audit),
                "--out", str(out)]
    assert R.main() == 0
    assert ("hand audit of 2 randomly drawn replies: 1 wrong (50%), 0 leaking the method, "
            "0 uninformative, 0 minor, 1 right") in out.read_text()


def test_audit_sample_is_seeded_and_counts_every_verdict():
    import audit_sample as A
    replies = [{"id": "c%d" % i, "label": lab} for i in range(20) for lab in ("base", "served")]
    first = A.sample(replies, 5, seed=1)
    assert first == A.sample(list(reversed(replies)), 5, seed=1)
    assert A.summarize({"a": {"verdict": "ok"}, "b": {"verdict": "leak"}}) == {
        "ok": 1, "wrong": 0, "uninformative": 0, "leak": 1, "minor": 0}


# --- transcripts for hand analysis ------------------------------------------

def _run_records():
    """turn1, turn2, reply and verdict records for CASE, as a run writes them."""
    import score as S
    t1 = _t1()
    t1["question"]["raw"] = " Which word?"
    t1["question"]["latency_s"] = 0.1
    for k in ("direct", "noask", "placebo"):
        t1[k].update(prompt_tokens=40, latency_s=0.1)
    t1["offer"].update(first_consistent=True)
    t2 = {"answer": dict(_answer("wdaw", tokens=4), second_ask=False, prompt_tokens=60,
                         latency_s=0.1)}
    rep = {"reply": "The second one.", "check": "line 1 loses beta", "think": "beta goes",
           "think_tokens": 9, "forced": False, "raw": "Check: x\nReply: The second one.",
           "reply_tokens": 8, "latency_s": 1.0, "leak_flags": [], "quotes_result": False,
           "simulator": {"model": "qwen3.5-9b"}}
    ex = S.execute([CASE], {"t-1": S.answers_for(t1, t2)}, workers=1)
    verdict = S.verdict_rows([CASE], "base", {"t-1": t1}, {"t-1": t2}, {"t-1": rep}, ex)[0]
    return t1, t2, rep, verdict


def test_transcript_holds_every_call_in_protocol_order_with_its_messages():
    import transcripts as T
    t1, t2, rep, verdict = _run_records()
    row = T.build_row(CASE, t1, t2, rep, verdict, "base")
    steps = [c["step"] for c in row["calls"]]
    assert steps == ["direct", "offer", "noask", "question", "simulator", "placebo", "real"]
    calls = {c["step"]: c for c in row["calls"]}
    assert calls["direct"]["messages"] == messages_for_case(CASE)
    assert calls["noask"]["messages"] == P.turn1_messages(CASE)
    assert calls["noask"]["ban_first"] == ["ask", "Ask"]
    assert calls["question"]["prefill"] == "ask:"
    assert calls["simulator"]["messages"] == P.sim_messages(CASE, "Which word?")
    assert calls["simulator"]["think"] == "beta goes"
    assert calls["real"]["messages"][-1] == {"role": "user", "content": "The second one."}
    assert calls["placebo"]["messages"][-1]["content"] == P.PLACEBO_REPLY
    # the gold is in the case record, never in a student or simulator prompt
    assert all("wdaw" not in m["content"] for c in row["calls"] for m in c["messages"])
    assert row["outcomes"]["real"]["outcome"] == "match"
    assert row["outcomes"]["real"]["diff_vs_expected"] is None
    d = row["outcomes"]["noask"]["diff_vs_expected"]
    assert "-alpha gamma" in d and "+beta gamma" in d


def test_prompt_fingerprint_covers_prompts_not_the_leak_screen(tmp_path, monkeypatch):
    import transcripts as T
    before = T.prompt_fingerprint()
    monkeypatch.setattr(P, "_VIM_TALK", [])                 # the screen is not a prompt
    assert T.prompt_fingerprint() == before
    monkeypatch.setattr(P, "SIM_SYSTEM", P.SIM_SYSTEM + " ")  # the simulator's prompt is
    assert T.prompt_fingerprint() != before
    path, new = T.record_fingerprint(tmp_path)
    assert new and json.loads(path.read_text())["prompt_fingerprint"] == T.prompt_fingerprint()
    assert T.record_fingerprint(tmp_path) == (path, False)   # recorded once, at launch


def test_transcript_markdown_renders_a_case():
    import transcripts as T
    t1, t2, rep, verdict = _run_records()
    md = T.to_markdown(T.build_row(CASE, t1, t2, rep, verdict, "base"))
    assert md.startswith("## t-1 — base (operator-motion")
    assert "**Asks:** Which word?" in md and "**Reply:** The second one." in md
    assert "noask left a different buffer:" in md
    assert T.fence("a ``` b").startswith("````")


# --- the paired wording test ------------------------------------------------

def test_paired_wording_matches_only_rows_whose_edit_is_unchanged(tmp_path):
    import paired_wording as W
    v1 = [dict(CASE, id="a", instruction="vague"), dict(CASE, id="b", instruction="vague"),
          dict(CASE, id="c", instruction="vague", gold_command="other")]
    v2 = [dict(CASE, id="a", instruction="clear"), dict(CASE, id="b", instruction="clear"),
          dict(CASE, id="c", instruction="clear")]
    ids, skipped = W.pairs(v1, v2)
    assert ids == ["a", "b"] and skipped == ["c"]
    cell = lambda m: {"match": bool(m)}                                 # noqa: E731
    r1 = {"a": {"direct": cell(0), "real": cell(1), "question": "q", "reply": "r"},
          "b": {"direct": cell(0), "real": cell(0), "question": "q", "reply": "r"}}
    r2 = {"a": {"direct": cell(1), "real": cell(1)}, "b": {"direct": cell(1), "real": cell(0)}}
    t = W.table(ids, r1, r2)
    assert t[("v1", "direct")] == [0, 0] and t[("v2", "direct")] == [1, 1]
    md = W.report_for("served", ids, r1, r2, [dict(c, v2_instruction="clear") for c in v1])
    assert "Of the 2 cases the reword turned right (a, b), asking on the vague wording " \
           "also turns right 1 (a)." in md
    assert "| `a` | vague → *clear* |" in md
    # a fix on a case the reword did not gain is reported apart, never as recovery
    r1b = dict(r1, b={"direct": cell(0), "real": cell(1), "question": "q", "reply": "r"})
    r2b = dict(r2, b={"direct": cell(0), "real": cell(0)})
    r1b["a"] = {"direct": cell(0), "real": cell(0), "question": "q", "reply": "r"}
    md = W.report_for("served", ids, r1b, r2b, [dict(c, v2_instruction="clear") for c in v1])
    assert "turned right (a), asking on the vague wording also turns right 0." in md
    assert "also turns right 1 case the reword did not: b." in md


# --- the linear probe -------------------------------------------------------

def test_probe_folds_are_stratified_and_the_probe_finds_a_planted_signal():
    pytest.importorskip("torch")
    import probe_ask as PA
    labels = [1] * 20 + [0] * 40
    fold = PA.stratified_folds(labels, 5, seed=1)
    for f in range(5):
        assert sum(1 for i, y in enumerate(labels) if fold[i] == f and y) == 4
    rng = random.Random(0)
    signal = [[y + rng.gauss(0, 0.5)] + [rng.gauss(0, 1) for _ in range(9)] for y in labels]
    noise = [[rng.gauss(0, 1) for _ in range(10)] for _ in labels]
    assert PA.cv_auroc(signal, labels, 1e-2) > 0.85       # held-out folds, planted signal
    assert 0.25 < PA.cv_auroc(noise, labels, 1e-2) < 0.75  # nothing to find: near chance


def test_probe_surface_baselines_read_the_case_not_the_model():
    pytest.importorskip("torch")
    import probe_ask as PA
    labels = [1] * 20 + [0] * 20
    groups = ["insert"] * 18 + ["global"] * 2 + ["global"] * 18 + ["insert"] * 2
    assert PA.prior_auroc(groups, labels) > 0.8      # a category that tracks the label
    assert PA.prior_auroc(["same"] * 40, labels) == 0.5
    cases = [{"instruction": "add a heading" if y else "delete every line with foo in it",
              "category": "c", "source": "s"} for y in labels]
    b = PA.surface_baselines(cases, labels)
    assert b["shorter instruction"] == 1.0           # the positives are all shorter
    assert set(b) == {"shorter instruction", "category (cross-validated prior)",
                      "data source (cross-validated prior)",
                      "bag of words on the instruction (best λ)"}


def test_probe_tasks_label_by_note_class_and_by_fixability():
    import probe_ask as PA
    cases = {"diag": {"a": {"note_class": "unstated"}, "b": {"note_class": "ambiguous"},
                      "c": {"note_class": "model"}}}
    cell = lambda m: {"match": bool(m)}                                 # noqa: E731
    ver = {"val": {"x": {"real": cell(1), "noask": cell(0)}, "y": {"real": cell(1), "noask": cell(1)}}}
    t = PA.tasks(cases, ver)
    assert t["unstated vs model"] == ([("diag", "a"), ("diag", "c")], [1, 0])
    assert t["unstated+ambiguous vs model"][1] == [1, 1, 0]
    assert t["fixable (asking turns it right)"] == ([("val", "x"), ("val", "y")], [1, 0])


# --- the diagnostic set -----------------------------------------------------

def test_diag_cases_take_every_ambiguous_and_unstated_row_and_a_seeded_control():
    import make_diag_cases as M
    train = [dict(CASE, id="r%d" % i) for i in range(10)]
    notes = {"r0": {"id": "r0", "class": "ambiguous", "comment": "a"},
             "r1": {"id": "r1", "class": "unstated", "comment": "u"},
             "r2": {"id": "r2", "class": "defect", "comment": "d"},
             "gone": {"id": "gone", "class": "ambiguous", "comment": "x"}}
    notes.update({"r%d" % i: {"id": "r%d" % i, "class": "model", "comment": "m"}
                  for i in range(3, 10)})
    rows, missing = M.build(train, notes, n_control=3, seed=1)
    classes = sorted(r["note_class"] for r in rows)
    assert classes == ["ambiguous", "model", "model", "model", "unstated"]
    assert missing == 1
    again, _ = M.build(train, notes, n_control=3, seed=1)
    assert [r["id"] for r in again] == [r["id"] for r in rows]
