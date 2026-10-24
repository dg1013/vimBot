"""Tests for scripts/ask_sft/ (stage 1 of training the ask decision).

Run:  python3 -m pytest scripts/ask_sft          (not collected by `pytest tests/`)

The trainer copy's drift guard; the split; the row shapes against the
clarify protocol (and the chat template's prefix property on a real
turn-2 row); the rejection filter; the no-held-out-rows guarantee;
verification through the real sandbox; and the gate.
"""
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))

import data as D                                       # noqa: E402
import protocol as P                                   # noqa: E402
from prompt import render_target                       # noqa: E402

ORIGINAL = REPO / "scripts" / "cliff_sft" / "train_sft_cliff.py"
COPY = HERE / "train_sft_ask.py"


def _case(i, cls="unstated", gold="x"):
    return {"id": "r%02d" % i, "note_class": cls, "category": "insert-edit",
            "instruction": "do %d" % i, "mode": "keys", "buffer_before": "alpha beta\n",
            "gold_command": gold, "buffer_after": "lpha beta\n", "cursor": [1, 1]}


# --- the trainer copy ---------------------------------------------------------

def _unmarked(text):
    out, inside = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# [ask] begin"):
            inside = True
            continue
        if s.startswith("# [ask] end"):
            inside = False
            continue
        if inside or "[ask]" in line:
            continue
        out.append(line)
    return out


def test_every_change_to_the_cliff_trainer_is_marked():
    orig = ORIGINAL.read_text(encoding="utf-8").splitlines()
    copy = _unmarked(COPY.read_text(encoding="utf-8"))
    replaced = {   # the cliff copy's lines the [ask] copy replaced: the whole diff
        '        args.run_name = reg["name"] + "-sft-cliff"',
        '        run_name=args.run_name, condition="sft-cliff" if adapter is not None else "sft", '
        'wandb_state=wandb_state,   # [cliff]',
        '        print("  (label it: --condition sft-cliff; then the next gated GRPO leg continues "',
        '              "runs/%s-sft-cliff/best: MODELS=%s LEG=-leg2 ADAPTER_FROM=sft-cliff "',
        '              "bash runs/grpo_gated_2026-09-14/launch.sh)" % (reg["name"], reg["name"]))',
    }
    missing = [l for l in orig if l not in set(copy)]
    assert set(missing) == replaced, missing
    assert [l for l in copy if l not in set(orig)] == []


def test_the_copy_passes_dialogue_rows_through_and_refuses_a_mix():
    pytest.importorskip("datasets")
    import train_sft_ask as T
    reg = {"chat_template_kwargs": {"enable_thinking": False}}
    rows = [D.answer_row(_case(1, "model")), D.placebo_row(_case(2), "Which?")]
    ds = T.build_dataset(rows, reg)
    assert ds[0]["prompt"] == rows[0]["prompt"] and ds[1]["completion"] == rows[1]["completion"]
    assert ds[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert set(ds.column_names) == {"prompt", "completion", "chat_template_kwargs"}
    with pytest.raises(SystemExit):
        T.build_dataset(rows + [_case(3)], reg)


# --- the split ----------------------------------------------------------------

def test_split_is_stratified_seeded_and_disjoint():
    cases = ([_case(i, "unstated") for i in range(20)] + [_case(20 + i, "ambiguous") for i in range(10)]
             + [_case(30 + i, "model") for i in range(10)])
    train, held = D.split_noted(cases, 0.3, seed=1)
    assert {c["id"] for c in train}.isdisjoint({c["id"] for c in held})
    assert len(train) + len(held) == 40
    by = lambda rows, k: sum(1 for c in rows if c["note_class"] == k)       # noqa: E731
    assert (by(held, "unstated"), by(held, "ambiguous"), by(held, "model")) == (6, 3, 3)
    assert [c["id"] for c in D.split_noted(list(reversed(cases)), 0.3, seed=1)[1]] == \
        [c["id"] for c in held]


# --- the rows -------------------------------------------------------------------

def test_rows_are_the_inference_prompts_with_one_target_turn():
    c = _case(1)
    t1, t2 = D.ask_rows(c, "Which letter?", "The first one.")
    assert t1["prompt"] == P.turn1_messages(c)
    assert t1["completion"] == [{"role": "assistant", "content": "ask: Which letter?"}]
    assert P.parse_turn1(t1["completion"][0]["content"]) == ("ask", "Which letter?")
    assert t2["prompt"] == P.turn2_messages(c, "Which letter?", "The first one.")
    assert t2["completion"][0]["content"] == render_target("keys", "x")
    pl = D.placebo_row(c, "Which letter?")
    assert pl["prompt"][-1] == {"role": "user", "content": P.PLACEBO_REPLY}
    assert pl["prompt"][2] == {"role": "assistant", "content": "ask: Which letter?"}
    assert D.answer_row(c)["prompt"] == P.turn1_messages(c)


def test_a_turn2_row_trains_on_exactly_what_inference_generates():
    """TRL takes the completion as the rendering of prompt+completion minus
    the rendering of the prompt with its generation prompt: the prefix must
    hold, and the remainder must be the answer line and its end token."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    from baseline import load_registry
    reg = load_registry("qwen3.5-4b")
    try:
        tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"],
                                            local_files_only=True)
    except OSError:
        pytest.skip("tokenizer not cached")
    kw = reg["chat_template_kwargs"]
    for row in (D.ask_rows(_case(1), "Which?", "The first.")[1], D.placebo_row(_case(2), "Which?"),
                D.ask_rows(_case(3), "Which?", "The first.")[0]):
        p = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True, **kw)
        full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False, **kw)
        assert full.startswith(p)
        assert full[len(p):].startswith(row["completion"][0]["content"] + "<|im_end|>")


def test_unique_questions_drops_repeats_empties_and_the_greedy_one():
    got = D.unique_questions(["Which?", " which? ", "", "Where?", "WHERE?"], seen=["Greedy?"])
    assert got == ["Which?", "Where?"]
    assert D.unique_questions(["Greedy?", "Other?"], seen=["greedy?"]) == ["Other?"]


def test_pick_dialogues_keeps_only_verified_clean_distinct_ones():
    d = lambda i, q, match=True, flags=(), quotes=False: {   # noqa: E731
        "id": "r%02d" % i, "question": q, "reply": "r", "match": match,
        "leak_flags": list(flags), "quotes_result": quotes}
    picked = D.pick_dialogues([d(1, "A?"), d(1, "a?"), d(1, "B?"), d(1, "C?"),
                               d(2, "A?", match=False), d(3, "A?", flags=["key notation"]),
                               d(4, "A?", quotes=True)], max_per_row=2)
    assert [x["question"] for x in picked["r01"]] == ["A?", "B?"]
    assert set(picked) == {"r01"}


def test_build_teaches_only_verified_rows_and_never_a_held_out_one():
    import build_set as B
    diag_train = [_case(1, "unstated"), _case(2, "ambiguous"), _case(3, "model")]
    held = ["r09"]
    dialogues = [{"id": "r01", "question": "Which?", "reply": "That.", "match": True,
                  "leak_flags": [], "quotes_result": False},
                 {"id": "r02", "question": "Which?", "reply": "That.", "match": False,
                  "leak_flags": [], "quotes_result": False}]
    greedy = {"r01": "G1?", "r02": "G2?", "r03": "G3?", "r09": "G9?"}
    train_rows = [_case(i, None) for i in range(1, 15)]
    notes = {"r03": "model", "r09": "model", "r10": "model", "r11": "model", "r12": "unstated"}
    rows, manifest = B.build(diag_train, held, dialogues, greedy, train_rows, notes,
                             max_dialogues=2, n_model=5, n_other=2, seed=0)
    kinds = manifest["rows by kind"]
    assert kinds == {"ask-turn1": 1, "ask-turn2": 1, "placebo-turn2": 3, "answer": 1 + 2 + 2}
    assert manifest["ask rows with a verified dialogue"] == 1
    assert manifest["ask rows without one (not taught)"] == 1
    assert "r09" not in {r["id"] for r in rows}
    answered = {r["id"] for r in rows if r["kind"] == "answer"}
    assert {"r03", "r10", "r11"} <= answered and "r12" not in answered
    with pytest.raises(AssertionError):
        B.build(diag_train, ["r01"], dialogues, greedy, train_rows, notes, 2, 5, 2, 0)


# --- verification ----------------------------------------------------------------

def test_verify_joins_and_executes_through_the_sandbox(tmp_path):
    import verify as V
    case = dict(_case(1), buffer_before="alpha beta gamma\n", gold_command="wdaw",
                buffer_after="alpha gamma\n")
    (tmp_path / "questions_q1.jsonl").write_text(json.dumps(
        {"id": "r01", "label": "q1", "question": {"text": "Which word?"}}) + "\n")
    (tmp_path / "replies.jsonl").write_text(json.dumps(
        {"id": "r01", "label": "q1", "reply": "The second one."}) + "\n")
    (tmp_path / "turn2_q1.jsonl").write_text(json.dumps({"id": "r01", "answer": {
        "parse_ok": True, "mode": "keys", "command": "wdaw", "completion": "keys: wdaw",
        "output_tokens": 3}}) + "\n")
    rows = V.sampled_dialogues([case], ["q1"], tmp_path, workers=1)
    assert rows == [{"id": "r01", "label": "q1", "question": "Which word?",
                     "reply": "The second one.", "mode": "keys", "command": "wdaw", "match": True}]
    verdicts = [{"id": "r01", "question": "G?", "reply": "R.",
                 "real": {"mode": "ex", "command": "x", "match": False}},
                {"id": "zz", "question": "G?", "reply": "R.", "real": {"mode": "ex", "command": "x", "match": True}}]
    assert [g["label"] for g in V.greedy_dialogues(verdicts, {"r01"})] == ["q0"]


def test_row_seeds_are_stable_and_distinct():
    import sample_questions as S
    assert S.row_seed(1, "hc-0001") == S.row_seed(1, "hc-0001")
    assert S.row_seed(1, "hc-0001") != S.row_seed(1, "hc-0002")


# --- the gate ----------------------------------------------------------------------

def _v(i, direct, noask, real, placebo, chose, cls=None):
    m = lambda x: {"match": bool(x)}                                     # noqa: E731
    row = {"id": "v%02d" % i, "direct": m(direct), "noask": m(noask), "real": m(real),
           "placebo": m(placebo), "chose_ask": chose}
    if cls:
        row["note_class"] = cls
    return row


def test_gate_reads_own_choice_fair_ask_rate_and_val():
    import compare as C
    held_new = [_v(0, 0, 0, 1, 0, True, "unstated"), _v(1, 0, 0, 1, 0, True, "unstated"),
                _v(2, 1, 1, 1, 1, False, "model"), _v(3, 1, 1, 0, 1, False, "model")]
    held_ref = [_v(i, 0, 0, 0, 0, False, c) for i, c in enumerate(["unstated"] * 2 + ["model"] * 2)]
    val_new = [_v(0, 1, 1, 1, 1, False), _v(1, 0, 0, 1, 0, True)]
    val_ref = [_v(0, 1, 1, 1, 1, False), _v(1, 0, 0, 0, 0, False)]
    checks = C.gate(held_new, held_ref, val_new, val_ref)
    assert [ok for _, ok, _ in checks] == [True, True, True]
    over_asking = [dict(r, chose_ask=True) for r in held_new]
    assert [ok for _, ok, _ in C.gate(over_asking, held_ref, val_new, val_ref)][1] is False
    lines, new, ref = C.section("held", held_new, held_ref)
    assert any("asks on *model* rows (2) | 0.0% | 0.0%" in l for l in lines)


# --- the analysis past the gate ---------------------------------------------------------

def _leak_case(i, instruction, buffer, gold, **kw):
    return dict({"id": "k%02d" % i, "instruction": instruction, "mode": "keys",
                 "buffer_before": buffer, "gold_command": gold}, **kw)


def test_links_find_shared_buffers_commands_and_paraphrases_only():
    import analyze as A
    trained = [_leak_case(1, "Delete the first word", "one two\n", "dw"),
               _leak_case(2, "Uppercase the line", "abc\n", "gUU", paraphrase_of="k13")]
    held = [_leak_case(11, "Remove the line's last word", "one two\n", "$diw"),    # same buffer
            _leak_case(12, "Wipe out the opening word", "red blue\n", "dw"),        # same command
            _leak_case(13, "Make the whole line capitals", "xyz\n", "VU"),          # paraphrased by k02
            _leak_case(14, "Join the two lines", "a\nb\n", "J")]                    # clean
    assert A.links(held, trained) == {"k11": ["buffer"], "k12": ["command"], "k13": ["paraphrase"]}


def _o(i, p, noask, real, cls="unstated", direct=None):
    return {"id": "o%02d" % i, "p": p, "noask": noask, "real": real, "cls": cls,
            "direct": noask if direct is None else direct, "chose": p > 0.5}


def test_thresholds_are_scored_exactly_and_picked_by_cost():
    import analyze as A
    rows = [_o(0, 0.9, 0, 1), _o(1, 0.6, 0, 1), _o(2, 0.3, 1, 0, "model"), _o(3, 0.1, 1, 1, "model")]
    ask, out = A.at_threshold(rows, 0.5)
    assert ask == [True, True, False, False] and out == [1, 1, 1, 1]
    # every row right at half the asks: 0.6 is the highest grid point that asks on rows 0-1
    assert A.pick(rows, 0.2) == 0.6
    # asking pays iff it fixes with probability > c, so certain fixes win at c = 0.9 ...
    assert A.pick(rows, 0.9) == 0.6
    # ... and a question dearer than any fix (c > 1) is never asked
    assert A.pick(rows, 1.5) == 1.01
    # equal utility at 0.9 and 0.8 (row 1 sits below both): the higher threshold, fewer asks
    assert A.pick([_o(0, 0.95, 0, 1), _o(1, 0.1, 1, 1)], 0.1, grid=(0.9, 0.8, 0.0)) == 0.9
    lines = A.sweep_lines(rows, [r["direct"] for r in rows], with_classes=True)
    assert lines[2].startswith("| greedy decoding") and lines[-1].startswith("| oracle | 2 (50.0%) | 100.0%")
