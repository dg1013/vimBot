"""Tests for scripts/ask_grpo/ (stage 2 of training the ask decision).

Run:  python3 -m pytest scripts/ask_grpo          (not collected by `pytest tests/`)

The trainer copy's drift guard; the reply bank; the rollout's orchestration
on a scripted model (ask detection, bank lookup, the greedy turn 2, the
masks and the final answers); the ask-cost reward; the prompt set.
"""
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))

import bank as B                      # noqa: E402
import rollout as R                   # noqa: E402
import protocol as P                  # noqa: E402

ORIGINAL = REPO / "scripts" / "train_grpo.py"
COPY = HERE / "train_grpo_ask.py"


def _case(i, cls="unstated"):
    return {"id": "r%02d" % i, "note_class": cls, "instruction": "do %d" % i, "mode": "keys",
            "buffer_before": "alpha beta\n", "gold_command": "x", "buffer_after": "lpha beta\n",
            "cursor": [1, 1]}


# --- the trainer copy ---------------------------------------------------------

def _unmarked(text):
    out, inside = [], False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("# [ask2] begin"):
            inside = True
            continue
        if s.startswith("# [ask2] end"):
            inside = False
            continue
        if inside or "[ask2]" in line:
            continue
        out.append(line)
    return out


def test_every_change_to_the_grpo_trainer_is_marked():
    orig = ORIGINAL.read_text(encoding="utf-8").splitlines()
    copy = _unmarked(COPY.read_text(encoding="utf-8"))
    replaced = {
        "REPO = pathlib.Path(__file__).resolve().parents[1]",
        '    rows = [{"prompt": messages_for_case(c),',
        '             "cursor": [int(x) for x in c.get("cursor", [1, 1])]}',
        "                          train_dataset=train_ds, processing_class=tok)",
    }
    missing = [l for l in orig if l not in set(copy)]
    assert set(missing) == replaced, missing
    assert [l for l in copy if l not in set(orig)] == []


def test_the_copy_trains_on_the_offer_prompt():
    pytest.importorskip("datasets")
    import train_grpo_ask as T
    ds = T.build_dataset([_case(1), dict(_case(2), note_class=None)])
    assert ds[0]["prompt"] == P.turn1_messages(_case(1))
    assert ds[0]["prompt"][0]["content"] == P.OFFER_SYSTEM
    assert ds["note_class"] == ["unstated", "replay"]
    assert "gold_command" not in ds.column_names       # the reward judges by execution


# --- the reply bank -----------------------------------------------------------

def test_bank_answers_the_nearest_question_and_placebo_elsewhere():
    pairs = [{"id": "r01", "question": "Which word should I delete?", "reply": "The second."},
             {"id": "r01", "question": "Should the space go too?", "reply": "Yes, the space too."},
             {"id": "r01", "question": "Which word should I delete?", "reply": "duplicate"},
             {"id": "r02", "question": "Q?", "reply": "leaky", "leak_flags": ["key notation"]}]
    bank = B.ReplyBank(pairs, P.PLACEBO_REPLY)
    assert len(bank) == 2                                 # duplicate kept once, leak dropped
    reply, sim, src = bank.reply("r01", "which word do I delete")
    assert (reply, src) == ("The second.", "bank") and 0 < sim < 1
    assert bank.reply("r01", "should the space go too")[0] == "Yes, the space too."
    assert bank.reply("r02", "Q?") == (P.PLACEBO_REPLY, None, "placebo")
    assert bank.reply("r99", "anything") == (P.PLACEBO_REPLY, None, "placebo")


def test_bank_pairs_join_the_diag_verdicts_and_stage1_dialogues():
    verdicts = [{"id": "r01", "question": "G?", "reply": "g", "leak_flags": [], "quotes_result": False},
                {"id": "zz", "question": "G?", "reply": "g"}]
    dialogues = [{"id": "r01", "question": "S?", "reply": "s", "leak_flags": []}]
    pairs = B.bank_pairs(verdicts, dialogues, {"r01"})
    assert [p["question"] for p in pairs] == ["G?", "S?"]


# --- the rollout on a scripted model --------------------------------------------

class FakeTok:
    """Characters are tokens; 0 = end of turn, 1 = pad."""
    eos_token_id, pad_token_id = 0, 1
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "|".join(m["content"] for m in messages) + "#"

    def __call__(self, texts, return_tensors=None, padding=False):
        import torch
        if isinstance(texts, str):
            return {"input_ids": [ord(c) for c in texts]}
        ids = [[ord(c) for c in t] for t in texts]
        width = max(len(i) for i in ids)
        rows = [[1] * (width - len(i)) + i for i in ids]          # left padding

        class Enc(dict):
            def to(self, device):
                return self
        return Enc(input_ids=torch.tensor(rows), attention_mask=torch.ones(len(rows), width))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids if not (skip_special_tokens and i in (0, 1)))


class FakeModel:
    """Turn 1: the first sample of every prompt asks, the rest answer.
    Turn 2 (prompts carrying a reply): answers with the reply's first word."""
    training = False

    class config:
        use_cache = True

    def __init__(self):
        self.calls = []

    def eval(self):
        self.training = False

    def train(self, mode=True):
        self.training = mode

    def parameters(self):
        import torch
        yield torch.zeros(1)

    def generate(self, input_ids, attention_mask=None, do_sample=False, **kw):
        import torch
        self.calls.append(("sample" if do_sample else "greedy", input_ids.shape[0]))
        outs = []
        for k, row in enumerate(input_ids.tolist()):
            text = "".join(chr(t) for t in row if t != 1)
            if do_sample:
                gen = "ask: which one?" if k % 2 == 0 else "keys: x"
            else:
                gen = "keys: " + text.split("|")[-1].split()[0].rstrip("#.")
            outs.append(row + [ord(c) for c in gen] + [0, 1, 1])
        width = max(len(o) for o in outs)
        return torch.tensor([o + [1] * (width - len(o)) for o in outs])


class FakeTrainer:
    def __init__(self, model):
        class Acc:
            @staticmethod
            def unwrap_model(m):
                return m
        self.accelerator, self.model = Acc(), model


def test_rollout_asks_answers_from_the_bank_and_masks_the_question():
    pytest.importorskip("torch")
    tok, model = FakeTok(), FakeModel()
    cases = [_case(1), _case(2, "replay")]
    bank = B.ReplyBank([{"id": "r01", "question": "Which one?", "reply": "lpha please"}],
                       P.PLACEBO_REPLY)
    by_user = {P.turn1_messages(c)[-1]["content"]: c for c in cases}
    stats = {}
    roll = R.make_rollout(tok, {}, by_user, bank, max_new_tokens=32, temperature=1.0, stats=stats)
    prompts = [P.turn1_messages(cases[0])] * 2 + [P.turn1_messages(cases[1])] * 2
    out = roll(prompts, FakeTrainer(model))
    assert out["logprobs"] is None
    assert out["asked"] == [1, 0, 1, 0]
    assert [tok.decode(c) for c in out["completion_ids"]] == ["ask: which one?", "keys: x"] * 2
    assert all(c[-1] == 0 for c in out["completion_ids"])            # each ends at end-of-turn
    assert out["env_mask"][0] == [1] + [0] * (len(out["completion_ids"][0]) - 1)
    assert out["env_mask"][1] == [1] * len(out["completion_ids"][1])
    # turn 2 answered the bank's reply for r01 and the placebo for the replay row
    assert out["final_text"] == ["keys: lpha", "keys: x", "keys: Please", "keys: x"]
    assert model.calls == [("sample", 4), ("greedy", 2)]
    assert stats == {"asks": 2, "bank_hits": 1, "placebo": 1, "mean_similarity": 1.0}


def test_ask_cost_scores_the_final_command_and_charges_the_question():
    seen = {}

    def executor_reward(prompts, completions, completion_ids, **kw):
        seen["texts"] = [c[0]["content"] for c in completions]
        seen["ids"] = completion_ids
        seen["kw"] = sorted(kw)
        return [1.0 if c[0]["content"] == "keys: right" else 0.1 for c in completions]

    logged = {}
    reward = R.with_ask_cost(executor_reward, 0.2, {"bank_hits": 1, "placebo": 0,
                                                    "mean_similarity": 0.5})
    got = reward(["p"] * 3, ["ask: q", "keys: wrong", "ask: q"], [[5], [6], [7]],
                 asked=[1, 0, 1], final_text=["keys: right", "keys: wrong", "keys: wrong"],
                 final_ids=[[8], [6], [9]], id=["a", "b", "c"], note_class=["unstated"] * 3,
                 log_metric=lambda k, v: logged.__setitem__(k, v))
    assert got == pytest.approx([0.8, 0.1, -0.1])
    assert seen["texts"] == ["keys: right", "keys: wrong", "keys: wrong"]
    assert seen["ids"] == [[8], [6], [9]]
    assert "asked" not in seen["kw"] and "id" in seen["kw"]
    assert logged["ask/rate"] == pytest.approx(2 / 3)
    assert logged["ask/rate_unstated"] == pytest.approx(2 / 3)
    assert reward.__name__ == "executor_reward"


def test_masks_and_eos_cutting():
    assert R.ask_mask(4, True) == [1, 0, 0, 0] and R.ask_mask(3, False) == [1, 1, 1]
    assert R.cut_at_eos([5, 6, 0, 7], 0) == [5, 6, 0]
    assert R.cut_at_eos([5, 6], 0) == [5, 6]


# --- the prompt set -------------------------------------------------------------

def test_prompt_set_keeps_held_out_rows_out_and_labels_replay():
    import build_prompts as BP
    diag_train = [_case(1), _case(2, "model")]
    train = [_case(i, None) for i in range(1, 12)]
    notes = {"r02": "model", "r05": "unstated", "r06": "model"}
    rows = BP.build(diag_train, ["r03"], train, notes, n_replay=20, seed=0)
    ids = {r["id"] for r in rows}
    assert "r03" not in ids and "r05" not in ids          # held out; a noted non-model row
    assert {"r01", "r02", "r06"} <= ids
    assert {r["note_class"] for r in rows if r["id"] not in ("r01", "r02")} == {"replay"}
    with pytest.raises(AssertionError):
        BP.build(diag_train, ["r01"], train, notes, 5, 0)
