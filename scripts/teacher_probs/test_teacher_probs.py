"""Tests for the 27B's recorded distributions (CPU only): `pytest scripts/teacher_probs`."""
import importlib.util
import math
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load("teacher_probs_record", HERE / "record.py")
A = load("teacher_probs_analyze", HERE / "analyze.py")


class FakeTok:
    """A character-level stand-in with a chat template whose generation prompt is a prefix of the
    rendered assistant turn, as Qwen3.5's is."""
    def apply_chat_template(self, messages, tokenize=True, return_dict=True, add_generation_prompt=False, **kw):
        text = "".join("[%s]%s|" % (m["role"], m["content"]) for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        return {"input_ids": [ord(c) for c in text]}


def case(cid, mode, command):
    return {"id": cid, "instruction": "do it", "buffer_before": "a\nb", "cursor": [1, 1],
            "mode": mode, "gold_command": command, "tier": "T1"}


# ------------------------------------------------------------ record.py

def test_divergence_and_answer_length():
    assert R.divergence([1, 2, 3], [1, 2, 4]) == 2
    assert R.divergence([1, 2], [1, 2, 3]) is None
    assert R.answer_len([5, 6, 99, 7], 99) == 2
    assert R.answer_len([5, 6], 99) == 2


def test_prepare_finds_retargeted_rows_and_where_the_gold_departs():
    distilled = [case("same", "ex", "%s/a/b/"), case("within", "keys", "x"), case("switch", "ex", ".,+1d")]
    golds = {c["id"]: c for c in [case("same", "ex", "%s/a/b/"), case("within", "keys", "dl"),
                                  case("switch", "keys", "dj")]}
    items, mismatched = R.prepare(FakeTok(), distilled, golds, {}, ord("|"))
    assert mismatched == []
    by = {it["id"]: it for it in items}
    assert by["same"]["gold"] is None
    comp = "".join(map(chr, by["within"]["completion"]))
    assert comp == "keys: x|" and by["within"]["n_answer"] == len("keys: x")
    assert by["within"]["gold"] == {"d": 6, "token": ord("d"), "mode_switch": False}
    assert by["switch"]["gold"] == {"d": 0, "token": ord("k"), "mode_switch": True}


def test_row_stats_matches_log_softmax_ranks_and_the_topk_probe():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, 0.0]])
    st = R.row_stats(logits, torch.tensor([0, 2]), topk=2, gold=(1, 3))
    lp = torch.log_softmax(logits, -1)
    assert st["tlp1"] == [round(float(lp[0, 0]), 4), round(float(lp[1, 2]), 4)]
    assert st["trank"] == [1, 2] and st["topk_ids"] == [[0, 1], [1, 2]]
    g = st["gold"]
    assert (g["d"], g["token"], g["rank"]) == (1, 3, 3)         # two logits above it
    assert g["q2"] == 0.0                                       # outside the top 2
    assert g["target_q2"] == round(1 / (1 + math.e), 5)         # softmax([3, 1] / 2)[1]


def test_both_topk_probes_agree():
    torch = pytest.importorskip("torch")
    z, ids = [4.0, 2.5, 1.0], [7, 3, 9]
    for tok_id in (7, 3, 9, 5):
        a = R.q_topk(torch.tensor(z), torch.tensor(ids), tok_id, 2.0)
        assert abs(a - A.q_topk(z, ids, tok_id, 2.0)) < 1e-6


def test_forward_rows_reads_the_positions_that_predict_the_completion():
    """On a tiny Qwen3.5 under LoRA (full-attention layers: the linear-attention kernels need a
    GPU), the padded batch gives each row's completion log-probabilities as its own full pass."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                            layer_types=["full_attention", "full_attention"], full_attention_interval=1,
                            tie_word_embeddings=False)
    torch.manual_seed(0)
    model = get_peft_model(AutoModelForCausalLM.from_config(cfg, dtype=torch.float32),
                           LoraConfig(r=4, target_modules="all-linear")).eval()
    inner = model.get_base_model()
    items = [{"ids": [5, 6, 7, 8, 9], "P": 3, "completion": [8, 9], "gold": None},
             {"ids": [5, 6, 10, 11], "P": 2, "completion": [10, 11],
              "gold": {"d": 1, "token": 12, "mode_switch": False}}]
    with torch.no_grad():
        batched = R.forward_rows(inner.model, inner.lm_head, items, "cpu", topk=5)
        for it, st in zip(items, batched):
            logits = model(input_ids=torch.tensor([it["ids"]])).logits[0].float()
            lp = torch.log_softmax(logits[it["P"] - 1:len(it["ids"]) - 1], -1)
            assert st["tlp1"] == pytest.approx([float(lp[j, t]) for j, t in enumerate(it["completion"])], abs=1e-3)
    assert batched[0].get("gold") is None and batched[1]["gold"]["token"] == 12


def test_encode_matches_the_trained_positions_with_the_real_tokenizer():
    """The masked part of a real row is exactly the §6 answer line and the template's end."""
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    reg = load_registry("qwen3.5-27b")
    try:
        tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"], local_files_only=True)
    except Exception:
        pytest.skip("the tokenizer is not in the local cache")
    c = case("x", "ex", "%s/foo/bar/g")
    p, full = R.encode(tok, c, c["mode"], c["gold_command"], reg.get("chat_template_kwargs") or {})
    assert full[:len(p)] == p
    comp = full[len(p):]
    n = R.answer_len(comp, tok.convert_tokens_to_ids("<|im_end|>"))
    assert tok.decode(comp[:n]) == "ex: %s/foo/bar/g"
    assert "<think>" not in tok.decode(comp)


# ------------------------------------------------------------ analyze.py

def test_verdict_bounds():
    assert A.verdict(0.50, 0.9, 0.02) == "informative"
    assert A.verdict(0.49, 0.9, 0.05) == "marginal"
    assert A.verdict(0.80, 0.9, 0.019) == "marginal"
    assert A.verdict(0.10, 0.24, 0.05) == "near one-hot"
    assert A.verdict(0.60, 0.90, 0.004) == "near one-hot"


def rec(rid, tlp1, gold=None):
    n = len(tlp1)
    return {"id": rid, "n": n + 1, "n_answer": n, "ids": [1] * (n + 1), "tlp1": tlp1 + [0.0],
            "ent1": [0.1] * (n + 1), "trank": [1] * (n + 1), "topk_ids": [[1, 2]] * (n + 1),
            "topk_logits": [[5.0, 0.0]] * (n + 1), "gold": gold}


def test_token_and_alternative_stats():
    recs = [rec("a", [math.log(0.95), math.log(0.5)]),
            rec("b", [0.0], gold={"d": 0, "token": 2, "rank": 2, "lp1": math.log(0.01), "q2": 0.07,
                                  "target_q2": 0.9, "mode_switch": False}),
            rec("c", [0.0], gold={"d": 0, "token": 3, "rank": 9, "lp1": math.log(1e-6), "q2": 0.0,
                                  "target_q2": 0.95, "mode_switch": True}),
            rec("d", [0.0], gold={"d": None, "mode_switch": False})]
    t = A.token_stats(recs, True)
    assert t["tokens"] == 5 and t["rows_with_branch"] == 0.25
    assert abs(t["m1"] - (0.05 + 0.5) / 5) < 1e-4
    rep = A.build(recs, {"b": "T1", "c": "T3"})
    a = rep["alternative"]["all re-targeted"]
    assert (rep["retargeted"], rep["no_divergence"], a["rows"]) == (3, 1, 2)
    assert a["top2"] == 0.5 and a["top5"] == 0.5 and a["q2_median"] == 0.035
    assert rep["verdict"] == "informative"
    assert rep["alternative"]["mode switches"]["rows"] == 1 and rep["alternative"]["T2"] == {"rows": 0}


def test_no_model_ids_and_no_thinking_in_this_folder():
    for f in ("record.py", "analyze.py", "run.sh"):
        src = (HERE / f).read_text(encoding="utf-8")
        assert "Qwen/" not in src and "enable_thinking=True" not in src, f
