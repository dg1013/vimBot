"""Tests for the soft-target KD trainer (CPU only): `pytest scripts/kd27b`."""
import importlib.util
import json
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


K = load("kd27b_train_kd", HERE / "train_kd.py")


def case(cid, mode, command, instruction="do it"):
    return {"id": cid, "instruction": instruction, "buffer_before": "alpha\nbeta\ngamma\n", "cursor": [1, 1],
            "mode": mode, "gold_command": command, "buffer_after": "", "category": "x", "tier": "T1"}


# ------------------------------------------------------------ the loss

def test_kd_sums_match_a_hand_computation_and_reduce_to_the_nll_at_alpha_zero():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, 0.5]])
    target = torch.tensor([0, 1])
    t_ids = torch.tensor([[0, 2], [1, 3]])
    t_logits = torch.tensor([[4.0, 2.0], [5.0, 1.0]])
    T, a = 2.0, 0.5
    loss, nll, soft, t_ent, correct, ent = K.kd_sums(logits, target, t_ids, t_logits, a, T)
    lp1, lpT = torch.log_softmax(logits, -1), torch.log_softmax(logits / T, -1)
    want_nll = -(lp1[0, 0] + lp1[1, 1])
    q = torch.softmax(t_logits / T, -1)
    want_soft = -sum(float(q[r, j] * lpT[r, t_ids[r, j]]) for r in range(2) for j in range(2))
    assert float(nll) == pytest.approx(float(want_nll), abs=1e-6)
    assert float(soft) == pytest.approx(want_soft, abs=1e-6)
    assert float(loss) == pytest.approx((1 - a) * float(want_nll) + a * T * T * want_soft, abs=1e-5)
    assert float(correct) == 2.0
    assert float(t_ent) == pytest.approx(-float((q * q.log()).sum()), abs=1e-6)
    zero = K.kd_sums(logits, target, t_ids, t_logits, 0.0, T)
    assert float(zero[0]) == pytest.approx(float(want_nll), abs=1e-6) and float(zero[2]) == 0.0


def test_the_soft_term_is_smallest_where_the_student_matches_the_teacher():
    """KL(q || p_T) >= 0, with equality when the student's tempered distribution over the top k
    is the teacher's and it puts no mass elsewhere (up to the vocabulary's tail)."""
    torch = pytest.importorskip("torch")
    t_logits = torch.tensor([[3.0, 1.0]])
    t_ids = torch.tensor([[0, 1]])
    match = torch.tensor([[3.0, 1.0, -1e4, -1e4]])
    off = torch.tensor([[1.0, 3.0, -1e4, -1e4]])
    kl = lambda lg: float(K.kd_sums(lg, torch.tensor([0]), t_ids, t_logits, 1.0, 2.0)[2]
                          - K.kd_sums(lg, torch.tensor([0]), t_ids, t_logits, 1.0, 2.0)[3])
    assert kl(match) == pytest.approx(0.0, abs=1e-5)
    assert kl(off) > 0.1


# ------------------------------------------------------------ data

def test_load_teacher_keeps_the_top_k_and_refuses_a_short_record(tmp_path):
    p = tmp_path / "probs.jsonl"
    good = {"id": "a", "n": 2, "ids": [5, 6], "topk_ids": [[5, 1, 2], [6, 3, 4]],
            "topk_logits": [[9.0, 1.0, 0.5], [8.0, 2.0, 1.0]]}
    p.write_text(json.dumps(good) + "\n", encoding="utf-8")
    t = K.load_teacher(p, 2)
    assert t["a"] == ([5, 6], [[5, 1], [6, 3]], [[9.0, 1.0], [8.0, 2.0]])
    bad = dict(good, id="b", ids=[5, 6, 7], n=3)
    p.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        K.load_teacher(p, 2)


def test_build_dataset_adds_the_teacher_columns_and_refuses_missing_rows():
    pytest.importorskip("datasets")
    sys.path.insert(0, str(REPO / "scripts"))
    import train_sft
    teacher = {"a": ([1, 2], [[1, 3], [2, 4]], [[5.0, 1.0], [6.0, 0.0]])}
    build = K.patched_build_dataset(train_sft.build_dataset, teacher)
    ds = build([case("a", "ex", "%s/a/b/")], {"chat_template_kwargs": {"enable_thinking": False}})
    row = ds[0]
    assert row["teacher_target"] == [1, 2] and row["teacher_ids"] == [[1, 3], [2, 4]]
    assert row["completion"][0]["content"] == "ex: %s/a/b/"
    with pytest.raises(SystemExit):
        build([case("zz", "ex", "x")], {})


def test_collator_concatenates_in_label_order_and_refuses_a_shift():
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    col = K.KDCollator(DataCollatorForLanguageModeling(pad_token_id=0))
    ex = [{"input_ids": [7, 8, 9, 10], "labels": [-100, -100, 9, 10], "teacher_target": [9, 10],
           "teacher_ids": [[9, 1], [10, 2]], "teacher_logits": [[3.0, 0.0], [2.0, 1.0]]},
          {"input_ids": [7, 11, 12], "labels": [-100, 11, 12], "teacher_target": [11, 12],
           "teacher_ids": [[11, 1], [12, 1]], "teacher_logits": [[1.0, 0.0], [4.0, 0.0]]}]
    batch = col(ex)
    assert batch["input_ids"].shape == (2, 4)
    assert batch["teacher_ids"].tolist() == [[9, 1], [10, 2], [11, 1], [12, 1]]
    lbl = batch["labels"][:, 1:]
    assert lbl[lbl != -100].tolist() == [9, 10, 11, 12]
    ex[1]["teacher_target"] = [12, 11]
    with pytest.raises(RuntimeError):
        col(ex)


# ------------------------------------------------------------ against TRL's own loss

def tiny_trainer(tmp_path, tok, cases, teacher, alpha):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from trl import SFTConfig, SFTTrainer
    sys.path.insert(0, str(REPO / "scripts"))
    import train_sft
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            layer_types=["full_attention", "full_attention"], full_attention_interval=1,
                            tie_word_embeddings=False, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    reg = {"chat_template_kwargs": {"enable_thinking": False}}
    ds = K.patched_build_dataset(train_sft.build_dataset, teacher)(cases, reg)
    args = SFTConfig(output_dir=str(tmp_path), per_device_train_batch_size=len(cases), max_steps=1,
                     max_length=1024, completion_only_loss=True, bf16=False, use_cpu=True,
                     report_to="none", save_strategy="no", logging_steps=1, seed=0)
    K.KD.update(alpha=alpha, temperature=2.0, topk=2)
    return K.make_kd_trainer(SFTTrainer)(model=model, args=args, train_dataset=ds, processing_class=tok,
                                         peft_config=LoraConfig(r=4, lora_alpha=8, target_modules="all-linear")), SFTTrainer


def test_alpha_zero_is_trls_own_loss_and_the_targets_survive_its_pipeline(tmp_path):
    """On a tiny Qwen3.5 under LoRA with the real tokenizer: the teacher columns survive TRL's
    preparation and the column filter, the collator's alignment check passes, and at alpha = 0
    the KD loss equals TRL's chunked NLL on the same batch. At alpha = 0.5 it differs, and the
    gradient reaches the adapters."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    reg = load_registry("qwen3.5-4b")
    try:
        tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"], local_files_only=True)
    except Exception:
        pytest.skip("the tokenizer is not in the local cache")
    R = load("kd27b_record", REPO / "scripts" / "teacher_probs" / "record.py")
    cases = [case("a", "ex", "%s/alpha/omega/g"), case("b", "keys", "dd", "delete the line"),
             case("c", "ex", "2,3d", "delete lines two and three")]
    teacher = {}
    gen = torch.Generator().manual_seed(1)
    for c in cases:
        p, full = R.encode(tok, c, c["mode"], c["gold_command"], reg["chat_template_kwargs"])
        comp = full[len(p):]
        other = torch.randint(0, len(tok), (len(comp),), generator=gen).tolist()
        teacher[c["id"]] = (comp, [[t, o] for t, o in zip(comp, other)], [[3.0, 1.0]] * len(comp))
    saved = dict(K.KD)
    try:
        trainer, SFTTrainer = tiny_trainer(tmp_path, tok, cases, teacher, alpha=0.0)
        batch = next(iter(trainer.get_train_dataloader()))
        assert {"teacher_ids", "teacher_logits"} <= set(batch)
        assert batch["teacher_ids"].shape[0] == int((batch["labels"][:, 1:] != -100).sum())
        trl_inputs = {k: v for k, v in batch.items() if not k.startswith("teacher_")}
        ours = trainer.compute_loss(trainer.model, dict(batch))
        theirs = SFTTrainer.compute_loss(trainer, trainer.model, dict(trl_inputs))
        assert float(ours) == pytest.approx(float(theirs), abs=1e-5)
        assert trainer._metrics["train"]["nll"][-1] == pytest.approx(float(theirs), abs=1e-5)
        K.KD["alpha"] = 0.5
        mixed = trainer.compute_loss(trainer.model, dict(batch))
        assert math.isfinite(float(mixed)) and abs(float(mixed) - float(theirs)) > 1e-3
        mixed.backward()
        grads = [p.grad for n, p in trainer.model.named_parameters() if p.requires_grad and "lora" in n]
        assert grads and any(g is not None and float(g.abs().sum()) > 0 for g in grads)
    finally:
        K.KD.clear()
        K.KD.update(saved)


# ------------------------------------------------------------ the report and the driver

def test_k_needs_the_pooled_interval_and_both_seeds_on_one_side():
    Q = load("kd27b_compare_kd", HERE / "compare_kd.py")
    up, s = {"ci95": [0.4, 5.0]}, lambda d: {"diff_pts": d}
    assert Q.call(up, s(3.0), s(1.0)) == "helps"
    assert Q.call(up, s(6.0), s(-0.5)) == "tied"                 # one seed carries the interval
    assert Q.call({"ci95": [-1.0, 4.0]}, s(2.0), s(2.0)) == "tied"
    assert Q.call({"ci95": [-6.0, -0.2]}, s(-3.0), s(-2.0)) == "hurts"
    assert set(Q.ARMS) >= {"kd-s42", "kd-s7", "distill-s42", "distill-s7"}
    assert Q.ARMS["distill-s7"] == "runs/qwen3.5-4b-sft-distill-s7"


def test_driver_parses_and_runs_the_preregistered_commands():
    import subprocess
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert "KD_ARGS=(--teacher-probs \"$PROBS\" --kd-alpha 0.5 --kd-temperature 2 --kd-topk 20)" in src
    assert "DISTILLED=runs/distill_2026-10-31/train_distilled.jsonl" in src
    assert "kd_arm kd-s42 42 runs/qwen3.5-4b-kd27b-s42" in src and "kd_arm kd-s7 7 runs/qwen3.5-4b-kd27b-s7" in src
    assert "--eval-every $every --no-eval-at-start --seed \"$seed\" --wandb" in src


def test_no_model_ids_and_no_thinking_in_this_folder():
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh") and f.name != "test_kd27b.py":
            src = f.read_text(encoding="utf-8")
            assert "Qwen/" not in src and "enable_thinking=True" not in src, f.name
