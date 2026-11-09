"""Tests for R2 (CPU only): `pytest scripts/think_train`."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
R1 = REPO / "runs" / "think_label_2026-11-01"


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = load("think_train_build_r2", HERE / "build_r2.py")
T = load("think_train_train_r2", HERE / "train_r2.py")


def case(cid, mode="ex", command="d", instruction="do it"):
    return {"id": cid, "instruction": instruction, "buffer_before": "alpha\nbeta\ngamma\n", "cursor": [1, 1],
            "mode": mode, "gold_command": command, "buffer_after": "", "category": "x", "tier": "T1"}


def label(cid, lab=True, mode="ex", command="2d", check="equivalent"):
    return {"id": cid, "in_cliff": True, "in_never": False, "tier": "T2", "category": "x", "gold": ["ex", "d"],
            B.THINK: {"mode": mode, "command": command, "match": lab, "check": check if lab else None, "label": lab}}


def r1row(cid, closed=True, cap=False, forced=False, mode="ex", command="2d"):
    return {"id": cid, "closed": closed, "hit_cap": cap, "forced_close": forced, "mode": mode, "command": command,
            "reasoning": "think about it\n", "answer": "\n\n%s: %s" % (mode, command), "new_tokens": 9,
            "think_tokens": 5}


def tokenizer():
    from transformers import AutoTokenizer
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    reg = load_registry(B.STUDENT)
    try:
        return AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"], local_files_only=True), reg
    except Exception:
        pytest.skip("the tokenizer is not in the local cache")


# ------------------------------------------------------------------ the data

def test_kept_rows_need_a_label_and_a_trace_that_closed_on_its_own():
    labels = [label("a"), label("b", lab=False), label("c"), label("d"), label("e"), label("f")]
    rows = {"a": r1row("a"), "b": r1row("b"), "c": r1row("c", cap=True), "d": r1row("d", forced=True),
            "e": r1row("e", closed=False)}                            # f has no R1 row
    assert [k for k, _, _ in B.kept_rows(labels, rows)] == ["a"]
    with pytest.raises(SystemExit):                                   # the two files must agree
        B.kept_rows([label("a", command="3d")], {"a": r1row("a")})


def test_a1_keeps_approach_1_and_adds_one_row_per_label_with_the_teachers_answer():
    a0 = [case("a", "ex", "2d"), case("b", "keys", "dd"), case("c")]
    kept = [("a", {"mode": "ex", "command": "2d"}, None), ("b", {"mode": "ex", "command": "1d"}, None)]
    a1 = B.a1_rows(a0, kept)
    assert a1[:3] == a0 and len(a1) == 5
    assert [r["id"] for r in a1[3:]] == ["a" + B.EXTRA, "b" + B.EXTRA]
    assert (a1[4]["mode"], a1[4]["gold_command"]) == ("ex", "1d")       # a mode switch
    assert {r["id"] for r in a1} == {r["id"] for r in a0} | {"a@think", "b@think"}
    assert B.same_as_target(a0, kept) == {"a": True, "b": False}
    with pytest.raises(SystemExit):
        B.a1_rows(a0, [("zz", {"mode": "ex", "command": "d"}, None)])


def test_the_gate_waits_for_both_arms_and_for_l_to_help(tmp_path):
    assert B.gate(tmp_path)[0] is False                              # no report yet
    rep = {"n": 795, "arms": {B.CONTROL: {"attempted": 795}, B.THINK: {"attempted": 714}},
           "readings": {"L": {"verdict": "stronger"}}, "verdict": "helps"}
    (tmp_path / "labels.jsonl").write_text("{}\n")
    (tmp_path / "report.json").write_text(json.dumps(rep))
    ok, why = B.gate(tmp_path)
    assert not ok and "714 of 795" in why
    rep["arms"][B.THINK]["attempted"] = 795
    (tmp_path / "report.json").write_text(json.dumps(rep))
    assert B.gate(tmp_path) == (True, "R1 complete (795 rows per arm); L helps")
    rep["verdict"] = "tied"
    (tmp_path / "report.json").write_text(json.dumps(rep))
    assert B.gate(tmp_path)[0] is False


def test_the_student_entry_renders_r1s_recorded_prompts_and_traces():
    """The training-only entry reproduces R1's prompts token for token, and a trace decodes to the
    recorded text and ends with the template's end of turn (on R1's own rows, where present)."""
    if not (R1 / "prompts_vllm.jsonl").exists() or not (R1 / ("rows_%s.jsonl" % B.THINK)).exists():
        pytest.skip("R1's rows are local-only")
    tok, reg = tokenizer()
    kwargs = reg["chat_template_kwargs"]
    assert B.turn_end(tok, kwargs) == "<|im_end|>\n"
    train = {c["id"]: c for c in B.load_jsonl(str(B.TRAIN))}
    plan = {r["id"]: r for r in B.load_jsonl(str(R1 / "plan_retr8.jsonl"))}
    prompts = {r["id"]: r for r in B.load_jsonl(str(R1 / "prompts_vllm.jsonl"))}
    rows = B.last_copy(B.load_jsonl(str(R1 / ("rows_%s.jsonl" % B.THINK))))
    done = [r for r in rows.values() if r["closed"] and not r.get("forced_close")][:12]
    assert done
    for r in done:
        tr = B.render_trace(tok, kwargs, train[r["id"]], [train[j] for j in plan[r["id"]]["examples"]],
                            prompts[r["id"]][B.THINK], r, B.turn_end(tok, kwargs))
        assert tr["prompt_ids"] == prompts[r["id"]][B.THINK]["ids"]
        assert tok.decode(tr["completion_ids"][-2:]) == "<|im_end|>\n"


# ------------------------------------------------------------------ the trainer

def test_trace_order_is_seeded_and_each_pass_covers_every_trace():
    import itertools
    a = list(itertools.islice(T.trace_order(5, 42), 15))
    assert a == list(itertools.islice(T.trace_order(5, 42), 15))
    assert a != list(itertools.islice(T.trace_order(5, 7), 15))
    for k in range(3):
        assert sorted(a[5 * k:5 * k + 5]) == list(range(5))
    assert a[:5] != a[5:10] or a[5:10] != a[10:15]                   # a fresh shuffle per pass
    assert T.longest_index([("x", [1], [2, 3]), ("y", [1, 2], [3, 4]), ("z", [1], [2])]) == 1


def tiny_model(tok):
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                            layer_types=["full_attention", "full_attention"], full_attention_interval=1,
                            tie_word_embeddings=False, pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)


def tiny_traces(tok):
    msgs = [{"role": "user", "content": "Instruction: delete line 2"}]
    out = []
    for k, body in enumerate(["first, find line 2\n", "line two is beta; delete it with 2d, a longer thought\n"]):
        p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=True)
        c = body + "</think>\n\nex: 2d<|im_end|>\n"
        out.append(("t%d" % k, tok(p)["input_ids"], tok(c, add_special_tokens=False)["input_ids"]))
    return out


def train_tiny(tmp_path, tok, trainer_cls, steps=2):
    import torch
    from peft import LoraConfig
    from trl import SFTConfig
    sys.path.insert(0, str(REPO / "scripts"))
    import train_sft
    model = tiny_model(tok)
    cases = [case("a", "ex", "2d"), case("b", "keys", "dd", "delete the line"), case("c", "ex", "3d"),
             case("d", "ex", "%s/a/b/")]
    ds = train_sft.build_dataset(cases, {"chat_template_kwargs": {"enable_thinking": False}})
    args = SFTConfig(output_dir=str(tmp_path), per_device_train_batch_size=1, gradient_accumulation_steps=2,
                     max_steps=steps, learning_rate=1e-2, max_length=1024, completion_only_loss=True, bf16=False,
                     use_cpu=True, report_to="none", save_strategy="no", logging_steps=1, seed=0)
    torch.manual_seed(0)
    trainer = trainer_cls(model=model, args=args, train_dataset=ds, processing_class=tok,
                          peft_config=LoraConfig(r=4, lora_alpha=8, target_modules="all-linear"))
    trainer.train()
    weights = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    return trainer, weights


def test_weight_zero_is_trls_training_and_weight_one_adds_one_trace_per_optimizer_step(tmp_path):
    """Through TRL's real training loop on a tiny Qwen3.5 under LoRA: at weight 0 the trained adapter
    is plain TRL's, bit for bit, and no trace runs; at weight 1 one trace runs per optimizer step
    (2 steps of 2 micro-batches: 2 traces, not 4), and the adapter moves."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    trl = pytest.importorskip("trl")
    tok, _ = tokenizer()
    saved = dict(T.TRACE)
    try:
        T.TRACE.update(items=tiny_traces(tok), seed=42, check=False)
        _, plain = train_tiny(tmp_path / "plain", tok, trl.SFTTrainer)
        T.TRACE["weight"] = 0.0
        t0, w0 = train_tiny(tmp_path / "w0", tok, T.make_trace_trainer(trl.SFTTrainer))
        assert t0.traces_run == 0
        assert plain.keys() == w0.keys() and all(torch.equal(plain[k], w0[k]) for k in plain)
        T.TRACE["weight"] = 1.0
        t1, w1 = train_tiny(tmp_path / "w1", tok, T.make_trace_trainer(trl.SFTTrainer))
        assert t1.traces_run == 2
        assert any(not torch.equal(plain[k], w1[k]) for k in plain)
    finally:
        T.TRACE.clear()
        T.TRACE.update(saved)


def test_trace_nll_is_the_mean_cross_entropy_over_the_completion_in_any_chunking():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    tok, _ = tokenizer()
    model = tiny_model(tok)
    _, p, c = tiny_traces(tok)[1]
    ids = torch.tensor([p + c])
    logits = model(input_ids=ids).logits[0, len(p) - 1:-1]
    want = F.cross_entropy(logits.float(), ids[0, len(p):])
    saved = T.CHUNK
    try:
        for chunk in (1024, 3):
            T.CHUNK = chunk
            got, n = T.trace_nll(model, p, c, "cpu")
            assert n == len(c) and float(got) == pytest.approx(float(want), rel=1e-5)
    finally:
        T.CHUNK = saved


def test_check_runs_the_longest_trace_once_more_and_stops(tmp_path):
    pytest.importorskip("peft")
    trl = pytest.importorskip("trl")
    tok, _ = tokenizer()
    saved = dict(T.TRACE)
    try:
        T.TRACE.update(items=tiny_traces(tok), weight=1.0, seed=42, check=True)
        with pytest.raises(T.CheckDone) as done:
            train_tiny(tmp_path, tok, T.make_trace_trainer(trl.SFTTrainer), steps=1)
        info = done.value.args[0]
        assert info["trace"] == "t1" and info["tokens"] == len(T.TRACE["items"][1][1]) + len(T.TRACE["items"][1][2])
        assert info["trace_seconds"] >= 0 and info["trace_seconds_first"] >= 0
    finally:
        T.TRACE.clear()
        T.TRACE.update(saved)


# ------------------------------------------------------------------ the report and the driver

def test_the_report_uses_the_shared_rule_and_pairs_the_seeds():
    Q = load("think_train_compare_r2", HERE / "compare_r2.py")
    up, s = {"ci95": [0.4, 5.0]}, lambda d: {"diff_pts": d}
    assert Q.call(up, s(3.0), s(1.0)) == "helps" and Q.call(up, s(6.0), s(-0.5)) == "tied"
    assert Q.ARMS["a0-s42"] == "runs/qwen3.5-4b-sft-distill" and Q.ARMS["a0-s7"] == "runs/qwen3.5-4b-sft-distill-s7"
    assert Q.READINGS["P1"][:2] == ("a1", "a2") and Q.READINGS["P2"][:2] == ("a0", "a1")


def test_driver_parses_and_runs_the_preregistered_commands():
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert 'TRACE_ARGS=(--traces "$TRACES" --trace-weight 1)' in src
    assert "A2_MAX_GIB=${A2_MAX_GIB:-20.5}" in src and "BUDGET_H=${BUDGET_H:-12}" in src
    assert 'for spec in "a1-42 42 a1" "a2-42 42 a2" "a1-7 7 a1" "a2-7 7 a2"; do' in src
    assert '--eval-every $every --no-eval-at-start --seed "$seed" --wandb' in src
    assert 'while kill -0 "$WAIT_PID"' in src and "\n  wait\n" not in src     # never a bare wait (OPD's hang)


def test_no_model_ids_and_no_hardcoded_template_flags_in_this_folder():
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh") and not f.name.startswith("test_"):
            src = f.read_text(encoding="utf-8")
            assert "Qwen/" not in src and "enable_thinking" not in src, f.name
