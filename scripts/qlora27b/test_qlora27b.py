"""Tests for the 27B's QLoRA SFT (CPU only): `pytest scripts/qlora27b`.

The model tests run a tiny Qwen3.5 built from a config, full-attention layers only: the
linear-attention layers dispatch to Triton kernels whenever flash-linear-attention is installed,
and those need a GPU. The swap touches the embedding, generation and the LoRA backward, which
the full-attention layers exercise the same way.
"""
import importlib.util
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    """Import a module under a unique name: scripts/teacher27b has a report.py too, and pytest
    runs both folders in one process."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = load("qlora27b_train", HERE / "train_27b.py")
P = load("qlora27b_probe", HERE / "probe.py")


def need(path):
    p = pathlib.Path(path)
    if not p.exists():
        pytest.skip("%s not present" % p)
    return p


def tiny_model(seed=0):
    torch = pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    cfg = Qwen3_5TextConfig(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=16, layer_types=["full_attention", "full_attention"],
                            full_attention_interval=1, tie_word_embeddings=False)
    torch.manual_seed(seed)
    return AutoModelForCausalLM.from_config(cfg, dtype=torch.float32).eval()


# ------------------------------------------------------------ patch 1: the RAM embedding

def test_ram_embedding_is_an_exact_lookup():
    torch = pytest.importorskip("torch")
    emb = torch.nn.Embedding(50, 8)
    ram = W.ram_embedding(emb, "cpu")
    ids = torch.tensor([[0, 3, 49, 3], [7, 7, 1, 0]])
    assert torch.equal(ram(ids), emb(ids))
    assert ram.num_embeddings == 50 and ram.embedding_dim == 8


def test_ram_embedding_table_is_no_parameter_so_nothing_moves_it():
    torch = pytest.importorskip("torch")
    ram = W.ram_embedding(torch.nn.Embedding(50, 8), "cpu")
    assert list(ram.parameters()) == [] and ram.state_dict() == {}
    ram.to(torch.float64)                       # .to() walks parameters and buffers only
    assert ram.weight.dtype == torch.float32 and ram.weight.device.type == "cpu"
    assert not ram.weight.requires_grad


def test_swap_leaves_logits_and_generation_unchanged():
    torch = pytest.importorskip("torch")
    model = tiny_model()
    ids = torch.randint(0, 97, (2, 9), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        logits = model(input_ids=ids).logits
        gen = model.generate(ids[:1], max_new_tokens=4, do_sample=False)
    old = model.get_input_embeddings().weight
    freed = W.move_input_embedding_to_ram(model)
    assert freed == pytest.approx(97 * 32 * 4 / 2 ** 30)
    assert type(model.get_input_embeddings()).__name__ == "RamEmbedding"
    # generate_predictions places its inputs on next(model.parameters()).device
    assert next(model.parameters()) is not old
    assert all(p is not old for p in model.parameters())
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids).logits, logits)
        assert torch.equal(model.generate(ids[:1], max_new_tokens=4, do_sample=False), gen)


def test_swap_frees_the_old_table():
    """The first probe's leak: a RamEmbedding method that closed over the original module kept
    the GPU table alive (allocated 16.44 GiB before and after the swap)."""
    import gc
    import weakref
    model = tiny_model()
    ref = weakref.ref(model.get_input_embeddings())
    W.move_input_embedding_to_ram(model)
    gc.collect()
    assert ref() is None


def test_patched_loader_collects_the_previous_model_first():
    """train_sft's `del trainer, model` leaves reference cycles; the reload must not meet them."""
    import gc
    import weakref
    prev = tiny_model()
    ref = weakref.ref(prev)
    holder = {"model": prev}
    holder["self"] = holder                      # a cycle, as Trainer and accelerate leave
    del prev, holder
    seen = {}

    def orig(reg, q=False):
        seen["previous_alive"] = ref() is not None
        return tiny_model()

    gc.disable()                                 # only the loader's own collection may run
    try:
        W.patched_load_base(orig)({"name": "tiny"}, True)
    finally:
        gc.enable()
        W.LOAD.clear()
    assert seen == {"previous_alive": False}


def test_lora_trains_through_the_swap_with_checkpointing():
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    model = tiny_model()
    W.move_input_embedding_to_ram(model)
    lora = peft.get_peft_model(model, peft.LoraConfig(r=4, lora_alpha=8, target_modules="all-linear",
                                                      task_type="CAUSAL_LM"))
    # TRL's chunked_nll loss needs an unwrapped LM head; all-linear leaves it alone
    from peft.tuners.tuners_utils import BaseTunerLayer
    assert not isinstance(lora.get_base_model().get_output_embeddings(), BaseTunerLayer)
    lora.enable_input_require_grads()           # what TRL calls for PEFT + checkpointing
    lora.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lora.train()
    ids = torch.randint(0, 97, (2, 9), generator=torch.Generator().manual_seed(2))
    labels = ids.clone()
    labels[:, :5] = -100                        # the answer-only mask
    lora(input_ids=ids, labels=labels).loss.backward()
    grads = [p.grad for n, p in lora.named_parameters() if "lora_" in n]
    assert grads and all(g is not None for g in grads)
    assert any(float(g.abs().sum()) > 0 for g in grads)
    assert lora.get_base_model().get_input_embeddings().weight.grad is None


def test_patched_loader_moves_the_table_and_records_it():
    pytest.importorskip("torch")
    W.LOAD.clear()
    loader = W.patched_load_base(lambda reg, q=False: tiny_model())
    model = loader({"name": "tiny"}, True)
    assert type(model.get_input_embeddings()).__name__ == "RamEmbedding"
    assert W.LOAD["embedding_moved_to_ram_gib"] == pytest.approx(0.0, abs=0.01)
    assert set(W.LOAD) == {"allocated_after_load_gib", "allocated_after_patch_gib",
                           "embedding_moved_to_ram_gib"}
    W.LOAD.clear()


# ------------------------------------------------------------ patch 2: pace

def test_projected_hours_counts_the_evaluations_left():
    # 614 steps, eval every 204: evals at 204, 408, 612 and the off-cadence one at 614
    h = W.projected_hours(3600, 30.0, 100, 614, 204, 300)
    assert h == pytest.approx((3600 + 30 * 514 + 4 * 300) / 3600)
    assert W.projected_hours(0, 30.0, 612, 614, 204, 300) == pytest.approx((60 + 300) / 3600)
    assert W.projected_hours(0, 10.0, 0, 100, 0, 999) == pytest.approx(1000 / 3600)


def test_wrapper_flags_are_split_from_train_sft_flags():
    own, rest = W.build_arg_parser().parse_known_args(
        ["--model", "qwen3.5-27b", "--load-in-4bit", "--pace-every", "2", "--batch-size", "4",
         "--grad-accum", "8", "--eval-every", "204", "--no-eval-at-start", "--embedding-on-gpu"])
    assert own.pace_every == 2 and own.embedding_on_gpu and own.reproject_every == 50
    args = W.train_sft.build_arg_parser().parse_args(rest)
    assert (args.model, args.load_in_4bit, args.batch_size, args.grad_accum, args.eval_every) == \
        ("qwen3.5-27b", True, 4, 8, 204)


def test_final_eval_only_takes_train_sft_flags():
    own, rest = W.build_arg_parser().parse_known_args(
        ["--final-eval-only", "--model", "qwen3.5-27b", "--load-in-4bit", "--out-dir", "x"])
    assert own.final_eval_only
    args = W.train_sft.build_arg_parser().parse_args(rest)
    assert (args.model, args.load_in_4bit, args.out_dir) == ("qwen3.5-27b", True, "x")
    for name in ("full_val_eval", "load_registry", "load_jsonl"):  # what final_eval_only calls
        assert callable(getattr(W.train_sft, name))


def test_install_patches_what_train_sft_main_looks_up():
    S = pytest.importorskip("trl.trainer.sft_trainer")
    orig_load, orig_cb, orig_chunk = W.train_sft.load_base, W.train_sft.make_callbacks, S._chunk
    try:
        W.install(W.build_arg_parser().parse_args([]))
        assert W.train_sft.load_base is not orig_load
        assert W.train_sft.make_callbacks is not orig_cb
        assert getattr(S._chunk, "lean", False)
        lean = S._chunk
        W.install(W.build_arg_parser().parse_args([]))       # idempotent for the chunk
        assert S._chunk is lean
    finally:
        W.train_sft.load_base, W.train_sft.make_callbacks, S._chunk = orig_load, orig_cb, orig_chunk


# ------------------------------------------------------------ patch 3: the lean loss chunk

def chunk_inputs(seed=0):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(48, 64, generator=g).to(torch.bfloat16)
    w = (torch.randn(700, 64, generator=g) * 0.05).to(torch.bfloat16)
    lbl = torch.randint(0, 700, (48,), generator=g)
    lbl[-6:] = -100                                     # a chunk's padded tail
    return h, w, lbl


def test_lean_chunk_is_bf16_logits_then_fp32_softmax():
    """The arithmetic TRL's chunk performs under CUDA bf16 autocast (measured bit-identical on
    the GPU, 2026-10-31): a bf16 matmul, then the softmax, loss and metrics in fp32."""
    torch = pytest.importorskip("torch")
    S = pytest.importorskip("trl.trainer.sft_trainer")
    import torch.nn.functional as F
    h, w, lbl = chunk_inputs()
    loss, correct, entropy = W.lean_chunk(S)(h, w, None, lbl, 1.0, None)
    logits = (h @ w.t()).float()
    assert torch.equal(loss, F.cross_entropy(logits, lbl, ignore_index=-100, reduction="sum"))
    valid = lbl != -100
    assert float(correct) == float(((logits.argmax(-1) == lbl) & valid).sum())
    log_p = F.log_softmax(logits, -1)
    assert torch.allclose(entropy, (-(log_p.exp() * log_p).sum(-1) * valid).sum())


def test_lean_chunk_agrees_with_a_true_fp32_matmul_to_rounding():
    torch = pytest.importorskip("torch")
    S = pytest.importorskip("trl.trainer.sft_trainer")
    h, w, lbl = chunk_inputs(1)
    lean = W.lean_chunk(S)(h, w, None, lbl, 1.0, None)
    ref = S._chunk(h, w, None, lbl, 1.0, None) if not getattr(S._chunk, "lean", False) else None
    if ref is None:
        pytest.skip("TRL's own chunk is patched in this process")
    assert abs(float(lean[0]) - float(ref[0])) / float(ref[0]) < 1e-3
    assert abs(float(lean[2]) - float(ref[2])) / float(ref[2]) < 1e-3


def test_trl_chunked_loss_calls_the_lean_chunk_once_installed():
    torch = pytest.importorskip("torch")
    S = pytest.importorskip("trl.trainer.sft_trainer")
    h, w, lbl = chunk_inputs(2)
    hidden = h.reshape(2, 24, 64)                       # [batch, seq, hidden]
    labels = torch.full((2, 24), -100)
    labels[:, 15:] = torch.randint(0, 700, (2, 9))      # the answer tokens
    orig = S._chunk
    try:
        S._chunk = W.lean_chunk(S)
        loss, correct, entropy, n = S._chunked_cross_entropy_loss(hidden, w, 256, labels=labels)
    finally:
        S._chunk = orig
    shift_h, shift_l = hidden[:, :-1].reshape(-1, 64), labels[:, 1:].reshape(-1)
    keep = shift_l != -100
    want = W.lean_chunk(S)(shift_h[keep], w, None, shift_l[keep], 1.0, None)
    assert int(n) == int(keep.sum()) == 18              # 9 answer tokens x 2 rows
    assert torch.allclose(loss, want[0] / 18) and float(correct) == float(want[1])


def test_paced_guard_extends_the_wall_clock_guard():
    pytest.importorskip("transformers")
    from transformers import TrainerCallback
    make = W.patched_make_callbacks(W.train_sft.make_callbacks, 5, 50)
    HarnessEval, Guard = make(TrainerCallback)
    _, WallClockGuard = W.train_sft.make_callbacks(TrainerCallback)
    assert Guard.__name__ == "PacedGuard" and Guard.__mro__[1].__name__ == WallClockGuard.__name__
    assert HarnessEval.__name__ == "HarnessEval"


# ------------------------------------------------------------ the probe

def case(i, n):
    return {"id": "c%03d" % i, "instruction": "x" * n, "buffer_before": "", "gold_command": "dd",
            "category": "line-ops"}


def test_probe_takes_the_longest_rows_with_a_stable_tie_break():
    rows = [case(1, 5), case(2, 50), case(3, 50), case(4, 20)]
    assert [c["id"] for c in P.longest(rows, 3)] == ["c002", "c003", "c004"]
    assert P.row_length(case(9, 7)) == 9


def test_probe_verdict_passes_and_names_every_failure():
    mem = {"peak_reserved_gib": 18.0, "peak_allocated_gib": 17.0}
    ev = [{"n": 16, "parse_failures": 0}]
    ok = P.verdict(mem, ev, {"metrics": {"exec_match_pct": 50.0}}, 0)
    assert ok["pass"] and ok["reasons"] == []
    bad = P.verdict({"peak_reserved_gib": 21.0}, [{"n": 16, "parse_failures": 9}], None, 1)
    assert not bad["pass"] and len(bad["reasons"]) == 4
    assert any("21.00 GiB is over the 20.5" in r for r in bad["reasons"])
    assert P.verdict(None, [], {"metrics": {}}, 0)["reasons"] == [
        "no memory.json: training did not reach its end", "no evaluation ran"]
    leaked = dict(mem, embedding_moved_to_ram_gib=2.37, allocated_after_load_gib=16.44,
                  allocated_after_patch_gib=16.44)
    assert P.verdict(leaked, ev, {"metrics": {}}, 0)["reasons"] == [
        "the input embedding did not leave the GPU (allocated fell 0.00 GiB, not 2.37)"]
    freed = dict(leaked, allocated_after_patch_gib=14.07)
    assert P.verdict(freed, ev, {"metrics": {}}, 0)["pass"]


def test_probe_prepare_writes_both_files(tmp_path):
    train = tmp_path / "train.jsonl"
    val = tmp_path / "val.jsonl"
    P.write_jsonl(train, [case(i, i) for i in range(40)])
    P.write_jsonl(val, [dict(case(i, 3), category="c%d" % (i % 3)) for i in range(30)])
    assert P.main(["prepare", "--train", str(train), "--val", str(val), "--out", str(tmp_path / "o"),
                   "--rows", "8", "--val-rows", "6"]) == 0
    got = [json.loads(l) for l in (tmp_path / "o" / "probe_train.jsonl").read_text().splitlines()]
    assert [c["id"] for c in got] == ["c%03d" % i for i in range(39, 31, -1)]
    assert len((tmp_path / "o" / "probe_val.jsonl").read_text().splitlines()) == 6


# ------------------------------------------------------------ the report

def test_report_readings_and_rules():
    R = load("qlora27b_report", HERE / "report.py")
    ids = ["r%d" % i for i in range(40)]
    clean = ids[:20]
    match = {R.ARM: {i: 1 for i in ids},
             "9b": {i: int(k % 2 == 0) for k, i in enumerate(ids)},
             "4b-distill": {i: 0 for i in ids}}
    out = R.readings(match, clean, ids)
    assert set(out) == {"Q1", "Q2"}                       # 27b-prompted missing: no Q3
    assert out["Q2"]["untouched"]["verdict"] == "stronger" and out["Q2"]["full"]["n"] == 40
    assert out["Q1"]["untouched"]["gained"] == 10 and out["Q1"]["untouched"]["lost"] == 0
    assert "becomes the teacher" in R.next_step("stronger", "stronger")
    assert "uses the 9B" in R.next_step("tied", "stronger")
    for _, comp, _ in R.READINGS:
        assert comp in R.COMPARATORS


def test_explore_share_diff_calls_by_the_interval():
    E = load("qlora27b_explore", HERE / "explore.py")
    ids = ["r%d" % i for i in range(60)]
    half = {i: 0.5 for i in ids}
    assert E.share_diff({i: 1 for i in ids}, half, ids) == {
        "diff_pts": 50.0, "ci95": [50.0, 50.0], "verdict": "stronger"}
    assert E.share_diff({i: 0 for i in ids}, half, ids)["verdict"] == "weaker"
    r = E.share_diff({i: int(k % 2 == 0) for k, i in enumerate(ids)}, half, ids)
    assert r["diff_pts"] == 0.0 and r["verdict"] == "tied" and r["ci95"][0] < 0 < r["ci95"][1]


# ------------------------------------------------------------ contracts

def test_no_model_ids_and_no_thinking_in_this_folder():
    """§4: model ids live in the registry only; §14: thinking stays off.
    (This file is not scanned: it holds the needles.)"""
    for f in ("train_27b.py", "probe.py", "report.py", "explore.py", "test_report.py", "run.sh"):
        src = (HERE / f).read_text(encoding="utf-8")
        assert "Qwen/" not in src and "hf_id: " not in src, f
        assert "enable_thinking\": True" not in src and "enable_thinking=True" not in src, f


def test_built_probe_files_come_from_their_sources():
    run = need(REPO / "runs" / "qlora27b_2026-10-31" / "probe_train.jsonl").parent
    train = {json.loads(l)["id"]: json.loads(l)
             for l in need(P.TRAIN).read_text(encoding="utf-8").splitlines()}
    val = {json.loads(l)["id"] for l in need(P.VAL).read_text(encoding="utf-8").splitlines()}
    probe = [json.loads(l) for l in (run / "probe_train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(probe) == P.PROBE_ROWS and all(train[c["id"]] == c for c in probe)
    pval = [json.loads(l)["id"] for l in (run / "probe_val.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(pval) == P.PROBE_VAL_ROWS and set(pval) <= val
