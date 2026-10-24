"""Tests for scripts/train_sft.py's model-free parts (CLAUDE.md §12).

The training loop needs the weights and the GPU (graphics processing unit) and is exercised by running the
script with --limit/--max-steps. These pin the decisions: what the model is
trained on, which tokens carry loss, how the val subset is drawn, and that the
training-time score and the results-table score share one generation path.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import train_sft  # noqa: E402
from prompt import messages_for_case, render_target  # noqa: E402

CASE = {"id": "x-1", "category": "substitute", "tier": "T1", "mode": "ex",
        "instruction": "replace cat with dog everywhere",
        "buffer_before": "cat\n", "buffer_after": "dog\n",
        "gold_command": "%s/cat/dog/g", "cursor": [1, 1]}


def _cases(n, cats=("substitute", "global", "join")):
    return [dict(CASE, id="c-%03d" % i, category=cats[i % len(cats)]) for i in range(n)]


# ------------------------------------------------------------------- data

def test_training_example_is_the_inference_prompt_plus_the_gold_answer():
    """Decision 1: train on exactly what is served. The prompt half must be
    byte-identical to what baseline.generate_predictions sends."""
    pytest.importorskip("datasets")
    ds = train_sft.build_dataset([CASE], {"chat_template_kwargs": {}})
    row = ds[0]
    assert row["prompt"] == messages_for_case(CASE)
    assert row["prompt"][-1]["role"] == "user"       # the answer is not in the prompt
    assert row["completion"] == [
        {"role": "assistant", "content": render_target(CASE["mode"], CASE["gold_command"])}]
    assert row["completion"][0]["content"] == "ex: %s/cat/dog/g"


def test_prompt_completion_shape_is_what_gives_an_exact_loss_mask():
    """Decision 2: TRL (Transformer Reinforcement Learning) tokenizes the two halves separately and concatenates, so
    the answer-only mask cannot drift at the boundary. That path is selected by
    the row having `prompt` and `completion` -- a single `text` field would take
    the other path and train on the buffer too."""
    pytest.importorskip("datasets")
    ds = train_sft.build_dataset([CASE], {"chat_template_kwargs": {}})
    assert set(ds.column_names) >= {"prompt", "completion"}
    assert "text" not in ds.column_names
    assert len(ds[0]["completion"]) == 1


def test_registry_template_flags_ride_on_the_row_not_the_config():
    """TRL reads chat_template_kwargs per example; SFTConfig has no such field.
    An empty dict is dropped -- an all-empty struct column has no Arrow type."""
    pytest.importorskip("datasets")
    with_flags = train_sft.build_dataset(
        [CASE], {"chat_template_kwargs": {"enable_thinking": False}})
    assert with_flags[0]["chat_template_kwargs"] == {"enable_thinking": False}
    without = train_sft.build_dataset([CASE], {"chat_template_kwargs": {}})
    assert "chat_template_kwargs" not in without.column_names


# ----------------------------------------------------------- val subsetting

def test_eval_subset_is_deterministic_and_category_proportional():
    cases = _cases(90)                      # 30 of each of three categories
    a = train_sft.stratified_subset(cases, 30)
    b = train_sft.stratified_subset(cases, 30)
    assert [c["id"] for c in a] == [c["id"] for c in b], "same seed, same rows"
    assert len(a) == 30
    counts = {}
    for c in a:
        counts[c["category"]] = counts.get(c["category"], 0) + 1
    assert set(counts.values()) == {10}, "val's category mix is preserved"


def test_eval_subset_returns_everything_when_it_is_not_smaller():
    cases = _cases(10)
    assert len(train_sft.stratified_subset(cases, 10)) == 10
    assert len(train_sft.stratified_subset(cases, 999)) == 10
    assert len(train_sft.stratified_subset(cases, 0)) == 10      # 0 disables


def test_eval_subset_keeps_a_rare_category_represented():
    cases = _cases(60) + [dict(CASE, id="rare-1", category="misc-ex")]
    picked = train_sft.stratified_subset(cases, 20)
    assert len(picked) == 20


# ------------------------------------------------------- generation context

class FakeModel:
    """Enough of a model to pin the state juggling around generation."""

    def __init__(self, checkpointing=True, training=True):
        self.training = training
        self.is_gradient_checkpointing = checkpointing
        self.config = SimpleNamespace(use_cache=False)
        self.log = []

    def eval(self):
        self.training = False
        self.log.append("eval")

    def train(self):
        self.training = True
        self.log.append("train")

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False
        self.log.append("ckpt_off")

    def gradient_checkpointing_enable(self, **kw):
        self.is_gradient_checkpointing = True
        self.log.append("ckpt_on")


def test_inference_mode_enables_the_cache_and_restores_training_state():
    """Gradient checkpointing forces use_cache=False; generating under it
    recomputes the prefix for every token. Both must come back on."""
    m = FakeModel()
    with train_sft.inference_mode(m):
        assert m.training is False
        assert m.config.use_cache is True
        assert m.is_gradient_checkpointing is False
    assert m.training is True
    assert m.config.use_cache is False
    assert m.is_gradient_checkpointing is True
    assert m.log == ["eval", "ckpt_off", "ckpt_on", "train"]


def test_inference_mode_restores_after_an_exception():
    """A failed eval must not leave training running without checkpointing --
    that comes back as an out-of-memory error hundreds of steps later."""
    m = FakeModel()
    with pytest.raises(RuntimeError):
        with train_sft.inference_mode(m):
            raise RuntimeError("sandbox blew up")
    assert m.training is True and m.is_gradient_checkpointing is True
    assert m.config.use_cache is False


def test_inference_mode_handles_a_model_without_checkpointing():
    m = FakeModel(checkpointing=False)
    with train_sft.inference_mode(m):
        assert m.config.use_cache is True
    assert m.log == ["eval", "train"]


# ------------------------------------------------------------- bookkeeping

def test_resolved_targets_sorts_the_set_peft_leaves_behind():
    """`all-linear` is rewritten by PEFT (parameter-efficient fine-tuning) into the modules it matched, as a set:
    no JSON (JavaScript Object Notation) type, and the expansion is the fact worth recording."""
    assert train_sft.resolved_targets(
        SimpleNamespace(target_modules={"v_proj", "q_proj"})) == ["q_proj", "v_proj"]
    assert train_sft.resolved_targets(
        SimpleNamespace(target_modules="all-linear")) == "all-linear"


def test_defaults_are_the_contract_in_section_8():
    d = {a.dest: a.default for a in train_sft.build_arg_parser()._actions}
    assert (d["lora_r"], d["lora_alpha"]) == (16, 32)
    assert d["lr"] == 1e-4
    assert d["batch_size"] * d["grad_accum"] == 32      # §8: effective batch 32-64
    assert d["max_length"] == 1024
    assert 2 <= d["epochs"] <= 3
    assert d["max_hours"] == 12.0                       # §14b hard stop


def test_one_generation_path_shared_with_the_results_table():
    """If this script grows its own generation loop, a training-time score and
    a RESULTS.md score stop being comparable."""
    src = (REPO / "scripts" / "train_sft.py").read_text(encoding="utf-8")
    assert "from baseline import" in src and "generate_predictions" in src
    assert "def generate_predictions" not in src
    assert ".generate(" not in src


def test_no_model_ids_outside_the_registry():
    src = (REPO / "scripts" / "train_sft.py").read_text(encoding="utf-8")
    assert "Qwen/" not in src and "hf_id: " not in src
    assert 'reg["hf_id"]' in src


def test_thinking_is_never_re_enabled_here():
    """§14: non-thinking everywhere, and no script may quietly flip it."""
    src = (REPO / "scripts" / "train_sft.py").read_text(encoding="utf-8")
    assert "enable_thinking\": True" not in src and "enable_thinking=True" not in src


def test_cli_help_needs_no_torch():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "train_sft.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0
    for flag in ("--model", "--inspect", "--eval-subset", "--max-hours"):
        assert flag in p.stdout


# ------------------------------------------------- QLoRA path (2026-09-09)

def test_four_bit_is_opt_in_and_the_recipe_is_otherwise_shared():
    """--load-in-4bit changes how the frozen base is stored and nothing else:
    the defaults the 4B trains with are the defaults the 9B trains with."""
    d = {a.dest: a.default for a in train_sft.build_arg_parser()._actions}
    assert d["load_in_4bit"] is False
    src = (REPO / "scripts" / "train_sft.py").read_text(encoding="utf-8")
    assert "load_base(reg, args.load_in_4bit)" in src        # training and the final reload
    assert "write_base_marker(" in src                        # best/ and final/ both say which base
    assert src.count("write_base_marker(") == 2
    # adapters are fp32 in both paths -- the upcast is unconditional, not gated on 4-bit
    assert "prm.data = prm.data.float()" in src


def test_cli_help_lists_the_qlora_flag():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "train_sft.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--load-in-4bit" in p.stdout
