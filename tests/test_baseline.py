"""Tests for scripts/baseline.py's model-free parts.

The generation loop needs the weights and the GPU (graphics processing unit) and is exercised by running
the script; these pin the pieces that decide what the numbers mean.
"""

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import baseline  # noqa: E402


class FakeTok:
    eos_token_id = 7

    def encode(self, s, add_special_tokens=False):
        return {"<|im_end|>": [7], "<|end|>": [9], "two tokens": [1, 2]}[s]


def test_stop_ids_are_single_tokens_plus_eos_without_duplicates():
    assert baseline.stop_token_ids(FakeTok(), ["<|im_end|>", "<|end|>", "two tokens"]) == [7, 9]


def test_percentile_bounds():
    assert baseline.percentile([], 50) is None
    assert baseline.percentile([3, 1, 2], 50) == 2
    assert baseline.percentile([3, 1, 2], 95) == 3
    assert baseline.percentile([3, 1, 2], 0) == 1


def test_summary_counts_parse_failures_as_failures():
    cases = [{"id": "a", "category": "x"}, {"id": "b", "category": "x"}]
    records = [
        {"id": "a", "output_tokens": 4, "prompt_tokens": 50, "latency_s": 0.5,
         "parse_ok": True, "think_leak": False},
        {"id": "b", "output_tokens": 40, "prompt_tokens": 60, "latency_s": 1.5,
         "parse_ok": False, "think_leak": True},
    ]
    scored = {"n": 2, "n_valid": 1, "n_match": 1, "per_cat": {"x": [2, 1, 1]},
              "failures": [], "per_case": {}}
    reg = {"name": "m", "hf_id": "org/m", "revision": "abc", "chat_template_kwargs": {}}
    args = SimpleNamespace(split="test", cases="c.jsonl", max_new_tokens=192,
                           wandb_state="off", run_name="m", condition="zero-shot",
                           adapter=None)
    s = baseline.summarize(cases, records, scored, reg, args, {"vim": "v"})
    m = s["metrics"]
    assert m["parse_failures"] == 1 and m["think_leaks"] == 1
    assert m["exec_match"] == 1 and m["exec_match_pct"] == 50.0
    # nearest-rank median of two values is the lower one
    assert m["output_tokens_mean"] == 22.0 and m["output_tokens_p50"] == 4
    assert m["cost_per_solved_task_usd"] is None      # derived at serving time (§11)
    assert s["per_category"] == [{"category": "x", "n": 2, "valid": 1, "match": 1,
                                  "valid_pct": 50.0, "match_pct": 50.0}]
    assert s["model"] == "m" and s["revision"] == "abc"
    # a row must say what it is: zero-shot and SFT (supervised fine-tuning) differ only by the adapter
    assert s["run"]["condition"] == "zero-shot" and s["run"]["adapter"] is None


def test_adapter_makes_the_row_an_sft_row():
    """--adapter is how the SFT/GRPO (Group Relative Policy Optimization) rows reach the table: same script, same
    generation path, one more argument. The name and condition follow it."""
    cases = [{"id": "a", "category": "x"}]
    records = [{"id": "a", "output_tokens": 4, "prompt_tokens": 50, "latency_s": 0.5,
                "parse_ok": True, "think_leak": False}]
    scored = {"n": 1, "n_valid": 1, "n_match": 1, "per_cat": {"x": [1, 1, 1]},
              "failures": [], "per_case": {}}
    reg = {"name": "m", "hf_id": "org/m", "revision": "abc", "chat_template_kwargs": {}}
    args = SimpleNamespace(split="test", cases="c.jsonl", max_new_tokens=192,
                           wandb_state="off", run_name="m-sft", condition="sft",
                           adapter="runs/m-sft/best")
    s = baseline.summarize(cases, records, scored, reg, args, {"vim": "v"})
    assert s["run"]["condition"] == "sft"
    assert s["run"]["adapter"] == "runs/m-sft/best"
    assert s["run"]["name"] == "m-sft"
    # the model identity is still the registry's, never the adapter's
    assert s["hf_id"] == "org/m" and s["revision"] == "abc"


def test_registry_is_the_only_source_of_model_ids():
    src = (REPO / "scripts" / "baseline.py").read_text(encoding="utf-8")
    assert "Qwen/" not in src and "hf_id: " not in src
    assert 'reg["hf_id"]' in src


def test_cli_help_needs_no_model():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "baseline.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--model" in p.stdout and "--limit" in p.stdout
    assert "--adapter" in p.stdout


def test_generation_is_a_reusable_function_not_inlined_in_main():
    """train_sft.py's val callback imports this; if it goes back inside main()
    the training-time score and the table score stop sharing a code path."""
    assert callable(baseline.generate_predictions)
    src = (REPO / "scripts" / "baseline.py").read_text(encoding="utf-8")
    body = src.split("def main()", 1)[1]
    assert "model.generate(" not in body, "generation belongs in generate_predictions"


def test_wandb_mode_falls_back_to_offline_without_a_key(monkeypatch):
    """Found 2026-09-09: after reading wandb.api.api_key, a WANDB_MODE set in
    os.environ is ignored and init() demands a login, so the mode is passed to
    init() explicitly. This pins the detection that feeds it."""
    import types
    fake = types.ModuleType("wandb")
    fake.api = SimpleNamespace(api_key=None)
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert baseline.wandb_mode() == "offline"
    fake.api.api_key = "k"
    assert baseline.wandb_mode() == "online"
    fake.api.api_key = None
    monkeypatch.setenv("WANDB_API_KEY", "k")
    assert baseline.wandb_mode() == "online"


def test_wandb_init_receives_the_mode_explicitly():
    """The env-var route is the bug; both scripts must hand mode= to init()."""
    for name in ("baseline.py", "train_sft.py"):
        src = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert 'os.environ["WANDB_MODE"]' not in src, name
        assert "mode=mode" in src, name


# ------------------------------------------------- base storage (QLoRA, 2026-09-09)

def test_nf4_is_the_qlora_recipe_and_bf16_needs_no_config():
    """A bf16 base loads with no quantization config at all; the 4-bit path is
    the QLoRA paper's recipe -- NF4, double quantization, bf16 compute."""
    assert baseline.bnb_config(False) is None
    assert baseline.NF4 == {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_use_double_quant": True,
                            "bnb_4bit_compute_dtype": "bfloat16"}
    assert baseline.base_quant_name(False) == "bf16"
    assert baseline.base_quant_name(True) == "nf4"


def test_base_marker_round_trips_beside_the_adapter(tmp_path):
    """train_sft.py records which base an adapter was trained against; the eval
    script reads it back so a QLoRA adapter is scored on the base it saw."""
    baseline.write_base_marker(tmp_path / "best", True)
    assert baseline.resolve_base_quant(tmp_path / "best") is True
    marker = json.loads((tmp_path / "best" / baseline.BASE_MARKER).read_text())
    assert marker["quantization"] == "nf4" and marker["bnb"] == baseline.NF4
    baseline.write_base_marker(tmp_path / "final", False)
    assert baseline.resolve_base_quant(tmp_path / "final") is False


def test_resolve_base_quant_falls_back_to_train_config_then_bf16(tmp_path):
    """No marker: the run's train_config.json one level up decides; nothing at
    all means bf16, the only base that existed before markers did."""
    run = tmp_path / "m-sft"
    (run / "best").mkdir(parents=True)
    assert baseline.resolve_base_quant(run / "best") is False
    (run / "train_config.json").write_text(json.dumps({"base": {"load_in_4bit": True}}))
    assert baseline.resolve_base_quant(run / "best") is True
    (run / "train_config.json").write_text(json.dumps({"lora": {"r": 16}}))
    assert baseline.resolve_base_quant(run / "best") is False


def test_summary_records_which_base_the_row_was_measured_on():
    cases = [{"id": "a", "category": "x"}]
    records = [{"id": "a", "output_tokens": 4, "prompt_tokens": 50, "latency_s": 0.5,
                "parse_ok": True, "think_leak": False}]
    scored = {"n": 1, "n_valid": 1, "n_match": 1, "per_cat": {"x": [1, 1, 1]},
              "failures": [], "per_case": {}}
    reg = {"name": "m", "hf_id": "org/m", "revision": "abc", "chat_template_kwargs": {}}
    args = SimpleNamespace(split="test", cases="c.jsonl", max_new_tokens=192,
                           wandb_state="off", run_name="m-sft", condition="sft",
                           adapter="runs/m-sft/best", base_quant="nf4")
    assert baseline.summarize(cases, records, scored, reg, args, {})["run"]["base"] == "nf4"
    del args.base_quant           # every row before 2026-09-09 was a bf16 row
    assert baseline.summarize(cases, records, scored, reg, args, {})["run"]["base"] == "bf16"


def test_cli_offers_base_dtype_with_auto_as_default():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "baseline.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--base-dtype" in p.stdout
    assert "auto" in p.stdout and "nf4" in p.stdout


def test_one_loader_for_every_row():
    """Zero-shot, SFT and GRPO rows must load the base the same way, and the
    trainer must reuse it, or a QLoRA adapter could be scored on a base it
    never saw."""
    src = (REPO / "scripts" / "baseline.py").read_text(encoding="utf-8")
    assert src.count("AutoModelForCausalLM.from_pretrained(") == 1
    assert "def load_base(" in src
    trn = (REPO / "scripts" / "train_sft.py").read_text(encoding="utf-8")
    assert "AutoModelForCausalLM" not in trn and trn.count("load_base(") == 2


def test_every_scored_run_persists_its_execution_records():
    """score() is the only place that sees what vim did with each prediction;
    if a run does not write exec_<split>.jsonl those details are printed and
    lost, and the error-analysis page has to re-run the whole split."""
    for name, split in (("baseline.py", 'exec_%s.jsonl" % args.split'),
                        ("train_sft.py", 'exec_val.jsonl')):
        src = (REPO / "scripts" / name).read_text(encoding="utf-8")
        assert "write_exec_log(" in src, "%s scores but keeps nothing" % name
        assert split in src, "%s writes the log under the wrong name" % name
