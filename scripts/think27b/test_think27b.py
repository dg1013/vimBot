"""Tests for the thinking run (CPU only): `pytest scripts/think27b`."""
import importlib.util
import json
import pathlib
import subprocess
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


T = load("think27b_think27b", HERE / "think27b.py")


# ------------------------------------------------------------ completions

def test_split_reasoning_takes_the_text_after_the_first_close_tag():
    assert T.split_reasoning("hmm, line 2.\n</think>\n\nex: 2d", True) == ("hmm, line 2.\n", "\n\nex: 2d", True)
    assert T.split_reasoning("still thinking about it", True) == ("still thinking about it", None, False)
    assert T.split_reasoning("ex: 2d", False) == ("", "ex: 2d", True)


def test_trim_stops_at_the_first_stop_token_and_drops_pads():
    assert T.trim([5, 6, 99, 7], {99, 98}, 0) == [5, 6]
    assert T.trim([0, 0, 5, 98], {99, 98}, 0) == [5]
    assert T.trim([5, 6], {99}, 0) == [5, 6]
    assert T.think_tokens([4, 4, 7, 9], 7) == 2 and T.think_tokens([4, 4], 7) == 2


def test_each_arm_has_its_own_fixed_order():
    ids = ["r%d" % i for i in range(20)]
    a, b = T.order_for("think-trained", ids), T.order_for("think-retr8", ids)
    assert sorted(a) == sorted(ids) and a == T.order_for("think-trained", ids) and a != b


def test_controls_run_first_and_the_thinking_arms_alternate():
    """Addendum 1: at one row per batch, a deadline must cut both thinking arms, not starve one."""
    todo = {"nothink-trained": ["a", "b"], "nothink-retr8": ["c"], "think-trained": ["d", "e", "f"],
            "think-retr8": ["g"]}
    thinks = {"nothink-trained": False, "nothink-retr8": False, "think-trained": True, "think-retr8": True}
    order = [(arm, ids) for arm, _, ids in T.schedule(todo, 1, thinks)]
    assert order == [("nothink-trained", ["a"]), ("nothink-trained", ["b"]), ("nothink-retr8", ["c"]),
                     ("think-trained", ["d"]), ("think-retr8", ["g"]), ("think-trained", ["e"]),
                     ("think-trained", ["f"])]
    offsets = [(arm, b) for arm, b, _ in T.schedule(todo, 2, thinks) if arm == "think-trained"]
    assert offsets == [("think-trained", 0), ("think-trained", 2)]   # each arm counts its own rows
    resumed = dict(todo, **{"nothink-trained": [], "nothink-retr8": [], "think-retr8": []})
    assert [ids for _, _, ids in T.schedule(resumed, 8, thinks)] == [["d", "e", "f"]]
    assert list(T.ARMS)[:2] == ["nothink-trained", "nothink-retr8"]  # the controls are the first two


def test_judge_parses_the_answer_and_runs_it_in_the_sandbox():
    case = {"id": "x", "category": "line-ops", "instruction": "delete line 2", "mode": "ex",
            "buffer_before": "a\nb\nc\n", "buffer_after": "a\nc\n", "gold_command": "2d", "cursor": [1, 1]}
    other = dict(case, id="y")
    none = dict(case, id="z")
    parsed, per = T.judge([case, other, none], ["\n\nex: 2d", "ex: 3d", None], 5.0)
    assert parsed["x"] == ("ex", "2d") and per["x"]["match"]
    assert parsed["y"] == ("ex", "3d") and per["y"]["valid"] and not per["y"]["match"]
    assert parsed["z"] is None and not per["z"]["match"] and not per["z"]["valid"]
    bad, _ = T.judge([case], ["<think>again</think>ex: 2d"], 5.0)
    assert bad["x"] is None                                  # §6: any think tag is a parse failure


TINY = r"""
import sys
sys.modules["fla"] = None                # the torch path for the linear-attention layer (fla needs a GPU)
import importlib.util, pathlib, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
spec = importlib.util.spec_from_file_location("t27", sys.argv[1])
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
from baseline import load_registry
reg = load_registry(T.THINK_MODEL)
try:
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"], local_files_only=True)
except Exception:
    print("SKIP"); sys.exit(0)
tok.padding_side = "left"
cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                        layer_types=["linear_attention", "full_attention"], full_attention_interval=2,
                        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16,
                        linear_value_head_dim=16, tie_word_embeddings=False, pad_token_id=tok.pad_token_id)
torch.manual_seed(0)
model = get_peft_model(AutoModelForCausalLM.from_config(cfg, dtype=torch.float32),
                       LoraConfig(r=4, target_modules="all-linear")).eval()
case = {"instruction": "delete line 2", "buffer_before": "a\nb\nc\n", "cursor": [1, 1]}
longer = dict(case, buffer_before="alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\n")
texts = [tok.apply_chat_template(T.messages_for_case(c), add_generation_prompt=True, tokenize=False,
                                 **reg["chat_template_kwargs"]) for c in (case, longer)]
assert texts[0].endswith("<think>\n")
assert len(tok(texts[0])["input_ids"]) < len(tok(texts[1])["input_ids"])      # the batch is padded
info = {"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "stop_ids": [tok.convert_tokens_to_ids("<|im_end|>")], "pad_id": tok.pad_token_id}
for adapter_on in (True, False):
    outs, lens = T.generate(model, tok, texts, adapter_on, 6, info, seed=1)
    assert len(outs) == 2 and all(n == 6 for n in lens) and all(len(o) <= 6 for o in outs)
g = max(T.padding_guard(model, tok, texts, on) for on in (True, False))
assert g < 1e-3, g
print("OK %.2e" % g)
"""


def test_generation_and_the_padding_guard_on_a_tiny_qwen35_under_lora():
    """A tiny Qwen3.5 with a linear-attention and a full-attention layer, under LoRA, with the real
    tokenizer, in a subprocess that blocks fla so the recurrent layer takes transformers' torch path:
    left-padded generation runs with the adapter on and off, and a padded batch's next-token
    distributions equal one-row passes (transformers zeroes padded inputs to the recurrent layer)."""
    pytest.importorskip("peft")
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH="%s:%s" % (REPO / "src", REPO / "scripts"))
    out = subprocess.run([sys.executable, "-c", TINY, str(HERE / "think27b.py")], capture_output=True,
                         text=True, env=env, timeout=600)
    if out.stdout.strip() == "SKIP":
        pytest.skip("the tokenizer is not in the local cache")
    assert out.returncode == 0 and out.stdout.startswith("OK"), out.stderr[-2000:]


# ------------------------------------------------------------ rows and arms

def test_rows_are_the_misses_and_the_never_solved():
    if not (T.MISSES.exists() and T.SERVED_ROWS.exists() and T.VAL.exists()):
        pytest.skip("the local runs are not on disk")
    rows = T.select_rows()
    assert len(rows) == 213
    assert sum(r["in_fail"] for r in rows) == 208 and sum(r["in_hard"] for r in rows) == 70
    assert sum(r["in_fail"] and r["in_hard"] for r in rows) == 65


def test_arms_take_their_templates_from_the_registry():
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    think, plain = load_registry(T.THINK_MODEL), load_registry(T.PLAIN_MODEL)
    assert (think["hf_id"], think["revision"]) == (plain["hf_id"], plain["revision"])
    assert think["chat_template_kwargs"]["enable_thinking"] is True
    assert plain["chat_template_kwargs"]["enable_thinking"] is False
    assert list(T.ARMS)[:2] == ["nothink-trained", "nothink-retr8"]          # controls first
    for name, (reg, adapter_on, retr) in T.ARMS.items():
        assert reg == (T.THINK_MODEL if name.startswith("think") else T.PLAIN_MODEL)
        assert adapter_on == name.endswith("trained") and retr == name.endswith("retr8")


# ------------------------------------------------------------ the report

def test_report_pairs_each_thinking_arm_with_its_control(tmp_path):
    val_path = REPO / "data" / "splits" / "val.jsonl"
    if not val_path.exists():
        pytest.skip("the local data is not on disk")
    R = load("think27b_report", HERE / "report.py")
    ids = [json.loads(line)["id"] for line in val_path.open(encoding="utf-8")][:6]
    rows = [{"id": i, "in_fail": k < 4, "in_hard": k >= 3} for k, i in enumerate(ids)]
    (tmp_path / "rows.json").write_text(json.dumps({"rows": rows, "n": 6, "fail": 4, "hard": 3}), encoding="utf-8")

    def write(arm, solved, thinking):
        with (tmp_path / ("rows_%s.jsonl" % arm)).open("w", encoding="utf-8") as fh:
            for k, i in enumerate(ids):
                fh.write(json.dumps({"id": i, "in_fail": k < 4, "in_hard": k >= 3, "match": k in solved,
                                     "valid": True, "closed": k != 5 or not thinking, "parse_ok": True,
                                     "think_tokens": 100 * (k + 1) if thinking else 0,
                                     "batch_seconds": 80.0, "batch_size": 8}) + "\n")
    write("nothink-trained", {0}, False)
    write("think-trained", {0, 1, 2}, True)
    rep = R.build(tmp_path)
    ta = rep["readings"]["TA"]
    assert (ta["n"], ta["gained"], ta["lost"]) == (6, 2, 0) and rep["verdicts"]["TA"] == "tied"
    assert rep["arms"]["think-trained"]["counts"]["misses"] == {"n": 4, "solved": 3}
    assert rep["arms"]["think-trained"]["thinking"]["unclosed"] == 1
    assert "TB" not in rep["readings"]                       # its arms have not run
    assert "think-trained" in R.render(rep)


# ------------------------------------------------------------ the driver

def test_driver_parses_and_runs_the_preregistered_stages():
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert "HOURS=${HOURS:-10}" in src and 'STEPS=${STEPS:-"rows run report"}' in src
    assert "think27b.py run --run \"$RUN\" --deadline \"$deadline\"" in src
    assert "RUN=runs/think27b_2026-11-01" in src


def test_no_model_ids_and_no_thinking_flag_in_this_folder():
    """The thinking switch lives in the registry (`qwen3.5-27b-think`), never in code."""
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh") and f.name != "test_think27b.py":
            src = f.read_text(encoding="utf-8")
            assert "Qwen/" not in src and "enable_thinking=True" not in src, f.name
            assert "enable_thinking\": True" not in src and "enable_thinking': True" not in src, f.name
