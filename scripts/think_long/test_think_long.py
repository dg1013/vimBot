"""Tests for the long-thinking extension (CPU only): `pytest scripts/think_long`."""
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


E = load("think_long_extend", HERE / "extend.py")
R = load("think_long_report", HERE / "report.py")
T = E.T


def rec(i, closed, cut, match=False, budget=4096, think=None, reasoning="r", answer=None, continuation=None, cap=None):
    """A completion record: `cut` means it ran its whole budget without a stop token."""
    n = budget if cut else budget // 2
    r = {"id": i, "closed": closed, "match": match, "valid": match, "steps": n, "new_tokens": n,
         "max_new_tokens": budget, "think_tokens": think if think is not None else n, "reasoning": reasoning,
         "answer": answer if closed else None, "seconds": 1.0}
    if continuation is not None:
        r["continuation"] = continuation
    if cap is not None:
        r["cap"] = cap
    return r


# ------------------------------------------------------------ the work

def test_cut_means_the_cap_stopped_the_completion():
    assert E.cut({"steps": 4096, "new_tokens": 4096, "max_new_tokens": 4096})
    assert not E.cut({"steps": 4096, "new_tokens": 4095, "max_new_tokens": 4096})   # a stop token at the last step
    assert not E.cut({"steps": 900, "new_tokens": 899, "max_new_tokens": 4096})
    assert E.cut(rec("x", True, True, answer="\n\nex: 2"))                            # closed, but the answer was cut


def test_each_stage_runs_what_the_previous_cap_cut_and_fresh_rows_only_when_asked():
    orders = {"think-trained": ["a", "b", "c", "e", "f"], "think-retr8": ["d", "g"]}
    byid = {"think-trained": {"a": rec("a", True, False, answer="ex: 2d"), "b": rec("b", False, True),
                              "c": rec("c", False, True)},
            "think-retr8": {"d": rec("d", False, True)}}
    stages = {cap: {arm: {} for arm in E.THINK_ARMS} for cap in E.CAPS}
    assert E.work(orders, byid, stages, 8192) == {"think-trained": ["b", "c"], "think-retr8": ["d"]}
    assert E.work(orders, byid, stages, 8192, fresh=True) == {"think-trained": ["b", "c", "e", "f"],
                                                              "think-retr8": ["d", "g"]}
    assert E.work(orders, byid, stages, 16384, fresh=True) == {"think-trained": [], "think-retr8": []}
    stages[8192]["think-trained"]["b"] = rec("b", True, False, answer="ex: 3d", continuation="x</think>ex: 3d", cap=8192)
    stages[8192]["think-trained"]["c"] = rec("c", False, True, continuation="more", cap=8192)
    stages[8192]["think-trained"]["e"] = rec("e", False, True, budget=8192, continuation="all", cap=8192)
    assert E.work(orders, byid, stages, 8192, fresh=True) == {"think-trained": ["f"], "think-retr8": ["d", "g"]}
    assert E.work(orders, byid, stages, 16384) == {"think-trained": ["c", "e"], "think-retr8": []}


def test_a_continuation_starts_from_everything_generated_so_far():
    first = rec("c", False, True, reasoning="Thinking... wait")
    mid = rec("c", False, True, continuation=" maybe 3d", budget=4096, cap=8192)
    byid = {"think-trained": {"c": first}}
    stages = {8192: {"think-trained": {"c": mid}}, 16384: {"think-trained": {}}}
    assert E.chain("c", byid, stages, "think-trained", 8192) == [first]
    records = E.chain("c", byid, stages, "think-trained", 16384)
    assert E.text_so_far(records) == "Thinking... wait maybe 3d"
    assert E.tokens_so_far(records) == 8192
    closed_cut = rec("x", True, True, reasoning="ok", answer="\n\nex: 2")
    assert E.text_so_far([closed_cut]) == "ok</think>\n\nex: 2"
    # A row the thinking run never reached starts from nothing, and its 8k text carries on at 16k.
    assert E.chain("n", byid, stages, "think-trained", 8192) == [] and E.text_so_far([]) == ""
    fresh = rec("n", False, True, budget=8192, continuation="from the start", cap=8192)
    stages[8192]["think-trained"]["n"] = fresh
    assert E.chain("n", byid, stages, "think-trained", 16384) == [fresh]
    assert E.text_so_far([fresh]) == "from the start" and E.tokens_so_far([fresh]) == 8192
    done = rec("a", True, False, answer="ex: 1")                                  # it stopped at 4k
    byid["think-trained"]["a"] = done
    assert E.chain("a", byid, stages, "think-trained", 16384) == [done]


def test_the_prompt_is_the_thinking_runs():
    class Tok:
        def apply_chat_template(self, messages, add_generation_prompt, tokenize, **kwargs):
            assert add_generation_prompt and not tokenize
            return json.dumps([messages, kwargs])
    case = {"id": "v1", "instruction": "delete line 2", "buffer_before": "a\nb\n", "cursor": [1, 1], "mode": "ex"}
    train = {"t1": {"id": "t1", "instruction": "delete line 3", "buffer_before": "a\nb\nc\n", "cursor": [1, 1],
                    "mode": "ex", "gold_command": "3d"}}
    plan = {"v1": {"examples": ["t1"]}}
    info = {"kwargs": {T.THINK_MODEL: {"enable_thinking": "from the registry"}, T.PLAIN_MODEL: {}}}
    retr = json.loads(E.prompt_text(Tok(), case, "think-retr8", train, plan, info))
    plain = json.loads(E.prompt_text(Tok(), case, "think-trained", train, plan, info))
    assert retr == json.loads(json.dumps([E.messages_for_case(case, [train["t1"]]), {"enable_thinking": "from the registry"}]))
    assert plain[0] == json.loads(json.dumps(E.messages_for_case(case, []))) and plain[1] == retr[1]


def test_seeds_are_fixed_and_apart_from_the_earlier_runs():
    ids = ["hc-%d" % i for i in range(300)]
    seeds = {E.seed_for(a, i, c) for a in E.THINK_ARMS for i in ids for c in E.CAPS}
    assert E.seed_for("think-trained", "hc-1", 8192) == E.seed_for("think-trained", "hc-1", 8192)
    assert len({E.seed_for(a, "hc-1", c) for a in E.THINK_ARMS for c in E.CAPS}) == 4   # per arm and stage
    earlier = range(T.SEED, T.SEED + 100000 * (len(T.ARMS) + 1))    # the thinking run's and the labeling pass's
    assert not any(s in earlier for s in seeds) and all(s < 2 ** 32 for s in seeds)


def test_the_arms_are_the_thinking_runs_thinking_arms():
    for arm in E.THINK_ARMS:
        assert T.ARMS[arm][0] == T.THINK_MODEL
    assert T.ARMS["think-trained"][1] is True and T.ARMS["think-retr8"][1] is False      # adapter on / off
    assert E.CAPS == (8192, 16384) and E.FIRST_CAP == 4096
    assert R.CONTROL == {"think-trained": "nothink-trained", "think-retr8": "nothink-retr8"}


def test_real_rows_are_the_start_of_each_arms_order():
    if not (E.FIRST_RUN / "rows.json").exists():
        pytest.skip("the thinking run is not on disk")
    assert all(E.is_prefix().values())


# ------------------------------------------------------------ the report

def test_report_uses_each_caps_prefix_and_pairs_with_the_controls(tmp_path):
    val_path = REPO / "data" / "splits" / "val.jsonl"
    if not val_path.exists():
        pytest.skip("the local data is not on disk")
    val = [json.loads(line) for line in val_path.open(encoding="utf-8")][:8]
    ids = [c["id"] for c in val]
    first, ext = tmp_path / "first", tmp_path / "ext"
    first.mkdir()
    ext.mkdir()
    (first / "rows.json").write_text(json.dumps({"rows": [{"id": i, "in_fail": True, "in_hard": False} for i in ids]}),
                                     encoding="utf-8")

    def write(path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    write(first / "rows_nothink-trained.jsonl", [{"id": i, "match": k == 0} for k, i in enumerate(ids)])
    write(first / "rows_nothink-retr8.jsonl", [{"id": i, "match": False} for i in ids])
    order = T.order_for("think-trained", ids)
    # 4k: rows 0-1 finished (row 0 right), 2-5 cut. 8k: rows 2 and 3 continued (2 finishes right, 3 still cut).
    write(first / "rows_think-trained.jsonl",
          [rec(order[0], True, False, match=True, answer="ex: 1"), rec(order[1], True, False, answer="ex: 1")]
          + [rec(i, False, True) for i in order[2:6]])
    order_b = T.order_for("think-retr8", ids)
    write(first / "rows_think-retr8.jsonl", [rec(i, False, True) for i in order_b[:2]])
    # think-retr8 at 8k: both cut rows continued (one right), then a fresh row from the start (right).
    write(ext / "rows_8192_think-retr8.jsonl",
          [rec(order_b[0], True, False, match=True, answer="ex: 4", continuation="c", cap=8192),
           rec(order_b[1], False, True, continuation="c", cap=8192),
           dict(rec(order_b[2], True, False, match=True, budget=8192, answer="ex: 5", continuation="all", cap=8192),
                fresh=True)])
    write(ext / "rows_8192_think-trained.jsonl",
          [rec(order[2], True, False, match=True, answer="ex: 2", continuation="c", cap=8192),
           rec(order[3], False, True, continuation="c", cap=8192)])
    write(ext / "rows_16384_think-trained.jsonl",
          [rec(order[3], True, False, match=True, budget=8192, answer="ex: 3", continuation="c", cap=16384)])
    rep = R.build(first_run=first, run=ext)
    arm = rep["arms"]["think-trained"]
    assert (arm["4096"]["rows"], arm["8192"]["rows"], arm["16384"]["rows"]) == (6, 4, 4)
    assert (arm["4096"]["solved"], arm["8192"]["solved"], arm["16384"]["solved"]) == (1, 2, 3)
    b8 = rep["arms"]["think-retr8"]["8192"]
    assert (b8["rows"], b8["fresh"], b8["solved"]) == (3, 1, 2)            # the fresh row counts at 8k
    assert rep["arms"]["think-retr8"]["16384"]["rows"] == 1               # finished at 8k; the next is still cut
    assert rep["readings"]["TB room 8k"]["n"] == 2                        # only rows the 4k run had
    assert rep["primary_cap"] == 8192
    think16 = {order[0]: 1, order[1]: 0, order[2]: 1, order[3]: 1}
    control = {i: int(i == ids[0]) for i in think16}
    ta = rep["readings"]["TA16k"]
    assert (ta["n"], ta["gained"], ta["lost"]) == (4, sum(think16[i] > control[i] for i in think16),
                                                   sum(think16[i] < control[i] for i in think16))
    room = rep["readings"]["TA room 16k"]
    assert (room["n"], room["gained"], room["lost"]) == (4, 2, 0)
    assert rep["extra_room"]["think-trained"]["cut_at_4k"] == 4               # all four; two reached so far
    assert rep["extra_room"]["think-trained"]["solved_by"] == {"8192": 1, "16384": 2}
    assert rep["extra_room"]["think-trained"]["reached_by"] == {"8192": 2, "16384": 2}
    assert "TA16k" in R.render(rep)


# ------------------------------------------------------------ the GPU stage, end to end on a tiny model

TINY_RUN = r"""
import sys
sys.modules["fla"] = None                # the torch path for the linear-attention layer (fla needs a GPU)
import importlib.util, json, pathlib, torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
spec = importlib.util.spec_from_file_location("think_long_extend", sys.argv[1])
E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
T = E.T
from baseline import load_registry
think, plain = load_registry(T.THINK_MODEL), load_registry(T.PLAIN_MODEL)
try:
    tok = AutoTokenizer.from_pretrained(think["hf_id"], revision=think["revision"], local_files_only=True)
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
info = {"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "stop_ids": [tok.convert_tokens_to_ids("<|im_end|>")], "pad_id": tok.pad_token_id,
        "kwargs": {T.THINK_MODEL: think["chat_template_kwargs"], T.PLAIN_MODEL: plain["chat_template_kwargs"]}}
loads = []
T.load_model = lambda: loads.append(1) or (tok, model, info)
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: 0
tmp = pathlib.Path(sys.argv[2])
def case(i, line=2):
    return {"id": i, "category": "line-ops", "tier": "T1", "instruction": "delete line %d" % line, "mode": "ex",
            "buffer_before": "a\nb\nc\n", "buffer_after": "a\nc\n", "gold_command": "2d", "cursor": [1, 1]}
def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
val = [case("v%d" % k) for k in range(3)]
write(tmp / "val.jsonl", val)
write(tmp / "train.jsonl", [case("t0", 3), case("t1", 1)])
write(tmp / "plan.jsonl", [{"id": c["id"], "examples": ["t0", "t1"]} for c in val])
T.VAL, T.TRAIN, T.PLAN = tmp / "val.jsonl", tmp / "train.jsonl", tmp / "plan.jsonl"
first = tmp / "first"
first.mkdir()
ids = [c["id"] for c in val]
(first / "rows.json").write_text(json.dumps({"rows": [{"id": i, "in_fail": True, "in_hard": False} for i in ids]}))
E.FIRST_CAP, E.CAPS = 16, (24, 40)       # tiny caps: an 8-token stage, then a 16-token one
for arm in E.THINK_ARMS:
    o = T.order_for(arm, ids)
    write(first / ("rows_%s.jsonl" % arm), [
        {"id": o[0], "closed": True, "answer": "\n\nex: 2d", "reasoning": "fine", "match": True, "steps": 9,
         "new_tokens": 8, "max_new_tokens": 16, "think_tokens": 5},
        {"id": o[1], "closed": False, "answer": None, "reasoning": "still thinking", "match": False, "steps": 16,
         "new_tokens": 16, "max_new_tokens": 16, "think_tokens": 16}])
run = tmp / "ext"
phase_a = ["run", "--first-run", str(first), "--run", str(run), "--deadline", "9999999999", "--stages", "24", "--fresh"]
assert E.main(phase_a) == 0
assert not list(run.glob("rows_40_*.jsonl"))                       # phase a runs stage 8k only
for arm in E.THINK_ARMS:
    o = T.order_for(arm, ids)
    a = [json.loads(l) for l in open(run / ("rows_24_%s.jsonl" % arm))]
    assert [r["id"] for r in a] == [o[1], o[2]], a                   # the cut row, then the unreached one
    assert (a[0]["tokens_before"], a[0]["max_new_tokens"], a[0]["fresh"]) == (16, 8, False)
    assert (a[1]["tokens_before"], a[1]["max_new_tokens"], a[1]["fresh"]) == (0, 24, True)
    assert a[0]["prefix_tokens"] > 0 and a[1]["prefix_tokens"] == 0 and a[0]["new_tokens"] <= 8
    assert a[0]["seed"] == E.seed_for(arm, o[1], 24)
phase_b = ["run", "--first-run", str(first), "--run", str(run), "--deadline", "9999999999"]
assert E.main(phase_b) == 0
for arm in E.THINK_ARMS:
    a = [json.loads(l) for l in open(run / ("rows_24_%s.jsonl" % arm))]
    path = run / ("rows_40_%s.jsonl" % arm)
    b = [json.loads(l) for l in open(path)] if path.exists() else []
    assert [r["id"] for r in b] == [r["id"] for r in a if E.cut(r)]
    for r in b:
        assert (r["tokens_before"], r["max_new_tokens"]) == (24, 16)
sizes = {p.name: p.read_text().count("\n") for p in run.glob("rows_*.jsonl")}
assert E.main(phase_b) == 0 and len(loads) == 2                    # a rerun resumes: nothing left, no load
assert sizes == {p.name: p.read_text().count("\n") for p in run.glob("rows_*.jsonl")}
print("OK")
"""


def test_the_gpu_stage_end_to_end_on_a_tiny_qwen35(tmp_path):
    """`extend.py run` on a tiny Qwen3.5 under LoRA with the real tokenizer and template, in a
    subprocess that blocks fla: both stages, the budgets, the seeds, the sandbox judge, and resume."""
    pytest.importorskip("peft")
    env = dict(__import__("os").environ, CUDA_VISIBLE_DEVICES="",
               PYTHONPATH="%s:%s" % (REPO / "src", REPO / "scripts"))
    out = subprocess.run([sys.executable, "-c", TINY_RUN, str(HERE / "extend.py"), str(tmp_path)],
                         capture_output=True, text=True, env=env, timeout=600)
    if out.stdout.strip().endswith("SKIP"):
        pytest.skip("the tokenizer is not in the local cache")
    assert out.returncode == 0 and out.stdout.strip().endswith("OK"), (out.stdout[-2000:], out.stderr[-3000:])


# ------------------------------------------------------------ the driver and hygiene

def test_driver_parses_and_runs_the_preregistered_stages():
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert 'CUTOFF=${CUTOFF:-"2026-11-02 07:30"}' in src and 'STEPS=${STEPS:-"plan run report"}' in src
    assert "RUN=runs/think_long_2026-11-01" in src
    assert 'extend.py run --run "$RUN" --deadline "$deadline" $RUN_ARGS' in src
    assert 'PHASE=${PHASE:-b}' in src and '"$RUN/deadline_$PHASE"' in src


def test_no_model_ids_and_no_thinking_flag_in_this_folder():
    """The thinking switch lives in the registry (`qwen3.5-27b-think`), never in code."""
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh") and f.name != "test_think_long.py":
            src = f.read_text(encoding="utf-8")
            assert "Qwen/" not in src and "enable_thinking" not in src, f.name
