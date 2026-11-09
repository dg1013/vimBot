"""Tests for the loop breaker (CPU only): `pytest scripts/think_loop`.

The drivers' runs use the real tokenizer and template with a scripted generator in place of the
model; they and the calibration skip when the tokenizer or the local runs are not on disk."""
import contextlib
import importlib.util
import json
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LB = load("think_loop_loop", HERE / "loop.py")
LINE = "* I need `/^###$/`? No.\n"              # the shape of a recorded loop


# ------------------------------------------------------------ the detector

def test_a_block_said_reps_times_in_a_row_is_a_loop():
    intro = "The user wants the headings gone.\nFirst, the regex.\n"
    assert LB.find_loop(intro + LINE * 7) is None
    loop = LB.find_loop(intro + LINE * 8)
    assert loop["period"] == 1 and loop["reps"] == 8 and loop["block"] == [LINE.strip()]
    assert (loop["start"], loop["keep"]) == (len(intro), len(intro) + len(LINE))
    assert LB.find_loop(intro + LINE * 8 + "Okay, something new.\n") is None      # only the end counts
    assert LB.find_loop(intro + LINE * 8 + "* I need `/^##") is not None           # a line still being written


def test_a_longer_block_with_blank_lines_and_an_earlier_start():
    block = ["* Wait, I'll check if I should use `ex: %s/old\\/path/new\\/path/g`.\n",   # a longer shape
             "* Wait, I'll check if `new\\/path` is correct.\n", "* Yes.\n", "* Okay.\n"]
    intro = "Reasoning.\n* Yes.\n* Okay.\n"
    text = intro + "".join(("\n" if k % 2 else "") + "".join(block) for k in range(12))
    loop = LB.find_loop(text)
    assert loop["period"] == 4 and loop["reps"] == 12.5          # it began at intro's "* Yes."
    assert loop["start"] == intro.index("* Yes.") and loop["block"][0] == "* Yes."
    assert text[loop["start"]:loop["keep"]] == "* Yes.\n* Okay.\n" + "".join(block[:2])


def test_twice_in_a_row_is_not_a_loop_and_quoted_short_lines_never_are():
    assert LB.find_loop(("* `2,$s/$/,/`\n* Check line 3.\n" * 2 + "So the answer.\n") * 6) is None
    assert LB.find_loop("    }\n" * 200) is None                     # a quoted buffer of closing braces
    assert LB.find_loop("* Yes.\n" * 31) is None                     # 5 characters a line: a block needs 4
    assert LB.find_loop("* Yes.\n" * 32)["block"] == ["* Yes."] * 4  # lines (16), said 8 times: 32 lines


# ------------------------------------------------------------ during generation

class CharTok:
    """One token per character, and one each for the reasoning tags."""
    vocab = ["<think>", "</think>", "\n"] + [chr(c) for c in range(32, 127)]
    ids = {t: i for i, t in enumerate(vocab)}

    def encode(self, s):
        out, i = [], 0
        while i < len(s):
            tag = next((t for t in ("</think>", "<think>") if s.startswith(t, i)), s[i])
            out.append(self.ids[tag])
            i += len(tag)
        return out

    def decode(self, ids, **kwargs):
        return "".join(self.vocab[i] for i in ids)

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": self.encode(s)}


def test_the_stopping_check_fires_on_the_newline_that_completes_the_loop():
    torch = pytest.importorskip("torch")
    tok = CharTok()
    stop = LB.stopping(tok, LB.DEFAULT)
    looping = tok.encode("user: fix it\n<think>\nThe regex first.\n" + LINE * 9)
    closed = tok.encode("user: fix it\n<think>\nEasy.</think>\n\n" + LINE * 12)[:len(looping)]
    assert len(closed) == len(looping)
    fired = [k for k in range(1, len(looping) + 1)
             if stop(torch.tensor([looping[:k], closed[:k]]), None).tolist() != [False, False]]
    end8 = len(tok.encode("user: fix it\n<think>\nThe regex first.\n" + LINE * 8))
    assert fired[0] == end8                                        # row 0 only, at the 8th copy's newline
    assert all(stop(torch.tensor([looping[:k], closed[:k]]), None).tolist()[1] is False for k in fired)


def test_break_loop_keeps_one_copy_closes_and_answers():
    tok, seen = CharTok(), []

    def generate(model, tok_, texts, adapter_on, max_new, info, seed):
        seen.append((texts, adapter_on, max_new, seed))
        return [tok.encode("ex: %d,$d")], [9]

    prompt, intro = "user: fix it\n<think>\n", "The regex first.\n"
    answer, loop = LB.break_loop(generate, None, tok, prompt, intro + LINE * 8, False, {}, 7, LB.DEFAULT)
    assert answer == "ex: %d,$d" and seen == [([prompt + intro + LINE.rstrip("\n") + LB.CLOSE], False, 192, 7)]
    assert (loop["period"], loop["reps"], loop["start_tokens"]) == (1, 8, len(intro))
    assert loop["kept_tokens"] == len(intro + LINE) - 1 and loop["dropped_tokens"] == 7 * len(LINE)
    assert LB.break_loop(generate, None, tok, prompt, intro + LINE * 3, False, {}, 7, LB.DEFAULT) == (None, None)
    assert len(seen) == 1                                          # no loop, no second call


# ------------------------------------------------------------ per run directory

def test_a_run_directory_keeps_the_setting_it_started_with(tmp_path):
    new = tmp_path / "new"
    assert LB.resolve(new) == LB.DEFAULT                           # on for a new directory, and recorded
    assert json.loads((new / LB.SETTINGS).read_text())["settings"] == LB.DEFAULT
    assert LB.resolve(new) == LB.DEFAULT and LB.resolve(new, True) == LB.DEFAULT
    with pytest.raises(SystemExit):
        LB.resolve(new, False)
    off = tmp_path / "off"
    assert LB.resolve(off, False) is None and LB.resolve(off) is None
    with pytest.raises(SystemExit):
        LB.resolve(off, True)
    old = tmp_path / "old"                                         # rows from before the breaker
    old.mkdir()
    (old / "rows_8192_think-retr8.jsonl").write_text("")
    assert LB.resolve(old) is None and not (old / LB.SETTINGS).exists()
    with pytest.raises(SystemExit):
        LB.resolve(old, True)


def test_the_finished_run_stays_without_the_breaker():
    run = REPO / "runs" / "think_long_2026-11-01"
    if not run.exists():
        pytest.skip("the local runs are not on disk")
    assert LB.resolve(run) is None and not (run / LB.SETTINGS).exists()


# ------------------------------------------------------------ the drivers

def tokenizer_or_skip():
    from transformers import AutoTokenizer
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    reg = load_registry("qwen3.5-27b-think")
    try:
        return AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"], local_files_only=True)
    except Exception:
        pytest.skip("the tokenizer is not in the local cache")


def test_the_close_is_the_chat_templates_and_the_tag_the_runs_split_on():
    tok = tokenizer_or_skip()
    T = load("think27b_think27b", REPO / "scripts" / "think27b" / "think27b.py")
    from baseline import load_registry
    kwargs = load_registry(T.THINK_MODEL)["chat_template_kwargs"]
    text = tok.apply_chat_template([{"role": "user", "content": "x"},
                                    {"role": "assistant", "content": "<think>\nR\n</think>\n\nex: 2d"}],
                                   tokenize=False, **kwargs)
    assert ("<think>\nR" + LB.CLOSE + "ex: 2d") in text
    assert LB.THINK_CLOSE == T.THINK_END and LB.THINK_OPEN in tok.apply_chat_template(
        [{"role": "user", "content": "x"}], add_generation_prompt=True, tokenize=False, **kwargs)


def test_drivers_take_the_flag():
    for name, path, base in (("think_long_extend", "think_long/extend.py", ["run", "--deadline", "1"]),
                             ("think_label_label", "think_label/label.py", ["run", "--deadline", "1"])):
        D = load(name, REPO / "scripts" / path)
        ap = D.build_parser()
        assert ap.parse_args(base).loop_break is None
        assert ap.parse_args(base + ["--loop-break"]).loop_break is True
        assert ap.parse_args(base + ["--no-loop-break"]).loop_break is False


LOOP_SCRIPT = "The user wants line 2 gone.\n" + "* Maybe `2d` deletes the wrong line? No.\n" * 40
CLOSED_SCRIPT = "Line 3 goes.\n</think>\n\nex: 3d"


def scripted(tok, T, LBm, calls):
    """A stand-in for think27b.generate: the loop script, the closed script, or after a forced close
    the answer, one token at a time, with the stopping check after each token as model.generate has."""
    import torch
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")

    def generate(model, tok_, texts, adapter_on, max_new, info, seed, stopping_criteria=None):
        outs, lens = [], []
        for text in texts:
            calls.append(text)
            if text.endswith(LBm.CLOSE):
                script = tok("ex: 2d", add_special_tokens=False)["input_ids"] + [stop_id]
            elif "delete line 2" in text.rsplit("<|im_start|>user", 1)[-1]:
                script = tok(LOOP_SCRIPT, add_special_tokens=False)["input_ids"]
            else:
                script = tok(CLOSED_SCRIPT, add_special_tokens=False)["input_ids"] + [stop_id]
            prompt, new = tok(text)["input_ids"], []
            for x in script[:max_new]:
                new.append(x)
                if x == stop_id:
                    break
                if stopping_criteria is not None and bool(stopping_criteria(torch.tensor([prompt + new]), None)[0]):
                    break
            outs.append(T.trim(new, {stop_id}, info["pad_id"]))
            lens.append(len(new))
        return outs, lens
    return generate


def case(i, line):
    return {"id": i, "category": "line-ops", "tier": "T1", "instruction": "delete line %d" % line, "mode": "ex",
            "buffer_before": "a\nb\nc\n", "buffer_after": {2: "a\nc\n", 3: "a\nb\n"}[line],
            "gold_command": "%dd" % line, "cursor": [1, 1]}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def info_for(tok, T):
    from baseline import load_registry
    return {"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
            "stop_ids": [tok.convert_tokens_to_ids("<|im_end|>")], "pad_id": tok.pad_token_id,
            "kwargs": {n: load_registry(n)["chat_template_kwargs"] for n in (T.THINK_MODEL, T.PLAIN_MODEL)}}


def test_think_long_breaks_a_loop_and_its_finished_run_would_not(tmp_path, monkeypatch):
    tok = tokenizer_or_skip()
    import torch
    E = load("think_long_extend_lb", REPO / "scripts" / "think_long" / "extend.py")
    T = E.T
    val = [case("v0", 2)]
    write(tmp_path / "val.jsonl", val)
    write(tmp_path / "train.jsonl", [case("t0", 3)])
    write(tmp_path / "plan.jsonl", [{"id": "v0", "examples": ["t0"]}])
    monkeypatch.setattr(T, "VAL", tmp_path / "val.jsonl")
    monkeypatch.setattr(T, "TRAIN", tmp_path / "train.jsonl")
    monkeypatch.setattr(T, "PLAN", tmp_path / "plan.jsonl")
    first = tmp_path / "first"
    first.mkdir()
    (first / "rows.json").write_text(json.dumps({"rows": [{"id": "v0", "in_fail": True, "in_hard": False}]}))
    monkeypatch.setattr(E, "FIRST_CAP", 16)
    monkeypatch.setattr(E, "CAPS", (300, 600))
    calls = []
    monkeypatch.setattr(T, "load_model", lambda: (tok, None, info_for(tok, T)))
    monkeypatch.setattr(T, "generate", scripted(tok, T, E.LB, calls))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    copy = "* Maybe `2d` deletes the wrong line? No.\n"
    for name, extra in (("on", []), ("off", ["--no-loop-break"])):
        run = tmp_path / name
        assert E.main(["run", "--first-run", str(first), "--run", str(run), "--deadline", "9999999999",
                       "--stages", "300", "--fresh"] + extra) == 0
        for arm in E.THINK_ARMS:
            r = json.loads((run / ("rows_300_%s.jsonl" % arm)).read_text())
            if name == "on":
                assert r["forced_close"] and r["closed"] and not r["hit_cap"] and r["match"], r
                assert r["answer"] == "ex: 2d" and r["continuation"].count(copy) == 8 and r["new_tokens"] < 300
                assert (r["loop"]["period"], r["loop"]["reps"]) == (1, 8) and r["loop_break"] == LB.DEFAULT
            else:
                assert not r["forced_close"] and r["loop"] is None and r["loop_break"] is None
                assert not r["closed"] and r["hit_cap"] and r["answer"] is None and r["new_tokens"] == 300
    forced = [c for c in calls if c.endswith(LB.CLOSE)]
    assert len(forced) == 2 and all(c.endswith("gone.\n" + copy.rstrip("\n") + LB.CLOSE) for c in forced)


def test_think_label_breaks_a_loop_in_a_batch(tmp_path, monkeypatch):
    tok = tokenizer_or_skip()
    import torch
    L = load("think_label_label_lb", REPO / "scripts" / "think_label" / "label.py")
    T = L.T
    train = [case("t0", 2), case("t1", 3)]
    write(tmp_path / "train.jsonl", train)
    monkeypatch.setattr(L, "TRAIN", tmp_path / "train.jsonl")
    run = tmp_path / "label"
    run.mkdir()
    (run / "rows.json").write_text(json.dumps({"rows": [{"id": c["id"], "in_cliff": True, "in_never": False}
                                                        for c in train]}))
    write(run / "plan_retr8.jsonl", [{"id": c["id"], "examples": []} for c in train])
    calls = []
    monkeypatch.setattr(L, "load_model", lambda: (tok, None, info_for(tok, T)))
    monkeypatch.setattr(T, "generate", scripted(tok, T, L.LB, calls))
    monkeypatch.setattr(T, "padding_guard", lambda *a, **k: 0.0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    assert L.main(["run", "--run", str(run), "--deadline", "9999999999", "--arms", "think-retr8",
                   "--max-new-tokens", "300", "--batch-size", "2"]) == 0
    rows = {r["id"]: r for r in map(json.loads, (run / "rows_think-retr8.jsonl").read_text().splitlines())}
    assert rows["t0"]["forced_close"] and rows["t0"]["match"] and rows["t0"]["loop"]["reps"] == 8
    assert not rows["t1"]["forced_close"] and rows["t1"]["closed"] and rows["t1"]["match"]
    assert rows["t1"]["loop"] is None and rows["t0"]["loop_break"] == rows["t1"]["loop_break"] == LB.DEFAULT
    assert sum(c.endswith(LB.CLOSE) for c in calls) == 1


# ------------------------------------------------------------ the calibration and hygiene

# The traces calibration.md was built from: the first 45 and 36 rows of each arm's order. Phase C
# (addendum 2 of runs/think_long_2026-11-01) appends later rows.
AT_16_44 = {"think-trained": 45, "think-retr8": 36}


@contextlib.contextmanager
def rows_as_of_16_44(E):
    """E.stage_rows limited to the rows each arm had at 16:44, inside the `with`."""
    keep = {arm: set(order[:AT_16_44[arm]]) for arm, order in E.arm_orders(E.FIRST_RUN).items()}
    real = E.stage_rows
    E.stage_rows = lambda run, cap, arm: {i: r for i, r in real(run, cap, arm).items() if i in keep[arm]}
    try:
        yield
    finally:
        E.stage_rows = real


def test_the_calibration_reproduces_its_report():
    C = load("think_loop_calibrate", HERE / "calibrate.py")
    if not (C.E.RUN / "rows_16384_think-retr8.jsonl").exists():
        pytest.skip("the local runs are not on disk")
    tok = tokenizer_or_skip()
    with rows_as_of_16_44(C.E):
        rep = C.sweep(tok=tok)
    assert (rep["traces"], rep["finished"], rep["longest_in_finished"]) == (81, 79, 1)
    assert rep["longest_in_finished_any_length"] == 2              # hc-1540: a 10-character line pair, twice
    assert all(x["false_alarms"] == 0 for x in rep["sweep"].values())
    assert all(len(x["stopped"]) == 2 for x in rep["sweep"].values())
    stops = {x["id"]: x for x in rep["stops"]}
    assert set(stops) == {"hc-2179", "hc-0904"} and not any(x["finished"] for x in stops.values())
    assert (stops["hc-2179"]["loop_starts_at"], stops["hc-2179"]["stops_at"]) == (5469, 5581)
    assert (stops["hc-0904"]["loop_starts_at"], stops["hc-0904"]["stops_at"]) == (6603, 7067)
    assert set(rep["buffers"]) == {"train", "val"}                 # never the test split
    assert all(b["stopped_at_default"] == 0 for b in rep["buffers"].values())
    assert max(max(b["longest"].values()) for b in rep["buffers"].values()) < LB.DEFAULT["reps"]


def test_no_model_ids_and_no_thinking_flag_in_this_folder():
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh") and f.name != "test_think_loop.py":
            src = f.read_text(encoding="utf-8")
            assert "Qwen/" not in src and "enable_thinking" not in src, f.name
