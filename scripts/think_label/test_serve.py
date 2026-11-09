"""Tests for R1's engine path (CPU only): `pytest scripts/think_label`."""
import http.server
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import threading
import time

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load("think_label_serve_t", HERE / "serve.py")
L, T, LB = S.L, S.T, S.LB


class FakeServer:
    """vLLM's /v1/completions, scripted: reply(body) -> the generated token ids (a stop included)."""

    def __init__(self, reply, delay=0.0, on_request=None):
        self.calls = []
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls.append(body)
                if on_request:
                    on_request(len(outer.calls))
                time.sleep(delay)
                ids = reply(body)
                stops = set(body["stop_token_ids"])
                finish = "stop" if ids and ids[-1] in stops else "length"
                out = json.dumps({"choices": [{"text": "", "token_ids": ids, "finish_reason": finish}],
                                  "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": len(ids)}})
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(out.encode())

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def served(script, max_tokens, stops):
    """What vLLM returns for a script: its tokens up to the cap, through the first stop."""
    out = []
    for x in script[:max_tokens]:
        out.append(x)
        if x in stops:
            break
    return out


def case(i, line):
    return {"id": i, "category": "line-ops", "tier": "T1", "instruction": "delete line %d" % line, "mode": "ex",
            "buffer_before": "a\nb\nc\n", "buffer_after": {2: "a\nc\n", 3: "a\nb\n"}[line],
            "gold_command": "%dd" % line, "cursor": [1, 1]}


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def setup_run(run, train):
    run.mkdir()
    (run / "rows.json").write_text(json.dumps({"rows": [{"id": c["id"], "in_cliff": True, "in_never": c["id"] == "t1"}
                                                        for c in train]}))
    write(run / "plan_retr8.jsonl", [{"id": c["id"], "examples": []} for c in train])


# ------------------------------------------------------------ the records are label.py's

# scripts/think_loop/test_think_loop.py's scripts: row t0 loops, row t1 closes, a forced close answers.
LOOP_SCRIPT = "The user wants line 2 gone.\n" + "* Maybe `2d` deletes the wrong line? No.\n" * 40
CLOSED_SCRIPT = "Line 3 goes.\n</think>\n\nex: 3d"


def tokenizer_or_skip():
    from transformers import AutoTokenizer
    from baseline import load_registry
    reg = load_registry("qwen3.5-27b-think")
    try:
        return AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"], local_files_only=True)
    except OSError:
        pytest.skip("the 27B's tokenizer is not in the local cache")


def script_for(tok, text, stop_id):
    if text.endswith(LB.CLOSE):
        return tok("ex: 2d", add_special_tokens=False)["input_ids"] + [stop_id]
    if "delete line 2" in text.rsplit("<|im_start|>user", 1)[-1]:
        return tok(LOOP_SCRIPT, add_special_tokens=False)["input_ids"]
    return tok(CLOSED_SCRIPT, add_special_tokens=False)["input_ids"] + [stop_id]


def test_served_rows_are_label_pys_rows_for_the_same_tokens(tmp_path, monkeypatch):
    """label.py on transformers, with the live loop check, against serve.py on a server returning the same
    tokens without it: every field label.py writes agrees, the broken loop's included."""
    tok = tokenizer_or_skip()
    torch = pytest.importorskip("torch")
    train = [case("t0", 2), case("t1", 3)]
    write(tmp_path / "train.jsonl", train)
    monkeypatch.setattr(L, "TRAIN", tmp_path / "train.jsonl")
    stop_id = tok.convert_tokens_to_ids("<|im_end|>")
    from baseline import load_registry
    info = {"sampling": dict(T.PREREGISTERED_SAMPLING), "stop_ids": [stop_id], "pad_id": tok.pad_token_id,
            "kwargs": {n: load_registry(n)["chat_template_kwargs"] for n in (T.THINK_MODEL, T.PLAIN_MODEL)}}
    hf_calls = []

    def generate(model, tok_, texts, adapter_on, max_new, info_, seed, stopping_criteria=None):
        outs, lens = [], []
        for text in texts:
            hf_calls.append(text)
            prompt, new = tok(text)["input_ids"], []
            for x in script_for(tok, text, stop_id)[:max_new]:
                new.append(x)
                if x == stop_id:
                    break
                if stopping_criteria is not None and bool(stopping_criteria(torch.tensor([prompt + new]), None)[0]):
                    break
            outs.append(T.trim(new, {stop_id}, info_["pad_id"]))
            lens.append(len(new))
        return outs, lens

    monkeypatch.setattr(L, "load_model", lambda: (tok, None, info))
    monkeypatch.setattr(T, "generate", generate)
    monkeypatch.setattr(T, "padding_guard", lambda *a, **k: 0.0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    hf = tmp_path / "hf"
    setup_run(hf, train)
    assert L.main(["run", "--run", str(hf), "--deadline", "9999999999", "--max-new-tokens", "300",
                   "--batch-size", "1"]) == 0

    served_run = tmp_path / "served"
    setup_run(served_run, train)
    assert S.main(["prompts", "--run", str(served_run)]) == 0
    monkeypatch.setattr(S, "CAP", 300)
    server = FakeServer(lambda b: served(script_for(tok, tok.decode(b["prompt"]), stop_id), b["max_tokens"],
                                         set(b["stop_token_ids"])))
    try:
        assert S.main(["run", "--run", str(served_run), "--port", str(server.port), "--attempt", "2",
                       "--in-flight", "2"]) == 0
    finally:
        server.close()

    timing = {"batch_seconds", "time"}
    for arm in L.ARMS:
        a = {r["id"]: r for r in map(json.loads, (hf / ("rows_%s.jsonl" % arm)).read_text().splitlines())}
        b = {r["id"]: r for r in map(json.loads, (served_run / ("rows_%s.jsonl" % arm)).read_text().splitlines())}
        assert set(a) == set(b) == {"t0", "t1"}
        for i in a:
            for k in set(a[i]) - timing:
                if k == "loop" and a[i]["loop"]:
                    la, lb = dict(a[i]["loop"]), dict(b[i]["loop"])
                    la.pop("answer_seconds"), lb.pop("answer_seconds")
                    assert lb.pop("overrun_tokens") > 0
                    assert la == lb, (arm, i)
                else:
                    assert a[i][k] == b[i][k], (arm, i, k)
            assert b[i]["engine"] == "vllm" and b[i]["batch_size"] == 1
    think = {r["id"]: r for r in map(json.loads, (served_run / "rows_think-retr8.jsonl").read_text().splitlines())}
    assert think["t0"]["forced_close"] and think["t0"]["match"] and think["t0"]["loop"]["reps"] == 8
    assert not think["t1"]["forced_close"] and think["t1"]["closed"] and think["t1"]["match"]
    # The control's prompts end in LB.CLOSE too (the template closes an empty reasoning when thinking is
    # off); a forced close is the one that carries the kept reasoning.
    forced = [b for b in server.calls if tok.decode(b["prompt"]).endswith("No." + LB.CLOSE)]
    assert len(forced) == 1 and forced[0]["max_tokens"] == LB.DEFAULT["answer_tokens"]
    assert forced[0]["seed"] == L.batch_seed("think-retr8", "t0")
    assert all(b[k] == T.PREREGISTERED_SAMPLING[k] for b in server.calls for k in ("temperature", "top_p", "top_k"))


# ------------------------------------------------------------ pause and resume

class CharTok:
    """One token per character, and one for the stop."""
    STOP = 1000

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in s]}

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids if i != self.STOP)


def char_run(tmp_path, monkeypatch, train):
    write(tmp_path / "train.jsonl", train)
    monkeypatch.setattr(L, "TRAIN", tmp_path / "train.jsonl")
    run = tmp_path / "run"
    setup_run(run, train)
    from baseline import load_registry
    meta = {"sampling": dict(T.PREREGISTERED_SAMPLING), "stop_ids": [CharTok.STOP], "think_end_id": 999,
            "kwargs": {n: load_registry(n)["chat_template_kwargs"] for n in (T.THINK_MODEL, T.PLAIN_MODEL)}}
    (run / S.META).write_text(json.dumps(meta))
    write(run / S.PROMPTS, [{"id": c["id"], "k_examples": 0,
                             **{arm: {"text": "p%s" % c["id"], "ids": [ord(x) for x in "p%s" % c["id"]]}
                                for arm in L.ARMS}} for c in train])
    monkeypatch.setattr(S.P, "engine_info", lambda: (CharTok(), None, meta))
    return run


def test_a_pause_finishes_the_row_in_flight_and_the_next_run_resumes(tmp_path, monkeypatch):
    run = char_run(tmp_path, monkeypatch, [case("t0", 2), case("t1", 2)])
    answer = [ord(c) for c in "ex: 2d"] + [CharTok.STOP]
    server = FakeServer(lambda b: answer, delay=0.3,
                        on_request=lambda n: (run / S.PAUSE).touch() if n == 1 else None)
    args = ["run", "--run", str(run), "--port", str(server.port), "--attempt", "2", "--in-flight", "1",
            "--arms", "nothink-retr8"]
    try:
        assert S.main(args) == S.RC_PAUSED
        rows = (run / "rows_nothink-retr8.jsonl").read_text().splitlines()
        assert len(rows) == 1 and json.loads(rows[0])["match"] and len(server.calls) == 1
        assert S.main(args) == S.RC_PAUSED            # still paused: nothing is asked
        assert len(server.calls) == 1
        (run / S.PAUSE).unlink()
        assert S.main(args) == 0
    finally:
        server.close()
    ids = [json.loads(r)["id"] for r in (run / "rows_nothink-retr8.jsonl").read_text().splitlines()]
    assert sorted(ids) == ["t0", "t1"] and len(server.calls) == 2
    sessions = [json.loads(r) for r in (run / "serve_runs.jsonl").read_text().splitlines()]
    assert [s["rc"] for s in sessions] == [S.RC_PAUSED, 0] and [s["rows"] for s in sessions] == [1, 1]


def test_failed_requests_stop_the_pass_and_write_nothing(tmp_path, monkeypatch):
    run = char_run(tmp_path, monkeypatch, [case("t%d" % k, 2) for k in range(4)])
    assert S.main(["run", "--run", str(run), "--port", "9", "--attempt", "2", "--in-flight", "1",
                   "--arms", "nothink-retr8"]) == 1
    assert not (run / "rows_nothink-retr8.jsonl").read_text()


# ------------------------------------------------------------ the paging guard, the trigger, the arguments

def stats_line(t, gen, running):
    return ("(APIServer pid=1) INFO %s [loggers.py:127] Engine 000: Avg prompt throughput: 0.0 tokens/s, "
            "Avg generation throughput: %.1f tokens/s, Running: %d reqs, Waiting: 0 reqs, "
            "GPU KV cache usage: 40.0%%, Prefix cache hit rate: 0.0%%\n" % (time.strftime("%m-%d %H:%M:%S",
                                                                                     time.localtime(t)), gen, running))


def test_the_paging_guard_reads_tokens_per_running_sequence(tmp_path):
    now = time.time()
    log = tmp_path / "vllm.log"
    log.write_text("".join(stats_line(now - 600 + 5 * k, 86.4, 8) for k in range(120)))
    rate, n = S.per_seq_rate(log, now)
    assert n >= 100 and rate == pytest.approx(10.8) and rate > S.SLOW_PER_SEQ
    log.write_text("".join(stats_line(now - 600 + 5 * k, 9.0, 6) for k in range(120)))   # a tenth: paging
    assert S.per_seq_rate(log, now)[0] < S.SLOW_PER_SEQ
    log.write_text("".join(stats_line(now - 120 + 5 * k, 9.0, 6) for k in range(24)))    # too few intervals
    assert S.per_seq_rate(log, now)[0] is None
    assert S.per_seq_rate(tmp_path / "missing.log", now) == (None, 0)


def test_the_trigger_is_tb_at_8k(tmp_path):
    path = tmp_path / "report.json"
    assert S.trigger(path) is None
    for tb, rc in (("helps", 0), ("tied", L.RC_SKIP)):
        path.write_text(json.dumps({"verdicts": {"TB4k": "tied", "TB8k": tb}}))
        assert S.main(["trigger", "--report", str(path)]) == rc
    if S.THINK_LONG_REPORT.exists():
        assert S.trigger() == "helps"


def test_the_server_is_r0s_attempt_2_at_this_passs_length(tmp_path):
    (tmp_path / S.META).write_text(json.dumps({"max_model_len": 10240}))
    out = subprocess.run([sys.executable, str(HERE / "serve.py"), "serve-args", "--run", str(tmp_path), "--attempt",
                          "2"], capture_output=True, text=True, check=True).stdout.splitlines()
    assert out == S.P.serve_args(2, S.PORT, {"max_model_len": 10240})
    assert "--enforce-eager" in out and out[out.index("--max-num-seqs") + 1] == "8"
    assert out[out.index("--max-model-len") + 1] == "10240" and "--language-model-only" in out
    assert out[out.index("--quantization") + 1] == "bitsandbytes"


# ------------------------------------------------------------ the driver and hygiene

def test_the_driver_parses_waits_by_pid_and_stops_its_server():
    src = (HERE / "r1.sh").read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(HERE / "r1.sh")], check=True)
    assert not [ln for ln in src.splitlines() if re.match(r"\s*wait\s*(#.*)?$", ln)]
    assert "trap stop_server EXIT" in src and "FREE_MIB=${FREE_MIB:-2600}" in src
    assert 'STEPS=${STEPS:-"trigger plan prompts run report"}' in src and "$SERVE run" in src
    assert "for a in 2 3; do start_server" in src
