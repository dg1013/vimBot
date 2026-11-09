"""CPU tests for scripts/opd27b (on-policy distillation from the trained 27B)."""
import importlib.util
import json
import pathlib
import re
import subprocess

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = load("opd27b_opd_t", HERE / "opd.py")
TCH = load("opd27b_teacher_t", HERE / "teacher.py")
LOCAL = (REPO / "data" / "splits" / "train.jsonl").exists()
local_only = pytest.mark.skipif(not LOCAL, reason="the splits are local-only")


def teacher_buckets(t_logits, k):
    """What teacher.py records, as tensors: top-k ids, their log-probabilities, the rest's."""
    st = TCH.position_stats(t_logits, t_logits.argmax(-1), topk=k)
    return (torch.tensor(st["topk_ids"]), torch.tensor(st["topk_lp1"]), torch.tensor(st["lrest1"]))


def full_kl(s_logits, t_logits):
    ls, lt = torch.log_softmax(s_logits, -1), torch.log_softmax(t_logits, -1)
    return (ls.exp() * (ls - lt)).sum(-1)


def test_bucket_kl_by_hand():
    s = torch.tensor([[2.0, 1.0, 0.0, -1.0, -1.0, -3.0]])
    t = torch.tensor([[0.5, 2.5, 0.0, -2.0, -1.0, -4.0]])
    ids, lp, rest = teacher_buckets(t, 2)                    # the teacher's top 2: tokens 1 and 0
    assert ids.tolist() == [[1, 0]]
    ps, pt = torch.softmax(s, -1)[0], torch.softmax(t, -1)[0]
    qs = torch.stack([ps[1], ps[0], ps[2:].sum()])
    qt = torch.stack([pt[1], pt[0], pt[2:].sum()])
    want = float((qs * (qs / qt).log()).sum())
    got = float(O.bucket_kl(s, ids, lp, rest)[0])
    assert got == pytest.approx(want, abs=2e-4)               # the recording rounds to 4 decimals


def test_bucket_kl_is_zero_when_the_buckets_match_and_bounded_by_the_full_kl():
    g = torch.Generator().manual_seed(0)
    t = torch.randn(64, 50, generator=g) * 3
    ids, lp, rest = teacher_buckets(t, 5)
    assert O.bucket_kl(t.clone(), ids, lp, rest).abs().max() < 1e-3
    s = torch.randn(64, 50, generator=g) * 3
    b = O.bucket_kl(s, ids, lp, rest)
    assert (b >= -1e-5).all()
    assert (b <= full_kl(s, t) + 1e-3).all()                   # coarsening can only lower a KL


def test_bucket_kl_equals_the_full_kl_when_the_top_k_hold_the_mass():
    s = torch.tensor([[1.0, 0.2, -0.5, 0.0, -1e4]])
    t = torch.tensor([[0.0, 1.5, 0.3, -0.2, -1e4]])
    ids, lp, rest = teacher_buckets(t, 4)
    assert float(O.bucket_kl(s, ids, lp, rest)[0]) == pytest.approx(float(full_kl(s, t)[0]), abs=2e-4)


def test_a_gradient_step_moves_the_student_toward_the_teacher():
    g = torch.Generator().manual_seed(1)
    t = torch.randn(8, 30, generator=g) * 2
    ids, lp, rest = teacher_buckets(t, 20)
    s = torch.randn(8, 30, generator=g, requires_grad=True)
    before = O.bucket_kl(s, ids, lp, rest).sum()
    before.backward()
    with torch.no_grad():
        after = O.bucket_kl(s - 0.5 * s.grad, ids, lp, rest).sum()
    assert after < before


def test_position_stats_buckets_sum_to_one():
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(5, 300, generator=g) * 4
    st = TCH.position_stats(logits, torch.tensor([3, 1, 4, 1, 5]), topk=20)
    lq = torch.cat([torch.tensor(st["topk_lp1"]), torch.tensor(st["lrest1"])[:, None]], 1)
    assert torch.logsumexp(lq, -1).abs().max() < 1e-3
    assert st["tlp1"][0] == pytest.approx(float(torch.log_softmax(logits, -1)[0, 3]), abs=1e-4)


def test_build_rows_refuses_a_shifted_record():
    prompts = {"a": [1, 2, 3]}
    good = {"id": "a", "ids": [7, 8], "prompt_sha": O.ids_sha([1, 2, 3]), "topk_ids": [[1] * 20] * 2,
            "topk_lp1": [[0.0] * 20] * 2, "lrest1": [0.0, 0.0]}
    rows = O.build_rows([{"id": "a", "ids": [7, 8]}], [good], prompts)
    assert rows[0][:3] == ("a", [1, 2, 3], [7, 8])
    for bad in (dict(good, ids=[7, 9]), dict(good, prompt_sha="x"), dict(good, lrest1=[0.0])):
        with pytest.raises(SystemExit):
            O.build_rows([{"id": "a", "ids": [7, 8]}], [bad], prompts)


def test_micro_batch_positions_predict_each_sampled_token():
    k = O.TOPK
    rows = [("a", [10, 11, 12], [20, 21], [[1] * k] * 2, [[0.0] * k] * 2, [0.0, 0.0]),
            ("b", [10], [30, 31, 32], [[2] * k] * 3, [[0.0] * k] * 3, [0.0] * 3)]
    mb = O.micro_batch(rows, "cpu")
    assert mb["input_ids"].tolist() == [[10, 11, 12, 20, 21], [10, 30, 31, 32, 0]]
    assert mb["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert list(zip(mb["pos_b"].tolist(), mb["pos_t"].tolist())) == [(0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    ids = mb["input_ids"]
    assert [int(ids[b, t + 1]) for b, t in zip(mb["pos_b"].tolist(), mb["pos_t"].tolist())] == mb["sampled"].tolist()


def test_choice_seeds_and_orders():
    assert O.choose({0: {"exec_match": 10}, 1: {"exec_match": 10}, 2: {"exec_match": 9}}) == 0
    assert O.choose({0: {"exec_match": 10}, 1: {"exec_match": 12}, 2: {"exec_match": 12}}) == 1
    ids = ["c", "a", "b", "d"]
    assert O.round_order(42, 1, ids) == O.round_order(42, 1, list(reversed(ids)))
    assert O.round_order(42, 1, ids) != O.round_order(42, 2, ids) or O.round_order(7, 1, ids) != O.round_order(42, 1, ids)
    assert O.sample_seed(42, 1, "a") != O.sample_seed(42, 2, "a") != O.sample_seed(7, 1, "a")
    assert O.adapter_for(42, 0) == O.STARTS[42] / "best" and O.adapter_for(7, 2).parts[-3:] == (
        "qwen3.5-4b-opd27b-s7", "round2", "adapter")


def test_final_chooses_then_calls_the_results_table_path(tmp_path, monkeypatch):
    import sys
    import types
    for r, m in enumerate((10, 12, 12, 11)):
        d = tmp_path / ("round%d" % r)
        d.mkdir()
        (d / "eval.json").write_text(json.dumps({"exec_match": m, "exec_match_pct": m / 2.5}))
    seen = {}

    def stub(reg, tok, adapter, val_cases, out_dir, **kw):
        seen.update(adapter=adapter, out_dir=out_dir, rows=len(val_cases), **kw)
        return {"metrics": {"exec_match_pct": 50.0}}
    monkeypatch.setitem(sys.modules, "train_sft", types.SimpleNamespace(full_val_eval=stub))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: "tok")
    monkeypatch.setattr(O, "load_jsonl", lambda path: [{"id": "v1"}, {"id": "v2"}])
    args = O.build_parser().parse_args(["final", "--seed", "42", "--root", str(tmp_path)])
    assert O.cmd_final(args) == 0
    pick = json.loads((tmp_path / "chosen.json").read_text())
    assert pick["chosen_round"] == 1 and seen["adapter"] == tmp_path / "round1" / "adapter"
    assert seen["out_dir"] == tmp_path and seen["summary_name"] == "opd_val.json" and seen["load_in_4bit"] is False
    assert seen["rows"] == 2 and seen["max_new_tokens"] == O.MAX_NEW


def test_sampler_helpers():
    S = load("opd27b_sample_t", HERE / "sample.py")
    assert S.engine_kwargs({"vllm_extra_args": ["--max-num-seqs", "128"]}) == {"max_num_seqs": 128}

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return ",".join(map(str, ids))
    assert S.text_of(Tok(), [5, 6, 248046], [248044, 248046]) == "5,6"
    assert S.text_of(Tok(), [5, 6], [248046]) == "5,6"


def test_no_model_ids_or_thinking_in_code():
    for f in ("opd.py", "sample.py", "teacher.py", "report.py", "run.sh"):
        text = (HERE / f).read_text(encoding="utf-8")
        assert "Qwen/" not in text, f
        assert "enable_thinking=True" not in text and "enable_thinking: true" not in text, f


def test_the_driver_waits_for_its_judges_by_pid(tmp_path):
    """bash 5.1's bare `wait` also waits for the tee behind run.sh's `exec > >(tee ...)`, which never
    exits: the check stage hung there for 8 h on 2026-11-02. The driver waits for the judges' pids."""
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert not [ln for ln in src.splitlines() if re.match(r"\s*wait\s*(#.*)?$", ln)]
    fn = re.search(r"^JUDGES=\(\)$.*?^}$", src, re.S | re.M).group(0)
    done = tmp_path / "returned"
    script = "\n".join(["exec > >(cat > /dev/null) 2>&1", "status() { :; }", fn,
                        "sleep 0.1 &", "JUDGES+=($!)", "wait_judges", "echo yes > %s" % done])
    subprocess.run(["bash", "-c", script], timeout=20, check=True)
    assert done.read_text() == "yes\n"


@local_only
def test_prompt_ids_are_trls_and_the_teachers():
    path = O.RUN / "prompts.jsonl"
    if not path.exists():
        pytest.skip("prompts.jsonl not built yet")
    REC = load("teacher_probs_record_t", REPO / "scripts" / "teacher_probs" / "record.py")
    from transformers import AutoTokenizer
    from baseline import load_registry
    reg = load_registry(O.TEACHER)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    from evaluate import load_jsonl
    case = load_jsonl(str(O.TRAIN))[0]
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    p, _ = REC.encode(tok, case, case["mode"], case["gold_command"], reg.get("chat_template_kwargs") or {})
    assert row["id"] == case["id"] and row["ids"] == p


@local_only
def test_the_selection_subset_is_the_untouched_complement():
    R = load("opd27b_report_t", HERE / "report.py")
    from evaluate import load_jsonl
    val = load_jsonl(str(O.VAL))
    sub = {c["id"] for c in O.subset_cases()}
    assert len(sub) == 250 and set(R.C.untouched_ids(val)) == {c["id"] for c in val} - sub
    rep = R.build()                                           # partial results work: no verdict until both arms are in
    both = all("opd-s%d" % s in rep["arms"] for s in O.STARTS)
    assert rep["verdict"] in (("helps", "tied", "hurts") if both else (None,))
