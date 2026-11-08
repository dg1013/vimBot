"""Tests for the prompted-27B check (CPU only): `pytest scripts/teacher27b`."""
import json
import pathlib
import re
import sys
from types import SimpleNamespace

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import teacher27b as T       # noqa: E402
import report as R           # noqa: E402


def need(path):
    p = pathlib.Path(path)
    if not p.exists():
        pytest.skip("%s not present" % p)
    return p


def probe_default(flag):
    src = (REPO / "scripts" / "probe_difficulty.py").read_text(encoding="utf-8")
    m = re.search(r'"%s", type=\w+, default=([0-9.]+)' % re.escape(flag), src)
    assert m, flag
    return float(m.group(1))


# ------------------------------------------------------------ contracts

def test_registry_entry_is_pinned_non_thinking_and_shaped_like_the_4b():
    reg, ref = T.load_registry(T.MODEL), T.load_registry(T.REFERENCE_MODEL)
    family = lambda hf_id: hf_id.rsplit("-", 1)[0]                # noqa: E731
    assert family(reg["hf_id"]) == family(ref["hf_id"]) and reg["hf_id"] != ref["hf_id"]
    assert re.fullmatch(r"[0-9a-f]{40}", reg["revision"])
    assert reg["chat_template_kwargs"] == {"enable_thinking": False}
    assert reg["stop"] == ref["stop"]
    assert reg["max_seq_len"] == ref["max_seq_len"]          # the plans' token budget assumes it


def test_sampling_is_the_val_probes_sampling():
    assert T.G == probe_default("--num-generations")
    assert T.TEMPERATURE == probe_default("--temperature")
    assert T.MAX_COMPLETION == probe_default("--max-completion-length")
    assert T.SEED == probe_default("--seed")
    src = (REPO / "scripts" / "probe_difficulty.py").read_text(encoding="utf-8")
    assert "temperature=args.temperature, top_p=1.0, top_k=0" in src
    assert (T.TOP_P, T.TOP_K) == (1.0, 0)


def test_greedy_cap_is_the_results_table_cap():
    src = (REPO / "scripts" / "baseline.py").read_text(encoding="utf-8")
    assert re.search(r'"--max-new-tokens", type=int, default=%d\b' % T.MAX_COMPLETION, src)


def test_both_stages_take_the_examples_run_arm_takes():
    src = (REPO / "scripts" / "retrieval" / "run_arms.py").read_text(encoding="utf-8")
    assert 'return [train_by_id[i] for i in plan[case["id"]]["examples"]]' in src
    train = {"a": {"id": "a"}, "b": {"id": "b"}}
    assert T.examples_for({"examples": ["b", "a"]}, train) == [train["b"], train["a"]]
    with pytest.raises(KeyError):
        T.examples_for({"examples": ["zzz"]}, train)


def test_no_model_ids_and_no_thinking_in_this_folder():
    """§4: model ids live in the registry only; §14: thinking stays off.
    (This file is not scanned: it holds the needles.)"""
    for f in ("teacher27b.py", "report.py", "run.sh"):
        src = (HERE / f).read_text(encoding="utf-8")
        assert "Qwen/" not in src and "hf_id: " not in src, f
        assert "enable_thinking\": True" not in src and "enable_thinking=True" not in src, f


def test_readings_name_known_rows():
    for _, arm, comp, _ in R.READINGS:
        assert arm in R.ARMS and comp in T.GREEDY_COMPARATORS


# ------------------------------------------------------------ plans

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_check_plans_flags_order_held_out_and_unknown_examples(tmp_path):
    write_jsonl(tmp_path / "plan_k0.jsonl", [{"id": "v1", "examples": []}, {"id": "v2", "examples": []}])
    write_jsonl(tmp_path / "plan_retr8.jsonl", [{"id": "v1", "examples": ["t1", "t2"]},
                                                {"id": "v2", "examples": ["t1"]}])
    ok = T.check_plans(tmp_path, ["v1", "v2"], {"t1", "t2"}, {"v1", "v2"})
    assert all(r["ok"] for r in ok.values()) and ok["plan_retr8"]["k_max"] == 2
    assert not T.check_plans(tmp_path, ["v2", "v1"], {"t1", "t2"}, set())["plan_k0"]["ok"]
    assert T.check_plans(tmp_path, ["v1", "v2"], {"t1"}, set())["plan_retr8"]["not_in_train"] == 1
    assert T.check_plans(tmp_path, ["v1", "v2"], {"t1", "t2"}, {"t2"})["plan_retr8"]["held_out"] == 1


def test_the_val_plans_cover_val_in_order_and_draw_only_from_train():
    plans, cases, train, test = (need(p) for p in (T.PLANS_SRC, T.CASES, T.TRAIN, T.TEST))
    case_ids = [r["id"] for r in T.load_jsonl(str(cases))]
    held = set(case_ids) | {r["id"] for r in T.load_jsonl(str(test))}
    rep = T.check_plans(plans, case_ids, {r["id"] for r in T.load_jsonl(str(train))}, held)
    assert all(r["ok"] for r in rep.values()), rep
    assert rep["plan_k0"]["examples"] == 0 and rep["plan_retr8"]["k_max"] == 8


# ------------------------------------------------------------ guards

def test_latency_guard_skips_warmup_and_one_slow_call():
    g = T.LatencyGuard(1.0, window=3, warmup=2)
    for s in (60.0, 60.0):                  # kernel compilation on first use: not counted
        g.add(s)
    for s in (0.2, 5.0, 0.2, 0.2):          # one outlier: the median stays at 0.2
        g.add(s)
    assert g.baseline == 0.2


def test_latency_guard_trips_above_the_limit_or_on_a_slowdown():
    absolute = T.LatencyGuard(1.0, window=3, warmup=0)
    absolute.add(1.1)
    absolute.add(1.2)
    with pytest.raises(T.SlowCard):
        absolute.add(1.3)                   # paging from the start: over the ceiling
    relative = T.LatencyGuard(10.0, factor=4.0, window=3, warmup=0)
    for s in (0.1, 0.1, 0.1, 0.5, 0.5):     # baseline 0.1, then 5x with the ceiling far away
        try:
            relative.add(s)
        except T.SlowCard:
            break
    else:
        pytest.fail("a 5x slowdown did not trip the guard")


class Shape:
    def __init__(self, n):
        self.shape = (1, n)


def test_latency_guard_wraps_generate_and_counts_new_steps():
    model = SimpleNamespace(generate=lambda *a, **k: Shape(10))
    g = T.LatencyGuard(10.0, window=2, warmup=0)
    g.wrap(model)
    assert model.generate(input_ids=Shape(4), x=2).shape == (1, 10)
    assert model.generate(Shape(4)).shape == (1, 10)      # positional prompt too
    assert len(g.times) == 2 and all(t < 1.0 for t in g.times)   # seconds over 6 new steps


def test_patch_summary_relabels_and_leaves_the_metrics(tmp_path):
    p = tmp_path / "baseline_val.json"
    p.write_text(json.dumps({"metrics": {"exec_match_pct": 40.0},
                             "run": {"experiment": "retrieval_2026-10-18", "name": "27b-retr8"}}))
    T.patch_summary(p, {"experiment": T.EXPERIMENT, "peak_gpu_gib": 19.1})
    d = json.loads(p.read_text())
    assert d["metrics"] == {"exec_match_pct": 40.0}
    assert d["run"] == {"experiment": T.EXPERIMENT, "name": "27b-retr8", "peak_gpu_gib": 19.1}


# ------------------------------------------------------------ report math

def test_verdict_is_three_way():
    assert R.verdict(5.0, 0.01) == "stronger"
    assert R.verdict(-5.0, 0.01) == "weaker"
    assert R.verdict(5.0, 0.2) == "tied"


def test_compare_counts_discordant_rows():
    a = {"1": 1, "2": 0, "3": 0, "4": 1}
    b = {"1": 1, "2": 1, "3": 1, "4": 0}
    r = R.compare(a, b, list(a), n_boot=200)
    assert (r["gained"], r["lost"], r["n"], r["diff_pts"]) == (2, 1, 4, 25.0)
    assert r["p"] == 1.0 and r["verdict"] == "tied"
    assert r["ci95"][0] <= 25.0 <= r["ci95"][1]


def test_plurality_reads_both_probe_formats_and_breaks_ties_by_order():
    listed = {"commands": [{"command": "x", "state": "match", "n": 3},
                           {"command": "y", "state": "valid_wrong", "n": 3}]}
    as_repr = {"commands": repr([{"command": "y", "state": "valid_wrong", "n": 5},
                                 {"command": "x", "state": "match", "n": 3}])}
    assert R.plurality_right(listed) == 1
    assert R.plurality_right(as_repr) == 0
    assert R.plurality_right({"commands": []}) == 0


def test_complementarity():
    arm = {"a": 1, "b": 1, "c": 0}
    others = [{"a": 1, "b": 0, "c": 0}, {"a": 0, "b": 0, "c": 1}]
    c = R.complementarity(arm, others, ["a", "b", "c"])
    assert c["only_arm"] == ["b"]
    assert (c["union_without_pct"], c["union_with_pct"]) == (66.7, 100.0)


def test_next_step_follows_the_two_primary_verdicts():
    assert "train scale" in R.next_step("stronger", "stronger")
    assert "bigger card" in R.next_step("weaker", "weaker")
    assert "diversity" in R.next_step("tied", "tied")
    assert "diversity" in R.next_step("weaker", "tied")


# ------------------------------------------------------------ the report end to end

def fake_arm(folder, outcomes, mode="ex"):
    """A run folder as run_arm / full_val_eval write it, from id -> (match, valid)."""
    write_jsonl(folder / "exec_val.jsonl", [{"id": i, "match": m, "valid": v} for i, (m, v) in outcomes.items()])
    write_jsonl(folder / "preds_val.jsonl", [{"id": i, "command": "x", "mode": mode} for i in outcomes])
    write_jsonl(folder / "completions_val.jsonl", [
        {"id": i, "completion": "ex: x", "output_tokens": 5, "prompt_tokens": 100, "k_examples": 0,
         "latency_s": 1.0, "parse_ok": True, "think_leak": False, "mode": mode, "command": "x"}
        for i in outcomes])
    (folder / "baseline_val.json").write_text(json.dumps({"run": {"load": {"load_s": 200}, "peak_gpu_gib": 19.0}}))


def probe_rows(path, ks):
    write_jsonl(path, [{"id": i, "k": k, "G": 8, "p_hat": k / 8, "states": {"match": k, "valid_wrong": 8 - k},
                        "tokens_mean": 9.0,
                        "commands": [{"command": "x", "state": "match" if k >= 4 else "valid_wrong", "n": 8}]}
                       for i, k in ks.items()])


def test_report_builds_and_renders_on_a_fixture_run(tmp_path, monkeypatch):
    ids = ["r%d" % n for n in range(6)]
    cases = tmp_path / "val.jsonl"
    write_jsonl(cases, [{"id": i, "tier": "T%d" % (1 + n % 3)} for n, i in enumerate(ids)])
    run = tmp_path / "run"
    fake_arm(run / "27b-retr8", {i: (n < 4, 1) for n, i in enumerate(ids)})
    fake_arm(run / "27b-k0", {i: (n < 1, n < 3) for n, i in enumerate(ids)})
    comps = {}
    for label, wins in (("4b-base-k0", 0), ("4b-base-retr8", 2), ("4b-served", 3), ("9b", 3)):
        fake_arm(tmp_path / label, {i: (n < wins, 1) for n, i in enumerate(ids)})
        comps[label] = (str(tmp_path / label), label)
    probe_rows(run / T.PROBE_OUT.name, {i: (8 if n < 3 else 1) for n, i in enumerate(ids)})
    for label in ("4b-served", "9b"):
        probe_rows(tmp_path / ("probe_%s.jsonl" % label), {i: (8 if n < 2 else 0) for n, i in enumerate(ids)})
    monkeypatch.setattr(T, "CASES", cases)
    monkeypatch.setattr(T, "GREEDY_COMPARATORS", comps)
    monkeypatch.setattr(T, "PROBE_COMPARATORS", {k: (str(tmp_path / ("probe_%s.jsonl" % k)), k)
                                                 for k in ("4b-served", "9b")})
    rep = R.build(SimpleNamespace(run=str(run), timeout=5.0, no_rescore=True))
    assert rep["arms"]["27b-retr8"]["exec_match_pct"] == 66.7
    assert rep["readings"]["P1"]["gained"] == 1 and rep["readings"]["P1"]["lost"] == 0
    assert rep["S3"]["only_arm"] == ["r3"]
    assert rep["S4"]["27b-retr8"]["pass_at_8_pct"] == 100.0
    assert rep["S4"]["coverage"]["only_arm"] == ["r2", "r3", "r4", "r5"]
    md = R.render(rep)
    assert "| 27b-retr8 | **66.7%**" in md and "Next step, by the pre-registered rule" in md
    assert "S4: 8 samples" in md and "Re-score audit" in md
