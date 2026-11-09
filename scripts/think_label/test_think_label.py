"""Tests for the labeling pass (CPU only): `pytest scripts/think_label`."""
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


L = load("think_label_label", HERE / "label.py")
R = load("think_label_report", HERE / "report.py")


# ------------------------------------------------------------ the trigger

def test_trigger_runs_the_pass_only_when_tb_helps(tmp_path):
    path = tmp_path / "report.json"
    assert L.trigger(path) is None
    for tb, rc in (("helps", 0), ("tied", L.RC_SKIP), ("hurts", L.RC_SKIP)):
        path.write_text(json.dumps({"verdicts": {"TA": "helps", "TB": tb}}), encoding="utf-8")
        assert L.trigger(path) == tb
        assert L.main(["trigger", "--report", str(path)]) == rc
    path.write_text(json.dumps({"verdicts": {"TA": "helps"}}), encoding="utf-8")   # TB never finished
    assert L.main(["trigger", "--report", str(path)]) == L.RC_SKIP


# ------------------------------------------------------------ rows and plans

def test_rows_are_the_cliff_and_the_never_solved():
    if not (L.CLIFF.exists() and L.TARGETS.exists() and L.TRAIN.exists()):
        pytest.skip("the local runs are not on disk")
    rows = L.select_rows()
    assert len(rows) == 795
    assert sum(r["in_cliff"] for r in rows) == 770 and sum(r["in_never"] for r in rows) == 214
    assert sum(r["in_cliff"] and r["in_never"] for r in rows) == 189


def toy(i, instruction, buffer, command):
    return {"id": "t%d" % i, "instruction": instruction, "buffer_before": buffer, "gold_command": command,
            "mode": "ex", "cursor": [1, 1]}


def test_planner_leaves_out_the_rows_whole_component():
    train = [toy(0, "delete line two of the file", "a\nb\n", "2d"),
             toy(1, "remove the second line of the file", "a\nb\n", "2,2d"),       # t0's buffer
             toy(2, "delete line two of the list please", "x\ny\nz\n", "2,2d"),    # t1's command: t0's component
             toy(3, "delete line three of the file", "p\nq\nr\n", "3d"),
             toy(4, "delete the last line of the file", "m\nn\n", "$d"),
             toy(5, "sort the lines of the file", "c\nb\na\n", "sort")]
    plans = L.plan_rows([train[0]], train, lambda messages: 10, 10 ** 6, k=3)
    assert plans[0]["component_size"] == 3
    assert set(plans[0]["examples"]) == {"t3", "t4", "t5"}
    assert plans[0]["guard_excluded"] == 0                   # the component already holds every shared key
    assert plans[0]["examples"][-1] == "t3"                  # the nearest sits next to the query


def test_no_real_hard_row_sees_its_own_component():
    if not (L.CLIFF.exists() and L.TARGETS.exists() and L.TRAIN.exists()):
        pytest.skip("the local runs are not on disk")
    sys.path.insert(0, str(REPO / "scripts"))
    from build_splits import build_components
    train = L.load_jsonl(str(L.TRAIN))
    comp = {}
    for k, members in enumerate(build_components(train)):
        for i in members:
            comp[train[i]["id"]] = k
    byid = {c["id"]: c for c in train}
    cases = [byid[r["id"]] for r in L.select_rows()[:60]]
    for p in L.plan_rows(cases, train, lambda messages: 10, 10 ** 6):
        assert p["effective_k"] == L.K and p["id"] not in p["examples"]
        assert all(comp[e] != comp[p["id"]] for e in p["examples"])


def test_each_arm_has_its_own_fixed_order_and_seeds():
    ids = ["r%d" % i for i in range(30)]
    a, b = L.order_for("think-retr8", ids), L.order_for("nothink-retr8", ids)
    assert sorted(a) == sorted(ids) and a == L.order_for("think-retr8", ids) and a != b
    assert L.order_for("think-retr8", ids) != L.T.order_for("think-retr8", ids)   # not the thinking run's draw
    assert L.batch_seed("think-retr8", "r1") != L.batch_seed("nothink-retr8", "r1")


# ------------------------------------------------------------ the check and the report

def test_check_tells_the_gold_an_equivalent_and_a_coincidence():
    case = {"id": "x", "mode": "ex", "gold_command": "2d", "instruction": "delete line 2",
            "buffer_before": "a\nb\n", "buffer_after": "a\n", "cursor": [1, 1]}
    assert R.check(case, "ex", "2d") == "gold" and R.check(case, "ex", ":2d") == "gold"
    assert R.check(case, "keys", "jdd") == "equivalent"      # line 2 on every copy of the buffer
    assert R.check(case, "ex", "$d") == "diverges"           # the last line only while it is line 2


def test_report_pairs_thinking_with_its_control(tmp_path):
    if not L.TRAIN.exists():
        pytest.skip("the local data is not on disk")
    train = L.load_jsonl(str(L.TRAIN))[:6]
    rows = [{"id": c["id"], "in_cliff": k < 4, "in_never": k >= 3} for k, c in enumerate(train)]
    (tmp_path / "rows.json").write_text(json.dumps({
        "rows": rows, "n": 6, "cliff": 4, "never": 3, "both": 1,
        "plan": {"effective_k_mean": 8.0, "prompt_tokens_mean": 700.0}}), encoding="utf-8")

    def write(arm, matched, thinking, other=()):
        with (tmp_path / ("rows_%s.jsonl" % arm)).open("w", encoding="utf-8") as fh:
            for k, c in enumerate(train):
                cmd = ("%s xx" % c["gold_command"]) if k in other else c["gold_command"]
                fh.write(json.dumps({"id": c["id"], "tier": c["tier"], "category": c["category"],
                                     "in_cliff": k < 4, "in_never": k >= 3, "match": k in matched,
                                     "valid": True, "parse_ok": True, "closed": k != 5 or not thinking,
                                     "hit_cap": k == 5 and thinking, "mode": c["mode"], "command": cmd,
                                     "think_tokens": 100 * (k + 1) if thinking else 0,
                                     "batch_seconds": 80.0, "batch_size": 8}) + "\n")
    write("nothink-retr8", {0}, False)
    write("think-retr8", {0, 1, 2}, True, other={2})         # row 2's match is not the gold

    def coincidental(case, ranking):                         # stands in for the vim check
        return {"choice": None, "reason": "none_equivalent", "tried": [{"verdict": "diverges"}]}
    rep = R.build(tmp_path, decide=coincidental)
    lr = rep["readings"]["L"]
    assert (lr["n"], lr["gained"], lr["lost"]) == (6, 1, 0) and rep["verdict"] == "tied"
    assert rep["readings"]["matches"]["gained"] == 2
    think = rep["arms"]["think-retr8"]
    assert (think["matched"], think["labels"], think["checks"]) == (3, 2, {"gold": 2, "diverges": 1})
    assert think["thinking"]["unclosed"] == 1 and think["by_source"]["never"] == {"n": 3, "labels": 0}
    assert rep["forms"]["think-retr8"]["gold"] == 2 and rep["labeled_by_either"] == 2
    labels = [json.loads(line) for line in (tmp_path / "labels.jsonl").open(encoding="utf-8")]
    assert len(labels) == 6 and labels[2]["think-retr8"]["check"] == "diverges"
    assert "think-retr8" in R.render(rep)


# ------------------------------------------------------------ arms, driver, hygiene

def test_arms_take_their_templates_from_the_registry():
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_registry
    think, plain = load_registry(L.ARMS["think-retr8"]), load_registry(L.ARMS["nothink-retr8"])
    assert (think["hf_id"], think["revision"]) == (plain["hf_id"], plain["revision"])
    assert think["chat_template_kwargs"]["enable_thinking"] is True
    assert plain["chat_template_kwargs"]["enable_thinking"] is False
    assert list(L.ARMS) == ["nothink-retr8", "think-retr8"]                     # the control first


def test_the_labeler_is_the_base_27b_without_the_trained_adapter():
    """The trained 27B learned these rows' targets, so its answers on them are not labels."""
    src = (HERE / "label.py").read_text(encoding="utf-8")
    assert "PeftModel" not in src and "T.ADAPTER" not in src and "T.load_model" not in src


def test_driver_parses_and_runs_the_preregistered_stages():
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert "HOURS=${HOURS:-12}" in src and 'CUTOFF=${CUTOFF:-"2026-11-02 08:00"}' in src
    assert 'STEPS=${STEPS:-"trigger plan run report"}' in src
    assert "label.py run --run \"$RUN\" --deadline \"$deadline\"" in src
    assert "RUN=runs/think_label_2026-11-01" in src
    assert 'if [ "$cutoff" -lt "$deadline" ]; then deadline=$cutoff; fi' in src


def test_no_model_ids_and_no_thinking_flag_in_this_folder():
    """The thinking switch lives in the registry (`qwen3.5-27b-think`), never in code."""
    for f in HERE.iterdir():
        if f.suffix in (".py", ".sh") and f.name != "test_think_label.py":
            src = f.read_text(encoding="utf-8")
            assert "Qwen/" not in src and "enable_thinking=True" not in src, f.name
            assert "enable_thinking\": True" not in src and "enable_thinking': True" not in src, f.name
