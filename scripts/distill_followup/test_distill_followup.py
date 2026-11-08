"""Tests for approach 1's second seed and GRPO on the distilled SFT (CPU only):
`pytest scripts/distill_followup`."""
import importlib.util
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S = load("distill_followup_seeds", HERE / "compare_seeds.py")
G = load("distill_followup_grpo", HERE / "report_grpo.py")


def test_part_a_verdict_follows_the_primary_reading():
    assert S.call({"verdict": "stronger", "diff_pts": 6.0}) == "replicates"
    assert S.call({"verdict": "tied", "diff_pts": 2.0}) == "direction holds"
    assert S.call({"verdict": "tied", "diff_pts": 0.0}) == "does not replicate"
    assert S.call({"verdict": "weaker", "diff_pts": -4.0}) == "does not replicate"


def test_pooled_is_the_mean_of_the_two_seeds_differences():
    ids = ["a", "b", "c", "d"]
    zero = dict.fromkeys(ids, 0)
    r = S.pooled(zero, {"a": 1, "b": 1, "c": 0, "d": 0}, zero, {"a": 1, "b": 0, "c": 0, "d": 0}, ids, n_boot=200)
    assert r["diff_pts"] == 37.5                    # (2/4 + 1/4) / 2
    assert r["ci95"][0] <= 37.5 <= r["ci95"][1]


def test_grpo_readings_name_known_arms_and_the_served_comparator():
    for _, a, b, _ in G.READINGS:
        assert a in G.ARMS and b in G.ARMS
    assert G.READINGS[0][:3] == ("R", "served", "distill-grpo-gated")
    assert G.ARMS["served"][0] == "runs/qwen3.5-4b-grpo-gated-leg2"
    assert G.ARMS["distill-sft"][0] == "runs/qwen3.5-4b-sft-distill"


def test_driver_parses_and_runs_the_preregistered_commands():
    subprocess.run(["bash", "-n", str(HERE / "run.sh")], check=True)
    src = (HERE / "run.sh").read_text(encoding="utf-8")
    assert "SEED2=7" in src and "--seed $SEED2" in src
    assert "START=runs/qwen3.5-4b-sft-distill/best" in src
    assert 'grpo_round grpo1 scripts/train_grpo.py "$START" "$G1"' in src
    assert 'grpo_round grpo2 scripts/train_grpo_gated.py "$G1/best" "$G2" --difficulty "$PROBE"' in src


def test_no_model_ids_and_no_thinking_in_this_folder():
    for f in ("run.sh", "compare_seeds.py", "report_grpo.py"):
        src = (HERE / f).read_text(encoding="utf-8")
        assert "Qwen/" not in src and "enable_thinking=True" not in src, f
