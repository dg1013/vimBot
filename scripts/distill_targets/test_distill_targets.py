"""Tests for approach 1 (CPU only): `pytest scripts/distill_targets`."""
import json
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build_targets as B    # noqa: E402
import compare as C          # noqa: E402


def need(path):
    p = pathlib.Path(path)
    if not p.exists():
        pytest.skip("%s not present" % p)
    return p


# ------------------------------------------------------------ the target rule

def test_perturbations_keep_the_original_on_top_and_the_trailing_newline():
    a, b = B.perturb("x\ny\n")
    assert a == "x\ny\nx\ny\n" and b.startswith("x\ny\n") and b.endswith("\n") and b.count("\n") > 3
    a, b = B.perturb("x")
    assert a == "x\nx" and not b.endswith("\n")


def test_key_drops_a_leading_colon_in_ex_only():
    assert B.key("ex", " :%s/a/b/ ") == ("ex", "%s/a/b/")
    assert B.key("keys", ":wq") == ("keys", ":wq")


def probe(cmds):
    return {"commands": [{"mode": m, "command": c, "state": s, "n": n} for m, c, s, n in cmds]}


def test_ranking_sums_votes_ignores_wrong_answers_and_gives_ties_to_the_gold():
    gold = ("ex", "g")
    probes = {"p1": {"r": probe([("ex", "a", "match", 3), ("ex", "g", "match", 2), ("ex", "w", "valid_wrong", 9)])},
              "p2": {"r": probe([("ex", ":g", "match", 1)])}}
    r = B.ranked("r", gold, probes)
    assert r == [(("ex", "g"), 3, ["p1", "p2"]), (("ex", "a"), 3, ["p1"])]          # tie: the gold first
    assert B.ranked("absent", gold, probes) == []
    as_repr = {"p": {"r": {"commands": repr(probe([("keys", "x", "match", 8)])["commands"])}}}
    assert B.ranked("r", gold, as_repr) == [(("keys", "x"), 8, ["p"])]


# A tiny editor for decide(): buffers are text, commands are named transformations.
OPS = {
    "upper": lambda t: t.upper(),                                            # the gold's meaning
    "UPPER": lambda t: t.upper(),                                            # an equivalent spelling
    "upper1": lambda t: t.split("\n", 1)[0].upper() + ("\n" + t.split("\n", 1)[1] if "\n" in t else ""),
    "lower": lambda t: t.lower(),
}


def fake_run(buffer_before, command, mode="ex", cursor=(1, 1), timeout=5.0, invalid_on_long=False):
    if invalid_on_long and buffer_before.count("\n") > 1:
        return {"was_valid": False, "buffer_after": buffer_before, "timeout": False}
    return {"was_valid": True, "buffer_after": OPS[command](buffer_before), "timeout": False}


ROW = {"id": "r", "mode": "ex", "gold_command": "upper", "buffer_before": "ab", "buffer_after": "AB",
       "cursor": [1, 1], "keys_raw": None}


def rank(*cmds):
    return [(("ex", c), n, ["p"]) for c, n in cmds]


def test_an_equivalent_command_above_the_gold_replaces_it():
    d = B.decide(ROW, rank(("UPPER", 5), ("upper", 3)), run=fake_run)
    assert d["choice"] == ("ex", "UPPER") and d["reason"] == "equivalent"


def test_a_coincidental_command_is_skipped_for_the_next_one():
    # upper1 matches on the one-line buffer but not on the longer copies
    d = B.decide(ROW, rank(("upper1", 6), ("UPPER", 5), ("upper", 3)), run=fake_run)
    assert [t["verdict"] for t in d["tried"]] == ["diverges", "equivalent"] and d["choice"] == ("ex", "UPPER")


def test_the_walk_stops_at_the_gold_and_never_uses_a_stale_match():
    assert B.decide(ROW, rank(("upper", 7), ("UPPER", 5)), run=fake_run)["reason"] == "gold_is_consensus"
    d = B.decide(ROW, rank(("lower", 5), ("upper", 3)), run=fake_run)      # its probe said match; now it is not
    assert d["choice"] is None and d["tried"][0]["verdict"] == "no_match_now"
    assert B.decide(ROW, [], run=fake_run)["reason"] == "never_solved"
    broken = dict(ROW, buffer_after="XX")
    assert B.decide(broken, rank(("UPPER", 5)), run=fake_run)["reason"] == "gold_fails_now"


def test_without_a_determinable_perturbation_nothing_is_replaced():
    run = lambda *a, **k: fake_run(*a, invalid_on_long=True, **k)              # noqa: E731
    d = B.decide(dict(ROW, buffer_before="ab\nc", buffer_after="AB\nC"), rank(("UPPER", 5)), run=run)
    assert d["choice"] is None and d["tried"][0]["verdict"] == "undetermined"


def test_apply_changes_only_the_target_fields():
    row = dict(ROW, mode="keys", gold_command="gU$", keys_raw="gU$", tier="T1")
    assert B.apply(row, {"choice": None}) == row
    out = B.apply(row, {"choice": ("ex", "s/.*/\\U&/")})
    assert {k for k in row if row[k] != out[k]} == {"mode", "gold_command", "keys_raw"}
    assert out["keys_raw"] is None


# ------------------------------------------------------------ the comparison

def test_the_verdict_must_beat_the_noise_floor():
    strong = {"verdict": "stronger", "diff_pts": 3.0}
    assert C.reading(strong, {"diff_pts": -1.0}) == "helps"
    assert C.reading(strong, {"diff_pts": 4.0}) == "no detectable effect"
    assert C.reading(strong, None) == "helps"
    assert C.reading({"verdict": "weaker", "diff_pts": -3.0}, None) == "hurts"
    assert C.reading({"verdict": "tied", "diff_pts": 1.0}, None) == "no detectable effect"


def test_compare_counts_discordant_rows():
    a = {"1": 1, "2": 0, "3": 0, "4": 1}
    b = {"1": 1, "2": 1, "3": 1, "4": 0}
    r = C.compare(a, b, list(a), n_boot=200)
    assert (r["gained"], r["lost"], r["diff_pts"], r["verdict"]) == (2, 1, 25.0, "tied")


def test_train_curve_reads_the_trainer_lines(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("x\r   {'loss': '1.798', 'grad_norm': '3.6', 'entropy': '1.463', 'mean_token_accuracy': "
                   "'0.5875', 'epoch': '0.04884'}\n 10/615 [00:43]\n{'loss': '0.2', 'entropy': '0.3', "
                   "'mean_token_accuracy': '0.9', 'epoch': '3'}\n", encoding="utf-8")
    curve = C.train_curve(log)
    assert [c["loss"] for c in curve] == [1.798, 0.2]
    assert C.fit(curve, last=1) == {"loss": 0.2, "entropy": 0.3, "mean_token_accuracy": 0.9}


def test_both_arms_use_the_data_scaling_command():
    ours = (HERE / "run.sh").read_text(encoding="utf-8")
    theirs = (REPO / "scripts" / "data_scaling" / "run_scaling.sh").read_text(encoding="utf-8")
    for flags in ("--eval-every $every --no-eval-at-start --wandb", "every=$(( rows / 32 ))"):
        assert flags in ours and flags in theirs
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in ours and \
           "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in theirs


# ------------------------------------------------------------ the built file

def test_the_distilled_file_is_the_train_split_with_only_targets_changed():
    built, train = need(B.RUN / "train_distilled.jsonl"), need(B.TRAIN)
    decisions = {d["id"]: d for d in B.load_jsonl(str(need(B.RUN / "targets.jsonl")))}
    rows = B.load_jsonl(str(train))
    new = B.load_jsonl(str(built))
    assert [r["id"] for r in new] == [r["id"] for r in rows]
    for old, cur in zip(rows, new):
        changed = {k for k in old if old[k] != cur[k]}
        if decisions[old["id"]]["choice"] is None:
            assert not changed, old["id"]
        else:
            assert decisions[old["id"]]["reason"] == "equivalent"
            assert changed <= {"mode", "gold_command", "keys_raw"} and changed & {"mode", "gold_command"}
