"""Tests for the near-cliff leg: `pytest scripts/nearcliff_grpo`."""
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nearcliff as N        # noqa: E402


def test_decide_follows_the_preregistered_rule():
    assert N.decide({"solvable_share": 0.25, "solvable_ci": [0.22, 0.28]})[0]
    assert not N.decide({"solvable_share": 0.19, "solvable_ci": [0.16, 0.22]})[0]    # under 20%
    assert not N.decide({"solvable_share": 0.21, "solvable_ci": [0.14, 0.30]})[0]    # bound too low


def test_leg_steps_give_three_visits_and_respect_the_cap():
    assert N.leg_steps(200) == 375                   # 3 * 200 / (2 * 0.8)
    assert N.leg_steps(5000) == N.MAX_STEPS


def test_build_leg_takes_solvable_prompts_and_known_anchors():
    train = [{"id": "p%d" % i, "tier": "T3", "category": "substitute"} for i in range(10)]
    recheck = [{"id": "p0", "k": 0, "G": 32}, {"id": "p1", "k": 3, "G": 32},
               {"id": "p2", "k": 1, "G": 32}, {"id": "zz", "k": 5, "G": 32}]
    probe = [{"id": "p%d" % i, "k": 8, "G": 8} for i in range(3, 10)] + [{"id": "p1", "k": 0, "G": 8}]
    rows, near, anchors = N.build_leg(train, recheck, probe, anchor_ratio=1.0, seed=1)
    assert near == ["p1", "p2"]                      # p0 stays never; zz is not a train id
    assert len(anchors) == 2 and set(anchors) <= {"p%d" % i for i in range(3, 10)}
    assert [r["shelf"] for r in rows] == ["near-cliff", "near-cliff", "anchor", "anchor"]


def test_mcnemar_and_paired_row():
    assert N.mcnemar_p(0, 0) == 1.0 and abs(N.mcnemar_p(5, 0) - 0.0625) < 1e-12
    line = N.paired_row("x", {"a": 0, "b": 1}, {"a": 1, "b": 1}, ["a", "b"])
    assert "| x | 2 | 50.0% | 100.0% | +50.0 | +1 / −0 |" in line


def test_build_cli_writes_the_leg(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps({"id": "p%d" % i, "tier": "T2", "category": "join"}) + "\n" for i in range(6)))
    recheck = tmp_path / "r.jsonl"
    recheck.write_text("".join(json.dumps({"id": "p%d" % i, "k": i % 2, "G": 32}) + "\n" for i in range(4)))
    probe = tmp_path / "p.jsonl"
    probe.write_text("".join(json.dumps({"id": "p%d" % i, "k": 8, "G": 8}) + "\n" for i in range(4, 6)))
    assert N.main(["build", "--recheck", str(recheck), "--probe", str(probe), "--train", str(train),
                   "--out-dir", str(tmp_path / "leg")]) == 0
    m = json.loads((tmp_path / "leg" / "leg_manifest.json").read_text())
    assert m["near_cliff"] == 2 and m["anchors"] == 1
    assert (tmp_path / "leg" / "nearcliff_ids.txt").read_text().split() == ["p1", "p3"]
