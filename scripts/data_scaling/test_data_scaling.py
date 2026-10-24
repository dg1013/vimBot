"""Tests for the data-scaling subsets and report math: `pytest scripts/data_scaling`."""
import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data_scaling as D        # noqa: E402


def case(i, cat="substitute", tier="T2", buf=None, cmd=None):
    return {"id": "c%04d" % i, "category": cat, "tier": tier, "mode": "ex",
            "instruction": "edit number %d of the file in its own words" % i,
            "buffer_before": buf if buf is not None else "line %d\n" % i,
            "gold_command": cmd if cmd is not None else "s/%d/x/" % i}


def corpus():
    """120 singletons over two strata, plus one 30-row component sharing a buffer."""
    cases = [case(i, cat="substitute" if i % 2 else "global", tier="T3" if i % 3 else "T1")
             for i in range(120)]
    cases += [case(1000 + j, cat="increment", tier="T2", buf="shared\n") for j in range(30)]
    cases += [case(2000 + j, cat="increment", tier="T2") for j in range(30)]
    return cases


def test_subsets_are_nested_and_never_split_a_component():
    cases = corpus()
    comps = D.components_of(cases)
    subs = D.deal(cases, comps, seed=1)
    small, half = set(subs[0.25]), set(subs[0.5])
    assert small < half
    comp_of = {i: ci for ci, comp in enumerate(comps) for i in comp}
    for chosen in (small, half):
        for ci in {comp_of[i] for i in chosen}:
            assert set(comps[ci]) <= chosen          # whole components only


def test_fractions_land_near_target_in_every_stratum():
    cases = corpus()
    subs = D.deal(cases, D.components_of(cases), seed=1)
    for frac, idx in subs.items():
        assert abs(len(idx) / len(cases) - frac) < 0.12
        for key in {D.stratum(c) for c in cases}:
            total = sum(1 for c in cases if D.stratum(c) == key)
            got = sum(1 for i in idx if D.stratum(cases[i]) == key)
            # within one component of the target (the 30-row group is the coarse one)
            assert abs(got - frac * total) <= 30


def test_closer_rule_skips_a_component_bigger_than_twice_the_gap():
    # one stratum: a 40-row component and 20 singletons; target at 25% = 15 rows
    cases = [case(1000 + j, cat="join", tier="T1", buf="b\n") for j in range(40)]
    cases += [case(j, cat="join", tier="T1") for j in range(20)]
    subs = D.deal(cases, D.components_of(cases), fractions=(0.25,), seed=3)
    assert len(subs[0.25]) == 15                   # singletons only; the 40 would overshoot


def test_deal_is_deterministic_for_a_seed():
    cases = corpus()
    comps = D.components_of(cases)
    assert D.deal(cases, comps, seed=7) == D.deal(cases, comps, seed=7)
    assert D.deal(cases, comps, seed=7) != D.deal(cases, comps, seed=8)


def test_fit_log2_recovers_a_known_slope():
    xs = [1000, 2000, 4000]
    a, b = D.fit_log2(xs, [40.0, 43.0, 46.0])
    assert abs(b - 3.0) < 1e-9 and abs(a + 3.0 * math.log2(1000) - 40.0) < 1e-9


def test_bootstrap_slope_brackets_the_point_estimate():
    ids = ["v%d" % i for i in range(200)]
    # point rates 30% / 40% / 50% on nested-looking rows
    rows = [{i: (n < k) for n, i in enumerate(ids)} for k in (60, 80, 100)]
    xs = [1000, 2000, 4000]
    boot = D.bootstrap_slope(rows, xs, ids, n_boot=300, seed=5)
    lo, hi = D.quantile(boot, 0.025), D.quantile(boot, 0.975)
    assert lo < 10.0 < hi


def test_build_writes_nested_files_and_manifest(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("".join(json.dumps(c) + "\n" for c in corpus()))
    assert D.main(["build", "--train", str(train), "--out-dir", str(tmp_path / "out"), "--seed", "1"]) == 0
    a = {json.loads(l)["id"] for l in (tmp_path / "out" / "train_25.jsonl").open()}
    b = {json.loads(l)["id"] for l in (tmp_path / "out" / "train_50.jsonl").open()}
    assert a < b
    m = json.loads((tmp_path / "out" / "subsets_manifest.json").read_text())
    assert m["subsets"]["0.25"]["rows"] == len(a) and m["full"]["rows"] == 180
