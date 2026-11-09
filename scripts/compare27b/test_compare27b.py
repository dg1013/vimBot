"""Tests for the 27B comparison (CPU only): `pytest scripts/compare27b`.

The rebuild must reproduce every figure its sources' own reports recorded. Skipped when the local
runs are not on disk (runs/ stays off GitHub)."""
import contextlib
import importlib.util
import pathlib

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("compare27b", HERE / "compare.py")
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

NEEDED = [REPO / "data" / "splits" / "val.jsonl", REPO / M.TEACHER / "report.json", M.E.RUN / "rows_8192_think-trained.jsonl"]
NEEDED += [REPO / f / "exec_val.jsonl" for f, _ in M.GREEDY.values()] + [REPO / p for p, _ in M.PROBES.values()]
NEEDED += [REPO / f / "exec_test.jsonl" for f in M.TEST.values()]

# The rows the 8k report covered: the first 45 and 36 of each arm's order. Phase C (addendum 2 of
# runs/think_long_2026-11-01) appends later rows, and the figures below are the 13:45 readings.
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


@pytest.fixture(scope="module")
def rep():
    missing = [str(p) for p in NEEDED if not p.exists()]
    if missing:
        pytest.skip("the local runs are not on disk: %s" % missing[0])
    with rows_as_of_16_44(M.E):
        return M.build()


def test_val_rows_reproduce_their_reports(rep):
    m = rep["val"]["methods"]
    # (untouched, all val): the QLoRA, prompted-27B and KD reports' figures
    assert (m["27b-trained"]["untouched"]["pct"], m["27b-trained"]["all"]["pct"]) == (60.7, 61.8)
    assert (m["27b-retr8"]["untouched"]["pct"], m["27b-retr8"]["all"]["pct"]) == (54.6, 54.5)
    assert (m["27b-k0"]["untouched"]["pct"], m["27b-k0"]["all"]["pct"]) == (27.8, 29.5)
    assert (m["4b-served"]["untouched"]["pct"], m["4b-served"]["all"]["pct"]) == (50.5, 51.9)
    assert m["27b-retr8@8"]["all"]["pct"] == 48.3
    assert m["27b-trained"]["all"]["seconds_p50"] == 1.5 and m["27b-retr8"]["all"]["seconds_p50"] == 1.57
    c = rep["val"]["ceilings"]
    assert c["27B any of 8 samples (8 examples, T = 1)"]["all"] == 69.7
    assert c["served 4B any of 8 samples (T = 1)"]["all"] == 64.2


def test_the_matrix_is_antisymmetric_and_matches_the_preregistered_readings(rep):
    mx = rep["val"]["matrix"]
    for part in ("untouched", "all"):
        for key, r in mx[part].items():
            a, b = key.split("|")
            back = mx[part]["%s|%s" % (b, a)]
            assert (r["diff_pts"], r["gained"], r["lost"]) == (-back["diff_pts"], back["lost"], back["gained"])
            assert r["p"] == back["p"]
    q3 = mx["untouched"]["27b-trained|27b-retr8"]          # the QLoRA report's Q3
    assert (q3["diff_pts"], q3["gained"], q3["lost"]) == (6.1, 46, 28)
    p1b = mx["all"]["27b-retr8|4b-served"]                 # the prompted-27B report's P1b
    assert p1b["diff_pts"] == 2.6 and p1b["ci95"] == [-2.0, 7.0]


def test_thinking_panels_match_the_8k_report(rep):
    ta, tb = rep["thinking"]["think-trained"], rep["thinking"]["think-retr8"]
    assert (ta["rows"], ta["solved"]) == (45, 23) and (tb["rows"], tb["solved"]) == (36, 21)
    ctl = "no thinking, sampled at T 0.6 (its control)"
    assert ta["columns"][ctl]["solved"] == 6 and tb["columns"][ctl]["solved"] == 9
    assert ta["columns"][ctl]["thinking_vs"]["gained"] == 19 and ta["columns"][ctl]["thinking_vs"]["lost"] == 2


def test_test_v2_matches_its_report(rep):
    t = rep["test"]
    assert t["n"] == 429
    assert t["methods"]["27b-trained"]["pct"] == 67.6 and t["methods"]["4b-served"]["pct"] == 54.3
    assert (t["reading"]["gained"], t["reading"]["lost"], t["reading"]["diff_pts"]) == (91, 34, 13.3)


def test_no_model_id_in_code():
    text = (HERE / "compare.py").read_text(encoding="utf-8")
    assert "Qwen/" not in text and "enable_thinking" not in text
