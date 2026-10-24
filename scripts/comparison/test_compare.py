"""Tests for the side-by-side comparison (CPU only): `pytest scripts/comparison`."""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_test_v2 as C        # noqa: E402


def test_wilson_interval_is_inside_0_100_and_contains_the_rate():
    lo, hi = C.wilson(233, 429)
    assert 0 <= lo < 100 * 233 / 429 < hi <= 100
    assert round(lo, 1) == 49.6 and round(hi, 1) == 59.0
    lo0, hi0 = C.wilson(0, 10)
    assert lo0 == 0 or abs(lo0) < 1e-9 and hi0 > 0


def test_paired_counts_and_exact_mcnemar():
    ids = list(range(10))
    ref = {i: int(i < 5) for i in ids}
    row = {i: int(i < 3 or i in (7, 8, 9)) for i in ids}
    g, l, p = C.paired(ref, row, ids)
    assert (g, l) == (3, 2)
    assert C.mcnemar_p(0, 0) == 1.0 and C.mcnemar_p(37, 13) < 0.002 and C.mcnemar_p(5, 5) == 1.0


def test_unions_count_each_row_once():
    ids = list(range(6))
    rows = {"verifier-4b": {"outcome": {i: int(i in (0, 1, 2)) for i in ids}},
            "sample-rule": {"outcome": {i: int(i in (2, 3)) for i in ids}}}
    old = C.UNION_PAIRS
    try:
        C.UNION_PAIRS = [("verifier-4b", "sample-rule")]
        u = C.unions(rows, ids)[0]
    finally:
        C.UNION_PAIRS = old
    assert (u["both"], u["only_a"], u["only_b"]) == (1, 2, 1) and u["either_pct"] == round(100 * 4 / 6, 1)


def test_buffer_key_ignores_one_trailing_newline_only():
    assert C.key("a\nb\n") == C.key("a\nb") == ("a", "b")
    assert C.key("a\nb\n\n") != C.key("a\nb")


def test_every_reference_figure_is_pinned():
    assert C.RECORDED["served"] == 233 and C.RECORDED["sample-rule"] == 256
