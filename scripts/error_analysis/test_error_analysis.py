"""Tests for the error analysis (CPU only): `pytest scripts/error_analysis`."""
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import taxonomy as T          # noqa: E402
import analyze as A           # noqa: E402

B = "foo bar\nfoo baz\nqux\n"          # before
E = "FOO bar\nFOO baz\nqux\n"          # expected: lines 1 and 2 changed


def kind(got, before=B, expected=E):
    return T.classify(before, expected, got)["kind"]


def test_match_noop_and_near_kinds_take_precedence():
    assert kind(E) == "match"
    assert kind(B) == "noop"
    assert kind("FOO  bar\n\tFOO baz\nqux\n") == "ws_only"
    assert kind("FOO bar\n\nFOO baz\nqux\n") == "blank_lines_only"
    assert kind("Foo bar\nFOO baz\nqux\n") == "case_only"


def test_scope_kinds_compare_touched_lines():
    b, e = "foo bar\nfoo baz\nqux\n", "XX bar\nXX baz\nqux\n"            # gold changes lines 1 and 2
    assert T.classify(b, e, "XX bar\nfoo baz\nqux\n")["kind"] == "scope_under"          # only line 1: range too narrow
    assert T.classify(b, e, "XX bar\nXX baz\nXX\n")["kind"] == "scope_over"              # lines 1-2 and line 3 too
    assert T.classify(b, e, "YY bar\nYY baz\nqux\n")["kind"] == "same_lines_wrong_text"  # right lines, wrong edit
    assert T.classify(b, e, "foo bar\nfoo baz\nXX\n")["kind"] == "scope_disjoint"        # the wrong line only
    assert T.classify(b, e, "XX bar\nfoo baz\nXX\n")["kind"] == "scope_partial"          # one right line, one wrong
    # insertions and deletions count as touching the place they act on
    assert T.classify("a\nb\nc\n", "a\nb\n", "a\nc\n")["kind"] == "scope_disjoint"
    assert T.classify("a\nb\n", "a\nX\nb\n", "a\nb\nX\n")["kind"] == "scope_disjoint"


def test_within_line_refines_same_lines_wrong_text():
    b, e = "a a a\nz\n", "b b b\nz\n"
    assert T.classify(b, e, "b a a\nz\n")["within"] == "under-applied"        # a missing g flag
    assert T.classify(b, e, "c c c\nz\n")["within"] == "wrong replacement"    # same characters, other text
    assert T.classify("x1 y\n", "x2 y\n", "x2 yz\n")["within"] == "over-applied"
    assert T.classify("x1 y\n", "x2 y\n", "x1 yz\n")["within"] == "elsewhere in the line"
    assert T.classify("ab\ncd\n", "AB\nCD\n", "aB\nCDx\n")["within"] == "mixed"
    assert T.classify(b, e, "b a a\nZ\n")["within"] is None                   # scope_over, not same lines
    assert T.classify("a\nb\n", "a\nB\n", "a\nB\nB\n")["within"] == "unalignable"   # one line became two


def test_distance_fields_and_trailing_newline_rule():
    r = T.classify(B, E, "FOO bar\nfoo baz\nqux\n")
    assert r["diff_lines"] == 1 and r["d_lines"] == 0 and 0 < r["similarity"] < 1
    assert T.classify(B, E, "FOO bar\nFOO baz\nqux")["kind"] == "match"       # one missing trailing newline is equal
    assert T.classify(B, E, "FOO bar\nFOO baz\nqux\n\n")["kind"] != "match"   # an extra empty line is not
    assert T.classify(B, E, None)["kind"] == "no_buffer"
    assert T.line_diff_size(["a", "b", "c"], ["a", "c"]) == 1


def test_buffer_hash_matches_the_ceiling_files_rule():
    assert T.buffer_hash("a\nb\n") == T.buffer_hash("a\nb") and len(T.buffer_hash("x")) == 16
    assert T.buffer_hash("a\nb\n") != T.buffer_hash("a\nb\n\n")


def test_add_g_handles_delimiters_chains_and_escapes():
    assert T.add_g("%s/a/b/") == "%s/a/b/g"
    assert T.add_g("s/a/b/g") is None
    assert T.add_g("2,4s#x#y#|%s/q/r/") == "2,4s#x#y#g|%s/q/r/g"
    assert T.add_g("s/a/b") == "s/a/b/g"
    assert T.add_g(r"%s/\/x\//y/") == r"%s/\/x\//y/g"              # escaped delimiters inside the pattern
    assert T.add_g(r"%s/a\|b/c/") == r"%s/a\|b/c/g"                 # an escaped bar is alternation, not a chain
    assert T.add_g("g/x/d") is None and T.add_g("sort") is None


def test_mutations_by_mode():
    rules = dict(T.mutations("s/a/b/", "ex"))
    assert rules == {"add g flag": "s/a/b/g", "prefix %": "%s/a/b/"}
    assert dict(T.mutations("%s/a/b/g", "ex")) == {"drop %": "s/a/b/g"}
    assert dict(T.mutations("g/^#/d", "ex")) == {"g <-> v": "v/^#/d", "prefix %": "%g/^#/d"}
    assert dict(T.mutations("2,$v/x/d", "ex")) == {"g <-> v": "2,$g/x/d"}
    assert dict(T.mutations("dd", "keys")) == {"prefix gg": "ggdd", "prefix 0": "0dd"}
    assert T.mutations(None, "ex") == []


def test_features_read_the_case_not_a_model():
    case = {"gold_command": r"%s/\(\w\+\)/\u\1/g", "mode": "ex", "instruction": 'Capitalise every word, keep "x" as is',
            "buffer_before": "a b\nc d\n", "buffer_after": "A B\nC D\n", "cursor": [1, 1], "tier": "T3",
            "source": "handcrafted", "instruction_provenance": "claude-authored"}
    f = T.features(case)
    assert f["tier"] == "T3" and f["gold mode"] == "ex" and f["cursor"] == "(1,1)"
    assert f["ex: capture group \\("] == "yes" and f["ex: chained with |"] == "no" and f["ex: range given"] == "yes"
    assert f["instruction has a scope word"] == "yes" and f["instruction has a restriction word"] == "yes"
    assert f["instruction quotes text"] == "yes" and f["lines the gold touches"] == "2-3"
    assert f["gold changes the line count"] == "no" and f["buffer lines"] == "1-3"
    k = T.features({**case, "mode": "keys", "gold_command": "3ciwfoo<Esc>"})
    assert k["keys: inserts text (<Esc>)"] == "yes" and k["keys: has a count"] == "yes" and k["keys: text object"] == "yes"


def test_auroc_ranks_with_ties():
    assert A.auroc([(0.9, 1), (0.8, 1), (0.3, 0), (0.1, 0)]) == 1.0
    assert A.auroc([(0.5, 1), (0.5, 0)]) == 0.5
    assert A.auroc([(0.2, 1), (0.9, 0)]) == 0.0
    assert A.auroc([(0.2, 1), (0.9, 1)]) is None


def test_greedy_state_from_an_exec_log_row():
    assert A.greedy_state({"command": None}) == "no prediction"
    assert A.greedy_state({"command": "x", "blocked": "shell", "match": False, "valid": False}) == "blocked"
    assert A.greedy_state({"command": "x", "timeout": True, "match": False, "valid": False}) == "timeout"
    assert A.greedy_state({"command": "x", "match": True, "valid": True}) == "match"
    assert A.greedy_state({"command": "x", "match": False, "valid": True}) == "valid_wrong"
    assert A.greedy_state({"command": "x", "match": False, "valid": False}) == "invalid"


ARTIFACTS = [A.CASES, A.GREEDY["4b"], A.GREEDY["9b"], A.EX["4b"], A.EX["9b"], A.SCORES["4b"], A.SCORES["9b"],
             A.SAMP["4b"], A.SAMP["9b"], A.S128, A.VERD4] + [p for m in A.STAGES for _, p in A.STAGES[m]]


@pytest.mark.skipif(not all(p.exists() for p in ARTIFACTS), reason="recorded runs not on this machine")
def test_every_system_reproduces_its_recorded_figure():
    D = A.build(A.Sandbox(False))
    assert D["checks"] == A.RECORDED
    assert len(D["ids"]) == 429
    levels = {D["rows"][i]["ladder"][:2] for i in D["ids"]}
    assert levels == {"P1", "P2", "P3", "W1", "W2", "W3", "W4"}
    never = [i for i in D["ids"] if not D["rows"][i]["sys"]["pass@128 either"]]
    assert len(never) == 47                                                                # the ceiling's uncovered core
    # one of the 47 has a right answer among the fresh 16 samples of the 8 + 8 run, so it is a verifier miss, not a W4
    assert sum(1 for i in D["ids"] if D["rows"][i]["ladder"].startswith("W4")) == 46
    assert [i for i in never if D["rows"][i]["sys"]["any of 16"]] == ["hc-3047"]
