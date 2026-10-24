"""Tests for scripts/difficulty.py, scripts/select_frontier.py and the
model-free parts of scripts/probe_difficulty.py (difficulty-gated GRPO,
DECISIONS.md 2026-09-14).

The claim under test: a prompt carries gradient only while the policy's
samples disagree about it, so the bookkeeping must (1) pool observations so a
single lucky or unlucky group does not misfile a prompt, (2) put every prompt
on exactly one shelf, (3) draw distinct prompts per chunk mostly from the
frontier while still visiting the other shelves at the stated rates, and
(4) keep coverage of the frontier even across refreshes.
"""

import collections
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import difficulty  # noqa: E402
import select_frontier  # noqa: E402
from difficulty import ChunkPlanner, DifficultyBook, merge_unprobed  # noqa: E402

IDS = ["p%02d" % i for i in range(20)]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# ---------------------------------------------------------------- basics

def test_pool_spec_parsing():
    assert difficulty.parse_pool("latest") == 1
    assert difficulty.parse_pool("last:1") == 1
    assert difficulty.parse_pool("last:3") == 3
    assert difficulty.parse_pool("all") is None
    for bad in ("last:0", "last:x", "recent", ""):
        with pytest.raises(ValueError):
            difficulty.parse_pool(bad)


def test_fuel_is_the_group_sum_of_squared_advantages():
    """k(G-k)/G: zero at both ends, maximal at k = G/2; the quantity the
    gate is trying to raise."""
    assert difficulty.fuel(0, 8) == 0 and difficulty.fuel(8, 8) == 0
    assert difficulty.fuel(4, 8) == pytest.approx(2.0)
    assert difficulty.fuel(1, 8) == pytest.approx(7 / 8)
    assert difficulty.fuel(3, 0) == 0.0


def test_shelf_of():
    assert difficulty.shelf_of(0, 8) == "cliff"
    assert difficulty.shelf_of(8, 8) == "known"
    assert difficulty.shelf_of(1, 8) == "frontier" and difficulty.shelf_of(7, 8) == "frontier"
    assert difficulty.shelf_of(0, 0) == "unprobed"


# ------------------------------------------------------------------ book

def test_book_loads_a_difficulty_file_and_skips_what_is_not_its_business(tmp_path):
    f = tmp_path / "d.jsonl"
    _write(f, [{"id": "p00", "k": 8, "G": 8, "step": 0, "source": "probe", "commands": []},
               {"id": "p01", "k": 0, "G": 8}, {"id": "p02", "k": 3, "G": 8},
               {"id": "zzz", "k": 1, "G": 8},                 # not in the book
               {"id": "p03", "k": "x", "G": 8}, {"id": "p04"}])   # malformed
    book = DifficultyBook(IDS)
    assert book.load(f) == 3
    assert book.unknown_ids == 1 and book.malformed == 2
    assert book.shelf("p00") == "known" and book.shelf("p01") == "cliff"
    assert book.shelf("p02") == "frontier" and book.shelf("p03") == "unprobed"
    assert book.observed() == 3
    counts = book.counts()
    assert counts == {"cliff": 1, "frontier": 1, "known": 1, "unprobed": 17}
    assert sum(counts.values()) == len(IDS)


def test_book_refuses_duplicate_ids():
    with pytest.raises(ValueError):
        DifficultyBook(["a", "a"])


def test_pooling_last_two_needs_two_perfect_groups_to_leave_the_frontier():
    """The debounce: one 8-of-8 after a mixed group keeps the prompt on the
    frontier (a p = 0.9 prompt shows 8 of 8 43% of the time); two in a row
    move it to the known shelf; one failure in an anchor visit brings it back."""
    book = DifficultyBook(IDS, pool="last:2")
    book.record("p00", 5, 8, step=1)
    assert book.shelf("p00") == "frontier"
    book.record("p00", 8, 8, step=2)
    assert book.shelf("p00") == "frontier"          # pooled 13 of 16
    book.record("p00", 8, 8, step=3)
    assert book.shelf("p00") == "known"             # pooled 16 of 16
    book.record("p00", 7, 8, step=9)
    assert book.shelf("p00") == "frontier"          # regression noticed
    assert book.pooled("p00") == (15, 16, 2)


def test_pooling_latest_and_all():
    latest, every = DifficultyBook(IDS, pool="latest"), DifficultyBook(IDS, pool="all")
    for b in (latest, every):
        b.record("p00", 0, 8, step=1)
        b.record("p00", 8, 8, step=2)
    assert latest.shelf("p00") == "known" and latest.pooled("p00") == (8, 8, 1)
    assert every.shelf("p00") == "frontier" and every.pooled("p00") == (8, 16, 2)
    assert latest.p_hat("p00") == 1.0 and every.p_hat("p00") == 0.5
    assert latest.p_hat("p01") is None


def test_record_appends_to_the_attached_log_which_is_itself_a_difficulty_file(tmp_path):
    log = tmp_path / "sub" / "difficulty_log.jsonl"
    book = DifficultyBook(IDS)
    book.attach_log(log)
    row = book.record_states("p00", ["match", "match", "invalid", "valid_wrong"], step=7, distinct=3)
    assert row["k"] == 2 and row["G"] == 4 and row["states"] == {"match": 2, "invalid": 1, "valid_wrong": 1}
    assert book.record("nope", 1, 8, step=1) is None and book.unknown_ids == 1
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    assert lines == [{"id": "p00", "k": 2, "G": 4, "step": 7, "source": "train",
                      "states": {"match": 2, "invalid": 1, "valid_wrong": 1}, "distinct": 3}]
    again = DifficultyBook(IDS)
    assert again.load(log) == 1 and again.pooled("p00") == (2, 4, 1)


def test_shelves_partition_the_indices_in_id_order():
    book = DifficultyBook(IDS)
    book.record("p03", 8, 8, 1)
    book.record("p01", 0, 8, 1)
    book.record("p19", 4, 8, 1)
    shelves = book.shelves()
    assert shelves["known"] == [3] and shelves["cliff"] == [1] and shelves["frontier"] == [19]
    assert sorted(i for v in shelves.values() for i in v) == list(range(20))


def test_expected_fuel_and_histogram_and_summary():
    cases = [{"id": cid, "category": "substitute" if i < 10 else "join", "tier": "T1", "mode": "ex"}
             for i, cid in enumerate(IDS)]
    book = DifficultyBook(IDS, pool="latest")
    for i, cid in enumerate(IDS[:16]):
        book.record(cid, i % 9, 8, 1)                     # k = 0..8, 0..6
    assert book.expected_fuel([0]) == 0.0                 # k = 0
    assert book.expected_fuel([4]) == pytest.approx(1.75)   # k = 4 of 8
    hist = book.histogram()
    assert sum(hist.values()) == 16 and hist[0] == 2 and hist[8] == 1
    s = book.summary(cases)
    assert s["prompts"] == 20 and s["observed"] == 16
    assert s["shelves"]["unprobed"] == 4
    assert s["by_category"]["join"]["unprobed"] == 4
    assert s["by_category"]["substitute"]["k_sum"] == sum(i % 9 for i in range(10))
    assert s["by_tier"]["T1"]["G_sum"] == 16 * 8


# --------------------------------------------------------------- planner

def _pools(frontier, known=(), cliff=()):
    return {"frontier": list(frontier), "known": list(known), "cliff": list(cliff)}


def test_chunks_are_distinct_and_frontier_only_when_the_other_rates_are_zero():
    planner = ChunkPlanner(_pools(range(0, 30), range(30, 40), range(40, 45)), batch_size=8,
                           anchor_frac=0.0, cliff_frac=0.0, seed=1)
    seen = collections.Counter()
    for _ in range(60):
        chunk = planner.chunk()
        assert len(chunk) == 8 and len(set(chunk)) == 8
        assert all(i < 30 for i in chunk)
        seen.update(chunk)
    assert set(seen) == set(range(30))
    assert planner.drawn == {"frontier": 480}


def test_each_frontier_prompt_is_served_once_per_pass():
    planner = ChunkPlanner(_pools(range(24)), batch_size=8, anchor_frac=0, cliff_frac=0, seed=3)
    first_pass = [i for _ in range(3) for i in planner.chunk()]
    assert sorted(first_pass) == list(range(24))
    assert planner.passes["frontier"] == 0
    planner.chunk()
    assert planner.passes["frontier"] == 1


def test_anchor_and_cliff_rates_are_honoured_in_the_long_run():
    planner = ChunkPlanner(_pools(range(0, 200), range(200, 300), range(300, 400)), batch_size=8,
                           anchor_frac=0.15, cliff_frac=0.05, seed=11)
    for _ in range(2000):
        planner.chunk()
    n = 16000
    assert abs(planner.drawn["known"] / n - 0.15) < 0.015
    assert abs(planner.drawn["cliff"] / n - 0.05) < 0.01
    assert planner.starved == 0


def test_a_small_frontier_falls_back_to_the_other_shelves_never_duplicating():
    """Four frontier prompts cannot fill a chunk of eight: the rest comes from
    the known shelf, then the cliff, with no repeats inside the chunk."""
    planner = ChunkPlanner(_pools(range(4), range(4, 7), range(7, 9)), batch_size=8,
                           anchor_frac=0, cliff_frac=0, seed=5)
    chunk = planner.chunk()
    assert len(set(chunk)) == 8 and set(range(4)) <= set(chunk)
    assert planner.starved == 0
    assert planner.drawn["frontier"] == 4 and planner.drawn["known"] == 3 and planner.drawn["cliff"] == 1


def test_starvation_completes_the_chunk_from_the_universe():
    planner = ChunkPlanner(_pools(range(3)), batch_size=8, anchor_frac=0, cliff_frac=0, seed=5,
                           universe=range(50))
    chunk = planner.chunk()
    assert len(chunk) == 8 and len(set(chunk)) == 8
    assert planner.starved == 5 and planner.drawn["starved"] == 5
    tiny = ChunkPlanner(_pools(range(3)), batch_size=8, anchor_frac=0, cliff_frac=0, seed=5)
    assert len(tiny.chunk()) == 3                   # nothing else exists at all


def test_refresh_keeps_the_unvisited_remainder_of_the_pass():
    planner = ChunkPlanner(_pools(range(16)), batch_size=4, anchor_frac=0, cliff_frac=0, seed=2)
    first = planner.chunk()
    remaining = planner.cycles["frontier"][planner.cursor["frontier"]:]
    assert len(remaining) == 12
    # 3 of the remaining prompts get solved, 4 new ones join the frontier
    solved = set(remaining[:3])
    new_frontier = [i for i in range(16) if i not in solved] + [100, 101, 102, 103]
    planner.set_pools(_pools(new_frontier, known=sorted(solved)))
    cycle = planner.cycles["frontier"]
    assert cycle[:9] == [i for i in remaining if i not in solved]        # order kept
    assert set(cycle[9:]) == set(first) | {100, 101, 102, 103}           # newcomers behind
    assert planner.refreshes == 1 and planner.cursor["frontier"] == 0


def test_planner_is_deterministic_under_a_seed():
    a = ChunkPlanner(_pools(range(50), range(50, 60)), batch_size=8, seed=9)
    b = ChunkPlanner(_pools(range(50), range(50, 60)), batch_size=8, seed=9)
    assert [a.chunk() for _ in range(10)] == [b.chunk() for _ in range(10)]


def test_planner_validates_its_rates_and_size():
    with pytest.raises(ValueError):
        ChunkPlanner(_pools(range(4)), batch_size=0)
    with pytest.raises(ValueError):
        ChunkPlanner(_pools(range(4)), batch_size=2, anchor_frac=0.8, cliff_frac=0.3)
    with pytest.raises(ValueError):
        ChunkPlanner(_pools(range(4)), batch_size=2, anchor_frac=-0.1)


def test_merge_unprobed_policies():
    shelves = {"cliff": [1], "frontier": [2], "known": [3], "unprobed": [4, 5]}
    assert merge_unprobed(shelves, "frontier")["frontier"] == [2, 4, 5]
    assert merge_unprobed(shelves, "known")["known"] == [3, 4, 5]
    assert merge_unprobed(shelves, "cliff")["cliff"] == [1, 4, 5]
    skipped = merge_unprobed(shelves, "skip")
    assert skipped == {"frontier": [2], "known": [3], "cliff": [1]}
    with pytest.raises(ValueError):
        merge_unprobed(shelves, "elsewhere")


# ------------------------------------------------------- select_frontier

def _cases():
    return [{"id": cid, "category": "substitute" if i % 2 else "join", "tier": "T%d" % (i % 3 + 1),
             "mode": "ex", "instruction": "do %d" % i, "buffer_before": "a\n", "buffer_after": "b\n",
             "gold_command": "%s/a/b/", "cursor": [1, 1]} for i, cid in enumerate(IDS)]


def test_anchor_count_makes_the_requested_share_of_the_file():
    import random
    known = list(range(100))
    anchors = select_frontier.anchors_for(85, known, 0.15, random.Random(0))
    assert len(anchors) == 15 and anchors == sorted(anchors)
    assert select_frontier.anchors_for(85, known, 0.0, random.Random(0)) == []
    assert select_frontier.anchors_for(85, [], 0.5, random.Random(0)) == []
    assert len(select_frontier.anchors_for(10, known, 1.0, random.Random(0))) == 100
    assert len(select_frontier.anchors_for(10, list(range(3)), 0.5, random.Random(0))) == 3


def test_shelve_annotates_rows_and_builds_the_train_file():
    cases = _cases()
    book = DifficultyBook(IDS, pool="latest")
    for i, cid in enumerate(IDS[:15]):
        book.record(cid, (0, 4, 8)[i % 3], 8, 0)          # 5 cliff, 5 frontier, 5 known; 5 unprobed
    rows, train, anchors = select_frontier.shelve(cases, book, anchor_frac=0.5, unprobed="frontier", seed=1)
    assert {r["shelf"] for r in rows["cliff"]} == {"cliff"} and len(rows["cliff"]) == 5
    assert rows["frontier"][0]["k_hat"] == 4 and rows["frontier"][0]["g_hat"] == 8
    assert len(anchors) == 5                                # 0.5 of the file: 5 anchors for 5 frontier
    assert [r["shelf"] for r in train] == ["frontier"] * 5 + ["unprobed"] * 5 + ["anchor"] * 5
    assert all(set(c) <= set(r) for c, r in zip(cases, [next(x for x in train if x["id"] == c["id"])
                                                          for c in cases if c["id"] in {t["id"] for t in train}]))
    _, dropped, _ = select_frontier.shelve(cases, book, anchor_frac=0.0, unprobed="drop", seed=1)
    assert [r["shelf"] for r in dropped] == ["frontier"] * 5


def test_select_frontier_cli_writes_train_cliff_and_summary(tmp_path):
    cases = tmp_path / "cases.jsonl"
    _write(cases, _cases())
    probe = tmp_path / "probe.jsonl"
    _write(probe, [{"id": cid, "k": (0, 4, 8)[i % 3], "G": 8, "step": 0, "source": "probe"}
                   for i, cid in enumerate(IDS)])
    later = tmp_path / "log.jsonl"
    _write(later, [{"id": "p01", "k": 8, "G": 8, "step": 50, "source": "train"},   # frontier -> known? (last:2 says no)
                   {"id": "p00", "k": 2, "G": 8, "step": 50, "source": "train"}])  # cliff -> frontier
    out, cliff, summary = tmp_path / "train.jsonl", tmp_path / "cliff.jsonl", tmp_path / "s.md"
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "select_frontier.py"),
                        "--cases", str(cases), "--difficulty", str(probe), str(later),
                        "--out", str(out), "--cliff-out", str(cliff), "--summary", str(summary),
                        "--anchor-frac", "0.2", "--pool", "last:2"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    train = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    shelves = collections.Counter(r["shelf"] for r in train)
    # probe: 7 cliff (i%3==0), 7 frontier, 6 known; the log moves p00 to the
    # frontier (2 of 16 pooled) and keeps p01 there (12 of 16 pooled)
    assert shelves["frontier"] == 8 and shelves["unprobed"] == 0
    assert shelves["anchor"] == 2                            # 0.2 of the file
    assert {r["id"] for r in train if r["shelf"] == "anchor"} <= {IDS[i] for i in range(20) if i % 3 == 2}
    assert [json.loads(l)["id"] for l in cliff.read_text(encoding="utf-8").splitlines()] == [
        IDS[i] for i in range(20) if i % 3 == 0 and i != 0]
    text = summary.read_text(encoding="utf-8")
    assert "| frontier | 8 |" in text and "## By category" in text and "anchor share" in text
    assert "1.75 for a prompt at p = 1/2" in text
    for r in train:                                          # still §5 rows the trainers can read
        assert {"id", "instruction", "buffer_before", "buffer_after", "gold_command", "mode"} <= set(r)


def test_select_frontier_help_needs_no_torch():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "select_frontier.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--anchor-frac" in p.stdout and "--pool" in p.stdout


# ---------------------------------------------------------------- probe

def test_probe_trims_completions_like_the_trainer_and_resumes_by_id(tmp_path):
    probe = pytest.importorskip("probe_difficulty")
    assert probe.trim_completion([5, 6, 7, 0, 0], eos_id=7, pad_id=0) == [5, 6, 7]
    assert probe.trim_completion([5, 6, 0, 0], eos_id=7, pad_id=0) == [5, 6]
    out = tmp_path / "d.jsonl"
    assert probe.already_probed(out) == set()
    _write(out, [{"id": "a", "k": 1, "G": 8}, {"id": "b", "k": 2, "G": 8}])
    assert probe.already_probed(out) == {"a", "b"}


def test_probe_observe_scores_a_group_with_the_trainers_states(monkeypatch):
    probe = pytest.importorskip("probe_difficulty")
    import train_grpo

    class Tok:
        def decode(self, ids, skip_special_tokens=True):
            return {(1,): "ex: %s/a/b/", (2,): "ex: %s/a/c/", (3,): "ex: %s/x/", (4,): "nope"}[tuple(ids)]

    monkeypatch.setattr(probe, "execution_state",
                        lambda case, mode, command, timeout: {"%s/a/b/": "match", "%s/a/c/": "valid_wrong",
                                                              "%s/x/": "invalid"}[command])
    EOS = 9
    groups = [[1, EOS], [1, EOS], [2, EOS], [3, EOS], [4, EOS], [4]]       # last: truncated
    case = {"id": "c1", "category": "substitute", "tier": "T1", "mode": "ex",
            "buffer_before": "a\n", "buffer_after": "b\n", "cursor": [1, 1]}

    class Pool:
        def map(self, fn, keys):
            return [fn(k) for k in keys]

    row = probe.observe(case, groups, Tok(), EOS, 5.0, Pool())
    assert (row["k"], row["G"], row["shelf"]) == (2, 6, "frontier")
    assert row["p_hat"] == pytest.approx(2 / 6, abs=1e-4)          # rounded to 4 places in the file
    assert row["states"] == {"match": 2, "valid_wrong": 1, "invalid": 1, "parse": 1, "truncated": 1}
    assert row["distinct"] == 5 and row["commands"][0] == {"mode": "ex", "command": "%s/a/b/",
                                                           "state": "match", "n": 2}
    assert row["fuel"] == pytest.approx(difficulty.fuel(2, 6), abs=1e-4)
    assert set(row["states"]) <= set(train_grpo.REWARD_STATES)


def test_probe_pause_request_consumes_the_stop_file_and_names_its_marker(tmp_path):
    probe = pytest.importorskip("probe_difficulty")
    stop = tmp_path / "out" / "STOP"
    stop.parent.mkdir()
    pause = probe.PauseRequest(stop)
    assert pause.check() is None and pause.marker() == tmp_path / "out" / "PAUSED"
    stop.write_text("")
    assert pause.check() == "STOP file" and not stop.exists()      # consumed
    assert pause.check() == "STOP file"                            # sticky
    assert probe.PauseRequest(None).marker() is None and probe.PauseRequest(None).check() is None
    assert probe.PAUSED_RC == 3


def test_probe_help_needs_no_torch():
    p = subprocess.run([sys.executable, str(REPO / "scripts" / "probe_difficulty.py"), "--help"],
                       capture_output=True, text=True)
    assert p.returncode == 0 and "--num-generations" in p.stdout and "--ids" in p.stdout
