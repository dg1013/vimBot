"""The shipped split files, checked as data.

`tests/test_datagen.py` exercises the ALLOCATOR on a synthetic corpus, but
the files under `data/splits/` are frozen and are mutated afterwards by
`scripts/refresh_splits.py` — appends, refreshes and moves the allocator
never sees. Nothing checked those files until a leak got through: `--add-new`
guarded a new row's gold command against test alone, so eight quota-fill rows
joined train while six val rows already held their commands (found and
repaired 2026-09-06, `data/fixes/split_command_leak_2026-09-06.json`).

These tests are the standing check. They read the real files, so they fail on
a bad curation pass rather than on a bad allocator.

They compare on the ALLOCATOR's keys, not on raw strings. An earlier version
compared bytes, which is a weaker check than the invariant it asserts: two
buffers differing only by a final newline are one template to `norm_buffer`
and to `evaluate.py`, so the raw check passed while `1\n2\n3\n4\n5` sat in
test and `1\n2\n3\n4\n5\n` sat in train (2026-09-06). 2,419 of 7,512 rows
carry no final newline -- the mined corpora store none and the hand-authored
rows do -- so the gap runs along the seam where duplicate content is likeliest.
"""

import collections
import itertools
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data" / "splits"
NAMES = ("train", "val", "test")

sys.path.insert(0, str(REPO / "scripts"))

from refresh_splits import buffer_key, command_key   # noqa: E402
from generate_data import norm_instruction           # noqa: E402


def _rows(name):
    p = SPLITS / f"{name}.jsonl"
    if not p.exists():
        pytest.skip(f"{p} not present")
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


@pytest.fixture(scope="module")
def splits():
    return {n: _rows(n) for n in NAMES}


@pytest.fixture(scope="module")
def corpus():
    p = REPO / "data" / "combined_cases.jsonl"
    if not p.exists():
        pytest.skip("corpus not built")
    return {json.loads(l)["id"]: json.loads(l)
            for l in p.open(encoding="utf-8") if l.strip()}


def test_every_corpus_row_is_placed_exactly_once(splits, corpus):
    seen = collections.Counter(r["id"] for rows in splits.values() for r in rows)
    assert not [i for i, n in seen.items() if n > 1], "id in two splits or twice in one"
    assert set(seen) == set(corpus), (
        "unplaced: %s / not in corpus: %s"
        % (sorted(set(corpus) - set(seen))[:5], sorted(set(seen) - set(corpus))[:5]))


def test_split_rows_match_the_corpus(splits, corpus):
    stale = [r["id"] for rows in splits.values() for r in rows
             if any(r.get(f) != corpus[r["id"]].get(f)
                    for f in ("instruction", "gold_command", "buffer_before",
                              "buffer_after", "cursor", "mode"))]
    assert not stale, "split copy differs from the corpus: %s" % stale[:5]


@pytest.mark.parametrize("a, b", list(itertools.combinations(NAMES, 2)))
@pytest.mark.parametrize("key, label", [(buffer_key, "buffer template"),
                                        (command_key, "gold command")])
def test_no_field_spans_two_splits(splits, a, b, key, label):
    """Buffer templates are disjoint across splits (CLAUDE.md §7.3), and a
    gold command stays inside one split (DECISIONS.md 2026-09-06). Keyed the
    way the allocator keys them, so a trailing newline cannot hide a twin."""
    shared = {key(r) for r in splits[a]} & {key(r) for r in splits[b]}
    assert not shared, "%d %s(s) in both %s and %s, e.g. %r" % (
        len(shared), label, a, b, sorted(shared)[0])


def test_paraphrase_clusters_do_not_straddle_splits(splits):
    """Rows sharing (buffer, cursor, command) are paraphrases of one case;
    CLAUDE.md §7.3 keeps a cluster whole."""
    home = {}
    for name, rows in splits.items():
        for r in rows:
            home.setdefault(
                (buffer_key(r), tuple(r.get("cursor") or (1, 1)),
                 command_key(r)), set()).add(name)
    straddling = [k for k, v in home.items() if len(v) > 1]
    assert not straddling, "%d cluster(s) straddle splits" % len(straddling)


def test_instructions_are_unique_per_buffer(splits):
    """The §7.3 dedup key: (normalized instruction, buffer_before)."""
    key = collections.Counter(
        (norm_instruction(r["instruction"]), buffer_key(r))
        for rows in splits.values() for r in rows)
    dup = [k for k, n in key.items() if n > 1]
    assert not dup, "%d duplicate (instruction, buffer) pair(s)" % len(dup)


# ------------------------------------------------ the tool that moves rows

def _write(dirpath, name, rows):
    (dirpath / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture_splits(tmp_path):
    """train holds a row whose command val also holds — the 2026-09-06 leak."""
    def row(i, buf, cmd):
        return {"id": i, "instruction": "do " + i, "buffer_before": buf,
                "gold_command": cmd, "buffer_after": "x", "cursor": [1, 1],
                "mode": "ex"}
    train = [row("t-1", "a\n", "%s/a/b/"), row("t-2", "b\n", "3,8d")]
    val = [row("v-1", "c\n", "3,8d"), row("v-2", "d\n", "%s/x/y/")]
    test = [row("e-1", "e\n", "1d")]
    for name, rows in (("train", train), ("val", val), ("test", test)):
        _write(tmp_path, name, rows)
    return {r["id"]: r for r in train + val + test}


class _Args:
    def __init__(self, splits, to, dry_run=False):
        self.splits, self.to, self.dry_run = str(splits), to, dry_run


def test_move_repairs_a_command_that_spans_two_splits(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from refresh_splits import move_rows
    corpus = _fixture_splits(tmp_path)
    move_rows(_Args(tmp_path, "train"), ["v-1"], corpus)
    val = [json.loads(l) for l in (tmp_path / "val.jsonl").open() if l.strip()]
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").open() if l.strip()]
    assert [r["id"] for r in val] == ["v-2"]
    assert "v-1" in [r["id"] for r in train]
    assert not ({r["gold_command"] for r in train} & {r["gold_command"] for r in val})


def test_move_is_refused_when_it_would_create_a_leak(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from refresh_splits import move_rows
    corpus = _fixture_splits(tmp_path)
    # t-2 carries 3,8d, which val's v-1 also carries: moving t-2 to test would
    # put that command in both val and test instead of repairing anything
    with pytest.raises(SystemExit) as e:
        move_rows(_Args(tmp_path, "test"), ["t-2"], corpus)
    assert "gold command" in str(e.value)
    assert len([l for l in (tmp_path / "test.jsonl").open() if l.strip()]) == 1


def test_move_refuses_an_unknown_or_already_placed_id(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from refresh_splits import move_rows
    corpus = _fixture_splits(tmp_path)
    with pytest.raises(SystemExit):
        move_rows(_Args(tmp_path, "train"), ["nope"], corpus)
    with pytest.raises(SystemExit):
        move_rows(_Args(tmp_path, "train"), ["t-1"], corpus)


def test_move_is_refused_when_only_the_normalized_keys_collide(tmp_path):
    """The 2026-09-06 hole: test held `1\n2\n3` and train `1\n2\n3\n`, which
    are one template to the allocator and two strings to a byte comparison.

    A clean fixture on purpose -- `_fixture_splits` carries a deliberate
    command leak, which would refuse the move for the wrong reason.
    """
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from refresh_splits import move_rows

    def row(i, buf, cmd):
        return {"id": i, "instruction": "do " + i, "buffer_before": buf,
                "gold_command": cmd, "buffer_after": "x", "cursor": [1, 1],
                "mode": "ex"}
    train = [row("t-1", "1\n2\n3\n", "1d")]          # with the final newline
    val = [row("v-1", "c\n", "2d")]
    test = [row("e-1", "1\n2\n3", "3d")]              # without it
    for name, rows in (("train", train), ("val", val), ("test", test)):
        _write(tmp_path, name, rows)
    corpus = {r["id"]: r for r in train + val + test}
    # the byte comparison the guard used until 2026-09-06 sees nothing here,
    # which is what makes this a regression test rather than a tautology
    assert not ({r["buffer_before"] for r in train}
                & {r["buffer_before"] for r in test})
    with pytest.raises(SystemExit) as e:
        move_rows(_Args(tmp_path, "test"), ["v-1"], corpus)
    msg = str(e.value)
    assert "1 buffer(s) in both" in msg and "test" in msg and "train" in msg


def test_add_new_refuses_a_buffer_that_differs_only_by_a_trailing_newline(
        tmp_path, monkeypatch, capsys):
    """The append guard reads the same keys the move guard does."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import refresh_splits

    _fixture_splits(tmp_path)
    # test holds this template as "e\n"; the new row spells it without the
    # final newline, the way every mined corpus in this repo (repository) does
    unplaced = {"id": "n-1", "instruction": "brand new sentence",
                "buffer_before": "e", "gold_command": "9d",
                "buffer_after": "x", "cursor": [1, 1], "mode": "ex"}
    cases = tmp_path / "corpus.jsonl"
    rows = [json.loads(l) for name in NAMES
            for l in (tmp_path / f"{name}.jsonl").open() if l.strip()]
    cases.write_text("".join(json.dumps(r) + "\n" for r in rows + [unplaced]),
                     encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["refresh_splits.py", "--cases", str(cases),
                                      "--splits", str(tmp_path), "--add-new", "train"])
    refresh_splits.main()
    out = capsys.readouterr().out
    # different strings, one template: a raw comparison admits the row
    assert unplaced["buffer_before"] not in {r["buffer_before"] for r in rows}
    assert "refused n-1: buffer already in test" in out
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").open() if l.strip()]
    assert "n-1" not in [r["id"] for r in train]


def test_add_new_keys_a_command_by_mode(tmp_path, monkeypatch, capsys):
    """keys `dd` and ex `dd` are different commands; the allocator keys them
    apart, so the guard must not refuse one for the other."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import refresh_splits

    _fixture_splits(tmp_path)
    test = [json.loads(l) for l in (tmp_path / "test.jsonl").open() if l.strip()]
    test[0]["gold_command"] = "dd"
    test[0]["mode"] = "ex"
    _write(tmp_path, "test", test)
    unplaced = {"id": "n-2", "instruction": "an unrelated sentence",
                "buffer_before": "z\ny\n", "gold_command": "dd",
                "buffer_after": "x", "cursor": [1, 1], "mode": "keys"}
    cases = tmp_path / "corpus.jsonl"
    rows = [json.loads(l) for name in NAMES
            for l in (tmp_path / f"{name}.jsonl").open() if l.strip()]
    cases.write_text("".join(json.dumps(r) + "\n" for r in rows + [unplaced]),
                     encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["refresh_splits.py", "--cases", str(cases),
                                      "--splits", str(tmp_path), "--add-new", "train"])
    refresh_splits.main()
    assert "appended 1 rows" in capsys.readouterr().out


# ------------------------- the curation tool answers to the same invariant

def _curation_fixture():
    """index and home as apply_curation sees them: a train row and a test row
    holding one template in the two spellings the corpora use."""
    train = {"id": "t-1", "instruction": "delete the second line",
             "buffer_before": "1\n2\n3\n", "gold_command": "2d",
             "buffer_after": "1\n3\n", "cursor": [1, 1], "mode": "ex"}
    test = {"id": "e-1", "instruction": "blank every digit",
            "buffer_before": "1\n2\n3", "gold_command": "%s/\\d/0/g",
            "buffer_after": "0\n0\n0", "cursor": [1, 1], "mode": "ex"}
    other = {"id": "t-2", "instruction": "delete the last line",
             "buffer_before": "a\nb\n", "gold_command": "$d",
             "buffer_after": "a\n", "cursor": [1, 1], "mode": "ex"}
    index = {r["id"]: ("v3", "handcrafted_cases.jsonl", r)
             for r in (train, test, other)}
    home = {"t-1": "train", "e-1": "test", "t-2": "train"}
    return train, test, other, index, home


def test_curation_refuses_an_edit_that_crosses_a_split():
    """Giving a train row a test row's buffer was accepted until 2026-09-06."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from apply_curation import crosses_a_split
    train, test, _other, index, home = _curation_fixture()
    edited = dict(train, buffer_before=test["buffer_before"])
    why = crosses_a_split(edited, index, home)
    assert why and "buffer template" in why and "e-1" in why


def test_curation_sees_through_the_trailing_newline():
    """The two spellings are one template, so the raw comparison is not enough:
    train's `1\n2\n3\n` already collides with test's `1\n2\n3`."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from apply_curation import crosses_a_split
    train, test, _other, index, home = _curation_fixture()
    assert train["buffer_before"] != test["buffer_before"]      # not raw-equal
    why = crosses_a_split(train, index, home)
    assert why and "buffer template" in why


def test_curation_refuses_a_command_or_instruction_that_crosses_a_split():
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from apply_curation import crosses_a_split
    train, test, _other, index, home = _curation_fixture()
    train = dict(train, buffer_before="z\n")                   # clear the buffer clash
    index["t-1"] = ("v3", "handcrafted_cases.jsonl", train)
    assert "gold command" in crosses_a_split(
        dict(train, gold_command=test["gold_command"]), index, home)
    assert "instruction" in crosses_a_split(
        dict(train, instruction="  Blank   Every Digit  "), index, home)


def test_curation_allows_a_collision_inside_one_split():
    """A command may be shared with the split the row is already in -- that is
    what keeps a command inside one split, not what violates it."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from apply_curation import crosses_a_split
    train, _test, other, index, home = _curation_fixture()
    train = dict(train, buffer_before="z\n")
    index["t-1"] = ("v3", "handcrafted_cases.jsonl", train)
    assert crosses_a_split(dict(train, gold_command=other["gold_command"]),
                           index, home) is None


def test_curation_skips_the_check_for_an_unplaced_row():
    """A row in no split is guarded by --add-new when it is placed."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from apply_curation import crosses_a_split
    train, test, _other, index, home = _curation_fixture()
    del home["t-1"]
    assert crosses_a_split(dict(train, buffer_before=test["buffer_before"]),
                           index, home) is None


def test_curation_and_the_allocator_share_one_definition():
    """Three copies of `norm_buffer` is how the split leak survived review."""
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import apply_curation
    from generate_data import norm_buffer, norm_instruction
    assert apply_curation.norm("a\nb\n") == norm_buffer("a\nb\n")
    assert apply_curation.norm("a\nb") == apply_curation.norm("a\nb\n")
    assert apply_curation.norm(None) is None
    assert apply_curation.norm_ins is norm_instruction


def test_move_dry_run_writes_nothing(tmp_path):
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    from refresh_splits import move_rows
    corpus = _fixture_splits(tmp_path)
    before = (tmp_path / "val.jsonl").read_text(encoding="utf-8")
    move_rows(_Args(tmp_path, "train", dry_run=True), ["v-1"], corpus)
    assert (tmp_path / "val.jsonl").read_text(encoding="utf-8") == before
