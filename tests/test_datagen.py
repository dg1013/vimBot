"""Tests for the Week-1 data tooling: taxonomy, scope screens, the
validate/ingest pipeline (real vim, like the executor tests), resumability,
dedup hashing, and split-leakage guarantees. No network anywhere.

Run:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from generate_data import (  # noqa: E402
    Corpus,
    candidate_id,
    instruction_token_hash,
    load_taxonomy,
    norm_buffer,
    process_candidate,
    screen_ex,
    screen_keys,
    screen_keys_notation,
)
from build_splits import (  # noqa: E402
    allocate,
    build_components,
    norm_command,
    stratum_of,
    verify_no_leakage,
)

TAXONOMY_PATH = REPO / "data" / "taxonomy.yaml"
TAXONOMY = load_taxonomy(TAXONOMY_PATH) if TAXONOMY_PATH.exists() else None


def _categories():
    if TAXONOMY is None:
        pytest.skip(f"{TAXONOMY_PATH} not present")
    return TAXONOMY["categories"]


# ------------------------------------------------------------------- taxonomy

def test_taxonomy_shape():
    # 15 original + visual + visual-block (admitted 2026-08-18; added to the
    # taxonomy 2026-09-03 when the Coverage Ledger was found still labelling
    # them 'outside taxonomy v1')
    categories = _categories()
    assert len(categories) == 17
    assert set(TAXONOMY["tiers"]) == {"T1", "T2", "T3"}
    for name, cat in categories.items():
        assert cat["mode"] in ("ex", "keys", "both"), name
        assert cat["description"].strip(), name
        assert set(cat["quotas"]) == {"T1", "T2", "T3"}, name
        for tier, quota in cat["quotas"].items():
            assert isinstance(quota, int) and quota > 0, (name, tier)


def test_taxonomy_quota_total_in_target_range():
    total = sum(sum(c["quotas"].values()) for c in _categories().values())
    assert 5000 - 500 <= total <= 20000


# ---------------------------------------------------------------- scope: ex

@pytest.mark.parametrize("cmd", [
    "%s/cat/dog/g", "2,3s/x/y/g", "g/^$/d", "v/keep/d", "1m$", "2t2",
    "sort n", "sort u", ":2,4j", "%s/\\s\\+$//e", "g/^/m0", "retab!",
    "2,5>", "g/foo/s//bar/g", "$put ='x'", "%s/\\(\\w\\+\\) \\(\\w\\+\\)/\\2 \\1/",
    "g/^<<<<<<</.,/^=======/d", "/start/,/end/d", "?back?,.j",
])
def test_ex_in_scope_commands_pass(cmd):
    assert screen_ex(cmd) is None, cmd


@pytest.mark.parametrize("cmd", [
    "w", "w /tmp/exfil", "wq", "q!", "x", ":wq!", "e /etc/passwd",
    "r /etc/passwd", "sp other.txt", "normal dd", "norm! x", "call foo()",
    "while 1 | endwhile", "g/x/w", "!ls", "s/a/b/ | wq", "bn", "tabnew",
    "so script.vim", "let g:x=1",
])
def test_ex_forbidden_commands_screened(cmd):
    assert screen_ex(cmd) is not None, cmd


# --------------------------------------------------------------- scope: keys

@pytest.mark.parametrize("cmd", [
    "dd", "3dd", "ci\"error<Esc>", "gUiw", "<C-a>", "/error<CR>dd",
    "oworld<Esc>", "A;<Esc>", "3rx", "df,", "dap", "xj.", "J",
])
def test_keys_in_scope_commands_pass(cmd):
    assert screen_keys(cmd) is None, cmd


@pytest.mark.parametrize("cmd", [
    "@a", "ZZ", "ZQ", ":wq<CR>", ":w /tmp/x<CR>", "Qi",
    # expression register: evaluates vimscript from inside insert mode
    "i<C-r>=writefile(['x'], '/tmp/x')<CR>",
])
def test_keys_forbidden_commands_screened(cmd):
    assert screen_keys(cmd) is not None, cmd


@pytest.mark.parametrize("cmd", [
    # Covered scope since 2026-08-18 (data/command_coverage.md). None of these
    # executes anything: a register stores text, `q` records keys. Every path
    # that EXECUTES a register still runs through `@`, screened above.
    "qqddq", "2qad", "\"ayy", "vd", "Vd", "3Vd", "<C-v>jjd",
])
def test_keys_newly_covered_commands_are_allowed(cmd):
    assert screen_keys(cmd) is None, cmd


@pytest.mark.parametrize("cmd", ["dd", "ci\"x<Esc>", "<C-a>", "a<lt>b<Esc>",
                                 "/foo<CR>x", "o<Tab>indented<Esc>"])
def test_keys_notation_supported(cmd):
    assert screen_keys_notation(cmd) is None, cmd


@pytest.mark.parametrize("cmd", ["<F5>", "d<Leader>x", "<M-a>", "<S-Tab>"])
def test_keys_notation_unsupported(cmd):
    assert screen_keys_notation(cmd) is not None, cmd


# ------------------------------------------- ex fragments anywhere in a keys cmd
#
# The screen used to check only a leading ':'...<CR> fragment, so an ex
# command placed after a normal-mode edit, or after the first <CR>, slipped
# through entirely. Regression cases for that gap.

@pytest.mark.parametrize("cmd", [
    ":set write<CR>:w! /tmp/x<CR>",     # second fragment carried the write
    "dd:set write<CR>:w! /tmp/x<CR>",   # command did not start with ':'
    "dd:e /etc/passwd<CR>",
    "x:w /tmp/x<CR>",
    "/foo<CR>:w! /tmp/x<CR>",
])
def test_ex_fragment_anywhere_in_keys_is_screened(cmd):
    assert screen_keys(cmd) is not None, cmd


def test_ex_fragments_from_keys_finds_every_span():
    from scope import ex_fragments_from_keys
    assert ex_fragments_from_keys("dd:set write<CR>:w! /x<CR>") == [
        "set write", "w! /x",
    ]
    assert ex_fragments_from_keys("dd") == []
    assert ex_fragments_from_keys(":") == [""]     # lone ':' must not crash


# ------------------------------------------------------------------- pipeline

def make_candidate(**over):
    cand = {
        "category": "substitute", "tier": "T1", "mode": "ex",
        "buffer_before": "the cat sat\na cat too",
        "command": "%s/cat/dog/g",
        "buffer_after": "the dog sat\na dog too",
        "instructions": ["Replace every cat with dog",
                         "change all occurrences of 'cat' to 'dog'"],
        "domain": "prose",
    }
    cand.update(over)
    return cand


def run_one(tmp_path, cand, corpus=None):
    corpus = corpus or Corpus()
    out = tmp_path / "cases.jsonl"
    quar = tmp_path / "quarantine.jsonl"
    kept, reason = process_candidate(
        cand, corpus, _categories(), out, quar, timeout=5.0, vim_bin="vim",
    )
    return kept, reason, out, quar, corpus


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def test_good_candidate_kept_with_paraphrase_expansion(tmp_path):
    kept, reason, out, _, _ = run_one(tmp_path, make_candidate())
    assert reason is None and kept == 2
    rows = read_jsonl(out)
    assert [r["id"] for r in rows] == ["substitute-T1-001a", "substitute-T1-001b"]
    for r in rows:
        assert r["template_id"] == r["paraphrase_group"] == "substitute-T1-001"
        assert r["buffer_after"] == "the dog sat\na dog too"
        assert r["teacher"]


def test_good_keys_candidate(tmp_path):
    cand = make_candidate(
        category="line-ops", mode="keys", command="dd", cursor=[2, 1],
        buffer_before="keep\ndelete me\nalso keep",
        buffer_after="keep\nalso keep",
        instructions=["Delete the current line"],
    )
    kept, reason, out, _, _ = run_one(tmp_path, cand)
    assert reason is None and kept == 1
    assert read_jsonl(out)[0]["id"] == "line-ops-T1-001a"


def test_mismatch_rejected_with_executed_buffer(tmp_path):
    cand = make_candidate(buffer_after="wrong claim")
    kept, reason, _, quar, _ = run_one(tmp_path, cand)
    assert (kept, reason) == (0, "mismatch")
    row = read_jsonl(quar)[0]
    assert row["reject_reason"] == "mismatch"
    assert "dog" in row["buffer_executed"]  # what vim actually produced


def test_noop_rejected(tmp_path):
    cand = make_candidate(
        command="%s/zebra/dog/ge",
        buffer_after="the cat sat\na cat too",
    )
    _, reason, _, quar, _ = run_one(tmp_path, cand)
    assert reason == "noop"


def test_invalid_command_rejected(tmp_path):
    cand = make_candidate(command="%s/unclosed")
    _, reason, _, _, _ = run_one(tmp_path, cand)
    assert reason == "invalid"


def test_excluded_scope_rejected_without_execution(tmp_path):
    cand = make_candidate(command=":wq")
    _, reason, _, quar, _ = run_one(tmp_path, cand)
    assert reason == "excluded_scope"
    assert read_jsonl(quar)[0]["buffer_executed"] is None


def test_bad_notation_rejected(tmp_path):
    cand = make_candidate(
        category="line-ops", mode="keys", command="<F5>dd",
        buffer_before="a\nb", buffer_after="b",
        instructions=["delete line via F5"],
    )
    _, reason, _, _, _ = run_one(tmp_path, cand)
    assert reason == "bad_keys_notation"


def test_exact_duplicate_rejected(tmp_path):
    kept, _, out, quar, corpus = run_one(tmp_path, make_candidate())
    assert kept == 2
    dup = make_candidate(command="%s/cat/dog/g ")  # same content, new cand id
    kept2, reason = process_candidate(
        dup, corpus, _categories(), out, quar, timeout=5.0, vim_bin="vim",
    )
    assert (kept2, reason) == (0, "dup")


def test_near_dup_hash_is_order_insensitive():
    a = instruction_token_hash("Delete the blank lines")
    b = instruction_token_hash("the BLANK lines -- delete!")
    c = instruction_token_hash("delete every comment line")
    assert a == b and a != c


def test_resumability_skips_processed_and_advances_seq(tmp_path):
    kept, _, out, quar, _ = run_one(tmp_path, make_candidate())
    cand2 = make_candidate(
        buffer_before="one cat here", buffer_after="one dog here",
        instructions=["swap cat for dog on this line"],
    )
    corpus2 = Corpus()
    corpus2.load_existing(out, quar)
    assert candidate_id(make_candidate()) in corpus2.processed
    assert corpus2.next_seq[("substitute", "T1")] == 2
    kept2, reason2 = process_candidate(
        cand2, corpus2, _categories(), out, quar, timeout=5.0, vim_bin="vim",
    )
    assert reason2 is None
    rows = read_jsonl(out)
    assert rows[-1]["id"] == "substitute-T1-002a"


# --------------------------------------------------------------------- splits

def synthetic_corpus():
    cases = []
    cats = ["substitute", "global", "line-ops"]
    for t in range(30):
        cat = cats[t % 3]
        tier = f"T{(t % 3) + 1}"
        template = f"{cat}-{tier}-{t:03d}"
        for p, letter in enumerate("ab"):
            cases.append({
                "id": f"{template}{letter}",
                "category": cat, "tier": tier,
                "instruction": f"instruction variant {p} for scenario {t}",
                "mode": "ex",
                "buffer_before": f"line one {t}\nline two {t}",
                "gold_command": f"%s/{t}/x/",
                "buffer_after": f"line one x\nline two {t}",
                "template_id": template, "paraphrase_group": template,
            })
    return cases


def test_split_components_group_paraphrases_and_shared_buffers():
    cases = synthetic_corpus()
    components = build_components(cases)
    assert all(len(c) >= 2 for c in components)  # paraphrase pairs never split
    by_id = {cases[i]["id"]: comp for comp in components for i in comp}
    a, b = by_id["substitute-T1-000a"], by_id["substitute-T1-000b"]
    assert a is b


def test_split_allocation_has_no_leakage():
    cases = synthetic_corpus()
    components = build_components(cases)
    splits = allocate(cases, components, test_min=300, test_max=500,
                      val_frac=0.2, seed=13, smoke=True)
    assert sorted(i for s in splits.values() for i in s) == list(range(len(cases)))
    assert verify_no_leakage(cases, splits) == []
    assert all(splits[s] for s in ("train", "val", "test"))


def test_split_leakage_detector_fires_on_manufactured_leak():
    cases = synthetic_corpus()
    n = len(cases)
    splits = {"train": list(range(0, n, 2)), "val": [], "test": list(range(1, n, 2))}
    # paraphrase pairs a/b are adjacent, so this straddles every template
    assert verify_no_leakage(cases, splits)


def _bare_case(id_, instruction, buffer_before, command, mode="ex", **extra):
    """A mined-style row: no template_id, no tier unless given."""
    return {"id": id_, "category": "line-ops", "instruction": instruction,
            "mode": mode, "buffer_before": buffer_before,
            "gold_command": command, "buffer_after": "x", **extra}


def test_components_survive_rows_without_template_id():
    # v3/v4 rows carry no template_id at all — must not crash (KeyError)
    cases = [
        _bare_case("a1", "delete the first line", "one\ntwo", "1d"),
        _bare_case("a2", "remove line one", "alpha\nbeta", "1d"),
    ]
    assert build_components(cases)


def test_command_key_merges_reworded_instructions():
    # round-three audit: same command, deliberately reworded instructions,
    # different buffers — the token hash is blind to this, the command key
    # must merge it into one component
    cases = [
        _bare_case("c1", "increment every number in the file",
                   "3 apples\n7 pears", "VGg<C-a>", mode="keys"),
        _bare_case("c2", "bump all the counts up by one",
                   "score 10\nscore 22", "VGg<C-a>", mode="keys"),
        _bare_case("c3", "delete the trailing blank rows",
                   "text\n\n\n", "v/./d", mode="ex"),
    ]
    comps = build_components(cases)
    by_id = {cases[i]["id"]: comp for comp in comps for i in comp}
    assert by_id["c1"] is by_id["c2"]
    assert by_id["c3"] is not by_id["c1"]
    splits = {"train": [0], "val": [], "test": [1, 2]}
    assert any("command" in v for v in verify_no_leakage(cases, splits))


def test_norm_command_forms():
    assert norm_command(" :%s/a/b/ ", "ex") == "%s/a/b/"      # colon + space
    assert norm_command("<C-a>", "keys") == norm_command("<C-A>", "keys")
    assert norm_command(":m2\n", "keys") == ":m2\n"           # keystrokes kept
    assert norm_command("dd ", "keys") != norm_command("dd", "keys")


def test_paraphrase_of_links_cases():
    cases = [
        _bare_case("p1", "indent the block", "a\nb", "1,2>"),
        _bare_case("p2", "shift those rows right", "c\nd", "3,4>",
                   paraphrase_of="p1"),
    ]
    comps = build_components(cases)
    by_id = {cases[i]["id"]: comp for comp in comps for i in comp}
    assert by_id["p1"] is by_id["p2"]


def test_blank_tier_becomes_pseudo_tier():
    # 206 mined rows carry tier="" — one "T?" stratum, not T1, not ""
    assert stratum_of(_bare_case("t1", "x", "y", "dd", tier=""))[2] == "T?"
    assert stratum_of(_bare_case("t2", "x", "y", "dd"))[2] == "T?"
    assert stratum_of(_bare_case("t3", "x", "y", "dd", tier="T2"))[2] == "T2"


# ------------------------------------------------- provenance passthrough

def test_provenance_fields_pass_through_to_kept_rows(tmp_path):
    cand = make_candidate()
    cand.update({"source": "deep-regex", "source_corpus": ["NL-RX-Synth"],
                 "source_index": ["NL-RX-Synth:7"], "source_regex": ".*cat.*",
                 "source_license": "MIT"})
    kept, reason, out, _, _ = run_one(tmp_path, cand)
    assert reason is None and kept
    row = read_jsonl(out)[0]
    assert row["source"] == "deep-regex"
    assert row["source_regex"] == ".*cat.*"
    assert row["source_license"] == "MIT"


def test_rows_without_provenance_stay_clean(tmp_path):
    kept, _, out, _, _ = run_one(tmp_path, make_candidate())
    assert kept and "source" not in read_jsonl(out)[0]


# --------------------------------------------- feature-bucket stratification

from pattern_features import case_bucket, feature_bucket  # noqa: E402
from build_splits import coverage_report  # noqa: E402


@pytest.mark.parametrize("cmd,mode,bucket", [
    ("dd", "keys", "no-pattern"),
    ("diw", "keys", "no-pattern"),
    (r"%s/\(\w\+\) \(\w\+\)/\2 \1/", "ex", "capture"),
    (r"g/^\%(foo\|bar\)$/d", "ex", "alternation"),
    (r"g/cat\|dog/d", "ex", "alternation"),
    (r"v/x\{2,}/d", "ex", "counted-rep"),
    (r"g/\<word\>/d", "ex", "boundary"),
    (r"%s/[^a-z]//g", "ex", "negclass"),
    (r"g/[0-9]/d", "ex", "class"),
    (r"%s/ \+$//", "ex", "quantifier"),
    ("g/DEBUG/d", "ex", "literal"),
    ("/target<CR>dw", "keys", "literal"),
])
def test_feature_bucket_ladder(cmd, mode, bucket):
    assert feature_bucket(cmd, mode) == bucket


def two_source_corpus():
    """A big imported slice plus a small hand-authored slice with a rare
    bucket (capture groups) spread across many components."""
    cases = []
    for t in range(60):
        template = f"global-T2-{t:03d}"
        for p, letter in enumerate("ab"):
            cases.append({
                "id": f"{template}{letter}", "category": "global",
                "tier": "T2",
                "instruction": f"delete matching lines variant {p} number {t}",
                "mode": "ex", "buffer_before": f"keep {t}\ndrop {t}",
                "gold_command": f"g/[0-9]{t}/d",
                "buffer_after": f"keep {t}",
                "template_id": template, "paraphrase_group": template,
                "source": "deep-regex",
            })
    for t in range(20):
        template = f"substitute-T3-{t:03d}"
        for p, letter in enumerate("ab"):
            cases.append({
                "id": f"{template}{letter}", "category": "substitute",
                "tier": "T3",
                "instruction": f"swap the two words around variant {p} case {t}",
                "mode": "ex", "buffer_before": f"alpha beta {t}",
                # command varies per template: identical commands would merge
                # all templates into one leakage component via the command key
                "gold_command": rf"%s/\(alpha\) \(beta {t}\)/\2 \1/",
                "buffer_after": f"beta {t} alpha",
                "template_id": template, "paraphrase_group": template,
            })
    return cases


def test_coverage_repair_rescues_multi_row_component_sources():
    """The greedy fill takes singletons first, so a source living only in
    multi-row components (the emulator sources on v3+v4) never reaches
    test/val without the repair pass."""
    from build_splits import repair_coverage
    cases = []
    # 60 singleton fillers spread over 20 strata, mirroring the real corpus:
    # per-stratum caps never bind, so test/val fill entirely from singletons
    for t in range(60):
        cases.append({
            "id": f"fill-{t:03d}", "category": f"cat{t % 20}", "tier": "T2",
            "instruction": f"delete rows matching pattern number {t}",
            "mode": "ex", "buffer_before": f"keep {t}\ndrop f{t}",
            "gold_command": f"g/f{t}/d", "buffer_after": f"keep {t}",
            "source": "deep-regex",
        })
    for t in range(6):  # 6 components of 3 rows each, linked by buffer
        for p in range(3):
            cases.append({
                "id": f"emu-{t}{p}", "category": "line-ops", "tier": "T1",
                # s{t}/w{p} prefixes keep the order-insensitive token hash
                # from colliding across (t, p) swaps
                "instruction": f"emulator scenario s{t} wording w{p}",
                "mode": "ex", "buffer_before": f"emu shared buffer {t}",
                "gold_command": f"s/e{t}{p}/x/", "buffer_after": "x",
                "source": "emulator-x",
            })
    components = build_components(cases)
    splits = allocate(cases, components, test_min=300, test_max=500,
                      val_frac=0.15, seed=13, smoke=True)
    moved = repair_coverage(cases, components, splits, test_max=500)
    assert moved > 0
    assert verify_no_leakage(cases, splits) == []
    src_table, _, violations = coverage_report(cases, splits)
    assert violations == []
    for split in ("train", "val", "test"):
        assert src_table["emulator-x"][split] > 0


def test_stratified_allocation_covers_sources_and_buckets_everywhere():
    cases = two_source_corpus()
    components = build_components(cases)
    splits = allocate(cases, components, test_min=300, test_max=500,
                      val_frac=0.15, seed=13, smoke=True)
    assert verify_no_leakage(cases, splits) == []
    src_table, bkt_table, violations = coverage_report(cases, splits)
    assert violations == []
    # the rare capture bucket and the small source appear in EVERY split
    for split in ("train", "val", "test"):
        assert bkt_table["capture"][split] > 0
        assert src_table["hand-authored"][split] > 0
        assert src_table["deep-regex"][split] > 0
    assert case_bucket(cases[0]) == "class"


# ------------------------- category coverage floor (2026-08-27)
#
# Tracking only source and bucket let a whole taxonomy category fall out of
# the test slice unnoticed: on the 5,141-case corpus `visual-block` had 62
# cases across 27 components and ZERO test rows, so its per-category
# exec-match was unmeasurable. Category is now a tracked group, and the
# repair aims at a small floor rather than bare presence -- one test row
# reports 0% or 100% and is noise, not a measurement.

def concentrated_category_corpus():
    """A category whose rows live only in multi-row components.

    Mirrors the real failure: the fill takes singletons first, so a category
    with no singleton components never gets dealt into test or val.
    """
    cases = []
    for t in range(90):                       # singleton filler, many strata
        cases.append({
            "id": f"fill-{t:03d}", "category": f"cat{t % 15}", "tier": "T2",
            "instruction": f"remove the rows matching marker number {t}",
            "mode": "ex", "buffer_before": f"keep {t}\ndrop m{t}",
            "gold_command": f"g/m{t}/d", "buffer_after": f"keep {t}",
            "source": "filler",
        })
    for t in range(9):                        # 9 components of 4 rows each
        for p in range(4):
            cases.append({
                "id": f"blk-{t}{p}", "category": "visual-block", "tier": "T3",
                "instruction": f"column edit scenario s{t} wording w{p}",
                "mode": "keys", "buffer_before": f"col shared buffer {t}",
                "gold_command": f"<C-v>j{p}lx{t}", "buffer_after": "x",
                "source": "mined",
            })
    return cases


def test_category_is_a_tracked_coverage_group():
    from build_splits import _group_keys
    kinds = {k for k, _ in _group_keys(concentrated_category_corpus()[0])}
    assert "category" in kinds, \
        "a category can vanish from test unnoticed if it is not tracked"


def test_concentrated_category_reaches_the_test_floor():
    from build_splits import repair_coverage, _COVERAGE_FLOOR
    cases = concentrated_category_corpus()
    components = build_components(cases)
    splits = allocate(cases, components, test_min=300, test_max=500,
                      val_frac=0.15, seed=13, smoke=True)

    def n_in(split):
        return sum(1 for i in splits[split]
                   if cases[i]["category"] == "visual-block")

    before = n_in("test")
    moved = repair_coverage(cases, components, splits, test_max=500)
    assert moved > 0, "the repair did not fire"
    assert verify_no_leakage(cases, splits) == [], \
        "the repair moved a partial component and broke leakage"

    n = sum(1 for c in cases if c["category"] == "visual-block")
    share = len(splits["test"]) / len(cases)
    want = min(_COVERAGE_FLOOR, max(1, round(n * share)))
    assert n_in("test") >= want, (
        f"visual-block has {n_in('test')} test rows, floor is {want} "
        f"(was {before} before repair)")
    assert n_in("val") >= 1
    _, _, violations = coverage_report(cases, splits)
    assert violations == []


def test_floor_never_drags_a_group_above_its_proportional_share():
    """A rare group must not be topped up past what its size justifies."""
    from build_splits import _repair_targets
    cases = concentrated_category_corpus()
    components = build_components(cases)
    splits = allocate(cases, components, test_min=300, test_max=500,
                      val_frac=0.15, seed=13, smoke=True)
    total = len(cases)
    for kind, name, split, have, want in _repair_targets(cases, splits, components):
        n = sum(1 for c in cases
                if (c["category"] if kind == "category"
                    else c.get("source", "hand-authored")) == name) or total
        proportional = max(1, round(n * len(splits[split]) / total))
        assert want <= max(proportional, 1), (
            f"{kind} {name!r} targeted {want} in {split}, above its "
            f"proportional {proportional}")


# ------------------------------------------------- authored tier (2026-09-06)

def test_authored_draft_may_state_its_own_tier():
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts"))
    from ingest_handcrafted import draft_tier
    # stated wins, and is recorded as authored
    assert draft_tier({"tier": "T2", "mode": "keys", "command": "<C-v>jjlld"}) == ("T2", "authored")
    assert draft_tier({"tier": "t1", "mode": "keys", "command": "<C-v>jjd"}) == ("T1", "authored")
    # absent falls back to the heuristic, still marked heuristic
    assert draft_tier({"mode": "keys", "command": "Vjd"}) == ("T1", "heuristic")


def test_the_tier_heuristic_miscounts_blockwise_notation():
    # Why an authored tier exists: `<` and `>` of <C-v>/<Esc> count as shift
    # operators, so every blockwise command scores T3. Pinned so a future fix
    # to guess_tier is a deliberate change, not a silent one.
    import sys as _sys
    _sys.path.insert(0, str(REPO / "scripts"))
    from make_v3_emulator import guess_tier
    assert guess_tier("keys", "<C-v>jjd") == "T3"        # taxonomy calls this T1
    assert guess_tier("keys", "<C-v>jjIx<Esc>") == "T3"  # taxonomy calls this T1
    assert guess_tier("keys", "Vjd") == "T1"             # charwise is fine
