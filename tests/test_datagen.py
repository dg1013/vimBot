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
    verify_no_leakage,
)

TAXONOMY = load_taxonomy(REPO / "data" / "taxonomy.yaml")
CATEGORIES = TAXONOMY["categories"]


# ------------------------------------------------------------------- taxonomy

def test_taxonomy_shape():
    assert len(CATEGORIES) == 15
    assert set(TAXONOMY["tiers"]) == {"T1", "T2", "T3"}
    for name, cat in CATEGORIES.items():
        assert cat["mode"] in ("ex", "keys", "both"), name
        assert cat["description"].strip(), name
        assert set(cat["quotas"]) == {"T1", "T2", "T3"}, name
        for tier, quota in cat["quotas"].items():
            assert isinstance(quota, int) and quota > 0, (name, tier)


def test_taxonomy_quota_total_in_target_range():
    total = sum(sum(c["quotas"].values()) for c in CATEGORIES.values())
    assert 5000 - 500 <= total <= 20000


# ---------------------------------------------------------------- scope: ex

@pytest.mark.parametrize("cmd", [
    "%s/cat/dog/g", "2,3s/x/y/g", "g/^$/d", "v/keep/d", "1m$", "2t2",
    "sort n", "sort u", ":2,4j", "%s/\\s\\+$//e", "g/^/m0", "retab!",
    "2,5>", "g/foo/s//bar/g", "$put ='x'", "%s/\\(\\w\\+\\) \\(\\w\\+\\)/\\2 \\1/",
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
    "qqddq", "@a", "ZZ", "ZQ", "\"ayy", "vd", "Vd", "3Vd", "<C-v>jjd",
    ":wq<CR>", ":w /tmp/x<CR>", "Qi", "2qad",
])
def test_keys_forbidden_commands_screened(cmd):
    assert screen_keys(cmd) is not None, cmd


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
        cand, corpus, CATEGORIES, out, quar, timeout=5.0, vim_bin="vim",
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
        dup, corpus, CATEGORIES, out, quar, timeout=5.0, vim_bin="vim",
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
        cand2, corpus2, CATEGORIES, out, quar, timeout=5.0, vim_bin="vim",
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
