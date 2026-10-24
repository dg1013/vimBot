"""Dedup-aware train/val/test splits with scripted leakage checks (§7.3).

Cases are grouped into leakage components with a union-find over these keys:
same exact-dup key (normalized instruction, buffer), same near-dup
instruction token hash, same normalized buffer_before (buffer template),
same normalized (mode, gold_command) — the round-three redundancy audit
found 42 repeated commands whose instructions were deliberately reworded,
which the token hash is blind to — plus, when the fields are present, same
template_id (v2 paraphrase cluster) and the paraphrase_of id link. Whole
components — never individual cases — are then allocated test -> val ->
train, so no buffer template and no paraphrase cluster can straddle splits
by construction.
The same invariants are re-verified after allocation and the script exits
nonzero on any violation ("leakage checks scripted, not eyeballed").

Test slice targets 300-500 cases balanced across category x tier, with at
least 150 flagged needs_review for human verification. --smoke relaxes the
size targets for pilot-scale corpora.

Usage:
    python scripts/build_splits.py [--smoke]
"""

import argparse
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import (  # noqa: E402
    REPO, append_jsonl, instruction_token_hash, iter_jsonl, norm_buffer,
    norm_instruction,
)
from pattern_features import case_bucket  # noqa: E402
from settings import load_settings  # noqa: E402


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


_CTRL_KEY = re.compile(r"<C-[a-zA-Z]>")


def norm_command(command: str, mode: str) -> str:
    """Leakage-key form of a gold command.

    ex mode: outer whitespace and a leading ':' are noise. keys mode: every
    character is a keystroke (a space is a motion, a trailing newline is
    Enter), so nothing is stripped; only <C-x> letter case is folded, since
    vim treats <C-a> and <C-A> as the same key.
    """
    if mode == "keys":
        return _CTRL_KEY.sub(lambda m: m.group(0).upper(), command)
    return command.strip().lstrip(":")


def leakage_keys(case: dict) -> list[tuple]:
    """The keys that must not straddle splits. Shared by DSU and verifier."""
    mode = case.get("mode", "ex")
    keys = [
        ("exact", norm_instruction(case["instruction"]),
         norm_buffer(case["buffer_before"])),
        ("near", instruction_token_hash(case["instruction"])),
        ("buffer", norm_buffer(case["buffer_before"])),
        ("command", mode, norm_command(case["gold_command"], mode)),
        # every case anchors its own id so paraphrase_of links resolve
        ("paraphrase", case["id"]),
    ]
    if case.get("template_id"):
        keys.append(("template", case["template_id"]))
    if case.get("paraphrase_of"):
        keys.append(("paraphrase", case["paraphrase_of"]))
    return keys


def build_components(cases: list[dict]) -> list[list[int]]:
    dsu = DSU(len(cases))
    first_seen: dict[tuple, int] = {}
    for i, case in enumerate(cases):
        for key in leakage_keys(case):
            if key in first_seen:
                dsu.union(first_seen[key], i)
            else:
                first_seen[key] = i
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(cases)):
        groups[dsu.find(i)].append(i)
    return list(groups.values())


def stratum_of(case: dict) -> tuple[str, str, str, str]:
    """Stratification key: (source, category, tier, feature bucket).

    Rare structural kinds — capture groups, no-pattern keys cases,
    alternation — must be dealt across ALL splits instead of vanishing into
    one by shuffle luck, and each ingested source must be represented
    proportionally everywhere.
    """
    # mined rows can carry a blank tier; group them as one "T?" pseudo-tier
    # rather than pretending T1 or fragmenting on ""-vs-missing.
    return (case.get("source", "hand-authored"), case["category"],
            case.get("tier") or "T?", case_bucket(case))


def majority_stratum(cases: list[dict], comp: list[int]) -> tuple:
    counts: dict[tuple, int] = defaultdict(int)
    for i in comp:
        counts[stratum_of(cases[i])] += 1
    return max(counts, key=lambda c: (counts[c], c))


def allocate(
        cases: list[dict],
        components: list[list[int]],
        test_min: int, test_max: int, val_frac: float,
        seed: int, smoke: bool,
) -> dict[str, list[int]]:
    rng = random.Random(seed)
    comps = sorted(components, key=lambda c: min(cases[i]["id"] for i in c))
    rng.shuffle(comps)

    total = len(cases)
    test_target = min(test_max, max(test_min, (test_min + test_max) // 2))
    if smoke:
        test_target = min(test_target, max(1, total // 5))
    val_target = round(val_frac * total)

    stratum_total: dict[tuple, int] = defaultdict(int)
    for case in cases:
        stratum_total[stratum_of(case)] += 1

    def targets(overall: int) -> dict[tuple, int]:
        return {s: max(1, round(overall * n / total))
                for s, n in stratum_total.items()}

    splits: dict[str, list[int]] = {"test": [], "val": [], "train": []}
    taken: set[int] = set()

    # Stratified passes for test then val: fill each stratum toward its
    # proportional target, smallest components first so rare strata are not
    # blown past their budget by one giant component.
    for split, overall in (("test", test_target), ("val", val_target)):
        stratum_target = targets(overall)
        stratum_got: dict[tuple, int] = defaultdict(int)
        for comp in sorted(comps, key=len):
            if len(splits[split]) >= overall:
                break
            if any(i in taken for i in comp):
                continue
            stratum = majority_stratum(cases, comp)
            if stratum_got[stratum] + len(comp) \
                    > stratum_target[stratum] + 2:
                continue
            splits[split].extend(comp)
            stratum_got[stratum] += len(comp)
            taken.update(comp)

    for comp in comps:
        if any(i in taken for i in comp):
            continue
        splits["train"].extend(comp)
        taken.update(comp)
    return splits


def _group_keys(case: dict) -> list[tuple[str, str]]:
    """The (kind, name) groups the coverage invariant tracks.

    'category' was added 2026-08-27. Tracking only source and bucket let a
    whole taxonomy category fall out of the test slice unnoticed: measured
    then, `visual-block` had 62 cases across 27 components and **zero test
    rows**, so that capability was not being measured at all. Category is the
    axis RESULTS.md reports per-category exec-match on, so its absence is the
    one gap that silently invalidates a published number.
    """
    return [("source", case.get("source", "hand-authored")),
            ("bucket", case_bucket(case)),
            ("category", case["category"])]


def coverage_gaps(cases: list[dict], splits: dict[str, list[int]],
                  components: list[list[int]]):
    """Qualifying groups missing from some split.

    A group (source, bucket or category) with >= 15 cases spread over >= 3
    leakage components must appear in every split; smaller groups are
    best-effort (leakage protection outranks coverage for one-cluster
    rarities). Note this is a PRESENCE invariant, not a proportional one --
    see category_balance() for why proportion cannot be guaranteed here.
    Returns (gaps, qualifying) where gaps is a list of
    (kind, name, missing_split, n_cases, n_components).
    """
    table: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0})
    for split, indices in splits.items():
        for i in indices:
            for gk in _group_keys(cases[i]):
                table[gk][split] += 1
    spread: dict[tuple, set[int]] = defaultdict(set)
    for ci, comp in enumerate(components):
        for i in comp:
            for gk in _group_keys(cases[i]):
                spread[gk].add(ci)
    qualifying = {gk for gk, row in table.items()
                  if sum(row.values()) >= 15 and len(spread[gk]) >= 3}
    gaps = [(*gk, split, sum(table[gk].values()), len(spread[gk]))
            for gk in sorted(qualifying)
            for split in ("train", "val", "test") if table[gk][split] == 0]
    return gaps, qualifying


# A group present in a split with one or two rows satisfies the presence
# invariant but reports a per-category exec-match of 0% or 100% -- noise, not
# a measurement. The repair therefore aims for a small FLOOR rather than mere
# presence, capped by what a proportional share would be so a rare group is
# never dragged above its own weight.
_COVERAGE_FLOOR = 5


def _repair_targets(cases, splits, components, floor=_COVERAGE_FLOOR):
    """(kind, name, split, have, want) for qualifying groups under the floor.

    Presence gaps (have == 0) come first so the hard invariant is satisfied
    before any group is topped up toward the floor.
    """
    _, qualifying = coverage_gaps(cases, splits, components)
    total = sum(len(v) for v in splits.values())
    table: dict[tuple, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0})
    for split, indices in splits.items():
        for i in indices:
            for gk in _group_keys(cases[i]):
                table[gk][split] += 1

    out = []
    for gk in sorted(qualifying):
        n = sum(table[gk].values())
        for split in ("val", "test"):
            share = len(splits[split]) / total if total else 0.0
            want = min(floor, max(1, round(n * share)))
            have = table[gk][split]
            if have < want:
                out.append((gk[0], gk[1], split, have, want))
    # presence first, then the shallowest top-ups
    out.sort(key=lambda t: (t[3] != 0, t[3]))
    return out


def repair_coverage(cases: list[dict], components: list[list[int]],
                    splits: dict[str, list[int]], test_max: int) -> int:
    """Post-pass enforcing coverage, then topping up toward _COVERAGE_FLOOR.

    The greedy stratified fill takes smallest components first, so a source,
    bucket or category whose rows all sit in multi-case components can end up
    absent from test/val entirely (seen on v3+v4: the emulator sources, whose
    rows share buffers; and `visual-block`, which reached ZERO test rows).
    Move the smallest train component containing the short group into the gap
    — whole components only, so leakage invariants hold by construction.
    Guarded so no move empties a qualifying group's train presence, and every
    move strictly reduces the shortfall, so the loop terminates.
    """
    moved = 0
    while True:
        gaps = _repair_targets(cases, splits, components)
        _, qualifying = coverage_gaps(cases, splits, components)
        if not gaps:
            return moved
        train_set = set(splits["train"])
        group_train_comps: dict[tuple, list[int]] = defaultdict(list)
        for ci, comp in enumerate(components):
            if all(i in train_set for i in comp):
                for gk in {g for i in comp for g in _group_keys(cases[i])}:
                    group_train_comps[gk].append(ci)

        made_move = False
        for kind, name, split, _have, _want in gaps:
            candidates = []
            for ci in group_train_comps[(kind, name)]:
                comp = components[ci]
                if split == "test" \
                        and len(splits["test"]) + len(comp) > test_max:
                    continue
                donor_groups = {g for i in comp
                                for g in _group_keys(cases[i])}
                if any(g in qualifying and len(group_train_comps[g]) <= 1
                       for g in donor_groups):
                    continue
                candidates.append(ci)
            if not candidates:
                continue
            ci = min(candidates, key=lambda c: (
                len(components[c]), min(cases[i]["id"] for i in components[c])))
            comp = set(components[ci])
            splits["train"] = [i for i in splits["train"] if i not in comp]
            splits[split].extend(sorted(comp))
            moved += 1
            made_move = True
            break
        if not made_move:
            return moved  # remaining gaps are unrepairable; report + exit 1


def category_balance(cases: list[dict], splits: dict[str, list[int]]):
    """Per-category observed vs proportional share, with a skew flag.

    Presence is an invariant (coverage_gaps); proportion deliberately is NOT,
    and cannot be. Whole leakage components are dealt atomically, and the
    command key merges whole command FAMILIES transitively through the buffer
    key -- `dw` links to another `dw` by command, that one links to `daw` by
    buffer, and so on. Measured 2026-08-27 the largest such component held 158
    cases (70 operator-motion, 37 text-objects, 36 visual). A component that
    size exceeds every stratum's test budget, so it is refused by test and val
    and lands whole in train, dragging its categories' test share down with it.

    Forcing proportion would mean either splitting components (which is the
    leakage hole the keys exist to close) or letting one component occupy ~40%
    of a 400-case test slice. Neither is worth it, so the skew is REPORTED
    instead of silently absorbed -- a category reading 3.5% where 7.8% was
    expected is a fact the reader of RESULTS.md needs, because its
    per-category exec-match rests on that many rows and no more.

    Returns (rows, notes): rows are dicts per category, notes are human-
    readable strings for the categories whose test share is badly off.
    """
    total = sum(len(v) for v in splits.values())
    want = {s: len(splits[s]) / total for s in ("train", "val", "test")}

    per: dict[str, dict[str, int]] = defaultdict(
        lambda: {"train": 0, "val": 0, "test": 0})
    for split, indices in splits.items():
        for i in indices:
            per[cases[i]["category"]][split] += 1

    rows, notes = [], []
    for cat in sorted(per, key=lambda c: -sum(per[c].values())):
        got = per[cat]
        n = sum(got.values())
        exp_test = n * want["test"]
        row = {"category": cat, "n": n, **got,
               "test_share": got["test"] / n if n else 0.0,
               "expected_test": exp_test}
        rows.append(row)
        # only meaningful once the expectation is a few rows; below that the
        # granularity of whole components dominates and the ratio is noise
        if exp_test >= 4 and got["test"] < exp_test * 0.5:
            notes.append(
                f"{cat}: {got['test']} test rows where ~{exp_test:.0f} would be "
                f"proportional ({got['test'] / n:.1%} of {n}) -- its cases are "
                f"concentrated in large leakage components that cannot be split")
    return rows, notes


def coverage_report(cases: list[dict], splits: dict[str, list[int]]):
    """Per-source and per-bucket split tables + hard coverage violations."""
    def group_tables(keyfn):
        table: dict[str, dict[str, int]] = defaultdict(
            lambda: {"train": 0, "val": 0, "test": 0})
        for split, indices in splits.items():
            for i in indices:
                table[keyfn(cases[i])][split] += 1
        return table

    src_table = group_tables(lambda c: c.get("source", "hand-authored"))
    bkt_table = group_tables(case_bucket)

    gaps, _ = coverage_gaps(cases, splits, build_components(cases))
    by_group: dict[tuple, list[str]] = defaultdict(list)
    meta = {}
    for kind, name, split, n, spread in gaps:
        by_group[(kind, name)].append(split)
        meta[(kind, name)] = (n, spread)
    violations = [
        f"{kind} {name!r} ({meta[(kind, name)][0]} cases, "
        f"{meta[(kind, name)][1]} components) "
        f"missing from: {', '.join(missing)}"
        for (kind, name), missing in by_group.items()]
    return src_table, bkt_table, violations


def verify_no_leakage(cases: list[dict], splits: dict[str, list[int]]) -> list[str]:
    violations = []
    key_owner: dict[tuple, str] = {}
    for split, indices in splits.items():
        for i in indices:
            for key in leakage_keys(cases[i]):
                owner = key_owner.setdefault(key, split)
                if owner != split:
                    violations.append(
                        f"{key[0]} key straddles {owner}/{split}: case {cases[i]['id']}"
                    )
    return violations


def main() -> None:
    # Two-pass parse: --settings supplies the defaults for the other flags.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings", help="settings TOML "
                                        "(default: configs/settings.toml)")
    known, _ = pre.parse_known_args()
    cfg = load_settings(known.settings)["splits"]

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(REPO / "data" / "combined_cases.jsonl"))
    ap.add_argument("--outdir", default=str(REPO / "data" / "splits"))
    ap.add_argument("--test-min", type=int, default=cfg["test_min"])
    ap.add_argument("--test-max", type=int, default=cfg["test_max"])
    ap.add_argument("--val-frac", type=float, default=cfg["val_frac"])
    ap.add_argument("--review-min", type=int, default=cfg["review_min"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    ap.add_argument("--smoke", action="store_true",
                    help="pilot-scale corpus: relax test-size targets")
    args = ap.parse_args()

    cases = list(iter_jsonl(Path(args.cases)))
    if not cases:
        sys.exit(f"no cases in {args.cases}")

    components = build_components(cases)
    splits = allocate(cases, components, args.test_min, args.test_max,
                      args.val_frac, args.seed, args.smoke)
    moved = repair_coverage(cases, components, splits, args.test_max)
    if moved:
        print(f"coverage repair: moved {moved} component(s) out of train")

    violations = verify_no_leakage(cases, splits)
    if violations:
        for v in violations:
            print(f"LEAKAGE: {v}", file=sys.stderr)
        sys.exit(1)

    if not args.smoke and not (args.test_min <= len(splits["test"]) <= args.test_max):
        sys.exit(
            f"test slice {len(splits['test'])} outside "
            f"[{args.test_min}, {args.test_max}] (use --smoke for pilot data)"
        )

    # flag test cases for human verification, round-robin across cells
    review_target = len(splits["test"]) if args.smoke else max(
        args.review_min, len(splits["test"]) // 2
    )
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i in splits["test"]:
        by_cell[(cases[i]["category"], cases[i].get("tier") or "T?")].append(i)
    flagged: set[int] = set()
    while len(flagged) < min(review_target, len(splits["test"])):
        progressed = False
        for cell in sorted(by_cell):
            queue = [i for i in by_cell[cell] if i not in flagged]
            if queue:
                flagged.add(queue[0])
                progressed = True
            if len(flagged) >= min(review_target, len(splits["test"])):
                break
        if not progressed:
            break

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for split, indices in splits.items():
        path = outdir / f"{split}.jsonl"
        path.write_text("", encoding="utf-8")
        for i in sorted(indices, key=lambda i: cases[i]["id"]):
            row = dict(cases[i])
            if split == "test" and i in flagged:
                row["needs_review"] = True
            append_jsonl(path, row)

    src_table, bkt_table, coverage_violations = coverage_report(cases, splits)

    report = [
        "# Split report", "",
        f"cases: {len(cases)}   components: {len(components)} "
        f"(largest {max(len(c) for c in components)})", "",
        "| split | cases | share |", "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        n = len(splits[split])
        report.append(f"| {split} | {n} | {100 * n / len(cases):.0f}% |")
    for title, table in (("source", src_table),
                         ("feature bucket", bkt_table)):
        report += ["", f"## By {title}", "",
                   f"| {title} | train | val | test |", "|---|---|---|---|"]
        for name in sorted(table):
            row = table[name]
            report.append(f"| {name} | {row['train']} | {row['val']} "
                          f"| {row['test']} |")
    bal_rows, bal_notes = category_balance(cases, splits)
    report += ["", "## By category", "",
               "Presence in every split is an invariant; proportion is not — "
               "whole leakage components are dealt atomically, so a category "
               "concentrated in large components lands mostly in train.", "",
               "| category | cases | train | val | test | test share | "
               "proportional |", "|---|---|---|---|---|---|---|"]
    for r in bal_rows:
        report.append(
            f"| {r['category']} | {r['n']} | {r['train']} | {r['val']} "
            f"| {r['test']} | {r['test_share']:.1%} "
            f"| {r['expected_test']:.0f} |")
    if bal_notes:
        report += ["", "### Under-represented in test", ""]
        report += [f"- {n}" for n in bal_notes]

    report += ["", f"test cases flagged needs_review: {len(flagged)}", "",
               "Leakage checks: PASS (exact, near-dup, buffer, command, "
               "template, paraphrase keys all split-disjoint)"]
    if coverage_violations:
        report += ["", "## COVERAGE VIOLATIONS", ""]
        report += [f"- {v}" for v in coverage_violations]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"train {len(splits['train'])}  val {len(splits['val'])}  "
          f"test {len(splits['test'])} ({len(flagged)} flagged for review)")
    print(f"leakage checks: PASS  ->  {outdir}/report.md")
    for note in bal_notes:
        print(f"BALANCE: {note}", file=sys.stderr)
    if coverage_violations:
        for v in coverage_violations:
            print(f"COVERAGE: {v}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
