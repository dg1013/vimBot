"""Dedup-aware train/val/test splits with scripted leakage checks (§7.3).

Cases are grouped into leakage components with a union-find over four keys:
same template_id (paraphrase cluster), same exact-dup key
(normalized instruction, buffer), same near-dup instruction token hash, and
same normalized buffer_before (buffer template). Whole components — never
individual cases — are then allocated test -> val -> train, so no buffer
template and no paraphrase cluster can straddle splits by construction.
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
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import (  # noqa: E402
    REPO, append_jsonl, instruction_token_hash, iter_jsonl, norm_buffer,
    norm_instruction,
)
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


def leakage_keys(case: dict) -> list[tuple]:
    """The keys that must not straddle splits. Shared by DSU and verifier."""
    return [
        ("template", case["template_id"]),
        ("exact", norm_instruction(case["instruction"]),
         norm_buffer(case["buffer_before"])),
        ("near", instruction_token_hash(case["instruction"])),
        ("buffer", norm_buffer(case["buffer_before"])),
    ]


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


def majority_cell(cases: list[dict], comp: list[int]) -> tuple[str, str]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for i in comp:
        counts[(cases[i]["category"], cases[i].get("tier", "T1"))] += 1
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

    cell_total: dict[tuple[str, str], int] = defaultdict(int)
    for case in cases:
        cell_total[(case["category"], case.get("tier", "T1"))] += 1
    cell_target = {
        cell: max(1, round(test_target * n / total))
        for cell, n in cell_total.items()
    }

    splits: dict[str, list[int]] = {"test": [], "val": [], "train": []}
    cell_test: dict[tuple[str, str], int] = defaultdict(int)

    # pass 1: fill test respecting per-cell balance, smallest components first
    for comp in sorted(comps, key=len):
        if len(splits["test"]) >= test_target:
            break
        cell = majority_cell(cases, comp)
        if cell_test[cell] + len(comp) > cell_target[cell] + 2:
            continue
        splits["test"].extend(comp)
        cell_test[cell] += len(comp)

    taken = set(splits["test"])
    val_target = round(val_frac * total)
    for comp in comps:
        if comp[0] in taken or any(i in taken for i in comp):
            continue
        if len(splits["val"]) < val_target:
            splits["val"].extend(comp)
            taken.update(comp)
        else:
            splits["train"].extend(comp)
            taken.update(comp)
    return splits


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
    ap.add_argument("--cases", default=str(REPO / "data" / "generated_cases.jsonl"))
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
        by_cell[(cases[i]["category"], cases[i].get("tier", "T1"))].append(i)
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

    report = [
        "# Split report", "",
        f"cases: {len(cases)}   components: {len(components)} "
        f"(largest {max(len(c) for c in components)})", "",
        "| split | cases | share |", "|---|---|---|",
    ]
    for split in ("train", "val", "test"):
        n = len(splits[split])
        report.append(f"| {split} | {n} | {100 * n / len(cases):.0f}% |")
    report += ["", f"test cases flagged needs_review: {len(flagged)}", "",
               "Leakage checks: PASS (template, exact, near-dup, buffer keys "
               "all split-disjoint)"]
    (outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print(f"train {len(splits['train'])}  val {len(splits['val'])}  "
          f"test {len(splits['test'])} ({len(flagged)} flagged for review)")
    print(f"leakage checks: PASS  ->  {outdir}/report.md")


if __name__ == "__main__":
    main()
