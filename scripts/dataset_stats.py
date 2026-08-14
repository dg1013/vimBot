"""Dataset statistics report (§7.4): writes data/stats.md.

Reports yield rate (kept vs rejected candidates, by reject_reason), per-cell
kept counts vs taxonomy quota, mode/domain balance, length distributions
(chars and whitespace tokens — approximate, no tokenizer dependency), and
N seeded-random samples.

Usage:
    python scripts/dataset_stats.py
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_data import REPO, iter_jsonl, load_taxonomy  # noqa: E402
from settings import load_settings  # noqa: E402


def dist(values: list[int]) -> str:
    """min / median / mean / p90 / max on one line."""
    if not values:
        return "-"
    vs = sorted(values)
    n = len(vs)
    median = vs[n // 2]
    p90 = vs[min(n - 1, (n * 9) // 10)]
    mean = sum(vs) / n
    return f"min {vs[0]} / med {median} / mean {mean:.1f} / p90 {p90} / max {vs[-1]}"


def fence(text: str) -> list[str]:
    return ["```", *text.split("\n"), "```"]


def main() -> None:
    # Two-pass parse: --settings supplies the defaults for the other flags.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings", help="settings TOML "
                                        "(default: configs/settings.toml)")
    known, _ = pre.parse_known_args()
    cfg = load_settings(known.settings)["stats"]

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre])
    ap.add_argument("--cases", default=str(REPO / "data" / "generated_cases.jsonl"))
    ap.add_argument("--quarantine", default=str(REPO / "data" / "quarantine.jsonl"))
    ap.add_argument("--taxonomy", default=str(REPO / "data" / "taxonomy.yaml"))
    ap.add_argument("--out", default=str(REPO / "data" / "stats.md"))
    ap.add_argument("--samples", type=int, default=cfg["samples"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    args = ap.parse_args()

    cases = list(iter_jsonl(Path(args.cases)))
    quarantine = list(iter_jsonl(Path(args.quarantine)))
    taxonomy = load_taxonomy(Path(args.taxonomy))
    categories = taxonomy["categories"]

    if not cases:
        sys.exit(f"no cases in {args.cases}")

    kept_scenarios = {c.get("source_candidate") for c in cases if c.get("source_candidate")}
    rejected_ids = {q.get("candidate_id") for q in quarantine if q.get("candidate_id")}
    n_scenarios = len(kept_scenarios) + len(rejected_ids)
    reject_by_reason: dict[str, int] = defaultdict(int)
    for q in quarantine:
        reject_by_reason[q.get("reject_reason", "unknown")] += 1

    kept_per_cell: dict[tuple[str, str], int] = defaultdict(int)
    mode_counts: dict[str, int] = defaultdict(int)
    domain_counts: dict[str, int] = defaultdict(int)
    for c in cases:
        kept_per_cell[(c["category"], c.get("tier", "T1"))] += 1
        mode_counts[c["mode"]] += 1
        domain_counts[c.get("domain", "unknown")] += 1

    lines: list[str] = []
    w = lines.append

    w("# Dataset stats — vimBot")
    w("")
    w(f"Source: `{Path(args.cases).name}` — teacher: "
      f"{cases[0].get('teacher', 'unknown')}")
    w("")
    w("## Yield")
    w("")
    w(f"- candidate scenarios processed: **{n_scenarios}** "
      f"(kept {len(kept_scenarios)}, rejected {len(rejected_ids)})")
    if n_scenarios:
        w(f"- scenario yield rate: **{100 * len(kept_scenarios) / n_scenarios:.0f}%**")
    w(f"- kept cases after paraphrase expansion: **{len(cases)}**")
    w("")
    if reject_by_reason:
        w("| reject_reason | count |")
        w("|---|---|")
        for reason in sorted(reject_by_reason):
            w(f"| {reason} | {reject_by_reason[reason]} |")
        w("")

    w("## Per-cell fill vs quota")
    w("")
    w("| category | T1 | T2 | T3 | total | quota | fill |")
    w("|---|---|---|---|---|---|---|")
    grand_kept = grand_quota = 0
    for cat in categories:
        quotas = categories[cat].get("quotas", {})
        row_kept = [kept_per_cell.get((cat, t), 0) for t in ("T1", "T2", "T3")]
        total = sum(row_kept)
        quota = sum(quotas.get(t, 0) for t in ("T1", "T2", "T3"))
        grand_kept += total
        grand_quota += quota
        fill = f"{100 * total / quota:.1f}%" if quota else "-"
        w(f"| {cat} | {row_kept[0]} | {row_kept[1]} | {row_kept[2]} "
          f"| {total} | {quota} | {fill} |")
    fill = f"{100 * grand_kept / grand_quota:.1f}%" if grand_quota else "-"
    w(f"| **all** | | | | **{grand_kept}** | **{grand_quota}** | **{fill}** |")
    w("")

    w("## Balance")
    w("")
    n = len(cases)
    w("- mode: " + ", ".join(
        f"{m} {c} ({100 * c / n:.0f}%)" for m, c in sorted(mode_counts.items())))
    w("- domains: " + ", ".join(
        f"{d} {c}" for d, c in sorted(domain_counts.items(),
                                      key=lambda kv: -kv[1])))
    w("")

    w("## Length distributions (approximate: chars / whitespace tokens)")
    w("")
    w(f"- instruction chars: {dist([len(c['instruction']) for c in cases])}")
    w(f"- instruction tokens: {dist([len(c['instruction'].split()) for c in cases])}")
    w(f"- command chars: {dist([len(c['gold_command']) for c in cases])}")
    w(f"- buffer_before lines: "
      f"{dist([len(c['buffer_before'].split(chr(10))) for c in cases])}")
    w("")

    rng = random.Random(args.seed)
    sample = rng.sample(cases, min(args.samples, len(cases)))
    w(f"## {len(sample)} random samples (seed {args.seed})")
    for c in sample:
        w("")
        w(f"### {c['id']}  ({c['category']}/{c.get('tier', '?')}, {c['mode']}, "
          f"domain {c.get('domain', '?')})")
        w("")
        w(f"- instruction: {c['instruction']}")
        w(f"- command: `{c['gold_command']}`  (cursor {c.get('cursor', [1, 1])})")
        w("")
        w("buffer_before:")
        lines.extend(fence(c["buffer_before"]))
        w("")
        w("buffer_after:")
        lines.extend(fence(c["buffer_after"]))

    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(cases)} cases, "
          f"{len(kept_scenarios)}/{n_scenarios} scenarios kept)")


if __name__ == "__main__":
    main()
