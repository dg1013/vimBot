#!/usr/bin/env python3
"""Test v2 numbers for every results-table run, from saved predictions plus
the regenerated reworded rows (scripts/remeasure/README.md).

For each run in plan.TABLE:
  * the 398 rows that did not change are re-scored from the run's own saved
    greedy predictions (`runs/<run>/preds_test.jsonl`) against test v2 --
    identical cases, so identical verdicts to the run's v1 measurement;
  * the 31 reworded rows come from its regeneration,
    `<R>/regen/<config>/exec_test.jsonl` and `completions_test.jsonl`;
  * the 16 dropped rows are gone.
Writes <R>/report.md, <R>/summary.json and, per run, the merged 429-row
execution log `<R>/v2/<run>/exec_test.jsonl` -- the v2 audit trail in the
same shape as every other exec log.

    python scripts/remeasure/v2_report.py --run runs/remeasure_2026-09-25
"""
import argparse
import datetime
import json
import pathlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl, score, write_exec_log   # noqa: E402
from plan import plan                                      # noqa: E402

RUNS = REPO / "runs"
FIXES = REPO / "data" / "fixes" / "always_failing_review_2026-09-25.json"
FLA_SINCE = "2026-09-16"          # the fused linear-attention kernels (RESULTS.md)


def by_id(rows):
    return {r["id"]: r for r in rows}


def rate(xs):
    return 100.0 * sum(xs) / len(xs) if xs else float("nan")


def measure(row, v1_cases, v2_cases, reworded, dropped, regen_dir):
    """One run's v1 figures (as recorded) and v2 figures (merged)."""
    run = RUNS / row["run"]
    summary = json.loads((run / "baseline_test.json").read_text(encoding="utf-8"))
    preds = {p["id"]: p for p in load_jsonl(str(run / "preds_test.jsonl"))}
    tokens = {c["id"]: c["output_tokens"] for c in load_jsonl(str(run / "completions_test.jsonl"))}

    unchanged = [c for c in v2_cases if c["id"] not in reworded]
    kept = score(unchanged, preds)["per_case"]
    old_rew = score([v1_cases[i] for i in sorted(reworded)], preds)["per_case"]
    old_drop = score([v1_cases[i] for i in sorted(dropped)], preds)["per_case"]

    regen = by_id(load_jsonl(str(regen_dir / "exec_test.jsonl")))
    regen_tok = {c["id"]: c["output_tokens"]
                 for c in load_jsonl(str(regen_dir / "completions_test.jsonl"))}
    missing = sorted(set(reworded) - set(regen))
    if missing:
        sys.exit("%s: regeneration lacks %s" % (row["config"], missing))

    merged = dict(kept)
    for i in reworded:
        entry = dict(regen[i])
        entry.pop("id", None)
        entry.pop("category", None)
        merged[i] = entry
    verdicts = [merged[c["id"]] for c in v2_cases]
    toks = [tokens.get(c["id"], 0) if c["id"] not in reworded else regen_tok.get(c["id"], 0)
            for c in v2_cases]
    m1 = summary["metrics"]
    qwen35 = row["model"].startswith("qwen3.5")
    return {
        "label": row["label"], "run": row["run"], "config": row["config"],
        "v1": {"n": summary["n"], "exec_match_pct": m1["exec_match_pct"],
               "valid_pct": m1["valid_pct"], "tokens_mean": m1.get("output_tokens_mean")},
        "v2": {"n": len(v2_cases), "exec_match": sum(v["match"] for v in verdicts),
               "exec_match_pct": round(rate([v["match"] for v in verdicts]), 1),
               "valid_pct": round(rate([v["valid"] for v in verdicts]), 1),
               "tokens_mean": round(statistics.mean(toks), 1)},
        "reworded_solved_before": sum(v["match"] for v in old_rew.values()),
        "reworded_solved_after": sum(regen[i]["match"] for i in reworded),
        "dropped_solved": sum(v["match"] for v in old_drop.values()),
        "mixed_kernels": bool(qwen35 and (row["date"] or "") < FLA_SINCE),
        "_merged": merged,
    }


def build(r_dir):
    r_dir = pathlib.Path(r_dir)
    v1_cases = by_id(load_jsonl(str(r_dir / "test_v1_445.jsonl")))
    v2_cases = load_jsonl(str(REPO / "data" / "splits" / "test.jsonl"))
    fixes = json.loads(FIXES.read_text(encoding="utf-8"))
    reworded, dropped = set(fixes["edits"]), set(fixes["deletes"])
    if reworded - {c["id"] for c in v2_cases} or dropped & {c["id"] for c in v2_cases}:
        sys.exit("test v2 does not match %s" % FIXES)
    rows, _configs = plan()

    def one(row):
        return measure(row, v1_cases, v2_cases, reworded, dropped, r_dir / "regen" / row["config"])
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(one, rows))

    for res in results:
        out = r_dir / "v2" / res["run"].replace("/", "__") / "exec_test.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_exec_log(str(out), v2_cases, {"per_case": res.pop("_merged")})

    md = []
    w = md.append
    w("# Test v2: every results-table run on the 429-row slice")
    w("")
    w("Built %s by `scripts/remeasure/v2_report.py`. Test v2 is the frozen test slice after the "
      "always-failing review (DECISIONS.md 2026-09-25): %d rows dropped and %d reworded (one "
      "with a new gold), %d unchanged. Each run's unchanged rows are its own saved greedy "
      "predictions re-scored (so identical to its v1 verdicts); its reworded rows are "
      "regenerated greedily with the same model, adapter and base. v1 figures are the "
      "recorded ones on the 445-row slice."
      % (datetime.date.today().isoformat(), len(dropped), len(reworded),
         len(v2_cases) - len(reworded)))
    w("")
    w("| run | v1 exec-match (445) | **v2 exec-match (429)** | Δ | v2 valid | v2 tokens | "
      "reworded rows solved, old → new wording (of %d) | dropped rows it had solved (of %d) |"
      % (len(reworded), len(dropped)))
    w("|---|---|---|---|---|---|---|---|")
    for res in results:
        v1, v2 = res["v1"], res["v2"]
        w("| %s%s | %.1f%% | **%.1f%%** (%d) | %+.1f | %.1f%% | %.1f | %d → %d | %d |"
          % (res["label"], " ¹" if res["mixed_kernels"] else "", v1["exec_match_pct"],
             v2["exec_match_pct"], v2["exec_match"], v2["exec_match_pct"] - v1["exec_match_pct"],
             v2["valid_pct"], v2["tokens_mean"], res["reworded_solved_before"],
             res["reworded_solved_after"], res["dropped_solved"]))
    w("")
    w("¹ Measured on the pure-torch path before the fused kernels were installed "
      "(%s); its %d regenerated rows ran under the kernels, which change about 10%% of "
      "greedy completions. The unchanged rows keep their recorded verdicts."
      % (FLA_SINCE, len(reworded)))
    w("")
    w("Latency is not re-measured: the 398 reused rows keep their recorded latency and 31 "
      "new rows cannot move a p50 honestly. The merged per-case logs are under `v2/`.")
    (r_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (r_dir / "summary.json").write_text(json.dumps(
        {"built": datetime.datetime.now().isoformat(timespec="seconds"),
         "reworded": sorted(reworded), "dropped": sorted(dropped), "runs": results},
        indent=1) + "\n", encoding="utf-8")
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="the re-measurement directory")
    args = ap.parse_args(argv)
    results = build(args.run)
    print("wrote %s/report.md for %d runs" % (args.run, len(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
