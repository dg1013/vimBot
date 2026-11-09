#!/usr/bin/env python3
"""Not pre-registered: the 27B's own samples with the verifier, and how light the timing rows are.

The report's largest gains sit in arms the pre-registration made descriptive, so they carry no
verdict. This puts the same rule and the same interval on them, after their numbers had been seen.
Nothing here changes a reading in report.md, and a claim from it needs its own pre-registration.

Only arms report.md already showed are used (a test pins that), cheapest first by measured GPU time.
Also here, because the report's cost rows rest on them: how the timing rows compare with all rows,
and what the verifier does to the 27B's answers in the three-model pool.

    python scripts/serve27b/posthoc27b.py --run runs/serve27b_2026-11-05
"""
import argparse
import datetime
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pool27b as P                                               # noqa: E402

A, C = P.A, P.C

FAMILY = ("27B pick at 1", "27B pick at 2", "27B pick at 4", "27B greedy + 4, pick", "27B pick at 8",
          "27B greedy + 8, pick", "27B pick at 16", "27B pick at 32")


def composed(k, timing, pool, acc_pct):
    """GPU time adds: the 27B's answers, then one verifier pass as long as the pooled pick's. Timed
    on the head of val, like every cost row, and the k samples on its first 30 rows."""
    gpu = pool["pooled44"]["score_mean_s"]
    if "27b" in k:
        gpu += timing["samples"][str(k["27b"])]["mean_s"]
    if "27g" in k:
        gpu += timing["greedy"]["mean_s"]
    return P.per_solved(gpu, acc_pct)


def overrides(rows, k):
    """On one run of arm k: the verifier's pick against the 27B's own answer, over rows where they
    differ. (both right, only the pick, only the 27B, neither, the rows where they agree)."""
    out = {"both": 0, "pick_only": 0, "27b_only": 0, "neither": 0, "agree": 0}
    for r in rows.values():
        _, _, _, key, _ = P.first_k_arm(r, k)
        own = r["samples"]["27g"][0]
        if key is not None and own is not None and key == tuple(own):
            out["agree"] += 1
            continue
        lab = P.S.row_maps(r)[1]
        pick, mine = key is not None and lab[key], own is not None and lab[tuple(own)]
        out["both" if pick and mine else "pick_only" if pick else "27b_only" if mine else "neither"] += 1
    return out


def row_weights(run, val_ids, test_ids):
    """How the timing rows compare with all rows, and each stage's own rate over all of them."""
    greedy = {r["id"]: r for sp in A.SPLITS for r in A.load_jsonl(run / ("greedy_%s.jsonl" % sp))}
    sets = {"val[:30], the k-sample timing": val_ids[:30], "val[:96], throughput at 8 and 32": val_ids[:96],
            "val[:100], the pooled pick": val_ids[:100], "val[:200], one-at-a-time latency": val_ids[:200],
            "all 974 rows": val_ids + test_ids}
    table = {name: {"rows": len(ids), "prompt": round(statistics.fmean(greedy[i]["prompt_tokens"] for i in ids), 1),
                    "output": round(statistics.fmean(sum(greedy[i]["output_tokens"]) for i in ids), 1)}
             for name, ids in sets.items()}
    stamps = {}
    for line in (run / "status.txt").read_text().splitlines():
        for mark in ("greedy: start", "greedy: done"):
            if mark in line:
                stamps[mark] = datetime.datetime.fromisoformat(line.split()[0])
    greedy_wall = (stamps["greedy: done"] - stamps["greedy: start"]).total_seconds()
    samples = A.load_jsonl(run / "samples.jsonl")
    wall = sum(r["wall_s"] for r in A.load_jsonl(run / "passes.jsonl"))
    tokens = sum(sum(r["output_tokens"]) for r in samples)
    return {"sets": table,
            "greedy_stage": {"rows": len(greedy), "wall_s": greedy_wall, "gpu_s_per_request": round(greedy_wall / len(greedy), 3)},
            "samples_stage": {"rows": len(samples), "wall_s": wall, "tokens_per_row": round(tokens / len(samples), 1),
                              "tokens_per_s": round(tokens / wall, 1), "s_per_sample": round(wall / sum(len(r["samples"]) for r in samples), 3)}}


def build(run):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    untouched = C.untouched_ids(cases["val"])
    clean = set(untouched)
    rows = {sp: P.load_rows(run, sp, [c["id"] for c in cases[sp]])[0] for sp in A.SPLITS}
    every = {**rows["val"], **rows["test"]}
    choosing = [c["id"] for c in cases["val"] if c["id"] not in clean and c["id"] in every]
    parts = {"untouched": [i for i in untouched if i in every], "test": [c["id"] for c in cases["test"] if c["id"] in every]}

    base = {}
    for sp in A.SPLITS:
        base.update(P.base_shares(rows[sp], list(rows[sp])))
    g27 = {i: P.first_k_arm(r, {"27g": 1})[0] for i, r in every.items()}
    pool_k = dict(P.base_k(), **{"27g": 1})
    pool = {i: s[0] for i, s in P.arm_shares(every, list(every), pool_k).items()}

    arms = dict(P.descriptive(32))
    pick = {name: {i: s[0] for i, s in P.arm_shares(every, list(every), arms[name]).items()} for name in FAMILY}
    means = {name: P.mean(pick[name], choosing) for name in FAMILY}
    chosen = P.choose(means, list(FAMILY))
    one = {i: P.first_k_arm(r, arms[chosen])[0] for i, r in every.items()}
    one_base = {i: P.first_k_arm(r, P.base_k())[0] for i, r in every.items()}
    one_pool = {i: P.first_k_arm(r, pool_k)[0] for i, r in every.items()}

    timing = json.loads((run / "timing.json").read_text())
    pool_t = json.loads((run / "pool_timing.json").read_text())
    rep = {"choosing_rows": len(choosing), "means": means, "chosen": chosen,
           "by_split": {name: {"untouched": P.mean(pick[name], parts["untouched"]), "test": P.mean(pick[name], parts["test"])}
                        for name in FAMILY},
           "readings": {"against the 27B's one answer": P.reading(g27, pick[chosen], parts, g27, one),
                        "against the pooled pick, 4 + 4": P.reading(base, pick[chosen], parts, one_base, one),
                        "against the three-model pool": P.reading(pool, pick[chosen], parts, one_pool, one)},
           "cost": {name: composed(arms[name], timing, pool_t, P.mean(pick[name], parts["test"])) for name in FAMILY},
           "overrides_in_the_pool": overrides(every, pool_k),
           "overrides_in_the_chosen_arm": overrides(every, arms[chosen]),
           "weights": row_weights(run, [c["id"] for c in cases["val"]], [c["id"] for c in cases["test"]])}
    return rep


def render(rep):
    out = ["# Post hoc: the 27B's samples with the verifier, and the timing rows", "",
           "**Not pre-registered.** Read after `report.md`, whose numbers for these arms had been seen. "
           "Nothing here changes a reading there, and a claim from it needs its own pre-registration. "
           "Built by `scripts/serve27b/posthoc27b.py`.", "",
           "## The rule on the %d choosing rows" % rep["choosing_rows"], "",
           "Only arms `report.md` showed, cheapest first; the first within 1.0 point of the best is chosen. "
           "GPU time is composed from parts: this cannot run on one card, since the verifier needs about 8 GB "
           "beside the 27B's 22.7.", "",
           "| arm | choosing rows | untouched | test | GPU-s / request | $ / 1,000 solved |", "|---|---|---|---|---|---|"]
    for name, value in rep["means"].items():
        s, c = rep["by_split"][name], rep["cost"][name]
        mark = "**" if name == rep["chosen"] else ""
        out.append("| %s%s%s | %s%% | %s%% | %s%% | %.2f | $%.4f |" % (mark, name, mark, value, s["untouched"], s["test"],
                                                                      c["gpu_s_per_request"], c["usd_per_1000_solved"]))
    out += ["", "Chosen: **%s**." % rep["chosen"], "", "## Its readings on the 724 rows", "",
            "| %s | pooled rows | untouched | test | the rule's word | one run |" % rep["chosen"], "|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        o = r["one_run"]
        out.append("| %s | **%s** | %s | %s | %s | %+.1f, +%d / −%d, p %.3g |" % (
            name, P.fmt(r["pooled"]), P.fmt(r["untouched"]), P.fmt(r["test"]), r["verdict"], o["diff_pts"], o["gained"], o["lost"], o["p"]))
    o = rep["overrides_in_the_pool"]
    c = rep["overrides_in_the_chosen_arm"]
    head = "| where the pick and the 27B's one answer differ | agree | both right | only the pick | only the 27B | neither |"
    out += ["", "## What the verifier does to the 27B's answers (one run, all 974 rows)", "", head, "|---|---|---|---|---|---|",
            "| three-model pool, 4 + 4 + greedy | %d | %d | %d | %d | %d |" % (o["agree"], o["both"], o["pick_only"], o["27b_only"], o["neither"]),
            "| %s | %d | %d | %d | %d | %d |" % (rep["chosen"], c["agree"], c["both"], c["pick_only"], c["27b_only"], c["neither"]),
            "", "## The timing rows are lighter than the rows", "",
            "`val.jsonl` is not shuffled, and every timing pass takes its head, as every earlier cost table did. "
            "So the GPU-seconds in the report are low for all rows. The 27B and the pooled pick were timed on nearly "
            "the same rows, so comparing them is like for like, but neither figure carries over to all rows.", "",
            "| rows | n | prompt tokens | output tokens |", "|---|---|---|---|"]
    w = rep["weights"]
    for name, r in w["sets"].items():
        out.append("| %s | %d | %s | %s |" % (name, r["rows"], r["prompt"], r["output"]))
    g, s = w["greedy_stage"], w["samples_stage"]
    out += ["", "- The greedy stage, on all %d rows: %.0f s, **%.3f GPU-s per request**." % (g["rows"], g["wall_s"], g["gpu_s_per_request"]),
            "- The samples stage, on all %d rows: %.0f s, %.1f tokens per row, %.1f tokens/s, %.3f s per sample." % (
                s["rows"], s["wall_s"], s["tokens_per_row"], s["tokens_per_s"], s["s_per_sample"]), ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(A.REPO / "runs" / "serve27b_2026-11-05"))
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    rep = build(run)
    (run / "posthoc.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (run / "posthoc.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
