#!/usr/bin/env python3
"""Report for the prompted 27B on val (CPU): every arm and comparator on the
same 545 rows, the pre-registered paired readings and their verdicts.

The comparators' saved predictions are re-scored against the current val
through evaluate.score before pairing: the trained models' logs date from
2026-09-15 to -18, before the splits were last written (2026-09-25). The
audit table lists every verdict that moved. The 27B's own arms were scored on
the current val when they ran.

    python scripts/teacher27b/report.py          # -> runs/teacher27b_2026-10-30/report.{md,json}
"""
import argparse
import ast
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "retrieval", HERE):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl, score                           # noqa: E402
from retrieval import mcnemar_p, paired                          # noqa: E402
import teacher27b as T                                           # noqa: E402

ARMS = ("27b-retr8", "27b-k0")
TIERS = ("T1", "T2", "T3")
N_BOOT = 10000
BOOT_SEED = 20261030
ALPHA = 0.05
# The pre-registered paired readings: (name, arm, comparator, question).
READINGS = [
    ("P1", "27b-retr8", "9b", "the prompted 27B against the trained 9B"),
    ("P1b", "27b-retr8", "4b-served", "the prompted 27B against the served 4B"),
    ("S1", "27b-retr8", "4b-base-retr8", "what size buys under the same prompting"),
    ("S2", "27b-k0", "4b-base-k0", "the zero-shot row against the 4B's"),
]


def pct(k, n):
    return round(100.0 * k / n, 1) if n else None


def verdict(diff, p, alpha=ALPHA):
    """Three-way reading of a paired difference: significant either way, or tied."""
    if p < alpha and diff > 0:
        return "stronger"
    if p < alpha and diff < 0:
        return "weaker"
    return "tied"


def compare(a, b, ids, n_boot=N_BOOT, seed=BOOT_SEED):
    """b against a on the same rows (id -> 0/1): gained, lost, the difference
    in points with a paired bootstrap 95% interval, exact McNemar p, verdict."""
    A, B = {i: a[i] for i in ids}, {i: b[i] for i in ids}
    g, l, n = paired(A, B)
    diff = 100.0 * (g - l) / n
    d = [B[i] - A[i] for i in ids]
    rng = random.Random(seed)
    boots = sorted(100.0 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    p = mcnemar_p(g, l)
    return {"gained": g, "lost": l, "n": n, "diff_pts": round(diff, 1),
            "ci95": [round(lo, 1), round(hi, 1)], "p": p, "verdict": verdict(diff, p)}


def commands(row):
    c = row["commands"]
    return ast.literal_eval(c) if isinstance(c, str) else c      # the 4B probe stores a repr


def plurality_right(row):
    """The most frequent command of the G is a match (ties: the first listed,
    the probe's most_common order; the rule of the 2026-10-30 headroom check)."""
    cs = commands(row)
    return int(bool(cs) and max(cs, key=lambda c: c["n"])["state"] == "match")


def probe_stats(rows, ids):
    n = len(ids)
    states = {}
    for i in ids:
        for s, k in rows[i]["states"].items():
            states[s] = states.get(s, 0) + k
    total = sum(states.values())
    return {"n": n, "pass_at_8_pct": pct(sum(rows[i]["k"] > 0 for i in ids), n),
            "plurality_pct": pct(sum(plurality_right(rows[i]) for i in ids), n),
            "mean_p_hat": round(statistics.fmean(rows[i]["p_hat"] for i in ids), 3),
            "truncated_pct": pct(states.get("truncated", 0), total),
            "parse_fail_pct": pct(states.get("parse", 0), total),
            "tokens_mean": round(statistics.fmean(rows[i]["tokens_mean"] for i in ids), 1)}


def complementarity(arm, others, ids):
    """Rows the arm solves and none of `others` does, and the union oracles."""
    only = sorted(i for i in ids if arm[i] and not any(o[i] for o in others))
    before = sum(any(o[i] for o in others) for i in ids)
    after = sum(any(o[i] for o in others) or arm[i] for i in ids)
    return {"only_arm": only, "union_without_pct": pct(before, len(ids)),
            "union_with_pct": pct(after, len(ids))}


def arm_stats(verd, comp, ids):
    n = len(ids)
    c = [comp[i] for i in ids if i in comp]
    parsed = [r for r in c if r["parse_ok"]]
    out = {"n": n, "exec_match_pct": pct(sum(verd[i]["match"] for i in ids), n),
           "valid_pct": pct(sum(verd[i]["valid"] for i in ids), n)}
    if c:
        out.update({
            "parse_failures": sum(not r["parse_ok"] for r in c),
            "think_leaks": sum(bool(r.get("think_leak")) for r in c),
            "at_cap": sum(r["output_tokens"] >= T.MAX_COMPLETION for r in c),
            "out_tokens_mean": round(statistics.fmean(r["output_tokens"] for r in c), 1),
            "out_tokens_p50": statistics.median(r["output_tokens"] for r in c),
            "prompt_tokens_mean": round(statistics.fmean(r["prompt_tokens"] for r in c), 1),
            "latency_p50_s": round(statistics.median(r["latency_s"] for r in c), 2),
            "ex_share_pct": pct(sum(r["mode"] == "ex" for r in parsed), len(parsed))})
    return out


def load_greedy(folder, cases, rescore, timeout):
    """(id -> {match, valid}, completions by id, re-score audit or None)."""
    folder = pathlib.Path(folder)
    logged = {r["id"]: r for r in load_jsonl(str(folder / "exec_val.jsonl"))}
    cpath = folder / "completions_val.jsonl"
    comp = {r["id"]: r for r in load_jsonl(str(cpath))} if cpath.exists() else {}
    as01 = {i: {"match": int(bool(r["match"])), "valid": int(bool(r["valid"]))} for i, r in logged.items()}
    if not rescore:
        return as01, comp, None
    preds = {r["id"]: r for r in load_jsonl(str(folder / "preds_val.jsonl"))}
    per = score(cases, preds, timeout=timeout)["per_case"]
    now = {i: {"match": int(bool(v["match"])), "valid": int(bool(v["valid"]))} for i, v in per.items()}
    moved = sorted(i for i in now if i in as01 and now[i]["match"] != as01[i]["match"])
    return now, comp, {"logged_match": sum(v["match"] for v in as01.values()),
                       "rescored_match": sum(v["match"] for v in now.values()), "moved": moved}


def next_step(p1, p1b):
    if p1 == "stronger":
        return ("The prompted 27B out-solves the trained 9B: the strongest teacher on hand, with no "
                "training. Next: a cost check for train scale (approaches 1-3 need all 6,548 train "
                "prompts, 12x these val rows, so this run's seconds per row decide whether that "
                "needs a vLLM path).")
    if p1b == "weaker":
        return ("Below the served 4B: not a teacher for logit or on-policy distillation (a student "
                "cannot learn more than its teacher knows). The 27B question becomes a fine-tuned 27B "
                "on a bigger card (an owner decision), or dropping it; approach 1 goes ahead with the "
                "five trained models' samples.")
    return ("Between the served 4B and the 9B, or tied with them: no stronger than what we have. "
            "Its value is diversity (the complementarity line), which only approach 1 can use, and "
            "only after a train-scale cost check.")


def build(args):
    run = pathlib.Path(args.run)
    cases = load_jsonl(str(T.CASES))
    ids = [c["id"] for c in cases]
    tier = {c["id"]: c["tier"] for c in cases}
    rep = {"run": str(run), "n": len(ids), "arms": {}, "comparators": {}, "audit": {}}
    verd, comps = {}, {}
    missing = [a for a in ARMS if not (run / a / "baseline_val.json").exists()]
    if "27b-retr8" in missing:
        sys.exit("no 27b-retr8 arm in %s yet: run the greedy stage first" % run)
    for a in ARMS:
        if a in missing:
            continue
        v, c, _ = load_greedy(run / a, cases, False, args.timeout)
        verd[a], comps[a] = v, c
        rep["arms"][a] = arm_stats(v, c, ids)
        rep["arms"][a]["summary_run"] = json.loads((run / a / "baseline_val.json").read_text())["run"]
    for label, (folder, desc) in T.GREEDY_COMPARATORS.items():
        v, c, audit = load_greedy(REPO / folder, cases, not args.no_rescore, args.timeout)
        verd[label], comps[label] = v, c
        rep["comparators"][label] = dict(arm_stats(v, c, ids), source=folder, what=desc)
        rep["audit"][label] = audit
    match = {k: {i: v[i]["match"] for i in ids} for k, v in verd.items()}
    rep["readings"] = {}
    for name, arm, comp, question in READINGS:
        if arm in match:
            rep["readings"][name] = dict(compare(match[comp], match[arm], ids), arm=arm,
                                         comparator=comp, question=question)
    rep["tiers"] = {t: {k: pct(sum(match[k][i] for i in ids if tier[i] == t),
                               sum(tier[i] == t for i in ids))
                        for k in ("27b-retr8", "27b-k0", "4b-base-retr8", "4b-served", "9b") if k in match}
                    for t in TIERS}
    rep["tier_n"] = {t: sum(tier[i] == t for i in ids) for t in TIERS}
    rep["S3"] = complementarity(match["27b-retr8"], [match["4b-served"], match["9b"]], ids)
    probe_path = run / T.PROBE_OUT.name
    if probe_path.exists():
        ours = {r["id"]: r for r in load_jsonl(str(probe_path))}
        if set(ours) >= set(ids):
            theirs = {k: {r["id"]: r for r in load_jsonl(str(REPO / p))}
                      for k, (p, _) in T.PROBE_COMPARATORS.items()}
            rep["S4"] = {"27b-retr8": probe_stats(ours, ids),
                         **{k: probe_stats(v, ids) for k, v in theirs.items()}}
            pl = {k: {i: plurality_right(v[i]) for i in ids} for k, v in
                  {"27b-retr8": ours, **theirs}.items()}
            cover = {k: {i: int(v[i]["k"] > 0) for i in ids} for k, v in
                     {"27b-retr8": ours, **theirs}.items()}
            rep["S4"]["plurality_vs_9b"] = compare(pl["9b"], pl["27b-retr8"], ids)
            rep["S4"]["plurality_vs_4b"] = compare(pl["4b-served"], pl["27b-retr8"], ids)
            rep["S4"]["coverage"] = complementarity(cover["27b-retr8"],
                                                    [cover["4b-served"], cover["9b"]], ids)
        else:
            rep["S4"] = {"incomplete": "%d of %d rows sampled" % (len(set(ours) & set(ids)), len(ids))}
    p1 = rep["readings"]["P1"]["verdict"]
    p1b = rep["readings"]["P1b"]["verdict"]
    rep["next_step"] = next_step(p1, p1b)
    return rep


def fmt_ci(r):
    return "%+.1f [%+.1f, %+.1f]" % (r["diff_pts"], r["ci95"][0], r["ci95"][1])


def render(rep):
    L = ["# The prompted Qwen3.5-27B on val (%d rows)" % rep["n"], "",
         "Pre-registration: `preregistration.md` beside this file. Built by "
         "`scripts/teacher27b/report.py`.", ""]
    retr = rep["arms"]["27b-retr8"]["summary_run"]
    L += ["27B load: %s s, %s GiB free after load, peak %s GiB in the k = 8 arm, %s s for the arm." % (
        retr.get("load", {}).get("load_s"), retr.get("load", {}).get("free_gib_after_load"),
        retr.get("peak_gpu_gib"), retr.get("arm_seconds")), ""]
    L += ["## Greedy arms and comparators", "",
          "| row | exec-match | valid | parse fail | think leaks | at cap | out tok mean / p50 | prompt tok | p50 s | ex share |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rows = [(a, rep["arms"][a]) for a in ARMS if a in rep["arms"]] + list(rep["comparators"].items())
    for name, s in rows:
        L.append("| %s | **%s%%** | %s%% | %s | %s | %s | %s / %s | %s | %s | %s%% |" % (
            name, s["exec_match_pct"], s["valid_pct"], s.get("parse_failures", "—"),
            s.get("think_leaks", "—"), s.get("at_cap", "—"), s.get("out_tokens_mean", "—"),
            s.get("out_tokens_p50", "—"), s.get("prompt_tokens_mean", "—"), s.get("latency_p50_s", "—"),
            s.get("ex_share_pct", "—")))
    L += ["", "Comparators: " + "; ".join("`%s` = %s (`%s`)" % (k, v["what"], v["source"])
                                          for k, v in rep["comparators"].items()) + ".", ""]
    L += ["## Pre-registered readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | question | arm vs comparator | gained / lost | difference, points | p | verdict |",
          "|---|---|---|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | %s | %s vs %s | +%d / −%d | %s | %.3g | **%s** |" % (
            name, r["question"], r["arm"], r["comparator"], r["gained"], r["lost"], fmt_ci(r), r["p"],
            r["verdict"]))
    L += ["", "**Next step, by the pre-registered rule:** " + rep["next_step"], ""]
    L += ["## By tier", "", "| tier | n | " + " | ".join(next(iter(rep["tiers"].values())).keys()) + " |",
          "|---|---|" + "---|" * len(next(iter(rep["tiers"].values())))]
    for t, d in rep["tiers"].items():
        L.append("| %s | %d | %s |" % (t, rep["tier_n"][t], " | ".join("%s%%" % v for v in d.values())))
    s3 = rep["S3"]
    L += ["", "## S3: complementarity (greedy)", "",
          "Rows the k = 8 arm solves and neither the served 4B nor the 9B does: **%d**. Greedy union "
          "oracle 4B ∪ 9B %s%% → with the 27B %s%%." % (len(s3["only_arm"]), s3["union_without_pct"],
                                                       s3["union_with_pct"]), ""]
    if "S4" in rep:
        s4 = rep["S4"]
        L += ["## S4: 8 samples at T = 1 (probe format)", ""]
        if "incomplete" in s4:
            L.append("Not reported: %s." % s4["incomplete"])
        else:
            L += ["| row | pass@8 | plurality of 8 | mean p̂ | truncated | parse fail | tokens |",
                  "|---|---|---|---|---|---|---|"]
            for k in ("27b-retr8", "4b-served", "9b"):
                s = s4[k]
                L.append("| %s | %s%% | %s%% | %s | %s%% | %s%% | %s |" % (
                    k, s["pass_at_8_pct"], s["plurality_pct"], s["mean_p_hat"], s["truncated_pct"],
                    s["parse_fail_pct"], s["tokens_mean"]))
            cv = s4["coverage"]
            L += ["", "Plurality vs the 9B: %s (p %.3g, %s); vs the served 4B: %s (p %.3g, %s). Rows "
                  "only the 27B's samples solve: **%d**; union pass@8 4B ∪ 9B %s%% → %s%%." % (
                      fmt_ci(s4["plurality_vs_9b"]), s4["plurality_vs_9b"]["p"],
                      s4["plurality_vs_9b"]["verdict"], fmt_ci(s4["plurality_vs_4b"]),
                      s4["plurality_vs_4b"]["p"], s4["plurality_vs_4b"]["verdict"], len(cv["only_arm"]),
                      cv["union_without_pct"], cv["union_with_pct"])]
        L.append("")
    L += ["## Re-score audit (comparators on the current val)", "",
          "| comparator | logged matches | re-scored | verdicts moved |", "|---|---|---|---|"]
    for k, a in rep["audit"].items():
        if a:
            L.append("| %s | %d | %d | %d %s |" % (k, a["logged_match"], a["rescored_match"], len(a["moved"]),
                                                  ", ".join(a["moved"][:8])))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(T.RUN))
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--no-rescore", action="store_true", help="pair the comparators' logged verdicts")
    args = ap.parse_args(argv)
    rep = build(args)
    run = pathlib.Path(args.run)
    (run / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (run / "report.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
