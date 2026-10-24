#!/usr/bin/env python3
"""Similarity-gated retrieval: retrieve only when the nearest train example is
similar enough; otherwise answer zero-shot.

Every row of a split already has an answer from each arm (zero-shot, retrieved
k = 1 / 3 / 5 / 8), so a gate that chooses an arm per row from the plan's BM25
scores is scored exactly, offline, with no new generation. Two policies:

  on-off      nearest score >= tau  ->  the retrieved arm (k = 5 or 8), else zero-shot
  variable-k  k = the largest of {8, 5, 3, 1} whose k-th nearest example scores >= tau,
              else 0: take only the examples that are similar enough

The threshold is picked on VAL by a pre-registered rule and read on TEST: the
smallest gated-on fraction that keeps at least `--keep` (default 90%) of the
retrieved arm's gain over zero-shot. The whole curve on test is also reported,
as description, never as the pick.

    python scripts/retrieval/gate.py --run runs/retrieval_2026-10-18 \\
        --pick-split val --pick-plans val_plans --read-split test

Writes gate_report.md and gate_report.json into the run folder.
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from retrieval import load_jsonl, mcnemar_p, paired           # noqa: E402

K_ARMS = {0: "base-k0", 1: "base-retr1", 3: "base-retr3", 5: "base-retr5", 8: "base-retr8"}


def load_rows(run, arm, split):
    """id -> {match, tokens} for one arm on one split (tokens = prompt + output)."""
    d = pathlib.Path(run) / arm
    ex = {r["id"]: int(bool(r["match"])) for r in load_jsonl(d / ("exec_%s.jsonl" % split))}
    comp = {r["id"]: r for r in load_jsonl(d / ("completions_%s.jsonl" % split))}
    return {i: {"match": ex[i], "tokens": comp[i]["prompt_tokens"] + comp[i]["output_tokens"],
                "prompt": comp[i]["prompt_tokens"]} for i in ex if i in comp}


def load_scores(plan_dir, k=8):
    """id -> the BM25 scores of the retrieved examples, nearest first, from plan_retr<k>."""
    rows = load_jsonl(pathlib.Path(plan_dir) / ("plan_retr%d.jsonl" % k))
    return {r["id"]: sorted(r["scores"], reverse=True) for r in rows}


def choose_on_off(scores, tau, on_k):
    return on_k if scores and scores[0] >= tau else 0


def choose_variable(scores, tau, ks=(8, 5, 3, 1)):
    for k in ks:
        if len(scores) >= k and scores[k - 1] >= tau:
            return k
    return 0


def evaluate(policy, tau, ids, scores, arms, on_k=5):
    """Metrics of the gated policy at tau: exec-match, mean tokens, tokens per
    solved task, gated-on fraction, mean k, and the per-row choices."""
    choice, match, tokens, prompt = {}, {}, [], []
    for i in ids:
        s = scores.get(i, [])
        k = choose_on_off(s, tau, on_k) if policy == "on-off" else choose_variable(s, tau)
        r = arms[K_ARMS[k]][i]
        choice[i] = k
        match[i] = r["match"]
        tokens.append(r["tokens"])
        prompt.append(r["prompt"])
    n = len(ids)
    m = sum(match.values())
    return {"tau": tau, "n": n, "exec_match": m, "exec_match_pct": 100.0 * m / n,
            "gated_on_pct": 100.0 * sum(1 for i in ids if choice[i] > 0) / n,
            "mean_k": statistics.fmean(choice[i] for i in ids),
            "prompt_tokens_mean": statistics.fmean(prompt), "tokens_mean": statistics.fmean(tokens),
            "tokens_per_solved": (sum(tokens) / m) if m else float("inf"),
            "match": match, "choice": choice}


def curve(policy, ids, scores, arms, on_k=5):
    """The policy at every threshold the data offers (each distinct nearest or
    k-th score), from all-off to all-on; ascending gated-on fraction."""
    grid = sorted({s for i in ids for s in scores.get(i, [])} | {float("inf")}, reverse=True)
    out = []
    for tau in grid:
        e = evaluate(policy, tau, ids, scores, arms, on_k)
        out.append({k: v for k, v in e.items() if k not in ("match", "choice")})
    return out


def pick_tau(cv, zero_pct, full_pct, keep=0.9):
    """The largest tau (fewest gated rows) keeping >= keep of the full gain."""
    target = zero_pct + keep * (full_pct - zero_pct)
    ok = [p for p in cv if p["exec_match_pct"] >= target]
    if not ok:
        return cv[-1]["tau"]
    return max(ok, key=lambda p: (p["tau"], -p["gated_on_pct"]))["tau"]      # inf = never retrieve


def fmt_tau(t):
    return "∞ (never)" if t == float("inf") else "%.1f" % t


def summarize_curve(cv, fractions=(0, 10, 25, 50, 75, 100)):
    """One point per requested gated-on fraction: the closest at or above it."""
    rows = []
    for f in fractions:
        cands = [p for p in cv if p["gated_on_pct"] >= f]
        if cands:
            rows.append(min(cands, key=lambda p: p["gated_on_pct"]))
    seen, out = set(), []
    for p in rows:
        if p["tau"] not in seen:
            seen.add(p["tau"]); out.append(p)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--read-split", default="test")
    ap.add_argument("--read-plans", default=None, help="plan folder for the read split (default: the run folder)")
    ap.add_argument("--pick-split", default=None, help="split the threshold is picked on (omit: descriptive only)")
    ap.add_argument("--pick-plans", default=None, help="plan folder for the pick split")
    ap.add_argument("--keep", type=float, default=0.9)
    ap.add_argument("--on-k", type=int, default=5, help="the retrieved arm of the on-off gate")
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    out = {"read_split": args.read_split, "pick_split": args.pick_split, "keep": args.keep, "policies": {}}
    lines = ["# Similarity-gated retrieval", "",
             "Offline: each row takes one arm's recorded answer by its BM25 scores. Threshold picked on "
             "`%s` (keep ≥ %.0f%% of the retrieved arm's gain over zero-shot with the fewest gated rows), read on `%s`."
             % (args.pick_split or "—", 100 * args.keep, args.read_split), ""]
    read_plans = pathlib.Path(args.read_plans) if args.read_plans else run

    def load_split(split, plans):
        arms = {a: load_rows(run, a, split) for a in K_ARMS.values()}
        ids = [i for i in arms["base-k0"] if all(i in arms[a] for a in arms)]
        return ids, load_scores(plans), arms

    r_ids, r_scores, r_arms = load_split(args.read_split, read_plans)
    if args.pick_split:
        p_ids, p_scores, p_arms = load_split(args.pick_split, pathlib.Path(args.pick_plans))
    for policy, on_k in (("on-off", args.on_k), ("on-off", 8), ("variable-k", None)):
        name = policy + ("" if on_k is None else " k=%d" % on_k)
        full_arm = K_ARMS[on_k] if on_k else "base-retr8"
        cv = curve(policy, r_ids, r_scores, r_arms, on_k or 5)
        zero = 100.0 * sum(r_arms["base-k0"][i]["match"] for i in r_ids) / len(r_ids)
        full = 100.0 * sum(r_arms[full_arm][i]["match"] for i in r_ids) / len(r_ids)
        rec = {"policy": policy, "on_k": on_k, "zero_shot_pct": zero, "full_pct": full,
               "curve": summarize_curve(cv)}
        lines += ["## %s (zero-shot %.1f%%, always-on %.1f%%)" % (name, zero, full), "",
                  "| gated on | τ | k (mean) | exec-match | prompt tok | tokens / solved |", "|---|---|---|---|---|---|"]
        for p in rec["curve"]:
            lines.append("| %.0f%% | %s | %.2f | %.1f%% | %.0f | %.0f |" % (
                p["gated_on_pct"], fmt_tau(p["tau"]), p["mean_k"], p["exec_match_pct"], p["prompt_tokens_mean"], p["tokens_per_solved"]))
        if args.pick_split:
            pcv = curve(policy, p_ids, p_scores, p_arms, on_k or 5)
            pz = 100.0 * sum(p_arms["base-k0"][i]["match"] for i in p_ids) / len(p_ids)
            pf = 100.0 * sum(p_arms[full_arm][i]["match"] for i in p_ids) / len(p_ids)
            tau = pick_tau(pcv, pz, pf, args.keep)
            picked_on_pick = evaluate(policy, tau, p_ids, p_scores, p_arms, on_k or 5)
            e = evaluate(policy, tau, r_ids, r_scores, r_arms, on_k or 5)
            zero_m = {i: r_arms["base-k0"][i]["match"] for i in r_ids}
            full_m = {i: r_arms[full_arm][i]["match"] for i in r_ids}
            g0, l0, _ = paired(zero_m, e["match"]); gf, lf, _ = paired(e["match"], full_m)
            tps_zero = sum(r_arms["base-k0"][i]["tokens"] for i in r_ids) / max(1, sum(zero_m.values()))
            tps_full = sum(r_arms[full_arm][i]["tokens"] for i in r_ids) / max(1, sum(full_m.values()))
            rec["picked"] = {"tau": tau, "pick_split": {"zero_shot_pct": pz, "full_pct": pf,
                             "gated_on_pct": picked_on_pick["gated_on_pct"], "exec_match_pct": picked_on_pick["exec_match_pct"]},
                             "read_split": {k: v for k, v in e.items() if k not in ("match", "choice")},
                             "vs_zero_shot": {"gained": g0, "lost": l0, "p": mcnemar_p(g0, l0)},
                             "always_on_vs_gated": {"gained": gf, "lost": lf, "p": mcnemar_p(gf, lf)},
                             "tokens_per_solved": {"zero_shot": tps_zero, "gated": e["tokens_per_solved"], "always_on": tps_full},
                             "gain_kept_pct": 100.0 * (e["exec_match_pct"] - zero) / (full - zero) if full > zero else None}
            lines += ["", "**Picked on %s:** τ = %s gates on %.0f%% there (%.1f%% → keeps the rule's share of %.1f − %.1f). "
                      "**Read on %s:** gated on %.0f%%, mean k %.2f, exec-match **%.1f%%** (zero-shot %.1f%%, always-on %.1f%%: "
                      "%.0f%% of the gain kept), prompt tokens %.0f (always-on %.0f), tokens per solved task %.0f "
                      "(zero-shot %.0f, always-on %.0f). Gated vs zero-shot +%d / −%d (p %s); always-on vs gated +%d / −%d (p %s)." % (
                          args.pick_split, fmt_tau(tau), picked_on_pick["gated_on_pct"], picked_on_pick["exec_match_pct"], pf, pz,
                          args.read_split, e["gated_on_pct"], e["mean_k"], e["exec_match_pct"], zero, full,
                          rec["picked"]["gain_kept_pct"] or 0, e["prompt_tokens_mean"],
                          statistics.fmean(r_arms[full_arm][i]["prompt"] for i in r_ids), e["tokens_per_solved"], tps_zero, tps_full,
                          g0, l0, "%.3f" % mcnemar_p(g0, l0), gf, lf, "%.3f" % mcnemar_p(gf, lf))]
        lines.append("")
        out["policies"][name] = rec
    text = "\n".join(lines)
    (run / "gate_report.md").write_text(text + "\n", encoding="utf-8")
    (run / "gate_report.json").write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
