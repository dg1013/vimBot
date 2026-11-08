#!/usr/bin/env python3
"""How close to one-hot the trained 27B is on its own training rows: the readings and the rule of
runs/teacher_probs_2026-10-31/preregistration.md, from record.py's probs.jsonl. CPU only; writes
analysis.md and analysis.json beside it.

    python scripts/teacher_probs/analyze.py
"""
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = REPO / "runs" / "teacher_probs_2026-10-31"
TRAIN = REPO / "runs" / "distill_2026-10-31" / "train_distilled.jsonl"
TOP2_MIN, Q2_MIN = 0.50, 0.02          # informative: both
TOP5_MAX, Q2_FLOOR = 0.25, 0.005       # near one-hot: either
BRANCH = 0.10                          # off-target mass that makes a token a branch point
TIERS = ("T1", "T2", "T3")


def q_topk(logits, ids, token, temperature):
    """A token's probability in the top-k renormalized at `temperature` (0 outside the k)."""
    if token not in ids:
        return 0.0
    m = max(logits)
    w = [math.exp((z - m) / temperature) for z in logits]
    return w[ids.index(token)] / sum(w)


def verdict(top2, top5, q2_median):
    if top2 >= TOP2_MIN and q2_median >= Q2_MIN:
        return "informative"
    if top5 < TOP5_MAX or q2_median < Q2_FLOOR:
        return "near one-hot"
    return "marginal"


def token_stats(recs, answer_only=True):
    """Peakedness over answer tokens (before <|im_end|>) or over every completion token."""
    off1, off2, ent, not_top, nll, rows_branch = [], [], [], [], [], 0
    for r in recs:
        n = r["n_answer"] if answer_only else r["n"]
        branch = False
        for j in range(n):
            o1 = 1.0 - math.exp(r["tlp1"][j])
            off1.append(o1)
            off2.append(1.0 - q_topk(r["topk_logits"][j], r["topk_ids"][j], r["ids"][j], 2.0))
            ent.append(r["ent1"][j])
            not_top.append(r["trank"][j] > 1)
            nll.append(-r["tlp1"][j])
            branch = branch or o1 >= BRANCH
        rows_branch += branch
    mean = lambda xs: round(statistics.fmean(xs), 4) if xs else None
    return {"tokens": len(off1), "m1": mean(off1), "m2_top20": mean(off2), "entropy_t1": mean(ent),
            "branch_share": mean([o >= BRANCH for o in off1]), "rows_with_branch": round(rows_branch / len(recs), 4),
            "target_not_top": mean(not_top), "mean_nll": mean(nll)}


def alt_stats(recs):
    """The alternative probe on rows whose original gold departs from the target."""
    probed = [r for r in recs if r.get("gold") and r["gold"].get("d") is not None]
    if not probed:
        return {"rows": 0}
    ranks = [r["gold"]["rank"] for r in probed]
    q2 = [r["gold"]["q2"] for r in probed]
    p1 = [math.exp(r["gold"]["lp1"]) for r in probed]
    out = {"rows": len(probed),
           "top2": round(sum(k <= 2 for k in ranks) / len(ranks), 4),
           "top5": round(sum(k <= 5 for k in ranks) / len(ranks), 4),
           "rank1": round(sum(k == 1 for k in ranks) / len(ranks), 4),
           "q2_median": round(statistics.median(q2), 5),
           "p1_median": statistics.median(p1),
           "target_q2_median": round(statistics.median(r["gold"]["target_q2"] for r in probed), 5)}
    out["verdict"] = verdict(out["top2"], out["top5"], out["q2_median"])
    return out


def build(recs, tier):
    retargeted = [r for r in recs if r.get("gold")]
    groups = {"all re-targeted": retargeted,
              "mode switches": [r for r in retargeted if r["gold"].get("mode_switch")],
              "within mode": [r for r in retargeted if not r["gold"].get("mode_switch")]}
    for t in TIERS:
        groups[t] = [r for r in retargeted if tier.get(r["id"]) == t]
    rep = {"rows": len(recs), "retargeted": len(retargeted),
           "no_divergence": sum(1 for r in retargeted if r["gold"].get("d") is None),
           "alternative": {name: alt_stats(g) for name, g in groups.items()},
           "peakedness": {"answer tokens": token_stats(recs, True), "all completion tokens": token_stats(recs, False)}}
    rep["verdict"] = rep["alternative"]["all re-targeted"].get("verdict")
    return rep


MEANING = {
    "informative": "a T = 2 top-20 KD target would teach valid alternatives the hard targets lack: the 4B "
                   "KD run gets its own pre-registration and its trainer is built.",
    "near one-hot": "the teacher has collapsed onto its targets: no KD trainer for this teacher on these "
                    "rows; the alternatives are a cross-fitted teacher or no logit KD.",
    "marginal": "neither rule holds: reported, and the owner decides.",
}


def render(rep, meta):
    L = ["# How close to one-hot is the trained 27B on its own training rows?", "",
         "Pre-registration: `preregistration.md` beside this file. Built by `scripts/teacher_probs/analyze.py` "
         "from `probs.jsonl` (%s rows; the batched-vs-single guard moved log-probabilities by at most %s)."
         % (rep["rows"], meta.get("guard_max_abs_dlogp", "?")), "",
         "## A: the alternative probe (primary)", "",
         "On the %d rows approach 1 re-targeted, the original gold is a verified alternative. At the first token "
         "where its completion departs from the target's (%d rows had no such token):"
         % (rep["retargeted"], rep["no_divergence"]), "",
         "| rows | n | gold token in the top 2 | in the top 5 | ranked first | median q₂ (top 20, T = 2) | median p (T = 1) | target's median q₂ |",
         "|---|---|---|---|---|---|---|---|"]
    for name, a in rep["alternative"].items():
        if not a.get("rows"):
            L.append("| %s | 0 | | | | | | |" % name)
            continue
        L.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% | %.4f | %.2g | %.4f |"
                 % (name, a["rows"], 100 * a["top2"], 100 * a["top5"], 100 * a["rank1"], a["q2_median"],
                    a["p1_median"], a["target_q2_median"]))
    v = rep["verdict"]
    L += ["", "**Verdict, by the pre-registered rule: %s** — %s" % (v, MEANING.get(v, "")), "",
          "## Peakedness (descriptive)", "",
          "| | answer tokens | all completion tokens |", "|---|---|---|"]
    pa, pc = rep["peakedness"]["answer tokens"], rep["peakedness"]["all completion tokens"]
    for key, label in (("tokens", "tokens"), ("m1", "m₁: off-target probability, T = 1, full vocabulary"),
                       ("m2_top20", "m₂: off-target probability, top 20 at T = 2"),
                       ("entropy_t1", "entropy at T = 1 (nats)"),
                       ("branch_share", "share with off-target mass ≥ %.2f at T = 1" % BRANCH),
                       ("rows_with_branch", "rows with such a token"),
                       ("target_not_top", "target not the teacher's top token"),
                       ("mean_nll", "mean −log p(target) (training's last loss: 0.051)")):
        L.append("| %s | %s | %s |" % (label, pa[key], pc[key]))
    return "\n".join(L) + "\n"


def main():
    probs = RUN / "probs.jsonl"
    if not probs.exists():
        sys.exit("no %s yet: run record.py first" % probs)
    recs = [json.loads(line) for line in probs.open(encoding="utf-8") if line.strip()]
    tier = {json.loads(line)["id"]: json.loads(line)["tier"] for line in TRAIN.open(encoding="utf-8") if line.strip()}
    meta_path = RUN / "record_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    rep = build(recs, tier)
    (RUN / "analysis.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep, meta)
    (RUN / "analysis.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
