#!/usr/bin/env python3
"""Approach 1's comparison (CPU): the 4B SFT on distilled targets against the
same recipe on the golds (`ds100`, 2026-10-16), with a rerun of that recipe as
the noise floor. The rules are in runs/distill_2026-10-31/preregistration.md.

The primary rows are the 295 val rows no run used for checkpoint selection
(data_scaling.selection_subset_ids is the 250 used); all 545 are secondary.
Arms that have not run yet are left out of the tables.

    python scripts/distill_targets/compare.py      # -> runs/distill_2026-10-31/report.{md,json}
"""
import argparse
import ast
import json
import pathlib
import random
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "retrieval", REPO / "scripts" / "data_scaling"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                  # noqa: E402
from retrieval import mcnemar_p, paired                          # noqa: E402

RUN = REPO / "runs" / "distill_2026-10-31"
VAL = REPO / "data" / "splits" / "val.jsonl"
ARMS = {
    "ds100": ("runs/qwen3.5-4b-sft-ds100", "SFT on the golds (the data-scaling run, 2026-10-16)"),
    "distill": ("runs/qwen3.5-4b-sft-distill", "SFT on the distilled targets"),
    "rerun": ("runs/qwen3.5-4b-sft-ds100-rerun", "the ds100 recipe again, same seed: the noise floor"),
}
TIERS = ("T1", "T2", "T3")
N_BOOT = 10000
BOOT_SEED = 20261031
ALPHA = 0.05
LOSS_RE = re.compile(r"\{'loss': [^{}]*\}")


def verdict(diff, p, alpha=ALPHA):
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
    p = mcnemar_p(g, l)
    return {"gained": g, "lost": l, "n": n, "diff_pts": round(diff, 1),
            "ci95": [round(boots[int(0.025 * n_boot)], 1), round(boots[int(0.975 * n_boot) - 1], 1)],
            "p": p, "verdict": verdict(diff, p)}


def reading(primary, noise):
    """The pre-registered verdict on approach 1."""
    if primary["verdict"] == "stronger" and (noise is None or primary["diff_pts"] > abs(noise["diff_pts"])):
        return "helps"
    if primary["verdict"] == "weaker":
        return "hurts"
    return "no detectable effect"


NEXT = {
    "helps": "The models' preferred forms teach better than the golds. Next: GRPO on top of the distilled SFT "
             "(the gated chain, several hours; ask first), and soft-target distillation with whichever teacher "
             "the 27B check names.",
    "hurts": "The preferred forms teach worse than the golds: keep the golds. Distillation would then need a "
             "teacher that is stronger, not just different (the 27B check's question).",
    "no detectable effect": "Re-targeting does not move SFT on this data. The remaining routes need a stronger "
                            "teacher (the 27B check decides whether one is on hand) or per-token feedback "
                            "(on-policy distillation).",
}


def train_curve(log):
    """The trainer's logged dicts ({'loss': '1.798', 'entropy': ...}) as floats."""
    if not pathlib.Path(log).exists():
        return []
    out = []
    for m in LOSS_RE.finditer(pathlib.Path(log).read_text(encoding="utf-8", errors="replace")):
        d = ast.literal_eval(m.group(0))
        out.append({k: float(v) for k, v in d.items() if isinstance(v, (int, float, str))
                    and re.fullmatch(r"[-+0-9.e]+", str(v))})
    return out


def fit(curve, last=5):
    tail = curve[-last:]
    keys = ("loss", "entropy", "mean_token_accuracy")
    return {k: round(statistics.fmean(c[k] for c in tail if k in c), 4) for k in keys
            if any(k in c for c in tail)} if tail else {}


def load_arm(folder):
    d = REPO / folder
    if not (d / "sft_val.json").exists():
        return None
    ex = {r["id"]: r for r in load_jsonl(str(d / "exec_val.jsonl"))}
    comp = {r["id"]: r for r in load_jsonl(str(d / "completions_val.jsonl"))}
    summary = json.loads((d / "sft_val.json").read_text(encoding="utf-8"))
    return {"match": {i: int(bool(r["match"])) for i, r in ex.items()},
            "valid": {i: int(bool(r["valid"])) for i, r in ex.items()},
            "comp": comp, "summary": summary, "fit": fit(train_curve(d / "train.log"))}


def pct(k, n):
    return round(100.0 * k / n, 1) if n else None


def untouched_ids(val):
    from data_scaling import selection_subset_ids
    sel = selection_subset_ids(val)
    return [c["id"] for c in val if c["id"] not in sel]


def build(args):
    val = load_jsonl(str(VAL))
    ids = [c["id"] for c in val]
    tier = {c["id"]: c["tier"] for c in val}
    clean = untouched_ids(val)
    arms = {k: load_arm(f) for k, (f, _) in ARMS.items()}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {}, "readings": {}}
    for k, a in arms.items():
        if a is None:
            rep["arms"][k] = {"status": "not run", "source": ARMS[k][0]}
            continue
        parsed = [r for r in a["comp"].values() if r["parse_ok"]]
        rep["arms"][k] = {
            "source": ARMS[k][0], "what": ARMS[k][1],
            "full_pct": pct(sum(a["match"][i] for i in ids), len(ids)),
            "untouched_pct": pct(sum(a["match"][i] for i in clean), len(clean)),
            "valid_pct": pct(sum(a["valid"][i] for i in ids), len(ids)),
            "out_tokens_mean": round(statistics.fmean(r["output_tokens"] for r in a["comp"].values()), 1),
            "ex_share_pct": pct(sum(r["mode"] == "ex" for r in parsed), len(parsed)),
            "best_step": (a["summary"].get("training") or {}).get("best_step"),
            "fit": a["fit"],
            "tiers": {t: pct(sum(a["match"][i] for i in ids if tier[i] == t), sum(tier[i] == t for i in ids))
                      for t in TIERS}}
    base = arms["ds100"]
    for name, other, rows in (("P", "distill", clean), ("P_full", "distill", ids),
                              ("N", "rerun", clean), ("N_full", "rerun", ids)):
        if base is not None and arms[other] is not None:
            rep["readings"][name] = compare(base["match"], arms[other]["match"], rows)
    if "P" in rep["readings"]:
        rep["verdict"] = reading(rep["readings"]["P"], rep["readings"].get("N"))
        rep["next_step"] = NEXT[rep["verdict"]]
    tpath = RUN / "targets_summary.json"
    if tpath.exists():
        rep["targets"] = json.loads(tpath.read_text(encoding="utf-8"))
    return rep


def fmt(r):
    return "+%d / −%d | %+.1f [%+.1f, %+.1f] | %.3g | **%s**" % (
        r["gained"], r["lost"], r["diff_pts"], r["ci95"][0], r["ci95"][1], r["p"], r["verdict"])


def render(rep):
    L = ["# Approach 1: SFT on distilled targets vs on the golds (val)", "",
         "Pre-registration: `preregistration.md` beside this file. Primary rows: the %d val rows never used "
         "for checkpoint selection; all %d are secondary." % (rep["n_untouched"], rep["n"]), ""]
    L += ["| arm | untouched val | full val | valid | out tokens | ex share | best step | final loss / entropy / token acc |",
          "|---|---|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        if a.get("status") == "not run":
            L.append("| %s | not run | | | | | | |" % k)
            continue
        f = a["fit"]
        L.append("| %s | **%s%%** | %s%% | %s%% | %s | %s%% | %s | %s / %s / %s |" % (
            k, a["untouched_pct"], a["full_pct"], a["valid_pct"], a["out_tokens_mean"], a["ex_share_pct"],
            a["best_step"], f.get("loss"), f.get("entropy"), f.get("mean_token_accuracy")))
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | comparison | rows | gained / lost | difference, points | p | verdict |",
          "|---|---|---|---|---|---|---|"]
    names = {"P": ("distill vs ds100", "untouched"), "P_full": ("distill vs ds100", "full"),
             "N": ("rerun vs ds100 (noise floor)", "untouched"), "N_full": ("rerun vs ds100 (noise floor)", "full")}
    for k, r in rep["readings"].items():
        L.append("| %s | %s | %s (%d) | %s |" % (k, names[k][0], names[k][1], r["n"], fmt(r)))
    if "verdict" in rep:
        L += ["", "**Verdict, by the pre-registered rule: %s.** %s" % (rep["verdict"], rep["next_step"])]
    L += ["", "## By tier (full val)", "", "| arm | " + " | ".join(TIERS) + " |", "|---|---|---|---|"]
    for k, a in rep["arms"].items():
        if "tiers" in a:
            L.append("| %s | %s |" % (k, " | ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    t = rep.get("targets")
    if t:
        L += ["", "## The distilled targets", "",
              "%d of %d train rows re-targeted (T1 %s, T2 %s, T3 %s); reasons %s; mode changes %s; candidate "
              "verdicts %s; mean length gold %s vs chosen %s characters." % (
                  t["replaced"], t["rows"], t["replaced_by_tier"].get("T1"), t["replaced_by_tier"].get("T2"),
                  t["replaced_by_tier"].get("T3"), t["reasons"], t["mode_changes"], t["verdicts"],
                  *t.get("chars_mean_gold_vs_choice", [None, None]))]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RUN))
    args = ap.parse_args(argv)
    rep = build(args)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "report.md").write_text(render(rep), encoding="utf-8")
    print(render(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
