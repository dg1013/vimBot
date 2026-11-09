#!/usr/bin/env python3
"""Soft-target KD from the trained 27B against approach 1's hard targets, by
runs/kd27b_2026-10-31/preregistration.md: the two seeds pooled (primary), each seed, the fit. CPU
only: reads the arms' exec logs, writes kd.md and kd.json beside the pre-registration. Works on
partial results (the verdict waits for both KD arms).

    python scripts/kd27b/compare_kd.py
"""
import importlib.util
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# One copy of each: approach 1's compare(), load_arm(), untouched_ids(), pct(), train_curve(), and
# the second seed's pooled bootstrap.
F = load("distill_followup_compare_seeds", REPO / "scripts" / "distill_followup" / "compare_seeds.py")
C = F.C

RUN = REPO / "runs" / "kd27b_2026-10-31"
ARMS = {
    "kd-s42": "runs/qwen3.5-4b-kd27b-s42",
    "kd-s7": "runs/qwen3.5-4b-kd27b-s7",
    "distill-s42": "runs/qwen3.5-4b-sft-distill",
    "distill-s7": "runs/qwen3.5-4b-sft-distill-s7",
    "golds-s42": "runs/qwen3.5-4b-sft-ds100",
    "golds-s7": "runs/qwen3.5-4b-sft-ds100-s7",
}
TIERS = ("T1", "T2", "T3")
FIT_KEYS = ("loss", "nll", "kd_kl", "entropy", "mean_token_accuracy")


def call(pooled, s42, s7):
    """K's verdict: the pooled interval clear of zero AND both seeds on the same side."""
    lo, hi = pooled["ci95"]
    if lo > 0 and s42["diff_pts"] > 0 and s7["diff_pts"] > 0:
        return "helps"
    if hi < 0 and s42["diff_pts"] < 0 and s7["diff_pts"] < 0:
        return "hurts"
    return "tied"


def fit(folder, last=5):
    tail = C.train_curve(REPO / folder / "train.log")[-last:]
    return {k: round(statistics.fmean(c[k] for c in tail if k in c), 4) for k in FIT_KEYS
            if any(k in c for c in tail)}


def build():
    val = C.load_jsonl(str(C.VAL))
    ids = [c["id"] for c in val]
    tier = {c["id"]: c["tier"] for c in val}
    clean = C.untouched_ids(val)
    arms = {k: C.load_arm(f) for k, f in ARMS.items()}
    need = [k for k in ("distill-s42", "distill-s7", "golds-s42", "golds-s7") if arms[k] is None]
    if need:
        sys.exit("comparison arms without sft_val.json: %s" % ", ".join(need))
    m = {k: a["match"] for k, a in arms.items() if a}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {}, "readings": {}, "pooled": {}}
    for k, a in arms.items():
        if a is None:
            continue
        ex = [i for i in ids if (a["comp"].get(i) or {}).get("mode") == "ex"]
        rep["arms"][k] = {"source": ARMS[k],
                          "untouched_pct": C.pct(sum(a["match"][i] for i in clean), len(clean)),
                          "full_pct": C.pct(sum(a["match"][i] for i in ids), len(ids)),
                          "valid_pct": C.pct(sum(a["valid"][i] for i in ids), len(ids)),
                          "ex_share_pct": C.pct(len(ex), len(ids)),
                          "tokens": (a["summary"].get("metrics") or {}).get("output_tokens_mean"),
                          "best_step": (a["summary"].get("training") or {}).get("best_step"),
                          "fit": fit(ARMS[k]),
                          "tiers": {t: C.pct(sum(a["match"][i] for i in ids if tier[i] == t),
                                             sum(tier[i] == t for i in ids)) for t in TIERS}}
    pairs = {"KD vs hard targets, seed 42": ("distill-s42", "kd-s42"),
             "KD vs hard targets, seed 7": ("distill-s7", "kd-s7"),
             "KD vs golds, seed 42": ("golds-s42", "kd-s42"),
             "KD vs golds, seed 7": ("golds-s7", "kd-s7")}
    for name, (a, b) in pairs.items():
        if a in m and b in m:
            rep["readings"][name] = {"untouched": C.compare(m[a], m[b], clean), "full": C.compare(m[a], m[b], ids)}
    both = "kd-s42" in m and "kd-s7" in m
    if both:
        for name, base in (("K: KD vs hard targets, two seeds pooled", "distill"),
                           ("KD vs golds, two seeds pooled", "golds")):
            rep["pooled"][name] = {part: F.pooled(m[base + "-s42"], m["kd-s42"], m[base + "-s7"], m["kd-s7"], sub)
                                   for part, sub in (("untouched", clean), ("full", ids))}
        r = rep["readings"]
        rep["verdict"] = call(rep["pooled"]["K: KD vs hard targets, two seeds pooled"]["untouched"],
                              r["KD vs hard targets, seed 42"]["untouched"],
                              r["KD vs hard targets, seed 7"]["untouched"])
    else:
        rep["verdict"] = "pending (both KD arms are needed)"
    return rep


MEANING = {"helps": "KD becomes the 4B's SFT recipe for this data; GRPO on the KD adapter and a test v2 row "
                    "are the owner's call.",
           "tied": "the hard targets stay; KD from this teacher is not worth its cost here.",
           "hurts": "the hard targets stay; reported."}


def render(rep):
    L = ["# Soft-target KD from the trained 27B (val)", "",
         "By `preregistration.md` beside this file; built by `scripts/kd27b/compare_kd.py`. Primary rows: the %d "
         "never used for checkpoint selection; all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| arm | untouched val | full val | valid | ex share | tokens | best step | fit (last 5 logs) | T1 / T2 / T3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        fit_s = ", ".join("%s %s" % (key, v) for key, v in a["fit"].items())
        L.append("| `%s` | **%s%%** | %s%% | %s%% | %s%% | %s | %s | %s | %s |" % (
            k, a["untouched_pct"], a["full_pct"], a["valid_pct"], a["ex_share_pct"], a["tokens"],
            a["best_step"], fit_s, " / ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    L += ["", "The KD arms' `loss` is the mixed loss; `nll` is the term comparable with the other arms' `loss`.",
          "", "## Readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | untouched (primary rows) | all val |", "|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | %s | %s |" % (name, F.fmt(r["untouched"]), F.fmt(r["full"])))
    for name, p in rep["pooled"].items():
        pu, pf = p["untouched"], p["full"]
        L.append("| %s (bootstrap) | %+.1f [%+.1f, %+.1f] | %+.1f [%+.1f, %+.1f] |" % (
            name, pu["diff_pts"], pu["ci95"][0], pu["ci95"][1], pf["diff_pts"], pf["ci95"][0], pf["ci95"][1]))
    v = rep["verdict"]
    L += ["", "**Verdict, by the pre-registered rule: %s**%s" % (v, " — " + MEANING[v] if v in MEANING else "")]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "kd.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "kd.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
