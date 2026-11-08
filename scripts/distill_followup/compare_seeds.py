#!/usr/bin/env python3
"""Approach 1 at a second seed, by Part A of runs/distill_followup_2026-10-31/preregistration.md:
distilled targets against golds at seed 7, the two seeds pooled, and the seed-to-seed spread. CPU
only: reads the arms' exec logs, writes seeds.md and seeds.json beside the pre-registration.

    python scripts/distill_followup/compare_seeds.py
"""
import importlib.util
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

# Approach 1's comparison module by path, under its own name (other folders have a compare.py).
_spec = importlib.util.spec_from_file_location("distill_targets_compare",
                                               REPO / "scripts" / "distill_targets" / "compare.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)      # compare(), load_arm(), untouched_ids(), pct(): one copy of each

RUN = REPO / "runs" / "distill_followup_2026-10-31"
ARMS = {
    "golds-s42": "runs/qwen3.5-4b-sft-ds100",
    "distill-s42": "runs/qwen3.5-4b-sft-distill",
    "golds-s7": "runs/qwen3.5-4b-sft-ds100-s7",
    "distill-s7": "runs/qwen3.5-4b-sft-distill-s7",
    "rerun-s42": "runs/qwen3.5-4b-sft-ds100-rerun",
}
TIERS = ("T1", "T2", "T3")


def call(s):
    """Part A's verdict from the primary reading S."""
    if s["verdict"] == "stronger":
        return "replicates"
    return "direction holds" if s["diff_pts"] > 0 else "does not replicate"


def pooled(g42, d42, g7, d7, ids, n_boot=C.N_BOOT, seed=C.BOOT_SEED):
    """Each row's mean of the two seeds' paired differences, with a paired bootstrap interval."""
    d = [((d42[i] - g42[i]) + (d7[i] - g7[i])) / 2.0 for i in ids]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(100.0 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"diff_pts": round(100.0 * sum(d) / n, 1),
            "ci95": [round(boots[int(0.025 * n_boot)], 1), round(boots[int(0.975 * n_boot) - 1], 1)]}


def build():
    val = C.load_jsonl(str(C.VAL))
    ids = [c["id"] for c in val]
    tier = {c["id"]: c["tier"] for c in val}
    clean = C.untouched_ids(val)
    arms = {k: C.load_arm(f) for k, f in ARMS.items()}
    missing = [k for k in ("golds-s7", "distill-s7", "golds-s42", "distill-s42") if arms[k] is None]
    if missing:
        sys.exit("arms without sft_val.json: %s" % ", ".join(missing))
    m = {k: a["match"] for k, a in arms.items() if a}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {}, "readings": {}}
    for k, a in arms.items():
        if a is None:
            continue
        rep["arms"][k] = {"source": ARMS[k],
                          "untouched_pct": C.pct(sum(a["match"][i] for i in clean), len(clean)),
                          "full_pct": C.pct(sum(a["match"][i] for i in ids), len(ids)),
                          "valid_pct": C.pct(sum(a["valid"][i] for i in ids), len(ids)),
                          "best_step": (a["summary"].get("training") or {}).get("best_step"),
                          "fit": a["fit"],
                          "tiers": {t: C.pct(sum(a["match"][i] for i in ids if tier[i] == t),
                                             sum(tier[i] == t for i in ids)) for t in TIERS}}
    pairs = {"S": ("golds-s7", "distill-s7"), "seed 42 (the night's P)": ("golds-s42", "distill-s42"),
             "golds: seed 7 vs 42": ("golds-s42", "golds-s7"),
             "distilled: seed 7 vs 42": ("distill-s42", "distill-s7")}
    if "rerun-s42" in m:
        pairs["golds: rerun vs seed 42 (kernel noise)"] = ("golds-s42", "rerun-s42")
    for name, (a, b) in pairs.items():
        rep["readings"][name] = {"untouched": C.compare(m[a], m[b], clean), "full": C.compare(m[a], m[b], ids)}
    rep["pooled"] = {part: pooled(m["golds-s42"], m["distill-s42"], m["golds-s7"], m["distill-s7"], sub)
                     for part, sub in (("untouched", clean), ("full", ids))}
    rep["verdict"] = call(rep["readings"]["S"]["untouched"])
    return rep


MEANING = {"replicates": "the distilled file becomes the default SFT data for the 4B.",
           "direction holds": "the golds stay the default; the pooled estimate is the number to quote.",
           "does not replicate": "the golds stay the default; seed 42's result is reported with seed 7 beside it."}


def fmt(r):
    return "%+.1f [%+.1f, %+.1f], +%d / −%d, p %.3g → %s" % (
        r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["p"], r["verdict"])


def render(rep):
    L = ["# Approach 1 at a second seed (val)", "",
         "Part A of `preregistration.md` beside this file; built by `scripts/distill_followup/compare_seeds.py`. "
         "Primary rows: the %d never used for checkpoint selection; all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| arm | untouched val | full val | valid | best step | final loss / entropy / token accuracy | T1 / T2 / T3 |",
         "|---|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        f = a["fit"]
        L.append("| `%s` | **%s%%** | %s%% | %s%% | %s | %s / %s / %s | %s |" % (
            k, a["untouched_pct"], a["full_pct"], a["valid_pct"], a["best_step"], f.get("loss"), f.get("entropy"),
            f.get("mean_token_accuracy"), " / ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | untouched (primary) | all val |", "|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | %s | %s |" % (name, fmt(r["untouched"]), fmt(r["full"])))
    pu, pf = rep["pooled"]["untouched"], rep["pooled"]["full"]
    L += ["| the two seeds pooled (bootstrap only) | %+.1f [%+.1f, %+.1f] | %+.1f [%+.1f, %+.1f] |"
          % (pu["diff_pts"], pu["ci95"][0], pu["ci95"][1], pf["diff_pts"], pf["ci95"][0], pf["ci95"][1]), "",
          "**Verdict, by the pre-registered rule: %s** — %s" % (rep["verdict"], MEANING[rep["verdict"]])]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (RUN / "seeds.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "seeds.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
