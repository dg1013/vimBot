#!/usr/bin/env python3
"""GRPO on the distilled SFT, by Part B of runs/distill_followup_2026-10-31/preregistration.md: the
chain's final adapter against the served 4B (R, primary), round 1 against the served chain's
round 1 (R1), what RL adds, tokens, and the test v2 rows. CPU only: reads the runs' logs, writes
grpo.md and grpo.json beside the pre-registration. Arms not finished yet are reported as missing.

    python scripts/distill_followup/report_grpo.py
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("distill_targets_compare",
                                               REPO / "scripts" / "distill_targets" / "compare.py")
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)      # compare(), untouched_ids(), pct(), load_jsonl: one copy of each

RUN = REPO / "runs" / "distill_followup_2026-10-31"
TEST = REPO / "data" / "splits" / "test.jsonl"
ARMS = {    # name: (run folder, its summary file, what it is)
    "distill-grpo-gated": ("runs/qwen3.5-4b-distill-grpo-gated", "grpo_val.json", "distilled SFT → GRPO → gated GRPO"),
    "distill-grpo": ("runs/qwen3.5-4b-distill-grpo", "grpo_val.json", "distilled SFT → GRPO (round 1)"),
    "served": ("runs/qwen3.5-4b-grpo-gated-leg2", "grpo_val.json", "the served 4B: golds SFT → GRPO → gated GRPO"),
    "served-round1": ("runs/qwen3.5-4b-grpo", "grpo_val.json", "the served chain's round 1 (before the fla kernels)"),
    "distill-sft": ("runs/qwen3.5-4b-sft-distill", "sft_val.json", "the distilled SFT (approach 1, seed 42)"),
    "golds-sft": ("runs/qwen3.5-4b-sft-ds100", "sft_val.json", "golds SFT, the round-1 recipe (ds100)"),
}
TEST_COMPARATOR = REPO / "runs" / "remeasure_2026-09-25" / "v2" / "qwen3.5-4b-grpo-gated__fla-kernels"
READINGS = [   # (name, comparator, arm, question)
    ("R", "served", "distill-grpo-gated", "the chain's final adapter against the served 4B (primary)"),
    ("R1", "served-round1", "distill-grpo", "round 1 against the served chain's round 1 (descriptive)"),
    ("RL gain", "distill-sft", "distill-grpo-gated", "what RL adds on top of the distilled SFT"),
    ("served RL gain", "golds-sft", "served", "what RL added on top of golds SFT"),
]
TIERS = ("T1", "T2", "T3")


def load_run(folder, summary_name, split="val"):
    d = REPO / folder
    ex, summ = d / ("exec_%s.jsonl" % split), d / (summary_name if split == "val" else "baseline_test.json")
    if not ex.exists() or not summ.exists():
        return None
    rows = C.load_jsonl(str(ex))
    return {"match": {r["id"]: int(bool(r["match"])) for r in rows},
            "valid": {r["id"]: int(bool(r["valid"])) for r in rows},
            "summary": json.loads(summ.read_text(encoding="utf-8"))}


def best_step(folder):
    hist = REPO / folder / "eval_history.jsonl"
    if not hist.exists():
        return None
    evals = [json.loads(line) for line in hist.open(encoding="utf-8") if line.strip()]
    best = max(evals, key=lambda e: (e["exec_match_pct"], -e["step"])) if evals else None
    return {"best_step": best["step"], "best_subset_pct": best["exec_match_pct"],
            "step0_subset_pct": next((e["exec_match_pct"] for e in evals if e["step"] == 0), None),
            "evals": len(evals)} if best else None


def build():
    val = C.load_jsonl(str(C.VAL))
    ids = [c["id"] for c in val]
    clean = C.untouched_ids(val)
    tier = {c["id"]: c["tier"] for c in val}
    runs = {k: load_run(f, s) for k, (f, s, _) in ARMS.items()}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {}, "readings": {}, "test": {}}
    for k, (folder, _, what) in ARMS.items():
        r = runs[k]
        if r is None:
            rep["arms"][k] = {"status": "missing", "source": folder, "what": what}
            continue
        met = r["summary"].get("metrics", {})
        sel = best_step(folder)
        trained = r["summary"].get("training") or {}
        if sel and trained.get("best_step") is not None:     # the trainer's own selection wins
            sel.update(best_step=trained["best_step"],
                       best_subset_pct=trained.get("best_subset_exec_match_pct", sel["best_subset_pct"]))
        rep["arms"][k] = {"source": folder, "what": what,
                          "untouched_pct": C.pct(sum(r["match"][i] for i in clean), len(clean)),
                          "full_pct": C.pct(sum(r["match"][i] for i in ids), len(ids)),
                          "valid_pct": met.get("valid_pct"), "tokens": met.get("output_tokens_mean"),
                          "selection": sel,
                          "tiers": {t: C.pct(sum(r["match"][i] for i in ids if tier[i] == t),
                                             sum(tier[i] == t for i in ids)) for t in TIERS}}
    for name, a, b, question in READINGS:
        if runs[a] and runs[b]:
            rep["readings"][name] = {"question": question, "comparator": a, "arm": b,
                                     "untouched": C.compare(runs[a]["match"], runs[b]["match"], clean),
                                     "full": C.compare(runs[a]["match"], runs[b]["match"], ids)}
    r = rep["readings"].get("R")
    rep["verdict"] = r["untouched"]["verdict"] if r else None
    # Test v2: reporting only.
    test_ids = [c["id"] for c in C.load_jsonl(str(TEST))]
    served_rows = C.load_jsonl(str(TEST_COMPARATOR / "exec_test.jsonl"))
    served = {x["id"]: int(bool(x["match"])) for x in served_rows}
    rep["test"]["served"] = C.pct(sum(served.values()), len(served))
    for k in ("distill-grpo", "distill-grpo-gated"):
        t = load_run(ARMS[k][0], None, split="test")
        if t:
            rep["test"][k] = {"pct": C.pct(sum(t["match"][i] for i in test_ids), len(test_ids)),
                              "tokens": t["summary"].get("metrics", {}).get("output_tokens_mean"),
                              "vs_served": C.compare(served, t["match"], test_ids)}
    return rep


MEANING = {"stronger": "the distilled chain's adapter is the candidate to replace the served 4B; the swap is the owner's call.",
           "tied": "the served adapter stays.", "weaker": "the served adapter stays."}


def fmt(r):
    return "%+.1f [%+.1f, %+.1f], +%d / −%d, p %.3g → %s" % (
        r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["p"], r["verdict"])


def render(rep):
    L = ["# GRPO on the distilled SFT (val; test v2 for reporting)", "",
         "Part B of `preregistration.md` beside this file; built by `scripts/distill_followup/report_grpo.py`. "
         "Primary rows: the %d never used for checkpoint selection; all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| arm | untouched val | full val | valid | tokens | best step (subset; step 0) | T1 / T2 / T3 |",
         "|---|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        if a.get("status") == "missing":
            L.append("| `%s` (%s) | missing | | | | | |" % (k, a["what"]))
            continue
        s = a["selection"]
        sel = "%s (%s%%; %s)" % (s["best_step"], s["best_subset_pct"], "—" if s["step0_subset_pct"] is None
                                 else "%s%%" % s["step0_subset_pct"]) if s else "—"
        L.append("| `%s` (%s) | **%s%%** | %s%% | %s%% | %s | %s | %s |" % (
            k, a["what"], a["untouched_pct"], a["full_pct"], a["valid_pct"], a["tokens"], sel,
            " / ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | question | untouched (primary) | all val |", "|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | %s | %s | %s |" % (name, r["question"], fmt(r["untouched"]), fmt(r["full"])))
    if rep["verdict"]:
        L += ["", "**R, by the pre-registered rule: %s** — %s" % (rep["verdict"], MEANING[rep["verdict"]])]
    L += ["", "## Test v2 (reporting only)", "", "| adapter | test v2 | tokens | against the served 4B (%s%%) |"
          % rep["test"]["served"], "|---|---|---|---|"]
    for k in ("distill-grpo", "distill-grpo-gated"):
        t = rep["test"].get(k)
        L.append("| `%s` | %s | %s | %s |" % (k, ("%s%%" % t["pct"]) if t else "missing",
                                            t["tokens"] if t else "", fmt(t["vs_served"]) if t else ""))
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (RUN / "grpo.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "grpo.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
