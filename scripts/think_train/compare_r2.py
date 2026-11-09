#!/usr/bin/env python3
"""R2's report (CPU), by runs/think_train_2026-11-08/preregistration.md: P1 (A2 against A1, the
reasoning task) and P2 (A1 against A0, the thinking teacher's answers as rows), each per seed and
pooled over the two seeds, judged by the plan's shared rule; A2 against A0, the fit, any thinking in
the answers, and R1's reading L, descriptive. Reads the arms' exec logs and writes r2.md and r2.json
beside the pre-registration. Works on partial results: a verdict waits for both seeds of its arms.

    python scripts/think_train/compare_r2.py
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


# One copy of each: approach 1's compare(), load_arm(), untouched_ids(), pct(), train_curve(), the
# second seed's pooled bootstrap, and KD's verdict rule (the plan's shared rule).
F = load("distill_followup_compare_seeds", REPO / "scripts" / "distill_followup" / "compare_seeds.py")
C = F.C
K = load("kd27b_compare_kd", REPO / "scripts" / "kd27b" / "compare_kd.py")
call = K.call

RUN = REPO / "runs" / "think_train_2026-11-08"
R1 = REPO / "runs" / "think_label_2026-11-01"
ARMS = {
    "a2-s42": "runs/qwen3.5-4b-r2-a2-s42",
    "a2-s7": "runs/qwen3.5-4b-r2-a2-s7",
    "a1-s42": "runs/qwen3.5-4b-r2-a1-s42",
    "a1-s7": "runs/qwen3.5-4b-r2-a1-s7",
    "a0-s42": "runs/qwen3.5-4b-sft-distill",
    "a0-s7": "runs/qwen3.5-4b-sft-distill-s7",
}
TIERS = ("T1", "T2", "T3")
FIT_KEYS = ("loss", "entropy", "mean_token_accuracy", "trace_nll", "trace_tokens")
READINGS = {"P1": ("a1", "a2", "the reasoning task: A2 against A1 (primary)"),
            "P2": ("a0", "a1", "the thinking teacher's answers as rows: A1 against A0"),
            "A2 vs A0": ("a0", "a2", "both together: A2 against A0 (descriptive)")}
MEANING = {
    "P1": {"helps": "reasoning is a training signal here. Traces on more rows, GRPO on top, and a standing "
                    "exception to §4 in DECISIONS are the owner's call.",
           "tied": "reasoning on about 5% of the rows does not move the 4B. That does not show that traces on most "
                   "rows would fail; testing that would take about ten times R1.",
           "hurts": "reported, with the check for thinking leaking into answers below."},
    "P2": {"helps": "the thinking labels are useful data without the reasoning, the cheaper recipe.",
           "tied": "the thinking teacher's answers as extra rows do not move the 4B.",
           "hurts": "reported, with the check for thinking leaking into answers below."},
}


def fit(folder, last=5):
    path = REPO / folder / "train.log"
    if not path.exists():
        return {}
    tail = C.train_curve(path)[-last:]
    return {k: round(statistics.fmean(c[k] for c in tail if k in c), 4) for k in FIT_KEYS
            if any(k in c for c in tail)}


def thinking_in_answers(arm):
    comp = arm["comp"].values()
    return {"think_leaks": sum(bool(r.get("think_leak")) for r in comp),
            "think_text": sum("<think>" in (r.get("completion") or "") or "</think>" in (r.get("completion") or "")
                              for r in comp),
            "parse_failures": sum(not r.get("parse_ok", True) for r in comp)}


def build():
    val = C.load_jsonl(str(C.VAL))
    ids = [c["id"] for c in val]
    tier = {c["id"]: c["tier"] for c in val}
    clean = C.untouched_ids(val)
    arms = {k: C.load_arm(f) for k, f in ARMS.items()}
    need = [k for k in ("a0-s42", "a0-s7") if arms[k] is None]
    if need:
        sys.exit("approach 1's arms without sft_val.json: %s" % ", ".join(need))
    m = {k: a["match"] for k, a in arms.items() if a}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {}, "readings": {}, "pooled": {}, "verdicts": {}}
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
                          "fit": fit(ARMS[k]), "thinking": thinking_in_answers(a),
                          "tiers": {t: C.pct(sum(a["match"][i] for i in ids if tier[i] == t),
                                             sum(tier[i] == t for i in ids)) for t in TIERS}}
    for name, (base, arm, _) in READINGS.items():
        for seed in ("s42", "s7"):
            a, b = "%s-%s" % (base, seed), "%s-%s" % (arm, seed)
            if a in m and b in m:
                rep["readings"]["%s, seed %s" % (name, seed[1:])] = {
                    "untouched": C.compare(m[a], m[b], clean), "full": C.compare(m[a], m[b], ids)}
        keys = ["%s-%s" % (x, s) for x in (base, arm) for s in ("s42", "s7")]
        if all(k in m for k in keys):
            pooled = {part: F.pooled(m[base + "-s42"], m[arm + "-s42"], m[base + "-s7"], m[arm + "-s7"], sub)
                      for part, sub in (("untouched", clean), ("full", ids))}
            rep["pooled"][name] = pooled
            r = rep["readings"]
            rep["verdicts"][name] = call(pooled["untouched"], r["%s, seed 42" % name]["untouched"],
                                         r["%s, seed 7" % name]["untouched"])
        else:
            rep["verdicts"][name] = "pending (both seeds of both arms are needed)"
    for path, key in ((RUN / "build.json", "build"), (RUN / "check" / "check.json", "check")):
        if path.exists():
            rep[key] = json.loads(path.read_text(encoding="utf-8"))
    r1 = R1 / "report.json"
    if r1.exists():
        r1rep = json.loads(r1.read_text(encoding="utf-8"))
        rep["r1"] = {"verdict": r1rep.get("verdict"), "L": (r1rep.get("readings") or {}).get("L"),
                     "labels": {a: s.get("labels") for a, s in (r1rep.get("arms") or {}).items()}}
    return rep


def render(rep):
    L = ["# R2: the thinking teacher's labels and reasoning as training signals (val)", "",
         "By `preregistration.md` beside this file; built by `scripts/think_train/compare_r2.py`. Primary rows: the "
         "%d never used for checkpoint selection; all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| arm | untouched val | full val | valid | ex share | tokens | best step | fit (last 5 logs) | "
         "thinking in answers | T1 / T2 / T3 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        fit_s = ", ".join("%s %s" % (key, v) for key, v in a["fit"].items())
        th = a["thinking"]
        L.append("| `%s` | **%s%%** | %s%% | %s%% | %s%% | %s | %s | %s | %d leaks, %d with think tags | %s |" % (
            k, a["untouched_pct"], a["full_pct"], a["valid_pct"], a["ex_share_pct"], a["tokens"], a["best_step"],
            fit_s, th["think_leaks"], th["think_text"], " / ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    L += ["", "A2's `loss` is the answer loss alone, comparable with the other arms'; `trace_nll` is the "
          "reasoning task's.", "", "## Readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | untouched (primary rows) | all val |", "|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | %s | %s |" % (name, F.fmt(r["untouched"]), F.fmt(r["full"])))
    for name, p in rep["pooled"].items():
        pu, pf = p["untouched"], p["full"]
        L.append("| %s: two seeds pooled (bootstrap) | %+.1f [%+.1f, %+.1f] | %+.1f [%+.1f, %+.1f] |" % (
            name, pu["diff_pts"], pu["ci95"][0], pu["ci95"][1], pf["diff_pts"], pf["ci95"][0], pf["ci95"][1]))
    L.append("")
    for name in ("P1", "P2"):
        v = rep["verdicts"].get(name)
        mean = MEANING[name].get(v)
        L.append("**%s (%s), by the pre-registered rule: %s**%s" % (name, READINGS[name][2], v,
                                                                     " — " + mean if mean else ""))
        L.append("")
    if "build" in rep:
        b = rep["build"]
        L += ["## The data", "",
              "- Kept rows: **%d** (%s); A1 has %d rows (%d + %d)." % (
                  b["kept"], ", ".join("%s %d" % kv for kv in sorted(b["by_check"].items())), b["a1_rows"],
                  b["a0_rows"], b["a1_rows"] - b["a0_rows"]),
              "- The teacher's answer equals approach 1's target on %d of them; %d switch mode." % (
                  b["same_as_target"], b["mode_switches_vs_target"]),
              "- Traces: %s tokens with the prompt (median %s, p90 %s); the longest is `%s`." % (
                  "%s–%s" % (b["trace_tokens"]["min"], b["trace_tokens"]["max"]), b["trace_tokens"]["median"],
                  b["trace_tokens"]["p90"], b["longest"]["id"]), ""]
    if "check" in rep:
        c = rep["check"]
        L += ["- The check: the longest trace (%s tokens) through one step, %s s, peak reserved %s GiB." % (
            c["tokens"], c["trace_seconds"], c["peak_reserved_gib"]), ""]
    if "r1" in rep:
        r = rep["r1"]
        L += ["R1's reading L: **%s** (%s)." % (r["verdict"], F.fmt(r["L"]) if r.get("L") else "—"), ""]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "r2.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "r2.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
