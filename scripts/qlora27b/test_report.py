#!/usr/bin/env python3
"""The 27B's test v2 row beside the 9B, the served 4B and the served pooled pick, by addendum 3
of runs/qlora27b_2026-10-31/preregistration.md. Descriptive: no decision depends on it. CPU
only: reads the runs' test logs, writes test_report.md and test_report.json beside report.md.

    python scripts/qlora27b/test_report.py
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

# explore.py by path, under its own name; it loads this folder's report.py as E.R.
_spec = importlib.util.spec_from_file_location("qlora27b_explore", HERE / "explore.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)
R = E.R

TEST = REPO / "data" / "splits" / "test.jsonl"
V2 = REPO / "runs" / "remeasure_2026-09-25" / "v2"
COMPARATORS = {     # test v2 execution logs: remeasure's merged 429-row files
    "9b": (V2 / "qwen3.5-9b-grpo-gated", "the 9B, SFT + GRPO (the pooled pick's 9B adapter, NF4)"),
    "4b-served": (V2 / "qwen3.5-4b-grpo-gated__fla-kernels", "the served 4B, SFT + GRPO"),
}
POOLED_K = 4        # the rule's served setting (2026-10-21)


def load_test(folder):
    """id -> 0/1 from a run's exec_test.jsonl."""
    path = pathlib.Path(folder) / "exec_test.jsonl"
    if not path.exists():
        sys.exit("no %s yet" % path)
    return {r["id"]: int(bool(r["match"])) for r in R.load_jsonl(str(path))}


def build():
    cases = R.load_jsonl(str(TEST))
    ids = [c["id"] for c in cases]
    tier = {c["id"]: c["tier"] for c in cases}
    arm_dir = REPO / R.ARM_DIR
    match = {R.ARM: load_test(arm_dir)}
    for name, (folder, _) in COMPARATORS.items():
        match[name] = load_test(folder)
    for name, m in match.items():
        if set(m) != set(ids):
            sys.exit("%s does not cover test v2 (%d of %d ids)" % (name, len(set(m) & set(ids)), len(ids)))
    summary = json.loads((arm_dir / "baseline_test.json").read_text(encoding="utf-8"))
    share = E.pooled_shares(ids, "test")
    rep = {"n": len(ids), "arm": R.ARM, "metrics": summary["metrics"],
           "served_check": E.check_against_served(share, ids, "test"),
           "rows": {name: R.T27.pct(sum(m.values()), len(ids)) for name, m in match.items()},
           "readings": {name: R.T27.compare(match[name], match[R.ARM], ids, seed=R.BOOT_SEED)
                        for name in COMPARATORS},
           "pooled@%d" % POOLED_K: round(100.0 * sum(share[POOLED_K][i] for i in ids) / len(ids), 1),
           "vs_pooled": E.share_diff(match[R.ARM], share[POOLED_K], ids)}
    rep["tiers"] = {}
    for t in R.TIERS:
        sub = [i for i in ids if tier[i] == t]
        row = {name: R.T27.pct(sum(m[i] for i in sub), len(sub)) for name, m in match.items()}
        row["pooled@%d" % POOLED_K] = round(100.0 * sum(share[POOLED_K][i] for i in sub) / len(sub), 1)
        rep["tiers"][t] = dict(row, n=len(sub))
    return rep


def render(rep):
    m = rep["metrics"]
    L = ["# The 27B, QLoRA SFT on the distilled targets: test v2", "",
         "Addendum 3 of `preregistration.md` beside this file; built by `scripts/qlora27b/test_report.py`. "
         "Reporting only: no decision depends on these numbers.", "",
         "| row | test v2 (%d) |" % rep["n"], "|---|---|",
         "| **27B, QLoRA SFT** (valid %s%%, %s tokens, p50 %s s) | **%s%%** |"
         % (m["valid_pct"], m["output_tokens_mean"], m["latency_s_p50"], rep["rows"][R.ARM])]
    for name, (_, what) in COMPARATORS.items():
        L.append("| %s | %s%% |" % (what, rep["rows"][name]))
    L.append("| the served pooled pick, %d + %d samples + verifier (mean of %d draws; served report %s) | %s%% |"
             % (POOLED_K, POOLED_K, E.DRAWS, rep["served_check"][POOLED_K], rep["pooled@%d" % POOLED_K]))
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", ""]
    for name, r in rep["readings"].items():
        L.append("- against %s: %+.1f [%+.1f, %+.1f], +%d / −%d, p %.3g → **%s**"
                 % (COMPARATORS[name][1], r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"],
                    r["p"], r["verdict"]))
    v = rep["vs_pooled"]
    L.append("- against the pooled pick at %d + %d (bootstrap only): %+.1f [%+.1f, %+.1f] → **%s**"
             % (POOLED_K, POOLED_K, v["diff_pts"], v["ci95"][0], v["ci95"][1], v["verdict"]))
    names = list(rep["rows"]) + ["pooled@%d" % POOLED_K]
    L += ["", "## By tier", "", "| tier | n | " + " | ".join(names) + " |",
          "|---|---|" + "---|" * len(names)]
    for t, row in rep["tiers"].items():
        L.append("| %s | %d | %s |" % (t, row["n"], " | ".join("%s%%" % row[k] for k in names)))
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (R.RUN / "test_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (R.RUN / "test_report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
