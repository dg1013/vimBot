#!/usr/bin/env python3
"""Exploratory, not pre-registered: the trained 27B beside the served pooled pick, and how much of
val the greedy models cover together. CPU only: reads the runs' logs, writes explore.md and
explore.json beside report.md.

The pooled pick is the deployed system: 4B and 9B samples, the verifier picks one. Its per-row
share of right picks over 20 random draws of k samples per model is rebuilt from the served pass's
rows exactly as scripts/serve_pooled/served_at_k.py builds it, and checked against that report.
The 27B's one greedy answer is compared with that share by a paired bootstrap. McNemar needs 0/1
rows, and a share is not one.

    python scripts/qlora27b/explore.py
"""
import importlib.util
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

# This folder's report.py by path, under its own name: scripts/teacher27b has a report.py too.
_spec = importlib.util.spec_from_file_location("qlora27b_report", HERE / "report.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)      # paths, comparators, loaders, BOOT_SEED: one copy of each

sys.path.insert(0, str(REPO / "scripts" / "serve_pooled"))
import pick_at_k as K                                             # noqa: E402
import served_at_k as S                                           # noqa: E402

SERVED = REPO / "runs" / "serve_pooled" / "served_at_k"
DRAWS, SEED = 20, "served@k|"    # served_at_k.py's estimator
KS = (4, 8)                      # the rule's served setting (2026-10-21), and 2026-10-19's
N_BOOT = 10000


def share_diff(a, share, ids, n_boot=N_BOOT, seed=R.BOOT_SEED):
    """a (id -> 0/1) against share (id -> fraction right) on the same rows: the difference in
    points with a paired bootstrap 95% interval, called by the interval alone."""
    d = [a[i] - share[i] for i in ids]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(100.0 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    call = "stronger" if lo > 0 else "weaker" if hi < 0 else "tied"
    return {"diff_pts": round(100.0 * sum(d) / n, 1), "ci95": [round(lo, 1), round(hi, 1)],
            "verdict": call}


def pooled_shares(ids, split="val"):
    """{k: {id: share of the DRAWS draws whose pooled pick is right}} for k in KS."""
    rows = {}
    for r in S.load_jsonl(SERVED / ("rows_%s.jsonl" % split)):
        rows.setdefault(r["id"], r)
    missing = [i for i in ids if i not in rows]
    if missing:
        sys.exit("the served pass has no row for %d %s ids (first: %s)" % (len(missing), split, missing[0]))
    crow = {m: {i: S.ceiling_row(rows[i], m) for i in ids} for m in ("4b", "9b")}
    maps = {i: S.row_maps(rows[i])[0] for i in ids}
    _, per_row = K.curves_for(ids, crow, lambda i, mode, cmd: maps[i][(mode, cmd)], DRAWS, seed=SEED)
    return {k: dict(zip(ids, per_row["4B + 9B"][k])) for k in KS}


def check_against_served(share, ids, split="val"):
    """The rebuilt means over the whole split must equal the served report's, or nothing here
    is the deployed system."""
    pub = json.loads((SERVED / "report.json").read_text(encoding="utf-8"))
    curves = pub["splits"][split]["mean_of_draws"]["curves"]["4B + 9B"]
    out = {}
    for k in KS:
        mine = round(100.0 * sum(share[k][i] for i in ids) / len(ids), 1)
        if mine != curves[str(k)]["pick"]:
            sys.exit("rebuilt pooled pick at %d is %.1f, the served report says %.1f"
                     % (k, mine, curves[str(k)]["pick"]))
        out[k] = mine
    return out


def coverage(match, ids):
    """Greedy unions: the share of rows at least one of the named models solves."""
    union = lambda names: R.T27.pct(sum(any(match[n][i] for n in names) for i in ids), len(ids))
    return {"27b-sft + 9b": union((R.ARM, "9b")),
            "9b + 4b-served": union(("9b", "4b-served")),
            "27b-sft + 9b + 4b-served": union((R.ARM, "9b", "4b-served")),
            "only 27b-sft of the three": sum(1 for i in ids if match[R.ARM][i]
                                             and not match["9b"][i] and not match["4b-served"][i])}


def identical(val, ids):
    """Answers equal as strings (mode and command) to the gold and to the other models'."""
    def preds(folder):
        return {r["id"]: (r.get("mode"), r["command"])
                for r in R.load_jsonl(str(REPO / folder / "preds_val.jsonl"))}
    p = {R.ARM: preds(R.ARM_DIR), "9b": preds(R.COMPARATORS["9b"][0]),
         "4b-distill": preds(R.COMPARATORS["4b-distill"][0])}
    gold = {c["id"]: (c["mode"], c["gold_command"]) for c in val}
    same = lambda a, b: sum(1 for i in ids if a.get(i) == b.get(i))
    return {"27b-sft = gold": same(p[R.ARM], gold), "9b = gold": same(p["9b"], gold),
            "4b-distill = gold": same(p["4b-distill"], gold),
            "27b-sft = 9b": same(p[R.ARM], p["9b"]), "27b-sft = 4b-distill": same(p[R.ARM], p["4b-distill"])}


def build():
    val = R.load_jsonl(str(R.VAL))
    ids = [c["id"] for c in val]
    clean = R.untouched_ids(val)
    tier = {c["id"]: c["tier"] for c in val}
    match = {}
    for name, folder in [(R.ARM, R.ARM_DIR), ("9b", R.COMPARATORS["9b"][0]),
                         ("4b-served", R.COMPARATORS["4b-served"][0])]:
        verd, _, _ = R.T27.load_greedy(REPO / folder, val, False, 5.0)
        match[name] = {i: verd[i]["match"] for i in ids}
    share = pooled_shares(ids)
    rep = {"n": len(ids), "n_untouched": len(clean), "draws": DRAWS,
           "served_check": check_against_served(share, ids), "vs_pooled": {}, "coverage": {}}
    for part, sub in (("untouched", clean), ("full", ids)):
        rep["vs_pooled"][part] = {
            "27b-sft": R.T27.pct(sum(match[R.ARM][i] for i in sub), len(sub)),
            **{"pooled@%d" % k: round(100.0 * sum(share[k][i] for i in sub) / len(sub), 1) for k in KS},
            **{"27b-sft vs pooled@%d" % k: share_diff(match[R.ARM], share[k], sub) for k in KS}}
        rep["coverage"][part] = coverage(match, sub)
    rep["tiers"] = {t: {"n": sum(tier[i] == t for i in ids),
                        "27b-sft": R.T27.pct(sum(match[R.ARM][i] for i in ids if tier[i] == t),
                                             sum(tier[i] == t for i in ids)),
                        "pooled@4": round(100.0 * sum(share[4][i] for i in ids if tier[i] == t)
                                          / sum(tier[i] == t for i in ids), 1)}
                    for t in R.TIERS}
    rep["identical"] = identical(val, ids)
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f] → %s" % (r["diff_pts"], r["ci95"][0], r["ci95"][1], r["verdict"])


def render(rep):
    u, f = rep["vs_pooled"]["untouched"], rep["vs_pooled"]["full"]
    L = ["# Exploratory: the trained 27B beside the served pooled pick (val)", "",
         "Not pre-registered. Built by `scripts/qlora27b/explore.py`. The pooled pick is the deployed "
         "system (4B + 9B samples, the verifier picks): its per-row share of right picks over %d draws, "
         "rebuilt from `runs/serve_pooled/served_at_k/rows_val.jsonl` and equal to that report's "
         "full-val means (%s). Paired bootstrap 95%% intervals, 10,000 resamples; a share has no "
         "McNemar test." % (rep["draws"], ", ".join("%.1f at %d" % (v, k)
                                                     for k, v in rep["served_check"].items())), "",
         "| | untouched val (%d) | full val (%d) |" % (rep["n_untouched"], rep["n"]), "|---|---|---|",
         "| 27B, QLoRA SFT: one greedy answer | %s%% | %s%% |" % (u["27b-sft"], f["27b-sft"])]
    for k in KS:
        L.append("| pooled pick, %d samples per model + verifier | %s%% | %s%% |"
                 % (k, u["pooled@%d" % k], f["pooled@%d" % k]))
    for k in KS:
        L.append("| 27B − pooled pick at %d | %s | %s |"
                 % (k, fmt(u["27b-sft vs pooled@%d" % k]), fmt(f["27b-sft vs pooled@%d" % k])))
    L += ["", "## Greedy coverage", "", "| union | untouched | full |", "|---|---|---|"]
    cu, cf = rep["coverage"]["untouched"], rep["coverage"]["full"]
    for key in cu:
        unit = "" if key.startswith("only") else "%"
        L.append("| %s | %s%s | %s%s |" % (key, cu[key], unit, cf[key], unit))
    L += ["", "## By tier (full val)", "", "| tier | n | 27B | pooled pick at 4 |", "|---|---|---|---|"]
    for t, r in rep["tiers"].items():
        L.append("| %s | %d | %s%% | %s%% |" % (t, r["n"], r["27b-sft"], r["pooled@4"]))
    L += ["", "## Identical answers (mode and command, of %d)" % rep["n"], ""]
    L += ["- %s: %d" % (k, v) for k, v in rep["identical"].items()]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (R.RUN / "explore.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (R.RUN / "explore.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
