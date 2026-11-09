#!/usr/bin/env python3
"""GRPO on OPD's adapters, by runs/grpo_opd_2026-11-04/preregistration.md: G (GRPO against its OPD
start, the two seeds pooled, primary), S (against the served 4B, pooled), each seed, and the
descriptive readings. CPU only: reads the arms' files, writes grpo_opd.md and grpo_opd.json beside
the pre-registration. Works on partial results (the verdicts wait for both arms).

    python scripts/grpo_opd/report.py
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load("opd27b_report", REPO / "scripts" / "opd27b" / "report.py")   # exec_rows(), arm_rows(), fmt()
O, K, F, C = R.O, R.K, R.F, R.C    # chosen paths; call(): the conjunction rule; pooled(); compare()
RUN = REPO / "runs" / "grpo_opd_2026-11-04"
SEEDS = (42, 7)
SERVED = "runs/qwen3.5-4b-grpo-gated-leg2"
BOOT_SEED = 20261104
TIERS = ("T1", "T2", "T3")


def arm_folder(seed):
    return "runs/qwen3.5-4b-opd27b-grpo-s%d" % seed


def meaning(g, s):
    if g is None:
        return None
    if g != "helps":
        return ("OPD's better samples gave GRPO nothing it could turn into better single answers. The plateau of about "
                "50.5% stands, no more RL rounds run on the 4B on these prompts, and the served adapter stays.")
    if s == "helps":
        return "the adapter is a candidate to replace the served 4B; a test v2 row and the swap are the owner's call."
    return "GRPO used what OPD left, but only up to the plateau; the served adapter stays."


def selection(seed):
    """The round's choice, and step 0 beside the subset number OPD recorded for the same adapter."""
    d = REPO / arm_folder(seed)
    if not (d / "grpo_val.json").exists():
        return None
    tr = json.loads((d / "grpo_val.json").read_text(encoding="utf-8"))["training"]
    pick = json.loads((O.out_dir(seed) / "chosen.json").read_text(encoding="utf-8"))
    out = {"best_step": tr["best_step"], "best_subset_pct": tr["best_subset_exec_match_pct"],
           "step0_subset_pct": next((e["exec_match_pct"] for e in tr["eval_history"] if e["step"] == 0), None),
           "opd_subset_pct": pick["subset"][str(pick["chosen_round"])],
           "train_hours": round(tr["train_seconds"] / 3600, 2)}
    if (d / "gate_history.json").exists():
        gate = json.loads((d / "gate_history.json").read_text(encoding="utf-8"))
        out["shelves_start"], out["shelves_end"] = gate["initial_shelves"], gate["final_shelves"]
        out["book_fuel"] = [round(gate["refreshes"][i]["expected_fuel_all"], 3) for i in (0, -1)]
        out["shelves_chosen"] = next((r["shelves"] for r in gate["refreshes"] if r["step"] == tr["best_step"]),
                                     gate["initial_shelves"])
    return out


def shelf_pcts(shelves):
    names = ("cliff", "frontier", "known")
    total = sum(shelves[k] for k in names)
    return " / ".join("%.1f%%" % (100 * shelves[k] / total) for k in names)


def describe(a, ids, clean, tier):
    return {"untouched_pct": C.pct(sum(a["match"][i] for i in clean), len(clean)),
            "full_pct": C.pct(sum(a["match"][i] for i in ids), len(ids)),
            "valid_pct": C.pct(sum(a["valid"][i] for i in ids), len(ids)),
            "tokens": round(a["tokens"], 1) if a["tokens"] else None,
            "tiers": {t: C.pct(sum(a["match"][i] for i in ids if tier[i] == t),
                               sum(tier[i] == t for i in ids)) for t in TIERS}}


def build():
    val = C.load_jsonl(str(C.VAL))
    ids = [c["id"] for c in val]
    tier = {c["id"]: c["tier"] for c in val}
    clean = C.untouched_ids(val)
    served = R.exec_rows(SERVED)
    starts = {s: R.arm_rows(s)[0] for s in SEEDS}
    arms = {s: R.exec_rows(arm_folder(s)) for s in SEEDS}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {"served": describe(served, ids, clean, tier)},
           "readings": {}, "pooled": {}, "verdict": {"G": None, "S": None}, "selection": {}, "fuel": {}}
    for s in SEEDS:
        rep["arms"]["opd-s%d" % s] = describe(starts[s], ids, clean, tier)
        fuel = RUN / ("fuel_s%d.json" % s)
        if fuel.exists():
            rep["fuel"]["s%d" % s] = json.loads(fuel.read_text(encoding="utf-8"))
        if arms[s] is None:
            continue
        rep["arms"]["grpo-s%d" % s] = describe(arms[s], ids, clean, tier)
        rep["selection"]["s%d" % s] = selection(s)
        for name, base in (("G", starts[s]), ("S", served)):
            rep["readings"].setdefault(name, {})["s%d" % s] = {
                key: C.compare(base["match"], arms[s]["match"], rows, seed=BOOT_SEED)
                for rows, key in ((clean, "untouched"), (ids, "all"))}
    if all(arms[s] is not None for s in SEEDS):
        for name, base in (("G", starts), ("S", {s: served for s in SEEDS})):
            rep["pooled"][name] = {key: F.pooled(base[42]["match"], arms[42]["match"], base[7]["match"], arms[7]["match"],
                                                 rows, seed=BOOT_SEED)
                                   for rows, key in ((clean, "untouched"), (ids, "all"))}
            one = rep["readings"][name]
            rep["verdict"][name] = K.call(rep["pooled"][name]["untouched"], one["s42"]["untouched"], one["s7"]["untouched"])
    return rep


def render(rep):
    L = ["# GRPO on OPD's adapters (val)", "",
         "By `preregistration.md` beside this file; built by `scripts/grpo_opd/report.py`. Primary rows: the %d never "
         "used for choosing; all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| arm | untouched val | full val | valid | tokens | T1 / T2 / T3 |", "|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        L.append("| `%s` | **%s%%** | %s%% | %s%% | %s | %s |" % (
            k, a["untouched_pct"], a["full_pct"], a["valid_pct"], a["tokens"],
            " / ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | untouched (primary rows) | all val |", "|---|---|---|"]
    names = {"G": "G: GRPO vs its OPD start", "S": "S: GRPO vs the served 4B"}
    for name, what in names.items():
        for s, r in rep["readings"].get(name, {}).items():
            L.append("| %s, seed %s | %s | %s |" % (what, s[1:], R.fmt(r["untouched"]), R.fmt(r["all"])))
        if name in rep["pooled"]:
            p, q = rep["pooled"][name]["untouched"], rep["pooled"][name]["all"]
            L.append("| %s, two seeds pooled (bootstrap) | %+.1f [%+.1f, %+.1f] | %+.1f [%+.1f, %+.1f] |" % (
                what, p["diff_pts"], p["ci95"][0], p["ci95"][1], q["diff_pts"], q["ci95"][0], q["ci95"][1]))
    if rep["verdict"]["G"]:
        L += ["", "**G, by the pre-registered rule: %s. S: %s.** %s" % (
            rep["verdict"]["G"], rep["verdict"]["S"], meaning(rep["verdict"]["G"], rep["verdict"]["S"]))]
    L += ["", "## The rounds (descriptive)", "",
          "| seed | fuel at the start | cliff / frontier / known | the same at the chosen step | the same at the end "
          "| the book's fuel, step 50 → end | best step (subset) | step 0 (OPD recorded) | train hours |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        f, x = rep["fuel"].get("s%d" % s), rep["selection"].get("s%d" % s)
        shelves = " / ".join("%s%%" % f["shelves_pct"][k] for k in ("cliff", "frontier", "known")) if f else "—"
        chosen, end = "—", "—"
        if x and "shelves_end" in x:
            chosen, end = shelf_pcts(x["shelves_chosen"]), shelf_pcts(x["shelves_end"])
        L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            s, f["fuel"] if f else "—", shelves, chosen, end,
            "%s → %s" % tuple(x["book_fuel"]) if x and "book_fuel" in x else "—",
            "%s (%s%%)" % (x["best_step"], x["best_subset_pct"]) if x else "—",
            "%s%% (%s%%)" % (x["step0_subset_pct"], x["opd_subset_pct"]) if x else "—",
            x["train_hours"] if x else "—"))
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (RUN / "grpo_opd.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "grpo_opd.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
