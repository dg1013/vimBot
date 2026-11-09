#!/usr/bin/env python3
"""OPD against its starts, by runs/opd27b_2026-11-02/preregistration.md: O (the two seeds pooled,
primary), each seed, the descriptive readings. CPU only: reads the arms' files, writes opd.md and
opd.json beside the pre-registration. Works on partial results (the verdict waits for both arms).

    python scripts/opd27b/report.py
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


O = load("opd27b_opd", HERE / "opd.py")
K = load("kd27b_compare_kd", REPO / "scripts" / "kd27b" / "compare_kd.py")   # call(): the conjunction rule
F, C = K.F, K.C                    # pooled(), and compare(), untouched_ids(), pct()
BOOT_SEED = 20261102
TIERS = ("T1", "T2", "T3")
OTHERS = {"distill-grpo": ("runs/qwen3.5-4b-distill-grpo", "GRPO from the seed-42 start (round 1)"),
          "served": ("runs/qwen3.5-4b-grpo-gated-leg2", "the served 4B")}
MEANING = {
    "helps": ("OPD replaces GRPO as the 4B's stage after SFT (GRPO added +0.0 on this start); GRPO after OPD, a test "
              "v2 row and serving are the owner's call."),
    "tied": ("neither off-policy nor on-policy distillation from the trained 27B moves the 4B on these prompts, so "
             "the 27B's lead is capacity; no more distillation from this teacher on these prompts."),
    "hurts": "as tied, and reported.",
}


def exec_rows(folder):
    d = REPO / folder
    path = d / "exec_val.jsonl"
    if not path.exists():
        return None
    rows = {r["id"]: r for r in C.load_jsonl(str(path))}
    comp = {r["id"]: r for r in C.load_jsonl(str(d / "completions_val.jsonl"))} if (d / "completions_val.jsonl").exists() else {}
    return {"match": {i: int(bool(r["match"])) for i, r in rows.items()},
            "valid": {i: int(bool(r["valid"])) for i, r in rows.items()},
            "tokens": statistics.fmean(c["output_tokens"] for c in comp.values()) if comp else None}


def arm_rows(seed):
    """The arm's val rows: the chosen round's full val, or the start's when round 0 was chosen."""
    od = O.out_dir(seed)
    pick = od / "chosen.json"
    if not pick.exists():
        return None, None
    chosen = json.loads(pick.read_text(encoding="utf-8"))
    if chosen["chosen_round"] == 0:
        return exec_rows(str(O.STARTS[seed].relative_to(REPO))), chosen
    return exec_rows(str(od.relative_to(REPO))), chosen


def round_stats(seed, r):
    rd = O.round_dir(seed, r)
    out = {}
    if (rd / "eval.json").exists():
        out["subset_pct"] = json.loads((rd / "eval.json").read_text(encoding="utf-8"))["exec_match_pct"]
    if r == 0:
        return out
    if (rd / "judged.jsonl").exists():
        j = {x["id"]: x for x in C.load_jsonl(str(rd / "judged.jsonl"))}
        out["train_match_pct"] = C.pct(sum(x["match"] for x in j.values()), len(j))
        if (rd / "teacher.jsonl").exists() and (rd / "samples.jsonl").exists():
            t = {x["id"]: sum(x["tlp1"]) for x in C.load_jsonl(str(rd / "teacher.jsonl"))}
            s = {x["id"]: sum(x["slp"]) for x in C.load_jsonl(str(rd / "samples.jsonl"))}
            right = [i for i, x in j.items() if x["match"] and i in t and i in s]
            out["correct_pushed_down_pct"] = C.pct(sum(t[i] < s[i] for i in right), len(right))
            out["teacher_seq_lp_median"] = round(statistics.median(t.values()), 3)
            out["student_seq_lp_median"] = round(statistics.median(s.values()), 3)
    if (rd / "train_log.jsonl").exists():
        log = C.load_jsonl(str(rd / "train_log.jsonl"))
        if log:
            out["kl_first5"] = round(statistics.fmean(x["kl"] for x in log[:5]), 4)
            out["kl_last5"] = round(statistics.fmean(x["kl"] for x in log[-5:]), 4)
            out["steps"] = len(log)
    if (rd / "samples_meta.json").exists():
        m = json.loads((rd / "samples_meta.json").read_text(encoding="utf-8"))
        out["sample_tokens_median"], out["hit_cap"] = m["median_tokens"], m["hit_cap"]
    return out


def pass8(seed, label):
    path = O.out_dir(seed) / ("pass8_%s.jsonl" % label)
    if not path.exists():
        return None
    rows = C.load_jsonl(str(path))
    return C.pct(sum(r["any"] for r in rows), len(rows))


def build():
    val = C.load_jsonl(str(C.VAL))
    ids = [c["id"] for c in val]
    tier = {c["id"]: c["tier"] for c in val}
    clean = C.untouched_ids(val)
    starts = {s: exec_rows(str(O.STARTS[s].relative_to(REPO))) for s in O.STARTS}
    arms, chosen = {}, {}
    for s in O.STARTS:
        arms[s], chosen[s] = arm_rows(s)
    others = {k: exec_rows(f) for k, (f, _) in OTHERS.items()}
    rep = {"n": len(ids), "n_untouched": len(clean), "arms": {}, "readings": {}, "pooled": {}, "rounds": {},
           "pass8": {}, "verdict": None}
    for s in O.STARTS:
        for name, a in (("start-s%d" % s, starts[s]), ("opd-s%d" % s, arms[s])):
            if a is None:
                continue
            rep["arms"][name] = {"untouched_pct": C.pct(sum(a["match"][i] for i in clean), len(clean)),
                                 "full_pct": C.pct(sum(a["match"][i] for i in ids), len(ids)),
                                 "valid_pct": C.pct(sum(a["valid"][i] for i in ids), len(ids)),
                                 "tokens": round(a["tokens"], 1) if a["tokens"] else None,
                                 "tiers": {t: C.pct(sum(a["match"][i] for i in ids if tier[i] == t),
                                                    sum(tier[i] == t for i in ids)) for t in TIERS}}
        rep["rounds"]["s%d" % s] = {r: round_stats(s, r) for r in range(O.ROUNDS + 1)}
        rep["pass8"]["s%d" % s] = {"start": pass8(s, "start"), "chosen": pass8(s, "chosen")}
        if chosen[s]:
            rep["arms"]["opd-s%d" % s]["chosen_round"] = chosen[s]["chosen_round"]
        if arms[s] is not None:
            for rows, key in ((clean, "untouched"), (ids, "all")):
                rep["readings"].setdefault("OPD vs its start, seed %d" % s, {})[key] = C.compare(
                    starts[s]["match"], arms[s]["match"], rows, seed=BOOT_SEED)
            for k, (_, what) in OTHERS.items():
                if others[k] is not None and (k != "distill-grpo" or s == 42):
                    rep["readings"].setdefault("OPD seed %d vs %s" % (s, what), {})["untouched"] = C.compare(
                        others[k]["match"], arms[s]["match"], clean, seed=BOOT_SEED)
    if all(arms[s] is not None for s in O.STARTS):
        for rows, key in ((clean, "untouched"), (ids, "all")):
            rep["pooled"][key] = F.pooled(starts[42]["match"], arms[42]["match"], starts[7]["match"], arms[7]["match"],
                                          rows, seed=BOOT_SEED)
        r42 = rep["readings"]["OPD vs its start, seed 42"]["untouched"]
        r7 = rep["readings"]["OPD vs its start, seed 7"]["untouched"]
        rep["verdict"] = K.call(rep["pooled"]["untouched"], r42, r7)
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f], +%d / −%d, p %.3g → %s" % (
        r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["p"], r["verdict"])


def render(rep):
    L = ["# On-policy distillation from the trained 27B (val)", "",
         "By `preregistration.md` beside this file; built by `scripts/opd27b/report.py`. Primary rows: the %d never "
         "used for choosing; all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| arm | untouched val | full val | valid | tokens | chosen round | T1 / T2 / T3 |", "|---|---|---|---|---|---|---|"]
    for k, a in rep["arms"].items():
        L.append("| `%s` | **%s%%** | %s%% | %s%% | %s | %s | %s |" % (
            k, a["untouched_pct"], a["full_pct"], a["valid_pct"], a["tokens"], a.get("chosen_round", "—"),
            " / ".join("%s%%" % a["tiers"][t] for t in TIERS)))
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", "", "| reading | untouched (primary rows) | all val |",
          "|---|---|---|"]
    for k, r in rep["readings"].items():
        L.append("| %s | %s | %s |" % (k, fmt(r["untouched"]), fmt(r["all"]) if "all" in r else "—"))
    if rep["pooled"]:
        p, q = rep["pooled"]["untouched"], rep["pooled"]["all"]
        L.append("| O: OPD vs its start, two seeds pooled (bootstrap) | %+.1f [%+.1f, %+.1f] | %+.1f [%+.1f, %+.1f] |" % (
            p["diff_pts"], p["ci95"][0], p["ci95"][1], q["diff_pts"], q["ci95"][0], q["ci95"][1]))
    if rep["verdict"]:
        L += ["", "**Verdict, by the pre-registered rule: %s** — %s" % (rep["verdict"], MEANING[rep["verdict"]])]
    L += ["", "## By round (descriptive)", "", "| seed | round | subset | train match (own samples) | mean KL, first / last 5 steps "
          "| correct samples pushed down | median sample tokens | at the cap |", "|---|---|---|---|---|---|---|---|"]
    for s, rounds in rep["rounds"].items():
        for r, x in rounds.items():
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                s, r, ("%s%%" % x["subset_pct"]) if "subset_pct" in x else "—",
                ("%s%%" % x["train_match_pct"]) if "train_match_pct" in x else "—",
                ("%s / %s" % (x["kl_first5"], x["kl_last5"])) if "kl_first5" in x else "—",
                ("%s%%" % x["correct_pushed_down_pct"]) if "correct_pushed_down_pct" in x else "—",
                x.get("sample_tokens_median", "—"), x.get("hit_cap", "—")))
    L += ["", "pass@8 on the 250-row subset (T 1, the same sampler): " + "; ".join(
        "seed %s start %s%%, chosen %s%%" % (s[1:], v["start"], v["chosen"]) for s, v in rep["pass8"].items()) + "."]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (O.RUN / "opd.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (O.RUN / "opd.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
