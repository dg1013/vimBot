#!/usr/bin/env python3
"""The 27B's QLoRA SFT against the models it is meant to beat, on val, by the rules fixed in
runs/qlora27b_2026-10-31/preregistration.md. CPU only: reads the runs' exec logs, writes
report.md and report.json beside the pre-registration.

Primary rows are the 295 val rows never used for checkpoint selection (every adapter here was
selected on the same 250-row subset); all 545 rows are secondary.

    python scripts/qlora27b/report.py
"""
import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "retrieval",
          REPO / "scripts" / "data_scaling", REPO / "scripts" / "distill_targets"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                   # noqa: E402
from compare import fit, train_curve, untouched_ids               # noqa: E402  (distill_targets)

_spec = importlib.util.spec_from_file_location("teacher27b_report",
                                               REPO / "scripts" / "teacher27b" / "report.py")
T27 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T27)    # compare(), load_greedy(), arm_stats(), pct(): one copy of each

RUN = REPO / "runs" / "qlora27b_2026-10-31"
VAL = REPO / "data" / "splits" / "val.jsonl"
ARM = "27b-sft"
ARM_DIR = "runs/qwen3.5-27b-sft-distill"
COMPARATORS = {
    "9b": ("runs/qwen3.5-9b-grpo-gated", "the 9B, SFT + GRPO (the pooled pick's 9B adapter, NF4)"),
    "4b-distill": ("runs/qwen3.5-4b-sft-distill", "the 4B, the same recipe on the same distilled targets"),
    "27b-prompted": ("runs/teacher27b_2026-10-30/27b-retr8", "the 27B untrained, 8 retrieved examples"),
    "4b-served": ("runs/qwen3.5-4b-grpo-gated-leg2", "the served 4B, SFT + GRPO"),
    "27b-k0": ("runs/teacher27b_2026-10-30/27b-k0", "the 27B untrained, zero-shot"),
}
# The pre-registered paired readings: (name, comparator, question). Each is read on the
# untouched rows (primary) and on all of val.
READINGS = [
    ("Q1", "9b", "a stronger teacher than the 9B?"),
    ("Q2", "4b-distill", "what size buys under identical training"),
    ("Q3", "27b-prompted", "what training adds to the 27B"),
]
BOOT_SEED = 20261031
TIERS = ("T1", "T2", "T3")


def readings(match, clean, ids):
    """Every pre-registered reading, primary (untouched rows) and full, from id -> 0/1 dicts."""
    out = {}
    for name, comp, question in READINGS:
        if comp in match and ARM in match:
            out[name] = {"question": question, "comparator": comp,
                         "untouched": T27.compare(match[comp], match[ARM], clean, seed=BOOT_SEED),
                         "full": T27.compare(match[comp], match[ARM], ids, seed=BOOT_SEED)}
    return out


def next_step(q1, q2):
    if q1 == "stronger":
        s = ("The trained 27B out-solves the 9B, so it becomes the teacher for soft-target "
             "distillation: record its top-k token probabilities on the train targets offline "
             "(the 27B cannot sit beside a training 4B on the card), then train the 4B against "
             "them under their own pre-registration.")
    else:
        s = ("Not a stronger teacher than the 9B, which needs about a third of the memory: "
             "soft-target distillation, if it is run, uses the 9B. The 27B row stands as the top "
             "of the size curve.")
    if q2:
        s += " Under identical training the 27B is %s against the 4B (Q2)." % q2
    return s


def build():
    val = load_jsonl(str(VAL))
    ids = [c["id"] for c in val]
    clean = untouched_ids(val)
    tier = {c["id"]: c["tier"] for c in val}
    arm_dir = REPO / ARM_DIR
    if not (arm_dir / "exec_val.jsonl").exists():
        sys.exit("no %s/exec_val.jsonl yet: the full run has not finished" % ARM_DIR)
    rep = {"n": len(ids), "n_untouched": len(clean), "arm": ARM, "arm_dir": ARM_DIR,
           "rows": {}}
    match = {}
    for name, (folder, what) in [(ARM, (ARM_DIR, "Qwen3.5-27B, QLoRA SFT on the distilled targets"))] \
            + list(COMPARATORS.items()):
        d = REPO / folder
        if not (d / "exec_val.jsonl").exists():
            rep["rows"][name] = {"status": "missing", "source": folder}
            continue
        verd, comp, _ = T27.load_greedy(d, val, False, 5.0)
        match[name] = {i: verd[i]["match"] for i in ids}
        row = dict(T27.arm_stats(verd, comp, ids), source=folder, what=what)
        row["untouched_pct"] = T27.pct(sum(match[name][i] for i in clean), len(clean))
        rep["rows"][name] = row
    rep["readings"] = readings(match, clean, ids)
    rep["tiers"] = {t: {k: T27.pct(sum(match[k][i] for i in ids if tier[i] == t),
                                   sum(tier[i] == t for i in ids)) for k in match}
                    for t in TIERS}
    rep["tier_n"] = {t: sum(tier[i] == t for i in ids) for t in TIERS}
    read = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    summary = read(arm_dir / "sft_val.json") or {}
    rep["training"] = {"best_step": (summary.get("training") or {}).get("best_step"),
                       "fit": fit(train_curve(arm_dir / "train.log")),
                       "memory": read(arm_dir / "memory.json"),
                       "config": (read(arm_dir / "train_config.json") or {}).get("optimization")}
    q = rep["readings"]
    rep["next_step"] = next_step(q.get("Q1", {}).get("untouched", {}).get("verdict"),
                                 q.get("Q2", {}).get("untouched", {}).get("verdict"))
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f], +%d / −%d, p %.3g → **%s**" % (
        r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["p"], r["verdict"])


def render(rep):
    L = ["# The 27B, QLoRA SFT on the distilled targets (val)", "",
         "Pre-registration: `preregistration.md` beside this file. Built by "
         "`scripts/qlora27b/report.py`. Primary rows: the %d never used for checkpoint selection; "
         "all %d are secondary." % (rep["n_untouched"], rep["n"]), "",
         "| row | untouched val | full val | valid | parse fail | out tok mean | p50 s | ex share |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r in rep["rows"].items():
        if r.get("status") == "missing":
            L.append("| %s | missing (`%s`) | | | | | | |" % (name, r["source"]))
            continue
        L.append("| %s | **%s%%** | %s%% | %s%% | %s | %s | %s | %s%% |" % (
            name, r["untouched_pct"], r["exec_match_pct"], r["valid_pct"], r.get("parse_failures"),
            r.get("out_tokens_mean"), r.get("latency_p50_s"), r.get("ex_share_pct")))
    L += ["", "Rows: " + "; ".join("`%s` = %s" % (k, r.get("what", r.get("source")))
                                  for k, r in rep["rows"].items()) + ".", "",
          "## Pre-registered readings (exact McNemar, paired bootstrap 95% interval)", "",
          "| reading | question | untouched (primary) | all val |", "|---|---|---|---|"]
    for name, r in rep["readings"].items():
        L.append("| %s | %s (vs `%s`) | %s | %s |" % (name, r["question"], r["comparator"],
                                                     fmt(r["untouched"]), fmt(r["full"])))
    L += ["", "**Next step, by the pre-registered rule:** " + rep["next_step"], "",
          "## By tier (full val)", "", "| tier | n | " + " | ".join(rep["tiers"]["T1"]) + " |",
          "|---|---|" + "---|" * len(rep["tiers"]["T1"])]
    for t in TIERS:
        L.append("| %s | %d | %s |" % (t, rep["tier_n"][t], " | ".join(
            "%s%%" % v for v in rep["tiers"][t].values())))
    tr = rep["training"]
    m = tr.get("memory") or {}
    L += ["", "## Training", "",
          "- best step %s; final loss / entropy / token accuracy %s / %s / %s (mean of the last 5 "
          "logged steps)" % (tr["best_step"], tr["fit"].get("loss"), tr["fit"].get("entropy"),
                             tr["fit"].get("mean_token_accuracy")),
          "- memory: peak allocated %s GiB, reserved %s GiB; the input embedding (%s GiB) in "
          "system RAM; %s s per step (median); %s h of training" % (
              m.get("peak_allocated_gib"), m.get("peak_reserved_gib"),
              m.get("embedding_moved_to_ram_gib"), m.get("median_s_per_step_all"),
              m.get("elapsed_h"))]
    return "\n".join(L) + "\n"


def main():
    rep = build()
    (RUN / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (RUN / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
