#!/usr/bin/env python3
"""Score reranking offline: what each rule would have answered, judged by the
executor labels already on the candidates.

Rules, per prompt:
- greedy: the served adapter's §6 answer;
- majority: the most frequent command among the 8 samples, by text (ties split);
- verifier: the candidate with the highest verifier score among the samples,
  optionally plus lambda * log(sample count) (a frequency prior, lambda tuned
  on the val selection subset only);
- oracle: some sample is right (the ceiling of reranking these samples).
With --ask-report and --verdicts the rules also run end to end beside the
confirmed ask rule (ask when the top result holds at most 5 of the 8 samples):
where it asks, the clarification dialogue's answer; elsewhere, the rule's pick.

    # val: pick lambda on the selection subset, report the untouched rows
    python scripts/reranker/rerank_eval.py --examples runs/reranker_2026-10-16/val_examples.jsonl \\
        --scores 4b=runs/reranker_2026-10-16/verifier-4b/val_scores_step300.jsonl \\
        --scores 2b=... --split val --ask-report runs/sample_rule_2026-10-09/val/report.json \\
        --verdicts runs/clarify_2026-10-06/val/verdicts_served.jsonl --out .../val_report.md
"""
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", HERE):
    sys.path.insert(0, str(p))
from verifier_lib import auroc, group, mcnemar_p, paired, rule_outcomes   # noqa: E402

LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
ASK_AT_MOST = 5


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_scores(path):
    return {(r["id"], r["cand_mode"], r["cand_command"]): float(r["score"]) for r in load_jsonl(path)}


def select_ids_for(split):
    if split != "val":
        return set()
    from train_sft import SUBSET_SEED, stratified_subset
    val = load_jsonl(REPO / "data" / "splits" / "val.jsonl")
    return {c["id"] for c in stratified_subset(val, 250, seed=SUBSET_SEED)}


def rate(outcomes, rule, ids):
    vals = [outcomes[i][rule] for i in ids if outcomes[i].get(rule) is not None]
    return 100 * sum(vals) / len(ids) if ids else float("nan")


def asks(report_path, ids):
    view = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
    return {i: view[i]["top_share"] * view[i]["n"] <= ASK_AT_MOST + 1e-9 for i in ids if i in view}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", required=True)
    ap.add_argument("--scores", action="append", default=[], help="LABEL=scores.jsonl")
    ap.add_argument("--split", choices=("val", "test"), required=True)
    ap.add_argument("--lam", action="append", default=[],
                    help="LABEL=lambda, fixed (test); default on val: tuned on the selection subset")
    ap.add_argument("--ask-report", default=None, help="sample_rule.py report.json for this split")
    ap.add_argument("--verdicts", default=None, help="clarify verdicts (the dialogue's answer)")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    rows = load_jsonl(args.examples)
    groups = group(rows)
    ids = sorted(groups)
    sel = select_ids_for(args.split)
    views = [("all", ids)]
    if sel:
        views = [("selection subset", [i for i in ids if i in sel]),
                 ("untouched", [i for i in ids if i not in sel]), ("all", ids)]
    cases = {c["id"]: c for c in load_jsonl(args.cases or REPO / "data" / "splits" / ("%s.jsonl" % args.split))}
    base = rule_outcomes(groups)
    fixed = dict(s.split("=", 1) for s in args.lam)
    verifiers = {}
    for spec in args.scores:
        label, path = spec.split("=", 1)
        scores = load_scores(path)
        missing = sum(1 for r in rows if (r["id"], r["cand_mode"], r["cand_command"]) not in scores)
        if missing:
            raise SystemExit("%s: %d candidates have no score" % (label, missing))
        tune_ids = [i for i in ids if i in sel] if sel else ids
        if label in fixed:
            lam, tuning = float(fixed[label]), None
        else:
            tuning = {lam: rate(rule_outcomes(groups, scores, lam), "verifier_samples", tune_ids) for lam in LAMBDAS}
            lam = max(LAMBDAS, key=lambda l: (round(tuning[l], 6), -l))   # ties to the smaller prior
        out = rule_outcomes(groups, scores, lam)
        pos = [scores[(r["id"], r["cand_mode"], r["cand_command"])] for r in rows if r["label"] == 1]
        neg = [scores[(r["id"], r["cand_mode"], r["cand_command"])] for r in rows if r["label"] == 0]
        # within-prompt: of the prompts holding a right and a wrong candidate, how often the
        # top-scored candidate is right (pure score, no prior)
        mixed = [i for i in ids if 0 < sum(c["label"] for c in groups[i]) < len(groups[i])]
        pure = rule_outcomes(groups, scores, 0.0)
        verifiers[label] = {"lam": lam, "tuning": tuning, "outcomes": out, "auroc": auroc(pos, neg),
                            "mixed_prompts": len(mixed),
                            "mixed_top_right": rate(pure, "verifier_all", mixed) if mixed else None}

    L = ["# Reranking the served adapter's samples: %s" % args.split, "",
         "Generated by `scripts/reranker/rerank_eval.py`. %d prompts, %d candidates "
         "(%d right). Every pick is judged by the executor labels on the candidates." % (
             len(ids), len(rows), sum(r["label"] for r in rows)), ""]
    hdr = "| rule | " + " | ".join("%s (%d)" % (n, len(v)) for n, v in views) + " |"
    L += [hdr, "|---|" + "---|" * len(views)]
    for rule, name in (("greedy", "greedy §6 answer (served adapter)"),
                       ("majority", "majority of the 8 samples, by command text")):
        L.append("| %s | %s |" % (name, " | ".join("%.1f%%" % rate(base, rule, v) for _, v in views)))
    for label, v in verifiers.items():
        L.append("| **verifier %s** over the samples (λ = %g) | %s |" % (
            label, v["lam"], " | ".join("**%.1f%%**" % rate(v["outcomes"], "verifier_samples", vv) for _, vv in views)))
        L.append("| verifier %s over the samples + greedy | %s |" % (
            label, " | ".join("%.1f%%" % rate(v["outcomes"], "verifier_all", vv) for _, vv in views)))
    L.append("| oracle: some sample is right | %s |" % " | ".join("%.1f%%" % rate(base, "oracle_samples", v) for _, v in views))
    L.append("")
    report_ids = views[1][1] if sel else ids
    L += ["## Against the greedy answer (%s rows)" % ("untouched val" if sel else args.split), "",
          "| verifier | AUROC (candidates) | top pick right where the prompt has both | gained / lost vs greedy | exact McNemar p |",
          "|---|---|---|---|---|"]
    greedy = {i: base[i]["greedy"] for i in report_ids}
    for label, v in verifiers.items():
        pick = {i: v["outcomes"][i]["verifier_samples"] for i in report_ids}
        g, l, n = paired(greedy, pick)
        L.append("| %s | %.3f | %.1f%% of %d | +%d / −%d | %.3f |" % (
            label, v["auroc"], v["mixed_top_right"] or float("nan"), v["mixed_prompts"], g, l, mcnemar_p(g, l)))
    L.append("")
    if any(v["tuning"] for v in verifiers.values()):
        L += ["## λ (frequency prior), tuned on the selection subset only", "",
              "| verifier | " + " | ".join("λ = %g" % x for x in LAMBDAS) + " |", "|---|" + "---|" * len(LAMBDAS)]
        for label, v in verifiers.items():
            if v["tuning"]:
                L.append("| %s | %s |" % (label, " | ".join("%.1f%%" % v["tuning"][x] for x in LAMBDAS)))
        L.append("")
    if args.ask_report and args.verdicts:
        ask = asks(args.ask_report, ids)
        ver = {r["id"]: r for r in load_jsonl(args.verdicts)}
        dialogue = {i: int(bool((ver.get(i, {}).get("real") or {}).get("match"))) for i in ids}
        L += ["## Beside the confirmed ask rule (asks where the top result holds ≤ 5 of 8)", "",
              "| answer where it does not ask | " + " | ".join("%s (%d)" % (n, len(v)) for n, v in views) + " |",
              "|---|" + "---|" * len(views)]
        def e2e(rule_vals):
            return {i: (dialogue[i] if ask.get(i) else rule_vals[i]) for i in ids}
        rows_e2e = [("greedy", e2e({i: base[i]["greedy"] for i in ids}))]
        for label, v in verifiers.items():
            rows_e2e.append(("verifier %s" % label, e2e({i: v["outcomes"][i]["verifier_samples"] for i in ids})))
        for name, vals in rows_e2e:
            L.append("| %s | %s |" % (name, " | ".join(
                "%.1f%%" % (100 * sum(vals[i] or 0 for i in vv) / len(vv)) for _, vv in views)))
        rule_rate = sum(1 for i in ids if ask.get(i)) / len(ids)
        L += ["", "Asks on %.1f%% of rows. Where it asks the answer is the dialogue's, the same for every row of this table. "
              "Where it does not ask, 6 or more of the 8 samples agree, so a reranker has little to change there." % (
                  100 * rule_rate), ""]
        # The alternative: let the verifier decide. Ask on the least confident prompts
        # (lowest top score among the samples), answer the verifier's pick elsewhere.
        L += ["## The verifier deciding when to ask", "",
              "Ask on the given share of prompts where the verifier's best score is lowest; "
              "elsewhere answer the verifier's pick. A curve, not a tuned point.", "",
              "| verifier | " + " | ".join("asks %d%%" % round(100 * a) for a in (0, .1, .2, .3, round(rule_rate, 2))) +
              " | the 5-of-8 rule + greedy |", "|---|" + "---|" * 6]
        greedy_e2e = e2e({i: base[i]["greedy"] for i in ids})
        for label, v in verifiers.items():
            scores = load_scores(dict(s.split("=", 1) for s in args.scores)[label])
            conf = {}
            for i in ids:
                sampled = [c for c in groups[i] if c.get("n", 0) > 0] or groups[i]
                conf[i] = max(scores[(i, c["cand_mode"], c["cand_command"])] for c in sampled)
            order = sorted(ids, key=lambda i: conf[i])
            cells = []
            for a in (0, .1, .2, .3, round(rule_rate, 2)):
                asked = set(order[:round(a * len(ids))])
                vals = [dialogue[i] if i in asked else (v["outcomes"][i]["verifier_samples"] or 0) for i in ids]
                cells.append("%.1f%%" % (100 * sum(vals) / len(ids)))
            L.append("| %s | %s | %.1f%% |" % (label, " | ".join(cells),
                                              100 * sum(greedy_e2e[i] or 0 for i in ids) / len(ids)))
            # the pre-registered point (2026-10-16): asks on 20%, against the rule + greedy
            asked = set(order[:round(0.2 * len(ids))])
            gated = {i: (dialogue[i] if i in asked else (v["outcomes"][i]["verifier_samples"] or 0)) for i in ids}
            base_rule = {i: greedy_e2e[i] or 0 for i in ids}
            g, l, _ = paired(base_rule, gated)
            v["gated20"] = (100 * sum(gated.values()) / len(ids), g, l, mcnemar_p(g, l))
        L += ["", "| verifier | asks 20%: accuracy | vs the 5-of-8 rule + greedy | gained / lost | exact McNemar p |",
              "|---|---|---|---|---|"]
        for label, v in verifiers.items():
            acc, g, l, p = v["gated20"]
            L.append("| %s | %.1f%% | %.1f%% | +%d / −%d | %.3f |" % (
                label, acc, 100 * sum(greedy_e2e[i] or 0 for i in ids) / len(ids), g, l, p))
        L.append("")
    L += ["## By tier (%s rows)" % ("untouched val" if sel else args.split), "",
          "| tier | rows | greedy | majority | " + " | ".join("verifier %s" % k for k in verifiers) + " | oracle |",
          "|---|---|---|---|" + "---|" * len(verifiers) + "---|"]
    for t in ("T1", "T2", "T3"):
        tid = [i for i in report_ids if cases.get(i, {}).get("tier") == t]
        if not tid:
            continue
        L.append("| %s | %d | %.1f%% | %.1f%% | %s | %.1f%% |" % (
            t, len(tid), rate(base, "greedy", tid), rate(base, "majority", tid),
            " | ".join("%.1f%%" % rate(v["outcomes"], "verifier_samples", tid) for v in verifiers.values()),
            rate(base, "oracle_samples", tid)))
    L += ["", "## By category (%s rows; under 25 rows is noise)" % ("untouched val" if sel else args.split), "",
          "| category | rows | greedy | " + " | ".join("verifier %s" % k for k in verifiers) + " | oracle |",
          "|---|---|---|" + "---|" * len(verifiers) + "---|"]
    cats = collections.Counter(cases.get(i, {}).get("category", "?") for i in report_ids)
    for cat, n in cats.most_common():
        cid = [i for i in report_ids if cases.get(i, {}).get("category") == cat]
        L.append("| %s | %d | %.1f%% | %s | %.1f%% |" % (
            cat, n, rate(base, "greedy", cid),
            " | ".join("%.1f%%" % rate(v["outcomes"], "verifier_samples", cid) for v in verifiers.values()),
            rate(base, "oracle_samples", cid)))
    L.append("")
    pathlib.Path(args.out).write_text("\n".join(L), encoding="utf-8")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps({
            label: {"lam": v["lam"], "tuning": v["tuning"], "auroc": v["auroc"],
                    **{name: rate(v["outcomes"], "verifier_samples", vv) for name, vv in views}}
            for label, v in verifiers.items()}, indent=1) + "\n", encoding="utf-8")
    print("report -> %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
