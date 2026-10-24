#!/usr/bin/env python3
"""Score every ask-and-pick policy on val, choose by the pre-registered rules,
read test once, and write the report (CPU; torch for the probe fit).

    python scripts/ask_pick/ask_pick_eval.py --run runs/ask_pick_2026-10-18

Reads, from the run folder: turn2_<split>.jsonl and turn2_scores_<split>.jsonl
(the samples after the reply and their verifier scores) and probe_features.pt.
Everything else is on disk from earlier runs (ask_pick_lib paths). The rules
are runs/ask_pick_2026-10-18/preregistration.md.
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
import ask_pick_lib as L                                     # noqa: E402

LAYERS = (8, 16, 24, 32)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)
TRIGGER_ORDER = {t: k for k, t in enumerate(L.TRIGGERS)}
AFTER_ORDER = {a: k for k, a in enumerate(L.AFTER)}


def single_turn(split):
    """Per row: the verifier's pick label and its best score over the 8 samples."""
    ex_path, sc_path = L.SINGLE_TURN[split]
    groups = {}
    for e in L.load_jsonl(ex_path):
        if e.get("n", 0) > 0:
            groups.setdefault(e["id"], []).append(e)
    scores = {(r["id"], r["cand_mode"], r["cand_command"]): r["score"] for r in L.load_jsonl(sc_path)}
    pick, best = {}, {}
    for cid, cands in groups.items():
        pick[cid] = L.pick_label(cands, scores, cid)
        best[cid] = L.best_score(cands, scores, cid)
    return pick, best


def turn2(run, split):
    """Per row: labels of the answers after the reply -- the verifier's pick
    (plain and clarified), the majority, the oracle -- and the sample count."""
    samples = {r["id"]: r for r in L.load_jsonl(run / ("turn2_%s.jsonl" % split))}
    sc = {}
    for r in L.load_jsonl(run / ("turn2_scores_%s.jsonl" % split)):
        sc[(r["id"], r["cand_mode"], r["cand_command"])] = (r["plain"], r["clarified"])
    plain = {k: v[0] for k, v in sc.items()}
    clar = {k: v[1] for k, v in sc.items()}
    out = {}
    for cid, row in samples.items():
        cands = L.candidates(row)
        out[cid] = {"pick-plain": L.pick_label(cands, plain, cid), "pick-clarified": L.pick_label(cands, clar, cid),
                    "majority": L.majority_label(cands), "oracle": int(int(row["k"]) > 0),
                    "distinct": len(cands), "k": int(row["k"])}
    return out


def fit_probes(feats, diag_cases, diag_dial):
    """U and F probes on diag: layer and λ by 5-fold CV AUROC, then fit on all."""
    import torch
    from probe_ask import cv_auroc, fit_logistic
    targets = {}
    keys = [i for i, c in diag_cases.items() if c.get("note_class") in ("unstated", "ambiguous", "model")]
    targets["U"] = (keys, [int(diag_cases[i]["note_class"] in ("unstated", "ambiguous")) for i in keys])
    keys = [i for i in diag_cases if i in diag_dial]
    targets["F"] = (keys, [int(bool(diag_dial[i]["real"]["match"]) and not bool(diag_dial[i]["direct"]["match"]))
                           for i in keys])
    probes, grid = {}, {}
    for name, (keys, labels) in targets.items():
        best = None
        for layer in LAYERS:
            x = torch.stack([feats["diag|%s" % i][str(layer)] for i in keys]).double()
            for lam in LAMBDAS:
                a = cv_auroc(x.numpy(), labels, lam)
                grid[(name, layer, lam)] = a
                if best is None or a > best[0]:
                    best = (a, layer, lam)
        a, layer, lam = best
        x = torch.stack([feats["diag|%s" % i][str(layer)] for i in keys]).double()
        y = torch.as_tensor(labels, dtype=torch.long)
        mu, sd = x.mean(0), x.std(0) + 1e-6
        w, b = fit_logistic((x - mu) / sd, y, lam)
        probes[name] = {"layer": layer, "lam": lam, "cv_auroc": a, "w": w, "b": b, "mu": mu, "sd": sd,
                        "n": len(labels), "pos": sum(labels)}
    return probes, grid


def apply_probe(probe, feats, split, ids):
    import torch
    x = torch.stack([feats["%s|%s" % (split, i)][str(probe["layer"])] for i in ids]).double()
    s = (((x - probe["mu"]) / probe["sd"]) @ probe["w"] + probe["b"]).tolist()
    return dict(zip(ids, s))


def build_split(run, split, feats, probes):
    cases = {c["id"]: c for c in L.load_jsonl(L.CASES[split])}
    dial = {r["id"]: r for r in L.load_jsonl(L.DIALOGUES[split])}
    pick, best = single_turn(split)
    t2 = turn2(run, split)
    ids = [i for i in cases if i in dial and i in pick and i in t2]
    after = {"greedy": {i: int(bool(dial[i]["real"]["match"])) for i in ids},
             "pick-plain": {i: t2[i]["pick-plain"] for i in ids},
             "pick-clarified": {i: t2[i]["pick-clarified"] for i in ids},
             "majority": {i: t2[i]["majority"] for i in ids}, "oracle": {i: t2[i]["oracle"] for i in ids}}
    trig = {"V": {i: -best[i] for i in ids}}
    for name in ("U", "F"):
        trig[name] = apply_probe(probes[name], feats, split, ids)
    trig["V+U"] = L.combine(ids, trig["V"], trig["U"])
    trig["V+F"] = L.combine(ids, trig["V"], trig["F"])
    rule_path = L.RUNS / "sample_rule_2026-10-09" / split / "report.json"
    rule = json.loads(rule_path.read_text())
    direct = {i: int(bool(dial[i]["direct"]["match"])) for i in ids}
    rule_asks = {i for i in ids if rule[i]["top_share"] * 8 <= 5}
    round_trip = {i: (after["greedy"][i] if i in rule_asks else direct[i]) for i in ids}
    return {"ids": ids, "cases": cases, "pick": {i: pick[i] for i in ids}, "best": best, "after": after,
            "trig": trig, "direct": direct, "round_trip": round_trip, "rule_asks": rule_asks, "t2": t2}


def grid(S):
    """{(trigger, rate, after): (outcome, asked)} for the 45 policies."""
    out = {}
    for t in L.TRIGGERS:
        for r in L.RATES:
            for a in L.AFTER:
                out[(t, r, a)] = L.policy(S["ids"], S["trig"][t], r, S["after"][a], S["pick"])
    return out


def acc(outcome):
    return 100 * sum(outcome.values()) / len(outcome)


def select(val_grid):
    def key(item):
        (t, r, a), (o, _) = item
        return (-acc(o), r, TRIGGER_ORDER[t], AFTER_ORDER[a])
    return min(val_grid.items(), key=key)[0]


def threshold_read(V, T, pol):
    """The selected policy with its rate turned into a fixed cutoff from val."""
    t, r, a = pol
    import math

    def zscores(S, name, ref):
        vals = [ref["trig"][name][i] for i in ref["ids"]]
        mu, sd = statistics.fmean(vals), statistics.pstdev(vals) or 1.0
        return {i: (S["trig"][name][i] - mu) / sd for i in S["ids"]}

    def score(S):
        if "+" in t:
            p, q = t.split("+")
            zp, zq = zscores(S, p, V), zscores(S, q, V)
            return {i: (zp[i] + zq[i]) / 2 for i in S["ids"]}
        return S["trig"][t]
    sv, st = score(V), score(T)
    k = round(r * len(V["ids"]))
    cut = sorted(sv.values(), reverse=True)[k - 1] if k else math.inf
    asked = {i for i in T["ids"] if st[i] >= cut}
    out = {i: (T["after"][a][i] if i in asked else T["pick"][i]) for i in T["ids"]}
    return out, asked, cut


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(L.RUNS / "ask_pick_2026-10-18"))
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    import torch
    feats = torch.load(run / "probe_features.pt")
    diag_cases = {c["id"]: c for c in L.load_jsonl(L.CASES["diag"])}
    diag_dial = {r["id"]: r for r in L.load_jsonl(L.DIALOGUES["diag"])}
    probes, pgrid = fit_probes(feats, diag_cases, diag_dial)
    V = build_split(run, "val", feats, probes)
    T = build_split(run, "test", feats, probes)
    gv, gt = grid(V), grid(T)
    ref_key = ("V", 0.20, "greedy")
    ref_t = gt[ref_key][0]
    if sum(ref_t.values()) != 263 or len(T["ids"]) != 429:
        raise SystemExit("the verifier-gated ask does not reproduce 263/429 on test: %d/%d" % (
            sum(ref_t.values()), len(T["ids"])))
    sel = select(gv)
    c1_variant = max(("pick-plain", "pick-clarified"), key=lambda a: (acc(gv[("V", 0.20, a)][0]), -AFTER_ORDER[a]))
    c2_probe = max(("U", "F"), key=lambda t: (acc(gv[(t, 0.20, "greedy")][0]), -TRIGGER_ORDER[t]))
    ids = T["ids"]
    n = len(ids)

    def test_line(label, a_out, b_out):
        g, l = L.paired(a_out, b_out, ids)
        d, lo, hi = L.paired_ci(g, l, n)
        return {"label": label, "b_pct": acc(b_out), "a_pct": acc(a_out), "gained": g, "lost": l,
                "delta": d, "ci": [lo, hi], "p": L.mcnemar_p(g, l)}
    tests = [test_line("primary: selected policy %s vs the verifier-gated ask" % (sel,), ref_t, gt[sel][0]),
             test_line("C1: pick after asking (%s) vs greedy after asking, both V at 20%%" % c1_variant, ref_t,
                       gt[("V", 0.20, c1_variant)][0]),
             test_line("C2: trigger %s vs V, both 20%% with greedy after asking" % c2_probe, ref_t,
                       gt[(c2_probe, 0.20, "greedy")][0]),
             test_line("selected policy vs the deployed round trip", T["round_trip"], gt[sel][0])]
    thr_out, thr_asked, cut = threshold_read(V, T, sel)
    tests.append(test_line("selected policy with a fixed cutoff from val (asks %.1f%% of test)" % (
        100 * len(thr_asked) / n), ref_t, thr_out))

    # diagnostics on test: does each trigger find wrong picks, and the rows a question fixes?
    diag = {}
    wrong = [i for i in ids if not T["pick"][i]]
    fix = {i: T["after"]["greedy"][i] for i in wrong}
    for t in L.TRIGGERS:
        s = T["trig"][t]
        diag[t] = {"pick_wrong": L.auroc([s[i] for i in ids if not T["pick"][i]], [s[i] for i in ids if T["pick"][i]]),
                   "asking_fixes": L.auroc([s[i] for i in wrong if fix[i]], [s[i] for i in wrong if not fix[i]])}
    # what picking after asking does on its own, on every row (not a policy)
    t2_all = {a: acc(T["after"][a]) for a in ("greedy", "pick-plain", "pick-clarified", "majority", "oracle")}
    v2_all = {a: acc(V["after"][a]) for a in ("greedy", "pick-plain", "pick-clarified", "majority", "oracle")}

    lines = ["# Pick after asking, and a better trigger for asking (2026-10-18)", "",
             "Rules: `preregistration.md`. Built by `scripts/ask_pick/`. Exec-match on val (545) and test v2 (429); "
             "every row has every arm's answer, so policies are scored offline. The simulated user knows the wanted result.", "",
             "## The probes (fit on the 422 diag train rows only)", "",
             "| probe | target | rows (positives) | layer | λ | CV AUROC on diag |", "|---|---|---|---|---|---|"]
    for name, desc in (("U", "unstated or ambiguous vs fair-but-wrong"), ("F", "asking fixes the greedy answer")):
        p = probes[name]
        lines.append("| %s | %s | %d (%d) | %d | %g | %.3f |" % (name, desc, p["n"], p["pos"], p["layer"], p["lam"], p["cv_auroc"]))
    lines += ["", "## The answer after asking, on every row (before any policy)", "",
              "| answer after the reply | val | test |", "|---|---|---|"]
    for a, lab in (("greedy", "greedy (recorded)"), ("majority", "most common of 8"), ("pick-plain", "verifier pick, plain"),
                   ("pick-clarified", "verifier pick, clarified"), ("oracle", "any of 8 right (ceiling)")):
        lines.append("| %s | %.1f%% | %.1f%% |" % (lab, v2_all[a], t2_all[a]))
    lines += ["", "## Val: all 45 policies (exec-match %)", "",
              "| trigger | asks | greedy after | pick-plain after | pick-clarified after |", "|---|---|---|---|---|"]
    for t in L.TRIGGERS:
        for r in L.RATES:
            lines.append("| %s | %d%% | %s |" % (t, round(100 * r), " | ".join(
                ("**%.1f**" if (t, r, a) == sel else "%.1f") % acc(gv[(t, r, a)][0]) for a in L.AFTER)))
    lines += ["", "Selected on val: **%s at %d%% asks, %s after asking** (%.1f%% on val). Component choices on val: "
              "pick variant **%s**; probe **%s**. Val references: verifier pick alone %.1f%%; round trip %.1f%%." % (
                  sel[0], round(100 * sel[1]), sel[2], acc(gv[sel][0]), c1_variant, c2_probe, acc(V["pick"]),
                  acc(V["round_trip"])), "",
              "## Test v2: the pre-registered reads", "",
              "| comparison | first | second | gained / lost | Δ points (95% CI) | p |", "|---|---|---|---|---|---|"]
    for tl in tests:
        lines.append("| %s | %.1f%% | %.1f%% | +%d / −%d | %+.1f (%+.1f to %+.1f) | %s |" % (
            tl["label"], tl["a_pct"], tl["b_pct"], tl["gained"], tl["lost"], tl["delta"], tl["ci"][0], tl["ci"][1],
            "%.3f" % tl["p"] if tl["p"] >= 0.001 else "< 0.001"))
    lines += ["", "Reference: the verifier-gated ask reproduces 263/429 (61.3%%) on test; the round trip %.1f%%; the "
              "verifier pick alone %.1f%%." % (acc(T["round_trip"]), acc(T["pick"])), "",
              "## Test diagnostics: what each trigger finds (AUROC; 0.5 = chance)", "",
              "| trigger | the verifier's pick is wrong | among wrong picks, asking fixes it |", "|---|---|---|"]
    for t in L.TRIGGERS:
        lines.append("| %s | %.2f | %.2f |" % (t, diag[t]["pick_wrong"], diag[t]["asking_fixes"]))
    lines += ["", "## Test: all 45 policies (description only; never a pick)", "",
              "| trigger | asks | greedy after | pick-plain after | pick-clarified after |", "|---|---|---|---|---|"]
    for t in L.TRIGGERS:
        for r in L.RATES:
            lines.append("| %s | %d%% | %s |" % (t, round(100 * r), " | ".join("%.1f" % acc(gt[(t, r, a)][0]) for a in L.AFTER)))
    text = "\n".join(lines) + "\n"
    (run / "report.md").write_text(text, encoding="utf-8")
    (run / "report.json").write_text(json.dumps({
        "probes": {k: {kk: v[kk] for kk in ("layer", "lam", "cv_auroc", "n", "pos")} for k, v in probes.items()},
        "probe_grid": {"%s|%d|%g" % k: v for k, v in pgrid.items()},
        "selected": list(sel), "c1_variant": c1_variant, "c2_probe": c2_probe,
        "val_grid": {"%s|%g|%s" % k: acc(v[0]) for k, v in gv.items()},
        "test_grid": {"%s|%g|%s" % k: acc(v[0]) for k, v in gt.items()},
        "tests": tests, "diagnostics": diag, "turn2_all": {"val": v2_all, "test": t2_all},
        "threshold": {"cut": cut, "asked_test": len(thr_asked)}}, indent=1, default=str) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
