#!/usr/bin/env python3
"""The report on the served 27B and the three-model pool (CPU).

By runs/serve27b_2026-11-05/preregistration.md, from what pass27b.py wrote. Rows are the served
pass's (runs/serve_pooled/served_at_k), which hold the 4B's and 9B's 128 samples per row and their
verifier scores, with the 27B's answers added. Each row then has the served pass's own shape, so
its helpers read it unchanged. Every answer of all three models carries the verdict of one oracle,
the box as it stands today.

An arm is what each model contributes to the pool, {model: k}: "4b", "9b", "27b" (k samples at
T = 1) and "27g" (the 27B's one greedy answer). Its estimate is the served pass's: the mean over 20
random draws of k samples per model, the verifier picking among the drawn candidates (a row's value
is its share of right picks), and one run from each model's first k. Differences are paired over
rows with a bootstrap, which takes shares and 0/1 alike; McNemar needs 0/1, so it goes with the one
runs.

    python scripts/serve27b/pool27b.py --run runs/serve27b_2026-11-05
"""
import argparse
import importlib.metadata
import importlib.util
import json
import pathlib
import random
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "serve_pooled",
          REPO / "scripts" / "methods_9b", HERE):
    sys.path.insert(0, str(p))
import pass27b as A                                               # noqa: E402
import pick_at_k as K                                             # noqa: E402
import served_at_k as S                                           # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load("distill_targets_compare", REPO / "scripts" / "distill_targets" / "compare.py")      # compare(), untouched_ids()

DRAWS, SEED = 20, "served@k|"         # served_at_k.py's estimator
BASE_K = 4                            # the pooled pick's val-chosen setting (2026-10-21)
MARGIN = 1.0                          # points: the rule of the served pass
N_BOOT = 10000
BOOT_SEED = 20261105
GPU_HOUR = 0.30                       # USD, as stated in every cost table so far
TIERS = ("T1", "T2", "T3")


def base_k():
    return {"4b": BASE_K, "9b": BASE_K}


def family(n):
    """What the 27B adds to the pooled pick, cheapest first: the rule's settings."""
    out = [("g", dict(base_k(), **{"27g": 1}))]
    return out + [(str(k), dict(base_k(), **{"27b": k})) for k in A.TIMED_KS if k <= n]


def descriptive(n):
    """Everything else the report shows: the 27B alone, each pair, every model at k."""
    ks = [k for k in A.TIMED_KS if k <= n]
    arms = [("27B pick at %d" % k, {"27b": k}) for k in ks]
    arms += [("27B greedy + %d, pick" % k, {"27g": 1, "27b": k}) for k in ks if k in (4, 8)]
    arms += [("4B + 27B greedy", {"4b": BASE_K, "27g": 1}), ("9B + 27B greedy", {"9b": BASE_K, "27g": 1}),
             ("4B + 27B at 4", {"4b": BASE_K, "27b": 4}), ("9B + 27B at 4", {"9b": BASE_K, "27b": 4})]
    return arms + [("%d + %d + %d" % (k, k, k), {"4b": k, "9b": k, "27b": k}) for k in ks if k != BASE_K]


def label(c):
    return "4 + 4 + greedy" if c == "g" else "4 + 4 + %s" % c


# ------------------------------------------------------------ rows

def merged_row(old, sample, greedy, scores, states):
    """A served-pass row with the 27B's answers added, verdicts overlaid. scores: {(mode, command):
    score} for the 27B's new answers; states: {(mode, command): verdict} from judge. Raises if an
    answer has no score or no verdict: a pick on a missing number would be a silent error."""
    cands = {(c["mode"], c["command"]): dict(c) for c in old["cands"]}
    for key, value in scores.items():
        cands.setdefault(key, {"mode": key[0], "command": key[1], "score": value, "state": None})
    samples = dict(old["samples"], **{"27b": sample["samples"], "27g": greedy["samples"]})
    for row in samples.values():
        for s in row:
            if s is not None and tuple(s) not in cands:
                raise KeyError("%s: no score for %r" % (old["id"], s))
    for key, c in cands.items():
        if key in states:
            c["state"] = states[key]
        if c["state"] is None:
            raise KeyError("%s: no verdict for %r" % (old["id"], key))
    tokens = dict(old["output_tokens"], **{"27b": sample["output_tokens"], "27g": greedy["output_tokens"]})
    return {"id": old["id"], "samples": samples, "cands": list(cands.values()), "output_tokens": tokens}


def load_rows(run, split, ids):
    """({id: merged row}, {changed: verdicts that differ from the served pass}) for rows with every part."""
    old = A.old_rows(split)
    greedy = {r["id"]: r for r in A.load_jsonl(run / ("greedy_%s.jsonl" % split))}
    samples = {r["id"]: r for r in A.load_jsonl(run / "samples.jsonl") if r["split"] == split}
    scores, states = {}, {}
    for r in A.load_jsonl(run / "scores27.jsonl"):
        scores.setdefault(r["id"], {})[(r["mode"], r["command"])] = r["score"]
    for r in A.load_jsonl(run / "judged.jsonl"):
        states.setdefault(r["id"], {})[(r["mode"], r["command"])] = r["state"]
    rows, changed, total = {}, [], 0
    for i in ids:
        if i in greedy and i in samples and i in old:
            for c in old[i]["cands"]:
                new = states.get(i, {}).get((c["mode"], c["command"]))
                if new is not None:
                    total += 1
                    if new != c["state"]:
                        changed.append({"id": i, "mode": c["mode"], "command": c["command"], "was": c["state"], "now": new})
            rows[i] = merged_row(old[i], samples[i], greedy[i], scores.get(i, {}), states.get(i, {}))
    return rows, {"judged": total, "changed": changed}


# ------------------------------------------------------------ estimates

def row_pools(row, models):
    """(candidates by model, sample populations by model) in pick_at_k's format."""
    cb, pops = {}, {}
    for m in models:
        cb[m], pops[m] = K.expand(S.ceiling_row(row, m))
    return cb, pops


def arm_shares(rows, ids, k, draws=DRAWS, seed=SEED):
    """{id: (share of right picks, share with a right candidate, share of right votes)} over draws."""
    out = {}
    for i in ids:
        cb, pops = row_pools(rows[i], k)
        scores = S.row_maps(rows[i])[0]
        rng = random.Random("%s%s|%s" % (seed, sorted(k.items()), i))
        acc = [0, 0, 0]
        for _ in range(draws):
            r = K.draw_eval(pops, cb, scores, rng, k)
            acc = [a + b for a, b in zip(acc, r)]
        out[i] = tuple(a / draws for a in acc)
    return out


def first_k_arm(row, k):
    """One run: each model's first k samples -> (pick right, pass, maj right, picked key, candidates)."""
    sc, lab = S.row_maps(row)
    counts = {}
    for m, kk in k.items():
        for s in row["samples"][m][:kk]:
            if s is not None:
                counts[tuple(s)] = counts.get(tuple(s), 0) + 1
    if not counts:
        return 0, 0, 0, None, 0
    pick = max(counts, key=lambda key: (sc[key], counts[key], key))
    maj = max(counts, key=lambda key: (counts[key], key))
    return lab[pick], int(any(lab[c] for c in counts)), lab[maj], pick, len(counts)


def base_shares(rows, ids):
    """The pooled pick at 4 + 4 by served_at_k's own code and random stream, so with the served
    pass's verdicts it is the published curve to the digit."""
    crow = {m: {i: S.ceiling_row(rows[i], m) for i in ids} for m in ("4b", "9b")}
    maps = {i: S.row_maps(rows[i])[0] for i in ids}
    _, per_row = K.curves_for(ids, crow, lambda i, mode, cmd: maps[i][(mode, cmd)], DRAWS, seed=SEED)
    return dict(zip(ids, per_row["4B + 9B"][BASE_K]))


def mean(shares, ids):
    return round(100.0 * statistics.fmean(shares[i] for i in ids), 1) if ids else None


def choose(means, order, margin=MARGIN):
    """The rule: the first setting in `order` (cheapest first) within `margin` points of the best."""
    best = max(means[c] for c in order)
    return next(c for c in order if means[c] >= best - margin)


def paired(a, b, ids, n_boot=N_BOOT, seed=BOOT_SEED):
    """b minus a over the same rows (shares or 0/1), in points, with a paired bootstrap 95% interval."""
    d = [b[i] - a[i] for i in ids]
    n = len(d)
    rng = random.Random(seed)
    boots = sorted(100.0 * sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return {"n": n, "diff_pts": round(100.0 * sum(d) / n, 1),
            "ci95": [round(boots[int(0.025 * n_boot)], 1), round(boots[int(0.975 * n_boot) - 1], 1)]}


def verdict(pooled, parts):
    """The conjunction rule: an interval clear of zero, and both slices on that side of it."""
    lo, hi = pooled["ci95"]
    if lo > 0 and all(p["diff_pts"] > 0 for p in parts):
        return "helps"
    if hi < 0 and all(p["diff_pts"] < 0 for p in parts):
        return "hurts"
    return "tied"


def reading(a, b, parts, one_a=None, one_b=None):
    """b against a on the pooled rows and on each slice; McNemar on the one runs where given."""
    pooled_ids = [i for ids in parts.values() for i in ids]
    rep = {"pooled": paired(a, b, pooled_ids)}
    for name, ids in parts.items():
        rep[name] = paired(a, b, ids)
    rep["verdict"] = verdict(rep["pooled"], [rep[name] for name in parts])
    if one_a is not None:
        rep["one_run"] = C.compare(one_a, one_b, pooled_ids)
    return rep


# ------------------------------------------------------------ cost

def best_rate(throughput):
    return max(t["req_per_s"] for t in throughput)


def per_solved(gpu_s, acc_pct):
    s = gpu_s / (acc_pct / 100.0)
    return {"exec_match": acc_pct, "gpu_s_per_request": round(gpu_s, 3), "gpu_s_per_solved": round(s, 3),
            "usd_per_1000_solved": round(s * 1000 / 3600 * GPU_HOUR, 4)}


def costs(timing, pool, acc):
    """GPU-seconds per request, as every cost table so far has it: 1 / requests per second at the
    best concurrency for a §6 answer, the mean one-at-a-time latency for a system whose one request
    fills the card. The rows with the 27B and the verifier cannot run on one card, so theirs are
    composed from the parts: GPU time adds, and two cards would overlap the sampling."""
    rows = {}
    g4, gp = pool["greedy4b"], pool["pooled44"]
    rows["served 4B, one answer"] = dict(per_solved(1.0 / best_rate(pool["throughput4b"]), acc["4b"]),
                                         p50_s=g4["p50_s"], p95_s=g4["p95_s"], tokens=g4["tokens_mean"], measured=True)
    rows["pooled pick, 4 + 4"] = dict(per_solved(gp["mean_s"], acc["base"]), p50_s=gp["p50_s"], p95_s=gp["p95_s"],
                                      tokens=gp["tokens_mean"], measured=True)
    g27 = timing["greedy"]
    rows["trained 27B, one answer"] = dict(per_solved(1.0 / best_rate(timing["throughput"]), acc["27b"]),
                                           p50_s=g27["p50_s"], p95_s=g27["p95_s"], tokens=g27["tokens_mean"], measured=True)
    return rows


def composed(timing, pool, c, acc_pct):
    """The three-model pool at setting c: the pooled pick's GPU time plus the 27B's, and its latency
    if the 27B had a card of its own (the slower sampling, then the verifier). A sum of medians is
    not the median of the sums, so the pooled pick's own p50 is the floor: this pool does more."""
    gp = pool["pooled44"]
    part = timing["greedy"] if c == "g" else timing["samples"][str(c)]
    row = per_solved(gp["mean_s"] + part["mean_s"], acc_pct)
    row["p50_s"] = round(max(gp["p50_s"], max(gp["sample_p50_s"], part["p50_s"]) + gp["score_p50_s"]), 3)
    row["tokens"] = round(gp["tokens_mean"] + part["tokens_mean"], 1)
    row["measured"] = False
    return row


# ------------------------------------------------------------ the report

def build(run):
    cases = {sp: A.cases(sp) for sp in A.SPLITS}
    untouched = C.untouched_ids(cases["val"])
    clean = set(untouched)
    choosing_all = [c["id"] for c in cases["val"] if c["id"] not in clean]
    rows, rejudged = {}, {}
    for sp in A.SPLITS:
        rows[sp], rejudged[sp] = load_rows(run, sp, [c["id"] for c in cases[sp]])
    every = {**rows["val"], **rows["test"]}
    choosing = [i for i in choosing_all if i in every]
    parts = {"untouched": [i for i in untouched if i in every], "test": [c["id"] for c in cases["test"] if c["id"] in every]}
    val_ids = [c["id"] for c in cases["val"] if c["id"] in every]
    tier = {c["id"]: c["tier"] for sp in A.SPLITS for c in cases[sp]}
    n = min(len(r["samples"]["27b"]) for r in every.values())

    rep = {"n_samples": n, "rows": {"val": len(rows["val"]), "test": len(rows["test"]), "choosing": len(choosing),
                                    "untouched": len(parts["untouched"])},
           "of": {sp: len(cases[sp]) for sp in A.SPLITS},
           "rejudged": {sp: {"judged": r["judged"], "changed": len(r["changed"]), "first": r["changed"][:12]}
                        for sp, r in rejudged.items()}}

    # the two systems everything is read against
    base = {}
    published = json.loads((A.SERVED / "report.json").read_text())["splits"]
    rep["base_check"] = {}
    for sp in A.SPLITS:
        ids = list(rows[sp])
        base.update(base_shares(rows[sp], ids))
        rep["base_check"][sp] = {"rebuilt": mean(base, ids), "rows": len(ids), "of": len(cases[sp]),
                                 "published": published[sp]["mean_of_draws"]["curves"]["4B + 9B"][str(BASE_K)]["pick"]}
    g27 = {i: first_k_arm(r, {"27g": 1})[0] for i, r in every.items()}

    # the rule's family on the choosing rows, then the reading rows
    fam = family(n)
    fam_shares = {c: arm_shares(every, list(every), k) for c, k in fam}
    order = [c for c, _ in fam]
    choosing_means = {c: mean({i: s[0] for i, s in fam_shares[c].items()}, choosing) for c in order}
    chosen = choose(choosing_means, order)
    pick_of = {c: {i: s[0] for i, s in fam_shares[c].items()} for c in order}
    k_chosen = dict(fam)[chosen]
    one_chosen = {i: first_k_arm(r, k_chosen)[0] for i, r in every.items()}
    one_base = {i: first_k_arm(r, base_k())[0] for i, r in every.items()}
    rep["rule"] = {"choosing_rows": len(choosing), "means": choosing_means, "chosen": chosen, "label": label(chosen)}
    rep["readings"] = {
        "27B greedy vs pooled 4 + 4": reading(base, g27, parts, one_base, g27),
        "three-model pool vs 27B greedy": reading(g27, pick_of[chosen], parts, g27, one_chosen),
        "three-model pool vs pooled 4 + 4": reading(base, pick_of[chosen], parts, one_base, one_chosen)}

    def split_means(shares):
        return {"val": mean(shares, val_ids), "untouched": mean(shares, parts["untouched"]), "test": mean(shares, parts["test"]),
                "choosing": mean(shares, choosing)}

    rep["systems"] = {"27B greedy, served": split_means(g27), "pooled 4 + 4": split_means(base)}

    # every arm, for the table (pick / pass / vote; the one-run pick beside the mean of draws)
    table = {}
    for c, k in fam:
        s = fam_shares[c]
        table[label(c)] = {
            "pick": split_means({i: v[0] for i, v in s.items()}), "pass": split_means({i: v[1] for i, v in s.items()}),
            "maj": split_means({i: v[2] for i, v in s.items()}),
            "one_run": split_means({i: first_k_arm(r, k)[0] for i, r in every.items()})}
    for name, k in descriptive(n):
        s = arm_shares(every, list(every), k)
        table[name] = {
            "pick": split_means({i: v[0] for i, v in s.items()}), "pass": split_means({i: v[1] for i, v in s.items()}),
            "maj": split_means({i: v[2] for i, v in s.items()}),
            "one_run": split_means({i: first_k_arm(r, k)[0] for i, r in every.items()})}
    rep["arms"] = table

    # who supplies the chosen pool's picks, and what the verifier leaves on the table
    who = {"27B only": 0, "4B or 9B only": 0, "both": 0}
    right = dict(who)
    for i, r in every.items():
        _, _, _, key, _ = first_k_arm(r, k_chosen)
        if key is None:
            continue
        from27 = any(tuple(s) == key for m in ("27b", "27g") if m in k_chosen for s in r["samples"][m][:k_chosen[m]] if s)
        from_old = any(tuple(s) == key for m in ("4b", "9b") for s in r["samples"][m][:k_chosen[m]] if s)
        name = "both" if from27 and from_old else "27B only" if from27 else "4B or 9B only"
        who[name] += 1
        right[name] += S.row_maps(r)[1][key]
    rep["picked_from"] = {k: {"rows": who[k], "right": right[k], "pct": C.pct(right[k], who[k])} for k in who}
    rep["tiers"] = {}
    for t in TIERS:
        ids = [i for i in parts["untouched"] + parts["test"] if tier[i] == t]
        rep["tiers"][t] = {"n": len(ids), "27B greedy": mean(g27, ids), "pooled 4 + 4": mean(base, ids),
                           "chosen pool": mean(pick_of[chosen], ids)}

    # the 27B through vLLM against the same weights through transformers
    rep["parity"] = {}
    for sp in A.SPLITS:
        served = {r["id"]: r for r in A.load_jsonl(run / ("exec_%s.jsonl" % sp))}
        rec = {r["id"]: r for r in A.load_jsonl(run / ("recorded_exec_%s.jsonl" % sp))}
        preds = {r["id"]: r for r in A.load_jsonl(A.RECORDED / ("preds_%s.jsonl" % sp))}
        greedy = {r["id"]: r for r in A.load_jsonl(run / ("greedy_%s.jsonl" % sp))}
        ids = [c["id"] for c in cases[sp] if c["id"] in greedy]
        m = {i: int(served[i]["match"]) for i in ids}
        w = {i: int(rec[i]["match"]) for i in ids}
        rep["parity"][sp] = {"rows": len(ids), "identical": sum(A.same_answer(greedy[i]["samples"][0], preds[i]) for i in ids),
                             "served": C.pct(sum(m.values()), len(ids)), "recorded": C.pct(sum(w.values()), len(ids)),
                             "valid": C.pct(sum(int(served[i]["valid"]) for i in ids), len(ids)),
                             "tokens": round(statistics.fmean(sum(greedy[i]["output_tokens"]) for i in ids), 1),
                             "think": sum("<think>" in greedy[i]["texts"][0] for i in ids),
                             "unparsed": sum(greedy[i]["samples"][0] is None for i in ids),
                             "served_vs_recorded": C.compare(w, m, ids)}

    # the served 4B on the same rows, from the harness's own log, for the cost table
    served_4b = {r["id"]: int(r["match"]) for r in A.load_jsonl(REPO / "runs" / "remeasure_2026-09-25" / "v2" /
                                                                "qwen3.5-4b-grpo-gated__fla-kernels" / "exec_test.jsonl")}
    test_ids = [i for i in parts["test"] if i in served_4b]
    rep["cost"] = {}
    timing, pool = (json.loads((run / f).read_text()) if (run / f).exists() else None for f in ("timing.json", "pool_timing.json"))
    if timing and pool:
        acc = {"4b": mean(served_4b, test_ids), "base": mean(base, parts["test"]), "27b": mean(g27, parts["test"])}
        rep["cost"] = costs(timing, pool, acc)
        rep["cost"]["three-model pool, %s" % label(chosen)] = composed(timing, pool, chosen, mean(pick_of[chosen], parts["test"]))
        rep["timing"] = {"27b": timing, "pool": pool}
    rep["versions"] = {"vllm": importlib.metadata.version("vllm")}
    for name in ("smoke.json", "rescore.json"):
        if (run / name).exists():
            rep[name.split(".")[0]] = json.loads((run / name).read_text())
    return rep


def fmt(r):
    return "%+.1f [%+.1f, %+.1f]" % (r["diff_pts"], r["ci95"][0], r["ci95"][1])


def render(rep):
    L = ["# The trained 27B served, and in the pool with the 4B, the 9B and the verifier", "",
         "By `preregistration.md` beside this file; built by `scripts/serve27b/pool27b.py`. vLLM %s, one RTX 3090. "
         "Rows with every part: val %d of %d, test %d of %d. %d samples of the 27B per row. Means of %d draws unless "
         "marked." % (rep["versions"]["vllm"], rep["rows"]["val"], rep["of"]["val"], rep["rows"]["test"], rep["of"]["test"],
                      rep["n_samples"], DRAWS), ""]
    L += ["## Is the served 27B the measured 27B?", "",
          "| split | rows | identical answers | served | recorded, re-judged | served vs recorded | valid | tokens | `<think>` | unparsed |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for sp, r in rep["parity"].items():
        c = r["served_vs_recorded"]
        L.append("| %s | %d | %d | **%.1f%%** | %.1f%% | %+.1f [%+.1f, %+.1f], +%d / −%d, p %.3g | %.1f%% | %.1f | %d | %d |" % (
            sp, r["rows"], r["identical"], r["served"], r["recorded"], c["diff_pts"], c["ci95"][0], c["ci95"][1],
            c["gained"], c["lost"], c["p"], r["valid"], r["tokens"], r["think"], r["unparsed"]))
    L += ["", "Served and recorded are the same weights: NF4 base, trained adapter. The rule: a split beyond 3 points is a "
              "serving fault to find before anything below is read.", ""]
    if rep.get("cost"):
        L += ["## Serving cost (test v2 accuracy; all timed in this run)", "",
              "| system | exec-match on test | latency p50 | p95 | tokens / request | GPU-s / request | GPU-s / solved | $ / 1,000 solved |",
              "|---|---|---|---|---|---|---|---|"]
        for name, r in rep["cost"].items():
            star = "" if r["measured"] else " ¹"
            L.append("| %s%s | %.1f%% | %.2f s | %s | %.0f | %.3f | %.3f | $%.4f |" % (
                name, star, r["exec_match"], r["p50_s"], "%.2f s" % r["p95_s"] if "p95_s" in r else "—", r["tokens"],
                r["gpu_s_per_request"], r["gpu_s_per_solved"], r["usd_per_1000_solved"]))
        L += ["", "GPU-s per request: 1 / requests per second at the best of 1 / 8 / 32 concurrent for a §6 answer, the mean "
                  "one-at-a-time latency for a system whose one request fills the card. Dollars at $%.2f per GPU-hour. Every "
                  "timing takes the head of val, whose rows are lighter than the rest: `posthoc.md` has the sizes, and the "
                  "27B's rate over all rows." % GPU_HOUR,
              "", "¹ Composed, not measured: the 27B fills the card, so this pool cannot run on one 3090. GPU time adds; the "
                  "latency is what two cards would give.", ""]
        t = rep["timing"]["27b"]
        L += ["### Throughput", "", "| concurrent | 27B req/s | p50 | p95 | 4B req/s | p50 | p95 |", "|---|---|---|---|---|---|---|"]
        for a, b in zip(t["throughput"], rep["timing"]["pool"]["throughput4b"]):
            L.append("| %d | %.2f | %.2f s | %.2f s | %.2f | %.2f s | %.2f s |" % (
                a["concurrency"], a["req_per_s"], a["p50_s"], a["p95_s"], b["req_per_s"], b["p50_s"], b["p95_s"]))
        L += ["", "### The 27B's k samples, one request at a time", "", "| k | p50 | mean | p95 | tokens |", "|---|---|---|---|---|"]
        for k, r in t["samples"].items():
            L.append("| %s | %.2f s | %.2f s | %.2f s | %.0f |" % (k, r["p50_s"], r["mean_s"], r["p95_s"], r["tokens_mean"]))
        L.append("")
    r = rep["rule"]
    L += ["## The rule's choice", "",
          "On the %d choosing rows, what the 27B adds to the pooled pick at 4 + 4, cheapest first; the first within %.1f point "
          "of the best is chosen." % (r["choosing_rows"], MARGIN), "",
          "| setting | pick on the choosing rows |", "|---|---|"]
    for c, v in r["means"].items():
        L.append("| %s | %s%.1f%%%s |" % (label(c), "**" if c == r["chosen"] else "", v, "**" if c == r["chosen"] else ""))
    L += ["", "Chosen: **%s**." % r["label"], "", "## Readings", "",
          "On the %d untouched val rows plus the %d test rows, which chose nothing: paired bootstrap on the means of draws, "
          "and the conjunction rule (*helps* needs the pooled interval above zero and both slices positive; *hurts* the mirror; "
          "otherwise *tied*). The one-run line is the same pair on what one request returns." % (
              rep["rows"]["untouched"], rep["rows"]["test"]), "",
          "| reading | pooled rows | untouched | test | verdict | one run |", "|---|---|---|---|---|---|"]
    for name, x in rep["readings"].items():
        o = x["one_run"]
        L.append("| %s | **%s** | %s | %s | **%s** | %+.1f, +%d / −%d, p %.3g |" % (
            name, fmt(x["pooled"]), fmt(x["untouched"]), fmt(x["test"]), x["verdict"], o["diff_pts"], o["gained"], o["lost"], o["p"]))
    L += ["", "| system | val | untouched | test |", "|---|---|---|---|"]
    for name, x in rep["systems"].items():
        L.append("| %s | %.1f%% | %.1f%% | %.1f%% |" % (name, x["val"], x["untouched"], x["test"]))
    L += ["", "## Every arm (descriptive)", "",
          "pick / pass / vote, means of draws; the last column is the pick on one run (each model's first k).", "",
          "| arm | val | untouched | test | test, one run |", "|---|---|---|---|---|"]
    for name, x in rep["arms"].items():
        cell = lambda part: "**%.1f** / %.1f / %.1f" % (x["pick"][part], x["pass"][part], x["maj"][part])
        L.append("| %s | %s | %s | %s | %.1f |" % (name, cell("val"), cell("untouched"), cell("test"), x["one_run"]["test"]))
    L += ["", "## Where the chosen pool's picks come from (one run, all rows)", "",
          "| the picked answer was drawn from | rows | right |", "|---|---|---|"]
    for name, x in rep["picked_from"].items():
        L.append("| %s | %d | %s%% |" % (name, x["rows"], x["pct"]))
    L += ["", "## By tier (the reading rows)", "", "| tier | n | 27B greedy | pooled 4 + 4 | chosen pool |", "|---|---|---|---|---|"]
    for t, x in rep["tiers"].items():
        L.append("| %s | %d | %.1f%% | %.1f%% | %.1f%% |" % (t, x["n"], x["27B greedy"], x["pooled 4 + 4"], x["chosen pool"]))
    L += ["", "## Checks", ""]
    for sp, x in rep["base_check"].items():
        L.append("- Pooled pick at 4 + 4 rebuilt on %s: %.1f%%, published %.1f%% (%d of %d rows; they can differ only by "
                 "re-judged verdicts or missing rows)." % (sp, x["rebuilt"], x["published"], x["rows"], x["of"]))
    for sp, x in rep["rejudged"].items():
        L.append("- %s: %d old answers re-judged by the box, %d verdicts changed." % (sp, x["judged"], x["changed"]))
    if rep.get("rescore"):
        x = rep["rescore"]
        L.append("- Verifier: %d old answers scored again, %d identical, mean |difference| %s, max %s." % (
            x["rows"], x["identical"], x["mean_abs"], x["max_abs"]))
    if rep.get("smoke"):
        L.append("- Smoke: %d of %d greedy answers identical to the recorded ones." % (rep["smoke"]["identical"], rep["smoke"]["rows"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(A.REPO / "runs" / "serve27b_2026-11-05"))
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    rep = build(run)
    (run / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
