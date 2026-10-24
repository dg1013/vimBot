#!/usr/bin/env python3
"""The top five methods on the 9B against the same methods on the 4B (CPU).

Rebuilds each method's per-row outcome for both models from recorded runs and
compares them on the same rows with exact McNemar tests. The rules are fixed
in runs/qwen9b_methods_2026-10-19/preregistration.md.

    python scripts/methods_9b/compare_9b.py --run runs/qwen9b_methods_2026-10-19 --lam-v9 <λ from val>

Methods (asked rows take the answer after the reply):
  1 verifier-gated ask   ask on the 20% with the lowest verifier best score, else the pick
  2 rule + verifier      ask when the top result holds <= 5 of 8, else the pick
  3 verifier pick        the best-scored of the 8 sampled candidates
  4 sample rule          ask when the top result holds <= 5 of 8, else the §6 answer
  5 always ask           the answer after the reply on every row
plus the pooled pick (the 4B's and the 9B's candidates together) and the
comparison points (§6 answer, most common of 8, any of 8 right, placebo
reply, asking only where it helps).
"""
import argparse
import json
import math
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"
ASK_AT_MOST = 5
GATE_RATE = 0.20
SUBSET_SEED = 20260908


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mcnemar_p(g, l):
    n = g + l
    if n == 0:
        return 1.0
    k = min(g, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired_ci(g, l, n):
    d = (g - l) / n
    se = math.sqrt(max(g + l - (g - l) ** 2 / n, 0.0)) / n
    return 100 * d, 100 * (d - 1.96 * se), 100 * (d + 1.96 * se)


def group(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["id"]].append(r)
    return out


def load_scores(path):
    return {(r["id"], r["cand_mode"], r["cand_command"]): r["score"] for r in load_jsonl(path)}


def pick(cands, scores, lam=0.0):
    """(label, best score) of the verifier's pick among sampled candidates:
    score + λ·log(count); ties by count, the greedy flag, then text (the
    reranker's rule)."""
    sampled = [c for c in cands if c.get("n", 0) > 0]
    if not sampled:
        return 0, float("-inf")

    def s(c):
        return scores[(c["id"], c["cand_mode"], c["cand_command"])] + lam * math.log(max(c.get("n", 0), 1))
    top = max(sampled, key=lambda c: (s(c), c.get("n", 0), bool(c.get("greedy")), c["cand_mode"], c["cand_command"]))
    return top["label"], s(top)


def pooled(c4, c9):
    """The two models' sampled candidates as one pool: one row per distinct
    command, counts summed, labels kept (the same command on the same buffer
    has one label). The greedy flag is the 9B's."""
    pool = {}
    for c, src in [(c, "4b") for c in c4] + [(c, "9b") for c in c9]:
        if c.get("n", 0) <= 0:
            continue
        k = (c["cand_mode"], c["cand_command"])
        if k not in pool:
            pool[k] = dict(c, n=0, greedy=False, sources=set())
        pool[k]["n"] += c["n"]
        pool[k]["sources"].add(src)
        pool[k]["greedy"] = pool[k]["greedy"] or (src == "9b" and bool(c.get("greedy")))
        if pool[k]["label"] != c["label"]:
            raise ValueError("one command, two labels: %r" % (k,))
    return list(pool.values())


def pooled_scores(pool, s4, s9, cid):
    """Scores for a pool from the two scored files of one verifier (the same
    verifier on the same input gives one score; the 9B file wins a tie)."""
    out = {}
    for c in pool:
        k = (cid, c["cand_mode"], c["cand_command"])
        out[k] = s9[k] if k in s9 else s4[k]
    return out


def model_methods(examples, scores, lam, verdicts, asks, ids):
    """Per-row outcomes of every method and comparison point for one model."""
    g = group(examples)
    out = defaultdict(dict)
    best = {}
    for i in ids:
        lab, b = pick(g.get(i, []), scores, lam)
        out["verifier pick"][i] = lab
        best[i] = b
        sampled = [c for c in g.get(i, []) if c.get("n", 0) > 0]
        out["any of 8 right"][i] = int(any(c["label"] for c in sampled))
        top = max(sampled, key=lambda c: (c["n"], bool(c.get("greedy")), c["cand_mode"], c["cand_command"])) if sampled else None
        out["most common of 8"][i] = top["label"] if top else 0
        v = verdicts[i]
        real, direct = int(bool(v["real"]["match"])), int(bool(v["direct"]["match"]))
        out["§6 answer"][i] = direct
        out["always ask"][i] = real
        out["placebo reply"][i] = int(bool(v["placebo"]["match"]))
        out["ask only where it helps"][i] = max(real, direct)
        out["sample rule"][i] = real if asks[i] else direct
        out["rule + verifier"][i] = real if asks[i] else lab
    order = sorted(ids, key=lambda i: (best[i], i))
    gated = set(order[:round(GATE_RATE * len(ids))])
    for i in ids:
        out["verifier-gated ask"][i] = out["always ask"][i] if i in gated else out["verifier pick"][i]
    return out, {"rule_asks": sum(asks[i] for i in ids) / len(ids), "gated": gated}


def rule_asks(report_path, ids):
    view = json.loads(pathlib.Path(report_path).read_text(encoding="utf-8"))
    return {i: view[i]["top_share"] * view[i]["n"] <= ASK_AT_MOST + 1e-9 for i in ids}


def select_ids(split):
    if split != "val":
        return set()
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    from train_sft import stratified_subset
    cases = load_jsonl(REPO / "data" / "splits" / "val.jsonl")
    return {c["id"] for c in stratified_subset(cases, 250, SUBSET_SEED)}


def build(run, split, lam9, smoke=False):
    cases = {c["id"]: c for c in load_jsonl(REPO / "data" / "splits" / ("%s.jsonl" % split))}
    ex9 = load_jsonl(run / ("examples_%s.jsonl" % split))
    ver9 = {r["id"]: r for r in load_jsonl(run / "clarify" / split / "verdicts_served9b.jsonl")}
    ids = [i for i in cases if i in ver9 and i in {r["id"] for r in ex9}]
    if not smoke and len(ids) != len(cases):
        raise SystemExit("%s: the 9B covers %d of %d rows" % (split, len(ids), len(cases)))
    four_dir = RUNS / "reranker_2026-10-16"
    ex4 = load_jsonl(four_dir / ("%s_examples.jsonl" % split))
    s4_4 = load_scores(four_dir / "verifier-4b" / ("test_scores.jsonl" if split == "test" else "val_scores_step818.jsonl"))
    ver4 = {r["id"]: r for r in load_jsonl(RUNS / ("clarify_2026-10-07/test" if split == "test" else "clarify_2026-10-06/val")
                                           / "verdicts_served.jsonl")}
    asks4 = rule_asks(RUNS / "sample_rule_2026-10-09" / split / "report.json", ids)
    asks9 = rule_asks(run / ("sample_rule_%s" % split) / "report.json", ids)
    s4_9 = load_scores(run / ("v4_scores_%s.jsonl" % split))
    s9_9 = load_scores(run / ("v9_scores_%s.jsonl" % split))
    s9_4 = load_scores(run / ("v9_scores_4b_%s.jsonl" % split))
    m4, info4 = model_methods(ex4, s4_4, 0.0, ver4, asks4, ids)
    m9v4, info9 = model_methods(ex9, s4_9, 0.0, ver9, asks9, ids)
    m9v9, _ = model_methods(ex9, s9_9, lam9, ver9, asks9, ids)
    m4v9, _ = model_methods(ex4, s9_4, lam9, ver4, asks4, ids)
    g4, g9 = group(ex4), group(ex9)
    pool = {}
    for name, sa, sb, lam in (("V4", s4_4, s4_9, 0.0), ("V9", s9_4, s9_9, lam9)):
        lab = {}
        for i in ids:
            p = pooled(g4.get(i, []), g9.get(i, []))
            lab[i] = pick(p, pooled_scores(p, sa, sb, i), lam)[0]
        pool["pooled pick (%s)" % name] = lab
    pool["any of the 16 right"] = {i: int(m4["any of 8 right"][i] or m9v4["any of 8 right"][i]) for i in ids}
    recorded9 = {}
    rec_path = (RUNS / "remeasure_2026-09-25" / "v2" / "qwen3.5-9b-grpo-gated" / "exec_test.jsonl" if split == "test"
                else RUNS / "qwen3.5-9b-grpo-gated" / "exec_val.jsonl")
    if rec_path.exists():
        recorded9 = {r["id"]: int(bool(r["match"])) for r in load_jsonl(rec_path)}
    parity = sum(1 for i in ids if i in recorded9 and recorded9[i] == m9v4["§6 answer"][i]) / max(1, len(ids))
    return {"ids": ids, "m4": m4, "m9v4": m9v4, "m9v9": m9v9, "m4v9": m4v9, "pool": pool, "parity": parity,
            "recorded9": recorded9, "asks4": info4["rule_asks"], "asks9": info9["rule_asks"],
            "select": select_ids(split)}


METHODS = ["verifier-gated ask", "rule + verifier", "verifier pick", "sample rule", "always ask"]
POINTS = ["§6 answer", "most common of 8", "placebo reply", "any of 8 right", "ask only where it helps"]


def pct(o, ids):
    return 100 * sum(o[i] for i in ids) / len(ids)


def line(label, a, b, ids):
    g = sum(1 for i in ids if b[i] and not a[i])
    l = sum(1 for i in ids if a[i] and not b[i])
    d, lo, hi = paired_ci(g, l, len(ids))
    return "| %s | %.1f%% | %.1f%% | +%d / −%d | %+.1f (%+.1f to %+.1f) | %s |" % (
        label, pct(a, ids), pct(b, ids), g, l, d, lo, hi, ("%.3f" % mcnemar_p(g, l)) if mcnemar_p(g, l) >= 0.001 else "< 0.001")


def report(splits):
    L = ["# The top five methods on the 9B (2026-10-19)", "",
         "Rules: `preregistration.md`. Paired exact McNemar on the same rows; asked rows use each model's own "
         "dialogue with the simulated user (Qwen3.5-9B, which knows the wanted result).", ""]
    for split, S in splits.items():
        ids = S["ids"]
        L += ["## %s (%d rows)" % (split, len(ids)), "",
              "Parity: the harness's §6 answers for the 9B agree with the recorded greedy answers on %.1f%% of rows. "
              "The sample rule asks on %.1f%% of rows for the 4B and %.1f%% for the 9B." % (
                  100 * S["parity"], 100 * S["asks4"], 100 * S["asks9"]), "",
              "| method | 4B (V4) | 9B (V4) | 9B (V9) | 4B (V9) |", "|---|---|---|---|---|"]
        for m in METHODS + POINTS:
            L.append("| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
                m, pct(S["m4"][m], ids), pct(S["m9v4"][m], ids), pct(S["m9v9"][m], ids), pct(S["m4v9"][m], ids)))
        for name, o in S["pool"].items():
            L.append("| %s | — | %.1f%% | — | — |" % (name, pct(o, ids)))
        L += ["", "| comparison | first | second | gained / lost | Δ (95% CI) | p |", "|---|---|---|---|---|---|"]
        base9 = S["m9v4"]["§6 answer"]
        tag = " **(P1)**" if split == "test" else ""
        L.append(line("9B: verifier-gated ask (V4) vs its §6 answer" + tag, base9, S["m9v4"]["verifier-gated ask"], ids))
        tag = " **(P2)**" if split == "test" else ""
        L.append(line("verifier-gated ask (V4): 4B vs 9B" + tag, S["m4"]["verifier-gated ask"], S["m9v4"]["verifier-gated ask"], ids))
        for m in METHODS[1:]:
            L.append(line("9B: %s (V4) vs its §6 answer" % m, base9, S["m9v4"][m], ids))
        for m in METHODS:
            L.append(line("%s: 4B vs 9B (V4)" % m, S["m4"][m], S["m9v4"][m], ids))
        for m in METHODS[:3]:
            L.append(line("9B, %s: V4 vs V9" % m, S["m9v4"][m], S["m9v9"][m], ids))
        L.append(line("pooled pick (V4) vs the 9B's own pick (V4)", S["m9v4"]["verifier pick"], S["pool"]["pooled pick (V4)"], ids))
        L.append(line("pooled pick (V4) vs the 4B's own pick (V4)", S["m4"]["verifier pick"], S["pool"]["pooled pick (V4)"], ids))
        L.append(line("pooled pick (V9) vs the 9B's own pick (V9)", S["m9v9"]["verifier pick"], S["pool"]["pooled pick (V9)"], ids))
        if S["select"]:
            un = [i for i in ids if i not in S["select"]]
            L += ["", "V9's checkpoint and λ were chosen on the 250-row selection subset; on the %d untouched val rows: "
                  "9B verifier pick V4 %.1f%%, V9 %.1f%%; pooled V4 %.1f%%, V9 %.1f%%." % (
                      len(un), pct(S["m9v4"]["verifier pick"], un), pct(S["m9v9"]["verifier pick"], un),
                      pct(S["pool"]["pooled pick (V4)"], un), pct(S["pool"]["pooled pick (V9)"], un))]
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(RUNS / "qwen9b_methods_2026-10-19"))
    ap.add_argument("--lam-v9", type=float, required=True)
    ap.add_argument("--smoke", action="store_true", help="partial rows; no coverage checks")
    args = ap.parse_args(argv)
    run = pathlib.Path(args.run)
    splits = {s: build(run, s, args.lam_v9, args.smoke) for s in ("test", "val")
              if (run / "clarify" / s / "verdicts_served9b.jsonl").exists()}
    text = report(splits)
    (run / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
