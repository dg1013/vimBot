#!/usr/bin/env python3
"""Retrieved in-context examples (kNN few-shot) for the §6 prompt: plan and report.

Built 2026-10-18 as a new folder (owner's proposal: "find the k most similar
training examples and put them in the prompt; compare zero-shot, random
examples, retrieved examples and the fine-tuned model on test v2, with
McNemar tests"). The pre-registration is runs/retrieval_2026-10-18/
preregistration.md; the prompt side is prompt.build_messages(examples), which
§6 already sanctions for baselines; the generation side is the one path in
scripts/baseline.py (generate_predictions, `examples_for`).

Two subcommands, both CPU only:

  plan    choose the examples for every case of a split, one file per arm:
            retrieved k   BM25 over instruction tokens, train split only
            random k      per-row draws, seeded by (seed, case id)
            k = 0         the zero-shot plan (no examples), for uniformity
          Every candidate passes the leakage guard (no shared leakage key with
          the query, build_splits.leakage_keys) and the token budget
          (max_seq_len - max_new_tokens); when a prompt overflows, the least
          similar example is dropped first and the effective k is recorded.
          Examples are ordered least similar first, so the nearest one sits
          next to the query.
  report  fold the arms' artifacts (completions, exec logs, summaries, plans)
          into report.md and report.json: the table, the paired exact McNemar
          tests, the gap closed with a bootstrap CI, and the breakdowns.

Retrieval sees the TRAIN split only; val never enters the index. No gold
command string of test v2 exists in train (checked 2026-10-18), but the
near-duplicate key is a token-set hash, so instructions differing by one
literal ("Delete 2 characters" / "Delete 3 characters") can sit on either side
of the split: 48 of 429 test rows have a nearest train row with the same
command shape up to literals and numbers. The plan records that flag per row
(`same_shape_neighbor`) and the report splits every arm by it.
"""
import argparse
import json
import math
import pathlib
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from prompt import messages_for_case          # noqa: E402

TOKEN = re.compile(r"[a-z0-9]+")
MAX_NEW_TOKENS = 192                          # baseline.py's default completion cap
OVERFETCH = 4                                 # candidates fetched per wanted example
SHOT_SEED = 20261018


def tokens(text: str) -> list[str]:
    """Lower-cased alphanumeric runs: the same token family as the near-dup hash."""
    return TOKEN.findall((text or "").lower())


class BM25:
    """Okapi BM25 over tokenized documents, with an inverted index."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = [len(d) for d in docs]
        self.avg = (sum(self.len) / len(docs)) if docs else 0.0
        self.post = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.post[t].append(i)
        n = len(docs)
        self.idf = {t: math.log((n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in self.post.items()}

    def scores(self, query_tokens):
        """{doc index: score} for every document sharing a term with the query."""
        acc = defaultdict(float)
        for t in set(query_tokens):
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i in self.post[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                acc[i] += idf * f * (self.k1 + 1) / denom
        return acc

    def top(self, query_tokens, k, exclude=()):
        """The k best (score, index) pairs, highest first; ties broken by index."""
        acc = self.scores(query_tokens)
        ranked = sorted(((s, i) for i, s in acc.items() if i not in exclude), key=lambda x: (-x[0], x[1]))
        return ranked[:k]


# ------------------------------------------------------------ guards and flags

def shared_leakage_keys(case, example):
    """Leakage keys (build_splits.leakage_keys) the two rows have in common."""
    from build_splits import leakage_keys
    return sorted(set(leakage_keys(case)) & set(leakage_keys(example)))


def command_shape(mode, command):
    """A gold command with its literals masked: digits -> N, and in ex mode
    letter runs -> W. Two commands with one shape differ only in literals.
    Analysis flag only; never used to choose examples."""
    s = re.sub(r"\d+", "N", command.strip().lstrip(":") if mode == "ex" else command)
    if mode == "ex":
        s = re.sub(r"[A-Za-z_]+", "W", s)
    return (mode, s)


# ------------------------------------------------------------ planning

def prompt_tokens(count, case, examples):
    """Prompt length in tokens under `count(messages)`."""
    return count(messages_for_case(case, examples))


def plan_case(case, train, arm, k, count, budget, index=None, seed=SHOT_SEED):
    """Choose up to k examples for one case. `arm` is "retrieved" or "random".

    Returns the plan row: example ids (in prompt order, least similar first),
    scores, effective k, prompt tokens, how many were dropped for the budget,
    how many the leakage guard excluded, and the same-shape flag."""
    if k == 0:
        return {"id": case["id"], "arm": "zero-shot", "k": 0, "examples": [], "scores": [],
                "effective_k": 0, "prompt_tokens": prompt_tokens(count, case, []),
                "dropped_for_budget": 0, "guard_excluded": 0, "same_shape_neighbor": False,
                "nearest_score": None}
    if arm == "retrieved":
        cands = index.top(tokens(case["instruction"]), OVERFETCH * k)
    elif arm == "random":
        rng = random.Random("%d:%s" % (seed, case["id"]))
        cands = [(0.0, i) for i in rng.sample(range(len(train)), min(len(train), OVERFETCH * k))]
    else:
        raise ValueError(arm)
    chosen, guard_excluded = [], 0
    for score, i in cands:
        if shared_leakage_keys(case, train[i]):
            guard_excluded += 1
            continue
        chosen.append((score, i))
        if len(chosen) == k:
            break
    chosen.reverse()                                   # least similar first, nearest last
    dropped = 0
    while chosen:
        n = prompt_tokens(count, case, [train[i] for _, i in chosen])
        if n <= budget:
            break
        chosen.pop(0)                                  # drop the least similar
        dropped += 1
    n = prompt_tokens(count, case, [train[i] for _, i in chosen])
    shape = command_shape(case["mode"], case["gold_command"])
    return {"id": case["id"], "arm": arm, "k": k,
            "examples": [train[i]["id"] for _, i in chosen],
            "scores": [round(s, 3) for s, _ in chosen],
            "effective_k": len(chosen), "prompt_tokens": n,
            "dropped_for_budget": dropped, "guard_excluded": guard_excluded,
            "same_shape_neighbor": any(command_shape(train[i]["mode"], train[i]["gold_command"]) == shape
                                       for _, i in chosen),
            "nearest_score": (round(chosen[-1][0], 3) if chosen and arm == "retrieved" else None)}


def plan_split(cases, train, arm, k, count, budget, seed=SHOT_SEED):
    index = BM25([tokens(c["instruction"]) for c in train]) if arm == "retrieved" else None
    return [plan_case(c, train, arm, k, count, budget, index, seed) for c in cases]


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def tokenizer_counter(model_name):
    """count(messages) under the registry model's real chat template."""
    from baseline import load_registry
    from transformers import AutoTokenizer
    reg = load_registry(model_name)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    kw = reg.get("chat_template_kwargs") or {}

    def count(messages):
        out = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kw)
        if hasattr(out, "keys"):                      # newer transformers return a BatchEncoding
            out = out["input_ids"]
        if hasattr(out, "tolist"):
            out = out.tolist()
        if out and isinstance(out[0], list):
            out = out[0]
        return len(out)
    return count, reg


def cmd_plan(args):
    cases = load_jsonl(args.cases)
    train = load_jsonl(args.train)
    count, reg = tokenizer_counter(args.model)
    budget = int(reg["max_seq_len"]) - args.max_new_tokens
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"cases": str(args.cases), "n_cases": len(cases), "train": str(args.train), "n_train": len(train),
                "model": args.model, "max_seq_len": reg["max_seq_len"], "budget_tokens": budget,
                "seed": args.seed, "plans": {}}
    for spec in args.plans:
        arm, k = spec.split(":")
        k = int(k)
        rows = plan_split(cases, train, arm, k, count, budget, args.seed)
        name = "plan_%s%d.jsonl" % ({"retrieved": "retr", "random": "random", "zero-shot": "k"}[arm], k)
        write_jsonl(out / name, rows)
        eff = [r["effective_k"] for r in rows]
        summary = {"arm": arm, "k": k, "file": name,
                   "effective_k_mean": round(statistics.fmean(eff), 2) if eff else 0,
                   "rows_short_of_k": sum(1 for e in eff if e < k),
                   "guard_excluded_total": sum(r["guard_excluded"] for r in rows),
                   "prompt_tokens_mean": round(statistics.fmean(r["prompt_tokens"] for r in rows), 1),
                   "prompt_tokens_max": max(r["prompt_tokens"] for r in rows),
                   "same_shape_neighbor_rows": sum(r["same_shape_neighbor"] for r in rows)}
        manifest["plans"][name] = summary
        print("%-22s effective k %.2f (short on %d rows)  guard excluded %d  prompt tokens mean %.0f max %d  same-shape rows %d" % (
            name, summary["effective_k_mean"], summary["rows_short_of_k"], summary["guard_excluded_total"],
            summary["prompt_tokens_mean"], summary["prompt_tokens_max"], summary["same_shape_neighbor_rows"]), flush=True)
    (out / "plan_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------ statistics

def mcnemar_p(gained, lost):
    """Exact two-sided McNemar (binomial on the discordant pairs); the same
    function as scripts/reranker/verifier_lib.mcnemar_p."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a, b):
    """(gained, lost, n) of b over a; a and b map id -> 0/1 on the same ids."""
    ids = [i for i in a if i in b]
    g = sum(1 for i in ids if b[i] and not a[i])
    l = sum(1 for i in ids if a[i] and not b[i])
    return g, l, len(ids)


def gap_closed(zero, retr, tuned, ids, n_boot=10000, seed=SHOT_SEED):
    """(retrieved - zero-shot) / (fine-tuned - zero-shot) on the rows, with a
    paired bootstrap percentile interval. None when the denominator is <= 0."""
    def stat(sel):
        z = sum(zero[i] for i in sel); r = sum(retr[i] for i in sel); t = sum(tuned[i] for i in sel)
        return None if t - z <= 0 else (r - z) / (t - z)
    point = stat(ids)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        sel = [ids[rng.randrange(len(ids))] for _ in ids]
        s = stat(sel)
        if s is not None:
            boots.append(s)
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else None
    hi = boots[min(len(boots) - 1, int(0.975 * len(boots)))] if boots else None
    return point, lo, hi


# ------------------------------------------------------------ report

def load_arm(run_dir, arm, split):
    d = pathlib.Path(run_dir) / arm
    ex = {r["id"]: r for r in load_jsonl(d / ("exec_%s.jsonl" % split))}
    comp = {r["id"]: r for r in load_jsonl(d / ("completions_%s.jsonl" % split))}
    summary = json.loads((d / ("baseline_%s.json" % split)).read_text(encoding="utf-8"))
    return {"exec": ex, "comp": comp, "summary": summary}


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def cmd_report(args):
    run = pathlib.Path(args.run)
    cases = load_jsonl(args.cases)
    ids = [c["id"] for c in cases]
    byid = {c["id"]: c for c in cases}
    arms = {}
    for arm in args.arms:
        try:
            arms[arm] = load_arm(run, arm, args.split)
        except FileNotFoundError:
            print("  (arm %s has no complete artifacts yet; skipped)" % arm)
    if not arms:
        print("nothing to report"); return 1
    plans = {p.stem: {r["id"]: r for r in load_jsonl(p)} for p in sorted(run.glob("plan_*.jsonl"))}
    arm_plan = {arm: plans.get(args.arm_plans.get(arm, "")) for arm in arms}
    match = {arm: {i: int(bool(a["exec"][i]["match"])) for i in ids if i in a["exec"]} for arm, a in arms.items()}
    valid = {arm: {i: int(bool(a["exec"][i]["valid"])) for i in ids if i in a["exec"]} for arm, a in arms.items()}
    base_arm = args.zero_shot if args.zero_shot in arms else next(iter(arms))

    lines = ["# Retrieved in-context examples on %s (%d rows)" % (args.split, len(ids)), "",
             "Run folder `%s`. Greedy decoding through `baseline.generate_predictions`; k-shot pairs "
             "rendered by `prompt.build_messages`; BM25 over instruction tokens on the train split; per-row "
             "seeded random draws as the control. Pre-registration: `preregistration.md`." % run, ""]
    # 1. the table
    lines += ["## Table", "",
              "| arm | k (effective) | parse fail | valid | **exec-match** | prompt tok | output tok | latency p50 | tokens / solved (× zero-shot) |",
              "|---|---|---|---|---|---|---|---|---|"]
    def tokens_per_solved(arm):
        a = arms[arm]; n = len(match[arm]); m = sum(match[arm].values())
        tot = sum(a["comp"][i]["prompt_tokens"] + a["comp"][i]["output_tokens"] for i in match[arm])
        return (tot / m) if m else float("inf")
    tps0 = tokens_per_solved(base_arm)
    table = {}
    for arm, a in arms.items():
        m = a["summary"]["metrics"]; n = len(match[arm])
        keff = statistics.fmean(a["comp"][i].get("k_examples", 0) for i in match[arm])
        tps = tokens_per_solved(arm)
        table[arm] = {"n": n, "k_effective": round(keff, 2), "parse_failures": m["parse_failures"],
                      "valid_pct": round(pct(sum(valid[arm].values()), n), 1),
                      "exec_match": sum(match[arm].values()), "exec_match_pct": round(pct(sum(match[arm].values()), n), 1),
                      "prompt_tokens_mean": m["prompt_tokens_mean"], "output_tokens_mean": m["output_tokens_mean"],
                      "latency_p50": m["latency_s_p50"], "tokens_per_solved": round(tps, 1),
                      "tokens_per_solved_rel": round(tps / tps0, 2) if tps0 else None}
        t = table[arm]
        lines.append("| %s | %.2f | %d | %.1f%% | **%.1f%%** (%d) | %.0f | %.1f | %.2f s | %.2f× |" % (
            arm, t["k_effective"], t["parse_failures"], t["valid_pct"], t["exec_match_pct"], t["exec_match"],
            t["prompt_tokens_mean"], t["output_tokens_mean"], t["latency_p50"], t["tokens_per_solved_rel"]))
    lines.append("")
    # 2. paired tests
    lines += ["## Paired tests (exact McNemar on discordant rows)", "",
              "| comparison | gained | lost | Δ points | p |", "|---|---|---|---|---|"]
    tests = {}
    for a_name, b_name, label in args.pairs:
        if a_name in match and b_name in match:
            g, l, n = paired(match[a_name], match[b_name])
            p = mcnemar_p(g, l)
            tests[label] = {"a": a_name, "b": b_name, "gained": g, "lost": l, "n": n,
                            "delta_points": round(100 * (g - l) / n, 1), "p": p}
            lines.append("| %s: %s vs %s | +%d | −%d | %+.1f | %s |" % (
                label, b_name, a_name, g, l, 100 * (g - l) / n, "%.3f" % p if p >= 0.001 else "< 0.001"))
    lines.append("")
    # 3. gap closed
    gaps = {}
    if args.zero_shot in match and args.tuned in match:
        lines += ["## Gap closed without training", "",
                  "(retrieved − zero-shot) / (fine-tuned − zero-shot), paired bootstrap 95% interval over rows.", ""]
        for arm in arms:
            if arm in (args.zero_shot, args.tuned) or not arm.startswith(args.base_prefix):
                continue
            common = [i for i in ids if i in match[arm] and i in match[args.zero_shot] and i in match[args.tuned]]
            pt, lo, hi = gap_closed(match[args.zero_shot], match[arm], match[args.tuned], common)
            gaps[arm] = {"point": pt, "lo": lo, "hi": hi}
            if pt is not None:
                lines.append("- **%s**: %.0f%% of the gap (95%% CI %.0f%% to %.0f%%)" % (arm, 100 * pt, 100 * lo, 100 * hi))
        lines.append("")
    # 4. breakdowns
    def breakdown(title, key_fn, min_n=10):
        groups = defaultdict(list)
        for i in ids:
            groups[key_fn(byid[i])].append(i)
        cols = [a for a in args.breakdown_arms if a in match]
        out = ["## " + title, "", "| group | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        rows = {}
        for g in sorted(groups, key=lambda x: str(x)):
            sel = [i for i in groups[g] if all(i in match[a] for a in cols)]
            if len(sel) < min_n:
                continue
            rows[str(g)] = {a: round(pct(sum(match[a][i] for i in sel), len(sel)), 1) for a in cols}
            rows[str(g)]["n"] = len(sel)
            out.append("| %s | %d | " % (g, len(sel)) + " | ".join("%.1f%%" % rows[str(g)][a] for a in cols) + " |")
        return out + [""], rows
    cat_lines, cat_rows = breakdown("By category (n ≥ 10)", lambda c: c["category"])
    tier_lines, tier_rows = breakdown("By tier", lambda c: c.get("tier") or "T?", min_n=1)
    lines += cat_lines + tier_lines
    # 5. the same-shape neighbour split, from the k=5 retrieved plan
    shape_rows = {}
    ref_plan = arm_plan.get(args.shape_arm) if args.shape_arm in arm_plan else None
    if ref_plan:
        flagged = [i for i in ids if ref_plan.get(i, {}).get("same_shape_neighbor")]
        rest = [i for i in ids if i not in set(flagged)]
        cols = [a for a in args.breakdown_arms if a in match]
        lines += ["## Rows whose retrieved examples include a same-shape gold (literals aside)", "",
                  "Flag from the `%s` plan: a retrieved example's gold has the query gold's shape with digits and, "
                  "in ex mode, letter runs masked. Not a leak of the answer string (none repeats), but the strongest "
                  "kind of help; the effect should be read on both halves." % args.shape_arm, "",
                  "| rows | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
        for label, sel in (("same-shape neighbour retrieved", flagged), ("no same-shape neighbour", rest)):
            sel = [i for i in sel if all(i in match[a] for a in cols)]
            shape_rows[label] = {"n": len(sel), **{a: round(pct(sum(match[a][i] for i in sel), len(sel)), 1) for a in cols}}
            lines.append("| %s | %d | " % (label, len(sel)) + " | ".join("%.1f%%" % shape_rows[label][a] for a in cols) + " |")
        lines.append("")
    # 6. plan facts
    man_path = run / "plan_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text(encoding="utf-8"))
        lines += ["## Plans", "", "Index: %d train rows; budget %d prompt tokens (max_seq_len %s − %d)." % (
            man["n_train"], man["budget_tokens"], man["max_seq_len"], MAX_NEW_TOKENS), "",
            "| plan | effective k | rows short of k | guard excluded | prompt tokens mean / max | same-shape rows |",
            "|---|---|---|---|---|---|"]
        for name, s in man["plans"].items():
            lines.append("| %s | %.2f | %d | %d | %.0f / %d | %d |" % (
                name, s["effective_k_mean"], s["rows_short_of_k"], s["guard_excluded_total"],
                s["prompt_tokens_mean"], s["prompt_tokens_max"], s["same_shape_neighbor_rows"]))
        lines.append("")
    text = "\n".join(lines)
    (run / "report.md").write_text(text + "\n", encoding="utf-8")
    (run / "report.json").write_text(json.dumps({"table": table, "tests": tests, "gap_closed": gaps,
                                                 "by_category": cat_rows, "by_tier": tier_rows,
                                                 "same_shape": shape_rows}, indent=1) + "\n", encoding="utf-8")
    print(text)
    return 0


DEFAULT_ARMS = ["base-k0", "base-random5", "base-retr1", "base-retr3", "base-retr5", "base-retr8",
                "adapter-k0", "adapter-retr5"]
DEFAULT_ARM_PLANS = {"base-k0": "plan_k0", "base-random5": "plan_random5", "base-retr1": "plan_retr1",
                     "base-retr3": "plan_retr3", "base-retr5": "plan_retr5", "base-retr8": "plan_retr8",
                     "adapter-k0": "plan_k0", "adapter-retr5": "plan_retr5"}
DEFAULT_PAIRS = [("base-random5", "base-retr5", "primary: retrieval vs random, k = 5"),
                 ("base-k0", "base-retr5", "retrieved k = 5 vs zero-shot"),
                 ("base-k0", "base-random5", "random k = 5 vs zero-shot"),
                 ("base-retr5", "adapter-k0", "fine-tuned vs retrieved k = 5"),
                 ("adapter-k0", "adapter-retr5", "fine-tuned + retrieved k = 5 vs fine-tuned"),
                 ("base-k0", "base-retr1", "retrieved k = 1 vs zero-shot"),
                 ("base-k0", "base-retr3", "retrieved k = 3 vs zero-shot"),
                 ("base-k0", "base-retr8", "retrieved k = 8 vs zero-shot"),
                 ("base-retr5", "base-retr8", "retrieved k = 8 vs k = 5")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="choose examples per case, per arm (CPU; needs the tokenizer)")
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--cases", default=str(REPO / "data/splits/test.jsonl"))
    p.add_argument("--train", default=str(REPO / "data/splits/train.jsonl"))
    p.add_argument("--out", required=True, help="run folder; plan_<arm><k>.jsonl files land here")
    p.add_argument("--plans", nargs="+", default=["zero-shot:0", "random:5", "retrieved:1", "retrieved:3",
                                                   "retrieved:5", "retrieved:8"],
                   help="arm:k specs; arm in zero-shot, random, retrieved")
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--seed", type=int, default=SHOT_SEED)
    p.set_defaults(func=cmd_plan)
    r = sub.add_parser("report", help="fold the arms' artifacts into report.md")
    r.add_argument("--run", required=True)
    r.add_argument("--cases", default=str(REPO / "data/splits/test.jsonl"))
    r.add_argument("--split", default="test")
    r.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    r.add_argument("--zero-shot", default="base-k0")
    r.add_argument("--tuned", default="adapter-k0")
    r.add_argument("--base-prefix", default="base-")
    r.add_argument("--shape-arm", default="base-retr5")
    r.add_argument("--breakdown-arms", nargs="+", default=["base-k0", "base-random5", "base-retr5", "adapter-k0", "adapter-retr5"])
    r.set_defaults(func=cmd_report, arm_plans=DEFAULT_ARM_PLANS, pairs=DEFAULT_PAIRS)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
