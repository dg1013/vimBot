#!/usr/bin/env python3
"""Every approach measured on test v2, side by side, from the recorded runs.

Built 2026-10-18 as a new folder (owner: "Let's look at how everything looks
when compared to each other and the original best performing models (the post
trained model, along with the clarification question one)"). Nothing here
generates or trains: each row's per-case outcome on the 429 test v2 rows is
rebuilt from artifacts already on disk, so every comparison is paired on the
same rows. Each rebuilt row is checked against the figure its own report
recorded (`RECORDED`), and the script stops if one does not reproduce.

The two reference rows:
  served       the served gated-GRPO adapter, one greedy answer (54.3%)
  sample-rule  the deployed round trip: 8 samples; ask when the most common
               result holds <= 5 of 8 (the simulated user replies); otherwise
               the greedy answer (59.7%)

Writes runs/comparison_2026-10-18/report.md and data.json (the page's data).

    python scripts/comparison/compare_test_v2.py
"""
import json
import math
import pathlib
import sys
from collections import Counter, defaultdict

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "retrieval"))
OUT = REPO / "runs" / "comparison_2026-10-18"
INPUTS = OUT / "inputs"
RUNS = REPO / "runs"

# What each source's own report recorded, as exec-match rows of 429.
RECORDED = {"served": 233, "sample-rule": 256, "verifier-4b": 257, "verifier-2b": 250,
            "rule+verifier-4b": 260, "verifier-ask-20": 263, "always-ask": 251, "oracle8": 291,
            "base-k0": 59, "retr5": 146, "retr8": 162, "adapter-retr5": 225, "majority-result": 248}


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def mcnemar_p(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(ref, row, ids):
    g = sum(1 for i in ids if row[i] and not ref[i])
    l = sum(1 for i in ids if ref[i] and not row[i])
    return g, l, mcnemar_p(g, l)


def key(buf):
    s = buf or ""
    return tuple((s[:-1] if s.endswith("\n") else s).split("\n"))


def exec_match(path):
    return {r["id"]: int(bool(r["match"])) for r in load_jsonl(path)}


def majority_by_result(probe_rows, cases):
    """Per case: does the most common RESULT among the samples equal the gold?
    (The rule buckets by what each command does, so it needs the sandbox; as a
    way to pick the answer it is outside the execution boundary.)"""
    from executor import run_vim
    out = {}
    for r in probe_rows:
        c = cases.get(r["id"])
        if c is None:
            continue
        masses = Counter()
        for e in r["commands"]:
            res = run_vim(c["buffer_before"], e["command"], mode=e["mode"], cursor=tuple(c.get("cursor") or (1, 1)))
            if res["buffer_after"] is None or (e["mode"] == "ex" and not res["was_valid"]):
                continue
            masses[key(res["buffer_after"])] += e["n"]
        # ties go to the bucket seen first, i.e. the one holding the most frequent command text
        # (probe rows list commands by count); max() keeps the first of equal keys
        top = max(masses.items(), key=lambda kv: kv[1])[0] if masses else None
        out[r["id"]] = int(top == key(c["buffer_after"]))
    return out


def build(cases):
    ids = [c["id"] for c in cases]
    byid = {c["id"]: c for c in cases}
    rows = {}

    def add(name, label, group, outcome, **meta):
        missing = [i for i in ids if i not in outcome]
        if missing:
            raise SystemExit("%s: no outcome for %d rows (first %s)" % (name, len(missing), missing[0]))
        rows[name] = {"name": name, "label": label, "group": group, "outcome": {i: int(outcome[i]) for i in ids}, **meta}

    # ---- the served adapter and its line
    served_comp = {r["id"]: r for r in load_jsonl(RUNS / "qwen3.5-4b-grpo-nearcliff" / "completions_test.jsonl")}
    served = exec_match(RUNS / "qwen3.5-4b-grpo-nearcliff" / "exec_test.jsonl")
    greedy_tok = sum(served_comp[i]["output_tokens"] for i in ids) / len(ids)
    one = dict(gens=1, sandbox="no", asks=0.0, extra="—", deploy="served", gen_tokens=greedy_tok,
               prompt_tokens=sum(served_comp[i]["prompt_tokens"] for i in ids) / len(ids), latency="0.22 s")
    rv = RUNS / "remeasure_2026-09-25" / "v2"
    train_rows = [("sft-r1", "SFT round 1", rv / "qwen3.5-4b-sft"),
                  ("sft-grpo", "SFT + GRPO", rv / "qwen3.5-4b-grpo"),
                  ("served", "SFT + GRPO + difficulty-gated GRPO (the served adapter)", None),
                  ("cliff-chain", "served → cliff SFT → gated GRPO (stage C)", rv / "qwen3.5-4b-grpo-gated-cliffleg"),
                  ("sft-ds100", "SFT re-run on 100% of train (data-scaling point)", RUNS / "qwen3.5-4b-sft-ds100"),
                  ("sft-plusglobal", "SFT on train + 926 proposed global prompts", RUNS / "qwen3.5-4b-sft-plusglobal")]
    retr = RUNS / "retrieval_2026-10-18"
    add("base-k0", "Qwen3.5-4B base, zero-shot", "A. One greedy answer: the 4B, trained", exec_match(retr / "base-k0" / "exec_test.jsonl"),
        **{**one, "deploy": "—", "note": "fla-kernel rerun 2026-10-18; recorded 14.0% on the pure-torch path"})
    for name, label, d in train_rows:
        out = served if d is None else exec_match(d / "exec_test.jsonl")
        add(name, label, "A. One greedy answer: the 4B, trained", out,
            **{**one, "deploy": "served" if name == "served" else "—"})
    others = [("9b-gated", "Qwen3.5-9B + SFT + GRPO + gated (step 0, NF4)", "qwen3.5-9b-grpo-gated"),
              ("seed-sft", "Seed-Coder-8B + SFT (NF4)", "seed-coder-8b-sft"),
              ("coder-sft", "Qwen2.5-Coder-7B + SFT (NF4)", "qwen2.5-coder-7b-sft"),
              ("granite-sft", "Granite-4.2-8B + SFT (NF4)", "granite-4.2-8b-sft"),
              ("q3-4b-sft", "Qwen3-4B-2507 + SFT", "qwen3-4b-2507-sft"),
              ("2b-sft", "Qwen3.5-2B + SFT", "qwen3.5-2b-sft"),
              ("0.8b-sft", "Qwen3.5-0.8B + SFT", "qwen3.5-0.8b-sft")]
    for name, label, d in others:
        add(name, label, "B. One greedy answer: other models, same recipe", exec_match(rv / d / "exec_test.jsonl"),
            **{**one, "deploy": "—", "latency": "—", "gen_tokens": None, "prompt_tokens": None})

    # ---- training-free: retrieved in-context examples on the base model
    def retr_meta(arm):
        comp = {r["id"]: r for r in load_jsonl(retr / arm / "completions_test.jsonl")}
        return dict(gens=1, sandbox="no", asks=0.0, extra="BM25 index", deploy="possible",
                    gen_tokens=sum(comp[i]["output_tokens"] for i in ids) / len(ids),
                    prompt_tokens=sum(comp[i]["prompt_tokens"] for i in ids) / len(ids), latency="—")
    for arm, label in (("base-random5", "base + 5 random train examples"), ("base-retr1", "base + 1 retrieved example"),
                       ("base-retr3", "base + 3 retrieved examples"), ("base-retr5", "base + 5 retrieved examples"),
                       ("base-retr8", "base + 8 retrieved examples")):
        add({"base-retr5": "retr5", "base-retr8": "retr8"}.get(arm, arm), label, "C. Training-free: the base model with examples",
            exec_match(retr / arm / "exec_test.jsonl"), **retr_meta(arm))
    import gate as G                                                   # the similarity gate, offline
    g_arms = {a: G.load_rows(retr, a, "test") for a in G.K_ARMS.values()}
    g_scores = G.load_scores(retr)
    for k, tau in ((5, 15.5), (8, 15.1)):
        e = G.evaluate("on-off", tau, ids, g_scores, g_arms, k)
        add("gated-retr%d" % k, "base + %d retrieved, only when similar (τ %.1f from val)" % (k, tau),
            "C. Training-free: the base model with examples", e["match"],
            gens=1, sandbox="no", asks=0.0, extra="BM25 index", deploy="possible", gen_tokens=None,
            prompt_tokens=e["prompt_tokens_mean"], latency="—", note="retrieves on %.0f%% of rows" % e["gated_on_pct"])

    # ---- the served adapter plus inference-time work, no question
    add("adapter-retr5", "served adapter + 5 retrieved examples", "D. The served adapter, more inference, no question",
        exec_match(retr / "adapter-retr5" / "exec_test.jsonl"), **{**retr_meta("adapter-retr5"), "deploy": "possible"})
    probe = {r["id"]: r for r in load_jsonl(RUNS / "sample_rule_2026-10-09" / "probe_test.jsonl")}
    samp_tok = {i: 8 * probe[i]["tokens_mean"] for i in ids}
    mean_samp = sum(samp_tok.values()) / len(ids)
    examples = load_jsonl(RUNS / "reranker_2026-10-16" / "test_examples.jsonl")
    groups = defaultdict(list)
    for e in examples:
        groups[e["id"]].append(e)
    maj_text = {}
    for i in ids:
        sampled = [c for c in groups[i] if c.get("n", 0) > 0]
        top = max(sampled, key=lambda c: (c["n"], bool(c.get("greedy")), c["cand_mode"], c["cand_command"]))
        maj_text[i] = top["label"]
    eight = dict(gens=8, asks=0.0, gen_tokens=mean_samp, prompt_tokens=one["prompt_tokens"], latency="—")
    add("majority-text", "8 samples, most common command text", "D. The served adapter, more inference, no question",
        maj_text, sandbox="no", extra="—", deploy="possible", **eight, note="ties go to the greedy answer, then text")
    add("majority-result", "8 samples, most common result", "D. The served adapter, more inference, no question",
        majority_by_result(load_jsonl(RUNS / "sample_rule_2026-10-09" / "probe_test.jsonl"), byid),
        sandbox="picks", extra="—", deploy="✗ executor picks", **eight)
    for name, f in (("topp09-result", "samples_top_p0.9.jsonl"), ("minp01-result", "samples_min_p0.1.jsonl")):
        rows_ = load_jsonl(INPUTS / f)
        add(name, "8 samples at %s, most common result" % ("top-p 0.9" if "top" in f else "min-p 0.1"),
            "D. The served adapter, more inference, no question", majority_by_result(rows_, byid),
            sandbox="picks", extra="—", deploy="✗ executor picks", gens=8, asks=0.0,
            gen_tokens=sum(8 * r["tokens_mean"] for r in rows_) / len(rows_), prompt_tokens=one["prompt_tokens"],
            latency="—", note="exploratory, one seed (2026-10-18)")
    scores = {}
    for size in ("4b", "2b"):
        s = {(r["id"], r["cand_mode"], r["cand_command"]): r["score"]
             for r in load_jsonl(RUNS / "reranker_2026-10-16" / ("verifier-%s" % size) / "test_scores.jsonl")}
        lam = 0.0 if size == "4b" else 1.0
        pick, best = {}, {}
        for i in ids:
            sampled = [c for c in groups[i] if c.get("n", 0) > 0]
            def sc(c):
                return s[(i, c["cand_mode"], c["cand_command"])] + lam * math.log(max(c.get("n", 0), 1))
            top = max(sampled, key=lambda c: (sc(c), c.get("n", 0), bool(c.get("greedy")), c["cand_mode"], c["cand_command"]))
            pick[i], best[i] = top["label"], sc(top)
        scores[size] = (pick, best)
        add("verifier-%s" % size, "8 samples, %s yes/no verifier picks%s" % (size.upper(), "" if lam == 0 else " (λ = 1)"),
            "D. The served adapter, more inference, no question", pick, sandbox="no",
            extra="verifier adapter", deploy="possible", **{**eight, "gens": 8})

    # ---- the clarifying question (simulated user replies)
    verd = {r["id"]: r for r in load_jsonl(RUNS / "clarify_2026-10-07" / "test" / "verdicts_served.jsonl")}
    rule = json.loads((RUNS / "sample_rule_2026-10-09" / "test" / "report.json").read_text())
    asks = {i: rule[i]["top_share"] * 8 <= 5 for i in ids}
    dial_tok = {i: verd[i]["question_tokens"] + (verd[i]["real"].get("tokens") or 0) for i in ids}
    real = {i: int(bool(verd[i]["real"]["match"])) for i in ids}
    direct = {i: int(bool(verd[i]["direct"]["match"])) for i in ids}
    ask_rate = 100 * sum(asks.values()) / len(ids)
    qgroup = "E. The served adapter + a clarifying question (simulated user)"
    add("always-ask", "always ask, then answer", qgroup, real, gens=1, sandbox="no", asks=100.0, extra="user turn",
        deploy="possible", gen_tokens=sum(dial_tok.values()) / len(ids), prompt_tokens=None, latency="—")
    add("sample-rule", "ask when ≤ 5 of 8 sampled results agree (the deployed round trip)", qgroup,
        {i: real[i] if asks[i] else direct[i] for i in ids}, gens=9, sandbox="ask decision", asks=ask_rate,
        extra="user turn", deploy="served (MCP)", latency="0.54 s",
        gen_tokens=greedy_tok + mean_samp + sum(dial_tok[i] for i in ids if asks[i]) / len(ids), prompt_tokens=None)
    pick4, best4 = scores["4b"]
    add("rule+verifier-4b", "the sample rule; where it does not ask, the 4B verifier picks", qgroup,
        {i: real[i] if asks[i] else pick4[i] for i in ids}, gens=8, sandbox="ask decision", asks=ask_rate,
        extra="user turn + verifier", deploy="possible", latency="—",
        gen_tokens=mean_samp + sum(dial_tok[i] for i in ids if asks[i]) / len(ids), prompt_tokens=None)
    order = sorted(ids, key=lambda i: (best4[i], i))
    vask = set(order[:round(0.20 * len(ids))])
    add("verifier-ask-20", "ask on the 20% of rows the 4B verifier doubts most; else its pick", qgroup,
        {i: real[i] if i in vask else pick4[i] for i in ids}, gens=8, sandbox="no", asks=20.0,
        extra="user turn + verifier", deploy="possible", latency="—",
        gen_tokens=mean_samp + sum(dial_tok[i] for i in vask) / len(ids), prompt_tokens=None,
        note="the pre-registered 20% point of a curve")

    # ---- ceilings: need the answer key, not deployable
    cgroup = "F. Ceilings (use the answer key; not deployable)"
    add("oracle-ask", "ask only where the question fixes the answer", cgroup,
        {i: max(direct[i], real[i]) for i in ids}, gens=1, sandbox="—", asks=100 * sum(1 for i in ids if real[i] and not direct[i]) / len(ids),
        extra="answer key", deploy="✗", gen_tokens=None, prompt_tokens=None, latency="—")
    add("oracle8", "any of the 8 samples right", cgroup, {i: int(int(probe[i]["k"]) > 0) for i in ids},
        gens=8, sandbox="—", asks=0.0, extra="answer key", deploy="✗", gen_tokens=mean_samp,
        prompt_tokens=None, latency="—")
    forced = {r["id"]: r for r in load_jsonl(INPUTS / "forced_test.jsonl")}
    branch = {r["id"]: r for r in load_jsonl(INPUTS / "branch_test.jsonl")}
    add("oracle-modes", "either forced mode right (ex: or keys:)", cgroup,
        {i: int(forced[i]["ex_match"] or forced[i]["keys_match"]) for i in ids}, gens=2, sandbox="—", asks=0.0,
        extra="answer key", deploy="✗", gen_tokens=None, prompt_tokens=None, latency="—", note="exploratory (2026-10-18)")
    other = lambda i: forced[i]["keys_match"] if forced[i]["free_mode"] == "ex" else forced[i]["ex_match"]
    add("oracle-3struct", "greedy, its branch at the least likely token, or the other mode right", cgroup,
        {i: int(branch[i]["greedy_match"] or branch[i]["branch_match"] or other(i)) for i in ids}, gens=3, sandbox="—",
        asks=0.0, extra="answer key", deploy="✗", gen_tokens=None, prompt_tokens=None, latency="—",
        note="exploratory (2026-10-18)")
    return rows


def check_recorded(rows):
    bad = []
    for name, want in RECORDED.items():
        got = sum(rows[name]["outcome"].values())
        if got != want:
            bad.append("%s: rebuilt %d, recorded %d" % (name, got, want))
    if bad:
        raise SystemExit("rows that do not reproduce their recorded figure:\n  " + "\n  ".join(bad))


def summarize(rows, cases):
    ids = [c["id"] for c in cases]
    tier = {c["id"]: c.get("tier") or "T?" for c in cases}
    ref1, ref2 = rows["served"]["outcome"], rows["sample-rule"]["outcome"]
    out = []
    for name, r in rows.items():
        o = r["outcome"]
        k = sum(o.values())
        lo, hi = wilson(k, len(ids))
        g1, l1, p1 = paired(ref1, o, ids)
        g2, l2, p2 = paired(ref2, o, ids)
        tiers = {}
        for t in ("T1", "T2", "T3"):
            sel = [i for i in ids if tier[i] == t]
            tiers[t] = round(100 * sum(o[i] for i in sel) / len(sel), 1)
        out.append({k_: v for k_, v in r.items() if k_ != "outcome"} | {
            "exec_match": k, "pct": round(100 * k / len(ids), 1), "ci": [round(lo, 1), round(hi, 1)],
            "vs_served": {"gained": g1, "lost": l1, "delta": round(100 * (g1 - l1) / len(ids), 1), "p": p1},
            "vs_rule": {"gained": g2, "lost": l2, "delta": round(100 * (g2 - l2) / len(ids), 1), "p": p2},
            "tiers": tiers})
    return out


UNION_PAIRS = [("verifier-4b", "sample-rule"), ("verifier-4b", "always-ask"), ("served", "9b-gated"),
               ("served", "seed-sft"), ("served", "retr8"), ("served", "cliff-chain")]


def unions(rows, ids):
    """For pairs of approaches: both right, only the first, only the second, and
    the ceiling of a perfect chooser between them (either right)."""
    out = []
    for a, b in UNION_PAIRS:
        oa, ob = rows[a]["outcome"], rows[b]["outcome"]
        both = sum(1 for i in ids if oa[i] and ob[i])
        only_a = sum(1 for i in ids if oa[i] and not ob[i])
        only_b = sum(1 for i in ids if ob[i] and not oa[i])
        out.append({"a": a, "b": b, "both": both, "only_a": only_a, "only_b": only_b,
                    "either_pct": round(100 * (both + only_a + only_b) / len(ids), 1)})
    return out


def fmt_p(p):
    return "< 0.001" if p < 0.001 else ("%.3f" % p if p < 0.1 else "%.2f" % p)


def report_unions(un, labels):
    lines = ["", "## Do two approaches fix the same rows?", "",
             "A perfect chooser between two approaches would score \"either right\"; the two \"only\" columns are the rows "
             "one fixes and the other misses.", "",
             "| first | second | both right | only the first | only the second | either right |", "|---|---|---|---|---|---|"]
    for u in un:
        lines.append("| %s | %s | %d | %d | %d | %.1f%% |" % (labels[u["a"]], labels[u["b"]], u["both"], u["only_a"],
                                                             u["only_b"], u["either_pct"]))
    return lines


def report(summary, n):
    lines = ["# Every approach on test v2, side by side (2026-10-18)", "",
             "Built by `scripts/comparison/compare_test_v2.py` from recorded runs only (no new generation); each row's "
             "per-case outcome is rebuilt and checked against the figure its own report recorded. %d rows. Paired exact "
             "McNemar against the two references: **the served adapter** (one greedy answer, 54.3%%) and **the sample rule** "
             "(the deployed round trip that asks when ≤ 5 of 8 sampled results agree, 59.7%%). Rows marked ✗ cannot be "
             "deployed (the executor would pick the answer, or they need the answer key)." % n, "",
             "| group | approach | exec-match (95% CI) | vs served: +/−, Δ, p | vs sample rule: +/−, Δ, p | T1 / T2 / T3 | asks | samples / request | generated tokens / request | sandbox at inference | deployable |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        v1, v2 = s["vs_served"], s["vs_rule"]
        lines.append("| %s | %s%s | **%.1f%%** (%.1f–%.1f) | +%d / −%d, %+.1f, %s | +%d / −%d, %+.1f, %s | %.1f / %.1f / %.1f | %s | %s | %s | %s | %s |" % (
            s["group"].split(".")[0], s["label"], (" — " + s["note"]) if s.get("note") else "",
            s["pct"], s["ci"][0], s["ci"][1],
            v1["gained"], v1["lost"], v1["delta"], fmt_p(v1["p"]), v2["gained"], v2["lost"], v2["delta"], fmt_p(v2["p"]),
            s["tiers"]["T1"], s["tiers"]["T2"], s["tiers"]["T3"],
            ("%.0f%%" % s["asks"]) if s["asks"] else "—", s["gens"],
            ("%.0f" % s["gen_tokens"]) if s.get("gen_tokens") else "—", s["sandbox"], s["deploy"]))
    lines += ["", "Groups: " + "; ".join(sorted({s["group"] for s in summary})) + ".", "",
              "Generated tokens per request count every sampled answer, plus the question and the answer after the reply "
              "where the row asks; verifier rows also run 8 scoring passes (prefill only). Latency is measured only for "
              "the two served paths (vLLM, one request at a time, val): 0.22 s for the greedy answer, 0.54 s for the round trip."]
    return "\n".join(lines)


SHORT = {"base-k0": "4B base, zero-shot", "sft-r1": "SFT round 1", "sft-grpo": "SFT + GRPO",
         "served": "Served adapter (gated GRPO)", "cliff-chain": "Cliff SFT → gated GRPO",
         "sft-ds100": "SFT re-run, 100% of train", "sft-plusglobal": "SFT + 926 global prompts",
         "9b-gated": "9B, gated GRPO", "seed-sft": "Seed-Coder-8B, SFT", "coder-sft": "Qwen2.5-Coder-7B, SFT",
         "granite-sft": "Granite-4.2-8B, SFT", "q3-4b-sft": "Qwen3-4B-2507, SFT", "2b-sft": "Qwen3.5-2B, SFT",
         "0.8b-sft": "Qwen3.5-0.8B, SFT", "base-random5": "Base + 5 random examples",
         "base-retr1": "Base + 1 retrieved", "base-retr3": "Base + 3 retrieved", "retr5": "Base + 5 retrieved",
         "retr8": "Base + 8 retrieved", "gated-retr5": "Base + 5 retrieved, gated", "gated-retr8": "Base + 8 retrieved, gated",
         "adapter-retr5": "Adapter + 5 retrieved", "majority-text": "Most common of 8, by text",
         "majority-result": "Most common of 8, by result", "topp09-result": "Most common result, top-p 0.9",
         "minp01-result": "Most common result, min-p 0.1", "verifier-4b": "4B verifier picks of 8",
         "verifier-2b": "2B verifier picks of 8", "always-ask": "Always ask",
         "sample-rule": "Sample rule (deployed round trip)", "rule+verifier-4b": "Sample rule + verifier",
         "verifier-ask-20": "Verifier-gated ask", "oracle-ask": "Ask only where it helps",
         "oracle8": "Any of 8 samples right", "oracle-modes": "Either forced mode right",
         "oracle-3struct": "Greedy, branch or other mode right"}
FAMILY = {"A": "The 4B, trained", "B": "Other models, same recipe", "C": "Training-free: base model + examples",
          "D": "Served adapter + more inference", "E": "Served adapter + a clarifying question",
          "F": "Ceilings (need the answer key)"}


def page_rows(summary, un):
    """Flat rows for the comparison page: one per approach, and one per pair."""
    def status(s):
        if s["deploy"].startswith("served"):
            return "in use"
        if s["deploy"] == "✗ executor picks":
            return "executor picks"
        if s["deploy"] == "✗":
            return "ceiling"
        return "possible"
    rows = []
    for order, s in enumerate(summary):
        g = s["group"].split(".")[0]
        rows.append({"name": s["name"], "order": order, "short": SHORT[s["name"]], "label": s["label"],
                     "family_key": g, "family": FAMILY[g], "pct": s["pct"], "ci_lo": s["ci"][0], "ci_hi": s["ci"][1],
                     "exec_match": s["exec_match"], "n": 429,
                     "d_served": s["vs_served"]["delta"], "g_served": s["vs_served"]["gained"],
                     "l_served": s["vs_served"]["lost"], "p_served": s["vs_served"]["p"],
                     "d_rule": s["vs_rule"]["delta"], "g_rule": s["vs_rule"]["gained"], "l_rule": s["vs_rule"]["lost"],
                     "p_rule": s["vs_rule"]["p"],
                     # display text, so a page can show exactly what the data holds
                     "d_served_txt": "%+.1f" % s["vs_served"]["delta"] if s["vs_served"]["delta"] else "0.0",
                     "d_rule_txt": "%+.1f" % s["vs_rule"]["delta"] if s["vs_rule"]["delta"] else "0.0",
                     "p_served_txt": fmt_p(s["vs_served"]["p"]), "p_rule_txt": fmt_p(s["vs_rule"]["p"]),
                     "t1": s["tiers"]["T1"], "t2": s["tiers"]["T2"], "t3": s["tiers"]["T3"],
                     "asks_pct": round(s["asks"], 1), "samples": s["gens"],
                     "gen_tokens": round(s["gen_tokens"], 1) if s.get("gen_tokens") else None,
                     "sandbox": s["sandbox"], "status": status(s), "note": s.get("note") or ""})
    pairs = [{"pair": "%s|%s" % (u["a"], u["b"]), "first": SHORT[u["a"]], "second": SHORT[u["b"]], "both": u["both"],
              "only_first": u["only_a"], "only_second": u["only_b"], "either_pct": u["either_pct"]} for u in un]
    return rows, pairs


def main():
    cases = load_jsonl(REPO / "data" / "splits" / "test.jsonl")
    rows = build(cases)
    check_recorded(rows)
    summary = summarize(rows, cases)
    un = unions(rows, [c["id"] for c in cases])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data.json").write_text(json.dumps({"n": len(cases), "rows": summary, "unions": un}, indent=1,
                                              ensure_ascii=False) + "\n", encoding="utf-8")
    prow, ppair = page_rows(summary, un)
    (OUT / "page_approaches.json").write_text(json.dumps(prow, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "page_pairs.json").write_text(json.dumps(ppair, ensure_ascii=False) + "\n", encoding="utf-8")
    text = report(summary, len(cases)) + "\n".join(report_unions(un, {r["name"]: r["label"] for r in summary}))
    (OUT / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
