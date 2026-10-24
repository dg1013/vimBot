#!/usr/bin/env python3
"""Error analysis of the best runs on test v2 (CPU; recorded artifacts, plus a few
hundred sandbox re-executions for the buffer taxonomy and the one-edit probe).

The systems (test v2, 429 rows; every figure is rebuilt and checked against the
number its own report recorded before anything else runs):

  4B greedy    the served adapter (gated GRPO, step 750), one greedy answer, fla kernels   54.3%
  9B greedy    the 9B GRPO adapter (gated round kept step 0), NF4, one greedy answer      56.4%
  4B pick      the 4B yes/no verifier (V4) over the served adapter's 8 samples            59.9%
  9B pick      V4 over the 9B's 8 samples                                                  61.8%
  pooled pick  V4 over both models' 8 + 8 samples -- the best measured system              65.3%
  gated ask    ask on the 20% the verifier doubts most, else the 4B pick (best with asking) 61.3%
  ceilings     any of the 16; pass@128 per model and for either (the 47-row uncovered core)

Writes <out>/report.md (tables), examples.md (a gallery of failures), rows.jsonl
(one joined record per test row) and summary.json (every table as data).

    python scripts/error_analysis/analyze.py                  # full: ~1,500 sandbox runs, about a minute
    python scripts/error_analysis/analyze.py --no-exec        # tables that need no re-execution
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import difflib
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"
for p in (REPO / "src", REPO / "scripts" / "methods_9b", HERE):
    sys.path.insert(0, str(p))
import taxonomy as T                       # noqa: E402
import compare_9b as C9                    # noqa: E402  the recorded pick / pooled rules, one definition

CASES = REPO / "data" / "splits" / "test.jsonl"
RV = RUNS / "remeasure_2026-09-25" / "v2"
GREEDY = {"4b": RV / "qwen3.5-4b-grpo-gated__fla-kernels" / "exec_test.jsonl",
          "9b": RV / "qwen3.5-9b-grpo-gated" / "exec_test.jsonl"}
STAGES = {"4b": [("SFT round 1", RV / "qwen3.5-4b-sft" / "exec_test.jsonl"),
                 ("+ GRPO", RV / "qwen3.5-4b-grpo" / "exec_test.jsonl"),
                 ("+ gated GRPO = the served adapter (pure-torch path)", RV / "qwen3.5-4b-grpo-gated" / "exec_test.jsonl"),
                 ("the served adapter under the fla kernels (this report's 4B greedy)", GREEDY["4b"]),
                 ("served → cliff SFT → gated GRPO (stage C)", RV / "qwen3.5-4b-grpo-gated-cliffleg" / "exec_test.jsonl")],
          "9b": [("SFT round 1 (NF4)", RV / "qwen3.5-9b-sft" / "exec_test.jsonl"),
                 ("+ GRPO (NF4)", RV / "qwen3.5-9b-grpo" / "exec_test.jsonl"),
                 ("gated round, step 0, fla (this report's 9B greedy)", GREEDY["9b"])]}
RERANK = RUNS / "reranker_2026-10-16"
METH = RUNS / "qwen9b_methods_2026-10-19"
CEIL = METH / "ceiling"
EX = {"4b": RERANK / "test_examples.jsonl", "9b": METH / "examples_test.jsonl"}
SCORES = {"4b": RERANK / "verifier-4b" / "test_scores.jsonl", "9b": METH / "v4_scores_test.jsonl"}
SAMP = {"4b": CEIL / "samples_served4b.jsonl", "9b": CEIL / "samples_served9b.jsonl"}
S128 = RUNS / "serve_pooled" / "scores_128.jsonl"
VERD4 = RUNS / "clarify_2026-10-07" / "test" / "verdicts_served.jsonl"
GATE_RATE = 0.20

# What each system's own report recorded, as exec-match rows of 429.
RECORDED = {"4B greedy": 233, "9B greedy": 242, "4B pick": 257, "9B pick": 265, "pooled pick": 280,
            "gated ask": 263, "any of 16": 343, "pass@128 4B": 335, "pass@128 9B": 359, "pass@128 either": 382}
MODEL_LABEL = {"4b": "4B greedy", "9b": "9B greedy"}
SYSTEMS = ["4B greedy", "9B greedy", "4B pick", "9B pick", "pooled pick", "gated ask"]
MIN_CAT = 13                                 # categories with fewer rows are listed, not read


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def pct(k, n):
    return 100.0 * k / n if n else float("nan")


def fmt(k, n):
    return "%.1f%%" % pct(k, n) if n else "—"


def greedy_state(row):
    if row.get("command") is None:
        return "no prediction"
    if row.get("blocked"):
        return "blocked"
    if row.get("timeout"):
        return "timeout"
    if row["match"]:
        return "match"
    if row["valid"]:
        return "valid_wrong"
    return "invalid"


def rank_key(scores, c):
    return (scores[(c["id"], c["cand_mode"], c["cand_command"])], c.get("n", 0), bool(c.get("greedy")),
            c["cand_mode"], c["cand_command"])


def ranked(cands, scores):
    """The verifier's order over sampled candidates: the reranker's tie rule (compare_9b.pick)."""
    sampled = [c for c in cands if c.get("n", 0) > 0]
    return sorted(sampled, key=lambda c: rank_key(scores, c), reverse=True)


def auroc(pairs):
    """Rank AUROC over (score, label) pairs, ties at average rank; None if one class is missing."""
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return None
    allv = sorted(pairs, key=lambda x: x[0])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rp = sum(ranks[k] for k, (s, l) in enumerate(allv) if l)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def code(s):
    s = "" if s is None else str(s)
    return "`" + s.replace("`", "ˋ").replace("|", "\\|") + "`"


def ex_mode(mode, cmd):
    return code("%s: %s" % (mode, cmd))


class Sandbox:
    """Re-executions, cached; disabled by --no-exec (then every call says None)."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.cache = {}
        self.calls = 0
        if enabled:
            from executor import run_vim
            self.run_vim = run_vim

    def run(self, case, command, mode):
        if not self.enabled or command is None:
            return None
        k = (case["id"], mode, command)
        if k not in self.cache:
            self.calls += 1
            r = self.run_vim(case["buffer_before"], command, mode=mode, cursor=tuple(case.get("cursor") or (1, 1)))
            self.cache[k] = r
        return self.cache[k]

    def classify(self, case, command, mode):
        r = self.run(case, command, mode)
        if r is None:
            return None
        if r["buffer_after"] is None:
            return {"kind": "no_buffer", "within": None}
        if mode == "ex" and not r["was_valid"]:
            return {"kind": "invalid", "within": None}
        return T.classify(case["buffer_before"], case["buffer_after"], r["buffer_after"])

    def kind(self, case, command, mode):
        c = self.classify(case, command, mode)
        return None if c is None else c["kind"]

    def matches(self, case, command, mode):
        r = self.run(case, command, mode)
        if r is None or r["buffer_after"] is None:
            return False
        return T.norm(r["buffer_after"]) == T.norm(case["buffer_after"])


# ---------------------------------------------------------------- loading

def build(sandbox):
    cases = load_jsonl(CASES)
    ids = [c["id"] for c in cases]
    byid = {c["id"]: c for c in cases}
    R = {i: {"id": i, "category": byid[i]["category"], "tier": byid[i].get("tier") or "T?", "gold_mode": byid[i]["mode"],
             "features": T.features(byid[i]), "sys": {}} for i in ids}

    # -- greedy answers, with their buffer taxonomy (buffers are in the exec logs: no re-execution)
    for m in ("4b", "9b"):
        log = {r["id"]: r for r in load_jsonl(GREEDY[m])}
        for i in ids:
            r = log[i]
            st = greedy_state(r)
            cl = T.classify(byid[i]["buffer_before"], byid[i]["buffer_after"], r.get("buffer_after")) if st in ("valid_wrong", "match") \
                else {"kind": "no_buffer" if r.get("buffer_after") is None else
                      T.classify(byid[i]["buffer_before"], byid[i]["buffer_after"], r.get("buffer_after"))["kind"],
                      "diff_lines": None, "similarity": None, "d_lines": None}
            R[i]["g" + m] = {"command": r.get("command"), "mode": r.get("mode"), "state": st, "error": r.get("error") or "",
                             "kind": cl["kind"], "within": cl.get("within"), "diff_lines": cl.get("diff_lines"),
                             "similarity": cl.get("similarity"), "d_lines": cl.get("d_lines"), "match": int(bool(r["match"])),
                             "buffer_after": r.get("buffer_after"), "stderr": (r.get("stderr") or "")[:200]}
            R[i]["sys"][MODEL_LABEL[m]] = int(bool(r["match"]))

    # -- the 8-sample candidates and the verifier's order (4B, 9B, pooled)
    ex = {m: C9.group(load_jsonl(EX[m])) for m in ("4b", "9b")}
    sc = {m: C9.load_scores(SCORES[m]) for m in ("4b", "9b")}
    best4 = {}
    for i in ids:
        for m in ("4b", "9b"):
            order = ranked(ex[m].get(i, []), sc[m])
            top = order[0]
            right = [c for c in order if c["label"]]
            R[i]["pick" + m] = {"label": top["label"], "command": top["cand_command"], "mode": top["cand_mode"], "state": top["state"],
                                "n": top["n"], "score": sc[m][(i, top["cand_mode"], top["cand_command"])],
                                "n_cands": len(order), "n_right": len(right),
                                "rank_right": (order.index(right[0]) + 1) if right else None}
            R[i]["sys"][m.upper() + " pick"] = top["label"]
            if m == "4b":
                best4[i] = R[i]["pick4b"]["score"]
        pool = C9.pooled(ex["4b"].get(i, []), ex["9b"].get(i, []))
        ps = C9.pooled_scores(pool, sc["4b"], sc["9b"], i)
        order = ranked(pool, ps)
        top = order[0]
        right = [c for c in order if c["label"]]
        maj = max(order, key=lambda c: (c["n"], bool(c.get("greedy")), c["cand_mode"], c["cand_command"]))
        n4 = sum(c["n"] for c in ex["4b"].get(i, []) if (c["cand_mode"], c["cand_command"]) == (top["cand_mode"], top["cand_command"]))
        n9 = sum(c["n"] for c in ex["9b"].get(i, []) if (c["cand_mode"], c["cand_command"]) == (top["cand_mode"], top["cand_command"]))
        br = right[0] if right else None
        R[i]["pooled"] = {
            "label": top["label"], "command": top["cand_command"], "mode": top["cand_mode"], "state": top["state"],
            "n": top["n"], "n4": n4, "n9": n9, "sources": "+".join(sorted(top["sources"])),
            "score": ps[(i, top["cand_mode"], top["cand_command"])],
            "n_cands": len(order), "n_right": len(right), "any16": int(bool(right)),
            "rank_right": (order.index(br) + 1) if br else None,
            "margin": (ps[(i, top["cand_mode"], top["cand_command"])] - ps[(i, br["cand_mode"], br["cand_command"])]) if (br and not top["label"]) else None,
            "best_right": None if not br else {"command": br["cand_command"], "mode": br["cand_mode"], "n": br["n"],
                                              "sources": "+".join(sorted(br["sources"])),
                                              "score": ps[(i, br["cand_mode"], br["cand_command"])]},
            "majority_label": maj["label"], "majority_is_pick": maj is top,
            "kind": None, "within": None,
        }
        R[i]["sys"]["pooled pick"] = top["label"]
        if not top["label"] and sandbox.enabled:
            cl = sandbox.classify(byid[i], top["cand_command"], top["cand_mode"])
            R[i]["pooled"]["kind"], R[i]["pooled"]["within"] = cl["kind"], cl.get("within")
            rr = sandbox.run(byid[i], top["cand_command"], top["cand_mode"])
            R[i]["pooled"]["buffer_after"] = rr.get("buffer_after")
            R[i]["pooled"]["stderr"] = (rr.get("stderr") or "")[:200]

    # -- exploratory: the same pick when both greedy answers join the pool (they carry verifier scores; n = 0 when unsampled)
    for i in ids:
        pool = {}
        for m in ("4b", "9b"):
            for c in ex[m].get(i, []):
                k = (c["cand_mode"], c["cand_command"])
                slot = pool.setdefault(k, dict(c, n=0, greedy=False))
                slot["n"] += c.get("n", 0)
                slot["greedy"] = slot["greedy"] or bool(c.get("greedy"))
        cands = list(pool.values())
        ps = C9.pooled_scores(cands, sc["4b"], sc["9b"], i)
        top = max(cands, key=lambda c: rank_key(ps, c))
        R[i]["pooled_plus_greedy"] = {"label": top["label"], "command": top["cand_command"], "mode": top["cand_mode"], "n": top["n"]}

    # -- the verifier-gated ask (asks on the 20% with the lowest 4B best score)
    verd = {r["id"]: r for r in load_jsonl(VERD4)}
    order = sorted(ids, key=lambda i: (best4[i], i))
    gated = set(order[:round(GATE_RATE * len(ids))])
    for i in ids:
        real = int(bool(verd[i]["real"]["match"]))
        R[i]["ask20"] = {"asked": int(i in gated), "label": real if i in gated else R[i]["pick4b"]["label"],
                         "after_reply": real}
        R[i]["sys"]["gated ask"] = R[i]["ask20"]["label"]

    # -- 128 samples per model, and the verifier over all of them
    samp = {m: {r["id"]: r for r in load_jsonl(SAMP[m])} for m in ("4b", "9b")}
    s128 = {(r["id"], r["mode"], r["command"]): r["score"] for r in load_jsonl(S128)}
    missing_scores = 0
    for i in ids:
        c = byid[i]
        gold_h, before_h = T.buffer_hash(c["buffer_after"]), T.buffer_hash(c["buffer_before"])
        info = {}
        union = {}
        for m in ("4b", "9b"):
            r = samp[m][i]
            cmds = [e for e in r["commands"] if e.get("mode") and e.get("command") is not None]
            top = max(cmds, key=lambda e: (e["n"], e["command"]))
            info[m] = {"c": r["c"], "n": r["n"], "distinct": r["distinct"], "distinct_buffers": r["distinct_buffers"],
                       "top_command": top["command"], "top_mode": top["mode"], "top_share": top["n"] / r["n"],
                       "top_right": int(top["state"] == "match"), "top_state": top["state"],
                       "noop_share": sum(e["n"] for e in cmds if e.get("buf") == before_h) / r["n"],
                       "invalid_share": sum(e["n"] for e in cmds if e["state"] == "invalid") / r["n"],
                       "gold_mode_share": sum(e["n"] for e in cmds if e["mode"] == c["mode"]) / r["n"]}
            for e in cmds:
                k = (e["mode"], e["command"])
                slot = union.setdefault(k, {"n4": 0, "n9": 0, "label": int(e["state"] == "match"), "state": e["state"], "buf": e.get("buf")})
                slot["n" + m[0]] += e["n"]
        scored = []
        for k, v in union.items():
            s = s128.get((i,) + k)
            if s is None:
                missing_scores += 1
                continue
            scored.append((s, v["n4"] + v["n9"], k, v))
        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        right_idx = [j for j, x in enumerate(scored) if x[3]["label"]]
        top = scored[0] if scored else None
        info["pick128"] = {"label": top[3]["label"] if top else 0, "command": top[2][1] if top else None,
                           "mode": top[2][0] if top else None, "state": top[3]["state"] if top else None,
                           "n": top[1] if top else 0, "score": top[0] if top else None,
                           "n_cands": len(scored), "n_right": len(right_idx),
                           "rank_right": (right_idx[0] + 1) if right_idx else None,
                           "margin": (top[0] - scored[right_idx[0]][0]) if (right_idx and top and not top[3]["label"]) else None}
        info["either"] = int(info["4b"]["c"] > 0 or info["9b"]["c"] > 0)
        info["shared_top"] = int(info["4b"]["top_command"] == info["9b"]["top_command"] and info["4b"]["top_mode"] == info["9b"]["top_mode"])
        info["gold_hash"] = gold_h
        R[i]["ceil"] = info
        R[i]["sys"]["pass@128 4B"] = int(info["4b"]["c"] > 0)
        R[i]["sys"]["pass@128 9B"] = int(info["9b"]["c"] > 0)
        R[i]["sys"]["pass@128 either"] = info["either"]
        R[i]["sys"]["any of 16"] = R[i]["pooled"]["any16"]

    # -- the ladder
    for i in ids:
        s = R[i]["sys"]
        g = s["4B greedy"] + s["9B greedy"]
        if s["pooled pick"]:
            lvl = "P1 pooled pick right; both greedies right" if g == 2 else \
                  "P2 pooled pick right; one greedy right" if g == 1 else \
                  "P3 pooled pick right; neither greedy right (selection gain)"
        else:
            lvl = "W1 pooled pick wrong; a greedy answer was right (selection loss)" if g else \
                  "W2 pooled pick wrong; a right answer was among the 16 (verifier miss)" if s["any of 16"] else \
                  "W3 none of the 16 right; some of the 256 right (coverage miss at 8 + 8)" if s["pass@128 either"] else \
                  "W4 never right in 256 samples"
        R[i]["ladder"] = lvl

    # -- training stages (greedy exec logs on test v2)
    stages = {m: [(lab, {r["id"]: int(bool(r["match"])) for r in load_jsonl(p)}) for lab, p in STAGES[m]] for m in STAGES}

    # -- one mechanical edit away: the two greedies' wrong answers and the pooled pick's
    probes = {}
    if sandbox.enabled:
        for name, getter in (("4B greedy", lambda i: R[i]["g4b"]), ("9B greedy", lambda i: R[i]["g9b"]),
                             ("pooled pick", lambda i: R[i]["pooled"])):
            rule_tried, rule_fixed, fixed_rows, tried_rows = collections.Counter(), collections.Counter(), set(), set()
            fixed_kinds = collections.Counter()
            for i in ids:
                a = getter(i)
                if a.get("label", a.get("match")) or a.get("state") in ("blocked", "timeout", "no prediction") or not a.get("command"):
                    continue
                muts = T.mutations(a["command"], a["mode"])
                if not muts:
                    continue
                tried_rows.add(i)
                hit = []
                for rule, cmd in muts:
                    rule_tried[rule] += 1
                    if sandbox.matches(byid[i], cmd, a["mode"]):
                        rule_fixed[rule] += 1
                        hit.append(rule)
                if hit:
                    fixed_rows.add(i)
                    fixed_kinds[a.get("kind") or "?"] += 1
                a["mutation_fix"] = hit
            probes[name] = {"tried_rows": len(tried_rows), "fixed_rows": len(fixed_rows), "rules": {r: (rule_tried[r], rule_fixed[r]) for r in rule_tried},
                            "fixed_kinds": dict(fixed_kinds), "fixed_ids": sorted(fixed_rows)}

    # -- the uncovered core and the stable wrong answers: what kind of wrong the favourite answer is
    if sandbox.enabled:
        for i in ids:
            for m in ("4b", "9b"):
                t = R[i]["ceil"][m]
                if not t["top_right"]:
                    cl = sandbox.classify(byid[i], t["top_command"], t["top_mode"])
                    t["top_kind"], t["top_within"] = cl["kind"], cl.get("within")

    checks = {"4B greedy": sum(R[i]["sys"]["4B greedy"] for i in ids), "9B greedy": sum(R[i]["sys"]["9B greedy"] for i in ids),
              "4B pick": sum(R[i]["sys"]["4B pick"] for i in ids), "9B pick": sum(R[i]["sys"]["9B pick"] for i in ids),
              "pooled pick": sum(R[i]["sys"]["pooled pick"] for i in ids), "gated ask": sum(R[i]["sys"]["gated ask"] for i in ids),
              "any of 16": sum(R[i]["sys"]["any of 16"] for i in ids),
              "pass@128 4B": sum(R[i]["sys"]["pass@128 4B"] for i in ids), "pass@128 9B": sum(R[i]["sys"]["pass@128 9B"] for i in ids),
              "pass@128 either": sum(R[i]["sys"]["pass@128 either"] for i in ids)}
    bad = ["%s: rebuilt %d, recorded %d" % (k, checks[k], RECORDED[k]) for k in RECORDED if checks[k] != RECORDED[k]]
    if bad:
        raise SystemExit("systems that do not reproduce their recorded figure:\n  " + "\n  ".join(bad))
    return {"ids": ids, "cases": byid, "rows": R, "stages": stages, "probes": probes, "checks": checks,
            "missing_scores": missing_scores, "sandbox_calls": sandbox.calls}


# ------------------------------------------------------------- the report

def by(rows, ids, keyf):
    g = collections.defaultdict(list)
    for i in ids:
        g[keyf(rows[i])].append(i)
    return g


def sys_cols(rows, sel, systems):
    return [fmt(sum(rows[i]["sys"][s] for i in sel), len(sel)) for s in systems]


def report(D, out, exec_on):
    ids, rows, cases = D["ids"], D["rows"], D["cases"]
    n = len(ids)
    S = {}
    L = ["# Error analysis of the best runs (test v2, %d rows; %s)" % (n, dt.date.today().isoformat()), "",
         "Built by `scripts/error_analysis/analyze.py` from recorded runs; the only new executions are the sandbox "
         "re-runs for the buffer taxonomy of picked answers and the one-edit probe (%s). Every system is rebuilt from its "
         "own artifacts and checked against the figure its report recorded before anything is counted."
         % ("%d sandbox calls" % D["sandbox_calls"] if exec_on else "skipped: --no-exec"), ""]
    L += table(["system", "exec-match", "rows", "source"], [
        ("4B greedy — the served adapter, one greedy answer (fla kernels)", fmt(D["checks"]["4B greedy"], n), D["checks"]["4B greedy"], code("runs/remeasure_2026-09-25/v2/qwen3.5-4b-grpo-gated__fla-kernels/exec_test.jsonl")),
        ("9B greedy — the 9B GRPO adapter (gated round, step 0), NF4", fmt(D["checks"]["9B greedy"], n), D["checks"]["9B greedy"], code("runs/remeasure_2026-09-25/v2/qwen3.5-9b-grpo-gated/exec_test.jsonl")),
        ("4B pick — the 4B verifier over the served adapter's 8 samples", fmt(D["checks"]["4B pick"], n), D["checks"]["4B pick"], code("runs/reranker_2026-10-16/")),
        ("9B pick — the same verifier over the 9B's 8 samples", fmt(D["checks"]["9B pick"], n), D["checks"]["9B pick"], code("runs/qwen9b_methods_2026-10-19/")),
        ("**pooled pick — the verifier over both models' 8 + 8 samples (best measured)**", "**%s**" % fmt(D["checks"]["pooled pick"], n), D["checks"]["pooled pick"], "same"),
        ("gated ask — ask on the 20% the verifier doubts most, else the 4B pick", fmt(D["checks"]["gated ask"], n), D["checks"]["gated ask"], code("runs/clarify_2026-10-07/test/")),
        ("any of the 16 samples right (ceiling of the pooled pick)", fmt(D["checks"]["any of 16"], n), D["checks"]["any of 16"], ""),
        ("pass@128, 4B / 9B / either", "%s / %s / %s" % (fmt(D["checks"]["pass@128 4B"], n), fmt(D["checks"]["pass@128 9B"], n), fmt(D["checks"]["pass@128 either"], n)),
         "%d / %d / %d" % (D["checks"]["pass@128 4B"], D["checks"]["pass@128 9B"], D["checks"]["pass@128 either"]), code("runs/qwen9b_methods_2026-10-19/ceiling/"))])
    L.append("")

    # ---- 1. the ladder
    L += ["## 1. Where the 429 rows end up", "",
          "Each row is placed once, from the pooled pick's point of view: did the best system get it, and if not, what was the nearest "
          "thing that would have. `any of the 16` = some sampled candidate in the 8 + 8 was right; `256` = the two models' 128 samples each.", ""]
    lad = by(rows, ids, lambda r: r["ladder"])
    lad_rows = []
    for lvl in sorted(lad):
        sel = lad[lvl]
        tiers = collections.Counter(rows[i]["tier"] for i in sel)
        modes = collections.Counter(rows[i]["gold_mode"] for i in sel)
        cats = collections.Counter(rows[i]["category"] for i in sel).most_common(3)
        lad_rows.append((lvl, len(sel), fmt(len(sel), n), "%d / %d / %d" % (tiers["T1"], tiers["T2"], tiers["T3"]),
                         "%d / %d" % (modes["ex"], modes["keys"]), ", ".join("%s %d" % c for c in cats)))
    L += table(["level", "rows", "share", "T1 / T2 / T3", "ex / keys", "largest categories"], lad_rows)
    S["ladder"] = {lvl: len(sel) for lvl, sel in lad.items()}
    wrong = [i for i in ids if not rows[i]["sys"]["pooled pick"]]
    L += ["", "The pooled pick's %d misses split into: %d where one of the two greedy answers was right and the verifier chose something else, "
          "%d where a right answer sat among the 16 samples and the verifier did not pick it, %d where none of the 16 was right but a "
          "right answer appears within 128 samples of one model, and %d that no sample of either model ever solves." % (
              len(wrong), len(lad.get([k for k in lad if k.startswith("W1")][0], [])) if any(k.startswith("W1") for k in lad) else 0,
              sum(len(v) for k, v in lad.items() if k.startswith("W2")), sum(len(v) for k, v in lad.items() if k.startswith("W3")),
              sum(len(v) for k, v in lad.items() if k.startswith("W4"))), ""]

    w1 = [i for i in ids if rows[i]["ladder"].startswith("W1")]
    w1a = [i for i in w1 if rows[i]["sys"]["any of 16"]]
    pg = {i: rows[i]["pooled_plus_greedy"]["label"] for i in ids}
    pg_g = [i for i in ids if pg[i] and not rows[i]["sys"]["pooled pick"]]
    pg_l = [i for i in ids if rows[i]["sys"]["pooled pick"] and not pg[i]]
    L += ["Of the %d selection losses, %d had a right answer among the 16 samples (a verifier miss in all but name) and %d did not: the right greedy "
          "answer was not among the 16 temperature-1 samples at all. Exploratory, read on test and not pre-registered: adding both greedy answers to "
          "the pool as candidates (they carry verifier scores; unsampled ones count 0) gives %s (+%d / −%d against the pooled pick), and solves %d of "
          "those %d." % (len(w1), len(w1a), len(w1) - len(w1a), fmt(sum(pg.values()), n), len(pg_g), len(pg_l),
                         sum(pg[i] for i in w1 if i not in w1a), len(w1) - len(w1a)), ""]
    caught = [i for i in ids if rows[i]["sys"]["any of 16"] and not rows[i]["sys"]["pass@128 either"]]
    if caught:
        L += ["The ceiling run's 47 uncovered rows and the W4 level differ by %d: %s has a right answer among the fresh 16 samples of the 8 + 8 run "
              "(a different draw than the 128), so by all evidence %d rows are unsolved." % (len(caught), ", ".join(code(i) for i in caught), 47 - len(caught)), ""]
    S["w1_split"] = {"w1": len(w1), "right_among_16": len(w1a)}
    S["pooled_plus_greedy"] = {"right": sum(pg.values()), "gained": len(pg_g), "lost": len(pg_l), "w1b_fixed": sum(pg[i] for i in w1 if i not in w1a)}
    S["caught_never"] = caught

    # by tier and category: pooled misses by level
    L += ["### 1a. The pooled pick's misses by tier and category", "",
          "Rows lost = rows the pooled pick gets wrong; the four columns after it are the W1–W4 levels above. Sorted by rows lost; "
          "categories under %d rows are small." % MIN_CAT, ""]
    def miss_row(label, sel):
        w = [i for i in sel if not rows[i]["sys"]["pooled pick"]]
        c = collections.Counter(rows[i]["ladder"][:2] for i in w)
        return (label, len(sel), fmt(len(sel) - len(w), len(sel)), len(w), c["W1"], c["W2"], c["W3"], c["W4"],
                fmt(sum(rows[i]["sys"]["pass@128 either"] for i in sel), len(sel)))
    hdr = ["slice", "rows", "pooled pick", "rows lost", "W1 selection loss", "W2 verifier miss", "W3 coverage", "W4 never", "pass@128 either"]
    tier_rows = [miss_row(t, sel) for t, sel in sorted(by(rows, ids, lambda r: r["tier"]).items())]
    cat_rows = sorted((miss_row(c, sel) for c, sel in by(rows, ids, lambda r: r["category"]).items()), key=lambda x: -x[3])
    L += table(hdr, tier_rows + [("**all**", n, fmt(n - len(wrong), n), len(wrong), *[sum(r[k] for r in tier_rows) for k in (4, 5, 6, 7)], fmt(D["checks"]["pass@128 either"], n))])
    L.append("")
    L += table(hdr, cat_rows)
    S["misses_by_tier"] = tier_rows
    S["misses_by_category"] = cat_rows
    L.append("")
    ct = collections.defaultdict(list)
    for i in ids:
        ct[(rows[i]["category"], rows[i]["tier"])].append(i)
    cells = sorted(((c, t, sel) for (c, t), sel in ct.items()), key=lambda x: -sum(1 for i in x[2] if not rows[i]["sys"]["pooled pick"]))[:12]
    L += ["The twelve category × tier cells with the most rows lost (they hold %d of the %d misses):" % (
        sum(1 for c, t, sel in cells for i in sel if not rows[i]["sys"]["pooled pick"]), len(wrong)), ""]
    L += table(["category", "tier", "rows", "4B greedy", "9B greedy", "pooled pick", "rows lost", "W2 verifier miss", "W3 coverage", "W4 never"],
               [(c, t, len(sel), *sys_cols(rows, sel, ["4B greedy", "9B greedy", "pooled pick"]), sum(1 for i in sel if not rows[i]["sys"]["pooled pick"]),
                 sum(1 for i in sel if rows[i]["ladder"].startswith("W2")), sum(1 for i in sel if rows[i]["ladder"].startswith("W3")),
                 sum(1 for i in sel if rows[i]["ladder"].startswith("W4"))) for c, t, sel in cells])
    L.append("")

    # ---- 2. greedy anatomy
    L += ["## 2. The greedy answers: what went wrong, by kind", "",
          "States from the execution logs; the buffer taxonomy (`taxonomy.classify`) is at line granularity: which of the buffer's lines the "
          "answer touched against which the gold touches. `same lines, wrong text` is the right place with the wrong edit (a missing `g` flag, "
          "a wrong replacement, an off-by-one column); the `scope` kinds are the wrong place.", ""]
    st_rows, kind_rows = [], []
    all_states = ["match", "valid_wrong", "invalid", "blocked", "timeout", "no prediction"]
    for m in ("4b", "9b"):
        g = [rows[i]["g" + m] for i in ids]
        st = collections.Counter(x["state"] for x in g)
        st_rows.append((MODEL_LABEL[m], *[("%d (%s)" % (st[s], fmt(st[s], n))) for s in all_states]))
        kinds = collections.Counter(x["kind"] for x in g if x["state"] == "valid_wrong")
        kind_rows.append((MODEL_LABEL[m], *["%d" % kinds[k] for k in T.KINDS if k not in ("match", "no_buffer")]))
    L += table(["model"] + all_states, st_rows)
    L.append("")
    L += ["Valid but wrong, by kind:", ""]
    L += table(["model"] + [k for k in T.KINDS if k not in ("match", "no_buffer")], kind_rows)
    L += ["", "Key: " + "; ".join("`%s` = %s" % (k, T.KIND_TEXT[k]) for k in T.KINDS if k not in ("match", "no_buffer")), ""]
    S["greedy_states"] = st_rows
    S["greedy_kinds"] = kind_rows
    within_rows = []
    for m in ("4b", "9b"):
        w = collections.Counter(rows[i]["g" + m]["within"] for i in ids if rows[i]["g" + m]["kind"] == "same_lines_wrong_text")
        within_rows.append((MODEL_LABEL[m], sum(w.values()), *[w[k] for k in T.WITHIN]))
    if exec_on:
        w = collections.Counter(rows[i]["pooled"]["within"] for i in ids if rows[i]["pooled"]["kind"] == "same_lines_wrong_text")
        within_rows.append(("pooled pick (wrong picks)", sum(w.values()), *[w[k] for k in T.WITHIN]))
    L += ["Inside `same_lines_wrong_text`, what the answer did to the characters of the lines it shares with the gold:", ""]
    L += table(["answers", "rows"] + list(T.WITHIN), within_rows)
    L += ["", "Key: " + "; ".join("`%s` = %s" % (k, T.WITHIN_TEXT[k]) for k in T.WITHIN), ""]
    S["greedy_within"] = within_rows
    # invalid by E-code
    err_rows = []
    codes = collections.Counter()
    for m in ("4b", "9b"):
        for i in ids:
            x = rows[i]["g" + m]
            if x["state"] == "invalid":
                codes[(m, x["error"] or "(no E-code)")] += 1
    top_codes = collections.Counter()
    for (m, c), k in codes.items():
        top_codes[c] += k
    ERR_TEXT = {"E486": "pattern not found", "E488": "trailing characters (malformed command)", "E16": "invalid range", "E492": "not an editor command",
                "E20": "mark not set", "E21": "cannot make changes, modifiable off", "E64": "a multi follows nothing", "E65": "illegal back reference",
                "E71": "invalid character after \\%", "E33": "no previous substitute regular expression", "E35": "no previous regular expression",
                "E54": "unmatched \\(", "E55": "unmatched \\)", "E493": "backwards range given", "E15": "invalid expression", "E121": "undefined variable",
                "E474": "invalid argument", "E475": "invalid argument", "E476": "invalid command", "E518": "unknown option", "E872": "unmatched (",
                "E873": "unmatched \\%(", "E14": "invalid address", "E477": "no ! allowed", "E117": "unknown function", "E481": "no range allowed",
                "E10": "\\ should be followed by /, ? or &", "E57": "unmatched \\z(", "E866": "misplaced \\%(", "E869": "unknown operator after \\@",
                "E61": "nested \\=", "E62": "nested \\=", "E749": "empty buffer", "E88": "cannot go before first buffer", "E108": "no such variable",
                "E118": "too many arguments for function", "E119": "not enough arguments", "E470": "command aborted", "E171": "missing :endif",
                "E53": "unmatched \\%(", "E114": "missing double quote", "E115": "missing single quote", "E554": "syntax error in \\%{...}",
                "E654": "missing delimiter after search pattern", "E877": "(NFA regexp) invalid character class"}
    for c, k in top_codes.most_common(14):
        err_rows.append((c, ERR_TEXT.get(c, ""), codes[("4b", c)], codes[("9b", c)]))
    L += ["Invalid ex answers by vim's error code (keys-mode validity is process-exit only, so keys slips land in `valid_wrong`, mostly as `noop`):", ""]
    L += table(["E-code", "meaning", "4B greedy", "9B greedy"], err_rows)
    S["greedy_errors"] = err_rows
    L.append("")
    # how far off: diff_lines for valid_wrong
    dl_rows = []
    for m in ("4b", "9b"):
        d = [rows[i]["g" + m]["diff_lines"] for i in ids if rows[i]["g" + m]["state"] == "valid_wrong" and rows[i]["g" + m]["kind"] != "noop"]
        c = collections.Counter(T.bucket(x, (1, 2, 3), ("1", "2", "3", "4+")) for x in d)
        sim = [rows[i]["g" + m]["similarity"] for i in ids if rows[i]["g" + m]["state"] == "valid_wrong" and rows[i]["g" + m]["kind"] != "noop"]
        dl_rows.append((MODEL_LABEL[m], len(d), c["1"], c["2"], c["3"], c["4+"], "%.2f" % statistics.median(sim) if sim else "—"))
    L += ["How far the wrong buffer is from the gold (valid-but-wrong answers that changed something): lines that differ, and the median "
          "character similarity of the whole buffer.", ""]
    L += table(["model", "answers", "1 line off", "2", "3", "4+", "median similarity"], dl_rows)
    S["greedy_distance"] = dl_rows
    L.append("")
    # mode cross-tab
    L += ["### 2a. Mode: what the model answered in, against the gold's mode", "",
          "Buffers are the judge, so an answer in the other mode can still solve the row.", ""]
    mode_rows = []
    for m in ("4b", "9b"):
        for gm in ("ex", "keys"):
            for pm in ("ex", "keys"):
                sel = [i for i in ids if rows[i]["gold_mode"] == gm and rows[i]["g" + m]["mode"] == pm]
                if sel:
                    mode_rows.append((MODEL_LABEL[m], gm, pm, len(sel), fmt(sum(rows[i]["g" + m]["match"] for i in sel), len(sel))))
    L += table(["model", "gold mode", "answered in", "rows", "exec-match"], mode_rows)
    S["greedy_modes"] = mode_rows
    L.append("")
    cur_rows = []
    for m in ("4b", "9b"):
        for gm in ("ex", "keys"):
            for cur in ("(1,1)", "elsewhere"):
                sel = [i for i in ids if rows[i]["gold_mode"] == gm and rows[i]["features"]["cursor"] == cur]
                cur_rows.append((MODEL_LABEL[m], gm, cur, len(sel), fmt(sum(rows[i]["g" + m]["match"] for i in sel), len(sel)),
                                 fmt(sum(rows[i]["sys"]["pooled pick"] for i in sel), len(sel))))
    L += ["Cursor dependence (keys-mode golds often start from a stated cursor; ex-mode golds rarely do):", ""]
    L += table(["model", "gold mode", "cursor", "rows", "greedy", "pooled pick"], cur_rows)
    S["cursor"] = cur_rows
    L.append("")

    # ---- 3. one edit away
    if D["probes"]:
        L += ["## 3. One mechanical edit away", "",
              "Each wrong answer was re-run in the sandbox with single mechanical edits (`taxonomy.mutations`): a `g` flag on every substitute lacking "
              "one, a `%` range added or removed, `g` ↔ `v` on a global command, and for keys `gg` or `0` prefixed. A hit means the edited command "
              "exec-matches. This is a diagnostic of how near the misses are, never a repair surface (CLAUDE.md §1).", ""]
        pr = []
        for name, p in D["probes"].items():
            for rule, (tried, fixed) in sorted(p["rules"].items()):
                pr.append((name, rule, tried, fixed, fmt(fixed, tried)))
            pr.append(("**%s**" % name, "any rule (rows)", p["tried_rows"], p["fixed_rows"], fmt(p["fixed_rows"], p["tried_rows"])))
        L += table(["system", "edit", "answers tried", "now exec-match", "rate"], pr)
        L += ["", "By the kind of the answer it fixed: " + "; ".join(
            "%s: %s" % (name, ", ".join("%s %d" % kv for kv in sorted(p["fixed_kinds"].items(), key=lambda kv: -kv[1]))) for name, p in D["probes"].items()), ""]
        S["probes"] = {k: {kk: vv for kk, vv in v.items() if kk != "fixed_ids"} for k, v in D["probes"].items()}

    # ---- 4. correlates
    L += ["## 4. What goes with failure: the rows' own properties", "",
          "Exec-match by property, for the two greedy answers, the pooled pick and the 256-sample ceiling. Properties come from the case "
          "(gold, instruction, buffer), never from any model's output. Read differences of a few points on small groups as noise.", ""]
    feat_names = []
    for i in ids:
        for k in rows[i]["features"]:
            if k not in feat_names:
                feat_names.append(k)
    frows = []
    for fn in feat_names:
        groups = by(rows, [i for i in ids if fn in rows[i]["features"]], lambda r: r["features"][fn])
        order = sorted(groups, key=lambda v: (-len(groups[v]) if fn in ("source", "instruction provenance", "category") else str(v)))
        for v in order:
            sel = groups[v]
            frows.append((fn, v, len(sel), *sys_cols(rows, sel, ["4B greedy", "9B greedy", "pooled pick", "pass@128 either"])))
    L += table(["property", "value", "rows", "4B greedy", "9B greedy", "pooled pick", "pass@128 either"], frows)
    S["features"] = frows
    mix = {}
    for split in ("train", "val", "test"):
        rs = load_jsonl(REPO / "data" / "splits" / ("%s.jsonl" % split))
        cnt = collections.Counter(r.get("source") for r in rs)
        mix[split] = "handcrafted %.0f%%, deep-regex-kb13 %.0f%%, structured-regex %.0f%%, emulator and candidate corpora %.0f%%" % (
            pct(cnt["handcrafted"], len(rs)), pct(cnt["deep-regex-kb13"], len(rs)), pct(cnt["structured-regex"], len(rs)),
            pct(sum(v for k, v in cnt.items() if k not in ("handcrafted", "deep-regex-kb13", "structured-regex")), len(rs)))
    L += ["", "The source mix is the same across splits (train: %s; test: %s), so the regex-corpus rows are not over-represented in test; they are "
          "simply the solved part of it." % (mix["train"], mix["test"]), ""]
    S["source_mix"] = mix
    L.append("")

    # ---- 5. the verifier
    L += ["## 5. The verifier's errors (the pooled pick, 8 + 8 samples)", "",
          "The pooled pick is the 4B verifier's top-scored candidate among both models' distinct sampled commands; ties by sample count, "
          "then the 9B's greedy flag, then text. Where a right candidate existed, where did it rank, and what did the verifier prefer instead?", ""]
    P = {i: rows[i]["pooled"] for i in ids}
    have_right = [i for i in ids if P[i]["n_right"] > 0]
    both = [i for i in have_right if P[i]["n_right"] < P[i]["n_cands"]]
    miss = [i for i in have_right if not P[i]["label"]]
    L += table(["", "rows"], [
        ("prompts with a right candidate among the 16", len(have_right)),
        ("… of which the pick is right", "%d (%s)" % (len(have_right) - len(miss), fmt(len(have_right) - len(miss), len(have_right)))),
        ("prompts holding both right and wrong candidates", len(both)),
        ("… of which the pick is right", "%d (%s)" % (sum(P[i]["label"] for i in both), fmt(sum(P[i]["label"] for i in both), len(both)))),
        ("prompts with only wrong candidates (the pick cannot be right)", n - len(have_right)),
        ("mean distinct candidates per prompt / mean right ones", "%.1f / %.1f" % (statistics.mean(P[i]["n_cands"] for i in ids), statistics.mean(P[i]["n_right"] for i in ids)))])
    L.append("")
    rk = collections.Counter(T.bucket(P[i]["rank_right"], (1, 2, 3), ("1 (picked)", "2", "3", "4+")) for i in have_right)
    L += ["Rank of the first right candidate in the verifier's order, over prompts that have one:", ""]
    L += table(["rank", "prompts", "share"], [(k, rk[k], fmt(rk[k], len(have_right))) for k in ("1 (picked)", "2", "3", "4+")])
    L.append("")
    margins = sorted(P[i]["margin"] for i in miss)
    if margins:
        q = statistics.quantiles(margins, n=4) if len(margins) >= 4 else margins
        L += ["Verifier misses (%d prompts): the wrong pick's score minus the best right candidate's score — median %.2f, quartiles %.2f / %.2f; "
              "%d misses within 1 point, %d within 2, %d beyond 4 (scores are logit(yes) − logit(no))." % (
                  len(miss), statistics.median(margins), q[0], q[-1], sum(1 for x in margins if x <= 1), sum(1 for x in margins if x <= 2), sum(1 for x in margins if x > 4)), ""]
    # bias checks on misses
    br = {i: P[i]["best_right"] for i in miss}
    longer = sum(1 for i in miss if len(P[i]["command"]) > len(br[i]["command"]))
    shorter = sum(1 for i in miss if len(P[i]["command"]) < len(br[i]["command"]))
    mode_sw = collections.Counter((br[i]["mode"], P[i]["mode"]) for i in miss if br[i]["mode"] != P[i]["mode"])
    n_lower = sum(1 for i in miss if P[i]["n"] < br[i]["n"])
    n_higher = sum(1 for i in miss if P[i]["n"] > br[i]["n"])
    inval = sum(1 for i in miss if P[i]["state"] == "invalid")
    src = collections.Counter(P[i]["sources"] for i in miss)
    src_r = collections.Counter(br[i]["sources"] for i in miss)
    gold_mode_pick = sum(1 for i in miss if P[i]["mode"] == rows[i]["gold_mode"])
    L += ["What the wrong pick looks like next to the best right candidate, over the %d misses:" % len(miss), ""]
    L += table(["check", "count", "reading"], [
        ("pick is longer than the right candidate / shorter", "%d / %d" % (longer, shorter), "a length bias would show as one side dominating"),
        ("pick was sampled fewer times than the right one / more", "%d / %d" % (n_lower, n_higher), "fewer = the verifier overrode a more frequent right answer"),
        ("pick is in a different mode from the right candidate", "%d (%s)" % (sum(mode_sw.values()), ", ".join("%s→%s %d" % (a, b, k) for (a, b), k in mode_sw.items())), "ex↔keys confusions"),
        ("pick is in the gold's mode", "%d of %d" % (gold_mode_pick, len(miss)), ""),
        ("pick is an invalid command (vim error)", inval, "the verifier cannot run anything; an invalid pick is text that looked right"),
        ("pick came from the 4B only / 9B only / both", "%d / %d / %d" % (src["4b"], src["9b"], src["4b+9b"]), "right candidate: %d / %d / %d" % (src_r["4b"], src_r["9b"], src_r["4b+9b"])),
    ])
    L.append("")
    maj_over = sum(1 for i in ids if P[i]["majority_label"] and not P[i]["label"])
    maj_resc = sum(1 for i in ids if not P[i]["majority_label"] and P[i]["label"])
    L += ["Against plain voting over the same 16 samples: the verifier overrode a right plurality on %d prompts and rescued a right minority on %d "
          "(net +%d; the most common command is right on %d prompts, the pick on %d)." % (
              maj_over, maj_resc, maj_resc - maj_over, sum(P[i]["majority_label"] for i in ids), sum(P[i]["label"] for i in ids)), ""]
    S["verifier"] = {"have_right": len(have_right), "miss": len(miss), "both": len(both), "rank": dict(rk), "longer": longer, "shorter": shorter,
                     "n_lower": n_lower, "n_higher": n_higher, "mode_switch": {"%s->%s" % k: v for k, v in mode_sw.items()}, "invalid_pick": inval,
                     "pick_sources": dict(src), "right_sources": dict(src_r), "majority_overridden": maj_over, "minority_rescued": maj_resc,
                     "margins": margins}
    if exec_on:
        kinds = collections.Counter(P[i]["kind"] for i in wrong)
        L += ["What the wrong picks do to the buffer (all %d wrong picks, re-executed; `invalid` = an ex command vim rejected, buffer unchanged):" % len(wrong), ""]
        L += table(["kind", "wrong picks", "of them with a right candidate among the 16"],
                   [(k, kinds[k], sum(1 for i in miss if P[i]["kind"] == k)) for k in ("invalid",) + T.KINDS if kinds[k]])
        L.append("")
        S["verifier"]["pick_kinds"] = dict(kinds)
        det = [i for i in wrong if P[i]["kind"] in ("noop", "invalid")]
        det_miss = [i for i in det if P[i]["n_right"] > 0]
        det_fixed = sum(rows[i]["ask20"]["after_reply"] for i in det)
        L += ["A pick that vim rejects, or that leaves the buffer as it was, is recognisable without an answer key, and no right pick can be either "
              "(every gold changes the buffer). %d of the %d wrong picks are such (%d invalid, %d valid no-ops), %d of them among the %d verifier "
              "misses. Exploratory: if those picks were turned into a question instead (the sandbox deciding only whether to ask, which is what the "
              "2026-10-13 exception allows), the 4B's answer after the simulated reply is right on %d of the %d, i.e. %s would become %s; %d of the "
              "%d have a right answer within 128 samples of one model." % (
                  len(det), len(wrong), kinds["invalid"], kinds["noop"], len(det_miss), len(miss), det_fixed, len(det),
                  fmt(D["checks"]["pooled pick"], n), fmt(D["checks"]["pooled pick"] + det_fixed, n),
                  sum(rows[i]["sys"]["pass@128 either"] for i in det), len(det)), ""]
        S["verifier"]["detectable_picks"] = {"n": len(det), "invalid": kinds["invalid"], "noop": kinds["noop"], "of_misses": len(det_miss), "ask_fixes": det_fixed}
    # AUROC per slice over candidates
    cand_pairs = collections.defaultdict(list)
    for m in ("4b", "9b"):
        pass
    ex = {m: C9.group(load_jsonl(EX[m])) for m in ("4b", "9b")}
    sc = {m: C9.load_scores(SCORES[m]) for m in ("4b", "9b")}
    for i in ids:
        pool = C9.pooled(ex["4b"].get(i, []), ex["9b"].get(i, []))
        ps = C9.pooled_scores(pool, sc["4b"], sc["9b"], i)
        for c in pool:
            pr = (ps[(i, c["cand_mode"], c["cand_command"])], c["label"])
            cand_pairs["all"].append(pr)
            cand_pairs["tier " + rows[i]["tier"]].append(pr)
            cand_pairs["gold " + rows[i]["gold_mode"]].append(pr)
            cand_pairs["candidate " + c["cand_mode"]].append(pr)
            cand_pairs["category " + rows[i]["category"]].append(pr)
    au_rows = []
    cat_n = collections.Counter(rows[i]["category"] for i in ids)
    for k in sorted(cand_pairs, key=lambda k: (k.split()[0] != "all", k.split()[0], -len(cand_pairs[k]))):
        if k.startswith("category") and cat_n[k.split(" ", 1)[1]] < MIN_CAT:
            continue
        a = auroc(cand_pairs[k])
        au_rows.append((k, len(cand_pairs[k]), sum(l for s, l in cand_pairs[k]), "%.3f" % a if a is not None else "—"))
    L += ["Candidate-level AUROC of the verifier's score (one candidate per distinct command per prompt, 8 + 8):", ""]
    L += table(["slice", "candidates", "right", "AUROC"], au_rows)
    S["verifier"]["auroc"] = au_rows
    L.append("")
    # calibration
    bins = [(-1e9, -4, "≤ −4"), (-4, -2, "−4 … −2"), (-2, 0, "−2 … 0"), (0, 2, "0 … 2"), (2, 4, "2 … 4"), (4, 1e9, "> 4")]
    cal = []
    for lo, hi, lab in bins:
        c = [(s, l) for s, l in cand_pairs["all"] if lo < s <= hi]
        p = [i for i in ids if lo < P[i]["score"] <= hi]
        cal.append((lab, len(c), fmt(sum(l for s, l in c), len(c)), len(p), fmt(sum(P[i]["label"] for i in p), len(p))))
    L += ["Calibration: how often a candidate in a score band is right, and how often the pick is right when its score lands in that band "
          "(the second is what an ask-or-answer threshold reads):", ""]
    L += table(["score band", "candidates", "right", "picks", "pick right"], cal)
    S["verifier"]["calibration"] = cal
    L.append("")
    # at 128 + 128
    C = {i: rows[i]["ceil"] for i in ids}
    hr = [i for i in ids if C[i]["pick128"]["n_right"] > 0]
    rk128 = collections.Counter(T.bucket(C[i]["pick128"]["rank_right"], (1, 2, 3, 10), ("1 (picked)", "2", "3", "4-10", "11+")) for i in hr)
    L += ["### 5a. The same verifier over all 256 samples", "",
          "With every distinct command of both models' 128 samples as candidates (%.1f per prompt), the pick is right on %d of %d prompts (%s) and on "
          "%d of the %d prompts that hold a right candidate (%s). Rank of the first right candidate:" % (
              statistics.mean(C[i]["pick128"]["n_cands"] for i in ids), sum(C[i]["pick128"]["label"] for i in ids), n, fmt(sum(C[i]["pick128"]["label"] for i in ids), n),
              sum(C[i]["pick128"]["label"] for i in hr), len(hr), fmt(sum(C[i]["pick128"]["label"] for i in hr), len(hr))), ""]
    L += table(["rank", "prompts", "share"], [(k, rk128[k], fmt(rk128[k], len(hr))) for k in ("1 (picked)", "2", "3", "4-10", "11+")])
    L.append("")
    inv128 = sum(1 for i in ids if not C[i]["pick128"]["label"] and C[i]["pick128"]["state"] == "invalid")
    L += ["Of the %d wrong picks at 256, %d are invalid commands. %d candidate commands had no verifier score and were skipped." % (
        n - sum(C[i]["pick128"]["label"] for i in ids), inv128, D["missing_scores"]), ""]
    S["verifier"]["at128"] = {"right": sum(C[i]["pick128"]["label"] for i in ids), "have_right": len(hr), "rank": dict(rk128), "invalid_pick": inv128}

    # ---- 6. confidently wrong
    L += ["## 6. Confidently wrong: the answers the models give almost every time", "",
          "From the 128 samples per model: the most common command, its share, and whether it is right. A model that answers one way in "
          "≥ 90% of samples has no diversity for a verifier or a vote to work with; where the answer is wrong, only training can move it.", ""]
    conf = []
    for m in ("4b", "9b"):
        sel = [i for i in ids if C[i][m]["top_share"] >= 0.9]
        w = [i for i in sel if not C[i][m]["top_right"]]
        conf.append((MODEL_LABEL[m].split()[0], len(sel), fmt(len(sel), n), len(w), fmt(len(w), n),
                     fmt(sum(rows[i]["sys"]["pooled pick"] for i in w), len(w)), fmt(sum(rows[i]["sys"]["pass@128 either"] for i in w), len(w))))
    L += table(["model", "prompts answered ≥ 90% one way", "share", "… and that answer is wrong", "share of all rows", "pooled pick right on those", "pass@128 either on those"], conf)
    S["confident"] = conf
    L.append("")
    shared = [i for i in ids if C[i]["shared_top"] and not C[i]["4b"]["top_right"] and min(C[i]["4b"]["top_share"], C[i]["9b"]["top_share"]) >= 0.5]
    shared.sort(key=lambda i: -(C[i]["4b"]["top_share"] + C[i]["9b"]["top_share"]))
    L += ["### 6a. Shared wrong answers", "",
          "Both models' most common answer is the same wrong command, each giving it in at least half of its samples: %d rows. The same text from two "
          "differently sized models points at the instruction's reading (or the training data), not at a sampling accident. Listed for the owner's "
          "eye; nothing here was changed." % len(shared), ""]
    sh_rows = []
    for i in shared:
        c = cases[i]
        k = C[i]["4b"].get("top_kind") or ""
        sh_rows.append((i, rows[i]["category"], rows[i]["tier"], code(c["instruction"][:110] + ("…" if len(c["instruction"]) > 110 else "")),
                        ex_mode(c["mode"], c["gold_command"]), ex_mode(C[i]["4b"]["top_mode"], C[i]["4b"]["top_command"]),
                        "%.0f%% / %.0f%%" % (100 * C[i]["4b"]["top_share"], 100 * C[i]["9b"]["top_share"]), k))
    L += table(["id", "category", "tier", "instruction", "gold", "both models' favourite answer", "share 4B / 9B", "kind"], sh_rows)
    S["shared_wrong"] = [r[0] for r in sh_rows]
    L.append("")
    never = [i for i in ids if not rows[i]["sys"]["pass@128 either"]]
    if exec_on:
        nk = collections.Counter(C[i]["9b"].get("top_kind") for i in never)
        nw = collections.Counter(C[i]["9b"].get("top_within") for i in never if C[i]["9b"].get("top_kind") == "same_lines_wrong_text")
        L += ["### 6b. The %d rows no sample of the 256 solves: what kind of wrong the favourite answer is" % len(never), "",
              "The ceiling report lists these rows; here the 9B's most common answer on each is re-executed and classified: "
              + "; ".join("%s %d" % (k, v) for k, v in nk.most_common()) + ". Inside `same_lines_wrong_text`: "
              + "; ".join("%s %d" % (k, v) for k, v in nw.most_common()) + ".", ""]
        nmode = collections.Counter((rows[i]["gold_mode"], C[i]["9b"]["top_mode"]) for i in never)
        L += ["Mode of the favourite answer against the gold's mode on these rows: " + ", ".join("gold %s → %s %d" % (a, b, k) for (a, b), k in sorted(nmode.items())) + ".", ""]
        S["never_kinds"] = dict(nk)

    # ---- 7. 4B vs 9B
    L += ["## 7. The 4B against the 9B: different rows", ""]
    o4 = [i for i in ids if rows[i]["g4b"]["match"] and not rows[i]["g9b"]["match"]]
    o9 = [i for i in ids if rows[i]["g9b"]["match"] and not rows[i]["g4b"]["match"]]
    bothr = [i for i in ids if rows[i]["g4b"]["match"] and rows[i]["g9b"]["match"]]
    L += ["Both greedy answers right on %d rows, only the 4B on %d, only the 9B on %d, neither on %d (a perfect chooser between the two would score %s)." % (
        len(bothr), len(o4), len(o9), n - len(bothr) - len(o4) - len(o9), fmt(len(bothr) + len(o4) + len(o9), n)), ""]
    def comp(sel, other):
        tiers = collections.Counter(rows[i]["tier"] for i in sel)
        modes = collections.Counter(rows[i]["gold_mode"] for i in sel)
        st = collections.Counter(rows[i][other]["state"] for i in sel)
        kd = collections.Counter(rows[i][other]["kind"] for i in sel if rows[i][other]["state"] == "valid_wrong")
        return ("%d / %d / %d" % (tiers["T1"], tiers["T2"], tiers["T3"]), "%d / %d" % (modes["ex"], modes["keys"]),
                ", ".join("%s %d" % kv for kv in st.most_common()), ", ".join("%s %d" % kv for kv in kd.most_common(4)))
    L += table(["rows", "n", "T1 / T2 / T3", "ex / keys", "the other model's state there", "its kinds (valid-wrong)"],
               [("only the 4B right", len(o4), *comp(o4, "g9b")), ("only the 9B right", len(o9), *comp(o9, "g4b"))])
    L.append("")
    cat_cmp = []
    for c, sel in sorted(by(rows, ids, lambda r: r["category"]).items(), key=lambda kv: -len(kv[1])):
        a, b = sum(rows[i]["g4b"]["match"] for i in sel), sum(rows[i]["g9b"]["match"] for i in sel)
        cat_cmp.append((c, len(sel), fmt(a, len(sel)), fmt(b, len(sel)), "%+d" % (b - a), sum(1 for i in sel if i in o4), sum(1 for i in sel if i in o9)))
    L += table(["category", "rows", "4B greedy", "9B greedy", "9B − 4B (rows)", "only 4B", "only 9B"], cat_cmp)
    S["complement"] = {"both": len(bothr), "only4": len(o4), "only9": len(o9), "by_category": cat_cmp}
    L.append("")
    t2 = []
    for c, sel in sorted(by(rows, [i for i in ids if rows[i]["tier"] == "T2"], lambda r: r["category"]).items(), key=lambda kv: -len(kv[1])):
        t2.append((c, len(sel), *sys_cols(rows, sel, ["4B greedy", "9B greedy", "pooled pick"])))
    L += ["T2 by category (the tier where the 9B trails the 4B, 55.0% against 64.9%):", ""]
    L += table(["category (T2)", "rows", "4B greedy", "9B greedy", "pooled pick"], t2)
    L.append("")

    # ---- 8. stages
    L += ["## 8. Training stages: which rows each stage gained and lost", "",
          "Greedy answers of the stage checkpoints on test v2 (`runs/remeasure_2026-09-25/v2/`). Churn is the point: the net change hides rows "
          "that flip both ways.", ""]
    for m in ("4b", "9b"):
        st = D["stages"][m]
        srows = []
        prev = None
        for lab, o in st:
            k = sum(o[i] for i in ids)
            if prev is None:
                srows.append((lab, k, fmt(k, n), "", "", ""))
            else:
                g = [i for i in ids if o[i] and not prev[i]]
                l = [i for i in ids if prev[i] and not o[i]]
                tg = collections.Counter(rows[i]["tier"] for i in g)
                tl = collections.Counter(rows[i]["tier"] for i in l)
                srows.append((lab, k, fmt(k, n), "+%d / −%d" % (len(g), len(l)),
                              "%d / %d / %d" % (tg["T1"], tg["T2"], tg["T3"]), "%d / %d / %d" % (tl["T1"], tl["T2"], tl["T3"])))
            prev = o
        first, last = st[0][1], st[-1][1]
        ever = sum(1 for i in ids if any(o[i] for _, o in st))
        lost = [i for i in ids if first[i] and not last[i]]
        gained = [i for i in ids if last[i] and not first[i]]
        L += ["%s:" % ("Qwen3.5-4B" if m == "4b" else "Qwen3.5-9B"), ""]
        L += table(["stage", "right", "exec-match", "vs previous stage", "gained T1 / T2 / T3", "lost T1 / T2 / T3"], srows)
        L += ["", "Some stage of this model solves %d rows (%s). From the first stage to the last: %d rows gained, %d lost; of the lost, %d are solved "
              "by the pooled pick and %d by neither model in 256 samples." % (
                  ever, fmt(ever, n), len(gained), len(lost), sum(rows[i]["sys"]["pooled pick"] for i in lost),
                  sum(1 for i in lost if not rows[i]["sys"]["pass@128 either"])), ""]
        S["stages_" + m] = srows

    # ---- 9. the ask
    asked = [i for i in ids if rows[i]["ask20"]["asked"]]
    L += ["## 9. The clarifying question, where the verifier asks", "",
          "The gated ask asks on the %d rows whose best verifier score is lowest and answers the 4B pick elsewhere. On the asked rows: the 4B pick "
          "would have been right on %d, the answer after the simulated user's reply is right on %d, the pooled pick on %d, and some sample of the "
          "16 was right on %d. On the %d rows it does not ask, the 4B pick is right on %d and the pooled pick on %d." % (
              len(asked), sum(rows[i]["pick4b"]["label"] for i in asked), sum(rows[i]["ask20"]["after_reply"] for i in asked),
              sum(rows[i]["sys"]["pooled pick"] for i in asked), sum(rows[i]["sys"]["any of 16"] for i in asked),
              n - len(asked), sum(rows[i]["pick4b"]["label"] for i in ids if not rows[i]["ask20"]["asked"]),
              sum(rows[i]["sys"]["pooled pick"] for i in ids if not rows[i]["ask20"]["asked"])), ""]
    S["ask"] = {"asked": len(asked), "pick_right_asked": sum(rows[i]["pick4b"]["label"] for i in asked),
                "after_reply_right": sum(rows[i]["ask20"]["after_reply"] for i in asked), "pooled_right_asked": sum(rows[i]["sys"]["pooled pick"] for i in asked)}

    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return S


# ------------------------------------------------------------ the gallery

def udiff(expected, got, limit=14):
    e, g = T.norm(expected) or [], T.norm(got) or []
    lines = list(difflib.unified_diff(e, g, "expected", "got", lineterm="", n=1))[2:]
    if len(lines) > limit:
        lines = lines[:limit] + ["… (%d more)" % (len(lines) - limit)]
    return lines


def example(D, i, title, answers):
    c, r = D["cases"][i], D["rows"][i]
    buf = T.norm(c["buffer_before"]) or []
    shown = buf[:10] + (["… (%d more lines)" % (len(buf) - 10)] if len(buf) > 10 else [])
    L = ["### %s — `%s` (%s, %s, gold %s mode)" % (title, i, r["category"], r["tier"], r["gold_mode"]), "",
         "Instruction: %s  " % c["instruction"], "Cursor: line %d, column %d" % tuple(c.get("cursor") or (1, 1)), "", "```"] + shown + ["```", ""]
    L.append("| who | answer | outcome |")
    L.append("|---|---|---|")
    L.append("| gold | %s | — |" % ex_mode(c["mode"], c["gold_command"]))
    for who, mode, cmd, outcome in answers:
        L.append("| %s | %s | %s |" % (who, ex_mode(mode, cmd), outcome))
    L.append("")
    return L


def gallery(D, out):
    ids, rows, cases = D["ids"], D["rows"], D["cases"]
    L = ["# Failure gallery (test v2)", "",
         "Companion to `report.md`: concrete rows for each kind of failure, chosen by rule (first ids in order, or the largest margins), "
         "not by hand. Diffs are expected → got for the 4B greedy answer where it applies.", ""]
    g4log = {r["id"]: r for r in load_jsonl(GREEDY["4b"])}
    g9log = {r["id"]: r for r in load_jsonl(GREEDY["9b"])}

    def outcome(x):
        if x["state"] == "match":
            return "exec-match"
        if x["state"] == "valid_wrong":
            return "valid, wrong: %s%s" % (x["kind"], (", %d line(s) off" % x["diff_lines"]) if x.get("diff_lines") else "")
        if x["state"] == "invalid":
            return "invalid %s" % x["error"]
        return x["state"]

    L += ["## A. The greedy answers, by kind (the 4B served adapter)", ""]
    for kind in [k for k in T.KINDS if k not in ("match", "no_buffer")]:
        sel = [i for i in ids if rows[i]["g4b"]["state"] == "valid_wrong" and rows[i]["g4b"]["kind"] == kind][:2]
        for i in sel:
            x, y = rows[i]["g4b"], rows[i]["g9b"]
            L += example(D, i, "%s — %s" % (kind, T.KIND_TEXT[kind]), [
                ("4B greedy", x["mode"], x["command"], outcome(x) + (" — fixed by: %s" % ", ".join(x["mutation_fix"]) if x.get("mutation_fix") else "")),
                ("9B greedy", y["mode"], y["command"], outcome(y)),
                ("pooled pick", rows[i]["pooled"]["mode"], rows[i]["pooled"]["command"], "right" if rows[i]["pooled"]["label"] else "wrong (%s)" % (rows[i]["pooled"].get("kind") or rows[i]["pooled"]["state"]))])
            L += ["```diff"] + udiff(cases[i]["buffer_after"], g4log[i]["buffer_after"]) + ["```", ""]
    L += ["## B. Invalid greedy answers (the 4B), by the most common error codes", ""]
    seen = collections.Counter()
    for i in ids:
        x = rows[i]["g4b"]
        if x["state"] == "invalid" and seen[x["error"]] < 2 and x["error"] in ("E486", "E488", "E16", "E492"):
            seen[x["error"]] += 1
            L += example(D, i, "invalid %s" % x["error"], [
                ("4B greedy", x["mode"], x["command"], "invalid: %s" % (g4log[i]["stderr"].strip()[:120] or x["error"])),
                ("9B greedy", rows[i]["g9b"]["mode"], rows[i]["g9b"]["command"], outcome(rows[i]["g9b"])),
                ("pooled pick", rows[i]["pooled"]["mode"], rows[i]["pooled"]["command"], "right" if rows[i]["pooled"]["label"] else "wrong")])
    L += ["## C. Verifier misses with the largest margin (a right candidate was among the 16)", ""]
    miss = sorted((i for i in ids if rows[i]["pooled"]["margin"] is not None), key=lambda i: -rows[i]["pooled"]["margin"])[:8]
    for i in miss:
        p = rows[i]["pooled"]
        br = p["best_right"]
        L += example(D, i, "verifier miss, margin %.2f" % p["margin"], [
            ("pooled pick (score %.2f, sampled %d×, from %s)" % (p["score"], p["n"], p["sources"]), p["mode"], p["command"], "wrong: %s" % (p.get("kind") or p["state"])),
            ("best right candidate (score %.2f, sampled %d×, from %s)" % (br["score"], br["n"], br["sources"]), br["mode"], br["command"], "right"),
            ("4B greedy", rows[i]["g4b"]["mode"], rows[i]["g4b"]["command"], outcome(rows[i]["g4b"])),
            ("9B greedy", rows[i]["g9b"]["mode"], rows[i]["g9b"]["command"], outcome(rows[i]["g9b"]))])
    L += ["## D. Selection losses: a greedy answer was right and the pooled pick is not", ""]
    sel = [i for i in ids if rows[i]["ladder"].startswith("W1")][:6]
    for i in sel:
        p = rows[i]["pooled"]
        L += example(D, i, "selection loss", [
            ("pooled pick (score %.2f, sampled %d×, from %s)" % (p["score"], p["n"], p["sources"]), p["mode"], p["command"], "wrong: %s" % (p.get("kind") or p["state"])),
            ("4B greedy", rows[i]["g4b"]["mode"], rows[i]["g4b"]["command"], outcome(rows[i]["g4b"])),
            ("9B greedy", rows[i]["g9b"]["mode"], rows[i]["g9b"]["command"], outcome(rows[i]["g9b"]))])
    L += ["## E. Coverage misses: none of the 16 right, but a right answer appears within 128 samples", ""]
    sel = [i for i in ids if rows[i]["ladder"].startswith("W3")][:6]
    for i in sel:
        c = rows[i]["ceil"]
        L += example(D, i, "coverage miss (4B right in %d of 128, 9B in %d of 128)" % (c["4b"]["c"], c["9b"]["c"]), [
            ("4B favourite (%.0f%% of samples)" % (100 * c["4b"]["top_share"]), c["4b"]["top_mode"], c["4b"]["top_command"], "right" if c["4b"]["top_right"] else "wrong: %s" % (c["4b"].get("top_kind") or c["4b"]["top_state"])),
            ("9B favourite (%.0f%% of samples)" % (100 * c["9b"]["top_share"]), c["9b"]["top_mode"], c["9b"]["top_command"], "right" if c["9b"]["top_right"] else "wrong: %s" % (c["9b"].get("top_kind") or c["9b"]["top_state"]))])
    (out / "examples.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RUNS / "error_analysis_2026-10-21"))
    ap.add_argument("--no-exec", action="store_true", help="skip the sandbox re-executions (buffer kinds of picks, the one-edit probe)")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sandbox = Sandbox(not args.no_exec)
    D = build(sandbox)
    S = report(D, out, sandbox.enabled)
    gallery(D, out)
    with open(out / "rows.jsonl", "w", encoding="utf-8") as fh:
        for i in D["ids"]:
            fh.write(json.dumps(D["rows"][i], ensure_ascii=False) + "\n")
    S["checks"] = D["checks"]
    S["sandbox_calls"] = D["sandbox_calls"]
    page = []
    for i in D["ids"]:
        c, r = D["cases"][i], D["rows"][i]
        page.append({"id": i, "category": c["category"], "tier": r["tier"], "mode": c["mode"], "source": c.get("source"),
                     "instruction": c["instruction"], "cursor": c.get("cursor") or [1, 1], "buffer_before": c["buffer_before"],
                     "buffer_gold": c["buffer_after"], "gold": c["gold_command"], "ladder": r["ladder"], "sys": r["sys"],
                     "features": r["features"], "g4": r["g4b"], "g9": r["g9b"], "pooled": r["pooled"],
                     "pick4": {k: r["pick4b"][k] for k in ("label", "command", "mode", "state", "score")},
                     "pick9": {k: r["pick9b"][k] for k in ("label", "command", "mode", "state", "score")},
                     "ask": r["ask20"], "ceil": {m: {k: v for k, v in r["ceil"][m].items()} for m in ("4b", "9b")},
                     "pick128": r["ceil"]["pick128"]})
    # the page host refuses U+FFFD; a few sampled commands carry one (undecodable model output), shown as a visible marker
    text = json.dumps(page, ensure_ascii=False, separators=(",", ":")).replace("\ufffd", "[U+FFFD]")
    (out / "page_data.json").write_text(text, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(S, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print("wrote %s: report.md, examples.md, rows.jsonl, summary.json, page_data.json (%d sandbox calls)" % (out, D["sandbox_calls"]))
    print("checks:", D["checks"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
