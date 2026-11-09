#!/usr/bin/env python3
"""V timed as a system: the verifier-gated ask on the served 4B, one request at a time (runs/vtime4b_2026-11-09/plan.md).

sys4b's V (scripts/retr27b/fewer.py) was read on the CPU from stored samples and scores, so its cost row was composed
from parts timed apart. Here it runs live, built from the calls those parts came from:

  1. the §6 answer and 8 samples at T = 1, requested together: scripts/serve/roundtrip.py's first step;
  2. the verifier on every distinct parsed sample, in one batched request: scripts/serve_pooled/pooled.py's call, on
     the plain prompt, since the verifier never saw examples;
  3. fewer.py's gate: ask when the best score is at most τ (sys4b's, from its report), else the pick (highest score,
     ties by count, then mode and text). Where the question comes back empty it answers as if it had not asked;
  4. the question and the turn after the reply: the round trip's, with the simulated user's recorded reply.

Nothing is executed while V runs: it reads text. The 5-of-8 round trip at 8 runs on the same rows as the same-session
control, interleaved row by row with alternate rows first, so a faster or slower card moves both alike.

    timing  4B   V and the round trip on the 200 timing rows (ask27b's), arm a (no examples) or b (8 BM25 examples)
    report  CPU  GPU-seconds, waits and cost per solved beside sys4b's composed rows; the answers through the box

    python scripts/vtime4b/vtime4b.py timing --arm a
    python scripts/vtime4b/vtime4b.py report
"""
import argparse
import json
import math
import pathlib
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "clarify",
          REPO / "scripts" / "readings", REPO / "scripts" / "reranker", REPO / "scripts" / "serve_pooled",
          REPO / "scripts" / "retr27b", REPO / "scripts" / "sys4b"):
    sys.path.insert(0, str(p))
import sys4b as S                                                 # noqa: E402  the 27B's stages, pointed at the 4B
import pooled as PL                                               # noqa: E402  the verifier's call, merge and pick
from prompt import messages_for_case, parse_output                # noqa: E402
from protocol import (ASK_PREFIX, PLACEBO_REPLY, QUESTION_MAX_TOKENS,   # noqa: E402
                      cut_question, turn1_messages, turn2_messages)

Q, A = S.Q, S.A
RUN = REPO / "runs" / "vtime4b_2026-11-09"
K = 8
MAX_AGREE = 5
TIMING_N = 200
GPU_HOUR = 0.30                                                   # USD, as every cost table so far
ARMS = ("a", "b")
STORED_SCORES = {"a": S.ROOT / "a" / "served" / "scores27.jsonl", "b": S.ROOT / "b" / "scores.jsonl"}


def case_of(c):
    """The case the round trip builds from a request (roundtrip.RoundTrip.case), so both systems send the same prompts."""
    from roundtrip import RoundTrip
    return RoundTrip.case(c["instruction"], c["buffer_before"], c.get("cursor", (1, 1)), uuid.uuid4().hex[:12])


def candidates(texts):
    """The distinct parsed samples with their counts (pooled.merge), in first-seen order."""
    return PL.merge([("4b",) + tuple(p) for p in map(parse_output, texts) if p])


def gate(cands, tau):
    """(ask, pick): fewer.view's pick, which is pooled.pick's, and V's question when its score is at most τ. No parsed
    sample is the lowest score, so V asks."""
    pick = PL.pick(cands)
    return (pick["score"] if pick else -math.inf) <= tau, pick


def answer_of(text):
    p = parse_output(text)
    return {"mode": p[0], "command": p[1]} if p else {"mode": None, "command": None}


class LiveV:
    """fewer.py's V as one request. `ep` renders every model turn (with examples in arm b); `pp.scores` is the
    verifier's batched call on the plain 4B endpoint."""

    def __init__(self, ep, pp, tau, k=K):
        self.ep, self.pp, self.tau, self.k = ep, pp, tau, k
        self.pool = ThreadPoolExecutor(max_workers=4)

    def translate(self, case):
        t0 = time.perf_counter()
        prompt = self.ep.render(messages_for_case(case))
        direct_f = self.pool.submit(self.ep.complete, prompt, Q.ANSWER_MAX_TOKENS)
        samples_f = self.pool.submit(self.ep.complete, prompt, Q.ANSWER_MAX_TOKENS, temperature=1.0, n=self.k)
        direct, samples = direct_f.result(), samples_f.result()
        t1 = time.perf_counter()
        cands = candidates(samples["texts"])
        scores, _ = self.pp.scores(case, cands)
        for c, s in zip(cands, scores):
            c["score"] = s
        t2 = time.perf_counter()
        ask, pick = gate(cands, self.tau)
        timings = {"direct_s": direct["latency_s"], "samples_s": samples["latency_s"],
                   "models_wall_s": round(t1 - t0, 4), "verifier_s": round(t2 - t1, 4)}
        out = {"ask": ask, "best": pick["score"] if pick else None,
               "cands": [{k: c[k] for k in ("mode", "command", "n", "score")} for c in cands], "timings": timings,
               "tokens": {"direct": direct["output_tokens"][0], "samples": sum(samples["output_tokens"])}}
        if ask:
            q = self.ep.complete(self.ep.render(turn1_messages(case), prefill=ASK_PREFIX), QUESTION_MAX_TOKENS,
                                 stop=["\n"])
            timings["question_s"] = q["latency_s"]
            out["tokens"]["question"] = q["output_tokens"][0]
            question = cut_question(q["texts"][0])
            if question:
                timings["total_s"] = round(time.perf_counter() - t0, 4)
                return dict(out, type="question", question=question)
            out["fallback"] = "empty question: answered as if it had not asked"
        timings["total_s"] = round(time.perf_counter() - t0, 4)
        got = {"mode": pick["mode"], "command": pick["command"]} if pick else answer_of(direct["texts"][0])
        return dict(out, type="command", **got)

    def reply(self, case, question, reply):
        r = self.ep.complete(self.ep.render(turn2_messages(case, question, reply)), Q.ANSWER_MAX_TOKENS)
        return dict(answer_of(r["texts"][0]), turn2_s=r["latency_s"], tokens=r["output_tokens"][0])


def gpu_s(t):
    """A request's GPU-seconds, one request at a time: the models' wall time, the verifier's, and the question and
    turn 2 where they ran. The sandbox is CPU (report_retr.request_gpu leaves it out too)."""
    return sum(t.get(k) or 0.0 for k in ("models_wall_s", "verifier_s", "question_s", "turn2_s"))


def v_row(v, c, replies):
    case = case_of(c)
    res = v.translate(case)
    row = {"id": c["id"], "system": "V", "type": res["type"], "ask": res["ask"], "best": res["best"],
           "n_cands": len(res["cands"]), "cands": res["cands"], "timings": res["timings"], "tokens": res["tokens"]}
    if res["type"] == "question":
        rec = replies.get(c["id"], {})
        row["question"] = res["question"]
        row["same_question"] = res["question"] == rec.get("question")
        done = v.reply(case, res["question"], rec.get("reply", PLACEBO_REPLY))
        row["timings"]["turn2_s"] = done.pop("turn2_s")
        row["tokens"]["turn2"] = done.pop("tokens")
        row["answer"] = done
    else:
        row["answer"] = {"mode": res["mode"], "command": res["command"]}
        if res.get("fallback"):
            row["fallback"] = res["fallback"]
    row["gpu_s"] = round(gpu_s(row["timings"]), 4)
    return row


def rule_row(rt, c, replies):
    res = rt.translate(c["instruction"], c["buffer_before"], c.get("cursor", (1, 1)))
    row = {"id": c["id"], "system": "rule", "type": res["type"], "ask": res["rule"]["ask"],
           "timings": dict(res["timings"]), "tokens": dict(res.get("tokens", {}))}
    if res["type"] == "question":
        rec = replies.get(c["id"], {})
        row["same_question"] = res["question"] == rec.get("question")
        done = rt.reply(res["dialogue_id"], rec.get("reply", PLACEBO_REPLY))
        if "timings" in done:
            row["timings"]["turn2_s"] = done["timings"]["turn2_s"]
            row["tokens"]["turn2"] = done.get("tokens", {}).get("turn2")
        row["answer"] = {"mode": done.get("mode"), "command": done.get("command")}
    else:
        row["answer"] = {"mode": res.get("mode"), "command": res.get("command")}
    row["gpu_s"] = round(gpu_s(row["timings"]), 4)
    return row


def tau_of(arm):
    return json.loads((S.ROOT / arm / "report.json").read_text())["fewer"]["tau"]


def endpoints(arm):
    """(the model endpoint for every turn, the verifier's Pooled) after sys4b points the stages at the 4B."""
    import endpoint as E
    S.use_4b(arm)
    ep = A.endpoint27()
    e4 = E.load(S.MODEL, S.URL_SERVED, S.NAME_SERVED)             # plain: the verifier never saw examples
    return ep, PL.Pooled(e4, E.load(S.MODEL, S.URL_SERVED, S.NAME_SERVED))


def summarize(rows, mem=None):
    from timing import stats
    asks = [r for r in rows if r["type"] == "question"]
    ans = [r for r in rows if r["type"] != "question"]
    out = {"n": len(rows), "asks": round(len(asks) / max(1, len(rows)), 4),
           "first_response_s": stats(r["timings"]["total_s"] for r in rows),
           "when_it_answers_s": stats(r["timings"]["total_s"] for r in ans),
           "when_it_asks_s": stats(r["timings"]["total_s"] for r in asks),
           "models_wall_s": stats(r["timings"]["models_wall_s"] for r in rows),
           "question_s": stats(r["timings"].get("question_s") for r in rows),
           "turn2_s": stats(r["timings"].get("turn2_s") for r in rows),
           "gpu_s": stats(r["gpu_s"] for r in rows),
           "tokens": round(statistics.fmean(sum(v or 0 for v in r["tokens"].values()) for r in rows), 1),
           "same_question": round(sum(r.get("same_question", False) for r in asks) / max(1, len(asks)), 3),
           "memory": mem}
    if rows and rows[0]["system"] == "V":
        n_c = sum(r["n_cands"] for r in rows)
        out.update(verifier_s=stats(r["timings"]["verifier_s"] for r in rows),
                   candidates=round(n_c / len(rows), 2),
                   verifier_s_per_candidate=round(sum(r["timings"]["verifier_s"] for r in rows) / n_c, 4) if n_c else None,
                   fallbacks=sum(bool(r.get("fallback")) for r in rows))
    else:
        out["sandbox_s"] = stats(r["timings"].get("sandbox_s") for r in rows)
    return out


def cmd_timing(args):
    import measure as M
    from roundtrip import RoundTrip
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    tau = tau_of(args.arm)
    ep, pp = endpoints(args.arm)
    v, rt = LiveV(ep, pp, tau), RoundTrip(ep, samples=K, max_agree=MAX_AGREE)
    rows = Q.timing_rows(A.cases("val"), TIMING_N)[:args.n]
    replies = {r["id"]: r for r in A.load_jsonl(S.ROOT / args.arm / "replies.jsonl")}
    for c in rows[:5]:                                               # warm-up, as ask27b's timing
        ep.complete(ep.render(messages_for_case(c)), Q.ANSWER_MAX_TOKENS)
    v_row(v, rows[0], replies)
    rule_row(rt, rows[0], replies)
    got = {"V": [], "rule": []}
    t0 = time.time()
    with (out / "v.jsonl").open("w") as fv, (out / "rule.jsonl").open("w") as fr:
        for i, c in enumerate(rows):
            if (args.out / "PAUSE").exists():
                print("paused after %d rows" % i, flush=True)
                return 4
            order = ("V", "rule") if i % 2 == 0 else ("rule", "V")
            for name in order:
                r = v_row(v, c, replies) if name == "V" else rule_row(rt, c, replies)
                r["order"] = order.index(name)
                got[name].append(r)
                (fv if name == "V" else fr).write(json.dumps(r, ensure_ascii=False) + "\n")
            fv.flush()
            fr.flush()
            if (i + 1) % 50 == 0:
                print("  %d / %d rows, %.0f s" % (i + 1, len(rows), time.time() - t0), flush=True)
    mem = M.gpu_mem()
    rep = {"measured": time.strftime("%Y-%m-%d %H:%M"), "arm": args.arm, "tau": tau, "rows": len(rows),
           "V": summarize(got["V"], mem), "rule": summarize(got["rule"], mem)}
    A.write_json(out / "timing.json", rep)
    print("arm %s: V first response p50 %.3f s, %.3f GPU-s a request, asks %.0f%%, %.3f s a candidate; "
          "the rule p50 %.3f s, %.3f GPU-s" % (
              args.arm, rep["V"]["first_response_s"]["p50"], rep["V"]["gpu_s"]["mean"], 100 * rep["V"]["asks"],
              rep["V"]["verifier_s_per_candidate"] or 0, rep["rule"]["first_response_s"]["p50"],
              rep["rule"]["gpu_s"]["mean"]), flush=True)
    return 0


# ------------------------------------------------------------ report (CPU)

def per_solved(gpu, acc_pct):
    """pool27b.per_solved: GPU-seconds per request over the accuracy, at GPU_HOUR."""
    s = gpu / (acc_pct / 100.0)
    return {"exec_match": acc_pct, "gpu_s_per_request": round(gpu, 3), "gpu_s_per_solved": round(s, 3),
            "usd_per_1000_solved": round(s * 1000 / 3600 * GPU_HOUR, 4)}


def stored_scores(arm):
    path = STORED_SCORES[arm]
    return {(r["id"], r["mode"], r["command"]): r["score"] for r in A.load_jsonl(path)} if path.exists() else {}


def rescore(rows, stored):
    diffs = [abs(c["score"] - stored[(r["id"], c["mode"], c["command"])]) for r in rows for c in r["cands"]
             if (r["id"], c["mode"], c["command"]) in stored]
    return {"pairs": len(diffs), "identical": sum(d == 0 for d in diffs),
            "mean_abs": round(statistics.fmean(diffs), 4) if diffs else None, "max_abs": max(diffs) if diffs else None}


def judged(rows, cases):
    """{id: matched} for the final answers through the box (evaluate.score), descriptive."""
    from evaluate import score
    preds = {r["id"]: r["answer"] for r in rows if r["answer"].get("command")}
    res = score([cases[r["id"]] for r in rows], preds)
    return {i: bool(v["match"]) for i, v in res["per_case"].items()}


def build(out):
    cases = {c["id"]: c for c in A.cases("val")}
    rep = {"arms": {}}
    for arm in ARMS:
        path = out / arm / "timing.json"
        if not path.exists():
            continue
        t = json.loads(path.read_text())
        s4 = json.loads((S.ROOT / arm / "report.json").read_text())
        old = json.loads((S.ROOT / arm / "timing.json").read_text())["summary"]["k8"]
        rows = {n: A.load_jsonl(out / arm / f) for n, f in (("V", "v.jsonl"), ("rule", "rule.jsonl"))}
        acc_v, acc_r = s4["fewer"]["arms"]["V"]["test"], s4["cost"]["round trip at 8"]["exec_match"]
        bm25_s = json.loads((S.ROOT / "b" / "bm25.json").read_text())["query_ms_p50"] / 1000 if arm == "b" else 0.0
        m = {n: judged(rows[n], cases) for n in rows}
        a = {"tau": t["tau"], "measured": t["measured"], "rows": t["rows"], "bm25_wait_s": bm25_s,
             "V": t["V"], "rule": t["rule"],
             "cost": {"V, measured": per_solved(t["V"]["gpu_s"]["mean"], acc_v),
                      "V, composed (sys4b)": s4["cost"]["V on one card (composed)"],
                      "the rule at 8, measured now": per_solved(t["rule"]["gpu_s"]["mean"], acc_r),
                      "the rule at 8 (sys4b's timing pass)": s4["cost"]["round trip at 8"]},
             "wait_p50_s": {"V": round(t["V"]["first_response_s"]["p50"] + bm25_s, 4),
                            "rule": round(t["rule"]["first_response_s"]["p50"] + bm25_s, 4),
                            "rule, sys4b's pass": round(old["first_response_s"]["p50"] + bm25_s, 4)},
             "wait_p95_s": {"V": round(t["V"]["first_response_s"]["p95"] + bm25_s, 4),
                            "rule": round(t["rule"]["first_response_s"]["p95"] + bm25_s, 4)},
             "rule_then": {"asks": old["asks"], "models_wall_mean_s": old["models_wall_s"]["mean"],
                           "first_p50_s": old["first_response_s"]["p50"]},
             "v_asks_offline": {"test": s4["fewer"]["arms"]["V"]["asks_test"], "choosing": s4["fewer"]["arms"]["V"]["asks_choosing"]},
             "rescore": rescore(rows["V"], stored_scores(arm)),
             "answers_on_these_rows": {n: {"matched": sum(v.values()), "n": len(v)} for n, v in m.items()}}
        rep["arms"][arm] = a
    return rep


def render(rep):
    L = ["# V timed as a system on the served 4B", "",
         "By `plan.md` beside this file; built by `scripts/vtime4b/vtime4b.py report`. 200 timing rows, one request at a time, "
         "V and the 5-of-8 round trip at 8 interleaved row by row.", ""]
    for arm, a in rep["arms"].items():
        v, r = a["V"], a["rule"]
        L += ["## Arm %s (%s), τ %s, measured %s" % (arm, "no examples" if arm == "a" else "8 BM25 examples", a["tau"], a["measured"]),
              "", "| system | per 1,000 solved | GPU-s a request | wait p50 / p95 | asks | tokens |", "|---|---|---|---|---|---|"]
        c = a["cost"]
        L.append("| **V, measured** | **$%.4f** | %.3f | %.2f / %.2f s | %.1f%% | %.1f |" % (
            c["V, measured"]["usd_per_1000_solved"], v["gpu_s"]["mean"], a["wait_p50_s"]["V"], a["wait_p95_s"]["V"],
            100 * v["asks"], v["tokens"]))
        L.append("| V, composed (sys4b) | $%.4f | %.3f | — | %.1f%% (test) | — |" % (
            c["V, composed (sys4b)"]["usd_per_1000_solved"], c["V, composed (sys4b)"]["gpu_s_per_request"], a["v_asks_offline"]["test"]))
        L.append("| the rule at 8, measured now | $%.4f | %.3f | %.2f / %.2f s | %.1f%% | %.1f |" % (
            c["the rule at 8, measured now"]["usd_per_1000_solved"], r["gpu_s"]["mean"], a["wait_p50_s"]["rule"],
            a["wait_p95_s"]["rule"], 100 * r["asks"], r["tokens"]))
        L.append("| the rule at 8, sys4b's pass | $%.4f | %.3f | %.2f s p50 | %.1f%% | — |" % (
            c["the rule at 8 (sys4b's timing pass)"]["usd_per_1000_solved"],
            c["the rule at 8 (sys4b's timing pass)"]["gpu_s_per_request"], a["wait_p50_s"]["rule, sys4b's pass"],
            100 * a["rule_then"]["asks"]))
        L += ["", "- V's parts: models %.3f s (mean), verifier %.3f s for %.2f candidates (%.4f s a candidate; composed: %.4f), "
              "question %s, turn 2 %s." % (
                  v["models_wall_s"]["mean"], v["verifier_s"]["mean"], v["candidates"], v["verifier_s_per_candidate"] or 0,
                  c["V, composed (sys4b)"].get("verifier_s_per_candidate") or 0,
                  "%.3f s" % v["question_s"]["mean"] if v["question_s"] else "—",
                  "%.3f s" % v["turn2_s"]["mean"] if v["turn2_s"] else "—"),
              "- The verifier's live scores against its stored ones: %d pairs, %d identical, mean |diff| %s, max %s." % (
                  a["rescore"]["pairs"], a["rescore"]["identical"], a["rescore"]["mean_abs"], a["rescore"]["max_abs"]),
              "- Answers on these 200 val rows through the box (descriptive; recorded replies): V %d, the rule %d." % (
                  a["answers_on_these_rows"]["V"]["matched"], a["answers_on_these_rows"]["rule"]["matched"]),
              "- Empty questions answered as if not asked: %d. Live questions equal to the recorded ones: V %.0f%%, the rule %.0f%%." % (
                  v.get("fallbacks", 0), 100 * v["same_question"], 100 * r["same_question"]), ""]
        if a["bm25_wait_s"]:
            L += ["- BM25 (CPU, %.4f s) is added to every wait, not to the GPU." % a["bm25_wait_s"], ""]
    return "\n".join(L) + "\n"


def cmd_report(args):
    rep = build(args.out)
    A.write_json(args.out / "report.json", rep)
    text = render(rep)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("timing", "report"))
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--out", default=str(RUN))
    ap.add_argument("--n", type=int, default=TIMING_N, help="the first n of the 200 timing rows (a smoke)")
    args = ap.parse_args(argv)
    args.out = pathlib.Path(args.out)
    if args.stage == "timing":
        if not args.arm:
            ap.error("timing needs --arm")
        return cmd_timing(args)
    return cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
