#!/usr/bin/env python3
"""What the question round trip costs in time, against the §6 answer alone.

One user at a time (concurrency 1), through the served endpoint, on a
seeded sample of val rows:

    direct       the §6 answer alone: the served row today
    k = 2, 4, 8  the round trip (serve/roundtrip.py). The §6 answer and k
                 samples are requested together, the sandbox checks the
                 distinct samples (the validation path), and where the rule
                 asks (the top result holds at most floor(5/8 k) of k), the
                 question is generated. Measured: the time to the first
                 response, and each part.
    turn 2       after a question, the answer to the user's reply (the
                 stored 2026-10-06 simulated reply to the row's question)

Warm-up requests run first. Reports the mean, p50 and p95 of each step,
the share of requests that ask, and the tokens said per request.

    python scripts/serve/timing.py --n 200 --out runs/serve/timing_val.json
"""
import argparse
import json
import math
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                            # noqa: E402
from prompt import messages_for_case                       # noqa: E402

KS = (2, 4, 8)


def stats(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    q = lambda f: xs[min(len(xs) - 1, int(round(f * (len(xs) - 1))))]      # noqa: E731
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 4), "p50": q(0.5), "p95": q(0.95)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "val.jsonl"))
    ap.add_argument("--ref", default=str(REPO / "runs" / "clarify_2026-10-06" / "val" / "verdicts_served.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import endpoint
    from roundtrip import RoundTrip
    ep = endpoint.load(args.model, args.endpoint)
    cases = load_jsonl(args.cases)
    cases = random.Random(args.seed).sample(cases, min(args.n, len(cases)))
    ref = {r["id"]: r for r in load_jsonl(args.ref)}

    for c in cases[:5]:                                                   # warm-up
        ep.complete(ep.render(messages_for_case(c)), 192)
    RoundTrip(ep, samples=8, settings_path=args.settings).translate(
        cases[0]["instruction"], cases[0]["buffer_before"], cases[0].get("cursor", (1, 1)))

    rows = {"direct": [], **{"k%d" % k: [] for k in KS}}
    for c in cases:
        t0 = time.perf_counter()
        r = ep.complete(ep.render(messages_for_case(c)), 192)
        rows["direct"].append({"id": c["id"], "total_s": round(time.perf_counter() - t0, 4),
                               "model_s": r["latency_s"], "tokens": r["output_tokens"][0]})
    for k in KS:
        rt = RoundTrip(ep, samples=k, max_agree=math.floor(5 / 8 * k + 1e-9),
                       settings_path=args.settings)
        for c in cases:
            out = rt.translate(c["instruction"], c["buffer_before"], c.get("cursor", (1, 1)))
            row = {"id": c["id"], "type": out["type"], **out["timings"],
                   "tokens": sum(out.get("tokens", {}).values())}
            if out["type"] == "question":
                done = rt.reply(out["dialogue_id"], ref[c["id"]]["reply"])
                row["turn2_s"] = done["timings"]["turn2_s"]
                row["tokens"] += done["tokens"]["turn2"]
                row["same_question_as_evaluated"] = out["question"] == ref[c["id"]]["question"]
            rows["k%d" % k].append(row)

    summary = {"n": len(cases), "direct": {"total_s": stats(r["total_s"] for r in rows["direct"]),
                                           "tokens": statistics.fmean(r["tokens"] for r in rows["direct"])}}
    for k in KS:
        rr = rows["k%d" % k]
        summary["k%d" % k] = {
            "asks": sum(r["type"] == "question" for r in rr) / len(rr),
            "first_response_s": stats(r["total_s"] for r in rr),
            "models_wall_s": stats(r["models_wall_s"] for r in rr),
            "samples_s": stats(r["samples_s"] for r in rr),
            "sandbox_s": stats(r["sandbox_s"] for r in rr),
            "question_s": stats(r.get("question_s") for r in rr),
            "turn2_s": stats(r.get("turn2_s") for r in rr),
            "tokens": statistics.fmean(r["tokens"] for r in rr),
            "same_question_as_evaluated": (sum(r.get("same_question_as_evaluated", False) for r in rr)
                                           / max(1, sum(r["type"] == "question" for r in rr)))}
    pathlib.Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    d = summary["direct"]["total_s"]
    print("timing on %d val rows, one request at a time" % len(cases))
    print("  direct (§6 answer)   p50 %.2f s  p95 %.2f s  tokens %.1f" % (
        d["p50"], d["p95"], summary["direct"]["tokens"]))
    for k in KS:
        s = summary["k%d" % k]
        print("  k = %d round trip    first response p50 %.2f s  p95 %.2f s | sandbox p50 %.2f s | "
              "asks %.0f%% | turn 2 p50 %s | tokens %.1f" % (
                  k, s["first_response_s"]["p50"], s["first_response_s"]["p95"],
                  s["sandbox_s"]["p50"], 100 * s["asks"],
                  "%.2f s" % s["turn2_s"]["p50"] if s["turn2_s"] else "—", s["tokens"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
