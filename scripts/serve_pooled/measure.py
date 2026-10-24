#!/usr/bin/env python3
"""The serving cost of the pooled pick on one RTX 3090 (both endpoints up; serve_pooled.sh).

  accuracy    every val row through the served pooled pick (concurrency 8), scored by
              the harness; the same requests also give the 4B-only and 9B-only picks
              (scores are per command, so a single model's pick is the best of its own
              candidates). Compared with the offline numbers (val: pooled 59.6%).
  latency     one request at a time on the first N val rows: the 4B's §6 answer (greedy),
              the 4B's 8 samples + the verifier, and the pooled pick; p50 / p95.
  throughput  requests/s, p50 / p95 at concurrency 1 / 8 / 32 for the §6 answer and the
              pooled pick; GPU memory in use.
  cost        GPU-seconds per request = 1 / requests/s at the best concurrency (the
              deployment holds the whole card); per solved task = per request / exec-match;
              dollars at --gpu-hour (stated in the report, never assumed silently).

    python scripts/serve_pooled/measure.py --out runs/serve_pooled
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", HERE):
    sys.path.insert(0, str(p))


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, max(0, round(q / 100 * (len(v) - 1))))] if v else None


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return {"used_mib": int(out[0]), "total_mib": int(out[1])}
    except Exception:
        return None


def single_pick(cands, model):
    own = [c for c in cands if model in c["from"]]
    return max(own, key=lambda c: (c["score"], c["from"][model], c["mode"], c["command"])) if own else None


def run_concurrent(fn, cases, conc):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(fn, cases))
    wall = time.perf_counter() - t0
    lat = [r["latency_s"] for r in res]
    return res, {"concurrency": conc, "requests": len(cases), "wall_s": round(wall, 2),
                 "req_per_s": round(len(cases) / wall, 3), "p50_s": round(pct(lat, 50), 3), "p95_s": round(pct(lat, 95), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "serve_pooled"))
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--throughput-n", type=int, default=96)
    ap.add_argument("--gpu-hour", type=float, default=0.30, help="USD per GPU-hour used for the dollar column")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    import endpoint as E
    import pooled as P
    from evaluate import score
    from prompt import messages_for_case, parse_output
    e4 = E.load("qwen3.5-4b", "http://127.0.0.1:8011", "qwen3.5-4b-served")
    e9 = E.load("qwen3.5-9b", "http://127.0.0.1:8012", "served9b")
    pp = P.Pooled(e4, e9)
    pp.pool = ThreadPoolExecutor(max_workers=128)
    val = load_jsonl(REPO / "data" / "splits" / "val.jsonl")
    report = {"gpu_hour_usd": args.gpu_hour, "memory_idle": gpu_mem()}

    def greedy(case):
        t0 = time.perf_counter()
        r = e4.complete(e4.render(messages_for_case(case)), max_tokens=192)
        p = parse_output(r["texts"][0])
        return {"id": case["id"], "mode": p[0] if p else None, "command": p[1] if p else None,
                "latency_s": time.perf_counter() - t0}

    # accuracy: every val row, concurrency 8
    if not args.skip_accuracy:
        res, thr = run_concurrent(pp.answer, val, 8)
        preds = {"pooled": {}, "4b": {}, "9b": {}}
        for r in res:
            if r["command"] is not None:
                preds["pooled"][r["id"]] = {"command": r["command"], "mode": r["mode"]}
            for m in ("4b", "9b"):
                c = single_pick(r["candidates"], m)
                if c:
                    preds[m][r["id"]] = {"command": c["command"], "mode": c["mode"]}
        acc = {k: score(val, v)["n_match"] / len(val) for k, v in preds.items()}
        report["accuracy_val"] = {k: round(100 * v, 1) for k, v in acc.items()}
        report["accuracy_run"] = thr
        with (out / "served_val.jsonl").open("w") as fh:
            for r in res:
                fh.write(json.dumps({k: r[k] for k in ("id", "mode", "command", "latency_s", "sample_s", "score_s",
                                                       "output_tokens")}
                                    | {"candidates": [{"mode": c["mode"], "command": c["command"], "n": c["n"],
                                                       "from": c["from"], "score": c["score"]} for c in r["candidates"]]}) + "\n")
        print("accuracy on val:", report["accuracy_val"], thr, flush=True)
    elif (out / "served_val.jsonl").exists():
        # the accuracy pass of an earlier launch: re-score its saved picks (the 9B engine crashed
        # after it on 2026-10-19, so the timing phases were rerun on their own)
        saved = load_jsonl(out / "served_val.jsonl")
        preds = {"pooled": {}, "4b": {}, "9b": {}}
        for r in saved:
            if r["command"] is not None:
                preds["pooled"][r["id"]] = {"command": r["command"], "mode": r["mode"]}
            for m in ("4b", "9b"):
                c = single_pick(r["candidates"], m)
                if c:
                    preds[m][r["id"]] = {"command": c["command"], "mode": c["mode"]}
        report["accuracy_val"] = {k: round(100 * score(val, v)["n_match"] / len(val), 1) for k, v in preds.items()}
        lat8 = [r["latency_s"] for r in saved]
        report["accuracy_run"] = {"concurrency": 8, "requests": len(saved), "p50_s": round(pct(lat8, 50), 3),
                                  "p95_s": round(pct(lat8, 95), 3), "note": "from the saved accuracy pass"}
        print("accuracy on val (saved pass):", report["accuracy_val"], flush=True)

    # latency, one at a time
    cases = val[:args.latency_n]
    lat = {}
    for name, fn in (("§6 answer (4B greedy)", greedy), ("4B 8 samples + verifier", lambda c: pp.answer(c, models=("4b",))),
                     ("pooled pick (4B + 9B, 16 samples + verifier)", pp.answer)):
        res, thr = run_concurrent(fn, cases, 1)
        row = {"p50_s": thr["p50_s"], "p95_s": thr["p95_s"]}
        if "sample_s" in res[0]:
            row["sample_p50_s"] = round(pct([r["sample_s"] for r in res], 50), 3)
            row["score_p50_s"] = round(pct([r["score_s"] for r in res], 50), 3)
            row["candidates_mean"] = round(statistics.fmean(len(r["candidates"]) for r in res), 2)
            row["output_tokens_mean"] = round(statistics.fmean(sum(r["output_tokens"].values()) for r in res), 1)
        lat[name] = row
        print("latency", name, row, flush=True)
    report["latency_one_at_a_time"] = lat

    # throughput at 1 / 8 / 32
    tp = {}
    for name, fn in (("§6 answer (4B greedy)", greedy), ("pooled pick", pp.answer)):
        tp[name] = []
        for conc in (1, 8, 32):
            n = min(args.throughput_n, len(val)) if conc > 1 else 32
            _, thr = run_concurrent(fn, val[:n], conc)
            thr["memory"] = gpu_mem()
            tp[name].append(thr)
            print("throughput", name, thr, flush=True)
    report["throughput"] = tp

    # cost: GPU-seconds per request at the best measured throughput; per solved task by val accuracy
    cost = {}
    acc_map = report.get("accuracy_val", {})
    for name, key in (("§6 answer (4B greedy)", None), ("pooled pick", "pooled")):
        best = max(tp[name], key=lambda r: r["req_per_s"])
        gpu_s = 1.0 / best["req_per_s"]
        acc = (acc_map.get(key) if key else 51.9) / 100           # 51.9%: the 4B §6 answer on val (recorded)
        cost[name] = {"best_concurrency": best["concurrency"], "gpu_s_per_request": round(gpu_s, 3),
                      "val_exec_match": round(100 * acc, 1), "gpu_s_per_solved": round(gpu_s / acc, 3),
                      "usd_per_1000_solved": round(gpu_s / acc * 1000 / 3600 * args.gpu_hour, 4)}
    report["cost"] = cost
    (out / "measure.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["cost"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
