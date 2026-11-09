#!/usr/bin/env python3
"""The trained 27B through vLLM, then in the pool with the 4B, the 9B and the verifier.

The rules are in runs/serve27b_2026-11-05/preregistration.md. One stage per call, each resumable;
run.sh runs them in order. The 27B fills the card (serve27b.sh), so it cannot sit beside the 4B
endpoint that carries the verifier: its answers are drawn first, and scored when the card has been
handed back to the pool's endpoints (serve_pooled.sh). Nothing here touches a served system.

  pool-timing  4B + 9B endpoints: the 4B's §6 answer and the pooled pick at 4 + 4, one request at a
               time, the same night as the 27B's timing
  smoke        27B endpoint: greedy answers against the recorded transformers ones; exit 3 below
               the gate, because then this endpoint is not the model that was measured
  timing       27B endpoint: greedy one request at a time, throughput at 1 / 8 / 32 concurrent, and
               k samples for every k the report uses
  greedy       27B endpoint: every val and test row, as served
  samples      27B endpoint: N samples per row at T = 1, top-p 1 (pooled.Pooled's settings), rows in
               a fixed mixed order so that a deadline leaves both splits sampled alike
  score        4B + 9B endpoints: the verifier on every distinct 27B answer the served pass has not
               already scored, and a sample of the old scores again as a check
  judge        CPU: every distinct answer of all three models through the box as it stands today,
               and the 27B's greedy answers (served and recorded) through the harness

    python scripts/serve27b/pass27b.py smoke --run runs/serve27b_2026-11-05
"""
import argparse
import json
import os
import pathlib
import random
import statistics
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "reranker",
          REPO / "scripts" / "ceiling", REPO / "scripts" / "serve_pooled", HERE):
    sys.path.insert(0, str(p))

SERVED = REPO / "runs" / "serve_pooled" / "served_at_k"       # the 4B's and 9B's 128 samples per row
RECORDED = REPO / "runs" / "qwen3.5-27b-sft-distill"          # the trained 27B through transformers
MODEL27, NAME27, URL27 = "qwen3.5-27b", "served27b", "http://127.0.0.1:8015"
URL4, URL9 = "http://127.0.0.1:8011", "http://127.0.0.1:8012"
SPLITS = ("val", "test")
MAX_TOKENS = 192                                              # pooled.Pooled's and measure.py's cap
TIMED_KS = (1, 2, 4, 8, 16, 32)
SMOKE_GATE = 0.80


def load_jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def cases(split, limit=0):
    rows = load_jsonl(REPO / "data" / "splits" / ("%s.jsonl" % split))
    return rows[:limit] if limit else rows


def mix(ids):
    """A fixed order that does not depend on the split: a prefix holds both splits alike."""
    return sorted(ids, key=lambda i: (zlib.crc32(i.encode("utf-8")), i))


def same_answer(parsed, want):
    """A parsed (mode, command) against a recorded prediction row."""
    return parsed is not None and tuple(parsed) == (want.get("mode"), want["command"])


def pace_fell(seconds, warmup=25, window=10, factor=3.0):
    """Has the pace of the last `window` rows fallen `factor` times below the first `warmup`'s? WSL2
    pages GPU memory at about a tenth of the speed once something on the desktop wants the card
    back, with no error anywhere, and the card is shared."""
    if len(seconds) < warmup + window:
        return False
    return statistics.median(seconds[-window:]) > factor * statistics.median(seconds[:warmup])


def distinct_answers(greedy_row, sample_row):
    """A row's distinct 27B answers as [(mode, command)], greedy first; unparsed ones are dropped."""
    seen, out = set(), []
    samples = (greedy_row["samples"] if greedy_row else []) + (sample_row["samples"] if sample_row else [])
    for s in samples:
        if s is not None and tuple(s) not in seen:
            seen.add(tuple(s))
            out.append(tuple(s))
    return out


def old_rows(split):
    """{id: row} of the served pass (first occurrence wins, as served_at_k.report reads them)."""
    rows = {}
    for r in load_jsonl(SERVED / ("rows_%s.jsonl" % split)):
        rows.setdefault(r["id"], r)
    return rows


def to_score(wanted, known):
    """The answers of `wanted` whose score is not in `known`, in order."""
    return [k for k in wanted if k not in known]


def endpoint27():
    import endpoint as E
    e = E.load(MODEL27, URL27, NAME27)
    e.timeout = 900.0
    return e


def pool_endpoints(k=4):
    import endpoint as E
    import pooled as P
    e4 = E.load("qwen3.5-4b", URL4, "qwen3.5-4b-served")
    e9 = E.load("qwen3.5-9b", URL9, "served9b")
    return e4, P.Pooled(e4, e9, k=k)


def request(e, case, n, temperature):
    """One completion request: n answers for a case, parsed by the §6 grammar."""
    from prompt import messages_for_case, parse_output
    r = e.complete(e.render(messages_for_case(case)), max_tokens=MAX_TOKENS, temperature=temperature, n=n)
    return {"id": case["id"], "samples": [list(p) if p else None for p in map(parse_output, r["texts"])],
            "texts": r["texts"], "output_tokens": r["output_tokens"], "prompt_tokens": r["prompt_tokens"],
            "latency_s": r["latency_s"]}


def summary(res, tokens=lambda r: sum(r["output_tokens"])):
    lat = sorted(r["latency_s"] for r in res)
    return {"rows": len(res), "mean_s": round(statistics.fmean(lat), 4), "p50_s": round(statistics.median(lat), 4),
            "p95_s": round(lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))], 4),
            "tokens_mean": round(statistics.fmean(tokens(r) for r in res), 1)}


def greedy_timing(fn, val, latency_n, throughput_n):
    """(one at a time, throughput at 1 / 8 / 32): measure.py's protocol."""
    import measure as M
    res, _ = M.run_concurrent(fn, val[:latency_n], 1)
    tp = []
    for conc in (1, 8, 32):
        n = 32 if conc == 1 else min(throughput_n, len(val))
        _, thr = M.run_concurrent(fn, val[:n], conc)
        thr["memory"] = M.gpu_mem()
        tp.append(thr)
        print("throughput at %d:" % conc, thr, flush=True)
    return summary(res), tp


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


# ------------------------------------------------------------ stages

def cmd_pool_timing(args):
    """The two systems the 27B is priced against, measured the same night on the same desktop."""
    import measure as M
    e4, pp = pool_endpoints(k=4)
    val = cases("val", args.limit)
    rep = {"measured": time.strftime("%Y-%m-%d %H:%M"), "memory_idle": M.gpu_mem()}
    rep["greedy4b"], rep["throughput4b"] = greedy_timing(lambda c: request(e4, c, 1, 0.0), val, args.latency_n, args.throughput_n)
    print("4B greedy:", rep["greedy4b"], flush=True)
    res = [pp.answer(c) for c in val[:args.pool_n]]
    rep["pooled44"] = dict(summary(res, tokens=lambda r: sum(r["output_tokens"].values())),
                           sample_p50_s=round(statistics.median(r["sample_s"] for r in res), 4),
                           score_p50_s=round(statistics.median(r["score_s"] for r in res), 4),
                           score_mean_s=round(statistics.fmean(r["score_s"] for r in res), 4),
                           candidates_mean=round(statistics.fmean(len(r["candidates"]) for r in res), 2))
    print("pooled 4 + 4:", rep["pooled44"], flush=True)
    write_json(args.run / "pool_timing.json", rep)
    return 0


def cmd_smoke(args):
    e = endpoint27()
    rec = {r["id"]: r for r in load_jsonl(RECORDED / "preds_val.jsonl")}
    rows = cases("val", min(args.n, args.limit) if args.limit else args.n)
    res = [request(e, c, 1, 0.0) for c in rows]
    same = sum(same_answer(r["samples"][0], rec[r["id"]]) for r in res)
    rep = {"rows": len(res), "identical": same, "gate": SMOKE_GATE, **summary(res)}
    write_json(args.run / "smoke.json", rep)
    print(rep, flush=True)
    return 0 if same >= SMOKE_GATE * len(res) else 3


def cmd_timing(args):
    import measure as M
    e = endpoint27()
    val = cases("val", args.limit)
    path = args.run / "timing.json"
    rep = json.loads(path.read_text()) if path.exists() else {}
    rep.update(measured=time.strftime("%Y-%m-%d %H:%M"), memory_idle=M.gpu_mem())
    rep["greedy"], rep["throughput"] = greedy_timing(lambda c: request(e, c, 1, 0.0), val, args.latency_n, args.throughput_n)
    print("27B greedy:", rep["greedy"], flush=True)
    write_json(path, rep)
    rep.setdefault("samples", {})
    for k in TIMED_KS:
        res = [request(e, c, k, 1.0) for c in val[:args.timing_n]]
        rep["samples"][str(k)] = summary(res)
        print("27B %d samples:" % k, rep["samples"][str(k)], flush=True)
        write_json(path, rep)
    return 0


def cmd_greedy(args):
    e = endpoint27()
    for sp in SPLITS:
        path = args.run / ("greedy_%s.jsonl" % sp)
        done = {r["id"] for r in load_jsonl(path)} if path.exists() else set()
        todo = [c for c in cases(sp, args.limit) if c["id"] not in done]
        print("%s: %d rows to do, %d done" % (sp, len(todo), len(done)), flush=True)
        with path.open("a") as fh, ThreadPoolExecutor(max_workers=8) as ex:
            for r in ex.map(lambda c: request(e, c, 1, 0.0), todo):
                fh.write(json.dumps(dict(r, split=sp)) + "\n")
                fh.flush()
    return 0


def cmd_samples(args):
    e = endpoint27()
    path = args.run / "samples.jsonl"
    done = {r["id"]: r for r in load_jsonl(path)} if path.exists() else {}
    wrong = [i for i, r in done.items() if len(r["samples"]) != args.n]
    if wrong:
        sys.exit("%d rows on disk do not hold %d samples (first: %s)" % (len(wrong), args.n, wrong[0]))
    by_id = {c["id"]: (sp, c) for sp in SPLITS for c in cases(sp, args.limit)}
    todo = [i for i in mix(by_id) if i not in done]
    print("%d rows to do, %d done, %d samples per row" % (len(todo), len(done), args.n), flush=True)
    stop = args.run / "PAUSE"
    deadline = time.time() + args.budget_s
    slow = threading.Event()

    def one(i):
        if stop.exists() or slow.is_set() or time.time() > deadline:
            return None
        sp, c = by_id[i]
        t0 = time.perf_counter()
        r = request(e, c, args.n, 1.0)
        return {k: r[k] for k in ("id", "samples", "output_tokens", "prompt_tokens")} | {
            "split": sp, "sample_s": round(time.perf_counter() - t0, 3)}

    t0, n_done, seconds = time.time(), 0, []
    with path.open("a") as fh, ThreadPoolExecutor(max_workers=args.inflight) as ex:
        for r in ex.map(one, todo):
            if r is None:
                continue
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            n_done += 1
            seconds.append(r["sample_s"])
            if pace_fell(seconds):
                slow.set()
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.time() - t0
                print("  %d / %d rows, %.0f s, %.1f s per row, %.0f min to go" % (
                    n_done, len(todo), el, el / n_done, el / n_done * (len(todo) - n_done) / 60), flush=True)
    with (args.run / "passes.jsonl").open("a") as fh:
        fh.write(json.dumps({"rows": n_done, "wall_s": round(time.time() - t0, 1), "inflight": args.inflight,
                             "n": args.n, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
    left = len(todo) - n_done
    print("%d rows sampled, %d left" % (n_done, left), flush=True)
    if left and stop.exists():
        return 4                                     # paused, not finished
    if left and slow.is_set():
        print("the pace fell to %.1f s per row from %.1f; yielding the card" % (
            statistics.median(seconds[-10:]), statistics.median(seconds[:25])), flush=True)
        return 5                                     # something else wants the card
    return 0                                         # a deadline goes on with the rows it has


def answers27(run, split, limit=0):
    """{id: [(mode, command)]}: the 27B's distinct answers on a split's rows."""
    greedy = {r["id"]: r for r in load_jsonl(run / ("greedy_%s.jsonl" % split))}
    samples = {r["id"]: r for r in load_jsonl(run / "samples.jsonl") if r["split"] == split}
    return {c["id"]: distinct_answers(greedy.get(c["id"]), samples.get(c["id"])) for c in cases(split, limit)}


def cmd_score(args):
    _, pp = pool_endpoints()
    path = args.run / "scores27.jsonl"
    known = {(r["id"], r["mode"], r["command"]): r["score"] for r in load_jsonl(path)} if path.exists() else {}
    rng = random.Random("rescore")
    picked = []
    with path.open("a") as fh:
        for sp in SPLITS:
            old = old_rows(sp)
            answers = answers27(args.run, sp, args.limit)
            t0, n = time.time(), 0
            for row, c in enumerate(cases(sp, args.limit)):
                i = c["id"]
                have = {(x["mode"], x["command"]) for x in old[i]["cands"]}
                want = [k for k in to_score(answers[i], have) if (i,) + k not in known]
                for s in range(0, len(want), 64):
                    chunk = [{"mode": m, "command": cmd} for m, cmd in want[s:s + 64]]
                    sc, _ = pp.scores(c, chunk)
                    for x, v in zip(chunk, sc):
                        known[(i, x["mode"], x["command"])] = v
                        fh.write(json.dumps({"id": i, "split": sp, "mode": x["mode"], "command": x["command"], "score": v}) + "\n")
                    fh.flush()
                n += len(want)
                picked += [(c, x) for x in rng.sample(old[i]["cands"], min(1, len(old[i]["cands"])))]
                if (row + 1) % 100 == 0:
                    print("  %s: %d rows, %d new answers scored, %.0f s" % (sp, row + 1, n, time.time() - t0), flush=True)
            print("%s: %d new answers scored in %.0f s" % (sp, n, time.time() - t0), flush=True)
    # the old scores again: the same verifier on the same endpoint settings, two weeks later
    check = rng.sample(picked, min(args.rescore_n, len(picked)))
    diffs = []
    for c, x in check:
        sc, _ = pp.scores(c, [x])
        diffs.append(abs(sc[0] - x["score"]))
    write_json(args.run / "rescore.json", {"rows": len(diffs), "identical": sum(d == 0 for d in diffs),
                                           "mean_abs": round(statistics.fmean(diffs), 4) if diffs else None,
                                           "max_abs": round(max(diffs), 4) if diffs else None})
    print("rescored %d old answers: mean |diff| %.3f, max %.3f" % (len(diffs), statistics.fmean(diffs) if diffs else 0,
                                                                   max(diffs) if diffs else 0), flush=True)
    return 0


def cmd_judge(args):
    """Every answer of all three models by one oracle: the box of 2026-11-04, on vim.basic."""
    import sample as S
    from evaluate import score, write_exec_log
    path = args.run / "judged.jsonl"
    done = {(r["id"], r["mode"], r["command"]) for r in load_jsonl(path)} if path.exists() else set()
    with path.open("a") as fh:
        for sp in SPLITS:
            old = old_rows(sp)
            answers = answers27(args.run, sp, args.limit)
            todo = []
            for c in cases(sp, args.limit):
                keys = list(dict.fromkeys(answers[c["id"]] + [(x["mode"], x["command"]) for x in old[c["id"]]["cands"]]))
                todo += [(c, k) for k in keys if (c["id"],) + k not in done]
            print("%s: %d answers to judge" % (sp, len(todo)), flush=True)
            t0 = time.time()

            def judge(item):
                c, (mode, command) = item
                return S.execute(c, mode, command, args.timeout)[0]

            with ThreadPoolExecutor(max_workers=8) as ex:
                for n, ((c, (mode, command)), state) in enumerate(zip(todo, ex.map(judge, todo)), 1):
                    fh.write(json.dumps({"id": c["id"], "split": sp, "mode": mode, "command": command, "state": state}) + "\n")
                    if n % 5000 == 0:
                        fh.flush()
                        print("  %s: %d / %d, %.0f s" % (sp, n, len(todo), time.time() - t0), flush=True)
            fh.flush()
            print("%s: judged in %.0f s" % (sp, time.time() - t0), flush=True)
    for sp in SPLITS:
        rows = cases(sp, args.limit)
        ids = {c["id"] for c in rows}
        served = {r["id"]: r for r in load_jsonl(args.run / ("greedy_%s.jsonl" % sp))}
        preds = {i: {"command": r["samples"][0][1], "mode": r["samples"][0][0]} for i, r in served.items()
                 if i in ids and r["samples"][0] is not None}
        with (args.run / ("preds_%s.jsonl" % sp)).open("w") as fh:
            for i, p in preds.items():
                fh.write(json.dumps({"id": i, **p}) + "\n")
        write_exec_log(str(args.run / ("exec_%s.jsonl" % sp)), rows, score(rows, preds, timeout=args.timeout))
        rec = {r["id"]: {"command": r["command"], "mode": r.get("mode")}
               for r in load_jsonl(RECORDED / ("preds_%s.jsonl" % sp)) if r["id"] in ids}
        write_exec_log(str(args.run / ("recorded_exec_%s.jsonl" % sp)), rows, score(rows, rec, timeout=args.timeout))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("pool-timing", "smoke", "timing", "greedy", "samples", "score", "judge"))
    ap.add_argument("--run", default=str(REPO / "runs" / "serve27b_2026-11-05"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows of each split (a dry run)")
    ap.add_argument("--n", type=int, default=None, help="samples per row; rows in the smoke")
    ap.add_argument("--inflight", type=int, default=2, help="rows in flight in the samples stage")
    ap.add_argument("--budget-s", type=float, default=18000.0, help="no new row starts this long after the samples stage does")
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--throughput-n", type=int, default=96)
    ap.add_argument("--timing-n", type=int, default=30)
    ap.add_argument("--pool-n", type=int, default=100)
    ap.add_argument("--rescore-n", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=5.0, help="executor timeout per command, seconds (§9)")
    args = ap.parse_args()
    args.run = pathlib.Path(args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    if args.n is None:
        args.n = 40 if args.stage == "smoke" else 32
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return {"pool-timing": cmd_pool_timing, "smoke": cmd_smoke, "timing": cmd_timing, "greedy": cmd_greedy,
            "samples": cmd_samples, "score": cmd_score, "judge": cmd_judge}[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
