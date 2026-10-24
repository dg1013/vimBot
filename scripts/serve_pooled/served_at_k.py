#!/usr/bin/env python3
"""The served pooled pick at 1, 4, 8, 32, 64 and 128 samples per model, on whole val / test splits.

One pass per row through the two endpoints (serve_pooled.sh), as pooled.Pooled.answer
serves a request but with 128 samples per model:
  1. 128 samples from the 4B and 128 from the 9B at T = 1, top-p 1, in sample order;
  2. each parsed by the §6 grammar, distinct commands merged across models;
  3. every distinct command scored once by the verifier (pooled.Pooled.scores);
  4. every distinct command executed once to judge it (scripts/ceiling/sample.execute,
     which is evaluate.score): evaluation only, the pick never sees it.
The pick with k samples per model uses each model's first k. The first k of 128
independent samples are k independent samples, so one pass gives every k up to 128
for the price of the 128 alone. Reported per split, for the 4B, the 9B and the two pooled:
  - one run: each model's first k samples, i.e. what a k-sample request returns, with
    pick at k against pick at 8 by exact McNemar;
  - the mean over 20 random draws of k of the 128 (pick_at_k.curves_for), the estimator
    of the test curves from the ceiling samples, so the two tables compare directly.
Rows are appended to <out>/rows_<split>.jsonl as they finish; a relaunch resumes.
Two rows are in flight, so one row's scoring and execution overlap the other's
sampling. The pass's throughput is reported; per-request latency is measured one
request at a time by pick_at_k.py --timing.

    python scripts/serve_pooled/served_at_k.py --splits val,test      # sample, score, judge, report
    python scripts/serve_pooled/served_at_k.py --splits val,test --report-only
"""
import argparse
import importlib.metadata
import json
import pathlib
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "ceiling",
          REPO / "scripts" / "methods_9b", HERE):
    sys.path.insert(0, str(p))
import pick_at_k as K   # noqa: E402

N = 128
REPORT_KS = (1, 4, 8, 32, 64, 128)      # the owner's settings (2026-10-20); every k up to 128 is in report.json
TIMING = REPO / "runs" / "serve_pooled" / "pick_at_k.json"
# recorded pooled picks with 8 samples per model, for reference beside the served ones
RECORDED = {
    "test": [("methods run, transformers samples (RESULTS.md 2026-10-19)", 65.3),
             ("ceiling samples, mean of 20 draws (runs/serve_pooled/pick_at_k.md)", 66.0)],
    "val": [("methods run, transformers samples (RESULTS.md 2026-10-19)", 59.6),
            ("served, 8-sample requests (runs/serve_pooled/measure.json)", 60.6)],
}


def load_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def row_maps(row):
    """{(mode, command): score} and {(mode, command): right} for a row's distinct candidates."""
    sc = {(c["mode"], c["command"]): c["score"] for c in row["cands"]}
    lab = {(c["mode"], c["command"]): int(c["state"] == "match") for c in row["cands"]}
    return sc, lab


def first_k(row, models, k):
    """One run with each model's first k samples -> (pick right, pass, maj right, distinct
    candidates). The pick is pooled.pick's: highest score, ties by count, then mode and
    text; the vote is pick_at_k's: count, then mode and text."""
    sc, lab = row_maps(row)
    counts = {}
    for m in models:
        for s in row["samples"][m][:k]:
            if s is not None:
                counts[tuple(s)] = counts.get(tuple(s), 0) + 1
    if not counts:
        return 0, 0, 0, 0
    pick = max(counts, key=lambda key: (sc[key], counts[key], key))
    maj = max(counts, key=lambda key: (counts[key], key))
    return lab[pick], int(any(lab[c] for c in counts)), lab[maj], len(counts)


def ceiling_row(row, model):
    """A model's samples in the ceiling format, for pick_at_k.curves_for."""
    _, lab = row_maps(row)
    counts = {}
    for s in row["samples"][model]:
        if s is not None:
            counts[tuple(s)] = counts.get(tuple(s), 0) + 1
    return {"n": len(row["samples"][model]),
            "commands": [{"mode": m, "command": c, "n": n, "state": "match" if lab[(m, c)] else "wrong"}
                         for (m, c), n in counts.items()]}


def do_row(pp, S, parse_output, case, timeout):
    """Sample, score and judge one row (the pass's unit of work)."""
    t0 = time.perf_counter()
    futs = {m: pp.pool.submit(pp._sample, ep, case) for m, ep in (("4b", pp.e4), ("9b", pp.e9))}
    outs = {m: f.result() for m, f in futs.items()}
    t_sample = time.perf_counter() - t0
    samples = {}
    for m, o in outs.items():
        samples[m] = [list(p) if p else None for p in (parse_output(t) for t in o["texts"])]
    distinct, seen = [], set()
    for m in ("4b", "9b"):
        for s in samples[m]:
            if s is not None and tuple(s) not in seen:
                seen.add(tuple(s))
                distinct.append({"mode": s[0], "command": s[1]})
    t1 = time.perf_counter()
    scores = []
    for s in range(0, len(distinct), 64):
        scores += pp.scores(case, distinct[s:s + 64])[0]
    t_score = time.perf_counter() - t1
    t2 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        states = list(ex.map(lambda c: S.execute(case, c["mode"], c["command"], timeout)[0], distinct))
    return {"id": case["id"], "samples": samples,
            "output_tokens": {m: o["output_tokens"] for m, o in outs.items()},
            "prompt_tokens": {m: o["prompt_tokens"] for m, o in outs.items()},
            "cands": [dict(c, score=v, state=st) for c, v, st in zip(distinct, scores, states)],
            "sample_s": round(t_sample, 3), "score_s": round(t_score, 3),
            "exec_s": round(time.perf_counter() - t2, 3)}


def run_pass(args, out, splits, cases):
    import endpoint as E
    import pooled as P
    import sample as S
    from prompt import parse_output
    e4 = E.load("qwen3.5-4b", "http://127.0.0.1:8011", "qwen3.5-4b-served")
    e9 = E.load("qwen3.5-9b", "http://127.0.0.1:8012", "served9b")
    for e in (e4, e9):
        e.timeout = 900.0            # 256 samples queued per engine with two rows in flight
    pp = P.Pooled(e4, e9, k=N)
    pp.pool = ThreadPoolExecutor(max_workers=2 * args.inflight)
    for sp in splits:
        path = out / ("rows_%s.jsonl" % sp)
        done = {r["id"] for r in load_jsonl(path)} if path.exists() else set()
        todo = [c for c in cases[sp] if c["id"] not in done]
        print("%s: %d rows to do, %d already done" % (sp, len(todo), len(done)), flush=True)
        if not todo:
            continue
        t0, n_done = time.time(), 0
        try:
            with path.open("a") as fh, ThreadPoolExecutor(max_workers=args.inflight) as ex:
                for r in ex.map(lambda c: do_row(pp, S, parse_output, c, args.timeout), todo):
                    fh.write(json.dumps(dict(r, split=sp)) + "\n")
                    fh.flush()
                    n_done += 1
                    if n_done % 25 == 0 or n_done == len(todo):
                        el = time.time() - t0
                        print("  %s %d / %d rows, %.0f s, %.1f s per row, %.0f min to go" % (
                            sp, n_done, len(todo), el, el / n_done, el / n_done * (len(todo) - n_done) / 60), flush=True)
        finally:
            with (out / "passes.jsonl").open("a") as fh:
                fh.write(json.dumps({"split": sp, "rows": n_done, "wall_s": round(time.time() - t0, 1),
                                     "inflight": args.inflight, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")


def mcnemar_line(a, b):
    """b against a, both lists of 0/1 over the same rows: '+g / -l, p'."""
    import compare_9b as C
    g = sum(1 for x, y in zip(a, b) if y and not x)
    lo = sum(1 for x, y in zip(a, b) if x and not y)
    return "+%d / −%d, p %.3f" % (g, lo, C.mcnemar_p(g, lo)), {"gained": g, "lost": lo, "p": C.mcnemar_p(g, lo)}


def report(args, out, splits, cases):
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = "unknown"
    passes = load_jsonl(out / "passes.jsonl") if (out / "passes.jsonl").exists() else []
    lines = ["# The served pooled pick with up to %d samples per model" % N, "",
             "Served through serve_pooled.sh on one RTX 3090 (vLLM %s): the 4B (`runs/serve/qwen3.5-4b-served`, "
             "merged bf16) with the verifier as a LoRA (`runs/reranker_2026-10-16/verifier-4b/best`), and the 9B "
             "(`Qwen/Qwen3.5-9B` text weights, NF4 at load, `runs/qwen3.5-9b-grpo-gated/best` as a LoRA). %d samples "
             "per model per row at T = 1, top-p 1; every distinct command scored by the verifier and executed once "
             "to judge it. **One run** uses each model's first k samples (what a k-sample request returns); the "
             "**mean of %d draws** takes random k of the %d, as the test curves from the ceiling samples do."
             % (vllm_version, N, args.draws, N), ""]
    rep = {"vllm": vllm_version, "n": N, "splits": {}}
    for sp in splits:
        rows = {}
        if (out / ("rows_%s.jsonl" % sp)).exists():
            for r in load_jsonl(out / ("rows_%s.jsonl" % sp)):
                rows.setdefault(r["id"], r)
        ids = [c["id"] for c in cases[sp] if c["id"] in rows]
        n = len(ids)
        if not n:
            continue
        srep = rep["splits"][sp] = {"rows": n, "of": len(cases[sp])}
        lines += ["## %s (%d of %d rows)" % (sp, n, len(cases[sp])), "",
                  "### One run: each model's first k samples", "",
                  "| k per model | 4B pick / pass / maj | 9B pick / pass / maj | 4B + 9B pick / pass / maj "
                  "| distinct candidates (4B + 9B) | tokens generated (4B + 9B) |", "|---|---|---|---|---|---|"]
        outcome = {}
        for k in REPORT_KS:
            cells = []
            for name, models in K.CONFIGS:
                res = [first_k(rows[i], models, k) for i in ids]
                outcome[(name, k)] = [r[0] for r in res]
                p, a, mj = (100 * statistics.fmean(r[j] for r in res) for j in range(3))
                srep.setdefault("one_run", {}).setdefault(name, {})[k] = {"pick": round(p, 1), "pass": round(a, 1),
                                                                          "maj": round(mj, 1)}
                cells.append("**%.1f** / %.1f / %.1f" % (p, a, mj))
                if name == "4B + 9B":
                    cand = statistics.fmean(r[3] for r in res)
            toks = statistics.fmean(sum(sum(rows[i]["output_tokens"][m][:k]) for m in ("4b", "9b")) for i in ids)
            srep["one_run"]["4B + 9B"][k].update(candidates=round(cand, 1), tokens=round(toks, 1))
            lines.append("| %d | %s | %.1f | %.0f |" % (k, " | ".join(cells), cand, toks))
        lines += ["", "Pick at k against pick at 8, same rows (exact McNemar):", "",
                  "| | 4B | 9B | 4B + 9B |", "|---|---|---|---|"]
        for k in REPORT_KS:
            if k == 8:
                continue
            cells = []
            for name, _ in K.CONFIGS:
                text, d = mcnemar_line(outcome[(name, 8)], outcome[(name, k)])
                srep.setdefault("one_run_vs_8", {}).setdefault(name, {})[k] = d
                cells.append(text)
            lines.append("| %d vs 8 | %s |" % (k, " | ".join(cells)))
        crow = {m: {i: ceiling_row(rows[i], m) for i in ids} for m in ("4b", "9b")}
        maps = {i: row_maps(rows[i])[0] for i in ids}
        curves, per_row = K.curves_for(ids, crow, lambda i, mode, cmd: maps[i][(mode, cmd)], args.draws, seed="served@k|")
        cl, crep = K.curve_report(curves, per_row, n, "Mean of %d draws of k of the %d" % (args.draws, N),
                                  "pick / pass / maj, as in the test curves from the ceiling samples.", h="###",
                                  ks=REPORT_KS)
        lines += [""] + cl
        srep["mean_of_draws"] = crep
        timing = K.load_timing(getattr(args, "timing_json", TIMING))
        timed = [k for k in REPORT_KS if k in timing]
        if timed:
            gpu_hour = getattr(args, "gpu_hour", 0.30)
            lines += ["", "### Cost per solved task on %s" % sp, "",
                      "GPU-seconds per request = the mean one-at-a-time latency of the served pooled pick at that k "
                      "(`pick_at_k.json`, 30 val rows); per solved task, divided by this split's pooled pick@k "
                      "(mean of draws); dollars at $%.2f per GPU-hour." % gpu_hour, "",
                      "| k per model | GPU-s per request | pooled pick@k | GPU-s per solved | $ per 1,000 solved |",
                      "|---|---|---|---|---|"]
            for k in timed:
                acc = crep["curves"]["4B + 9B"][k]["pick"] / 100
                per = timing[k]["mean_s"] / acc if acc else None          # nothing solved: no price
                usd = per * 1000 / 3600 * gpu_hour if per is not None else None
                srep.setdefault("cost", {})[k] = {"gpu_s_per_request": timing[k]["mean_s"], "pick": round(100 * acc, 1),
                                                  "gpu_s_per_solved": round(per, 3) if per is not None else None,
                                                  "usd_per_1000_solved": round(usd, 4) if usd is not None else None}
                lines.append("| %d | %.2f | %.1f%% | %s | %s |" % (
                    k, timing[k]["mean_s"], 100 * acc, "%.2f" % per if per is not None else "—",
                    "$%.3f" % usd if usd is not None else "—"))
        lines += ["", "For reference, the pooled pick with 8 samples per model recorded earlier on %s:" % sp, ""]
        lines += ["- %s: %.1f%%" % (what, v) for what, v in RECORDED.get(sp, [])]
        sp_passes = [p for p in passes if p["split"] == sp and p["rows"]]
        if sp_passes:
            rows_done, wall = sum(p["rows"] for p in sp_passes), sum(p["wall_s"] for p in sp_passes)
            srep["throughput"] = {"rows": rows_done, "wall_s": wall, "s_per_row": round(wall / rows_done, 2)}
            lines += ["", "Pass throughput at %d samples per model, %d rows in flight: %.1f s per row "
                      "(%d rows in %.1f h, judging included)." % (N, sp_passes[-1]["inflight"], wall / rows_done,
                                                                  rows_done, wall / 3600)]
        lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--out", default=str(REPO / "runs" / "serve_pooled" / "served_at_k"))
    ap.add_argument("--inflight", type=int, default=2, help="rows in flight at once")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5.0, help="executor timeout per command, seconds (§9)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows of each split (a smoke run)")
    ap.add_argument("--timing-json", default=str(TIMING), help="pick_at_k.json whose timing prices each k")
    ap.add_argument("--gpu-hour", type=float, default=0.30, help="USD per GPU-hour for the dollar column")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    splits = args.splits.split(",")
    cases = {sp: load_jsonl(REPO / "data" / "splits" / ("%s.jsonl" % sp)) for sp in splits}
    if args.limit:
        cases = {sp: v[:args.limit] for sp, v in cases.items()}
    if not args.report_only:
        run_pass(args, out, splits, cases)
    report(args, out, splits, cases)
    return 0


if __name__ == "__main__":
    sys.exit(main())
