#!/usr/bin/env python3
"""Short serving follow-ups, with both endpoints up (serve_pooled.sh): the cost of the other
served systems, the pooled pick at small k under load, the two asking turns, and asymmetric
sampling. Stages (each skipped on a relaunch if report.json already holds it):

  systems   as served, every val row at concurrency 8 (exec-match by the harness) and
            throughput at 1 / 8 / 32: the 4B's and the 9B's §6 answer (greedy), the single-model
            picks (8 samples + verifier), and the pooled pick at 1 and 4 per model; plus the
            throughput of the sample rule's first turn (the 4B's 8 samples, no verifier).
  ask       the two generations a clarifying question costs the model, on the 4B endpoint:
            the question (forced "ask:" prefix, one line, 64 tokens) and the answer after the
            reply (turn-2 prompt, 192 tokens), with the questions and replies recorded for the
            served 4B on val (runs/clarify_2026-10-06/val/replies.jsonl). One at a time and at
            8 / 32 concurrent. The asking systems' GPU time per request is then composed:
            verifier-gated ask = the 4B pick + 20% x (question + answer); the sample rule's
            round trip = the 8 samples + 41.8% x (question + answer); always ask = question +
            answer. Their exec-match is the recorded one (RECORDED below).
  asym      k samples from the 4B and k9 from the 9B: (8, 2), (8, 4), (4, 2), (8, 1), (4, 1),
            which cut the 9B's share, and (2, 4), (1, 4), which cut the 4B's; one request at a
            time on 30 val rows; accuracy on test from the ceiling samples' saved verifier
            scores (pick_at_k: 20 draws), beside the symmetric settings.

Cost: GPU-seconds per request = 1 / requests per second at the best measured concurrency (the
deployment holds the card); per solved task = that / exec-match; dollars at --gpu-hour.

    python scripts/serve_pooled/followups.py --out runs/serve_pooled/followups
"""
import argparse
import json
import pathlib
import random
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
import measure as M      # noqa: E402
import pick_at_k as K    # noqa: E402

GATE_RATE = 0.20                               # the verifier-gated ask's share of asked rows (methods_9b)
RULE_ASKS = {"val": 0.418, "test": 0.401}      # the sample rule's ask share for the 4B (methods_9b report)
# exec-match of the 4B systems whose cost is composed from measured parts
# (runs/qwen9b_methods_2026-10-19/report.md, the 4B with V4)
RECORDED = {"verifier-gated ask (4B, 20%)": {"val": 57.2, "test": 61.3},
            "sample rule round trip (4B, the deployed system)": {"val": 55.4, "test": 59.7},
            "always ask (4B)": {"val": 50.6, "test": 58.5}}
REPLIES = REPO / "runs" / "clarify_2026-10-06" / "val" / "replies.jsonl"
ASYM = ((8, 2), (8, 4), (4, 2), (8, 1), (4, 1),      # fewer 9B samples (the GPU-time view)
        (2, 4), (1, 4))                              # fewer 4B samples (the token frontier; owner, 2026-10-20)
SYMMETRIC = ((8, 8), (4, 4), (2, 2), (1, 1))


def load_jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def composed(base, q, a, rate):
    """GPU-seconds per request of a system that makes the base request and, on `rate` of
    requests, a question and then an answer after the reply."""
    return base + rate * (q + a)


def cost_row(gpu_s, acc, gpu_hour):
    acc = acc / 100
    return {"gpu_s_per_request": round(gpu_s, 3), "exec_match": round(100 * acc, 1),
            "gpu_s_per_solved": round(gpu_s / acc, 3) if acc else None,
            "usd_per_1000_solved": round(gpu_s / acc * 1000 / 3600 * gpu_hour, 4) if acc else None}


def best_gpu_s(thrs):
    best = max(thrs, key=lambda r: r["req_per_s"])
    return 1.0 / best["req_per_s"], best["concurrency"]


def throughput(fn, val, concs=(1, 8, 32), n=96):
    out = []
    for conc in concs:
        _, thr = M.run_concurrent(fn, val[:(n if conc > 1 else 32)], conc)
        thr["memory"] = M.gpu_mem()
        out.append(thr)
    return out


def save_rows(path, rows):
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def stage_systems(args, out, rep, e4, e9, val, score):
    import pooled as P
    from prompt import messages_for_case, parse_output

    def greedy(ep):
        def fn(case):
            t0 = time.perf_counter()
            r = ep.complete(ep.render(messages_for_case(case)), max_tokens=192)
            p = parse_output(r["texts"][0])
            return {"id": case["id"], "mode": p[0] if p else None, "command": p[1] if p else None,
                    "output_tokens": r["output_tokens"][0], "latency_s": time.perf_counter() - t0}
        return fn

    def samples8(case):     # the sample rule's first turn: 8 samples, nothing scored
        t0 = time.perf_counter()
        r = e4.complete(e4.render(messages_for_case(case)), max_tokens=192, temperature=1.0, n=8)
        return {"id": case["id"], "output_tokens": sum(r["output_tokens"]), "latency_s": time.perf_counter() - t0}

    pools = {k: P.Pooled(e4, e9, k=k) for k in (1, 4, 8)}
    systems = [
        ("4B §6 answer (greedy)", greedy(e4), True),
        ("9B §6 answer (greedy)", greedy(e9), True),
        ("4B 8 samples + verifier (pick)", lambda c: pools[8].answer(c, models=("4b",)), True),
        ("9B 8 samples + verifier (pick)", lambda c: pools[8].answer(c, models=("9b",)), True),
        ("pooled pick, 1 per model", pools[1].answer, True),
        ("pooled pick, 4 per model", pools[4].answer, True),
        ("4B 8 samples, no verifier (the sample rule's first turn)", samples8, False),
    ]
    res = {}
    for name, fn, with_acc in systems:
        t0 = time.time()
        entry = {}
        if with_acc:
            rows, thr8, = M.run_concurrent(fn, val, 8)
            preds = {r["id"]: {"command": r["command"], "mode": r["mode"]} for r in rows if r.get("command") is not None}
            entry["exec_match_val"] = round(100 * score(val, preds)["n_match"] / len(val), 1)
            entry["output_tokens_mean"] = round(statistics.fmean(
                (sum(r["output_tokens"].values()) if isinstance(r["output_tokens"], dict) else r["output_tokens"])
                for r in rows), 1)
            save_rows(out / ("rows_%s.jsonl" % name.split(" (")[0].replace(" ", "_").replace(",", "").replace("§", "s")),
                      [{k: v for k, v in r.items() if k != "candidates"} for r in rows])
            thr8.update(requests=len(val), note="the accuracy pass")
            thrs = throughput(fn, val, concs=(1,)) + [thr8] + throughput(fn, val, concs=(32,))
        else:
            thrs = throughput(fn, val)
        entry["throughput"] = thrs
        entry["p50_one_at_a_time_s"] = thrs[0]["p50_s"]
        entry["gpu_s_per_request"], entry["best_concurrency"] = best_gpu_s(thrs)
        entry["seconds"] = round(time.time() - t0)
        res[name] = entry
        print("system", name, {k: v for k, v in entry.items() if k != "throughput"}, flush=True)
        rep["systems"] = res
        write(out, rep)
    return res


def stage_ask(args, out, rep, e4, val):
    from prompt import parse_output
    from protocol import ASK_PREFIX, QUESTION_MAX_TOKENS, cut_question, turn1_messages, turn2_messages
    replies = {r["id"]: r for r in load_jsonl(REPLIES) if r["label"] == "served"}
    rows = [c for c in val if c["id"] in replies][:args.ask_n]

    def question(case):
        t0 = time.perf_counter()
        r = e4.complete(e4.render(turn1_messages(case), prefill=ASK_PREFIX), max_tokens=QUESTION_MAX_TOKENS, stop=["\n"])
        return {"id": case["id"], "question": cut_question(r["texts"][0]), "output_tokens": r["output_tokens"][0],
                "latency_s": time.perf_counter() - t0}

    def answer_after(case):
        rep_ = replies[case["id"]]
        t0 = time.perf_counter()
        r = e4.complete(e4.render(turn2_messages(case, rep_["question"], rep_["reply"])), max_tokens=192)
        p = parse_output(r["texts"][0])
        return {"id": case["id"], "mode": p[0] if p else None, "command": p[1] if p else None,
                "output_tokens": r["output_tokens"][0], "latency_s": time.perf_counter() - t0}

    res = {}
    for name, fn in (("question (forced ask:, one line)", question), ("answer after the reply (turn 2)", answer_after)):
        rows1, thr1 = M.run_concurrent(fn, rows, 1)
        save_rows(out / ("rows_ask_%s.jsonl" % name.split(" ")[0]), rows1)
        thrs = [thr1] + throughput(fn, rows, concs=(8, 32), n=len(rows))
        entry = {"rows": len(rows), "p50_one_at_a_time_s": thr1["p50_s"], "p95_one_at_a_time_s": thr1["p95_s"],
                 "output_tokens_mean": round(statistics.fmean(r["output_tokens"] for r in rows1), 1),
                 "throughput": thrs}
        entry["gpu_s_per_request"], entry["best_concurrency"] = best_gpu_s(thrs)
        if "command" in rows1[0]:
            entry["parsed_share"] = round(sum(r["command"] is not None for r in rows1) / len(rows1), 3)
        res[name] = entry
        print("ask", name, {k: v for k, v in entry.items() if k != "throughput"}, flush=True)
    rep["ask"] = res
    write(out, rep)
    return res


def asym_accuracy(settings, draws=20):
    """Pooled pick / pass / maj on test for (k4, k9) settings, from the ceiling samples and the
    saved verifier scores, like pick_at_k's curves."""
    test = {c["id"]: c for c in load_jsonl(REPO / "data" / "splits" / "test.jsonl")}
    rows = {m: {r["id"]: r for r in load_jsonl(K.CEIL / ("samples_served%s.jsonl" % m))} for m in ("4b", "9b")}
    ids = [i for i in test if all(i in rows[m] for m in rows)]
    done = {(r["id"], r["mode"], r["command"]): r["score"] for r in load_jsonl(REPO / "runs" / "serve_pooled" / "scores_128.jsonl")}
    acc = {s: [0.0, 0.0, 0.0] for s in settings}
    for i in ids:
        cb, pops = {}, {}
        for m in ("4b", "9b"):
            cb[m], pops[m] = K.expand(rows[m][i])
        scores = {(c["mode"], c["command"]): done[(i, c["mode"], c["command"])] for m in cb for c in cb[m]}
        rng = random.Random("asym|" + i)
        for s in settings:
            tot = [0, 0, 0]
            for _ in range(draws):
                r = K.draw_eval(pops, cb, scores, rng, {"4b": s[0], "9b": s[1]})
                tot = [a + b for a, b in zip(tot, r)]
            acc[s] = [x + t / draws for x, t in zip(acc[s], tot)]
    return {s: {"pick": round(100 * v[0] / len(ids), 1), "pass": round(100 * v[1] / len(ids), 1),
                "maj": round(100 * v[2] / len(ids), 1)} for s, v in acc.items()}, len(ids)


def stage_asym(args, out, rep, e4, e9, val):
    import pooled as P
    rows = val[:args.asym_n]
    accs, n = asym_accuracy(ASYM + SYMMETRIC, args.draws)
    res = {}
    for k4, k9 in ASYM:
        pp = P.Pooled(e4, e9, k=k4, k9=k9)
        rs = [pp.answer(c) for c in rows]
        lat = [r["latency_s"] for r in rs]
        entry = {"k4": k4, "k9": k9, "mean_s": round(statistics.fmean(lat), 3), "p50_s": round(M.pct(lat, 50), 3),
                 "p95_s": round(M.pct(lat, 95), 3), "sample_p50_s": round(M.pct([r["sample_s"] for r in rs], 50), 3),
                 "candidates_mean": round(statistics.fmean(len(r["candidates"]) for r in rs), 1),
                 "tokens_mean": round(statistics.fmean(sum(r["output_tokens"].values()) for r in rs), 0),
                 "test": accs[(k4, k9)]}
        res["%d + %d" % (k4, k9)] = entry
        print("asym", k4, k9, entry, flush=True)
    rep["asym"] = {"timed_rows": len(rows), "test_rows": n, "settings": res,
                   "symmetric_test": {"%d + %d" % s: accs[s] for s in SYMMETRIC}}
    write(out, rep)
    return rep["asym"]


def write(out, rep):
    (out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


def report_md(rep, gpu_hour):
    L = ["# Serving follow-ups (val as served; one RTX 3090, both endpoints up)", "",
         "GPU-seconds per request = 1 / requests per second at the best measured concurrency; per solved task = "
         "that / exec-match; dollars at $%.2f per GPU-hour. Memory in use: %s MiB idle." % (
             gpu_hour, (rep.get("memory_idle") or {}).get("used_mib", "?")), ""]
    sy = rep.get("systems", {})
    if sy:
        L += ["## Served systems", "",
              "| system | exec-match (val, as served) | tokens / request | p50, one at a time | requests/s at 1 / 8 / 32 "
              "| GPU-s / request | GPU-s / solved | $ / 1,000 solved |", "|---|---|---|---|---|---|---|---|"]
        for name, e in sy.items():
            acc = e.get("exec_match_val")
            c = cost_row(e["gpu_s_per_request"], acc, gpu_hour) if acc is not None else None
            L.append("| %s | %s | %s | %.2f s | %s | %.3f | %s | %s |" % (
                name, "%.1f%%" % acc if acc is not None else "—",
                "%.0f" % e["output_tokens_mean"] if "output_tokens_mean" in e else "—", e["p50_one_at_a_time_s"],
                " / ".join("%.1f" % t["req_per_s"] for t in e["throughput"]), e["gpu_s_per_request"],
                "%.3f" % c["gpu_s_per_solved"] if c else "—", "$%.4f" % c["usd_per_1000_solved"] if c else "—"))
        pooled8 = REPO / "runs" / "serve_pooled" / "measure.json"
        if pooled8.exists():
            m = json.loads(pooled8.read_text())["cost"].get("pooled pick")
            if m:
                L.append("| pooled pick, 8 per model (measure.json, 2026-10-19) | %.1f%% | — | — | — | %.3f | %.3f | $%.4f |" % (
                    m["val_exec_match"], m["gpu_s_per_request"], m["gpu_s_per_solved"], m["usd_per_1000_solved"]))
    ask = rep.get("ask", {})
    if ask:
        L += ["", "## The asking turns (4B endpoint, %d val rows with recorded questions and replies)" % next(iter(ask.values()))["rows"], "",
              "| turn | p50 / p95, one at a time | tokens | requests/s at 1 / 8 / 32 | GPU-s / request |", "|---|---|---|---|---|"]
        for name, e in ask.items():
            L.append("| %s | %.2f / %.2f s | %.0f | %s | %.3f |" % (
                name, e["p50_one_at_a_time_s"], e["p95_one_at_a_time_s"], e["output_tokens_mean"],
                " / ".join("%.1f" % t["req_per_s"] for t in e["throughput"]), e["gpu_s_per_request"]))
        q = ask["question (forced ask:, one line)"]["gpu_s_per_request"]
        a = ask["answer after the reply (turn 2)"]["gpu_s_per_request"]
        pick4 = sy.get("4B 8 samples + verifier (pick)", {}).get("gpu_s_per_request")
        s8 = sy.get("4B 8 samples, no verifier (the sample rule's first turn)", {}).get("gpu_s_per_request")
        if pick4 is not None and s8 is not None:
            comp = {"verifier-gated ask (4B, 20%)": composed(pick4, q, a, GATE_RATE),
                    "sample rule round trip (4B, the deployed system)": composed(s8, q, a, RULE_ASKS["val"]),
                    "always ask (4B)": composed(0.0, q, a, 1.0)}
            rep["composed"] = {}
            L += ["", "## Systems that ask, composed from the parts above", "",
                  "The user's reply time is not GPU time. Exec-match is the recorded one (methods_9b report, the 4B with V4).", "",
                  "| system | GPU-s / request | exec-match val / test (recorded) | GPU-s / solved (val) | $ / 1,000 solved |",
                  "|---|---|---|---|---|"]
            for name, g in comp.items():
                c = cost_row(g, RECORDED[name]["val"], gpu_hour)
                c["exec_match_test"] = RECORDED[name]["test"]
                rep["composed"][name] = c
                L.append("| %s | %.3f | %.1f%% / %.1f%% | %.3f | $%.4f |" % (
                    name, g, RECORDED[name]["val"], RECORDED[name]["test"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
    asym = rep.get("asym")
    if asym:
        L += ["", "## Asymmetric sampling: k from the 4B + k9 from the 9B", "",
              "Latency one request at a time on %d val rows; accuracy on test (%d rows) from the ceiling samples' saved "
              "verifier scores, 20 draws. GPU-s per solved = mean latency / test pick." % (asym["timed_rows"], asym["test_rows"]), "",
              "| 4B + 9B samples | p50 / p95 | tokens | candidates | test pick / pass / maj | GPU-s / solved | $ / 1,000 solved |",
              "|---|---|---|---|---|---|---|"]
        for name, e in asym["settings"].items():
            c = cost_row(e["mean_s"], e["test"]["pick"], gpu_hour)
            L.append("| %s | %.2f / %.2f s | %.0f | %.1f | **%.1f** / %.1f / %.1f | %.2f | $%.3f |" % (
                name, e["p50_s"], e["p95_s"], e["tokens_mean"], e["candidates_mean"], e["test"]["pick"], e["test"]["pass"],
                e["test"]["maj"], c["gpu_s_per_solved"], c["usd_per_1000_solved"]))
        L += ["", "Symmetric settings on test by the same draws: " + "; ".join(
            "%s → %.1f / %.1f / %.1f" % (s, v["pick"], v["pass"], v["maj"]) for s, v in asym["symmetric_test"].items())]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO / "runs" / "serve_pooled" / "followups"))
    ap.add_argument("--stages", default="systems,ask,asym")
    ap.add_argument("--ask-n", type=int, default=100)
    ap.add_argument("--asym-n", type=int, default=30)
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--gpu-hour", type=float, default=0.30)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
    if not args.report_only:
        import endpoint as E
        from evaluate import score
        e4 = E.load("qwen3.5-4b", "http://127.0.0.1:8011", "qwen3.5-4b-served")
        e9 = E.load("qwen3.5-9b", "http://127.0.0.1:8012", "served9b")
        e9.name = "served9b"
        val = load_jsonl(REPO / "data" / "splits" / "val.jsonl")
        rep.setdefault("memory_idle", M.gpu_mem())
        rep.setdefault("started", time.strftime("%Y-%m-%d %H:%M"))
        stages = args.stages.split(",")
        if "systems" in stages and "systems" not in rep:
            stage_systems(args, out, rep, e4, e9, val, score)
        if "ask" in stages and "ask" not in rep:
            stage_ask(args, out, rep, e4, val)
        if "asym" in stages and "asym" not in rep:
            stage_asym(args, out, rep, e4, e9, val)
    md = report_md(rep, args.gpu_hour)
    write(out, rep)
    (out / "report.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
