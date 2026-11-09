#!/usr/bin/env python3
"""R0: can vLLM batch the base 27B's thinking on this card? (runs/engine27b_2026-11-02/preregistration.md)

Part 2's gate in runs/teacher_channels_2026-11-02/plan.md. The thinking runs went one row at a time,
because a left-padded batch moved the 27B's log-probabilities by 0.72. That ran at about 17 generated
tokens per second, which puts the 795-row trace pass at about 77 hours. vLLM batches sequences without
padding; this measures how fast it runs the base 27B's thinking with 1, 4 and 8 sequences in flight.

The prompts are the first 16 rows of the base 27B arm's order that have an 8,192-token outcome in
runs/think_long_2026-11-01, rendered as that arm renders them. The answers are judged as the thinking
run judges them, so the recorded outcomes on the same rows sit beside these (one sample each).

    python scripts/engine27b/probe.py prompts                  # CPU: the 16 rows -> prompts.jsonl
    python scripts/engine27b/probe.py serve-args --attempt 1   # the `vllm serve` arguments, one per line
    python scripts/engine27b/probe.py stage A --port 8013      # one stage against the running server
    python scripts/engine27b/probe.py report                   # CPU: probe.md, probe.json
"""
import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
import zlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                   # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = load("think_long_report", REPO / "scripts" / "think_long" / "report.py")   # prefix(): each row's 8k outcome
E = R.E
T = E.T

RUN = REPO / "runs" / "engine27b_2026-11-02"
ARM = "think-retr8"            # the base 27B with 8 retrieved examples, thinking
PLAIN_ARM = "nothink-trained"  # its prompt shape only: the zero-shot §6 prompt, thinking off
CAP = 8192                     # generated tokens, reasoning plus answer: phase C's cap
PLAIN_CAP = 192
N_ROWS = 16
SEED = 20261102
SERVED_NAME = "engine27b"
GPU_UTIL = "0.90"
GATE_ONE, GATE_TWO = 120.0, 60.0   # tokens/s: one 12-hour window for R1, or two
MIN_INTERVALS = 3
REQUEST_TIMEOUT = 3600
# stage: (sequences in flight, thinking, [(row index, draw)])
STAGES = {
    "A": (1, True, [(k, 0) for k in range(0, 2)]),
    "B": (4, True, [(k, 0) for k in range(2, 8)]),
    "C": (8, True, [(k, 0) for k in range(8, 16)] + [(k, 1) for k in range(0, 4)]),
    "D1": (1, False, [(k, 0) for k in range(N_ROWS)]),
    "D8": (8, False, [(k, 1) for k in range(N_ROWS)]),
}
THINK_STAGES = [s for s, (_, thinking, _) in STAGES.items() if thinking]
# attempt: (extra flags, --max-num-seqs). The first that comes up is used (the pre-registration's fallbacks).
ATTEMPTS = {1: ([], 8), 2: (["--enforce-eager"], 8), 3: (["--enforce-eager"], 4)}
STATS_RE = re.compile(
    r"(\d\d-\d\d \d\d:\d\d:\d\d)(?:\.\d+)? .*?Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs"
    r"(?:, Deferred: \d+ reqs)?(?:, Preemptions: (\d+))?, GPU KV cache usage: ([\d.]+)%")
KV_TOKENS_RE = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
KV_CONC_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")


# ------------------------------------------------------------ prompts (CPU)

def select_rows():
    """[(id, recorded completion)]: the first N_ROWS of the arm's order with an outcome at CAP."""
    byid = {arm: {r["id"]: r for r in rows} for arm, rows in E.first_rows(E.FIRST_RUN).items()}
    orders = E.arm_orders(E.FIRST_RUN)
    stages = {cap: {arm: E.stage_rows(E.RUN, cap, arm) for arm in E.THINK_ARMS} for cap in E.CAPS}
    rows = R.prefix(orders[ARM], byid, stages, ARM, CAP)
    if len(rows) < N_ROWS:
        sys.exit("only %d rows have an outcome at %d" % (len(rows), CAP))
    return rows[:N_ROWS]


def engine_info():
    """The tokenizer, the registry kwargs, the stop ids and the card's sampling (checked)."""
    from transformers import AutoTokenizer, GenerationConfig
    from baseline import load_registry, stop_token_ids
    regs = {n: load_registry(n) for n in (T.THINK_MODEL, T.PLAIN_MODEL)}
    a, b = regs[T.THINK_MODEL], regs[T.PLAIN_MODEL]
    if (a["hf_id"], a["revision"]) != (b["hf_id"], b["revision"]):
        sys.exit("%s and %s must name the same weights" % (T.THINK_MODEL, T.PLAIN_MODEL))
    tok = AutoTokenizer.from_pretrained(a["hf_id"], revision=a["revision"])
    gen = GenerationConfig.from_pretrained(a["hf_id"], revision=a["revision"])
    sampling = {"temperature": gen.temperature, "top_p": gen.top_p, "top_k": gen.top_k}
    if sampling != T.PREREGISTERED_SAMPLING:
        sys.exit("the model card's sampling is %s, not the pre-registered %s" % (sampling, T.PREREGISTERED_SAMPLING))
    stops = set(stop_token_ids(tok, a["stop"]))
    eos = gen.eos_token_id
    stops |= set(eos if isinstance(eos, list) else [eos])
    info = {"kwargs": {n: r.get("chat_template_kwargs") or {} for n, r in regs.items()}}
    return tok, info, {"sampling": sampling, "stop_ids": sorted(stops),
                       "think_end_id": tok.convert_tokens_to_ids(T.THINK_END)}


def cmd_prompts(args):
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    rows = select_rows()
    tok, info, meta = engine_info()
    val = {c["id"]: c for c in load_jsonl(str(T.VAL))}
    train = {c["id"]: c for c in load_jsonl(str(T.TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(T.PLAN))}
    out = []
    for k, (i, o) in enumerate(rows):
        c = val[i]
        think = tok(E.prompt_text(tok, c, ARM, train, plan, info))["input_ids"]
        plain = tok(E.prompt_text(tok, c, PLAIN_ARM, train, plan, info))["input_ids"]
        out.append({"k": k, "id": i, "tier": c.get("tier"), "category": c.get("category"),
                    "think_ids": think, "plain_ids": plain,
                    "recorded": {"match": bool(o["match"]), "closed": bool(o["closed"]),
                                 "think_tokens": o.get("think_tokens"), "fresh": bool(o.get("fresh"))}})
    longest = max(len(p["think_ids"]) for p in out)
    meta.update({"rows": len(out), "longest_prompt": longest, "cap": CAP, "plain_cap": PLAIN_CAP,
                 "max_model_len": -(-(longest + CAP) // 256) * 256,
                 "source": str(E.RUN.relative_to(REPO)), "arm": ARM})
    with (run / "prompts.jsonl").open("w", encoding="utf-8") as fh:
        for p in out:
            fh.write(json.dumps(p) + "\n")
    (run / "prompts_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("%d prompts: thinking %d-%d tokens, zero-shot %d-%d; max-model-len %d; recorded solves %d"
          % (len(out), min(len(p["think_ids"]) for p in out), longest, min(len(p["plain_ids"]) for p in out),
             max(len(p["plain_ids"]) for p in out), meta["max_model_len"], sum(p["recorded"]["match"] for p in out)))
    return 0


# ------------------------------------------------------------ the server's arguments

def serve_args(attempt, port, meta):
    from baseline import load_registry
    reg = load_registry(T.THINK_MODEL)
    extra, seqs = ATTEMPTS[attempt]
    return ([reg["hf_id"], "--revision", reg["revision"], "--tokenizer-revision", reg["revision"],
             "--served-model-name", SERVED_NAME, "--language-model-only",
             "--quantization", "bitsandbytes", "--dtype", "bfloat16",
             "--max-model-len", str(meta["max_model_len"]), "--max-num-seqs", str(seqs),
             "--gpu-memory-utilization", GPU_UTIL, "--generation-config", "vllm", "--seed", "0",
             "--port", str(port), "--host", "127.0.0.1"] + extra + list(reg.get("vllm_extra_args") or []))


def cmd_serve_args(args):
    meta = json.loads((pathlib.Path(args.run) / "prompts_meta.json").read_text(encoding="utf-8"))
    print("\n".join(serve_args(args.attempt, args.port, meta)))
    return 0


# ------------------------------------------------------------ one stage (against the server)

def seed_for(row_id, draw, thinking):
    return SEED + zlib.crc32(("%s|%d|%d" % (row_id, draw, thinking)).encode()) % 1000000


def request_body(p, draw, thinking, meta):
    return {"model": SERVED_NAME, "prompt": p["think_ids"] if thinking else p["plain_ids"],
            "max_tokens": CAP if thinking else PLAIN_CAP, **meta["sampling"],
            "seed": seed_for(p["id"], draw, thinking), "stop_token_ids": meta["stop_ids"],
            "skip_special_tokens": False, "return_token_ids": True}


def post(port, body, timeout=REQUEST_TIMEOUT):
    req = urllib.request.Request("http://127.0.0.1:%d/v1/completions" % port, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def completion_record(stage, p, draw, thinking, meta, resp, t0, t1):
    ch = resp["choices"][0]
    ids = ch.get("token_ids") or []
    reasoning, answer, closed = T.split_reasoning(ch["text"], thinking)
    think_end = meta["think_end_id"]
    return {"stage": stage, "k": p["k"], "id": p["id"], "draw": draw, "thinking": thinking,
            "seed": seed_for(p["id"], draw, thinking), "t0": round(t0, 3), "t1": round(t1, 3),
            "seconds": round(t1 - t0, 2), "prompt_tokens": resp["usage"]["prompt_tokens"],
            "completion_tokens": resp["usage"]["completion_tokens"], "finish_reason": ch.get("finish_reason"),
            "think_tokens": (ids.index(think_end) if think_end in ids else len(ids)) if thinking else 0,
            "closed": closed, "answer": answer, "reasoning": reasoning}


def cmd_stage(args):
    run = pathlib.Path(args.run)
    name = args.name
    in_flight, thinking, plan = STAGES[name]
    meta = json.loads((run / "prompts_meta.json").read_text(encoding="utf-8"))
    prompts = load_jsonl(str(run / "prompts.jsonl"))
    summary_path, rows_path = run / ("stage_%s.json" % name), run / ("rows_%s.jsonl" % name)
    if summary_path.exists():
        print("stage %s: done already" % name)
        return 0
    if rows_path.exists():                       # a partial stage reruns whole, so its timing is clean
        rows_path.rename(run / ("rows_%s.partial-%s.jsonl" % (name, datetime.datetime.now().strftime("%H%M%S"))))
    val = {c["id"]: c for c in load_jsonl(str(T.VAL))}
    print("stage %s: %d requests, %d in flight, thinking %s" % (name, len(plan), in_flight, thinking), flush=True)
    recs, errors = [], []
    t_start = time.time()

    def one(kd):
        k, draw = kd
        p = prompts[k]
        t0 = time.time()
        resp = post(args.port, request_body(p, draw, thinking, meta))
        return completion_record(name, p, draw, thinking, meta, resp, t0, time.time())

    with concurrent.futures.ThreadPoolExecutor(max_workers=in_flight) as pool:
        futs = {pool.submit(one, kd): kd for kd in plan}
        for f in concurrent.futures.as_completed(futs):
            try:
                r = f.result()
            except (urllib.error.URLError, OSError, KeyError, ValueError) as exc:
                errors.append({"k": futs[f][0], "draw": futs[f][1], "error": repr(exc)})
                print("  request %s failed: %r" % (futs[f], exc), flush=True)
                continue
            recs.append(r)
            print("  %s k=%d draw %d: %d tokens in %.0f s, %s, closed %s" % (
                name, r["k"], r["draw"], r["completion_tokens"], r["seconds"], r["finish_reason"], r["closed"]), flush=True)
    t_end = time.time()
    # Judged after the clock stops, the thinking run's way (§6 grammar, then the sandbox).
    cases = [val[r["id"]] for r in recs]
    parsed, per = T.judge(cases, [r["answer"] for r in recs], args.timeout) if recs else ({}, {})
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in sorted(recs, key=lambda r: (r["draw"], r["k"])):
            p, s = parsed.get(r["id"]), per.get(r["id"], {})
            r.update({"parse_ok": p is not None, "mode": p[0] if p else None, "command": p[1] if p else None,
                      "valid": bool(s.get("valid")), "match": bool(s.get("match")), "error": s.get("error")})
            fh.write(json.dumps(r) + "\n")
    tokens = sum(r["completion_tokens"] for r in recs)
    summary = {"stage": name, "in_flight": in_flight, "thinking": thinking, "requests": len(plan),
               "completed": len(recs), "errors": errors, "t_start": round(t_start, 3), "t_end": round(t_end, 3),
               "wall_s": round(t_end - t_start, 1), "completion_tokens": tokens,
               "client_tokens_per_s": round(tokens / max(t_end - t_start, 1e-9), 1),
               "attempt": args.attempt}
    if errors:                                   # the driver retries a failed stage under the next attempt
        (run / ("stage_%s.failed.json" % name)).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
        print("stage %s: %d of %d requests failed" % (name, len(errors), len(plan)), flush=True)
        return 1
    summary_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("stage %s: %d tokens in %.0f s = %.1f tokens/s (client side)" % (
        name, tokens, summary["wall_s"], summary["client_tokens_per_s"]), flush=True)
    return 0


# ------------------------------------------------------------ the report (CPU)

def parse_stats(text, year):
    """vLLM's periodic stats lines -> [{t (epoch, local time), prompt, gen, running, waiting, preemptions, kv}]."""
    out = []
    for m in STATS_RE.finditer(text):
        t = datetime.datetime.strptime("%d-%s" % (year, m.group(1)), "%Y-%m-%d %H:%M:%S").timestamp()
        out.append({"t": t, "prompt": float(m.group(2)), "gen": float(m.group(3)), "running": int(m.group(4)),
                    "waiting": int(m.group(5)), "preemptions": int(m.group(6) or 0), "kv": float(m.group(7))})
    return out


def by_running(stats, windows):
    """Generation tokens/s by sequences running, over the stats intervals inside the windows."""
    out = {}
    for s in stats:
        if any(a + 5 <= s["t"] <= b for a, b in windows) and s["running"] > 0:
            out.setdefault(s["running"], []).append(s["gen"])
    return {n: {"intervals": len(v), "median": round(statistics.median(v), 1)} for n, v in sorted(out.items())}


def gate(runs, top=8):
    """(G, the running count it was read at, verdict)."""
    ok = [n for n, v in runs.items() if v["intervals"] >= MIN_INTERVALS]
    if not ok:
        return None, None, "not on this card"
    at = top if top in ok else max(ok)
    g = runs[at]["median"]
    verdict = "fits one window" if g >= GATE_ONE else "two windows" if g >= GATE_TWO else "not on this card"
    return g, at, verdict


def build(run=RUN):
    run = pathlib.Path(run)
    attempts = load_jsonl(str(run / "attempts.jsonl")) if (run / "attempts.jsonl").exists() else []
    used = next((a for a in reversed(attempts) if a.get("up")), None)
    stages = {s: json.loads((run / ("stage_%s.json" % s)).read_text(encoding="utf-8"))
              for s in STAGES if (run / ("stage_%s.json" % s)).exists()}
    rows = {s: load_jsonl(str(run / ("rows_%s.jsonl" % s))) for s in stages}
    prompts = load_jsonl(str(run / "prompts.jsonl"))
    rep = {"attempts": attempts, "used_attempt": used["attempt"] if used else None, "stages": stages}
    stats, kv = [], {}
    for a in attempts:
        log = run / ("vllm_%d.log" % a["attempt"])
        if not log.exists():
            continue
        text = log.read_text(encoding="utf-8", errors="replace")
        year = datetime.datetime.fromtimestamp(a.get("t_start", time.time())).year
        stats += parse_stats(text, year)
        m, c = KV_TOKENS_RE.search(text), KV_CONC_RE.search(text)
        if m:
            kv[a["attempt"]] = {"kv_tokens": int(m.group(1).replace(",", "")),
                                "max_model_len": int(c.group(1).replace(",", "")) if c else None,
                                "max_concurrency": float(c.group(2)) if c else None}
    windows = [(s["t_start"], s["t_end"]) for name, s in stages.items() if name in THINK_STAGES]
    runs = by_running(stats, windows)
    g, at, verdict = gate(runs, max(n for n, _ in ATTEMPTS.values()) if not used else ATTEMPTS[used["attempt"]][1])
    if not used:
        verdict = "not on this card"
    rep.update({"kv": kv, "by_running": runs, "G": g, "G_at_running": at, "verdict": verdict,
                "preemptions": max((s["preemptions"] for s in stats), default=0)})
    first = {r["k"]: r for s in THINK_STAGES for r in rows.get(s, []) if r["draw"] == 0}
    ks = sorted(first)
    rep["solves"] = {
        "rows": len(ks), "vllm": sum(first[k]["match"] for k in ks), "recorded": sum(prompts[k]["recorded"]["match"] for k in ks),
        "both": sum(first[k]["match"] and prompts[k]["recorded"]["match"] for k in ks),
        "closed_vllm": sum(first[k]["closed"] for k in ks), "closed_recorded": sum(prompts[k]["recorded"]["closed"] for k in ks),
        "think_median_vllm": statistics.median([first[k]["think_tokens"] for k in ks]) if ks else None,
        "think_median_recorded": statistics.median([prompts[k]["recorded"]["think_tokens"] for k in ks]) if ks else None}
    plain = {s: rows[s] for s in ("D1", "D8") if s in rows}
    if "D1" in plain:
        secs = sorted(r["seconds"] for r in plain["D1"])
        rep["plain_latency"] = {"p50": round(statistics.median(secs), 2),
                                "p95": round(secs[min(len(secs) - 1, int(0.95 * len(secs)))], 2),
                                "tokens_median": statistics.median(r["completion_tokens"] for r in plain["D1"])}
    if "D8" in stages:
        rep["plain_8_tokens_per_s"] = stages["D8"]["client_tokens_per_s"]
    return rep


MEANING = {
    "fits one window": "R1 can run on this card in one night with this engine; it still needs its own addendum and the owner's go.",
    "two windows": "R1 runs over two nights or on fewer rows; that is the owner's call.",
    "not on this card": "R1 needs another engine or a larger card (a §3 deviation), or Part 2 stops; that is the owner's call.",
}


def render(rep):
    L = ["# R0: vLLM and the base 27B's thinking on this card", "",
         "By `preregistration.md` beside this file; built by `scripts/engine27b/probe.py report`.", ""]
    L += ["| attempt | flags | up | load s | card memory once up |", "|---|---|---|---|---|"]
    for a in rep["attempts"]:
        L.append("| %d | `%s` | %s | %s | %s |" % (a["attempt"], " ".join(ATTEMPTS[a["attempt"]][0]) or "default",
                                                   "yes" if a.get("up") else "no", a.get("load_s", "—"),
                                                   ("%s MiB" % a["memory_used_mib"]) if a.get("memory_used_mib") else "—"))
    for k, v in rep["kv"].items():
        L.append("")
        L.append("Attempt %s: KV cache %s tokens; vLLM's maximum concurrency at %s tokens per request: %sx." % (
            k, format(v["kv_tokens"], ","), format(v["max_model_len"] or 0, ","), v["max_concurrency"]))
    L += ["", "## Generated tokens per second, by sequences running (stages A–C, vLLM's 5-second stats)", "",
          "| running | intervals | median tokens/s |", "|---|---|---|"]
    for n, v in rep["by_running"].items():
        L.append("| %d | %d | %.1f |" % (n, v["intervals"], v["median"]))
    L += ["", "| stage | in flight | requests | tokens | wall s | tokens/s (client) |", "|---|---|---|---|---|---|"]
    for s, v in rep["stages"].items():
        L.append("| %s | %d | %d | %d | %.0f | %.1f |" % (s, v["in_flight"], v["completed"], v["completion_tokens"],
                                                         v["wall_s"], v["client_tokens_per_s"]))
    g = ("**G = %.1f tokens/s** with %d running" % (rep["G"], rep["G_at_running"])) if rep["G"] is not None else "**G: no reading**"
    note = "" if rep["G_at_running"] in (None, 8) else " (fewer than %d intervals at 8, so read at %d)" % (MIN_INTERVALS, rep["G_at_running"])
    L += ["", "%s%s. Preemptions: %d." % (g, note, rep["preemptions"]), "",
          "**Verdict, by the pre-registered rule: %s.** %s" % (rep["verdict"], MEANING[rep["verdict"]])]
    s = rep["solves"]
    if s["rows"]:
        L += ["", "## The same rows, beside the recorded outcomes (descriptive; one sample each)", "",
              "| | solved | closed | median thinking tokens |", "|---|---|---|---|",
              "| vLLM (first samples) | %d of %d | %d | %s |" % (s["vllm"], s["rows"], s["closed_vllm"], s["think_median_vllm"]),
              "| recorded (HF, one row at a time) | %d of %d | %d | %s |" % (s["recorded"], s["rows"], s["closed_recorded"], s["think_median_recorded"]),
              "", "Solved by both: %d." % s["both"]]
    if rep.get("plain_latency"):
        p = rep["plain_latency"]
        L += ["", "## Zero-shot answers, thinking off (descriptive: a lower bound on a served 27B answer)", "",
              "One in flight: p50 %.2f s, p95 %.2f s, median %s tokens. Eight in flight: %s tokens/s." % (
                  p["p50"], p["p95"], p["tokens_median"], rep.get("plain_8_tokens_per_s", "—"))]
    return "\n".join(L) + "\n"


def cmd_report(args):
    rep = build(args.run)
    run = pathlib.Path(args.run)
    (run / "probe.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (run / "probe.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--run", default=str(RUN))
    a.set_defaults(fn=cmd_prompts)
    a = sub.add_parser("serve-args")
    a.add_argument("--run", default=str(RUN))
    a.add_argument("--attempt", type=int, choices=sorted(ATTEMPTS), required=True)
    a.add_argument("--port", type=int, default=8013)
    a.set_defaults(fn=cmd_serve_args)
    a = sub.add_parser("stage")
    a.add_argument("name", choices=list(STAGES))
    a.add_argument("--run", default=str(RUN))
    a.add_argument("--port", type=int, default=8013)
    a.add_argument("--attempt", type=int, default=1)
    a.add_argument("--timeout", type=float, default=5.0, help="the sandbox's, per command")
    a.set_defaults(fn=cmd_stage)
    a = sub.add_parser("report")
    a.add_argument("--run", default=str(RUN))
    a.set_defaults(fn=cmd_report)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
