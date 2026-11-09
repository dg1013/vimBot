#!/usr/bin/env python3
"""R1: the labeling pass on R0's engine (runs/think_label_2026-11-01/preregistration.md, addendum 2).

R1 runs over two nights once OPD finishes, paused each morning until it is resumed. The pass is
label.py's as built: its rows and plans, arms and templates, orders and seeds, judging and records, and
report.py. Three things change, the plan's R1 (runs/teacher_channels_2026-11-02/plan.md):

- the thinking arm's cap is 8,192 tokens (reasoning plus answer), the cap at which TB helps;
- the loop breaker is on (scripts/think_loop's defaults);
- the engine is R0's (scripts/engine27b): vLLM serving the base 27B, quantized at load to NF4 with
  double quantization (baseline.NF4's settings), text only, 8 sequences in flight.

vLLM batches without padding, so the padding guard and the halving on OOM have nothing to guard.
Prompts are rendered by the same chat template and sent as token ids. The returned ids are trimmed at
the first stop and decoded by the same tokenizer, so a record is what label.py writes for the same
tokens (test_serve.py runs both on one script). Each row is a batch of one, seeded
label.batch_seed(arm, row).

The loop breaker runs when a row returns. The detector is replayed over its reasoning line by line
(scripts/think_loop's calibrate.first_stop), the row is cut at the token where the live check would
have stopped it, and think_loop's break_loop closes and answers it, with a generate that asks the
server. Stopping never changes the tokens before it, so the record is the live breaker's; what the
row generated past the cut is kept as loop.overrun_tokens.

Pause: a PAUSE file in the run folder. No row starts once it exists; rows in flight get DRAIN_S
seconds, then the pass exits RC_PAUSED, and the next run resumes from the rows on disk.

    python scripts/think_label/serve.py trigger                  # exit 0 iff think_long's TB8k is helps
    python scripts/think_label/serve.py prompts                  # CPU: both arms' prompts -> <run>/prompts_vllm.jsonl
    python scripts/think_label/serve.py serve-args --attempt 2   # the `vllm serve` arguments, one per line
    python scripts/think_label/serve.py run --attempt 2 --server-log <log>   # both arms against the server
"""
import argparse
import concurrent.futures
import datetime
import importlib.util
import json
import os
import pathlib
import statistics
import sys
import threading
import time
import urllib.error

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                   # noqa: E402
from prompt import messages_for_case                             # noqa: E402


def load(name, path):
    """Import a module by path under a unique name: several folders have same-named files."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L = load("think_label_label", HERE / "label.py")
P = load("engine27b_probe", REPO / "scripts" / "engine27b" / "probe.py")
CAL = load("think_loop_calibrate", REPO / "scripts" / "think_loop" / "calibrate.py")
T, LB = L.T, L.LB

RUN = L.RUN
CONTROL, THINK = "nothink-retr8", "think-retr8"
CAP, PLAIN_CAP = 8192, 192       # the thinking arm's reasoning plus answer; the control's §6 answers
ATTEMPTS = (2, 3)                # R0's: attempt 1 left no room for the KV cache; 2 came up; 3 is 4 sequences
PORT = 8014
THINK_LONG_REPORT = REPO / "runs" / "think_long_2026-11-01" / "report.json"
TRIGGER = "TB8k"
PROMPTS, META = "prompts_vllm.jsonl", "prompts_vllm_meta.json"
PAUSE = "PAUSE"
DRAIN_S = 600
RC_PAUSED, RC_SANITY = 6, 7
SANITY_ROWS = 16                 # an arm whose first 16 rows all fail to parse stops: something is broken
MAX_ERRORS = 3                   # failed requests before the pass stops, so the driver restarts the server
SLOW_PER_SEQ = 3.0               # generated tokens/s per running sequence; R0 measured 19.0 at 1, 10.8 at 8
SLOW_WINDOW_S, GUARD_EVERY_S, HEARTBEAT_S = 600, 300, 300
REQUEST_TIMEOUT = 3600


# ------------------------------------------------------------ the trigger and the prompts (CPU)

def trigger(report_path=THINK_LONG_REPORT, key=TRIGGER):
    """think_long's verdict for TB at 8,192 tokens ("helps", "tied", "hurts"), or None if it has none."""
    path = pathlib.Path(report_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("verdicts", {}).get(key)


def cmd_trigger(args):
    tb = trigger(args.report)
    if tb == "helps":
        print("TB at 8,192 tokens is helps: the labeling pass runs")
        return 0
    print("TB at 8,192 tokens is %s: the labeling pass does not run" % (tb or "missing"))
    return L.RC_SKIP


def cmd_prompts(args):
    """Both arms' prompts as label.py renders them (texts and token ids), and the server's max length."""
    run = pathlib.Path(args.run)
    rows = json.loads((run / "rows.json").read_text(encoding="utf-8"))["rows"]
    train = {c["id"]: c for c in load_jsonl(str(L.TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(run / "plan_retr8.jsonl"))}
    tok, info, meta = P.engine_info()
    longest = 0
    tmp = run / (PROMPTS + ".partial")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            c = train[r["id"]]
            exs = [train[j] for j in plan[r["id"]]["examples"]]
            rec = {"id": r["id"], "k_examples": len(exs)}
            for arm, model in L.ARMS.items():
                text = tok.apply_chat_template(messages_for_case(c, exs), add_generation_prompt=True, tokenize=False,
                                               **info["kwargs"][model])
                rec[arm] = {"text": text, "ids": tok(text)["input_ids"]}
            longest = max(longest, len(rec[THINK]["ids"]))
            fh.write(json.dumps(rec) + "\n")
    tmp.rename(run / PROMPTS)
    meta.update({"rows": len(rows), "longest_prompt": longest, "cap": CAP, "plain_cap": PLAIN_CAP,
                 "max_model_len": -(-(longest + CAP) // 256) * 256, "kwargs": info["kwargs"]})
    (run / META).write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("%d rows, both arms: thinking prompts up to %d tokens; max-model-len %d"
          % (len(rows), longest, meta["max_model_len"]))
    return 0


def cmd_serve_args(args):
    meta = json.loads((pathlib.Path(args.run) / META).read_text(encoding="utf-8"))
    print("\n".join(P.serve_args(args.attempt, args.port, meta)))
    return 0


# ------------------------------------------------------------ one row against the server

class LockedTokenizer:
    """The tokenizer behind a lock: worker threads encode and decode, and a fast tokenizer's state is
    not made to be shared across threads."""

    def __init__(self, tok):
        self.tok, self.lock = tok, threading.Lock()

    def __call__(self, *a, **k):
        with self.lock:
            return self.tok(*a, **k)

    def decode(self, *a, **k):
        with self.lock:
            return self.tok.decode(*a, **k)


def body(ids, max_tokens, seed, meta):
    """R0's request (probe.request_body), with this pass's prompt, cap and seed."""
    return {"model": P.SERVED_NAME, "prompt": ids, "max_tokens": max_tokens, **meta["sampling"], "seed": seed,
            "stop_token_ids": meta["stop_ids"], "skip_special_tokens": False, "return_token_ids": True}


def complete(port, ids, max_tokens, seed, meta, timeout=REQUEST_TIMEOUT):
    """(the generated token ids, a stop included; the finish reason; the prompt's tokens)."""
    resp = P.post(port, body(ids, max_tokens, seed, meta), timeout=timeout)
    ch = resp["choices"][0]
    return list(ch.get("token_ids") or []), ch.get("finish_reason"), resp["usage"]["prompt_tokens"]


def server_generate(port, meta):
    """scripts/think27b's generate contract, served: what break_loop calls for the forced answer."""
    stops = set(meta["stop_ids"])

    def generate(model, tok, texts, adapter_on, max_new, info, seed, stopping_criteria=None):
        outs, lens = [], []
        for text in texts:
            ids, _, _ = complete(port, tok(text)["input_ids"], max_new, seed, meta)
            outs.append(T.trim(ids, stops, None))
            lens.append(len(ids))
        return outs, lens
    return generate


def cut_at(tok, ids, reasoning, settings):
    """How many generated tokens the live breaker lets this row have, or None if it never fires.
    LB.stopping checks a row on each token holding a newline, so it fires on the token holding the
    newline of the first line at which the detector finds a loop (calibrate.first_stop)."""
    lines = LB.lines_of(reasoning)
    i, _ = CAL.first_stop(lines, settings)
    if i is None:
        return None
    end = lines[i][1]
    lo, hi = 1, len(ids)
    while lo < hi:                              # the fewest tokens whose text reaches that newline
        mid = (lo + hi) // 2
        if len(tok.decode(ids[:mid], skip_special_tokens=False)) >= end:
            hi = mid
        else:
            lo = mid + 1
    return lo


def run_row(arm, rid, thinking, ctx):
    """One row through the server, up to judging: what label.py computes for a batch of one."""
    p, meta, lb, tok = ctx["prompts"][rid][arm], ctx["meta"], ctx["lb"], ctx["tok"]
    max_new = CAP if thinking else PLAIN_CAP
    seed = L.batch_seed(arm, rid)
    t0 = time.time()
    raw, finish, prompt_tokens = complete(ctx["port"], p["ids"], max_new, seed, meta)
    secs = time.time() - t0
    o, n = T.trim(raw, set(meta["stop_ids"]), None), len(raw)
    reasoning, answer, closed = T.split_reasoning(tok.decode(o, skip_special_tokens=False), thinking)
    loop = None
    if lb and thinking:
        m = cut_at(tok, o, reasoning, lb)
        if m is not None and m < max_new:       # label.py breaks a row that stopped short of its cap
            overrun = len(o) - m
            o, n = o[:m], m
            reasoning = tok.decode(o, skip_special_tokens=False)
            forced, loop = LB.break_loop(server_generate(ctx["port"], meta), None, tok, p["text"], reasoning, True,
                                         None, seed, lb)
            if loop:
                answer, closed = forced, True
                loop["overrun_tokens"] = overrun
    return {"o": o, "n": n, "reasoning": reasoning, "answer": answer, "closed": closed, "loop": loop,
            "seconds": secs, "finish_reason": finish, "prompt_tokens": prompt_tokens, "generated_tokens": len(raw),
            "seed": seed, "max_new": max_new, "t0": t0}


def record(arm, case, flag, k_examples, res, parsed, per, thinking, ctx):
    """label.py's record for a batch of one, then the engine's own fields."""
    p, r = parsed[case["id"]], per.get(case["id"], {})
    o, n, closed = res["o"], res["n"], res["closed"]
    return {
        "arm": arm, "id": case["id"], "tier": case.get("tier"), "category": case.get("category"),
        "in_cliff": flag["in_cliff"], "in_never": flag["in_never"],
        "k_examples": k_examples, "new_tokens": len(o),
        "think_tokens": T.think_tokens(o, ctx["meta"]["think_end_id"]) if thinking else 0,
        "closed": closed, "hit_cap": n >= res["max_new"] and not closed,
        "answer": res["answer"], "reasoning": res["reasoning"], "parse_ok": p is not None,
        "mode": p[0] if p else None, "command": p[1] if p else None,
        "valid": bool(r.get("valid")), "match": bool(r.get("match")), "error": r.get("error"),
        "batch_seconds": round(res["seconds"], 1), "batch_size": 1, "steps": n,
        "seed": res["seed"], "max_new_tokens": res["max_new"], "loop_break": ctx["lb"] if thinking else None,
        "forced_close": res["loop"] is not None, "loop": res["loop"],
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "engine": "vllm", "server_attempt": ctx["attempt"], "finish_reason": res["finish_reason"],
        "prompt_tokens": res["prompt_tokens"], "generated_tokens": res["generated_tokens"],
        "request_t0": round(res["t0"], 1)}


# ------------------------------------------------------------ the paging guard

def per_seq_rate(log_path, now, window=SLOW_WINDOW_S, year=None):
    """(the median generated tokens/s per running sequence over the last `window` seconds, intervals)
    from the server's 5-second stats lines; (None, n) when fewer than half the window's intervals."""
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 4_000_000))
            text = fh.read().decode("utf-8", "replace")
    except OSError:
        return None, 0
    year = year or datetime.datetime.fromtimestamp(now).year
    st = [s for s in P.parse_stats(text, year) if now - window <= s["t"] <= now and s["running"] > 0]
    if len(st) < window / 10:
        return None, len(st)
    return statistics.median(s["gen"] / s["running"] for s in st), len(st)


# ------------------------------------------------------------ the pass

def run_arm(arm, ids, ctx):
    """One arm's rows through the server, appended as they finish. 0 when all are written, else
    RC_PAUSED, RC_SANITY, think27b's RC_SLOW, or 1 after MAX_ERRORS failed requests."""
    thinking = ctx["thinking"][arm]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=ctx["in_flight"])
    pending, inflight = list(reversed(ids)), {}
    written = parsed_ok = errors = tokens = slow = rc = 0
    paused_at = None
    t_start = last_guard = last_beat = time.time()
    with (ctx["run"] / ("rows_%s.jsonl" % arm)).open("a", encoding="utf-8") as fh:
        while pending or inflight:
            if paused_at is None and (ctx["run"] / PAUSE).exists():
                paused_at = time.time()
                print("PAUSE: no new rows; the %d in flight get up to %d s" % (len(inflight), DRAIN_S), flush=True)
            while pending and paused_at is None and len(inflight) < ctx["in_flight"]:
                rid = pending.pop()
                inflight[pool.submit(run_row, arm, rid, thinking, ctx)] = rid
            if not inflight:
                break
            done, _ = concurrent.futures.wait(inflight, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
            for f in done:
                rid = inflight.pop(f)
                try:
                    res = f.result()
                except (urllib.error.URLError, OSError, KeyError, IndexError, ValueError) as exc:
                    errors += 1
                    print("  %s %s: the request failed (%r); it reruns later" % (arm, rid, exc), flush=True)
                    continue
                case = ctx["train"][rid]
                parsed, per = T.judge([case], [res["answer"]], ctx["timeout"])
                rec = record(arm, case, ctx["flags"][rid], ctx["prompts"][rid]["k_examples"], res, parsed, per,
                             thinking, ctx)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                written += 1
                parsed_ok += rec["parse_ok"]
                tokens += res["generated_tokens"]
                print("  %s %d/%d %s: %d tokens in %.0f s, %s%s, match %s" % (
                    arm, written, len(ids), rid, res["generated_tokens"], res["seconds"],
                    "closed" if rec["closed"] else "open", ", loop broken" if rec["forced_close"] else "",
                    rec["match"]), flush=True)
            now = time.time()
            if written >= SANITY_ROWS and not parsed_ok:
                print("stopped: none of %s's first %d rows parsed" % (arm, written), flush=True)
                rc = RC_SANITY
                break
            if errors >= MAX_ERRORS:
                print("stopped: %d failed requests" % errors, flush=True)
                rc = 1
                break
            if thinking and ctx["server_log"] and now - last_guard >= GUARD_EVERY_S:
                last_guard = now
                rate, n = per_seq_rate(ctx["server_log"], now)
                slow = slow + 1 if rate is not None and rate < SLOW_PER_SEQ else 0
                if slow:
                    print("  slow: %.2f tokens/s per running sequence over %d intervals (%d in a row)"
                          % (rate, n, slow), flush=True)
                if slow >= 2:
                    print("stopped: the card is paging", flush=True)
                    rc = T.RC_SLOW
                    break
            if now - last_beat >= HEARTBEAT_S:
                last_beat = now
                print("  heartbeat: %s %d written, %d in flight, %d to go, %d tokens in %.0f min" % (
                    arm, written, len(inflight), len(pending), tokens, (now - t_start) / 60), flush=True)
            if paused_at is not None and now - paused_at >= DRAIN_S:
                print("drain over: %d rows in flight are dropped and rerun later" % len(inflight), flush=True)
                break
    pool.shutdown(wait=False, cancel_futures=True)
    secs = time.time() - t_start
    rc = rc or (RC_PAUSED if paused_at is not None else 0)
    with (ctx["run"] / "serve_runs.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"arm": arm, "t_start": round(t_start, 1), "seconds": round(secs, 1), "rows": written,
                             "tokens": tokens, "errors": errors, "rc": rc, "attempt": ctx["attempt"],
                             "in_flight": ctx["in_flight"]}) + "\n")
    print("%s: %d rows, %d tokens in %.1f min (%.1f tokens/s), rc %d" % (
        arm, written, tokens, secs / 60, tokens / max(secs, 1e-9), rc), flush=True)
    return rc


def cmd_run(args):
    run = pathlib.Path(args.run)
    if (run / PAUSE).exists():
        print("PAUSE is set in %s: remove it to resume" % run, flush=True)
        return RC_PAUSED
    rows = json.loads((run / "rows.json").read_text(encoding="utf-8"))["rows"]
    flags = {r["id"]: r for r in rows}
    meta = json.loads((run / META).read_text(encoding="utf-8"))
    arms = [a for a in L.ARMS if a in args.arms]
    lb = LB.resolve(run, args.loop_break)
    todo = {a: [i for i in L.order_for(a, flags) if i not in T.done_ids(run / ("rows_%s.jsonl" % a))] for a in arms}
    if args.limit:
        todo = {a: t[:args.limit] for a, t in todo.items()}
    print("loop breaker: %s; %s" % (lb or "off", ", ".join("%s %d to do" % (a, len(t)) for a, t in todo.items())),
          flush=True)
    if not any(todo.values()):
        return 0
    tok, _, now_meta = P.engine_info()
    for k in ("sampling", "stop_ids", "think_end_id"):
        if now_meta[k] != meta[k]:
            sys.exit("%s is %s now but %s when the prompts were built" % (k, now_meta[k], meta[k]))
    ctx = {"run": run, "meta": meta, "lb": lb, "tok": LockedTokenizer(tok), "port": args.port,
           "attempt": args.attempt, "in_flight": args.in_flight or P.ATTEMPTS[args.attempt][1],
           "server_log": args.server_log, "timeout": args.timeout, "flags": flags,
           "train": {c["id"]: c for c in load_jsonl(str(L.TRAIN))},
           "prompts": {r["id"]: r for r in load_jsonl(str(run / PROMPTS))},
           "thinking": {a: bool(meta["kwargs"][L.ARMS[a]].get("enable_thinking")) for a in arms}}
    for arm in arms:
        if todo[arm]:
            rc = run_arm(arm, todo[arm], ctx)
            if rc:
                return rc
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trigger", help="exit 0 iff think_long's TB reading at 8,192 tokens is helps (else exit 3)")
    t.add_argument("--report", default=str(THINK_LONG_REPORT))
    t.set_defaults(fn=cmd_trigger)
    p = sub.add_parser("prompts", help="CPU: both arms' prompts -> <run>/prompts_vllm.jsonl and its meta")
    p.add_argument("--run", default=str(RUN))
    p.set_defaults(fn=cmd_prompts)
    s = sub.add_parser("serve-args", help="the `vllm serve` arguments for a server attempt, one per line")
    s.add_argument("--run", default=str(RUN))
    s.add_argument("--attempt", type=int, choices=ATTEMPTS, required=True)
    s.add_argument("--port", type=int, default=PORT)
    s.set_defaults(fn=cmd_serve_args)
    r = sub.add_parser("run", help="both arms against the running server, resumable, until done or PAUSE")
    r.add_argument("--run", default=str(RUN))
    r.add_argument("--port", type=int, default=PORT)
    r.add_argument("--attempt", type=int, choices=ATTEMPTS, required=True, help="the server attempt (recorded)")
    r.add_argument("--server-log", default=None, help="the server's log, for the paging guard")
    r.add_argument("--arms", nargs="+", default=list(L.ARMS), choices=list(L.ARMS))
    r.add_argument("--in-flight", type=int, default=None, help="default: the attempt's --max-num-seqs")
    r.add_argument("--limit", type=int, default=None, help="at most N rows per arm")
    r.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    r.add_argument("--loop-break", action=argparse.BooleanOptionalAction, default=None,
                   help="scripts/think_loop's breaker; default: on for a new --run directory")
    r.set_defaults(fn=cmd_run)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)      # rows abandoned at a pause or a stop would otherwise hold the exit until they finish
