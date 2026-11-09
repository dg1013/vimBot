#!/usr/bin/env python3
"""The 5-of-8 ask rule on the trained 27B, one stage per call, each resumable.

By runs/ask27b_2026-11-06/preregistration.md. The rule is the one real use runs
(scripts/serve/roundtrip.py): ask when the most common result of 8 sampled answers holds at most 5
of them, else give the §6 answer. A sample that fails, changes nothing or does not parse counts
against agreement. Where it asks, the model writes its question, the audited simulated user replies
(scripts/clarify/simulate.py, unchanged, between `placebo` and `answer`), and the model answers.

    views     CPU   the 32 samples per row on disk, through the box: one result per sample
    question  27B   the question it asks, on every row: "ask:" prefilled, its first line
    placebo   27B   its answer after a reply that carries no information
    answer    27B   its answer after the simulated user's reply
    timing    27B   the round trip itself, one request at a time, on the 4B's 200 timing rows
    judge     CPU   every answer after a reply, through the box
    gate      CPU   is a dry run fit to go on?

Every row gets every turn, not only the rows the rule asks on: which rows it asks on depends on the
draw of samples, and always asking and the placebo are read from the same turns. Prompts come from
the functions the round trip calls, through the same client, so they are the prompts real use sends.

Rows run in a mixed order: val.jsonl is not shuffled and its head is light, which would read as a
falling pace. Exit codes, as scripts/serve27b/run.sh reads them: 3 a gate failed, 4 paused, 5 the
pace fell.
"""
import argparse
import hashlib
import json
import math
import pathlib
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "clarify",
          REPO / "scripts" / "readings", REPO / "scripts" / "serve27b"):
    sys.path.insert(0, str(p))
import pass27b as A                                               # noqa: E402
import readings_protocol as R                                     # noqa: E402
from protocol import (ASK_PREFIX, PLACEBO_REPLY, QUESTION_MAX_TOKENS,   # noqa: E402
                      cut_question, leak_flags, parse_turn1, parse_turn2,
                      quotes_result, turn1_messages, turn2_messages)

LABEL = "served27b"
SERVE27 = REPO / "runs" / "serve27b_2026-11-05"      # the 27B's samples and greedy answers, as served
ANSWER_MAX_TOKENS = 192                              # roundtrip.ANSWER_MAX_TOKENS
WORKERS = 8                                          # as the greedy stage ran
TIMING_KS = (4, 8)
TIMING_SEED = 20261013                               # scripts/serve/timing.py's: the rows the 4B was timed on
GATE = 0.75                                          # of a dry run's rows: asked, replied to, parsed


def settings_timeout():
    from settings import load_settings
    return load_settings(None)["executor"]["timeout"]


def result_key(case, entry):
    """A sample's result: a digest of the buffer it leaves, or None if it failed or changed nothing,
    which counts against agreement (roundtrip.sample_evidence)."""
    if R.reading_status(case, entry) != "ok":
        return None
    return hashlib.sha1("\n".join(R.norm_key(entry["buffer_after"])).encode("utf-8")).hexdigest()[:16]


def sample_results(case, samples, executed):
    """One result per sample, in order. executed: {(mode, command): the executor's entry}."""
    return [None if s is None else result_key(case, executed.get(tuple(s))) for s in samples]


def distinct(samples):
    return list(dict.fromkeys(tuple(s) for s in samples if s is not None))


def cmd_views(args):
    """No model: every distinct sample once through the validation path, as the round trip does."""
    from score import execute
    timeout = settings_timeout()
    samples = {r["id"]: r for r in A.load_jsonl(SERVE27 / "samples.jsonl")}
    with (args.run / "views.jsonl").open("w") as fh:
        for sp in A.SPLITS:
            cs = [c for c in A.cases(sp, args.limit) if c["id"] in samples]
            answers = {c["id"]: {"c%d" % n: {"parse_ok": True, "mode": m, "command": cmd}
                                 for n, (m, cmd) in enumerate(distinct(samples[c["id"]]["samples"]))} for c in cs}
            t0 = time.time()
            executed = execute(cs, answers, timeout=timeout, workers=8)
            for c in cs:
                i = c["id"]
                by_key = {(a["mode"], a["command"]): executed[(i, name)] for name, a in answers[i].items()}
                fh.write(json.dumps({"id": i, "split": sp, "results": sample_results(c, samples[i]["samples"], by_key)}) + "\n")
            print("%s: %d rows, %d commands through the box in %.0f s" % (
                sp, len(cs), sum(len(a) for a in answers.values()), time.time() - t0), flush=True)
    return 0


def by_id(limit):
    return {c["id"]: (sp, c) for sp in A.SPLITS for c in A.cases(sp, limit)}


def run_rows(run, name, todo, work, workers=WORKERS):
    """work(split, case) -> a row, over `todo` in the mixed order, appended to <name>.jsonl and
    resumable. 0 done, 4 paused, 5 the pace fell. An endpoint that dies raises, and the driver
    restarts it and resumes."""
    path = run / (name + ".jsonl")
    done = {r["id"] for r in A.load_jsonl(path)} if path.exists() else set()
    left = [i for i in A.mix(todo) if i not in done]
    print("%s: %d rows to do, %d done" % (name, len(left), len(done)), flush=True)
    stop, slow = run / "PAUSE", threading.Event()

    def one(i):
        if stop.exists() or slow.is_set():
            return None
        t0 = time.perf_counter()
        return dict(work(*todo[i]), row_s=round(time.perf_counter() - t0, 3))

    t0, seconds = time.time(), []
    with path.open("a") as fh, ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, left):
            if r is None:
                continue
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            seconds.append(r["row_s"])
            if A.pace_fell(seconds):
                slow.set()
            if len(seconds) % 100 == 0 or len(seconds) == len(left):
                el = time.time() - t0
                print("  %d / %d rows, %.0f s, %.2f s per row" % (len(seconds), len(left), el, el / len(seconds)), flush=True)
    if len(seconds) == len(left):
        print("%s: %d rows in %.0f s" % (name, len(left), time.time() - t0), flush=True)
        return 0
    if stop.exists():
        print("paused after %d of %d rows" % (len(seconds), len(left)), flush=True)
        return 4
    print("the pace fell to %.1f s per row from %.1f; yielding the card" % (
        statistics.median(seconds[-10:]), statistics.median(seconds[:25])), flush=True)
    return 5


def question_row(e, sp, case):
    q = e.complete(e.render(turn1_messages(case), prefill=ASK_PREFIX), QUESTION_MAX_TOKENS, stop=["\n"])
    raw = q["texts"][0]
    return {"id": case["id"], "label": LABEL, "split": sp,
            "question": {"text": cut_question(raw), "raw": raw, "output_tokens": q["output_tokens"][0],
                         "latency_s": q["latency_s"]}}


def answer_fields(text):
    parsed = parse_turn2(text)
    again = parse_turn1(text)
    return {"completion": text, "parse_ok": parsed is not None, "mode": parsed[0] if parsed else None,
            "command": parsed[1] if parsed else None, "second_ask": bool(again and again[0] == "ask"),
            "think": "<think>" in text}


def answer_row(e, sp, case, question, reply):
    r = e.complete(e.render(turn2_messages(case, question, reply)), ANSWER_MAX_TOKENS)
    return dict(answer_fields(r["texts"][0]), id=case["id"], split=sp, output_tokens=r["output_tokens"][0],
                prompt_tokens=r["prompt_tokens"], latency_s=r["latency_s"])


def asked(run):
    """{id: the question}, rows where the model wrote one. Real use answers directly where it did not."""
    return {r["id"]: r["question"]["text"] for r in A.load_jsonl(run / "question.jsonl") if r["question"]["text"]}


def cmd_question(args):
    e = A.endpoint27()
    return run_rows(args.run, "question", by_id(args.limit), lambda sp, c: question_row(e, sp, c))


def cmd_placebo(args):
    e, q = A.endpoint27(), asked(args.run)
    todo = {i: x for i, x in by_id(args.limit).items() if i in q}
    return run_rows(args.run, "placebo", todo, lambda sp, c: answer_row(e, sp, c, q[c["id"]], PLACEBO_REPLY))


def cmd_answer(args):
    e, q = A.endpoint27(), asked(args.run)
    replies = {r["id"]: r["reply"] for r in A.load_jsonl(args.run / "replies.jsonl")}
    todo = {i: x for i, x in by_id(args.limit).items() if i in q}
    missing = [i for i in todo if i not in replies]
    if missing:
        sys.exit("%d rows have no reply (first: %s)" % (len(missing), missing[0]))
    return run_rows(args.run, "answer", todo, lambda sp, c: answer_row(e, sp, c, q[c["id"]], replies[c["id"]]))


def timing_rows(val, n, seed=TIMING_SEED):
    return random.Random(seed).sample(val, min(n, len(val)))


def cmd_timing(args):
    """The round trip as real use runs it, one request at a time: scripts/serve/timing.py's protocol
    on its seeded rows, not the head of val."""
    import measure as M
    from prompt import messages_for_case
    from roundtrip import RoundTrip
    from timing import stats
    e = A.endpoint27()
    rows = timing_rows(A.cases("val", args.limit), args.timing_n)
    replies = {r["id"]: r for r in A.load_jsonl(args.run / "replies.jsonl")}
    stop = args.run / "PAUSE"
    for c in rows[:5]:                                               # warm-up
        e.complete(e.render(messages_for_case(c)), ANSWER_MAX_TOKENS)
    out = {"direct": []}
    for c in rows:
        if stop.exists():
            return 4
        t0 = time.perf_counter()
        r = e.complete(e.render(messages_for_case(c)), ANSWER_MAX_TOKENS)
        out["direct"].append({"id": c["id"], "total_s": round(time.perf_counter() - t0, 4), "tokens": r["output_tokens"][0]})
    for k in TIMING_KS:
        rt = RoundTrip(e, samples=k, max_agree=math.floor(5 / 8 * k + 1e-9))
        rt.translate(rows[0]["instruction"], rows[0]["buffer_before"], rows[0].get("cursor", (1, 1)))     # warm-up
        got = []
        for c in rows:
            if stop.exists():
                return 4
            res = rt.translate(c["instruction"], c["buffer_before"], c.get("cursor", (1, 1)))
            row = {"id": c["id"], "type": res["type"], **res["timings"], "tokens": sum(res.get("tokens", {}).values())}
            if res["type"] == "question":
                row["same_question"] = res["question"] == replies.get(c["id"], {}).get("question")
                done = rt.reply(res["dialogue_id"], replies.get(c["id"], {}).get("reply", PLACEBO_REPLY))
                if done.get("type") != "error" or "timings" in done:
                    row["turn2_s"] = done["timings"]["turn2_s"]
                    row["tokens"] += done["tokens"]["turn2"]
            got.append(row)
        out["k%d" % k] = got
        print("k = %d: %d rows, asks %.0f%%" % (k, len(got), 100 * statistics.fmean(r["type"] == "question" for r in got)), flush=True)
    summary = {"n": len(rows), "memory": M.gpu_mem(), "direct": {
        "total_s": stats(r["total_s"] for r in out["direct"]), "tokens": statistics.fmean(r["tokens"] for r in out["direct"])}}
    for k in TIMING_KS:
        rr = out["k%d" % k]
        answers, questions = [r for r in rr if r["type"] != "question"], [r for r in rr if r["type"] == "question"]
        summary["k%d" % k] = {"asks": len(questions) / len(rr), "first_response_s": stats(r["total_s"] for r in rr),
                              "when_it_answers_s": stats(r["total_s"] for r in answers), "when_it_asks_s": stats(r["total_s"] for r in questions),
                              "models_wall_s": stats(r["models_wall_s"] for r in rr), "sandbox_s": stats(r["sandbox_s"] for r in rr),
                              "question_s": stats(r.get("question_s") for r in rr), "turn2_s": stats(r.get("turn2_s") for r in rr),
                              "tokens": statistics.fmean(r["tokens"] for r in rr),
                              "same_question": sum(r.get("same_question", False) for r in questions) / max(1, len(questions))}
    A.write_json(args.run / "timing.json", {"measured": time.strftime("%Y-%m-%d %H:%M"), "summary": summary, "rows": out})
    print("round trip at 8: first response p50 %.2f s" % summary["k8"]["first_response_s"]["p50"], flush=True)
    return 0


def judged_row(case, sp, question, reply, real, placebo, entries):
    """What the report reads for one row: the verdicts, and the leak screen on the reply."""
    out = {"id": case["id"], "split": sp, "question": question, "reply": reply,
           "leak_flags": leak_flags(reply, case), "quotes_result": quotes_result(reply, case)}
    for name, a in (("real", real), ("placebo", placebo)):
        e = entries[name]
        out[name] = {"match": bool(e and e["match"]), "valid": bool(e and e["valid"]), "parse_ok": a["parse_ok"],
                     "second_ask": a["second_ask"], "think": a["think"], "tokens": a["output_tokens"],
                     "mode": a["mode"], "command": a["command"]}
    return out


def cmd_judge(args):
    from score import execute
    timeout = settings_timeout()
    q = asked(args.run)
    replies = {r["id"]: r["reply"] for r in A.load_jsonl(args.run / "replies.jsonl")}
    real = {r["id"]: r for r in A.load_jsonl(args.run / "answer.jsonl")}
    placebo = {r["id"]: r for r in A.load_jsonl(args.run / "placebo.jsonl")}
    with (args.run / "judged.jsonl").open("w") as fh:
        for sp in A.SPLITS:
            cs = [c for c in A.cases(sp, args.limit) if c["id"] in real and c["id"] in placebo]
            answers = {c["id"]: {"real": real[c["id"]], "placebo": placebo[c["id"]]} for c in cs}
            executed = execute(cs, answers, timeout=timeout, workers=8)
            for c in cs:
                i = c["id"]
                row = judged_row(c, sp, q[i], replies[i], real[i], placebo[i],
                                 {n: executed[(i, n)] for n in ("real", "placebo")})
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print("%s: %d rows judged" % (sp, len(cs)), flush=True)
    return 0


def dry_fit(rows, questions, replies, answers, need=GATE):
    """None if a dry run may go on, else why not: the turns must have happened on most rows."""
    if not rows:
        return "no rows"
    checks = (("wrote a question", len(questions)), ("got a reply", sum(1 for r in replies if r.strip())),
              ("gave an answer that parses", sum(1 for a in answers if a["parse_ok"])))
    for what, n in checks:
        if n < need * rows:
            return "%d of %d rows %s" % (n, rows, what)
    return None


def cmd_gate(args):
    rows = len(A.load_jsonl(args.run / "question.jsonl"))
    why = dry_fit(rows, asked(args.run), [r["reply"] for r in A.load_jsonl(args.run / "replies.jsonl")],
                  A.load_jsonl(args.run / "answer.jsonl"))
    print("the dry run is fit to go on" if why is None else "stop: %s" % why, flush=True)
    return 0 if why is None else 3


STAGES = {"views": cmd_views, "question": cmd_question, "placebo": cmd_placebo, "answer": cmd_answer,
          "timing": cmd_timing, "judge": cmd_judge, "gate": cmd_gate}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--run", default=str(REPO / "runs" / "ask27b_2026-11-06"))
    ap.add_argument("--limit", type=int, default=0, help="only the first rows of each split (a dry run)")
    ap.add_argument("--timing-n", type=int, default=200)
    args = ap.parse_args()
    args.run = pathlib.Path(args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    return STAGES[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
