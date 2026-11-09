#!/usr/bin/env python3
"""A 27B with 8 BM25 examples, with and without the 5-of-8 rule, one stage per call.

By runs/retr27b_2026-11-06/preregistration.md. Each stage resumes from what is on disk; run.sh runs them in order.
--arm base is the untrained 27B (Part 2), --arm trained the trained one (Part 3, addendum 2): the same stages, except that
the trained 27B's zero-shot answers are already on disk, and it has no recorded run with examples to compare the smoke with.

The examples go in through `WithExamples`, an endpoint that puts a row's pairs straight after the system message of
whatever it renders. So every turn carries them, and the question, placebo, answer, timing and judge stages are
scripts/ask27b/ask27b.py's own, and scripts/serve/roundtrip.py runs unchanged in the timing.

    plan      CPU  the two measured plan files, checked and kept beside the run; BM25's own time
    smoke     27B  is this the prompt and the model that were measured? Exit 3 if not
    greedy    27B  the §6 answer, with the examples and zero-shot, every val and test row
    samples   27B  N samples per row at T = 1, in requests of 8
    views     CPU  every distinct sample through the box: its result and its verdict
    question  27B  the question it asks, every row ("ask:" prefilled, the first line)
    placebo   27B  its answer after a reply with no information
    answer    27B  its answer after the simulated user's reply
    timing    27B  the round trip itself, one at a time, on the 200 rows the 4B and the trained 27B were timed on
    score     4B   the verifier on every distinct answer
    judge     CPU  the greedy answers and every answer after a reply, through the box
    gate      CPU  is a dry run fit to go on, and does its pace fit the night?

Exit codes as scripts/serve27b/run.sh reads them: 3 a gate failed, 4 paused, 5 the pace fell.
"""
import argparse
import collections
import json
import pathlib
import random
import shutil
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "clarify",
          REPO / "scripts" / "readings", REPO / "scripts" / "serve27b", REPO / "scripts" / "ask27b",
          REPO / "scripts" / "reranker", REPO / "scripts" / "serve_pooled", REPO / "scripts" / "retrieval"):
    sys.path.insert(0, str(p))
import ask27b as Q                                                # noqa: E402
from prompt import build_messages, render_user                    # noqa: E402

A = Q.A
BASE_NAME = "qwen3.5-27b-base"                                    # serve.sh's name for the model with no adapter
SERVED = {"base": BASE_NAME, "trained": A.NAME27}                 # the adapter's name, when serve.sh is given it
TRAINED_RUN = REPO / "runs" / "teacher27b_2026-10-30"
PLANS = {"val": TRAINED_RUN / "plans" / "plan_retr8.jsonl",       # the prompted 27B's, already measured on val
         "test": REPO / "runs" / "retrieval_2026-10-18" / "plan_retr8.jsonl"}
RECORDED = TRAINED_RUN / "27b-retr8" / "preds_val.jsonl"          # its greedy answers through transformers
RECORDED_ZERO = Q.SERVE27 / "preds_val.jsonl"                     # the trained 27B's, zero-shot, as served
ARMS = {"base": ("retr8", "k0"), "trained": ("retr8",)}           # the trained 27B's zero-shot pass is on disk
CHUNK = 8                                                         # samples per request
PARITY_GATE = 0.70                                                # an untrained model's near-ties flip more easily than the trained one's 93%
GATE_HOURS = 12.0                                                 # §14b
OTHER_HOURS = 4.0                                                 # every stage but samples, from the trained run plus longer prompts
NO_ANSWER = {"command": None}
ARM = "base"                                                      # main() sets it from --arm


def user_turn(case):
    return render_user(case["instruction"], case["buffer_before"], case.get("cursor", (1, 1)))


def insert_examples(messages, examples):
    """The k-shot pairs straight after the system message: prompt.build_messages' own pairs, so §6's copy."""
    return messages[:1] + build_messages("", "", examples=examples)[1:-1] + messages[1:]


class WithExamples:
    """An endpoint whose prompts carry each row's examples. `plan` maps a row's §6 user turn to them. Turn 2 repeats
    that turn, so one lookup serves every turn, and an unknown row raises instead of going out zero-shot."""

    def __init__(self, ep, plan):
        self.ep, self.plan = ep, plan

    def render(self, messages, prefill=""):
        return self.ep.render(insert_examples(messages, self.plan[messages[1]["content"]]), prefill)

    def __getattr__(self, name):
        return getattr(self.ep, name)


def load_plan():
    """{§6 user turn: the row's examples}, for every val and test row."""
    train = {r["id"]: r for r in A.load_jsonl(REPO / "data" / "splits" / "train.jsonl")}
    out = {}
    for sp in A.SPLITS:
        plan = {r["id"]: r for r in A.load_jsonl(PLANS[sp])}
        for c in A.cases(sp):
            key = user_turn(c)
            if key in out:
                raise SystemExit("two rows share a user turn (%s)" % c["id"])
            out[key] = [train[i] for i in plan[c["id"]]["examples"]]
    return out


def plain_endpoint():
    import endpoint as E
    e = E.load(A.MODEL27, A.URL27, SERVED[ARM])
    e.timeout = 900.0
    return e


def endpoint_with_examples():
    return WithExamples(plain_endpoint(), load_plan())


def agreeing(samples, recorded=None):
    """How many of a row's answers agree: with the recorded answer, or, where none exists, with each other."""
    if recorded is not None:
        return sum(A.same_answer(s, recorded) for s in samples)
    return max(collections.Counter(None if s is None else tuple(s) for s in samples).values())


def cmd_plan(args):
    """No model. Keeps the plans with the run, and times what real use would add: BM25 over the train rows."""
    import retrieval as B
    out = args.run / "plans"
    out.mkdir(exist_ok=True)
    train = A.load_jsonl(REPO / "data" / "splits" / "train.jsonl")
    known = {r["id"] for r in train}
    for sp in A.SPLITS:
        rows = {r["id"]: r for r in A.load_jsonl(PLANS[sp])}
        want = {c["id"] for c in A.cases(sp)}
        if set(rows) != want:
            raise SystemExit("the %s plan does not cover the split" % sp)
        if any(e not in known for r in rows.values() for e in r["examples"]):
            raise SystemExit("the %s plan names a row that is not in train" % sp)
        shutil.copyfile(PLANS[sp], out / ("plan_retr8_%s.jsonl" % sp))
    t0 = time.perf_counter()
    index = B.BM25([B.tokens(c["instruction"]) for c in train])
    build_s = time.perf_counter() - t0
    took = []
    for c in Q.timing_rows(A.cases("val"), 200):
        t0 = time.perf_counter()
        index.top(B.tokens(c["instruction"]), B.OVERFETCH * 8)
        took.append(time.perf_counter() - t0)
    A.write_json(args.run / "bm25.json", {"train_rows": len(train), "build_s": round(build_s, 3),
                                          "query_ms_mean": round(1000 * statistics.fmean(took), 3),
                                          "query_ms_p50": round(1000 * statistics.median(took), 3)})
    print("BM25: %d rows, built in %.2f s, %.2f ms a query" % (len(train), build_s, 1000 * statistics.fmean(took)), flush=True)
    return 0


def cmd_smoke(args):
    """Same prompt, same model. At the greedy stage's concurrency, so chunked prefill is exercised, and on the
    longest prompts with 8 requests at once, the way a sampling request arrives. The base 27B has a recorded run with
    these prompts. The trained 27B has none, so its zero-shot answers are checked, and its long rows against themselves."""
    e = A.endpoint27()
    trained = ARM == "trained"
    parity_e = plain_endpoint() if trained else e
    recorded = {r["id"]: r for r in A.load_jsonl(RECORDED_ZERO if trained else RECORDED)}
    tokens = {r["id"]: r["prompt_tokens"] for r in A.load_jsonl(PLANS["val"])}
    val = A.cases("val")
    rows = val[:args.smoke_n]                                     # a dry run too: 12 rows would make the gate a coin toss
    with ThreadPoolExecutor(max_workers=Q.WORKERS) as ex:
        res = list(ex.map(lambda c: A.request(parity_e, c, 1, 0.0), rows))
        carried = list(ex.map(lambda c: A.request(e, c, 1, 0.0), rows)) if trained else res
    same = sum(A.same_answer(r["samples"][0], recorded.get(r["id"], NO_ANSWER)) for r in res)     # an unparsed answer has no row
    off = [r["id"] for r in carried if r["prompt_tokens"] != tokens[r["id"]]]
    longest = sorted(val, key=lambda c: -tokens[c["id"]])[:args.long_n]
    long_same = 0
    for c in longest:
        with ThreadPoolExecutor(max_workers=CHUNK) as ex:
            got = list(ex.map(lambda _: A.request(e, c, 1, 0.0), range(CHUNK)))
        long_same += agreeing([g["samples"][0] for g in got], None if trained else recorded.get(c["id"], NO_ANSWER))
    # The longest turn a dialogue can reach must fit the endpoint: it raises if vLLM refuses it.
    e.complete(e.render(Q.turn2_messages(longest[0], "which one " * 30, "this one " * 45)), Q.ANSWER_MAX_TOKENS)
    rep = {"rows": len(res), "identical": same, "prompt_token_mismatches": off, "gate": PARITY_GATE,
           "longest_rows": len(longest), "longest_prompt": tokens[longest[0]["id"]], "longest_identical": long_same,
           "longest_answers": CHUNK * len(longest), **A.summary(res)}
    A.write_json(args.run / "smoke.json", rep)
    print(rep, flush=True)
    ok = not off and same >= PARITY_GATE * len(res) and long_same >= PARITY_GATE * CHUNK * len(longest)
    return 0 if ok else 3


def cmd_greedy(args):
    plain, carried = plain_endpoint(), A.endpoint27()
    for arm, e in (("retr8", carried), ("k0", plain)):
        if arm not in ARMS[ARM]:
            continue
        for sp in A.SPLITS:
            todo = {c["id"]: (sp, c) for c in A.cases(sp, args.limit)}

            def work(s, c, e=e):
                return dict(A.request(e, c, 1, 0.0), split=s)

            rc = Q.run_rows(args.run, "greedy_%s_%s" % (arm, sp), todo, work)
            if rc:
                return rc
    return 0


class OutOfBudget(Exception):
    pass


def sample_row(e, sp, case, n, chunk=CHUNK):
    """n samples in requests of `chunk`: 16 at once would hold about 13,600 tokens of a cache of 8,500."""
    got = [A.request(e, case, min(chunk, n - s), 1.0) for s in range(0, n, chunk)]
    return {"id": case["id"], "split": sp, "samples": [x for g in got for x in g["samples"]],
            "output_tokens": [x for g in got for x in g["output_tokens"]], "prompt_tokens": got[0]["prompt_tokens"]}


def cmd_samples(args):
    e = A.endpoint27()
    path = args.run / "samples.jsonl"
    wrong = [r["id"] for r in A.load_jsonl(path) if len(r["samples"]) != args.n] if path.exists() else []
    if wrong:
        sys.exit("%d rows on disk do not hold %d samples (first: %s)" % (len(wrong), args.n, wrong[0]))
    todo = {c["id"]: (sp, c) for sp in A.SPLITS for c in A.cases(sp, args.limit)}
    deadline = time.time() + args.budget_s

    def work(sp, c):
        if time.time() > deadline:
            raise OutOfBudget
        return sample_row(e, sp, c, args.n)

    try:
        return Q.run_rows(args.run, "samples", todo, work, workers=args.inflight)
    except OutOfBudget:
        print("the samples stage's budget ran out; going on with the rows it has", flush=True)
        return 0


def cmd_views(args):
    """No model: every distinct sample once through the validation path, as the round trip does."""
    from score import execute
    timeout = Q.settings_timeout()
    samples = {r["id"]: r for r in A.load_jsonl(args.run / "samples.jsonl")}
    with (args.run / "views.jsonl").open("w") as views, (args.run / "states.jsonl").open("w") as states:
        for sp in A.SPLITS:
            cs = [c for c in A.cases(sp, args.limit) if c["id"] in samples]
            answers = {c["id"]: {"c%d" % n: {"parse_ok": True, "mode": m, "command": cmd}
                                 for n, (m, cmd) in enumerate(Q.distinct(samples[c["id"]]["samples"]))} for c in cs}
            t0 = time.time()
            executed = execute(cs, answers, timeout=timeout, workers=8)
            for c in cs:
                i = c["id"]
                by_key = {(a["mode"], a["command"]): executed[(i, name)] for name, a in answers[i].items()}
                views.write(json.dumps({"id": i, "split": sp, "results": Q.sample_results(c, samples[i]["samples"], by_key)}) + "\n")
                for (mode, command), entry in by_key.items():
                    states.write(json.dumps({"id": i, "mode": mode, "command": command,
                                             "state": "match" if entry and entry["match"] else "no"}) + "\n")
            print("%s: %d rows, %d commands through the box in %.0f s" % (
                sp, len(cs), sum(len(a) for a in answers.values()), time.time() - t0), flush=True)
    return 0


def cmd_score(args):
    """The 4B verifier on every distinct answer. Text only. The 27B has left the card by now."""
    import endpoint as E
    import pooled as PP
    e4 = E.load("qwen3.5-4b", A.URL4, "qwen3.5-4b-served")
    pp = PP.Pooled(e4, SimpleNamespace(name=None))
    path = args.run / "scores.jsonl"
    known = {(r["id"], r["mode"], r["command"]) for r in A.load_jsonl(path)} if path.exists() else set()
    samples = {r["id"]: r for r in A.load_jsonl(args.run / "samples.jsonl")}
    with path.open("a") as fh:
        for sp in A.SPLITS:
            greedy = {r["id"]: r for r in A.load_jsonl(args.run / ("greedy_retr8_%s.jsonl" % sp))}
            t0, n = time.time(), 0
            for row, c in enumerate(A.cases(sp, args.limit)):
                i = c["id"]
                want = [k for k in A.distinct_answers(greedy.get(i), samples.get(i)) if (i,) + k not in known]
                for s in range(0, len(want), 64):
                    chunk = [{"mode": m, "command": cmd} for m, cmd in want[s:s + 64]]
                    values, _ = pp.scores(c, chunk)
                    for x, v in zip(chunk, values):
                        known.add((i, x["mode"], x["command"]))
                        fh.write(json.dumps({"id": i, "split": sp, "mode": x["mode"], "command": x["command"], "score": v}) + "\n")
                    fh.flush()
                n += len(want)
                if (row + 1) % 100 == 0:
                    print("  %s: %d rows, %d answers scored, %.0f s" % (sp, row + 1, n, time.time() - t0), flush=True)
            print("%s: %d answers scored in %.0f s" % (sp, n, time.time() - t0), flush=True)
    by_id = {c["id"]: c for sp in A.SPLITS for c in A.cases(sp)}
    old = A.load_jsonl(Q.SERVE27 / "scores27.jsonl")
    check = random.Random("rescore").sample(old, min(args.rescore_n, len(old)))
    diffs = [abs(pp.scores(by_id[r["id"]], [{"mode": r["mode"], "command": r["command"]}])[0][0] - r["score"]) for r in check]
    A.write_json(args.run / "rescore.json", {"rows": len(diffs), "mean_abs": round(statistics.fmean(diffs), 4),
                                             "max_abs": round(max(diffs), 4)})
    print("rescored %d stored answers: mean |diff| %.3f, max %.3f" % (len(diffs), statistics.fmean(diffs), max(diffs)), flush=True)
    return 0


def greedy_preds(rows):
    """{id: a prediction} from a greedy file's rows. An answer that did not parse has none: score() fails it."""
    return {r["id"]: {"command": r["samples"][0][1], "mode": r["samples"][0][0]} for r in rows if r["samples"][0] is not None}


def cmd_judge(args):
    from evaluate import score, write_exec_log
    timeout = Q.settings_timeout()
    for arm in ARMS[ARM]:
        for sp in A.SPLITS:
            rows = A.cases(sp, args.limit)
            preds = greedy_preds(A.load_jsonl(args.run / ("greedy_%s_%s.jsonl" % (arm, sp))))
            write_exec_log(str(args.run / ("exec_%s_%s.jsonl" % (arm, sp))), rows, score(rows, preds, timeout=timeout))
            print("%s, %s: %d rows judged" % (arm, sp, len(rows)), flush=True)
    return Q.cmd_judge(args)


def projected_hours(rows, plan_tokens):
    """The whole chain from a dry run's sampling pace, by prompt tokens: the dry rows are the head of each split,
    which are light, and nearly all of a sampled row's time here is reading its prompt 16 times."""
    seen = sum(plan_tokens[r["id"]] for r in rows)
    return sum(r["row_s"] for r in rows) / seen * sum(plan_tokens.values()) / 3600 + OTHER_HOURS


def cmd_gate(args):
    rc = Q.cmd_gate(args)
    if rc:
        return rc
    tokens = {r["id"]: r["prompt_tokens"] for sp in A.SPLITS for r in A.load_jsonl(PLANS[sp])}
    hours = projected_hours(A.load_jsonl(args.run / "samples.jsonl"), tokens)
    print("the pace projects the whole chain at %.1f h" % hours, flush=True)
    if hours > GATE_HOURS:
        print("stop: past the %.0f h stop point (§14b)" % GATE_HOURS, flush=True)
        return 3
    return 0


STAGES = {"plan": cmd_plan, "smoke": cmd_smoke, "greedy": cmd_greedy, "samples": cmd_samples, "views": cmd_views,
          "question": Q.cmd_question, "placebo": Q.cmd_placebo, "answer": Q.cmd_answer, "timing": Q.cmd_timing,
          "score": cmd_score, "judge": cmd_judge, "gate": cmd_gate}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--run", default=str(REPO / "runs" / "retr27b_2026-11-06"))
    ap.add_argument("--arm", choices=sorted(SERVED), default="base", help="which 27B the endpoint serves")
    ap.add_argument("--limit", type=int, default=0, help="only the first rows of each split (a dry run)")
    ap.add_argument("--n", type=int, default=16, help="samples per row")
    ap.add_argument("--inflight", type=int, default=1, help="rows in flight in the samples stage")
    ap.add_argument("--budget-s", type=float, default=18000.0, help="no new row starts this long into the samples stage")
    ap.add_argument("--smoke-n", type=int, default=40)
    ap.add_argument("--long-n", type=int, default=10)
    ap.add_argument("--timing-n", type=int, default=200)
    ap.add_argument("--rescore-n", type=int, default=100)
    args = ap.parse_args()
    args.run = pathlib.Path(args.run)
    args.run.mkdir(parents=True, exist_ok=True)
    global ARM
    ARM = args.arm
    A.endpoint27 = endpoint_with_examples                         # ask27b's stages read it at call time
    Q.LABEL = "%s27b-retr8" % ARM
    return STAGES[args.stage](args)


if __name__ == "__main__":
    sys.exit(main())
