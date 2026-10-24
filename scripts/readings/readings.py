#!/usr/bin/env python3
"""The readings pass: find the readings, run them, and ask where they differ.

For each case, with the student loaded (the served adapter by default):

1. readings  `--elicit iterative` (the default): its §6 answer, then, one
             at a time, ALT_SYSTEM with the readings found so far, each
             reply one line or "none" (elicit_iterative says when it stops).
             `--elicit list`: READINGS_SYSTEM, one reply, a command per line
             -- which this model does not follow (the 2026-10-09 smoke).
2. execute   every reading through the sandbox (clarify's score.execute ->
             evaluate.score -> run_vim). Readings that fail or change nothing
             drop; readings that leave the same buffer merge;
3. question  where two or more distinct results remain: QUESTION_SYSTEM, the
             user turn and each result's visible diff, with "ask:"
             prefilled; the first line, at most 64 tokens, decoded as the
             clarify question is.

Decoding is the clarify experiment's (dialogue.Decoder: greedy, the
registry's stops). Resumable and pausable like dialogue.py: ids already in
--out are skipped, and the stop file finishes the case in flight, writes
PAUSED beside it and exits 3.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # as scripts/baseline.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                                            # noqa: E402
from probe_difficulty import PAUSED_RC, PauseRequest, already_probed      # noqa: E402
from protocol import ASK_PREFIX, QUESTION_MAX_TOKENS, cut_question        # noqa: E402
import readings_protocol as R                                             # noqa: E402


def _call(step, res):
    return {"step": step, "completion": res["completion"],
            "output_tokens": res["output_tokens"], "prompt_tokens": res["prompt_tokens"],
            "latency_s": res["latency_s"]}


def elicit_list(dec, case, execute, timeout):
    """All readings in one reply (READINGS_SYSTEM), then run them all."""
    res = dec.complete(dec.render(R.readings_messages(case)),
                       max_new_tokens=R.READINGS_MAX_TOKENS)
    readings, tally = R.parse_readings(res["completion"])
    answers = {"r%d" % i: {"parse_ok": True, "mode": m, "command": c}
               for i, (m, c) in enumerate(readings)}
    executed = (execute([case], {case["id"]: answers}, timeout=timeout,
                        workers=min(4, len(answers)))
                if answers else {})
    entries = [executed.get((case["id"], "r%d" % i)) for i in range(len(readings))]
    meta = {"elicit": "list", "completion": res["completion"],
            "output_tokens": res["output_tokens"], "prompt_tokens": res["prompt_tokens"],
            "latency_s": res["latency_s"], "tally": tally}
    return readings, entries, meta


def elicit_iterative(dec, case, execute, timeout):
    """One reading at a time: the §6 answer, then ALT_SYSTEM for each next one.

    Each new reading runs at once. The search stops at "none", at a reply
    that does not parse, at a repeated command, at a reading that fails or
    changes nothing, at one whose result an earlier reading already gave, or
    at MAX_READINGS readings."""
    from dialogue import ANSWER_MAX_TOKENS
    from prompt import messages_for_case, parse_output

    def run_one(parsed):
        ans = {"r": {"parse_ok": True, "mode": parsed[0], "command": parsed[1]}}
        return execute([case], {case["id"]: ans}, timeout=timeout, workers=1)[(case["id"], "r")]

    res = dec.complete(dec.render(messages_for_case(case)))
    calls, readings, entries, stop = [_call("section6", res)], [], [], None
    first = parse_output(res["completion"])
    if first is None:
        stop = "unparsed"
    else:
        readings.append(first)
        entries.append(run_one(first))
    while stop is None and len(readings) < R.MAX_READINGS:
        res = dec.complete(dec.render(R.alt_messages(case, readings)),
                           max_new_tokens=ANSWER_MAX_TOKENS, stop_at_newline=True)
        calls.append(_call("alternative", res))
        kind, parsed = R.parse_alt(res["completion"])
        if kind != "reading":
            stop = "none" if kind == "none" else "unparsed"
            break
        if parsed in readings:
            stop = "repeat"
            break
        before = len(R.group_readings(case, entries)[0])
        readings.append(parsed)
        entries.append(run_one(parsed))
        groups, statuses = R.group_readings(case, entries)
        if statuses[-1] != "ok":
            stop = statuses[-1]
        elif len(groups) == before:
            stop = "same_result"
    meta = {"elicit": "iterative", "calls": calls, "stop": stop or "cap",
            "completion": "\n".join(c["completion"] for c in calls),
            "output_tokens": sum(c["output_tokens"] for c in calls),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "latency_s": round(sum(c["latency_s"] for c in calls), 4),
            "tally": {"lines": len(calls), "parsed": len(readings),
                      "unparsed": int(stop == "unparsed"), "marked": 0, "extra": 0,
                      "think": False}}
    return readings, entries, meta


ELICIT = {"iterative": elicit_iterative, "list": elicit_list}


def run_case(dec, case, execute, timeout=5.0, elicit="iterative"):
    """Everything the readings pass records for one case."""
    readings, entries, meta = ELICIT[elicit](dec, case, execute, timeout)
    groups, statuses = R.group_readings(case, entries)
    items = [{"mode": m, "command": c, "status": s, "group": None,
              "match": bool(e and e["match"]),
              "error": e["error"] if e else None,
              "buffer_after": e["buffer_after"] if e else None}
             for (m, c), s, e in zip(readings, statuses, entries)]
    for k, g in enumerate(groups):
        for i in g["members"]:
            items[i]["group"] = k
    row = {"readings": meta,
           "items": items,
           "groups": [{"members": g["members"], "match": g["match"]} for g in groups],
           "distinct": len(groups), "asked": len(groups) >= 2, "question": None}
    if row["asked"]:
        q = dec.complete(dec.render(R.question_messages(case, readings, groups),
                                    prefill=ASK_PREFIX),
                         max_new_tokens=QUESTION_MAX_TOKENS, stop_at_newline=True)
        row["question"] = {
            "text": cut_question(q["completion"]), "raw": q["completion"],
            # "ask:" is two tokens the model would have said itself
            "output_tokens": q["output_tokens"] + len(dec.tok.encode(
                ASK_PREFIX, add_special_tokens=False)),
            "prompt_tokens": q["prompt_tokens"], "latency_s": q["latency_s"]}
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-4b", help="registry name")
    ap.add_argument("--adapter", default=None, help="adapter dir, or 'none' for the base")
    ap.add_argument("--label", required=True, help="the checkpoint's name in the report")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True, help="JSONL, appended; resumable")
    ap.add_argument("--base-dtype", choices=("auto", "bf16", "nf4"), default="auto")
    ap.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per reading")
    ap.add_argument("--elicit", choices=sorted(ELICIT), default="iterative",
                    help="readings one at a time (default) or as one list")
    ap.add_argument("--limit", type=int, default=None, help="first N cases (smoke)")
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()
    adapter = None if args.adapter in (None, "none") else args.adapter

    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_probed(out_path)
    todo = [c for c in cases if c["id"] not in done]
    print("readings [%s]: %d cases, %d already in %s, %d to do" % (
        args.label, len(cases), len(done), out_path, len(todo)), flush=True)
    if not todo:
        return 0

    from baseline import generate_predictions
    from dialogue import ANSWER_MAX_TOKENS, Decoder, load_student
    from prompt import messages_for_case
    from score import execute
    t0 = time.time()
    reg, tok, model, base = load_student(args.model, adapter, args.base_dtype)
    print("loaded %s @ %s (%s base)%s in %.0f s" % (
        reg["hf_id"], reg["revision"][:12], base,
        " + adapter %s" % adapter if adapter else "", time.time() - t0), flush=True)
    dec = Decoder(model, tok, reg)
    # The clarify experiment's guard: on the §6 prompt, Decoder and the
    # results-table path must agree token for token, or a difference between
    # conditions could be an artifact of decoding.
    probe = todo[0]
    ref, _ = generate_predictions(model, tok, [probe], reg, ANSWER_MAX_TOKENS)
    mine = dec.complete(dec.render(messages_for_case(probe)))
    if mine["completion"] != ref[0]["completion"]:
        sys.exit("decoder self-check FAILED on %s: %r (Decoder) != %r (baseline)"
                 % (probe["id"], mine["completion"], ref[0]["completion"]))
    print("decoder self-check: identical to the baseline path on %s" % probe["id"], flush=True)
    pause = PauseRequest(args.stop_file)
    pause.install()

    t_start, asked = time.time(), 0
    with out_path.open("a", encoding="utf-8") as fh:
        for i, case in enumerate(todo, 1):
            row = {"id": case["id"], "label": args.label,
                   **run_case(dec, case, execute, args.timeout, args.elicit)}
            asked += row["asked"]
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_start) / i
                print("  %d/%d  %.1f s/case  asks so far %d  eta %.0f min" % (
                    i, len(todo), rate, asked, rate * (len(todo) - i) / 60), flush=True)
            if pause.check():
                marker = pause.marker()
                if marker:
                    marker.write_text("%s after %d cases\n" % (pause.requested, i))
                print("paused (%s) after %d cases" % (pause.requested, i), flush=True)
                return PAUSED_RC
            if i % 50 == 0:
                import torch
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
