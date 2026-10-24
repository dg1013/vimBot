#!/usr/bin/env python3
"""Execute every answer of the clarification experiment; write the verdicts.

One row per case and checkpoint, with the four conditions' outcomes side by
side (README.md): `direct` (the §6 prompt), `noask` (the offer prompt, ask
banned), `placebo` (asked, told nothing) and `real` (asked, answered by the
simulated user), plus P(ask), the question, the reply and its leak flags.

Scoring is src/evaluate.score -- the one comparison the whole project uses --
called on "virtual cases", one per distinct (case, mode, command), so an
answer two conditions share runs once; chunks run on a thread pool, as the
GRPO reward does around the same sandbox.
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
from evaluate import _norm, load_jsonl, score   # noqa: E402  (_norm: the comparison's own normalizer)
from protocol import QUESTION_MAX_TOKENS, leak_flags, quotes_result   # noqa: E402

CONDITIONS = ("direct", "noask", "placebo", "real")


def answers_for(t1, t2):
    """{condition: the answer record} for one case."""
    return {"direct": t1["direct"], "noask": t1["noask"],
            "placebo": t1["placebo"], "real": t2["answer"]}


def execute(cases, answers, timeout=5.0, workers=8):
    """{(id, condition): score()'s per-case entry, or None for a parse failure}."""
    by_id = {c["id"]: c for c in cases}
    virtual, preds, key_of = [], {}, {}
    for cid, conds in answers.items():
        for cond, a in conds.items():
            if not a["parse_ok"]:
                continue
            k = (cid, a["mode"], a["command"])
            if k not in key_of:
                vid = "%s#%d" % (cid, len(key_of))
                key_of[k] = vid
                virtual.append(dict(by_id[cid], id=vid))
                preds[vid] = {"id": vid, "command": a["command"], "mode": a["mode"]}
    chunks = [virtual[i::workers] for i in range(workers)]
    per_case = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(lambda ch: score(ch, preds, timeout=timeout), chunks):
            per_case.update(res["per_case"])
    out = {}
    for cid, conds in answers.items():
        for cond, a in conds.items():
            out[(cid, cond)] = (per_case[key_of[(cid, a["mode"], a["command"])]]
                                if a["parse_ok"] else None)
    return out


def free_consistent(t1):
    """When greedy would ask: is the forced question the one it asks?

    "same"; "cut" when the forced question is the free one stopped at the
    question cap (protocol.QUESTION_MAX_TOKENS); "diverged" otherwise; None
    when greedy does not ask. Divergence is numerical, not logical: forced,
    "ask:" is part of the prompt's prefill; free, it is decoded token by token
    through the cache, and bf16 rounds the two paths differently, which
    greedy decoding can split on at a near-tie (DECISIONS.md 2026-10-06).
    """
    free = t1["offer"].get("free_completion")
    if free is None:
        return None
    norm = lambda s: " ".join(s.split())                              # noqa: E731
    first = norm(free.lstrip().split("\n", 1)[0])
    forced = norm("ask: " + t1["question"]["text"])
    if first == forced:
        return "same"
    if first.startswith(forced) and question_capped(t1):
        return "cut"
    return "diverged"


# "ask:" is two tokens ("ask", ":") for every Qwen tokenizer in the registry;
# dialogue.py charges them to the question's token count.
ASK_PREFIX_TOKENS = 2


def question_capped(t1):
    """True if the forced question was stopped by the cap, not by its newline."""
    return t1["question"]["output_tokens"] - ASK_PREFIX_TOKENS >= QUESTION_MAX_TOKENS


OUTCOMES = ("match", "parse_fail", "second_ask", "blocked", "timeout", "vim_error",
            "no_change", "wrong_buffer")


def outcome_class(case, entry, answer):
    """Why an answer scored what it did, as one word, for error analysis.

    match; parse_fail (no command by the §6 grammar); second_ask (turn 2
    asked again, which the protocol forbids); blocked (the scope screen
    refused it before vim ran); timeout; vim_error (invalid by the
    executor's verdict -- keys mode's is best-effort, CLAUDE.md §2);
    no_change (ran, left the buffer as it was); wrong_buffer (ran, changed
    it, not to the result).
    """
    if not answer["parse_ok"]:
        return "second_ask" if answer.get("second_ask") else "parse_fail"
    if entry.get("blocked"):
        return "blocked"
    if entry["timeout"]:
        return "timeout"
    if entry["match"]:
        return "match"
    if not entry["valid"]:
        return "vim_error"
    if _norm(entry["buffer_after"]) == _norm(case["buffer_before"]):
        return "no_change"
    return "wrong_buffer"


def verdict_rows(cases, label, turn1, turn2, replies, executed):
    """One row per case: everything the report reads, and what vim did with
    every answer (the buffer it produced, its stderr), so error analysis
    never has to re-execute a condition -- the project's exec-log rule
    (DECISIONS.md 2026-09-10) applied to all four conditions."""
    rows = []
    for c in cases:
        cid = c["id"]
        t1, t2, rep = turn1[cid], turn2[cid], replies[cid]
        q_tokens = t1["question"]["output_tokens"]
        row = {"id": cid, "label": label, "category": c.get("category"),
               "tier": c.get("tier"), "gold_mode": c.get("mode"),
               "instruction": c["instruction"],
               "p_ask": t1["offer"]["p_ask"], "p_ask_norm": t1["offer"]["p_ask_norm"],
               "ask_margin": t1["offer"]["ask_margin"],
               "chose_ask": t1["offer"]["chose_ask"], "top": t1["offer"]["top"],
               "free_consistent": free_consistent(t1),
               "first_consistent": t1["offer"].get("first_consistent"),
               "question": t1["question"]["text"], "question_tokens": q_tokens,
               "question_capped": question_capped(t1),
               # screened again here, with the current screen, so every reply
               # of every set meets the same one (it was widened after the
               # val audit); the flags written at generation are kept beside
               "reply": rep["reply"], "leak_flags": leak_flags(rep["reply"], c),
               "quotes_result": quotes_result(rep["reply"], c),
               "leak_flags_at_generation": rep["leak_flags"]}
        for k in ("note_class", "note"):
            if k in c:
                row[k] = c[k]
        for cond, a in answers_for(t1, t2).items():
            e = executed[(cid, cond)]
            row[cond] = {
                "match": bool(e and e["match"]), "valid": bool(e and e["valid"]),
                "outcome": outcome_class(c, e, a),
                "parse_ok": a["parse_ok"], "mode": a["mode"], "command": a["command"],
                "completion": a["completion"],
                "error": e["error"] if e else "PARSE",
                # what vim did: the buffer it left, and what it said
                "buffer_after": e["buffer_after"] if e else None,
                "stderr": e["stderr"] if e else None,
                "return_code": e["return_code"] if e else None,
                "timeout": bool(e and e["timeout"]),
                "blocked": e.get("blocked") if e else None,
                # what the model said in the whole exchange: a question costs
                # its own tokens on top of the answer's
                "tokens": a["output_tokens"] + (q_tokens if cond in ("placebo", "real") else 0),
                "second_ask": a.get("second_ask", False)}
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--turn1", required=True)
    ap.add_argument("--turn2", required=True)
    ap.add_argument("--replies", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    turn1 = {r["id"]: r for r in load_jsonl(args.turn1)}
    turn2 = {r["id"]: r for r in load_jsonl(args.turn2)}
    replies = {r["id"]: r for r in load_jsonl(args.replies) if r["label"] == args.label}
    missing = [c["id"] for c in cases
               if c["id"] not in turn1 or c["id"] not in turn2 or c["id"] not in replies]
    if missing:
        sys.exit("%d cases are missing a stage (first: %s)" % (len(missing), missing[0]))

    answers = {c["id"]: answers_for(turn1[c["id"]], turn2[c["id"]]) for c in cases}
    executed = execute(cases, answers, args.timeout, args.workers)
    rows = verdict_rows(cases, args.label, turn1, turn2, replies, executed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(rows)
    print("%s: %d cases -> %s" % (args.label, n, out))
    for cond in CONDITIONS:
        kinds = {}
        for r in rows:
            kinds[r[cond]["outcome"]] = kinds.get(r[cond]["outcome"], 0) + 1
        print("  %-8s exec-match %5.1f%%   %s" % (
            cond, 100 * sum(r[cond]["match"] for r in rows) / n,
            " ".join("%s %d" % (k, kinds[k]) for k in OUTCOMES if k in kinds)))
    print("  greedy asks on %d (%.1f%%)" % (sum(r["chose_ask"] for r in rows),
                                           100 * sum(r["chose_ask"] for r in rows) / n))
    return 0


if __name__ == "__main__":
    sys.exit(main())
