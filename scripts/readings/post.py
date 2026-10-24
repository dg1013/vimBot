#!/usr/bin/env python3
"""The readings experiment's CPU stages (README.md beside it).

    prepare      readings rows -> the files the clarify harness reads:
                 questions_readings.jsonl (turn-1 shaped, label "readings"),
                 asked_cases.jsonl, placebo_replies.jsonl
    score        execute turn 2 (real reply and placebo) in the sandbox and
                 join the 2026-10-06 served verdicts on the same rows ->
                 verdicts_readings.jsonl
    report       verdicts -> report.md and summary.json
    transcripts  every prompt and action, case by case, with the student's
                 prompts rebuilt and checked against the recorded token counts
    fingerprint  record what the prompts are at launch (run.sh does)

The 2026-10-06 run supplies every counterfactual column for the served
adapter on these rows: the §6 direct answer, the unasked answer under the
offer prompt, the generic question with the simulated user's reply, and the
placebo. This experiment adds only what is new -- the readings, the question
written from them, its reply, and turn 2 -- so when to ask and what to ask
are compared on identical rows.
"""
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                                      # noqa: E402
from policies import (at_rate, auroc, flips, mcnemar_p, mean,        # noqa: E402
                      oracle, random_at_rate, threshold_curve)
from protocol import PLACEBO_REPLY, leak_flags, quotes_result, turn2_messages   # noqa: E402
import readings_protocol as R                                        # noqa: E402

LABEL, PLACEBO_LABEL = "readings", "readings_placebo"
CLASSES = ("unstated", "ambiguous", "model")
CONDS = ("readings_real", "readings_placebo")


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --- prepare --------------------------------------------------------------------

def prepare(readings_rows, cases):
    """(questions, asked_cases, placebo_replies) for the rows that ask."""
    by_id = {r["id"]: r for r in readings_rows}
    asked = [c for c in cases if c["id"] in by_id and by_id[c["id"]]["asked"]]
    questions = [{"id": c["id"], "label": LABEL, "question": by_id[c["id"]]["question"]}
                 for c in asked]
    placebo = [{"id": c["id"], "label": PLACEBO_LABEL,
                "question": by_id[c["id"]]["question"]["text"], "reply": PLACEBO_REPLY,
                "leak_flags": [], "quotes_result": False} for c in asked]
    return questions, asked, placebo


# --- score ----------------------------------------------------------------------

def cond_fields(case, entry, answer, extra_tokens):
    """One answer as the clarify verdicts record it (score.verdict_rows)."""
    from score import outcome_class
    return {"match": bool(entry and entry["match"]), "valid": bool(entry and entry["valid"]),
            "outcome": outcome_class(case, entry, answer),
            "parse_ok": answer["parse_ok"], "mode": answer["mode"],
            "command": answer["command"], "completion": answer["completion"],
            "error": entry["error"] if entry else "PARSE",
            "buffer_after": entry["buffer_after"] if entry else None,
            "stderr": entry["stderr"] if entry else None,
            "return_code": entry["return_code"] if entry else None,
            "timeout": bool(entry and entry["timeout"]),
            "blocked": entry.get("blocked") if entry else None,
            "tokens": answer["output_tokens"] + extra_tokens,
            "second_ask": answer.get("second_ask", False)}


def verdict_rows(cases, readings_rows, turn2_real, turn2_placebo, replies, ref, executed):
    """One row per case: the readings, the decision, the new dialogue's two
    answers (where it asked), and the 2026-10-06 columns on the same row."""
    rd = {r["id"]: r for r in readings_rows}
    t2r = {r["id"]: r for r in turn2_real}
    t2p = {r["id"]: r for r in turn2_placebo}
    rep = {r["id"]: r for r in replies if r["label"] == LABEL}
    rows = []
    for c in cases:
        cid = c["id"]
        if cid not in rd:
            continue
        r, v = rd[cid], ref[cid]
        q = r["question"]
        row = {"id": cid, "category": c.get("category"), "tier": c.get("tier"),
               "instruction": c["instruction"],
               "distinct": r["distinct"], "asked": r["asked"],
               "n_readings": len(r["items"]),
               "statuses": [it["status"] for it in r["items"]],
               "gold_in_readings": any(it["match"] for it in r["items"]),
               "readings_tokens": r["readings"]["output_tokens"],
               "readings_tally": r["readings"]["tally"],
               "elicit": r["readings"].get("elicit", "list"),
               "stop": r["readings"].get("stop"),
               "question": q["text"] if q else None,
               "question_tokens": q["output_tokens"] if q else 0,
               "reply": None, "reply_leak_flags": None, "reply_quotes_result": None,
               "readings_real": None, "readings_placebo": None,
               # the 2026-10-06 served verdict on the same row
               "direct": v["direct"], "noask": v["noask"], "real": v["real"],
               "placebo": v["placebo"], "p_ask_norm": v["p_ask_norm"],
               "chose_ask": v["chose_ask"], "generic_question": v["question"],
               "generic_question_tokens": v.get("question_tokens"),
               "generic_reply": v["reply"]}
        for k in ("note_class", "note"):
            if k in c:
                row[k] = c[k]
        if r["asked"] and cid in t2r and cid in t2p:
            reply = rep[cid]["reply"] if cid in rep else t2r[cid]["reply"]
            row.update(reply=reply, reply_leak_flags=leak_flags(reply, c),
                       reply_quotes_result=quotes_result(reply, c))
            for cond, t2 in (("readings_real", t2r[cid]), ("readings_placebo", t2p[cid])):
                row[cond] = cond_fields(c, executed[(cid, cond)], t2["answer"],
                                        q["output_tokens"])
        rows.append(row)
    return rows


# --- report ---------------------------------------------------------------------

def m(cond):
    return lambda r: int(r[cond]["match"])


def e2e(r):
    """The readings policy, deployed: its dialogue where it asks, else the §6 answer."""
    return int((r["readings_real"] if r["asked"] else r["direct"])["match"])


def e2e_tokens(r):
    """What the model says per request: the readings, then the question and
    its answer after the reply, or else the §6 answer."""
    if r["asked"]:
        return r["readings_tokens"] + r["readings_real"]["tokens"]
    return r["readings_tokens"] + r["direct"]["tokens"]


def pct(x):
    return "—" if x != x else "%.1f%%" % (100 * x)


def paired(a, b):
    g, l = flips(a, b)
    return "%+.1f (+%d/−%d, p %.3f)" % (100 * (mean(b) - mean(a)), g, l, mcnemar_p(g, l))


def policy_rows(rows):
    """Who to ask, with the 2026-10-06 generic question so only *when* differs:
    the asked rows get its dialogue's answer, the others the unasked answer."""
    pr = [{"id": r["id"], "real": m("real")(r), "noask": m("noask")(r), "p": r["p_ask_norm"]}
          for r in rows]
    ask = {r["id"] for r in rows if r["asked"]}
    rate = len(ask) / len(rows)
    by_d = mean(p["real"] if p["id"] in ask else p["noask"] for p in pr)
    o_rate, o_match = oracle(pr)
    return [("never ask (offer prompt, may not ask)", 0.0, mean(p["noask"] for p in pr)),
            ("**ask iff the readings differ**", rate, by_d),
            ("ask at random, same rate", rate, random_at_rate(pr, rate)),
            ("ask on the top P(ask), same rate", rate,
             at_rate(threshold_curve(pr, key="p"), rate)),
            ("oracle: ask only where it turns wrong to right", o_rate, o_match)]


def report_lines(rows, name, source):
    n = len(rows)
    has_cls = any(r.get("note_class") for r in rows)
    classes = [c for c in CLASSES if any(r.get("note_class") == c for r in rows)]
    asked = [r for r in rows if r["asked"]]
    iterative = all(r.get("elicit") == "iterative" for r in rows)
    how = ("one at a time: its §6 answer, then each further reading asked for alone, given "
           "those found so far" if iterative else "as one list in a single reply")
    out = ["# Readings first: %s" % name, "",
           "Generated by `scripts/readings/post.py report` from `%s`. The served adapter "
           "gives the readings of each instruction as commands, %s; the sandbox runs them; "
           "where two or more distinct results remain it asks one question written from them "
           "(`scripts/readings/readings_protocol.py`). The other columns are the 2026-10-06 "
           "run's on the same rows." % (source, how), ""]

    # 1. the readings
    nr = collections.Counter(r["n_readings"] for r in rows)
    st = collections.Counter(s for r in rows for s in r["statuses"])
    dd = collections.Counter(r["distinct"] for r in rows)
    tl = collections.Counter()
    for r in rows:
        for k, v in r["readings_tally"].items():
            tl[k] += int(v)
    out += ["## 1. The readings", "",
            "| | |", "|---|---|",
            "| readings per row (0 / 1 / 2 / 3 / 4) | %s |" % " / ".join(str(nr[k]) for k in range(5)),
            ("| model calls for readings / readings kept / replies that did not parse | %d / %d / %d |"
             % (tl["lines"], tl["parsed"], tl["unparsed"]) if iterative else
             "| lines written / parsed / unparsed / past the cap / with a list marker | %d / %d / %d / %d / %d |"
             % (tl["lines"], tl["parsed"], tl["unparsed"], tl["extra"], tl["marked"])),
            "| outputs with a think tag (failed whole) | %d |" % tl["think"],
            "| readings run: changed the buffer / no change / invalid | %d / %d / %d |"
            % (st["ok"], st["no_change"], st["invalid"]),
            "| distinct results per row (0 / 1 / 2 / 3 / 4) | %s |" % " / ".join(str(dd[k]) for k in range(5)),
            "| the expected result among the readings | %s |" % pct(mean(int(r["gold_in_readings"]) for r in rows)),
            "| the §6 direct answer right | %s |" % pct(mean(m("direct")(r) for r in rows)),
            "| tokens of readings per row (mean) | %.1f |" % mean(r["readings_tokens"] for r in rows)]
    stops = collections.Counter(r.get("stop") for r in rows if r.get("stop"))
    if stops:
        out.append("| why the search for readings stopped | %s |" % ", ".join(
            "%s %d" % kv for kv in stops.most_common()))
    out.append("")
    if has_cls:
        out += ["| class | rows | asks (readings differ) | expected result among readings | direct right |",
                "|---|---|---|---|---|"]
        for c in classes:
            sub = [r for r in rows if r.get("note_class") == c]
            out.append("| %s | %d | %s | %s | %s |" % (
                c, len(sub), pct(mean(int(r["asked"]) for r in sub)),
                pct(mean(int(r["gold_in_readings"]) for r in sub)),
                pct(mean(m("direct")(r) for r in sub))))
        out.append("")

    # 2. the decision
    fix = [int(r["real"]["match"] and not r["noask"]["match"]) for r in rows]
    a = [int(r["asked"]) for r in rows]
    hit = sum(x and y for x, y in zip(a, fix))
    out += ["## 2. When to ask: the readings against the other rules", "",
            "Asks on %d of %d rows (%s). Of the %d rows where the 2026-10-06 question turned a "
            "wrong answer right, the readings ask on %d (recall %s); of their asks, %s are such "
            "rows (precision), against %s for asking at random." % (
                sum(a), n, pct(mean(a)), sum(fix), hit,
                pct(hit / sum(fix)) if sum(fix) else "—",
                pct(hit / sum(a)) if sum(a) else "—", pct(mean(fix))), "",
            "| rule (each asks with the 2026-10-06 question) | asks | exec-match |", "|---|---|---|"]
    out += ["| %s | %s | %s |" % (lab, pct(rate), pct(match)) for lab, rate, match in policy_rows(rows)]
    if has_cls:
        out += ["", "AUROC against the fair (*model*) rows, 0.5 = chance:", "",
                "| task | distinct readings | the served P(ask) |", "|---|---|---|"]
        for task, pos in (("unstated vs fair", {"unstated"}),
                          ("unstated or ambiguous vs fair", {"unstated", "ambiguous"}),
                          ("ambiguous vs fair", {"ambiguous"})):
            sub = [r for r in rows if r.get("note_class") in pos | {"model"}]
            lab = [r["note_class"] in pos for r in sub]
            ad = auroc([r["distinct"] for r in sub], lab)
            ap = auroc([r["p_ask_norm"] for r in sub], lab)
            out.append("| %s | %s | %s |" % (task, "—" if ad is None else "%.2f" % ad,
                                             "—" if ap is None else "%.2f" % ap))
    out.append("")

    # 3. the question
    out += ["## 3. What to ask: the readings question against the generic one", ""]
    if asked:
        rr, gr = [m("readings_real")(r) for r in asked], [m("real")(r) for r in asked]
        rp, gp = [m("readings_placebo")(r) for r in asked], [m("placebo")(r) for r in asked]
        lf = sum(bool(r["reply_leak_flags"]) for r in asked)
        qr = sum(bool(r["reply_quotes_result"]) for r in asked)
        out += ["On the %d rows where the readings differ; both dialogues are answered by the "
                "same simulated user." % len(asked), "",
                "| question | simulated reply | placebo reply | the reply's information | tokens of question |",
                "|---|---|---|---|---|",
                "| generic (2026-10-06) | %s | %s | %s | %.1f |" % (
                    pct(mean(gr)), pct(mean(gp)), paired(gp, gr),
                    mean(r["generic_question_tokens"] or 0 for r in asked)),
                "| **from the readings** | %s | %s | %s | %.1f |" % (
                    pct(mean(rr)), pct(mean(rp)), paired(rp, rr),
                    mean(r["question_tokens"] for r in asked)), "",
                "- readings question against the generic one, simulated reply: %s" % paired(gr, rr),
                "- replies to the readings questions flagged by the leak screen: %d; quoting the "
                "result: %d. A question that names candidate results invites a reply that picks "
                "one, which is a choice, not a leak; read the transcripts." % (lf, qr), ""]
    else:
        out += ["No row asked.", ""]

    # 4. end to end
    d = [m("direct")(r) for r in rows]
    ee = [e2e(r) for r in rows] if all(r["readings_real"] or not r["asked"] for r in rows) else None
    out += ["## 4. End to end: the readings policy against the §6 answer", "",
            "Where the readings differ, the dialogue's answer; elsewhere the §6 answer.", ""]
    if ee is not None:
        o_rate, o_match = oracle([{"id": r["id"], "real": m("real")(r), "noask": m("noask")(r)}
                                  for r in rows])
        out += ["| | exec-match | asks | tokens said per request |", "|---|---|---|---|",
                "| §6 direct answer (the served row) | %s | 0%% | %.1f |" % (
                    pct(mean(d)), mean(r["direct"]["tokens"] for r in rows)),
                "| **readings first** | %s | %s | %.1f |" % (
                    pct(mean(ee)), pct(mean(a)), mean(e2e_tokens(r) for r in rows)),
                "| oracle with the generic question (2026-10-06) | %s | %s | — |" % (
                    pct(o_match), pct(o_rate)), "",
                "- readings first against the §6 answer: %s" % paired(d, ee)]
        if has_cls:
            for c in classes:
                idx = [i for i, r in enumerate(rows) if r.get("note_class") == c]
                out.append("- *%s* rows (%d): %s" % (c, len(idx), paired([d[i] for i in idx],
                                                                       [ee[i] for i in idx])))
    out.append("")

    # 5. examples
    out += ["## 5. Examples", ""]
    picks = []
    for c in (classes or [None]):
        sub = [r for r in asked if r.get("note_class") == c][:3]
        picks += sub
    for r in picks:
        out += ["- **%s** (%s) %s  " % (r["id"], r.get("note_class") or r.get("category"),
                                      r["instruction"]),
                "  readings: %d, distinct: %d; asks: %s  " % (r["n_readings"], r["distinct"],
                                                             r["question"]),
                "  reply: %s; turn 2: `%s` %s (direct `%s` %s)" % (
                    r["reply"], r["readings_real"]["completion"],
                    "✓" if r["readings_real"]["match"] else "✗",
                    r["direct"]["completion"], "✓" if r["direct"]["match"] else "✗")]
    return out


def summary(rows):
    a = [int(r["asked"]) for r in rows]
    s = {"n": len(rows), "ask_rate": mean(a),
         "gold_in_readings": mean(int(r["gold_in_readings"]) for r in rows),
         "direct": mean(m("direct")(r) for r in rows)}
    if all(r["readings_real"] or not r["asked"] for r in rows):
        s["readings_first"] = mean(e2e(r) for r in rows)
    for lab, rate, match in policy_rows(rows):
        s["policy: " + lab.strip("*")] = {"ask_rate": rate, "match": match}
    for c in CLASSES:
        sub = [r for r in rows if r.get("note_class") == c]
        if sub:
            s["ask_rate_" + c] = mean(int(r["asked"]) for r in sub)
    return s


# --- transcripts ----------------------------------------------------------------

READING_STEPS = ("readings", "reading 1 (the §6 answer)", "further reading")


def readings_calls(case, r):
    """The readings step's calls, with the messages rebuilt from the code:
    one call for the list elicitation; for the iterative one, the §6 prompt,
    then ALT_SYSTEM with the first k readings for the k-th further call."""
    meta = r["readings"]
    if meta.get("elicit") != "iterative":
        return [{"step": "readings", "model": "student", "messages": R.readings_messages(case),
                 "prefill": "", "completion": meta["completion"],
                 "prompt_tokens": meta["prompt_tokens"], "output_tokens": meta["output_tokens"]}]
    from prompt import messages_for_case
    readings = [(it["mode"], it["command"]) for it in r["items"]]
    out = []
    for k, call in enumerate(meta["calls"]):
        msgs = messages_for_case(case) if k == 0 else R.alt_messages(case, readings[:k])
        out.append({"step": "reading 1 (the §6 answer)" if k == 0 else "further reading",
                    "model": "student", "messages": msgs, "prefill": "",
                    "completion": call["completion"], "prompt_tokens": call["prompt_tokens"],
                    "output_tokens": call["output_tokens"]})
    return out


def transcript_row(case, r, rep, t2r, t2p, verdict):
    """Every call of one case, with the exact messages the model was given."""
    calls = readings_calls(case, r)
    runs = [{"reading": "%s: %s" % (it["mode"], it["command"]), "status": it["status"],
             "group": it["group"], "matches_expected": it["match"], "error": it["error"],
             "buffer_after": it["buffer_after"]} for it in r["items"]]
    if r["asked"]:
        readings = [(it["mode"], it["command"]) for it in r["items"]]
        groups = [{"members": g["members"],
                   "buffer_after": r["items"][g["members"][0]]["buffer_after"]}
                  for g in r["groups"]]
        q = r["question"]
        calls.append({"step": "question", "model": "student",
                      "messages": R.question_messages(case, readings, groups),
                      "prefill": "ask:", "completion": q["raw"],
                      "prompt_tokens": q["prompt_tokens"], "output_tokens": q["output_tokens"]})
        if rep:
            calls.append({"step": "simulated user", "model": "simulator",
                          "question": rep["question"], "reply": rep["reply"],
                          "check": rep.get("check"), "think_tokens": rep.get("think_tokens")})
        for step, t2, reply in (("turn 2, simulated reply", t2r, rep["reply"] if rep else None),
                                ("turn 2, placebo reply", t2p, PLACEBO_REPLY)):
            if t2 is None or reply is None:
                continue
            calls.append({"step": step, "model": "student",
                          "messages": turn2_messages(case, q["text"], reply), "prefill": "",
                          "completion": t2["answer"]["completion"],
                          "prompt_tokens": t2["answer"]["prompt_tokens"],
                          "output_tokens": t2["answer"]["output_tokens"]})
    return {"id": case["id"], "case": {k: case.get(k) for k in (
                "id", "category", "tier", "note_class", "instruction", "mode", "cursor",
                "buffer_before", "buffer_after", "gold_command")},
            "distinct": r["distinct"], "asked": r["asked"],
            "stop": r["readings"].get("stop"), "runs": runs, "calls": calls,
            "outcomes": {k: (verdict[k]["outcome"] if verdict.get(k) else None)
                         for k in ("direct", "readings_real", "readings_placebo")}}


def check_prompts(rows, tok, kwargs):
    """Render each rebuilt student prompt (plus its prefill) and compare its
    token count with the one recorded at generation. {step: [matched, checked]}."""
    tally = {}
    for row in rows:
        for call in row["calls"]:
            if call["model"] != "student":
                continue
            text = tok.apply_chat_template(call["messages"], tokenize=False,
                                           add_generation_prompt=True, **kwargs) + call["prefill"]
            call["prompt_check"] = len(tok(text)["input_ids"]) == call["prompt_tokens"]
            t = tally.setdefault(call["step"], [0, 0])
            t[0] += call["prompt_check"]
            t[1] += 1
    return tally


def fence(s):
    s = "" if s is None else str(s)
    tick = "```" if "```" not in s else "````"
    return "%s\n%s\n%s" % (tick, s, tick)


def to_markdown(row):
    c = row["case"]
    out = ["## %s (%s)" % (row["id"], c.get("note_class") or c.get("category")), "",
           "**Instruction:** %s  " % c["instruction"],
           "**Expected:** `%s: %s`; distinct readings %d; asked: %s%s" % (
               c["mode"], c["gold_command"], row["distinct"], row["asked"],
               "; the search stopped: %s" % row["stop"] if row.get("stop") else ""), "",
           "Buffer before:", fence(c["buffer_before"]), ""]
    n_read = sum(call["step"] in READING_STEPS for call in row["calls"])
    for k, call in enumerate(row["calls"]):
        out.append("### %s" % call["step"])
        if call["model"] == "simulator":
            out += ["question: %s" % call["question"], "", "reply: %s" % call["reply"], ""]
            continue
        for msg in call["messages"]:
            out += ["*%s*" % msg["role"], fence(msg["content"])]
        out += ["*assistant*%s" % (" (prefilled `%s`)" % call["prefill"] if call["prefill"] else ""),
                fence(call["completion"]), ""]
        if k == n_read - 1:                     # after the last readings call: what ran
            for run in row["runs"]:
                out.append("- `%s` -> %s%s%s" % (
                    run["reading"], run["status"],
                    "" if run["group"] is None else ", result %d" % (run["group"] + 1),
                    ", the expected result" if run["matches_expected"] else ""))
            out.append("")
    out.append("Outcomes: %s" % ", ".join("%s %s" % kv for kv in row["outcomes"].items()))
    return "\n".join(out) + "\n"


# --- fingerprint ----------------------------------------------------------------

def prompt_fingerprint():
    """sha256 over the readings prompts and builders, and the clarify
    fingerprint (turn 2 and the simulated user come from scripts/clarify/)."""
    import inspect
    import transcripts as clarify_transcripts
    parts = [R.READINGS_SYSTEM, R.ALT_SYSTEM, R.QUESTION_SYSTEM,
             clarify_transcripts.prompt_fingerprint()]
    for fn in (R.readings_messages, R.alt_user_turn, R.alt_messages, R.question_user_turn,
               R.question_messages, R.parse_readings, R.parse_alt):
        parts.append(inspect.getsource(fn))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


# --- main -----------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=("prepare", "score", "report", "transcripts", "fingerprint"))
    ap.add_argument("--cases", help="the set's case file")
    ap.add_argument("--dir", help="the set's directory in the run")
    ap.add_argument("--label", default="served", help="the readings file's label")
    ap.add_argument("--ref", help="score: the 2026-10-06 verdicts for this set and label")
    ap.add_argument("--name", help="report: the set's name")
    ap.add_argument("--run", help="fingerprint: the run directory")
    ap.add_argument("--model", default="qwen3.5-4b", help="transcripts: registry name")
    args = ap.parse_args()

    if args.phase == "fingerprint":
        path = pathlib.Path(args.run) / "prompt_fingerprint.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"prompt_fingerprint": prompt_fingerprint(),
                                        "recorded": datetime.datetime.now().isoformat(
                                            timespec="seconds")}, indent=1) + "\n")
            print("recorded %s" % path)
        else:
            print("fingerprint already recorded: %s" % path)
        return 0

    d = pathlib.Path(args.dir)
    cases = load_jsonl(args.cases)
    readings_rows = load_jsonl(str(d / ("readings_%s.jsonl" % args.label)))
    if args.phase == "prepare":
        questions, asked, placebo = prepare(readings_rows, cases)
        write_jsonl(d / "questions_readings.jsonl", questions)
        write_jsonl(d / "asked_cases.jsonl", asked)
        write_jsonl(d / "placebo_replies.jsonl", placebo)
        print("%d of %d rows ask -> %s" % (len(asked), len(readings_rows), d))
        return 0

    def opt(name):
        p = d / name
        return load_jsonl(str(p)) if p.exists() else []
    t2r, t2p, replies = opt("turn2_readings.jsonl"), opt("turn2_readings_placebo.jsonl"), opt("replies.jsonl")
    if args.phase == "score":
        from score import execute
        ref = {r["id"]: r for r in load_jsonl(args.ref)}
        answers = {}
        for cond, rows in (("readings_real", t2r), ("readings_placebo", t2p)):
            for r in rows:
                answers.setdefault(r["id"], {})[cond] = r["answer"]
        executed = execute(cases, answers)
        rows = verdict_rows(cases, readings_rows, t2r, t2p, replies, ref, executed)
        write_jsonl(d / "verdicts_readings.jsonl", rows)
        asked = [r for r in rows if r["asked"]]
        print("%d rows -> %s; asks on %d; readings-first exec-match %s (direct %s)" % (
            len(rows), d / "verdicts_readings.jsonl", len(asked),
            pct(mean(e2e(r) for r in rows)) if all(r["readings_real"] or not r["asked"] for r in rows)
            else "—", pct(mean(m("direct")(r) for r in rows))))
        return 0

    verdicts = load_jsonl(str(d / "verdicts_readings.jsonl"))
    if args.phase == "report":
        (d / "report.md").write_text("\n".join(report_lines(
            verdicts, args.name or d.name, d / "verdicts_readings.jsonl")) + "\n", encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(summary(verdicts), indent=1) + "\n")
        print("wrote %s and summary.json" % (d / "report.md"))
        return 0

    # transcripts
    from transformers import AutoTokenizer
    from baseline import load_registry
    reg = load_registry(args.model)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    by = lambda rows: {r["id"]: r for r in rows}                          # noqa: E731
    rd, rp, r2, p2, vd = by(readings_rows), by([r for r in replies if r["label"] == LABEL]), \
        by(t2r), by(t2p), by(verdicts)
    rows = [transcript_row(c, rd[c["id"]], rp.get(c["id"]), r2.get(c["id"]), p2.get(c["id"]),
                           vd[c["id"]]) for c in cases if c["id"] in rd]
    tally = check_prompts(rows, tok, reg.get("chat_template_kwargs") or {})
    write_jsonl(d / "transcripts_readings.jsonl", rows)
    (d / "transcripts_readings.md").write_text(
        "# Readings-first transcripts: %s\n\n" % d.name + "\n".join(to_markdown(r) for r in rows),
        encoding="utf-8")
    print("transcripts: %d cases; prompt check %s" % (len(rows), ", ".join(
        "%s %d/%d" % (k, v[0], v[1]) for k, v in tally.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
