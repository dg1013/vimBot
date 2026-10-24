#!/usr/bin/env python3
"""Every prompt and every action of a clarification run, one case at a time.

For hand analysis and error analysis. Joins, per case and checkpoint, the
case itself, every model call in the order the protocol makes it -- with the
exact chat messages it was given -- what each call said, the simulated user's
prompt and full reasoning, and what vim did with every answer (the buffer it
left, its stderr, and a diff against the result):

    calls      direct (the §6 prompt) -> offer (the first-token distribution)
               -> noask (ask banned) -> question ("ask:" prefilled)
               -> simulator (the 9B's view, reasoning, check and reply)
               -> placebo (turn 2 after the placebo reply)
               -> real (turn 2 after the simulated reply)
    outcomes   per condition: the outcome class (score.outcome_class), the
               command, the buffer vim left, its stderr, and a unified diff
               of the expected buffer against the produced one

The messages are rebuilt with the same protocol functions the run used; the
generation records hold no prompt text, only its token count. So every
rebuilt student prompt is rendered through the tokenizer's chat template and
its token count checked against the count recorded at generation time
(`prompt_check`): equal counts on every call mean the rebuilt prompt is the
one the model saw (on the 2026-10-06 val run: 3,270 of 3,270 before turn 2
existed). The simulator's prompt carries no recorded count, so
`transcripts_manifest.json` records every script's sha256 and modification
time beside the run's first status line: the generation code must predate
the launch for the rebuilt prompts to be the ones used.

Writes, per checkpoint, `transcripts_<label>.jsonl` (the data) and
`transcripts_<label>.md` (the same, readable: one section per case).
"""
import argparse
import datetime
import difflib
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl                                   # noqa: E402
from prompt import messages_for_case                              # noqa: E402
from protocol import (ASK_PREFIX, PLACEBO_REPLY, sim_messages,    # noqa: E402
                      turn1_messages, turn2_messages)

CASE_FIELDS = ("id", "category", "tier", "source", "mode", "cursor", "instruction",
               "buffer_before", "gold_command", "buffer_after", "note_class", "note")
ANSWER_FIELDS = ("completion", "output_tokens", "prompt_tokens", "latency_s", "parse_ok",
                 "mode", "command", "second_ask")


def diff_vs_expected(expected, produced):
    """Unified diff, expected -> produced; None when there is nothing to diff."""
    if produced is None:
        return None
    a, b = expected.split("\n"), produced.split("\n")
    lines = list(difflib.unified_diff(a, b, "expected", "produced", lineterm="", n=1))
    return "\n".join(lines) or None


def answer_call(step, messages, rec, **extra):
    call = {"step": step, "model": "student", "messages": messages}
    call.update(extra)
    call.update({k: rec[k] for k in ANSWER_FIELDS if k in rec})
    return call


def build_row(case, t1, t2, rep, verdict, label):
    """One case's complete record (see the module docstring)."""
    q = t1["question"]["text"]
    calls = [
        answer_call("direct", messages_for_case(case), t1["direct"]),
        {"step": "offer", "model": "student", "messages": turn1_messages(case),
         "kind": "one forward pass: the first-token distribution", **t1["offer"]},
        answer_call("noask", turn1_messages(case), t1["noask"], ban_first=["ask", "Ask"]),
        {"step": "question", "model": "student", "messages": turn1_messages(case),
         "prefill": ASK_PREFIX, "question": q, "raw": t1["question"]["raw"],
         "output_tokens": t1["question"]["output_tokens"],
         "latency_s": t1["question"]["latency_s"],
         "capped": verdict.get("question_capped")},
        {"step": "simulator", "model": rep.get("simulator"), "messages": sim_messages(case, q),
         **{k: rep.get(k) for k in ("think", "think_tokens", "forced", "check", "reply", "raw",
                                    "reply_tokens", "latency_s", "leak_flags",
                                    "quotes_result")}},
        answer_call("placebo", turn2_messages(case, q, PLACEBO_REPLY), t1["placebo"]),
        answer_call("real", turn2_messages(case, q, rep["reply"]), t2["answer"]),
    ]
    outcomes = {}
    for cond in ("direct", "noask", "placebo", "real"):
        v = dict(verdict[cond])
        v["diff_vs_expected"] = (diff_vs_expected(case["buffer_after"], v.get("buffer_after"))
                                 if not v["match"] else None)
        outcomes[cond] = v
    return {"id": case["id"], "label": label,
            "case": {k: case[k] for k in CASE_FIELDS if k in case},
            "calls": calls, "outcomes": outcomes,
            "policy": {k: verdict.get(k) for k in ("p_ask_norm", "ask_margin", "chose_ask",
                                                   "free_consistent", "first_consistent")}}


def check_prompts(rows, tok, kwargs):
    """Render each rebuilt student prompt; compare its token count with the record.

    Returns {step: [matched, checked]} and marks each call with
    ``prompt_check`` True/False. The question call carries no count of its
    own: its prompt is the noask prompt plus the prefill.
    """
    tally = {}
    for row in rows:
        for call in row["calls"]:
            if call["model"] != "student" or "prompt_tokens" not in call:
                continue
            text = tok.apply_chat_template(call["messages"], tokenize=False,
                                           add_generation_prompt=True, **kwargs)
            n = len(tok(text)["input_ids"])
            call["prompt_check"] = n == call["prompt_tokens"]
            t = tally.setdefault(call["step"], [0, 0])
            t[0] += call["prompt_check"]
            t[1] += 1
    return tally


def fence(s):
    """A fenced block that survives backticks inside the buffer."""
    s = "" if s is None else s
    ticks = "```"
    while ticks in s:
        ticks += "`"
    return "%s\n%s\n%s" % (ticks, s.rstrip("\n"), ticks)


def to_markdown(row):
    c = row["case"]
    o = row["outcomes"]
    calls = {x["step"]: x for x in row["calls"]}
    out = ["## %s — %s (%s, %s)" % (row["id"], row["label"], c.get("category"), c.get("tier")),
           "", "**Instruction:** %s  " % c["instruction"],
           "**Cursor:** line %d, column %d · **gold** (%s): `%s`" % (
               *c.get("cursor", (1, 1)), c.get("mode"), c["gold_command"])]
    if c.get("note_class"):
        out.append("**Note (%s):** %s" % (c["note_class"], c.get("note", "")))
    out += ["", "Buffer before:", fence(c["buffer_before"]), "Expected after:",
            fence(c["buffer_after"]), "",
            "| condition | outcome | command | vim |", "|---|---|---|---|"]
    for cond in ("direct", "noask", "placebo", "real"):
        v = o[cond]
        cmd = ("%s: %s" % (v["mode"], v["command"])) if v["command"] is not None else "—"
        out.append("| %s | %s | `%s` | %s |" % (cond, v["outcome"], cmd.replace("|", "\\|"),
                                               (v.get("error") or "").replace("|", "\\|")))
    p = row["policy"]
    out += ["", "P(ask) %.4g (log-odds %.2f); greedy %s." % (
        p["p_ask_norm"], p["ask_margin"], "asks" if p["chose_ask"] else "answers"),
        "", "**Asks:** %s" % calls["question"]["question"],
        "", "**Simulated user, reasoning** (%s tokens%s):" % (
            calls["simulator"].get("think_tokens"),
            ", cut at the budget" if calls["simulator"].get("forced") else ""),
        fence(calls["simulator"].get("think") or ""),
        "**Check:** %s  " % calls["simulator"].get("check"),
        "**Reply:** %s%s" % (calls["simulator"].get("reply"),
                             "  (flags: %s)" % ", ".join(calls["simulator"]["leak_flags"])
                             if calls["simulator"].get("leak_flags") else "")]
    for cond in ("direct", "noask", "placebo", "real"):
        v = o[cond]
        if v.get("diff_vs_expected"):
            out += ["", "%s left a different buffer:" % cond, fence(v["diff_vs_expected"])]
        if v.get("stderr"):
            out += ["%s, vim said: `%s`" % (cond, v["stderr"][:300].replace("`", "'"))]
    return "\n".join(out) + "\n"


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def prompt_fingerprint():
    """sha256 over everything that decides what a model is shown.

    The prompt texts and the source of every function that builds a message,
    from protocol.py and src/prompt.py -- and nothing else, so a change to,
    say, the leak screen leaves it alone. A run records it at launch
    (prompt_fingerprint.json); the transcripts' rebuilt prompts are the ones
    used only if the fingerprint is unchanged.
    """
    import inspect
    import prompt as section6
    import protocol as P
    parts = [section6.SYSTEM_TEXT, P.ASK_CLAUSE, P.PLACEBO_REPLY, P.ASK_PREFIX, P.SIM_SYSTEM]
    for fn in (section6.render_user, section6.build_messages, section6.messages_for_case,
               P.turn1_messages, P.turn2_messages, P.sim_messages, P.sim_user_turn,
               P.visible_diff, P.char_changes, P.cursor_char, P._q, P._lines, P._span):
        parts.append(inspect.getsource(fn))
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def record_fingerprint(run_dir):
    """Write $RUN/prompt_fingerprint.json once, at launch (run.sh does)."""
    path = pathlib.Path(run_dir) / "prompt_fingerprint.json"
    if path.exists():
        return path, False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "prompt_fingerprint": prompt_fingerprint(),
        "recorded": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1) + "\n")
    return path, True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record-fingerprint", metavar="RUN_DIR",
                    help="only record the prompt fingerprint for a run being launched, and exit")
    ap.add_argument("--cases")
    ap.add_argument("--dir", help="a set's directory, e.g. $RUN/val")
    ap.add_argument("--label", action="append")
    ap.add_argument("--model", default="qwen3.5-4b", help="the student's registry name")
    ap.add_argument("--no-check", action="store_true", help="skip the prompt token check")
    args = ap.parse_args()
    if args.record_fingerprint:
        path, new = record_fingerprint(args.record_fingerprint)
        print("%s %s" % ("recorded" if new else "kept the launch's", path))
        return 0
    if not (args.cases and args.dir and args.label):
        ap.error("--cases, --dir and --label are required to build transcripts")
    d = pathlib.Path(args.dir)
    cases = {c["id"]: c for c in load_jsonl(args.cases)}
    replies_all = load_jsonl(str(d / "replies.jsonl"))
    tok = kwargs = None
    if not args.no_check:
        from transformers import AutoTokenizer
        from baseline import load_registry
        reg = load_registry(args.model)
        tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
        kwargs = reg.get("chat_template_kwargs", {})
    # The code that generated the data must be the code that rebuilds its
    # prompts: every script's hash and modification time, beside the run's
    # first status line, lets a reader check none changed after the launch.
    manifest = {"built": datetime.datetime.now().isoformat(timespec="seconds"),
                "code": {p.name: {"sha256": sha256(p), "mtime": datetime.datetime.fromtimestamp(
                    p.stat().st_mtime).isoformat(timespec="seconds")}
                    for p in sorted(HERE.glob("*.py")) + [HERE / "run.sh"]},
                "labels": {}}
    status = d.parent / "status.txt"
    if status.exists():
        manifest["run_status_first_line"] = status.read_text().splitlines()[0]
    recorded = d.parent / "prompt_fingerprint.json"
    now = prompt_fingerprint()
    manifest["prompt_fingerprint"] = {"now": now}
    if recorded.exists():
        run_fp = json.loads(recorded.read_text())["prompt_fingerprint"]
        manifest["prompt_fingerprint"].update(run=run_fp, same=run_fp == now)
        if run_fp != now:
            print("WARNING: the prompt code changed since the run was launched; the rebuilt "
                  "messages may not be the ones the models saw")
    for label in args.label:
        t1 = {r["id"]: r for r in load_jsonl(str(d / ("turn1_%s.jsonl" % label)))}
        t2 = {r["id"]: r for r in load_jsonl(str(d / ("turn2_%s.jsonl" % label)))}
        rep = {r["id"]: r for r in replies_all if r["label"] == label}
        ver = {r["id"]: r for r in load_jsonl(str(d / ("verdicts_%s.jsonl" % label)))}
        rows = [build_row(cases[cid], t1[cid], t2[cid], rep[cid], ver[cid], label)
                for cid in ver]
        tally = check_prompts(rows, tok, kwargs) if tok is not None else {}
        with (d / ("transcripts_%s.jsonl" % label)).open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        (d / ("transcripts_%s.md" % label)).write_text(
            "# Transcripts: %s, %s\n\n" % (label, d.name)
            + "\n".join(to_markdown(r) for r in rows), encoding="utf-8")
        manifest["labels"][label] = {"cases": len(rows), "prompt_check": tally}
        print("%s: %d cases -> transcripts_%s.jsonl / .md; prompt check %s" % (
            label, len(rows), label,
            ", ".join("%s %d/%d" % (k, v[0], v[1]) for k, v in tally.items()) or "skipped"))
        bad = [k for k, v in tally.items() if v[0] != v[1]]
        if bad:
            print("  WARNING: rebuilt prompts differ from the recorded ones on %s" % bad)
    (d / "transcripts_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
