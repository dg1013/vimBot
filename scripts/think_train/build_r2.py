#!/usr/bin/env python3
"""R2's data (CPU): the kept labels and traces from R1, A1's training file, and A2's reasoning task.

By runs/think_train_2026-11-08/preregistration.md, which carries the plan's Part 2
(runs/teacher_channels_2026-11-02/plan.md) into rules.

    python scripts/think_train/build_r2.py gate     # R1 finished, and its reading L is helps
    python scripts/think_train/build_r2.py build    # -> kept.jsonl, train_a1.jsonl, traces.jsonl, build.json

**Kept.** A row of R1's thinking arm whose answer is a label (R1's report: it matches, and it is the
gold or passes approach 1's check on two longer buffers) and whose reasoning closed on its own: not
at the cap, and not closed by the loop breaker.

**A1's file** is approach 1's 6,548 rows unchanged, then one extra row per kept label: the hard row
with the thinking teacher's answer, its mode and command. Where that answer equals the row's
target, the extra row repeats the row.

**A2's reasoning task** (traces.jsonl) holds, per kept row, TB's own prompt with its 8 retrieved
examples and the recorded completion: the reasoning, `</think>`, the answer, and the template's end
of turn. The prompt is rendered by the student's training-only entry, qwen3.5-4b-think-train, and
must reproduce R1's recorded prompt text and token ids exactly, and the trace must decode to the
recorded text. A row that fails either check stops the build.
"""
import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl                                   # noqa: E402
from prompt import messages_for_case                              # noqa: E402

R1 = REPO / "runs" / "think_label_2026-11-01"
RUN = REPO / "runs" / "think_train_2026-11-08"
A0 = REPO / "runs" / "distill_2026-10-31" / "train_distilled.jsonl"
TRAIN = REPO / "data" / "splits" / "train.jsonl"
THINK, CONTROL = "think-retr8", "nothink-retr8"
STUDENT = "qwen3.5-4b-think-train"     # the registry entry that renders the reasoning task
THINK_END = "</think>"
EXTRA = "@think"                       # suffix of A1's extra rows' ids
MIN_KEPT = 100                         # pre-registered floor: fewer kept rows and R2 does not run
RC_NOT_READY = 3


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def last_copy(rows):
    """{id: row}; a row written twice (a resumed batch) keeps its last copy, as R1's report does."""
    return {r["id"]: r for r in rows}


# ------------------------------------------------------------------ the gate

def gate(r1=R1):
    """(ok, reason): R1's report covers every row of both arms, and L reads helps."""
    r1 = pathlib.Path(r1)
    rep_path = r1 / "report.json"
    if not rep_path.exists():
        return False, "R1's report.json is not written yet"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    n = rep.get("n")
    for arm in (CONTROL, THINK):
        got = (rep.get("arms") or {}).get(arm, {}).get("attempted")
        if got != n:
            return False, "R1's %s arm has %s of %s rows" % (arm, got, n)
    lab = r1 / "labels.jsonl"
    if not lab.exists() or lab.stat().st_mtime < rep_path.stat().st_mtime - 60:
        return False, "R1's labels.jsonl is missing or older than its report"
    verdict = rep.get("verdict")             # R1's report maps compare()'s "stronger" to "helps"
    if verdict != "helps":
        return False, "R1's reading L is %s, not helps" % verdict
    return True, "R1 complete (%d rows per arm); L helps" % n


# ------------------------------------------------------------------ the kept rows

def kept_rows(labels, rows):
    """[(id, label record, R1 row)] for the kept labels, in labels.jsonl's order."""
    out = []
    for lab in labels:
        t, r = lab.get(THINK), rows.get(lab["id"])
        if not t or not t.get("label") or r is None:
            continue
        if not r.get("closed") or r.get("hit_cap") or r.get("forced_close"):
            continue
        if (r["mode"], r["command"]) != (t["mode"], t["command"]):
            raise SystemExit("row %s: labels.jsonl and the R1 row disagree on the answer" % lab["id"])
        out.append((lab["id"], t, r))
    return out


def a1_rows(a0, kept):
    """Approach 1's rows unchanged, then one extra row per kept label with the teacher's answer."""
    by_id = {c["id"]: c for c in a0}
    if len(by_id) != len(a0):
        raise SystemExit("approach 1's file has duplicate ids")
    extra = []
    for rid, t, _ in kept:
        if rid not in by_id:
            raise SystemExit("kept row %s is not in approach 1's file" % rid)
        row = dict(by_id[rid])
        row.update(id=rid + EXTRA, mode=t["mode"], gold_command=t["command"], r2_source="r1-thinking-label")
        extra.append(row)
    return list(a0) + extra


def same_as_target(a0, kept):
    by_id = {c["id"]: c for c in a0}
    return {rid: (by_id[rid]["mode"], by_id[rid]["gold_command"]) == (t["mode"], t["command"])
            for rid, t, _ in kept}


# ------------------------------------------------------------------ the traces

def turn_end(tok, kwargs):
    """The text the template puts after an assistant turn's content (Qwen3.5: '<|im_end|>\\n')."""
    marker = "ZQXJ"
    text = tok.apply_chat_template([{"role": "user", "content": "u"}, {"role": "assistant", "content": marker}],
                                   tokenize=False, **kwargs)
    return text[text.rindex(marker) + len(marker):]


def render_trace(tok, kwargs, case, examples, recorded, row, end):
    """One trace's token ids, after the checks that it is the trace R1 recorded."""
    text = tok.apply_chat_template(messages_for_case(case, examples), add_generation_prompt=True,
                                   tokenize=False, **kwargs)
    if text != recorded["text"]:
        raise SystemExit("row %s: the student's template does not reproduce R1's prompt text" % case["id"])
    p_ids = tok(text)["input_ids"]
    if p_ids != recorded["ids"]:
        raise SystemExit("row %s: the student's tokens differ from R1's recorded prompt ids" % case["id"])
    completion = row["reasoning"] + THINK_END + row["answer"] + end
    c_ids = tok(completion, add_special_tokens=False)["input_ids"]
    if tok.decode(p_ids + c_ids, skip_special_tokens=False) != text + completion:
        raise SystemExit("row %s: the trace does not decode to the recorded text" % case["id"])
    end_ids = tok(end, add_special_tokens=False)["input_ids"]
    if c_ids[-len(end_ids):] != end_ids:
        raise SystemExit("row %s: the trace does not end with the template's end of turn" % case["id"])
    return {"id": case["id"], "prompt_ids": p_ids, "completion_ids": c_ids, "n_prompt": len(p_ids),
            "n_completion": len(c_ids), "recorded_new_tokens": row["new_tokens"],
            "same_count": len(c_ids) - len(end_ids) == row["new_tokens"]}


def quantiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    pick = lambda q: xs[min(len(xs) - 1, int(q * len(xs)))]          # noqa: E731
    return {"min": xs[0], "median": statistics.median(xs), "p90": pick(0.9), "max": xs[-1],
            "mean": round(statistics.fmean(xs), 1)}


def build(r1=R1, run=RUN, a0_path=A0, train_path=TRAIN, tok=None, kwargs=None, min_kept=MIN_KEPT):
    r1, run = pathlib.Path(r1), pathlib.Path(run)
    labels = load_jsonl(str(r1 / "labels.jsonl"))
    rows = last_copy(load_jsonl(str(r1 / ("rows_%s.jsonl" % THINK))))
    kept = kept_rows(labels, rows)
    a0 = load_jsonl(str(a0_path))
    if len(kept) < min_kept:
        return {"kept": len(kept), "stopped": "fewer than %d kept rows" % min_kept}
    if tok is None:
        from transformers import AutoTokenizer
        from baseline import load_registry
        reg = load_registry(STUDENT)
        tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
        kwargs = reg.get("chat_template_kwargs") or {}
    train = {c["id"]: c for c in load_jsonl(str(train_path))}
    plan = {r["id"]: r for r in load_jsonl(str(r1 / "plan_retr8.jsonl"))}
    prompts = {r["id"]: r for r in load_jsonl(str(r1 / "prompts_vllm.jsonl"))}
    end = turn_end(tok, kwargs)
    traces = [render_trace(tok, kwargs, train[rid], [train[j] for j in plan[rid]["examples"]],
                           prompts[rid][THINK], r, end) for rid, _, r in kept]
    a1 = a1_rows(a0, kept)
    same = same_as_target(a0, kept)
    run.mkdir(parents=True, exist_ok=True)
    with (run / "kept.jsonl").open("w", encoding="utf-8") as fh:
        for (rid, t, r), tr in zip(kept, traces):
            lab = next(x for x in labels if x["id"] == rid)
            fh.write(json.dumps({"id": rid, "in_cliff": lab["in_cliff"], "in_never": lab["in_never"],
                                 "tier": lab["tier"], "category": lab["category"], "check": t["check"],
                                 "mode": t["mode"], "command": t["command"], "gold": lab["gold"],
                                 "same_as_target": same[rid], "think_tokens": r["think_tokens"],
                                 "trace_tokens": tr["n_prompt"] + tr["n_completion"]},
                                ensure_ascii=False) + "\n")
    with (run / "train_a1.jsonl").open("w", encoding="utf-8") as fh:
        for c in a1:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    with (run / "traces.jsonl").open("w", encoding="utf-8") as fh:
        for tr in traces:
            fh.write(json.dumps({k: tr[k] for k in ("id", "prompt_ids", "completion_ids")}) + "\n")
    lab_by = {x["id"]: x for x in labels}
    totals = [tr["n_prompt"] + tr["n_completion"] for tr in traces]
    longest = max(traces, key=lambda tr: tr["n_prompt"] + tr["n_completion"])
    summary = {
        "kept": len(kept), "labels_in_thinking_arm": sum(bool((x.get(THINK) or {}).get("label")) for x in labels),
        "dropped": {"unclosed_or_cap": sum(1 for x in labels if (x.get(THINK) or {}).get("label")
                                           and (not rows[x["id"]].get("closed") or rows[x["id"]].get("hit_cap"))),
                    "loop_breaker": sum(1 for x in labels if (x.get(THINK) or {}).get("label")
                                        and rows[x["id"]].get("forced_close"))},
        "by_check": dict(collections.Counter(t["check"] for _, t, _ in kept)),
        "by_source": {"cliff": sum(lab_by[rid]["in_cliff"] for rid, _, _ in kept),
                      "never": sum(lab_by[rid]["in_never"] for rid, _, _ in kept)},
        "by_tier": dict(collections.Counter(lab_by[rid]["tier"] for rid, _, _ in kept)),
        "same_as_target": sum(same.values()),
        "mode_switches_vs_target": sum(1 for rid, t, _ in kept
                                       if t["mode"] != next(c for c in a0 if c["id"] == rid)["mode"]),
        "a0_rows": len(a0), "a1_rows": len(a1),
        "trace_tokens": quantiles(totals), "prompt_tokens": quantiles([tr["n_prompt"] for tr in traces]),
        "completion_tokens": quantiles([tr["n_completion"] for tr in traces]),
        "longest": {"id": longest["id"], "tokens": longest["n_prompt"] + longest["n_completion"]},
        "token_count_as_recorded": sum(tr["same_count"] for tr in traces),
        "turn_end": end, "student_entry": STUDENT,
        "inputs": {"labels": str(r1 / "labels.jsonl"), "labels_sha256": sha256(r1 / "labels.jsonl"),
                   "rows_sha256": sha256(r1 / ("rows_%s.jsonl" % THINK)), "a0": str(a0_path),
                   "a0_sha256": sha256(a0_path)},
    }
    (run / "build.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=("gate", "build"))
    ap.add_argument("--r1", default=str(R1))
    ap.add_argument("--run", default=str(RUN))
    args = ap.parse_args(argv)
    if args.stage == "gate":
        ok, why = gate(args.r1)
        print(why)
        return 0 if ok else RC_NOT_READY
    s = build(args.r1, args.run)
    if "stopped" in s:
        print("kept %d rows: %s; R2 does not run" % (s["kept"], s["stopped"]))
        return RC_NOT_READY
    print("kept %d rows (%s); A1 %d rows; traces %s tokens (median %s, max %s, %s); %d of %d token counts as "
          "recorded" % (s["kept"], ", ".join("%s %d" % kv for kv in sorted(s["by_check"].items())), s["a1_rows"],
                        s["trace_tokens"]["min"], s["trace_tokens"]["median"], s["trace_tokens"]["max"],
                        s["longest"]["id"], s["token_count_as_recorded"], s["kept"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
