#!/usr/bin/env python3
"""The loop breaker's calibration: the detector replayed over every recorded thinking trace, and the answers it would force.

`sweep` (CPU): every trace of the thinking run (runs/think27b_2026-11-01) with its continuations
(runs/think_long_2026-11-01) is rebuilt in full, and the detector runs after each complete line of
its reasoning, as it does during generation. For `reps` from 3 to 12 it reports which traces would
be stopped and where; a stop in a trace the model finished on its own is a false alarm. It also
runs the detector over every buffer and instruction in train and val, the rows the thinking runs
read, since the model quoting its own buffer must not look like a loop. The test split is not read.

`replay` (GPU, on the owner's go): each trace the default settings stop is cut where the detector
fires, and break_loop() writes the answer from the recorded reasoning, as it would have during the
run. The answer is judged in the sandbox.

    python scripts/think_loop/calibrate.py sweep     # -> runs/think_loop_2026-11-01/calibration.{md,json}
    python scripts/think_loop/calibrate.py replay    # GPU -> runs/think_loop_2026-11-01/replay.{jsonl,md}
"""
import argparse
import datetime
import importlib.util
import json
import pathlib
import sys
import zlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = load("think_long_extend", REPO / "scripts" / "think_long" / "extend.py")
T, LB = E.T, E.LB
OUT = REPO / "runs" / "think_loop_2026-11-01"
SPLITS = ("train", "val")                # the rows thinking runs on; the test split is not read
FIELDS = ("buffer_before", "buffer_after", "instruction")
SWEEP = (3, 4, 5, 6, 8, 10, 12)
SEED = 44000000                          # apart from the thinking runs' seeds


# ------------------------------------------------------------ the traces

def traces(first_run=E.FIRST_RUN, run=E.RUN):
    """Every thinking row, rebuilt from its records: the reasoning (the text before the first
    </think>), how it ended, its tokens and its GPU seconds."""
    byid = {arm: {r["id"]: r for r in rows} for arm, rows in E.first_rows(first_run).items()}
    stages = {cap: {arm: E.stage_rows(run, cap, arm) for arm in E.THINK_ARMS} for cap in E.CAPS}
    out = []
    for arm in E.THINK_ARMS:
        for i in E.arm_orders(first_run)[arm]:
            recs = E.chain(i, byid, stages, arm, float("inf"))
            if not recs:
                continue
            text = E.text_so_far(recs)
            k = text.find(T.THINK_END)
            last = recs[-1]
            out.append({"arm": arm, "id": i, "closed": last["closed"], "match": bool(last["match"]),
                        "cap": E.record_cap(last), "tokens": E.tokens_so_far(recs),
                        "seconds": sum(r.get("seconds", r.get("batch_seconds", 0.0)) for r in recs),
                        "reasoning": text if k < 0 else text[:k]})
    return out


def longest(lines, period_max, min_chars):
    """The most times any block of <= period_max lines (with at least min_chars non-space
    characters) is said in a row: the detector stops a trace at `reps` iff this reaches `reps`."""
    best = 1 if lines else 0
    for p in range(1, period_max + 1):
        run = 0
        for j in range(p, len(lines)):
            run = run + 1 if lines[j][2] == lines[j - p][2] else 0
            if run >= p and sum(len(x[2].replace(" ", "")) for x in lines[j - p + 1:j + 1]) >= min_chars:
                best = max(best, 1 + run // p)
    return best


def first_stop(lines, settings):
    """(line index, loop) where the detector first fires, line by line as during generation."""
    kw = LB.detector(settings)
    for i in range(len(lines)):
        loop = LB.loop_in(lines[:i + 1], **kw)
        if loop:
            return i, loop
    return None, None


# ------------------------------------------------------------ sweep

def sweep(settings=LB.DEFAULT, splits_dir=REPO / "data" / "splits", tok=None):
    if tok is None:
        from transformers import AutoTokenizer
        from baseline import load_registry
        reg = load_registry(T.THINK_MODEL)
        tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])

    def count(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    pm, mc = settings["period_max"], settings["min_chars"]
    rows = traces()
    for t in rows:
        t["lines"] = LB.lines_of(t["reasoning"])
        t["longest"] = longest(t["lines"], pm, mc)
    rep = {"settings": settings, "traces": len(rows), "finished": sum(t["closed"] for t in rows),
           "longest_in_finished": max(t["longest"] for t in rows if t["closed"]),
           "longest_in_finished_any_length": max(longest(t["lines"], pm, 0) for t in rows if t["closed"]),
           "sweep": {}, "stops": [], "buffers": {}}
    for r in SWEEP:
        hit = [t for t in rows if t["longest"] >= r]
        rep["sweep"][str(r)] = {"stopped": ["%s %s" % (t["arm"], t["id"]) for t in hit],
                                "false_alarms": sum(t["closed"] for t in hit)}
    for t in rows:
        if t["longest"] < settings["reps"]:          # it never fires here: skip the line-by-line replay
            continue
        i, loop = first_stop(t["lines"], settings)
        at = count(t["reasoning"][:t["lines"][i][1]])
        rep["stops"].append({
            "arm": t["arm"], "id": t["id"], "finished": t["closed"], "match": t["match"], "trace_tokens": t["tokens"],
            "period": loop["period"], "block": loop["block"], "loop_starts_at": count(t["reasoning"][:loop["start"]]),
            "stops_at": at, "kept_tokens": count(t["reasoning"][:loop["keep"]].rstrip("\n")),
            "tokens_saved": t["tokens"] - at,
            "gpu_minutes_saved": round((t["tokens"] - at) * t["seconds"] / max(t["tokens"], 1) / 60, 1)})
    for split in SPLITS:
        cases = [json.loads(line) for line in open(splits_dir / ("%s.jsonl" % split), encoding="utf-8")]
        worst = {f: max(longest(LB.lines_of((c.get(f) or "") + "\n"), pm, mc) for c in cases) for f in FIELDS}
        rep["buffers"][split] = {"rows": len(cases), "longest": worst,
                                 "stopped_at_default": sum(longest(LB.lines_of((c.get(f) or "") + "\n"), pm, mc)
                                                           >= settings["reps"] for c in cases for f in FIELDS)}
    return rep


def render(rep):
    s = rep["settings"]
    L = ["# The loop breaker, replayed over the recorded thinking traces", "",
         "Built by `scripts/think_loop/calibrate.py sweep` (CPU). A trace is stopped once its reasoning's last `reps` × p "
         "complete lines are one block of p ≤ %d lines said `reps` times in a row (blank lines skipped, whitespace "
         "collapsed, ≥ %d non-space characters per block). Default `reps` = %d." % (s["period_max"], s["min_chars"], s["reps"]),
         "", "- Traces: %d (%d finished by the model, %d never finished)." % (
             rep["traces"], rep["finished"], rep["traces"] - rep["finished"]),
         "- The most times a finished trace says one block in a row: **%d** (%d counting blocks under %d characters)."
         % (rep["longest_in_finished"], rep["longest_in_finished_any_length"], s["min_chars"]), "",
         "## Which traces each threshold stops", "", "| reps | stopped | false alarms (finished traces) |", "|---|---|---|"]
    for r, x in rep["sweep"].items():
        L.append("| %s | %d: %s | %d |" % (r, len(x["stopped"]), ", ".join("`%s`" % y for y in x["stopped"]) or "—",
                                           x["false_alarms"]))
    L += ["", "## Where the default stops them", "",
          "| trace | finished | period | loop starts | stopped at | kept | trace length | tokens saved | GPU minutes saved |",
          "|---|---|---|---|---|---|---|---|---|"]
    for x in rep["stops"]:
        L.append("| `%s %s` | %s | %d | %d | %d | %d | %d | %d | %.1f |" % (
            x["arm"], x["id"], x["finished"], x["period"], x["loop_starts_at"], x["stops_at"], x["kept_tokens"],
            x["trace_tokens"], x["tokens_saved"], x["gpu_minutes_saved"]))
    for x in rep["stops"]:
        L += ["", "`%s %s` repeats:" % (x["arm"], x["id"]), "", "```"] + x["block"] + ["```"]
    L += ["", "## The splits' own text", "",
          "The model quotes its buffer while it reasons, so no buffer or instruction of the rows thinking runs on "
          "may look like a loop. The test split is not read.", "",
          "| split | rows | most repeats: buffer before / after / instruction | fields stopped at the default |",
          "|---|---|---|---|"]
    for split, b in rep["buffers"].items():
        L.append("| %s | %d | %s | %d |" % (split, b["rows"], " / ".join(str(b["longest"][f]) for f in FIELDS),
                                            b["stopped_at_default"]))
    L += ["", "Not shown here: whether the forced answers are right. That needs the model (`calibrate.py replay`)."]
    return "\n".join(L) + "\n"


def cmd_sweep(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = sweep()
    (out / "calibration.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (out / "calibration.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


# ------------------------------------------------------------ replay (GPU)

def cmd_replay(args):
    out = pathlib.Path(args.out)
    rep = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    settings = rep["settings"]
    rows = {(t["arm"], t["id"]): t for t in traces()}
    val = {c["id"]: c for c in E.load_jsonl(str(T.VAL))}
    train = {c["id"]: c for c in E.load_jsonl(str(T.TRAIN))}
    plan = {r["id"]: r for r in E.load_jsonl(str(T.PLAN))}
    tok, model, info = T.load_model()
    done = []
    with (out / "replay.jsonl").open("w", encoding="utf-8") as fh:
        for x in rep["stops"]:
            t = rows[(x["arm"], x["id"])]
            lines = LB.lines_of(t["reasoning"])
            i, _ = first_stop(lines, settings)
            reasoning = t["reasoning"][:lines[i][1]]            # what the run had generated when the breaker fired
            case = val[t["id"]]
            prompt = E.prompt_text(tok, case, t["arm"], train, plan, info)
            seed = SEED + zlib.crc32(("%s|%s" % (t["arm"], t["id"])).encode()) % 100000
            answer, loop = LB.break_loop(T.generate, model, tok, prompt, reasoning, T.ARMS[t["arm"]][1], info,
                                         seed, settings)
            parsed, per = T.judge([case], [answer], args.timeout)
            p, r = parsed[case["id"]], per.get(case["id"], {})
            row = {"arm": t["arm"], "id": t["id"], "answer": answer, "parse_ok": p is not None,
                   "mode": p[0] if p else None, "command": p[1] if p else None, "valid": bool(r.get("valid")),
                   "match": bool(r.get("match")), "error": r.get("error"), "gold": case["gold_command"],
                   "loop": loop, "seed": seed, "time": datetime.datetime.now().isoformat(timespec="seconds")}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done.append(row)
            print("  %s %s: %r -> match %s" % (t["arm"], t["id"], answer, row["match"]), flush=True)
    L = ["# The loop breaker's forced answers on the recorded loops", "",
         "Built by `scripts/think_loop/calibrate.py replay`: each trace cut where the detector fires, reasoning kept to "
         "one copy of the block, closed, answered with %d tokens at most." % settings["answer_tokens"], "",
         "| trace | kept tokens | answer | parsed | valid | right | gold |", "|---|---|---|---|---|---|---|"]
    for row in done:
        L.append("| `%s %s` | %d | `%s` | %s | %s | %s | `%s` |" % (
            row["arm"], row["id"], row["loop"]["kept_tokens"], (row["answer"] or "").strip().replace("|", "\\|"),
            row["parse_ok"], row["valid"], row["match"], row["gold"].replace("|", "\\|")))
    (out / "replay.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="CPU: the detector over the recorded traces and the splits")
    s.add_argument("--out", default=str(OUT))
    g = sub.add_parser("replay", help="GPU: the forced answers on the traces the default stops")
    g.add_argument("--out", default=str(OUT))
    g.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return cmd_sweep(args) if args.cmd == "sweep" else cmd_replay(args)


if __name__ == "__main__":
    sys.exit(main())
