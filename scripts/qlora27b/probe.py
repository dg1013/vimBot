#!/usr/bin/env python3
"""The probe before the 27B's full run: build its inputs, then judge it.

The probe is train_27b.py for a few optimizer steps on the longest train rows. That is the worst
case for activation memory, because a micro-batch pads to its longest row. It also runs one
evaluation on a handful of val rows (generation through the RAM embedding) and the final pass that
reloads the adapter (the path full_val_eval takes). It passes when training finished, peak
reserved memory stayed under the bar, and the evaluation answered in the §6 grammar. Its pace is
not used, since long rows make it pessimistic; the full run re-projects its own.

    python scripts/qlora27b/probe.py prepare             # -> runs/qlora27b_2026-10-31/probe_*.jsonl
    python scripts/qlora27b/probe.py verdict <dir> <rc>  # -> <dir>/probe.json; exit 0 = pass
"""
import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl                                   # noqa: E402

RUN = REPO / "runs" / "qlora27b_2026-10-31"
TRAIN = REPO / "runs" / "distill_2026-10-31" / "train_distilled.jsonl"
VAL = REPO / "data" / "splits" / "val.jsonl"
PROBE_ROWS = 256           # 8 optimizer steps at an effective batch of 32
PROBE_VAL_ROWS = 16
BAR_RESERVED_GIB = 20.5    # ~1.5 GiB under the ~22 WSL2 leaves; the sampling run slowed at 21.3
MIN_PARSED_SHARE = 0.75


def row_length(case):
    """Characters the model reads and writes: a stand-in for tokens, used only to rank rows."""
    return len(case["instruction"]) + len(case["buffer_before"]) + len(case["gold_command"])


def longest(cases, n):
    return sorted(cases, key=lambda c: (-row_length(c), c["id"]))[:n]


def write_jsonl(path, rows):
    with pathlib.Path(path).open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def prepare(args):
    from train_sft import stratified_subset
    train, val = load_jsonl(str(args.train)), load_jsonl(str(args.val))
    rows = longest(train, args.rows)
    pval = stratified_subset(val, args.val_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "probe_train.jsonl", rows)
    write_jsonl(args.out / "probe_val.jsonl", pval)
    lens = sorted(row_length(c) for c in train)
    print("probe_train.jsonl: the %d longest of %d rows (%d-%d characters; train median %d)"
          % (len(rows), len(train), row_length(rows[-1]), row_length(rows[0]), lens[len(lens) // 2]))
    print("probe_val.jsonl: %d val rows, category-proportional" % len(pval))
    return 0


def verdict(memory, evals, summary, rc, bar=BAR_RESERVED_GIB, min_share=MIN_PARSED_SHARE):
    """Pass or fail, with every reason it failed."""
    reasons = []
    if rc != 0:
        reasons.append("train_27b.py exited %d" % rc)
    if not memory:
        reasons.append("no memory.json: training did not reach its end")
    elif memory["peak_reserved_gib"] > bar:
        reasons.append("peak reserved %.2f GiB is over the %.1f GiB bar"
                       % (memory["peak_reserved_gib"], bar))
    last = evals[-1] if evals else None
    if last is None:
        reasons.append("no evaluation ran")
    elif last["n"] - last["parse_failures"] < min_share * last["n"]:
        reasons.append("%d of %d answers failed the §6 grammar" % (last["parse_failures"], last["n"]))
    if not summary:
        reasons.append("no sft_val.json: the reload-and-evaluate pass did not finish")
    return {"pass": not reasons, "reasons": reasons, "bar_reserved_gib": bar,
            "memory": memory, "last_eval": last,
            "final_metrics": (summary or {}).get("metrics")}


def judge(args):
    d = pathlib.Path(args.dir)
    read = lambda p: json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    evals = load_jsonl(str(d / "eval_history.jsonl")) if (d / "eval_history.jsonl").exists() else []
    v = verdict(read(d / "memory.json"), evals, read(d / "sft_val.json"), args.rc)
    (d / "probe.json").write_text(json.dumps(v, indent=1) + "\n", encoding="utf-8")
    m = v["memory"] or {}
    print("probe %s: peak reserved %s GiB (allocated %s), %s s/step on the longest rows%s"
          % ("PASSED" if v["pass"] else "FAILED", m.get("peak_reserved_gib"),
             m.get("peak_allocated_gib"), m.get("median_s_per_step_all"),
             "" if v["pass"] else "; " + "; ".join(v["reasons"])))
    return 0 if v["pass"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--train", type=pathlib.Path, default=TRAIN)
    p.add_argument("--val", type=pathlib.Path, default=VAL)
    p.add_argument("--out", type=pathlib.Path, default=RUN)
    p.add_argument("--rows", type=int, default=PROBE_ROWS)
    p.add_argument("--val-rows", type=int, default=PROBE_VAL_ROWS)
    j = sub.add_parser("verdict")
    j.add_argument("dir")
    j.add_argument("rc", type=int)
    args = ap.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else judge(args)


if __name__ == "__main__":
    sys.exit(main())
