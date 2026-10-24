#!/usr/bin/env python3
"""Run the retrieval experiment's arms: one model load, several arms, each
written exactly as scripts/baseline.py writes a row (completions, preds, exec
log, baseline_<split>.json), so the report and RESULTS.md read them the same way.

    python scripts/retrieval/run_arms.py --model qwen3.5-4b --run runs/retrieval_2026-10-18 \\
        --arms base-k0:plan_k0 base-random5:plan_random5 base-retr5:plan_retr5
    python scripts/retrieval/run_arms.py --model qwen3.5-4b --run runs/retrieval_2026-10-18 \\
        --adapter runs/qwen3.5-4b-grpo-gated/best --arms adapter-k0:plan_k0 adapter-retr5:plan_retr5

An arm whose baseline_<split>.json exists is skipped, so the same command
resumes after a pause or a crash. Generation is `baseline.generate_predictions`
with `examples_for` reading the arm's plan: the one path every results-table
row went through, now with k-shot pairs when the plan has them.
"""
import argparse
import json
import pathlib
import platform
import statistics
import sys
import time
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl, score, write_exec_log                     # noqa: E402
from baseline import (base_quant_name, generate_predictions, load_base,      # noqa: E402
                      load_registry, percentile, resolve_base_quant, summarize, vim_version)


def run_arm(model, tok, reg, cases, train_by_id, plan_rows, out_dir, args, arm):
    out_dir.mkdir(parents=True, exist_ok=True)
    split = pathlib.Path(args.cases).stem
    plan = {r["id"]: r for r in plan_rows}
    missing = [c["id"] for c in cases if c["id"] not in plan]
    if missing:
        raise SystemExit("plan has no row for %d cases (first: %s)" % (len(missing), missing[0]))

    def examples_for(case):
        return [train_by_id[i] for i in plan[case["id"]]["examples"]]

    comp_path = out_dir / ("completions_%s.jsonl" % split)
    t0 = time.time()
    with comp_path.open("w", encoding="utf-8") as comp:
        seen = []

        def on_record(rec):
            seen.append(rec)
            comp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(seen) % 50 == 0 or len(seen) == len(cases):
                print("  [%s] %d/%d  parsed %d  prompt %.0f tok  out %.1f tok  p50 %.2f s  %.0f s" % (
                    arm, len(seen), len(cases), sum(1 for r in seen if r["parse_ok"]),
                    statistics.fmean(r["prompt_tokens"] for r in seen),
                    statistics.fmean(r["output_tokens"] for r in seen),
                    percentile([r["latency_s"] for r in seen], 50), time.time() - t0), flush=True)

        records, preds = generate_predictions(model, tok, cases, reg, args.max_new_tokens, on_record,
                                              examples_for=examples_for)
    with (out_dir / ("preds_%s.jsonl" % split)).open("w", encoding="utf-8") as fh:
        for cid in sorted(preds):
            fh.write(json.dumps(preds[cid]) + "\n")
    scored = score(cases, preds, timeout=args.timeout)
    write_exec_log(str(out_dir / ("exec_%s.jsonl" % split)), cases, scored)
    import torch, transformers                                              # noqa: E401
    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "vim": vim_version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    ns = SimpleNamespace(split=split, cases=args.cases, max_new_tokens=args.max_new_tokens,
                         run_name=arm, condition=arm, adapter=args.adapter,
                         base_quant=args.base_quant, wandb_state="off")
    summary = summarize(cases, records, scored, reg, ns, versions)
    summary["run"]["experiment"] = "retrieval_2026-10-18"
    summary["run"]["plan"] = str(args.plan_files[arm])
    summary["run"]["k_examples_mean"] = round(statistics.fmean(r["k_examples"] for r in records), 2)
    (out_dir / ("baseline_%s.json" % split)).write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    m = summary["metrics"]
    print("== %s: exec-match %.1f%%  valid %.1f%%  parse failures %d  prompt %.0f tok  out %.1f tok  p50 %.2f s  (%.0f s)" % (
        arm, m["exec_match_pct"], m["valid_pct"], m["parse_failures"], m["prompt_tokens_mean"],
        m["output_tokens_mean"], m["latency_s_p50"], time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="arm:plan_stem pairs, e.g. base-retr5:plan_retr5")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--cases", default=str(REPO / "data/splits/test.jsonl"))
    ap.add_argument("--train", default=str(REPO / "data/splits/train.jsonl"))
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--limit", type=int, default=None, help="first N cases (smoke)")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    split = pathlib.Path(args.cases).stem
    train_by_id = {r["id"]: r for r in load_jsonl(args.train)}
    args.plan_files = {}
    todo = []
    for spec in args.arms:
        arm, stem = spec.split(":")
        args.plan_files[arm] = run / (stem + ".jsonl")
        if (run / arm / ("baseline_%s.json" % split)).exists():
            print("skip %s: done" % arm, flush=True)
            continue
        todo.append(arm)
    if not todo:
        return 0
    reg = load_registry(args.model)
    load_in_4bit = resolve_base_quant(args.adapter) if args.adapter else False
    args.base_quant = base_quant_name(load_in_4bit)
    import torch
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = load_base(reg, load_in_4bit)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("loaded %s (%s)%s in %.0f s on %s; arms: %s" % (
        reg["hf_id"], args.base_quant, " + " + args.adapter if args.adapter else "", time.time() - t0,
        torch.cuda.get_device_name(0), ", ".join(todo)), flush=True)
    for arm in todo:
        plan_rows = load_jsonl(args.plan_files[arm])
        run_arm(model, tok, reg, cases, train_by_id, plan_rows, run / arm, args, arm)
    return 0


if __name__ == "__main__":
    sys.exit(main())
