#!/usr/bin/env python3
"""Score an examples file with a trained verifier (GPU): logit(yes) - logit(no)
per candidate. The val scores already come out of training; this is for the
pre-registered test run and for re-scoring.

    python scripts/reranker/score.py --model qwen3.5-4b \\
        --init-adapter runs/qwen3.5-4b-grpo-gated/best \\
        --verifier runs/reranker_2026-10-16/verifier-4b/best \\
        --examples runs/reranker_2026-10-16/test_examples.jsonl \\
        --cases data/splits/test.jsonl --out runs/reranker_2026-10-16/verifier-4b/test_scores.jsonl
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", HERE):
    sys.path.insert(0, str(p))
from verifier_lib import (answer_token_ids, check_scoring, load_policy_merged,   # noqa: E402
                          score_examples)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--verifier", required=True, help="the verifier adapter (best/)")
    ap.add_argument("--examples", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args(argv)
    from baseline import load_registry
    from transformers import AutoTokenizer
    from peft import PeftModel
    reg = load_registry(args.model)
    kw = reg.get("chat_template_kwargs") or {}
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    yes_id, no_id = answer_token_ids(tok, kw)
    t0 = time.time()
    model = PeftModel.from_pretrained(load_policy_merged(reg, args.init_adapter), args.verifier)
    rows = [json.loads(l) for l in open(args.examples, encoding="utf-8") if l.strip()]
    cases = {json.loads(l)["id"]: json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()}
    probe = sorted(rows, key=lambda r: len(cases[r["id"]]["buffer_before"]))
    print("scoring check: max diff %.4f" % check_scoring(model, tok, probe[:3] + probe[-3:], cases, kw,
                                                         yes_id, no_id), flush=True)
    scores, trunc = score_examples(model, tok, rows, cases, kw, yes_id, no_id, batch_size=args.batch_size)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r, s in zip(rows, scores):
            fh.write(json.dumps({"id": r["id"], "cand_mode": r["cand_mode"],
                                 "cand_command": r["cand_command"], "score": s}) + "\n")
    print("%d candidates scored in %.0f s (%d truncated) -> %s" % (len(rows), time.time() - t0, trunc, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
