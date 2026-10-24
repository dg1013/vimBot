#!/usr/bin/env python3
"""Score the turn-2 samples with the 4B verifier, plain and clarified (GPU).

plain       the verifier's input as trained: instruction, cursor, buffer, candidate
clarified   the same, with the recorded question and reply appended to the
            instruction (ask_pick_lib.clarified_case)

    python scripts/ask_pick/score_turn2.py --split val --samples runs/ask_pick_2026-10-18/turn2_val.jsonl \\
        --out runs/ask_pick_2026-10-18/turn2_scores_val.jsonl
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
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "reranker", HERE):
    sys.path.insert(0, str(p))
import ask_pick_lib as L                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--init-adapter", default=str(REPO / "runs" / "qwen3.5-4b-grpo-gated" / "best"))
    ap.add_argument("--verifier", default=str(REPO / "runs" / "reranker_2026-10-16" / "verifier-4b" / "best"))
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    from baseline import load_registry
    from transformers import AutoTokenizer
    from peft import PeftModel
    from verifier_lib import answer_token_ids, check_scoring, load_policy_merged, score_examples

    reg = load_registry(args.model)
    kw = reg.get("chat_template_kwargs") or {}
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    yes_id, no_id = answer_token_ids(tok, kw)
    t0 = time.time()
    model = PeftModel.from_pretrained(load_policy_merged(reg, args.init_adapter), args.verifier)
    cases = {c["id"]: c for c in L.load_jsonl(L.CASES[args.split])}
    dial = {r["id"]: r for r in L.load_jsonl(L.DIALOGUES[args.split])}
    rows = [c for r in L.load_jsonl(args.samples) for c in L.candidates(r)]
    clar = {i: L.clarified_case(cases[i], dial[i]["question"], dial[i]["reply"]) for i in {r["id"] for r in rows}}
    probe = sorted(rows, key=lambda r: len(cases[r["id"]]["buffer_before"]))
    print("scoring check: max diff %.4f" % check_scoring(model, tok, probe[:3] + probe[-3:], cases, kw, yes_id, no_id),
          flush=True)
    plain, t1 = score_examples(model, tok, rows, cases, kw, yes_id, no_id, batch_size=args.batch_size)
    clarified, t2 = score_examples(model, tok, rows, clar, kw, yes_id, no_id, batch_size=args.batch_size)
    L.write_jsonl(args.out, [{"id": r["id"], "cand_mode": r["cand_mode"], "cand_command": r["cand_command"],
                              "plain": a, "clarified": b} for r, a, b in zip(rows, plain, clarified)])
    print("%d candidates scored twice in %.0f s (truncated %d / %d) -> %s" % (
        len(rows), time.time() - t0, t1, t2, args.out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
