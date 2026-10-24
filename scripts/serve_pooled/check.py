#!/usr/bin/env python3
"""Parity checks for the pooled pick's endpoints, before any timing (GPU via the servers).

1. The verifier through vLLM against its recorded transformers scores, on test
   candidates (runs/reranker_2026-10-16/verifier-4b/test_scores.jsonl).
2. Greedy §6 answers of both endpoints against the recorded evaluated ones
   (the 4B: runs/qwen3.5-4b-grpo-gated/completions_val.jsonl; the 9B:
   runs/qwen3.5-9b-grpo-gated/completions_val.jsonl), identical-completion share.

    python scripts/serve_pooled/check.py --n 40
"""
import argparse
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "reranker", HERE):
    sys.path.insert(0, str(p))


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=str(REPO / "runs" / "serve_pooled" / "check.json"))
    args = ap.parse_args()
    import endpoint as E
    from prompt import messages_for_case
    import pooled as P
    e4 = E.load("qwen3.5-4b", "http://127.0.0.1:8011", "qwen3.5-4b-served")
    e9 = E.load("qwen3.5-9b", "http://127.0.0.1:8012", "served9b")
    pp = P.Pooled(e4, e9)
    # 1. verifier scores
    test = {c["id"]: c for c in load_jsonl(REPO / "data" / "splits" / "test.jsonl")}
    rec = load_jsonl(REPO / "runs" / "reranker_2026-10-16" / "verifier-4b" / "test_scores.jsonl")[:args.n]
    cands = [{"mode": r["cand_mode"], "command": r["cand_command"]} for r in rec]
    diffs = []
    for r, c in zip(rec, cands):
        s, _ = pp.scores(test[r["id"]], [c])
        diffs.append(abs(s[0] - r["score"]))
    print("verifier via vLLM vs recorded: %d candidates, |diff| mean %.3f, max %.3f" % (
        len(diffs), statistics.fmean(diffs), max(diffs)), flush=True)
    # 2. greedy parity
    val = load_jsonl(REPO / "data" / "splits" / "val.jsonl")[:args.n]
    out = {"verifier_absdiff_mean": statistics.fmean(diffs), "verifier_absdiff_max": max(diffs)}
    for name, ep, path in (("4b", e4, "runs/qwen3.5-4b-grpo-gated/completions_val.jsonl"),
                           ("9b", e9, "runs/qwen3.5-9b-grpo-gated/completions_val.jsonl")):
        recd = {r["id"]: r["completion"] for r in load_jsonl(REPO / path)}
        same = 0
        for c in val:
            text = ep.complete(ep.render(messages_for_case(c)), max_tokens=192)["texts"][0].strip()
            same += text == recd.get(c["id"], "").strip()
        out["greedy_identical_" + name] = same / len(val)
        print("%s greedy via vLLM identical to the recorded completion on %d of %d val rows" % (name, same, len(val)), flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
