#!/usr/bin/env python3
"""Does the served model answer as the evaluated one did? Greedy §6 answers on val.

Every number for the sample rule was measured through transformers on the
adapter (scripts/clarify/dialogue.py). Real use answers through vLLM on the
merged weights (serve/merge_adapter.py, serve/serve.sh). This puts val
through the endpoint with the same rendered prompts and compares, row by
row, with the 2026-10-06 run's §6 answers: identical completions, and
exec-match through the same sandbox (evaluate.score, the settings timeout).

    python scripts/serve/parity.py --out runs/serve/parity_val.json
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl, score                     # noqa: E402
from prompt import messages_for_case, parse_output         # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8011")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "val.jsonl"))
    ap.add_argument("--ref", default=str(REPO / "runs" / "clarify_2026-10-06" / "val" / "verdicts_served.jsonl"))
    ap.add_argument("--settings", default=None)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import endpoint
    from settings import load_settings
    ep = endpoint.load(args.model, args.endpoint)
    cases = load_jsonl(args.cases)
    ref = {r["id"]: r for r in load_jsonl(args.ref)}

    def one(c):
        return ep.complete(ep.render(messages_for_case(c)), 192)["texts"][0]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        served = dict(zip((c["id"] for c in cases), pool.map(one, cases)))
    preds = {}
    for cid, text in served.items():
        p = parse_output(text)
        if p:
            preds[cid] = {"mode": p[0], "command": p[1]}
    res = score(cases, preds, timeout=load_settings(args.settings)["executor"]["timeout"])
    same = sum(served[c["id"]] == ref[c["id"]]["direct"]["completion"] for c in cases)
    hf_match = sum(ref[c["id"]]["direct"]["match"] for c in cases)
    both = sum(res["per_case"][c["id"]]["match"] and ref[c["id"]]["direct"]["match"] for c in cases)
    out = {"n": len(cases), "identical_completions": same,
           "served_exec_match": res["n_match"], "evaluated_exec_match": hf_match,
           "match_in_both": both,
           "only_served": res["n_match"] - both, "only_evaluated": hf_match - both,
           "differing": [{"id": c["id"], "served": served[c["id"]],
                          "evaluated": ref[c["id"]]["direct"]["completion"],
                          "served_match": res["per_case"][c["id"]]["match"],
                          "evaluated_match": ref[c["id"]]["direct"]["match"]}
                         for c in cases if served[c["id"]] != ref[c["id"]]["direct"]["completion"]]}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("parity on %d rows: identical completions %d (%.1f%%); exec-match served %d vs evaluated %d "
          "(both %d, only served %d, only evaluated %d) -> %s" % (
              out["n"], same, 100 * same / out["n"], res["n_match"], hf_match, both,
              out["only_served"], out["only_evaluated"], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
