#!/usr/bin/env python3
"""Sample G answers after the recorded question and reply (GPU).

For every row of a split with a recorded dialogue, render the turn-2 prompt
exactly as the clarify harness does (`dialogue.Decoder.render` over
`protocol.turn2_messages`), sample G answers at T = 1 (top-p 1, top-k 0), and
execute each through the harness for its label (`probe_difficulty.observe`):
the same row format as the difficulty probe. Resumable by id.

    python scripts/ask_pick/sample_turn2.py --split val --out runs/ask_pick_2026-10-18/turn2_val.jsonl
    python scripts/ask_pick/sample_turn2.py --split test --parity 12 --out ...   # greedy parity first

`--parity N`: decode N rows greedily, one at a time, through this path and
compare with the recorded turn-2 completions; prints the agreement and stops.
"""
import argparse
import json
import os
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
import ask_pick_lib as L                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", required=True, choices=("val", "test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--adapter", default=str(REPO / "runs" / "qwen3.5-4b-grpo-gated" / "best"))
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--prompts-per-call", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--parity", type=int, default=0)
    args = ap.parse_args()

    import torch
    from transformers import set_seed
    from dialogue import Decoder, load_student
    from protocol import turn2_messages
    from probe_difficulty import observe, trim_completion
    from train_sft import inference_mode

    cases = {c["id"]: c for c in L.load_jsonl(L.CASES[args.split])}
    dial = {r["id"]: r for r in L.load_jsonl(L.DIALOGUES[args.split])}
    ids = [i for i in cases if i in dial]
    if args.limit:
        ids = ids[:args.limit]
    reg, tok, model, base = load_student(args.model, args.adapter, "auto")
    dec = Decoder(model, tok, reg)
    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id

    def prompt(i):
        return dec.render(turn2_messages(cases[i], dial[i]["question"], dial[i]["reply"]))

    if args.parity:
        same = 0
        for i in ids[:args.parity]:
            res = dec.complete(prompt(i))
            rec = dial[i]["real"]["completion"]
            same += res["completion"] == rec
            print("  %-16s %s  %r | recorded %r" % (i, "same" if res["completion"] == rec else "DIFF",
                                                   res["completion"][:50], rec[:50]), flush=True)
        print("parity: %d of %d greedy turn-2 completions identical to the recorded ones" % (same, args.parity))
        return 0

    out = pathlib.Path(args.out)
    done = {json.loads(l)["id"] for l in out.open()} if out.exists() else set()
    todo = [i for i in ids if i not in done]
    print("turn 2 on %s: %d rows, %d done, %d to do; G=%d, T=%.2f" % (
        args.split, len(ids), len(done), len(todo), args.num_generations, args.temperature), flush=True)
    set_seed(args.seed)
    tok.padding_side = "left"
    G = args.num_generations
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool, out.open("a", encoding="utf-8") as fh:
        for s in range(0, len(todo), args.prompts_per_call):
            batch = todo[s:s + args.prompts_per_call]
            enc = tok([prompt(i) for i in batch for _ in range(G)], return_tensors="pt", padding=True)
            enc = {k: v.to(dec.device) for k, v in enc.items()}
            with inference_mode(model), torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=192, do_sample=True, temperature=args.temperature,
                                     top_p=1.0, top_k=0, eos_token_id=eos_id, pad_token_id=pad_id)
            gen = gen[:, enc["input_ids"].shape[1]:].tolist()
            for j, i in enumerate(batch):
                rows = [trim_completion(r, eos_id, pad_id) for r in gen[j * G:(j + 1) * G]]
                row = observe(cases[i], rows, tok, eos_id, args.timeout, pool)
                row.update({"split": args.split, "greedy_turn2_match": bool(dial[i]["real"]["match"]),
                            "adapter": args.adapter, "temperature": args.temperature})
                fh.write(json.dumps(row) + "\n")
            fh.flush()
            n = s + len(batch)
            if n % 40 < args.prompts_per_call or n == len(todo):
                print("  %d/%d  %.0f s" % (n, len(todo), time.time() - t0), flush=True)
    print("done in %.0f s -> %s" % (time.time() - t0, out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
