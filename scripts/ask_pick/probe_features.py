#!/usr/bin/env python3
"""Hidden states for the ask probes (GPU): the served adapter's last-token
hidden state at layers 8 / 16 / 24 / 32 on the turn-1 offer prompt, for diag,
val and test -- the features scripts/clarify/probe_ask.py measured, extracted
again in one pass so all three sets come from the same code and kernels.

    python scripts/ask_pick/probe_features.py --out runs/ask_pick_2026-10-18/probe_features.pt
"""
import argparse
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
import ask_pick_lib as L                                     # noqa: E402

LAYERS = (8, 16, 24, 32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--adapter", default=str(REPO / "runs" / "qwen3.5-4b-grpo-gated" / "best"))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    import torch
    from dialogue import Decoder, load_student
    from protocol import turn1_messages
    reg, tok, model, base = load_student(args.model, args.adapter, "auto")
    dec = Decoder(model, tok, reg)
    feats = {}
    t0 = time.time()
    with torch.no_grad():
        for split in ("diag", "val", "test"):
            cases = L.load_jsonl(L.CASES[split])[:args.limit] if args.limit else L.load_jsonl(L.CASES[split])
            for c in cases:
                enc = tok(dec.render(turn1_messages(c)), return_tensors="pt").to(dec.device)
                hs = model(**enc, output_hidden_states=True).hidden_states
                feats["%s|%s" % (split, c["id"])] = {str(Lr): hs[Lr][0, -1].float().cpu() for Lr in LAYERS}
            print("  %s: %d rows, %.0f s" % (split, len(cases), time.time() - t0), flush=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, args.out)
    print("saved %d hidden states x %d layers -> %s" % (len(feats), len(LAYERS), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
