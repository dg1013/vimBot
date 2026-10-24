#!/usr/bin/env python3
"""The simulated user: replies to every question the student asked.

Reads the turn-1 files of one or more checkpoints and, for each (label, case),
writes the reply of the person who wrote the instruction. The simulator sees
the instruction, the buffer before, the buffer it wants after and the exact
changes, and **never the gold command**: a user knows the result they want,
not the keystrokes. Its rules (protocol.SIM_SYSTEM) are to answer only what
was asked, in words about the text, not to agree with a proposal the result
does not show, and to say "it doesn't matter" when the result does not settle
the question. It writes a one-sentence "Check:" on what the result does, then
its "Reply:"; only the reply reaches the student.

Model: the registry's `qwen3.5-9b`, the largest local model, held in NF4 (the
QLoRA storage recipe, ~6 GB) so it stays well inside the card next to the
desktop compositor; greedy. The student never thinks (CLAUDE.md §4); the
simulator is part of the measuring instrument, and how it reasons was
chosen by a hand audit of 50 replies (DECISIONS.md 2026-10-06):

* without thinking, 6-7 of 50 replies contradicted the result, whatever the
  prompt -- "yes" to a proposal the result does not make, a matching rule
  that the removed lines refute;
* unbounded thinking ran 7 of the first 8 replies past 2,048 tokens without
  finishing, at 13 s a reply;
* so thinking is **budget-forced** (Muennighoff et al. 2025, "s1: Simple
  test-time scaling"): at most `--think-budget` tokens of reasoning, stopped
  at the model's own ``</think>`` if it comes sooner; a block still open at
  the budget is closed by appending ``</think>``, and the reply is generated
  after it. `--think-budget 0` is the non-thinking simulator.

Batched with left padding like the difficulty probe. The replies file is the
artifact of record, so batch-composition rounding does not touch the
report's reproducibility: the report reads the file, it does not re-simulate.

Every reply is screened (protocol.leak_flags, protocol.quotes_result) and the
flags are stored with it; the report counts them and re-reads the headline
with flagged replies excluded.

Resumable and pausable like dialogue.py.
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl                                       # noqa: E402
from baseline import (base_quant_name, load_base, load_registry,     # noqa: E402
                      stop_token_ids)
from probe_difficulty import PAUSED_RC, PauseRequest                  # noqa: E402
from protocol import (clean_reply, leak_flags, quotes_result,         # noqa: E402
                      sim_messages, split_check)

REPLY_MAX_TOKENS = 160           # a one-sentence check, then a short reply
THINK_CLOSE = "</think>"


def done_keys(path):
    keys = set()
    p = pathlib.Path(path)
    if p.exists():
        for row in load_jsonl(str(p)):
            keys.add((row["label"], row["id"]))
    return keys


class Generator:
    """Greedy batched completion of rendered prompts, cut at the first stop."""

    def __init__(self, model, tok, stops):
        import torch
        self.torch, self.model, self.tok, self.stops = torch, model, tok, stops
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.device = next(model.parameters()).device

    def __call__(self, texts, max_new, stops=None):
        """[(token ids up to the stop, the stop id or None)] per text."""
        stops = stops or self.stops
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                      eos_token_id=stops, pad_token_id=self.pad)
        res = []
        for ids in out[:, enc["input_ids"].shape[1]:].tolist():
            cut = next((i for i, t in enumerate(ids) if t in stops), None)
            res.append((ids, None) if cut is None else (ids[:cut], ids[cut]))
        return res


def reply_batch(gen, tok, prompts, budget):
    """One reply per prompt: (reply text record) dicts, in order.

    budget 0: a single non-thinking pass. budget > 0: reasoning up to the
    budget (or the model's own close), then the reply after a closed block.
    """
    if budget <= 0:
        return [{"raw": tok.decode(ids, skip_special_tokens=True), "think_tokens": 0,
                 "forced": False, "think": "", "reply_tokens": len(ids)}
                for ids, _ in gen(prompts, REPLY_MAX_TOKENS)]
    close_id = tok.encode(THINK_CLOSE, add_special_tokens=False)
    assert len(close_id) == 1, "%s must be one token" % THINK_CLOSE
    phase1 = gen(prompts, budget, stops=gen.stops + close_id)
    thinks, conts = [], []
    for p, (ids, stop) in zip(prompts, phase1):
        think = tok.decode(ids, skip_special_tokens=True)
        thinks.append((think, len(ids), stop is None))
        conts.append(p + think.rstrip("\n") + "\n" + THINK_CLOSE + "\n\n")
    out = []
    for (think, n, forced), (ids, _) in zip(thinks, gen(conts, REPLY_MAX_TOKENS)):
        out.append({"raw": tok.decode(ids, skip_special_tokens=True), "think_tokens": n,
                    "forced": forced, "think": think, "reply_tokens": len(ids)})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-9b", help="registry name of the simulator")
    ap.add_argument("--base-dtype", choices=("bf16", "nf4"), default="nf4")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--turn1", action="append", required=True,
                    help="a turn-1 file (dialogue.py --phase turn1 --out); repeatable")
    ap.add_argument("--out", required=True, help="JSONL, appended; resumable")
    ap.add_argument("--think-budget", type=int, default=0,
                    help="tokens of reasoning before the reply, budget-forced; 0 = none")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N cases per label (smoke)")
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()

    cases = {c["id"]: c for c in load_jsonl(args.cases)}
    order = list(cases)
    if args.limit:
        order = order[:args.limit]
    keep = set(order)
    jobs = []
    for path in args.turn1:
        for row in load_jsonl(path):
            if row["id"] in keep:
                jobs.append((row["label"], row["id"], row["question"]["text"]))
    done = done_keys(args.out)
    todo = [j for j in jobs if (j[0], j[1]) not in done]
    print("simulate: %d questions, %d already in %s, %d to do; think budget %d" % (
        len(jobs), len(jobs) - len(todo), args.out, len(todo), args.think_budget), flush=True)
    if not todo:
        return 0

    from transformers import AutoTokenizer
    reg = load_registry(args.model)
    nf4 = args.base_dtype == "nf4"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    tok.padding_side = "left"
    model = load_base(reg, nf4)
    model.eval()
    print("simulator %s @ %s (%s) loaded in %.0f s" % (
        reg["hf_id"], reg["revision"][:12], base_quant_name(nf4), time.time() - t0), flush=True)
    kwargs = dict(reg.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = args.think_budget > 0
    gen = Generator(model, tok, stop_token_ids(tok, reg.get("stop")))

    texts = {(lab, cid): tok.apply_chat_template(sim_messages(cases[cid], q), tokenize=False,
                                                 add_generation_prompt=True, **kwargs)
             for lab, cid, q in todo}
    # Similar lengths together: less padding, and padding is all a batch costs.
    todo.sort(key=lambda j: len(texts[(j[0], j[1])]))

    pause = PauseRequest(args.stop_file)
    pause.install()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"model": reg["name"], "hf_id": reg["hf_id"], "revision": reg["revision"],
            "base": base_quant_name(nf4), "think_budget": args.think_budget}
    t_start = time.time()
    n = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), args.batch):
            batch = todo[start:start + args.batch]
            t1 = time.perf_counter()
            results = reply_batch(gen, tok, [texts[(lab, cid)] for lab, cid, _ in batch],
                                  args.think_budget)
            latency = (time.perf_counter() - t1) / len(batch)
            for (lab, cid, q), res in zip(batch, results):
                case = cases[cid]
                reply = clean_reply(res["raw"])
                fh.write(json.dumps({
                    "id": cid, "label": lab, "question": q, "reply": reply,
                    "check": split_check(res["raw"])[0], "raw": res["raw"],
                    "think": res["think"], "think_tokens": res["think_tokens"],
                    "forced": res["forced"], "reply_tokens": res["reply_tokens"],
                    "latency_s": round(latency, 3),
                    "leak_flags": leak_flags(reply, case),
                    "quotes_result": quotes_result(reply, case),
                    "simulator": meta,
                }, ensure_ascii=False) + "\n")
            fh.flush()
            n += len(batch)
            rate = (time.time() - t_start) / n
            if (start // args.batch) % 10 == 0 or n == len(todo):
                print("  %d/%d  %.2f s/reply  eta %.0f min" % (
                    n, len(todo), rate, rate * (len(todo) - n) / 60), flush=True)
            if pause.check():
                marker = pause.marker()
                if marker:
                    marker.write_text("%s after %d replies\n" % (pause.requested, n))
                print("paused (%s) after %d replies" % (pause.requested, n), flush=True)
                return PAUSED_RC
    print("done: %d replies in %.0f s -> %s" % (n, time.time() - t_start, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
