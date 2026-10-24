#!/usr/bin/env python3
"""Stage 1 sampling: k more questions per ask row, at T = 1, under the offer prompt.

The rejection-sampling half of stage 1 (README.md). For every train-split row
noted *unstated* or *ambiguous*, the served adapter writes k questions with
"ask:" prefilled -- the same prompt dialogue.py's forced question uses -- but
sampled at T = 1 (top-p 1, top-k off: the distribution GRPO samples from)
instead of greedy. The greedy question already exists in the diag run
(`--greedy`); repeats of it, or of each other, are dropped, since asking the
simulator the same question twice buys nothing.

Writes, per sample index i, `questions_q<i>.jsonl` (rows shaped like a
dialogue.py turn-1 record: {id, label, question: {text, raw, output_tokens}})
and `cases_q<i>.jsonl` (the rows that have a question at that index). That is
exactly what scripts/clarify/simulate.py and `dialogue.py --phase turn2`
read, so the reply and the answer come from the audited pipeline unchanged.

Resumable: ids already in `questions_all.jsonl` are skipped.
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                                    # noqa: E402
from protocol import ASK_PREFIX, QUESTION_MAX_TOKENS, cut_question, turn1_messages  # noqa: E402
from data import unique_questions                                  # noqa: E402


def row_seed(seed, cid):
    """A per-row seed: the same row draws the same questions on a rerun."""
    return seed + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16)


def sample(dec, case, k, temperature, seed):
    """k sampled continuations of the "ask:" prefill, each cut at its newline."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    tok = dec.tok
    enc = tok(dec.render(turn1_messages(case), prefill=ASK_PREFIX), return_tensors="pt").to(dec.device)
    n = int(enc["input_ids"].shape[1])

    class NewlinePerRow(StoppingCriteria):
        """Each sequence stops at its own first newline (dialogue.py's
        criterion looks at row 0 only, which is enough at batch size 1)."""
        def __call__(self, input_ids, scores, **kw):
            if input_ids.shape[1] <= n:
                return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            last = input_ids[:, -1].tolist()
            return torch.tensor(["\n" in tok.decode([t]) for t in last],
                                dtype=torch.bool, device=input_ids.device)

    torch.manual_seed(row_seed(seed, case["id"]))
    with torch.no_grad():
        out = dec.model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0,
                                 top_k=0, num_return_sequences=k,
                                 max_new_tokens=QUESTION_MAX_TOKENS, eos_token_id=dec.stops,
                                 pad_token_id=dec.pad,
                                 stopping_criteria=StoppingCriteriaList([NewlinePerRow()]))
    raws = []
    for row in out[:, n:].tolist():
        cut = next((i for i, t in enumerate(row) if t in dec.stops), len(row))
        raws.append(tok.decode(row[:cut], skip_special_tokens=True))
    return raws


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--adapter", default=str(REPO / "runs" / "qwen3.5-4b-grpo-gated" / "best"))
    ap.add_argument("--cases", required=True, help="the train-split ask rows")
    ap.add_argument("--greedy", required=True,
                    help="the diag run's turn1_<label>.jsonl: each row's greedy question")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261007)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    greedy = {r["id"]: r["question"]["text"] for r in load_jsonl(args.greedy)}
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_path = out / "questions_all.jsonl"
    done = {r["id"] for r in load_jsonl(str(all_path))} if all_path.exists() else set()
    todo = [c for c in cases if c["id"] not in done]
    print("sample: %d rows, %d done, %d to do; k=%d at T=%g" % (
        len(cases), len(done), len(todo), args.k, args.temperature), flush=True)

    if todo:
        sys.path.insert(0, str(REPO / "scripts" / "clarify"))
        from dialogue import Decoder, load_student
        reg, tok, model, base = load_student(args.model, args.adapter)
        dec = Decoder(model, tok, reg)
        t0 = time.time()
        with all_path.open("a", encoding="utf-8") as fh:
            for i, case in enumerate(todo, 1):
                raws = sample(dec, case, args.k, args.temperature, args.seed)
                texts = unique_questions([cut_question(r) for r in raws],
                                         seen=[greedy.get(case["id"], "")])
                fh.write(json.dumps({"id": case["id"], "greedy": greedy.get(case["id"]),
                                     "raws": raws, "questions": texts},
                                    ensure_ascii=False) + "\n")
                fh.flush()
                if i % 25 == 0 or i == len(todo):
                    print("  %d/%d  %.1f s/row" % (i, len(todo), (time.time() - t0) / i), flush=True)

    # One turn-1-shaped file per sample index, for simulate.py and dialogue.py.
    by_id = {c["id"]: c for c in cases}
    rows = load_jsonl(str(all_path))
    for i in range(1, args.k + 1):
        label = "q%d" % i
        have = [r for r in rows if r["id"] in by_id and len(r["questions"]) >= i]
        with (out / ("questions_%s.jsonl" % label)).open("w", encoding="utf-8") as fh:
            for r in have:
                q = r["questions"][i - 1]
                fh.write(json.dumps({"id": r["id"], "label": label, "question": {
                    "text": q, "raw": q, "output_tokens": None}}, ensure_ascii=False) + "\n")
        with (out / ("cases_%s.jsonl" % label)).open("w", encoding="utf-8") as fh:
            for r in have:
                fh.write(json.dumps(by_id[r["id"]], ensure_ascii=False) + "\n")
        print("  %s: %d questions" % (label, len(have)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
