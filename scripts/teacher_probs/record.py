#!/usr/bin/env python3
"""Record the trained 27B's next-token distributions on its own training targets, by the rules in
runs/teacher_probs_2026-10-31/preregistration.md. GPU: one teacher-forced forward pass per batch;
one JSON line per train row, appended, so a restart skips the rows already on disk.

Tokens come from TRL's own tokenizing function, called the way SFTTrainer calls it, so every
position recorded here is a position the 27B was trained on. On the rows approach 1 re-targeted,
the original gold is a verified alternative, and the gold token's probability is kept at the first
position where its completion departs from the target's.

    python scripts/teacher_probs/record.py
"""
import argparse
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                   # noqa: E402
from prompt import messages_for_case, render_target              # noqa: E402

MODEL = "qwen3.5-27b"
ADAPTER = REPO / "runs" / "qwen3.5-27b-sft-distill" / "best"
TRAIN = REPO / "runs" / "distill_2026-10-31" / "train_distilled.jsonl"
GOLDS = REPO / "data" / "splits" / "train.jsonl"
RUN = REPO / "runs" / "teacher_probs_2026-10-31"
TOPK = 20
GUARD_TOL = 0.05     # largest |Δ log p| allowed between the batched and the one-row pass


def encode(tok, case, mode, command, template_kwargs):
    """(prompt ids, prompt + completion ids), exactly as TRL 1.12's SFT tokenize_fn builds them
    for a conversational prompt-completion row."""
    from trl.data_utils import _tokenize
    kw = {"chat_template": None, "tools": None, **template_kwargs}
    prompt = messages_for_case(case)
    completion = [{"role": "assistant", "content": render_target(mode, command)}]
    prompt_ids = _tokenize(tok, prompt, add_generation_prompt=True, **kw)["input_ids"]
    full_ids = _tokenize(tok, prompt + completion, return_assistant_tokens_mask=False, **kw)["input_ids"]
    return list(prompt_ids), list(full_ids)


def divergence(a, b):
    """The first index where two token lists differ, or None when one is a prefix of the other."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None


def answer_len(completion, im_end):
    """Completion tokens before the first <|im_end|>: the answer line itself."""
    return completion.index(im_end) if im_end in completion else len(completion)


def q_topk(top_logits, top_ids, token, temperature):
    """A token's probability in the top-k renormalized at `temperature` (0 when outside the k)."""
    import torch
    hit = (top_ids == token).nonzero()
    if hit.numel() == 0:
        return 0.0
    return float(torch.softmax(top_logits.float() / temperature, -1)[hit[0, 0]])


def row_stats(logits, target, topk=TOPK, gold=None):
    """Statistics for one row from its completion positions' logits [n, V] (fp32) and the target
    ids [n]. gold = (position, token) at the divergence point, or None."""
    import torch
    lp1 = torch.log_softmax(logits, -1)
    lp2 = torch.log_softmax(logits / 2.0, -1)
    tgt = target[:, None]
    tlogit = logits.gather(1, tgt)
    top_v, top_i = logits.topk(topk, -1)
    r4 = lambda t: [round(float(x), 4) for x in t]
    out = {"tlp1": r4(lp1.gather(1, tgt)[:, 0]),
           "tlp2": r4(lp2.gather(1, tgt)[:, 0]),
           "ent1": r4(-(lp1.exp() * lp1).sum(-1)),
           "trank": [int(x) for x in (logits > tlogit).sum(-1) + 1],
           "lse1": r4(torch.logsumexp(logits, -1)),
           "topk_ids": [[int(x) for x in row] for row in top_i],
           "topk_logits": [[round(float(x), 3) for x in row] for row in top_v]}
    if gold is not None:
        d, g = gold
        z = logits[d]
        out["gold"] = {"d": d, "token": int(g), "lp1": round(float(lp1[d, g]), 4),
                       "rank": int((z > z[g]).sum()) + 1,
                       "q2": round(q_topk(top_v[d], top_i[d], g, 2.0), 5),
                       "target_q2": round(q_topk(top_v[d], top_i[d], int(target[d]), 2.0), 5)}
    return out


def prepare(tok, cases, golds, template_kwargs, im_end):
    """Every row's ids, completion, answer length and (for re-targeted rows) the divergence point."""
    items, mismatched = [], []
    for c in cases:
        p, full = encode(tok, c, c["mode"], c["gold_command"], template_kwargs)
        if full[:len(p)] != p:
            mismatched.append(c["id"])
        comp = full[len(p):]
        item = {"id": c["id"], "ids": full, "P": len(p), "completion": comp,
                "n_answer": answer_len(comp, im_end), "gold": None}
        g = golds.get(c["id"])
        if g is not None and (g["mode"], g["gold_command"]) != (c["mode"], c["gold_command"]):
            gp, gfull = encode(tok, c, g["mode"], g["gold_command"], template_kwargs)
            d = divergence(comp, gfull[len(gp):]) if gp == p else None
            item["gold"] = {"d": d, "token": gfull[len(gp) + d] if d is not None else None,
                            "mode_switch": g["mode"] != c["mode"]}
        items.append(item)
    return items, mismatched


def forward_rows(backbone, head, batch, device, topk):
    """One right-padded forward pass; per row, the logits at the positions that predict its
    completion tokens, reduced to row_stats."""
    import torch
    width = max(len(it["ids"]) for it in batch)
    ids = torch.zeros((len(batch), width), dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for b, it in enumerate(batch):
        ids[b, :len(it["ids"])] = torch.tensor(it["ids"])
        mask[b, :len(it["ids"])] = 1
    hidden = backbone(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False).last_hidden_state
    out = []
    for b, it in enumerate(batch):
        P, T = it["P"], len(it["ids"])
        logits = head(hidden[b, P - 1:T - 1]).float()
        target = torch.tensor(it["completion"], device=device)
        g = it["gold"]
        gold = (g["d"], g["token"]) if g and g["d"] is not None else None
        out.append(row_stats(logits, target, topk, gold))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", default=str(ADAPTER))
    ap.add_argument("--train", default=str(TRAIN))
    ap.add_argument("--golds", default=str(GOLDS))
    ap.add_argument("--out", default=str(RUN / "probs.jsonl"))
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N rows still to do (smoke)")
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from baseline import load_base, load_registry, resolve_base_quant

    t0 = time.time()
    reg = load_registry(args.model)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.open(encoding="utf-8") if line.strip()}
    cases = [c for c in load_jsonl(args.train) if c["id"] not in done]
    if args.limit:
        cases = cases[:args.limit]
    golds = {c["id"]: c for c in load_jsonl(args.golds)}
    items, mismatched = prepare(tok, cases, golds, reg.get("chat_template_kwargs") or {}, im_end)
    if mismatched:
        sys.exit("%d rows whose prompt is not a prefix of prompt + completion (first: %s): positions "
                 "would not be the trained ones" % (len(mismatched), mismatched[0]))
    retargeted = sum(1 for it in items if it["gold"])
    print("%d rows to record (%d on disk already), %d re-targeted; tokenized in %.0f s"
          % (len(items), len(done), retargeted, time.time() - t0), flush=True)
    if not items:
        return 0

    load_in_4bit = resolve_base_quant(args.adapter)
    model = PeftModel.from_pretrained(load_base(reg, load_in_4bit), args.adapter)
    model.eval()
    inner = model.get_base_model()
    backbone, head, device = inner.model, inner.lm_head, next(inner.parameters()).device
    print("loaded %s @ %s (%s) + %s in %.0f s; %.2f GiB allocated"
          % (reg["hf_id"], reg["revision"][:12], "nf4" if load_in_4bit else "bf16", args.adapter,
             time.time() - t0, torch.cuda.memory_allocated() / 2 ** 30), flush=True)

    items.sort(key=lambda it: (len(it["ids"]), it["id"]))     # little padding per batch
    batches = [items[i:i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    meta = {"model": reg["hf_id"], "revision": reg["revision"], "adapter": args.adapter,
            "base": "nf4" if load_in_4bit else "bf16", "topk": args.topk, "im_end": im_end,
            "rows": len(items) + len(done), "retargeted_this_pass": retargeted,
            "batch_size": args.batch_size, "torch": torch.__version__}
    with torch.inference_mode():
        # The guard: padding must not move any recorded number.
        first = batches[0]
        batched = forward_rows(backbone, head, first, device, args.topk)
        worst, singles = 0.0, []
        for it, rb in zip(first, batched):
            singles.append(forward_rows(backbone, head, [it], device, args.topk)[0])
            worst = max([worst] + [abs(a - b) for a, b in zip(rb["tlp1"], singles[-1]["tlp1"])])
        meta["guard_max_abs_dlogp"] = round(worst, 5)
        print("guard: batched vs one row at a time, max |Δ log p| = %.5f" % worst, flush=True)
        if worst > GUARD_TOL:
            # Padding moves the numbers: record one row per pass instead (slower, exact).
            print("guard: over %.2f, so every row is recorded alone" % GUARD_TOL, flush=True)
            meta["guard_fallback_batch_size"] = 1
            batched = singles
            batches = [first] + [[it] for b in batches[1:] for it in b]
        t1 = time.time()
        with out_path.open("a", encoding="utf-8") as fh:
            for k, batch in enumerate(batches):
                stats = batched if k == 0 else forward_rows(backbone, head, batch, device, args.topk)
                for it, st in zip(batch, stats):
                    rec = {"id": it["id"], "n": len(it["completion"]), "n_answer": it["n_answer"],
                           "ids": it["completion"], **st}
                    if it["gold"]:
                        rec.setdefault("gold", {}).update({"mode_switch": it["gold"]["mode_switch"]})
                        if it["gold"]["d"] is None:
                            rec["gold"]["d"] = None
                    fh.write(json.dumps(rec) + "\n")
                fh.flush()
                if (k + 1) % 50 == 0 or k + 1 == len(batches):
                    rate = (time.time() - t1) / (k + 1)
                    print("  %d/%d batches  %.2f s per batch  ~%.0f min left  %.2f GiB peak"
                          % (k + 1, len(batches), rate, rate * (len(batches) - k - 1) / 60,
                             torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    meta.update(seconds=round(time.time() - t0), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    (out_path.parent / "record_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("done in %.1f min -> %s" % ((time.time() - t0) / 60, out_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
