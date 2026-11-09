#!/usr/bin/env python3
"""OPD's teacher pass: the trained 27B scores the 4B's samples, teacher-forced (runs/opd27b_2026-11-02/preregistration.md).

The 27B (`runs/qwen3.5-27b-sft-distill/best` on its NF4 base, thinking off) reads each train prompt
plus the 4B's sampled ids in right-padded batches of 8. Right padding is exact for a causal model:
a position never sees what follows it. The batched-against-single guard is
scripts/teacher_probs/record.py's: above 0.05 in any log-probability, every row runs alone.

Recorded at each sampled position, at T = 1 over the full vocabulary:
- the sampled token's log-probability (`tlp1`);
- the top 20 ids and their log-probabilities (`topk_ids`, `topk_lp1`);
- the log of the mass outside those 20 (`lrest1`).
That is exactly what the loss's 21 buckets need. `prompt_sha` ties each record to its prompt ids.

    python scripts/opd27b/teacher.py --seed 42 --round 1
"""
import argparse
import importlib.util
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                   # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


O = load("opd27b_opd", HERE / "opd.py")
REC = load("teacher_probs_record", REPO / "scripts" / "teacher_probs" / "record.py")   # GUARD_TOL


def position_stats(logits, sampled, topk=O.TOPK):
    """One row's record from its sampled positions' logits [n, V] (fp32) and the sampled ids [n]."""
    import torch
    lse = torch.logsumexp(logits, -1)
    top_v, top_i = logits.topk(topk, -1)
    rest = torch.logsumexp(logits.scatter(1, top_i, float("-inf")), -1) - lse
    tlp = logits.gather(1, sampled[:, None])[:, 0] - lse
    r4 = lambda t: [round(float(x), 4) for x in t]
    return {"tlp1": r4(tlp), "topk_ids": [[int(x) for x in row] for row in top_i],
            "topk_lp1": [r4(row) for row in top_v - lse[:, None]], "lrest1": r4(rest)}


def forward_rows(backbone, head, batch, device):
    """One right-padded forward pass; per row, the stats at the positions that predict its samples."""
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
        out.append(position_stats(logits, torch.tensor(it["sample"], device=device)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, choices=sorted(O.STARTS), required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--run", default=str(O.RUN))
    ap.add_argument("--root", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args(argv)

    import torch
    from peft import PeftModel
    from baseline import load_base, load_registry, resolve_base_quant

    rd = O.round_dir(args.seed, args.round, args.root)
    out_path = rd / "teacher.jsonl"
    samples = load_jsonl(str(rd / "samples.jsonl"))
    prompts = O.load_prompts(args.run)
    done = {r["id"] for r in load_jsonl(str(out_path))} if out_path.exists() else set()
    items = [{"id": s["id"], "ids": prompts[s["id"]] + s["ids"], "P": len(prompts[s["id"]]), "sample": s["ids"],
              "prompt_sha": O.ids_sha(prompts[s["id"]])} for s in samples if s["id"] not in done]
    print("%d samples to score (%d on disk already)" % (len(items), len(done)), flush=True)
    if not items:
        return 0
    t0 = time.time()
    reg = load_registry(O.TEACHER)
    if (reg.get("chat_template_kwargs") or {}).get("enable_thinking"):
        sys.exit("the teacher's registry entry must have thinking off")
    load_in_4bit = resolve_base_quant(str(O.TEACHER_ADAPTER))
    model = PeftModel.from_pretrained(load_base(reg, load_in_4bit), str(O.TEACHER_ADAPTER))
    model.eval()
    inner = model.get_base_model()
    backbone, head, device = inner.model, inner.lm_head, next(inner.parameters()).device
    print("loaded %s @ %s (%s) + %s in %.0f s; %.2f GiB allocated" % (
        reg["hf_id"], reg["revision"][:12], "nf4" if load_in_4bit else "bf16", O.TEACHER_ADAPTER.name,
        time.time() - t0, torch.cuda.memory_allocated() / 2 ** 30), flush=True)

    items.sort(key=lambda it: (len(it["ids"]), it["id"]))      # little padding per batch
    batches = [items[i:i + args.batch_size] for i in range(0, len(items), args.batch_size)]
    meta = {"model": reg["hf_id"], "revision": reg["revision"], "adapter": str(O.TEACHER_ADAPTER),
            "base": "nf4" if load_in_4bit else "bf16", "topk": O.TOPK, "batch_size": args.batch_size,
            "samples": len(items) + len(done)}
    rd.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        first = batches[0]
        batched = forward_rows(backbone, head, first, device)
        worst, singles = 0.0, []
        for it, rb in zip(first, batched):
            singles.append(forward_rows(backbone, head, [it], device)[0])
            worst = max([worst] + [abs(a - b) for a, b in zip(rb["tlp1"], singles[-1]["tlp1"])])
        meta["guard_max_abs_dlogp"] = round(worst, 5)
        print("guard: batched vs one row at a time, max |Δ log p| = %.5f" % worst, flush=True)
        if worst > REC.GUARD_TOL:
            print("guard: over %.2f, so every row is scored alone" % REC.GUARD_TOL, flush=True)
            meta["guard_fallback_batch_size"] = 1
            batched = singles
            batches = [first] + [[it] for b in batches[1:] for it in b]
        t1 = time.time()
        with out_path.open("a", encoding="utf-8") as fh:
            for k, batch in enumerate(batches):
                stats = batched if k == 0 else forward_rows(backbone, head, batch, device)
                for it, st in zip(batch, stats):
                    fh.write(json.dumps({"id": it["id"], "ids": it["sample"], "prompt_sha": it["prompt_sha"],
                                         **st}) + "\n")
                fh.flush()
                if (k + 1) % 100 == 0 or k + 1 == len(batches):
                    rate = (time.time() - t1) / (k + 1)
                    print("  %d/%d batches  %.2f s per batch  ~%.0f min left  %.2f GiB peak" % (
                        k + 1, len(batches), rate, rate * (len(batches) - k - 1) / 60,
                        torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    meta.update(seconds=round(time.time() - t0), peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2))
    (rd / "teacher_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("round %d: scored in %.1f min -> %s" % (args.round, (time.time() - t0) / 60, out_path), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
