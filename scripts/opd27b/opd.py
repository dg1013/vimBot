#!/usr/bin/env python3
"""On-policy distillation (OPD) of the trained 27B into the 4B: the shared pieces, and the stages
other than sampling (sample.py) and the teacher pass (teacher.py).

By runs/opd27b_2026-11-02/preregistration.md (Part 1 of runs/teacher_channels_2026-11-02/plan.md).
The 4B writes its own answers to the train prompts, the trained 27B scores every token of them, and
the 4B moves toward the teacher on its own text.

The loss, at each sampled position t, is the reverse KL from the student to the teacher over 21
buckets: the teacher's top 20 tokens at t, and everything else as one bucket.

    KL_t = sum_b q_s(b) * (log q_s(b) - log q_t(b))

- q_t is the teacher at T = 1, from the recorded full-vocabulary log-probabilities of its top 20
  and the log of the mass left outside them. It is renormalized over the buckets, because the
  recorded values are rounded.
- q_s is the student's full softmax at T = 1, gathered into the same buckets.
- The sum over the positions of 32 samples is divided by their number of positions, TRL's
  normalization. There is no NLL term and no gold.

Reverse KL, not KD's forward direction: forward KL asks the student to cover everything the teacher
finds plausible (it spreads probability); reverse KL asks it to put probability only where the
teacher does (it commits). The 4B's measured weakness is commitment.

    python scripts/opd27b/opd.py prompts                       # CPU: train prompt ids
    python scripts/opd27b/opd.py eval --seed 42 --round 0      # GPU: the start on the 250-row subset
    python scripts/opd27b/opd.py train --seed 42 --round 1     # GPU: one pass over round 1's samples
    python scripts/opd27b/opd.py final --seed 42               # GPU: choose a round, full val on it
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import random
import sys
import time
import zlib

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl, score, write_exec_log             # noqa: E402
from prompt import messages_for_case                              # noqa: E402

RUN = REPO / "runs" / "opd27b_2026-11-02"
STUDENT, TEACHER = "qwen3.5-4b", "qwen3.5-27b"
TEACHER_ADAPTER = REPO / "runs" / "qwen3.5-27b-sft-distill" / "best"
STARTS = {42: REPO / "runs" / "qwen3.5-4b-sft-distill", 7: REPO / "runs" / "qwen3.5-4b-sft-distill-s7"}
TRAIN = REPO / "data" / "splits" / "train.jsonl"
VAL = REPO / "data" / "splits" / "val.jsonl"
BASE_TEXT = RUN / "base-text"          # the 4B's text model for vLLM (make_text_9b.py --model qwen3.5-4b)
ROUNDS = 3
TOPK = 20
LR = 2e-5
WARMUP = 10
MICRO, ACCUM = 8, 4                    # 32 samples per optimizer step
CLIP = 1.0
MAX_NEW = 192
CHUNK = 256                            # sampled positions per checkpointed loss chunk
SUBSET = 250


# ------------------------------------------------------------ paths and seeds

def out_dir(seed):
    return REPO / "runs" / ("qwen3.5-4b-opd27b-s%d" % seed)


def round_dir(seed, r, root=None):
    """Absolute, since vLLM takes a LoRA module's path as given."""
    return (pathlib.Path(root).resolve() if root else out_dir(seed)) / ("round%d" % r)


def adapter_for(seed, r, root=None):
    """The adapter at the end of round r: the start for round 0."""
    return STARTS[seed] / "best" if r == 0 else round_dir(seed, r, root) / "adapter"


def round_order(seed, r, ids):
    out = sorted(ids)
    random.Random("opd|%d|%d" % (seed, r)).shuffle(out)
    return out


def sample_seed(seed, r, row_id):
    return (1000003 * seed + 7919 * r + zlib.crc32(row_id.encode())) % (2 ** 31)


def ids_sha(ids):
    return hashlib.sha1(json.dumps(list(ids)).encode()).hexdigest()[:16]


# ------------------------------------------------------------ prompts (CPU)

def encode_prompt(tok, case, template_kwargs):
    """The §6 prompt's ids exactly as TRL 1.12 tokenizes a conversational prompt for SFT
    (scripts/teacher_probs/record.py's encode, its prompt part)."""
    from trl.data_utils import _tokenize
    kw = {"chat_template": None, "tools": None, **template_kwargs}
    return list(_tokenize(tok, messages_for_case(case), add_generation_prompt=True, **kw)["input_ids"])


def cmd_prompts(args):
    from transformers import AutoTokenizer
    from baseline import load_registry
    regs = {n: load_registry(n) for n in (STUDENT, TEACHER)}
    toks = {n: AutoTokenizer.from_pretrained(r["hf_id"], revision=r["revision"]) for n, r in regs.items()}
    cases = load_jsonl(str(TRAIN))
    out, differ = [], []
    for c in cases:
        a = encode_prompt(toks[STUDENT], c, regs[STUDENT].get("chat_template_kwargs") or {})
        b = encode_prompt(toks[TEACHER], c, regs[TEACHER].get("chat_template_kwargs") or {})
        if a != b:
            differ.append(c["id"])
        out.append({"id": c["id"], "ids": a})
    if differ:
        sys.exit("%d prompts tokenize differently for the student and the teacher (first: %s)" % (len(differ), differ[0]))
    path = pathlib.Path(args.run) / "prompts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    lens = sorted(len(r["ids"]) for r in out)
    print("%d train prompts, identical for %s and %s: %d-%d tokens, median %d -> %s"
          % (len(out), STUDENT, TEACHER, lens[0], lens[-1], lens[len(lens) // 2], path))
    return 0


def load_prompts(run=RUN):
    return {r["id"]: r["ids"] for r in load_jsonl(str(pathlib.Path(run) / "prompts.jsonl"))}


# ------------------------------------------------------------ the loss

def bucket_kl(logits, t_ids, t_lp, t_lrest):
    """Per position, KL(q_s || q_t) over the teacher's top k and the rest. logits [N, V] fp32 (the
    student's); t_ids [N, k]; t_lp [N, k] (the teacher's full-vocabulary log-probabilities of those
    k); t_lrest [N] (the log of the teacher's mass outside them). Returns the KL per position [N]."""
    import torch
    lse = torch.logsumexp(logits, -1)
    ls_k = logits.gather(1, t_ids) - lse[:, None]
    ls_rest = torch.logsumexp(logits.scatter(1, t_ids, float("-inf")), -1) - lse
    lq_s = torch.cat([ls_k, ls_rest[:, None]], 1)
    lq_t = torch.cat([t_lp, t_lrest[:, None]], 1)
    lq_t = lq_t - torch.logsumexp(lq_t, -1, keepdim=True)
    return (lq_s.exp() * (lq_s - lq_t)).sum(-1)


def _chunk(h, w, sampled, t_ids, t_lp, t_lrest):
    """(KL sum, the student's NLL of its own sampled tokens, argmax == sampled) over a chunk."""
    import torch
    logits = (h.to(w.dtype) @ w.t()).float()
    kl = bucket_kl(logits, t_ids, t_lp, t_lrest)
    nll = -(logits.gather(1, sampled[:, None])[:, 0] - torch.logsumexp(logits, -1))
    top = (logits.argmax(-1) == sampled).float()
    return kl.sum(), nll.sum(), top.sum()


# ------------------------------------------------------------ the round's rows

def build_rows(samples, teacher, prompts):
    """[(id, prompt ids, sampled ids, top ids, top log-probs, rest log-prob)]; refuses any row whose
    teacher record is not of the same prompt and sampled ids."""
    t = {r["id"]: r for r in teacher}
    rows, bad = [], []
    for s in samples:
        r = t.get(s["id"])
        p = prompts[s["id"]]
        if (r is None or r["ids"] != s["ids"] or r["prompt_sha"] != ids_sha(p) or len(r["topk_ids"]) != len(s["ids"])
                or len(r["topk_lp1"]) != len(s["ids"]) or len(r["lrest1"]) != len(s["ids"])):
            bad.append(s["id"])
            continue
        rows.append((s["id"], p, s["ids"], r["topk_ids"], r["topk_lp1"], r["lrest1"]))
    if bad:
        raise SystemExit("%d samples have no matching teacher record (first: %s): the targets would be shifted"
                         % (len(bad), bad[0]))
    return rows


def micro_batch(rows, device):
    """Right-padded ids and mask, and the flat per-position targets in row order."""
    import torch
    width = max(len(p) + len(s) for _, p, s, *_ in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    pos_b, pos_t, sampled, t_ids, t_lp, t_lrest = [], [], [], [], [], []
    for b, (_, p, s, ti, tl, tr) in enumerate(rows):
        seq = p + s
        ids[b, :len(seq)] = torch.tensor(seq)
        mask[b, :len(seq)] = 1
        P = len(p)
        pos_b += [b] * len(s)
        pos_t += list(range(P - 1, P - 1 + len(s)))       # the positions that predict each sampled token
        sampled += s
        t_ids += ti
        t_lp += tl
        t_lrest += tr
    to = lambda x, dt: torch.tensor(x, dtype=dt, device=device)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device),
            "pos_b": to(pos_b, torch.long), "pos_t": to(pos_t, torch.long), "sampled": to(sampled, torch.long),
            "t_ids": to(t_ids, torch.long), "t_lp": to(t_lp, torch.float32), "t_lrest": to(t_lrest, torch.float32)}


def micro_loss(model, mb):
    """(KL sum, NLL sum, top-1 agreement sum, positions) for one micro-batch, the logits computed only
    at the sampled positions, in checkpointed chunks (scripts/kd27b/train_kd.py's way)."""
    import torch
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.autocast("cuda", dtype=torch.bfloat16):
        hidden = inner.base_model(input_ids=mb["input_ids"], attention_mask=mb["attention_mask"],
                                  use_cache=False).last_hidden_state
    h = hidden[mb["pos_b"], mb["pos_t"]]
    w = inner.get_output_embeddings().weight
    sums = [h.new_zeros((), dtype=torch.float32) for _ in range(3)]
    for s in range(0, h.shape[0], CHUNK):
        sl = slice(s, s + CHUNK)
        part = torch.utils.checkpoint.checkpoint(_chunk, h[sl], w, mb["sampled"][sl], mb["t_ids"][sl],
                                                 mb["t_lp"][sl], mb["t_lrest"][sl], use_reentrant=False)
        sums = [a + b for a, b in zip(sums, part)]
    return sums[0], sums[1], sums[2], h.shape[0]


# ------------------------------------------------------------ models

def load_student(adapter, trainable):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from baseline import load_base, load_registry, resolve_base_quant
    reg = load_registry(STUDENT)
    if resolve_base_quant(str(adapter)):
        sys.exit("%s was trained on the NF4 base; OPD's starts are bf16" % adapter)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = PeftModel.from_pretrained(load_base(reg, False), str(adapter), is_trainable=trainable)
    if trainable:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.train()
    else:
        model.eval()
    return reg, tok, model, torch


# ------------------------------------------------------------ train (GPU)

def cmd_train(args):
    seed, r = args.seed, args.round
    rd = round_dir(seed, r, args.root)
    done = rd / "adapter" / "adapter_config.json"
    if done.exists():
        print("round %d: trained already (%s)" % (r, done.parent))
        return 0
    samples = load_jsonl(str(rd / "samples.jsonl"))
    teacher = load_jsonl(str(rd / "teacher.jsonl"))
    rows = build_rows(samples, teacher, load_prompts(args.run))
    by_id = {row[0]: row for row in rows}
    order = [by_id[i] for i in round_order(seed, r, by_id)]
    if args.limit_steps:
        order = order[:args.limit_steps * MICRO * ACCUM]
    steps = [order[i:i + MICRO * ACCUM] for i in range(0, len(order), MICRO * ACCUM)]
    prev = adapter_for(seed, r - 1, args.root)
    t0 = time.time()
    reg, tok, model, torch = load_student(prev, trainable=True)
    from transformers import get_constant_schedule_with_warmup, set_seed
    set_seed(seed + r)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, fused=True)
    sched = get_constant_schedule_with_warmup(opt, WARMUP)
    device = next(model.parameters()).device
    print("round %d: %d samples, %d steps of %d, from %s; loaded in %.0f s" % (
        r, len(order), len(steps), MICRO * ACCUM, prev, time.time() - t0), flush=True)
    rd.mkdir(parents=True, exist_ok=True)
    log = (rd / "train_log.jsonl").open("w", encoding="utf-8")
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    for k, step in enumerate(steps):
        n_pos = sum(len(s) for _, _, s, *_ in step)
        kl_t = nll_t = top_t = 0.0
        for m in range(0, len(step), MICRO):
            mb = micro_batch(step[m:m + MICRO], device)
            kl, nll, top, _ = micro_loss(model, mb)
            (kl / n_pos).backward()
            kl_t, nll_t, top_t = kl_t + kl.item(), nll_t + nll.item(), top_t + top.item()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, CLIP))
        lr = sched.get_last_lr()[0]
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": k + 1, "lr": lr, "kl": round(kl_t / n_pos, 5), "nll": round(nll_t / n_pos, 5),
               "top1": round(top_t / n_pos, 4), "positions": n_pos, "grad_norm": round(gnorm, 4),
               "seconds": round(time.time() - t1, 1)}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if (k + 1) % 20 == 0 or k + 1 == len(steps):
            print("  step %d/%d  kl %.4f  nll %.4f  top1 %.3f  |g| %.3f  lr %.1e  %.1f s/step  peak %.1f GiB" % (
                k + 1, len(steps), rec["kl"], rec["nll"], rec["top1"], gnorm, lr, (time.time() - t1) / (k + 1),
                torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    log.close()
    from baseline import write_base_marker
    tmp = rd / "adapter.partial"
    model.save_pretrained(str(tmp))
    write_base_marker(tmp, False)
    (tmp / "opd_round.json").write_text(json.dumps({
        "seed": seed, "round": r, "from": str(prev), "samples": len(order), "steps": len(steps), "lr": LR,
        "warmup": WARMUP, "effective_batch": MICRO * ACCUM, "clip": CLIP, "topk": TOPK,
        "loss": "reverse KL over the teacher's top 20 and the rest, per sampled position",
        "seconds": round(time.time() - t0), "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
        "time": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1) + "\n", encoding="utf-8")
    tmp.rename(rd / "adapter")
    print("round %d: saved %s in %.1f min" % (r, rd / "adapter", (time.time() - t0) / 60), flush=True)
    return 0


# ------------------------------------------------------------ evaluate (GPU)

def subset_cases():
    from train_sft import SUBSET_SEED, stratified_subset
    return stratified_subset(load_jsonl(str(VAL)), SUBSET, seed=SUBSET_SEED)


def cmd_eval(args):
    seed, r = args.seed, args.round
    rd = round_dir(seed, r, args.root)
    path = rd / "eval.json"
    if path.exists():
        print("round %d: evaluated already (%s)" % (r, path))
        return 0
    from baseline import generate_predictions
    cases = subset_cases()
    if args.limit:
        cases = cases[:args.limit]
    adapter = adapter_for(seed, r, args.root)
    t0 = time.time()
    reg, tok, model, torch = load_student(adapter, trainable=False)
    records, preds = generate_predictions(model, tok, cases, reg, MAX_NEW)
    scored = score(cases, preds, timeout=args.timeout)
    rd.mkdir(parents=True, exist_ok=True)
    write_exec_log(str(rd / "eval_exec.jsonl"), cases, scored)
    rep = {"seed": seed, "round": r, "adapter": str(adapter), "n": len(cases), "exec_match": scored["n_match"],
           "exec_match_pct": round(100.0 * scored["n_match"] / len(cases), 2),
           "valid_pct": round(100.0 * scored["n_valid"] / len(cases), 2),
           "parse_failures": sum(1 for x in records if not x["parse_ok"]),
           "think_leaks": sum(1 for x in records if x["think_leak"]),
           "output_tokens_mean": round(sum(x["output_tokens"] for x in records) / len(records), 2),
           "seconds": round(time.time() - t0)}
    path.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("round %d on the %d-row subset: %d matched (%.1f%%), valid %.1f%%, in %.0f s" % (
        r, rep["n"], rep["exec_match"], rep["exec_match_pct"], rep["valid_pct"], rep["seconds"]), flush=True)
    return 0


# ------------------------------------------------------------ choose, and full val (GPU)

def choose(evals):
    """The round with the highest subset exec-match; a tie goes to the earlier round."""
    return max(sorted(evals), key=lambda r: (evals[r]["exec_match"], -r))


def cmd_final(args):
    seed = args.seed
    od = pathlib.Path(args.root) if args.root else out_dir(seed)
    evals = {r: json.loads((round_dir(seed, r, args.root) / "eval.json").read_text(encoding="utf-8"))
             for r in range(ROUNDS + 1)}
    chosen = choose(evals)
    pick = {"seed": seed, "chosen_round": chosen, "adapter": str(adapter_for(seed, chosen, args.root)),
            "subset": {r: e["exec_match_pct"] for r, e in evals.items()}}
    (od / "chosen.json").write_text(json.dumps(pick, indent=1) + "\n", encoding="utf-8")
    print("seed %d: subset by round %s -> round %d" % (seed, pick["subset"], chosen), flush=True)
    if chosen == 0:
        print("round 0 chosen: OPD picked no change; the start's own full-val files stand", flush=True)
        return 0
    if (od / "opd_val.json").exists():
        print("full val done already (%s)" % (od / "opd_val.json"))
        return 0
    from transformers import AutoTokenizer
    from baseline import load_registry
    from train_sft import full_val_eval
    reg = load_registry(STUDENT)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    summary = full_val_eval(reg, tok, adapter_for(seed, chosen, args.root), load_jsonl(str(VAL)), od,
                            load_in_4bit=False, max_new_tokens=MAX_NEW, timeout=args.timeout, val_path=str(VAL),
                            run_name=od.name, condition="opd27b-s%d" % seed, wandb_state="offline",
                            training={"method": "on-policy distillation", "seed": seed, "chosen_round": chosen,
                                      "rounds": ROUNDS, "preregistration": str((RUN / "preregistration.md").relative_to(REPO))},
                            summary_name="opd_val.json")
    print("seed %d: full val %.1f%%" % (seed, summary["metrics"]["exec_match_pct"]), flush=True)
    return 0


def cmd_adapter(args):
    print(adapter_for(args.seed, args.round, args.root))
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prompts")
    a.add_argument("--run", default=str(RUN))
    a.set_defaults(fn=cmd_prompts)
    a = sub.add_parser("adapter", help="print the adapter at the end of a round (round 0: the start)")
    a.add_argument("--seed", type=int, choices=sorted(STARTS), required=True)
    a.add_argument("--round", type=int, required=True)
    a.add_argument("--root", default=None)
    a.set_defaults(fn=cmd_adapter)
    for name, fn in (("train", cmd_train), ("eval", cmd_eval), ("final", cmd_final)):
        a = sub.add_parser(name)
        a.add_argument("--seed", type=int, choices=sorted(STARTS), required=True)
        if name != "final":
            a.add_argument("--round", type=int, required=True)
        a.add_argument("--run", default=str(RUN), help="where prompts.jsonl is")
        a.add_argument("--root", default=None, help="the arm's directory (default runs/qwen3.5-4b-opd27b-s<seed>)")
        a.add_argument("--timeout", type=float, default=5.0)
        a.add_argument("--limit", type=int, default=None, help="eval: first N subset rows (check)")
        a.add_argument("--limit-steps", type=int, default=None, help="train: first N steps (check)")
        a.set_defaults(fn=fn)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
