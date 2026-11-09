#!/usr/bin/env python3
"""The 27B with thinking on, on the val rows it misses and the rows no model solves: does reasoning rescue them?

Exploratory, by runs/think27b_2026-11-01/preregistration.md. The owner, 2026-10-31: "we can try
27B with thinking for the hard rows and rows that failed to see how it works", with thinking
allowed for this teacher only (CLAUDE.md §14; DECISIONS.md 2026-10-31). Only the 27B thinks: no
student, results-table row or served model does.

Rows (`rows` stage, CPU): the 208 val rows the trained 27B answers wrong
(runs/qwen3.5-27b-sft-distill/exec_val.jsonl) and the 70 that neither the 4B nor the 9B solves in
128 samples each (runs/serve_pooled/served_at_k/rows_val.jsonl): 213 rows, 65 in both. The test
split is not touched.

Arms (`run` stage, GPU), from one load of the NF4 base with the trained adapter. All are sampled
with the model card's thinking settings (its generation_config.json: T 0.6, top-p 0.95, top-k 20;
greedy decoding is advised against in thinking mode), so each thinking arm has a control that
differs from it only in thinking:

  nothink-trained  the trained 27B, the §6 prompt, thinking off (control)
  nothink-retr8    the base 27B (adapter disabled), the 8 retrieved examples, thinking off (control)
  think-trained    the trained 27B, the §6 prompt, thinking on
  think-retr8      the base 27B, the 8 retrieved examples, thinking on

The thinking arms render their prompts through registry `qwen3.5-27b-think` (enable_thinking:
true), the controls through `qwen3.5-27b`; both name the same weights. A thinking completion is
split at its first `</think>`, and the text after it must satisfy the §6 grammar like any answer.
A completion that does not close its reasoning within --max-new-tokens has no answer: a miss,
counted apart.

Batches of 8, left-padded. A guard first compares a padded batch's next-token distributions with
one-row passes, and falls back to one row per batch if they differ by more than --guard-tol. The
input embedding sits in system RAM (scripts/qlora27b's exact lookup), which leaves room for the
reasoning's cache, and a batch that runs out of memory is retried in halves. The run stops starting batches at --deadline (epoch seconds) and resumes from
the rows on disk. A card that slows to a third of its own pace (paging) stops it with exit 5.

    python scripts/think27b/think27b.py rows
    python scripts/think27b/think27b.py run --deadline 1762020000
"""
import argparse
import contextlib
import datetime
import importlib.util
import itertools
import json
import os
import pathlib
import random
import statistics
import sys
import time
import zlib

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl, score                            # noqa: E402
from prompt import messages_for_case, parse_output               # noqa: E402

THINK_MODEL, PLAIN_MODEL = "qwen3.5-27b-think", "qwen3.5-27b"
RUN = REPO / "runs" / "think27b_2026-11-01"
VAL = REPO / "data" / "splits" / "val.jsonl"
TRAIN = REPO / "data" / "splits" / "train.jsonl"
ADAPTER = REPO / "runs" / "qwen3.5-27b-sft-distill" / "best"
MISSES = REPO / "runs" / "qwen3.5-27b-sft-distill" / "exec_val.jsonl"
SERVED_ROWS = REPO / "runs" / "serve_pooled" / "served_at_k" / "rows_val.jsonl"
PLAN = REPO / "runs" / "teacher27b_2026-10-30" / "plans" / "plan_retr8.jsonl"
THINK_END = "</think>"
PREREGISTERED_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}   # the card's, checked at load
SEED = 20261101
RC_SLOW = 5

# name: (registry entry for the template, adapter on, retrieved examples). The order is the run
# order: the two quick controls first, so a deadline can only cut a thinking arm.
ARMS = {
    "nothink-trained": (PLAIN_MODEL, True, False),
    "nothink-retr8": (PLAIN_MODEL, False, True),
    "think-trained": (THINK_MODEL, True, False),
    "think-retr8": (THINK_MODEL, False, True),
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ rows

def select_rows(misses_path=MISSES, served_path=SERVED_ROWS):
    """[{id, in_fail, in_hard}] in val order: the trained 27B's greedy misses, and the rows no 4B
    or 9B sample matched."""
    fail = {r["id"] for r in load_jsonl(str(misses_path)) if not r["match"]}
    hard = {r["id"] for r in load_jsonl(str(served_path))
            if not any(c["state"] == "match" for c in r["cands"])}
    order = [c["id"] for c in load_jsonl(str(VAL))]
    return [{"id": i, "in_fail": i in fail, "in_hard": i in hard} for i in order if i in fail | hard]


def cmd_rows(args):
    rows = select_rows()
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    rep = {"rows": rows, "n": len(rows), "fail": sum(r["in_fail"] for r in rows),
           "hard": sum(r["in_hard"] for r in rows), "both": sum(r["in_fail"] and r["in_hard"] for r in rows),
           "sources": {"misses": str(MISSES.relative_to(REPO)), "never_solved": str(SERVED_ROWS.relative_to(REPO))}}
    (run / "rows.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("%d rows: %d the trained 27B misses, %d no 4B/9B sample solves, %d both"
          % (rep["n"], rep["fail"], rep["hard"], rep["both"]))
    return 0


# ------------------------------------------------------------ completions

def trim(ids, stop_ids, pad_id):
    """Generated ids up to (not including) the first stop token, pads dropped."""
    out = []
    for t in ids:
        if t in stop_ids:
            break
        if t != pad_id:
            out.append(t)
    return out


def split_reasoning(text, thinking):
    """(reasoning, answer, closed). A thinking completion's answer is the text after its first
    </think>; without one there is no answer. A plain completion is all answer."""
    if not thinking:
        return "", text, True
    k = text.find(THINK_END)
    if k < 0:
        return text, None, False
    return text[:k], text[k + len(THINK_END):], True


def think_tokens(ids, think_end_id):
    return ids.index(think_end_id) if think_end_id in ids else len(ids)


def order_for(arm, ids):
    """A fixed shuffle per arm, so rows cut by a deadline are a random subset."""
    out = list(ids)
    random.Random("%s|%s" % (SEED, arm)).shuffle(out)
    return out


def schedule(todo, batch_size, thinking):
    """The run order as (arm, offset, ids) batches: the controls one after the other, then the
    thinking arms alternating batch by batch. At one row per batch a deadline would otherwise cut
    the second thinking arm to almost nothing; alternating leaves each arm a random subset of about
    the same size (addendum 1 of the pre-registration)."""
    def batches(arm):
        ids = todo[arm]
        return [(arm, b, ids[b:b + batch_size]) for b in range(0, len(ids), batch_size)]
    out = [x for a in todo if not thinking[a] for x in batches(a)]
    for group in itertools.zip_longest(*(batches(a) for a in todo if thinking[a])):
        out += [x for x in group if x is not None]
    return out


def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.open(encoding="utf-8") if line.strip()}


# ------------------------------------------------------------ the GPU stage

class SlowCard(RuntimeError):
    pass


def load_model():
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from baseline import load_base, load_registry, resolve_base_quant, stop_token_ids
    regs = {n: load_registry(n) for n in (THINK_MODEL, PLAIN_MODEL)}
    a, b = regs[THINK_MODEL], regs[PLAIN_MODEL]
    if (a["hf_id"], a["revision"]) != (b["hf_id"], b["revision"]):
        sys.exit("%s and %s must name the same weights" % (THINK_MODEL, PLAIN_MODEL))
    if not resolve_base_quant(str(ADAPTER)):
        sys.exit("the adapter at %s was not trained on the NF4 base" % ADAPTER)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a["hf_id"], revision=a["revision"])
    tok.padding_side = "left"
    base = load_base(a, True)
    Q = load("qlora27b_train_27b", REPO / "scripts" / "qlora27b" / "train_27b.py")
    moved = Q.move_input_embedding_to_ram(base)
    gen = base.generation_config
    sampling = {"temperature": gen.temperature, "top_p": gen.top_p, "top_k": gen.top_k}
    if sampling != PREREGISTERED_SAMPLING:
        sys.exit("the model card's sampling is %s, not the pre-registered %s" % (sampling, PREREGISTERED_SAMPLING))
    model = PeftModel.from_pretrained(base, str(ADAPTER))
    model.eval()
    stops = set(stop_token_ids(tok, a["stop"]))
    eos = gen.eos_token_id
    stops |= set(eos if isinstance(eos, list) else [eos])
    info = {"load_s": round(time.time() - t0), "embedding_in_ram_gib": round(moved, 2),
            "allocated_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
            "sampling": sampling, "stop_ids": sorted(stops), "pad_id": tok.pad_token_id,
            "kwargs": {n: r.get("chat_template_kwargs") or {} for n, r in regs.items()},
            "hf_id": a["hf_id"], "revision": a["revision"], "adapter": str(ADAPTER.relative_to(REPO))}
    print("loaded %s @ %s (NF4) + %s in %ds; %.2f GiB allocated (embedding %.2f GiB in RAM); "
          "sampling %s" % (a["hf_id"], a["revision"][:12], info["adapter"], info["load_s"],
                           info["allocated_gib"], moved, sampling), flush=True)
    return tok, model, info


def generate(model, tok, texts, adapter_on, max_new, info, seed, stopping_criteria=None):
    """Token ids per text (trimmed at the first stop), retrying in halves on a CUDA OOM.
    `stopping_criteria` goes to model.generate: scripts/think_loop's loop breaker."""
    import torch
    from transformers import set_seed
    try:
        set_seed(seed)
        enc = tok(texts, return_tensors="pt", padding=True)
        enc = {k: v.to(next(model.parameters()).device) for k, v in enc.items()}
        ctx = contextlib.nullcontext() if adapter_on else model.disable_adapter()
        with ctx, torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, **info["sampling"],
                                 eos_token_id=info["stop_ids"], pad_token_id=info["pad_id"],
                                 stopping_criteria=stopping_criteria)
        gen = out[:, enc["input_ids"].shape[1]:].tolist()
        return [trim(r, set(info["stop_ids"]), info["pad_id"]) for r in gen], [len(r) for r in gen]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(texts) == 1:
            raise
        h = len(texts) // 2
        print("  out of memory at batch %d: retrying in halves" % len(texts), flush=True)
        a_ids, a_len = generate(model, tok, texts[:h], adapter_on, max_new, info, seed, stopping_criteria)
        b_ids, b_len = generate(model, tok, texts[h:], adapter_on, max_new, info, seed + 1, stopping_criteria)
        return a_ids + b_ids, a_len + b_len


def padding_guard(model, tok, texts, adapter_on):
    """Largest |Δ log p| of the next token between a left-padded batch and one-row passes. The
    linear-attention layers zero padded inputs (transformers' recurrent mask), so this should be
    numerical noise; if not, the run falls back to one row per batch."""
    import torch
    device = next(model.parameters()).device
    ctx = contextlib.nullcontext() if adapter_on else model.disable_adapter()
    with ctx, torch.no_grad():
        enc = tok(texts, return_tensors="pt", padding=True)
        batched = model(**{k: v.to(device) for k, v in enc.items()}, logits_to_keep=1).logits[:, -1].float()
        worst = 0.0
        for j, t in enumerate(texts):
            one = tok([t], return_tensors="pt")
            single = model(**{k: v.to(device) for k, v in one.items()}, logits_to_keep=1).logits[0, -1].float()
            d = (torch.log_softmax(batched[j], -1) - torch.log_softmax(single, -1)).abs().max()
            worst = max(worst, float(d))
    return worst


def judge(cases, answers, timeout):
    """Parse each answer by the §6 grammar and run the parsed ones in the sandbox."""
    parsed = {c["id"]: parse_output(a) if a is not None else None for c, a in zip(cases, answers)}
    preds = {i: {"mode": p[0], "command": p[1]} for i, p in parsed.items() if p}
    per = score(cases, preds, timeout=timeout)["per_case"]
    return parsed, per


def cmd_run(args):
    import torch
    run = pathlib.Path(args.run)
    rows = json.loads((run / "rows.json").read_text(encoding="utf-8"))["rows"]
    flags = {r["id"]: r for r in rows}
    val = {c["id"]: c for c in load_jsonl(str(VAL))}
    train = {c["id"]: c for c in load_jsonl(str(TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(PLAN))}
    arms = [a for a in ARMS if a in args.arms]
    todo = {a: [i for i in order_for(a, flags) if i not in done_ids(run / ("rows_%s.jsonl" % a))] for a in arms}
    print("%s" % ", ".join("%s %d to do" % (a, len(t)) for a, t in todo.items()), flush=True)
    if not any(todo.values()):
        return 0
    if time.time() > args.deadline:
        print("deadline already passed: nothing started", flush=True)
        return 0
    tok, model, info = load_model()
    (run / ("load_%s.json" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))).write_text(
        json.dumps(info, indent=1) + "\n", encoding="utf-8")
    think_end_id = tok.convert_tokens_to_ids(THINK_END)
    # The guard: left padding must not move the next-token distribution (prompts of the longest
    # arm, the retrieved examples, on the base and on the adapter).
    sample = [val[i] for i in order_for("guard", flags)[:args.batch_size]]
    texts = [tok.apply_chat_template(messages_for_case(c, [train[j] for j in plan[c["id"]]["examples"]]),
                                     add_generation_prompt=True, tokenize=False,
                                     **info["kwargs"][THINK_MODEL]) for c in sample]
    guard = max(padding_guard(model, tok, texts, on) for on in (True, False))
    batch_size = args.batch_size if guard <= args.guard_tol else 1
    print("guard: left-padded batch vs one row at a time, max |Δ log p| = %.4f -> batch size %d"
          % (guard, batch_size), flush=True)
    with (run / "guard.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"max_abs_dlogp": round(guard, 5), "tol": args.guard_tol, "batch_size": batch_size,
                             "time": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
    torch.cuda.reset_peak_memory_stats()
    pace = []
    slow = 0
    thinks = {a: bool(info["kwargs"][ARMS[a][0]].get("enable_thinking")) for a in arms}
    with contextlib.ExitStack() as stack:
        files = {a: stack.enter_context((run / ("rows_%s.jsonl" % a)).open("a", encoding="utf-8")) for a in arms}
        for arm, b, batch_ids in schedule(todo, batch_size, thinks):
            reg_name, adapter_on, retr = ARMS[arm]
            kwargs = info["kwargs"][reg_name]
            thinking = thinks[arm]
            max_new = args.max_new_tokens if thinking else args.max_new_plain
            ids, fh = todo[arm], files[arm]
            if time.time() > args.deadline:
                print("deadline reached in %s after %d of %d rows; the report uses what is done"
                      % (arm, b, len(ids)), flush=True)
                return 0
            batch = [val[i] for i in batch_ids]
            exs = [[train[j] for j in plan[c["id"]]["examples"]] if retr else [] for c in batch]
            texts = [tok.apply_chat_template(messages_for_case(c, e), add_generation_prompt=True,
                                             tokenize=False, **kwargs) for c, e in zip(batch, exs)]
            # Keyed by the arm and the batch's first row, so a resumed run draws the same seeds.
            seed = SEED + 100000 * list(ARMS).index(arm) + zlib.crc32(batch[0]["id"].encode()) % 100000
            t0 = time.time()
            outs, lens = generate(model, tok, texts, adapter_on, max_new, info, seed)
            secs = time.time() - t0
            steps = max(lens)
            texts_out = [tok.decode(o, skip_special_tokens=False) for o in outs]
            splits = [split_reasoning(t, thinking) for t in texts_out]
            parsed, per = judge(batch, [s[1] for s in splits], args.timeout)
            stamp = datetime.datetime.now().isoformat(timespec="seconds")
            for c, e, o, n, (reasoning, answer, closed) in zip(batch, exs, outs, lens, splits):
                p, r = parsed[c["id"]], per.get(c["id"], {})
                fh.write(json.dumps({
                    "arm": arm, "id": c["id"], "tier": c.get("tier"), "category": c.get("category"),
                    "in_fail": flags[c["id"]]["in_fail"], "in_hard": flags[c["id"]]["in_hard"],
                    "k_examples": len(e), "new_tokens": len(o),
                    "think_tokens": think_tokens(o, think_end_id) if thinking else 0,
                    "closed": closed, "hit_cap": n >= max_new and not closed,
                    "answer": answer, "reasoning": reasoning, "parse_ok": p is not None,
                    "mode": p[0] if p else None, "command": p[1] if p else None,
                    "valid": bool(r.get("valid")), "match": bool(r.get("match")), "error": r.get("error"),
                    "batch_seconds": round(secs, 1), "batch_size": len(batch), "steps": steps,
                    "seed": seed, "max_new_tokens": max_new, "time": stamp}) + "\n")
            fh.flush()
            done = b + len(batch)
            print("  %s %d/%d  %d steps in %.0f s (%.2f s/step)  matched %d/%d  think tokens %s  "
                  "peak %.1f GiB" % (arm, done, len(ids), steps, secs, secs / max(steps, 1),
                                     sum(per.get(c["id"], {}).get("match", False) for c in batch), len(batch),
                                     [think_tokens(o, think_end_id) for o in outs] if thinking else "-",
                                     torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
            if steps >= 256:                 # per-step time is prefill-bound below this
                per_step = secs / steps
                if len(pace) >= 3 and per_step > 3 * statistics.median(pace):
                    slow += 1
                    if slow >= 2:
                        raise SlowCard("%.2f s/step against a median of %.2f: the card is paging"
                                       % (per_step, statistics.median(pace)))
                else:
                    slow = 0
                    pace.append(per_step)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rows", help="CPU: select the rows -> <run>/rows.json")
    r.add_argument("--run", default=str(RUN))
    g = sub.add_parser("run", help="GPU: the four arms, resumable, until --deadline")
    g.add_argument("--run", default=str(RUN))
    g.add_argument("--deadline", type=float, required=True, help="epoch seconds; no batch starts after it")
    g.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--max-new-tokens", type=int, default=4096, help="thinking arms: reasoning + answer")
    g.add_argument("--max-new-plain", type=int, default=192, help="the controls (§6 answers)")
    g.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    g.add_argument("--guard-tol", type=float, default=0.1, help="largest |Δ log p| allowed for batching")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "rows":
        return cmd_rows(args)
    try:
        return cmd_run(args)
    except SlowCard as e:
        print("stopped: %s" % e, flush=True)
        return RC_SLOW


if __name__ == "__main__":
    sys.exit(main())
