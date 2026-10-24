#!/usr/bin/env python3
"""Train a yes/no verifier for reranking the served adapter's samples.

The model starts from a policy (the registry base with a policy adapter merged
in, so it begins with the task's vocabulary), gets a fresh LoRA, and is trained
with TRL's SFTTrainer on (prompt, "yes"/"no") pairs from build_examples.py,
loss on the answer only -- the same stack as scripts/train_sft.py. Selection
is by reranking, not loss: every --eval-every steps the val candidates are
scored (logit yes - logit no at the answer position) and the adapter that best
picks a right command among each prompt's sampled candidates on the 250-row
selection subset (train_sft.py's seeded subset) is kept as best/. The other
295 val rows are reported, never selected on.

    python scripts/reranker/train_verifier.py --model qwen3.5-4b \\
        --init-adapter runs/qwen3.5-4b-grpo-gated/best \\
        --train runs/reranker_2026-10-16/train_examples.jsonl \\
        --val runs/reranker_2026-10-16/val_examples.jsonl \\
        --out-dir runs/reranker_2026-10-16/verifier-4b --wandb
    ... --max-steps 3 --eval-every 2 --eval-limit 40     # smoke
"""
import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", HERE):
    sys.path.insert(0, str(p))
from verifier_lib import (NO, YES, answer_token_ids, auroc, check_scoring,   # noqa: E402
                          group, load_policy_merged, rule_outcomes, score_examples,
                          verifier_messages)

PAUSED_RC = 3


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def build_dataset(rows, cases, chat_kwargs):
    from datasets import Dataset
    out = []
    for r in rows:
        row = {"prompt": verifier_messages(cases[r["id"]], r["cand_mode"], r["cand_command"]),
               "completion": [{"role": "assistant", "content": YES if r["label"] else NO}]}
        if chat_kwargs:
            row["chat_template_kwargs"] = chat_kwargs
        out.append(row)
    return Dataset.from_list(out)


def val_metrics(rows, scores, select_ids):
    """Rerank metrics on the selection subset, the untouched rest, and all of val."""
    smap = {(r["id"], r["cand_mode"], r["cand_command"]): s for r, s in zip(rows, scores)}
    outcomes = rule_outcomes(group(rows), smap)
    out = {}
    for name, keep in (("select", lambda i: i in select_ids), ("untouched", lambda i: i not in select_ids),
                       ("all", lambda i: True)):
        ids = [i for i in outcomes if keep(i)]
        if not ids:
            continue
        m = {"n": len(ids)}
        for rule in ("greedy", "majority", "verifier_samples", "verifier_all", "oracle_samples"):
            vals = [outcomes[i][rule] for i in ids if outcomes[i].get(rule) is not None]
            m[rule] = round(100 * sum(vals) / len(ids), 2) if vals else None
        pos = [s for r, s in zip(rows, scores) if keep(r["id"]) and r["label"] == 1]
        neg = [s for r, s in zip(rows, scores) if keep(r["id"]) and r["label"] == 0]
        m["auroc"] = round(auroc(pos, neg), 4)
        out[name] = m
    return out


def make_callback(TrainerCallback):
    class RerankEval(TrainerCallback):
        def __init__(self, tok, rows, cases, chat_kwargs, ids, select_ids, out_dir, every,
                     wandb_run, max_hours):
            self.tok, self.rows, self.cases, self.kw = tok, rows, cases, chat_kwargs
            self.yes, self.no = ids
            self.select_ids, self.out_dir, self.every = select_ids, out_dir, every
            self.wandb_run, self.max_hours, self.t0 = wandb_run, max_hours, time.time()
            self.best, self.best_step, self.last_eval = None, None, None
            self.history = out_dir / "eval_history.jsonl"

        def evaluate(self, model, step):
            t = time.time()
            was_training = model.training
            scores, trunc = score_examples(model, self.tok, self.rows, self.cases, self.kw,
                                           self.yes, self.no)
            if was_training:
                model.train()
            m = val_metrics(self.rows, scores, self.select_ids)
            rec = {"step": step, "eval_seconds": round(time.time() - t, 1), "truncated": trunc, **m}
            with self.history.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            view = m.get("select") or m["all"]          # a smoke's few prompts may miss the subset
            sel = view["verifier_samples"]
            print("  [eval] step %d: select rerank %.1f%% (greedy %.1f, majority %.1f, oracle %.1f), "
                  "untouched %.1f%%, AUROC %.3f, %.0f s" % (
                      step, sel, view["greedy"], view["majority"],
                      view["oracle_samples"], m.get("untouched", {}).get("verifier_samples") or 0,
                      m["all"]["auroc"], rec["eval_seconds"]), flush=True)
            if self.wandb_run is not None:
                self.wandb_run.log({"rerank/select": sel, "rerank/auroc": m["all"]["auroc"],
                                    "train/global_step": step})
            if self.best is None or sel > self.best:
                self.best, self.best_step = sel, step
                best = self.out_dir / "best"
                model.save_pretrained(str(best))
                (best / "selection.json").write_text(json.dumps(
                    {"step": step, "metric": "val select-subset rerank accuracy (verifier over samples)",
                     "value": sel, "n": view["n"]}, indent=1) + "\n", encoding="utf-8")
            self.last_eval = step
            # scored rows for this step, so a later report needs no GPU
            with (self.out_dir / ("val_scores_step%d.jsonl" % step)).open("w", encoding="utf-8") as fh:
                for r, s in zip(self.rows, scores):
                    fh.write(json.dumps({"id": r["id"], "cand_mode": r["cand_mode"],
                                         "cand_command": r["cand_command"], "score": s}) + "\n")

        def on_step_end(self, args, state, control, model=None, **kw):
            if self.every and state.global_step % self.every == 0:
                self.evaluate(model, state.global_step)
            if (pathlib.Path(self.out_dir) / "STOP").exists():
                control.should_training_stop = True
                (pathlib.Path(self.out_dir) / "PAUSED").write_text(
                    json.dumps({"step": state.global_step}) + "\n", encoding="utf-8")
            if self.max_hours and time.time() - self.t0 > self.max_hours * 3600:
                print("  [guard] --max-hours reached; stopping", flush=True)
                control.should_training_stop = True
            return control

        def on_train_end(self, args, state, control, model=None, **kw):
            if self.last_eval != state.global_step:
                self.evaluate(model, state.global_step)
    return RerankEval


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="registry name (configs/models.yaml)")
    ap.add_argument("--init-adapter", default=None, help="policy adapter merged in before the new LoRA")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--train-cases", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--val-cases", default=str(REPO / "data" / "splits" / "val.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--eval-every", type=int, default=None, help="default: a third of an epoch")
    ap.add_argument("--eval-limit", type=int, default=None, help="score only the first N val prompts (smoke)")
    ap.add_argument("--no-eval-at-start", action="store_true")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--max-hours", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args(argv)

    from baseline import load_registry, wandb_mode
    from train_sft import SUBSET_SEED, stratified_subset
    reg = load_registry(args.model)
    kw = reg.get("chat_template_kwargs") or {}
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(args.train)
    val_rows = load_jsonl(args.val)
    cases = {c["id"]: c for c in load_jsonl(args.train_cases)}
    val_cases = load_jsonl(args.val_cases)
    cases.update({c["id"]: c for c in val_cases})
    select_ids = {c["id"] for c in stratified_subset(val_cases, 250, seed=SUBSET_SEED)}
    if args.eval_limit:
        keep = sorted({r["id"] for r in val_rows})[:args.eval_limit]
        val_rows = [r for r in val_rows if r["id"] in set(keep)]
    overlap = {r["id"] for r in train_rows} & {r["id"] for r in val_rows}
    if overlap:
        raise SystemExit("train and val examples share %d prompt ids" % len(overlap))

    import torch
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    yes_id, no_id = answer_token_ids(tok, kw)
    model = load_policy_merged(reg, args.init_adapter)
    print("loaded %s%s in %.0f s; answer tokens yes=%d no=%d" % (
        reg["name"], " + %s (merged)" % args.init_adapter if args.init_adapter else "",
        time.time() - t0, yes_id, no_id), flush=True)

    steps_per_epoch = max(1, len(train_rows) // (args.batch_size * args.grad_accum))
    total = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    every = args.eval_every or max(1, steps_per_epoch // 3)
    cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=max(1, round(0.03 * total)),
        bf16=True, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length, packing=False, completion_only_loss=True,
        eval_strategy="no", save_strategy="no", logging_steps=10,
        report_to="wandb" if args.wandb else "none", run_name=out_dir.name, seed=args.seed)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
                      task_type="CAUSAL_LM", target_modules=reg.get("lora_target_modules", "all-linear"))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=build_dataset(train_rows, cases, kw),
                         processing_class=tok, peft_config=lora)
    for prm in trainer.model.parameters():                       # fp32 adapters, as train_sft.py
        if prm.requires_grad and prm.dtype != torch.float32:
            prm.data = prm.data.float()

    # The answer token the loss sees must be the one the scorer reads: in a
    # collated batch, the first label that carries loss is yes or no.
    batch = next(iter(trainer.get_train_dataloader()))
    for i in range(batch["input_ids"].shape[0]):
        keep = (batch["labels"][i] != -100).nonzero()
        if len(keep) == 0:
            raise SystemExit("an example carries no loss: the answer was truncated away")
        first = int(batch["input_ids"][i][int(keep[0])])
        if first not in (yes_id, no_id):
            raise SystemExit("first loss-bearing token %r is not the yes/no token the scorer reads"
                             % tok.decode([first]))
    print("answer-position check: %d examples, loss starts on yes/no" % batch["input_ids"].shape[0])
    # ... and the fast scorer must agree with full logits (padding, decoder, head rows).
    probe = sorted(val_rows, key=lambda r: len(cases[r["id"]]["buffer_before"]))
    probe = probe[:3] + probe[-3:]                       # short and long prompts in one padded batch
    worst = check_scoring(trainer.model, tok, probe, cases, kw, yes_id, no_id)
    trainer.model.train()
    print("scoring check: batched two-row scores match full logits (max diff %.4f)" % worst, flush=True)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="vimBot", name=out_dir.name, mode=wandb_mode(),
                               config={"model": reg["name"], "init_adapter": args.init_adapter,
                                       "train_examples": len(train_rows), "lr": args.lr,
                                       "epochs": args.epochs, "lora_r": args.lora_r})
    Eval = make_callback(TrainerCallback)
    evaluator = Eval(tok, val_rows, cases, kw, (yes_id, no_id), select_ids, out_dir, every,
                     wandb_run, args.max_hours)
    trainer.add_callback(evaluator)
    (out_dir / "train_config.json").write_text(json.dumps({
        "model": reg["name"], "hf_id": reg["hf_id"], "revision": reg["revision"],
        "init_adapter": args.init_adapter, "train": args.train, "val": args.val,
        "train_examples": len(train_rows), "train_positives": sum(r["label"] for r in train_rows),
        "val_examples": len(val_rows), "steps_per_epoch": steps_per_epoch, "total_steps": total,
        "eval_every": every, "lr": args.lr, "epochs": args.epochs, "batch": [args.batch_size, args.grad_accum],
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha}, "max_length": args.max_length,
        "seed": args.seed, "answer_tokens": {"yes": yes_id, "no": no_id},
        "selection": "val subset (train_sft.py stratified_subset, seed %d), verifier over samples" % SUBSET_SEED,
    }, indent=1) + "\n", encoding="utf-8")
    print("train %d examples (%d positive), val %d candidates; %d steps/epoch, %d total, eval every %d"
          % (len(train_rows), sum(r["label"] for r in train_rows), len(val_rows), steps_per_epoch,
             total, every), flush=True)
    if not args.no_eval_at_start:
        evaluator.evaluate(trainer.model, 0)
    trainer.train()
    trainer.model.save_pretrained(str(out_dir / "final"))
    if (out_dir / "PAUSED").exists():
        print("paused; partial run kept", flush=True)
        return PAUSED_RC
    summary = {"best_step": evaluator.best_step, "best_select": evaluator.best,
               "minutes": round((time.time() - t0) / 60, 1)}
    (out_dir / "verifier_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("done: best step %s, select-subset rerank %.1f%%" % (evaluator.best_step, evaluator.best), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
