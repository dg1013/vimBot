#!/usr/bin/env python3
"""LoRA (low-rank adaptation) SFT (supervised fine-tuning) on the vimBot corpus, TRL (Transformer Reinforcement Learning) path (CLAUDE.md §8).

Teaches the registry's student model to answer the §6 prompt with one line,
`ex: <command>` or `keys: <keystrokes>`, and judges it the only way this repo (repository)
accepts: generate on val, run the answer in the sandboxed vim, compare buffers.
Loss curves are diagnostics here, never the evaluation.

Six decisions are baked in. Each is a place where a plausible alternative
exists, so each is stated with the reason:

1. **The training example is the inference prompt plus the gold answer.**
   The rows are handed to TRL in prompt-completion form -- `prompt` is the §6
   message list from src/prompt.py, `completion` is one assistant turn holding
   the target line -- and TRL renders both through the model's own chat
   template. Rendering training data any other way trains on a distribution
   that is never served. Per-row `chat_template_kwargs` carry the registry's
   template flags (`enable_thinking: false` for Qwen3.5), which is TRL's
   mechanism for them; there is no such field on the config (configuration) object.

2. **Loss on the answer tokens only** (`completion_only_loss=True`). Prompt
   tokens are labelled -100 and contribute nothing. The §6 prompt carries the
   whole buffer, so it is roughly 110 of the ~120 tokens in a median example:
   training on it would spend most of the gradient teaching the model to
   predict buffers and instructions, text it never has to generate. TRL builds
   this mask exactly -- it tokenizes prompt and completion separately and
   concatenates, rather than tokenizing the whole string and guessing where the
   answer starts -- so the mask cannot drift off by a token at the boundary.

3. **LoRA r=16 alpha=32 on all linear layers** (§8), base weights frozen. On
   this model that is 248 modules and 32.5 M trainable parameters, 0.77% of
   4.21 B, which is what keeps the optimizer state small enough to train a 4B
   model on one 24 GB (gigabytes) card. `all-linear` reaches both attention families here
   (8 full-attention layers, 24 gated-delta layers) and every MLP. The registry
   owns the target list, so a model swap needs no code change.

4. **Gradient checkpointing on.** Activations are recomputed in the backward
   pass instead of being stored: measured 16.3 GB peak with it on, and without
   it the same shape ran the card to its 24 GB ceiling. The cost is roughly a
   third more compute. This model's recurrent layers also run in pure PyTorch
   because the fused kernels need a C compiler this box does not have
   (DECISIONS.md 2026-09-05); correctness is unaffected, throughput is not.

5. **Effective batch 32 by accumulation** (8 x 4), not by a bigger micro-batch,
   which would not fit. The update is statistically the update of a batch of
   32. Packing stays off: it would raise throughput by filling the 1024-token
   window with several ~120-token examples, but correct packing needs
   attention that does not cross example boundaries, which here means kernels
   that are not installed.

6. **Checkpoint selection by val exec-match, not by val loss.** They disagree
   in the direction that matters: one wrong character is a small loss change
   and a failed command. The callback generates through
   `baseline.generate_predictions` -- the same function that produces the
   results table -- so a training-time number and a table number are
   comparable. Mid-training evals run on a seeded, category-proportional subset
   because a full val pass is ~7 minutes at batch size 1; the final number is
   over the full slice, with the best adapter reloaded exactly as
   `baseline.py --adapter` will load it.

Artifacts, all under runs/<model>-sft/:

    best/                    the adapter with the highest val exec-match
    final/                   the adapter as training left it
    eval_history.jsonl       one row per evaluation: step, exec-match, valid,
                             tokens, seconds
    train_config.json        every hyperparameter, library and hardware version
    completions_val.jsonl    the final full-val generations
    preds_val.jsonl          the §5 prediction file for them
    exec_val.jsonl           what vim did with each of them (see src/evaluate.py)
    sft_val.json             the §11 summary, same shape as baseline_<split>.json

Then the results-table row comes from the same script as every other row:

    python scripts/baseline.py --model qwen3.5-4b --adapter runs/qwen3.5-4b-sft/best

Usage:
    python scripts/train_sft.py --model qwen3.5-4b
    python scripts/train_sft.py --model qwen3.5-4b --inspect 2      # show the
                                # rendered example and exactly which tokens
                                # carry loss, then exit without training
    python scripts/train_sft.py --model qwen3.5-4b --limit 64 --max-steps 2 \
                                --eval-subset 8                     # smoke
    python scripts/train_sft.py --model qwen3.5-9b --load-in-4bit   # QLoRA:
                                # the base held in NF4 because 18 GB of bf16
                                # weights do not fit beside training on 24 GB.
                                # Everything else in the recipe is unchanged,
                                # adapters included (fp32 in both paths), and
                                # the choice is recorded beside the adapter so
                                # baseline.py evaluates it on the same base.
"""
import argparse
import contextlib
import datetime
import json
import os
import pathlib
import platform
import random
import statistics
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl, score, write_exec_log       # noqa: E402
from prompt import messages_for_case, render_target          # noqa: E402
from baseline import (NF4, base_quant_name, generate_predictions,   # noqa: E402
                      load_base, load_registry, summarize, vim_version,
                      wandb_mode, write_base_marker)

SUBSET_SEED = 20260908


# --------------------------------------------------------------------- data

def build_dataset(cases, reg):
    """§5 rows -> TRL prompt-completion rows, rendered by the model's template.

    `prompt` is exactly what `baseline.generate_predictions` sends at inference
    (`messages_for_case`), and `completion` is the one line §6 demands. Keeping
    both sides in src/prompt.py is what makes the contract single-copy.
    """
    from datasets import Dataset
    kwargs = reg.get("chat_template_kwargs") or {}
    rows = []
    for case in cases:
        row = {"prompt": messages_for_case(case),
               "completion": [{"role": "assistant",
                               "content": render_target(case["mode"],
                                                        case["gold_command"])}]}
        if kwargs:
            # TRL reads template flags per row; there is no config field for
            # them. Omitted entirely when empty -- an all-empty struct column
            # has no Arrow type.
            row["chat_template_kwargs"] = kwargs
        rows.append(row)
    return Dataset.from_list(rows)


def stratified_subset(cases, n, seed=SUBSET_SEED):
    """A category-proportional sample, deterministic across runs.

    Proportional rather than balanced: the point of a mid-training number is to
    approximate the full-val number, so the subset must keep val's category mix
    rather than over-weighting the rare categories.
    """
    if not n or n >= len(cases):
        return list(cases)
    by_cat = {}
    for case in cases:
        by_cat.setdefault(case.get("category", "?"), []).append(case)
    quotas, remainders = {}, []
    for cat, rows in by_cat.items():
        exact = n * len(rows) / len(cases)
        quotas[cat] = int(exact)
        remainders.append((exact - int(exact), cat))
    for _, cat in sorted(remainders, reverse=True)[:n - sum(quotas.values())]:
        quotas[cat] += 1
    rng = random.Random(seed)
    picked = []
    for cat in sorted(by_cat):
        rows = sorted(by_cat[cat], key=lambda r: r["id"])    # order before the
        rng.shuffle(rows)                                    # shuffle, so the
        picked.extend(rows[:quotas[cat]])                    # seed decides all
    return sorted(picked, key=lambda r: r["id"])


# ---------------------------------------------------------------- callbacks

@contextlib.contextmanager
def inference_mode(model):
    """Training leaves a model in a state generation cannot use.

    Gradient checkpointing forces `use_cache=False`, so generating under it
    recomputes the whole prefix for every token. Both are restored on the way
    out, including after an exception, or training silently resumes without
    checkpointing and runs out of memory.
    """
    was_training = model.training
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    cfg = model.config
    cached = getattr(cfg, "use_cache", None)
    model.eval()
    if checkpointing:
        model.gradient_checkpointing_disable()
    cfg.use_cache = True
    try:
        yield model
    finally:
        cfg.use_cache = cached
        if checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if was_training:
            model.train()


def _trainer_callback_base():
    from transformers import TrainerCallback
    return TrainerCallback


def make_callbacks(base):
    """Defined against the imported base class so `--help` needs no torch."""

    class HarnessEval(base):
        """Evaluate by generation through the sandbox (§8), and keep the best.

        Selection is on exec-match. The adapter is saved by this callback rather
        than by the Trainer's own checkpointing, which would write optimizer
        state for a metric it cannot see.
        """

        def __init__(self, tok, reg, cases, out_dir, every, max_new_tokens,
                     timeout, wandb_run=None, load_in_4bit=False):
            self.tok, self.reg, self.cases = tok, reg, cases
            self.out_dir, self.every = out_dir, every
            self.max_new_tokens, self.timeout = max_new_tokens, timeout
            self.wandb_run = wandb_run
            self.load_in_4bit = load_in_4bit
            self.best = -1.0
            self.best_step = None
            self.history = []
            self.last_seconds = 0.0

        def on_step_end(self, args, state, control, model=None, **kwargs):
            if self.every and state.global_step % self.every == 0:
                self.run(model, state.global_step)
            return control

        def on_train_end(self, args, state, control, model=None, **kwargs):
            """The last window competes for best/ too: a run whose length is
            not a multiple of the cadence would otherwise end with steps that
            were trained but never scored, and final/ could never be chosen.
            Not on a pause (`final_eval` is cleared by the GRPO pause
            callback): a paused run continues, and an off-cadence evaluation
            would cost minutes and bend the curve."""
            if not getattr(self, "final_eval", True):
                return control
            if self.every and state.global_step % self.every != 0 and state.global_step > 0:
                self.run(model, state.global_step)
            return control

        def restore(self, upto_step):
            """Rebuild the selection state from this run's eval_history.jsonl
            when training resumes from a checkpoint at `upto_step`: rows after
            it belong to steps that will be trained again and are dropped
            (the file is rewritten without them); the best-so-far is the first
            row with the highest exec-match, the same tie rule `run` applies."""
            path = self.out_dir / "eval_history.jsonl"
            rows = []
            if path.exists():
                with path.open(encoding="utf-8") as fh:
                    rows = [json.loads(line) for line in fh if line.strip()]
            rows = [r for r in rows if r["step"] <= upto_step]
            with path.open("w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            self.history = rows
            self.best, self.best_step = -1.0, None
            for r in rows:
                if r["exec_match_pct"] > self.best:
                    self.best, self.best_step = r["exec_match_pct"], r["step"]
            # best/ must be a row of this history. The eval at step N runs
            # before checkpoint-N is written, so a kill in between leaves a
            # best/ from a step the resumed run will train again; it is set
            # aside (never deleted) and the retrained window re-selects.
            best_dir = self.out_dir / "best"
            selection = best_dir / "selection.json"
            if selection.exists():
                saved_step = json.loads(selection.read_text(encoding="utf-8")).get("step")
                if saved_step is not None and saved_step > upto_step:
                    target = self.out_dir / ("best.orphaned-%s" % saved_step)
                    if target.exists():
                        target = self.out_dir / ("best.orphaned-%s.%s" % (
                            saved_step, time.strftime("%Y%m%d-%H%M%S")))
                    best_dir.rename(target)
                    print("  [resume] best/ was from step %s, past checkpoint step %s: "
                          "moved to %s" % (saved_step, upto_step, target.name), flush=True)
            return rows

        def run(self, model, step):
            t0 = time.time()
            with inference_mode(model):
                records, preds = generate_predictions(
                    model, self.tok, self.cases, self.reg, self.max_new_tokens)
            scored = score(self.cases, preds, timeout=self.timeout)
            n = len(self.cases)
            self.last_seconds = time.time() - t0
            row = {"step": step, "n": n,
                   "exec_match": scored["n_match"],
                   "exec_match_pct": round(100 * scored["n_match"] / n, 2),
                   "valid_pct": round(100 * scored["n_valid"] / n, 2),
                   "parse_failures": sum(1 for r in records if not r["parse_ok"]),
                   "think_leaks": sum(1 for r in records if r["think_leak"]),
                   "output_tokens_mean": round(statistics.fmean(
                       r["output_tokens"] for r in records), 1),
                   "eval_seconds": round(self.last_seconds, 1)}
            self.history.append(row)
            with (self.out_dir / "eval_history.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            better = row["exec_match_pct"] > self.best
            print("  [eval] step %-5s exec-match %5.2f%%  valid %5.2f%%  "
                  "%4.1f tok  %3.0f s%s" % (
                      step, row["exec_match_pct"], row["valid_pct"],
                      row["output_tokens_mean"], row["eval_seconds"],
                      "   <- best" if better else ""), flush=True)
            if self.wandb_run is not None:
                # The step goes in as a field, the way transformers' own
                # WandbCallback logs it: an explicit step= would be rejected
                # as out of order if a trainer log landed at a later
                # internal step first.
                self.wandb_run.log({"val/exec_match_pct": row["exec_match_pct"],
                                    "val/valid_pct": row["valid_pct"],
                                    "val/output_tokens_mean": row["output_tokens_mean"],
                                    "val/parse_failures": row["parse_failures"],
                                    "train/global_step": step})
            if better:
                self.best, self.best_step = row["exec_match_pct"], step
                model.save_pretrained(str(self.out_dir / "best"))
                write_base_marker(self.out_dir / "best", self.load_in_4bit)
                (self.out_dir / "best" / "selection.json").write_text(
                    json.dumps({"step": step, "metric": "val exec_match_pct",
                                "value": row["exec_match_pct"], "n": n,
                                "subset_seed": SUBSET_SEED}, indent=1) + "\n",
                    encoding="utf-8")
            return row

    class WallClockGuard(base):
        """CLAUDE.md §14(b): no run past ~12 h without the owner saying so.

        Projected from ten real optimizer steps plus the measured cost of one
        evaluation, because a guess made before the run is not a projection.
        """

        def __init__(self, max_hours, evaluator, eval_every, probe_step=10):
            self.max_hours, self.evaluator = max_hours, evaluator
            self.eval_every, self.probe_step = eval_every, probe_step
            self.t0 = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = time.time()
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step != self.probe_step or not self.max_hours:
                return control
            per_step = (time.time() - self.t0) / max(1, state.global_step)
            evals_left = (state.max_steps // self.eval_every) if self.eval_every else 0
            hours = (per_step * state.max_steps
                     + evals_left * self.evaluator.last_seconds) / 3600
            print("  [guard] %.1f s/step over %d steps + %d evals -> %.1f h projected"
                  % (per_step, state.max_steps, evals_left, hours), flush=True)
            if hours > self.max_hours:
                raise SystemExit(
                    "projected %.1f h exceeds --max-hours %.1f (CLAUDE.md §14b: a run "
                    "over ~12 h needs the owner's sign-off).\nRerun with a higher "
                    "--max-hours, fewer --epochs, or a smaller --eval-subset."
                    % (hours, self.max_hours))
            return control

    return HarnessEval, WallClockGuard


# --------------------------------------------------------------------- main

def resolved_targets(lora):
    """PEFT (parameter-efficient fine-tuning) rewrites `all-linear` into the concrete module names it matched, as
    a set. Record them sorted: the expansion is the interesting fact (which
    layer families got adapters), and a set has no JSON (JavaScript Object Notation) type."""
    targets = lora.target_modules
    return sorted(targets) if isinstance(targets, (set, frozenset)) else targets


def stack_versions() -> dict:
    """The exact versions a number was measured under (CLAUDE.md §14)."""
    import torch
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "vim": vim_version(),
            "gpu": torch.cuda.get_device_name(0)}


def full_val_eval(reg, tok, adapter, val_cases, out_dir, *, load_in_4bit, max_new_tokens,
                  timeout, val_path, run_name, condition, wandb_state, training,
                  summary_name):
    """A training run's headline number: the FULL val slice, with `adapter`
    reloaded on the base it was trained against exactly the way
    `baseline.py --adapter` loads it, so this number and the results table
    cannot diverge. Shared by the SFT and GRPO scripts (the GRPO row is judged
    by the same code, DECISIONS.md 2026-09-12). Writes completions_val.jsonl,
    preds_val.jsonl, exec_val.jsonl and `summary_name` under `out_dir`;
    returns the summary.
    """
    import torch
    from peft import PeftModel
    adapter = pathlib.Path(adapter)
    base = load_base(reg, load_in_4bit)      # the base it was trained on
    final_model = PeftModel.from_pretrained(base, str(adapter))
    final_model.eval()
    print("\nfinal evaluation: full val (%d rows) with %s" % (len(val_cases), adapter.name),
          flush=True)
    comp_path = out_dir / "completions_val.jsonl"
    with comp_path.open("w", encoding="utf-8") as comp:
        records, preds = generate_predictions(
            final_model, tok, val_cases, reg, max_new_tokens,
            lambda rec: comp.write(json.dumps(rec, ensure_ascii=False) + "\n"))
    with (out_dir / "preds_val.jsonl").open("w", encoding="utf-8") as fh:
        for cid in sorted(preds):
            fh.write(json.dumps(preds[cid]) + "\n")
    del final_model, base
    torch.cuda.empty_cache()
    scored = score(val_cases, preds, timeout=timeout)
    write_exec_log(str(out_dir / "exec_val.jsonl"), val_cases, scored)

    summary_args = SimpleNamespace(
        split="val", cases=val_path, max_new_tokens=max_new_tokens,
        wandb_state=wandb_state, run_name=run_name, condition=condition,
        adapter=str(adapter), base_quant=base_quant_name(load_in_4bit))
    summary = summarize(val_cases, records, scored, reg, summary_args, stack_versions())
    summary["training"] = training
    (out_dir / summary_name).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="registry name in configs/models.yaml")
    ap.add_argument("--train", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--val", default=str(REPO / "data" / "splits" / "val.jsonl"))
    ap.add_argument("--out-dir", default=None, help="default: runs/<model>-sft/")
    ap.add_argument("--epochs", type=float, default=3.0, help="§8: 2-3")
    ap.add_argument("--lr", type=float, default=1e-4, help="§8: 1e-4 cosine")
    ap.add_argument("--warmup-steps", type=int, default=None,
                    help="default: 3%% of total steps (transformers 5.x dropped "
                         "warmup_ratio, so this is in steps)")
    ap.add_argument("--batch-size", type=int, default=8, help="micro-batch")
    ap.add_argument("--grad-accum", type=int, default=4, help="8 x 4 = §8's 32")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--max-length", type=int, default=1024,
                    help="§8's ceiling; the longest corpus example is 416 tokens, "
                         "so nothing is truncated and batches pad to their own longest")
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps")
    ap.add_argument("--eval-subset", type=int, default=250,
                    help="category-proportional val sample for mid-training evals; "
                         "0 or >= len(val) uses all of val")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    ap.add_argument("--max-hours", type=float, default=12.0, help="§14b guard; 0 disables")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb", action="store_true", help="log to W&B project vimBot")
    ap.add_argument("--limit", type=int, default=None, help="first N train rows (smoke)")
    ap.add_argument("--max-steps", type=int, default=-1, help="cap optimizer steps (smoke)")
    ap.add_argument("--inspect", type=int, default=0, metavar="N",
                    help="print N rendered examples with the tokens that carry "
                         "loss, then exit without training")
    ap.add_argument("--no-eval-at-start", action="store_true",
                    help="skip the step-0 evaluation (it is the wiring check and "
                         "gives the curve its zero-shot origin)")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA: hold the frozen base in NF4 (4-bit, double-quantized, "
                         "bf16 compute) -- needed for the 9B on 24 GB, not for the 4B")
    return ap


def show_masked_batch(trainer, tok, n):
    """Print what the model is actually trained on: the text, and the tokens
    that carry loss. This is the check that the answer-only mask is right."""
    batch = next(iter(trainer.get_train_dataloader()))
    ids, labels = batch["input_ids"], batch["labels"]
    print("collated batch: input_ids %s, labels %s" % (tuple(ids.shape), tuple(labels.shape)))
    for i in range(min(n, ids.shape[0])):
        keep = labels[i] != -100
        print("\n--- example %d ---" % (i + 1))
        print("full sequence:\n%s" % tok.decode(ids[i][batch["attention_mask"][i] == 1]))
        print("tokens carrying loss (%d of %d): %r" % (
            int(keep.sum()), int((batch["attention_mask"][i] == 1).sum()),
            tok.decode(ids[i][keep])))
    return 0


def main() -> int:
    args = build_arg_parser().parse_args()
    reg = load_registry(args.model)
    args.run_name = reg["name"] + "-sft"
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else REPO / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cases = load_jsonl(args.train)
    val_cases = load_jsonl(args.val)
    if args.limit:
        train_cases = train_cases[:args.limit]
    subset = stratified_subset(val_cases, args.eval_subset)
    print("train %d rows, val %d rows (mid-training evals on %d)"
          % (len(train_cases), len(val_cases), len(subset)))

    import torch
    from transformers import AutoTokenizer, TrainerCallback, set_seed
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = load_base(reg, args.load_in_4bit)
    base_quant = base_quant_name(args.load_in_4bit)
    print("loaded %s @ %s (%s base) in %.0f s on %s" % (
        reg["hf_id"], reg["revision"][:12], base_quant, time.time() - t0,
        torch.cuda.get_device_name(0)), flush=True)

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha,
                      lora_dropout=args.lora_dropout, bias="none",
                      task_type="CAUSAL_LM",
                      target_modules=reg.get("lora_target_modules", "all-linear"))

    steps_per_epoch = max(1, len(train_cases) // (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else int(steps_per_epoch * args.epochs)
    warmup = args.warmup_steps if args.warmup_steps is not None else max(1, round(0.03 * total_steps))

    cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},   # required with PEFT
        max_length=args.max_length,
        packing=False,                     # see decision 5 in the module docstring
        completion_only_loss=True,         # decision 2 -- stated, not inherited
        eval_strategy="no",                # the harness callback is the evaluation
        save_strategy="no",                # the callback saves the adapter it selects
        logging_steps=10,
        report_to="wandb" if args.wandb else "none",
        run_name=args.run_name,
        seed=args.seed,
    )

    train_ds = build_dataset(train_cases, reg)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds,
                         processing_class=tok, peft_config=lora)
    # Adapter weights are fp32 in both paths. PEFT upcasts them on a bf16 base;
    # on a quantized base TRL then casts them back to bf16 (the QLoRA paper's
    # memory saving, ~0.3 GB here). Undone so the 4B and 9B recipes differ in
    # base storage only -- bf16 adapter weights also lose small updates to
    # rounding, which the fp32 path does not.
    for prm in trainer.model.parameters():
        if prm.requires_grad and prm.dtype != torch.float32:
            prm.data = prm.data.float()
    adapter_dtype = str(next(prm.dtype for prm in trainer.model.parameters()
                             if prm.requires_grad))

    if args.inspect:
        return show_masked_batch(trainer, tok, args.inspect)

    wandb_run, wandb_state = None, "off"
    if args.wandb:
        import wandb
        mode = wandb_mode()
        wandb_state = "online" if mode == "online" else "offline (no API key)"
        wandb_run = wandb.init(project="vimBot", name=args.run_name, mode=mode,
                               config={"model": reg["name"], "hf_id": reg["hf_id"],
                                       "revision": reg["revision"],
                                       "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                                       "lr": args.lr, "epochs": args.epochs,
                                       "effective_batch": args.batch_size * args.grad_accum,
                                       "base": base_quant})

    HarnessEval, WallClockGuard = make_callbacks(TrainerCallback)
    evaluator = HarnessEval(tok, reg, subset, out_dir, args.eval_every,
                            args.max_new_tokens, args.timeout, wandb_run,
                            load_in_4bit=args.load_in_4bit)
    trainer.add_callback(evaluator)
    trainer.add_callback(WallClockGuard(args.max_hours, evaluator, args.eval_every))

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    targets = resolved_targets(lora)
    print("LoRA r=%d alpha=%d on %s: %.1f M trainable of %.2f B (%.2f%%)"
          % (args.lora_r, args.lora_alpha,
             ", ".join(targets) if isinstance(targets, list) else targets,
             trainable / 1e6, total / 1e9, 100 * trainable / total))
    print("effective batch %d (%d x %d), %d steps/epoch, ~%d total, warmup %d"
          % (args.batch_size * args.grad_accum, args.batch_size, args.grad_accum,
             steps_per_epoch, total_steps, warmup), flush=True)

    (out_dir / "train_config.json").write_text(json.dumps({
        "model": reg["name"], "hf_id": reg["hf_id"], "revision": reg["revision"],
        "chat_template_kwargs": reg.get("chat_template_kwargs", {}),
        "base": {"load_in_4bit": args.load_in_4bit, "quantization": base_quant,
                 "bnb": NF4 if args.load_in_4bit else None,
                 "adapter_dtype": adapter_dtype},
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout,
                 "target_modules": targets,
                 "trainable_params": trainable, "total_params": total},
        "optimization": {"lr": args.lr, "schedule": "cosine", "warmup_steps": warmup,
                         "epochs": args.epochs, "micro_batch": args.batch_size,
                         "grad_accum": args.grad_accum,
                         "effective_batch": args.batch_size * args.grad_accum,
                         "max_length": args.max_length, "packing": False,
                         "completion_only_loss": True, "seed": args.seed,
                         "optim": cfg.optim},
        "data": {"train": args.train, "train_rows": len(train_cases),
                 "val": args.val, "val_rows": len(val_cases),
                 "eval_subset": len(subset), "subset_seed": SUBSET_SEED},
        "eval": {"every_steps": args.eval_every, "metric": "val exec_match_pct",
                 "max_new_tokens": args.max_new_tokens, "decoding": "greedy",
                 "sandbox_timeout_s": args.timeout},
        "versions": stack_versions(),
        "run": {"date": datetime.date.today().isoformat(), "host": platform.node(),
                "name": args.run_name, "wandb": wandb_state},
    }, indent=1) + "\n", encoding="utf-8")

    if not args.no_eval_at_start:
        # Step 0 is the wiring check and the curve's origin: it should land on
        # the zero-shot number for this subset.
        print("evaluating before training (step 0)...", flush=True)
        evaluator.run(trainer.model, 0)

    trainer.train()
    trainer.model.save_pretrained(str(out_dir / "final"))
    write_base_marker(out_dir / "final", args.load_in_4bit)
    print("saved final adapter to %s" % (out_dir / "final"))

    # Final number on the FULL val slice, with the best adapter loaded the way
    # baseline.py --adapter loads it, so this run's headline and the results
    # table cannot diverge.
    best_dir = out_dir / "best"
    adapter = best_dir if best_dir.exists() else out_dir / "final"
    del trainer, model
    torch.cuda.empty_cache()
    summary = full_val_eval(
        reg, tok, adapter, val_cases, out_dir, load_in_4bit=args.load_in_4bit,
        max_new_tokens=args.max_new_tokens, timeout=args.timeout, val_path=args.val,
        run_name=args.run_name, condition="sft", wandb_state=wandb_state,
        training={"best_step": evaluator.best_step,
                  "best_subset_exec_match_pct": evaluator.best,
                  "eval_history": evaluator.history},
        summary_name="sft_val.json")

    m = summary["metrics"]
    print("\n%s on full val (%d cases)" % (args.run_name, summary["n"]))
    print("  parse failures  %d   think leaks %d" % (m["parse_failures"], m["think_leaks"]))
    print("  valid           %d  (%.1f%%)" % (m["valid"], m["valid_pct"]))
    print("  exec_match      %d  (%.1f%%)" % (m["exec_match"], m["exec_match_pct"]))
    print("  output tokens   mean %.1f  p50 %s" % (m["output_tokens_mean"], m["output_tokens_p50"]))
    if wandb_run is not None:
        wandb_run.log({"final/%s" % k: v for k, v in m.items() if isinstance(v, (int, float))})
        wandb_run.finish()
    print("\nnext: the test row through the same script as every other row\n"
          "  python scripts/baseline.py --model %s --adapter %s" % (reg["name"], adapter))
    return 0


if __name__ == "__main__":
    sys.exit(main())
