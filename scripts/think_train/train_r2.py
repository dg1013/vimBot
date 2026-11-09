#!/usr/bin/env python3
"""R2's A2 arm: train_sft.py plus the reasoning task, one recorded trace per optimizer step.

scripts/train_sft.py runs unchanged: data, prompt, mask, LoRA, schedule, evaluation, checkpoint
choice and the final full-val pass are its code, so A2 differs from A1 only in the added term. The
rules are in runs/think_train_2026-11-08/preregistration.md.

**The added term.** At the last micro-batch of every optimizer step (the Trainer's sync step), one
trace runs forward and backward. A trace is TB's prompt with its 8 examples, then the thinking
teacher's recorded reasoning and answer, as scripts/think_train/build_r2.py rendered and checked
them through the training-only registry entry qwen3.5-4b-think-train. Its loss is the mean NLL over
the completion tokens, times --trace-weight (1, Hsieh et al.'s default), and its gradient joins the
step's beside the answer batch's (TRL's mean over the batch's answer tokens).
- It is added per step, not as rows, because TRL averages over every token in a batch: one
  6,000-token trace among 32 seventeen-token answers would take over 90% of the gradient.
- Logits are computed only at the completion positions, in checkpointed chunks, from the backbone's
  hidden states, as scripts/kd27b/train_kd.py does. No padding mask is passed: one sequence, causal.
- The traces cycle in an order shuffled by the run's seed, a fresh shuffle per pass.
- At weight 0 no trace runs and the step is TRL's exactly (a test pins it).
- The logged `loss` stays the answer loss; the trace's is `trace_nll`.

**--check** runs one optimizer step whose trace is the longest, then stops: it writes the peak
memory and the trace step's seconds to <out-dir>/check.json and exits before any evaluation.

    python scripts/think_train/train_r2.py --traces runs/think_train_2026-11-08/traces.jsonl \\
        --trace-weight 1 --model qwen3.5-4b --train runs/think_train_2026-11-08/train_a1.jsonl \\
        --out-dir runs/qwen3.5-4b-r2-a2-s42 --eval-every 212 --no-eval-at-start --seed 42 --wandb
"""
import argparse
import hashlib
import json
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import train_sft                                                  # noqa: E402

TRACES = REPO / "runs" / "think_train_2026-11-08" / "traces.jsonl"
STUDENT = "qwen3.5-4b-think-train"
CHUNK = 1024        # completion positions per checkpointed loss chunk
TRACE = {"weight": 1.0, "items": [], "seed": 42, "check": False}     # set by install()


class CheckDone(Exception):
    """Raised after --check's one step; carries what check.json records."""


def load_traces(path):
    items = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if not r["prompt_ids"] or not r["completion_ids"]:
                    raise SystemExit("trace %s is empty" % r["id"])
                items.append((r["id"], r["prompt_ids"], r["completion_ids"]))
    if not items:
        raise SystemExit("no traces in %s" % path)
    return items


def trace_order(n, seed):
    """An endless stream of trace indices: each pass is a fresh shuffle from one seeded generator."""
    rng = random.Random("r2-traces|%d" % seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        yield from perm


def longest_index(items):
    return max(range(len(items)), key=lambda i: len(items[i][1]) + len(items[i][2]))


def _nll_chunk(h, w, y):
    import torch.nn.functional as F
    logits = (h.to(w.dtype) @ w.t()).float()
    return F.cross_entropy(logits, y, reduction="sum")


def trace_nll(inner, prompt_ids, completion_ids, device):
    """(mean NLL over the completion tokens, their count) for one trace, on the unwrapped model."""
    import torch
    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], dtype=torch.long, device=device)
    hidden = inner.base_model(input_ids=ids, use_cache=False).last_hidden_state
    n_p = len(prompt_ids)
    h, y = hidden[0, n_p - 1:-1], ids[0, n_p:]
    w = inner.get_output_embeddings().weight
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, y.numel(), CHUNK):
        total = total + torch.utils.checkpoint.checkpoint(_nll_chunk, h[s:s + CHUNK], w, y[s:s + CHUNK],
                                                          use_reentrant=False)
    return total / y.numel(), y.numel()


def make_trace_trainer(SFTTrainer):
    """SFTTrainer whose sync steps also train on one trace."""
    import torch

    class TraceTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            items = TRACE["items"]
            self._trace_stream = (iter([longest_index(items)]) if TRACE["check"]
                                  else trace_order(len(items), TRACE["seed"]))
            self.traces_run = 0

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            if TRACE["weight"] and self.accelerator.sync_gradients:
                self._trace_step(model)
            return loss

        def _one_trace(self, model, inner, p, c):
            cuda = torch.cuda.is_available()
            t0 = time.time()
            model.train()
            with self.compute_loss_context_manager():
                nll, n = trace_nll(inner, p, c, self.args.device)
                loss = TRACE["weight"] * nll
            self.accelerator.backward(loss)
            if cuda:
                torch.cuda.synchronize()
            return nll, n, time.time() - t0

        def _trace_step(self, model):
            rid, p, c = TRACE["items"][next(self._trace_stream)]
            inner = self.accelerator.unwrap_model(model)
            if hasattr(inner, "get_base_model"):
                inner = inner.get_base_model()
            nll, n, dt = self._one_trace(model, inner, p, c)
            self.traces_run += 1
            m = self._metrics["train"]
            m["trace_nll"].append(float(nll.detach()))
            m["trace_tokens"].append(n)
            if TRACE["check"]:
                # Twice: the first call pays for kernel compilation, which would inflate the projection.
                _, _, warm = self._one_trace(model, inner, p, c)
                cuda = torch.cuda.is_available()
                gib = lambda b: round(b / 2 ** 30, 2)                    # noqa: E731
                raise CheckDone({
                    "trace": rid, "prompt_tokens": len(p), "completion_tokens": len(c), "tokens": len(p) + len(c),
                    "trace_seconds": round(warm, 2), "trace_seconds_first": round(dt, 2),
                    "trace_nll": round(float(nll.detach()), 4),
                    "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()) if cuda else None,
                    "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()) if cuda else None})

    return TraceTrainer


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 epilog="Every other flag goes to scripts/train_sft.py unchanged.")
    ap.add_argument("--traces", default=str(TRACES), help="build_r2.py's traces.jsonl")
    ap.add_argument("--trace-weight", type=float, default=TRACE["weight"],
                    help="weight of the reasoning task's mean NLL; 0 is train_sft.py exactly")
    ap.add_argument("--check", action="store_true",
                    help="one optimizer step with the longest trace, then check.json and exit")
    return ap


def install(own, seed):
    """Put the trace trainer where train_sft.main looks SFTTrainer up (it imports it from `trl` at
    call time), as scripts/kd27b/train_kd.py does."""
    import trl
    TRACE.update(weight=own.trace_weight, items=load_traces(own.traces), seed=seed, check=own.check)
    trl.SFTTrainer = make_trace_trainer(trl.SFTTrainer)


def main(argv=None):
    own, rest = build_arg_parser().parse_known_args(argv)
    if own.trace_weight < 0:
        raise SystemExit("--trace-weight must be >= 0")
    sft = train_sft.build_arg_parser().parse_args(rest)
    from baseline import load_registry
    student, renderer = load_registry(sft.model), load_registry(STUDENT)
    if (student["hf_id"], student["revision"]) != (renderer["hf_id"], renderer["revision"]):
        raise SystemExit("the traces were rendered for %s; --model %s has other weights" % (STUDENT, sft.model))
    install(own, sft.seed)
    out_dir = pathlib.Path(sft.out_dir) if sft.out_dir else REPO / "runs" / (sft.model + "-sft")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trace_config.json").write_text(json.dumps({
        "weight": TRACE["weight"], "traces": own.traces,
        "traces_sha256": hashlib.sha256(pathlib.Path(own.traces).read_bytes()).hexdigest(),
        "n_traces": len(TRACE["items"]), "seed": TRACE["seed"], "chunk": CHUNK, "check": TRACE["check"],
        "rendered_by": STUDENT,
        "loss": "answer batch (TRL) + weight * mean NLL of one trace's completion tokens, once per optimizer step",
        "order": "a fresh seeded shuffle per pass over the traces"}, indent=1) + "\n", encoding="utf-8")
    print("R2 reasoning task: %d traces, weight %g, seed %d%s"
          % (len(TRACE["items"]), TRACE["weight"], TRACE["seed"], ", CHECK (one step, the longest trace)"
             if TRACE["check"] else ""), flush=True)
    sys.argv = [str(REPO / "scripts" / "train_sft.py")] + rest
    try:
        return train_sft.main()
    except CheckDone as done:
        info = dict(done.args[0], micro_batch=sft.batch_size, grad_accum=sft.grad_accum)
        (out_dir / "check.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        print("check: %s" % json.dumps(info), flush=True)
        return 0


if __name__ == "__main__":
    sys.exit(main())
