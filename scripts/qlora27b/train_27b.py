#!/usr/bin/env python3
"""QLoRA SFT of the 27B on the RTX 3090: scripts/train_sft.py, unchanged, plus two runtime patches.

Why a 27B needs anything at all. Measured 2026-10-31, the NF4 base holds 16.46 GiB, and WSL2
leaves about 22 of the card's 24 GiB: Windows' compositor keeps ~2 GB, and past the rest memory
spills to system RAM at ~10x slower per step instead of failing. Training adds the LoRA adapters
with their Adam state (116.7 M fp32 parameters, 1.7 GiB), the checkpointed activations (~1 GiB)
and the cross-entropy. TRL 1.12's default `chunked_nll` loss runs the 248,320-way LM head on the
answer tokens only, which would make the loss small, but for the transient copies patch 3
removes. With them gone that is ~20.5 GiB with slack: inside the budget, but at the level where
the 27B's sampling run started slowing down (21.3 GiB peak).

Patch 1, the input embedding in system RAM. The input table is 248,320 x 5,120 bf16, 2.37 GiB,
and the model only ever looks rows up in it: no matmul, and no gradient (it is frozen, and LoRA
does not target it). `RamEmbedding` keeps the table on the CPU, looks the rows up there and sends
the result (~10 MB per micro-batch) to the GPU. A lookup is exact, so training and generation are
numerically unchanged. The table is a plain attribute, not a parameter, so `.to()` never moves it
back, and `next(model.parameters()).device` -- which baseline.generate_predictions uses to place
its inputs -- still names the GPU.

Patch 3, the chunked loss without copies of the LM head (found by the first probe, which died at
its first loss). TRL's `_chunk` computes `h.float() @ w.float().t()`: an fp32 copy of the whole
248,320 x 5,120 head (4.74 GiB) for every chunk, forward and checkpointed backward alike, and
under bf16 autocast -- which the Trainer wraps the forward in -- the matmul casts it straight
back to bf16, another 2.37 GiB. So the arithmetic was a bf16 matmul all along, behind ~7 GiB of
transient copies. `lean_chunk` does that bf16 matmul directly and upcasts the [chunk, vocab]
logits afterwards. Measured on the GPU under autocast, loss, correct-token count and entropy are
bit-identical to TRL's.

Patch 2, pace and memory. Every --pace-every steps one line goes to <out-dir>/pace.jsonl: the
median seconds per step over the last 10, peak allocated and reserved memory, free memory. Every
--reproject-every steps, the §14b projection that train_sft.py makes once at step 10 is made
again from the measured pace, so a run that starts spilling to RAM stops at --max-hours instead
of crawling for days. memory.json, written when training ends, holds the peaks.

Everything else -- data, prompt, loss mask, LoRA config, schedule, evaluation, checkpoint
selection, the final full-val pass -- is train_sft.py's code, so the 27B and 4B rows share one
recipe.

    python scripts/qlora27b/train_27b.py --model qwen3.5-27b --load-in-4bit \\
        --train runs/distill_2026-10-31/train_distilled.jsonl \\
        --out-dir runs/qwen3.5-27b-sft-distill --batch-size 4 --grad-accum 8 \\
        --eval-every 204 --no-eval-at-start --wandb
"""
import argparse
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import train_sft                                                  # noqa: E402

GIB = 2 ** 30
LOAD = {}           # filled by the patched loader, written into memory.json


def ram_embedding(emb, out_device):
    """A drop-in for the model's input nn.Embedding whose table lives in system RAM."""
    import torch

    class RamEmbedding(torch.nn.Module):
        # Built from plain values, never from `emb` itself: a method that read `emb` from this
        # enclosing scope would hold the GPU table in its closure for as long as the class
        # lives -- the first probe's 2.37 GiB that never left the card.
        def __init__(self, weight, padding_idx, out_device):
            super().__init__()
            # A plain tensor attribute: not a Parameter or buffer, so .to(), .cuda(),
            # .parameters() and state_dict() all leave it alone.
            self.weight = weight
            self.num_embeddings, self.embedding_dim = weight.shape
            self.padding_idx = padding_idx
            self.out_device = torch.device(out_device)

        def forward(self, input_ids):
            rows = torch.nn.functional.embedding(input_ids.to("cpu"), self.weight, self.padding_idx)
            return rows.to(self.out_device)

        def extra_repr(self):
            return "%d, %d, table on cpu -> %s" % (self.num_embeddings, self.embedding_dim,
                                                 self.out_device)

    return RamEmbedding(emb.weight.detach().to("cpu"), emb.padding_idx, out_device)


def move_input_embedding_to_ram(model):
    """Swap the model's input embedding for a RamEmbedding; returns the GiB it held on its device."""
    import gc
    import torch
    emb = model.get_input_embeddings()
    device = emb.weight.device
    size = emb.weight.numel() * emb.weight.element_size() / GIB
    model.set_input_embeddings(ram_embedding(emb, device))
    del emb
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return size


def patched_load_base(orig):
    def load_base(reg, load_in_4bit=False):
        import gc
        import torch
        # train_sft frees the training model (`del trainer, model`) before its final pass loads
        # the base again, but Trainer and accelerate objects hold reference cycles that only the
        # collector frees. Without this the reload met the old 16 GiB still allocated and failed
        # (WSL2 reports an allocation past the card as "CUDA driver error: device not ready").
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            print("  [qlora27b] loading the base with %.2f GiB still allocated"
                  % (torch.cuda.memory_allocated() / GIB), flush=True)
        model = orig(reg, load_in_4bit)
        on_gpu = model.get_input_embeddings().weight.device.type == "cuda"
        allocated = lambda: torch.cuda.memory_allocated() / GIB if on_gpu else 0.0
        before = allocated()
        freed = move_input_embedding_to_ram(model)
        after = allocated()
        LOAD.update({"allocated_after_load_gib": round(before, 2),
                     "allocated_after_patch_gib": round(after, 2),
                     "embedding_moved_to_ram_gib": round(freed, 2)})
        print("  [qlora27b] input embedding (%.2f GiB) moved to system RAM: allocated %.2f -> "
              "%.2f GiB" % (freed, before, after), flush=True)
        return model
    return load_base


def lean_chunk(S):
    """TRL's `_chunk` (module `S`, trl.trainer.sft_trainer) with the logits from a bf16 matmul
    upcast afterwards, instead of a matmul of fp32 copies (see patch 3 above)."""
    import torch
    import torch.nn.functional as F

    def _chunk(h, w, b, lbl, logit_scale, final_logit_softcapping):
        with S.maybe_gather_lm_head_ctx(w, b):
            logits = (h.to(w.dtype) @ w.t()).float()
            if b is not None:
                logits = logits + b.float()
        if logit_scale != 1.0:
            logits = logits * logit_scale
        if final_logit_softcapping is not None:
            logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
        log_p = F.log_softmax(logits, dim=-1)
        chunk_loss = F.nll_loss(log_p, lbl, ignore_index=-100, reduction="sum")
        valid = lbl != -100
        chunk_correct = ((logits.argmax(dim=-1) == lbl) & valid).sum().float()
        chunk_entropy = (-(log_p.exp() * log_p).sum(dim=-1) * valid).sum()
        return chunk_loss, chunk_correct, chunk_entropy

    _chunk.lean = True
    return _chunk


def projected_hours(elapsed_s, median_step_s, step, max_steps, eval_every, eval_s):
    """Hours from the start of training to its end at the measured pace, with the evaluations
    still to come at the last one's cost (train_sft's WallClockGuard rule, re-made later)."""
    evals_left = 0
    if eval_every:
        evals_left = max_steps // eval_every - step // eval_every
        if max_steps % eval_every:
            evals_left += 1                       # HarnessEval.on_train_end's off-cadence eval
    return (elapsed_s + median_step_s * (max_steps - step) + evals_left * eval_s) / 3600


def patched_make_callbacks(orig, pace_every, reproject_every):
    def make_callbacks(base):
        HarnessEval, WallClockGuard = orig(base)

        class PacedGuard(WallClockGuard):
            """WallClockGuard, plus pace and memory lines and a projection re-made every
            `reproject_every` steps from the median of the last ten step times."""

            def on_train_begin(self, args, state, control, **kwargs):
                import torch
                torch.cuda.reset_peak_memory_stats()
                self.durations, self.last, self.t_begin = [], time.time(), time.time()
                self.out = pathlib.Path(args.output_dir)
                return super().on_train_begin(args, state, control, **kwargs)

            def median_step(self):
                return statistics.median(self.durations[-10:]) if self.durations else 0.0

            def line(self, state):
                import torch
                free, _ = torch.cuda.mem_get_info()
                return {"step": state.global_step, "max_steps": state.max_steps,
                        "median_s_per_step": round(self.median_step(), 2),
                        "elapsed_h": round((time.time() - self.t_begin) / 3600, 3),
                        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 2),
                        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 2),
                        "free_gib": round(free / GIB, 2)}

            def on_step_end(self, args, state, control, **kwargs):
                now = time.time()
                self.durations.append(now - self.last)       # an eval step's outlier is
                self.last = now                              # absorbed by the median
                step = state.global_step
                if step % pace_every == 0 or step == state.max_steps:
                    row = self.line(state)
                    with (self.out / "pace.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(row) + "\n")
                if (self.max_hours and step > self.probe_step and step % reproject_every == 0):
                    hours = projected_hours(now - self.t_begin, self.median_step(), step,
                                            state.max_steps, self.eval_every,
                                            self.evaluator.last_seconds)
                    print("  [pace] step %d: %.1f s/step, %.1f h projected"
                          % (step, self.median_step(), hours), flush=True)
                    if hours > self.max_hours:
                        raise SystemExit(
                            "projected %.1f h at step %d exceeds --max-hours %.1f: the pace fell "
                            "(spilling to RAM?), so the run stops (CLAUDE.md §14b)"
                            % (hours, step, self.max_hours))
                return super().on_step_end(args, state, control, **kwargs)

            def on_train_end(self, args, state, control, **kwargs):
                row = self.line(state)
                import trl.trainer.sft_trainer as S
                row.update({"steps_timed": len(self.durations),
                            "median_s_per_step_all": round(statistics.median(self.durations), 2)
                            if self.durations else None,
                            "ram_embedding": bool(LOAD),
                            "lean_chunk": bool(getattr(S._chunk, "lean", False)), **LOAD})
                (self.out / "memory.json").write_text(json.dumps(row, indent=1) + "\n",
                                                      encoding="utf-8")
                return control

        return HarnessEval, PacedGuard
    return make_callbacks


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Every other flag goes to scripts/train_sft.py unchanged.")
    ap.add_argument("--pace-every", type=int, default=5, help="optimizer steps per pace.jsonl line")
    ap.add_argument("--reproject-every", type=int, default=50,
                    help="optimizer steps between re-made §14b projections")
    ap.add_argument("--embedding-on-gpu", action="store_true",
                    help="leave the input embedding on the GPU (patch 1 off)")
    ap.add_argument("--final-eval-only", action="store_true",
                    help="only the final full-val pass, in this fresh process, for a run whose "
                         "training finished but whose in-process pass did not")
    return ap


def final_eval_only(rest):
    """train_sft's own full_val_eval on the run's best/ adapter (final/ if none), in a fresh
    process, so a failed final pass costs minutes instead of a retrained run. Writes the same
    files the in-process pass would; the training summary is rebuilt from best/selection.json
    and eval_history.jsonl."""
    from transformers import AutoTokenizer
    args = train_sft.build_arg_parser().parse_args(rest)
    reg = train_sft.load_registry(args.model)
    out_dir = pathlib.Path(args.out_dir)
    adapter = out_dir / "best" if (out_dir / "best").exists() else out_dir / "final"
    sel_path, hist_path = adapter / "selection.json", out_dir / "eval_history.jsonl"
    sel = json.loads(sel_path.read_text(encoding="utf-8")) if sel_path.exists() else {}
    history = train_sft.load_jsonl(str(hist_path)) if hist_path.exists() else []
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    summary = train_sft.full_val_eval(
        reg, tok, adapter, train_sft.load_jsonl(args.val), out_dir,
        load_in_4bit=args.load_in_4bit, max_new_tokens=args.max_new_tokens, timeout=args.timeout,
        val_path=args.val, run_name=reg["name"] + "-sft", condition="sft", wandb_state="off",
        training={"best_step": sel.get("step"), "best_subset_exec_match_pct": sel.get("value"),
                  "eval_history": history, "final_pass": "separate process"},
        summary_name="sft_val.json")
    print("full val: %s%% exec-match" % summary["metrics"]["exec_match_pct"], flush=True)
    return 0


def install(own):
    """Apply the patches where they are looked up: train_sft's module namespace for the loader
    and the callbacks, TRL's for the loss chunk (`_chunked_cross_entropy_loss` reads `_chunk`
    from its module globals at call time)."""
    import trl.trainer.sft_trainer as S
    if not own.embedding_on_gpu:
        train_sft.load_base = patched_load_base(train_sft.load_base)
    if not getattr(S._chunk, "lean", False):
        S._chunk = lean_chunk(S)
    train_sft.make_callbacks = patched_make_callbacks(train_sft.make_callbacks,
                                                      own.pace_every, own.reproject_every)


def main(argv=None):
    own, rest = build_arg_parser().parse_known_args(argv)
    install(own)
    if own.final_eval_only:
        return final_eval_only(rest)
    sys.argv = [str(REPO / "scripts" / "train_sft.py")] + rest
    return train_sft.main()


if __name__ == "__main__":
    sys.exit(main())
