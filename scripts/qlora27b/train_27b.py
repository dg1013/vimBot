#!/usr/bin/env python3
"""QLoRA SFT of the 27B on the RTX 3090: scripts/train_sft.py, unchanged, plus two runtime patches.

Why a 27B needs anything at all. Measured 2026-10-31, the NF4 base holds 16.46 GiB, and WSL2
leaves about 22 of the card's 24 GiB: Windows' compositor keeps ~2 GB, and past the rest memory
spills to system RAM at ~10x slower per step instead of failing. Training adds the LoRA adapters
with their Adam state (116.7 M fp32 parameters, 1.7 GiB), the checkpointed activations (~1 GiB)
and the cross-entropy, which is small because TRL 1.12's default `chunked_nll` loss runs the
248,320-way LM head on the answer tokens only. That is ~20.5 GiB with slack: inside the budget,
but at the level where the 27B's sampling run started slowing down (21.3 GiB peak).

Patch 1, the input embedding in system RAM. The input table is 248,320 x 5,120 bf16, 2.37 GiB,
and the model only ever looks rows up in it: no matmul, and no gradient (it is frozen, and LoRA
does not target it). `RamEmbedding` keeps the table on the CPU, looks the rows up there and sends
the result (~10 MB per micro-batch) to the GPU. A lookup is exact, so training and generation are
numerically unchanged. The table is a plain attribute, not a parameter, so `.to()` never moves it
back, and `next(model.parameters()).device` -- which baseline.generate_predictions uses to place
its inputs -- still names the GPU.

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
        def __init__(self):
            super().__init__()
            # A plain tensor attribute: not a Parameter or buffer, so .to(), .cuda(),
            # .parameters() and state_dict() all leave it alone.
            self.weight = emb.weight.detach().to("cpu")
            self.num_embeddings, self.embedding_dim = emb.num_embeddings, emb.embedding_dim
            self.padding_idx = emb.padding_idx
            self.out_device = torch.device(out_device)

        def forward(self, input_ids):
            rows = torch.nn.functional.embedding(input_ids.to("cpu"), self.weight, self.padding_idx)
            return rows.to(self.out_device)

        def extra_repr(self):
            return "%d, %d, table on cpu -> %s" % (self.num_embeddings, self.embedding_dim,
                                                 self.out_device)

    return RamEmbedding()


def move_input_embedding_to_ram(model):
    """Swap the model's input embedding for a RamEmbedding; returns the GiB freed on the GPU."""
    import torch
    emb = model.get_input_embeddings()
    device = emb.weight.device
    freed = emb.weight.numel() * emb.weight.element_size() / GIB
    model.set_input_embeddings(ram_embedding(emb, device))
    del emb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return freed


def patched_load_base(orig):
    def load_base(reg, load_in_4bit=False):
        import torch
        model = orig(reg, load_in_4bit)
        on_gpu = model.get_input_embeddings().weight.device.type == "cuda"
        allocated = lambda: torch.cuda.memory_allocated() / GIB if on_gpu else 0.0
        before = allocated()
        freed = move_input_embedding_to_ram(model)
        after = allocated()
        LOAD.update({"allocated_after_load_gib": round(before, 2),
                     "allocated_after_patch_gib": round(after, 2),
                     "embedding_moved_to_ram_gib": round(freed, 2)})
        print("  [qlora27b] input embedding moved to system RAM: %.2f GiB freed, %.2f GiB allocated"
              % (freed, after), flush=True)
        return model
    return load_base


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
                row.update({"steps_timed": len(self.durations),
                            "median_s_per_step_all": round(statistics.median(self.durations), 2)
                            if self.durations else None,
                            "ram_embedding": bool(LOAD), **LOAD})
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
    return ap


def install(own):
    """Apply the patches to train_sft's module namespace, where main() looks them up."""
    if not own.embedding_on_gpu:
        train_sft.load_base = patched_load_base(train_sft.load_base)
    train_sft.make_callbacks = patched_make_callbacks(train_sft.make_callbacks,
                                                      own.pace_every, own.reproject_every)


def main(argv=None):
    own, rest = build_arg_parser().parse_known_args(argv)
    install(own)
    sys.argv = [str(REPO / "scripts" / "train_sft.py")] + rest
    return train_sft.main()


if __name__ == "__main__":
    sys.exit(main())
