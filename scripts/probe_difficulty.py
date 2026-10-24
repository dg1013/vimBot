#!/usr/bin/env python3
"""Measure p̂ for every train prompt: sample G completions per prompt from a
given adapter at the training temperature, score each in the sandboxed vim
with the trainer's own reward states, and write one observation per prompt
(`k` matches of `G`) as a difficulty file for `select_frontier.py` and
`train_grpo_gated.py --difficulty`.

This is the offline pre-pass of difficulty-gated GRPO (DECISIONS.md
2026-09-14): the same generation call shape the trainer uses (G samples of one
prompt, several prompts per call, temperature 1.0, top-p 1, top-k off) and
the same verdicts (`train_grpo.classify` + `train_grpo.execution_state`, so
"match" here is "match" in the results table). Nothing about the gold command
is read: the probe sees the buffers, exactly as the reward does.

Cost on the 4B, bf16: 64 completions per call take 4.6-8.7 s, so the 6,551
train prompts at G = 8 are about 1.6 h with 8 prompts per call; 16 per call
(the default) is faster on this card because the calls are bounded by the
longest sample, not by the batch. Resumable: rows already in --out are
skipped, so a killed probe is relaunched with the same command.

Pausable like the trainer: one SIGTERM/SIGINT, or the file named by
--stop-file appearing, makes the probe finish the call in flight, write its
rows, leave a PAUSED marker beside the stop file and exit 3; a second signal
aborts at once. The same command later continues from the rows on disk.

Output row (one per prompt; `select_frontier.py` reads k, G, step, source):

    {"id", "category", "tier", "mode", "k", "G", "p_hat", "states": {...},
     "distinct", "commands": [{"mode", "command", "state", "n"}, ...],
     "step": 0, "source": "probe", "adapter", "temperature", "time"}

Usage:
    python scripts/probe_difficulty.py --model qwen3.5-4b \\
        --adapter runs/qwen3.5-4b-grpo/best \\
        --out runs/grpo_gated_2026-09-14/difficulty_probe.jsonl
    python scripts/probe_difficulty.py --model qwen3.5-4b --adapter ... \\
        --ids runs/.../frontier_ids.txt --out ...        # re-probe a subset
    python scripts/probe_difficulty.py --model qwen3.5-4b --limit 16 --out /tmp/x.jsonl   # smoke
"""
import argparse
import collections
import datetime
import json
import os
import pathlib
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl                                    # noqa: E402
from prompt import messages_for_case                              # noqa: E402
from baseline import (base_quant_name, load_base, load_registry,  # noqa: E402
                      resolve_base_quant)
from train_grpo import classify, execution_state                  # noqa: E402
from train_sft import inference_mode                              # noqa: E402
from difficulty import fuel, shelf_of                             # noqa: E402

PAUSED_RC = 3
PAUSED_MARKER = "PAUSED"


class PauseRequest:
    """One termination signal or the stop file = "finish this call, then
    stop"; a second signal aborts. `check()` is polled between calls."""

    def __init__(self, stop_file=None):
        self.stop_file = pathlib.Path(stop_file) if stop_file else None
        self.requested = None
        self._previous = {}

    def install(self):
        def handler(signum, frame):
            if self.requested is None:
                self.requested = signal.Signals(signum).name
                print("\n  [pause] %s received: finishing this call, then stopping "
                      "(send it again to abort)" % self.requested, flush=True)
                return
            signal.signal(signum, self._previous.get(signum, signal.SIG_DFL))
            raise KeyboardInterrupt
        for sig in (signal.SIGTERM, signal.SIGINT):
            self._previous[sig] = signal.signal(sig, handler)

    def check(self):
        if self.requested is None and self.stop_file is not None and self.stop_file.exists():
            self.requested = "STOP file"
            self.stop_file.unlink()
        return self.requested

    def marker(self):
        """Where the PAUSED marker goes: beside the stop file, if any."""
        if self.stop_file is None:
            return None
        return self.stop_file.with_name(PAUSED_MARKER)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="registry name in configs/models.yaml")
    ap.add_argument("--adapter", default=None,
                    help="adapter to measure (default runs/<model>-sft/best; the next GRPO "
                         "round continues from runs/<model>-grpo/best, so pass that)")
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--out", required=True, help="difficulty file (JSONL, appended; resumable)")
    ap.add_argument("--ids", default=None,
                    help="file with one case id per line: probe only these (a refresh)")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    ap.add_argument("--num-generations", type=int, default=8, help="G, as in training")
    ap.add_argument("--prompts-per-call", type=int, default=16,
                    help="prompts per generate() call (x G completions)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-completion-length", type=int, default=192)
    ap.add_argument("--base-dtype", choices=("auto", "bf16", "nf4"), default="auto",
                    help="auto: the base the adapter was trained against")
    ap.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    ap.add_argument("--reward-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop-file", default=None,
                    help="pause when this file appears (it is consumed); PAUSED is written beside it")
    return ap


def already_probed(path):
    ids = set()
    p = pathlib.Path(path)
    if not p.exists():
        return ids
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
    return ids


def trim_completion(row, eos_id, pad_id):
    """What the trainer's reward sees: the generated ids up to and including
    the end-of-turn token, or every non-pad token if it never came."""
    if eos_id in row:
        return row[:row.index(eos_id) + 1]
    return [t for t in row if t != pad_id]


def observe(case, groups_ids, tok, eos_id, timeout, pool):
    """Score one prompt's G completions. Returns the probe row (without the
    provenance fields the caller adds)."""
    states, cmds, tokens = [], [], []
    jobs = {}
    for i, ids in enumerate(groups_ids):
        text = tok.decode(ids[:-1] if ids and ids[-1] == eos_id else ids, skip_special_tokens=True)
        state, mode, command, ntok = classify(text, ids, eos_id)
        states.append(state)
        cmds.append((mode, command))
        tokens.append(ntok)
        if state is None:
            jobs.setdefault((mode, command), []).append(i)

    def run(key):
        c = {"id": case["id"], "buffer_before": case["buffer_before"],
             "buffer_after": case["buffer_after"], "cursor": list(case.get("cursor", [1, 1]))}
        return key, execution_state(c, key[0], key[1], timeout)

    if jobs:
        for key, state in pool.map(run, list(jobs)):
            for i in jobs[key]:
                states[i] = state
    counts = collections.Counter(states)
    k, g = counts.get("match", 0), len(states)
    by_cmd = collections.Counter((m, c, s) for (m, c), s in zip(cmds, states))
    return {"id": case["id"], "category": case.get("category"), "tier": case.get("tier"),
            "mode": case.get("mode"), "k": k, "G": g, "p_hat": round(k / g, 4) if g else None,
            "shelf": shelf_of(k, g), "fuel": round(fuel(k, g), 4),
            "states": dict(counts), "distinct": len(by_cmd),
            "tokens_mean": round(sum(tokens) / len(tokens), 2) if tokens else None,
            "commands": [{"mode": m, "command": c, "state": s, "n": n}
                         for (m, c, s), n in by_cmd.most_common()]}


def main() -> int:
    args = build_arg_parser().parse_args()
    reg = load_registry(args.model)
    adapter = pathlib.Path(args.adapter) if args.adapter else REPO / "runs" / (reg["name"] + "-sft") / "best"
    if not (adapter / "adapter_config.json").exists():
        sys.exit("no adapter at %s" % adapter)
    load_in_4bit = resolve_base_quant(adapter) if args.base_dtype == "auto" else args.base_dtype == "nf4"

    cases = load_jsonl(args.cases)
    if args.ids:
        wanted = {l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]
    done = already_probed(args.out)
    todo = [c for c in cases if c["id"] not in done]
    pause = PauseRequest(args.stop_file)
    if pause.marker() is not None:
        pause.marker().unlink(missing_ok=True)          # a new attempt starts clean
    print("probe: %d prompts requested, %d already in %s, %d to do; G=%d, %d prompts per call, T=%.2f"
          % (len(cases), len(cases) - len(todo), args.out, len(todo), args.num_generations,
             args.prompts_per_call, args.temperature), flush=True)
    if not todo:
        return 0

    import torch
    from transformers import AutoTokenizer, set_seed
    from peft import PeftModel

    set_seed(args.seed)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    tok.padding_side = "left"
    base = load_base(reg, load_in_4bit)
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model.eval()
    print("loaded %s @ %s (%s base) + %s in %.0f s" % (
        reg["hf_id"], reg["revision"][:12], base_quant_name(load_in_4bit), adapter,
        time.time() - t0), flush=True)
    kwargs = reg.get("chat_template_kwargs") or {}
    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id
    device = next(model.parameters()).device
    G = args.num_generations
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    tally = collections.Counter()
    t_start = time.time()
    pause.install()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.reward_workers) as pool, \
            out_path.open("a", encoding="utf-8") as fh:
        for start in range(0, len(todo), args.prompts_per_call):
            if pause.check() is not None:
                break
            batch = todo[start:start + args.prompts_per_call]
            texts = [tok.apply_chat_template(messages_for_case(c), add_generation_prompt=True,
                                             tokenize=False, **kwargs) for c in batch]
            enc = tok([t for t in texts for _ in range(G)], return_tensors="pt", padding=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            with inference_mode(model), torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_completion_length, do_sample=True,
                                     temperature=args.temperature, top_p=1.0, top_k=0,
                                     eos_token_id=eos_id, pad_token_id=pad_id)
            gen = out[:, enc["input_ids"].shape[1]:].tolist()
            for j, case in enumerate(batch):
                ids = [trim_completion(row, eos_id, pad_id) for row in gen[j * G:(j + 1) * G]]
                row = observe(case, ids, tok, eos_id, args.timeout, pool)
                row.update({"step": 0, "source": "probe", "adapter": str(adapter),
                            "temperature": args.temperature, "time": stamp})
                fh.write(json.dumps(row) + "\n")
                tally[row["shelf"]] += 1
                tally["fuel"] += row["fuel"]
            fh.flush()
            n_done = start + len(batch)
            elapsed = time.time() - t_start
            print("  %5d / %d  %.0f s  eta %.0f min  | cliff %d  frontier %d  known %d  | mean fuel %.2f"
                  % (n_done, len(todo), elapsed, elapsed / n_done * (len(todo) - n_done) / 60,
                     tally["cliff"], tally["frontier"], tally["known"],
                     tally["fuel"] / n_done), flush=True)
    if pause.check() is not None and n_done < len(todo):
        info = {"reason": pause.requested, "rows_done": len(done) + n_done, "rows_total": len(cases),
                "out": str(out_path), "time": datetime.datetime.now().isoformat(timespec="seconds")}
        if pause.marker() is not None:
            pause.marker().write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        print("\npaused (%s) after %d of %d prompts; %d rows in %s. Relaunch the same command to continue."
              % (pause.requested, n_done, len(todo), info["rows_done"], out_path), flush=True)
        return PAUSED_RC
    n = len(todo)
    print("\nprobe done: %d prompts in %.0f s -> %s\n  cliff %d (%.0f%%)  frontier %d (%.0f%%)  known %d (%.0f%%)"
          "\n  mean fuel k(G-k)/G %.2f (a prompt at p=1/2 expects %.2f)"
          % (n, time.time() - t_start, out_path, tally["cliff"], 100 * tally["cliff"] / n,
             tally["frontier"], 100 * tally["frontier"] / n, tally["known"], 100 * tally["known"] / n,
             tally["fuel"] / n, (G - 1) / 4))
    return 0


if __name__ == "__main__":
    sys.exit(main())
