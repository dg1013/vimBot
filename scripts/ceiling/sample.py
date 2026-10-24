#!/usr/bin/env python3
"""Sample n completions per prompt from one checkpoint of a registry model --
the base weights alone, or a LoRA adapter on them -- run every distinct answer
through the sandboxed vim, and write one row per prompt: how many of the n
samples exec-match and what each distinct answer did. report.py turns the rows
into pass@k and maj@k (README.md beside this file).

It is the instrument of scripts/probe_difficulty.py (the gated round's p̂
probe): the §6 prompt through the registry's chat template, sampling at the
training temperature with the distribution untruncated (top-p 1, top-k off),
verdicts from `evaluate.score`. Three things differ, each because a ceiling
needs it:

  * the base model is a checkpoint too (`--adapter none`): whether the
    pretrained weights hold an answer at all is the first question, and the
    probe requires an adapter;
  * a completion is judged the way the results table judges it -- the §6
    grammar on its first line, whether or not the model then stopped. The
    trainer's stricter reading (a completion that ran into the cap is a
    failure) is kept beside it as `c_strict`;
  * each distinct answer records a hash of the buffer it produced, so answers
    can be grouped by effect as well as by spelling (maj@k by buffer).

One prompt per generate() call with the n samples as num_return_sequences, so
nothing is padded: every sequence in a call shares its prompt. `--chunk` caps
the sequences per call (n = 256 runs as two calls of 128). Each prompt is
seeded from (--seed, its id), so a resumed run draws what an uninterrupted
one would have, up to GPU nondeterminism.

Resumable and pausable like the probe: rows already in --out are skipped; one
SIGTERM/SIGINT or the --stop-file appearing finishes the prompt in flight,
writes PAUSED beside the stop file and exits 3; the same command continues.
`<out>.meta.json` records every launch with its versions, and a relaunch
whose sampling settings differ from the first launch's is refused -- one
samples file is one distribution.

Output row (one per prompt):

    {"id", "category", "tier", "mode", "n", "c", "c_strict", "unparsed",
     "think", "truncated", "unrunnable", "tokens_mean", "tokens_max", "states": {...},
     "distinct", "distinct_buffers",
     "commands": [{"mode", "command", "n", "truncated", "state", "buf"}, ...],
     "checkpoint", "prompt_tokens", "gen_s", "exec_s", "peak_gb",
     "reserved_gb", "time"}

`gen_s` is the wall clock of the prompt's generate() calls, which HF bounds
by the LONGEST of the n samples (finished sequences keep occupying the
batch), so `tokens_max`, not `tokens_mean`, predicts it.

Usage:
    python scripts/ceiling/sample.py --model qwen3.5-4b --adapter none \\
        --label base --out runs/ceiling_2026-09-22/samples_base.jsonl
    python scripts/ceiling/sample.py --model qwen3.5-4b \\
        --adapter runs/qwen3.5-4b-sft/best --label sft --out ...
    python scripts/ceiling/sample.py --model qwen3.5-4b --adapter none \\
        --label smoke --limit 4 --n 16 --out /tmp/x.jsonl            # smoke
"""
import argparse
import collections
import datetime
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py
# 128 sequences of up to 192 new tokens put the card near its 24 GB ceiling,
# where the caching allocator fragments and every step slows several-fold
# (the GRPO chain's stage C, RESULTS.md 2026-09-17). Expandable segments
# avoid it; set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import _norm, load_jsonl, score                      # noqa: E402
from prompt import messages_for_case, parse_output                 # noqa: E402
from baseline import (base_quant_name, load_base, load_registry,   # noqa: E402
                      resolve_base_quant, stop_token_ids)
from probe_difficulty import PAUSED_RC, PauseRequest, already_probed   # noqa: E402
from train_grpo import TIMEOUT_RETRIES                              # noqa: E402

# The settings that define the distribution being measured. A relaunch onto
# an existing samples file must match them exactly.
SAMPLING_KEYS = ("model", "adapter", "cases", "n", "temperature", "top_p", "top_k",
                 "max_new_tokens", "base_dtype", "seed")


def buffer_hash(buf):
    """A short digest of a buffer under the exec-match normalization, so two
    answers share a hash exactly when exec-match would call their buffers
    equal. JSON of the line list, not the joined text: [] and [""] differ."""
    lines = _norm(buf)
    if lines is None:
        return None
    return hashlib.sha1(json.dumps(lines).encode("utf-8")).hexdigest()[:16]


def unrunnable(mode, command):
    """An ex command holding a NUL byte, which the harness cannot run as
    given. `run_vim` passes ex commands in argv, and argv cannot carry a NUL
    (execve takes NUL-terminated strings; subprocess raises ValueError rather
    than truncate) -- the base model sampled `2\\x00a` at T = 1 on 2026-09-22
    and crashed the first launch. Vim itself can hold one (keys-mode
    `:s/foo/<C-v><C-@>/` substitutes a NUL), but no other channel reproduces
    the command either: a sourced script cuts the line at the NUL, so
    `s/foo/<NUL>/` would run as `s/foo/` and exit 0. The verdict is invalid,
    without running it. Keys mode is unaffected: keys go through a file,
    and there a NUL is a key vim accepts (insert-mode CTRL-@). Since the
    same day `run_vim` returns this verdict itself (DECISIONS.md
    2026-09-22); the check stays for the row's `unrunnable` count."""
    return mode == "ex" and "\x00" in command


def execute(case, mode, command, timeout, retries=TIMEOUT_RETRIES, scorer=score):
    """(state, buffer hash) for one answer: `train_grpo.execution_state` plus
    the buffer. Through `evaluate.score`, so "match" and "valid" mean exactly
    what they mean in the results table, and a timeout is retried the way the
    trainer retries it. An invalid answer has no hash -- it abstains from the
    vote by buffer."""
    if unrunnable(mode, command):
        return "invalid", None
    pred = {case["id"]: {"command": command, "mode": mode}}
    for _ in range(retries + 1):
        entry = scorer([case], pred, timeout=timeout)["per_case"][case["id"]]
        if not entry["timeout"]:
            break
    if entry["match"]:
        return "match", buffer_hash(entry["buffer_after"])
    if entry["valid"]:
        return "valid_wrong", buffer_hash(entry["buffer_after"])
    return "invalid", None


def cut(ids, stops, pad_id):
    """(content ids, terminated): the generated ids before the first stop
    token, or -- when none came -- every non-pad id, a completion that ran
    into the cap. The stop token is not content and is not counted (§11)."""
    for i, t in enumerate(ids):
        if t in stops:
            return ids[:i], True
    return [t for t in ids if t != pad_id], False


def observe(case, samples, run):
    """One prompt's row from its samples, a list of (text, tokens, terminated).

    `run(keys)` executes the distinct (mode, command) pairs and returns
    {key: (state, buf)}; each distinct answer runs once however often it was
    sampled. States are the trainer's (`train_grpo.REWARD_STATES`) with the
    results table's reading of a capped completion: it is judged by its first
    line like any other, and `truncated` / `c_strict` say how many ran on."""
    parsed = [parse_output(text) for text, _, _ in samples]
    verdict = run(sorted({p for p in parsed if p is not None}))
    per_cmd = {}
    states = collections.Counter()
    c = c_strict = unparsed = truncated = think = nul = 0
    for (text, _, terminated), p in zip(samples, parsed):
        truncated += not terminated
        think += "<think>" in text or "</think>" in text
        if p is None:
            unparsed += 1
            states["parse"] += 1
            continue
        nul += unrunnable(*p)
        state, buf = verdict[p]
        states[state] += 1
        if state == "match":
            c += 1
            c_strict += terminated
        e = per_cmd.setdefault(p, {"mode": p[0], "command": p[1], "n": 0, "truncated": 0,
                                   "state": state, "buf": buf})
        e["n"] += 1
        e["truncated"] += not terminated
    commands = sorted(per_cmd.values(), key=lambda e: (-e["n"], e["mode"], e["command"]))
    tokens = [t for _, t, _ in samples]
    return {"id": case["id"], "category": case.get("category"), "tier": case.get("tier"),
            "mode": case.get("mode"), "n": len(samples), "c": c, "c_strict": c_strict,
            "unparsed": unparsed, "think": think, "truncated": truncated, "unrunnable": nul,
            "tokens_mean": round(sum(tokens) / len(tokens), 2) if tokens else None,
            "tokens_max": max(tokens) if tokens else None,
            "states": dict(states), "distinct": len(commands),
            "distinct_buffers": len({e["buf"] for e in commands if e["buf"] is not None}),
            "commands": commands}


def prompt_seed(seed, case_id):
    return (seed * 1_000_003 + zlib.crc32(case_id.encode("utf-8"))) % (2 ** 32)


def sampling_config(args, adapter):
    return {"model": args.model, "adapter": str(adapter) if adapter else None,
            "cases": str(args.cases), "n": args.n, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
            "base_dtype": args.base_dtype, "seed": args.seed}


def check_meta(meta_path, config):
    """Refuse to append samples from a different distribution to a file."""
    if not meta_path.exists():
        return {"sampling": config, "launches": []}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    first = meta.get("sampling", {})
    diff = {k: (first.get(k), config.get(k)) for k in SAMPLING_KEYS if first.get(k) != config.get(k)}
    if diff:
        sys.exit("refusing to append to %s: sampling differs from its first launch %s"
                 % (meta_path, json.dumps(diff)))
    return meta


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="registry name in configs/models.yaml")
    ap.add_argument("--adapter", required=True,
                    help="LoRA adapter directory, or 'none' for the base weights alone")
    ap.add_argument("--label", required=True,
                    help="checkpoint name carried in every row (base, sft, served, ...)")
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "test.jsonl"))
    ap.add_argument("--out", required=True, help="samples file (JSONL, appended; resumable)")
    ap.add_argument("--ids", default=None, help="file with one case id per line: sample only these")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    ap.add_argument("--n", type=int, default=128, help="samples per prompt")
    ap.add_argument("--chunk", type=int, default=128, help="sequences per generate() call")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="default 1.0, the GRPO and probe temperature: the distribution RL samples")
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=0, help="0 = off")
    ap.add_argument("--max-new-tokens", type=int, default=192,
                    help="the results table's cap (the longest gold answer is 151 tokens)")
    ap.add_argument("--base-dtype", choices=("auto", "bf16", "nf4"), default="auto",
                    help="auto: the base the adapter was trained against (bf16 for --adapter none)")
    ap.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    ap.add_argument("--workers", type=int, default=8, help="executor threads")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stop-file", default=None,
                    help="pause when this file appears (it is consumed); PAUSED is written beside it")
    return ap


def versions():
    """CLAUDE.md §14: the exact stack a number was measured under."""
    from train_sft import stack_versions
    v = stack_versions()
    for dist in ("peft", "flash-linear-attention"):
        try:
            v[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            v[dist] = None
    return v


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.n < 1 or args.chunk < 1:
        sys.exit("--n and --chunk must be positive")
    reg = load_registry(args.model)
    adapter = None if args.adapter.lower() == "none" else pathlib.Path(args.adapter)
    if adapter is not None and not (adapter / "adapter_config.json").exists():
        sys.exit("no adapter at %s" % adapter)
    if args.base_dtype == "auto":
        load_in_4bit = resolve_base_quant(adapter) if adapter is not None else False
    else:
        load_in_4bit = args.base_dtype == "nf4"

    cases = load_jsonl(args.cases)
    if args.ids:
        wanted = {l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()}
        cases = [c for c in cases if c["id"] in wanted]
    if args.limit:
        cases = cases[:args.limit]
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.with_name(out_path.name + ".meta.json")
    meta = check_meta(meta_path, sampling_config(args, adapter))
    done = already_probed(out_path)
    todo = [c for c in cases if c["id"] not in done]
    pause = PauseRequest(args.stop_file)
    if pause.marker() is not None:
        pause.marker().unlink(missing_ok=True)          # a new attempt starts clean
    print("ceiling sample [%s]: %d prompts requested, %d already in %s, %d to do; n=%d, "
          "T=%.2f top-p %.2f top-k %d, cap %d"
          % (args.label, len(cases), len(cases) - len(todo), out_path, len(todo), args.n,
             args.temperature, args.top_p, args.top_k, args.max_new_tokens), flush=True)
    if not todo:
        return 0

    import torch
    from transformers import AutoTokenizer, set_seed

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = load_base(reg, load_in_4bit)
    if adapter is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    print("loaded %s @ %s (%s base)%s in %.0f s" % (
        reg["hf_id"], reg["revision"][:12], base_quant_name(load_in_4bit),
        " + %s" % adapter if adapter else ", no adapter", time.time() - t0), flush=True)
    stops = stop_token_ids(tok, reg.get("stop"))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    kwargs = reg.get("chat_template_kwargs") or {}
    device = next(model.parameters()).device

    launch = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
              "label": args.label, "argv": sys.argv[1:], "rows_before": len(done),
              "base": base_quant_name(load_in_4bit), "hf_id": reg["hf_id"],
              "revision": reg["revision"], "stop_ids": stops, "versions": versions(),
              "generation_config": model.generation_config.to_diff_dict()}
    meta["launches"].append(launch)
    meta_path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")

    tally = collections.Counter()
    t_start = time.time()
    pause.install()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            out_path.open("a", encoding="utf-8") as fh:

        def runner(case):
            def run(keys):
                results = pool.map(lambda key: execute(case, key[0], key[1], args.timeout), keys)
                return dict(zip(keys, results))
            return run

        for case in todo:
            if pause.check() is not None:
                break
            set_seed(prompt_seed(args.seed, case["id"]))
            text = tok.apply_chat_template(messages_for_case(case), add_generation_prompt=True,
                                           tokenize=False, **kwargs)
            enc = tok(text, return_tensors="pt").to(device)
            width = enc["input_ids"].shape[1]
            samples, left = [], args.n
            torch.cuda.reset_peak_memory_stats()
            t_gen = time.time()
            while left:
                m = min(args.chunk, left)
                with torch.no_grad():
                    out = model.generate(**enc, do_sample=True, num_return_sequences=m,
                                         temperature=args.temperature, top_p=args.top_p,
                                         top_k=args.top_k, max_new_tokens=args.max_new_tokens,
                                         eos_token_id=stops, pad_token_id=pad_id)
                for ids in out[:, width:].tolist():
                    content, terminated = cut(ids, stops, pad_id)
                    samples.append((tok.decode(content, skip_special_tokens=True),
                                    len(content), terminated))
                left -= m
            t_exec = time.time()
            row = observe(case, samples, runner(case))
            row.update({"checkpoint": args.label, "prompt_tokens": width,
                        "gen_s": round(t_exec - t_gen, 2), "exec_s": round(time.time() - t_exec, 2),
                        "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                        "reserved_gb": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2),
                        "time": datetime.datetime.now().isoformat(timespec="seconds")})
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            # Hand the prompt's cache back. The allocator otherwise keeps its
            # high-water mark for good: one long prompt (328 tokens x 128
            # copies, 2026-09-23) left 23.4 GB reserved, which with the
            # Windows compositor's 2.1 GB overfilled the 24 GB card, and WSL2
            # paged to system RAM at a tenth of the speed for every prompt
            # after it.
            torch.cuda.empty_cache()
            n_done += 1
            tally["c"] += row["c"]
            tally["n"] += row["n"]
            tally["covered"] += row["c"] > 0
            tally["truncated"] += row["truncated"]
            elapsed = time.time() - t_start
            print("  %4d / %d  %-24s %3d/%-3d  | pass@1 %.3f  covered %d/%d  capped %d"
                  "  | %.0f s  eta %.0f min"
                  % (n_done, len(todo), case["id"][:24], row["c"], row["n"],
                     tally["c"] / tally["n"], tally["covered"], n_done, tally["truncated"],
                     elapsed, elapsed / n_done * (len(todo) - n_done) / 60), flush=True)

    if pause.check() is not None and n_done < len(todo):
        info = {"reason": pause.requested, "rows_done": len(done) + n_done,
                "rows_total": len(cases), "out": str(out_path),
                "time": datetime.datetime.now().isoformat(timespec="seconds")}
        if pause.marker() is not None:
            pause.marker().write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
        print("\npaused (%s) after %d of %d prompts; %d rows in %s. Relaunch the same "
              "command to continue." % (pause.requested, n_done, len(todo), info["rows_done"],
                                        out_path), flush=True)
        return PAUSED_RC
    print("\ndone [%s]: %d prompts in %.0f s -> %s\n  pass@1 %.3f  covered (c > 0) %d of %d"
          "  capped completions %d of %d"
          % (args.label, n_done, time.time() - t_start, out_path, tally["c"] / tally["n"],
             tally["covered"], n_done, tally["truncated"], tally["n"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
