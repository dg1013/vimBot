#!/usr/bin/env python3
"""Student zero-shot baseline through the harness (CLAUDE.md §8).

Runs the registry's model (`--model <name>` from configs/models.yaml, never a
hardcoded id) over a case file, one request at a time, and scores the answers
exactly the way training and serving will be scored: the §6 output grammar
decides what counts as an answer, and src/evaluate.py decides valid and
exec-match by running the command in the sandbox and comparing buffers.

Reports the §11 metrics -- valid%, exec-match%, output tokens, latency --
overall and per category, and writes everything keyed by registry name so a
model swap produces an apples-to-apples row:

`generate_predictions` is the repo (repository)'s single generation path -- scripts/train_sft.py
imports it for its val callback, so a training-time score and a results-table
score are computed by the same code. `--adapter` puts a LoRA (low-rank adaptation) adapter on the base
model, which is how the SFT (supervised fine-tuning) and GRPO (Group Relative Policy Optimization) rows are measured through this same script;
`--condition grpo` labels such a row (default `sft`) and names its run directory.

    runs/<model>/completions_<split>.jsonl   every raw completion with its
                                             token count and latency
    runs/<model>/preds_<split>.jsonl         the §5 prediction file (parsed
                                             rows only; a parse failure has
                                             no prediction and scores as one)
    runs/<model>/exec_<split>.jsonl          what vim did with each
                                             prediction: the two verdicts,
                                             return code, error code, stderr
                                             and the buffer it produced
    runs/<model>/baseline_<split>.json       the summary: metrics, per
                                             category, decoding config (configuration), exact
                                             model/library/vim versions

Decoding is greedy and non-thinking. Greedy so the number is reproducible;
the model card's sampling settings are recorded in the registry but not used
here. `max_new_tokens` defaults to 192 because the longest gold answer in the
corpus is 151 tokens (p99 72), so no correct answer can be cut off, while a
model that rambles is capped rather than allowed to run to the context limit.
Any <think> content is a parse failure (§4), so thinking tokens cannot leak
into the token metric.

Latency is per-request wall clock around generate() at batch size 1 -- the
honest number for this transport; the §10 serving numbers come from vLLM.

TORCH_DISABLE_NATIVE_JIT is set before torch is imported: torch 2.13 routes
some CUDA (Compute Unified Device Architecture) matmuls through Triton kernels that need a C compiler at first use,
and this box has none (DECISIONS.md 2026-09-05). The switch is torch's own
and falls back to the standard cuBLAS path -- same math, no JIT (just-in-time).

W&B (Weights & Biases): `--wandb` logs the summary to project vimBot. Without an API (application programming interface) key the run
goes to WANDB_MODE=offline, which is stated in the summary.

Usage:
    python scripts/baseline.py --model qwen3.5-4b
    python scripts/baseline.py --model qwen3.5-4b --limit 5     # smoke
"""
import argparse
import datetime
import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see module docstring

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from evaluate import load_jsonl, score, write_exec_log   # noqa: E402
from prompt import messages_for_case, parse_output  # noqa: E402

REGISTRY = REPO / "configs" / "models.yaml"

# QLoRA (quantized low-rank adaptation) base storage -- CLAUDE.md §4's 9B
# fallback, DECISIONS.md 2026-09-09. NF4 with double quantization and bf16
# compute is the QLoRA paper's recipe. Only the 9B needs it on 24 GB; the 4B
# trains in bf16. A results-table row must say which base it was measured on,
# so train_sft.py records the choice beside every adapter it saves and
# --base-dtype auto reads it back: an adapter is evaluated on the base it was
# trained against unless the caller overrides that deliberately.
NF4 = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
       "bnb_4bit_use_double_quant": True, "bnb_4bit_compute_dtype": "bfloat16"}
BASE_MARKER = "base_quantization.json"


def base_quant_name(load_in_4bit: bool) -> str:
    return "nf4" if load_in_4bit else "bf16"


def bnb_config(load_in_4bit: bool):
    """None for a bf16 base; a BitsAndBytesConfig built from NF4 otherwise."""
    if not load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig
    kw = dict(NF4)
    kw["bnb_4bit_compute_dtype"] = getattr(torch, kw["bnb_4bit_compute_dtype"])
    return BitsAndBytesConfig(**kw)


def load_base(reg: dict, load_in_4bit: bool = False):
    """The one loader for every row: the registry's weights, bf16 on the GPU,
    or held in NF4 when the caller asks for QLoRA storage."""
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        reg["hf_id"], revision=reg["revision"], dtype=torch.bfloat16,
        device_map="cuda", quantization_config=bnb_config(load_in_4bit))


def write_base_marker(adapter_dir, load_in_4bit: bool) -> None:
    """Record beside an adapter which base it was trained against."""
    d = pathlib.Path(adapter_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / BASE_MARKER).write_text(json.dumps(
        {"load_in_4bit": bool(load_in_4bit),
         "quantization": base_quant_name(load_in_4bit),
         "bnb": NF4 if load_in_4bit else None}, indent=1) + "\n", encoding="utf-8")


def resolve_base_quant(adapter_dir) -> bool:
    """True if the adapter was trained on an NF4 base.

    Reads the marker train_sft.py writes beside the adapter, then the run's
    train_config.json one level up, and otherwise assumes bf16 -- the only base
    that existed before the marker did (the 2026-09-09 4B run).
    """
    d = pathlib.Path(adapter_dir)
    marker = d / BASE_MARKER
    if marker.exists():
        return bool(json.loads(marker.read_text(encoding="utf-8")).get("load_in_4bit", False))
    cfg = d.parent / "train_config.json"
    if cfg.exists():
        base = json.loads(cfg.read_text(encoding="utf-8")).get("base") or {}
        return bool(base.get("load_in_4bit", False))
    return False


def load_registry(name: str) -> dict:
    import yaml
    models = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))["models"]
    if name not in models:
        sys.exit("unknown model %r; registry has: %s" % (name, ", ".join(models)))
    return dict(models[name], name=name)


def stop_token_ids(tok, stops) -> list[int]:
    """Registry stop strings that are single tokens, plus EOS."""
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for s in stops or ():
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1 and enc[0] not in ids:
            ids.append(enc[0])
    return ids


def generate_predictions(model, tok, cases, reg, max_new_tokens=192, on_record=None,
                         examples_for=None):
    """Greedy, one case at a time; parsed by the §6 grammar.

    `examples_for(case)`, when given, returns the §5 rows to prepend as k-shot
    pairs for that case (prompt.build_messages); the default is none, so every
    existing caller renders the plain §6 prompt unchanged. Added 2026-10-18 for
    the retrieved-examples experiment (scripts/retrieval/), so that its arms
    and the zero-shot rows share this one generation path.

    This is the single generation path in the repo: the zero-shot baseline
    calls it, and so does the SFT trainer's val callback (scripts/train_sft.py),
    so a training-time number and a results-table number are produced by the
    same code and stay comparable. Batch size is 1 deliberately -- batching
    needs left padding, which perturbs greedy decoding slightly and would make
    the two paths disagree on the margin.

    `on_record(rec)` is called per case, for streaming and progress. Returns
    (records, preds): every completion with its tokens and latency, and the §5
    prediction map with parse failures left out, so they score as failures.
    """
    import torch
    stops = stop_token_ids(tok, reg.get("stop"))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    device = next(model.parameters()).device
    records, preds = [], {}
    for case in cases:
        examples = list(examples_for(case)) if examples_for is not None else []
        msgs = messages_for_case(case, examples)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       **reg.get("chat_template_kwargs", {}))
        enc = tok(text, return_tensors="pt").to(device)
        n_prompt = int(enc["input_ids"].shape[1])
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False, eos_token_id=stops,
                                 pad_token_id=pad)
        latency = time.perf_counter() - t1
        gen = out[0, n_prompt:].tolist()
        if gen and gen[-1] in stops:          # the stop token is not content
            gen = gen[:-1]
        completion = tok.decode(gen, skip_special_tokens=True)
        parsed = parse_output(completion)
        rec = {"id": case["id"], "completion": completion,
               "output_tokens": len(gen), "prompt_tokens": n_prompt,
               "k_examples": len(examples),
               "latency_s": round(latency, 4), "parse_ok": parsed is not None,
               "think_leak": "<think>" in completion or "</think>" in completion,
               "mode": parsed[0] if parsed else None,
               "command": parsed[1] if parsed else None}
        records.append(rec)
        if parsed:
            preds[case["id"]] = {"id": case["id"], "command": parsed[1],
                                 "mode": parsed[0]}
        if on_record is not None:
            on_record(rec)
    return records, preds


def wandb_mode() -> str:
    """'online' when a W&B API key is configured, else 'offline'.

    Pass the result to wandb.init(mode=...). Setting WANDB_MODE in os.environ
    instead does not work here: reading wandb.api.api_key to detect the key
    initializes wandb's settings singleton, after which the environment is
    ignored and init() demands a login. Found 2026-09-09 with wandb 0.28.2: an
    offline baseline run raised UsageError after its measurement was written.
    """
    import wandb
    return "online" if (os.environ.get("WANDB_API_KEY") or wandb.api.api_key) else "offline"

def vim_version() -> str:
    try:
        out = subprocess.run(["vim", "--version"], capture_output=True, text=True,
                             timeout=10).stdout.splitlines()
        return out[0].strip() if out else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def percentile(values, p):
    if not values:
        return None
    vals = sorted(values)
    k = max(0, min(len(vals) - 1, round(p / 100 * (len(vals) - 1))))
    return vals[k]


def summarize(cases, records, scored, reg, args, versions) -> dict:
    """Fold per-row records and the evaluator's result into the §11 metrics."""
    n = len(cases)
    toks = [r["output_tokens"] for r in records]
    lats = [r["latency_s"] for r in records]
    parse_fail = sum(1 for r in records if not r["parse_ok"])
    per_cat = []
    for cat in sorted(scored["per_cat"]):
        t, v, m = scored["per_cat"][cat]
        per_cat.append({"category": cat, "n": t, "valid": v, "match": m,
                        "valid_pct": round(100 * v / t, 1),
                        "match_pct": round(100 * m / t, 1)})
    return {
        "model": reg["name"], "hf_id": reg["hf_id"], "revision": reg["revision"],
        "chat_template_kwargs": reg.get("chat_template_kwargs", {}),
        "split": args.split, "cases": str(args.cases), "n": n,
        "decoding": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens,
                     "batch_size": 1, "dtype": "bfloat16"},
        "metrics": {
            "parse_failures": parse_fail,
            "think_leaks": sum(1 for r in records if r["think_leak"]),
            "valid": scored["n_valid"], "valid_pct": round(100 * scored["n_valid"] / n, 1),
            "exec_match": scored["n_match"],
            "exec_match_pct": round(100 * scored["n_match"] / n, 1),
            "output_tokens_mean": round(statistics.fmean(toks), 1) if toks else None,
            "output_tokens_p50": percentile(toks, 50),
            "latency_s_p50": round(percentile(lats, 50), 3) if lats else None,
            "latency_s_p95": round(percentile(lats, 95), 3) if lats else None,
            "prompt_tokens_mean": round(statistics.fmean(
                r["prompt_tokens"] for r in records), 1) if records else None,
            # §11: local $/token is derived from measured throughput at a
            # stated GPU-hour rate at serving time; not a property of this run
            "cost_per_solved_task_usd": None,
        },
        "per_category": per_cat,
        "versions": versions,
        "run": {"date": datetime.date.today().isoformat(),
                "host": platform.node(),
                "name": args.run_name,
                "condition": args.condition,
                "adapter": args.adapter,
                "base": getattr(args, "base_quant", "bf16"),   # bf16 or nf4 (QLoRA)
                "wandb": args.wandb_state},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="registry name in configs/models.yaml")
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "test.jsonl"))
    ap.add_argument("--out-dir", default=None, help="default: runs/<model>/")
    ap.add_argument("--max-new-tokens", type=int, default=192)
    ap.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    ap.add_argument("--limit", type=int, default=None, help="first N cases only (smoke)")
    ap.add_argument("--wandb", action="store_true", help="log the summary to W&B project vimBot")
    ap.add_argument("--adapter", default=None,
                    help="LoRA adapter directory to load on the base model (an SFT or "
                         "GRPO row); without it this is the zero-shot row")
    ap.add_argument("--condition", default=None,
                    help="what the row is (zero-shot, sft, grpo, ...); recorded in the "
                         "summary and used for the default run name. Default: sft "
                         "when --adapter is given, else zero-shot")
    ap.add_argument("--run-name", default=None,
                    help="artifact/W&B name; default: the registry name, or "
                         "<model>-<condition> when --adapter is given")
    ap.add_argument("--base-dtype", choices=("auto", "bf16", "nf4"), default="auto",
                    help="how the base weights are held: bf16, or nf4 (QLoRA 4-bit "
                         "storage); auto = whatever the adapter was trained on, "
                         "and bf16 without an adapter")
    args = ap.parse_args()
    if args.base_dtype == "auto":
        args.load_in_4bit = resolve_base_quant(args.adapter) if args.adapter else False
    else:
        args.load_in_4bit = args.base_dtype == "nf4"
    args.base_quant = base_quant_name(args.load_in_4bit)

    reg = load_registry(args.model)
    args.split = pathlib.Path(args.cases).stem
    args.condition = args.condition or ("sft" if args.adapter else "zero-shot")
    if not args.run_name:
        args.run_name = reg["name"] + ("-" + args.condition if args.adapter else "")
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else REPO / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]

    import torch
    import transformers
    from transformers import AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = load_base(reg, args.load_in_4bit)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print("loaded %s @ %s (%s base)%s in %.0f s on %s" % (
        reg["hf_id"], reg["revision"][:12], args.base_quant,
        " + adapter %s" % args.adapter if args.adapter else "",
        time.time() - t0, torch.cuda.get_device_name(0)), flush=True)

    comp_path = out_dir / ("completions_%s.jsonl" % args.split)
    with comp_path.open("w", encoding="utf-8") as comp:
        seen = []

        def on_record(rec):
            seen.append(rec)
            comp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if len(seen) % 25 == 0 or len(seen) == len(cases):
                print("  %d/%d  parsed %d  mean %.1f tok  p50 %.2f s" % (
                    len(seen), len(cases), sum(1 for r in seen if r["parse_ok"]),
                    statistics.fmean(r["output_tokens"] for r in seen),
                    percentile([r["latency_s"] for r in seen], 50)), flush=True)

        records, preds = generate_predictions(model, tok, cases, reg,
                                              args.max_new_tokens, on_record)

    preds_path = out_dir / ("preds_%s.jsonl" % args.split)
    with preds_path.open("w", encoding="utf-8") as fh:
        for cid in sorted(preds):
            fh.write(json.dumps(preds[cid]) + "\n")

    del model
    torch.cuda.empty_cache()
    print("scoring %d predictions through the sandbox..." % len(preds), flush=True)
    scored = score(cases, preds, timeout=args.timeout)
    # The sandbox runs just happened; keeping them costs one pass over a dict
    # and saves every consumer (error-analysis page, GRPO reward work) from
    # re-executing the whole split to find out what vim said.
    exec_path = out_dir / ("exec_%s.jsonl" % args.split)
    print("  %d execution records -> %s" % (
        write_exec_log(str(exec_path), cases, scored), exec_path), flush=True)

    versions = {"torch": torch.__version__, "transformers": transformers.__version__,
                "python": platform.python_version(), "vim": vim_version(),
                "gpu": torch.cuda.get_device_name(0)}
    args.wandb_state = "off"
    summary = None
    if args.wandb:
        import wandb
        mode = wandb_mode()
        args.wandb_state = "online" if mode == "online" else "offline (no API key)"
    summary = summarize(cases, records, scored, reg, args, versions)
    (out_dir / ("baseline_%s.json" % args.split)).write_text(
        json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.wandb:
        run = wandb.init(project="vimBot", name="baseline-%s-%s" % (args.run_name, args.split),
                         mode=mode,
                         config={k: summary[k] for k in ("model", "hf_id", "revision",
                                                          "decoding", "split", "n")})
        wandb.log(summary["metrics"])
        run.finish()

    m = summary["metrics"]
    print("\n%s %s (%s base) on %s (%d cases)" % (args.run_name, args.condition,
                                                    args.base_quant, args.split, summary["n"]))
    print("  parse failures  %d   think leaks %d" % (m["parse_failures"], m["think_leaks"]))
    print("  valid           %d  (%.1f%%)" % (m["valid"], m["valid_pct"]))
    print("  exec_match      %d  (%.1f%%)" % (m["exec_match"], m["exec_match_pct"]))
    print("  output tokens   mean %.1f  p50 %s" % (m["output_tokens_mean"], m["output_tokens_p50"]))
    print("  latency         p50 %.2f s  p95 %.2f s" % (m["latency_s_p50"], m["latency_s_p95"]))
    print("\n  %-16s %4s %6s %6s" % ("category", "n", "valid", "match"))
    for row in summary["per_category"]:
        print("  %-16s %4d %6d %6d" % (row["category"], row["n"], row["valid"], row["match"]))
    print("\nRESULTS.md row: | %s %s, %s base | %.1f%% | %.1f%% | %.1f | %.2f s | n/a |" % (
        args.run_name, args.condition, args.base_quant, m["valid_pct"], m["exec_match_pct"],
        m["output_tokens_mean"], m["latency_s_p50"]))
    print("wrote %s, %s, %s" % (comp_path.name, preds_path.name, "baseline_%s.json" % args.split))
    return 0


if __name__ == "__main__":
    sys.exit(main())
