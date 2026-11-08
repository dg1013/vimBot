#!/usr/bin/env python3
"""The prompted Qwen3.5-27B on val: is an untrained 27B, given retrieved
examples, a better teacher than the trained 4B and 9B?

The 27B (registry `qwen3.5-27b`) never trains here: 55.6 GB in bf16 does not
fit beside training on 24 GB, so each process loads it once in NF4 through
`baseline.load_base`, the loader of every results-table row. It runs the 545
val rows on the retrieval experiment's val plans (BM25 over train
instructions, `runs/retrieval_2026-10-18/val_plans/`; the 27B's tokenizer is
byte-identical to the 4B's the plans were budgeted with):

  preflight  CPU: the registry entry, the weights in the HF cache at the
             pinned revision, the tokenizer identity, the plans against the
             current splits, the comparators; copies the plans into the run
  greedy     arms through `retrieval/run_arms.run_arm`, i.e.
             `baseline.generate_predictions`: greedy, one row at a time, the
             path the base 4B's val arms took on the same plans
  samples    8 samples per row at T = 1 on the k = 8 prompts, scored and
             written by `probe_difficulty.observe`: the settings and the row
             format of the 4B and 9B val probes

Two guards protect the shared card. A process with less than --min-free-gib
free after loading stops before generating (exit 4). One whose generate calls
slow to a crawl stops too (exit 5): an over-full card under WSL2 pages to
system memory at about a tenth of the speed instead of failing. The samples
stage resumes from its rows on disk and pauses like the probe (exit 3).

    python scripts/teacher27b/teacher27b.py preflight
    python scripts/teacher27b/teacher27b.py greedy                      # 27b-retr8, then 27b-k0
    python scripts/teacher27b/teacher27b.py greedy --run runs/teacher27b_2026-10-30/smoke --limit 5 \\
        --arms 27b-retr8:plan_retr8
    python scripts/teacher27b/teacher27b.py samples --samples-per-call 8
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "retrieval"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                     # noqa: E402
from prompt import messages_for_case                               # noqa: E402
from baseline import load_base, load_registry                      # noqa: E402

MODEL = "qwen3.5-27b"
EXPERIMENT = "teacher27b_2026-10-30"
RUN = REPO / "runs" / EXPERIMENT
PLANS_SRC = REPO / "runs" / "retrieval_2026-10-18" / "val_plans"
PLAN_STEMS = ("plan_k0", "plan_retr8")
SAMPLES_PLAN = "plan_retr8"
CASES = REPO / "data" / "splits" / "val.jsonl"
TRAIN = REPO / "data" / "splits" / "train.jsonl"
TEST = REPO / "data" / "splits" / "test.jsonl"
PROBE_OUT = RUN / "probe_val_27b-retr8.jsonl"
REFERENCE_MODEL = "qwen3.5-4b"       # the plans' token budget was counted with its tokenizer
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "chat_template.jinja")

# The 4B and 9B val probes' sampling (probe_difficulty.py's defaults; pinned by a test).
G = 8
TEMPERATURE = 1.0
TOP_P = 1.0
TOP_K = 0
MAX_COMPLETION = 192                 # also baseline.py's greedy cap
SEED = 42

# What the report pairs the 27B against (preflight checks they exist).
GREEDY_COMPARATORS = {
    "4b-base-k0": ("runs/retrieval_2026-10-18/base-k0",
                   "Qwen3.5-4B base, zero-shot, the same plan"),
    "4b-base-retr8": ("runs/retrieval_2026-10-18/base-retr8",
                      "Qwen3.5-4B base, 8 retrieved examples, the same plan"),
    "4b-served": ("runs/qwen3.5-4b-grpo-gated-leg2",
                  "the served 4B adapter (leg 2's step 0, byte-identical to "
                  "runs/qwen3.5-4b-grpo-gated/best), fused kernels"),
    "9b": ("runs/qwen3.5-9b-grpo-gated",
           "the pooled pick's 9B adapter (gated round, step 0, NF4), fused kernels"),
}
PROBE_COMPARATORS = {
    "4b-served": ("runs/sample_rule_2026-10-09/probe_val.jsonl", "the served 4B, G = 8, T = 1"),
    "9b": ("runs/qwen9b_methods_2026-10-19/probe_val.jsonl", "the 9B adapter, G = 8, T = 1"),
}
SERVED_4B_ADAPTERS = ("runs/qwen3.5-4b-grpo-gated/best", "runs/qwen3.5-4b-grpo-gated-leg2/best")

RC_PAUSED, RC_NO_ROOM, RC_SLOW = 3, 4, 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def examples_for(plan_row, train_by_id):
    """The k-shot rows a plan row names, in order: what run_arm's
    examples_for returns for the same row, so both stages see one prompt."""
    return [train_by_id[i] for i in plan_row["examples"]]


# ------------------------------------------------------------ preflight

def cached_snapshot(reg):
    """The cached snapshot folder of the registry revision, or None."""
    from huggingface_hub import snapshot_download
    try:
        return pathlib.Path(snapshot_download(reg["hf_id"], revision=reg["revision"],
                                              local_files_only=True))
    except Exception:            # not cached (LocalEntryNotFoundError and kin)
        return None


def weights_report(reg):
    """Every shard the index names, plus config and tokenizer, on disk."""
    snap = cached_snapshot(reg)
    if snap is None:
        return {"ok": False, "why": "revision %s not in the HF cache" % reg["revision"][:12]}
    index = snap / "model.safetensors.index.json"
    if not index.exists():
        return {"ok": False, "why": "no model.safetensors.index.json in %s" % snap}
    shards = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
    missing = [s for s in shards if not (snap / s).exists()]
    missing += [f for f in ("config.json",) + TOKENIZER_FILES if not (snap / f).exists()]
    size = sum((snap / s).stat().st_size for s in shards if (snap / s).exists())
    return {"ok": not missing, "snapshot": str(snap), "shards": len(shards),
            "gb": round(size / 1e9, 2), "missing": missing}


def tokenizer_identity(reg, ref_reg):
    """File-by-file sha256 equality with the reference model's tokenizer."""
    a, b = cached_snapshot(reg), cached_snapshot(ref_reg)
    out = {}
    for f in TOKENIZER_FILES:
        pa, pb = (a / f) if a else None, (b / f) if b else None
        if not (pa and pb and pa.exists() and pb.exists()):
            out[f] = None                      # cannot tell: a file is not cached
        else:
            out[f] = sha256(pa) == sha256(pb)
    return out


def check_plans(plans_dir, case_ids, train_ids, heldout_ids):
    """Each plan covers the val rows in order and draws only from train."""
    report = {}
    for stem in PLAN_STEMS:
        rows = load_jsonl(str(pathlib.Path(plans_dir) / (stem + ".jsonl")))
        ex = [e for r in rows for e in r["examples"]]
        r = {"rows": len(rows), "ids_match_val": [x["id"] for x in rows] == list(case_ids),
             "examples": len(ex), "not_in_train": sum(e not in train_ids for e in ex),
             "held_out": sum(e in heldout_ids for e in ex),
             "k_max": max((len(x["examples"]) for x in rows), default=0)}
        r["ok"] = r["ids_match_val"] and not r["not_in_train"] and not r["held_out"]
        report[stem] = r
    return report


def check_comparators():
    out = {}
    for label, (folder, _) in GREEDY_COMPARATORS.items():
        d = REPO / folder
        need = [d / f for f in ("preds_val.jsonl", "exec_val.jsonl", "completions_val.jsonl")]
        out["greedy:" + label] = all(p.exists() for p in need)
    for label, (path, _) in PROBE_COMPARATORS.items():
        out["probe:" + label] = (REPO / path).exists()
    a, b = (REPO / p / "adapter_model.safetensors" for p in SERVED_4B_ADAPTERS)
    out["served_4b_identical"] = a.exists() and b.exists() and sha256(a) == sha256(b)
    return out


def cmd_preflight(args):
    run = pathlib.Path(args.run)
    (run / "plans").mkdir(parents=True, exist_ok=True)
    reg, ref = load_registry(MODEL), load_registry(REFERENCE_MODEL)
    report = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
              "registry": {"hf_id": reg["hf_id"], "revision": reg["revision"],
                           "chat_template_kwargs": reg.get("chat_template_kwargs"),
                           "stop": reg.get("stop")}}
    report["weights"] = weights_report(reg)
    report["tokenizer_identical_to_4b"] = tokenizer_identity(reg, ref)
    for stem in PLAN_STEMS:                   # copy once; a copy that drifted is an error
        src, dst = PLANS_SRC / (stem + ".jsonl"), run / "plans" / (stem + ".jsonl")
        if not dst.exists():
            shutil.copy2(src, dst)
        report.setdefault("plan_sha256", {})[stem] = sha256(dst)
        report.setdefault("plan_copy_matches_source", {})[stem] = sha256(dst) == sha256(src)
    case_ids = [c["id"] for c in load_jsonl(str(CASES))]
    train_ids = {r["id"] for r in load_jsonl(str(TRAIN))}
    heldout = set(case_ids) | {r["id"] for r in load_jsonl(str(TEST))}
    report["plans"] = check_plans(run / "plans", case_ids, train_ids, heldout)
    report["comparators"] = check_comparators()
    problems = []
    if not report["weights"]["ok"]:
        problems.append("weights: %s" % (report["weights"].get("why") or report["weights"]["missing"]))
    tok_same = report["tokenizer_identical_to_4b"]
    if tok_same.get("tokenizer.json") is not True or any(v is False for v in tok_same.values()):
        problems.append("tokenizer not confirmed equal to the 4B's: the plans' token budget may not hold")
    problems += ["plan %s" % s for s, r in report["plans"].items() if not r["ok"]]
    problems += ["plan copy %s drifted" % s for s, ok in report["plan_copy_matches_source"].items() if not ok]
    problems += ["comparator %s" % k for k, ok in report["comparators"].items() if not ok]
    report["problems"] = problems
    (run / "preflight.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1))
    print("preflight: %s" % ("OK" if not problems else "FAILED -- " + "; ".join(problems)))
    return 0 if not problems else 1


# ------------------------------------------------------------ the GPU stages

class SlowCard(RuntimeError):
    pass


class LatencyGuard:
    """Stops a run whose generate calls have slowed to a crawl.

    It times seconds per generated step, not per call: paging slows every
    step about tenfold, while a ramble to the token cap only makes a call
    long. SlowCard is raised when the median of the last `window` calls is
    above `limit`, or more than `factor` times this run's own first full
    window (another job taking the card mid-run). The first `warmup` calls
    are not counted: the fused linear-attention kernels compile on first use
    at the 27B's head counts.
    """

    def __init__(self, limit, factor=4.0, window=10, warmup=3):
        self.limit, self.factor = float(limit), float(factor)
        self.window, self.warmup = int(window), int(warmup)
        self.calls, self.times, self.baseline = 0, [], None

    def add(self, seconds_per_step):
        self.calls += 1
        if self.calls <= self.warmup:
            return
        self.times.append(float(seconds_per_step))
        recent = self.times[-self.window:]
        if len(recent) < self.window:
            return
        med = statistics.median(recent)
        if self.baseline is None:
            self.baseline = med
        if med > self.limit or med > self.factor * self.baseline:
            raise SlowCard("median of the last %d generate calls is %.2f s per generated step (this "
                           "run began at %.2f; limits %.2f, or %.0fx the start): the card is probably "
                           "paging or shared" % (self.window, med, self.baseline, self.limit, self.factor))

    def wrap(self, model):
        """Time every model.generate call; generation itself is unchanged."""
        inner = model.generate

        def generate(*a, **k):
            t = time.perf_counter()
            out = inner(*a, **k)
            prompt = k["input_ids"] if "input_ids" in k else a[0]
            steps = max(1, out.shape[-1] - prompt.shape[-1])
            self.add((time.perf_counter() - t) / steps)
            return out

        model.generate = generate
        return model


def load_27b(min_free_gib):
    import torch
    from transformers import AutoTokenizer
    reg = load_registry(MODEL)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = load_base(reg, True)
    model.eval()
    free, total = torch.cuda.mem_get_info()
    info = {"load_s": round(time.time() - t0, 1), "free_gib_after_load": round(free / 2 ** 30, 2),
            "total_gib": round(total / 2 ** 30, 2),
            "allocated_gib_after_load": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
            "gpu": torch.cuda.get_device_name(0)}
    print("loaded %s @ %s in NF4 in %.0f s: %.2f GiB allocated, %.2f of %.2f GiB free" % (
        reg["hf_id"], reg["revision"][:12], info["load_s"], info["allocated_gib_after_load"],
        info["free_gib_after_load"], info["total_gib"]), flush=True)
    if info["free_gib_after_load"] < min_free_gib:
        print("only %.2f GiB free after loading, %.2f needed: stopping before generating"
              % (info["free_gib_after_load"], min_free_gib), flush=True)
        sys.exit(RC_NO_ROOM)
    return reg, tok, model, info


def patch_summary(path, extra):
    """Relabel a run_arm summary as this experiment's (run_arm stamps the
    retrieval experiment's name) and add the load and memory facts. The
    metrics are not touched."""
    path = pathlib.Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    d["run"].update(extra)
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def cmd_greedy(args):
    from run_arms import run_arm
    run, plans = pathlib.Path(args.run), pathlib.Path(args.plans)
    cases = load_jsonl(str(CASES))
    if args.limit:
        cases = cases[:args.limit]
    train_by_id = {r["id"]: r for r in load_jsonl(str(TRAIN))}
    todo, plan_files = [], {}
    for spec in args.arms:
        arm, stem = spec.split(":")
        plan_files[arm] = plans / (stem + ".jsonl")
        if (run / arm / "baseline_val.json").exists():
            print("skip %s: done" % arm, flush=True)
            continue
        todo.append(arm)
    if not todo:
        return 0
    import torch
    reg, tok, model, info = load_27b(args.min_free_gib)
    LatencyGuard(args.max_s_per_step, args.slowdown).wrap(model)
    ns = SimpleNamespace(cases=str(CASES), timeout=args.timeout, max_new_tokens=MAX_COMPLETION,
                         adapter=None, base_quant="nf4", plan_files=plan_files)
    for arm in todo:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            run_arm(model, tok, reg, cases, train_by_id, load_jsonl(str(plan_files[arm])),
                    run / arm, ns, arm)
        except SlowCard as e:
            print("stopped %s: %s" % (arm, e), flush=True)
            return RC_SLOW
        patch_summary(run / arm / "baseline_val.json", {
            "experiment": EXPERIMENT, "load": info, "arm_seconds": round(time.time() - t0),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            "plan_sha256": sha256(plan_files[arm])})
    return 0


def cmd_samples(args):
    from concurrent.futures import ThreadPoolExecutor
    from probe_difficulty import PauseRequest, already_probed, observe, trim_completion
    from train_sft import inference_mode
    if G % args.samples_per_call:
        sys.exit("--samples-per-call must divide G = %d" % G)
    out_path = pathlib.Path(args.out)
    cases = load_jsonl(str(CASES))
    if args.limit:
        cases = cases[:args.limit]
    train_by_id = {r["id"]: r for r in load_jsonl(str(TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(pathlib.Path(args.plans) / (SAMPLES_PLAN + ".jsonl")))}
    done = already_probed(out_path)
    todo = [c for c in cases if c["id"] not in done]
    print("samples: %d rows, %d already in %s, %d to do; G = %d in calls of %d, T = %.1f"
          % (len(cases), len(cases) - len(todo), out_path, len(todo), G, args.samples_per_call,
             TEMPERATURE), flush=True)
    if not todo:
        return 0
    pause = PauseRequest(args.stop_file)
    if pause.marker() is not None:
        pause.marker().unlink(missing_ok=True)          # a new attempt starts clean
    import torch
    from transformers import set_seed
    set_seed(SEED)
    reg, tok, model, info = load_27b(args.min_free_gib)
    tok.padding_side = "left"
    LatencyGuard(args.max_s_per_step, args.slowdown).wrap(model)
    kwargs = reg.get("chat_template_kwargs") or {}
    eos_id, pad_id = tok.eos_token_id, tok.pad_token_id
    device = next(model.parameters()).device
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pause.install()
    torch.cuda.reset_peak_memory_stats()
    t0, n, solved = time.time(), 0, 0
    try:
        with ThreadPoolExecutor(max_workers=args.reward_workers) as pool, \
                out_path.open("a", encoding="utf-8") as fh:
            for case in todo:
                if pause.check() is not None:
                    break
                examples = examples_for(plan[case["id"]], train_by_id)
                text = tok.apply_chat_template(messages_for_case(case, examples),
                                               add_generation_prompt=True, tokenize=False, **kwargs)
                ids = []
                for _ in range(G // args.samples_per_call):
                    enc = tok([text] * args.samples_per_call, return_tensors="pt", padding=True)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    with inference_mode(model), torch.no_grad():
                        out = model.generate(**enc, max_new_tokens=MAX_COMPLETION, do_sample=True,
                                             temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                                             eos_token_id=eos_id, pad_token_id=pad_id)
                    gen = out[:, enc["input_ids"].shape[1]:].tolist()
                    ids += [trim_completion(r, eos_id, pad_id) for r in gen]
                row = observe(case, ids, tok, eos_id, args.timeout, pool)
                row.update({"step": 0, "source": "probe", "adapter": None, "model": MODEL,
                            "plan": SAMPLES_PLAN, "k_examples": len(examples),
                            "temperature": TEMPERATURE, "samples_per_call": args.samples_per_call,
                            "seed": SEED, "time": stamp})
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n += 1
                solved += row["k"] > 0
                if n % 25 == 0 or n == len(todo):
                    per = (time.time() - t0) / n
                    print("  [samples] %d/%d  %.1f s/row  eta %.0f min  any-right %d (%.1f%%)  peak %.1f GiB"
                          % (n, len(todo), per, per * (len(todo) - n) / 60, solved, 100.0 * solved / n,
                             torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    except SlowCard as e:
        print("stopped after %d rows: %s" % (n, e), flush=True)
        return RC_SLOW
    if pause.requested is not None:
        if pause.marker() is not None:
            pause.marker().write_text("paused %s after %d rows (%s)\n" % (
                datetime.datetime.now().isoformat(timespec="seconds"), n, pause.requested))
        print("paused after %d rows; the same command resumes" % n, flush=True)
        return RC_PAUSED
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preflight", help="CPU checks; copies the plans into the run folder")
    p.add_argument("--run", default=str(RUN))
    g = sub.add_parser("greedy", help="greedy arms through retrieval/run_arms.run_arm (GPU)")
    g.add_argument("--run", default=str(RUN))
    g.add_argument("--plans", default=str(RUN / "plans"))
    g.add_argument("--arms", nargs="+", default=["27b-retr8:plan_retr8", "27b-k0:plan_k0"],
                   help="arm:plan_stem pairs, run in this order")
    g.add_argument("--limit", type=int, default=None, help="first N val rows (smoke)")
    g.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    g.add_argument("--min-free-gib", type=float, default=1.0)
    g.add_argument("--max-s-per-step", type=float, default=1.5,
                   help="median seconds per generated step over 10 calls above this stops the stage (exit 5)")
    g.add_argument("--slowdown", type=float, default=4.0,
                   help="...or above this multiple of the run's own first 10 calls")
    s = sub.add_parser("samples", help="G = 8 samples per row at T = 1, probe rows (GPU)")
    s.add_argument("--plans", default=str(RUN / "plans"))
    s.add_argument("--out", default=str(PROBE_OUT))
    s.add_argument("--samples-per-call", type=int, default=G,
                   help="8 = one call per row; 4 = two calls, for less memory")
    s.add_argument("--limit", type=int, default=None, help="first N val rows (smoke)")
    s.add_argument("--timeout", type=float, default=5.0)
    s.add_argument("--reward-workers", type=int, default=8)
    s.add_argument("--min-free-gib", type=float, default=2.5)
    s.add_argument("--max-s-per-step", type=float, default=3.0)
    s.add_argument("--slowdown", type=float, default=4.0)
    s.add_argument("--stop-file", default=None,
                   help="pause when this file appears (it is consumed); PAUSED is written beside it")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"preflight": cmd_preflight, "greedy": cmd_greedy, "samples": cmd_samples}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
