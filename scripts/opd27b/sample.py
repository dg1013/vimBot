#!/usr/bin/env python3
"""OPD's sampling stage: the 4B writes its own answers, with vLLM (runs/opd27b_2026-11-02/preregistration.md).

vLLM's offline engine runs the 4B's text model in bf16 (`make_text_9b.py --model qwen3.5-4b`, saved
unmerged, as the served 9B's base is), with the current adapter as a LoRA module. vLLM batches
prompts without padding. The loss never uses the sampler's probabilities, so vLLM's small numeric
differences from the training stack do not enter it; they are kept only for the descriptive
readings.

GRPO's sampling: T 1, top-p 1, top-k off, at most 192 new tokens, the registry's stop ids, and a
seed per prompt. The stop token stays as each sample's last token, so the teacher grades the
decision to stop as well.

    python scripts/opd27b/sample.py train --seed 42 --round 1    # round 1's samples, from the start
    python scripts/opd27b/sample.py pass8 --seed 42 --label start --adapter runs/qwen3.5-4b-sft-distill/best
    python scripts/opd27b/sample.py judge --seed 42 --round 1    # CPU: §6 grammar and the sandbox
"""
import argparse
import concurrent.futures
import importlib.util
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl, score                            # noqa: E402
from prompt import parse_output                                  # noqa: E402

spec = importlib.util.spec_from_file_location("opd27b_opd", HERE / "opd.py")
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)

GPU_UTIL = 0.80
PASS_K = 8
JUDGE_WORKERS = 8


def engine_kwargs(reg):
    """The registry's vllm_extra_args (flag, value pairs) as LLM(...) keyword arguments."""
    extra, out = list(reg.get("vllm_extra_args") or []), {}
    for flag, value in zip(extra[::2], extra[1::2]):
        out[flag.lstrip("-").replace("-", "_")] = int(value) if str(value).isdigit() else value
    return out


def make_engine(base_text):
    from vllm import LLM
    from baseline import load_registry
    reg = load_registry(O.STUDENT)
    if not (pathlib.Path(base_text) / "merge.json").exists():
        sys.exit("no text model at %s (make_text_9b.py --model %s)" % (base_text, O.STUDENT))
    llm = LLM(model=str(base_text), dtype="bfloat16", max_model_len=reg["max_seq_len"], seed=0,
              gpu_memory_utilization=GPU_UTIL, enable_lora=True, max_lora_rank=16, max_loras=1,
              **engine_kwargs(reg))
    from transformers import AutoTokenizer, GenerationConfig
    from baseline import stop_token_ids
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    # vLLM also stops at the model's own EOS ids (its generation_config), so those count as stops too
    eos = GenerationConfig.from_pretrained(str(base_text)).eos_token_id
    stops = set(stop_token_ids(tok, reg.get("stop"))) | set(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    return llm, tok, reg, sorted(stops)


def sampled_lp(position, token):
    """The sampled token's log-probability from vLLM's per-position logprobs ({id: Logprob})."""
    x = position.get(token) if hasattr(position, "get") else position[token]
    return float(getattr(x, "logprob", x))


def generate(llm, prompts, adapter, seeds, stops, n=1):
    """[(id, [(ids, finish, sampled log-probs)] * n)] in prompt order."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    params = [SamplingParams(n=n, temperature=1.0, top_p=1.0, top_k=0, max_tokens=O.MAX_NEW, seed=seeds[i],
                             stop_token_ids=stops, logprobs=0, skip_special_tokens=False) for i, _ in prompts]
    outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids in prompts], params,
                        lora_request=LoRARequest("student", 1, str(adapter)))
    res = []
    for (i, ids), out in zip(prompts, outs):
        if list(out.prompt_token_ids) != list(ids):
            raise SystemExit("vLLM changed the prompt of %s" % i)
        samples = []
        for o in out.outputs:
            toks = list(o.token_ids)
            lps = [round(sampled_lp(lp, t), 4) for t, lp in zip(toks, o.logprobs or [])] if o.logprobs else []
            if o.finish_reason == "stop" and (not toks or toks[-1] not in stops):
                raise SystemExit("%s stopped without its stop token in the output" % i)
            samples.append((toks, o.finish_reason, lps))
        res.append((i, samples))
    return res


def text_of(tok, ids, stops):
    """The completion as generate_predictions decodes it: the stop token is not content."""
    body = ids[:-1] if ids and ids[-1] in stops else ids
    return tok.decode(body, skip_special_tokens=True)


def cmd_train(args):
    seed, r = args.seed, args.round
    rd = O.round_dir(seed, r, args.root)
    path = rd / "samples.jsonl"
    prompts = O.load_prompts(args.run)
    order = O.round_order(seed, r, prompts)
    if args.limit:
        order = order[:args.limit]
    if path.exists() and len(load_jsonl(str(path))) == len(order):
        print("round %d: sampled already (%s)" % (r, path))
        return 0
    adapter = O.adapter_for(seed, r - 1, args.root)
    t0 = time.time()
    llm, tok, reg, stops = make_engine(args.base_text)
    print("engine up in %.0f s; sampling %d prompts from %s" % (time.time() - t0, len(order), adapter), flush=True)
    t1 = time.time()
    res = generate(llm, [(i, prompts[i]) for i in order], adapter,
                   {i: O.sample_seed(seed, r, i) for i in order}, stops)
    secs = time.time() - t1
    rd.mkdir(parents=True, exist_ok=True)
    tmp = rd / "samples.jsonl.partial"
    with tmp.open("w", encoding="utf-8") as fh:
        for i, [(ids, finish, lps)] in res:
            fh.write(json.dumps({"id": i, "ids": ids, "n": len(ids), "finish": finish, "slp": lps,
                                 "text": text_of(tok, ids, stops), "seed": O.sample_seed(seed, r, i)}) + "\n")
    tmp.rename(path)
    lens = sorted(len(x[1][0][0]) for x in res)
    meta = {"seed": seed, "round": r, "adapter": str(adapter), "prompts": len(res), "seconds": round(secs, 1),
            "tokens": sum(lens), "median_tokens": lens[len(lens) // 2], "max_tokens": lens[-1],
            "hit_cap": sum(1 for x in res if x[1][0][1] == "length"), "stop_ids": stops}
    (rd / "samples_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print("round %d: %d samples in %.0f s (median %d tokens, %d at the cap) -> %s" % (
        r, len(res), secs, meta["median_tokens"], meta["hit_cap"], path), flush=True)
    return 0


def judge_rows(cases, texts, timeout):
    """{id: (parse_ok, valid, match)} for one sample per case, in parallel chunks."""
    preds, parsed = {}, {}
    for c, t in zip(cases, texts):
        p = parse_output(t)
        parsed[c["id"]] = p is not None
        if p:
            preds[c["id"]] = {"id": c["id"], "mode": p[0], "command": p[1]}
    chunks = [cases[k:k + 64] for k in range(0, len(cases), 64)]
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=JUDGE_WORKERS) as pool:
        for per in pool.map(lambda ch: score(ch, {c["id"]: preds[c["id"]] for c in ch if c["id"] in preds},
                                             timeout=timeout)["per_case"], chunks):
            for i, s in per.items():
                out[i] = (parsed[i], bool(s.get("valid")), bool(s.get("match")))
    return out


def cmd_judge(args):
    rd = O.round_dir(args.seed, args.round, args.root)
    path = rd / "judged.jsonl"
    if path.exists():
        print("round %d: judged already" % args.round)
        return 0
    samples = load_jsonl(str(rd / "samples.jsonl"))
    train = {c["id"]: c for c in load_jsonl(str(O.TRAIN))}
    cases = [train[s["id"]] for s in samples]
    t0 = time.time()
    res = judge_rows(cases, [s["text"] for s in samples], args.timeout)
    with (rd / "judged.jsonl.partial").open("w", encoding="utf-8") as fh:
        for s in samples:
            p, v, m = res[s["id"]]
            fh.write(json.dumps({"id": s["id"], "parse_ok": p, "valid": v, "match": m}) + "\n")
    (rd / "judged.jsonl.partial").rename(path)
    print("round %d: %d of %d samples match (%.1f%%), judged in %.0f s" % (
        args.round, sum(m for _, _, m in res.values()), len(res), 100.0 * sum(m for _, _, m in res.values()) / len(res),
        time.time() - t0), flush=True)
    return 0


def cmd_pass8(args):
    od = pathlib.Path(args.root) if args.root else O.out_dir(args.seed)
    path = od / ("pass8_%s.jsonl" % args.label)
    if path.exists():
        print("pass@%d %s: done already" % (PASS_K, args.label))
        return 0
    from baseline import load_registry
    reg = load_registry(O.STUDENT)
    cases = O.subset_cases()
    if args.limit:
        cases = cases[:args.limit]
    t0 = time.time()
    llm, tok, _, stops = make_engine(args.base_text)
    kw = reg.get("chat_template_kwargs") or {}
    prompts = [(c["id"], O.encode_prompt(tok, c, kw)) for c in cases]
    res = generate(llm, prompts, args.adapter, {c["id"]: O.sample_seed(args.seed, 100, c["id"]) for c in cases},
                   stops, n=PASS_K)
    del llm
    hits = {c["id"]: [] for c in cases}
    for k in range(PASS_K):
        judged = judge_rows(cases, [text_of(tok, s[k][0], stops) for _, s in res], args.timeout)
        for i, (_, _, m) in judged.items():
            hits[i].append(m)
    od.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps({"id": c["id"], "matches": hits[c["id"]], "any": any(hits[c["id"]])}) + "\n")
    print("pass@%d %s: %d of %d rows (%.1f%%) in %.0f s" % (
        PASS_K, args.label, sum(any(v) for v in hits.values()), len(cases),
        100.0 * sum(any(v) for v in hits.values()) / len(cases), time.time() - t0), flush=True)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("train", cmd_train), ("judge", cmd_judge), ("pass8", cmd_pass8)):
        a = sub.add_parser(name)
        a.add_argument("--seed", type=int, choices=sorted(O.STARTS), required=True)
        if name != "pass8":
            a.add_argument("--round", type=int, required=True)
        else:
            a.add_argument("--label", required=True)
            a.add_argument("--adapter", required=True)
        a.add_argument("--run", default=str(O.RUN))
        a.add_argument("--root", default=None)
        a.add_argument("--base-text", default=str(O.BASE_TEXT))
        a.add_argument("--timeout", type=float, default=5.0)
        a.add_argument("--limit", type=int, default=None, help="first N prompts (check)")
        a.set_defaults(fn=fn)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
