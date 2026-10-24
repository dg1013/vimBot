#!/usr/bin/env python3
"""Merge a LoRA adapter into its registry base and save weights vLLM can serve.

§10: "Merge LoRA -> serve on vLLM". The base loads the way it loads
everywhere in this repo (baseline.load_base: bf16 on the GPU, through
AutoModelForCausalLM, i.e. the text model of the multimodal checkpoint). The
adapter merges into it, and the result is saved with the tokenizer and a
provenance marker (merge.json). Merging in bf16 rounds the adapter's fp32
update into the weights, so one greedy answer is compared before and after
the merge and the result recorded; serve/parity.py measures the whole of val.

    python scripts/serve/merge_adapter.py --model qwen3.5-4b \
        --adapter runs/qwen3.5-4b-grpo-gated/best --out runs/serve/qwen3.5-4b-served
"""
import argparse
import datetime
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))


HF_PREFIX, VLLM_PREFIX = "model.language_model.", "model."


def vllm_names(out):
    """Rename the saved tensors to the names vLLM's Qwen3_5ForCausalLM loads.

    transformers saves the text model of the multimodal checkpoint under its
    multimodal names (`model.language_model.layers...`), with a text-only
    config (`Qwen3_5ForCausalLM`). vLLM's class for that config expects
    `model.layers...` and refused the merge on 2026-10-13 ("no module or
    parameter named 'language_model'"). Names only: no value changes."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    path = pathlib.Path(out) / "model.safetensors"
    with safe_open(str(path), "pt") as f:
        meta = f.metadata() or {}
        tensors = {(VLLM_PREFIX + k[len(HF_PREFIX):] if k.startswith(HF_PREFIX) else k): f.get_tensor(k)
                   for k in f.keys()}
    renamed = sum(1 for k in tensors if not k.startswith(HF_PREFIX))
    save_file(tensors, str(path), metadata=dict(meta, format="pt"))
    marker = pathlib.Path(out) / "merge.json"
    m = json.loads(marker.read_text()) if marker.exists() else {}
    m["tensor_names"] = "%s* -> %s* (vLLM's Qwen3_5ForCausalLM); values unchanged" % (HF_PREFIX, VLLM_PREFIX)
    marker.write_text(json.dumps(m, indent=1) + "\n")
    return renamed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-4b", help="registry name")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-case", default=str(REPO / "data" / "splits" / "val.jsonl"),
                    help="a case file; its first row is the before/after check")
    ap.add_argument("--rename-only", action="store_true",
                    help="only rename an existing merge's tensors for vLLM")
    args = ap.parse_args()
    if args.rename_only:
        print("renamed %d tensors for vLLM in %s" % (vllm_names(args.out), args.out))
        return 0
    if not args.adapter:
        sys.exit("--adapter is required to merge")
    from transformers import AutoTokenizer
    from peft import PeftModel
    from baseline import generate_predictions, load_base, load_registry, resolve_base_quant
    from evaluate import load_jsonl

    if resolve_base_quant(args.adapter):
        sys.exit("%s was trained on an NF4 base; a merge needs the bf16 base it was trained on"
                 % args.adapter)
    out = pathlib.Path(args.out)
    if (out / "merge.json").exists():
        print("already merged: %s" % out)
        return 0
    reg = load_registry(args.model)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = PeftModel.from_pretrained(load_base(reg, False), args.adapter)
    model.eval()
    probe = load_jsonl(args.probe_case)[:1]
    before, _ = generate_predictions(model, tok, probe, reg)
    merged = model.merge_and_unload()
    after, _ = generate_predictions(merged, tok, probe, reg)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok.save_pretrained(str(out))
    cfg = json.loads((out / "config.json").read_text())
    marker = {"hf_id": reg["hf_id"], "revision": reg["revision"], "adapter": args.adapter,
              "architectures": cfg.get("architectures"), "dtype": "bfloat16",
              "probe": {"id": probe[0]["id"], "adapter": before[0]["completion"],
                        "merged": after[0]["completion"],
                        "identical": before[0]["completion"] == after[0]["completion"]},
              "merged_at": datetime.datetime.now().isoformat(timespec="seconds")}
    (out / "merge.json").write_text(json.dumps(marker, indent=1) + "\n")
    vllm_names(out)
    print("merged %s into %s @ %s -> %s (%s); probe %s: %r vs %r" % (
        args.adapter, reg["hf_id"], reg["revision"][:12], out, marker["architectures"],
        "identical" if marker["probe"]["identical"] else "DIFFERS",
        before[0]["completion"], after[0]["completion"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
