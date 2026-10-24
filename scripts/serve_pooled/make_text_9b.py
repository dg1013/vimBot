#!/usr/bin/env python3
"""Save the 9B's text model for vLLM: bf16, no adapter merged, tensor names as
vLLM's Qwen3_5ForCausalLM loads them.

The served 9B is this base quantized to 4-bit by vLLM at load (bitsandbytes,
in flight) with the 9B's LoRA adapter applied at runtime: NF4 base plus a
LoRA, the way the adapter was trained (QLoRA). Built on the CPU, since the
bf16 9B (~18 GB) would fill the card.

    python scripts/serve_pooled/make_text_9b.py --out runs/serve/qwen3.5-9b-text
"""
import argparse
import datetime
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve"):
    sys.path.insert(0, str(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-9b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from baseline import load_registry
    from merge_adapter import vllm_names
    reg = load_registry(args.model)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = AutoModelForCausalLM.from_pretrained(reg["hf_id"], revision=reg["revision"], dtype=torch.bfloat16,
                                                 device_map="cpu", low_cpu_mem_usage=True)
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="100GB")
    tok.save_pretrained(str(out))
    (out / "merge.json").write_text(json.dumps({
        "hf_id": reg["hf_id"], "revision": reg["revision"], "adapter": None,
        "architectures": list(getattr(model.config, "architectures", None) or []), "dtype": "bfloat16",
        "note": "the 9B's text model, no adapter merged; served 4-bit with its LoRA at runtime",
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1) + "\n")
    print("saved; renamed %d tensors for vLLM -> %s" % (vllm_names(str(out)), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
