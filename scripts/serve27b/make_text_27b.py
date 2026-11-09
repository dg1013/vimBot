#!/usr/bin/env python3
"""Save the 27B's text model for vLLM, one shard at a time.

scripts/serve_pooled/make_text_9b.py loads the whole model on the CPU and saves it. The 27B is 52 GB
in bf16 against 45 GB of RAM here, so this copies the checkpoint shard by shard instead: every text
tensor keeps its value and takes the name vLLM's Qwen3_5ForCausalLM loads, and the vision tower and
the multi-token head are left out. A shard is about 5 GB in memory at its peak.

The served 27B is this base quantized to 4-bit by vLLM at load, with the trained adapter applied at
runtime: NF4 base plus a LoRA, the way the adapter was trained, as the served 9B is.

    python scripts/serve27b/make_text_27b.py --out runs/serve/qwen3.5-27b-text
    python scripts/serve27b/make_text_27b.py --check-against runs/serve/qwen3.5-9b-text --model qwen3.5-9b
"""
import argparse
import datetime
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve"):
    sys.path.insert(0, str(p))
from merge_adapter import HF_PREFIX, VLLM_PREFIX      # noqa: E402

COPIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt")


def text_name(key):
    """The name vLLM's class loads for a checkpoint tensor, or None to leave it out."""
    if key.startswith(HF_PREFIX):
        return VLLM_PREFIX + key[len(HF_PREFIX):]
    if key.startswith("lm_head."):
        return key
    return None


def text_config(full):
    """The text-only config, as transformers saves it for the 9B (runs/serve/qwen3.5-9b-text)."""
    cfg = dict(full["text_config"])
    cfg.update(architectures=["Qwen3_5ForCausalLM"], bos_token_id=None, pad_token_id=None,
               tie_word_embeddings=full.get("tie_word_embeddings", False),
               partial_rotary_factor=cfg["rope_parameters"]["partial_rotary_factor"])
    return cfg


def snapshot(model):
    """(registry entry, the local snapshot of its pinned revision)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    from baseline import load_registry
    reg = load_registry(model)
    snap = pathlib.Path(HF_HUB_CACHE) / ("models--" + reg["hf_id"].replace("/", "--")) / "snapshots" / reg["revision"]
    if not (snap / "model.safetensors.index.json").exists():
        sys.exit("no cached snapshot at %s" % snap)
    return reg, snap


def check_against(snap, copy):
    """The names text_name keeps from a checkpoint against a text copy built by transformers:
    returns (expected names, names in the copy)."""
    from safetensors import safe_open
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    want = {n for n in map(text_name, index["weight_map"]) if n}
    with safe_open(str(pathlib.Path(copy) / "model.safetensors"), "pt") as f:
        have = set(f.keys())
    return want, have


def convert(snap, out):
    from safetensors import safe_open
    from safetensors.torch import save_file
    index = json.loads((snap / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        dst = out / ("model-%05d-of-%05d.safetensors" % (i, len(shards)))
        tensors = {}
        with safe_open(str(snap / shard), "pt") as f:
            names = {k: text_name(k) for k in f.keys()}
            if dst.exists():                      # a relaunch keeps finished shards
                kept = {v for v in names.values() if v}
            else:
                tensors = {v: f.get_tensor(k) for k, v in names.items() if v}
                kept = set(tensors)
        if not kept:
            continue
        if tensors:
            part = dst.with_suffix(".part")
            save_file(tensors, str(part), metadata={"format": "pt"})
            part.rename(dst)
        with safe_open(str(dst), "pt") as f:      # read back: names and sizes as written
            got = set(f.keys())
        if got != kept:
            sys.exit("%s holds %d names, expected %d" % (dst.name, len(got), len(kept)))
        weight_map.update({k: dst.name for k in kept})
        print("%s: %d tensors" % (dst.name, len(kept)), flush=True)
        del tensors
    expected = {n for n in map(text_name, index["weight_map"]) if n}
    if set(weight_map) != expected:
        sys.exit("kept %d tensors, expected %d" % (len(weight_map), len(expected)))
    size = sum((out / n).stat().st_size for n in set(weight_map.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": size}, "weight_map": dict(sorted(weight_map.items()))}, indent=1) + "\n")
    return len(weight_map), size


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="qwen3.5-27b", help="registry name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-against", default=None, help="a text copy built by transformers: compare names and config")
    args = ap.parse_args()
    reg, snap = snapshot(args.model)
    full = json.loads((snap / "config.json").read_text())
    if args.check_against:
        want, have = check_against(snap, args.check_against)
        stored = json.loads((pathlib.Path(args.check_against) / "config.json").read_text())
        made = text_config(full)
        cfg_diff = {k: (made.get(k), stored.get(k)) for k in set(made) | set(stored)
                    if k != "transformers_version" and made.get(k) != stored.get(k)}
        print("names: %d expected, %d in the copy, %d only expected, %d only in the copy; config differences: %r"
              % (len(want), len(have), len(want - have), len(have - want), cfg_diff))
        return 0 if want == have and not cfg_diff else 1
    if not args.out:
        sys.exit("--out is required")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, size = convert(snap, out)
    (out / "config.json").write_text(json.dumps(text_config(full), indent=2) + "\n")
    gen = json.loads((snap / "generation_config.json").read_text())
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": gen["eos_token_id"]}, indent=2) + "\n")
    for name in COPIED:
        if (snap / name).exists():
            shutil.copy2(snap / name, out / name)
    (out / "merge.json").write_text(json.dumps({
        "hf_id": reg["hf_id"], "revision": reg["revision"], "adapter": None,
        "architectures": ["Qwen3_5ForCausalLM"], "dtype": "bfloat16",
        "note": "the 27B's text model, no adapter merged; served 4-bit with its LoRA at runtime",
        "tensors": n, "bytes": size, "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "tensor_names": "%s* -> %s* (vLLM's Qwen3_5ForCausalLM); values unchanged" % (HF_PREFIX, VLLM_PREFIX)}, indent=1) + "\n")
    print("saved %d tensors, %.1f GB -> %s" % (n, size / 1e9, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
