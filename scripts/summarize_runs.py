#!/usr/bin/env python3
"""Build the results table from the run artifacts under runs/ (CLAUDE.md §14).

Every row of RESULTS.md is reproducible from a JSON that baseline.py or
train_sft.py wrote; this script reads them all and prints the table, so a
batch of runs that finishes unattended is reviewable without reopening each
file. Stdlib only.

    python scripts/summarize_runs.py                 # markdown to stdout
    python scripts/summarize_runs.py --out runs/summary.md
"""
import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]

SIZE_ORDER = ["qwen3.5-0.8b", "qwen3.5-2b", "qwen3-4b-2507", "qwen3.5-4b",
              "qwen2.5-coder-7b", "granite-4.2-8b", "seed-coder-8b", "qwen3.5-9b"]


def _load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_key(name):
    base = name.replace("-sft", "").replace("-grpo", "").split("/")[0]
    return (SIZE_ORDER.index(base) if base in SIZE_ORDER else len(SIZE_ORDER), name)


def collect(runs_dir):
    """One record per summary JSON: zero-shot rows, SFT/GRPO val rows, test rows."""
    rows = []
    for path in sorted(runs_dir.rglob("*.json")):
        if path.name not in ("baseline_test.json", "baseline_val.json", "sft_val.json",
                             "grpo_val.json"):
            continue
        d = _load(path)
        if not d or "metrics" not in d:
            continue
        rel = path.relative_to(runs_dir)
        run = d.get("run", {})
        m = d["metrics"]
        rows.append({
            "path": str(rel),
            "model": d.get("model", "?"),
            "condition": run.get("condition", "zero-shot"),
            "base": run.get("base", "bf16"),
            "split": d.get("split", path.stem.split("_")[-1]),
            "n": d.get("n"),
            "valid_pct": m.get("valid_pct"),
            "exec_match_pct": m.get("exec_match_pct"),
            "tokens": m.get("output_tokens_mean"),
            "p50": m.get("latency_s_p50"),
            "parse_failures": m.get("parse_failures"),
            "date": run.get("date"),
            "adapter": run.get("adapter"),
            "training": d.get("training"),
        })
    rows.sort(key=lambda r: (_model_key(r["model"]), r["condition"] != "zero-shot",
                             r["split"] != "test", r["path"]))
    return rows


def eval_histories(runs_dir):
    out = {}
    for path in sorted(runs_dir.glob("*/eval_history.jsonl")):
        hist = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                hist.append(json.loads(line))
            except ValueError:
                pass
        cfg = _load(path.parent / "train_config.json") or {}
        out[path.parent.name] = (hist, cfg)
    return out


def render(rows, hists):
    fmt = lambda v, s="": ("%s%s" % (v, s)) if v is not None else "—"
    lines = ["| model | condition | base | split | n | valid | exec-match | tokens (mean) | latency p50 | parse fails | date |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | **%s** | %s | %s | %s | %s |" % (
            r["model"], r["condition"], r["base"], r["split"], fmt(r["n"]),
            fmt(r["valid_pct"], "%"), fmt(r["exec_match_pct"], "%"), fmt(r["tokens"]),
            fmt(r["p50"], " s"), fmt(r["parse_failures"]), fmt(r["date"])))
    if hists:
        lines += ["", "## Training curves (val subset exec-match by step)", ""]
        for name in sorted(hists, key=_model_key):
            hist, cfg = hists[name]
            base = (cfg.get("base") or {}).get("quantization", "bf16")
            opt = cfg.get("optimization", {})
            lines.append("**%s** — base %s, micro-batch %s × %s, lr %s, %s epochs" % (
                name, base, opt.get("micro_batch", "?"), opt.get("grad_accum", "?"),
                opt.get("lr", "?"), opt.get("epochs", "?")))
            lines.append("")
            lines.append("| step | exec-match | valid | tokens | eval s |")
            lines.append("|---|---|---|---|---|")
            for h in hist:
                lines.append("| %s | %s%% | %s%% | %s | %s |" % (
                    h.get("step"), h.get("exec_match_pct"), h.get("valid_pct"),
                    h.get("output_tokens_mean"), h.get("eval_seconds")))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(REPO / "runs"))
    ap.add_argument("--out", default=None, help="write the markdown here instead of stdout")
    args = ap.parse_args()
    runs_dir = pathlib.Path(args.runs)
    text = render(collect(runs_dir), eval_histories(runs_dir))
    if args.out:
        pathlib.Path(args.out).write_text(text, encoding="utf-8")
        print("wrote %s" % args.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
