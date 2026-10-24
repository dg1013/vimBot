#!/usr/bin/env python
"""Build the all-data workbook: every dataset, run, probe, fix and log of the
project, one tab each, with derived breakdown columns (vim command class,
regex features, keystroke features, buffer-diff shape, difficulty, split,
leakage component ...) so any question can be answered with a pivot table.

    conda run -n vimBot python exports/build_workbook.py [--out PATH] [--no-rescore]

Reads only; writes the workbook (default `exports/vimBot_data_<date>.xlsx`)
and a scoring cache under `exports/cache/` for the runs that predate the
per-case exec log (their predictions are re-executed through the sandbox
once, ~1 min with 8 threads). Needs the project env for pandas/pyarrow
(GRPO rollout parquet) and PyYAML (taxonomy, model registry); the xlsx
itself is written by the stdlib `xlsx_writer.py` beside this file.

Every derived column is a heuristic over the command string and is labelled
as such in the README tab; the measured columns (valid, match, tokens,
latency, p-hat) come straight from the run artifacts.
"""

import argparse
import collections
import difflib
import glob
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "exports"))

from xlsx_writer import Workbook                              # noqa: E402
from evaluate import score as harness_score                   # noqa: E402
from scope import _KEY_TOKEN                                  # noqa: E402
from build_splits import build_components, stratum_of         # noqa: E402
from pattern_features import feature_bucket, extract_patterns  # noqa: E402

try:
    import yaml
except ImportError:                                           # pragma: no cover
    yaml = None
try:
    import pandas as pd
except ImportError:                                           # pragma: no cover
    pd = None

RUNS = REPO / "runs"
DATA = REPO / "data"
CACHE = REPO / "exports" / "cache"
TODAY = date.today().isoformat()
FLA_FROM = "2026-09-16"          # rows measured from this date run on the fused kernels

# ------------------------------------------------------------------ helpers


def jl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def jload(path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def flat(v):
    """Cell-safe scalar: lists/dicts as compact JSON, None stays None."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return json.dumps(v, ensure_ascii=False)


def pct(a, b):
    return round(a / b, 4) if b else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else round((xs[m - 1] + xs[m]) / 2, 3)


def norm_lines(text):
    if text is None:
        return None
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


# ------------------------------------------------------ vim command parsing

from vimparse import parse_ex, parse_keys                     # noqa: E402


def regex_features(cmd, mode, parsed):
    pats = extract_patterns(cmd, mode)
    if parsed.get("ex_sub_pattern"):
        pats = pats + [parsed["ex_sub_pattern"]]
    p = " ".join(pats)
    rep = parsed.get("ex_sub_replacement") or ""
    whole = cmd or ""
    return {
        "re_has_pattern": bool(pats),
        "re_very_magic": "\\v" in p,
        "re_nomagic": "\\V" in p or "\\M" in p,
        "re_capture_group": "\\(" in p or ("\\v" in p and "(" in p),
        "re_backref": bool(re.search(r"\\[1-9]", whole)),
        "re_alternation": "\\|" in p or ("\\v" in p and "|" in p),
        "re_counted_rep": "\\{" in p or ("\\v" in p and "{" in p),
        "re_word_boundary": "\\<" in p or "\\>" in p or "\\b" in p,
        "re_char_class": bool(re.search(r"(?<!\\)\[", p)),
        "re_neg_class": "[^" in p,
        "re_shorthand_class": bool(re.search(r"\\[dDwWsSaAlLuUxXoOhHkKiIfFpP]", p)),
        "re_anchor_start": p.startswith("^") or " ^" in p or "/^" in whole,
        "re_anchor_end": p.endswith("$") or "$ " in p or "$/" in whole,
        "re_quantifier_star": bool(re.search(r"(?<!\\)\*", p)),
        "re_quantifier_plus": "\\+" in p or ("\\v" in p and "+" in p),
        "re_optional": "\\=" in p or "\\?" in p,
        "re_non_capturing": "\\%(" in p,
        "re_zs_ze": "\\zs" in p or "\\ze" in p,
        "re_lookaround": "\\@" in p,
        "re_case_flag": "\\c" in p or "\\C" in p,
        "re_multiline": "\\n" in p or "\\_" in p,
        "re_expression_replacement": rep.startswith("\\="),
        "re_submatch_call": "submatch(" in whole,
        "re_case_modifier_in_replacement": bool(re.search(r"\\[uUlLeE]", rep)),
        "re_newline_in_replacement": "\\r" in rep or "\\n" in rep or "^M" in rep,
        "re_ampersand_in_replacement": bool(re.search(r"(?<!\\)&", rep)),
        "re_whole_match_ref": "\\0" in rep,
        "re_pattern_chars": len(p) if pats else 0,
        "feature_bucket": feature_bucket(cmd or "", mode),
    }


def buffer_diff(before, after):
    a, b = norm_lines(before) or [], norm_lines(after) or []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    ins = dele = rep_a = rep_b = 0
    first = None
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        if first is None:
            first = i1 + 1
        if tag == "insert":
            ins += j2 - j1
        elif tag == "delete":
            dele += i2 - i1
        else:
            rep_a += i2 - i1
            rep_b += j2 - j1
    if a == b:
        kind = "no-change"
    elif sorted(a) == sorted(b):
        kind = "reorder"
    elif ins and not dele and not rep_a:
        kind = "insert-only"
    elif dele and not ins and not rep_a:
        kind = "delete-only"
    elif rep_a and not ins and not dele:
        kind = "modify-in-place"
    else:
        kind = "mixed"
    # character-level size of the change on the modified lines
    return {
        "buffer_before_lines": len(a), "buffer_before_chars": len(before or ""),
        "buffer_after_lines": len(b), "buffer_after_chars": len(after or ""),
        "lines_delta": len(b) - len(a), "chars_delta": len(after or "") - len(before or ""),
        "lines_inserted": ins, "lines_deleted": dele,
        "lines_modified": max(rep_a, rep_b), "lines_unchanged": len(a) - dele - rep_a,
        "first_changed_line": first, "change_kind": kind,
        "buffer_has_tabs": "\t" in (before or ""),
        "buffer_has_trailing_ws": bool(re.search(r"[ \t]+\n", before or "")),
        "buffer_has_blank_lines": "\n\n" in (before or "").rstrip("\n") + "\n",
        "buffer_max_line_len": max((len(x) for x in a), default=0),
        "buffer_ratio_after_before": round(len(after or "") / len(before), 3) if before else None,
    }


def buffer_kind(text):
    t = text or ""
    lines = [l for l in t.split("\n") if l.strip()]
    if not lines:
        return "empty"
    if re.search(r"^\s*(def |class |import |from \w+ import|print\()", t, re.M):
        return "python"
    if re.search(r"\b(function|const|let|var)\b|=>|console\.log", t):
        return "javascript"
    if re.search(r"#include|int main|printf\(|;\s*$", t, re.M) and "{" in t:
        return "c-like"
    if t.lstrip().startswith(("{", "[")) and ":" in t:
        return "json"
    if re.search(r"^#{1,6} |^\s*[-*] |\[.*\]\(.*\)", t, re.M):
        return "markdown"
    if len(lines) >= 2 and all(l.count(",") >= 1 for l in lines) and len({l.count(",") for l in lines}) <= 2:
        return "csv"
    if re.search(r"^\s*[\w.-]+\s*[=:]\s*\S", t, re.M) and len(lines) >= 2 and sum(bool(re.match(r"^\s*[\w.-]+\s*[=:]", l)) for l in lines) >= len(lines) * 0.6:
        return "config-or-kv"
    if re.search(r"\d{4}-\d{2}-\d{2}|\b(INFO|WARN|ERROR|DEBUG)\b", t):
        return "log"
    if re.search(r"^\s*(\$|#|sudo|cd |ls |grep )", t, re.M):
        return "shell"
    if len(lines) >= 3 and sum(len(l.split()) <= 3 for l in lines) >= len(lines) * 0.7:
        return "list"
    return "prose"


def instruction_features(ins):
    ins = ins or ""
    words = ins.split()
    n = len(words)
    return {
        "instruction_chars": len(ins), "instruction_words": n,
        "instruction_form": "question" if ins.rstrip().endswith("?") else
                            ("terse" if n <= 5 else "short" if n <= 12 else "long"),
        "instruction_mentions_line_number": bool(re.search(r"\bline\s+\d+|\b\d+(st|nd|rd|th)\b|\blines?\s+\d+", ins, re.I)),
        "instruction_quotes_text": bool(re.search(r"[\"'`“”‘’]", ins)),
        "instruction_mentions_cursor": bool(re.search(r"\bcursor\b|\bhere\b|\bthis line\b|\bcurrent\b", ins, re.I)),
        "instruction_mentions_all_or_every": bool(re.search(r"\b(all|every|each|whole|entire)\b", ins, re.I)),
        "instruction_negation": bool(re.search(r"\b(not|don't|except|without|only)\b", ins, re.I)),
        "instruction_first_word": words[0].strip(",.:;").lower() if words else None,
    }


# ----------------------------------------------------------------- loading


def load_corpus():
    corpus = jl(DATA / "combined_cases.jsonl")
    splits = {}
    for name in ("train", "val", "test"):
        for r in jl(DATA / "splits" / f"{name}.jsonl"):
            splits[r["id"]] = name
    return corpus, splits


def run_dirs():
    """Every folder under runs/ that holds a scored row, incl. sub-variants."""
    found = []
    for js in sorted(glob.glob(str(RUNS / "**" / "baseline_test.json"), recursive=True)) + \
            sorted(glob.glob(str(RUNS / "**" / "sft_val.json"), recursive=True)) + \
            sorted(glob.glob(str(RUNS / "**" / "grpo_val.json"), recursive=True)):
        d = str(Path(js).parent.relative_to(RUNS))
        if d not in found:
            found.append(d)
    return found


def model_meta(model):
    fam = "Qwen3.5 (hybrid linear+full attention)" if model.startswith("qwen3.5") else \
          "Qwen3 dense" if model.startswith("qwen3-") else "Qwen2.5-Coder dense" if model.startswith("qwen2.5") else \
          "IBM Granite 4 dense" if model.startswith("granite") else "Seed-Coder dense" if model.startswith("seed") else "?"
    m = re.search(r"(\d+(?:\.\d+)?)b", model)
    params = float(m.group(1)) if m else None
    return fam, params


def stage_label(cond, run):
    if run.startswith("qwen3.5-4b-sft-cliff-gentle"):
        return "cliff SFT, gentle recipe: 1 epoch, lr 1e-5, replay 1:1 (null: step 0 kept)"
    if run.startswith("qwen3.5-4b-sft-cliff"):
        return "cliff SFT round 1: 2 epochs, lr 5e-5, replay 1:2 (null: step 0 kept)"
    return {
        "zero-shot": "zero-shot (base model, non-thinking)",
        "sft": "SFT round 1 (LoRA r16)",
        "grpo": "SFT + GRPO, 1 epoch",
        "grpo-gated": "SFT + GRPO + difficulty-gated GRPO",
        "grpo-gated-leg2": "gated GRPO leg 2 (null: step 0 kept)",
        "sft-cliff": "cliff SFT (null: step 0 kept)",
        "grpo-gated-cliffleg": "gated GRPO from the cliff SFT's final adapter",
        "grpo-pilot": "GRPO pilot, 40 steps",
    }.get(cond, cond)


def run_note(run):
    notes = {
        "qwen3.5-4b-grpo-gated-leg2": "selection kept step 0: the scored adapter is qwen3.5-4b-grpo-gated/best re-measured (fla kernels)",
        "qwen3.5-4b-sft-cliff": "selection kept step 0: the scored adapter is qwen3.5-4b-grpo-gated/best re-measured (torch path)",
        "qwen3.5-4b-sft-cliff-gentle": "selection kept step 0: the scored adapter is qwen3.5-4b-grpo-gated/best re-measured (fla kernels)",
        "qwen3.5-9b-grpo-gated": "selection kept step 0: the scored adapter is qwen3.5-9b-grpo/best re-measured (fla kernels)",
        "qwen3.5-4b-grpo-gated/fla-kernels": "the served 4B adapter re-measured after the fused kernels were installed",
        "qwen3.5-9b-sft/bf16-base": "the QLoRA adapter moved onto the bf16 base (merge-and-serve delta)",
        "seed-coder-8b-sft/bf16-base": "the QLoRA adapter moved onto the bf16 base (merge-and-serve delta)",
        "qwen3.5-4b/2026-09-05_409-rows": "earlier 409-row test slice, before the 2026-09-08 freeze; not comparable",
        "qwen3.5-4b-grpo-gated": "the served 4B row (best step 750)",
        "qwen3.5-9b-grpo": "the served 9B row (best step 800)",
    }
    return notes.get(run)


def read_train_log(run):
    log = RUNS / run / "train.log"
    out = {"train_seconds": None, "peak_gpu_gb": None, "train_loss_final": None,
           "mean_token_accuracy_final": None, "entropy_final": None, "reward_final_window": None}
    if not log.exists():
        return out
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    m = re.findall(r"training (\d+) s, peak GPU memory ([\d.]+) GB", text)
    if m:
        out["train_seconds"] = int(m[-1][0])
        out["peak_gpu_gb"] = float(m[-1][1])
    rt = re.findall(r"'train_runtime': '([\d.e+]+)'", text)
    if rt and out["train_seconds"] is None:
        out["train_seconds"] = int(float(rt[-1]))          # last session's runtime
    tl = re.findall(r"'train_loss': '([\d.e+-]+)'", text)
    out["train_loss_final"] = float(tl[-1]) if tl else None
    acc = re.findall(r"'mean_token_accuracy': '([\d.e+-]+)'", text)
    out["mean_token_accuracy_final"] = float(acc[-1]) if acc else None
    ent = re.findall(r"'entropy': '([\d.e+-]+)'", text)
    out["entropy_final"] = float(ent[-1]) if ent else None
    rw = re.findall(r"'reward': '([\d.e+-]+)'", text)
    out["reward_final_window"] = float(rw[-1]) if rw else None
    return out


def per_case_results(run, split, cases_by_id, rescore):
    """{id: exec-log-like dict}; from exec_<split>.jsonl or by re-executing preds."""
    d = RUNS / run
    log = d / f"exec_{split}.jsonl"
    if log.exists():
        return {r["id"]: r for r in jl(log)}, "exec log"
    preds = d / f"preds_{split}.jsonl"
    if not preds.exists():
        return {}, None
    cache = CACHE / f"{run.replace('/', '__')}__exec_{split}.jsonl"
    if cache.exists():
        return {r["id"]: r for r in jl(cache)}, "re-scored (cached)"
    if not rescore:
        return {}, "preds only (re-scoring disabled)"
    pmap = {}
    for r in jl(preds):
        pmap[r["id"]] = {"command": r.get("command"), "mode": r.get("mode")}
    # a completion that failed the output grammar has no prediction row; it
    # is a parse failure (invalid, no match), scored as such
    for r in jl(d / f"completions_{split}.jsonl"):
        if r["id"] not in pmap and r.get("parse_ok") is False:
            pmap[r["id"]] = {"command": None, "mode": None, "parse_failure": True}
    cases = [cases_by_id[i] for i in pmap if i in cases_by_id]
    if not cases:
        return {}, None
    # parallel scoring: run the harness on 8 shards and merge per_case
    scorable = [c for c in cases if not pmap[c["id"]].get("parse_failure")]
    shards = [scorable[i::8] for i in range(8)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda sh: harness_score(sh, pmap) if sh else {"per_case": {}}, shards))
    per = {}
    for res in results:
        per.update(res["per_case"])
    for c in cases:
        if pmap[c["id"]].get("parse_failure"):
            per[c["id"]] = {"valid": False, "match": False, "error": "PARSE", "return_code": None,
                            "stderr": "output grammar failure (no `ex:`/`keys:` line)", "buffer_after": None,
                            "timeout": False, "mode": c.get("mode"), "command": None}
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    with cache.open("w", encoding="utf-8") as fh:
        for c in cases:
            e = per.get(c["id"])
            if e is None:
                continue
            rec = {"id": c["id"], "category": c.get("category")}
            rec.update(e)
            rec.setdefault("command", pmap[c["id"]]["command"])
            rec.setdefault("mode", pmap[c["id"]]["mode"] or c.get("mode"))
            rows.append(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {r["id"]: r for r in rows}, "re-scored %s" % TODAY


# --------------------------------------------------------------- the build


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO / "exports" / f"vimBot_data_{TODAY}.xlsx"))
    ap.add_argument("--no-rescore", action="store_true",
                    help="skip re-executing predictions for runs without an exec log")
    ap.add_argument("--no-rollouts", action="store_true", help="skip the GRPO rollout parquet")
    args = ap.parse_args()
    wb = Workbook()
    readme = []          # (tab, rows, description, sources)

    corpus, split_of = load_corpus()
    by_id = {r["id"]: r for r in corpus}
    print(f"corpus {len(corpus)} rows; splits {collections.Counter(split_of.values())}")

    # ---- derived columns for the corpus
    comps = build_components(corpus)
    comp_of, comp_size = {}, {}
    for k, comp in enumerate(sorted(comps, key=lambda c: (-len(c), min(corpus[i]['id'] for i in c))), start=1):
        for i in comp:
            comp_of[corpus[i]["id"]] = k
            comp_size[corpus[i]["id"]] = len(comp)

    # probes (train prompts) — p-hat per probe file
    probe_files = [
        ("phat_4b_gated_probe_0914", RUNS / "grpo_gated_2026-09-14" / "difficulty_probe_qwen3.5-4b.jsonl"),
        ("phat_9b_gated_probe_0915", RUNS / "grpo_gated_2026-09-14" / "difficulty_probe_qwen3.5-9b.jsonl"),
        ("phat_4b_leg2_probe_0917", RUNS / "grpo_gated_2026-09-14" / "difficulty_probe_qwen3.5-4b-leg2.jsonl"),
        ("phat_4b_cliffleg_probe_0917", RUNS / "grpo_gated_2026-09-14" / "difficulty_probe_qwen3.5-4b-cliffleg.jsonl"),
        ("phat_cliff_before_0916", RUNS / "cliff_2026-09-16" / "probe_before_qwen3.5-4b.jsonl"),
        ("phat_cliff_after_0916", RUNS / "cliff_2026-09-16" / "probe_after_qwen3.5-4b.jsonl"),
        ("phat_cliff_final_0916", RUNS / "cliff_2026-09-16" / "probe_final_qwen3.5-4b.jsonl"),
        ("phat_cliff_cliffleg_0916", RUNS / "cliff_2026-09-16" / "probe_cliffleg_qwen3.5-4b.jsonl"),
        ("phat_cliff_gentle_after_0917", RUNS / "cliff_2026-09-17" / "probe_after_qwen3.5-4b.jsonl"),
    ]
    probes = {}
    probe_long = []
    for label, path in probe_files:
        rows = jl(path)
        if not rows:
            continue
        probes[label] = {r["id"]: r for r in rows}
        for r in rows:
            top = (r.get("commands") or [{}])[0]
            st = r.get("states") or {}
            probe_long.append([label, str(path.relative_to(REPO)), r.get("adapter"), r.get("time"),
                               r["id"], r.get("category"), r.get("tier"), r.get("mode"),
                               r.get("k"), r.get("G"), r.get("p_hat"), r.get("shelf"), r.get("fuel"),
                               st.get("match", 0), st.get("valid_wrong", 0), st.get("invalid", 0),
                               st.get("parse", 0), st.get("truncated", 0), r.get("distinct"),
                               r.get("tokens_mean"), top.get("mode"), top.get("command"), top.get("state"),
                               top.get("n"), len(r.get("commands") or []), r.get("temperature")])
    cliff_sets = {}
    for label, path in (("cliff_set_0916", RUNS / "cliff_2026-09-16" / "train_cliff_set_qwen3.5-4b.jsonl"),
                        ("cliff_set_0917_gentle", RUNS / "cliff_2026-09-17" / "train_cliff_set_qwen3.5-4b.jsonl")):
        cliff_sets[label] = {r["id"]: r.get("shelf") for r in jl(path)}
    known_sample = set(Path(RUNS / "cliff_2026-09-16" / "known_sample_qwen3.5-4b.txt").read_text().split()) \
        if (RUNS / "cliff_2026-09-16" / "known_sample_qwen3.5-4b.txt").exists() else set()
    reprobe = set(Path(RUNS / "cliff_2026-09-16" / "reprobe_ids_qwen3.5-4b.txt").read_text().split()) \
        if (RUNS / "cliff_2026-09-16" / "reprobe_ids_qwen3.5-4b.txt").exists() else set()
    cliff_cands = set(Path(RUNS / "cliff_2026-09-16" / "cliff_candidates_qwen3.5-4b.txt").read_text().split()) \
        if (RUNS / "cliff_2026-09-16" / "cliff_candidates_qwen3.5-4b.txt").exists() else set()

    # owner verdicts on the test slice
    verdict = {}
    for f in ("test_slice_verification_2026-09-03.json", "restored_review_2026-09-04.json",
              "test_slice_verification_2026-09-08.json"):
        d = jload(DATA / "fixes" / f, {}) or {}
        for i, v in (d.get("review") or {}).items():
            verdict[i] = (v.get("verdict") if isinstance(v, dict) else str(v), f)

    # ---- runs: registry, results table, per-case results
    runs = run_dirs()
    print("runs:", len(runs))
    cases_by_split = {s: {i: by_id[i] for i, sp in split_of.items() if sp == s} for s in ("train", "val", "test")}
    results_rows, registry_rows, curve_rows = [], [], []
    per_case = {}           # (run, split) -> {id: rec}
    completions = {}        # (run, split) -> {id: rec}
    row_meta = {}           # (run, split) -> summary json
    for run in runs:
        d = RUNS / run
        base_run = run.split("/")[0]
        cfg = jload(RUNS / base_run / "train_config.json", {}) or {}
        for split, fname in (("test", "baseline_test.json"), ("val", "sft_val.json"), ("val", "grpo_val.json")):
            js = jload(d / fname)
            if not js:
                continue
            row_meta[(run, split)] = js
            met = js.get("metrics", {})
            rmeta = js.get("run", {})
            # the val summaries of the GRPO trainer say "grpo" whatever the leg; the
            # test row (baseline.py --condition) carries the leg's name, so prefer it
            test_cond = ((jload(d / "baseline_test.json", {}) or {}).get("run") or {}).get("condition")
            cond = test_cond or rmeta.get("condition") or ("sft" if base_run.endswith("-sft") else "zero-shot")
            model = js.get("model")
            fam, params = model_meta(model or base_run)
            rdate = rmeta.get("date")
            kernel = ("fla (fused linear attention)" if (rdate and rdate > FLA_FROM) or "fla-kernels" in run
                      else "pure torch")        # the kernels went in the evening of 2026-09-16; that day's other rows predate them
            cases = cases_by_split.get(split, {})
            pc, src = per_case_results(run, split, cases if split != "test" or "409" not in run else by_id, not args.no_rescore)
            per_case[(run, split)] = pc
            comp = {r["id"]: r for r in jl(d / f"completions_{split}.jsonl")}
            completions[(run, split)] = comp
            gold_modes = {i: by_id[i]["mode"] for i in pc if i in by_id}
            keys_chosen = sum(1 for r in pc.values() if r.get("mode") == "keys")
            agrees = sum(1 for i, r in pc.items() if gold_modes.get(i) == r.get("mode"))
            keys_gold = [i for i in pc if gold_modes.get(i) == "keys"]
            kg_ex = [i for i in keys_gold if pc[i].get("mode") == "ex"]
            invalid = sum(1 for r in pc.values() if not r.get("valid"))
            vw = sum(1 for r in pc.values() if r.get("valid") and not r.get("match"))
            timeouts = sum(1 for r in pc.values() if r.get("error") == "TIMEOUT" or r.get("timeout"))
            blocked = sum(1 for r in pc.values() if r.get("blocked"))
            toks = [c.get("output_tokens") for c in comp.values()]
            lats = [c.get("latency_s") for c in comp.values()]
            at_cap = sum(1 for t in toks if t is not None and t >= js.get("decoding", {}).get("max_new_tokens", 192))
            versions = js.get("versions", {})
            best_step = None
            hist = jl(RUNS / base_run / "eval_history.jsonl")
            if hist:
                best = max(hist, key=lambda r: (r.get("exec_match_pct", 0), -r.get("step", 0)))
                best_step = best.get("step")
            results_rows.append([
                run, model, fam, params, cond, stage_label(cond, run), split, kernel,
                rmeta.get("base") or cfg.get("base") or ("nf4" if "QLoRA" in run else "bf16"),
                rmeta.get("adapter"), cfg.get("adapter_in"), best_step,
                js.get("n"), met.get("parse_failures"), met.get("think_leaks"),
                met.get("valid"), round(met["valid_pct"] / 100, 4) if met.get("valid_pct") is not None else None,
                met.get("exec_match"), round(met["exec_match_pct"] / 100, 4) if met.get("exec_match_pct") is not None else None,
                invalid if pc else None, vw if pc else None, timeouts if pc else None, blocked if pc else None,
                met.get("output_tokens_mean"), met.get("output_tokens_p50"), max(t for t in toks if t is not None) if any(t is not None for t in toks) else None,
                at_cap if comp else None, met.get("prompt_tokens_mean"),
                met.get("latency_s_p50"), met.get("latency_s_p95"), round(sum(l for l in lats if l), 1) if any(lats) else None,
                met.get("cost_per_solved_task_usd"),
                keys_chosen if pc else None, agrees if pc else None, len(keys_gold) if pc else None,
                len(kg_ex) if pc else None, sum(pc[i].get("match", False) for i in kg_ex) if pc else None,
                rdate, rmeta.get("host"), js.get("hf_id"), js.get("revision"),
                flat(js.get("chat_template_kwargs")), flat(js.get("decoding")),
                versions.get("torch"), versions.get("transformers"), versions.get("python"),
                versions.get("vim"), versions.get("gpu"), src, run_note(run),
                str(Path(d / fname).relative_to(REPO)),
            ])
        # registry: one row per training run folder (top-level only, once)
        if "/" in run:
            continue
        hist = jl(d / "eval_history.jsonl")
        best = max(hist, key=lambda r: (r.get("exec_match_pct", 0), -r.get("step", 0))) if hist else {}
        last = hist[-1] if hist else {}
        js = jload(d / "baseline_test.json", {}) or {}
        cond = (js.get("run") or {}).get("condition") or ("sft" if run.endswith("-sft") else "grpo" if cfg.get("grpo") else "zero-shot")
        if True:
            lora = cfg.get("lora", {})
            opt = cfg.get("optimization", {})
            grpo = cfg.get("grpo", {})
            gate = cfg.get("gating", {})
            data = cfg.get("data", {})
            ev = cfg.get("eval", {})
            rw = cfg.get("reward", {})
            tl = read_train_log(run)
            vjs = jload(d / "sft_val.json", None) or jload(d / "grpo_val.json", {}) or {}
            rm = (js.get("run") or {})
            fam, params = model_meta(cfg.get("model") or js.get("model") or run)
            resumes = sorted(p.name for p in d.glob("train_config.resume-*.json"))
            registry_rows.append([
                run, cfg.get("model") or js.get("model"), fam, params, cond, stage_label(cond, run),
                "grpo" if grpo else ("sft" if lora else "eval-only"),
                cfg.get("adapter_in") or (lora and "base model") or None,
                cfg.get("base") or rm.get("base"), (cfg.get("run") or {}).get("date") or rm.get("date"),
                (cfg.get("run") or {}).get("host") or rm.get("host"),
                lora.get("r"), lora.get("alpha"), lora.get("dropout"), flat(lora.get("target_modules")),
                lora.get("trainable_params"), lora.get("total_params"),
                round(lora["trainable_params"] / lora["total_params"] * 100, 3) if lora.get("trainable_params") and lora.get("total_params") else None,
                opt.get("lr"), opt.get("schedule"), opt.get("warmup_steps"), opt.get("epochs"), opt.get("max_steps"),
                opt.get("micro_batch"), opt.get("grad_accum"), opt.get("effective_batch"), opt.get("max_length"),
                opt.get("packing"), opt.get("completion_only_loss"), opt.get("seed"), opt.get("optim"),
                opt.get("gradient_checkpointing"),
                grpo.get("num_generations"), grpo.get("prompts_per_step"), grpo.get("completions_per_step"),
                grpo.get("max_completion_length"), grpo.get("temperature"), grpo.get("beta"), grpo.get("epsilon"),
                grpo.get("loss_type"), grpo.get("scale_rewards"), grpo.get("num_iterations"), flat(rw) if rw else None,
                gate.get("enabled", False), gate.get("pool"), gate.get("anchor_frac"), gate.get("cliff_frac"),
                gate.get("refresh_every"), gate.get("chunk_size"), flat(gate.get("initial_shelves")),
                flat(gate.get("difficulty_files")),
                data.get("train_rows") or data.get("train_prompts"), data.get("val_rows"), data.get("eval_subset"), data.get("subset_seed"),
                ev.get("every_steps"), ev.get("metric"),
                len(hist) or None, last.get("step"), best.get("step"), round(best.get("exec_match_pct", 0) / 100, 4) if best else None,
                round(hist[0].get("exec_match_pct", 0) / 100, 4) if hist else None, round(last.get("exec_match_pct", 0) / 100, 4) if last else None,
                round(vjs["metrics"]["exec_match_pct"] / 100, 4) if (vjs.get("metrics") or {}).get("exec_match_pct") is not None else None,
                round(js["metrics"]["exec_match_pct"] / 100, 4) if (js.get("metrics") or {}).get("exec_match_pct") is not None else None,
                tl["train_seconds"], round(tl["train_seconds"] / 3600, 2) if tl["train_seconds"] else None,
                round(tl["train_seconds"] / last["step"], 1) if tl["train_seconds"] and last.get("step") else None,
                tl["peak_gpu_gb"], tl["train_loss_final"], tl["mean_token_accuracy_final"], tl["entropy_final"],
                tl["reward_final_window"], len(resumes), ", ".join(resumes) or None,
                flat((cfg.get("versions") or js.get("versions"))), run_note(run), str(d.relative_to(REPO)),
            ])
        for r in hist:
            curve_rows.append([run, cfg.get("model") or js.get("model"), cond, r.get("step"), r.get("n"),
                               r.get("exec_match"), round((r.get("exec_match_pct") or 0) / 100, 4),
                               round((r.get("valid_pct") or 0) / 100, 4), r.get("parse_failures"), r.get("think_leaks"),
                               r.get("output_tokens_mean"), r.get("eval_seconds"),
                               r.get("step") == best.get("step"), r.get("step") == last.get("step")])

    # ---- Results table
    hdr = ["run", "model", "family", "params_B", "condition", "stage", "split", "kernel_path", "base_dtype",
           "adapter", "adapter_continued_from", "best_step", "n", "parse_failures", "think_leaks",
           "valid", "valid_pct", "exec_match", "exec_match_pct", "invalid", "valid_but_wrong", "timeouts", "blocked",
           "output_tokens_mean", "output_tokens_p50", "output_tokens_max", "completions_at_cap", "prompt_tokens_mean",
           "latency_p50_s", "latency_p95_s", "generation_seconds_total", "cost_per_solved_task_usd",
           "keys_mode_chosen", "mode_agrees_with_gold", "keys_gold_rows", "keys_gold_answered_in_ex",
           "keys_gold_answered_in_ex_and_right", "run_date", "host", "hf_id", "revision", "chat_template_kwargs",
           "decoding", "torch", "transformers", "python", "vim", "gpu", "per_case_source", "note", "source_file"]
    order = {"test": 0, "val": 1}
    results_rows.sort(key=lambda r: (r[1] or "", order.get(r[6], 2), r[37] or "", r[0]))
    wb.add_sheet("Results table", hdr, results_rows,
                 formats={"valid_pct": "pct", "exec_match_pct": "pct", "latency_p50_s": "float3", "latency_p95_s": "float3"},
                 tab_color="1F77B4")
    readme.append(("Results table", len(results_rows), "One row per scored (run, split): the RESULTS.md table with every measured field, the mode-choice counts and the kernel path.", "runs/*/baseline_test.json, sft_val.json, grpo_val.json, completions_*.jsonl, exec_*.jsonl"))

    # ---- Runs registry
    hdr = ["run", "model", "family", "params_B", "condition", "stage", "kind", "adapter_continued_from", "base_dtype",
           "date", "host", "lora_r", "lora_alpha", "lora_dropout", "lora_target_modules", "trainable_params",
           "total_params", "trainable_pct", "lr", "schedule", "warmup_steps", "epochs", "max_steps", "micro_batch",
           "grad_accum", "effective_batch", "max_length", "packing", "completion_only_loss", "seed", "optim",
           "gradient_checkpointing", "grpo_G", "grpo_prompts_per_step", "grpo_completions_per_step",
           "grpo_max_completion_length", "grpo_temperature", "grpo_beta_kl", "grpo_epsilon", "grpo_loss_type",
           "grpo_scale_rewards", "grpo_num_iterations", "reward_config", "gated", "gate_pool", "gate_anchor_frac",
           "gate_cliff_frac", "gate_refresh_every", "gate_chunk_size", "gate_initial_shelves", "gate_difficulty_files",
           "train_rows", "val_rows", "eval_subset", "subset_seed", "eval_every_steps", "eval_metric",
           "n_evals", "last_step", "best_step", "best_val_subset_pct", "step0_val_subset_pct", "last_val_subset_pct",
           "full_val_exec_match_pct", "test_exec_match_pct", "train_seconds", "train_hours", "seconds_per_step",
           "peak_gpu_gb", "train_loss_final", "mean_token_accuracy_final", "policy_entropy_final",
           "reward_final_window", "n_resumes", "resume_configs", "versions", "note", "folder"]
    wb.add_sheet("Runs registry", hdr, registry_rows,
                 formats={"best_val_subset_pct": "pct", "step0_val_subset_pct": "pct", "last_val_subset_pct": "pct",
                          "full_val_exec_match_pct": "pct", "test_exec_match_pct": "pct"}, tab_color="1F77B4")
    readme.append(("Runs registry", len(registry_rows), "One row per run folder: the full recipe (LoRA, optimizer, GRPO, gating), data sizes, curve summary, wall clock and peak memory.", "runs/*/train_config.json, eval_history.jsonl, train.log"))

    hdr = ["run", "model", "condition", "step", "n", "exec_match", "exec_match_pct", "valid_pct", "parse_failures",
           "think_leaks", "output_tokens_mean", "eval_seconds", "is_best", "is_last"]
    wb.add_sheet("Training curves", hdr, curve_rows, formats={"exec_match_pct": "pct", "valid_pct": "pct"}, tab_color="1F77B4")
    readme.append(("Training curves", len(curve_rows), "Every mid-training harness eval on the seeded 250-row val subset (greedy), for every SFT/GRPO run.", "runs/*/eval_history.jsonl"))

    # ---- per-case: derived corpus columns first (needed by the breakdowns)
    derived = {}
    for r in corpus:
        mode = r.get("mode", "ex")
        cmd = r.get("gold_command") or ""
        ex = parse_ex(cmd) if mode == "ex" else parse_ex(None)
        ks = parse_keys(cmd) if mode == "keys" else parse_keys(None)
        rf = regex_features(cmd, mode, ex)
        bd = buffer_diff(r.get("buffer_before"), r.get("buffer_after"))
        inf = instruction_features(r.get("instruction"))
        if mode == "ex":
            vclass = "ex:" + str(ex["ex_command"]) + ("/" + ex["ex_inner_command"] if ex.get("ex_inner_command") else "")
        else:
            vclass = "keys:" + str(ks["keys_family"]) + ("/" + ks["keys_ex_command"] if ks.get("keys_ex_command") else "")
        derived[r["id"]] = dict(vim_command_class=vclass, gold_chars=len(cmd),
                                gold_has_special_tokens=bool(_KEY_TOKEN.search(cmd)), **ex, **ks, **rf, **bd, **inf,
                                buffer_kind_heuristic=buffer_kind(r.get("buffer_before")))

    # ---- wide per-case sheets (test, val) and the long predictions sheet
    run_order = [r for r in runs]
    long_rows = []
    for split in ("test", "val"):
        ids = sorted(cases_by_split[split])
        cols = [(run, per_case[(run, split)], completions.get((run, split), {})) for run in run_order
                if per_case.get((run, split)) and "409" not in run]
        hdr = ["id", "category", "tier", "mode", "source", "dataset", "vim_command_class", "feature_bucket",
               "change_kind", "instruction", "gold_command", "n_runs", "n_runs_solved", "solved_frac",
               "solved_by_served_4b", "solved_by_served_9b", "solved_by_4b_zero_shot", "owner_verdict"]
        for run, _pc, _c in cols:
            hdr += [f"{run} · cmd", f"{run} · match", f"{run} · valid", f"{run} · error"]
        rows = []
        served4 = per_case.get(("qwen3.5-4b-grpo-gated", split), {})
        served9 = per_case.get(("qwen3.5-9b-grpo", split), {})
        zs4 = per_case.get(("qwen3.5-4b", split), {})
        for i in ids:
            c = by_id[i]
            dv = derived[i]
            n = sum(1 for _r, pc, _c in cols if i in pc)
            solved = sum(1 for _r, pc, _c in cols if pc.get(i, {}).get("match"))
            row = [i, c.get("category"), c.get("tier"), c.get("mode"), c.get("source"), c.get("dataset"),
                   dv["vim_command_class"], dv["feature_bucket"], dv["change_kind"], c.get("instruction"),
                   c.get("gold_command"), n, solved, pct(solved, n),
                   bool(served4.get(i, {}).get("match")) if i in served4 else None,
                   bool(served9.get(i, {}).get("match")) if i in served9 else None,
                   bool(zs4.get(i, {}).get("match")) if i in zs4 else None,
                   verdict.get(i, (None,))[0]]
            for run, pc, comp in cols:
                e = pc.get(i)
                if e is None:
                    row += [None, None, None, None]
                else:
                    row += [e.get("command"), bool(e.get("match")), bool(e.get("valid")), e.get("error") or None]
            rows.append(row)
        wb.add_sheet(f"{split.title()} cases x models", hdr, rows, formats={"solved_frac": "pct"}, tab_color="FF7F0E")
        readme.append((f"{split.title()} cases x models", len(rows), f"The {split} slice, one row per case, with every run's command / match / valid / vim error side by side plus how many runs solve it.", "runs/*/exec_*.jsonl (or re-scored preds), data/splits"))
        for run, pc, comp in [(run, per_case[(run, split)], completions.get((run, split), {})) for run in run_order if per_case.get((run, split))]:
            for i, e in pc.items():
                c = by_id.get(i, {})
                cm = comp.get(i, {})
                pmode = e.get("mode") or cm.get("mode")
                pcmd = e.get("command")
                pex = parse_ex(pcmd) if pmode == "ex" and pcmd else None
                pks = parse_keys(pcmd) if pmode == "keys" and pcmd else None
                pclass = ("ex:" + str(pex["ex_command"]) + ("/" + pex["ex_inner_command"] if pex.get("ex_inner_command") else "")) if pex else \
                         ("keys:" + str(pks["keys_family"])) if pks else None
                long_rows.append([run, split, i, c.get("category"), c.get("tier"), c.get("mode"), pmode,
                                  pmode == c.get("mode"), pcmd, cm.get("completion"), cm.get("parse_ok"), cm.get("think_leak"),
                                  bool(e.get("valid")), bool(e.get("match")), e.get("error") or None, e.get("return_code"),
                                  bool(e.get("timeout")), e.get("blocked"),
                                  cm.get("output_tokens"), cm.get("prompt_tokens"), cm.get("latency_s"),
                                  pclass, derived.get(i, {}).get("vim_command_class"),
                                  (pcmd or "") == (c.get("gold_command") or ""),
                                  len(pcmd) if pcmd else None, derived.get(i, {}).get("gold_chars"),
                                  (e.get("stderr") or "")[:500] or None, e.get("buffer_after")])
    hdr = ["run", "split", "id", "category", "tier", "gold_mode", "pred_mode", "mode_agrees", "pred_command",
           "raw_completion", "parse_ok", "think_leak", "valid", "match", "vim_error", "return_code", "timeout", "blocked",
           "output_tokens", "prompt_tokens", "latency_s", "pred_vim_command_class", "gold_vim_command_class",
           "pred_equals_gold_string", "pred_chars", "gold_chars", "stderr", "pred_buffer_after"]
    wb.add_sheet("Predictions (long)", hdr, long_rows, formats={"latency_s": "float3"}, tab_color="FF7F0E")
    readme.append(("Predictions (long)", len(long_rows), "Every prediction of every run on test and val: raw completion, parsed command, verdicts, vim error, tokens, latency and the buffer vim produced.", "runs/*/completions_*.jsonl + exec_*.jsonl"))

    # ---- breakdowns
    def breakdown(name, keyfn, label):
        rows = []
        for (run, split), pc in per_case.items():
            if not pc:
                continue
            comp = completions.get((run, split), {})
            groups = collections.defaultdict(list)
            for i, e in pc.items():
                if i not in by_id:
                    continue
                groups[keyfn(by_id[i], derived[i])].append(i)
            for key, ids in sorted(groups.items(), key=lambda kv: str(kv[0])):
                n = len(ids)
                v = sum(1 for i in ids if pc[i].get("valid"))
                m = sum(1 for i in ids if pc[i].get("match"))
                toks = [comp.get(i, {}).get("output_tokens") for i in ids]
                lats = [comp.get(i, {}).get("latency_s") for i in ids]
                rows.append([run, split] + list(key if isinstance(key, tuple) else (key,)) +
                            [n, v, m, pct(v, n), pct(m, n), n - v, v - m, mean(toks), median(lats)])
        hdr = ["run", "split"] + label + ["n", "valid", "match", "valid_pct", "match_pct", "invalid", "valid_but_wrong",
                                          "output_tokens_mean", "latency_p50_s"]
        wb.add_sheet(name, hdr, rows, formats={"valid_pct": "pct", "match_pct": "pct", "latency_p50_s": "float3"}, tab_color="2CA02C")
        readme.append((name, len(rows), f"Per-run, per-split scores broken down by {', '.join(label)}.", "computed from the per-case results"))

    breakdown("By category", lambda c, d: c.get("category"), ["category"])
    breakdown("By tier", lambda c, d: c.get("tier"), ["tier"])
    breakdown("By mode", lambda c, d: c.get("mode"), ["mode"])
    breakdown("By source", lambda c, d: c.get("source"), ["source"])
    breakdown("By dataset", lambda c, d: c.get("dataset"), ["dataset"])
    breakdown("By feature bucket", lambda c, d: d["feature_bucket"], ["feature_bucket"])
    breakdown("By vim command class", lambda c, d: d["vim_command_class"], ["vim_command_class"])
    breakdown("By change kind", lambda c, d: d["change_kind"], ["change_kind"])
    breakdown("By category x tier", lambda c, d: (c.get("category"), c.get("tier")), ["category", "tier"])
    breakdown("By category x mode", lambda c, d: (c.get("category"), c.get("mode")), ["category", "mode"])
    breakdown("By instruction form", lambda c, d: d["instruction_form"], ["instruction_form"])
    breakdown("By buffer size", lambda c, d: ("1-3 lines" if d["buffer_before_lines"] <= 3 else "4-8 lines" if d["buffer_before_lines"] <= 8 else "9-15 lines" if d["buffer_before_lines"] <= 15 else "16+ lines"), ["buffer_size_bucket"])
    breakdown("By provenance", lambda c, d: c.get("instruction_provenance"), ["instruction_provenance"])

    # category matrices: rows categories (+ tiers, modes), columns runs (match %)
    for split in ("test", "val"):
        split_runs = [run for run in run_order if per_case.get((run, split)) and "409" not in run]
        ids_all = sorted(cases_by_split[split])
        groups = [("category", c) for c in sorted({by_id[i]["category"] for i in ids_all})] + \
                 [("tier", t) for t in sorted({by_id[i]["tier"] for i in ids_all})] + \
                 [("mode", m) for m in sorted({by_id[i]["mode"] for i in ids_all})] + [("ALL", "ALL")]
        hdr = ["dimension", "value", "n"] + split_runs
        rows = []
        for dim, val in groups:
            ids = [i for i in ids_all if dim == "ALL" or by_id[i][dim] == val]
            row = [dim, val, len(ids)]
            for run in split_runs:
                pc = per_case[(run, split)]
                scored = [i for i in ids if i in pc]
                row.append(pct(sum(1 for i in scored if pc[i].get("match")), len(scored)))
            rows.append(row)
        wb.add_sheet(f"Category matrix ({split})", hdr, rows, formats={r: "pct" for r in split_runs}, tab_color="2CA02C", max_width=22)
        readme.append((f"Category matrix ({split})", len(rows), f"Exec-match on the {split} slice by category, tier and mode × runs, for a one-glance comparison.", "computed"))

    # paired deltas between every two runs scored on the same split
    import math
    rows = []
    for split in ("test", "val"):
        split_runs = [run for run in run_order if per_case.get((run, split)) and "409" not in run]
        for a in split_runs:
            for b in split_runs:
                if a == b:
                    continue
                A, B = per_case[(a, split)], per_case[(b, split)]
                ids = [i for i in A if i in B]
                if not ids:
                    continue
                am = sum(1 for i in ids if A[i].get("match")); bm = sum(1 for i in ids if B[i].get("match"))
                gained = sum(1 for i in ids if B[i].get("match") and not A[i].get("match"))
                lost = sum(1 for i in ids if A[i].get("match") and not B[i].get("match"))
                disc = gained + lost
                se = math.sqrt(disc) / len(ids) if disc else 0.0
                delta = (bm - am) / len(ids)
                ident = sum(1 for i in ids if (A[i].get("command") or "") == (B[i].get("command") or ""))
                rows.append([split, a, b, len(ids), am, bm, pct(am, len(ids)), pct(bm, len(ids)), round(delta, 4), gained, lost,
                             round(se, 4), round(delta / se, 2) if se else None, ident, pct(ident, len(ids))])
    wb.add_sheet("Paired deltas", ["split", "run_a", "run_b", "n", "a_match", "b_match", "a_match_pct", "b_match_pct", "delta_b_minus_a",
                                   "gained_by_b", "lost_by_b", "paired_se", "z_score", "identical_commands", "identical_frac"], rows,
                 formats={"a_match_pct": "pct", "b_match_pct": "pct", "delta_b_minus_a": "pct", "paired_se": "pct", "identical_frac": "pct"}, tab_color="2CA02C")
    readme.append(("Paired deltas", len(rows), "Every ordered pair of runs on the same split: rows gained/lost, the paired standard error of the difference (sqrt(discordant)/n) and z, plus how many commands are byte-identical.", "computed"))

    # vim error codes and mode confusion per run
    err_rows, mode_rows = [], []
    for (run, split), pc in per_case.items():
        if not pc:
            continue
        errs = collections.Counter((e.get("error") or ("ok" if e.get("valid") else "none")) for e in pc.values())
        for code, n in errs.most_common():
            sample = next((e.get("command") for e in pc.values() if (e.get("error") or ("ok" if e.get("valid") else "none")) == code), None)
            rows_m = sum(1 for e in pc.values() if (e.get("error") or ("ok" if e.get("valid") else "none")) == code and e.get("match"))
            err_rows.append([run, split, code, n, pct(n, len(pc)), rows_m, sample])
        conf = collections.Counter((by_id[i]["mode"], e.get("mode")) for i, e in pc.items() if i in by_id)
        for (gm, pm), n in sorted(conf.items(), key=lambda kv: str(kv[0])):
            m = sum(1 for i, e in pc.items() if i in by_id and by_id[i]["mode"] == gm and e.get("mode") == pm and e.get("match"))
            v = sum(1 for i, e in pc.items() if i in by_id and by_id[i]["mode"] == gm and e.get("mode") == pm and e.get("valid"))
            mode_rows.append([run, split, gm, pm, n, v, m, pct(m, n)])
    wb.add_sheet("By vim error", ["run", "split", "error_code", "n", "share", "of_which_match", "example_command"], err_rows,
                 formats={"share": "pct"}, tab_color="2CA02C")
    readme.append(("By vim error", len(err_rows), "Per run: how the invalid predictions fail (vim E-codes, TIMEOUT, PARSE for grammar failures; 'ok' = executed cleanly).", "computed from exec logs"))
    wb.add_sheet("Mode confusion", ["run", "split", "gold_mode", "pred_mode", "n", "valid", "match", "match_pct"], mode_rows,
                 formats={"match_pct": "pct"}, tab_color="2CA02C")
    readme.append(("Mode confusion", len(mode_rows), "Per run: gold mode × predicted mode (the ex-for-keys drift), with how often the off-mode answers are right.", "computed"))

    # ---- Corpus sheet
    base_fields = ["id", "dataset", "category", "tier", "mode", "instruction", "gold_command", "buffer_before", "buffer_after"]
    other_fields = sorted({k for r in corpus for k in r} - set(base_fields) - {"cursor"})
    dkeys = list(next(iter(derived.values())).keys())
    probe_cols = list(probes.keys())
    hdr = (["id", "split", "dataset", "category", "tier", "mode", "instruction", "gold_command", "cursor_line", "cursor_col",
            "buffer_before", "buffer_after"] + other_fields +
           ["leak_component_id", "leak_component_size", "stratum", "owner_test_verdict"] + dkeys +
           probe_cols + ["shelf_4b_gated_probe_0914", "shelf_9b_gated_probe_0915", "in_cliff_set_0916", "in_cliff_set_0917_gentle",
                         "in_known_sample_0916", "in_reprobe_0916", "in_cliff_candidates_0916",
                         "n_runs_scored", "n_runs_solved", "solved_frac"])
    rows = []
    # per-case solve counts on test/val
    solve = collections.defaultdict(lambda: [0, 0])
    for (run, split), pc in per_case.items():
        if "409" in run:
            continue
        for i, e in pc.items():
            solve[i][0] += 1
            solve[i][1] += 1 if e.get("match") else 0
    for r in corpus:
        i = r["id"]
        cur = r.get("cursor") or [1, 1]
        dv = derived[i]
        row = [i, split_of.get(i, "unplaced"), r.get("dataset"), r.get("category"), r.get("tier"), r.get("mode"),
               r.get("instruction"), r.get("gold_command"), cur[0], cur[1], r.get("buffer_before"), r.get("buffer_after")]
        row += [flat(r.get(k)) for k in other_fields]
        row += [comp_of.get(i), comp_size.get(i), "|".join(stratum_of(r)), verdict.get(i, (None,))[0]]
        row += [flat(dv[k]) for k in dkeys]
        row += [probes[p].get(i, {}).get("p_hat") for p in probe_cols]
        row += [probes.get("phat_4b_gated_probe_0914", {}).get(i, {}).get("shelf"),
                probes.get("phat_9b_gated_probe_0915", {}).get(i, {}).get("shelf"),
                cliff_sets.get("cliff_set_0916", {}).get(i), cliff_sets.get("cliff_set_0917_gentle", {}).get(i),
                i in known_sample, i in reprobe, i in cliff_cands,
                solve[i][0] or None, solve[i][1] if solve[i][0] else None, pct(solve[i][1], solve[i][0]) if solve[i][0] else None]
        rows.append(row)
    wb.add_sheet("Corpus", hdr, rows, formats={"solved_frac": "pct"}, tab_color="9467BD", max_width=50)
    readme.append(("Corpus", len(rows), "All %d cases of data/combined_cases.jsonl with every original field, the split, leakage component, stratum, owner verdict, ~90 derived columns (ex-command structure, regex features, keystroke features, buffer-diff shape, instruction features) and the per-prompt difficulty probes." % len(rows), "data/combined_cases.jsonl, data/splits, probes"))

    # ---- Split counts / component summary
    rows = []
    dims = [("category", lambda c, d: c.get("category")), ("tier", lambda c, d: c.get("tier")),
            ("mode", lambda c, d: c.get("mode")), ("source", lambda c, d: c.get("source")),
            ("dataset", lambda c, d: c.get("dataset")), ("feature_bucket", lambda c, d: d["feature_bucket"]),
            ("vim_command_class", lambda c, d: d["vim_command_class"]), ("change_kind", lambda c, d: d["change_kind"]),
            ("instruction_provenance", lambda c, d: c.get("instruction_provenance")),
            ("instruction_form", lambda c, d: d["instruction_form"]), ("theme", lambda c, d: c.get("theme")),
            ("buffer_kind_heuristic", lambda c, d: d["buffer_kind_heuristic"]),
            ("keys_family", lambda c, d: d["keys_family"]), ("ex_command", lambda c, d: d["ex_command"]),
            ("ex_range_kind", lambda c, d: d["ex_range_kind"]), ("keys_visual", lambda c, d: d["keys_visual"]),
            ("category x tier", lambda c, d: f"{c.get('category')} {c.get('tier')}")]
    for dim, fn in dims:
        counts = collections.defaultdict(lambda: collections.Counter())
        for r in corpus:
            counts[fn(r, derived[r["id"]])][split_of.get(r["id"], "unplaced")] += 1
        for val, cnt in sorted(counts.items(), key=lambda kv: (-sum(kv[1].values()), str(kv[0]))):
            tot = sum(cnt.values())
            rows.append([dim, val, cnt.get("train", 0), cnt.get("val", 0), cnt.get("test", 0), cnt.get("unplaced", 0), tot,
                         pct(tot, len(corpus)), pct(cnt.get("test", 0), tot)])
    wb.add_sheet("Split counts", ["dimension", "value", "train", "val", "test", "unplaced", "total", "share_of_corpus", "test_share_of_value"],
                 rows, formats={"share_of_corpus": "pct", "test_share_of_value": "pct"}, tab_color="9467BD")
    readme.append(("Split counts", len(rows), "Corpus composition by every dimension × split (the pivot tables of data/stats.md and data/splits/report.md, recomputed).", "computed"))

    rows = []
    for k, comp in enumerate(sorted(comps, key=lambda c: (-len(c), min(corpus[i]['id'] for i in c))), start=1):
        ids = [corpus[i]["id"] for i in comp]
        cs = collections.Counter(by_id[i]["category"] for i in ids)
        rows.append([k, len(ids), collections.Counter(split_of.get(i) for i in ids).most_common(1)[0][0],
                     len({split_of.get(i) for i in ids}), len({by_id[i]["buffer_before"] for i in ids}),
                     len({by_id[i]["gold_command"] for i in ids}), len({by_id[i]["instruction"] for i in ids}),
                     ", ".join(f"{c}:{n}" for c, n in cs.most_common()), " ".join(ids[:12]) + (" …" if len(ids) > 12 else "")])
    wb.add_sheet("Leak components", ["component_id", "size", "split", "n_splits_touched", "distinct_buffers", "distinct_commands",
                                     "distinct_instructions", "categories", "ids"], rows, tab_color="9467BD")
    readme.append(("Leak components", len(rows), "Leakage components (union-find over shared buffers, commands, instruction hashes, paraphrase links); every component sits inside one split.", "scripts/build_splits.py"))

    # ---- Taxonomy
    rows = []
    tax = (yaml.safe_load((DATA / "taxonomy.yaml").read_text()) if yaml and (DATA / "taxonomy.yaml").exists() else {}) or {}
    cat_counts = collections.Counter((r["category"], r["tier"]) for r in corpus)
    cat_split = collections.Counter((r["category"], r["tier"], split_of.get(r["id"])) for r in corpus)
    for cat, spec in (tax.get("categories") or {}).items():
        for tier in ("T1", "T2", "T3"):
            q = (spec.get("quotas") or {}).get(tier)
            n = cat_counts.get((cat, tier), 0)
            rows.append([cat, tier, spec.get("mode"), (spec.get("description") or "").strip(), q, n, pct(n, q) if q else None,
                         cat_split.get((cat, tier, "train"), 0), cat_split.get((cat, tier, "val"), 0), cat_split.get((cat, tier, "test"), 0),
                         flat((spec.get("tier_examples") or {}).get(tier)), (tax.get("tiers") or {}).get(tier)])
    for cat in sorted({r["category"] for r in corpus} - set((tax.get("categories") or {}).keys())):
        for tier in ("T1", "T2", "T3"):
            n = cat_counts.get((cat, tier), 0)
            if n:
                rows.append([cat, tier, None, "(not in taxonomy.yaml)", None, n, None,
                             cat_split.get((cat, tier, "train"), 0), cat_split.get((cat, tier, "val"), 0), cat_split.get((cat, tier, "test"), 0), None, None])
    wb.add_sheet("Taxonomy", ["category", "tier", "mode", "description", "quota", "corpus_count", "fill_vs_quota",
                              "train", "val", "test", "tier_examples", "tier_definition"], rows, formats={"fill_vs_quota": "pct"}, tab_color="9467BD")
    readme.append(("Taxonomy", len(rows), "data/taxonomy.yaml (category × tier quotas, definitions, examples) against the corpus counts.", "data/taxonomy.yaml"))

    # ---- Command frequency
    freq = collections.defaultdict(list)
    for r in corpus:
        freq[(r["mode"], derived[r["id"]]["vim_command_class"])].append(r)
    rows = []
    for (mode, cls), rs in sorted(freq.items(), key=lambda kv: -len(kv[1])):
        cats = collections.Counter(x["category"] for x in rs)
        tiers = collections.Counter(x["tier"] for x in rs)
        cmds = collections.Counter(x["gold_command"] for x in rs)
        rows.append([mode, cls, len(rs), pct(len(rs), len(corpus)), len(cmds), len(cats),
                     ", ".join(f"{c}:{n}" for c, n in cats.most_common(6)),
                     tiers.get("T1", 0), tiers.get("T2", 0), tiers.get("T3", 0),
                     " ‖ ".join(c for c, _ in cmds.most_common(3)),
                     collections.Counter(split_of.get(x["id"]) for x in rs).get("test", 0)])
    wb.add_sheet("Command frequency", ["mode", "vim_command_class", "count", "share", "distinct_commands", "n_categories",
                                       "top_categories", "T1", "T2", "T3", "example_commands", "in_test"], rows, formats={"share": "pct"}, tab_color="9467BD")
    readme.append(("Command frequency", len(rows), "Gold commands grouped by their parsed vim class (ex command word / keys mechanism family).", "computed"))

    # ---- Difficulty: by prompt (wide) and probes (long)
    ids = sorted({i for p in probes.values() for i in p})
    hdr = ["id", "split", "category", "tier", "mode", "vim_command_class", "feature_bucket"] + probe_cols + \
          ["shelf_4b_gated_probe_0914", "shelf_9b_gated_probe_0915", "shelf_4b_cliffleg_probe_0917",
           "fuel_4b_gated_probe_0914", "fuel_9b_gated_probe_0915", "top_command_4b_probe", "top_state_4b_probe",
           "in_cliff_set_0916", "in_cliff_set_0917_gentle", "in_known_sample_0916", "in_reprobe_0916", "in_cliff_candidates_0916"] + \
          [f"visits_{run}" for run in ("qwen3.5-4b-grpo-gated", "qwen3.5-4b-grpo-gated-leg2", "qwen3.5-4b-grpo-gated-cliffleg", "qwen3.5-9b-grpo-gated")] + \
          [f"last_k_{run}" for run in ("qwen3.5-4b-grpo-gated", "qwen3.5-4b-grpo-gated-leg2", "qwen3.5-4b-grpo-gated-cliffleg", "qwen3.5-9b-grpo-gated")]
    visits, lastk, visit_rows = collections.defaultdict(collections.Counter), {}, []
    for run in ("qwen3.5-4b-grpo-gated", "qwen3.5-4b-grpo-gated-leg2", "qwen3.5-4b-grpo-gated-cliffleg", "qwen3.5-9b-grpo-gated"):
        for r in jl(RUNS / run / "difficulty_log.jsonl"):
            visits[run][r["id"]] += 1
            lastk[(run, r["id"])] = r.get("k")
            st = r.get("states") or {}
            c = by_id.get(r["id"], {})
            visit_rows.append([run, r.get("step"), r["id"], c.get("category"), c.get("tier"), c.get("mode"), r.get("source"),
                               r.get("k"), r.get("G"), pct(r.get("k", 0), r.get("G", 8)),
                               "cliff" if r.get("k") == 0 else "known" if r.get("k") == r.get("G") else "frontier",
                               st.get("match", 0), st.get("valid_wrong", 0), st.get("invalid", 0), st.get("parse", 0), r.get("distinct")])
    rows = []
    for i in ids:
        c = by_id.get(i, {})
        dv = derived.get(i, {})
        p4 = probes.get("phat_4b_gated_probe_0914", {}).get(i, {})
        p9 = probes.get("phat_9b_gated_probe_0915", {}).get(i, {})
        pc_ = probes.get("phat_4b_cliffleg_probe_0917", {}).get(i, {})
        top = (p4.get("commands") or [{}])[0]
        rows.append([i, split_of.get(i), c.get("category"), c.get("tier"), c.get("mode"), dv.get("vim_command_class"), dv.get("feature_bucket")] +
                    [probes[p].get(i, {}).get("p_hat") for p in probe_cols] +
                    [p4.get("shelf"), p9.get("shelf"), pc_.get("shelf"), p4.get("fuel"), p9.get("fuel"), top.get("command"), top.get("state"),
                     cliff_sets.get("cliff_set_0916", {}).get(i), cliff_sets.get("cliff_set_0917_gentle", {}).get(i),
                     i in known_sample, i in reprobe, i in cliff_cands] +
                    [visits[run].get(i, 0) for run in ("qwen3.5-4b-grpo-gated", "qwen3.5-4b-grpo-gated-leg2", "qwen3.5-4b-grpo-gated-cliffleg", "qwen3.5-9b-grpo-gated")] +
                    [lastk.get((run, i)) for run in ("qwen3.5-4b-grpo-gated", "qwen3.5-4b-grpo-gated-leg2", "qwen3.5-4b-grpo-gated-cliffleg", "qwen3.5-9b-grpo-gated")])
    wb.add_sheet("Difficulty by prompt", hdr, rows, tab_color="D62728")
    readme.append(("Difficulty by prompt", len(rows), "Train prompts × every probe: success rate p̂ (k of 8 samples at T=1) under each adapter, shelf, fuel, cliff-set membership, and how often each gated run visited the prompt.", "runs/grpo_gated_2026-09-14/difficulty_probe_*.jsonl, runs/cliff_2026-09-1*/probe_*.jsonl, runs/*/difficulty_log.jsonl"))
    wb.add_sheet("Difficulty probes (long)", ["probe", "file", "adapter", "time", "id", "category", "tier", "mode", "k", "G", "p_hat", "shelf",
                                              "fuel", "n_match", "n_valid_wrong", "n_invalid", "n_parse", "n_truncated", "distinct_commands",
                                              "tokens_mean", "top_mode", "top_command", "top_state", "top_n", "n_distinct_listed", "temperature"],
                 probe_long, tab_color="D62728")
    readme.append(("Difficulty probes (long)", len(probe_long), "Every probe row (one per prompt per probe) with the sample-state counts and the most frequent sampled command.", "same"))
    wb.add_sheet("Gated visits", ["run", "step", "id", "category", "tier", "mode", "source", "k", "G", "p_hat", "shelf_from_k",
                                  "n_match", "n_valid_wrong", "n_invalid", "n_parse", "distinct_commands"], visit_rows, tab_color="D62728")
    readme.append(("Gated visits", len(visit_rows), "The live difficulty log of the gated runs: every prompt visit during training with its k of G.", "runs/*/difficulty_log.jsonl"))

    rows = []
    for run in runs:
        g = jload(RUNS / run / "gate_history.json")
        if not g:
            continue
        for ref in g.get("refreshes", []):
            sh = ref.get("shelves") or {}
            pl = ref.get("planner") or {}
            dr = pl.get("drawn") or {}
            rows.append([run, ref.get("step"), sh.get("cliff"), sh.get("frontier"), sh.get("known"), sh.get("unprobed"),
                         ref.get("expected_fuel_frontier"), ref.get("expected_fuel_all"),
                         dr.get("frontier"), dr.get("known"), dr.get("cliff"), pl.get("starved"), pl.get("refreshes"),
                         flat(pl.get("passes"))])
        rows.append([run, "initial", *(g.get("initial_shelves", {}).get(k) for k in ("cliff", "frontier", "known", "unprobed")), None, None, None, None, None, None, None, None])
        rows.append([run, "final", *(g.get("final_shelves", {}).get(k) for k in ("cliff", "frontier", "known", "unprobed")), None, None,
                     *(g.get("planner", {}).get("drawn", {}).get(k) for k in ("frontier", "known", "cliff")), g.get("planner", {}).get("starved"), g.get("planner", {}).get("refreshes"), flat(g.get("planner", {}).get("passes"))])
    wb.add_sheet("Gate refreshes", ["run", "step", "cliff", "frontier", "known", "unprobed", "expected_fuel_frontier", "expected_fuel_all",
                                    "drawn_frontier", "drawn_known", "drawn_cliff", "starved", "refreshes", "passes"], rows, tab_color="D62728")
    readme.append(("Gate refreshes", len(rows), "Shelf sizes and draw counts at every refresh of the gated sampler.", "runs/*/gate_history.json"))

    # ---- GRPO telemetry + rollouts (parquet)
    tele_rows, roll_rows = [], []
    if pd is not None and not args.no_rollouts:
        idx_exact, idx_buf, idx_ins = {}, collections.defaultdict(set), collections.defaultdict(set)
        sources = corpus + jl(DATA / "dropped_cases.jsonl") + jl(RUNS / "cliff_2026-09-15" / "flagged_review_qwen3.5-4b.jsonl")
        for r in sources:
            if "instruction" not in r or "buffer_before" not in r:
                continue
            idx_exact.setdefault((r["instruction"], r["buffer_before"]), r["id"])
            idx_buf[r["buffer_before"]].add(r["id"])
            idx_ins[r["instruction"]].add(r["id"])
            if r.get("instruction_before_restore"):
                idx_exact.setdefault((r["instruction_before_restore"], r["buffer_before"]), r["id"])
        prm = re.compile(r"Instruction: (.*?)\nCursor: line (\d+), column (\d+)\nBuffer:\n<<<\n(.*?)>>>", re.S)
        matched = total = 0
        for run in runs:
            files = sorted(glob.glob(str(RUNS / run / "completions" / "*.parquet")))
            if not files:
                continue
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            pid = {}
            for p in df["prompt"].unique():
                m = prm.search(p)
                cid = None
                if m:
                    ins, buf = m.group(1), m.group(4)
                    cid = idx_exact.get((ins, buf))
                    if cid is None and len(idx_buf.get(buf, ())) == 1:
                        cid = next(iter(idx_buf[buf]))
                    if cid is None and len(idx_ins.get(ins, ())) == 1:
                        cid = next(iter(idx_ins[ins]))
                pid[p] = (cid, m.group(1)[:120] if m else p[:120])
            df["case_id"] = df["prompt"].map(lambda p: pid[p][0])
            df["instruction_head"] = df["prompt"].map(lambda p: pid[p][1])
            total += len(df); matched += int(df["case_id"].notna().sum())
            for step, g in df.groupby("step"):
                grp = g.groupby("prompt")
                nmatch = (g["state"] == "match").sum()
                allm = sum(1 for _, x in grp if (x["state"] == "match").all())
                nom = sum(1 for _, x in grp if not (x["state"] == "match").any())
                zero = sum(1 for _, x in grp if x["executor_reward"].nunique() == 1)
                dist = mean([x["completion"].nunique() for _, x in grp])
                tele_rows.append([run, int(step), len(g), grp.ngroups, round(float(g["executor_reward"].mean()), 4),
                                  round(float(g["executor_reward"].std()), 4), pct(int(nmatch), len(g)),
                                  pct(int((g["state"] == "valid_wrong").sum()), len(g)), pct(int((g["state"] == "invalid").sum()), len(g)),
                                  pct(int((g["state"] == "parse").sum()), len(g)), pct(int((~g["state"].isin(["match", "valid_wrong", "invalid", "parse"])).sum()), len(g)),
                                  pct(allm, grp.ngroups), pct(nom, grp.ngroups), pct(zero, grp.ngroups), dist,
                                  round(float(g["completion"].str.len().mean()), 1),
                                  pct(int(g["completion"].str.startswith("keys:").sum()), len(g)),
                                  round(float(g["advantage"].abs().mean()), 4)])
            for rec in df.itertuples(index=False):
                c = by_id.get(rec.case_id, {}) if rec.case_id else {}
                roll_rows.append([run, int(rec.step), rec.case_id, c.get("category"), c.get("tier"), c.get("mode"),
                                  rec.instruction_head, rec.completion, round(float(rec.executor_reward), 4), rec.state,
                                  round(float(rec.advantage), 4), rec.completion.startswith("keys:"),
                                  (rec.completion.split(":", 1)[1].strip() == c.get("gold_command")) if c and ":" in rec.completion else None])
        print(f"rollouts: {total} rows, {matched} matched to a case id")
    wb.add_sheet("GRPO telemetry", ["run", "step", "completions", "groups", "reward_mean", "reward_std", "match_frac", "valid_wrong_frac",
                                    "invalid_frac", "parse_frac", "other_state_frac", "all_match_groups_frac", "no_match_groups_frac",
                                    "zero_signal_groups_frac", "distinct_completions_per_group", "completion_chars_mean", "keys_share",
                                    "abs_advantage_mean"], tele_rows,
                 formats={k: "pct" for k in ("match_frac", "valid_wrong_frac", "invalid_frac", "parse_frac", "other_state_frac",
                                             "all_match_groups_frac", "no_match_groups_frac", "zero_signal_groups_frac", "keys_share")}, tab_color="D62728")
    readme.append(("GRPO telemetry", len(tele_rows), "Per logged step (every 5th) of every GRPO run: reward, state fractions, silent-group share, diversity, completion length, keys share.", "runs/*/completions/*.parquet"))
    wb.add_sheet("GRPO rollouts", ["run", "step", "case_id", "category", "tier", "gold_mode", "instruction_head", "completion", "reward",
                                   "state", "advantage", "is_keys", "equals_gold"], roll_rows, tab_color="D62728")
    readme.append(("GRPO rollouts", len(roll_rows), "Every logged rollout (64 completions per logged step): the sampled completion, executor reward, state and advantage, joined to its case id.", "runs/*/completions/*.parquet"))

    # ---- Cliff SFT sets
    rows = []
    for label, path in (("cliff_set_0916", RUNS / "cliff_2026-09-16" / "train_cliff_set_qwen3.5-4b.jsonl"),
                        ("cliff_set_0917_gentle", RUNS / "cliff_2026-09-17" / "train_cliff_set_qwen3.5-4b.jsonl")):
        for r in jl(path):
            rows.append([label, r["id"], r.get("shelf"), r.get("k_hat"), r.get("g_hat"), r.get("category"), r.get("tier"), r.get("mode"),
                         r.get("instruction"), r.get("gold_command"), r.get("source"), r.get("dataset"),
                         probes.get("phat_cliff_after_0916", {}).get(r["id"], {}).get("p_hat"),
                         probes.get("phat_cliff_final_0916", {}).get(r["id"], {}).get("p_hat"),
                         probes.get("phat_cliff_cliffleg_0916", {}).get(r["id"], {}).get("p_hat")])
    wb.add_sheet("Cliff SFT sets", ["set", "id", "role_shelf", "k_hat", "g_hat", "category", "tier", "mode", "instruction", "gold_command",
                                    "source", "dataset", "p_hat_after_round1", "p_hat_final_round1", "p_hat_after_cliffleg"], rows, tab_color="D62728")
    readme.append(("Cliff SFT sets", len(rows), "The supervised sets of the two cliff SFT rounds (cliff rows + replay rows) with what each row's p̂ became afterwards.", "runs/cliff_2026-09-16/17/train_cliff_set_*.jsonl"))

    # ---- Data fixes (every curation file, flattened)
    rows = []
    fix_files = sorted(glob.glob(str(DATA / "fixes" / "*.json"))) + [str(REPO / "quota_audit.json"), str(REPO / "authored.json")]
    for f in fix_files:
        d = jload(f)
        rel = str(Path(f).relative_to(REPO))
        dm = re.search(r"(\d{4}-\d{2}-\d{2})", rel)
        fdate = dm.group(1) if dm else None
        if isinstance(d, list):
            for item in d:
                rows.append([rel, fdate, "list", item.get("id") if isinstance(item, dict) else None, None, flat(item)])
            continue
        if not isinstance(d, dict):
            rows.append([rel, fdate, "value", None, None, flat(d)]); continue
        for section, val in d.items():
            if isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, dict):
                        for fld, x in v.items():
                            rows.append([rel, fdate, section, k, fld, flat(x)])
                    else:
                        rows.append([rel, fdate, section, k, None, flat(v)])
            elif isinstance(val, list):
                for item in val:
                    rows.append([rel, fdate, section, item.get("id") if isinstance(item, dict) else (item if isinstance(item, str) else None), None, flat(item)])
            else:
                rows.append([rel, fdate, section, None, None, flat(val)])
    wb.add_sheet("Data fixes", ["file", "date", "section", "id", "field", "value"], rows, tab_color="8C564B", max_width=50)
    readme.append(("Data fixes", len(rows), "Every curation change-set flattened: edits (id, field, new value), deletes with reasons, review verdicts, retargets, restores.", "data/fixes/*.json, quota_audit.json, authored.json"))

    # ---- Decisions timeline
    rows = []
    for name in ("DECISIONS.md",):
        text = (REPO / name).read_text(encoding="utf-8").split("\n")
        heads = [(n, l) for n, l in enumerate(text, start=1) if l.startswith("## ")]
        for k, (n, l) in enumerate(heads):
            end = heads[k + 1][0] - 1 if k + 1 < len(heads) else len(text)
            body = text[n:end]
            title = l[3:].strip()
            dm = re.match(r"(\d{4}-\d{2}-\d{2})\s*[—-]*\s*(.*)", title)
            rows.append([name, dm.group(1) if dm else None, dm.group(2) if dm else title, n, end - n, sum(len(x.split()) for x in body),
                         " ".join(x for x in body if x.strip())[:1500]])
    wb.add_sheet("Decisions", ["file", "date", "title", "line", "lines", "words", "opening_text"], rows, tab_color="8C564B", max_width=60)
    readme.append(("Decisions", len(rows), "Every DECISIONS.md entry: date, title, size, opening text.", "DECISIONS.md"))

    # ---- Models registry
    rows = []
    reg = (yaml.safe_load((REPO / "configs" / "models.yaml").read_text()) if yaml else {}) or {}
    for name, m in (reg.get("models") or {}).items():
        fam, params = model_meta(name)
        rows.append([name, fam, params, m.get("hf_id"), m.get("revision"), flat(m.get("chat_template_kwargs")), flat(m.get("stop")),
                     m.get("max_seq_len"), flat(m.get("lora_target_modules")), flat(m.get("ref_cost_per_mtok")), flat(m.get("vllm_extra_args")),
                     flat({k: v for k, v in m.items() if k not in ("hf_id", "revision", "chat_template_kwargs", "stop", "max_seq_len", "lora_target_modules", "ref_cost_per_mtok", "vllm_extra_args")}) or None])
    wb.add_sheet("Models registry", ["name", "family", "params_B", "hf_id", "revision", "chat_template_kwargs", "stop", "max_seq_len",
                                     "lora_target_modules", "ref_cost_per_mtok", "vllm_extra_args", "other"], rows, tab_color="8C564B")
    readme.append(("Models registry", len(rows), "configs/models.yaml, one row per registered model.", "configs/models.yaml"))

    # ---- Run logs
    rows = []
    for f in sorted(glob.glob(str(RUNS / "*" / "status.txt"))):
        rel = str(Path(f).relative_to(REPO))
        year = "2026"
        for line in Path(f).read_text(encoding="utf-8", errors="replace").split("\n"):
            m = re.match(r"(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2}) (.*)", line)
            if m:
                rows.append([rel, f"{year}-{m.group(1)}-{m.group(2)} {m.group(3)}", m.group(4)])
            elif line.strip():
                rows.append([rel, None, line])
    wb.add_sheet("Run logs", ["log", "timestamp", "message"], rows, tab_color="8C564B", max_width=100)
    readme.append(("Run logs", len(rows), "Every timestamped line of the driver status logs (launches, pauses, resumes, stage completions).", "runs/*/status.txt"))

    # ---- side datasets
    def dataset_sheet(name, path, desc, extra_derived=False):
        rs = jl(path)
        if not rs:
            return
        keys = []
        for r in rs:
            for k in r:
                if k not in keys:
                    keys.append(k)
        front = [k for k in ("id", "category", "tier", "mode", "instruction", "gold_command", "command", "buffer_before", "buffer_after", "cursor") if k in keys]
        keys = front + [k for k in keys if k not in front]
        hdr = list(keys)
        if extra_derived:
            hdr += ["vim_command_class", "feature_bucket", "change_kind", "buffer_before_lines", "in_current_corpus"]
        rows = []
        for r in rs:
            row = [flat(r.get(k)) for k in keys]
            if extra_derived:
                mode = r.get("mode", "ex")
                cmd = r.get("gold_command") or r.get("command") or ""
                ex = parse_ex(cmd) if mode == "ex" else None
                ks = parse_keys(cmd) if mode == "keys" else None
                vc = ("ex:" + str(ex["ex_command"])) if ex else ("keys:" + str(ks["keys_family"])) if ks else None
                bd = buffer_diff(r.get("buffer_before"), r.get("buffer_after"))
                row += [vc, feature_bucket(cmd, mode), bd["change_kind"], bd["buffer_before_lines"], r.get("id") in by_id]
            rows.append(row)
        wb.add_sheet(name, hdr, rows, tab_color="7F7F7F", max_width=50)
        readme.append((name, len(rows), desc, str(Path(path).relative_to(REPO))))

    dataset_sheet("Dropped cases", DATA / "dropped_cases.jsonl", "Cases dropped from the corpus by curation (with their drop reason).", True)
    dataset_sheet("Quarantine", DATA / "quarantine.jsonl", "Generation-time rejects (reject_reason) kept as future preference-pair negatives.", False)
    dataset_sheet("Quarantine review", DATA / "quarantine_review.jsonl", "Reviewed quarantine rows.", False)
    dataset_sheet("Seed cases", DATA / "seed_cases.jsonl", "The 14 hand-written seed cases (the schema reference).", True)
    dataset_sheet("v1 archived cases", DATA / "archive" / "v1-hand-authored" / "generated_cases.jsonl", "The archived v1 hand-authored set (354 cases, 15 categories).", True)
    dataset_sheet("v2 generated cases", DATA / "generated_cases.jsonl", "The v2 deep-regex-derived generation (superseded by v3/v4; the current corpus descends from it).", True)
    for name, path, desc in (("v3 cases", DATA / "v3" / "v3_cases.jsonl", "data/v3/v3_cases.jsonl as selected into the corpus (v3 dataset file)."),
                             ("v4 cases", DATA / "v4" / "v4_cases.jsonl", "data/v4/v4_cases.jsonl (v4 dataset file).")):
        if Path(path).exists():
            rs = jl(path)
            rows = [[r.get("id"), r.get("category"), r.get("tier"), r.get("mode"), r.get("source"), r.get("instruction_provenance"),
                     r.get("id") in by_id, split_of.get(r.get("id"))] for r in rs]
            wb.add_sheet(name, ["id", "category", "tier", "mode", "source", "instruction_provenance", "in_current_corpus", "split"], rows, tab_color="7F7F7F")
            readme.append((name, len(rows), desc + " Ids only — the full rows are in the Corpus tab.", str(Path(path).relative_to(REPO))))

    # ---- Regex conversions
    conv = jload(DATA / "external" / "deep-regex" / "conversions.json", {}) or {}
    rows = [[k, v.get("k"), v.get("v"), v.get("n"), v.get("exec"), v.get("pol")] for k, v in conv.items()] if isinstance(conv, dict) else []
    wb.add_sheet("Regex conversions", ["source_regex", "k", "vim_pattern", "note_or_normal_form", "exec", "policy"], rows, tab_color="7F7F7F", max_width=60)
    readme.append(("Regex conversions", len(rows), "The validated deep-regex → vim pattern conversion table (9,496 conversions, 0 failures).", "data/external/deep-regex/conversions.json"))

    # ---- README (first tab)
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    try:
        vimv = subprocess.run(["vim", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    except Exception:
        vimv = None
    intro = [
        ["vimBot — all project data", None, None, None],
        ["generated", datetime.now().strftime("%Y-%m-%d %H:%M"), None, None],
        ["repo commit", head, None, None],
        ["vim (the oracle)", vimv, None, None],
        ["corpus", f"{len(corpus)} cases; train {sum(1 for v in split_of.values() if v == 'train')} / val {sum(1 for v in split_of.values() if v == 'val')} / test {sum(1 for v in split_of.values() if v == 'test')}", None, None],
        ["runs scored", f"{len(results_rows)} (run, split) rows over {len(runs)} run folders", None, None],
        ["", None, None, None],
        ["metric definitions", "valid = parsed AND executed without a vim error / timeout (keys mode: process-exit only). exec_match = the produced buffer equals the expected buffer line for line (one trailing newline ignored). output_tokens = completion tokens, thinking excluded. latency = generate() wall clock per request at batch size 1 (transformers, not the serving layer). p_hat = successes of 8 samples at temperature 1.", None, None],
        ["percent columns", "hold fractions (0.521 = 52.1%) and are formatted as percentages", None, None],
        ["kernel path", "rows dated ≥ 2026-09-16 (and the fla-kernels folder) were measured with the fused linear-attention kernels; earlier rows on the pure-torch path. The two differ on ~10% of greedy completions (±0.5 points on the served adapters).", None, None],
        ["derived columns", "ex_*, keys_*, re_*, change_kind, buffer_kind_heuristic, instruction_* and vim_command_class are HEURISTIC parses of the command / buffer strings by exports/build_workbook.py, not project ground truth. category/tier/mode/split/verdicts and every measured number are authoritative.", None, None],
        ["re-scored rows", "runs from before 2026-09-10 have no exec log; their saved predictions were re-executed through the sandbox on %s (per_case_source column). Greedy decoding + a deterministic vim make this exact." % TODAY, None, None],
        ["rebuild", "conda run -n vimBot python exports/build_workbook.py", None, None],
        ["", None, None, None],
        ["tab", "rows", "what it holds", "source"],
    ]
    wb_readme = Workbook()
    wb_readme.add_sheet("README", ["item", "value", "what it holds", "source"], intro + [[t, n, d, s] for t, n, d, s in readme],
                        tab_color="000000", freeze=False, autofilter=False, max_width=110)
    # README first: rebuild the sheet list with README in front
    wb.sheets = wb_readme.sheets + wb.sheets
    wb._names |= wb_readme._names
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), {len(wb.sheets)} tabs")
    for t, n, _d, _s in readme:
        print(f"  {t:32s} {n:>8}")


if __name__ == "__main__":
    main()
