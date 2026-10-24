#!/usr/bin/env python3
"""Verifier examples from difficulty probes: one row per (prompt, distinct
candidate command), labelled by the executor's verdict in the probe.

A probe row (scripts/probe_difficulty.py) holds every distinct command the
adapter sampled for one prompt, with its execution state and sample count.
Each becomes an example: label 1 if it exec-matched, 0 if it ran wrong or vim
rejected it; parse failures and truncations are dropped (a reranker only
ranks commands that parse). Several probes of the same prompt are merged: the
sample counts add, and the state is the most common one (ties to "match";
a disagreement can only come from a flaky timeout, and is counted).

For val and test, `--greedy` adds the served adapter's greedy (§6) answer
from a clarify verdicts file as a candidate flagged greedy=True, with its own
match verdict, so "rerank the 8 samples + the greedy answer" is measurable.

Split hygiene: every example's id must be in --cases, and the train build
refuses ids that appear in val or test.

    python scripts/reranker/build_examples.py --cases data/splits/train.jsonl \\
        --probe runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-leg2.jsonl \\
        --probe runs/cliff_2026-09-16/probe_before_qwen3.5-4b.jsonl \\
        --probe runs/cliff_recheck_2026-10-16/recheck_g32_qwen3.5-4b.jsonl \\
        --keep-unanimous 0.25 --max-per-prompt 8 --out runs/reranker_2026-10-16/train_examples.jsonl
    python scripts/reranker/build_examples.py --cases data/splits/val.jsonl \\
        --probe runs/sample_rule_2026-10-09/probe_val.jsonl \\
        --greedy runs/clarify_2026-10-06/val/verdicts_served.jsonl \\
        --out runs/reranker_2026-10-16/val_examples.jsonl
"""
import argparse
import collections
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from verifier_lib import LABEL_STATES                     # noqa: E402

HELD_OUT = (REPO / "data" / "splits" / "val.jsonl", REPO / "data" / "splits" / "test.jsonl")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def merge_probes(probe_rows):
    """{id: {(mode, command): {"n": samples, "states": Counter}}} plus G per id."""
    cands = collections.defaultdict(dict)
    total = collections.Counter()
    for r in probe_rows:
        total[r["id"]] += int(r.get("G", 0))
        for c in r.get("commands") or []:
            if not c.get("command") or c.get("state") not in LABEL_STATES:
                continue
            key = (c["mode"], c["command"])
            slot = cands[r["id"]].setdefault(key, {"n": 0, "states": collections.Counter()})
            slot["n"] += int(c.get("n", 1))
            slot["states"][c["state"]] += int(c.get("n", 1))
    return cands, total


def resolve_state(states):
    top = max(states.values())
    tied = [s for s, v in states.items() if v == top]
    return "match" if "match" in tied else sorted(tied)[0]


def build(cases, probe_rows, greedy_rows=None, keep_unanimous=1.0, max_per_prompt=None,
          forbid_ids=(), seed=20261016):
    by_id = {c["id"]: c for c in cases}
    cands, total = merge_probes(probe_rows)
    bad = [i for i in cands if i not in by_id]
    if bad:
        raise ValueError("%d probe ids are not in the cases file, e.g. %s" % (len(bad), bad[:3]))
    leaked = sorted(set(cands) & set(forbid_ids))
    if leaked:
        raise ValueError("%d ids belong to a held-out split, e.g. %s" % (len(leaked), leaked[:3]))
    greedy = {}
    for g in greedy_rows or []:
        d = g.get("direct") or {}
        if g["id"] in by_id and d.get("parse_ok") and d.get("command"):
            greedy[g["id"]] = (d["mode"], d["command"], int(bool(d.get("match"))))
    rng = random.Random(seed)
    stats = collections.Counter()
    rows = []
    for cid in sorted(set(cands) | set(greedy)):
        case = by_id[cid]
        items = []
        for (mode, cmd), slot in cands.get(cid, {}).items():
            st = resolve_state(slot["states"])
            if len(slot["states"]) > 1:
                stats["state_conflicts"] += 1
            items.append({"cand_mode": mode, "cand_command": cmd, "state": st,
                          "label": LABEL_STATES[st], "n": slot["n"], "greedy": False})
        if cid in greedy:
            gm, gc, gl = greedy[cid]
            hit = [it for it in items if (it["cand_mode"], it["cand_command"]) == (gm, gc)]
            if hit:
                hit[0]["greedy"] = True
                if hit[0]["label"] != gl:
                    stats["greedy_label_conflicts"] += 1
            else:
                items.append({"cand_mode": gm, "cand_command": gc, "state": "match" if gl else "wrong",
                              "label": gl, "n": 0, "greedy": True})
        if not items:
            continue
        if keep_unanimous < 1.0 and all(it["label"] == 1 for it in items) and not any(it["greedy"] for it in items):
            if rng.random() >= keep_unanimous:
                stats["unanimous_dropped"] += 1
                continue
        if max_per_prompt and len(items) > max_per_prompt:
            # keep every positive, then the most-sampled negatives
            pos = [it for it in items if it["label"] == 1]
            neg = sorted((it for it in items if it["label"] == 0), key=lambda it: (-it["n"], it["cand_command"]))
            kept = pos + neg[:max(0, max_per_prompt - len(pos))]
            stats["capped_candidates"] += len(items) - len(kept)
            items = kept
        for it in items:
            rows.append({"id": cid, "category": case.get("category"), "tier": case.get("tier"),
                         "gold_mode": case.get("mode"), "G": total.get(cid, 0), **it})
    stats["prompts"] = len({r["id"] for r in rows})
    stats["examples"] = len(rows)
    stats["positives"] = sum(r["label"] for r in rows)
    return rows, dict(stats)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", required=True)
    ap.add_argument("--probe", action="append", default=[], help="probe file(s), merged")
    ap.add_argument("--greedy", default=None, help="clarify verdicts file: adds the greedy answer")
    ap.add_argument("--keep-unanimous", type=float, default=1.0,
                    help="keep this fraction of prompts whose candidates are all right")
    ap.add_argument("--max-per-prompt", type=int, default=None)
    ap.add_argument("--train", action="store_true", help="refuse ids in val/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261016)
    args = ap.parse_args(argv)
    cases = load_jsonl(args.cases)
    probe_rows = [r for p in args.probe for r in load_jsonl(p)]
    forbid = set()
    if args.train:
        for f in HELD_OUT:
            forbid |= {r["id"] for r in load_jsonl(f)}
    rows, stats = build(cases, probe_rows, load_jsonl(args.greedy) if args.greedy else None,
                        args.keep_unanimous, args.max_per_prompt, forbid, args.seed)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats.update({"cases": args.cases, "probes": args.probe, "greedy": args.greedy,
                  "keep_unanimous": args.keep_unanimous, "max_per_prompt": args.max_per_prompt})
    out.with_suffix(".manifest.json").write_text(json.dumps(stats, indent=1) + "\n", encoding="utf-8")
    print("%d examples over %d prompts (%d positive) -> %s" % (
        stats["examples"], stats["prompts"], stats["positives"], out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
