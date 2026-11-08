#!/usr/bin/env python3
"""Approach 1 of the distillation plan (DECISIONS.md 2026-10-31): sequence-level
distillation from the trained models' own verified answers.

The five trained models' train probes (G = 8 samples per prompt, T = 1) hold
their correct commands for every train row. A row keeps its gold unless a
command the models prefer over it is *equivalent* to it:
  - ranked by votes, i.e. samples across the five probes; the gold wins ties
    and ends the walk when it is reached;
  - re-executed now, it reproduces the row's buffer_after;
  - on two perturbed copies of buffer_before (the buffer duplicated after
    itself; four filler lines appended; cursor unchanged), it produces the
    gold's buffer.
The first such command replaces the gold, mode included. A command that
matches only on the original buffer is coincidental and never used: 13-18% of
the alternatives were, in the 2026-10-30 check.

Writes, into runs/distill_2026-10-31/:
  train_distilled.jsonl  the train split, schema unchanged; on replaced rows
                         mode and gold_command are the chosen command's and
                         keys_raw (the raw form of the old keys gold) is null
  targets.jsonl          one decision per row: the gold, the choice, every
                         candidate tried with its votes, voters and verdict
  targets_summary.json   counts by tier, category, reason and mode change

    python scripts/distill_targets/build_targets.py
"""
import argparse
import ast
import collections
import json
import pathlib
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from evaluate import _norm, load_jsonl                  # noqa: E402
from executor import run_vim                            # noqa: E402

RUN = REPO / "runs" / "distill_2026-10-31"
TRAIN = REPO / "data" / "splits" / "train.jsonl"
PROBES = {                                  # the trained models' train probes, G = 8, T = 1
    "4b": "runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b-leg2.jsonl",     # the served adapter
    "9b": "runs/qwen9b_methods_2026-10-19/train_probe_9b.jsonl",
    "seed-coder": "runs/fuel_probe_2026-09-26/difficulty_probe_seed-coder-8b.jsonl",
    "qwen-coder": "runs/fuel_probe_2026-09-26/difficulty_probe_qwen2.5-coder-7b.jsonl",
    "granite": "runs/fuel_probe_2026-09-26/difficulty_probe_granite-4.2-8b.jsonl",
}
FILLER = "alpha beta gamma\n\n    delta 42 epsilon\nzeta"
MAX_CANDIDATES = 5                          # alternatives tested above the gold, per row
TIMEOUT = 5.0


def commands(row):
    c = row["commands"]
    return ast.literal_eval(c) if isinstance(c, str) else c


def key(mode, command):
    command = command.strip()
    return mode, (command[1:] if mode == "ex" and command.startswith(":") else command)


def perturb(buf):
    """Two longer copies of the buffer; the original lines stay on top, so the
    cursor still points where it did."""
    nl = "\n" if buf.endswith("\n") else ""
    body = buf[:-1] if nl else buf
    return [body + "\n" + body + nl, body + "\n" + FILLER + nl]


def same(a, b):
    return _norm(a) == _norm(b)


def ranked(row_id, gold, probes):
    """Every correct command the probes sampled, as (key, votes, voters), most
    votes first; the gold wins a tie."""
    votes, voters = collections.Counter(), collections.defaultdict(list)
    for name, rows in probes.items():
        r = rows.get(row_id)
        if r is None:
            continue
        for c in commands(r):
            if c["state"] == "match" and c.get("command") is not None:
                k = key(c["mode"], c["command"])
                votes[k] += c["n"]
                voters[k].append(name)
    order = sorted(votes.items(), key=lambda kv: (-kv[1], kv[0] != gold))
    return [(k, n, voters[k]) for k, n in order]


def decide(row, ranking, run=run_vim, timeout=TIMEOUT):
    """The row's decision: {"choice": (mode, command) or None, "reason", "tried"}."""
    gold = key(row["mode"], row["gold_command"])
    cur = tuple(row.get("cursor") or (1, 1))
    above = []
    for k, n, who in ranking:
        if k == gold:
            break
        above.append((k, n, who))
    if not ranking:
        return {"choice": None, "reason": "never_solved", "tried": []}
    if not above:
        return {"choice": None, "reason": "gold_is_consensus", "tried": []}
    g0 = run(row["buffer_before"], row["gold_command"], mode=row["mode"], cursor=cur, timeout=timeout)
    if not same(g0["buffer_after"], row["buffer_after"]):
        return {"choice": None, "reason": "gold_fails_now", "tried": []}
    gold_out = []
    for b in perturb(row["buffer_before"]):
        g = run(b, row["gold_command"], mode=row["mode"], cursor=cur, timeout=timeout)
        ok = g["was_valid"] and not g.get("timeout") and not g.get("blocked")
        gold_out.append(g["buffer_after"] if ok else None)
    tried = []
    for (mode, command), n, who in above[:MAX_CANDIDATES]:
        entry = {"mode": mode, "command": command, "votes": n, "voters": who}
        r0 = run(row["buffer_before"], command, mode=mode, cursor=cur, timeout=timeout)
        if r0.get("timeout") or r0.get("blocked") or not same(r0["buffer_after"], row["buffer_after"]):
            tried.append(dict(entry, verdict="no_match_now"))
            continue
        seen, diverged = 0, False
        for b, g in zip(perturb(row["buffer_before"]), gold_out):
            if g is None:
                continue
            seen += 1
            c = run(b, command, mode=mode, cursor=cur, timeout=timeout)
            if not same(c["buffer_after"], g):
                diverged = True
                break
        verdict = "diverges" if diverged else ("equivalent" if seen else "undetermined")
        tried.append(dict(entry, verdict=verdict))
        if verdict == "equivalent":
            return {"choice": (mode, command), "reason": "equivalent", "tried": tried}
    return {"choice": None, "reason": "none_equivalent", "tried": tried}


def apply(row, decision):
    """The training row: the original unless a command was chosen."""
    out = dict(row)
    if decision["choice"] is not None:
        out["mode"], out["gold_command"] = decision["choice"]
        out["keys_raw"] = None
    return out


def summarize(rows, decisions):
    by = lambda f: collections.Counter(f(r, d) for r, d in zip(rows, decisions))     # noqa: E731
    changed = [(r, d) for r, d in zip(rows, decisions) if d["choice"] is not None]
    out = {"rows": len(rows), "replaced": len(changed),
           "reasons": dict(by(lambda r, d: d["reason"])),
           "replaced_by_tier": dict(collections.Counter(r["tier"] for r, _ in changed)),
           "rows_by_tier": dict(collections.Counter(r["tier"] for r in rows)),
           "replaced_by_category": dict(collections.Counter(r["category"] for r, _ in changed)),
           "mode_changes": dict(collections.Counter("%s->%s" % (r["mode"], d["choice"][0])
                                                    for r, d in changed if r["mode"] != d["choice"][0])),
           "verdicts": dict(collections.Counter(t["verdict"] for d in decisions for t in d["tried"])),
           "voters_of_choice": dict(collections.Counter(v for _, d in changed for v in d["tried"][-1]["voters"]))}
    if changed:
        out["chars_mean_gold_vs_choice"] = [round(statistics.fmean(len(r["gold_command"]) for r, _ in changed), 2),
                                            round(statistics.fmean(len(d["choice"][1]) for _, d in changed), 2)]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(RUN))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(str(TRAIN))
    probes = {name: {r["id"]: r for r in load_jsonl(str(REPO / p))} for name, p in PROBES.items()}
    missing = {name: sum(r["id"] not in p for r in rows) for name, p in probes.items()}
    print("train rows %d; rows missing per probe %s" % (len(rows), missing), flush=True)
    rankings = [ranked(r["id"], key(r["mode"], r["gold_command"]), probes) for r in rows]
    with ThreadPoolExecutor(args.workers) as pool:
        decisions = list(pool.map(lambda rr: decide(*rr), zip(rows, rankings)))
    with (out / "train_distilled.jsonl").open("w", encoding="utf-8") as fh:
        for r, d in zip(rows, decisions):
            fh.write(json.dumps(apply(r, d), ensure_ascii=False) + "\n")
    with (out / "targets.jsonl").open("w", encoding="utf-8") as fh:
        for r, d in zip(rows, decisions):
            fh.write(json.dumps({"id": r["id"], "tier": r["tier"], "category": r["category"],
                                 "gold": [r["mode"], r["gold_command"]], "keys_raw": r.get("keys_raw"),
                                 "choice": list(d["choice"]) if d["choice"] else None,
                                 "reason": d["reason"], "tried": d["tried"]}, ensure_ascii=False) + "\n")
    summary = summarize(rows, decisions)
    summary["probes"] = PROBES
    summary["rows_missing_per_probe"] = missing
    (out / "targets_summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
