#!/usr/bin/env python3
"""Shelve a case file by measured difficulty and write the pieces a training
round needs: the frontier (plus an anchor slice of solved prompts) as a GRPO
train file, the cliff for SFT round 2 or targeted authoring, and a summary
of where p̂ lives by category, tier and mode.

Inputs are difficulty files (see `difficulty.py`): the offline probe from
`probe_difficulty.py` and/or the live log a `train_grpo_gated.py` run writes
(`<out_dir>/difficulty_log.jsonl`). Observations are pooled per prompt over
the last M (`--pool last:M`, default 2) in the order the files are given.

The train file it writes is an ordinary §5 case file -- every row is a row of
--cases with three extra fields (`shelf`, `k_hat`, `g_hat`) that both GRPO
trainers ignore -- so the unmodified `scripts/train_grpo.py --train <file>` is
the zero-code version of gated training (one fixed selection, refreshed by
re-running this script between legs), and `train_grpo_gated.py` refreshes
the same selection in-process.

Usage:
    python scripts/select_frontier.py --cases data/splits/train.jsonl \\
        --difficulty runs/grpo_gated_2026-09-14/difficulty_probe.jsonl \\
        --out runs/grpo_gated_2026-09-14/train_frontier.jsonl \\
        --cliff-out runs/grpo_gated_2026-09-14/train_cliff.jsonl \\
        --summary runs/grpo_gated_2026-09-14/difficulty_summary.md
    python scripts/select_frontier.py ... --difficulty probe.jsonl runs/x/difficulty_log.jsonl
                                                        # refresh: the log's later rows win
"""
import argparse
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl                                    # noqa: E402
from difficulty import (DEFAULT_ANCHOR_FRAC, DEFAULT_POOL,         # noqa: E402
                        DifficultyBook)


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--difficulty", nargs="+", required=True,
                    help="difficulty files, in order; later observations pool after earlier ones")
    ap.add_argument("--pool", default=DEFAULT_POOL, help="latest | all | last:M (default %s)" % DEFAULT_POOL)
    ap.add_argument("--out", default=None, help="train file: frontier + anchors (+ unprobed, see --unprobed)")
    ap.add_argument("--cliff-out", default=None, help="the k = 0 prompts, for SFT round 2")
    ap.add_argument("--known-out", default=None, help="the k = G prompts, for the record")
    ap.add_argument("--anchor-frac", type=float, default=DEFAULT_ANCHOR_FRAC,
                    help="share of the train file drawn from the known shelf (default %.2f)" % DEFAULT_ANCHOR_FRAC)
    ap.add_argument("--unprobed", choices=("frontier", "drop"), default="frontier",
                    help="unobserved prompts go into the train file (frontier) or are left out")
    ap.add_argument("--seed", type=int, default=20260914, help="anchor sampling")
    ap.add_argument("--summary", default=None, help="write the shelf report as markdown")
    ap.add_argument("--ids-out", default=None,
                    help="write the frontier ids one per line (for probe_difficulty.py --ids)")
    return ap


def anchors_for(frontier_n, known, anchor_frac, rng):
    """Enough known prompts that they make up `anchor_frac` of the file."""
    if anchor_frac <= 0 or not known:
        return []
    if anchor_frac >= 1:
        return list(known)
    want = round(anchor_frac / (1 - anchor_frac) * frontier_n)
    return sorted(rng.sample(known, min(want, len(known))))


def shelve(cases, book, anchor_frac, unprobed, seed):
    """-> dict of shelf name -> case rows (annotated), plus the train rows."""
    shelves = book.shelves()
    rows = {name: [dict(cases[i], shelf=name, k_hat=book.pooled(cases[i]["id"])[0],
                        g_hat=book.pooled(cases[i]["id"])[1]) for i in idx]
            for name, idx in shelves.items()}
    rng = random.Random(seed)
    anchors = anchors_for(len(rows["frontier"]), shelves["known"], anchor_frac, rng)
    train = list(rows["frontier"])
    if unprobed == "frontier":
        train += rows["unprobed"]
    train += [dict(cases[i], shelf="anchor", k_hat=book.pooled(cases[i]["id"])[0],
                   g_hat=book.pooled(cases[i]["id"])[1]) for i in anchors]
    return rows, train, anchors


def write_jsonl(path, rows):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_summary(summary, rows, train, anchors, args):
    s = summary["shelves"]
    n = summary["prompts"]
    lines = ["# Difficulty shelves", "",
             "Cases `%s`; difficulty %s; pool `%s`." % (args.cases, ", ".join("`%s`" % d for d in args.difficulty),
                                                        summary["pool"]), "",
             "| shelf | prompts | share |", "|---|---|---|"]
    for name in ("cliff", "frontier", "known", "unprobed"):
        lines.append("| %s | %d | %.1f%% |" % (name, s[name], 100 * s[name] / n if n else 0))
    lines += ["", "Observed %d of %d prompts. Mean expected fuel (G−1)·p̂(1−p̂) over observed prompts: %s "
              "(1.75 for a prompt at p = 1/2)." % (summary["observed"], n,
                                                  "%.2f" % summary["expected_fuel_observed"]
                                                  if summary["expected_fuel_observed"] is not None else "n/a"), ""]
    hist = summary["histogram_k_of_8"]
    lines += ["| k of 8 | " + " | ".join(str(k) for k in hist) + " |",
              "|---|" + "---|" * len(hist),
              "| prompts | " + " | ".join(str(v) for v in hist.values()) + " |", ""]
    for key in ("category", "tier", "mode"):
        table = summary.get("by_" + key)
        if not table:
            continue
        lines += ["## By %s" % key, "",
                  "| %s | n | cliff | frontier | known | unprobed | mean p̂ |" % key,
                  "|---|---|---|---|---|---|---|"]
        for val, c in sorted(table.items(), key=lambda kv: -sum(kv[1].get(x, 0) for x in ("cliff", "frontier", "known", "unprobed"))):
            tot = sum(c.get(x, 0) for x in ("cliff", "frontier", "known", "unprobed"))
            p = ("%.2f" % (c["k_sum"] / c["G_sum"])) if c.get("G_sum") else "n/a"
            lines.append("| %s | %d | %d | %d | %d | %d | %s |" % (
                val, tot, c.get("cliff", 0), c.get("frontier", 0), c.get("known", 0), c.get("unprobed", 0), p))
        lines.append("")
    lines += ["## Output", "",
              "Train file: %d rows = %d frontier + %d unprobed + %d anchors (anchor share %.1f%%)." % (
                  len(train), len(rows["frontier"]),
                  len(rows["unprobed"]) if args.unprobed == "frontier" else 0, len(anchors),
                  100 * len(anchors) / len(train) if train else 0),
              "Cliff: %d rows. Known: %d rows." % (len(rows["cliff"]), len(rows["known"])), ""]
    return "\n".join(lines)


def main() -> int:
    args = build_arg_parser().parse_args()
    cases = load_jsonl(args.cases)
    book = DifficultyBook([c["id"] for c in cases], pool=args.pool)
    for path in args.difficulty:
        n = book.load(path)
        print("loaded %d observations from %s" % (n, path))
    if book.unknown_ids or book.malformed:
        print("  skipped %d rows for ids outside --cases, %d malformed" % (book.unknown_ids, book.malformed))
    rows, train, anchors = shelve(cases, book, args.anchor_frac, args.unprobed, args.seed)
    summary = book.summary(cases)
    s = summary["shelves"]
    print("shelves: cliff %d  frontier %d  known %d  unprobed %d  (of %d; pool %s)" % (
        s["cliff"], s["frontier"], s["known"], s["unprobed"], summary["prompts"], args.pool))
    if summary["expected_fuel_observed"] is not None:
        print("expected fuel over observed prompts %.2f; over the train file's observed rows %.2f" % (
            summary["expected_fuel_observed"],
            book.expected_fuel([book.index[r["id"]] for r in train]) or 0.0))
    if args.out:
        write_jsonl(args.out, train)
        print("train file: %d rows (%d frontier, %d unprobed, %d anchors) -> %s" % (
            len(train), len(rows["frontier"]), len(rows["unprobed"]) if args.unprobed == "frontier" else 0,
            len(anchors), args.out))
    if args.cliff_out:
        write_jsonl(args.cliff_out, rows["cliff"])
        print("cliff: %d rows -> %s" % (len(rows["cliff"]), args.cliff_out))
    if args.known_out:
        write_jsonl(args.known_out, rows["known"])
        print("known: %d rows -> %s" % (len(rows["known"]), args.known_out))
    if args.ids_out:
        pathlib.Path(args.ids_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.ids_out).write_text("".join(r["id"] + "\n" for r in rows["frontier"]), encoding="utf-8")
        print("frontier ids -> %s" % args.ids_out)
    if args.summary:
        pathlib.Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.summary).write_text(render_summary(summary, rows, train, anchors, args), encoding="utf-8")
        print("summary -> %s" % args.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
