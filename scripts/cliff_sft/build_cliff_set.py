#!/usr/bin/env python3
"""The data side of SFT round 2 on the cliff shelf (survey item #2, DECISIONS.md
2026-09-15): which prompts to teach, what to mix in so the policy is not
pulled off what it knows, and how to read the result.

Three subcommands, one per stage of scripts/cliff_sft/run_cliff.sh:

  candidates  from difficulty files (the offline probe of the adapter the
              round continues, plus the live log of the gated run that
              produced it), pool the observations the way the gated trainer
              does (`--pool last:2`) and write two id lists: the CLIFF
              candidates (pooled k = 0 -- never seen solved) and a seeded
              KNOWN sample (pooled k = G) that serves as the retention
              control. The union is re-probed on the input adapter before
              training and on the new adapter after it, so every number in
              the report is a paired before/after on the same prompts.

  build       with the re-probe as the last difficulty file (so it wins
              under the pooling window), write the SFT train file: every
              cliff row, plus a REPLAY slice of solved prompts (frontier
              and known, sampled uniformly so the mix follows the split's
              own category shape) sized as `--replay-frac` of the cliff.
              Rows carry `shelf` (cliff / replay), `k_hat`, `g_hat`, which
              the trainer ignores. Also an audit page: the cliff by
              category, tier and mode, and the prompts the policy answers
              WRONG CONSISTENTLY (one valid command in >= 6 of 8 samples):
              a confident, stable wrong answer is where a gold command or
              an instruction deserves a second look before it is taught.
              `--hold-flagged` keeps those rows OUT of the train file and
              writes them to a review file instead, so a suspect gold is
              not taught before someone has read it.

  report      pair `--before` and `--after` probe rows by id and write the
              round's page: conversion (cliff prompts that now have k >= 1,
              by category), retention (known-sample prompts still at k = G,
              mean p-hat before and after), the fuel the cliff will bring to
              the next gated leg, and the val / test rows of the new adapter
              beside the adapter it continued from.

Why a replay slice: on-policy RL forgets little because it moves the policy
little (it is implicitly the KL-minimal solution -- RL's Razor, ICLR 2026);
off-policy SFT on a narrow, hard subset is exactly the move that forgets.
The replay rows keep the update anchored; the harness selection on the val
subset is the second brake.

Stdlib only, plus `difficulty.DifficultyBook` and `evaluate.load_jsonl`.

Usage:
    python scripts/cliff_sft/build_cliff_set.py candidates \\
        --difficulty runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b.jsonl \\
                     runs/qwen3.5-4b-grpo-gated/difficulty_log.jsonl \\
        --cliff-ids-out runs/cliff_2026-09-15/cliff_candidates_qwen3.5-4b.txt \\
        --known-ids-out runs/cliff_2026-09-15/known_sample_qwen3.5-4b.txt \\
        --reprobe-ids-out runs/cliff_2026-09-15/reprobe_ids_qwen3.5-4b.txt
    python scripts/cliff_sft/build_cliff_set.py build \\
        --difficulty <probe> <log> runs/cliff_2026-09-15/probe_before_qwen3.5-4b.jsonl \\
        --out runs/cliff_2026-09-15/train_cliff_set_qwen3.5-4b.jsonl \\
        --audit runs/cliff_2026-09-15/cliff_audit_qwen3.5-4b.md
    python scripts/cliff_sft/build_cliff_set.py report \\
        --before runs/cliff_2026-09-15/probe_before_qwen3.5-4b.jsonl \\
        --after runs/cliff_2026-09-15/probe_after_qwen3.5-4b.jsonl \\
        --cliff-ids ... --known-ids ... --run-dir runs/qwen3.5-4b-sft-cliff \\
        --compare-dir runs/qwen3.5-4b-grpo-gated --out runs/cliff_2026-09-15/cliff_report_qwen3.5-4b.md
"""
import argparse
import collections
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from evaluate import load_jsonl                                    # noqa: E402
from difficulty import DEFAULT_POOL, DifficultyBook, fuel          # noqa: E402

DEFAULT_REPLAY_FRAC = 0.5
DEFAULT_KNOWN_SAMPLE = 400
DEFAULT_SEED = 20260915
CONSISTENT_MIN = 6          # one valid-wrong command in >= this many of the G samples


# ------------------------------------------------------------------ helpers

def read_ids(path):
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def write_ids(path, ids):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(i + "\n" for i in ids), encoding="utf-8")


def write_jsonl(path, rows):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_book(cases, difficulty, pool):
    book = DifficultyBook([c["id"] for c in cases], pool=pool)
    for path in difficulty:
        n = book.load(path)
        print("loaded %d observations from %s" % (n, path))
    if book.unknown_ids or book.malformed:
        print("  skipped %d rows for ids outside --cases, %d malformed" % (book.unknown_ids, book.malformed))
    return book


def latest_probe_rows(paths, wanted=None):
    """The last full probe row per id across the given files (rows that carry
    `commands`), for the audit. Later files and later rows win."""
    rows = {}
    for path in paths:
        p = pathlib.Path(path)
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                cid = row.get("id")
                if cid is None or "commands" not in row:
                    continue
                if wanted is not None and cid not in wanted:
                    continue
                rows[cid] = row
    return rows


def consistent_wrong(row, min_n=CONSISTENT_MIN):
    """The one valid-but-wrong command the policy gives in >= min_n samples,
    or None. Invalid commands and mixtures do not count: the signal is a
    stable answer the executor accepts and the gold disagrees with."""
    best = None
    for c in row.get("commands") or []:
        if c.get("state") == "valid_wrong" and c.get("n", 0) >= min_n:
            if best is None or c["n"] > best["n"]:
                best = c
    return best


def by_key(cases, ids, key):
    counts = collections.Counter()
    by_id = {c["id"]: c for c in cases}
    for cid in ids:
        counts[str(by_id.get(cid, {}).get(key, "?"))] += 1
    return counts


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return lines


# --------------------------------------------------------------- candidates

def cmd_candidates(args):
    cases = load_jsonl(args.cases)
    book = load_book(cases, args.difficulty, args.pool)
    shelves = book.shelves()
    cliff = [cases[i]["id"] for i in shelves["cliff"]]
    known = [cases[i]["id"] for i in shelves["known"]]
    rng = random.Random(args.seed)
    sample = sorted(rng.sample(known, min(args.known_sample, len(known))))
    s = book.counts()
    print("shelves: cliff %d  frontier %d  known %d  unprobed %d  (of %d; pool %s)" % (
        s["cliff"], s["frontier"], s["known"], s["unprobed"], len(cases), args.pool))
    print("cliff candidates %d; known sample %d of %d; re-probe set %d" % (
        len(cliff), len(sample), len(known), len(cliff) + len(sample)))
    write_ids(args.cliff_ids_out, cliff)
    write_ids(args.known_ids_out, sample)
    if args.reprobe_ids_out:
        write_ids(args.reprobe_ids_out, cliff + sample)
    for name, ids in (("cliff", cliff), ("known sample", sample)):
        top = by_key(cases, ids, "category").most_common(6)
        print("  %s by category: %s" % (name, ", ".join("%s %d" % kv for kv in top)))
    return 0


# -------------------------------------------------------------------- build

def build_rows(cases, book, replay_frac, seed):
    """-> (cliff rows, replay rows), annotated. Replay is a uniform sample of
    the solved shelves (frontier + known), so its category mix is the
    split's own rather than the cliff's."""
    shelves = book.shelves()

    def annotate(i, shelf):
        k, g, _ = book.pooled(cases[i]["id"])
        return dict(cases[i], shelf=shelf, k_hat=k, g_hat=g)

    cliff = [annotate(i, "cliff") for i in shelves["cliff"]]
    solved = sorted(shelves["frontier"] + shelves["known"])
    want = int(round(replay_frac * len(cliff))) if replay_frac > 0 else 0
    rng = random.Random(seed)
    picked = sorted(rng.sample(solved, min(want, len(solved)))) if want else []
    replay = [annotate(i, "replay") for i in picked]
    return cliff, replay


def render_audit(cases, book, cliff, replay, probe_rows, args, held=()):
    """`cliff` is every cliff row (taught or held); `held` the subset kept out."""
    by_id = {c["id"]: c for c in cases}
    cliff_ids = [r["id"] for r in cliff]
    s = book.counts()
    lines = ["# Cliff set for SFT round 2", "",
             "Cases `%s`; difficulty %s; pool `%s`; replay fraction %.2f; seed %d." % (
                 args.cases, ", ".join("`%s`" % d for d in args.difficulty), args.pool,
                 args.replay_frac, args.seed), "",
             "Shelves on the input adapter: cliff %d, frontier %d, known %d, unprobed %d of %d." % (
                 s["cliff"], s["frontier"], s["known"], s["unprobed"], len(cases)), "",
             "**Train file: %d rows = %d cliff + %d replay.**%s" % (
                 len(cliff) - len(held) + len(replay), len(cliff) - len(held), len(replay),
                 " Held out pending a label review (`--hold-flagged`): %d consistent-wrong cliff rows."
                 % len(held) if held else ""), ""]
    for key in ("category", "tier", "mode"):
        total = by_key(cases, [c["id"] for c in cases], key)
        cl = by_key(cases, cliff_ids, key)
        rp = by_key(cases, [r["id"] for r in replay], key)
        rows = [(val, total[val], cl.get(val, 0),
                 "%.1f%%" % (100 * cl.get(val, 0) / total[val]) if total[val] else "n/a", rp.get(val, 0))
                for val in sorted(total, key=lambda v: -total[v])]
        lines += ["## Cliff by %s" % key, ""] + md_table(
            [key, "split", "cliff", "cliff share of the cell", "replay"], rows) + [""]
    # Consistent wrong answers: label-review candidates.
    flagged = []
    for cid in cliff_ids:
        row = probe_rows.get(cid)
        if not row:
            continue
        c = consistent_wrong(row)
        if c is not None:
            case = by_id[cid]
            flagged.append((cid, case.get("category", "?"), case.get("instruction", ""),
                            "%s: %s" % (c.get("mode", "?"), c.get("command", "")), c["n"], row.get("G", "?"),
                            "%s: %s" % (case.get("mode", "?"), case.get("gold_command", ""))))
    lines += ["## Consistent wrong answers on the cliff (%d)" % len(flagged), "",
              "Cliff prompts where one executor-accepted command was given in at least %d of the G samples "
              "of the latest probe. The policy is sure and stable, and the gold disagrees: read the "
              "instruction and the gold before teaching the gold. Listed: up to %d." % (
                  CONSISTENT_MIN, args.max_flagged), ""]
    if flagged:
        rows = [(cid, cat, instr.replace("|", "\\|")[:90], mine.replace("|", "\\|")[:60],
                 "%s/%s" % (n, g), gold.replace("|", "\\|")[:60])
                for cid, cat, instr, mine, n, g, gold in flagged[:args.max_flagged]]
        lines += md_table(["id", "category", "instruction", "the policy's answer", "n", "gold"], rows)
    else:
        lines.append("(none, or the latest probe rows carry no per-command detail)")
    lines.append("")
    return "\n".join(lines), flagged


def cmd_build(args):
    cases = load_jsonl(args.cases)
    book = load_book(cases, args.difficulty, args.pool)
    cliff, replay = build_rows(cases, book, args.replay_frac, args.seed)
    probe_rows = latest_probe_rows(args.difficulty, wanted={r["id"] for r in cliff})
    held = []
    if args.hold_flagged:
        keep = []
        for r in cliff:
            row = probe_rows.get(r["id"])
            if row is not None and consistent_wrong(row) is not None:
                held.append(r)
            else:
                keep.append(r)
        cliff = keep
        if args.held_out:
            write_jsonl(args.held_out, held)
        print("held out %d consistent-wrong cliff rows pending a label review%s" % (
            len(held), " -> %s" % args.held_out if args.held_out else ""))
    rows = cliff + replay
    random.Random(args.seed).shuffle(rows)
    write_jsonl(args.out, rows)
    print("train file: %d rows = %d cliff + %d replay (%.2f of the cliff before any hold-out) -> %s" % (
        len(rows), len(cliff), len(replay), args.replay_frac, args.out))
    if args.cliff_ids_out:
        write_ids(args.cliff_ids_out, [r["id"] for r in cliff])
    if args.audit:
        text, flagged = render_audit(cases, book, cliff + held, replay, probe_rows, args, held=held)
        pathlib.Path(args.audit).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.audit).write_text(text, encoding="utf-8")
        print("audit -> %s (%d consistent wrong answers flagged for a label review)" % (args.audit, len(flagged)))
    return 0


# ------------------------------------------------------------------- report

def k_by_id(path):
    """id -> (k, G) from one probe file; the last row per id wins."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                out[row["id"]] = (int(row["k"]), int(row["G"]))
            except (ValueError, KeyError, TypeError):
                continue
    return out


def paired(before, after, ids):
    """[(id, k_before, k_after, G)] for ids present in both files."""
    rows = []
    for cid in ids:
        if cid in before and cid in after:
            (kb, gb), (ka, ga) = before[cid], after[cid]
            rows.append((cid, kb, ka, gb if gb == ga else min(gb, ga)))
    return rows


def conversion(pairs):
    """Cliff prompts (k_before = 0) by their k_after shelf."""
    c = collections.Counter()
    for _, kb, ka, g in pairs:
        if kb != 0:
            c["not_on_cliff_before"] += 1
        elif ka <= 0:
            c["still_cliff"] += 1
        elif ka >= g:
            c["now_known"] += 1
        else:
            c["now_frontier"] += 1
    return c


def retention(pairs):
    """Known-sample prompts (k_before = G) by their k_after shelf, plus mean p-hat."""
    c = collections.Counter()
    pb, pa = [], []
    for _, kb, ka, g in pairs:
        if not g:
            continue
        pb.append(kb / g)
        pa.append(ka / g)
        if kb < g:
            c["not_known_before"] += 1
        elif ka >= g:
            c["still_known"] += 1
        elif ka > 0:
            c["now_frontier"] += 1
        else:
            c["now_cliff"] += 1
    c["p_hat_before"] = sum(pb) / len(pb) if pb else None
    c["p_hat_after"] = sum(pa) / len(pa) if pa else None
    return c


def mean_fuel(pairs, which):
    vals = [fuel(kb if which == "before" else ka, g) for _, kb, ka, g in pairs if g]
    return sum(vals) / len(vals) if vals else None


def run_numbers(run_dir):
    """The val and test rows a run directory holds, if any."""
    out = {}
    d = pathlib.Path(run_dir) if run_dir else None
    if d is None:
        return out
    for name, key in (("sft_val.json", "val"), ("grpo_val.json", "val"), ("baseline_test.json", "test")):
        p = d / name
        if p.exists():
            try:
                m = json.loads(p.read_text(encoding="utf-8")).get("metrics", {})
            except ValueError:
                continue
            out[key] = {"exec_match_pct": m.get("exec_match_pct"), "valid_pct": m.get("valid_pct"),
                        "output_tokens_mean": m.get("output_tokens_mean"), "file": str(p)}
    hist = d / "eval_history.jsonl"
    if hist.exists():
        rows = [json.loads(l) for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()]
        out["curve"] = [(r["step"], r["exec_match_pct"]) for r in rows]
    sel = d / "best" / "selection.json"
    if sel.exists():
        out["best_step"] = json.loads(sel.read_text(encoding="utf-8")).get("step")
    return out


def render_report(cases, cliff_pairs, known_pairs, new, old, args):
    by_id = {c["id"]: c for c in cases}
    conv, ret = conversion(cliff_pairs), retention(known_pairs)
    n_cliff = sum(v for k, v in conv.items() if k != "not_on_cliff_before")
    moved = conv["now_frontier"] + conv["now_known"]
    lines = ["# SFT round 2 on the cliff: report", "",
             "Before `%s`, after `%s`; %d cliff prompts and %d known-sample prompts paired by id." % (
                 args.before, args.after, len(cliff_pairs), len(known_pairs)), ""]
    lines += ["## Conversion (cliff prompts: k = 0 before)", "",
              "**%d of %d cliff prompts (%.1f%%) now have at least one success**: %d on the frontier, "
              "%d solved 8 of 8; %d still on the cliff.%s" % (
                  moved, n_cliff, 100 * moved / n_cliff if n_cliff else 0, conv["now_frontier"],
                  conv["now_known"], conv["still_cliff"],
                  " (%d paired prompts were not on the cliff in the before-probe and are not counted.)"
                  % conv["not_on_cliff_before"] if conv["not_on_cliff_before"] else ""), "",
              "Mean fuel k(G−k)/G over the cliff prompts: before %.2f, after %.2f (a prompt at p = 1/2 "
              "gives 2.00 at G = 8) -- what these prompts will return to the next gated leg." % (
                  mean_fuel(cliff_pairs, "before") or 0.0, mean_fuel(cliff_pairs, "after") or 0.0), ""]
    percat = collections.defaultdict(collections.Counter)
    for cid, kb, ka, g in cliff_pairs:
        if kb != 0:
            continue
        cat = str(by_id.get(cid, {}).get("category", "?"))
        percat[cat]["n"] += 1
        if ka > 0:
            percat[cat]["moved"] += 1
        if ka >= g:
            percat[cat]["known"] += 1
    rows = [(cat, c["n"], c["moved"], "%.0f%%" % (100 * c["moved"] / c["n"]), c["known"])
            for cat, c in sorted(percat.items(), key=lambda kv: -kv[1]["n"])]
    lines += md_table(["category", "cliff", "now k ≥ 1", "share", "now 8 of 8"], rows) + [""]
    lines += ["## Retention (known sample: k = G before)", "",
              "**%d of %d still solved 8 of 8** (%.1f%%); %d slipped to the frontier, %d to the cliff. "
              "Mean p̂ before %.3f, after %.3f.%s" % (
                  ret["still_known"], ret["still_known"] + ret["now_frontier"] + ret["now_cliff"],
                  100 * ret["still_known"] / max(1, ret["still_known"] + ret["now_frontier"] + ret["now_cliff"]),
                  ret["now_frontier"], ret["now_cliff"], ret["p_hat_before"] or 0.0, ret["p_hat_after"] or 0.0,
                  " (%d paired prompts were not 8 of 8 before and are not counted.)" % ret["not_known_before"]
                  if ret["not_known_before"] else ""), "",
              "A slip to the frontier is one lost draw in eight at T = 1 and is mostly binomial noise; "
              "a slip to the cliff is forgetting.", ""]
    lines += ["## The adapter, on val and test", "",
              "| row | continued from (%s) | after SFT round 2 (%s) |" % (args.compare_dir or "-", args.run_dir or "-"),
              "|---|---|---|"]
    for key, label in (("val", "full val exec-match"), ("test", "test exec-match")):
        a, b = old.get(key), new.get(key)
        lines.append("| %s | %s | %s |" % (
            label, "%.1f%% (valid %.1f%%, %.1f tok)" % (a["exec_match_pct"], a["valid_pct"], a["output_tokens_mean"])
            if a and a.get("exec_match_pct") is not None else "n/a",
            "%.1f%% (valid %.1f%%, %.1f tok)" % (b["exec_match_pct"], b["valid_pct"], b["output_tokens_mean"])
            if b and b.get("exec_match_pct") is not None else "not yet measured"))
    if new.get("curve"):
        lines += ["", "Val-subset curve of the round (step: exec-match): " + ", ".join(
            "%s: %.1f%%" % (s, v) for s, v in new["curve"]) + "."]
        if new.get("best_step") == 0:
            lines += ["", "**Selection kept step 0** -- no later checkpoint beat the input adapter's own "
                      "subset score, which was itself the maximum of the gated run's checkpoints (an inflated "
                      "bar). `best/` is then the input adapter re-saved and the test row reproduces its number; "
                      "read the conversion and retention tables, and consider `final/` for the next leg."]
    lines.append("")
    return "\n".join(lines)


def cmd_report(args):
    cases = load_jsonl(args.cases)
    before, after = k_by_id(args.before), k_by_id(args.after)
    cliff_ids, known_ids = read_ids(args.cliff_ids), read_ids(args.known_ids)
    cliff_pairs, known_pairs = paired(before, after, cliff_ids), paired(before, after, known_ids)
    new, old = run_numbers(args.run_dir), run_numbers(args.compare_dir)
    text = render_report(cases, cliff_pairs, known_pairs, new, old, args)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(text, encoding="utf-8")
    conv = conversion(cliff_pairs)
    print("report -> %s: cliff %d -> frontier %d, known %d, still cliff %d; known sample still 8/8: %d of %d"
          % (args.out, len(cliff_pairs), conv["now_frontier"], conv["now_known"], conv["still_cliff"],
             retention(known_pairs)["still_known"], len(known_pairs)))
    return 0


# ---------------------------------------------------------------------- CLI

def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("candidates", help="cliff candidate ids + a known sample, for the before-probe")
    c.add_argument("--cases", default=str(REPO / "data" / "splits" / "train.jsonl"))
    c.add_argument("--difficulty", nargs="+", required=True, help="difficulty files, earliest first")
    c.add_argument("--pool", default=DEFAULT_POOL)
    c.add_argument("--known-sample", type=int, default=DEFAULT_KNOWN_SAMPLE, help="retention control size")
    c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c.add_argument("--cliff-ids-out", required=True)
    c.add_argument("--known-ids-out", required=True)
    c.add_argument("--reprobe-ids-out", default=None, help="cliff + known sample, one file for the probe")
    c.set_defaults(func=cmd_candidates)

    b = sub.add_parser("build", help="the SFT train file: cliff + replay, and the audit page")
    b.add_argument("--cases", default=str(REPO / "data" / "splits" / "train.jsonl"))
    b.add_argument("--difficulty", nargs="+", required=True,
                   help="difficulty files, earliest first; give the before-probe LAST so it wins")
    b.add_argument("--pool", default=DEFAULT_POOL)
    b.add_argument("--replay-frac", type=float, default=DEFAULT_REPLAY_FRAC,
                   help="replay rows as a fraction of the cliff rows (0 disables)")
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--out", required=True, help="the §5-shaped train file")
    b.add_argument("--cliff-ids-out", default=None, help="the cliff ids actually taught")
    b.add_argument("--audit", default=None, help="markdown audit page")
    b.add_argument("--max-flagged", type=int, default=60)
    b.add_argument("--hold-flagged", action="store_true",
                   help="leave the consistent-wrong cliff rows OUT of the train file (label review first)")
    b.add_argument("--held-out", default=None, help="where the held rows go (JSONL), with --hold-flagged")
    b.set_defaults(func=cmd_build)

    r = sub.add_parser("report", help="paired before/after conversion and retention, plus the rows")
    r.add_argument("--cases", default=str(REPO / "data" / "splits" / "train.jsonl"))
    r.add_argument("--before", required=True, help="probe of the re-probe ids on the input adapter")
    r.add_argument("--after", required=True, help="probe of the same ids on the new adapter")
    r.add_argument("--cliff-ids", required=True)
    r.add_argument("--known-ids", required=True)
    r.add_argument("--run-dir", default=None, help="runs/<model>-sft-cliff (val/test rows, curve)")
    r.add_argument("--compare-dir", default=None, help="the continued adapter's run dir (its rows)")
    r.add_argument("--out", required=True, help="markdown report")
    r.set_defaults(func=cmd_report)
    return ap


def main() -> int:
    args = build_arg_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
