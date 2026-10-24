#!/usr/bin/env python3
"""Stage 1 past its gate: leakage, the ranking, and a threshold on P(ask).

compare.py checks the gate fixed before the run. This reads the same
verdicts for three things the gate does not look at:

1. **Leakage into the held-out rows.** split.py holds noted rows out one by
   one, not by leakage component (build_splits.py), so a held-out row can
   share a buffer, a gold command or a near-identical instruction with a row
   the adapter trained on. Those rows are counted and compared, and left out
   of everything below: the rest are the *clean* held-out rows.
2. **P(ask) as a ranking** of where to ask: the AUROC of the first-token
   probability for *unstated* / *ambiguous* rows against the fair *model*
   rows, for this adapter and for the served one.
3. **A threshold on P(ask)** in place of greedy decoding's argmax. Every row
   was measured asked and unasked, so each threshold is scored exactly
   (policies.py). For an ask cost c the threshold is **picked on val** (the
   one maximizing exec-match - c * ask rate) and **read on the clean
   held-out rows**, so no held-out number is tuned on itself. It is the
   baseline stage 2 (GRPO with the same price on a question) has to beat.

    python scripts/ask_sft/analyze.py --run runs/ask_sft_2026-10-07

Without val verdicts yet, parts 1, 2 and the held-out sweep are written and
the val-picked thresholds are skipped.
"""
import argparse
import collections
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))
from evaluate import load_jsonl                          # noqa: E402
from build_splits import leakage_keys                    # noqa: E402
from policies import auroc, flips, mcnemar_p, mean       # noqa: E402
import compare                                           # noqa: E402

# 1.01 never asks; 0 always does
GRID = (1.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0)
COSTS = (0.1, 0.2, 0.3)
CLASSES = ("unstated", "ambiguous", "model")
FATES = {(0, 1): "fixed", (1, 0): "broke", (1, 1): "right both ways", (0, 0): "wrong both ways"}


def links(held, trained):
    """{held-out id: the key kinds it shares with a trained row}; clean rows are absent.

    Every leakage key counts, the paraphrase anchors included: a held-out
    row's own-id anchor can only meet a trained row's `paraphrase_of` link,
    since no held-out id is trained on."""
    owned = set()
    for c in trained:
        owned.update(leakage_keys(c))
    out = {}
    for c in held:
        kinds = sorted({k[0] for k in leakage_keys(c) if k in owned})
        if kinds:
            out[c["id"]] = kinds
    return out


def outcomes(v):
    """A verdict row as the policies read it: 0/1 outcomes and the score."""
    return {"id": v["id"], "noask": int(v["noask"]["match"]), "real": int(v["real"]["match"]),
            "direct": int(v["direct"]["match"]), "p": v["p_ask_norm"],
            "cls": v.get("note_class"), "chose": bool(v["chose_ask"])}


def at_threshold(rows, t):
    """(ask or not, 0/1 outcome) per row, asking iff P(ask) >= t."""
    ask = [r["p"] >= t for r in rows]
    return ask, [r["real"] if a else r["noask"] for r, a in zip(rows, ask)]


def pick(rows, cost, grid=GRID):
    """The grid threshold maximizing exec-match - cost * ask rate; a tie keeps fewer asks."""
    best = None
    for t in sorted(grid, reverse=True):
        ask, out = at_threshold(rows, t)
        u = mean(out) - cost * mean(ask)
        if best is None or u > best[0] + 1e-12:
            best = (u, t)
    return best[1]


def paired(a, b):
    g, l = flips(a, b)
    return "%+.1f (+%d/−%d, p %.3f)" % (100 * (mean(b) - mean(a)), g, l, mcnemar_p(g, l))


def cell(s):
    return str(s).replace("|", "\\|").replace("\n", " ⏎ ")


def class_rates(rows, ask):
    return ["%d/%d" % (sum(a for r, a in zip(rows, ask) if r["cls"] == c), sum(r["cls"] == c for r in rows))
            for c in CLASSES]


def sweep_lines(rows, served_direct, with_classes):
    """A row per threshold, plus greedy decoding and the oracle."""
    head = "| policy | asks | exec-match | vs its direct answer | vs the served direct answer |"
    rule = "|---|---|---|---|---|"
    if with_classes:
        head += "".join(" asks on *%s* |" % c for c in CLASSES)
        rule += "---|" * len(CLASSES)
    own_direct = [r["direct"] for r in rows]
    policies = [("P(ask) ≥ %s" % ("— never" if t > 1 else t), at_threshold(rows, t)) for t in GRID]
    greedy = [r["chose"] for r in rows]
    policies.insert(0, ("greedy decoding (its own choice)",
                        (greedy, [r["real"] if a else r["noask"] for r, a in zip(rows, greedy)])))
    helps = [bool(r["real"] and not r["noask"]) for r in rows]
    policies.append(("oracle", (helps, [max(r["real"], r["noask"]) for r in rows])))
    out = [head, rule]
    for name, (ask, res) in policies:
        line = "| %s | %d (%.1f%%) | %.1f%% | %s | %s |" % (
            name, sum(ask), 100 * mean(ask), 100 * mean(res), paired(own_direct, res),
            paired(served_direct, res))
        if with_classes:
            line += "".join(" %s |" % x for x in class_rates(rows, ask))
        out.append(line)
    return out


def ranking_lines(sets):
    """AUROC of P(ask) for each kind of noted row against the fair ones."""
    out = ["| rows | task | served adapter | this adapter |", "|---|---|---|---|"]
    for name, new, ref in sets:
        for task, pos in (("unstated vs fair", {"unstated"}),
                          ("unstated or ambiguous vs fair", {"unstated", "ambiguous"}),
                          ("ambiguous vs fair", {"ambiguous"})):
            vals = []
            for rows in (ref, new):
                sub = [r for r in rows if r["cls"] in pos | {"model"}]
                a = auroc([r["p"] for r in sub], [r["cls"] in pos for r in sub])
                vals.append("—" if a is None else "%.2f" % a)
            n_pos = sum(r["cls"] in pos for r in new)
            n_neg = sum(r["cls"] == "model" for r in new)
            out.append("| %s | %s (%d vs %d) | %s | %s |" % (name, task, n_pos, n_neg, *vals))
    return out


def asked_lines(verdicts, linked):
    """Every ask the adapter made of its own accord, with what it did."""
    asked = [v for v in verdicts if v["chose_ask"]]
    fate = lambda v: FATES[(int(v["noask"]["match"]), int(v["real"]["match"]))]   # noqa: E731
    counts = collections.Counter(fate(v) for v in asked)
    out = ["%d asks; against the answer it gives when it may not ask: %s." % (
               len(asked), ", ".join("%s %d" % (k, counts[k]) for k in FATES.values() if counts[k])), "",
           "| id | class | instruction | question | reply | turn 2 | unasked | fate |",
           "|---|---|---|---|---|---|---|---|"]
    order = {"fixed": 0, "broke": 1, "wrong both ways": 2, "right both ways": 3}
    for v in sorted(asked, key=lambda v: (order[fate(v)], v["id"])):
        out.append("| %s%s | %s | %s | %s | %s | `%s` %s | `%s` %s | %s |" % (
            v["id"], " (linked)" if v["id"] in linked else "", v.get("note_class"),
            cell(v["instruction"]), cell(v["question"]), cell(v["reply"]),
            cell(v["real"]["completion"]), "✓" if v["real"]["match"] else "✗",
            cell(v["noask"]["completion"]), "✓" if v["noask"]["match"] else "✗", fate(v)))
    return out


def leakage_lines(held_v, served_by_id, linked):
    kinds = collections.Counter(k for ks in linked.values() for k in ks)
    out = ["%d of %d held-out rows share a leakage key with a trained row (%s); %d are clean." % (
               len(linked), len(held_v), ", ".join("%s %d" % kv for kv in sorted(kinds.items())) or "none",
               len(held_v) - len(linked)), "",
           "| held-out rows | n | served direct | this adapter's direct | change |", "|---|---|---|---|---|"]
    for name, rows in (("linked", [v for v in held_v if v["id"] in linked]),
                       ("clean", [v for v in held_v if v["id"] not in linked])):
        if not rows:
            continue
        s = [int(served_by_id[v["id"]]["direct"]["match"]) for v in rows]
        n = [int(v["direct"]["match"]) for v in rows]
        out.append("| %s | %d | %d | %d | %s |" % (name, len(rows), sum(s), sum(n), paired(s, n)))
    return out


def threshold_lines(val, val_served, held, held_served):
    """For each cost: the threshold val picks, read on val and on the clean held-out rows."""
    out = ["| ask cost c | threshold picked on val | val: asks | val: exec-match vs served direct "
           "| held-out (clean): asks | exec-match vs its direct | asks on fair rows |",
           "|---|---|---|---|---|---|---|"]
    for c in COSTS:
        t = pick(val, c)
        va, vr = at_threshold(val, t)
        ha, hr = at_threshold(held, t)
        fair = [a for r, a in zip(held, ha) if r["cls"] == "model"]
        out.append("| %.1f | %s | %.1f%% | %.1f%%, %s | %.1f%% | %.1f%%, %s | %d/%d |" % (
            c, "never" if t > 1 else t, 100 * mean(va), 100 * mean(vr), paired(val_served, vr),
            100 * mean(ha), 100 * mean(hr), paired([r["direct"] for r in held], hr),
            sum(fair), len(fair)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="stage 1's RUN")
    ap.add_argument("--label", default="asksft", help="the checkpoint label in the verdicts")
    ap.add_argument("--eval-dir", help="default: <run>/eval")
    ap.add_argument("--heldout", help="default: <run>/heldout.jsonl")
    ap.add_argument("--trained", action="append",
                    help="a jsonl whose ids the adapter trained on (repeatable; "
                         "default: <run>/train_set.jsonl)")
    ap.add_argument("--train", default=str(REPO / "data" / "splits" / "train.jsonl"))
    ap.add_argument("--served-held", default=str(REPO / "runs" / "clarify_2026-10-06" / "diag"
                                                 / "verdicts_served.jsonl"))
    ap.add_argument("--served-val", default=str(REPO / "runs" / "clarify_2026-10-06" / "val"
                                                / "verdicts_served.jsonl"))
    ap.add_argument("--out", help="default: <run>/analysis.md")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    ev = pathlib.Path(args.eval_dir) if args.eval_dir else run / "eval"
    out_path = pathlib.Path(args.out) if args.out else run / "analysis.md"

    train = {c["id"]: c for c in load_jsonl(args.train)}
    trained_ids = set()
    for f in args.trained or [str(run / "train_set.jsonl")]:
        trained_ids |= {r["id"] for r in load_jsonl(f)}
    held_cases = load_jsonl(args.heldout or str(run / "heldout.jsonl"))
    assert not trained_ids & {c["id"] for c in held_cases}, "a held-out row was trained on"
    linked = links(held_cases, [train[i] for i in sorted(trained_ids)])

    held_v = load_jsonl(str(ev / "heldout" / ("verdicts_%s.jsonl" % args.label)))
    served_by_id = {v["id"]: v for v in load_jsonl(args.served_held)}
    clean_v = [v for v in held_v if v["id"] not in linked]
    held = [outcomes(v) for v in clean_v]
    held_served = [int(served_by_id[v["id"]]["direct"]["match"]) for v in clean_v]

    doc = ["# Stage 1 past its gate: leakage, the ranking, a threshold on P(ask)", "",
           "Generated by `scripts/ask_sft/analyze.py` from `%s` (label `%s`). The gate itself is "
           "`gate.md`." % (ev, args.label), "",
           "## 1. Leakage into the held-out rows", ""]
    doc += leakage_lines(held_v, served_by_id, linked)
    sec, _, _ = compare.section("Clean held-out rows", clean_v,
                                [served_by_id[v["id"]] for v in clean_v])
    doc += ["", "#" + sec[0]] + sec[1:]

    doc += ["", "## 2. P(ask) as a ranking of where to ask", "",
            "AUROC, 0.5 = chance. The served adapter asks nowhere, but its P(ask) still ranks.", ""]
    all_new = [outcomes(v) for v in held_v]
    all_ref = [outcomes(served_by_id[v["id"]]) for v in held_v]
    doc += ranking_lines([("all held-out", all_new, all_ref),
                          ("clean held-out", held, [outcomes(served_by_id[v["id"]]) for v in clean_v])])

    doc += ["", "## 3. The adapter's own asks (all held-out rows)", ""]
    doc += asked_lines(held_v, linked)

    doc += ["", "## 4. A threshold on P(ask) instead of greedy decoding", "",
            "Ask iff P(ask) ≥ t (`p_ask_norm`: the first-token probability, renormalized over "
            "ask / ex / keys). Every row was measured both ways, so these are exact.", "",
            "### Clean held-out rows (%d)" % len(held), ""]
    doc += sweep_lines(held, held_served, with_classes=True)

    val_path = ev / "val" / ("verdicts_%s.jsonl" % args.label)
    if val_path.exists():
        val_v = load_jsonl(str(val_path))
        val_served_by_id = {v["id"]: v for v in load_jsonl(args.served_val)}
        val = [outcomes(v) for v in val_v]
        val_served = [int(val_served_by_id[v["id"]]["direct"]["match"]) for v in val_v]
        doc += ["", "### Val (%d)" % len(val), ""]
        doc += sweep_lines(val, val_served, with_classes=False)
        doc += ["", "### The threshold picked on val, read on the clean held-out rows", "",
                "For each ask cost c, the grid threshold maximizing val exec-match − c · ask rate "
                "(a tie keeps fewer asks): the offline counterpart of stage 2's reward.", ""]
        doc += threshold_lines(val, val_served, held, held_served)
    else:
        doc += ["", "_No val verdicts at `%s` yet: the val sweep and the val-picked thresholds "
                "are skipped._" % val_path]

    out_path.write_text("\n".join(doc) + "\n", encoding="utf-8")
    print("wrote %s (%d clean of %d held-out rows; val %s)" % (
        out_path, len(held), len(held_v), "included" if val_path.exists() else "not yet"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
