#!/usr/bin/env python3
"""The clarification experiment's report: report.md and summary.json.

Reads the verdict files score.py writes (one per checkpoint) and answers, per
checkpoint, the four questions the README poses:

1. Does a clarifying question help at all? -- always asking with the
   simulated user's reply, against the §6 direct answer.
2. Is it the information or the extra turn? -- the same question answered
   with a placebo that says nothing.
3. Does the model know *when* to ask? -- its own greedy choice, and its
   P(ask) ranking at fixed question budgets, against random asking at the
   same rate and the oracle.
4. Is the simulated user handing over the answer? -- the leak screen's
   counts, and the headline re-read with every flagged reply excluded.

Pure computation over the verdict files; nothing is executed or generated.
"""
import argparse
import collections
import datetime
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl                                   # noqa: E402
from policies import (at_rate, auroc, flips, mcnemar_p, mean,    # noqa: E402
                      oracle, paired_se, policy_match, random_at_rate,
                      threshold_curve)
from protocol import ASK_CLAUSE, PLACEBO_REPLY, QUESTION_MAX_TOKENS   # noqa: E402
from audit_sample import summarize                                # noqa: E402

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


def pct(x):
    return "%.1f%%" % (100 * x)


def policy_rows(rows):
    """Verdicts as the policies module reads them: noask and real as 0/1, and
    the ranking score, the log-odds of asking (dialogue.first_token_view)."""
    return [{"id": r["id"], "noask": int(r["noask"]["match"]),
             "real": int(r["real"]["match"]), "score": r["ask_margin"]} for r in rows]


def own_choice(rows):
    """Exec-match, ask rate and mean tokens of the model's greedy choice."""
    asked = {r["id"] for r in rows if r["chose_ask"]}
    m = policy_match(policy_rows(rows), asked)
    toks = mean(r["real"]["tokens"] if r["id"] in asked else r["noask"]["tokens"] for r in rows)
    return m, len(asked) / len(rows), toks, asked


def headline(rows):
    """The condition table for one checkpoint, as (name, what, values) rows."""
    direct = [int(r["direct"]["match"]) for r in rows]
    out = []

    def add(name, what, vals, ask_rate, tokens):
        g, l = flips(direct, vals)
        d, se = paired_se(direct, vals)
        out.append({"name": name, "what": what, "match": mean(vals), "delta": d, "se": se,
                    "gained": g, "lost": l, "p": mcnemar_p(g, l),
                    "ask_rate": ask_rate, "tokens": tokens})

    add("direct", "the §6 prompt, unchanged: the results-table path", direct, 0.0,
        mean(r["direct"]["tokens"] for r in rows))
    add("offer, may not ask", "the prompt with the ask clause, \"ask\" banned at the "
        "first token", [int(r["noask"]["match"]) for r in rows], 0.0,
        mean(r["noask"]["tokens"] for r in rows))
    m, rate, toks, asked = own_choice(rows)
    add("offer, the model's choice", "asks where greedy decoding asks; the dialogue the "
        "product would see", [int((r["real"] if r["id"] in asked else r["noask"])["match"])
                              for r in rows], rate, toks)
    add("always ask, placebo reply", "asks every time; the reply is \"%s\"" % PLACEBO_REPLY,
        [int(r["placebo"]["match"]) for r in rows], 1.0,
        mean(r["placebo"]["tokens"] for r in rows))
    add("always ask, simulated reply", "asks every time; the simulated user answers",
        [int(r["real"]["match"]) for r in rows], 1.0, mean(r["real"]["tokens"] for r in rows))
    o_rate, _ = oracle(policy_rows(rows))
    helps = {r["id"] for r in rows if r["real"]["match"] and not r["noask"]["match"]}
    add("oracle timing", "asks only where asking turns a wrong answer right",
        [int((r["real"] if r["id"] in helps else r["noask"])["match"]) for r in rows],
        o_rate, mean(r["real"]["tokens"] if r["id"] in helps else r["noask"]["tokens"]
                     for r in rows))
    return out


def headline_md(rows, title):
    lines = ["| condition | exec-match | Δ vs direct (SE) | gained / lost | McNemar p | asks | "
             "tokens said (mean) |", "|---|---|---|---|---|---|---|"]
    for h in headline(rows):
        delta = "—" if h["name"] == "direct" else "%+.1f (%.1f)" % (100 * h["delta"], 100 * h["se"])
        gl = "—" if h["name"] == "direct" else "+%d / −%d" % (h["gained"], h["lost"])
        p = "—" if h["name"] == "direct" else ("%.3f" % h["p"] if h["p"] >= 0.001 else "<0.001")
        lines.append("| **%s** — %s | %s | %s | %s | %s | %s | %.1f |" % (
            h["name"], h["what"], pct(h["match"]), delta, gl, p, pct(h["ask_rate"]), h["tokens"]))
    return "#### %s (n = %d)\n\n" % (title, len(rows)) + "\n".join(lines)


def timing_md(rows):
    """Question 3: the P(ask) ranking at fixed budgets, random, oracle; AUROCs."""
    pr = policy_rows(rows)
    curve = threshold_curve(pr, key="score")
    o_rate, o_match = oracle(pr)
    lines = ["| ask budget | ranked by P(ask) | random | gain over random |",
             "|---|---|---|---|"]
    for b in (0.0,) + BUDGETS + (1.0,):
        ranked, rnd = at_rate(curve, b), random_at_rate(pr, b)
        lines.append("| %s | %s | %s | %+.1f |" % (pct(b), pct(ranked), pct(rnd),
                                                    100 * (ranked - rnd)))
    lines.append("| oracle: %s | %s | | |" % (pct(o_rate), pct(o_match)))
    scores = [r["ask_margin"] for r in rows]
    a_wrong = auroc(scores, [not r["noask"]["match"] for r in rows])
    a_helps = auroc(scores, [r["real"]["match"] and not r["noask"]["match"] for r in rows])
    fmt = lambda a: "n/a" if a is None else "%.3f" % a          # noqa: E731
    probs = sorted(r["p_ask_norm"] for r in rows)
    return "\n".join(lines) + (
        "\n\nCases are ranked by the log-odds of asking against answering at the first "
        "token. AUROC (area under the ROC curve) of that ranking: %s for \"the unasked "
        "answer is wrong\", %s for \"asking turns it right\". 0.5 = no better than "
        "chance.\n\nP(ask) among the three forms: median %.2g, 90th percentile %.2g, "
        "max %.2g; greedy asks on %d."
        % (fmt(a_wrong), fmt(a_helps), probs[len(probs) // 2],
           probs[int(0.9 * (len(probs) - 1))], probs[-1],
           sum(r["chose_ask"] for r in rows)))


def information_md(rows):
    """Question 2: real reply against placebo, placebo against no ask."""
    real = [int(r["real"]["match"]) for r in rows]
    plac = [int(r["placebo"]["match"]) for r in rows]
    noask = [int(r["noask"]["match"]) for r in rows]
    out = []
    for name, a, b in (("the reply's information (simulated − placebo)", plac, real),
                       ("the extra turn alone (placebo − may not ask)", noask, plac)):
        g, l = flips(a, b)
        d, se = paired_se(a, b)
        p = mcnemar_p(g, l)
        out.append("| %s | %+.1f (%.1f) | +%d / −%d | %s |" % (
            name, 100 * d, 100 * se, g, l, "%.3f" % p if p >= 0.001 else "<0.001"))
    return ("| effect | Δ exec-match (SE) | gained / lost | McNemar p |\n|---|---|---|---|\n"
            + "\n".join(out))


def hand_audit_md(audit, label):
    """The hand audit's verdicts for this checkpoint's replies (audit_sample.py)."""
    mine = {k: v for k, v in audit.items() if k.split("|", 1)[0] == label}
    if not mine:
        return None
    c = summarize(mine)
    n = len(mine)
    return ("- hand audit of %d randomly drawn replies: %d wrong (%.0f%%), %d leaking the "
            "method, %d uninformative, %d minor, %d right. \"Wrong\" = followed literally, the "
            "reply leads away from the result."
            % (n, c["wrong"], 100 * c["wrong"] / n, c["leak"], c["uninformative"], c["minor"],
               c["ok"]))


def audit_md(rows, audit=None, label=None):
    """Question 4: what the leak screen saw, and the headline without flags."""
    n = len(rows)
    reasons = collections.Counter(f for r in rows for f in r["leak_flags"])
    flagged = [r for r in rows if r["leak_flags"] or r["quotes_result"]]
    clean = [r for r in rows if not (r["leak_flags"] or r["quotes_result"])]
    empty_q = sum(1 for r in rows if not r["question"].strip())
    second = sum(1 for r in rows if r["real"]["second_ask"])
    second_p = sum(1 for r in rows if r["placebo"]["second_ask"])
    consistent = collections.Counter(r["free_consistent"] for r in rows
                                     if r["free_consistent"] is not None)
    capped = sum(1 for r in rows if r.get("question_capped"))
    lines = [
        "- replies flagged by the leak screen: %d of %d%s" % (
            sum(1 for r in rows if r["leak_flags"]), n,
            " (%s)" % ", ".join("%s %d" % kv for kv in reasons.most_common()) if reasons else ""),
        "- replies pasting a whole changed line of the result: %d" % sum(
            1 for r in rows if r["quotes_result"]),
        "- empty questions (the model ended the line right after \"ask:\"): %d" % empty_q,
        "- a second question in turn 2 (a parse failure by the protocol): %d after the "
        "simulated reply, %d after the placebo" % (second, second_p),
    ]
    if consistent:
        lines.append("- where greedy asks of its own accord (%d cases), the forced question is "
                     "the one it asks on %d, the same cut at the %d-token cap on %d, and a "
                     "different one on %d (bf16 prefill against cached decoding; score.py)" % (
                         sum(consistent.values()), consistent["same"], QUESTION_MAX_TOKENS,
                         consistent["cut"], consistent["diverged"]))
    lines.append("- questions that hit the %d-token cap: %d of %d" % (
        QUESTION_MAX_TOKENS, capped, n))
    first = [r.get("first_consistent") for r in rows if r.get("first_consistent") is not None]
    if first:
        lines.append("- P(ask) is read from a forward pass of its own; its argmax is "
                     "generation's first token on %d of %d cases" % (sum(first), len(first)))
    if flagged and clean:
        d_all = mean(int(r["real"]["match"]) - int(r["direct"]["match"]) for r in rows)
        d_clean = mean(int(r["real"]["match"]) - int(r["direct"]["match"]) for r in clean)
        lines.append("- always-ask gain over direct on all %d cases %+.1f points; on the %d "
                     "with unflagged replies %+.1f" % (n, 100 * d_all, len(clean), 100 * d_clean))
    if audit:
        line = hand_audit_md(audit, label)
        if line:
            lines.append(line)
    return "\n".join(lines)


def by_group_md(rows, key, title):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r.get(key) or "?"].append(r)
    own = own_choice(rows)[3]
    lines = ["| %s | n | direct | may not ask | model's choice (asks) | always ask: placebo | "
             "always ask: simulated |" % title, "|---|---|---|---|---|---|---|"]
    for g in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        rs = groups[g]
        asks = sum(1 for r in rs if r["id"] in own)
        choice = mean(int((r["real"] if r["id"] in own else r["noask"])["match"]) for r in rs)
        lines.append("| %s | %d | %s | %s | %s (%d) | %s | %s |" % (
            g, len(rs), pct(mean(r["direct"]["match"] for r in rs)),
            pct(mean(r["noask"]["match"] for r in rs)), pct(choice), asks,
            pct(mean(r["placebo"]["match"] for r in rs)),
            pct(mean(r["real"]["match"] for r in rs))))
    return "\n".join(lines)


def show(s, limit=160):
    s = (s or "").replace("|", "\\|").replace("\n", "⏎")
    return s if len(s) <= limit else s[:limit - 1] + "…"


def examples_md(rows, k=6):
    def dialogue(r):
        return ("- `%s` (%s) — *%s*\n  - asks: %s\n  - user: %s\n  - answer: `%s` %s; "
                "unasked: `%s` %s; log-odds of asking %.1f%s" % (
                    r["id"], r.get("category"), show(r["instruction"]), show(r["question"]),
                    show(r["reply"]), show(r["real"]["command"]),
                    "✓" if r["real"]["match"] else "✗", show(r["noask"]["command"]),
                    "✓" if r["noask"]["match"] else "✗", r["ask_margin"],
                    "; leak flags: " + ", ".join(r["leak_flags"]) if r["leak_flags"] else ""))
    won = [r for r in rows if r["real"]["match"] and not r["noask"]["match"]]
    lost = [r for r in rows if r["noask"]["match"] and not r["real"]["match"]]
    keen = sorted(rows, key=lambda r: -r["ask_margin"])
    out = []
    for title, rs in (("Asking turned a wrong answer right", won),
                      ("Asking turned a right answer wrong", lost),
                      ("Where the model most wants to ask (highest P(ask))", keen)):
        out.append("**%s** (%d)\n" % (title, len(rs)) + "\n".join(dialogue(r) for r in rs[:k]))
    return "\n\n".join(out)


def untouched_ids(val_path, subset=250):
    """Val rows the checkpoint selections never saw: val minus the eval subset."""
    from train_sft import SUBSET_SEED, stratified_subset
    val = load_jsonl(val_path)
    seen = {c["id"] for c in stratified_subset(val, subset, SUBSET_SEED)}
    return {c["id"] for c in val} - seen


def summary_for(rows):
    return {h["name"]: {k: h[k] for k in ("match", "delta", "se", "gained", "lost", "p",
                                          "ask_rate", "tokens")} for h in headline(rows)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verdicts", action="append", required=True,
                    help="LABEL=PATH of a score.py output; repeatable, in report order")
    ap.add_argument("--title", default="Clarification experiment")
    ap.add_argument("--cases-name", default="", help="what the cases are, for the header")
    ap.add_argument("--untouched", default=None, metavar="LABEL[,LABEL]=VAL_JSONL",
                    help="also report these labels on the val rows checkpoint selection "
                         "never saw")
    ap.add_argument("--audit", default=None,
                    help="the simulator's hand-audit verdicts (audit_sample.py), if any")
    ap.add_argument("--out", required=True, help="report.md (summary.json beside it)")
    args = ap.parse_args()
    audit = json.loads(pathlib.Path(args.audit).read_text()) if args.audit else None
    if audit and "verdicts" in audit:            # a file with its provenance beside the verdicts
        audit = audit["verdicts"]

    sections = ["# %s\n\nGenerated %s by `scripts/clarify/report.py` from %s.%s\n\n"
                "The ask clause added to the §6 system text:\n\n> %s\n" % (
                    args.title, datetime.date.today().isoformat(),
                    ", ".join("`%s`" % v.split("=", 1)[1] for v in args.verdicts),
                    " Cases: %s." % args.cases_name if args.cases_name else "",
                    ASK_CLAUSE.replace("\n", " "))]
    summary = {}
    untouched_labels, untouched = set(), None
    if args.untouched:
        labs, path = args.untouched.split("=", 1)
        untouched_labels = set(labs.split(","))
        untouched = untouched_ids(path)
    for spec in args.verdicts:
        label, path = spec.split("=", 1)
        rows = load_jsonl(path)
        summary[label] = {"n": len(rows), "conditions": summary_for(rows)}
        parts = ["## %s\n" % label,
                 "### 1. Does asking help? Every condition against the direct answer\n",
                 headline_md(rows, "%s, all cases" % label)]
        if label in untouched_labels:
            sub = [r for r in rows if r["id"] in untouched]
            parts.append(headline_md(sub, "%s, the val rows checkpoint selection never saw"
                                     % label))
            summary[label]["untouched"] = {"n": len(sub), "conditions": summary_for(sub)}
        parts += ["### 2. The information, or the extra turn?\n", information_md(rows),
                  "### 3. Does the model know when to ask?\n", timing_md(rows),
                  "### 4. The simulated user\n", audit_md(rows, audit, label)]
        if any("note_class" in r for r in rows):
            parts += ["### By the note's class (why the row was never solved)\n",
                      by_group_md(rows, "note_class", "class")]
        parts += ["### By category\n", by_group_md(rows, "category", "category"),
                  "### Dialogues\n", examples_md(rows)]
        sections.append("\n\n".join(parts))
    out = pathlib.Path(args.out)
    out.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    out.with_name("summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                             encoding="utf-8")
    print("wrote %s and summary.json" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
