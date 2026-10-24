"""Shared pieces of the ask-and-pick experiment (stdlib + optional torch).

The experiment combines three things already measured on their own: the
verifier's pick among the served adapter's 8 samples, the clarifying-question
dialogue (the model's own question, the simulated user's reply, an answer
after it), and the hidden-state probe that reads under-specification. Every
val and test row has every arm's answer, so a policy -- ask on a share of rows
chosen by a trigger score, else the verifier's pick -- is scored offline.
"""
import json
import math
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = REPO / "runs"
DIALOGUES = {"val": RUNS / "clarify_2026-10-06" / "val" / "verdicts_served.jsonl",
             "test": RUNS / "clarify_2026-10-07" / "test" / "verdicts_served.jsonl",
             "diag": RUNS / "clarify_2026-10-06" / "diag" / "verdicts_served.jsonl"}
CASES = {"val": REPO / "data" / "splits" / "val.jsonl", "test": REPO / "data" / "splits" / "test.jsonl",
         "diag": RUNS / "clarify_2026-10-06" / "diag_cases.jsonl"}
SINGLE_TURN = {"val": (RUNS / "reranker_2026-10-16" / "val_examples.jsonl",
                       RUNS / "reranker_2026-10-16" / "verifier-4b" / "val_scores_step818.jsonl"),
               "test": (RUNS / "reranker_2026-10-16" / "test_examples.jsonl",
                        RUNS / "reranker_2026-10-16" / "verifier-4b" / "test_scores.jsonl")}
TRIGGERS = ("V", "U", "F", "V+U", "V+F")
RATES = (0.10, 0.20, 0.30)
AFTER = ("greedy", "pick-plain", "pick-clarified")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def clarified_case(case, question, reply):
    """The case with the question and the reply appended to its instruction:
    what the *clarified* verifier variant reads. Never touches buffer_after or
    the gold command."""
    c = dict(case)
    c["instruction"] = "%s\nClarifying question: %s\nUser's answer: %s" % (case["instruction"], question, reply)
    return c


def candidates(row):
    """Distinct executed candidates of a sample row (probe format), as the
    verifier's examples: parse failures and truncations are never ranked."""
    out = []
    for e in row.get("commands", []):
        if e["state"] in ("match", "valid_wrong", "invalid"):
            out.append({"id": row["id"], "cand_mode": e["mode"], "cand_command": e["command"],
                        "n": e["n"], "label": int(e["state"] == "match"), "state": e["state"]})
    return out


def pick_label(cands, scores, cid):
    """The label of the verifier's pick: the highest score; ties by sample
    count, then mode and command text (deterministic). 0 if nothing to pick."""
    if not cands:
        return 0
    top = max(cands, key=lambda c: (scores[(cid, c["cand_mode"], c["cand_command"])], c.get("n", 0),
                                    c["cand_mode"], c["cand_command"]))
    return top["label"]


def best_score(cands, scores, cid):
    """The verifier's score of its own pick (its confidence); -inf if none."""
    if not cands:
        return float("-inf")
    return max(scores[(cid, c["cand_mode"], c["cand_command"])] for c in cands)


def majority_label(cands):
    """The label of the most frequent command (ties by mode and text)."""
    if not cands:
        return 0
    return max(cands, key=lambda c: (c.get("n", 0), c["cand_mode"], c["cand_command"]))["label"]


def ranks(values):
    """Average ranks (0-based) of a list, ties shared: for rank-averaging scores
    that live on different scales."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2
        i = j + 1
    return r


def combine(ids, a, b):
    """Rank-average of two trigger scores (higher = ask first)."""
    ra, rb = ranks([a[i] for i in ids]), ranks([b[i] for i in ids])
    return {i: (x + y) / 2 for i, x, y in zip(ids, ra, rb)}


def asked_set(ids, trigger, rate):
    """The rows a policy asks on: the top `rate` share by trigger score (ties
    broken by id, so the set is deterministic)."""
    k = round(rate * len(ids))
    order = sorted(ids, key=lambda i: (-trigger[i], i))
    return set(order[:k])


def policy(ids, trigger, rate, after, notasked):
    """Per-row outcome of: ask on the top `rate` by trigger, answer `after[i]`
    there and `notasked[i]` elsewhere. Returns (outcome dict, asked set)."""
    asked = asked_set(ids, trigger, rate)
    return {i: (after[i] if i in asked else notasked[i]) for i in ids}, asked


def mcnemar_p(gained, lost):
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(a, b, ids):
    """(gained, lost) of b over a."""
    return (sum(1 for i in ids if b[i] and not a[i]), sum(1 for i in ids if a[i] and not b[i]))


def auroc(pos, neg):
    """P(a positive outscores a negative), ties half."""
    if not pos or not neg:
        return float("nan")
    import bisect
    srt = sorted(neg)
    tot = 0.0
    for p in pos:
        lo, hi = bisect.bisect_left(srt, p), bisect.bisect_right(srt, p)
        tot += lo + 0.5 * (hi - lo)
    return tot / (len(pos) * len(neg))


def paired_ci(gained, lost, n):
    """Delta and its 95% interval, in points, for a paired difference."""
    d = (gained - lost) / n
    se = math.sqrt(max(gained + lost - (gained - lost) ** 2 / n, 0.0)) / n
    return 100 * d, 100 * (d - 1.96 * se), 100 * (d + 1.96 * se)
