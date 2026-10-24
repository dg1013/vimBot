"""When to ask: scoring a clarification policy offline (README.md).

Every case carries two outcomes, 1 for an exec-match and 0 otherwise:
``noask``, the answer when the model may not ask, and ``real``, the answer
after asking and hearing the simulated user. A policy is the set S of cases
it asks on, and its exec-match is

    (sum of real over S + sum of noask outside S) / n

so any policy is scored without generating anything new. The ones the report
reads:

* **the model's own choice** -- S = the cases where greedy decoding's first
  token is "ask";
* **a threshold on P(ask)** -- S = the top m cases by the model's
  first-token probability of asking, for every budget m: the model's
  ranking of where a question is worth it, separated from its calibration;
* **random at the same rate** -- the expectation is a straight line from
  "never ask" to "always ask", so a ranking is worth something only where
  its curve lies above that line;
* **the oracle** -- S = exactly the cases where asking turns a wrong answer
  right: the most any timing rule could get from these questions and
  replies.

Stdlib only.
"""
import math


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def policy_match(rows, ask_ids):
    """Exec-match of asking exactly on ``ask_ids``."""
    return mean(r["real"] if r["id"] in ask_ids else r["noask"] for r in rows)


def threshold_curve(rows, key="p_ask_norm"):
    """(ask rate, exec-match) after each group of equal scores, highest first.

    Asking on the top-scored cases first, with exact ties taken together: a
    threshold cannot separate tied cases, so the points are the group
    boundaries, and a budget that falls inside a group is read by linear
    interpolation -- the expectation under a random tie-break.
    """
    n = len(rows)
    if not n:
        return [(0.0, float("nan"))]
    order = sorted(rows, key=lambda r: -r[key])
    acc = sum(r["noask"] for r in rows)
    points = [(0.0, acc / n)]
    i = 0
    while i < n:
        j = i
        while j < n and order[j][key] == order[i][key]:
            acc += order[j]["real"] - order[j]["noask"]
            j += 1
        points.append((j / n, acc / n))
        i = j
    return points


def at_rate(curve, rate):
    """The curve read at an ask rate (linear between its points)."""
    for (r0, m0), (r1, m1) in zip(curve, curve[1:]):
        if r0 <= rate <= r1:
            return m0 if r1 == r0 else m0 + (m1 - m0) * (rate - r0) / (r1 - r0)
    return curve[-1][1]


def random_at_rate(rows, rate):
    """Expected exec-match of asking on a random ``rate`` of the cases."""
    return (1 - rate) * mean(r["noask"] for r in rows) + rate * mean(r["real"] for r in rows)


def oracle(rows):
    """(ask rate, exec-match) of asking exactly where it turns wrong into right."""
    helps = {r["id"] for r in rows if r["real"] and not r["noask"]}
    return len(helps) / len(rows), policy_match(rows, helps)


def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties counted half.

    The Mann-Whitney statistic: 0.5 is a ranking no better than chance.
    None when one class is empty.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg = (i + 1 + j) / 2              # mean of ranks i+1 .. j
        rank_sum += avg * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    u = rank_sum - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


def flips(a, b):
    """(gained, lost): cases b gets right that a missed, and the reverse."""
    gained = sum(1 for x, y in zip(a, b) if y and not x)
    lost = sum(1 for x, y in zip(a, b) if x and not y)
    return gained, lost


def mcnemar_p(gained, lost):
    """Exact two-sided McNemar p for paired binary outcomes.

    Only the discordant pairs carry information about a difference; under
    the null each is a fair coin, so the p-value is a binomial tail.
    """
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_se(a, b):
    """(mean difference b - a, its standard error) over paired outcomes."""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    if n < 2:
        return mean(d), float("nan")
    m = mean(d)
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return m, math.sqrt(var / n)
