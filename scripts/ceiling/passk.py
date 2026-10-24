"""pass@k and maj@k over sampled completions -- the arithmetic of the ceiling
probe (README.md beside this file). Stdlib only, so the report and the tests
run without the training stack.

A prompt's observation is n samples, c of which exec-match. Two questions are
asked of it:

pass@k  would at least one of k samples be right? The chance that a perfect
        verifier -- here the executor -- finds an answer when allowed k tries.
        Estimated without bias from n >= k samples as

            pass@k = 1 - C(n - c, k) / C(n, k)

        (Chen et al. 2021, the Codex paper): the probability that a k-subset
        of the n observed samples, drawn without replacement, holds a correct
        one. The plug-in form 1 - (1 - c/n)^k is biased low -- (1 - p)^k is
        convex, so by Jensen its expectation overshoots -- most at small n and
        large k, which is exactly where a ceiling is read.

maj@k   would the most common answer among k samples be right? The
        verifier-free way to spend samples (self-consistency, Wang et al.
        2023). Answers vote by a key; a sample without one abstains; a tie is
        scored as its expected value (r correct among t tied keys -> r/t), so
        no tie-breaking rule leaks into the number. Exact at k = 1 and k = n,
        a seeded Monte Carlo mean over k-subsets in between.
"""
import collections
import math
import random

MAJ_DRAWS = 200     # k-subsets per prompt per k for maj@k (1 < k < n)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one prompt with c correct of n samples."""
    if not 0 <= c <= n:
        raise ValueError("need 0 <= c <= n, got c=%d n=%d" % (c, n))
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n, got k=%d n=%d" % (k, n))
    if n - c < k:           # every k-subset must contain a correct sample
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def k_grid(n: int) -> list[int]:
    """1, 2, 4, ... up to n, and n itself if it is not a power of two."""
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    if ks[-1] != n:
        ks.append(n)
    return ks


def vote_value(population) -> float:
    """Expected correctness of a plurality vote over (key, correct) samples.

    A key of None abstains. If every sample abstains the vote has no answer
    and scores 0. Samples sharing a key must agree on `correct` -- true by
    construction here, because a key is either the command string or the
    buffer it produced, and exec-match is a function of both."""
    counts = collections.Counter()
    right = {}
    for key, ok in population:
        if key is None:
            continue
        counts[key] += 1
        right[key] = bool(ok)
    if not counts:
        return 0.0
    top = max(counts.values())
    tied = [right[key] for key, cnt in counts.items() if cnt == top]
    return sum(tied) / len(tied)


def maj_at_k(population, k: int, draws: int = MAJ_DRAWS, rng=None) -> float:
    """maj@k for one prompt; `population` is one (key, correct) per sample."""
    population = list(population)
    n = len(population)
    if not 1 <= k <= n:
        raise ValueError("need 1 <= k <= n, got k=%d n=%d" % (k, n))
    if k == 1:              # one sample is its own majority
        return sum(1 for key, ok in population if key is not None and ok) / n
    if k == n:
        return vote_value(population)
    rng = rng or random.Random(0)
    return sum(vote_value(rng.sample(population, k)) for _ in range(draws)) / draws


def mean_se(values) -> tuple[float, float]:
    """Mean and its standard error (sample standard deviation / sqrt(N))."""
    values = list(values)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(values) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var / n)


def paired_diff(a, b) -> tuple[float, float]:
    """Mean of a - b over the same prompts and its paired standard error --
    the error bar the results table uses for every checkpoint comparison."""
    a, b = list(a), list(b)
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal lengths")
    return mean_se(x - y for x, y in zip(a, b))
