#!/usr/bin/env python3
"""Difficulty bookkeeping for difficulty-gated GRPO (Group Relative Policy
Optimization): what the policy has been seen to do on each train prompt, and
which prompts are therefore worth a rollout. Stdlib only; shared by
`probe_difficulty.py`, `select_frontier.py` and `train_grpo_gated.py`.

The one fact everything here rests on: GRPO learns from a prompt only when
its G samples DISAGREE. Every completion's advantage is its reward minus the
group mean, so a group that is all right or all wrong has every advantage at
zero and returns no gradient for the rollouts it cost. For a prompt the
policy solves with probability p, the expected squared advantage in a group
of G binary rewards is (G - 1) * p * (1 - p): largest at p = 1/2, zero at
both ends. Measured on the 4B's first GRPO run (DECISIONS.md 2026-09-14), the
share of silent groups rose from 35% to 56% over the epoch and the realized
mean of k(G - k)/G was 0.53 against 1.75 for a prompt at p = 1/2, so the run
drew about 30% of the gradient its rollouts could have carried.

Vocabulary (the "shelves"):

    cliff      no success observed (pooled k = 0): nothing to rank a
               completion above, nothing for GRPO to reinforce
    frontier   some successes and some failures (0 < k < G): the only
               shelf that carries gradient
    known      every sample succeeded (k = G): only the brevity bonus can
               move it, and training on it sharpens the sampler (the
               entropy collapse of the first run)
    unprobed   never observed; callers decide whether to treat it as
               frontier (measure by rolling it out) or leave it alone

An observation is one group: `k` matches out of `G` samples at training
`step` (0 for an offline probe). Observations are pooled over a window of
the last M per prompt (`--pool last:M`): M = 1 is "latest wins"; M = 2, the
default, means a prompt leaves the frontier only after two consecutive
perfect groups, which halves the misfiling of a p = 0.9 prompt on a single
8-of-8 draw (43% -> 18%) at the price of one extra group per solved prompt.

Difficulty files are JSONL, one observation per line, the same shape whether
written by the probe or by the trainer's live log:

    {"id": "...", "k": 3, "G": 8, "step": 0, "source": "probe",
     "states": {"match": 3, "valid_wrong": 4, "invalid": 1}, "distinct": 5,
     ... any further fields (commands, category, adapter, time) are kept by
     the writer for analysis and ignored by the book}

`ChunkPlanner` turns shelves into the chunks a sampler needs -- `batch_size`
DISTINCT prompt indices per optimizer step -- drawing each slot from the
frontier by default, from the known shelf at `anchor_frac` (so solved
prompts stay visited and a regression is noticed) and from the cliff at
`cliff_frac` (so a prompt the policy has grown into is noticed too; GRESO's
"skip with a probability, never with certainty"). Frontier prompts are
served in shuffled passes, each once per pass, so coverage is even; when the
pools are refreshed the not-yet-visited part of the current pass is kept.
"""
import collections
import json
import pathlib
import random

SHELVES = ("cliff", "frontier", "known", "unprobed")
DEFAULT_POOL = "last:2"
DEFAULT_ANCHOR_FRAC = 0.15
DEFAULT_CLIFF_FRAC = 0.05


def parse_pool(spec):
    """'last:M' -> M, 'latest' -> 1, 'all' -> None (every observation)."""
    if spec is None:
        return None
    spec = str(spec).strip().lower()
    if spec == "all":
        return None
    if spec == "latest":
        return 1
    if spec.startswith("last:") and spec[5:].isdigit() and int(spec[5:]) >= 1:
        return int(spec[5:])
    raise ValueError("pool must be 'latest', 'all' or 'last:M' (M >= 1), not %r" % spec)


def fuel(k, g):
    """Sum of squared centered advantages in a group with k of g binary
    rewards, k(g - k)/g -- the gradient a group can carry, before scaling.
    Its expectation over k ~ Binomial(g, p) is (g - 1) p (1 - p)."""
    return k * (g - k) / g if g else 0.0


def shelf_of(k, g):
    if g <= 0:
        return "unprobed"
    if k <= 0:
        return "cliff"
    if k >= g:
        return "known"
    return "frontier"


class DifficultyBook:
    """Per-prompt observations, pooled into a shelf. Keyed by case id; the
    index of each id in the caller's case list is kept so a sampler can work
    in dataset indices."""

    def __init__(self, ids, pool=DEFAULT_POOL):
        self.ids = list(ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate case ids")
        self.index = {cid: i for i, cid in enumerate(self.ids)}
        self.obs = {cid: [] for cid in self.ids}          # (k, G, step, source)
        self.window = parse_pool(pool)
        self.pool_spec = pool
        self.log_path = None
        self.unknown_ids = 0
        self.malformed = 0

    # ------------------------------------------------------------ input

    def load(self, path):
        """Append every observation in a difficulty file, in file order.
        Rows for ids outside the book are counted and skipped, so a probe of
        the whole split can seed a book over a subset of it."""
        n = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    k, g = int(row["k"]), int(row["G"])
                except (ValueError, KeyError, TypeError):
                    self.malformed += 1
                    continue
                cid = row.get("id")
                if cid not in self.obs:
                    self.unknown_ids += 1
                    continue
                self.obs[cid].append((k, g, int(row.get("step") or 0), row.get("source") or "file"))
                n += 1
        return n

    def attach_log(self, path):
        """Every later `record` is also appended to this JSONL file, which is
        itself a valid difficulty file: a resumed run reloads it."""
        self.log_path = pathlib.Path(path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, cid, k, g, step, source="train", **extra):
        if cid not in self.obs:
            self.unknown_ids += 1
            return None
        self.obs[cid].append((int(k), int(g), int(step), source))
        row = {"id": cid, "k": int(k), "G": int(g), "step": int(step), "source": source}
        row.update(extra)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        return row

    def record_states(self, cid, states, step, source="train", **extra):
        """Record one group from its per-completion reward states (the
        trainer's vocabulary: truncated / parse / invalid / valid_wrong /
        match)."""
        counts = collections.Counter(states)
        return self.record(cid, counts.get("match", 0), len(states), step, source,
                           states=dict(counts), **extra)

    # ------------------------------------------------------------ pooling

    def pooled(self, cid):
        """(k, G, n_observations) over the pooling window for one prompt."""
        obs = self.obs[cid]
        if self.window is not None:
            obs = obs[-self.window:]
        return sum(o[0] for o in obs), sum(o[1] for o in obs), len(obs)

    def p_hat(self, cid):
        k, g, _ = self.pooled(cid)
        return k / g if g else None

    def shelf(self, cid):
        k, g, _ = self.pooled(cid)
        return shelf_of(k, g)

    def shelves(self):
        """{shelf: [dataset index, ...]} in id order, every id on exactly one shelf."""
        out = {s: [] for s in SHELVES}
        for i, cid in enumerate(self.ids):
            out[self.shelf(cid)].append(i)
        return out

    def observed(self):
        return sum(1 for cid in self.ids if self.obs[cid])

    # ------------------------------------------------------------ reporting

    def counts(self):
        return {s: len(v) for s, v in self.shelves().items()}

    def expected_fuel(self, indices=None):
        """Mean k(G-k)/G over the pooled observations of the given prompts
        (default: every observed prompt), the realized-gradient proxy. Uses
        each prompt's pooled p̂ at G = 8 so windows of different sizes compare."""
        ids = self.ids if indices is None else [self.ids[i] for i in indices]
        vals = []
        for cid in ids:
            k, g, _ = self.pooled(cid)
            if g:
                p = k / g
                vals.append(7 * p * (1 - p))      # (G - 1) p (1 - p) at G = 8
        return sum(vals) / len(vals) if vals else None

    def histogram(self, g_ref=8):
        """Counts of pooled p̂ mapped onto k-of-g_ref bins, for the summary."""
        hist = collections.Counter()
        for cid in self.ids:
            k, g, _ = self.pooled(cid)
            if g:
                hist[round(k / g * g_ref)] += 1
        return {k: hist.get(k, 0) for k in range(g_ref + 1)}

    def summary(self, cases=None, keys=("category", "tier", "mode")):
        """Shelf counts overall and, when the cases are given, per key."""
        out = {"prompts": len(self.ids), "observed": self.observed(),
               "pool": self.pool_spec, "shelves": self.counts(),
               "expected_fuel_observed": self.expected_fuel(),
               "histogram_k_of_8": self.histogram()}
        if cases is not None:
            by_id = {c["id"]: c for c in cases}
            for key in keys:
                table = {}
                for cid in self.ids:
                    val = str(by_id.get(cid, {}).get(key, "?"))
                    row = table.setdefault(val, collections.Counter())
                    row[self.shelf(cid)] += 1
                    k, g, _ = self.pooled(cid)
                    if g:
                        row["k_sum"] += k
                        row["G_sum"] += g
                out["by_" + key] = {v: dict(c) for v, c in sorted(table.items())}
        return out


class ChunkPlanner:
    """Draws chunks of `batch_size` distinct prompt indices from the shelves.

    pools        {"frontier": [...], "known": [...], "cliff": [...]}; the
                 caller has already merged the unprobed prompts into whichever
                 pool it wants them measured from
    anchor_frac  probability a slot is drawn from the known shelf
    cliff_frac   probability a slot is drawn from the cliff
    A slot whose pool is empty, or whose pool cannot supply another distinct
    index for this chunk, falls back frontier -> known -> cliff. If every pool
    is exhausted the chunk is completed from the whole index range so the
    trainer never starves; `starved` counts such slots.
    """

    ORDER = ("frontier", "known", "cliff")

    def __init__(self, pools, batch_size, anchor_frac=DEFAULT_ANCHOR_FRAC,
                 cliff_frac=DEFAULT_CLIFF_FRAC, seed=None, universe=None):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if anchor_frac < 0 or cliff_frac < 0 or anchor_frac + cliff_frac > 1:
            raise ValueError("anchor_frac and cliff_frac must be >= 0 and sum to <= 1")
        self.batch_size = batch_size
        self.anchor_frac = anchor_frac
        self.cliff_frac = cliff_frac
        self.rng = random.Random(seed)
        self.cycles = {name: [] for name in self.ORDER}
        self.cursor = {name: 0 for name in self.ORDER}
        self.passes = {name: 0 for name in self.ORDER}
        self.drawn = collections.Counter()      # slots served per pool
        self.starved = 0
        self.refreshes = 0
        self._wrapped = set()
        self.universe = list(universe) if universe is not None else None
        self.set_pools(pools, initial=True)

    def set_pools(self, pools, initial=False):
        """Replace the pools. The unvisited remainder of each current pass is
        kept (in order) for indices still in the pool; new indices join
        shuffled behind it, so a refresh does not restart coverage."""
        for name in self.ORDER:
            new = list(pools.get(name, []))
            new_set = set(new)
            keep = [i for i in self.cycles[name][self.cursor[name]:] if i in new_set]
            kept = set(keep)
            fresh = [i for i in new if i not in kept]
            self.rng.shuffle(fresh)
            self.cycles[name] = keep + fresh
            self.cursor[name] = 0
        if not initial:
            self.refreshes += 1

    def sizes(self):
        return {name: len(c) for name, c in self.cycles.items()}

    def _slot_pool(self):
        u = self.rng.random()
        if u < self.cliff_frac:
            return "cliff"
        if u < self.cliff_frac + self.anchor_frac:
            return "known"
        return "frontier"

    def _draw(self, name, taken):
        """Next index of `name`'s pass not already in this chunk, or None.
        A pass wraps at most once per chunk: after a wrap, every index the
        pool holds has been offered to this chunk, so a second wrap could
        only repeat the same refusals (and would inflate the pass count)."""
        cycle = self.cycles[name]
        if not cycle:
            return None
        while True:
            if self.cursor[name] >= len(cycle):
                if name in self._wrapped:
                    return None
                self._wrapped.add(name)
                self.rng.shuffle(cycle)
                self.cursor[name] = 0
                self.passes[name] += 1
            idx = cycle[self.cursor[name]]
            self.cursor[name] += 1
            if idx not in taken:
                return idx

    def chunk(self):
        out, taken = [], set()
        self._wrapped = set()
        for _ in range(self.batch_size):
            want = self._slot_pool()
            idx, served = None, None
            for name in (want,) + tuple(n for n in self.ORDER if n != want):
                idx = self._draw(name, taken)
                if idx is not None:
                    served = name
                    break
            if idx is None:
                self.starved += 1
                universe = self.universe if self.universe is not None else [
                    i for c in self.cycles.values() for i in c]
                rest = [i for i in universe if i not in taken]
                if not rest:
                    break
                idx, served = self.rng.choice(rest), "starved"
            out.append(idx)
            taken.add(idx)
            self.drawn[served] += 1
        return out

    def stats(self):
        return {"sizes": self.sizes(), "passes": dict(self.passes),
                "drawn": dict(self.drawn), "starved": self.starved,
                "refreshes": self.refreshes}


def merge_unprobed(shelves, policy="frontier"):
    """Fold the unprobed prompts into one pool: 'frontier' (roll them out and
    learn their p), 'known' (visit at the anchor rate), 'cliff', or 'skip'
    (never draw them)."""
    pools = {name: list(shelves.get(name, [])) for name in ChunkPlanner.ORDER}
    if policy in pools:
        pools[policy] = pools[policy] + list(shelves.get("unprobed", []))
    elif policy != "skip":
        raise ValueError("unprobed policy must be frontier, known, cliff or skip, not %r" % policy)
    return pools
