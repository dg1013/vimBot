#!/usr/bin/env python3
"""The ceiling report: pass@k and maj@k curves, coverage, and the rows no
sample reaches, from the samples files sample.py writes (README.md beside
this file). Stdlib only.

    python scripts/ceiling/report.py --run runs/ceiling_2026-09-22 \\
        --samples base=samples_base.jsonl --samples sft=samples_sft.jsonl \\
        --samples served=samples_served.jsonl \\
        --greedy base=runs/qwen3.5-4b/exec_test.jsonl \\
        --greedy sft=runs/ceiling_2026-09-22/greedy/qwen3.5-4b-sft_exec_test.jsonl \\
        --greedy served=runs/qwen3.5-4b-grpo-gated/fla-kernels/exec_test.jsonl \\
        --pool 4b='runs/qwen3.5-4b*/exec_test.jsonl' --pool all='runs/*/exec_test.jsonl'

`--samples` paths are resolved against --run when relative and not found
from the working directory; checkpoints are reported in the order given.
Every comparison is over the prompts present in every samples file, so a
report on a run still in progress is internally consistent. `--greedy` pairs
a checkpoint with its greedy exec log (the results-table row); `--pool NAME=GLOB`
(repeatable, globs accumulate per name) names a set of greedy runs whose union
is reported beside the sampled coverage.

Writes <run>/report.md and <run>/summary.json.
"""
import argparse
import collections
import datetime
import glob
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from passk import k_grid, maj_at_k, mean_se, paired_diff, pass_at_k   # noqa: E402
from evaluate import load_jsonl                                        # noqa: E402

VOTE_KEYS = ("command", "buffer")


# ------------------------------------------------------------------ loading

def labelled(specs, what):
    out = collections.OrderedDict()
    for spec in specs or ():
        if "=" not in spec:
            sys.exit("%s must be LABEL=PATH, got %r" % (what, spec))
        label, path = spec.split("=", 1)
        out.setdefault(label, []).append(path)
    return out


def resolve(path, run):
    p = pathlib.Path(path)
    if p.exists() or p.is_absolute():
        return p
    return run / p


def load_rows(path):
    rows = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows.setdefault(r["id"], r)       # first observation wins
    return rows


def load_greedy(path):
    """{id: matched} from an exec log (one row per case, `match` verdict)."""
    return {r["id"]: bool(r["match"]) for r in load_jsonl(str(path))}


def load_meta(path):
    meta = pathlib.Path(str(path) + ".meta.json")
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return None


# ------------------------------------------------------------------ metrics

def population(row, by):
    """One (key, correct) per sample of a row, for maj@k. by='command' votes
    on the answer's spelling, by='buffer' on the buffer it produced (an
    invalid answer abstains there). Unparsed samples abstain under both."""
    pop = []
    for e in row["commands"]:
        key = (e["mode"], e["command"]) if by == "command" else e.get("buf")
        pop.extend([(key, e["state"] == "match")] * e["n"])
    pop.extend([(None, False)] * row["unparsed"])
    return pop


def curves(label, rows, ids, ks, draws):
    """Per-prompt pass@k and maj@k values for one checkpoint, keyed by k."""
    out = {"pass": {}, "maj_command": {}, "maj_buffer": {}}
    for k in ks:
        out["pass"][k] = [pass_at_k(rows[i]["n"], rows[i]["c"], k) for i in ids]
        for by in VOTE_KEYS:
            vals = []
            for i in ids:
                rng = random.Random("%s|%s|%d|%s" % (label, i, k, by))
                vals.append(maj_at_k(population(rows[i], by), k, draws=draws, rng=rng))
            out["maj_" + by][k] = vals
    return out


def show(command):
    """A command safe inside a markdown table cell: control characters as
    \\xNN (a sampled answer can hold anything), pipes escaped."""
    out = "".join(ch if ch.isprintable() else "\\x%02x" % ord(ch) if ord(ch) < 256 else "\\u%04x" % ord(ch)
                  for ch in command)
    return out.replace("|", "\\|").replace("`", "'")


def pct(x):
    return "%.1f%%" % (100 * x)


def pct_se(pair):
    m, se = pair
    return "%.1f ± %.1f" % (100 * m, 100 * se)


def share(num, den):
    return "%d / %d (%.1f%%)" % (num, den, 100 * num / den) if den else "0 / 0"


def union_of(logs, ids):
    solved = set()
    for path in logs:
        g = load_greedy(path)
        solved |= {i for i in ids if g.get(i)}
    return solved


def expand_pool(globs):
    paths = []
    for pattern in globs:
        for p in sorted(glob.glob(pattern)):
            if p not in paths:
                paths.append(p)
    return paths


# ------------------------------------------------------------------ report

def build(args):
    run = pathlib.Path(args.run)
    cases = {c["id"]: c for c in load_jsonl(args.cases)}
    order = list(cases)
    samples = labelled(args.samples, "--samples")
    if not samples:
        sys.exit("give at least one --samples LABEL=PATH")
    rows, metas = {}, {}
    for label, paths in samples.items():
        path = resolve(paths[-1], run)
        rows[label] = load_rows(path)
        metas[label] = load_meta(path)
    labels = list(rows)
    ids = [i for i in order if all(i in rows[l] for l in labels)]
    if not ids:
        sys.exit("no prompt is present in every samples file")
    n_of = {l: min(rows[l][i]["n"] for i in ids) for l in labels}
    ks = k_grid(min(n_of.values()))
    greedy = {l: load_greedy(resolve(p[-1], run)) for l, p in labelled(args.greedy, "--greedy").items()}
    pools = {name: expand_pool(g) for name, g in labelled(args.pool, "--pool").items()}

    cur = {l: curves(l, rows[l], ids, ks, args.maj_draws) for l in labels}
    covered = {l: {i for i in ids if rows[l][i]["c"] > 0} for l in labels}
    covered_strict = {l: {i for i in ids if rows[l][i]["c_strict"] > 0} for l in labels}
    union = set().union(*covered.values())
    N = len(ids)
    kmax = ks[-1]

    md = []
    w = md.append
    w("# Ceiling probe — pass@k on the %s rows of `%s`" % (N, pathlib.Path(args.cases).name))
    w("")
    w("Built %s by `scripts/ceiling/report.py`. Checkpoints: %s. %s"
      % (datetime.date.today().isoformat(), ", ".join("`%s` (n = %d)" % (l, n_of[l]) for l in labels),
         "Every number is over the %d prompts present in every samples file%s."
         % (N, "" if N == len(order) else " (of %d in the split: a run is still in progress)" % len(order))))
    w("")
    w("## Sampling")
    w("")
    w("| checkpoint | adapter | T | top-p | top-k | cap | seed | base | versions |")
    w("|---|---|---|---|---|---|---|---|---|")
    for l in labels:
        m = metas[l] or {}
        s = m.get("sampling", {})
        last = (m.get("launches") or [{}])[-1]
        v = last.get("versions", {})
        w("| %s | `%s` | %s | %s | %s | %s | %s | %s | torch %s, transformers %s, peft %s, fla %s; %s |"
          % (l, s.get("adapter"), s.get("temperature"), s.get("top_p"), s.get("top_k"),
             s.get("max_new_tokens"), s.get("seed"), last.get("base"), v.get("torch"),
             v.get("transformers"), v.get("peft"), v.get("flash-linear-attention"), v.get("vim")))
    w("")

    # -- the ladder: one row per checkpoint, deployable → oracle
    k8 = 8 if 8 in ks else ks[min(3, len(ks) - 1)]
    w("## Headline — from the greedy row to the oracle bound")
    w("")
    w("Left to right: the results-table decoding; one sample at this run's temperature; the "
      "most common of n samples (no executor); the best of %d and of %d samples (the executor "
      "as an oracle verifier: not deployable, but the bound for RL on these prompts)." % (k8, kmax))
    w("")
    w("| checkpoint | greedy | pass@1 | maj@%d by command | pass@%d | pass@%d |" % (kmax, k8, kmax))
    w("|---|---|---|---|---|---|")
    for l in labels:
        gr = pct(sum(greedy[l].get(i, False) for i in ids) / N) if l in greedy else "—"
        w("| %s | %s | %s | %s | %s | %s |" % (
            l, gr, pct(mean_se(cur[l]["pass"][1])[0]), pct(mean_se(cur[l]["maj_command"][kmax])[0]),
            pct(mean_se(cur[l]["pass"][k8])[0]), pct(mean_se(cur[l]["pass"][kmax])[0])))
    w("| union of the checkpoints above (some sample of some checkpoint) | | | | | %s |"
      % pct(len(union) / N))
    w("")

    # -- the headline curve
    w("## pass@k — would any of k samples exec-match? (mean ± SE over prompts)")
    w("")
    w("| k | " + " | ".join(labels) + " |")
    w("|---|" + "---|" * len(labels))
    if greedy:
        w("| greedy (the results-table decoding) | " + " | ".join(
            pct(sum(greedy[l].get(i, False) for i in ids) / N) if l in greedy else "" for l in labels) + " |")
    for k in ks:
        w("| %d | " % k + " | ".join(pct_se(mean_se(cur[l]["pass"][k])) for l in labels) + " |")
    w("")
    w("Coverage at n — prompts with at least one exec-matching sample — and the union:")
    w("")
    w("| | " + " | ".join(labels) + " | union of these |")
    w("|---|" + "---|" * (len(labels) + 1))
    w("| covered (c ≥ 1) | " + " | ".join(share(len(covered[l]), N) for l in labels)
      + " | %s |" % share(len(union), N))
    w("| covered, trainer reading (a capped completion is a failure) | "
      + " | ".join(share(len(covered_strict[l]), N) for l in labels)
      + " | %s |" % share(len(set().union(*covered_strict.values())), N))
    blank = " | ".join("" for _ in labels)
    for name, logs in pools.items():
        solved = union_of(logs, ids)
        w("| greedy pool `%s` (%d runs): union solved | %s | %s |"
          % (name, len(logs), blank, share(len(solved), N)))
        w("| sampled union ∪ greedy pool `%s` | %s | %s |" % (name, blank, share(len(union | solved), N)))
    w("")

    # -- pairwise: where does one checkpoint overtake another?
    pairs = []
    for spec in args.pair or ():
        a, _, b = spec.partition(":")
        if a not in rows or b not in rows:
            sys.exit("--pair %r names a checkpoint without samples (have %s)" % (spec, ", ".join(labels)))
        pairs.append((a, b))
    if not args.pair:
        pairs = [(labels[i + 1], labels[i]) for i in range(len(labels) - 1)]
    if pairs:
        w("## Paired differences, A − B (points ± paired SE over the same prompts)")
        w("")
        w("| k | " + " | ".join("%s − %s" % p for p in pairs) + " |")
        w("|---|" + "---|" * len(pairs))
        for k in ks:
            cells = []
            for a, b in pairs:
                d, se = paired_diff(cur[a]["pass"][k], cur[b]["pass"][k])
                cells.append("%+.1f ± %.1f" % (100 * d, 100 * se))
            w("| %d | " % k + " | ".join(cells) + " |")
        w("")
        w("Coverage moves between checkpoints (prompts with c ≥ 1):")
        w("")
        w("| from → to | kept | gained | lost | never |")
        w("|---|---|---|---|---|")
        for a, b in pairs:
            A, B = covered[b], covered[a]
            w("| %s → %s | %d | %d | %d | %d |" % (b, a, len(A & B), len(B - A), len(A - B),
                                                 N - len(A | B)))
        w("")

    # -- verifier-free sampling
    w("## maj@k — is the most common answer right? (no executor at answer time)")
    w("")
    w("By command string: deployable under the execution boundary (voting reads strings, runs "
      "nothing). By resulting buffer: needs every answer executed, so it is an analysis number, "
      "not a real-use one (CLAUDE.md §1).")
    w("")
    w("| k | " + " | ".join("%s, by command | %s, by buffer" % (l, l) for l in labels) + " |")
    w("|---|" + "---|" * (2 * len(labels)))
    for k in ks:
        w("| %d | " % k + " | ".join("%s | %s" % (pct_se(mean_se(cur[l]["maj_command"][k])),
                                                   pct_se(mean_se(cur[l]["maj_buffer"][k])))
                                      for l in labels) + " |")
    w("")

    # -- what the samples look like
    w("## The samples themselves")
    w("")
    w("| per checkpoint | " + " | ".join(labels) + " |")
    w("|---|" + "---|" * len(labels))

    def tot(l, f):
        return sum(f(rows[l][i]) for i in ids)
    for name, f in (
            ("samples", lambda r: r["n"]),
            ("exec-match", lambda r: r["c"]),
            ("valid, wrong buffer", lambda r: r["states"].get("valid_wrong", 0)),
            ("invalid", lambda r: r["states"].get("invalid", 0)),
            ("unparsed (§6 grammar)", lambda r: r["unparsed"]),
            ("<think> content", lambda r: r["think"]),
            ("ran into the cap", lambda r: r["truncated"]),
            ("ex command with a NUL byte (invalid unrun)", lambda r: r.get("unrunnable", 0))):
        w("| %s | " % name + " | ".join(share(tot(l, f), tot(l, lambda r: r["n"])) if name != "samples"
                                        else "%d" % tot(l, f) for l in labels) + " |")
    w("| answered in the gold's mode | " + " | ".join(share(
        tot(l, lambda r: sum(e["n"] for e in r["commands"] if e["mode"] == r["mode"])),
        tot(l, lambda r: r["n"])) for l in labels) + " |")
    w("| mean distinct commands per prompt | " + " | ".join(
        "%.1f" % (tot(l, lambda r: r["distinct"]) / N) for l in labels) + " |")
    w("| mean distinct buffers per prompt | " + " | ".join(
        "%.1f" % (tot(l, lambda r: r["distinct_buffers"]) / N) for l in labels) + " |")
    w("| mean completion tokens | " + " | ".join(
        "%.1f" % (tot(l, lambda r: (r["tokens_mean"] or 0) * r["n"]) / tot(l, lambda r: r["n"]))
        for l in labels) + " |")
    w("| prompts answered one way in every sample | " + " | ".join(
        share(sum(1 for i in ids if rows[l][i]["distinct"] == 1 and rows[l][i]["unparsed"] == 0), N)
        for l in labels) + " |")
    w("")

    # -- by category and tier
    for field in ("category", "tier", "mode"):
        w("## By %s — greedy / pass@1 / pass@%d%s" % (field, kmax,
          " (mode = the gold's mode)" if field == "mode" else ""))
        w("")
        groups = collections.defaultdict(list)
        for i in ids:
            groups[cases[i].get(field)].append(i)
        w("| %s | rows | " % field + " | ".join(labels) + " | union covered |")
        w("|---|---|" + "---|" * (len(labels) + 1))
        for g in sorted(groups, key=lambda x: (-len(groups[x]), str(x))):
            gi = groups[g]
            cells = []
            for l in labels:
                p1 = sum(rows[l][i]["c"] / rows[l][i]["n"] for i in gi) / len(gi)
                pk = sum(pass_at_k(rows[l][i]["n"], rows[l][i]["c"], kmax) for i in gi) / len(gi)
                gr = ("%s / " % pct(sum(greedy[l].get(i, False) for i in gi) / len(gi))) if l in greedy else ""
                cells.append("%s%s / %s" % (gr, pct(p1), pct(pk)))
            w("| %s | %d | " % (g, len(gi)) + " | ".join(cells)
              + " | %s |" % pct(sum(1 for i in gi if i in union) / len(gi)))
        w("")

    # -- the rows nothing reaches
    all_logs = expand_pool(sum(labelled(args.pool, "--pool").values(), []))
    ever = union_of(all_logs, ids) if all_logs else set()
    core = [i for i in ids if i not in union]
    # whose answer to show: the checkpoint that covers the most prompts is
    # the one whose miss says most about the ceiling
    last = args.answers_from or max(labels, key=lambda l: len(covered[l]))
    if last not in rows:
        sys.exit("--answers-from %r has no samples (have %s)" % (last, ", ".join(labels)))
    w("## The uncovered core: %d rows no sample of any checkpoint solves" % len(core))
    w("")
    w("Test rows: read them to understand the ceiling, never to author training data from "
      "(that would leak the test slice into the next round). `greedy elsewhere` = solved by "
      "some greedy run in the pools above. The answer shown is `%s`'s most common one." % last)
    w("")
    w("| id | category | tier | gold | %s's top answer (share, state) | greedy elsewhere |" % last)
    w("|---|---|---|---|---|---|")
    for i in sorted(core, key=lambda x: (cases[x].get("category"), cases[x].get("tier"), x)):
        c = cases[i]
        top = rows[last][i]["commands"][:1]
        ans = ("`%s: %s` (%d/%d, %s)" % (top[0]["mode"], show(top[0]["command"]),
                                         top[0]["n"], rows[last][i]["n"], top[0]["state"])
               if top else "(no parseable answer)")
        w("| %s | %s | %s | `%s: %s` | %s | %s |"
          % (i, c.get("category"), c.get("tier"), c.get("mode"),
             show(c["gold_command"]), ans, "yes" if i in ever else ""))
    w("")

    summary = {
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
        "cases": args.cases, "prompts": N, "checkpoints": labels, "n": n_of, "k": ks,
        "pass_at_k": {l: {k: mean_se(cur[l]["pass"][k]) for k in ks} for l in labels},
        "maj_at_k_command": {l: {k: mean_se(cur[l]["maj_command"][k]) for k in ks} for l in labels},
        "maj_at_k_buffer": {l: {k: mean_se(cur[l]["maj_buffer"][k]) for k in ks} for l in labels},
        "greedy": {l: sum(greedy[l].get(i, False) for i in ids) / N for l in greedy},
        "covered": {l: len(covered[l]) for l in labels},
        "covered_strict": {l: len(covered_strict[l]) for l in labels},
        "union_covered": len(union),
        "pools": {name: {"runs": len(logs), "solved": len(union_of(logs, ids))}
                  for name, logs in pools.items()},
        "uncovered_core": sorted(core),
    }
    return "\n".join(md) + "\n", summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run directory; report.md and summary.json go here")
    ap.add_argument("--cases", default=str(REPO / "data" / "splits" / "test.jsonl"))
    ap.add_argument("--samples", action="append", help="LABEL=PATH, repeatable, in display order")
    ap.add_argument("--greedy", action="append", help="LABEL=EXEC_LOG: the checkpoint's greedy row")
    ap.add_argument("--pool", action="append", help="NAME=GLOB: greedy exec logs to union")
    ap.add_argument("--pair", action="append",
                    help="A:B, repeatable: report A − B (default: each checkpoint minus the one before)")
    ap.add_argument("--answers-from", default=None,
                    help="checkpoint whose top answer the uncovered-core table shows "
                         "(default: the one covering the most prompts)")
    ap.add_argument("--maj-draws", type=int, default=100,
                    help="Monte Carlo k-subsets per prompt for maj@k at 1 < k < n")
    args = ap.parse_args(argv)
    md, summary = build(args)
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "report.md").write_text(md, encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    print("wrote %s and %s (%d prompts, checkpoints %s)"
          % (run / "report.md", run / "summary.json", summary["prompts"], ", ".join(summary["checkpoints"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
