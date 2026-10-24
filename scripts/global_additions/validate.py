#!/usr/bin/env python3
"""Gate the candidate global prompts and select the proposed set.

Every gold is executed by the sandboxed executor (src/executor.run_vim). A
candidate is kept only if all of these hold:
  - vim accepts the command (no error, no timeout, not blocked by the scope screen);
  - the buffer changes (no-op rows teach nothing and invite reward hacking, §7.2);
  - case does not decide the result: re-running the command with every global
    pattern made case-insensitive (\\c) gives the same buffer. These
    instructions do not state case, and the corpus rule is that a
    case-dependent instruction must (DECISIONS.md 2026-09-05);
  - no leakage key (scripts/build_splits.leakage_keys: exact, near-duplicate
    instruction, buffer, normalized command) is shared with any val or test
    row, so training on it cannot leak into a measurement (§7.3);
  - its buffer is not a train buffer and its (instruction, buffer) pair is new.
Then the selection caps any gold command and any instruction sentence at 3 uses,
and balances skills and domains (seeded).

Accepted rows get an id (glb16-NNNN), the executor's buffer_after, and the tags
that keep them separable: dataset "add-global-2026-10-16", source
"claude-session-2026-10-16", needs_review true.

    python scripts/global_additions/validate.py --candidates runs/global_additions_2026-10-16/candidates.jsonl \\
        --out-dir data/proposed/global_2026-10-16 --target 1000
"""
import argparse
import collections
import json
import pathlib
import random
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim                                   # noqa: E402

DATASET = "add-global-2026-10-16"
SOURCE = "claude-session-2026-10-16"
ID_PREFIX = "glb16-"
SKILL_CAPS = {"sub-on-match": 220, "realistic-delete": 160, "capture": 180, "relative": 130,
              "move": 100, "join": 60, "range": 90, "copy": 70, "sub-on-nonmatch": 40}
DOMAIN_CAP = 110
PER_COMMAND, PER_SENTENCE = 3, 3
_GLOBAL = re.compile(r"(^|\||[\d$,.])([gv])/")


def load_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def norm(buf):
    return buf[:-1] if buf.endswith("\n") else buf


def case_blind(cmd):
    """The command with every :g/:v pattern made case-insensitive."""
    return _GLOBAL.sub(lambda m: "%s%s/\\c" % (m.group(1), m.group(2)), cmd)


def execute(row):
    res = run_vim(row["buffer_before"], row["gold_command"], mode="ex", cursor=tuple(row.get("cursor") or (1, 1)))
    return res


def held_out_keys():
    from build_splits import leakage_keys
    keys = set()
    for split in ("val", "test"):
        for r in load_jsonl(REPO / "data" / "splits" / ("%s.jsonl" % split)):
            for k in leakage_keys(r):
                if k[0] != "paraphrase":
                    keys.add(k)
    return keys


def gate(row, held, train_buffers, train_pairs):
    """None if the row passes, else the reject reason. Sets buffer_after."""
    from build_splits import leakage_keys
    res = execute(row)
    if res.get("blocked"):
        return "blocked by the scope screen: %s" % res["blocked"]
    if res.get("timeout"):
        return "timeout"
    if not res.get("was_valid"):
        return "vim rejected it: %s" % (res.get("stderr") or "").strip()[:120]
    after = res.get("buffer_after") or ""
    if norm(after) == norm(row["buffer_before"]):
        return "no-op"
    blind = run_vim(row["buffer_before"], case_blind(row["gold_command"]), mode="ex",
                    cursor=tuple(row.get("cursor") or (1, 1)))
    if norm(blind.get("buffer_after") or "") != norm(after):
        return "case decides the result"
    probe = dict(row, id="__candidate__")
    clash = [k for k in leakage_keys(probe) if k[0] != "paraphrase" and k in held]
    if clash:
        return "leakage key shared with val/test: %s" % clash[0][0]
    if norm(row["buffer_before"]) in train_buffers:
        return "buffer already in train"
    if (row["instruction"], norm(row["buffer_before"])) in train_pairs:
        return "duplicate of a train row"
    row["buffer_after"] = after
    return None


def select(rows, target, seed):
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    cmd_n, sent_n, skill_n, dom_n = (collections.Counter() for _ in range(4))
    chosen = []
    for r in order:
        if len(chosen) >= target:
            break
        if cmd_n[r["gold_command"]] >= PER_COMMAND or sent_n[r["instruction"]] >= PER_SENTENCE:
            continue
        if skill_n[r["skill"]] >= SKILL_CAPS.get(r["skill"], 60) or dom_n[r["domain"]] >= DOMAIN_CAP:
            continue
        chosen.append(r)
        cmd_n[r["gold_command"]] += 1
        sent_n[r["instruction"]] += 1
        skill_n[r["skill"]] += 1
        dom_n[r["domain"]] += 1
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--authored-on", default="2026-10-16")
    args = ap.parse_args(argv)
    cands = load_jsonl(args.candidates)
    held = held_out_keys()
    train = load_jsonl(REPO / "data" / "splits" / "train.jsonl")
    train_buffers = {norm(r["buffer_before"]) for r in train}
    train_pairs = {(r["instruction"], norm(r["buffer_before"])) for r in train}
    # gate in a seeded order, but only as many as selection can use: gating every
    # candidate costs two vim runs each
    rng = random.Random(args.seed)
    order = list(cands)
    rng.shuffle(order)
    passed, rejects, reasons = [], [], collections.Counter()
    cmd_seen = collections.Counter()
    for r in order:
        if cmd_seen[r["gold_command"]] >= PER_COMMAND + 2:     # selection will not need more
            continue
        why = gate(r, held, train_buffers, train_pairs)
        if why:
            reasons[why.split(":")[0]] += 1
            rejects.append(dict(r, reject_reason=why))
            continue
        cmd_seen[r["gold_command"]] += 1
        passed.append(r)
    chosen = select(passed, args.target, args.seed)
    chosen.sort(key=lambda r: (r["domain"], r["skill"], r["cand"]))
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(chosen, 1):
        rows.append({
            "id": "%s%04d" % (ID_PREFIX, i), "category": "global", "tier": r["tier"], "mode": "ex",
            "instruction": r["instruction"], "buffer_before": r["buffer_before"],
            "gold_command": r["gold_command"], "buffer_after": r["buffer_after"], "cursor": [1, 1],
            "dataset": DATASET, "source": SOURCE, "authored_on": args.authored_on,
            "instruction_provenance": "claude-in-session, templated per skill and domain (scripts/global_additions/generate.py)",
            "domain": r["domain"], "skill": r["skill"], "why": r["why"], "needs_review": True,
        })
    with (out_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (out_dir / "rejects.jsonl").open("w", encoding="utf-8") as fh:
        for r in rejects:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": DATASET, "candidates": len(cands), "gated": len(passed) + len(rejects), "passed": len(passed),
        "selected": len(rows), "reject_reasons": dict(reasons.most_common()),
        "distinct_commands": len({r["gold_command"] for r in rows}),
        "distinct_instructions": len({r["instruction"] for r in rows}),
        "distinct_buffers": len({r["buffer_before"] for r in rows}),
        "by_skill": dict(collections.Counter(r["skill"] for r in rows).most_common()),
        "by_domain": dict(collections.Counter(r["domain"] for r in rows).most_common()),
        "by_tier": dict(sorted(collections.Counter(r["tier"] for r in rows).items())),
        "caps": {"per_command": PER_COMMAND, "per_sentence": PER_SENTENCE, "skills": SKILL_CAPS, "domain": DOMAIN_CAP},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("candidates", "gated", "passed", "selected", "distinct_commands",
                                               "reject_reasons", "by_tier")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
