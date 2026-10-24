#!/usr/bin/env python3
"""Render paste-ready payloads for the in-session frontier baseline.

CLAUDE.md §8 requires a frontier row (k=0 and k=5 on the test slice) before
any training. Transport (DECISIONS 2026-08-13): a dedicated parallel Claude
chat, closed book, on Claude Opus 5. This script renders, per condition:

    runs/frontier/<k0|k5>/00_first_message.txt   protocol (+ k-shot examples)
    runs/frontier/<k0|k5>/batch_NN.txt           50 cases each, id-tagged

Workflow: paste 00_first_message.txt as the parallel chat's first message,
wait for READY, then paste each batch in order and save each reply verbatim
to a text file. Feed the saved replies to scripts/ingest_frontier_preds.py.

Payloads contain instructions/buffers only — never gold commands. The k=5
examples are seeded train-split cases, identical for every test case, shown
in the exact answer format the chat must produce.
"""

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "deep_regex"))
from garble_screen import classify as garble_rules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from prompt import SYSTEM_TEXT, render_user  # noqa: E402

# The §6 system text and user-turn layout come from src/prompt.py, the
# single implementation of the contract; this script only adds the paste
# protocol around them.

PROTOCOL = """\
This chat is a closed-book baseline run for a benchmark. Follow these rules
for the entire chat.

The task, exactly as the system prompt under evaluation states it:

<system>
{system}
</system>

Protocol:
- I will paste numbered batches of cases. Each case looks like:
  ### <case-id>
  Instruction: ...
  Cursor: line <L>, column <C>
  Buffer:
  <<<
  ...buffer lines...
  >>>
- The buffer is shown verbatim between the <<< and >>> markers (the markers
  are not part of it). The first buffer line is line 1.
- For every case, reply with exactly one line: the case id, one space, then
  your answer in the required form. Example:
  some-case-id ex: g/foo/d
- One line per case, in the same order as the batch. No other text: no
  commentary, no code fences, no blank lines between answers, no questions.
- Closed book: do not use tools, do not read or open any files, do not
  browse. Work only from the text of each case.
{examples}
Reply to this message with exactly: READY
"""

EXAMPLES_HEADER = """
Worked examples — these show the required behavior; answer every future case
the same way:
"""


def render_case(row):
    # id tag for the paste protocol, then the §6 user turn verbatim
    return f"### {row['id']}\n" + render_user(
        row["instruction"], row["buffer_before"], row.get("cursor", [1, 1]))


def render_example(row):
    return (
        f"{render_case(row)}\n"
        f"Correct answer:\n"
        f"{row['id']} {row['mode']}: {row['gold_command']}"
    )


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_examples(train_rows, shots, seed):
    rng = random.Random(seed)
    # garble-screened pool only — an exemplar must not model the defect
    # families the review purged (see scripts/deep_regex/garble_screen.py)
    clean = [r for r in train_rows
             if not garble_rules(r.get("source_regex"), r["instruction"])]
    # one example per template family so the shots aren't near-duplicates
    by_template = {}
    for row in rng.sample(clean, len(clean)):
        by_template.setdefault(row.get("template_id", row["id"]), row)
    pool = list(by_template.values())
    return rng.sample(pool, shots)


def write_condition(outdir, cases, batch_size, examples):
    outdir.mkdir(parents=True, exist_ok=True)
    ex_text = ""
    if examples:
        ex_text = EXAMPLES_HEADER + "\n" + "\n\n".join(
            render_example(r) for r in examples
        ) + "\n"
    first = PROTOCOL.format(system=SYSTEM_TEXT, examples=ex_text)
    (outdir / "00_first_message.txt").write_text(first, encoding="utf-8")

    batches = [cases[i:i + batch_size] for i in range(0, len(cases), batch_size)]
    for n, batch in enumerate(batches, 1):
        body = f"Batch {n} of {len(batches)} — {len(batch)} cases.\n\n"
        body += "\n\n".join(render_case(r) for r in batch) + "\n"
        (outdir / f"batch_{n:02d}.txt").write_text(body, encoding="utf-8")
    return len(batches)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default="data/splits/test.jsonl")
    ap.add_argument("--train", default="data/splits/train.jsonl")
    ap.add_argument("--out", default="runs/frontier")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260813)
    args = ap.parse_args()

    cases = load_jsonl(args.cases)
    train = load_jsonl(args.train)
    out = Path(args.out)

    n0 = write_condition(out / "k0", cases, args.batch_size, [])
    examples = pick_examples(train, args.shots, args.seed)
    n5 = write_condition(out / "k5", cases, args.batch_size, examples)

    manifest = {
        "cases": args.cases,
        "n_cases": len(cases),
        "batch_size": args.batch_size,
        "batches_per_condition": {"k0": n0, "k5": n5},
        "shot_seed": args.seed,
        "k5_example_ids": [r["id"] for r in examples],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(cases)} cases -> {n0} batches per condition under {out}/")
    print(f"k=5 examples: {', '.join(manifest['k5_example_ids'])}")


if __name__ == "__main__":
    main()
