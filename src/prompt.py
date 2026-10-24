"""The §6 model I/O contract, implemented once.

CLAUDE.md §6 fixes the prompt and the output grammar, and §4 requires the
prompt text to be identical for every model. This module is the single
implementation: baselines, SFT (supervised fine-tuning), GRPO (Group Relative Policy Optimization), the MCP (Model Context Protocol) server and the frontier batch
renderer import it. Nothing else may carry a copy of the system text, the
user-turn layout, or the output regex (regular expression). The neovim plugin is Lua and cannot
import it; ``python3 src/prompt.py --system`` prints the text so the plugin's
copy can be diffed against it by a test.

Prompt (rendered through the model's chat template, thinking disabled):

    system: You translate natural-language editing instructions into a single
            vim command. Output exactly one line in the form
            "ex: <command>" or "keys: <keystrokes>". No explanation.

    user:   Instruction: <instruction>
            Cursor: line <L>, column <C>
            Buffer:
            <<<
            <buffer_before>
            >>>

k-shot examples (baselines only, §8) are prepended as alternating
user/assistant turns, each assistant turn in the exact target form, so the
model sees the answer shape rather than a description of it.

Output grammar (strict): ``^(ex|keys): ?(.+)$`` on the first line; everything
else is a parse failure, scored invalid. Three details this module fixes,
recorded in DECISIONS.md 2026-09-04:

* Leading whitespace, including blank lines, is skipped before "the first
  line" is read. Chat templates and samplers routinely emit a newline before
  the content; counting that as a format failure would measure the template,
  not the model. Any *text* before the answer line still fails.
* Trailing whitespace on the answer line is dropped (a keystroke that is a
  literal trailing space is not expressible; none exists in the corpus).
  Lines after the first are ignored by the parser and charged by the token
  metric (§11), which is the penalty §6's "exactly one line" intends.
* Any ``<think>`` or ``</think>`` tag anywhere in the completion is a format
  failure for every model (§4), which keeps token and cost numbers
  comparable across model swaps.
"""

import re
import sys

SYSTEM_TEXT = (
    'You translate natural-language editing instructions into a single\n'
    'vim command. Output exactly one line in the form\n'
    '"ex: <command>" or "keys: <keystrokes>". No explanation.'
)

MODES = ("ex", "keys")

OUTPUT_RE = re.compile(r"^(ex|keys): ?(.+)$")
_THINK_RE = re.compile(r"</?think>")


def render_user(instruction: str, buffer_before: str, cursor=(1, 1)) -> str:
    """The user turn, byte for byte as §6 draws it."""
    line, col = cursor
    return (
        f"Instruction: {instruction}\n"
        f"Cursor: line {line}, column {col}\n"
        f"Buffer:\n<<<\n{buffer_before}\n>>>"
    )


def render_target(mode: str, command: str) -> str:
    """The one-line completion the model must produce: ``ex: <command>``."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return f"{mode}: {command}"


def build_messages(instruction: str, buffer_before: str, cursor=(1, 1),
                   examples=()) -> list:
    """Chat messages for one case: system, k-shot pairs, then the user turn.

    ``examples`` are §5 rows (instruction, buffer_before, cursor, mode,
    gold_command); each becomes a user/assistant pair ahead of the query.
    """
    messages = [{"role": "system", "content": SYSTEM_TEXT}]
    for ex in examples:
        messages.append({"role": "user", "content": render_user(
            ex["instruction"], ex["buffer_before"], ex.get("cursor", (1, 1)))})
        messages.append({"role": "assistant", "content": render_target(
            ex["mode"], ex["gold_command"])})
    messages.append({"role": "user",
                     "content": render_user(instruction, buffer_before, cursor)})
    return messages


def messages_for_case(row: dict, examples=()) -> list:
    """``build_messages`` from a §5 case row."""
    return build_messages(row["instruction"], row["buffer_before"],
                          row.get("cursor", (1, 1)), examples)


def parse_output(text):
    """Apply the §6 grammar to a completion.

    Returns ``(mode, command)`` or ``None`` for a parse failure. See the
    module docstring for the three fixed details.
    """
    if text is None or _THINK_RE.search(text):
        return None
    first = text.lstrip().split("\n", 1)[0].rstrip()
    m = OUTPUT_RE.match(first)
    if not m:
        return None
    return m.group(1), m.group(2)


def _main(argv=None) -> int:
    """CLI (command-line interface): print the system text, or render one case from a JSONL (JSON Lines) file."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Print the §6 prompt pieces.")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--system", action="store_true",
                      help="print the system text and exit")
    what.add_argument("--case", metavar="JSONL",
                      help="render one case's user turn from this file")
    ap.add_argument("--id", help="case id within --case (default: first row)")
    ap.add_argument("--messages", action="store_true",
                    help="with --case: print the full chat messages as JSON")
    args = ap.parse_args(argv)

    if args.system:
        print(SYSTEM_TEXT)
        return 0
    with open(args.case, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.id:
        row = next((r for r in rows if r["id"] == args.id), None)
    else:
        row = rows[0] if rows else None
    if row is None:
        print(f"no row with id {args.id!r} in {args.case}", file=sys.stderr)
        return 1
    if args.messages:
        print(json.dumps(messages_for_case(row), indent=2, ensure_ascii=False))
    else:
        print(render_user(row["instruction"], row["buffer_before"],
                          row.get("cursor", (1, 1))))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
