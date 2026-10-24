"""Pasteable rendering of model output for real-use surfaces (§10).

Owner decision 2026-09-04 (DECISIONS.md): keys-mode commands reach the user
as a single ex line they can paste on vim's command line,

    keys: ci"foo<Esc>       ->      :exe "normal! ci\\"foo\\<Esc>"

and ex-mode commands as themselves behind a leading colon. The model still
emits the §6 forms; this module is the last step before a human sees the
command, in the plugin, the MCP (Model Context Protocol) server and the CLI (command-line interface). Surfaces show the raw
keys beside the rendered line, so the user still learns the keystrokes.

It is deliberately NOT something the executor will run: src/scope.py refuses
:execute and :normal in model output, and tests/test_paste.py pins that the
rendered form is blocked. Rendering is for people, not for the sandbox.

Fidelity, measured 2026-09-04 over every keys row in the corpus: 1,812 of
1,819 sequences reproduce the identical buffer through this rendering. The
7 that do not share two causes, both inherent to :normal rather than to the
escaping: :normal aborts the rest of the sequence at the first failing
motion or beep, where typed keys just continue; and state-dependent
commands (changelist jumps g; and g,, macro recording q, the :s///c confirm
prompt, <C-q> standing in for <C-v>) behave differently non-interactively.

Rendering rules for the double-quoted vimscript string:

    \\   ->  \\\\          "  ->  \\"          |  ->  \\<Bar>   (keeps the bar
                                                          out of the ex parser)
    every notation token convert_keys() understands -- <Esc>, <CR>, <Tab>,
    <BS>, <Up>..., <Bar>, <C-x> -- becomes \\<Token>, spelling kept;
    <lt>, <gt> and <Space> become the literal characters, which need no
    escaping inside the string;
    any other <...> stays literal text, exactly as convert_keys() treats it;
    a leading space would be skipped by :normal, so it is rendered as "1 "
    (a count of one, then the space motion), per :help :normal.

The bang matters: `normal!` ignores the user's mappings, so the pasted line
does what the sandbox did, not what the user's configuration would make of
the same keys.
"""

import sys

from executor import _KEYMAP

MODES = ("ex", "keys")

# notation that is a plain character inside a double-quoted string
_LITERAL = {"<lt>": "<", "<gt>": ">", "<space>": " "}


def _is_notation(token_lower: str) -> bool:
    """The exact token set convert_keys() converts; everything else is text."""
    return token_lower in _KEYMAP or (
        token_lower.startswith("<c-") and len(token_lower) == 5)


def _escape_char(ch: str) -> str:
    if ch == "\\":
        return "\\\\"
    if ch == '"':
        return '\\"'
    if ch == "|":
        return "\\<Bar>"
    return ch


def escape_keys(keys: str) -> str:
    """Keys in corpus notation -> the body of a `normal!` string literal."""
    out, i = [], 0
    while i < len(keys):
        ch = keys[i]
        if ch == "<":
            close = keys.find(">", i + 1)
            if close != -1:
                token = keys[i:close + 1]
                low = token.lower()
                if low in _LITERAL:
                    out.append(_escape_char(_LITERAL[low]))
                    i = close + 1
                    continue
                if _is_notation(low):
                    out.append("\\" + token)
                    i = close + 1
                    continue
        out.append(_escape_char(ch))
        i += 1
    body = "".join(out)
    if body.startswith(" "):
        body = "1" + body
    return body


def pasteable(mode: str, command: str) -> str:
    """The one line a user pastes on vim's command line for this command."""
    if mode == "ex":
        return ":" + command.lstrip(":")
    if mode == "keys":
        return ':exe "normal! ' + escape_keys(command) + '"'
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _main(argv=None) -> int:
    """CLI: render a mode+command pair, or a whole §6 completion."""
    import argparse
    from prompt import parse_output

    ap = argparse.ArgumentParser(
        description="Render a vimBot command as one pasteable command line.")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument("--completion", metavar="TEXT",
                      help='a model completion such as \'keys: ci"x<Esc>\'')
    what.add_argument("--command", help="the command (with --mode)")
    ap.add_argument("--mode", choices=MODES, default="ex")
    args = ap.parse_args(argv)

    if args.completion is not None:
        parsed = parse_output(args.completion)
        if parsed is None:
            print("parse failure: not in the form 'ex: <command>' or "
                  "'keys: <keystrokes>'", file=sys.stderr)
            return 1
        mode, command = parsed
    else:
        mode, command = args.mode, args.command
    print(pasteable(mode, command))
    return 0


if __name__ == "__main__":
    sys.exit(_main())
