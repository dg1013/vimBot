"""Headless-vim executor

Runs a vim command against a buffer in a sandboxed headless vim and returns
the resulting buffer. This is the trust boundary for model-generated commands,
and the ground-truth oracle for both evaluation and (later) RL-style training
with verifiable rewards.

Two modes:
  - "ex":   Ex commands, e.g.  %s/foo/bar/g   or   g/^$/d
            (leading ":" optional; it's stripped). Wrapped in
            try/execute/catch so an invalid command exits nonzero.
  - "keys": Normal-mode keystroke sequences, e.g.  dap  or  ci"hello<esc>
            Angle-bracket notation (<esc>, <cr>, <Tab>, <BS>, <C-a>, ...)
            is converted to raw bytes and replayed with `vim -s`.

Two entry points:
  - run_vim(buffer_before, ...)   buffer supplied as a string
  - run_vim_on_file(path, ...)    buffer read from a file on disk. The source
                                  file is READ-ONLY by default: only its contents enter
                                  the sandbox, vim never sees the real path. Pass
                                  output_path to write the result elsewhere;
                                  an existing output_path is refused by
                                  default (on_exists="refuse").

Command line:
    python src/executor.py --command '%s/foo/bar/g' --file notes.txt
    python src/executor.py --command 'dd' --mode keys --text $'a\\nb'
    python src/executor.py --command 'sort u' --file n.txt --output s.txt \\
                           --on-exists suffix
Resulting buffer goes to stdout, diagnostics to stderr. Exit 0 on success,
1 if the command was invalid, 2 if the command ran but --output could not
be written (it already existed and --on-exists was "refuse").

Safety measures (model output is untrusted input):
  - a static scope screen (src/scope.py) refuses damage-capable commands
    before vim is spawned -- writes, other-file reads, shell, scripting, and
    'set write', which would otherwise re-enable writing and defeat the
    'nowrite' guard below (verified: it does, in BOTH modes)
  - vim runs with -Z (restricted mode): :!, system(), :shell are disabled
  - -u NONE -i NONE -n: no user config, no viminfo, no swap
  - cwd is a throwaway temp dir that is deleted afterwards
  - hard wall-clock timeout kills hung sessions (interactive prompts,
    runaway :g commands, recursive macros)

Skipping the user config also skips vim's own defaults.vim, so the option
baseline has to be stated rather than inherited -- see _OPTION_BASELINE.

Note that precedence is cli then settings then default

Requires vim (not vim-tiny) on PATH.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from scope import screen_dangerous
from settings import load_settings

# Angle-bracket key notation -> raw bytes
_KEYMAP = {
    "<esc>": "\x1b",
    "<cr>": "\r",  # enter key
    "<enter>": "\r",
    "<return>": "\r",
    "<tab>": "\t",
    "<bs>": "\x08",  # backspace
    "<space>": " ",
    "<lt>": "<",
    "<gt>": ">",
    "<bar>": "|",
    "<up>": "\x1bOA",
    "<down>": "\x1bOB",
    "<right>": "\x1bOC",
    "<left>": "\x1bOD",
}


# Options applied before the untrusted command runs, in both modes.
#
# 'nowrite' is the sandbox guard (see screen_dangerous for why it is not
# sufficient on its own) and must come first.
#
# 'nrformats' is a correctness pin, not a safety one. -u NONE skips vim's
# defaults.vim, whose line 60 is `set nrformats-=octal`; neovim excludes octal
# by default too. Without this, <C-a> on 007 yields 010 here and 008 in every
# editor a user actually runs -- so the reward oracle, the MCP server and the
# neovim plugin would disagree about the same keystroke. Measured 2026-08-11:
# this is the only option difference that changes any dataset buffer between
# a bare -u NONE vim and a defaults.vim one.
#
# Keep this the single source for both modes: a setting added to only one of
# the two code paths is exactly the bug tests/test_executor.py pins against.
_OPTION_BASELINE = ("set nowrite", "set nrformats-=octal")


def convert_keys(seq: str) -> str:
    """Convert <esc>/<cr>/<C-x> style notation into raw key bytes."""
    out: list[str] = []
    i = 0
    while i < len(seq):
        ch = seq[i]
        if ch == "<":
            close = seq.find(">", i + 1)
            if close != -1:
                token = seq[i: close + 1].lower()
                if token in _KEYMAP:
                    out.append(_KEYMAP[token])
                    i = close + 1
                    continue
                # <C-a> style control keys
                if token.startswith("<c-") and len(token) == 5:
                    out.append(chr(ord(token[3]) & 0x1F))  # convert to ASCII number value and then back to char
                    i = close + 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def run_vim(
        buffer_before: str,
        command: str,
        mode: str = "ex",
        cursor: tuple[int, int] = (1, 1),
        timeout: float = 5.0,
        restricted: bool = True,
        vim_bin: str = "vim",
) -> dict:
    """Execute `command` against `buffer_before` in headless vim.

    _OPTION_BASELINE is applied first in both modes, so vim's options are a
    deliberate choice rather than whatever `-u NONE` happens to leave behind.

    Parameters:
        buffer_before (str): input buffer
        command (str): vim command
        mode (str):  = "ex" or "keys"
        cursor (tuple[int, int]): (line, column)
        timeout (float):  = seconds before process is killed
        restricted (bool):  = passes -Z; never set to False
        vim_bin (str):  =  binary to invoke, if missing FileNotFoundError

    Returns a dict:
      was_valid        bool  -- vim exited cleanly (in "ex" mode this also means
                            the command itself raised no error; in "keys"
                            mode it is best-effort: correctness should be
                            judged by comparing buffer_after)
      buffer_after str|None -- file content after :wq!, None if unreadable
      return_code  int|None -- 0 = success, 1 = failure, 2 = command ran, but no output written
      timeout      bool
      stderr       str   -- truncated diagnostics
      blocked      str   -- present ONLY when the command was refused by the
                           scope screen before vim was spawned; the reason.
                           Absent on every command that actually ran.
    """
    if mode not in ("ex", "keys"):
        raise ValueError(f"unknown mode: {mode!r}")

    # Static screen before vim is ever spawned. 'nowrite' alone is not enough:
    # a command containing 'set write' re-enables writing in BOTH modes
    # (verified empirically), so the guard must also refuse such commands.
    blocked = screen_dangerous(command, mode)
    if blocked is not None:
        return {
            "was_valid": False,
            "buffer_after": None,
            "return_code": None,
            "timeout": False,
            "stderr": f"blocked: {blocked}",
            "blocked": blocked,
        }

    tmpdir = tempfile.mkdtemp(prefix="vimBot_")
    try:
        buf_path = os.path.join(tmpdir, "buffer.txt")
        text = buffer_before
        if text and not text.endswith("\n"):
            text += "\n"
        with open(buf_path, "w", encoding="utf-8") as fh:
            fh.write(text)

        argv = [vim_bin]
        if restricted:
            argv.append("-Z")
        argv += ["-N", "-n", "-u", "NONE", "-i", "NONE", "--not-a-term"]

        env = dict(os.environ, TERM="dumb")

        if mode == "ex":
            cmd = command.strip()
            if cmd.startswith(":"):
                cmd = cmd[1:]
            # execute-with-quoting makes bars inside patterns safe, and
            # try/catch turns any command error into a nonzero exit (cquit).
            # 'nowrite' is set around the untrusted command so it cannot
            # write files anywhere; our own wq! re-enables writing first.
            quoted = cmd.replace("'", "''")
            wrapped = f"try | execute '{quoted}' | catch | cquit! | endtry"
            argv += ["-es"]
            for opt in _OPTION_BASELINE:
                argv += ["-c", opt]
            argv += [
                "-c", f"call cursor({cursor[0]},{cursor[1]})",
                "-c", wrapped,
                "-c", "set write",
                "-c", "wq!",
                buf_path,
            ]
            script_path = None
        else:
            keys = convert_keys(command)
            payload = (
                    "".join(f":{opt}\r" for opt in _OPTION_BASELINE)
                    + f":call cursor({cursor[0]},{cursor[1]})\r"
                    + keys
                    + "\x1b:set write\r:wq!\r"
            )
            script_path = os.path.join(tmpdir, "keys.bin")
            with open(script_path, "wb") as fh:
                fh.write(payload.encode("latin-1", "replace"))
            argv += ["-s", script_path, buf_path]

        try:
            proc = subprocess.run(
                argv,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "was_valid": False,
                "buffer_after": None,
                "return_code": None,
                "timeout": True,
                "stderr": "timeout",
            }

        buffer_after: Optional[str] = None
        if os.path.exists(buf_path):
            with open(buf_path, "r", encoding="utf-8", errors="replace") as fh:
                buffer_after = fh.read()

        return {
            "was_valid": proc.returncode == 0,
            "buffer_after": buffer_after,
            "return_code": proc.returncode,
            "timeout": False,
            "stderr": proc.stderr.decode("utf-8", errors="replace")[-500:],
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


ON_EXISTS_MODES = ("refuse", "overwrite", "suffix")

# Upper bound on the numbered variants tried in "suffix" mode, so a directory
# full of collisions fails loudly instead of spinning.
_SUFFIX_LIMIT = 500


def _numbered(path: str, n: int) -> str:
    """out.txt -> out_1.txt  (before the extension, so globs and file-type
    detection keep working; out.txt_1 would break both)."""
    root, ext = os.path.splitext(path)
    return f"{root}_{n}{ext}"


def _write_output(output_path, text: str, on_exists: str) -> str:
    """Write `text`, honoring the conflict policy. Returns the path written.

    Uses open(..., "x") -- an atomic create-if-absent -- rather than
    os.path.exists() followed by a write. The two-step version has a
    time-of-check/time-of-use window that concurrent workers (the GRPO reward
    pool is ~8 threads) can slip through; "x" closes it in one syscall.
    """
    if on_exists == "overwrite":
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(text)
        return str(output_path)

    if on_exists == "refuse":
        try:
            with open(output_path, "x", encoding="utf-8") as fh:
                fh.write(text)
        except FileExistsError:
            raise FileExistsError(
                f"{output_path} already exists and nothing was written. Pass "
                f"on_exists='overwrite' to replace it, or on_exists='suffix' "
                f"to write alongside it (CLI: --on-exists)."
            ) from None
        return str(output_path)

    # "suffix": first free numbered variant. Losing the race to another
    # worker just means the next candidate is tried, so this stays correct
    # under concurrency instead of silently clobbering.
    candidate = str(output_path)
    for n in range(_SUFFIX_LIMIT + 1):
        if n:
            candidate = _numbered(str(output_path), n)
        try:
            with open(candidate, "x", encoding="utf-8") as fh:
                fh.write(text)
            return candidate
        except FileExistsError:
            continue
    raise FileExistsError(
        f"{output_path}: no free filename after {_SUFFIX_LIMIT} attempts"
    )


def run_vim_on_file(
        path,
        command: str,
        mode: str = "ex",
        cursor: tuple = (1, 1),
        timeout: float = 5.0,
        restricted: bool = True,
        vim_bin: str = "vim",
        output_path=None,
        on_exists: str = "refuse",
) -> dict:
    """Execute `command` against the contents of the file at `path`.

    The source file is READ-ONLY. Only its contents are copied into the
    sandbox; vim is never pointed at the real path, so every guarantee of
    run_vim() (restricted mode, 'nowrite', throwaway cwd, hard timeout)
    applies unchanged and no generated command can damage the original.

    `path` may be a str or os.PathLike. A missing file raises
    FileNotFoundError -- that is a usage error, distinct from the
    `was_valid: False` result that signals a failing *command*.

    If `output_path` is given, the resulting buffer is written there, but
    only when the command ran successfully. Writing to the source file is
    refused (ValueError) -- that is what makes "read-only" an enforced
    invariant rather than a convention.

    `on_exists` decides what happens when output_path is already taken:
      "refuse"    (default) raise FileExistsError, write nothing
      "overwrite" replace it
      "suffix"    write out_1.txt, out_2.txt, ... -- never overwrites, but
                  the caller MUST read written_to, since the result did not
                  land where it was asked to go

    Returns the run_vim() dict plus:
      source_path  str        the file that was read
      written_to   str|None   the file ACTUALLY written (may differ from
                              output_path under "suffix"), or None if
                              nothing was written
    """
    if on_exists not in ON_EXISTS_MODES:
        raise ValueError(
            f"unknown on_exists: {on_exists!r} (expected one of {ON_EXISTS_MODES})"
        )

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        buffer_before = fh.read()

    if output_path is not None:
        source_real = os.path.realpath(path)
        if os.path.realpath(output_path) == source_real:
            raise ValueError(
                "output_path must differ from the source file: "
                "run_vim_on_file never modifies its input"
            )

    result = run_vim(
        buffer_before, command, mode=mode, cursor=cursor,
        timeout=timeout, restricted=restricted, vim_bin=vim_bin,
    )
    result["source_path"] = str(path)
    result["written_to"] = None

    if output_path is not None and result["was_valid"] and result["buffer_after"] is not None:
        text = result["buffer_after"]
        if text and not text.endswith("\n"):
            text += "\n"
        result["written_to"] = _write_output(output_path, text, on_exists)

    return result


def _main(argv=None) -> int:
    """CLI: run one command against a file or a literal string.

    Flag defaults come from the settings file (configs/settings.toml), so
    precedence is: CLI flag > settings file > built-in default.
    """
    import argparse

    # Two-pass parse: --settings must be resolved before it can supply the
    # defaults for every other flag.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings", help="path to a settings TOML file "
                                        "(default: configs/settings.toml)")
    pre.add_argument("--show-settings", action="store_true",
                     help="print the effective settings and their source, "
                          "then exit")
    known, _ = pre.parse_known_args(argv)
    cfg = load_settings(known.settings)["executor"]

    if known.show_settings:
        print(load_settings(known.settings).describe())
        return 0

    ap = argparse.ArgumentParser(
        prog="executor", parents=[pre],
        description="Run a single vim command against a buffer in a sandboxed "
                    "headless vim and print the resulting buffer.",
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", help="read the buffer from this file "
                                    "(never modified)")
    src.add_argument("--text", help="use this string as the buffer")
    ap.add_argument("--command", required=True, help="the vim command to run")
    ap.add_argument("--mode", choices=("ex", "keys"), default=cfg["mode"])
    ap.add_argument("--line", type=int, default=1, help="cursor line (1-based)")
    ap.add_argument("--col", type=int, default=1, help="cursor column (1-based)")
    ap.add_argument("--timeout", type=float, default=cfg["timeout"])
    ap.add_argument("--vim", default=cfg["vim_bin"], help="vim binary to use")
    ap.add_argument("--output", help="write the resulting buffer here "
                                     "(must differ from --file)")
    ap.add_argument("--on-exists", choices=ON_EXISTS_MODES,
                    default=cfg["on_exists"],
                    help="what to do if --output already exists "
                         f"(default: {cfg['on_exists']}, from settings)")
    args = ap.parse_args(argv)

    # NOTE: the sandbox is deliberately not configurable from the CLI.
    common = dict(
        mode=args.mode, cursor=(args.line, args.col),
        timeout=args.timeout, vim_bin=args.vim,
    )
    write_error = None
    if args.file is not None:
        try:
            result = run_vim_on_file(args.file, args.command,
                                     output_path=args.output,
                                     on_exists=args.on_exists, **common)
        except FileExistsError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2  # nothing was written; re-run without --output to see it
    else:
        result = run_vim(args.text, args.command, **common)
        result["written_to"] = None
        if args.output is not None and result["was_valid"] \
                and result["buffer_after"] is not None:
            text = result["buffer_after"]
            if text and not text.endswith("\n"):
                text += "\n"
            try:
                result["written_to"] = _write_output(args.output, text,
                                                     args.on_exists)
            except FileExistsError as exc:
                write_error = exc

    if result["buffer_after"] is not None:
        sys.stdout.write(result["buffer_after"])

    status = "timeout" if result["timeout"] else f"rc={result['return_code']}"
    print(f"[was_valid={result['was_valid']} {status}]", file=sys.stderr)
    if result["stderr"].strip():
        print(result["stderr"].strip(), file=sys.stderr)
    if result["written_to"]:
        print(f"[wrote {result['written_to']}]", file=sys.stderr)
    if write_error is not None:
        print(f"error: {write_error}", file=sys.stderr)
        return 2  # command may be fine; the OUTPUT was not written

    return 0 if result["was_valid"] else 1


if __name__ == "__main__":
    sys.exit(_main())
