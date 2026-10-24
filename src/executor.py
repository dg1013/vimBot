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
                                  file is READ-ONLY: only its contents enter
                                  the sandbox, vim never sees the real path,
                                  and nothing is ever written back. The
                                  executor writes no files at all (owner
                                  decision 2026-09-04): persistence belongs
                                  to the harness, and in real use to the user.

Command line:
    python src/executor.py --command '%s/foo/bar/g' --file notes.txt
    python src/executor.py --command 'dd' --mode keys --text $'a\\nb'
Resulting buffer goes to stdout, diagnostics to stderr. Exit 0 on success,
1 if the command was invalid. Redirect stdout to keep a result.

Safety measures (model output is untrusted input):
  - a static scope screen (src/scope.py) refuses damage-capable commands
    before vim is spawned -- writes, other-file reads, shell, scripting, and
    'set write', which would otherwise re-enable writing and defeat the
    'nowrite' guard below (verified: it does, in BOTH modes)
  - vim runs with -Z (restricted mode): :!, system(), :shell are disabled
  - -u NONE -i NONE -n: no user config (configuration), no viminfo, no swap
  - cwd is a throwaway temp dir (directory) that is deleted afterwards
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
import threading
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
# editor a user actually runs -- so the reward oracle, the MCP (Model Context Protocol) server and the
# neovim plugin would disagree about the same keystroke. Measured 2026-08-11:
# this is the only option difference that changes any dataset buffer between
# a bare -u NONE vim and a defaults.vim one.
#
# 'textwidth' is deliberately NOT pinned here, and the reason is worth stating
# because it looks like it belongs. Found 2026-08-26: with textwidth at its
# default of 0, `gq` wraps at min(columns, 79), so reflow gold depends on the
# COLUMNS *environment variable* of whatever spawned vim -- the same `gqq`
# gives 2 lines under a default terminal and 4 under COLUMNS=40. That is the
# nrformats hazard again, sourced from the environment.
#
# Pinning `set textwidth=79` was tried and rejected: textwidth also drives
# insert-mode auto-wrap, so it silently rewraps long inserted text for every
# i/a/o/c command, not just for gq. Measured, it reproduced all 11 pre-existing
# gq golds but broke hc-0663 (a `cis` whose replacement sentence auto-wrapped).
# Trading a behavior change across the whole insert-edit category for 30 reflow
# rows is the wrong direction, and most editors a user runs have textwidth=0
# for code anyway.
#
# The convention instead: a reflow case states its own width in its command
# (`:set tw=72<CR>gqip`), which makes that row reproducible on any terminal and
# self-documenting about the width it expects. tests/test_executor.py pins both
# halves of this -- the hazard, so it is not rediscovered, and the fix.
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


# ---------------------------------------------------------------- isolation
#
# The static screen in src/scope.py is a denylist over a language with
# abbreviations and dynamic evaluation. On 2026-08-18 it was defeated sixteen
# different ways -- expression contexts reaching writefile(), plus four plain
# command words nobody had enumerated. The screen was fixed, but the lesson is
# the one src/scope.py's own docstring states: a denylist raises the bar, it
# does not close the door.
#
# This is the door. The vim process runs inside an unprivileged user +
# mount namespace where the whole filesystem is read-only EXCEPT the throwaway
# temp directory holding the buffer. writefile() to any other path then fails
# at the kernel, whatever the command string says.
#
# Requires only util-linux `unshare` and unprivileged user namespaces; no
# daemon, no image, no root. Where that is unavailable (macOS, hardened
# kernels), behaviour follows the [executor] isolation setting:
#   "namespace" -- use it when available, warn loudly once when not (default)
#   "required"  -- refuse to run without it
#   "none"      -- disabled; the screen is then the only protection
_ISOLATION_SH = (
    'mount --make-rprivate / 2>/dev/null;'
    'mount --bind "$1" "$1" 2>/dev/null;'
    'mount -o remount,bind,ro / 2>/dev/null;'
    'mount -o remount,bind,rw "$1" 2>/dev/null;'
    'shift; exec "$@"'
)

_ISOLATION_PROBE: bool | None = None
_ISOLATION_WARNED = False
# The probe takes ~200 ms (milliseconds). Without this lock, a thread pool (the sec 9 GRPO (Group Relative Policy Optimization)
# reward workers use ~8) races: the first caller has not finished probing when
# the others look, and every one of them silently runs vim with no OS (operating system) boundary.
# Found 2026-08-18 by watching the fallback warning fire under a parallel gate.
_ISOLATION_LOCK = threading.Lock()


def _isolate(argv: list, sandbox_dir: str) -> list:
    """Wrap argv so it runs with everything but `sandbox_dir` read-only."""
    return ["unshare", "--user", "--map-root-user", "--mount",
            "sh", "-c", _ISOLATION_SH, "sh", sandbox_dir] + argv


def isolation_available() -> bool:
    """Probe once whether the namespace actually blocks an outside write.

    Deliberately empirical: `unshare` existing proves nothing, because the
    remount can silently fail on some kernels and leave the process with a
    fully writable filesystem.
    """
    global _ISOLATION_PROBE
    if _ISOLATION_PROBE is not None:
        return _ISOLATION_PROBE
    with _ISOLATION_LOCK:
        # re-check: another thread may have finished probing while we waited
        if _ISOLATION_PROBE is not None:
            return _ISOLATION_PROBE
        if shutil.which("unshare") is None:
            _ISOLATION_PROBE = False
            return False
        outside = tempfile.mkdtemp(prefix="vimBot_probe_out_")
        inside = tempfile.mkdtemp(prefix="vimBot_probe_in_")
        result = False
        try:
            blocked = subprocess.run(
                _isolate(["touch", os.path.join(outside, "x")], inside),
                capture_output=True, timeout=10).returncode != 0
            allowed = subprocess.run(
                _isolate(["touch", os.path.join(inside, "x")], inside),
                capture_output=True, timeout=10).returncode == 0
            result = bool(blocked and allowed)
        except (OSError, subprocess.SubprocessError):
            result = False
        finally:
            shutil.rmtree(outside, ignore_errors=True)
            shutil.rmtree(inside, ignore_errors=True)
        _ISOLATION_PROBE = result
        return result


def _apply_isolation(argv: list, sandbox_dir: str, policy: str) -> list:
    global _ISOLATION_WARNED
    if policy == "none":
        return argv
    if isolation_available():
        return _isolate(argv, sandbox_dir)
    if policy == "required":
        raise RuntimeError(
            "[executor] isolation = 'required' but unprivileged user namespaces "
            "are unavailable here; refusing to run vim without an OS boundary."
        )
    if not _ISOLATION_WARNED:
        _ISOLATION_WARNED = True
        print("warning: OS isolation unavailable (no usable user namespace); "
              "src/scope.py is the only protection for this run.", file=sys.stderr)
    return argv


def run_vim(
        buffer_before: str,
        command: str,
        mode: str = "ex",
        cursor: tuple[int, int] = (1, 1),
        timeout: float = 5.0,
        restricted: bool = True,
        vim_bin: str = "vim",
        isolation: str = "namespace",
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

    An ex command holding a NUL byte cannot be passed to vim and comes back
    invalid (was_valid False, buffer_after None) without vim being spawned.
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

    # An ex command reaches vim in argv, and argv cannot carry a NUL byte
    # (execve takes NUL-terminated strings; subprocess raises ValueError
    # rather than truncate). Raising here aborted every caller -- a whole
    # evaluate.score call, the §9 reward -- when the base model sampled
    # `2\x00a` at T = 1 (2026-09-22, DECISIONS.md). No other channel carries
    # the command as written either: a sourced script cuts the line at the
    # NUL, so `s/foo/<NUL>/` would run as `s/foo/` and exit 0. So it is
    # invalid, returned like any vim error; never stripped, which would turn
    # a right answer plus a stray NUL into a match. After the screen, so a
    # dangerous command still reports `blocked`; keys mode is untouched, since
    # keys go through a file, where NUL is a key (insert-mode CTRL-@).
    if mode == "ex" and "\x00" in command:
        return {
            "was_valid": False,
            "buffer_after": None,
            "return_code": None,
            "timeout": False,
            "stderr": "invalid: NUL byte in an ex command (argv cannot carry it to vim)",
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
            # The exception text (e.g. "Vim(substitute):E486: Pattern not
            # found: foo") is written to a file INSIDE the sandbox dir -- the
            # one place vim may write -- and surfaces in the result's
            # `stderr`, so a failing command is diagnosable (2026-09-05, for
            # the §9 error analysis). 'nowrite' is set around the untrusted
            # command so it cannot write files anywhere; our own wq!
            # re-enables writing first.
            quoted = cmd.replace("'", "''")
            err_path = os.path.join(tmpdir, "exception.txt")
            wrapped = (f"try | execute '{quoted}' | catch | "
                       f"call writefile([v:exception], '{err_path}') | "
                       f"cquit! | endtry")
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

        argv = _apply_isolation(argv, tmpdir, isolation)

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

        result = {
            "was_valid": proc.returncode == 0,
            "buffer_after": buffer_after,
            "return_code": proc.returncode,
            "timeout": False,
            "stderr": proc.stderr.decode("utf-8", errors="replace")[-1000:],
        }
        exc_path = os.path.join(tmpdir, "exception.txt")
        if os.path.exists(exc_path):
            with open(exc_path, "r", encoding="utf-8", errors="replace") as fh:
                exc = fh.read().strip()
            if exc:
                result["stderr"] = ("vim: " + exc + "\n" + result["stderr"])[-1000:]
        return result
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_vim_on_file(
        path,
        command: str,
        mode: str = "ex",
        cursor: tuple = (1, 1),
        timeout: float = 5.0,
        restricted: bool = True,
        vim_bin: str = "vim",
) -> dict:
    """Execute `command` against the contents of the file at `path`.

    The source file is READ-ONLY. Only its contents are copied into the
    sandbox; vim is never pointed at the real path, so every guarantee of
    run_vim() (restricted mode, 'nowrite', throwaway cwd, hard timeout)
    applies unchanged and no generated command can damage the original.
    Nothing is written anywhere: the executor returns the buffer and the
    caller decides what to do with it (train/test/validation only, per the
    §1 execution boundary; owner decision 2026-09-04).

    `path` may be a str or os.PathLike. A missing file raises
    FileNotFoundError -- that is a usage error, distinct from the
    `was_valid: False` result that signals a failing *command*.

    Returns the run_vim() dict plus:
      source_path  str        the file that was read
    """
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        buffer_before = fh.read()

    result = run_vim(
        buffer_before, command, mode=mode, cursor=cursor,
        timeout=timeout, restricted=restricted, vim_bin=vim_bin,
    )
    result["source_path"] = str(path)
    return result


def _main(argv=None) -> int:
    """CLI (command-line interface): run one command against a file or a literal string.

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
    args = ap.parse_args(argv)

    # NOTE: the sandbox is deliberately not configurable from the CLI.
    common = dict(
        mode=args.mode, cursor=(args.line, args.col),
        timeout=args.timeout, vim_bin=args.vim,
    )
    if args.file is not None:
        result = run_vim_on_file(args.file, args.command, **common)
    else:
        result = run_vim(args.text, args.command, **common)

    if result["buffer_after"] is not None:
        sys.stdout.write(result["buffer_after"])

    status = "timeout" if result["timeout"] else f"rc={result['return_code']}"
    print(f"[was_valid={result['was_valid']} {status}]", file=sys.stderr)
    if result["stderr"].strip():
        print(result["stderr"].strip(), file=sys.stderr)

    return 0 if result["was_valid"] else 1


if __name__ == "__main__":
    sys.exit(_main())
