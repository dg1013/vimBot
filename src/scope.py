"""Static screens for damage-capable vim commands.

This is the *policy* half of the trust boundary; src/executor.py is the
mechanism half. run_vim() calls screen_dangerous() before spawning vim, so
every consumer of the executor -- the GRPO reward loop, the MCP server, the
neovim plugin -- inherits the same protection that the data pipeline has.

WHAT THIS IS: defense in depth. It raises the bar substantially against
commands that write files, read other files, shell out, or disable the
executor's own 'nowrite' guard.

WHAT THIS IS NOT: a boundary. It is a denylist over a language with
abbreviations, dynamic evaluation, and modal parsing, so a determined
adversary who controls the command string will eventually get past it. The
real boundary is OS-level -- run generation and RL workers in a container as
an unprivileged user (CLAUDE.md sec 15). Treat a bypass here as a bug worth
fixing, not as a breach of a guarantee we claimed.

Known deliberate imprecision, in the safe direction:
  - In keys mode every ':' begins a candidate ex fragment, even one typed as
    literal text (e.g. "oemail: x<Esc>"). Such a command is rejected rather
    than risk missing a real ':w'. No case in the current dataset has a ':'
    in a keys command.
  - '@' anywhere in a keys command is rejected (macro playback), which also
    rejects a literal '@' typed into a buffer.

Stdlib only.
"""

import re

# --- ex command words that can cause damage -------------------------------
#
# Grouped by why they are here, so the list stays auditable. Vim accepts any
# unambiguous prefix of a command name, so prefixes are enumerated too.

_WRITE_FILE = {
    # :write and friends -- the direct write vector
    "w", "wr", "wri", "writ", "write", "wa", "wall",
    "wq", "wqa", "wqall", "x", "xit", "xa", "xall", "exi", "exit",
    "up", "upd", "upda", "updat", "update", "sav", "save", "saveas",
    # commands whose whole purpose is writing a file somewhere
    "mk", "mke", "mkexrc", "mks", "mksession", "mkv", "mkvie", "mkview",
    "wv", "wviminfo", "wundo",
}

_READ_OTHER_FILE = {
    # pull another file's contents into the buffer -> exfiltration via
    # buffer_after, which the caller reads
    "e", "ed", "edi", "edit", "ene", "enew", "r", "re", "rea", "read",
    "vi", "view", "vie", "fin", "find", "n", "ne", "nex", "next",
    "prev", "previous", "rew", "rewind", "fir", "first", "la", "last",
    "b", "bu", "buf", "buffer", "bn", "bnext", "bp", "bprev", "bprevious",
    "ar", "arg", "args", "arga", "argadd",
    "sp", "split", "vs", "vsp", "vsplit", "new", "vnew",
    "tabe", "tabedit", "tabnew",
    "so", "sou", "sourc", "source", "ru", "runtime", "rv", "rviminfo",
    "cd", "chd", "chdir", "lcd", "tcd",
}

_SHELL = {
    # already blocked by -Z; kept as a second layer and for clear reporting
    "sh", "shell", "ter", "term", "terminal", "!",
}

_SCRIPTING = {
    # can construct a write dynamically, or replay arbitrary keys
    "norm", "norma", "normal", "exe", "exec", "execu", "execut", "execute",
    "cal", "call", "let", "unl", "unlet", "sil", "silent",
    "py", "pyt", "pyth", "pytho", "python", "py3", "python3", "pyx",
    "lua", "perl", "rub", "ruby", "mz", "mzscheme", "tcl",
    "fu", "fun", "func", "functi", "functio", "function",
    "sandbox", "debug",
}

_DANGEROUS_WORDS = _WRITE_FILE | _READ_OTHER_FILE | _SHELL | _SCRIPTING

# Commands whose arguments are checked for re-enabling writes. ':set write'
# defeats the executor's 'nowrite' guard -- verified empirically, in BOTH
# ex and keys mode.
_SET_WORDS = {"se", "set", "setl", "setlo", "setloc", "setloca", "setlocal",
              "setg", "setgl", "setglo", "setglob", "setgloba", "setglobal"}
_LET_WORDS = {"let"}

# 'write' and 'writeany' (abbrev 'wa'), with optional no/inv prefixes.
_WRITE_OPTION = re.compile(r"\b(?:no|inv)?(?:write|writeany|wa)\b")
_WRITE_OPTION_VAR = re.compile(r"&(?:l:|g:)?(?:write|writeany|wa)\b")

# A leading range/address, then the command word.
_EX_HEAD = re.compile(r"^[\s%$.,;+\-0-9']*([a-zA-Z]+|[<>]+|[&~=#*@!])(!?)\s*(.*)$",
                      re.DOTALL)

# :global / :vglobal run another command per matching line -- screen it too.
_GLOBAL_WORDS = {"g", "gl", "glo", "glob", "globa", "global",
                 "v", "vg", "vgl", "vglo", "vglob", "vgloba", "vglobal"}

_CR_TOKENS = ("<CR>", "<cr>", "<Enter>", "<enter>", "<Return>", "<return>")


def _screen_ex_segment(segment: str) -> str | None:
    """Screen one bar-separated ex segment. Returns a reason or None."""
    segment = segment.strip()
    if not segment:
        return None

    m = _EX_HEAD.match(segment)
    if not m:
        return None  # unparseable: let vim reject it, don't guess
    word, _bang, rest = m.group(1), m.group(2), m.group(3)

    if word == "!":
        return "shell escape (:!)"
    if word.isalpha() and word.lower() in _DANGEROUS_WORDS:
        return f"forbidden ex command {word!r}"

    if word.lower() in _SET_WORDS and _WRITE_OPTION.search(rest):
        return "re-enables writing ('set write' defeats the sandbox guard)"
    if word.lower() in _LET_WORDS and _WRITE_OPTION_VAR.search(rest):
        return "re-enables writing (assignment to &write)"

    if word.lower() in _GLOBAL_WORDS and rest:
        # :g/pat/cmd -- screen the trailing command after the pattern
        delim, i, n = rest[0], 1, len(rest)
        while i < n:
            if rest[i] == "\\":
                i += 2
                continue
            if rest[i] == delim:
                break
            i += 1
        sub = rest[i + 1:] if i < n else ""
        if sub.strip():
            return _screen_ex_segment(sub)
    return None


def screen_ex_command(command: str) -> str | None:
    """Screen a full ex command, including every bar-separated segment.

    Bars matter: vim runs 'set write | w! /path' as TWO commands, even
    inside :execute, which is how the guard gets defeated.
    """
    cmd = command.strip()
    if cmd.startswith(":"):
        cmd = cmd[1:]
    for segment in re.split(r"(?<!\\)\|", cmd):
        why = _screen_ex_segment(segment)
        if why:
            return why
    return None


def ex_fragments_from_keys(command: str) -> list[str]:
    """Every candidate ex fragment inside a keys-mode command.

    A keys command may enter command-line mode any number of times, at any
    position -- 'dd:set write<CR>:w! /path<CR>' has two. Splitting only on a
    leading ':' misses all but the first, which is the gap this closes.
    """
    normalized = command
    for tok in _CR_TOKENS:
        normalized = normalized.replace(tok, "\r")
    fragments = []
    for piece in normalized.split("\r"):
        idx = piece.find(":")
        if idx != -1:
            fragments.append(piece[idx + 1:])
    return fragments


# Normal-mode constructs that can write or replay arbitrary commands.
_KEYS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Z[ZQ]"), "save/quit (ZZ/ZQ)"),
    (re.compile(r"@"), "macro playback (@)"),
    (re.compile(r"^[0-9]*q[a-zA-Z0-9\"]"), "macro recording (q{reg})"),
    (re.compile(r"^[0-9]*\""), "named register prefix"),
    (re.compile(r"^g?Q"), "ex mode (Q/gQ)"),
]


def screen_keys_command(command: str) -> str | None:
    """Screen a keys-mode command for damage-capable constructs."""
    for pattern, why in _KEYS_PATTERNS:
        if pattern.search(command):
            return why
    for fragment in ex_fragments_from_keys(command):
        why = screen_ex_command(fragment)
        if why:
            return why
    return None


def screen_dangerous(command: str, mode: str) -> str | None:
    """Return a reason the command is unsafe to execute, or None.

    Called by run_vim() before vim is spawned.
    """
    if mode == "ex":
        return screen_ex_command(command)
    if mode == "keys":
        return screen_keys_command(command)
    return None
