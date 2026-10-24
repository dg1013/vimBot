"""Static screens for damage-capable vim commands.

This is the *policy* half of the trust boundary; src/executor.py is the
mechanism half. run_vim() calls screen_dangerous() before spawning vim, so
every consumer of the executor -- the GRPO (Group Relative Policy Optimization) reward loop, the MCP (Model Context Protocol) server, the
neovim plugin -- inherits the same protection that the data pipeline has.

WHAT THIS IS: defense in depth. It raises the bar substantially against
commands that write files, read other files, shell out, or disable the
executor's own 'nowrite' guard.

WHAT THIS IS NOT: a boundary. It is a denylist over a language with
abbreviations, dynamic evaluation, and modal parsing, so a determined
adversary who controls the command string will eventually get past it. The
real boundary is OS-level -- run generation and RL (reinforcement learning) workers in a container as
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
    # found 2026-08-18: each of these writes a file and none needs :w.
    # :redir/:profile/:hardcopy take a path; :helptags writes a tags file
    # into the directory it is given; the spell family writes .add/.spl files.
    "redi", "redir",
    "prof", "profi", "profil", "profile", "profd", "profde", "profdel",
    "ha", "har", "hard", "hardc", "hardco", "hardcop", "hardcopy",
    "helpt", "helpta", "helptag", "helptags",
    "mksp", "mkspe", "mkspel", "mkspell",
    "spellg", "spellgo", "spellgoo", "spellgood",
    "spellw", "spellwr", "spellwro", "spellwron", "spellwrong",
    "spellr", "spellra", "spellrar", "spellrare",
    "spellu", "spellun", "spellund", "spellundo",
    "spelld", "spelldu", "spelldum", "spelldump",
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

_EVALUATES_EXPRESSION = {
    # These evaluate an arbitrary vimscript expression, which reaches
    # writefile() without touching any word in _SCRIPTING. Found 2026-08-18.
    "ec", "ech", "echo", "echon", "echom", "echoms", "echomsg",
    "echoe", "echoer", "echoerr", "echoh", "echohl", "eval",
    # vimscript control flow: out of taxonomy scope anyway, and each body
    # is a fresh place to hide an expression.
    "if", "el", "els", "else", "elsei", "elseif", "en", "end", "endi", "endif",
    "wh", "whi", "whil", "while", "endw", "endwh", "endwhi", "endwhil", "endwhile",
    "for", "endfo", "endfor", "try", "cat", "catc", "catch",
    "fina", "final", "finall", "finally", "endt", "endtr", "endtry",
    "retu", "retur", "return", "throw",
}

_DANGEROUS_WORDS = (_WRITE_FILE | _READ_OTHER_FILE | _SHELL | _SCRIPTING
                    | _EVALUATES_EXPRESSION)

# ------------------------------------------------------- expression policy
#
# Expressions are not banned outright: `\=submatch(0)+3` is how the increment
# and case categories express themselves, and banning it would cost real
# coverage. Instead the functions an expression may call are ALLOWLISTED.
#
# The allowlist deliberately contains no primitive that can invoke a function
# by name -- no call(), function(), eval(), execute(), map(), filter(), sort().
# With any one of those present, `call(join(['write','file'],''), ...)` walks
# straight through a denylist of function names.
_PURE_FUNCTIONS = {
    "submatch", "line", "col", "virtcol", "indent", "getline",
    "printf", "string", "str2nr", "str2float", "nr2char", "char2nr",
    "strpart", "strcharpart", "strlen", "strchars", "strdisplaywidth",
    "toupper", "tolower", "tr", "trim", "repeat", "reverse",
    "split", "join", "range", "len", "empty", "type",
    "match", "matchstr", "matchend", "substitute",
    "abs", "max", "min", "round", "floor", "ceil", "float2nr",
}

_CALLED_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_#:.]*)\s*\(")

# substitute, and the two commands that repeat one
_SUB_WORDS = {"s", "su", "sub", "subs", "subst", "substi", "substit",
              "substitu", "substitut", "substitute", "&", "~"}
_PUT_WORDS = {"pu", "put"}


def _screen_expression(expr: str) -> str | None:
    """Allow an expression only if every function it calls is pure."""
    for name in _CALLED_NAME.findall(expr):
        if name.lower() not in _PURE_FUNCTIONS:
            return f"expression calls {name!r}, which is not on the pure-function allowlist"
    return None


def _substitute_replacement(rest: str) -> str | None:
    r"""Return the replacement half of :s{delim}pat{delim}repl{delim}flags.

    Needed because `\=` means two different things in vim: at the start of a
    REPLACEMENT it introduces an expression, but inside a PATTERN it is the
    'optional' quantifier -- which the v3 deep-regex commands use heavily.
    Screening every `\=` alike would reject most of the existing dataset.
    """
    if not rest:
        return None
    delim = rest[0]
    if delim.isalnum() or delim in " \\|\"":
        return None
    fields, cur, i, n = [], [], 1, len(rest)
    while i < n and len(fields) < 2:
        ch = rest[i]
        if ch == "\\" and i + 1 < n:
            cur.append(rest[i:i + 2])
            i += 2
            continue
        if ch == delim:
            fields.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(ch)
        i += 1
    if len(fields) < 2:
        fields.append("".join(cur))
    return fields[1] if len(fields) > 1 else None

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
#
# The address part must consume a MARK ('a, '<, '>, '') as a two-character
# unit, not as a bare quote. Demonstrated 2026-08-26: a plain character class
# containing "'" eats only the quote, and the mark's letter is then taken as
# the command word -- so ":'a,'bw! /path" screened as command 'a' and
# ":'<,'>w! /path" as command '<' (because [<>]+ is a command-word branch).
# Both sailed past the screen; ":1mark a|2mark b|'a,'bd" was verified to
# execute unscreened. '<,'> is not exotic: it is exactly what vim inserts the
# moment ':' is pressed in visual mode, so this is the form a visual-mode
# dataset row naturally takes.
#
# "'." is ordered before the class so a mark is always consumed whole. The
# quote is deliberately NOT in the class any more -- leaving it in both places
# would let "'" match alone again and reopen the hole.
_EX_HEAD = re.compile(
    r"^(?:'.|[\s%$.,;+\-0-9])*([a-zA-Z]+|[<>]+|[&~=#*@!])(!?)\s*(.*)$",
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

    low = word.lower()
    if low in _PUT_WORDS and rest.lstrip().startswith("="):
        why = _screen_expression(rest.lstrip()[1:])
        if why:
            return why
    if low in _SUB_WORDS:
        repl = _substitute_replacement(rest)
        if repl is not None and repl.lstrip().startswith("\\="):
            why = _screen_expression(repl.lstrip()[2:])
            if why:
                return why

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
# NOTE (2026-08-18): 'named register prefix' and 'macro recording (q{reg})'
# used to live here. Both are TAXONOMY rules, not safety ones -- a register
# stores text, it does not execute it, and every path that executes one runs
# through '@', which is still screened below. They moved to
# scripts/generate_data.py; see data/command_coverage.md sec 9.
# Screened wherever they appear, because they are meaningful INSIDE insert
# mode -- projecting them onto normal-mode positions would lose them.
_KEYS_PATTERNS_ANY: list[tuple[re.Pattern, str]] = [
    # <C-r>= is the insert-mode expression register: it evaluates arbitrary
    # vimscript from inside insert mode, reaching writefile() with no ex
    # command involved at all. Both the angle-bracket notation and the raw
    # 0x12 byte are matched, plus the <C-r><C-r>/<C-o>/<C-p> variants.
    (re.compile(r"(?i)<c-r>\s*(?:<c-[rop]>\s*)?="), "expression register (<C-r>=)"),
    (re.compile(r"\x12\s*[\x12\x0f\x10]?\s*="), "expression register (<C-r>=)"),
]

# Screened only where vim would read them as NORMAL or VISUAL mode keys.
# Matching these anywhere would refuse ordinary rows -- 12 corpus cases type a
# literal '!' inside insert mode ("i!", "a!", "$i !important<Esc>") and 4 type
# a literal 'Q' ("ci[the FAQ<Esc>", "oQ2<CR>Q3<Esc>"). _normal_mode_view()
# blanks insert-mode and command-line text so those cannot match.
_KEYS_PATTERNS_NORMAL: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Z[ZQ]"), "save/quit (ZZ/ZQ)"),
    (re.compile(r"@"), "macro playback (@)"),
    # Un-anchored as of 2026-08-26. It was r"^g?Q", so only a command STARTING
    # with Q/gQ was screened and "VjgQ..." or "ddgQ..." reached Ex mode
    # untouched. Ex mode takes ex commands with no ':' typed, so nothing
    # downstream screened them either.
    (re.compile(r"g?Q"), "ex mode (Q/gQ)"),
    # '!' in normal or visual mode is the filter operator: it opens the
    # command line as ":{range}!" and pipes the lines through a shell command.
    # ex_fragments_from_keys() cannot see it because no ':' is ever typed, so
    # before 2026-08-26 "Vj!echo ...", "!!sh<CR>" and "<C-v>j!..." all passed
    # the screen. -Z refused them, but the screen is what the GRPO reward
    # loop, the MCP server and the plugin inherit, so it screens them now.
    (re.compile(r"!"), "shell filter (!{motion})"),
]

_KEY_TOKEN = re.compile(r"<[^<>\s]{1,14}>")

# Every way back to normal mode from insert. Getting this set wrong is the
# expensive direction: a missed exit leaves the walker stuck in insert, so it
# blanks keys vim really does execute and the screen stops looking at them.
# Re-audited 2026-08-26 after the first version had only <Esc> and <C-c>:
# "ihi<C-[>@a" and "ihi<C-o>@a" both slipped a macro past a screen that had
# caught them before the mode view existed.
#   <C-[> and 0x1b are literally the same key as <Esc>; <C-c> is 0x03;
#   <C-\><C-n> forces normal mode from anywhere, so <C-\> (0x1c) is treated
#   as an exit on its own.
_ESC_TOKENS = {"<esc>", "<c-[>", "<c-c>", "<c-\\>",
               "\x1b", "\x03", "\x1c"}

# <C-o> runs exactly ONE normal-mode command and then returns to insert. The
# walker does not try to find where that one command ends -- it returns to
# normal and STAYS there, which over-screens the tail of such a command rather
# than under-screening it. Verified against the corpus: the five <C-o> rows
# ("i<C-o>l123", "a123<C-o>zz", "yejA<C-o>p") carry nothing screenable after
# the <C-o>, so fail-closed costs nothing here.
_ONESHOT_NORMAL_TOKENS = {"<c-o>", "\x0f"}
_CR_TOKEN_SET = {t.lower() for t in _CR_TOKENS}
# Normal-mode keys that put vim into insert mode outright.
_INSERT_KEYS = set("iIaAoOsSR")
# Keys that swallow exactly one following key (so its literal char is not a
# command): r{char}, f/t/F/T{char}, m{mark}, '{mark}, `{mark}, "{reg}, q{reg},
# Z{Z|Q} is handled by its own pattern before this ever runs.
_SWALLOW_NEXT = set("rfFtTm'`\"q")


def _normal_mode_view(command: str) -> str:
    """The command with insert-mode and command-line text blanked out.

    Length and indices are preserved (blanked runs become spaces) so a match
    position still refers to the original string. The walk is a deliberate
    approximation of vim's mode machine, and it fails CLOSED: anything it
    cannot classify stays visible as normal mode and is therefore screened.
    """
    out = list(command)
    i, n = 0, len(command)
    mode = "normal"

    def token_at(idx):
        if idx >= n:                      # a swallow-key can be the last char
            return "", 0
        m = _KEY_TOKEN.match(command, idx)
        if m:
            return m.group(0), len(m.group(0))
        return command[idx], 1

    def blank(start, end):
        for k in range(start, end):
            if not out[k].isspace():
                out[k] = " "

    while i < n:
        tok, size = token_at(i)
        low = tok.lower()
        if mode == "normal":
            if size > 1:
                # A <...> token is ONE key. Its letters are not commands --
                # the Q inside <C-Q> is not the Ex-mode Q (cand-vsv-0062).
                blank(i, i + size)
                i += size
                continue
            if tok in _INSERT_KEYS:
                mode = "insert"
            elif tok == "c":
                mode = "pending_c"
            elif tok in ":/?":
                mode = "cmdline"
            elif tok in _SWALLOW_NEXT:
                _, nxt = token_at(i + size)
                blank(i + size, i + size + nxt)
                i += size + nxt
                continue
            i += size
            continue
        if mode == "pending_c":
            # consume a motion, then insert. Counts and text objects first.
            if tok.isdigit():
                i += size
                continue
            if tok in "ia":                       # text object: iw, a", ...
                _, nxt = token_at(i + size)
                blank(i, i + size + nxt)
                i += size + nxt
                mode = "insert"
                continue
            if tok in "fFtT":                     # cf, ct, ...
                _, nxt = token_at(i + size)
                blank(i, i + size + nxt)
                i += size + nxt
                mode = "insert"
                continue
            if tok in "/?":                       # c/pattern<CR>
                mode = "cmdline"
                i += size
                continue
            blank(i, i + size)                    # single-char motion: cw, c$
            i += size
            mode = "insert"
            continue
        if mode == "insert":
            if low in _ESC_TOKENS or low in _ONESHOT_NORMAL_TOKENS:
                mode = "normal"
            else:
                blank(i, i + size)
            i += size
            continue
        # cmdline
        if low in _CR_TOKEN_SET or tok == "\r" or tok == "\n":
            mode = "normal"
        elif low in _ESC_TOKENS:
            mode = "normal"
        else:
            blank(i, i + size)
        i += size
    return "".join(out)


def screen_keys_command(command: str) -> str | None:
    """Screen a keys-mode command for damage-capable constructs."""
    for pattern, why in _KEYS_PATTERNS_ANY:
        if pattern.search(command):
            return why
    normal = _normal_mode_view(command)
    for pattern, why in _KEYS_PATTERNS_NORMAL:
        if pattern.search(normal):
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
