"""Heuristic structure of vim commands, for the data workbook's breakdown columns.

`parse_ex` reads one ex command line (range, command word, bang, the
substitute's pattern / replacement / flags, :g's inner command ...);
`parse_keys` walks a keys-mode command through a small model of vim's
normal / operator-pending / visual / insert / command-line modes and
collects what it saw (operators, motions, text objects, counts, visual
kinds, inserts, ex fragments, marks, registers, macros ...).

Both are approximations of vim's grammar written for aggregation, not for
execution: the executor and the sandbox screen (`src/scope.py`) are the
ground truth, and a row these functions misread is a wrong pivot cell, not
a wrong reward. Every column they produce is labelled heuristic in the
workbook's README tab.
"""

import re

# ------------------------------------------------------------------ ex mode

_EX_WORDS = [
    ("substitute", "s su sub subs subst substi substit substitu substitut substitute"),
    ("global", "g gl glo glob globa global"),
    ("vglobal", "v vg vgl vglo vglob vgloba vglobal"),
    ("delete", "d de del dele delet delete"),
    ("move", "m mo mov move"),
    ("copy", "t co cop copy"),
    ("sort", "sor sort"),
    ("join", "j jo joi join"),
    ("yank", "y ya yan yank"),
    ("put", "pu put"),
    ("normal", "norm norma normal"),
    ("retab", "ret reta retab"),
    ("left", "le lef left"), ("right", "ri rig righ right"),
    ("center", "ce cen cent cente center"),
    ("mark", "k ma mar mark"),
    ("print", "p pr pri prin print nu num numb numbe number # l li lis list"),
    ("undo", "u un und undo"), ("redo", "red redo"),
    ("append", "a ap app appe appen append"), ("insert", "i in ins inse inser insert"),
    ("change", "c ch cha chan chang change"),
    ("execute", "exe exec execu execut execute"), ("call", "cal call"), ("let", "let"),
    ("set", "se set setl setlo setloc setloca setlocal"),
    ("read", "r re rea read"), ("write", "w wr wri writ write"), ("quit", "q qu qui quit"),
    ("registers", "reg regi regis regist registe register registers"),
    ("smagic", "sm smagic"), ("snomagic", "sno snomagic"),
    ("startinsert", "star start starti startin startins startinse startinser startinsert"),
    ("echo", "ec ech echo"), ("if", "if"), ("for", "for"), ("while", "wh while"),
    ("lsubstitute", "&"), ("tilde-substitute", "~"),
]
EX_CANON = {}
for _canon, _words in _EX_WORDS:
    for _w in _words.split():
        EX_CANON.setdefault(_w, _canon)

_ADDR = (r"(?:\d+|[.$%]|'[a-zA-Z<>\[\]]|/(?:[^/\\]|\\.)*/|\?(?:[^?\\]|\\.)*\?|\\[/?&])")
_RANGE = re.compile(r"^(%s?(?:\s*[+-]\s*\d*)*(?:\s*[,;]\s*%s?(?:\s*[+-]\s*\d*)*)*)" % (_ADDR, _ADDR))
_WORD = re.compile(r"^\s*([a-zA-Z]+|&&|&|~|[<>]+|!|#|=)")

EX_KEYS = ("ex_range", "ex_range_kind", "ex_command", "ex_bang", "ex_bar_chain",
           "ex_inner_command", "ex_sub_pattern", "ex_sub_replacement", "ex_sub_flags",
           "ex_sub_delim", "ex_sub_count", "ex_sort_flags", "ex_target_address",
           "ex_normal_keys", "ex_shift_depth")


def _range_kind(rng):
    if not rng:
        return "none"
    if rng == "%":
        return "whole-file"
    if "'<" in rng or "'>" in rng:
        return "visual-marks"
    if "'" in rng:
        return "marks"
    if "/" in rng or "?" in rng or "\\&" in rng:
        return "pattern"
    if "," in rng or ";" in rng:
        return "relative-range" if ("+" in rng or "-" in rng) else "line-range"
    if "+" in rng or "-" in rng:
        return "relative-line"
    return "single-line"


def split_delim(text, delim):
    """`pat<delim>rep<delim>flags` -> parts, honouring backslash escapes."""
    parts, cur, i = [], [], 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            cur.append(text[i:i + 2])
            i += 2
            continue
        if c == "[":
            # a collection: vim skips it when looking for the delimiter, so
            # `[!-/:-@]` may hold an unescaped `/`
            j = i + 1
            if j < len(text) and text[j] == "^":
                j += 1
            if j < len(text) and text[j] == "]":
                j += 1
            while j < len(text) and text[j] != "]":
                j += 2 if text[j] == "\\" else 1
            if j < len(text):
                cur.append(text[i:j + 1])
                i = j + 1
                continue
        if c == delim:
            parts.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _bar_chain(text):
    """Number of `|`-chained commands, ignoring `\\|` (alternation) and bars
    inside a substitute/global pattern-replacement body."""
    n, i, depth_word = 1, 0, ""
    m = _RANGE.match(text)
    rest = text[m.end():] if m else text
    w = _WORD.match(rest)
    word = w.group(1) if w else ""
    canon = EX_CANON.get(word)
    body_start = (m.end() if m else 0) + (w.end() if w else 0)
    if canon in ("substitute", "global", "vglobal") and body_start < len(text):
        # skip the delimited body: pattern (+ replacement for :s)
        d = text[body_start] if text[body_start] not in " \t" else None
        if d and not d.isalnum():
            parts = split_delim(text[body_start + 1:], d)
            take = 2 if canon == "substitute" else 1
            skipped = sum(len(p) + 1 for p in parts[:take])
            tail = text[body_start + 1 + skipped:]
            if canon != "substitute":
                return 1 + _bar_chain(tail) - 1 if tail.strip() else 1
            return 1 + len(re.findall(r"(?<!\\)\|", tail))
    return 1 + len(re.findall(r"(?<!\\)\|", text))


def parse_ex(cmd):
    """Structure of the first ex command of `cmd` (heuristic)."""
    out = {k: None for k in EX_KEYS}
    out["ex_bang"] = False
    out["ex_bar_chain"] = None
    if cmd is None:
        return out
    text = cmd.strip()
    while text.startswith(":"):
        text = text[1:].lstrip()
    try:
        out["ex_bar_chain"] = _bar_chain(text)
    except Exception:
        out["ex_bar_chain"] = None
    m = _RANGE.match(text)
    rng = (m.group(1) or "").strip() if m else ""
    rest = text[m.end():] if m else text
    out["ex_range"] = rng or None
    out["ex_range_kind"] = _range_kind(rng)
    w = _WORD.match(rest)
    if not w:
        out["ex_command"] = "(range only)" if rng else "(unparsed)"
        return out
    word = w.group(1)
    after = rest[w.end():]
    canon = EX_CANON.get(word)
    if word.startswith(">"):
        canon, out["ex_shift_depth"] = "shift-right", len(word)
    elif word.startswith("<"):
        canon, out["ex_shift_depth"] = "shift-left", len(word)
    elif word == "!":
        canon = "filter"
    elif word == "=":
        canon = "line-number"
    if canon is None:
        canon = "?" + word
    if after.startswith("!"):
        out["ex_bang"] = True
        after = after[1:]
        if canon == "global":
            canon = "vglobal"
    out["ex_command"] = canon
    if canon == "substitute" and after:
        d = after[0]
        if not d.isalnum() and d not in " \t\\\"|":
            parts = split_delim(after[1:], d)
            out["ex_sub_delim"] = d
            out["ex_sub_pattern"] = parts[0] if parts else None
            out["ex_sub_replacement"] = parts[1] if len(parts) > 1 else None
            tail = (parts[2] if len(parts) > 2 else "").split("|")[0].strip()
            fm = re.match(r"([&cegiInp#lr]*)\s*(\d+)?", tail)
            if fm:
                out["ex_sub_flags"] = fm.group(1) or None
                out["ex_sub_count"] = int(fm.group(2)) if fm.group(2) else None
        else:
            out["ex_sub_flags"] = after.strip() or None      # :s g  (repeat last)
    elif canon in ("global", "vglobal") and after:
        d = after[0]
        parts = split_delim(after[1:], d)
        out["ex_sub_pattern"] = parts[0] if parts else None
        inner = parts[1].strip() if len(parts) > 1 else ""
        if inner:
            sub = parse_ex(inner)
            out["ex_inner_command"] = sub["ex_command"]
            for k in ("ex_sub_replacement", "ex_sub_flags", "ex_normal_keys", "ex_target_address", "ex_sort_flags"):
                if sub.get(k) is not None:
                    out[k] = sub[k]
        else:
            out["ex_inner_command"] = "print"
    elif canon == "sort":
        out["ex_sort_flags"] = after.strip() or None
    elif canon in ("move", "copy"):
        out["ex_target_address"] = after.strip().split("|")[0].strip() or None
    elif canon == "normal":
        out["ex_normal_keys"] = after.strip() or None
    return out


# ---------------------------------------------------------------- keys mode

_KEY_TOKEN = re.compile(r"<[^<>\s]{1,14}>")
_OPERATORS = {"d", "c", "y", "<", ">", "=", "!"}
_G_OPERATORS = {"g~", "gu", "gU", "gq", "gw", "g?", "g@"}
_INSERT = {"i": "insert-before", "I": "insert-line-start", "a": "append", "A": "append-line-end",
           "o": "open-below", "O": "open-above", "s": "substitute-char", "S": "substitute-line",
           "C": "change-to-eol", "R": "replace-mode", "gi": "insert-last", "gI": "insert-col1"}
_SIMPLE_MOTIONS = set("wbeWBE$0^hjklGHLM{}()%|_;,nN*#+-")
_MOTION_TOKENS = {"<cr>": "j", "<enter>": "j", "<return>": "j", "<space>": "l", "<bs>": "h",
                  "<left>": "h", "<right>": "l", "<up>": "k", "<down>": "j", "<tab>": "tab"}
_CHAR_MOTIONS = set("fFtT")
_OBJECT_CHARS = set("wWsp()[]{}<>\"'`bBt")
_ESC = {"<esc>", "<c-[>", "<c-c>"}
_CR = {"<cr>", "<enter>", "<return>"}
_LINEWISE_REPEAT = {"d": "dd", "y": "yy", "c": "cc", ">": ">>", "<": "<<", "=": "==", "!": "!!",
                    "g~": "g~~", "gu": "guu", "gU": "gUU", "gq": "gqq", "gw": "gww"}

KEYS_FLAG_NAMES = ("keys_has_count", "keys_uses_ex", "keys_uses_search", "keys_enters_insert",
                   "keys_has_esc", "keys_dot_repeat", "keys_macro_record", "keys_macro_play",
                   "keys_named_register", "keys_marks", "keys_replace_char", "keys_replace_mode",
                   "keys_increment", "keys_indent", "keys_case_op", "keys_join", "keys_undo",
                   "keys_paste", "keys_delete_char", "keys_search_repeat", "keys_ctrl_keys",
                   "keys_linewise_op", "keys_unclassified")
KEYS_VALUE_NAMES = ("keys_tokens", "keys_specials", "keys_operators", "keys_motions",
                    "keys_text_objects", "keys_counts", "keys_max_count", "keys_visual",
                    "keys_inserts", "keys_inserted_chars", "keys_ex_fragments", "keys_ex_command",
                    "keys_search_patterns", "keys_family", "keys_first_action", "keys_n_actions")


def _tokens(cmd):
    toks, i = [], 0
    while i < len(cmd):
        m = _KEY_TOKEN.match(cmd, i)
        if m:
            toks.append(m.group(0))
            i = m.end()
        else:
            toks.append(cmd[i])
            i += 1
    return toks


def parse_keys(cmd):
    """Walk a keys-mode command through a model of vim's modes (heuristic)."""
    out = {k: None for k in KEYS_VALUE_NAMES}
    out.update({k: False for k in KEYS_FLAG_NAMES})
    if cmd is None:
        return out
    toks = _tokens(cmd)
    out["keys_tokens"] = len(toks)
    specials = [t for t in toks if len(t) > 1]
    out["keys_specials"] = " ".join(specials) or None
    out["keys_ctrl_keys"] = any(t.lower().startswith("<c-") for t in specials)
    out["keys_has_esc"] = any(t.lower() in _ESC for t in specials)

    ops, motions, objects, counts, inserts, ins_chars = [], [], [], [], [], 0
    ex_frags, search_pats, actions = [], [], []
    visual = set()
    mode = "normal"          # normal | pending | visual | insert | cmdline
    pending = None           # operator awaiting its motion
    count = ""
    cmdline = []
    cmdkind = None
    cmd_from_visual = False
    recording = False
    i, n = 0, len(toks)
    unclassified = 0

    def push_count():
        nonlocal count
        if count:
            counts.append(int(count))
            count = ""

    def finish_op(motion):
        nonlocal pending, mode
        if motion:
            motions.append(motion)
        ops.append(pending + ("(" + motion + ")" if motion else ""))
        actions.append(pending)
        if pending in ("c", "cc"):
            mode = "insert"
            inserts.append("c")
        else:
            mode = "normal"
        pending = None

    while i < n:
        t = toks[i]
        low = t.lower()
        i += 1
        # ---------------------------------------------------- insert mode
        if mode == "insert":
            if low in _ESC:
                mode = "normal"
            elif low == "<c-o>":
                pass                                  # one-shot normal, ignored
            else:
                if low in ("<c-a>", "<c-x>"):
                    pass
                ins_chars += 1
            continue
        # ------------------------------------------------- command line
        if mode == "cmdline":
            if low in _CR:
                text = "".join(cmdline)
                if cmdkind == ":":
                    ex_frags.append(text)
                else:
                    search_pats.append(text)
                    motions.append(cmdkind + "pattern")
                cmdline = []
                if pending:
                    motions.pop()                       # finish_op records it
                    finish_op(cmdkind + "pattern")
                else:
                    mode = "visual" if (cmd_from_visual and cmdkind != ":") else "normal"
                cmd_from_visual = False
            elif low in _ESC:
                cmdline, mode = [], ("visual" if cmd_from_visual else "normal")
                cmd_from_visual = False
            else:
                cmdline.append(t)
            continue
        # ------------------------------------------ operator pending
        if mode == "pending":
            if t.isdigit() and (t != "0" or count):
                count += t
                continue
            push_count()
            if t in "ia" and i < n and toks[i] in _OBJECT_CHARS:
                objects.append(t + toks[i])
                i += 1
                finish_op("obj:" + objects[-1])
                continue
            if t in _CHAR_MOTIONS and i < n:
                i += 1
                finish_op(t + "{char}")
                continue
            if t == "g" and i < n and toks[i] in "gejkE_nN;,":
                i += 1
                finish_op("g" + toks[i - 1])
                continue
            if t in "[]" and i < n and toks[i] in "(){}[]":
                i += 1
                finish_op(t + toks[i - 1])
                continue
            if t in ("'", "`") and i < n:
                i += 1
                finish_op(t + "mark")
                continue
            if t in ("/", "?"):
                mode, cmdkind = "cmdline", t
                continue
            if low in _MOTION_TOKENS:
                finish_op(_MOTION_TOKENS[low])
                continue
            if t in _SIMPLE_MOTIONS:
                finish_op(t)
                continue
            if t == pending or (pending in _LINEWISE_REPEAT and t == pending[-1]):
                out["keys_linewise_op"] = True
                name = _LINEWISE_REPEAT.get(pending, pending + pending)
                finish_op(None)
                ops[-1] = name
                continue
            # anything else cancels the operator
            unclassified += 1
            pending, mode = None, "normal"
            i -= 1
            continue
        # ------------------------------------------------ visual mode
        if mode == "visual":
            if low in _ESC:
                mode = "normal"
                continue
            if t.isdigit() and (t != "0" or count):
                count += t
                continue
            push_count()
            if t in ("v", "V") or low in ("<c-v>", "<c-q>"):
                kind = {"v": "charwise", "V": "linewise"}.get(t, "blockwise")
                if kind in visual:
                    mode = "normal"                     # same key toggles off
                else:
                    visual.add(kind)
                continue
            if t in "ia" and i < n and toks[i] in _OBJECT_CHARS:
                objects.append(t + toks[i])
                i += 1
                continue
            if t in _CHAR_MOTIONS and i < n:
                motions.append(t + "{char}")
                i += 1
                continue
            if t in "[]" and i < n and toks[i] in "pP":
                i += 1
                ops.append("visual:" + t + toks[i - 1])
                actions.append(t + toks[i - 1])
                out["keys_paste"] = True
                mode = "normal"
                continue
            if t == "g" and i < n and (toks[i] in "gejkE_nNvJupPU~?q" or toks[i].lower() in ("<c-a>", "<c-x>")):
                nxt = toks[i]
                i += 1
                if nxt in "gejkE_nN":
                    motions.append("g" + nxt)
                elif nxt.lower() in ("<c-a>", "<c-x>"):
                    ops.append("visual:g" + nxt)
                    actions.append("g" + nxt)
                    out["keys_increment"] = True
                    mode = "normal"
                else:
                    ops.append("visual:g" + nxt)
                    actions.append("g" + nxt)
                    mode = "normal"
                continue
            if t in _SIMPLE_MOTIONS or low in _MOTION_TOKENS:
                motions.append(_MOTION_TOKENS.get(low, t))
                continue
            if t in ("/", "?"):
                mode, cmdkind, cmdline, cmd_from_visual = "cmdline", t, [], True
                continue
            if t == ":":
                mode, cmdkind, cmdline, cmd_from_visual = "cmdline", ":", [], True
                continue
            if t in "dxXDyYcCsSrRJ<>=~uUpPoOIAJ" or low in ("<c-a>", "<c-x>"):
                name = {"<c-a>": "<C-a>", "<c-x>": "<C-x>"}.get(low, t)
                ops.append("visual:" + name)
                actions.append(name)
                if t == "r" and i < n:
                    i += 1
                    out["keys_replace_char"] = True
                if t in "cCsSIA":
                    mode = "insert"
                    inserts.append("visual-" + t)
                elif t == "o" or t == "O":
                    mode = "visual"                     # o/O move the cursor end
                    ops.pop(); actions.pop()
                else:
                    mode = "normal"
                if t in "<>=":
                    out["keys_indent"] = True
                if t in "~uU":
                    out["keys_case_op"] = True
                if t == "J":
                    out["keys_join"] = True
                if t in "pP":
                    out["keys_paste"] = True
                if low in ("<c-a>", "<c-x>"):
                    out["keys_increment"] = True
                continue
            unclassified += 1
            continue
        # ------------------------------------------------ normal mode
        if t.isdigit() and (t != "0" or count):
            count += t
            continue
        push_count()
        if t == '"' and i < n:
            out["keys_named_register"] = True
            i += 1
            continue
        if t in _OPERATORS:
            pending, mode = t, "pending"
            continue
        if t == "g" and i < n:
            nxt = toks[i]
            if "g" + nxt in _G_OPERATORS:
                i += 1
                pending, mode = "g" + nxt, "pending"
                if nxt in "~uU":
                    out["keys_case_op"] = True
                continue
            if nxt in ("i", "I"):
                i += 1
                mode = "insert"
                inserts.append("g" + nxt)
                actions.append("g" + nxt)
                continue
            if nxt == "J":
                i += 1
                out["keys_join"] = True
                actions.append("gJ")
                continue
            if nxt == "v":
                i += 1
                mode = "visual"
                visual.add("reselect")
                continue
            if nxt in "pP":
                i += 1
                out["keys_paste"] = True
                actions.append("g" + nxt)
                continue
            if nxt in "gejkE_;,":
                i += 1
                motions.append("g" + nxt)
                continue
            if nxt in "nN":
                i += 1
                motions.append("g" + nxt)
                mode = "visual"
                visual.add("charwise")
                continue
            unclassified += 1
            i += 1
            continue
        if t in "[]" and i < n and toks[i] in "pP":
            i += 1
            out["keys_paste"] = True
            actions.append(t + toks[i - 1])
            continue
        if t in "[]" and i < n and toks[i] in "(){}[]":
            i += 1
            motions.append(t + toks[i - 1])
            continue
        if low in ("<c-g>", "<c-l>"):
            continue
        if t in ("v", "V") or low in ("<c-v>", "<c-q>"):
            visual.add({"v": "charwise", "V": "linewise"}.get(t, "blockwise"))
            mode = "visual"
            continue
        if t in _INSERT:
            mode = "insert"
            inserts.append(t)
            actions.append(t)
            if t == "R":
                out["keys_replace_mode"] = True
            continue
        if t == ":":
            mode, cmdkind, cmdline = "cmdline", ":", []
            continue
        if t in ("/", "?"):
            mode, cmdkind, cmdline = "cmdline", t, []
            continue
        if t == "r" and i < n:
            i += 1
            out["keys_replace_char"] = True
            actions.append("r")
            continue
        if t == "m" and i < n:
            i += 1
            out["keys_marks"] = True
            actions.append("m")
            continue
        if t in ("'", "`") and i < n:
            i += 1
            out["keys_marks"] = True
            motions.append(t + "mark")
            continue
        if t == "q":
            if recording:
                recording = False
            elif i < n:
                i += 1
                recording = True
                out["keys_macro_record"] = True
                actions.append("q")
            continue
        if t == "@" and i < n:
            i += 1
            out["keys_macro_play"] = True
            actions.append("@")
            continue
        if t in _CHAR_MOTIONS and i < n:
            i += 1
            motions.append(t + "{char}")
            continue
        if t in _SIMPLE_MOTIONS or low in _MOTION_TOKENS:
            mo = _MOTION_TOKENS.get(low, t)
            motions.append(mo)
            if t in "nN":
                out["keys_search_repeat"] = True
            continue
        if t in "xXDYpPJu.~&":
            actions.append(t)
            if t in "xX":
                out["keys_delete_char"] = True
            elif t in "pP":
                out["keys_paste"] = True
            elif t == "J":
                out["keys_join"] = True
            elif t == "u":
                out["keys_undo"] = True
            elif t == ".":
                out["keys_dot_repeat"] = True
            elif t == "~":
                out["keys_case_op"] = True
            elif t == "D":
                ops.append("D")
            elif t == "Y":
                ops.append("Y")
            continue
        if low in ("<c-a>", "<c-x>"):
            out["keys_increment"] = True
            actions.append(t)
            continue
        if low in ("<c-r>",):
            actions.append("redo")
            continue
        if low in _ESC:
            continue
        unclassified += 1

    push_count()
    out["keys_operators"] = " ".join(ops) or None
    out["keys_motions"] = " ".join(motions) or None
    out["keys_text_objects"] = " ".join(objects) or None
    out["keys_counts"] = " ".join(map(str, counts)) or None
    out["keys_max_count"] = max(counts) if counts else None
    out["keys_has_count"] = bool(counts)
    vis = [k for k in ("blockwise", "linewise", "charwise", "reselect") if k in visual]
    out["keys_visual"] = "+".join(vis) if vis else None
    out["keys_inserts"] = " ".join(inserts) or None
    out["keys_enters_insert"] = bool(inserts)
    out["keys_inserted_chars"] = ins_chars if inserts else None
    out["keys_ex_fragments"] = " ‖ ".join(ex_frags) or None
    out["keys_uses_ex"] = bool(ex_frags)
    if ex_frags:
        out["keys_ex_command"] = parse_ex(ex_frags[0])["ex_command"]
    out["keys_search_patterns"] = " ‖ ".join(search_pats) or None
    out["keys_uses_search"] = bool(search_pats)
    out["keys_indent"] = out["keys_indent"] or any(o.startswith(("<", ">", "=")) for o in ops)
    out["keys_case_op"] = out["keys_case_op"] or any(o.startswith(("g~", "gu", "gU")) for o in ops)
    out["keys_unclassified"] = bool(unclassified)
    out["keys_first_action"] = (ops[0] if ops else actions[0] if actions else
                                ("visual" if vis else motions[0] if motions else None))
    out["keys_n_actions"] = len(actions) + len(ops)
    # coarse mechanism family, by priority
    if "blockwise" in visual:
        fam = "visual-block"
    elif visual:
        fam = "visual"
    elif ex_frags and not ops and not inserts:
        fam = "ex-from-keys"
    elif objects:
        fam = "text-object"
    elif out["keys_increment"]:
        fam = "increment"
    elif any("(" in o and not o.startswith(("<", ">", "=", "g~", "gu", "gU")) for o in ops):
        fam = "operator-motion"
    elif out["keys_indent"]:
        fam = "indent"
    elif out["keys_case_op"]:
        fam = "case"
    elif out["keys_join"]:
        fam = "join"
    elif any(o in ("dd", "yy", "cc", "D", "Y") for o in ops) or out["keys_paste"] or out["keys_delete_char"]:
        fam = "line-or-char-op"
    elif out["keys_replace_char"] or out["keys_replace_mode"]:
        fam = "replace"
    elif search_pats:
        fam = "search"
    elif inserts:
        fam = "insert"
    elif out["keys_dot_repeat"]:
        fam = "repeat"
    elif out["keys_macro_record"] or out["keys_macro_play"]:
        fam = "macro"
    elif ex_frags:
        fam = "ex-from-keys"
    elif motions:
        fam = "motion-only"
    else:
        fam = "other"
    out["keys_family"] = fam
    return out


if __name__ == "__main__":
    import json
    import sys
    for c in sys.argv[1:] or ["diw", "ci\"hello<Esc>", "Vj:s/a/b/<CR>", "<C-v>jjIa<Esc>", "3ddp", "qaxq", "ma'ad",
                              "gUiw", ">>j.", "/foo<CR>ciwbar<Esc>", "d/four<CR>", "2yyGp", "5<C-a>", "r<Tab>", "yiwwep"]:
        r = parse_keys(c)
        print(c.ljust(24), r["keys_family"], "|", r["keys_operators"], "|", r["keys_motions"], "|", r["keys_text_objects"],
              "|", r["keys_visual"], "|", r["keys_inserts"], "|", r["keys_ex_fragments"], "| counts", r["keys_counts"])
    for c in ["%s/\\(\\w\\+\\), \\(\\w\\+\\)/\\2 \\1/", "g/^$/d", "v/keep/d", "2,4m0", "sort! n u", "'a,'bd", ".,+3s/a/b/g",
              "%s#a#b#g|%s/x/y/", "g/foo/s//bar/g", "g/^\\d/normal A;", ":%j!", "%>", "1,3>>", "s"]:
        print(c.ljust(40), json.dumps({k: v for k, v in parse_ex(c).items() if v not in (None, False)}, ensure_ascii=False))
