#!/usr/bin/env python3
"""Build the human-verification page for the flagged test slice (§7.3).

§7.3 asks for at least 150 test cases flagged for human verification, and
build_splits.py flags them round-robin across category x tier cells. This page
loads the ENTIRE test slice, not just that subset -- the unflagged rows are
equally unverified and equally part of the headline number, so the flag is a
priority marker here rather than a filter. Flagging is not verifying: every case is already machine-validated -- vim reproduces
the buffer, so the COMMAND is right -- but no gate can check that the
INSTRUCTION honestly describes the edit. That is exactly the defect the
adversarial audits caught 76 times in 1,140 authored cases: ambiguity with two
faithful readings, side effects the wording omits, instructions worded
backwards from their own range.

If that defect rate holds in the test slice, the headline exec-match number
carries about that much unquantified error -- and it penalises a CORRECT
model, which produces the faithful reading while the gold holds the other one.

Every row is re-executed at build time, so the page shows what vim actually
produces rather than trusting the stored buffer_after. A mismatch would be a
corpus bug and is flagged in red.

Usage:
    python scripts/build_test_review.py --out page.html
"""
import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim              # noqa: E402
from evaluate import _norm                # noqa: E402
from page_guard import check_page         # noqa: E402
from scope import _normal_mode_view       # noqa: E402



# ------------------------------------------------------- case sensitivity
#
# Letter case is the axis the audits found most often carried a genuine
# ambiguity: "capitalize" reads as initial-capital OR as all-caps, and the two
# produce different buffers. Worse, the corpus uses the word BOTH ways --
# hc-1024 "Capitalize the two letter state codes" means ALL CAPS, while
# hc-1625 "Capitalise the city name" means INITIAL CAP -- so no model can
# learn which is meant. Flagged here so a reviewer can go straight at them.

_CASE_CMD = [
    (re.compile(r"gU"), "gU uppercase operator"),
    (re.compile(r"gu"), "gu lowercase operator"),
    (re.compile(r"g~"), "g~ toggle operator"),
    (re.compile(r"\\U"), "\\U uppercase-rest"),
    (re.compile(r"\\L"), "\\L lowercase-rest"),
    (re.compile(r"\\u"), "\\u uppercase-next"),
    (re.compile(r"\\l"), "\\l lowercase-next"),
    (re.compile(r"\\c(?![a-zA-Z])"), "\\c force case-insensitive"),
    (re.compile(r"\\C(?![a-zA-Z])"), "\\C force case-sensitive"),
    (re.compile(r"sort!?\s+[a-z]*i"), "sort i (case-insensitive)"),
    (re.compile(r"s/[^/]*/[^/]*/[a-z]*i"), "s///i (case-insensitive)"),
]
_CASE_WORD = re.compile(
    r"\b(capitali[sz]\w*|upper[- ]?cas\w*|lower[- ]?cas\w*|title[- ]?cas\w*|"
    r"camel[- ]?cas\w*|pascal[- ]?cas\w*|snake[- ]?cas\w*|all ?caps|"
    r"case[- ]insensitiv\w*|case[- ]sensitiv\w*|flip the case|toggle the case|"
    r"swap the case|first letter|initial letter)\b", re.I)
_CAP_WORD = re.compile(r"\bcapitali[sz]\w*\b", re.I)


def _alt_case_spellings(cmd):
    """The other reading of a 'capitalize' instruction, as a command."""
    out = []
    if "\\U" in cmd:
        out.append(cmd.replace("\\U", "\\u"))
    elif "\\u" in cmd:
        out.append(cmd.replace("\\u", "\\U"))
    if re.search(r"gU[lh]", cmd):
        out.append(re.sub(r"gU[lh]", "gUiw", cmd))
    elif "gUiw" in cmd:
        out.append(cmd.replace("gUiw", "gUl"))
    return out


_QUOTED = re.compile(r"""['"‘’“”]([A-Za-z][A-Za-z0-9 _-]{0,24})['"‘’“”]""")


def _case_is_shown(ins):
    """Does the instruction state the case without using a case word?

    One way only, and it is unmistakable: two quoted literals that are the
    same word in different case -- "contain both a 'B' and a 'b'". The
    contrast IS the statement.

    A capitalised literal on its own is NOT enough, and counting it was a
    detector bug (owner, 2026-09-05). "Keep only lines containing 'GHF'"
    shows a capitalised string but never says the match is case-sensitive;
    a reader may take the capitals as incidental spelling, which is exactly
    the ambiguity this page exists to find. That branch was hiding 65
    case-dependent rows behind an assumption the corpus standard does not
    make -- the standard is that the instruction says case-sensitive, says
    case-insensitive, or names the letters as capitals.
    """
    lits = _QUOTED.findall(ins)
    return any(a != b and a.lower() == b.lower()
               for i, a in enumerate(lits) for b in lits[i + 1:])


# Widened 2026-09-05: the earlier list missed the owner's own phrasing.
# "with a capital N and a capital Z" (their rewording of kb13-0224a), the
# hyphenated "all-caps", and the compound-case names were all read as
# silent, which produced false flags on rows that do state the case.
_SAYS_CASE = re.compile(
    r"\b(capitali[sz]\w*|capitals?|upper[- ]?cas\w*|lower[- ]?cas\w*|"
    r"case[- ]sensitiv\w*|case[- ]insensitiv\w*|small letter\w*|"
    r"all[- ]?caps|camel[- ]?case|snake[- ]?case|title[- ]?case|"
    r"pascal[- ]?case|regardless of case|either case|any case|"
    r"ignoring case)\b", re.I)


def _match_sensitivity_matters(row, gold_norm):
    """Would reading the pattern case-INSENSITIVELY change the answer?

    This is the case question the wording usually leaves out. The mined regex (regular expression)
    rows quote a lowercase literal -- "delete lines using 2 instances of
    'irl'" -- over a buffer that deliberately also holds IRLIRL, so the
    distinction is exercised while nothing in the sentence names it.
    """
    cmd, mode = row["gold_command"], row.get("mode", "ex")
    alt = None
    if mode == "ex":
        m = re.match(r"^([^/]*)(s|g|v)/(.*)$", cmd)
        if m:
            alt = m.group(1) + m.group(2) + "/\\c" + m.group(3)
    else:
        m = re.match(r"^(.*?)([/?])(.+?)(<CR>|\r)(.*)$", cmd)
        if m:
            alt = (m.group(1) + m.group(2) + "\\c" + m.group(3)
                   + m.group(4) + m.group(5))
    if not alt:
        return None
    res = run_vim(row["buffer_before"], alt, mode=mode,
                  cursor=tuple(row.get("cursor") or (1, 1)))
    if res.get("blocked") or not res.get("was_valid"):
        return None
    return alt if _norm(res.get("buffer_after")) != gold_norm else None


def case_profile(row, gold_norm):
    """(signals, risk) -- 'ambiguous', 'unstated', 'relevant' or ''."""
    cmd, ins = row["gold_command"], row.get("instruction", "")
    signals = [label for rx, label in _CASE_CMD if rx.search(cmd)]
    word = _CASE_WORD.search(ins)
    if word:
        signals.append("wording: " + word.group(0))

    # the match-sensitivity axis, whether or not a case word appears
    alt = _match_sensitivity_matters(row, gold_norm)
    if alt:
        if _SAYS_CASE.search(ins) or _case_is_shown(ins):
            signals.append("match is case-sensitive, and the instruction "
                           "shows or states the case")
        else:
            signals.append("the pattern only matches ONE case and the buffer "
                           "holds the other, but the instruction never says "
                           "case matters (case-insensitive reading diverges: "
                           + alt + ")")
    if not signals:
        return [], ""
    risk = ("unstated"
            if (alt and not _SAYS_CASE.search(ins) and not _case_is_shown(ins))
            else "relevant")
    if _CAP_WORD.search(ins):
        for alt in _alt_case_spellings(cmd):
            res = run_vim(row["buffer_before"], alt,
                          mode=row.get("mode", "ex"),
                          cursor=tuple(row.get("cursor") or (1, 1)))
            if res.get("blocked") or not res.get("was_valid"):
                continue
            if _norm(res.get("buffer_after")) != gold_norm:
                signals.append("the other reading of \u201ccapitalize\u201d gives a "
                               "different buffer: " + alt)
                risk = "ambiguous"
                break
    return signals, risk



# ------------------------------------------------------- language ambiguity
#
# An edit that INSERTS language-specific syntax -- a comment marker, a
# statement terminator, True vs true, None vs null -- while nothing in the
# instruction names the language. Whether the buffer settles it is a
# judgment call, so the page shows the reasoning rather than a verdict.
# Three exclusions keep this from being noise (each was a real false
# positive on the first sweep): the instruction names the literal it
# inserts; the buffer already shows that style; or the language IS named,
# just not in a short list.

_LANG_TOKENS = [
    (re.compile(r"(^|\s)#\s"),                               "hash comment (#)"),
    (re.compile(r"(^|\s)//"),                                 "slash comment (//)"),
    (re.compile(r"(^|\s)--\s"),                              "dash comment (--)"),
    (re.compile(r"/\*|\*/"),                                 "block comment (/* */)"),
    (re.compile(r"<!--|-->"),                                  "html comment"),
    (re.compile(r";\s*$|;\s*<Esc>|;\\r"),                   "statement terminator (;)"),
    (re.compile(r"\b(True|False|None)\b"),                   "Python literal"),
    (re.compile(r"\b(true|false|null|undefined)\b"),         "C-family/JS literal"),
    (re.compile(r"\b(nil|NULL|TRUE|FALSE)\b"),               "other-language literal"),
    (re.compile(r"\b(def|elif|lambda|self)\b"),              "Python keyword"),
    (re.compile(r"\b(function|const|let|var|console)\b"),    "JS keyword"),
    (re.compile(r"\b(func|guard|fn)\b"),                     "Swift/Go/Rust keyword"),
    (re.compile(r"\b(public|private|static|void|String)\b"), "Java/C# keyword"),
    (re.compile(r"\b(int|char|struct|printf)\b(?!\()"),     "C keyword"),
    (re.compile(r"\b(SELECT|FROM|WHERE|JOIN|INSERT)\b"),     "SQL keyword"),
]
_LANG_NAME = re.compile(
    r"\b(python|javascript|typescript|java|c\+\+|c#|csharp|golang|rust|swift|"
    r"kotlin|ruby|php|sql|bash|shell|zsh|yaml|json|html|css|xml|markdown|lua|"
    r"perl|scala|makefile|dockerfile|toml|ini|nginx|graphql|protobuf|proto|"
    r"crontab|cron|env file|dotenv|systemd|terraform|hcl|regex)\b", re.I)
_LANG_STRONG = [
    (re.compile(r"^\s*def \w+\(|^\s*(import|from) \w|\bself\.|\belif\b|\bprint\(", re.M), "Python"),
    (re.compile(r"\bfunction\s*\w*\s*\(|=>|\bconst |\blet |console\.|require\(", re.M),  "JavaScript"),
    (re.compile(r"\bpublic (class|static|void)|System\.(out|Console)|\busing System", re.M), "Java/C#"),
    (re.compile(r"#include\s*<|\bint main\b|\bprintf\(|\bstd::|\bint \w+ = ", re.M),     "C/C++"),
    (re.compile(r"\b(SELECT|INSERT INTO|CREATE TABLE|UPDATE \w+ SET)\b", re.M),             "SQL"),
    (re.compile(r"\bfunc \w+\(|\bguard \w|\b(let|var) \w+: \w", re.M),                  "Swift/Go"),
    (re.compile(r"^#!/bin/|\bfi$|\besac\b|\$\{\w+\}|^export \w+=", re.M),               "shell"),
    (re.compile(r"^\S+ \S+ \S+ \S+ \S+ /", re.M),                                          "crontab"),
    (re.compile(r"<[a-zA-Z][^>]*>.*</[a-zA-Z]+>", re.M),                                      "HTML/XML"),
]


# Best-guess signals, used only when nothing strong fires. Each is a real
# shape from the test slice: shell build steps, a SQL (Structured Query Language) seed comment, a
# KEY=value config (configuration), a crontab row, a JSON (JavaScript Object Notation) filename.
_LANG_GUESS = [
    (re.compile(r"^(cd|make|cmake|npm|pip|git|export|echo|ls|cp|mv|rm)\b", re.M), "shell"),
    (re.compile(r"^--\s", re.M),                                            "SQL"),
    (re.compile(r"^\w+\s*=\s*\S+\s*;", re.M),                              "C-family / JS-like statements"),
    (re.compile(r"^[A-Za-z_][\w-]*=\S", re.M),                              "KEY=value config"),
    (re.compile(r"^\[\w+\]\s*$", re.M),                                    "INI"),
    (re.compile(r"\.json\b|^\s*\"\w+\":", re.M),                          "JSON"),
    (re.compile(r"\.ya?ml\b|^[\w-]+:\s+\S", re.M),                         "YAML"),
    (re.compile(r"\.py\b", re.M),                                          "Python"),
    (re.compile(r"\.(js|ts)\b", re.M),                                     "JavaScript/TypeScript"),
    (re.compile(r"\.(c|h|cpp|hpp|cc)\b", re.M),                            "C/C++"),
    (re.compile(r"\.sql\b", re.M),                                         "SQL"),
    (re.compile(r"\.(sh|bash)\b", re.M),                                   "shell"),
]


def _inserted_text(row):
    """Only the text a command ADDS can carry language syntax.

    ex: the replacement side of s/pat/repl/, any delimiter, skipping \\=
    expressions. keys: the insert-mode text, i.e. exactly what
    _normal_mode_view() blanks.
    """
    cmd = row["gold_command"]
    if row.get("mode") == "keys":
        view = _normal_mode_view(cmd)
        return "".join(ch if v == " " and ch != " " else " "
                       for ch, v in zip(cmd, view))
    outs = []
    for seg in re.split(r"(?<!\\)\|", cmd):
        m = re.search(r"(?:^|[^a-zA-Z])s(?:u|ub|ubs|ubst|ubsti|ubstit|ubstitu|"
                      r"ubstitut|ubstitute)?([^\w\s\\])", seg)
        if not m:
            continue
        d = re.escape(m.group(1))
        parts = re.split(r"(?<!\\)" + d, seg[m.end():])
        if len(parts) >= 2 and not parts[1].lstrip().startswith("\\="):
            outs.append(parts[1])
    return " ".join(outs)


def lang_profile(row):
    """(signals, risk, language) -- risk is 'weak', 'shown' or ''.

    'weak'  the buffer carries no strong language signal either -- the
            reviewer's first stop.
    'shown' the buffer does signal a language; the instruction still
            never names it, which may or may not matter.
    """
    ins = row.get("instruction", "")
    typed = _inserted_text(row)
    toks = [n for rx, n in _LANG_TOKENS if rx.search(typed)]
    if not toks or _LANG_NAME.search(ins):
        return [], "", ""
    buf = row["buffer_before"]
    # exclusion 1: the instruction names the literal it inserts
    words = set(re.findall(r"[A-Za-z_]+", typed))
    if any(re.search(r"\b" + re.escape(w) + r"\b", ins) for w in words if len(w) > 2):
        return [], "", ""
    # exclusion 2: the buffer already shows that style (its counterpart counts)
    pairs = {"true": "false", "false": "true", "True": "False", "False": "True",
             "null": "null", "None": "None"}
    for w in words:
        if w in buf or pairs.get(w, "\0") in buf:
            return [], "", ""
    signals = ["inserts " + ", ".join(toks), "typed: " + typed.strip()[:60]]
    strong = sorted({n for rx, n in _LANG_STRONG if rx.search(buf)})
    if strong:
        return signals, "shown", " / ".join(strong)
    guess = sorted({n for rx, n in _LANG_GUESS if rx.search(buf)})
    return signals, "weak", (" / ".join(guess) + " (best guess)") if guess else "unclear"


def collect(path=None, select=None, flag=None, sort_key=None, history=None):
    """Re-execute and profile the rows to review.

    path      case JSONL (default: the test split)
    select    optional row -> bool filter
    flag      optional (row, case_risk) -> bool; the page's priority marker
              (default: the §7.3 needs_review flag)
    sort_key  optional ordering over the built cards
              (default: category, tier, id)
    history   optional row -> str; one line of review history per card
    """
    path = pathlib.Path(path) if path else REPO / "data" / "splits" / "test.jsonl"
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if select is not None:
        rows = [r for r in rows if select(r)]
    out = []
    for r in rows:
        cur = tuple(r.get("cursor") or (1, 1))
        res = run_vim(r["buffer_before"], r["gold_command"],
                      mode=r.get("mode", "ex"), cursor=cur)
        got = res.get("buffer_after") or ""
        gold_norm = _norm(r["buffer_after"])
        agrees = _norm(got) == gold_norm
        sig, risk = case_profile(r, gold_norm)
        lsig, lrisk, llang = lang_profile(r)
        out.append({
            "id": r["id"], "cat": r["category"], "tier": r.get("tier") or "T?",
            "mode": r.get("mode", "ex"), "src": r.get("source", ""),
            "ins": r.get("instruction", ""), "cmd": r["gold_command"],
            "bef": r["buffer_before"], "exp": r["buffer_after"],
            "vim": got, "agrees": agrees,
            "valid": bool(res.get("was_valid")),
            "cur": list(cur),
            "flagged": bool(flag(r, risk) if flag else r.get("needs_review")),
            "was": (r.get("instruction_before_restore")
                    if r.get("instruction_before_restore") not in (None, r.get("instruction"))
                    else None),
            "note": r.get("restore_note"),
            "hist": history(r) if history else None,
            "case_signals": sig,
            "case_risk": risk,
            "lang_signals": lsig,
            "lang_risk": lrisk,
            "lang": llang,
        })
    out.sort(key=sort_key or (lambda c: (c["cat"], c["tier"], c["id"])))
    return out


CSS = r"""
:root{
  --ground:#F2F4F6; --surface:#FBFCFD; --sunk:#E6EAEE; --raise:#DFE5EA;
  --ink:#161B22; --ink2:#39424D; --muted:#66717E; --faint:#98A2AE;
  --line:#D3DAE1; --line2:#E3E8ED;
  --accent:#0F5F66; --accent2:#14797F; --accent-soft:#DCEDEE; --on-accent:#FBFCFD;
  --ok:#2C6A4A; --ok-bg:#DDEDE3; --ok-line:#A6CBB4;
  --warn:#8A5A12; --warn-bg:#F4E9D5; --warn-line:#DEC694;
  --stop:#98333C; --stop-bg:#F4E0E2; --stop-line:#DEAEB3;
  --cut:#98333C; --cut-bg:#F7E7E9; --add:#2C6A4A; --add-bg:#E1EFE7;
  --mono:"JetBrains Mono",ui-monospace,"SF Mono","DejaVu Sans Mono",Menlo,monospace;
  --body:"Public Sans",system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --disp:"Newsreader","Iowan Old Style",Georgia,serif;
}
:root:not([data-theme="light"]){ color-scheme:light dark; }
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
    --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
    --line:#2B333C; --line2:#222A32;
    --accent:#6FC3C8; --accent2:#8FD6DA; --accent-soft:#123134; --on-accent:#171C22;
    --ok:#6BBF92; --ok-bg:#15271D; --ok-line:#2B4634;
    --warn:#D6A75C; --warn-bg:#2A2215; --warn-line:#4A3B1F;
    --stop:#DE8790; --stop-bg:#2B1A1D; --stop-line:#4A2A30;
    --cut:#DE8790; --cut-bg:#2C1B1E; --add:#6BBF92; --add-bg:#152720;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --ground:#101418; --surface:#171C22; --sunk:#0C1014; --raise:#222A32;
  --ink:#E6EBF0; --ink2:#BAC4CE; --muted:#8492A0; --faint:#5D6874;
  --line:#2B333C; --line2:#222A32;
  --accent:#6FC3C8; --accent2:#8FD6DA; --accent-soft:#123134; --on-accent:#171C22;
  --ok:#6BBF92; --ok-bg:#15271D; --ok-line:#2B4634;
  --warn:#D6A75C; --warn-bg:#2A2215; --warn-line:#4A3B1F;
  --stop:#DE8790; --stop-bg:#2B1A1D; --stop-line:#4A2A30;
  --cut:#DE8790; --cut-bg:#2C1B1E; --add:#6BBF92; --add-bg:#152720;
}
*{ box-sizing:border-box; }
/* The whole slice is rendered at once -- 415 cards is roughly 200,000px of
   document, which overruns the height an artifact frame will size itself to,
   and the tail becomes unreachable. So the PAGE is pinned to the viewport and
   the LIST scrolls inside it: document height stays ~100vh no matter how many
   rows are loaded, and nothing is position:fixed. */
html,body{ height:100%; }
body{ margin:0; background:var(--ground); color:var(--ink);
  font-family:var(--body); font-size:15.5px; line-height:1.55;
  -webkit-font-smoothing:antialiased;
  display:flex; flex-direction:column; overflow:hidden; }
header{ flex:0 0 auto; }
.rail{ flex:0 0 auto; }
main{ flex:1 1 auto; min-height:0; overflow-y:auto; overflow-x:hidden; }
h1,h2{ font-family:var(--disp); font-weight:600; margin:0;
  letter-spacing:-.012em; text-wrap:balance; }
.mono{ font-family:var(--mono); }
.wrap{ max-width:1080px; margin:0 auto; padding:0 22px; }
.lab{ font-family:var(--mono); font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--muted); font-weight:700; }
:focus-visible{ outline:2px solid var(--accent); outline-offset:2px;
  border-radius:3px; }
header{ background:var(--surface); border-bottom:1px solid var(--line); }
header .wrap{ padding:22px 22px 16px; }
h1{ font-size:clamp(28px,4.2vw,40px); margin:9px 0 11px; display:inline; }
details#intro > summary{ list-style:none; cursor:pointer; display:flex;
  align-items:baseline; gap:12px; flex-wrap:wrap; user-select:none;
  border-radius:6px; margin:0 -6px; padding:0 6px; }
details#intro > summary::-webkit-details-marker{ display:none; }
details#intro > summary:hover{ background:var(--sunk); }
details#intro > summary .hint{ font-family:var(--mono); font-size:10.5px;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  white-space:nowrap; }
details#intro > summary .hint::before{ content:"\25BE  hide explanation"; }
details#intro:not([open]) > summary .hint::before{ content:"\25B8  why this page exists"; }
details#intro:not([open]) > summary h1{ margin-bottom:4px; }
details#intro > .lede:first-of-type{ margin-top:4px; }
.lede{ max-width:66ch; color:var(--ink2); font-size:16px; }
.lede b{ color:var(--ink); font-weight:600; }
.band{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:1px; background:var(--line); border:1px solid var(--line);
  border-radius:9px; overflow:hidden; margin-top:20px; }
.band div{ background:var(--surface); padding:12px 15px; }
.band .v{ font-family:var(--mono); font-size:21px; font-weight:700;
  display:block; font-variant-numeric:tabular-nums; }
.band .k{ font-family:var(--mono); font-size:9.5px; letter-spacing:.07em;
  text-transform:uppercase; color:var(--muted); margin-top:3px; display:block; }
.rail{ z-index:20; background:var(--surface);
  border-bottom:1px solid var(--line); border-top:1px solid var(--line); }
.rail .wrap{ display:flex; flex-wrap:wrap; gap:7px 9px; align-items:center;
  padding:10px 22px; }
.f{ font-family:var(--mono); font-size:11.5px; padding:4px 9px;
  border-radius:5px; cursor:pointer; background:var(--surface);
  border:1px solid var(--line); color:var(--ink2); }
.f:hover{ border-color:var(--accent); color:var(--accent); }
.f[aria-pressed="true"]{ background:var(--accent-soft);
  border-color:var(--accent); color:var(--accent); font-weight:700; }
select{ font:inherit; font-size:12px; padding:4px 7px; border-radius:5px;
  border:1px solid var(--line); background:var(--ground); color:inherit;
  font-family:var(--mono); }
.count{ font-family:var(--mono); font-size:12px; color:var(--muted);
  margin-left:auto; font-variant-numeric:tabular-nums; }
main{ padding:22px 0 40px; }
.cards{ display:flex; flex-direction:column; gap:13px; }
.card{ background:var(--surface); border:1px solid var(--line);
  border-left:3px solid var(--line); border-radius:8px; overflow:hidden; }
.card.v-ok{ border-left-color:var(--ok); }
.card.v-reword{ border-left-color:var(--warn); }
.card.v-drop{ border-left-color:var(--stop); }
.card.mismatch{ border-left-color:var(--stop); background:var(--stop-bg); }
.chead{ display:flex; flex-wrap:wrap; gap:6px 9px; align-items:center;
  padding:10px 14px; border-bottom:1px solid var(--line2); }
.cid{ font-family:var(--mono); font-size:11.5px; color:var(--muted); }
.chip{ font-family:var(--mono); font-size:10px; padding:2px 6px;
  border-radius:4px; background:var(--raise); color:var(--ink2);
  white-space:nowrap; }
.chip.flagchip{ background:var(--accent-soft); color:var(--accent);
  font-weight:700; }
.chip.edited{ background:var(--warn-bg); color:var(--warn);
  border:1px solid var(--warn-line); font-weight:700; }
.chip.case{ background:var(--accent-soft); color:var(--accent); }
.chip.caseamb{ background:var(--stop-bg); color:var(--stop);
  border:1px solid var(--stop-line); font-weight:700; }
.chip.caseuns{ background:var(--warn-bg); color:var(--warn);
  border:1px solid var(--warn-line); font-weight:700; }
.casenote.uns{ background:var(--warn-bg); border-color:var(--warn-line);
  color:var(--warn); }
.chip.langweak{ background:var(--warn-bg); color:var(--warn);
  border:1px solid var(--warn-line); font-weight:700; }
.chip.langshown{ background:var(--raise); color:var(--ink2); }
.casenote.lang{ background:var(--warn-bg); border-color:var(--warn-line);
  color:var(--warn); }
.casenote.langshown{ background:var(--sunk); border-color:var(--line);
  color:var(--ink2); }
.langlabel{ display:inline-block; font-family:var(--mono); font-size:11.5px;
  letter-spacing:.04em; padding:2px 8px; border-radius:4px; margin-right:6px;
  background:var(--surface); border:1px solid currentColor; }
.langlabel b{ font-weight:700; }
.casenote{ margin:0 15px 10px; padding:8px 11px; border-radius:6px;
  background:var(--stop-bg); border:1px solid var(--stop-line);
  color:var(--stop); font-size:12.5px; line-height:1.5; }
.casenote.mild{ background:var(--accent-soft); border-color:var(--accent);
  color:var(--accent); }
.casenote b{ font-weight:700; }
.casenote code{ font-family:var(--mono); font-size:.92em; }
.editform{ padding:11px 14px; border-top:1px solid var(--line2);
  display:flex; flex-direction:column; gap:9px; background:var(--sunk); }
.editrow{ display:flex; gap:9px; flex-wrap:wrap; align-items:flex-end; }
.editrow > label{ display:flex; flex-direction:column; gap:4px;
  flex:1 1 200px; min-width:0; }
.editrow > label.sm{ flex:0 0 92px; }
.editform label > span{ font-family:var(--mono); font-size:9.5px;
  letter-spacing:.12em; text-transform:uppercase; color:var(--muted);
  font-weight:700; }
.editform textarea, .editform input, .editform select{
  font-family:var(--mono); font-size:12.5px; padding:6px 8px;
  border:1px solid var(--line); border-radius:5px; background:var(--surface);
  color:var(--ink); width:100%; }
.editform textarea{ resize:vertical; min-height:64px; white-space:pre; }
.editform textarea.ins{ font-family:var(--body); font-size:14px;
  white-space:pre-wrap; min-height:52px; }
.editform .dirty{ border-color:var(--warn); background:var(--warn-bg); }
.stale{ font-family:var(--mono); font-size:11.5px; color:var(--warn);
  background:var(--warn-bg); border:1px solid var(--warn-line);
  border-radius:5px; padding:7px 10px; line-height:1.45; }
.keys{ font-family:var(--mono); font-size:12.5px; font-weight:700;
  background:var(--accent-soft); color:var(--accent); padding:3px 8px;
  border-radius:4px; overflow-wrap:anywhere; }
.spacer{ flex:1 1 auto; }
.ins{ padding:13px 15px 4px; font-size:16.5px; color:var(--ink);
  max-width:70ch; font-family:var(--disp); line-height:1.45; }
.was{ padding:0 15px 6px; font-size:13px; color:var(--muted); max-width:70ch;
  font-family:var(--body); line-height:1.4; }
.was .wl{ font-family:var(--mono); font-size:10px; letter-spacing:.1em;
  text-transform:uppercase; margin-right:8px; color:var(--faint); }
.was.note{ color:var(--warn); }
.was.hist{ color:var(--ok); }
.qn{ padding:2px 15px 10px; font-size:12.5px; color:var(--muted); }
.panes{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr));
  gap:1px; background:var(--line2); border-top:1px solid var(--line2); }
.pane{ background:var(--surface); padding:9px 13px; min-width:0;
  overflow-x:auto; }
.pane .lab{ margin-bottom:5px; display:block; }
.ln{ display:flex; white-space:pre; font-family:var(--mono); font-size:12.5px;
  line-height:1.5; }
.ln .g{ color:var(--faint); opacity:.55; user-select:none; text-align:right;
  min-width:2em; padding-right:9px; flex:0 0 auto;
  font-variant-numeric:tabular-nums; }
.ln.cut{ background:var(--cut-bg); } .ln.cut .t{ color:var(--cut); }
.ln.add{ background:var(--add-bg); } .ln.add .t{ color:var(--add); }
.ws{ color:var(--accent); opacity:.42; }
.agree{ font-family:var(--mono); font-size:10.5px; padding:2px 7px;
  border-radius:4px; border:1px solid; }
.agree.yes{ background:var(--ok-bg); color:var(--ok); border-color:var(--ok-line); }
.agree.no{ background:var(--stop-bg); color:var(--stop); border-color:var(--stop-line); }
.cfoot{ display:flex; flex-wrap:wrap; gap:7px; align-items:center;
  padding:9px 14px; border-top:1px solid var(--line2); }
.vb{ font-family:var(--mono); font-size:11.5px; padding:4px 11px;
  border-radius:5px; cursor:pointer; background:none;
  border:1px solid var(--line); color:var(--muted); }
.vb:hover{ border-color:var(--accent); color:var(--accent); }
.vb.ok[aria-pressed="true"]{ background:var(--ok-bg); color:var(--ok);
  border-color:var(--ok); font-weight:700; }
.vb.reword[aria-pressed="true"]{ background:var(--warn-bg); color:var(--warn);
  border-color:var(--warn); font-weight:700; }
.vb.drop[aria-pressed="true"]{ background:var(--stop-bg); color:var(--stop);
  border-color:var(--stop); font-weight:700; }
input.note{ flex:1 1 220px; font:inherit; font-size:13px; padding:5px 9px;
  border:1px solid var(--line); border-radius:5px; background:var(--ground);
  color:var(--ink); min-width:0; }
input.note:focus{ outline:none; border-color:var(--accent);
  box-shadow:0 0 0 2px var(--accent-soft); }
.dock{ flex:0 0 auto; background:var(--surface);
  border-bottom:1px solid var(--line); }
.dock .wrap{ display:flex; gap:10px; align-items:center; padding:9px 22px;
  flex-wrap:wrap; }
.btn{ font-family:var(--mono); font-size:12px; padding:6px 13px;
  border-radius:6px; cursor:pointer; background:var(--surface);
  border:1px solid var(--line); color:var(--ink2); }
.btn:hover{ border-color:var(--accent); color:var(--accent); }
.btn.pri{ background:var(--accent); border-color:var(--accent);
  color:var(--on-accent); font-weight:700; }
.bar{ flex:1 1 200px; height:7px; border-radius:4px; background:var(--raise);
  overflow:hidden; min-width:120px; }
.bar i{ display:block; height:100%; background:var(--accent); }
dialog{ max-height:90vh; overflow:auto; border:1px solid var(--line);
  border-radius:9px; background:var(--surface); color:var(--ink);
  max-width:min(860px,94vw); width:100%; padding:0; }
dialog::backdrop{ background:rgba(16,20,24,.55); }
dialog .dh{ padding:16px 19px 0; }
dialog .db{ padding:12px 19px 17px; display:flex; flex-direction:column; gap:9px; }
dialog textarea{ min-height:250px; font-family:var(--mono); font-size:11.5px;
  border:1px solid var(--line); border-radius:6px; background:var(--ground);
  color:var(--ink); padding:9px; }
.boot{ padding:34px 0; text-align:center; color:var(--muted);
  font-family:var(--mono); font-size:13px; }
.empty{ padding:40px 0; text-align:center; color:var(--muted);
  font-family:var(--mono); font-size:13px; }
@media (prefers-reduced-motion:reduce){ *{ transition:none !important; } }
"""

JS = r"""
const DATA = JSON.parse(document.getElementById('payload').textContent);
const KEY = 'vimbot-test-review-v1';
let verdicts = {};
try { verdicts = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { verdicts = {}; }
let fCat = 'all', fState = 'all', fFlag = 'all', fCase = 'all', fLang = 'all';
let edits = {};
const EKEY = 'vimbot-test-edits-v1';
try { edits = JSON.parse(localStorage.getItem(EKEY) || '{}'); } catch (e) { edits = {}; }
const openEdit = {};

function save(){
  try {
    localStorage.setItem(KEY, JSON.stringify(verdicts));
    localStorage.setItem(EKEY, JSON.stringify(edits));
  } catch (e) {}
}
function escAttr(s){
  return esc(s).replace(/"/g, '&quot;');
}
function edited(id){
  const e = edits[id];
  return !!(e && Object.keys(e).length);
}
// A change to the command, buffer, cursor or mode makes the stored
// buffer_after stale. The page cannot run vim, so it says so and lets
// apply_curation.py recompute -- the page proposes, vim decides.
function stale(id){
  const e = edits[id] || {};
  return ['command', 'buffer_before', 'cursor', 'mode']
    .some(function(k){ return e[k] !== undefined; });
}
function fieldVal(c, key){
  const e = edits[c.id] || {};
  if (e[key] !== undefined) return e[key];
  return key === 'cursor' ? c.cur
       : key === 'buffer_before' ? c.bef
       : key === 'instruction' ? c.ins
       : key === 'command' ? c.cmd
       : key === 'mode' ? c.mode : '';
}
function setField(c, key, value){
  const orig = key === 'cursor' ? JSON.stringify(c.cur)
             : key === 'buffer_before' ? c.bef
             : key === 'instruction' ? c.ins
             : key === 'command' ? c.cmd : c.mode;
  const now = key === 'cursor' ? JSON.stringify(value) : value;
  edits[c.id] = edits[c.id] || {};
  if (now === orig) delete edits[c.id][key];
  else edits[c.id][key] = value;
  if (!Object.keys(edits[c.id]).length) delete edits[c.id];
  save();
}
function esc(s){
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function vis(line){
  let out = esc(line).replace(/\t/g, '<span class="ws">--&gt;</span>\t');
  const m = out.match(/( +)$/);
  if (m) out = out.slice(0, out.length - m[1].length)
              + '<span class="ws">' + '.'.repeat(m[1].length) + '</span>';
  return out;
}
function lines(t){
  const l = String(t).split('\n');
  if (l.length && l[l.length-1] === '') l.pop();
  return l;
}
function pane(label, text, other, kind){
  const a = lines(text), b = lines(other);
  let rows = '';
  for (let i = 0; i < a.length; i++){
    const cls = a[i] !== b[i] ? ' ' + kind : '';
    rows += '<div class="ln' + cls + '"><span class="g">' + (i+1)
          + '</span><span class="t">' + vis(a[i]) + '</span></div>';
  }
  if (!rows) rows = '<div class="ln"><span class="g">-</span>'
                  + '<span class="t">(empty buffer)</span></div>';
  return '<div class="pane"><span class="lab">' + label + '</span>' + rows + '</div>';
}
function card(c){
  const v = verdicts[c.id] || {};
  let cls = c.agrees ? (v.verdict ? 'v-' + v.verdict : '') : 'mismatch';
  if (edited(c.id)) cls += ' v-reword';
  const agree = c.agrees
    ? '<span class="agree yes">vim agrees with expected</span>'
    : '<span class="agree no">VIM DISAGREES - corpus bug</span>';
  function btn(kind, label){
    const on = v.verdict === kind;
    return '<button class="vb ' + kind + '" data-act="' + kind + '" data-id="'
         + esc(c.id) + '" aria-pressed="' + on + '">' + label + '</button>';
  }
  return '<div class="card ' + cls + '" id="c-' + esc(c.id) + '">'
    + '<div class="chead">'
    +   '<span class="cid">' + esc(c.id) + '</span>'
    +   '<span class="keys">' + esc(c.mode) + ': ' + esc(c.cmd) + '</span>'
    +   '<span class="chip">' + esc(c.cat) + '</span>'
    +   '<span class="chip">' + esc(c.tier) + '</span>'
    +   '<span class="chip">cursor ' + c.cur[0] + ',' + c.cur[1] + '</span>'
    +   '<span class="spacer"></span>' + agree
    + '</div>'
    + '<div class="ins">' + esc(c.ins || '(no instruction)') + '</div>'
    + (c.was ? '<div class="was"><span class="wl">was</span>' + esc(c.was) + '</div>' : '')
    + (c.note ? '<div class="was note"><span class="wl">note</span>' + esc(c.note) + '</div>' : '')
    + (c.hist ? '<div class="was hist"><span class="wl">review</span>' + esc(c.hist) + '</div>' : '')
    + '<div class="qn">Does that sentence describe this edit completely and '
    + 'unambiguously &mdash; including any side effect?</div>'
    + caseNote(c)
    + langNote(c)
    + '<div class="panes">'
    +   pane('input buffer', c.bef, c.exp, 'cut')
    +   pane('expected output', c.exp, c.bef, 'add')
    +   pane('what vim produced', c.vim, c.exp, c.agrees ? 'add' : 'cut')
    + '</div>'
    + editBlock(c)
    + '<div class="cfoot">'
    +   btn('ok', 'Accurate') + btn('reword', 'Needs rewording') + btn('drop', 'Drop')
    +   '<button class="vb" data-act="edit" data-id="' + escAttr(c.id) + '">'
    +   (openEdit[c.id] ? 'Close editor' : 'Edit') + '</button>'
    +   (edited(c.id)
        ? '<button class="vb" data-act="revert" data-id="' + escAttr(c.id)
          + '">Revert</button>' : '')
    +   '<input class="note" data-id="' + escAttr(c.id)
    +   '" placeholder="note (optional)" value="' + escAttr(v.note || '') + '">'
    + '</div>'
    + '</div>';
}

function caseNote(c){
  if (!c.case_risk) return '';
  const list = c.case_signals.map(function(x){ return esc(x); }).join(' &middot; ');
  if (c.case_risk === 'ambiguous'){
    return '<div class="casenote"><b>Letter case is ambiguous here.</b> '
      + 'The other reading of the wording produces a different buffer, so a '
      + 'correct model can be marked wrong. ' + list + '</div>';
  }
  if (c.case_risk === 'unstated'){
    return '<div class="casenote uns"><b>Case-sensitivity is load-bearing but '
      + 'unstated.</b> The instruction quotes a literal without saying whether '
      + 'its case matters, and the buffer holds the other casing too. vim\u2019s '
      + 'default is case-sensitive, so the convention lands on the gold \u2014 '
      + 'but the sentence does not say it. ' + list + '</div>';
  }
  return '<div class="casenote mild"><b>Letter case is load-bearing.</b> '
    + list + '</div>';
}

function langNote(c){
  if (!c.lang_risk) return '';
  const list = c.lang_signals.map(function(x){ return esc(x); }).join(' &middot; ');
  const label = '<span class="langlabel">Language: <b>' + esc(c.lang) + '</b></span> ';
  if (c.lang_risk === 'weak'){
    return '<div class="casenote lang">' + label
      + 'The edit inserts language-specific syntax, no language is named, and '
      + 'the buffer does not settle it. ' + list + '</div>';
  }
  return '<div class="casenote langshown">' + label
    + 'The buffer shows the language but the instruction never names it. '
    + list + '</div>';
}

function editBlock(c){
  if (!openEdit[c.id]) return '';
  const e = edits[c.id] || {};
  const dc = function(k){ return e[k] !== undefined ? ' dirty' : ''; };
  const cur = fieldVal(c, 'cursor');
  const modes = ['ex', 'keys'].map(function(m){
    return '<option value="' + m + '"'
         + (fieldVal(c, 'mode') === m ? ' selected' : '') + '>' + m + '</option>';
  }).join('');
  return '<div class="editform">'
    + (stale(c.id)
       ? '<div class="stale">Command, buffer, cursor or mode changed &mdash; the '
       + 'expected output above is now stale. Export and run '
       + 'apply_curation.py, which re-executes the row in the sandbox and '
       + 'recomputes it. Nothing is taken on trust.</div>'
       : '')
    + '<label><span>instruction</span>'
    + '<textarea class="ins' + dc('instruction') + '" data-field="instruction" '
    + 'data-id="' + escAttr(c.id) + '" spellcheck="true">'
    + esc(fieldVal(c, 'instruction')) + '</textarea></label>'
    + '<div class="editrow">'
    +   '<label><span>command</span>'
    +   '<textarea class="' + dc('command').trim() + '" data-field="command" '
    +   'data-id="' + escAttr(c.id) + '" spellcheck="false" '
    +   'style="min-height:40px">' + esc(fieldVal(c, 'command')) + '</textarea></label>'
    +   '<label class="sm"><span>mode</span><select data-field="mode" '
    +   'data-id="' + escAttr(c.id) + '" class="' + dc('mode').trim() + '">'
    +   modes + '</select></label>'
    +   '<label class="sm"><span>cursor line</span>'
    +   '<input type="number" min="1" data-field="cursorline" data-id="'
    +   escAttr(c.id) + '" value="' + cur[0] + '" class="' + dc('cursor').trim()
    +   '"></label>'
    +   '<label class="sm"><span>cursor col</span>'
    +   '<input type="number" min="1" data-field="cursorcol" data-id="'
    +   escAttr(c.id) + '" value="' + cur[1] + '" class="' + dc('cursor').trim()
    +   '"></label>'
    + '</div>'
    + '<label><span>buffer before</span>'
    + '<textarea data-field="buffer_before" data-id="' + escAttr(c.id) + '" '
    + 'spellcheck="false">' + esc(fieldVal(c, 'buffer_before')) + '</textarea></label>'
    + '</div>';
}
function matches(c){
  if (fCat !== 'all' && c.cat !== fCat) return false;
  if (fFlag === 'flagged' && !c.flagged) return false;
  if (fFlag === 'unflagged' && c.flagged) return false;
  if (fCase === 'case' && !c.case_risk) return false;
  if (fCase === 'amb' && c.case_risk !== 'ambiguous') return false;
  if (fCase === 'uns' && c.case_risk !== 'unstated') return false;
  if (fLang === 'any' && !c.lang_risk) return false;
  if (fLang === 'weak' && c.lang_risk !== 'weak') return false;
  const v = (verdicts[c.id] || {}).verdict;
  if (fState === 'todo' && v) return false;
  if (fState === 'done' && !v) return false;
  if (fState === 'issues' && v !== 'reword' && v !== 'drop') return false;
  return true;
}
function render(resetScroll){
  const hits = DATA.filter(matches);
  const el = document.getElementById('cards');
  // every matching row is rendered -- no pagination. The whole test slice is
  // the review unit, and a hidden tail is a slice nobody checks.
  el.innerHTML = hits.length
    ? hits.map(card).join('')
    : '<div class="empty">nothing matches that filter</div>';
  document.getElementById('count').textContent =
    hits.length + ' of ' + DATA.length + ' shown';
  if (resetScroll){
    const sc = document.getElementById('scroller');
    if (sc) sc.scrollTop = 0;
  }
  const done = DATA.filter(function(c){ return (verdicts[c.id] || {}).verdict; }).length;
  document.getElementById('prog').textContent =
    done + ' / ' + DATA.length + ' reviewed';
  document.getElementById('barfill').style.width =
    (DATA.length ? (100 * done / DATA.length) : 0) + '%';
}
function byId(id){
  for (let i = 0; i < DATA.length; i++) if (DATA[i].id === id) return DATA[i];
  return null;
}
function onClick(ev){
  const b = ev.target.closest('[data-act]');
  if (!b) return;
  const id = b.getAttribute('data-id'), act = b.getAttribute('data-act');
  if (act === 'edit'){
    openEdit[id] = !openEdit[id];
    render();
    return;
  }
  if (act === 'revert'){
    if (!confirm('Discard the edits to ' + id + '?')) return;
    delete edits[id];
    save(); render();
    return;
  }
  const cur = verdicts[id] || {};
  cur.verdict = cur.verdict === act ? null : act;
  if (!cur.verdict) delete cur.verdict;
  verdicts[id] = cur;
  save(); render();
}
function onInput(ev){
  const note = ev.target.closest('input.note');
  if (note){
    const id = note.getAttribute('data-id');
    verdicts[id] = verdicts[id] || {};
    verdicts[id].note = note.value;
    save();
    return;
  }
  const fld = ev.target.closest('[data-field]');
  if (!fld) return;
  const id = fld.getAttribute('data-id');
  const key = fld.getAttribute('data-field');
  const c = byId(id);
  if (!c) return;
  if (key === 'cursorline' || key === 'cursorcol'){
    const cur = fieldVal(c, 'cursor').slice();
    const n = parseInt(fld.value, 10);
    cur[key === 'cursorline' ? 0 : 1] = isNaN(n) || n < 1 ? 1 : n;
    setField(c, 'cursor', cur);
  } else {
    setField(c, key, fld.value);
  }
  // repaint only this card's chrome; re-rendering would steal focus mid-type
  const card = document.getElementById('c-' + id);
  if (card){
    fld.classList.toggle('dirty', (edits[id] || {})[key] !== undefined
                                  || (key.indexOf('cursor') === 0
                                      && (edits[id] || {}).cursor !== undefined));
    const head = card.querySelector('.chead');
    const has = head && head.querySelector('.chip.edited');
    if (edited(id) && !has && head){
      const sp = document.createElement('span');
      sp.className = 'chip edited';
      sp.textContent = 'edited';
      head.insertBefore(sp, head.querySelector('.spacer'));
    } else if (!edited(id) && has){ has.remove(); }
  }
}
// Emitted in apply_curation.py's own change-set shape, so the export runs
// as-is:  python scripts/apply_curation.py --changes changes.json --dry-run
// A "drop" verdict becomes a delete, with the reviewer's note as the reason.
function exportJson(){
  const out = { edits: {}, deletes: {}, review: {} };
  let shadowed = 0;
  DATA.forEach(function(c){
    const v = verdicts[c.id] || {};
    const e = edits[c.id];
    // A drop wins over an edit left on the same row: apply_curation.py refuses
    // a change-set that lists one id as both, and editing a row then dropping
    // it means the drop is the later decision. The edit stays in this
    // browser, so clearing the verdict brings it back.
    if (e && Object.keys(e).length){
      if (v.verdict === 'drop') shadowed++;
      else out.edits[c.id] = e;
    }
    if (v.verdict === 'drop'){
      out.deletes[c.id] = v.note
        || 'dropped in test-slice verification: the instruction does not '
         + 'describe this edit';
    }
    if (v.verdict || v.note) out.review[c.id] = v;
  });
  document.getElementById('out').value = JSON.stringify(out, null, 1);
  const n = Object.keys(out.edits).length, d = Object.keys(out.deletes).length;
  document.getElementById('dlgsub').textContent =
    n + ' edit(s), ' + d + ' delete(s)'
    + (shadowed ? '. ' + shadowed + ' edited row(s) are marked drop, so their '
       + 'edits are left out — clear the verdict to export the edit instead'
       : '')
    + '. Every edited row is re-executed by apply_curation.py and its expected '
    + 'output recomputed; a row whose new command fails, no-ops, or leaks the '
    + 'answer is refused.';
  document.getElementById('dlg').showModal();
}
try{
  // the intro is a per-viewer convenience: remember whether it was collapsed
  const intro = document.getElementById('intro');
  try {
    if (localStorage.getItem('vimbot-test-intro') === 'closed') intro.open = false;
  } catch (e) {}
  intro.addEventListener('toggle', function(){
    try { localStorage.setItem('vimbot-test-intro', intro.open ? 'open' : 'closed'); }
    catch (e) {}
  });
  document.getElementById('boot').style.display = 'none';
  document.querySelectorAll('[data-cat]').forEach(function(b){
    b.addEventListener('click', function(){
      fCat = b.getAttribute('data-cat');
      document.querySelectorAll('[data-cat]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render(true);
    });
  });
  document.querySelectorAll('[data-state]').forEach(function(b){
    b.addEventListener('click', function(){
      fState = b.getAttribute('data-state');
      document.querySelectorAll('[data-state]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render(true);
    });
  });
  document.querySelectorAll('[data-lang]').forEach(function(b){
    b.addEventListener('click', function(){
      fLang = b.getAttribute('data-lang');
      document.querySelectorAll('[data-lang]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render(true);
    });
  });
  document.querySelectorAll('[data-case]').forEach(function(b){
    b.addEventListener('click', function(){
      fCase = b.getAttribute('data-case');
      document.querySelectorAll('[data-case]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render(true);
    });
  });
  document.querySelectorAll('[data-flag]').forEach(function(b){
    b.addEventListener('click', function(){
      fFlag = b.getAttribute('data-flag');
      document.querySelectorAll('[data-flag]').forEach(function(x){
        x.setAttribute('aria-pressed', String(x === b)); });
      render(true);
    });
  });
  document.getElementById('cards').addEventListener('click', onClick);
  document.getElementById('cards').addEventListener('input', onInput);
  document.getElementById('export').addEventListener('click', exportJson);
  document.getElementById('top').addEventListener('click', function(){
    document.getElementById('scroller').scrollTo({ top: 0, behavior: 'smooth' });
  });
  document.getElementById('close').addEventListener('click', function(){
    document.getElementById('dlg').close();
  });
  document.getElementById('copy').addEventListener('click', function(){
    const t = document.getElementById('out');
    t.select();
    try { document.execCommand('copy'); } catch (e) {}
  });
  document.getElementById('reset').addEventListener('click', function(){
    if (!confirm('Discard every verdict AND every edit recorded on this page?')) return;
    verdicts = {}; edits = {}; save(); render();
  });
  render();
}catch(err){
  const b = document.getElementById('boot');
  b.style.display = 'block';
  b.textContent = 'page failed to start: ' + err;
}
"""


# The words on the page. A sibling builder (build_kb13_review.py) reviews a
# different row set with the same machinery by overriding these.
DEFAULT_META = {
    "title": "Test Slice Verification",
    "lab": "vimBot &middot; test slice &middot; human verification (&sect;7.3)",
    "h1": "Is the instruction telling the truth?",
    "lede": None,                  # None -> TEST_LEDE % (n, flagged)
    "total_label": "test cases in total",
    "flag_label": "&sect;7.3 priority rows",
    "flag_button": "&sect;7.3 flagged",
    "key": "vimbot-test",          # localStorage namespace for verdicts/edits
}

TEST_LEDE = """\
  <p class="lede">
    This is the <b>entire %d-case test slice</b>. Every one already passes the
    machine gate &mdash; vim reproduces the buffer, so the
    <i>command</i> is right. What no gate can check is whether the
    <i>instruction</i> honestly describes the edit, and that is the defect the
    adversarial audits caught <b>76 times in 1,140</b> authored cases: two
    faithful readings, an unmentioned side effect, wording backwards from its
    own range. If that rate holds here, the headline exec-match number carries
    about that much hidden error &mdash; and it punishes a <b>correct</b> model,
    which answers the reading the gold did not take.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>%d rows carry the &sect;7.3 review flag</b>, dealt round-robin across
    category &times; tier. Treat that as the priority order, not the scope: the
    unflagged rows are just as unverified and count toward the same headline
    number. Every row is rendered &mdash; there is no hidden tail.
    <b>Edit</b> opens the instruction, command, mode, cursor and input buffer
    for correction; the export is a change-set
    <span class="mono">apply_curation.py</span> runs directly, and it
    re-executes every edited row rather than trusting the page.
  </p>
"""


def build(cases, meta=None):
    meta = {**DEFAULT_META, **(meta or {})}
    n = len(cases)
    flagged = sum(1 for c in cases if c["flagged"])
    casey = sum(1 for c in cases if c["case_risk"])
    ambig = sum(1 for c in cases if c["case_risk"] == "ambiguous")
    unstated = sum(1 for c in cases if c["case_risk"] == "unstated")
    lang_any = sum(1 for c in cases if c["lang_risk"])
    lang_weak = sum(1 for c in cases if c["lang_risk"] == "weak")
    mismatch = sum(1 for c in cases if not c["agrees"])
    cats = sorted({c["cat"] for c in cases})
    cat_btns = "".join(
        '<button class="f" data-cat="%s" aria-pressed="false">%s'
        '<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>'
        % (c, c, sum(1 for x in cases if x["cat"] == c)) for c in cats)
    payload = json.dumps(cases, ensure_ascii=False)
    lede = meta["lede"] if meta["lede"] is not None else TEST_LEDE % (n, flagged)
    js = (JS.replace("'vimbot-test-review-v1'", "'%s-review-v1'" % meta["key"])
            .replace("'vimbot-test-edits-v1'", "'%s-edits-v1'" % meta["key"])
            .replace("'vimbot-test-intro'", "'%s-intro'" % meta["key"]))
    return """<title>%s</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?\
family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;600&\
family=JetBrains+Mono:wght@400;600;700&display=swap">
<style>%s</style>

<header><div class="wrap">
  <div class="lab">%s</div>
  <details id="intro" open>
  <summary><h1>%s</h1>
    <span class="hint" aria-hidden="true"></span></summary>
%s  </details>
  <div class="band">
    <div><span class="v">%d</span><span class="k">%s</span></div>
    <div><span class="v">%d</span><span class="k">%s</span></div>
    <div><span class="v">%d</span><span class="k">categories covered</span></div>
    <div><span class="v" style="color:var(--warn)">%d</span><span class="k">case unstated</span></div>
    <div><span class="v"%s>%d</span><span class="k">case ambiguous</span></div>
    <div><span class="v"%s>%d</span><span class="k">vim disagrees (corpus bug)</span></div>
    <div><span class="v" id="prog">0 / %d</span><span class="k">reviewed here</span></div>
  </div>
</div></header>

<div class="rail"><div class="wrap">
  <span class="lab">category</span>
  <button class="f" data-cat="all" aria-pressed="true">all<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>
  %s
  <span class="lab" style="margin-left:8px">priority</span>
  <button class="f" data-flag="all" aria-pressed="true">all</button>
  <button class="f" data-flag="flagged" aria-pressed="false">%s</button>
  <button class="f" data-flag="unflagged" aria-pressed="false">the rest</button>
  <span class="lab" style="margin-left:8px">case</span>
  <button class="f" data-case="all" aria-pressed="true">any</button>
  <button class="f" data-case="case" aria-pressed="false">case matters<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>
  <button class="f" data-case="uns" aria-pressed="false">case unstated<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>
  <button class="f" data-case="amb" aria-pressed="false">case ambiguous<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>
  <span class="lab" style="margin-left:8px">language</span>
  <button class="f" data-lang="all" aria-pressed="true">any</button>
  <button class="f" data-lang="any" aria-pressed="false">unnamed<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>
  <button class="f" data-lang="weak" aria-pressed="false">unstated &amp; unclear<span class="c" style="opacity:.6;margin-left:5px">%d</span></button>
  <span class="lab" style="margin-left:8px">state</span>
  <button class="f" data-state="all" aria-pressed="true">any</button>
  <button class="f" data-state="todo" aria-pressed="false">not reviewed</button>
  <button class="f" data-state="done" aria-pressed="false">reviewed</button>
  <button class="f" data-state="issues" aria-pressed="false">has issues</button>
  <span class="count" id="count"></span>
</div></div>

<div class="dock"><div class="wrap">
  <div class="bar"><i id="barfill" style="width:0%%"></i></div>
  <button class="btn pri" id="export">Export verdicts &amp; edits</button>
  <button class="btn" id="reset">Discard</button>
  <button class="btn" id="top">Back to top</button>
</div></div>

<main class="wrap" id="scroller">
  <div class="boot" id="boot">Loading %d cases&hellip; if this message stays,
  the page failed to start and the reason will appear here.</div>
  <div class="cards" id="cards"></div>
</main>

<dialog id="dlg"><div class="dh"><h2>Verdicts</h2></div><div class="db">
  <div style="display:flex;gap:8px;align-items:center">
    <button class="btn" id="copy">Copy JSON</button>
    <span class="spacer"></span>
    <button class="btn" id="close">Close</button>
  </div>
  <textarea id="out" spellcheck="false" readonly></textarea>
  <p id="dlgsub" style="margin:0;color:var(--ink2);font-size:12.5px"></p>
  <div class="cmdline" style="font-family:var(--mono);font-size:11.5px;
    background:var(--sunk);border:1px solid var(--line);border-radius:5px;
    padding:7px 9px;color:var(--ink2);overflow-x:auto;white-space:pre">\
conda run -n vimBot python scripts/apply_curation.py \\
  --changes changes.json --dry-run</div>
  <p style="margin:0;color:var(--muted);font-size:12.5px">Verdicts and edits live
  in this browser only, and nothing here writes to the corpus.</p>
</div></dialog>

<script id="payload" type="application/json">%s</script>
<script>%s</script>
""" % (meta["title"], CSS, meta["lab"], meta["h1"], lede,
       n, meta["total_label"], flagged, meta["flag_label"], len(cats),
       unstated,
       ' style="color:var(--stop)"' if ambig else '', ambig,
       ' style="color:var(--stop)"' if mismatch else '', mismatch, n,
       n, cat_btns, meta["flag_button"], casey, unstated, ambig, lang_any,
       lang_weak, n, payload, js)


FIXES = REPO / "data" / "fixes"


def _load(name):
    path = FIXES / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def review_history():
    """id -> one line saying what review and repair the row has had.

    Read from the change-sets in data/fixes/, so the page reports what was
    actually applied rather than a hand-kept list.
    """
    verified = (_load("test_slice_verification_2026-09-03.json") or {}).get("review", {})
    reworded = set(_load("reword_restored_2026-09-04.json").get("edits", {}))
    reworded |= set(_load("reword_restored_2026-09-04b.json").get("edits", {}))
    reworded |= set(_load("reword_hc-0803_2026-09-04.json").get("edits", {}))
    case_clause = {r["id"] for r in
                   _load("case_clause_2026-09-05_audit.json").get("rows", [])}
    reverified = {rid for rid, v in _load("reverify_2026-09-05.json").get("review", {}).items()
                  if v.get("verdict") == "ok"}
    # the quota-fill rows that reached test by the command rule, read 2026-09-08
    late = (_load("test_slice_verification_2026-09-08.json") or {}).get("review", {})
    second = _load("restored_review_2026-09-04.json")
    second_verdicts = second.get("review", {})
    second_edits = set(second.get("edits", {}))

    def line(row):
        rid, parts = row["id"], []
        v = (verified.get(rid) or {}).get("verdict")
        if v == "drop":
            parts.append("dropped 2026-09-03")
        elif v:
            parts.append("verified 2026-09-03 (%s)" % v)
        if row.get("restore_edit", "none") != "none":
            parts.append("restored 2026-09-04, case clause added")
        elif row.get("restored_on"):
            parts.append("restored 2026-09-04 unchanged")
        if rid in reworded:
            parts.append("wording fixed")
        v2 = (second_verdicts.get(rid) or {}).get("verdict")
        if rid in second_edits:
            parts.append("reworded by owner and kept 2026-09-04")
        elif v2 in ("ok", "reword"):
            parts.append("re-verified 2026-09-04")
        if rid in case_clause:
            parts.append("case clause added 2026-09-05" + (
                " · re-verified 2026-09-05" if rid in reverified
                else " — NOT re-read since"))
        if rid in late:
            parts.append("verified 2026-09-08 (%s)"
                         % (late[rid].get("verdict") or "ok"))
        if not parts:
            # a row that reached the slice after the verification passes: it
            # carries no verdict at all, which the headline must not hide
            return "authored %s — NEVER REVIEWED" % (row.get("authored_on") or "later")
        return " · ".join(parts)

    return line


LEDE = """\
  <p class="lede">
    <b>%s</b>
    364 were verified on 2026-09-03; 51 were dropped that day, then restored on
    2026-09-04 with a case clause where the instruction was silent about case,
    reworded for natural phrasing, and re-reviewed &mdash; 45 kept, 6 dropped
    again. %s The <span class="mono">review</span> line on each card says what
    a row went through; the priority filter isolates the %d touched on
    2026-09-04.
  </p>
  <p class="lede" style="margin-top:10px">
    The machine gate was never the question here: vim reproduces every buffer,
    so the <i>command</i> is right. What no gate can check is whether the
    <i>instruction</i> honestly describes the edit &mdash; the defect the
    adversarial audits caught <b>76 times in 1,140</b> authored cases. This
    page stays live as the record and the working surface: verdicts and edits
    still export as a change-set <span class="mono">apply_curation.py</span>
    runs directly.
  </p>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    history = review_history()
    cases = collect(history=history,
                    flag=lambda r, risk: bool(r.get("restored_on")))
    if not cases:
        sys.exit("no rows in data/splits/test.jsonl")
    touched = sum(1 for c in cases if c["flagged"])
    stale = [c for c in cases
             if c["hist"] and ("NOT re-read" in c["hist"] or "NEVER REVIEWED" in c["hist"])]
    if stale:
        headline = ("%d of these %d rows carry a human verdict against the text "
                    "shown." % (len(cases) - len(stale), len(cases)))
        never = [c for c in stale if "NEVER REVIEWED" in c["hist"]]
        edited = [c for c in stale if c not in never]
        bits = []
        if never:
            bits.append("<b>%d joined the slice after the verification passes and "
                        "carry no verdict at all</b> (%s) — they were authored to fill "
                        "a quota cell and landed here because §7.3 keeps a gold command "
                        "inside one split, and theirs already lived in test"
                        % (len(never), ", ".join('<span class="mono">%s</span>' % c["id"]
                                                 for c in never)))
        if edited:
            bits.append("<b>%d were edited on 2026-09-05</b> (%s) after their verdict, "
                        "when the corrected case classifier found the wording still left "
                        "case open" % (len(edited), ", ".join(
                            '<span class="mono">%s</span>' % c["id"] for c in edited)))
        exception = "The exceptions: " + "; ".join(bits) + ". Those are the ones to read."
    else:
        headline = ("Every one of these %d rows carries a human verdict against "
                    "the text shown." % len(cases))
        exception = ("Three rows whose wording gained a case clause on 2026-09-05 "
                     "were re-read and approved the same day.")
    page = build(cases, {
        "lede": LEDE % (headline, exception, touched),
        "lab": "vimBot &middot; test slice &middot; human verification (&sect;7.3)",
        "h1": "Is the instruction telling the truth?",
        "total_label": "test cases",
        "flag_label": "restored and re-reviewed",
        "flag_button": "touched 2026-09-04",
    })
    check_page(page, required_js=["split('\\n')", "replace(/\\t/g"],
               name="test review")
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    bad = sum(1 for c in cases if not c["agrees"])
    flag = sum(1 for c in cases if c["flagged"])
    print("test rows: %d (%d touched 2026-09-04)   vim disagreements: %d   "
          "page: %d KB  -> %s"
          % (len(cases), flag, bad, len(page) // 1024, args.out))


if __name__ == "__main__":
    main()
