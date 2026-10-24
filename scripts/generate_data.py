"""Validate/ingest pipeline for vimBot synthetic data (CLAUDE.md §7.2).

Generation is split in two (DECISIONS.md 2026-08-11): candidate tuples are
authored by a frontier teacher (Claude, in-session) into JSONL (JSON Lines) files under
data/candidates/; this script is the deterministic half — it validates every
candidate through the sandboxed executor and keeps only tuples whose command
actually reproduces the claimed buffer AND changes the buffer.

Candidate schema (one scenario per line, 1-3 instruction paraphrases):
    category       str    taxonomy category
    tier           "T1" | "T2" | "T3"
    mode           "ex" | "keys"
    buffer_before  str
    cursor         [line, col]   optional, default [1, 1]
    command        str    the gold command
    buffer_after   str    the buffer the teacher claims the command produces
    instructions   [str]  natural-language paraphrases (each becomes a case)
    domain         str    optional buffer-content domain tag

Validation stages, each a reject_reason in data/quarantine.jsonl:
    parse              missing/ill-typed fields, multi-line command
    excluded_scope     frozen-scope screen (save/quit, macros, registers,
                       visual, multi-file, :normal, shell). Best-effort
                       static screens; the execution comparison below is the
                       real gate — anything that slips through and still
                       matches the claimed buffer did not save/quit/etc. in
                       any way that matters to the buffer contract.
    bad_keys_notation  a <...> token translate_keys does not support
    invalid            vim error or timeout
    mismatch           executed buffer != claimed buffer_after
    noop               buffer unchanged (reward-hacking guard)
    dup                exact (instruction, buffer) or near-dup
                       (instruction-token-hash, command) already kept

Keepers append to data/generated_cases.jsonl in the §5 schema plus additive
fields: tier, template_id, paraphrase_group, domain, teacher. Rejects keep
the executed buffer (buffer_executed) — Week-3 preference-pair negatives.

Resumable: on start the script rebuilds per-cell id counters, the dedup
index, and the set of already-processed candidate ids from the output files,
so re-running with the same candidate files is a no-op.

Usage:
    python scripts/generate_data.py --candidates data/candidates/*.jsonl
"""

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from executor import _KEYMAP, run_vim  # noqa: E402
from settings import load_settings  # noqa: E402
from scope import ex_fragments_from_keys, screen_dangerous  # noqa: E402

# Sourced from configs/settings.toml at import, so a settings change applies
# without editing code.
_GEN = load_settings()["generate"]

TEACHER = _GEN["teacher"]


# ---------------------------------------------------------------- normalizing

def norm_buffer(text: str) -> tuple[str, ...]:
    """Line-list form ignoring one trailing newline (matches evaluate.py)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return tuple(lines)


def norm_instruction(text: str) -> str:
    return " ".join(text.lower().split())


def instruction_token_hash(text: str) -> str:
    """Near-dup hash: order-insensitive normalized-token digest (§7.3)."""
    tokens = sorted(set(re.findall(r"[a-z0-9]+", text.lower())))
    return hashlib.sha1(" ".join(tokens).encode()).hexdigest()[:16]


def candidate_id(cand: dict) -> str:
    """Stable content id used for resumability."""
    payload = json.dumps(
        [
            cand.get("category"), cand.get("tier"), cand.get("mode"),
            cand.get("buffer_before"), cand.get("command"),
            sorted(cand.get("instructions") or []),
        ],
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


# ---------------------------------------------------------- scope screens (ex)

# Ex command names that violate the frozen v1 scope (§7.1). Matched against
# the command word after stripping ranges; abbreviations included.
_EX_FORBIDDEN = {
    # save / quit / persistence
    "w", "wr", "wri", "writ", "write", "wa", "wall", "wq", "wqa", "wqall",
    "x", "xit", "xa", "xall", "q", "qu", "qui", "quit", "qa", "qall",
    "exi", "exit", "up", "upd", "upda", "updat", "update", "sav", "saveas",
    # other files / buffers / windows / tabs
    "e", "ed", "edi", "edit", "ene", "enew", "fin", "find", "vi", "visual",
    "view", "n", "ne", "nex", "next", "prev", "previous", "fir", "first",
    "la", "last", "rew", "rewind", "b", "bu", "buf", "buffer", "bn", "bnext",
    "bp", "bprev", "bprevious", "bd", "bdelete", "ls", "buffers", "files",
    "ar", "arg", "args", "arga", "argadd", "argd", "argdelete",
    "sp", "split", "vs", "vsp", "vsplit", "new", "vnew",
    "tabe", "tabedit", "tabnew", "tabc", "tabclose",
    "r", "re", "rea", "read", "cd", "chd", "chdir", "lcd", "tcd",
    # scripting / :normal / shell
    "norm", "normal", "cal", "call", "exe", "exec", "execu", "execut",
    "execute", "so", "source", "ru", "runtime", "sh", "shell", "ter",
    "terminal", "mks", "mksession", "mkv", "mkview", "wv", "wviminfo",
    "rv", "rviminfo", "let", "unlet", "function", "while", "for", "if",
    "py", "python", "py3", "python3", "lua", "perl", "ruby",
}

# Addresses: line specs PLUS /pattern/ and ?pattern? search addresses
# (e.g. `.,/^=======/d` inside a :g — standard ex grammar).
_EX_RANGE = r"(?:[\s%$.,;+\-0-9']|/(?:[^/\\]|\\.)*/|\?(?:[^?\\]|\\.)*\?)*"
_EX_HEAD = re.compile(rf"^{_EX_RANGE}([a-zA-Z]+|[<>]+|[&~=#*@!])(!?)\s*(.*)$",
                      re.DOTALL)


def _screen_ex_segment(segment: str) -> str | None:
    """Return a violation description for one bar-separated ex segment."""
    segment = segment.strip()
    if not segment:
        return None
    m = _EX_HEAD.match(segment)
    if not m:
        return f"unparseable ex command: {segment!r}"
    word, _bang, rest = m.group(1), m.group(2), m.group(3)
    if word in ("!", "@", "*"):
        return f"forbidden ex command {word!r}"
    if word.isalpha() and word in _EX_FORBIDDEN:
        return f"forbidden ex command {word!r}"
    if word in ("g", "gl", "glo", "glob", "globa", "global",
                "v", "vg", "vgl", "vglo", "vglob", "vgloba", "vglobal"):
        # :g/pat/cmd — screen the trailing command recursively
        if not rest:
            return None
        delim = rest[0]
        i, n = 1, len(rest)
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


def screen_ex(command: str) -> str | None:
    """Return a violation description, or None if the ex command is in scope.

    Two layers: the shared security screen in src/scope.py (damage-capable
    commands, enforced by run_vim too), then this module's broader taxonomy
    policy (macros, scripting, quit, ... — out of scope but not dangerous).
    """
    why = screen_dangerous(command, "ex")
    if why:
        return why
    cmd = command.strip()
    if cmd.startswith(":"):
        cmd = cmd[1:]
    # split on unescaped bars: each side is its own command in vim
    segments = re.split(r"(?<!\\)\|", cmd)
    for seg in segments:
        why = _screen_ex_segment(seg)
        if why:
            return why
    return None


# -------------------------------------------------------- scope screens (keys)

# 2026-08-18: visual mode, visual block, named registers and macro RECORDING
# were removed from this list. They are covered scope now -- see
# data/command_coverage.md, which replaced the frozen-scope-v1 exclusion list.
# What remains is what is still out of scope, for reasons that are not
# taxonomy: macro playback executes register contents that may have come from
# the buffer rather than from the command, and ZZ/ZQ/Q/gQ save or open ex mode.
_KEYS_SCREENS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"@"), "macro playback (@)"),
    (re.compile(r"ZZ|ZQ"), "save/quit (ZZ/ZQ)"),
    (re.compile(r"^Q|^gQ"), "ex mode (Q/gQ)"),
]


def screen_keys(command: str) -> str | None:
    """Best-effort frozen-scope screen for a keys-mode command.

    Static screening cannot see mode context (a 'q' inside typed insert text
    is fine), so screens target normal-mode positions: start of command plus
    a few unambiguous substrings. Anything that slips through must still
    reproduce the claimed buffer to be kept.

    Ex fragments are screened via scope.ex_fragments_from_keys, which finds
    EVERY ':'...<CR> span rather than only a leading one. The old version
    checked just the first fragment of a command starting with ':', so
    'dd:set write<CR>:w! /path<CR>' passed unscreened.
    """
    why = screen_dangerous(command, "keys")
    if why:
        return why
    for pat, why in _KEYS_SCREENS:
        if pat.search(command):
            return why
    for frag in ex_fragments_from_keys(command):
        why = screen_ex(frag)
        if why:
            return why
    return None


_TOKEN = re.compile(r"<[^<>]*>")
_CTRL = re.compile(r"^<c-.>$")
_NAMED = {k for k in _KEYMAP}


def screen_keys_notation(command: str) -> str | None:
    """Every <...> token must be one translate_keys supports."""
    for tok in _TOKEN.findall(command):
        low = tok.lower()
        if low in _NAMED or _CTRL.match(low):
            continue
        return f"unsupported key token {tok!r}"
    return None


# ------------------------------------------------------------------- pipeline

REQUIRED_FIELDS = {
    "category": str, "tier": str, "mode": str, "buffer_before": str,
    "command": str, "buffer_after": str, "instructions": list,
}

MAX_BUFFER_LINES = _GEN["max_buffer_lines"]
MAX_COMMAND_LEN = _GEN["max_command_len"]


def parse_candidate(cand: dict, categories: dict) -> str | None:
    """Return a parse-violation description, or None."""
    for field, typ in REQUIRED_FIELDS.items():
        if field not in cand:
            return f"missing field {field!r}"
        if not isinstance(cand[field], typ):
            return f"field {field!r} is not {typ.__name__}"
    if cand["category"] not in categories:
        return f"unknown category {cand['category']!r}"
    if cand["tier"] not in ("T1", "T2", "T3"):
        return f"bad tier {cand['tier']!r}"
    if cand["mode"] not in ("ex", "keys"):
        return f"bad mode {cand['mode']!r}"
    if "\n" in cand["command"]:
        return "multi-line command"
    if not cand["command"].strip():
        return "empty command"
    if len(cand["command"]) > MAX_COMMAND_LEN:
        return "command too long"
    if not cand["buffer_before"].strip("\n"):
        return "empty buffer_before"
    if len(cand["buffer_before"].split("\n")) > MAX_BUFFER_LINES:
        return "buffer_before too long"
    if not cand["instructions"] or not all(
            isinstance(i, str) and i.strip() for i in cand["instructions"]
    ):
        return "instructions must be a non-empty list of non-empty strings"
    cursor = cand.get("cursor", [1, 1])
    if (
            not isinstance(cursor, list) or len(cursor) != 2
            or not all(isinstance(c, int) and c >= 1 for c in cursor)
    ):
        return f"bad cursor {cursor!r}"
    return None


class Corpus:
    """Kept-case index: dedup keys, per-cell counters, processed candidates."""

    def __init__(self) -> None:
        self.exact: set[tuple[str, tuple[str, ...]]] = set()
        self.near: set[tuple[str, str]] = set()
        self.next_seq: dict[tuple[str, str], int] = defaultdict(lambda: 1)
        self.kept_per_cell: dict[tuple[str, str], int] = defaultdict(int)
        self.processed: set[str] = set()

    def load_existing(self, out_path: Path, quarantine_path: Path) -> None:
        for row in iter_jsonl(out_path):
            self.register_case(row)
            if "source_candidate" in row:
                self.processed.add(row["source_candidate"])
            tid = row.get("template_id", "")
            m = re.match(r"^(.+)-(T\d)-(\d+)$", tid)
            if m:
                cell = (m.group(1), m.group(2))
                seq = int(m.group(3))
                self.next_seq[cell] = max(self.next_seq[cell], seq + 1)
        for row in iter_jsonl(quarantine_path):
            if "candidate_id" in row:
                self.processed.add(row["candidate_id"])

    def register_case(self, row: dict) -> None:
        self.exact.add(
            (norm_instruction(row["instruction"]), norm_buffer(row["buffer_before"]))
        )
        self.near.add(
            (instruction_token_hash(row["instruction"]), row["gold_command"].strip())
        )
        cell = (row["category"], row.get("tier", "T1"))
        self.kept_per_cell[cell] += 1

    def is_dup(self, instruction: str, buffer_before: str, command: str) -> bool:
        if (norm_instruction(instruction), norm_buffer(buffer_before)) in self.exact:
            return True
        if (instruction_token_hash(instruction), command.strip()) in self.near:
            return True
        return False


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                sys.exit(f"{path}:{lineno}: bad JSON ({exc})")


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")


def process_candidate(
        cand: dict,
        corpus: Corpus,
        categories: dict,
        out_path: Path,
        quarantine_path: Path,
        timeout: float,
        vim_bin: str,
) -> tuple[int, str | None]:
    """Validate one candidate scenario. Returns (n_kept, reject_reason)."""
    cid = cand.get("candidate_id") or candidate_id(cand)

    def reject(reason: str, detail: str, executed: str | None = None) -> tuple[int, str]:
        corpus.processed.add(cid)
        append_jsonl(quarantine_path, {
            "candidate_id": cid,
            "reject_reason": reason,
            "detail": detail,
            "buffer_executed": executed,
            **{k: cand.get(k) for k in (
                "category", "tier", "mode", "buffer_before", "cursor",
                "command", "buffer_after", "instructions", "domain",
            )},
        })
        return 0, reason

    why = parse_candidate(cand, categories)
    if why:
        return reject("parse", why)

    command, mode = cand["command"], cand["mode"]
    screen = screen_ex(command) if mode == "ex" else screen_keys(command)
    if screen:
        return reject("excluded_scope", screen)

    if mode == "keys":
        why = screen_keys_notation(command)
        if why:
            return reject("bad_keys_notation", why)

    cursor = tuple(cand.get("cursor", [1, 1]))
    result = run_vim(
        cand["buffer_before"], command, mode=mode, cursor=cursor,
        timeout=timeout, vim_bin=vim_bin,
    )
    executed = result["buffer_after"]
    if result["timeout"]:
        return reject("invalid", "timeout", executed)
    if executed is None:
        return reject("invalid", f"no buffer (rc={result['return_code']})", executed)
    if mode == "ex" and not result["was_valid"]:
        return reject("invalid", f"vim error (rc={result['return_code']})", executed)

    if norm_buffer(executed) != norm_buffer(cand["buffer_after"]):
        return reject(
            "mismatch",
            f"executed != claimed (executed {len(norm_buffer(executed))} lines)",
            executed,
        )
    if norm_buffer(executed) == norm_buffer(cand["buffer_before"]):
        return reject("noop", "buffer unchanged", executed)

    # canonical buffer_after: executed content, single trailing newline stripped
    canonical_after = "\n".join(norm_buffer(executed))

    fresh = [
        ins for ins in cand["instructions"]
        if not corpus.is_dup(ins, cand["buffer_before"], command)
    ]
    if not fresh:
        return reject("dup", "all instruction paraphrases duplicate kept cases")

    cell = (cand["category"], cand["tier"])
    seq = corpus.next_seq[cell]
    corpus.next_seq[cell] = seq + 1
    template_id = f"{cand['category']}-{cand['tier']}-{seq:03d}"

    kept = 0
    for idx, instruction in enumerate(fresh):
        row = {
            "id": f"{template_id}{chr(ord('a') + idx)}",
            "category": cand["category"],
            "tier": cand["tier"],
            "instruction": instruction,
            "mode": mode,
            "buffer_before": cand["buffer_before"],
            "gold_command": command,
            "buffer_after": canonical_after,
            "cursor": list(cursor),
            "template_id": template_id,
            "paraphrase_group": template_id,
            "domain": cand.get("domain", "unknown"),
            "teacher": TEACHER,
            "source_candidate": cid,
        }
        # provenance passthrough for imported corpora (additive §5 fields)
        for key in ("source", "source_corpus", "source_index",
                    "source_regex", "source_license", "source_theme"):
            if key in cand:
                row[key] = cand[key]
        append_jsonl(out_path, row)
        corpus.register_case(row)
        kept += 1
    # cell counter counts cases; register_case incremented per case already
    corpus.processed.add(cid)
    return kept, None


# ----------------------------------------------------------------------- main

def load_taxonomy(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    # Two-pass parse: --settings supplies the defaults for the other flags,
    # so it has to be resolved on its own first.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--settings", help="settings TOML "
                                        "(default: configs/settings.toml)")
    known, _ = pre.parse_known_args()
    _cfg = load_settings(known.settings)

    ap = argparse.ArgumentParser(description=__doc__, parents=[pre],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True,
                    help="candidate JSONL file(s) authored by the teacher")
    ap.add_argument("--taxonomy", default=str(REPO / "data" / "taxonomy.yaml"))
    ap.add_argument("--out", default=str(REPO / "data" / "generated_cases.jsonl"))
    ap.add_argument("--quarantine", default=str(REPO / "data" / "quarantine.jsonl"))
    ap.add_argument("--timeout", type=float, default=_cfg["executor"]["timeout"])
    ap.add_argument("--vim", default=_cfg["executor"]["vim_bin"])
    args = ap.parse_args()

    taxonomy = load_taxonomy(Path(args.taxonomy))
    categories = taxonomy["categories"]
    out_path, quarantine_path = Path(args.out), Path(args.quarantine)

    corpus = Corpus()
    corpus.load_existing(out_path, quarantine_path)

    n_candidates = n_skipped = n_kept_cases = n_rejected = 0
    reject_counts: dict[str, int] = defaultdict(int)

    for cand_file in args.candidates:
        for cand in iter_jsonl(Path(cand_file)):
            n_candidates += 1
            cid = cand.get("candidate_id") or candidate_id(cand)
            if cid in corpus.processed:
                n_skipped += 1
                continue
            kept, reason = process_candidate(
                cand, corpus, categories, out_path, quarantine_path,
                args.timeout, args.vim,
            )
            if reason:
                n_rejected += 1
                reject_counts[reason] += 1
            else:
                n_kept_cases += kept

    print(f"candidates     {n_candidates}")
    print(f"  skipped      {n_skipped} (already processed)")
    print(f"  rejected     {n_rejected}")
    for reason in sorted(reject_counts):
        print(f"    {reason:<18} {reject_counts[reason]}")
    print(f"  kept cases   {n_kept_cases} (paraphrases expanded)")

    print(f"\n  {'cell':<24} {'kept':>5} {'quota':>6} {'fill':>6}")
    for cat in categories:
        quotas = categories[cat].get("quotas", {})
        for tier in ("T1", "T2", "T3"):
            kept = corpus.kept_per_cell.get((cat, tier), 0)
            quota = quotas.get(tier, 0)
            fill = f"{100 * kept / quota:.0f}%" if quota else "-"
            print(f"  {cat + '-' + tier:<24} {kept:>5} {quota:>6} {fill:>6}")


if __name__ == "__main__":
    main()
