r"""Feature-bucket assignment for split stratification.

Extracts the regex (regular expression) pattern(s) from a gold command and assigns exactly ONE
bucket via a priority ladder, so structurally rare case kinds (capture
groups, no-pattern keys cases, alternation, ...) can be dealt evenly across
train/val/test instead of vanishing into one split by shuffle luck.

Ladder (highest present wins):
    no-pattern > capture > alternation > counted-rep > boundary >
    negclass > class > quantifier > literal
"""
import re

_EX_PATTERN = re.compile(
    r"(?:^|[%\d,.$ ;+\-]|\bg|\bv|\bs|sort )/((?:[^/\\]|\\.)*)")
_KEYS_PATTERN = re.compile(r"/((?:[^/\\<]|\\.)*?)<CR>")

BUCKETS = ("no-pattern", "capture", "alternation", "counted-rep",
           "boundary", "negclass", "class", "quantifier", "literal")


def extract_patterns(cmd, mode):
    """Regex pattern bodies inside a gold command (may be empty)."""
    if mode == "ex":
        c = cmd.lstrip(":")
        return [m.group(1) for m in _EX_PATTERN.finditer(c) if m.group(1)]
    return [m.group(1) for m in _KEYS_PATTERN.finditer(cmd) if m.group(1)]


def feature_bucket(cmd, mode):
    pats = extract_patterns(cmd, mode)
    if not pats:
        return "no-pattern"
    p = " ".join(pats)
    # capture means USED captures: a \( group referenced by a backref
    # anywhere in the command (e.g. the :s replacement). \%( is
    # non-capturing and grouping without a backref is incidental — the
    # imported g// commands wrap everything in groups.
    if re.search(r"\\[1-9]", cmd) and "\\(" in p:
        return "capture"
    if "\\|" in p:
        return "alternation"
    if "\\{" in p:
        return "counted-rep"
    if "\\<" in p or "\\>" in p or "\\b" in p:
        return "boundary"
    if "[^" in p:
        return "negclass"
    if re.search(r"(?<!\\)\[", p) or re.search(r"\\[dwaslu]", p):
        return "class"
    if re.search(r"(?<!\\)\*", p) or "\\+" in p or "\\=" in p:
        return "quantifier"
    return "literal"


def case_bucket(case):
    """Bucket for a §5 dataset row."""
    return feature_bucket(case["gold_command"], case.get("mode", "ex"))
