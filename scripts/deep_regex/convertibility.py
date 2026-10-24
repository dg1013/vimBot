r"""Text-level helpers shared across the deep-regex conversion tooling.

The corpus DSL (domain-specific language) uses & (language intersection) and ~ (complement) on top of
ordinary regex (regular expression) syntax; these helpers navigate that text without a full parse
(the real parser lives in dsl_ast.py).
"""


def strip_outer(e):
    """Remove parens that enclose the entire expression."""
    while len(e) >= 2 and e[0] == "(":
        depth, enclosed, i = 0, False, 0
        while i < len(e):
            c = e[i]
            if c == "\\":
                i += 2
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    enclosed = i == len(e) - 1
                    break
            i += 1
        if not enclosed:
            return e
        e = e[1:-1]
    return e


def split_top(e, op):
    """Split at depth-0 occurrences of op, honoring [] and backslash."""
    parts, depth, buf, i, in_class = [], 0, [], 0, False
    while i < len(e):
        c = e[i]
        if c == "\\" and i + 1 < len(e):
            buf.append(e[i:i + 2])
            i += 2
            continue
        if in_class:
            buf.append(c)
            if c == "]":
                in_class = False
            i += 1
            continue
        if c == "[":
            in_class = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if c == op and depth == 0 and not in_class:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    parts.append("".join(buf))
    return parts


def has_ops(e):
    """True if e contains a DSL op (& or ~) outside a character class."""
    i, in_class = 0, False
    while i < len(e):
        c = e[i]
        if c == "\\":
            i += 2
            continue
        if in_class:
            if c == "]":
                in_class = False
        elif c == "[":
            in_class = True
        elif c in "&~":
            return True
        i += 1
    return False
