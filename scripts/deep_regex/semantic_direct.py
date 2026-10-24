r"""Direct vim constructions for root shapes with no DSL (domain-specific language) closed form.

R1  \b ~(X) \b  as the whole pattern: boundaries at the string edges mean
    first and last char are word chars; the complement rides a negative
    lookahead:      ^\%(\%(X\)$\)\@!\w\%(.*\w\)\=$
    One-sided variants keep only that side's \w anchor.

R2  root alternation with negated branches:  ~(X) | Y
    ->  ^\%(\%(\%(X\)$\)\@!\|\%(Y\)$\)
    (a zero-width branch is a fine :g match)

X and Y must be op-free after the semantic rewrite; both paths are probe-
validated in vim like every other conversion.
"""
from convertibility import has_ops
from dsl2vim import esc_portable, group_style
from dsl_ast import serialize


def _op_free_vim(nd, exact_boundary=False):
    txt = serialize(nd)
    if has_ops(txt):
        return None
    return group_style(esc_portable(txt, exact_boundary))


def direct_vim(nd, exact_boundary=False):
    """-> dict(display, exec, pol) or None. nd is the rewritten root AST (abstract syntax tree)."""
    if nd is None or not getattr(nd, "kind", None):
        return None
    kids = list(nd.kids) if nd.kind == "seq" else [nd]

    # R1: [\b] ~(X) [\b]
    pre = post = False
    core = list(kids)
    while core and core[0].kind == "bound":
        pre = True
        core.pop(0)
    while core and core[-1].kind == "bound":
        post = True
        core.pop()
    if len(core) == 1 and core[0].kind == "not":
        vx = _op_free_vim(core[0].kids[0], exact_boundary)
        if vx is not None:
            look = f"\\%(\\%({vx}\\)$\\)\\@!"
            if pre and post:
                body = "\\w\\%(.*\\w\\)\\=$"
            elif pre:
                body = "\\w.*$"
            elif post:
                body = ".*\\w$"
            else:
                body = ".*$"
            ex = "^" + look + body
            return {"display": ex, "exec": ex, "pol": "g"}

    # R2: root alternation with ~ branches
    if nd.kind == "alt":
        branches = []
        for b in nd.kids:
            bk = list(b.kids) if b.kind == "seq" else [b]
            if len(bk) == 1 and bk[0].kind == "not":
                vx = _op_free_vim(bk[0].kids[0], exact_boundary)
                if vx is None:
                    return None
                branches.append(f"\\%(\\%({vx}\\)$\\)\\@!")
            else:
                vb = _op_free_vim(b, exact_boundary)
                if vb is None:
                    return None
                branches.append(f"\\%({vb}\\)$")
        ex = "^\\%(" + "\\|".join(branches) + "\\)"
        return {"display": ex, "exec": ex, "pol": "g"}

    return None
