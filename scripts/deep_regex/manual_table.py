r"""Hand-derived vim conversions for the residual pattern families.

Consulted only for patterns the rules leave stuck; every entry is still
probe-validated in vim before it may ship (a failing entry is dropped and
the pattern stays stuck — honest over convenient).

  BCHAIN  (\b~(X)\b){k,} / *
          Chains of boundary positions whose segments avoid X. Any string
          with word-char edges can take itself as one segment, so the
          language is word-edged strings minus X's own unsplittable
          members; {k,} pads with ε-units at any boundary, * adds the
          empty line. Both alternation branches stay ^-anchored (an
          unanchored second branch once matched mid-line and was caught by
          validation).

  BALT    \b(A)|(~(B))\b — precedence splits this into (\b A) | (~(B) \b):
          a lone A-word, or any line that is not B and ends in a word char.

  REPALT  ((A)|(~(B)))rep — the ε-admitting ~(B) chunk makes every count
          reachable; concatenations of "anything but a single B" recover
          everything except B's single characters not rescued by A.
          ~(.) excludes ALL single characters (dot = any char, not a
          literal dot).
"""
import re

from dsl2vim import charset as text_charset
from dsl2vim import esc_portable, set_to_class

W = "[A-Za-z]"
_CLS = r"\[[^\]]*\]|\.|[a-z]+"


def _esc(x):
    return esc_portable(x)


def _excluded_class(b_text, a_text):
    """singles(B) - singles(A) as a class, '.' for all, None for none."""
    if a_text == ".":
        return None                 # the dot branch rescues every single char
    if b_text == ".":
        aset = text_charset(a_text) if a_text.startswith("[") else set()
        return "." if not aset else None
    bset = text_charset(b_text) if b_text.startswith("[") else (
        set(b_text) if len(b_text) == 1 else set())
    if not bset:
        return None
    aset = set()
    if a_text.startswith("["):
        aset = text_charset(a_text)
    elif len(a_text) == 1:
        aset = set(a_text)
    ex = bset - aset
    return set_to_class(ex) if ex else None


def build(patterns):
    """-> {pattern_text: {display, exec, pol}} for every family match."""
    out = {}
    for t in patterns:
        # BCHAIN
        m = re.fullmatch(r"\(\\b~\((\[[^\]]*\]|\.|[a-z]+)\)\\b\)"
                         r"(\*|\+|\{\d+,\})", t)
        if m:
            x, rep = m.groups()
            vx = "." if x == "." else _esc(x)
            core = f"^\\%(\\%({vx}\\)$\\)\\@!\\w\\%(.*\\w\\)\\=$"
            exec_ = f"{core}\\|^$" if rep == "*" else core
            out[t] = {"display": exec_, "exec": exec_, "pol": "g"}
            continue

        # BALT: \b(A)|(~(B))\b
        m = re.fullmatch(rf"\\b\(({_CLS})\)\|\(~\(({_CLS})\)\)\\b", t)
        if m:
            a, b = m.groups()
            va = "." if a == "." else _esc(a)
            vb = "." if b == "." else _esc(b)
            exec_ = (f"^\\%(\\%({va}\\)$\\|"
                     f"\\%(\\%({vb}\\)$\\)\\@!.*\\w$\\)")
            out[t] = {"display": exec_, "exec": exec_, "pol": "g"}
            continue

        # BALT flipped: \b(~(B))|(A)\b -> (\b ~(B)) | (A \b)
        m = re.fullmatch(rf"\\b\(~\(({_CLS})\)\)\|\(({_CLS})\)\\b", t)
        if m:
            b, a = m.groups()
            va = "." if a == "." else _esc(a)
            vb = "." if b == "." else _esc(b)
            exec_ = (f"^\\%(\\%(\\%({vb}\\)$\\)\\@!\\w.*$\\|"
                     f"\\%({va}\\)$\\)")
            out[t] = {"display": exec_, "exec": exec_, "pol": "g"}
            continue

        # REPALT, both operand orders
        ab = re.fullmatch(rf"\(\(({_CLS})\)\|\(~\(({_CLS})\)\)\)"
                          r"(\*|\+|\{\d+,\})", t)
        ba = re.fullmatch(rf"\(\(~\(({_CLS})\)\)\|\(({_CLS})\)\)"
                          r"(\*|\+|\{\d+,\})", t)
        if ab or ba:
            if ab:
                a_text, b_text = ab.group(1), ab.group(2)
            else:
                b_text, a_text = ba.group(1), ba.group(2)
            ex = _excluded_class(b_text, a_text)
            if ex is None:
                exec_ = "^.*$"
            elif ex == ".":
                exec_ = "^\\%(.$\\)\\@!.*$"
            else:
                exec_ = f"^\\%(\\%({_esc(ex)}\\)$\\)\\@!.*$"
            out[t] = {"display": exec_, "exec": exec_, "pol": "g"}
            continue

    # ---- explicit specials --------------------------------------------
    specials = {
        # two ordered chains interleaved: wi<sh and st<ar in either weave
        ".*((.*wi.*sh.*)&(.*st.*ar.*)).*":
            "^\\%(" + "\\|".join(
                ".*" + ".*".join(runs) + ".*" for runs in [
                    ("wi", "sh", "st", "ar"), ("wi", "st", "sh", "ar"),
                    ("wi", "st", "ar", "sh"), ("st", "wi", "sh", "ar"),
                    ("st", "wi", "ar", "sh"), ("st", "ar", "wi", "sh"),
                ]) + "\\)$",
        # a word that starts with s and ends with t
        ".*((\\bs[A-Za-z]*\\b)&(\\b[A-Za-z]*t\\b)).*":
            f"^.*\\<s{W}*t\\>.*$",
        # a word containing both an upper- and a lower-case letter
        ".*\\b(([A-Za-z]*[A-Z][A-Za-z]*)&([A-Za-z]*[a-z][A-Za-z]*))\\b.*":
            (f"^.*\\<\\%({W}*[A-Z]{W}*[a-z]{W}*"
             f"\\|{W}*[a-z]{W}*[A-Z]{W}*\\)\\>.*$"),
        # word1 (span containing a word and a digit) word3, junk-edged
        "[^A-Za-z]*\\b[A-Za-z]+\\b((.*\\b[A-Za-z]+\\b.*)&(.*[0-9].*))"
        "\\b[A-Za-z]+\\b[^A-Za-z]*":
            (f"^[^A-Za-z]*\\<{W}\\+\\>"
             f"\\%(.*\\<{W}\\+\\>.*[0-9].*\\|.*[0-9].*\\<{W}\\+\\>.*\\)"
             f"\\<{W}\\+\\>[^A-Za-z]*$"),
        # digits only allowed after ABC and XYZ have both appeared
        "([^0-9]*)|((([^0-9]*)&(.*ABC.*)&(.*XYZ.*)).*)":
            ("^\\%([^0-9]*$\\|\\%([^0-9]*ABC[^0-9]*XYZ[^0-9]*"
             "\\|[^0-9]*XYZ[^0-9]*ABC[^0-9]*\\).*\\)$"),
        # up to five junk-edged words of ≥4 letters containing an 'a'
        "([^A-Za-z]*((\\b[A-Za-z]+\\b)&([A-Za-z]{4,})&(.*a.*))"
        "[^A-Za-z]*){0,5}":
            (f"^$\\|^\\%([^A-Za-z]*\\<\\%({W}\\{{4,}}\\>\\)\\@="
             f"{W}*a{W}*\\>[^A-Za-z]*\\)\\{{1,5}}$"),
    }
    for t, exec_ in specials.items():
        if t in patterns:
            out[t] = {"display": exec_, "exec": exec_, "pol": "g"}

    # every word ends in er — :v of "contains a word not ending er"
    t54 = "~(.*((\\b[A-Za-z]+\\b)&(~(.*er))).*)"
    if t54 in patterns:
        inner = f".*\\<\\%({W}*er\\>\\)\\@!{W}\\+\\>.*"
        out[t54] = {"display": f":v/^\\%({inner}\\)$/  (polarity flip)",
                    "exec": f"^\\%({inner}\\)$", "pol": "v"}
    return out
