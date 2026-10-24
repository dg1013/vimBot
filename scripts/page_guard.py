#!/usr/bin/env python3
"""Build-time guards for the generated artifact pages.

Three silent-failure modes have now shipped from this repo (repository), each producing a
blank page with no error in the browser:

  1. a non-raw Python template ate a JS escape, so `/\\n/` became a regex (regular expression)
     literal split across two lines           (Instruction Bench, 2026-08-19)
  2. `\\b` became 0x08, so a leak check matched backspace characters instead
     of word boundaries — wrong, and invisible
  3. a double-quoted JS string contained double quotes:
     "</span><span class="chip">" terminates at `class=` and leaves a bare
     identifier                                     (Corpus Curator, 2026-08-25)

None of these are catchable by looking at the HTML (HyperText Markup Language) — the markup is valid and
balanced in every case. The only reliable check is to parse the JavaScript,
so that is what this does. `esprima` (pure Python, ES2017) is used when it is
importable and skipped with a warning when it is not, so the guard hardens
the build without becoming a hard dependency of it.

    from page_guard import check_page
    check_page(html, required_js=[...])   # raises SystemExit on failure
"""
import re
import sys

# control characters that only appear when \b, \t, \f, \v or \0 were eaten by
# a non-raw template
FORBIDDEN = {chr(c) for c in list(range(0, 9)) + [11, 12] + list(range(14, 32))}


def check_page(page, required_js=(), name="page"):
    """Raise SystemExit unless the page's inline JS is parseable and clean."""
    for frag in required_js:
        if frag not in page:
            sys.exit("%s guard: %r missing from the output — is the template "
                     "still a raw string? Python ate a JS escape." % (name, frag))

    bad = sorted({c for c in page if c in FORBIDDEN})
    if bad:
        sys.exit("%s guard: control characters in output: %r — a JS escape was "
                 "interpreted by Python." % (name, bad))

    blocks = re.findall(r"<script>(.*?)</script>", page, re.S)
    if not blocks:
        return "no inline script to parse"
    try:
        import esprima
    except ImportError:
        return ("esprima not installed — JS NOT PARSE-CHECKED. "
                "pip install esprima to enable the guard that catches "
                "blank-page syntax errors.")
    for i, js in enumerate(blocks):
        try:
            esprima.parseScript(js, tolerant=False)
        except Exception as exc:                       # noqa: BLE001
            msg = str(exc)
            m = re.search(r"Line (\d+)", msg)
            where = ""
            if m:
                n = int(m.group(1))
                lines = js.split("\n")
                where = "\n" + "\n".join(
                    "%s%4d | %s" % (">>" if k == n - 1 else "  ", k + 1, lines[k][:150])
                    for k in range(max(0, n - 3), min(len(lines), n + 2)))
            sys.exit("%s guard: inline script %d does not parse: %s%s\n\n"
                     "The page would render blank with no error. Common cause: "
                     "a double-quoted JS string containing double quotes, e.g. "
                     '"</span><span class=\\"x\\">".' % (name, i, msg, where))
    return "%d inline script(s) parsed clean" % len(blocks)
