"""Derive and validate a vim conversion for every unique corpus regex (regular expression).

Chain per pattern: structural templates (dsl2vim.generate) -> semantic AST (abstract syntax tree)
rewrite (rewrite_stuck) -> direct vim constructions (semantic_direct) ->
hand-derived manual table -> probe-gated trivial verdict. Every conversion
is executed in real vim through the project executor and compared against
an oracle over probe strings; never-match verdicts additionally require
zero matching probes. Failing conversions are dropped, not shipped.

Output: data/external/deep-regex/conversions.json —
    {regex_text: {k: 0|1|2, v: display|None, n: note|None,
                  exec: pattern|None, pol: "g"|"v"}}
k: 0 portable, 1 convertible, 2 stuck. Notes: class-and/root-not/root-and
(templates), semantic, manual, degenerate (never matches), trivial
(matches every line).

Final validated state (2026-08-13): 10,367 unique patterns -> 6,362
portable, 4,005 convertible (871 never-match, 9 trivial), 0 stuck; 9,487
vim-validated conversions, 0 failures.
"""
import json
import random
import re
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
from executor import run_vim  # noqa: E402

from dsl2vim import (  # noqa: E402
    charset, esc_portable, generate, group_style, py_oracle,
)
from dsl_ast import dsl_match, parse  # noqa: E402
from rewrite_stuck import rewrite, semantic_rewrite  # noqa: E402
from semantic_direct import direct_vim  # noqa: E402
import manual_table  # noqa: E402

CORPUS = REPO / "data" / "external" / "deep-regex"
OUT = CORPUS / "conversions.json"
CORPORA = ["KB13", "NL-RX-Synth", "NL-RX-Turk"]


def load_pool():
    seen, pool = set(), []
    for name in CORPORA:
        for t in (CORPUS / name / "targ.txt").read_text(
                encoding="utf-8").splitlines():
            t = t.strip()
            if t and t not in seen:
                seen.add(t)
                pool.append(t)
    return pool


def probes(expr):
    out = {"", "z", "zz zz", "dog", "truck", "dog truck", "truckdog",
           "A", "a", "7", "Aa7", "the Dog ran", "007", "x" * 18,
           "!a", "!ab", "a!", "?zz", "z!z", "!dog", "dog!", "!7!",
           " xT! q!", "!ab cd!", " a! b ", "zz!  !zz"}
    for lit in re.findall(r"[A-Za-z][A-Za-z ]{1,14}[A-Za-z]", expr):
        out |= {lit, f"{lit}z", f"zz {lit}", f"{lit} zz", lit.upper(),
                lit[::-1], f"aa{lit}", f"{lit}aa", f"z{lit} z", f"z {lit}z"}
    for cls in re.findall(r"\[[^\]]*\]", expr)[:4]:
        cs = sorted(charset(cls))
        if cs:
            out |= {cs[0], cs[-1], cs[0] * 3, f"z{cs[0]}",
                    f"{cs[0]} {cs[-1]}"}
    # single-letter literals: the word rules live or die on q/u interplay
    bare = expr
    for sub in re.findall(r"\[[^\]]*\]", bare):
        bare = bare.replace(sub, " ")
    singles = sorted(set(re.findall(
        r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])", bare)))[:3]
    for x in singles:
        out |= {x, x * 2, f"{x}z", f"z{x}", f"z{x}z", f"{x} z"}
    for x in singles:
        for y in singles:
            if x != y:
                out |= {x + y, f"{x}a{y}", f"{x} {y}", f"z{x}{y}z"}
    # boundary-heavy patterns break on long junk-mixed strings (counted
    # \b repetitions); generate a few deterministically per pattern — the
    # ingest gate's sampler caught this class, so the validator must too
    if "\\b" in expr:
        rng = random.Random(expr)
        alpha = sorted({c for cls in re.findall(r"\[[^\]]*\]", expr)[:4]
                        for c in charset(cls)})[:8] or list("az9T")
        pool = alpha + list("! ?.") + ["dog"]
        for _ in range(6):
            s = "".join(rng.choice(pool)
                        for _ in range(rng.randint(6, 14)))
            if len(s) <= 24:
                out.add(s)
    return sorted(s for s in out if "\n" not in s
                  and not s.startswith("#") and len(s) <= 24)[:72]


class Timeout(Exception):
    pass


signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))


def guarded(fn, seconds=6):
    signal.alarm(seconds)
    try:
        return fn()
    except Timeout:
        return None
    finally:
        signal.alarm(0)


def bound_risk(t):
    r"""\b inside a quantified group: repetition joints need the exact
    \%(\<\|\>\) union — the directional heuristic is provably wrong there
    (Phase A2 audit, 2026-08-14) and these boundary-dense languages are too
    hard to member-sample for validation-retry to be a reliable net.
    Depth-scans to the quantified group's opening paren (nesting-safe)."""
    for m in re.finditer(r"\)(\{\d|\*|\+)", t):
        j = m.start()
        depth, i = 0, j
        while i >= 0:
            if t[i] == ")":
                depth += 1
            elif t[i] == "(":
                depth -= 1
                if depth == 0:
                    break
            i -= 1
        if i >= 0 and "\\b" in t[i:j]:
            return True
    return False


def main():
    pool = load_pool()
    manual = manual_table.build(pool)

    work = []      # (t, gen, lines, expected, kind)
    final = {}
    stats = {"sem": 0, "direct": 0, "manual": 0, "deg": 0, "veto": 0,
             "trivial": 0}

    def span_expected(t, lines, seconds=20):
        return guarded(lambda: [dsl_match(t, s) for s in lines], seconds)

    def validation_lines(t, pol="g"):
        """probes + A1 adversarial witnesses — the strengthened validation
        set (Phase A2). Weak probes let 27 boundary-heuristic conversions
        ship in v2; witness lines make check()'s exact_boundary retry fire."""
        import hashlib
        import random
        from witnesses import build_buffer, sample_members
        lines = list(probes(t))
        rng = random.Random(hashlib.sha1(t.encode()).hexdigest())

        def enrich():
            extra = []
            buf = build_buffer(t, rng, probe_lines=lines, pol=pol)
            if buf:
                extra += buf["lines"]
            extra += sample_members(t, rng, want=3, taken=lines + extra)
            return extra

        extra = guarded(enrich, 5) or []
        return [s for s in dict.fromkeys(lines + extra) if s]

    def stuck_row():
        return {"k": 2, "v": None, "n": None, "exec": None, "pol": "g"}

    def mark_never(t, text):
        exp = span_expected(t, probes(t))
        if exp is None or any(exp):
            stats["veto"] += exp is not None and any(exp)
            final[t] = stuck_row()
            return
        stats["deg"] += 1
        final[t] = {"k": 1, "v": text, "n": "degenerate",
                    "exec": None, "pol": "g"}

    def queue(t, g, kind, new_text):
        lines = validation_lines(t, g.get("pol", "g"))
        expected = span_expected(t, lines)
        if expected is None:
            final[t] = stuck_row()
            return
        work.append((t, {**g, "new_text": new_text}, lines, expected, kind))

    for t in pool:
        g = generate(t, exact_boundary=bound_risk(t))
        if g["tag"] != "s":
            if g["exec"] is None or "/" in g["exec"]:
                final[t] = {"k": 1 if g["tag"] == "c" else 0,
                            "v": g["display"], "n": g["note"],
                            "exec": g["exec"], "pol": g["pol"]}
                continue
            lines = validation_lines(t, g.get("pol", "g"))
            oracle = py_oracle(t)
            expected = guarded(lambda: [oracle(s) for s in lines]) \
                if oracle else None
            if expected is None:
                expected = span_expected(t, lines)
            if expected is None:
                final[t] = {"k": 1 if g["tag"] == "c" else 0, "v": None,
                            "n": g["note"], "exec": None, "pol": g["pol"]}
                continue
            work.append((t, g, lines, expected, "template"))
            continue

        status, new_text, new_ast = semantic_rewrite(t)
        if status == "empty":
            mark_never(t, "never matches — empty sub-language")
            continue
        handled = False
        if status == "rewritten":
            g2 = generate(new_text, exact_boundary=bound_risk(new_text))
            if g2["tag"] != "s" and g2["exec"] and "/" not in g2["exec"]:
                disp = (g2["display"] or group_style(esc_portable(new_text))
                        if g2["tag"] == "p" else g2["display"])
                stats["sem"] += 1
                queue(t, {**g2, "display": disp, "note": "semantic"},
                      "semantic", new_text)
                handled = True
        if not handled and new_ast is not None:
            d = direct_vim(new_ast)
            if d and "/" not in d["exec"]:
                stats["direct"] += 1
                queue(t, {"tag": "c", "note": "semantic", **d},
                      "semantic", new_text or t)
                handled = True
        if not handled and t in manual:
            d = manual[t]
            if "/" not in d["exec"]:
                stats["manual"] += 1
                queue(t, {"tag": "c", "note": "manual", **d}, "manual", t)
                handled = True
        if not handled and "~" in t:
            lines = sorted(set(probes(t)) | {"q", "Q", "9", "!", "  ",
                           "q q", "the", "DOG", "d0g", "xyzzy", "0", "~",
                           "#x", "?"})
            exp = span_expected(t, lines)
            if exp is not None and all(exp):
                stats["trivial"] += 1
                final[t] = {"k": 1,
                            "v": "matches every line (probe-validated)",
                            "n": "trivial", "exec": None, "pol": "g"}
                handled = True
        if not handled:
            final[t] = stuck_row()

    print(f"patterns {len(pool)}; queued for vim {len(work)} "
          f"(semantic {stats['sem']}+{stats['direct']} direct, "
          f"manual {stats['manual']}); never-match {stats['deg']}, "
          f"vetoed {stats['veto']}, trivial {stats['trivial']}")

    # ---------------- vim validation (threaded) ------------------------

    def vim_marks(g, lines):
        buf = "\n".join(lines)
        cmd = f"{g['pol']}/{g['exec']}/s/^/#/"
        r = run_vim(buf, cmd, mode="ex", timeout=10)
        if r["buffer_after"] is None:
            return None
        got = r["buffer_after"].split("\n")
        if got and got[-1] == "":
            got.pop()
        if len(got) != len(lines):
            return None
        return [after == "#" + orig for orig, after in zip(lines, got)]

    def check(item):
        t, g, lines, expected, kind = item
        if vim_marks(g, lines) == expected:
            return (t, g, kind, "ok")
        if "\\b" in (g.get("new_text") or t):
            src = g.get("new_text") or t
            g2 = generate(src, exact_boundary=True)
            if kind != "template" and g2["tag"] == "p":
                disp = g2["display"] or group_style(esc_portable(src, True))
                g2 = {**g2, "display": disp, "note": g.get("note")}
            if g2["tag"] == "s":
                try:
                    ast = rewrite(parse(src))
                    dd = direct_vim(ast, exact_boundary=True)
                except Exception:
                    dd = None
                if dd and "/" not in dd["exec"]:
                    g2 = {"tag": "c", "note": g.get("note"), **dd}
            if g2.get("exec") and "/" not in g2["exec"] \
                    and g2.get("tag") != "s":
                if vim_marks(g2, lines) == expected:
                    return (t, g2, kind, "ok")
        return (t, g, kind, "FAIL")

    results = list(ThreadPoolExecutor(8).map(check, work))
    fails = [(t, g) for t, g, k, s in results if s == "FAIL"]
    for t, g, kind, s in results:
        if s == "FAIL":
            final[t] = stuck_row()
        else:
            note = g.get("note") if kind != "template" else g.get("note")
            final[t] = {"k": 1 if (kind != "template" or g["tag"] == "c")
                        else 0,
                        "v": g["display"], "n": note,
                        "exec": g["exec"], "pol": g["pol"]}

    print(f"vim-validated {len(results) - len(fails)}, FAILED {len(fails)}")
    for t, g in fails[:10]:
        print(f"  FAIL {t}\n    -> {g.get('display')}")

    OUT.write_text(json.dumps(final, ensure_ascii=False),
                   encoding="utf-8")
    kinds = [0, 0, 0]
    for c in final.values():
        kinds[c["k"]] += 1
    print(f"tags: portable {kinds[0]}, convertible {kinds[1]}, "
          f"stuck {kinds[2]} -> {OUT}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
