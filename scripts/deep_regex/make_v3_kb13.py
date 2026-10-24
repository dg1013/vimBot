"""Dataset v3 core: KB13 (the Kushman and Barzilay 2013 corpus) human-authored scenarios only (owner decision
2026-08-14 — the instruction-provenance analysis measured KB13 at 99.1%
screen approval vs ~26% for the machine-seeded tiers, so v3 is built on
the human-authored slice and grows via further vetted-license imports).

Pipeline per scenario: KB13 NL(s) screened (garble + anchor rules) →
pattern normalized/deduped per the A3 simplifier (merge-per-class) →
command from the A2-verified conversions → A1 witness buffer → executed
in sandboxed vim and cross-checked against the span oracle AND the
unrestricted Python implementation (the full verifier triangle, per case).

Output: data/v3/kb13_cases.jsonl (+ a skip report on stdout). Does not
touch the live v2 files; split rebuild happens after the import round.
"""
import collections
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))

from executor import run_vim  # noqa: E402
from build_conversions import CORPUS, probes, bound_risk  # noqa: E402
from dsl2vim import generate  # noqa: E402
from garble_screen import anchor_mismatch, classify  # noqa: E402
from make_candidates import instruction_for, tier_for  # noqa: E402
from py_impl import py_matcher  # noqa: E402
from simplify_ast import normal_form  # noqa: E402
from witnesses import build_buffer  # noqa: E402


def norm_buf(buf):
    return buf[:-1].split("\n") if buf.endswith("\n") else buf.split("\n")


def main():
    conv = json.loads((CORPUS / "conversions.json").read_text())
    srcs = (CORPUS / "KB13" / "src.txt").read_text(encoding="utf-8").splitlines()
    targs = (CORPUS / "KB13" / "targ.txt").read_text(encoding="utf-8").splitlines()

    by_regex = collections.defaultdict(lambda: {"nls": [], "idx": []})
    for i, (s, t) in enumerate(zip(srcs, targs), 1):
        s, t = s.strip(), t.strip()
        if s and t:
            e = by_regex[t]
            if s not in e["nls"]:
                e["nls"].append(s)
            e["idx"].append(i)

    # merge per A3 normal-form equivalence class (owner dedup decision)
    classes = collections.defaultdict(lambda: {"nls": [], "idx": [],
                                               "members": []})
    skip = collections.Counter()
    for t, e in by_regex.items():
        c = conv.get(t)
        if not c or not isinstance(c, dict) or c.get("k") == 2 \
                or c.get("n") in ("degenerate", "trivial") or not c.get("exec"):
            skip["no-usable-conversion"] += 1
            continue
        try:
            nf, _ = normal_form(t)
        except Exception:
            nf = t
        cl = classes[nf]
        cl["members"].append(t)
        cl["idx"] += e["idx"]
        for nl in e["nls"]:
            if nl not in cl["nls"]:
                cl["nls"].append(nl)

    rows, stats = [], collections.Counter()
    n_scen = 0
    for nf in sorted(classes):
        cl = classes[nf]
        h = hashlib.sha1(nf.encode()).hexdigest()
        rng = random.Random(h)
        op = "delete" if int(h, 16) % 2 == 0 else "keep"

        # screened human instructions (the whole point of v3)
        instrs = []
        for nl in cl["nls"]:
            cand = instruction_for(nl, op)
            if classify(nf, cand) or anchor_mismatch(nf, cand):
                stats["instruction-screened-out"] += 1
                continue
            instrs.append(cand)
        if not instrs:
            skip["all-instructions-screened"] += 1
            continue

        c = conv.get(nf)
        if not c or not c.get("exec"):
            g = generate(nf, exact_boundary=bound_risk(nf))
            if g.get("exec") and "/" not in g["exec"]:
                c = {"exec": g["exec"], "pol": g["pol"]}
            else:
                c = next((conv[m] for m in cl["members"]
                          if conv.get(m, {}).get("exec")), None)
        if not c:
            skip["no-class-conversion"] += 1
            continue
        pol = c.get("pol", "g")

        buf = build_buffer(nf, rng, probe_lines=probes(nf), pol=pol)
        if buf is None:
            skip["no-buffer"] += 1
            continue
        lines, truth = buf["lines"], buf["truth"]
        if op == "delete":
            cmd = f"g/{c['exec']}/d" if pol == "g" else f"v/{c['exec']}/d"
            keep_members = False
        else:
            cmd = f"v/{c['exec']}/d" if pol == "g" else f"g/{c['exec']}/d"
            keep_members = True
        expected = [ln for ln in lines if truth[ln] == keep_members]

        res = run_vim("\n".join(lines), cmd, mode="ex")
        if not res["was_valid"] or norm_buf(res["buffer_after"]) != expected:
            # a freshly-generated nf conversion is single-shot and unaudited;
            # fall back to a member's A2-audited conversion before skipping
            fb = next((conv[m] for m in cl["members"]
                       if conv.get(m, {}).get("exec")), None)
            if fb and fb.get("exec") != c.get("exec"):
                pol = fb.get("pol", "g")
                if op == "delete":
                    cmd = f"g/{fb['exec']}/d" if pol == "g" else f"v/{fb['exec']}/d"
                else:
                    cmd = f"v/{fb['exec']}/d" if pol == "g" else f"g/{fb['exec']}/d"
                res = run_vim("\n".join(lines), cmd, mode="ex")
            if not res["was_valid"] or norm_buf(res["buffer_after"]) != expected:
                skip["vim-mismatch"] += 1
                continue
        f = py_matcher(nf)
        if f is not None:
            py = [ln for ln in lines if f(ln) == keep_members]
            if py != expected:
                skip["python-mismatch"] += 1
                continue
            stats["triangle-full"] += 1
        else:
            stats["triangle-two-leg"] += 1
        if not expected or len(expected) == len(lines):
            skip["degenerate-result"] += 1
            continue

        n_scen += 1
        wk = sorted({lab for lab in buf["labels"].values()
                     if lab not in ("probe", "member", "generic")})
        for j, instr in enumerate(instrs[:3]):
            rows.append({
                "id": f"kb13-{n_scen:04d}{chr(ord('a') + j)}",
                "category": "global", "tier": tier_for(nf), "mode": "ex",
                "domain": "probe", "buffer_before": "\n".join(lines),
                # §5 contract key — was `command` until 2026-08-17, which made
                # src/evaluate.py raise KeyError on this file (see DECISIONS.md)
                "cursor": [1, 1], "gold_command": cmd,
                "buffer_after": "\n".join(expected),
                "instruction": instr,
                "source": "deep-regex-kb13", "source_corpus": ["KB13"],
                "source_index": [f"KB13:{i}" for i in sorted(set(cl["idx"]))[:6]],
                "source_regex": cl["members"][0], "normal_form": nf,
                "class_members": len(cl["members"]),
                "source_license": "MIT", "instruction_provenance": "human",
                "buffer_witnesses": wk,
            })

    out = REPO / "data" / "v3"
    out.mkdir(exist_ok=True)
    path = out / "kb13_cases.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"KB13 regexes {len(by_regex)}; classes {len(classes)}; "
          f"scenarios kept {n_scen}; cases written {len(rows)} -> {path}")
    print("verifier:", dict(stats))
    print("skips:", dict(skip))


if __name__ == "__main__":
    main()
