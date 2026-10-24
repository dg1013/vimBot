#!/usr/bin/env python3
"""Review page for the cliff's CONSISTENT-WRONG rows (DECISIONS.md 2026-09-15):
the train prompts the policy answers with one stable, executor-accepted
command that the gold disagrees with. A confident, stable answer is where a
gold command or an instruction deserves a read before it is taught, so each
row is shown three ways -- the buffer before, what the gold does to it, what
the policy's answer does to it -- beside the instruction, with Claude's
reading of the row and controls for the owner's verdict.

Inputs: the held-out rows `build_cliff_set.py build --hold-flagged` writes,
the probe file(s) whose `commands` say what the policy answered (the latest
row per id wins), and optionally a JSON of Claude's verdicts
(`{id: {verdict, reason, command?, mode?, instruction?}}`). Every command is
executed here, in the sandbox, so the page shows what vim did and never what
anyone assumed.

Verdict vocabulary, mapped onto `apply_curation.py`'s change-set:
    keep       the gold is right; teach it                 -> no change
    retarget   the policy's reading is right; make its
               command the gold (buffer_after recomputed)  -> edits[id] = {command, mode}
    reword     the instruction is wrong or ambiguous;
               a new instruction                           -> edits[id] = {instruction}
    drop       the row should not exist                    -> deletes[id] = reason
    unsure     leave it for later                          -> nothing

Verdicts persist in the artifact's database (`verdicts/<id>`), so they can be
read back without an export; the page also keeps a copy in localStorage and
offers the change-set as text.

Usage:
    python scripts/cliff_sft/build_flagged_review.py \\
        --rows runs/cliff_2026-09-15/flagged_review_qwen3.5-4b.jsonl \\
        --probe runs/grpo_gated_2026-09-14/difficulty_probe_qwen3.5-4b.jsonl \\
        --claude runs/cliff_2026-09-15/flagged_claude_verdicts_qwen3.5-4b.json \\
        --out runs/cliff_2026-09-15/flagged_review_qwen3.5-4b.html
    ... --dump           # print the rows as compact text (for reading them) and exit
    ... --data-out x.json   # write the assembled rows (with executed buffers) as JSON
"""
import argparse
import html
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from executor import run_vim                                   # noqa: E402
from evaluate import load_jsonl                                # noqa: E402
from page_guard import check_page                              # noqa: E402

VERDICTS = ("keep", "retarget", "reword", "drop", "unsure")


def probe_commands(paths):
    """id -> the probe row's `commands` list (latest file / row wins)."""
    out = {}
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if "commands" in row and row.get("id"):
                    out[row["id"]] = row
    return out


def policy_answer(probe_row, min_n=6):
    best = None
    for c in probe_row.get("commands") or []:
        if c.get("state") == "valid_wrong" and c.get("n", 0) >= min_n:
            if best is None or c["n"] > best["n"]:
                best = c
    return best


def execute(case, mode, command, timeout):
    r = run_vim(case["buffer_before"], command, mode=mode,
                cursor=tuple(case.get("cursor") or (1, 1)), timeout=timeout)
    return {"valid": bool(r["was_valid"]), "buffer": r["buffer_after"],
            "error": (r.get("stderr") or "")[:200], "blocked": bool(r.get("blocked"))}


def assemble(rows, probes, timeout):
    out = []
    for case in rows:
        p = probes.get(case["id"])
        ans = policy_answer(p) if p else None
        rec = {k: case.get(k) for k in ("id", "category", "tier", "mode", "source", "instruction",
                                        "buffer_before", "gold_command", "buffer_after", "cursor",
                                        "instruction_provenance", "source_test", "dataset")}
        rec["cursor"] = list(case.get("cursor") or [1, 1])
        rec["gold_exec"] = execute(case, case["mode"], case["gold_command"], timeout)
        if ans:
            rec["policy"] = {"mode": ans["mode"], "command": ans["command"], "n": ans["n"],
                             "G": p.get("G", 8), "others": [c for c in p.get("commands", []) if c is not ans]}
            rec["policy_exec"] = execute(case, ans["mode"], ans["command"], timeout)
        else:
            rec["policy"], rec["policy_exec"] = None, None
        out.append(rec)
    return out


# ------------------------------------------------------------------ dump

def show_buf(s, cursor=None, width=110):
    """One line per buffer line, newline shown as ⏎, cursor as ▮ before the
    character it sits on."""
    lines = s.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    out = []
    for i, line in enumerate(lines, 1):
        if cursor and cursor[0] == i:
            c = max(0, min(len(line), cursor[1] - 1))
            line = line[:c] + "▮" + line[c:]
        out.append("      %2d│%s" % (i, line[:width] + ("…" if len(line) > width else "")))
    return "\n".join(out) if out else "      (empty)"


def dump(recs, start=0, count=None):
    sel = recs[start:start + count] if count else recs[start:]
    for i, r in enumerate(sel, start + 1):
        pol = r["policy"]
        print("#%d %s [%s %s %s] cursor %s" % (i, r["id"], r["category"], r["tier"], r["source"], r["cursor"]))
        print("   INSTR: %s" % r["instruction"])
        print("   BEFORE:\n%s" % show_buf(r["buffer_before"], r["cursor"]))
        print("   GOLD  %s: %s  ->\n%s" % (r["mode"], r["gold_command"], show_buf(r["gold_exec"]["buffer"])))
        if pol:
            same = r["policy_exec"]["buffer"] == r["gold_exec"]["buffer"]
            print("   POLICY %s: %s  (%d/%d)%s ->\n%s" % (
                pol["mode"], pol["command"], pol["n"], pol["G"], "  [same buffer!]" if same else "",
                show_buf(r["policy_exec"]["buffer"]) if r["policy_exec"]["valid"] else "      (invalid: %s)" % r["policy_exec"]["error"]))
        print()


# ------------------------------------------------------------------ page

def build_page(recs, claude, meta):
    data_json = json.dumps({"rows": recs, "claude": claude, "meta": meta}, ensure_ascii=False)
    data_json = data_json.replace("</", "<\\/")
    tpl = (pathlib.Path(__file__).parent / "flagged_review_template.html").read_text(encoding="utf-8")
    page = tpl.replace("/*__DATA__*/null", data_json).replace("__TITLE__", html.escape(meta["title"]))
    check_page(page)
    return page


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", required=True, help="the held-out rows (JSONL)")
    ap.add_argument("--probe", nargs="+", required=True, help="probe file(s) with `commands`")
    ap.add_argument("--claude", default=None, help="Claude's verdicts JSON")
    ap.add_argument("--out", default=None, help="HTML page")
    ap.add_argument("--data-out", default=None, help="assembled rows as JSON")
    ap.add_argument("--dump", action="store_true", help="print the rows as text and exit")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--title", default="Cliff Disagreements")
    args = ap.parse_args()

    rows = load_jsonl(args.rows)
    probes = probe_commands(args.probe)
    recs = assemble(rows, probes, args.timeout)
    missing = [r["id"] for r in recs if r["policy"] is None]
    gold_bad = [r["id"] for r in recs if r["gold_exec"]["buffer"] != r["buffer_after"]]
    print("%d rows; %d without a probe answer; %d whose gold does not reproduce buffer_after" % (
        len(recs), len(missing), len(gold_bad)), file=sys.stderr)
    if args.data_out:
        pathlib.Path(args.data_out).write_text(json.dumps(recs, ensure_ascii=False, indent=0), encoding="utf-8")
    if args.dump:
        dump(recs, args.start, args.count)
        return 0
    if args.out:
        claude = json.loads(pathlib.Path(args.claude).read_text(encoding="utf-8")) if args.claude else {}
        meta = {"title": args.title, "rows": str(args.rows), "probe": [str(p) for p in args.probe],
                "n": len(recs), "claude_n": sum(1 for r in recs if r["id"] in claude)}
        page = build_page(recs, claude, meta)
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out).write_text(page, encoding="utf-8")
        print("page -> %s (%d KB)" % (args.out, len(page.encode("utf-8")) // 1024), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
