#!/usr/bin/env python3
"""Build the review-queue artifact page from the live splits.

Takes the repo-tracked page shell (data/review_page_base.html — curated
review tabs + UI (user interface), no split data) and injects one entry per case from
data/splits/{train,val,test}.jsonl into three tabs. Groups are the static
garble screen's verdict (flagged/clean) so the corpus-quarantine decision
can be browsed directly; metas carry source pattern, feature bucket, fired
rules, and test-slice status markers. Tab counts are computed client-side,
so a rebuild after any dataset change stays consistent.

Usage (repo root):
    python scripts/build_review_page.py --out /path/to/page.html
Then publish the output over the existing artifact URL (uniform resource locator).
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "scripts" / "deep_regex"))
from pattern_features import case_bucket
from garble_screen import classify, anchor_mismatch, class_sequence

VERDICTS_PATH = REPO / "data" / "audit_2026-08-14_verdicts.json"
PROBES_PATH = REPO / "data" / "probe_results_2026-08-14.json"
VERDICTS = json.loads(VERDICTS_PATH.read_text()) if VERDICTS_PATH.exists() else {}
PROBES = ({p["id"]: p for p in json.loads(PROBES_PATH.read_text())}
          if PROBES_PATH.exists() else {})

# flagged-half borderlines still in test, owner verdict pending (2026-08-14)
PENDING_BORDERLINE = {
    "global-T2-1475a": "TS-017", "global-T2-1658a": "TS-041",
    "global-T3-4010a": "TS-140", "global-T3-4582a": "TS-190"}

PREFIX = {"train": ("TR", 5), "val": ("VA", 4), "test": ("TE", 3)}

# fixed priority for the single filter chip a flagged case files under
RULE_ORDER = ["R1-pos-conj", "R2a-neg-posconj", "R2b-neg-disjunct",
              "R2c-embedded-neg-literal", "R2d-neg-anchored-alt",
              "R2e-negclass-word-conj", "R3-open-star", "R3-zero-or-more",
              "R4-dot-disjunct", "R5-instr-character",
              "R6-start-mismatch", "R6-end-mismatch"]


def entry(tab, n, row):
    pre, width = PREFIX[tab]
    rules = sorted(classify(row.get("source_regex"), row["instruction"]))
    kind = ("delete matching" if row["gold_command"].startswith("g/")
            else "keep matching")
    meta = (f"source pattern: {row.get('source_regex', '—')} · "
            f"bucket: {case_bucket(row)} · "
            f"garble: {', '.join(rules) if rules else 'none'}")
    if row.get("backfill_round"):
        meta += f" · backfilled {row['backfill_round']}"
    if row["id"] in PENDING_BORDERLINE:
        meta += (f" · borderline {PENDING_BORDERLINE[row['id']]} — owner "
                 f"verdict pending")
    elif tab == "test" and row.get("needs_review"):
        meta += " · needs_review"
    # coarse error-class grouping (2026-08-14 full audit); specifics stay in
    # meta + badges, which are searchable
    verdict = VERDICTS.get(row["id"])
    anchor = anchor_mismatch(row.get("source_regex"), row["instruction"])
    if anchor:
        blind = PROBES.get(row["id"], {}).get("anchor_blind")
        anchor_group = "anchor-latent" if blind in (True, None) else "anchor-visible"
        meta_note = f"anchor-mismatch ({'latent' if blind in (True, None) else 'visible on this buffer'})"
    if verdict:
        group = ("wrong (audit)" if verdict["verdict"] == "wrong"
                 else verdict["verdict"])
        meta = meta + f" · audit: {verdict['reason']}"
    elif rules:
        group = "garble"
    elif anchor:
        group = anchor_group
        meta = meta + " · " + meta_note
    elif class_sequence(row.get("source_regex")):
        group = "seq-pending"
        meta = meta + " · class-sequence family (SEQ verdict pending)"
    else:
        group = "clean"
    primary = next((r for r in RULE_ORDER if r in rules), None)
    badges = [row["tier"], kind, row["id"]]
    if primary:
        badges.append(primary)
    return {
        "ref": f"{pre}-{n:0{width}d}", "tab": tab,
        "group": group,
        "badges": badges,
        "instructions": [row["instruction"]],
        "command": row["gold_command"], "cursor": None,
        "before": row["buffer_before"], "after": row["buffer_after"],
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=str(REPO / "data" / "review_page_base.html"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    page = Path(args.base).read_text(encoding="utf-8")

    # idempotence: strip any previously injected split entries
    dec = json.JSONDecoder()
    for tab in ("train", "val", "test"):
        while True:
            m = re.search(r'\{"ref": "(?:TR|VA|TE)-\d+"', page)
            if not m:
                break
            _, end = dec.raw_decode(page, m.start())
            start = m.start()
            if page[start - 2:start] == ", ":
                start -= 2
            page = page[:start] + page[end:]

    entries = []
    for tab in ("train", "val", "test"):
        rows = [json.loads(l) for l in
                open(REPO / "data" / "splits" / f"{tab}.jsonl", encoding="utf-8")
                if l.strip()]
        for n, row in enumerate(rows, 1):
            entries.append(entry(tab, n, row))

    # Fixes tab: before->after cards from per-phase manifests (v3 plan)
    fixes_dir = REPO / "data" / "fixes"
    n_fx = 0
    if fixes_dir.is_dir():
        for mf in sorted(fixes_dir.glob("*.json")):
            for f in json.loads(mf.read_text()):
                n_fx += 1
                ref = f"FX-{n_fx:03d}"
                case = f.get("case", {})
                if f["kind"] == "removal":
                    entries.append({
                        "ref": ref, "tab": "fixes", "group": f["phase"],
                        "badges": ["removal", case.get("tier", ""), f["id"]],
                        "instructions": [case.get("instruction", "")],
                        "command": case.get("gold_command", ""), "cursor": None,
                        "before": case.get("buffer_before", ""),
                        "after": case.get("buffer_after", ""),
                        "meta": f"APPLIED: removed to quarantine_review · {f['summary']}",
                    })
                else:
                    old, new = f.get("old", {}), f.get("new", {})
                    entries.append({
                        "ref": ref, "tab": "fixes", "group": f["phase"],
                        "badges": [f["kind"], f["id"]],
                        "instructions": [case.get("instruction", "")],
                        "command": new.get("command", ""), "cursor": None,
                        "before": (f"OLD command: {old.get('command', '')}\n\n"
                                   f"old result:\n{old.get('buffer_after', '')}"),
                        "after": (f"NEW command: {new.get('command', '')}\n\n"
                                  f"new result (vim-verified):\n{new.get('buffer_after', '')}"),
                        "meta": f"APPLIED: {f['summary']}",
                    })

    blob = ", " + ", ".join(json.dumps(e) for e in entries)
    blob = blob.replace("</", "<\\/")          # never form a </script> in data
    end = page.index("}];")
    page = page[:end + 1] + blob + page[end + 1:]

    size = len(page.encode("utf-8"))
    assert size < 15_000_000, f"page too large: {size} bytes"
    Path(args.out).write_text(page, encoding="utf-8")
    flagged = sum(1 for e in entries if e["group"] != "clean")
    print(f"{len(entries)} split entries injected ({flagged} flagged); "
          f"page {size / 1e6:.1f} MB -> {args.out}")


if __name__ == "__main__":
    main()
