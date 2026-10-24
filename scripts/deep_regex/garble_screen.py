"""Class-garble screen for deep-regex-derived cases.

Flags scenarios whose source DSL (domain-specific language) pattern uses operators that template into
unusable English: token-level &-conjunction over class atoms, negations of
position-level expressions, vacuous group-stars, bare-dot alternation
branches, plus two instruction-side signals ("a character" as an operand
noun; line-start/end phrasing over .*-slack patterns). Line-level
(.*-wrapped) and word-shape (\\b) operators template fine and are exempt.

Validated 2026-08-14 against the full manual review of the original
400-case test slice (data/review_sweep_2026-08-14.md): precision 1.00,
recall 0.92 on 382 labeled cases (18 borderlines excluded). The 9 known
misses are containment-vs-anchor and count-semantics mismatches — those are
covered by the execution-probe screen, not this static one.

Run from the repo (repository) root: `python scripts/deep_regex/garble_screen.py`
prints the validation confusion matrix and per-rule prevalence on every
split. The hand-label sets below are the frozen ground-truth record."""
import json, re, collections
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])

FLAGGED_BAD = {
    "global-T2-121a", "global-T2-121b", "global-T2-1418a", "global-T2-1501a",
    "global-T2-1501b", "global-T2-175a", "global-T2-175b", "global-T2-1803a",
    "global-T2-1821a", "global-T2-1822a", "global-T2-409a", "global-T2-409b",
    "global-T2-460a", "global-T3-1109a", "global-T3-1329a", "global-T3-1336a",
    "global-T3-3114a", "global-T3-3530a", "global-T3-500a", "global-T3-692a"}
UNFLAGGED_BAD = {f"global-T3-{s}" for s in (
    "089a", "089b", "096a", "096b", "1622a", "1622b", "2383a", "3004a",
    "3847a", "3847b", "3901a", "3901b", "4084a", "4084b", "4283a", "4283b",
    "5576a", "5978a", "5978b", "912a", "912b",
    "1102a", "1107a", "1331a", "2143a", "2143b", "2601a", "2602a", "2747a",
    "2853a", "2854a", "3003a", "3117a", "3136a", "3325a", "3325b", "3351a",
    "3351b", "3505a", "3506a", "3507a", "3509a", "3554a", "3555a", "495a",
    "495b", "499a", "4610a", "4610b", "4678a", "4678b", "5567a", "5567b",
    "691a", "782a", "782b", "906a")}
BORDERLINE = {"global-T2-1475a", "global-T2-1658a", "global-T3-4010a",
              "global-T3-4582a", "global-T3-3529a"} | {f"global-T3-{s}" for s in (
    "1148a", "1148b", "1333a", "1435a", "1435b", "2125a", "2125b",
    "2147a", "2147b", "483a", "483b", "582a", "582b")}

CLS, NEG, WB = "Ⓒ", "Ⓝ", "⋄"   # Ⓒ Ⓝ ⋄ — cannot collide with ASCII sources

def strip_classes(src):
    s = re.sub(r"\[\^[^\]]*\]", NEG, src)
    s = re.sub(r"\[[^\]]*\]", CLS, s)
    s = s.replace("\\b", WB)
    return s

def unwrap(b):
    b = b.strip()
    while b.startswith("(") and b.endswith(")"):
        depth = 0
        for i, c in enumerate(b):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i != len(b) - 1:
                    return b          # outer parens are not a single wrap
        b = b[1:-1].strip()
    return b

def groups_with(s, ch):
    """Minimal parenthesized groups containing ch at their own top level."""
    out = []
    for i, c in enumerate(s):
        if c != "(":
            continue
        depth, j = 0, i
        while j < len(s):
            if s[j] == "(": depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0: break
            j += 1
        g = s[i + 1:j]
        # ch at top level of g?
        d = 0; found = False
        for c2 in g:
            if c2 == "(": d += 1
            elif c2 == ")": d -= 1
            elif c2 == ch and d == 0: found = True
        if found:
            out.append(g)
    return out

def top_split(s, op):
    parts, depth, cur = [], 0, ""
    for c in s:
        if c == "(": depth += 1
        elif c == ")": depth -= 1
        if c == op and depth == 0:
            parts.append(cur); cur = ""
        else:
            cur += c
    parts.append(cur)
    return parts

def has_lit(x):
    return bool(re.search(r"[A-Za-z0-9]", x.replace(CLS, "").replace(NEG, "").replace(WB, "")))

def single_atomish(o):
    """Operand is a single token atom (class, negclass, dot, short literal)
    or an alternation of such — the token-level shape that garbles."""
    o = unwrap(o)
    branches = top_split(o, "|")
    if len(branches) > 1:
        return all(single_atomish(b) for b in branches)
    return bool(re.fullmatch("(" + CLS + "|" + NEG + r"|\.|[A-Za-z0-9]{1,8})", unwrap(o)))

def classify(src, instr):
    rules = set()
    s = strip_classes(src or "")
    il = (instr or "").lower()

    # R1: token-level & — some operand is a single atom (or alternation of
    # atoms); line-level .*-conj and word-shape conj are fine
    candidates = groups_with(s, "&")
    if len(top_split(s, "&")) > 1:
        candidates.append(s)                 # top-level conjunction
    for g in candidates:
        ops = top_split(g, "&")
        if any(WB in o for o in ops):       # word-shape conj: usually fine...
            # ...unless an operand is a bare negated-class star ([^I]* etc.)
            if any(re.fullmatch("\\(?" + NEG + "\\*\\)?", o.strip()) for o in ops):
                rules.add("R2e-negclass-word-conj")
            continue
        ops = [o for o in ops if not re.fullmatch("\\(?" + CLS + "\\*\\)?", o.strip())]  # drop match-empty
        if len(ops) < 2:
            continue
        if any(single_atomish(o) for o in ops):
            rules.add("R1-pos-conj")

    # R2a: negation containing a position-level & ; R2b: ~ as alternation branch
    for m in re.finditer(r"~\(", s):
        depth, j = 0, m.start() + 1
        while j < len(s):
            if s[j] == "(": depth += 1
            elif s[j] == ")":
                depth -= 1
                if depth == 0: break
            j += 1
        body = s[m.start() + 2:j]
        if "&" in body and ".*" not in body:
            rules.add("R2a-neg-posconj")
        bparts = top_split(body, "|")
        if len(bparts) > 1:                  # ~(A|B) with position-anchored branch
            for b in bparts:
                bs = b.strip().strip("()")
                if not (bs.startswith(".*") and bs.endswith(".*")):
                    rules.add("R2d-neg-anchored-alt")
        if ".*" not in body and "&" not in body and has_lit(body):
            # negated bare literal embedded in a sequence (dropped-negation risk)
            before, after = s[:m.start()], s[j + 1:]
            if (".*" in before or ".*" in after) and WB not in s:
                rules.add("R2c-embedded-neg-literal")
    parts = top_split(s, "|")
    if len(parts) > 1:
        for part in parts:
            if part.strip().lstrip("(").startswith("~"):
                rules.add("R2b-neg-disjunct")

    # R3: star groups whose content is open-ended (contains . or C), unless
    # the standard negated-class idiom (N) or pure literal alternation
    for g in groups_with(s + " ", "\x00") or []:
        pass
    for m in re.finditer(r"\)\*", s):
        # find the group this star closes
        depth, i = 0, m.start()
        while i >= 0:
            if s[i] == ")": depth += 1
            elif s[i] == "(":
                depth -= 1
                if depth == 0: break
            i -= 1
        content = s[i + 1:m.start()]
        inner = content.replace(NEG, "")
        if "." in inner or CLS in inner:
            rules.add("R3-open-star")
    if "zero or more" in il or "0 or more" in il:
        rules.add("R3-zero-or-more")

    # R4: bare-dot alternation branch (grouped or top-level)
    for g in groups_with(s, "|") + ([s] if len(parts) > 1 else []):
        for o in top_split(g, "|"):
            if o.strip("()") == ".":
                rules.add("R4-dot-disjunct")

    # R5: 'character' as a noun operand in relational phrasing
    if (re.search(r"\b(a|any) characters?\b", il)
            or re.search(r"\b(before|after|then|followed by|preceding) characters?\b", il)) \
            and re.search(r"followed by|before|after|then|preceding|as well as|\band\b|plus|a ?long with|\bor\b", il):
        rules.add("R5-instr-character")

    # R6: start/end phrasing vs .*-slack pattern edge (line-level only —
    # instructions about words keep their own start/end semantics)
    if not re.search(r"\bword|string|sequence|phrase\b", il):
        if re.search(r"\b(start|begin)(s|ing)?\s+(with|in)\b", il) and re.match(r"^\(*\.\*", s):
            rules.add("R6-start-mismatch")
        if re.search(r"\bend(s|ing)?\s+(with|in|containing)\b", il) and re.search(r"\.\*\)*$", s):
            rules.add("R6-end-mismatch")
    return rules

CONTAIN_WORDS = re.compile(
    r"\b(with|contain\w*|hav(e|ing)|has|us(e|ing)|feature|mention\w*|include\w*)\b", re.I)
ANCHOR_DISCLOSERS = re.compile(
    r"\b(start\w*|begin\w*|end\w*|finish\w*|(?<!keep )only|composed|comprised"
    r"|consist\w*|exactly|entire|whole|are|letter lines?|\d+ letter)\b", re.I)

def anchor_mismatch(src, instr):
    """A-class screen (2026-08-14 full audit): pattern is anchored on at
    least one side, instruction phrases containment without disclosing the
    anchor. Reported separately from classify() — folding it in is an owner
    decision (it overlaps the pending class-sequence verdict)."""
    if not src or not instr:
        return False
    s = strip_classes(src)
    if s.startswith("~"):
        return False                      # negations invert anchor semantics
    left = bool(re.match(r"^\(*(\.\*|\.\{)", s))
    right = bool(re.search(r"(\.\*|\})\)*\*?\+?$", s)) or s.rstrip(")").endswith(".*")
    if left and right:
        return False                      # slack both sides: containment is true
    return bool(CONTAIN_WORDS.search(instr)) and not ANCHOR_DISCLOSERS.search(instr)


def class_sequence(src):
    """Pure class-sequence family (VA-0094's kind): only class/dot atoms and
    slack, >=2 atoms, no literals/word-boundaries/operators. Subject of the
    pending SEQ-01..30 owner verdict (adjacency ambiguity)."""
    if not src:
        return False
    s = strip_classes(src)
    if any(ch in s for ch in "&|~{"):
        return False
    if re.search(r"[A-Za-z0-9]", s.replace(CLS, "").replace(NEG, "")):
        return False
    if WB in s:
        return False
    return len(re.findall("(" + CLS + "|" + NEG + r"|(?<!\.)\.(?!\*))", s)) >= 2


def main():
    """Validation against the frozen 2026-08-14 hand labels + prevalence.

    NOTE: the validation reconstructs the original 400-case slice from
    the review quarantine + the live test split, so its numbers are only
    exact while every removed case still carries an owner_review_* reason
    in data/quarantine_review.jsonl. The recorded reference numbers
    (precision 1.00 / recall 0.92) live in DECISIONS.md."""
    rows400 = []
    for line in open(f"{REPO}/data/quarantine_review.jsonl"):
        r = json.loads(line)
        if "owner_review" in r.get("reject_reason", ""):
            rows400.append(r)
    quarantined = {r["id"] for r in rows400}
    rows400 += [json.loads(l) for l in open(f"{REPO}/data/splits/test.jsonl")]
    BAD = quarantined | FLAGGED_BAD | UNFLAGGED_BAD

    tp = fp = fn = tn = 0
    fp_d, fn_d = [], []
    for r in rows400:
        if r["id"] in BORDERLINE:
            continue
        got = classify(r.get("source_regex"), r["instruction"])
        truth = r["id"] in BAD
        if got and truth: tp += 1
        elif got and not truth: fp += 1; fp_d.append((r["id"], sorted(got), r.get("source_regex", "")[:60]))
        elif not got and truth: fn += 1; fn_d.append((r["id"], r.get("source_regex", "")[:80], r["instruction"][:60]))
        else: tn += 1
    print(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}   precision {tp/(tp+fp):.2f}  recall {tp/(tp+fn):.2f}")
    print("\nFP detail:"); [print("  ", *x) for x in fp_d]
    print("\nFN detail:"); [print("  ", *x) for x in fn_d]

    for split in ("train", "val"):
        rows = [json.loads(l) for l in open(f"{REPO}/data/splits/{split}.jsonl")]
        hits = collections.Counter(); flagged = 0
        for r in rows:
            rs = classify(r.get("source_regex"), r["instruction"])
            if rs:
                flagged += 1
                for x in rs: hits[x] += 1
        print(f"\n{split}: {flagged}/{len(rows)} ({100*flagged/len(rows):.1f}%)  " +
              "  ".join(f"{k}:{v}" for k, v in hits.most_common()))


if __name__ == "__main__":
    main()
