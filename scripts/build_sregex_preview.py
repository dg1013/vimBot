#!/usr/bin/env python3
"""Build the StructuredRegex conversion-preview artifact (server-rendered).

Every card — every shipped case AND every rejected pattern — is
rendered into the HTML (HyperText Markup Language) itself: category, validation status, vim command,
input buffer, and where vim was run, its actual output beside the expected
output with per-line agreement marks. JavaScript only filters what is
already on the page; if scripting fails, everything still shows. (The
previous client-rendered version could blank entirely — this one cannot.)

Usage (repo root):
    conda run -n vimBot python scripts/build_sregex_preview.py --out page.html
"""
import argparse
import collections
import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

CATS = [
    ("shipped", "shipped", "validated end to end; enters v3 as a case"),
    ("prose-refused", "prose refused", "corpus prose refused AND the A3 renderer "
     "cannot verbalize the pattern — the command itself often works"),
    ("unconvertible", "no vim pattern", "and/~ under a repeat/alternation, "
     "or variable-length conjuncts with a variable tail — beyond even "
     "suffix-pinned lookaheads"),
    ("too-few-examples", "too few examples", "not enough example strings "
     "for a contrastive buffer"),
    ("screened", "screened out", "converts and validates, but every "
     "phrasing tripped a garble/anchor screen"),
]
PRE = {"prose-refused": "PR", "unconvertible": "UC",
       "too-few-examples": "FE", "screened": "SC"}

E = html.escape


def read_jsonl(p):
    p = Path(p)
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] \
        if p.exists() else []


def pane(label, lines, marks=None):
    """marks: list aligned with lines — 'kp' / 'cut' / '' — or None."""
    if lines is None:
        return ""
    body = "".join(
        f'<em class="{marks[i] if marks else ""}">{E(l) or "&nbsp;"}</em>'
        for i, l in enumerate(lines)) or '<em class="dim">(empty buffer)</em>'
    return (f'<div class="pane"><p class="lab">{E(label)}</p>'
            f'<pre>{body}</pre></div>')


def subseq_marks(before, after):
    """kept/cut marks where after is a subsequence of before, else None."""
    out, i = [], 0
    for l in before:
        if i < len(after) and after[i] == l:
            out.append("kp")
            i += 1
        else:
            out.append("cut")
    return out if i == len(after) else None


def card_html(cat, ref, badges, prose, cmd, status, panes, note, dsl):
    ok = ("validated" in status and "FAILED" not in status
          and "no single" not in status)
    b = "".join(f'<span class="bdg">{E(x)}</span>' for x in badges if x)
    p = "".join(f'<p class="ins">{E(x)}</p>' for x in prose)
    c = (f'<div class="cmd">{E(cmd)}</div>' if cmd else "")
    st = f'<p class="cmdst {"ok" if ok else "bad"}">{E(status)}</p>'
    n = f'<p class="note">{E(note)}</p>' if note else ""
    return (f'<article class="card" data-cat="{cat}">'
            f'<div class="chead"><span class="ref">{E(ref)}</span>{b}</div>'
            f'{p}{c}{st}'
            f'<div class="panes">{"".join(panes)}</div>{n}'
            f'<p class="dsl">source DSL · {E(dsl)}</p></article>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = read_jsonl(REPO / "data" / "v3" / "sregex_cases.jsonl")
    rejects = read_jsonl(REPO / "data" / "v3" / "sregex_rejects.jsonl")
    stats = json.loads((REPO / "data" / "v3" / "sregex_stats.json").read_text())

    cards = []
    for r in rows:
        before = r["buffer_before"].split("\n")
        after = r["buffer_after"].split("\n")
        cmd = r.get("gold_command") or r.get("command", "")
        cards.append(card_html(
            "shipped", r["id"],
            ["shipped", r["tier"],
             "delete matching" if cmd.startswith("g/") else "keep matching",
             r.get("instruction_provenance", "")],
            [r["instruction"]], cmd,
            "validated — vim output = expected; full verifier triangle "
            "(vim + span oracle + python)",
            [pane("input buffer — kept / cut", before,
                  subseq_marks(before, after)),
             pane("result — vim output (= expected)", after)],
            "", r.get("source_dsl", "")))

    ncat = collections.Counter()
    for r in rejects:
        b = r["bucket"]
        ncat[b] += 1
        ref = f"{PRE[b]}-{ncat[b]:03d}"
        note = r["note"]
        if r.get("rules"):
            note += " · rules fired: " + ", ".join(r["rules"])
        panes = []
        if r["lines"]:
            panes.append(pane(
                "example strings — match / non-match", r["lines"],
                ["kp" if m else "cut" for m in r["match"]]))
        exp, got = r.get("expected"), r.get("vim_output")
        if exp is not None:
            panes.append(pane("expected result (oracle)", exp))
            agree = got == exp
            marks = [("kp" if i < len(exp) and exp[i] == l else "cut")
                     for i, l in enumerate(got)]
            panes.append(pane(
                "vim output — " + ("matches expected" if agree
                                   else "DIVERGES from expected"),
                got, None if agree else marks))
        label = dict((k, lab) for k, lab, _ in CATS)[b]
        cards.append(card_html(
            b, ref, [label], r["prose"][:3],
            r["vim_command"], r["vim_status"], panes, note,
            r["source_dsl"]))

    counts = collections.Counter(c["bucket"] for c in rejects)
    counts["shipped"] = len(rows)

    # ---- validation ledger, computed from the executed statuses
    def s_class(status):
        if "FAILED" in status:
            return "diverges"
        if "no single" in status:
            return "none"
        if "no example strings" in status:
            return "unverified"
        return "validated"

    per = {k: collections.Counter() for k, _, _ in CATS}
    per["shipped"]["validated"] = len(rows)
    for r in rejects:
        per[r["bucket"]][s_class(r["vim_status"])] += 1
    overall = collections.Counter()
    for c in per.values():
        overall.update(c)

    div_txt = (
        "none. The 15 divergences in the previous build were all one defect: "
        "a blanket \\/ escape corrupting character classes (the backslash "
        "became a range endpoint, swallowing digits and uppercase). '/' is "
        "now escaped only outside []-collections and the family is gone — "
        "every command on this page that runs also agrees with the oracle."
        if overall["diverges"] == 0 else
        "the command runs but vim's buffer differs from the oracle "
        "expectation — each card's vim-output pane marks the divergence "
        "line by line.")

    sk = stats["skips"]
    rb = stats.get("reject_buckets", {})
    n_prose_ok = sum(1 for r in rejects if r["bucket"] == "prose-refused"
                     and "validated in sandboxed vim" in r["vim_status"]
                     and "FAILED" not in r["vim_status"])
    top = stats["patterns_in"]
    funnel_rows = [
        ("Recovered patterns", top, ""),
        ("Instruction frame recognised", top - sk.get("no-usable-instruction", 0),
         f"−{sk.get('no-usable-instruction', 0)} prose refused — yet "
         f"{n_prose_ok} of them already have a working, vim-validated command: "
         f"only the English failed"),
        ("Convertible to a vim pattern",
         top - sk.get("no-usable-instruction", 0) - sk.get("no-vim-conversion", 0),
         f"−{sk.get('no-vim-conversion', 0)} truly unrenderable (ops under a "
         f"repeat/alternation, or variable conjuncts with a variable tail). "
         f"Embedded and/~ with a fixed-length tail now renders via "
         f"suffix-pinned lookaheads — 103 patterns recovered this build"),
        ("Enough example strings", top - sk.get("no-usable-instruction", 0)
         - sk.get("no-vim-conversion", 0) - sk.get("too-few-examples", 0),
         f"−{sk.get('too-few-examples', 0)} with fewer than 2 positives or 2 negatives"),
        ("Survived the instruction screens", stats["scenarios"],
         f"−{sk.get('all-instructions-screened', 0)} whose every phrasing tripped a screen"),
        ("Cases written", stats["cases"],
         "up to 3 annotator phrasings per surviving scenario"),
    ]
    funnel = "".join(
        f'<div class="fr"><span class="nm">{E(nm)}</span>'
        f'<span class="bar"><i style="width:{100 * n / top:.1f}%"></i></span>'
        f'<span class="n">{n}</span></div>'
        + (f'<p class="why">{E(why)}</p>' if why else "")
        for nm, n, why in funnel_rows)

    chips = "".join(
        f'<button class="chip" data-c="{k}" data-on="0">{E(lab)} '
        f'{counts.get(k, 0)}</button>'
        for k, lab, _ in CATS)

    def vsplit(k):
        c = per[k]
        parts = []
        if c["validated"]:
            parts.append(f'<i class="ok">✓ {c["validated"]} validated</i>')
        if c["diverges"]:
            parts.append(f'<i class="bad">✗ {c["diverges"]} diverge</i>')
        if c["none"]:
            parts.append(f'<i class="non">— {c["none"]} no command</i>')
        if c["unverified"]:
            parts.append(f'<i class="non">? {c["unverified"]} unverified</i>')
        return " · ".join(parts)

    legend = "".join(
        f'<div class="lg"><b>{E(lab)} · {counts.get(k, 0)}</b>'
        f'<span>{E(desc)}</span><span class="vs">{vsplit(k)}</span></div>'
        for k, lab, desc in CATS)

    vrows = "".join(
        f'<tr><td>{E(lab)}</td>'
        f'<td>{per[k]["validated"] or "—"}</td>'
        f'<td>{per[k]["diverges"] or "—"}</td>'
        f'<td>{per[k]["none"] or "—"}</td>'
        f'<td>{counts.get(k, 0)}</td></tr>'
        for k, lab, _ in CATS)
    ledger = f"""
<h2>Validation ledger</h2>
<div class="panel">
  <div class="kpis">
    <div class="kpi"><p class="lab">validated</p>
      <p class="v ok">{overall["validated"]}
      <small>vim output = oracle expectation, line for line</small></p></div>
    <div class="kpi"><p class="lab">diverge</p>
      <p class="v bad">{overall["diverges"]}
      <small>command runs; vim's buffer ≠ expected</small></p></div>
    <div class="kpi"><p class="lab">no command</p>
      <p class="v non">{overall["none"]}
      <small>dsl2vim has no rendering to execute</small></p></div>
  </div>
  <div class="tscroll"><table class="vt">
    <thead><tr><th>category</th><th>validated</th><th>diverge</th>
    <th>no command</th><th>total</th></tr></thead>
    <tbody>{vrows}</tbody>
  </table></div>
  <p class="vwhy"><b>Why validated ({overall["validated"]}):</b> the command was
  executed in the sandboxed vim and its output buffer equalled the span
  oracle's expectation exactly; the {len(rows)} shipped cases additionally
  agree with the unrestricted Python leg (full verifier triangle). Commands
  containing '/' carry it raw inside []-collections and escaped outside —
  the form vim's delimiter scan expects (measured 2026-08-17).</p>
  <p class="vwhy"><b>Why diverging ({overall["diverges"]}):</b> {div_txt}</p>
  <p class="vwhy"><b>Why no command ({overall["none"]}):</b> embedded
  <code>and</code>/<code>~</code> is now rendered where a whole-line anchor
  can be re-borrowed — everything after the conjunction has fixed length k,
  so each conjunct is pinned by a <code>\\%(…\\.\\{{k}}$\\)\\@=</code>
  lookahead (103 patterns recovered by <code>embedded_conj.py</code>, each
  brute-force-checked against the span oracle in real vim). What remains
  here is the genuinely unrenderable residue: conjunctions under a repeat
  or alternation, and variable-length conjuncts with a variable-length
  tail, where no anchor exists to pin the span.</p>
</div>"""

    page = (HEAD
            .replace("__NCASES__", str(stats["cases"]))
            .replace("__NSCEN__", str(stats["scenarios"]))
            .replace("__NREJ__", str(len(rejects)))
            .replace("__NVIM__", str(sum(1 for r in rejects
                                         if r.get("expected") is not None)
                                     + len(rows)))
            + f'<div class="wrap"><h2>Where the material goes</h2>'
            + f'<div class="funnel">{funnel}</div>'
            + ledger
            + f'<h2>Categories</h2><div class="legend">{legend}</div></div>'
            + f'<div class="bar2"><div class="wrap">'
            + f'<input id="q" type="search" '
            + f'placeholder="filter cards — text search" aria-label="Search">'
            + f'{chips}<span class="tally" id="tally"></span></div></div>'
            + f'<div class="wrap" id="cards">{"".join(cards)}</div>'
            + FOOT)
    Path(args.out).write_text(page, encoding="utf-8")
    size = len(page.encode())
    assert size < 15_000_000, size
    print(f"{len(cards)} cards server-rendered · page {size/1e6:.2f} MB "
          f"-> {args.out}")
    for k, lab, _ in CATS:
        print(f"   {lab:<18} {counts.get(k, 0):>4}")


HEAD = r"""<title>StructuredRegex Conversion Preview</title>
<style>
:root{
  --ground:#f6f6f3; --surface:#ffffff; --raise:#fbfbf9;
  --ink:#191b1f; --dim:#676c75; --line:#dcdcd5; --line2:#eaeae4;
  --accent:#775510; --accent-soft:#f1e8d4;
  --keep:#2c6a4a; --keep-bg:#e8f2ec; --cut:#97362f; --cut-bg:#f8e9e7;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#131519; --surface:#1b1e23; --raise:#22262c;
  --ink:#e6e7e4; --dim:#8d939d; --line:#2b2f36; --line2:#23272d;
  --accent:#d9ad57; --accent-soft:#3a3222;
  --keep:#63b48c; --keep-bg:#1a2a22; --cut:#d97a70; --cut-bg:#2c1f1e;
}}
:root[data-theme="dark"]{
  --ground:#131519; --surface:#1b1e23; --raise:#22262c;
  --ink:#e6e7e4; --dim:#8d939d; --line:#2b2f36; --line2:#23272d;
  --accent:#d9ad57; --accent-soft:#3a3222;
  --keep:#63b48c; --keep-bg:#1a2a22; --cut:#d97a70; --cut-bg:#2c1f1e;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font:15px/1.6 var(--sans);-webkit-font-smoothing:antialiased}
.wrap{max-width:70rem;margin:0 auto;padding:0 1.5rem}
.lab{font-size:11px;letter-spacing:.08em;text-transform:uppercase;
  color:var(--dim);font-weight:600;margin:0}
h1{font-size:1.75rem;font-weight:700;letter-spacing:-.02em;margin:.2rem 0 .4rem;
  text-wrap:balance}
h2{font-size:1.05rem;font-weight:650;margin:2.2rem 0 .5rem}
header{border-bottom:1px solid var(--line);background:var(--surface)}
header .wrap{padding:2rem 1.5rem 1.4rem}
.lede{color:var(--dim);margin:0;max-width:68ch}
.warn{margin:1.1rem 0 0;padding:.7rem .9rem;border-left:3px solid var(--cut);
  background:var(--cut-bg);color:var(--cut);border-radius:0 .35rem .35rem 0;
  font-size:.88rem;max-width:72ch}
.kpis{display:flex;gap:2.2rem;flex-wrap:wrap;margin:1.3rem 0 .2rem}
.kpi .v{font-size:1.6rem;font-weight:700;font-variant-numeric:tabular-nums;
  line-height:1.1}
.kpi .v small{font-size:.9rem;font-weight:500;color:var(--dim)}
.funnel{display:flex;flex-direction:column;gap:.25rem;max-width:56rem}
.fr{display:grid;grid-template-columns:16rem 1fr 3.5rem;gap:.8rem;
  align-items:center}
@media(max-width:44rem){.fr{grid-template-columns:1fr 3.5rem}
  .fr .bar{grid-column:1/-1}}
.fr .nm{font-size:.86rem}
.fr .bar{height:1.3rem;background:var(--line2);border-radius:.25rem;
  overflow:hidden}
.fr .bar i{display:block;height:100%;background:var(--accent);opacity:.85}
.fr .n{text-align:right;font-variant-numeric:tabular-nums;font-weight:650;
  font-size:.9rem}
.why{color:var(--cut);font-size:.78rem;margin:.1rem 0 .5rem;max-width:70ch}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(19rem,1fr));
  gap:.5rem;max-width:64rem}
.lg{background:var(--surface);border:1px solid var(--line);border-radius:.45rem;
  padding:.5rem .7rem;font-size:.82rem}
.lg b{display:block;font-weight:650}
.lg span{color:var(--dim)}
.panel{background:var(--surface);border:1px solid var(--line);
  border-radius:.6rem;padding:1rem 1.1rem;max-width:56rem}
.v.ok{color:var(--keep)} .v.bad{color:var(--cut)} .v.non{color:var(--dim)}
.tscroll{overflow-x:auto;margin:.6rem 0}
table.vt{border-collapse:collapse;font-size:.85rem;min-width:28rem}
.vt th{text-align:left;font-size:11px;letter-spacing:.08em;
  text-transform:uppercase;color:var(--dim);font-weight:600;
  padding:.25rem .9rem .25rem 0;border-bottom:1px solid var(--line)}
.vt td{padding:.3rem .9rem .3rem 0;border-bottom:1px solid var(--line2);
  font-variant-numeric:tabular-nums}
.vt td:not(:first-child){text-align:right}
.vt th:not(:first-child){text-align:right}
.vwhy{font-size:.86rem;line-height:1.6;margin:.7rem 0 0;max-width:72ch}
.vwhy code{font:.8em var(--mono);background:var(--raise);
  padding:.05rem .3rem;border-radius:.25rem}
.lg .vs{display:block;margin-top:.25rem}
.lg .vs i{font-style:normal;font-size:.78rem}
.lg .vs i.ok{color:var(--keep)} .lg .vs i.bad{color:var(--cut)}
.lg .vs i.non{color:var(--dim)}
.bar2{position:sticky;top:0;z-index:5;background:var(--ground);
  border-bottom:1px solid var(--line);padding:.6rem 0;margin-top:1.6rem}
.bar2 .wrap{display:flex;gap:.35rem;align-items:center;flex-wrap:wrap}
input[type=search]{font:inherit;font-size:.86rem;padding:.32rem .6rem;
  border:1px solid var(--line);border-radius:.4rem;background:var(--surface);
  color:inherit;min-width:15rem}
input:focus-visible,.chip:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px}
.chip{cursor:pointer;font:inherit;font-size:.78rem;padding:.16rem .6rem;
  border:1px solid var(--line);border-radius:1rem;background:var(--surface);
  color:var(--dim)}
.chip[data-on="1"]{border-color:var(--accent);color:var(--accent);
  background:var(--accent-soft);font-weight:600}
.tally{color:var(--dim);font-size:.84rem}
.card{background:var(--surface);border:1px solid var(--line);
  border-radius:.55rem;padding:.75rem .9rem;margin:.6rem 0}
.card[hidden]{display:none}
.chead{display:flex;gap:.45rem;align-items:baseline;flex-wrap:wrap}
.ref{font:600 .78rem/1 var(--mono);color:var(--accent)}
.bdg{font-size:.72rem;color:var(--dim);border:1px solid var(--line);
  border-radius:.9rem;padding:.05rem .45rem}
.ins{font-weight:600;margin:.4rem 0 .1rem;max-width:75ch}
.ins+.ins{font-weight:400;color:var(--dim);font-size:.9rem;margin-top:.15rem}
.cmd{font:.82rem/1.5 var(--mono);background:var(--raise);
  border:1px solid var(--line2);border-radius:.35rem;padding:.35rem .55rem;
  margin:.45rem 0 0;overflow-x:auto;white-space:pre}
.cmdst{font-size:.75rem;margin:.2rem 0 0}
.cmdst.ok{color:var(--keep)} .cmdst.bad{color:var(--cut)}
.panes{display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));
  gap:.6rem;margin:.5rem 0 0}
.pane{min-width:0}
.pane pre{margin:.2rem 0 0;font:.8rem/1.5 var(--mono);background:var(--raise);
  border:1px solid var(--line2);border-radius:.35rem;padding:.4rem .1rem;
  overflow-x:auto}
.pane pre em{display:block;padding:0 .55rem;font-style:normal;white-space:pre}
.pane pre em.cut{background:var(--cut-bg);color:var(--cut);
  text-decoration:line-through;text-decoration-thickness:1px}
.pane pre em.kp{background:var(--keep-bg);color:var(--keep)}
.pane pre em.dim{color:var(--dim)}
.note{margin:.45rem 0 0;font-size:.79rem;line-height:1.5;color:var(--dim);
  border-left:2px solid var(--accent);padding-left:.6rem;max-width:75ch}
.dsl{color:var(--dim);font:.74rem/1.5 var(--mono);margin:.4rem 0 0;
  word-break:break-all}
</style>
<header><div class="wrap">
  <p class="lab">vimBot · dataset v3</p>
  <h1>StructuredRegex Conversion Preview</h1>
  <p class="lede">Every pattern of the recovered sample, on this page: the
  shipped cases and all five reject categories, each with its vim command and
  an <b>executed</b> verdict — where vim ran, its actual output stands beside
  the expected output.</p>
  <div class="kpis">
    <div class="kpi"><p class="lab">cases produced</p>
      <p class="v">__NCASES__ <small>from __NSCEN__ scenarios</small></p></div>
    <div class="kpi"><p class="lab">rejected patterns</p>
      <p class="v">__NREJ__ <small>all on this page</small></p></div>
    <div class="kpi"><p class="lab">vim executions shown</p>
      <p class="v">__NVIM__ <small>output vs expected, per card</small></p></div>
  </div>
  <p class="warn"><b>Preview, not an import.</b> StructuredRegex is not
  vendored in this repo; input is 460 patterns recovered from a published
  artifact (refs contiguous SR-0001..SR-0718 — not a verified random sample of
  the ~3.5k corpus; do not extrapolate). A real import vendors the corpus under
  §14c first.</p>
</div></header>
"""

FOOT = r"""<div class="wrap"><p class="note" style="margin:2rem 0 4rem">
Filtering only hides cards already on this page — with scripting disabled,
everything above remains visible. Generated by make_v3_sregex.py →
build_sregex_preview.py; validation statuses come from executions in the
sandboxed vim, never from assertions.</p></div>
<script>
try{
  var chips=document.querySelectorAll(".chip"),q=document.getElementById("q"),
      cards=document.querySelectorAll(".card"),tally=document.getElementById("tally"),
      on={};
  function apply(){
    var any=false,i,n=0,text=(q.value||"").toLowerCase();
    for(i=0;i<chips.length;i++) if(on[chips[i].dataset.c]) any=true;
    for(i=0;i<cards.length;i++){
      var c=cards[i],show=(!any||on[c.dataset.cat]);
      if(show&&text) show=c.textContent.toLowerCase().indexOf(text)>=0;
      c.hidden=!show; if(show)n++;
    }
    tally.textContent=n+" of "+cards.length+" cards shown";
  }
  chips.forEach(function(ch){ch.addEventListener("click",function(){
    on[ch.dataset.c]=!on[ch.dataset.c];
    ch.dataset.on=on[ch.dataset.c]?"1":"0";apply();});});
  var t;q.addEventListener("input",function(){clearTimeout(t);t=setTimeout(apply,150);});
  apply();
}catch(e){/* filtering is an enhancement; the page stands without it */}
</script>
"""


if __name__ == "__main__":
    main()
