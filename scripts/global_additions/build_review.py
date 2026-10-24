#!/usr/bin/env python3
"""Build the review page for the proposed global prompts (follow-up #2, 2026-10-16).

Same machinery as the test-slice and new-cases pages (scripts/build_test_review.py):
each card shows the instruction, the input, the expected output and what vim
produced. Verdicts (ok / drop / reword) and edits stay in the browser and
export as the usual {edits, deletes, review} change-set. Because these rows
are not in the corpus, that change-set is applied with
scripts/global_additions/apply_review.py to data/proposed/global_2026-10-16/
(not apply_curation.py). Each card's history line says which skill the row
teaches and why that skill is in the set.

The priority filter marks the relative-address, join and range rows. Their
instructions carry more structure (which line, which neighbour, which span),
so they are the likeliest place for a wording that does not match the edit.

    python scripts/global_additions/build_review.py --out page.html
"""
import argparse
import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from build_test_review import build, collect      # noqa: E402

CASES = REPO / "data" / "proposed" / "global_2026-10-16" / "cases.jsonl"
PRIORITY = {"relative", "join", "range"}

META = {
    "title": "Proposed Global Prompts",
    "lab": "vimBot &middot; proposed 2026-10-16 &middot; follow-up #2, not merged",
    "h1": "Does the instruction describe this edit?",
    "total_label": "proposed prompts",
    "flag_label": "relative, join or range",
    "flag_button": "priority",
    "key": "vimbot-proposed-global-1016",
}



# Verdicts for 926 rows are state that must survive a cleared browser, so the page
# also keeps them in the artifact's shared store (collection `decisions`, one
# document per row id: verdict, note, edits), the way the never-solved review did.
# It wraps the builder's own save() and render(); browser storage and Export keep
# working when the store is unavailable (a local file, a signed-out viewer).
SYNC = r"""
<script>
(function(){
  const COLL = 'decisions';
  let db = null, timer = null;
  const synced = {};
  function stateOf(id){
    const v = verdicts[id] || {}, e = edits[id] || {}, out = {};
    if (v.verdict) out.verdict = v.verdict;
    if (v.note) out.note = v.note;
    if (Object.keys(e).length) out.edits = e;
    return out;
  }
  const k = JSON.stringify;
  const host = document.createElement('div');
  host.className = 'wrap';
  const status = document.createElement('p');
  status.className = 'lede';
  status.style.margin = '12px 0 0';
  status.textContent = 'Connecting to the shared store…';
  host.appendChild(status);
  const header = document.querySelector('header');
  if (header) header.after(host); else document.body.prepend(host);
  function flush(){
    timer = null;
    if (!db) return;
    DATA.forEach(function(c){
      const s = stateOf(c.id);
      if (k(s) === k(synced[c.id] || {})) return;
      synced[c.id] = s;
      db.collection(COLL).doc(c.id).set(Object.assign({}, s, {updated: new Date().toISOString()})).catch(function(err){
        status.textContent = 'Could not save ' + c.id + ' to the shared store (' + (err && err.code) + '). It is kept in this browser; Export hands it over.';
      });
    });
  }
  const localSave = save;
  save = function(){ localSave(); if (db && !timer) timer = setTimeout(flush, 600); };
  (async function(){
    let d = null;
    try { d = window.claude && window.claude.use ? await window.claude.use('db') : null; } catch (e) { d = null; }
    if (!d){ status.textContent = 'Saving in this browser only. Export hands the verdicts over.'; return; }
    db = d;
    status.textContent = 'Saving to this page’s shared store, so Claude can read the verdicts back. Export still works.';
    let first = true;
    db.collection(COLL).onSnapshot(function(snap){
      snap.docs.forEach(function(doc){
        if (!doc.exists) return;
        const s = doc.data() || {}, id = doc.id;
        const remote = {};
        if (s.verdict) remote.verdict = s.verdict;
        if (s.note) remote.note = s.note;
        if (s.edits && Object.keys(s.edits).length) remote.edits = s.edits;
        const local = stateOf(id);
        // on the first answer, a change made in this browser before the store answered wins
        if (first && Object.keys(local).length && k(local) !== k(remote)) return;
        synced[id] = remote;
        if (remote.verdict || remote.note) verdicts[id] = {verdict: remote.verdict, note: remote.note}; else delete verdicts[id];
        if (remote.edits) edits[id] = remote.edits; else delete edits[id];
      });
      if (first){ first = false; flush(); }
      localSave();
      render();
    }, function(err){
      status.textContent = 'The shared store stopped answering (' + (err && err.code) + '). New verdicts stay in this browser; Export hands them over.';
      db = null;
    });
  })();
})();
</script>
"""


def lede(rows, manifest):
    by_skill = collections.Counter(r["skill"] for r in rows)
    skills = " &middot; ".join("%s <b>%d</b>" % (k, v) for k, v in by_skill.most_common())
    return """\
  <p class="lede">
    <b>%d proposed global prompts</b>, written on 2026-10-16 to test whether more
    training data in <span class="mono">global</span> still pays (the 25 / 50 / 100%%
    scaling runs measured +8.2 points per doubling of train, and +12 for global
    alone). They are <b>not in the corpus or the splits</b>: they live in
    <span class="mono">data/proposed/global_2026-10-16/</span>, tagged
    <span class="mono">add-global-2026-10-16</span>, and one tag filter removes them.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>Why these skills.</b> On train, the served adapter fails the global skills the
    corpus barely teaches. 1,099 of the 1,238 global train rows are deletes. Substitutes
    on matching lines (44 rows: 18%% never solved, 52%% only sometimes), moves, copies and
    joins (about 20 rows) and capture groups (16 rows: 25%% never solved) are rare. %s.
    %d distinct gold commands over %d buffers, mostly T3.
  </p>
  <p class="lede" style="margin-top:10px">
    Every row passed the machine gate. Vim ran it; the buffer changes; making the global
    pattern case-blind does not change the result, so the instruction needs no case
    clause; and no leakage key is shared with val or test. The question left is the one
    no gate answers: does the sentence describe the edit, completely and without a second
    faithful reading? <b>drop</b> removes a row; <b>reword</b> plus <b>Edit</b> changes it.
    Export, then run
    <span class="mono">python scripts/global_additions/apply_review.py --changes changes.json --dry-run</span>.
  </p>
  <p class="lede" style="margin-top:10px">
    <b>Already measured, before this review (2026-10-17).</b> SFT with these rows added moved the
    held-out global rows from 68.4%% to 69.4%% (+1.0, p 0.85), not the projected +10. Those rows are
    83%% deletes from the regex corpora, while these teach substitutes, moves, copies and joins the
    evaluation barely contains. So this review decides whether the rows teach real editing skills
    worth keeping. Merging them will not move today&rsquo;s headline, and measuring their skills
    needs a held-out slice that contains them.
  </p>
""" % (len(rows), skills, manifest.get("distinct_commands", 0), manifest.get("distinct_buffers", 0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=str(CASES))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    manifest = json.loads((pathlib.Path(args.cases).parent / "manifest.json").read_text(encoding="utf-8"))
    by_id = {r["id"]: r for r in rows}
    cases = collect(args.cases,
                    flag=lambda r, risk: r.get("skill") in PRIORITY,
                    sort_key=lambda c: (by_id[c["id"]]["skill"], by_id[c["id"]]["domain"], c["id"]),
                    history=lambda r: "%s · %s: %s" % (r["domain"], r["skill"], r["why"]))
    disagree = sum(1 for c in cases if not c["agrees"])
    page = build(cases, {**META, "lede": lede(rows, manifest)})
    page = page + SYNC
    pathlib.Path(args.out).write_text(page, encoding="utf-8")
    print("proposed prompts: %d (%d priority)   vim disagreements: %d   page: %d KB -> %s"
          % (len(cases), sum(1 for c in cases if c["flagged"]), disagree, len(page) // 1024, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
