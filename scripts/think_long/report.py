"""Report on the long-thinking extension (CPU): `python scripts/think_long/report.py`.

Each thinking arm at 4,096, 8,192 and 16,384 generated tokens, against its control from the thinking
run on the same rows (exact McNemar, paired bootstrap); 8,192 is primary (addendum 1). A cap's rows
are the start of the arm's fixed order up to the first row without an outcome at that cap, so a
deadline cut leaves a random subset. A row's outcome at 8k is its 4k completion if that stopped
before the cap, its continuation if the cap cut it, or a fresh 8k completion if the thinking run
never reached it. The rules are in runs/think_long_2026-11-01/preregistration.md.
"""
import importlib.util
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = load("think_long_extend", HERE / "extend.py")
C = load("distill_targets_compare", REPO / "scripts" / "distill_targets" / "compare.py")   # compare(), pct()
T = E.T
CAPS = (E.FIRST_CAP,) + E.CAPS
CONTROL = {"think-trained": "nothink-trained", "think-retr8": "nothink-retr8"}
NAME = {"think-trained": "TA", "think-retr8": "TB"}
VERDICT = {"stronger": "helps", "weaker": "hurts", "tied": "tied"}
PRIMARY = E.CAPS[0]                      # addendum 1: the owner moved the working cap to 8k
SNAPSHOT = "report_2026-11-01_1644.md"   # addendum 2: the report before phase C's new rows; its readings stay primary
TIERS = ("T1", "T2", "T3")


def outcome_at(i, byid, stages, arm, cap):
    """The completion that decides row `i` at `cap`, or None if nothing has reached it there."""
    r = byid[arm].get(i)
    if r is not None and (cap == E.FIRST_CAP or not E.cut(r)):
        return r
    if cap == E.FIRST_CAP:
        return None
    for c in E.CAPS:
        nxt = stages[c][arm].get(i)
        if nxt is None:
            return None
        if c == cap or not E.cut(nxt):
            return nxt
    return None


def prefix(order, byid, stages, arm, cap):
    """(id, completion) in the arm's order, up to the first row without a completion at `cap`."""
    out = []
    for i in order:
        o = outcome_at(i, byid, stages, arm, cap)
        if o is None:
            break
        out.append((i, o))
    return out


def summary(pairs, val):
    t = [o["think_tokens"] for _, o in pairs]
    return {"rows": len(pairs), "solved": sum(o["match"] for _, o in pairs),
            "fresh": sum(bool(o.get("fresh")) for _, o in pairs),
            "closed": sum(o["closed"] for _, o in pairs), "cut": sum(not o["closed"] for _, o in pairs),
            "forced": sum(bool(o.get("forced_close")) for _, o in pairs),
            "forced_solved": sum(bool(o.get("forced_close")) and o["match"] for _, o in pairs),
            "valid": sum(o["valid"] for _, o in pairs),
            "think_median": statistics.median(t) if t else None,
            "think_p90": round(statistics.quantiles(t, n=10)[-1]) if len(t) >= 10 else (max(t) if t else None),
            "tiers": {k: [sum(o["match"] for i, o in pairs if val[i]["tier"] == k),
                          sum(val[i]["tier"] == k for i, _ in pairs)] for k in TIERS}}


def build(first_run=E.FIRST_RUN, run=E.RUN):
    first_run, run = pathlib.Path(first_run), pathlib.Path(run)
    byid = {arm: {r["id"]: r for r in rows} for arm, rows in E.first_rows(first_run).items()}
    orders = E.arm_orders(first_run)
    stages = {cap: {arm: E.stage_rows(run, cap, arm) for arm in E.THINK_ARMS} for cap in E.CAPS}
    val = {c["id"]: c for c in E.load_jsonl(str(REPO / "data" / "splits" / "val.jsonl"))}
    controls = {a: {r["id"]: int(bool(r["match"])) for r in E.load_jsonl(str(first_run / ("rows_%s.jsonl" % a)))}
                for a in CONTROL.values()}
    rep = {"caps": list(CAPS), "primary_cap": PRIMARY, "snapshot": SNAPSHOT if (run / SNAPSHOT).exists() else None,
           "arms": {}, "readings": {}, "verdicts": {}, "extra_room": {}}
    for arm, order in orders.items():
        ctl = controls[CONTROL[arm]]
        rep["arms"][arm] = {}
        at = {}
        for cap in CAPS:
            pairs = [(i, o) for i, o in prefix(order, byid, stages, arm, cap) if i in ctl]
            at[cap] = dict(pairs)
            s = summary(pairs, val)
            s["control_solved"] = sum(ctl[i] for i, _ in pairs)
            rep["arms"][arm][str(cap)] = s
            if pairs:
                ids = [i for i, _ in pairs]
                r = C.compare(ctl, {i: int(bool(o["match"])) for i, o in pairs}, ids)
                key = "%s%dk" % (NAME[arm], cap // 1024)
                rep["readings"][key] = {"arm": arm, "control": CONTROL[arm], "cap": cap, **r}
                rep["verdicts"][key] = VERDICT[r["verdict"]]
        # What the extra room did: each cap against 4,096 on the rows both have (the thinking run's).
        for cap in E.CAPS:
            ids = [i for i in at[cap] if i in at[E.FIRST_CAP]]
            if ids:
                r = C.compare({i: int(bool(at[E.FIRST_CAP][i]["match"])) for i in ids},
                              {i: int(bool(at[cap][i]["match"])) for i in ids}, ids)
                rep["readings"]["%s room %dk" % (NAME[arm], cap // 1024)] = {
                    "arm": arm, "cap": cap, "against": E.FIRST_CAP, **r}
        cut4 = [i for i in at[E.FIRST_CAP] if E.cut(at[E.FIRST_CAP][i])]
        rep["extra_room"][arm] = {
            "cut_at_4k": len(cut4),
            "reached_by": {str(cap): sum(i in at[cap] for i in cut4) for cap in E.CAPS},
            "closed_by": {str(cap): sum(at[cap][i]["closed"] for i in cut4 if i in at[cap]) for cap in E.CAPS},
            "solved_by": {str(cap): sum(at[cap][i]["match"] for i in cut4 if i in at[cap]) for cap in E.CAPS},
            "control_solved": sum(ctl[i] for i in cut4)}
        secs = [r["seconds"] for cap in E.CAPS for r in stages[cap][arm].values()]
        rep["extra_room"][arm]["gpu_hours"] = round(sum(secs) / 3600, 2)
    return rep


MEANING = ("If thinking helps at 8,192 or 16,384 tokens, it is a candidate teacher for hard rows, but at several minutes per "
           "row on this card a labeling run would need a faster engine first; the next step is the owner's call. If it "
           "helps at neither cap, thinking stays off for this teacher. Either way, nothing changes for any student, "
           "results-table row or served model.")


def render(rep):
    L = ["# The thinking run's cut-off rows, continued to 8k and 16k tokens", "",
         "By `preregistration.md` beside this file; built by `scripts/think_long/report.py`. Each cap's rows are the "
         "start of the arm's fixed order, up to the first row not reached at that cap.", "",
         "| arm | cap | rows (fresh) | thinking solves | control solves | finished thinking | still cut | median / p90 think tokens | T1 / T2 / T3 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for arm, caps in rep["arms"].items():
        for cap, s in caps.items():
            L.append("| `%s` | %s | %d (%d) | %d | %d | %d | %d | %s / %s | %s |" % (
                arm, cap, s["rows"], s["fresh"], s["solved"], s["control_solved"], s["closed"], s["cut"],
                s["think_median"], s["think_p90"], " / ".join("%d of %d" % tuple(s["tiers"][t]) for t in TIERS)))
    forced = [(arm, cap, s) for arm, caps in rep["arms"].items() for cap, s in caps.items() if s.get("forced")]
    if forced:
        L += ["", "Finished by the loop breaker (`scripts/think_loop`: stopped repeating itself, answered from one "
              "copy of the repeated block): " + "; ".join("`%s` at %s: %d, %d solved" % (a, c, s["forced"], s["forced_solved"])
                                                          for a, c, s in forced) + "."]
    primary = "Primary: TA and TB at %d tokens (addendum 1). The 4,096 readings are the thinking run's." % rep["primary_cap"]
    if rep.get("snapshot"):
        primary += (" Addendum 2 then added new rows at 8,192 overnight, so the primary readings stay the ones in `%s`; "
                    "the readings below, on the longer prefix, are an update." % rep["snapshot"])
    L += ["", "## Readings (exact McNemar, paired bootstrap 95% interval)", "", primary, "",
          "| reading | rows | difference | + / − | p | verdict |", "|---|---|---|---|---|---|"]
    for key, r in rep["readings"].items():
        if "room" in key:
            what = "`%s` at %d against itself at %d" % (r["arm"], r["cap"], r["against"])
            verdict = VERDICT[r["verdict"]]
        else:
            what = "`%s` against `%s`, %d tokens" % (r["arm"], r["control"], r["cap"])
            verdict = rep["verdicts"][key]
        L.append("| %s: %s | %d | %+.1f [%+.1f, %+.1f] | +%d / −%d | %.3g | **%s** |" % (
            key, what, r["n"], r["diff_pts"], r["ci95"][0], r["ci95"][1], r["gained"], r["lost"], r["p"], verdict))
    L += ["", "## The rows the 4,096 cap cut", "", "| arm | cut at 4,096 | reached by 8k / 16k | finished by 8k / 16k | "
          "solved by 8k / 16k | control solves on them | GPU hours, all the arm's 8k / 16k rows |", "|---|---|---|---|---|---|---|"]
    for arm, x in rep["extra_room"].items():
        L.append("| `%s` | %d | %s | %s | %s | %d | %s |" % (
            arm, x["cut_at_4k"], " / ".join(str(x["reached_by"][str(c)]) for c in E.CAPS),
            " / ".join(str(x["closed_by"][str(c)]) for c in E.CAPS),
            " / ".join(str(x["solved_by"][str(c)]) for c in E.CAPS), x["control_solved"], x["gpu_hours"]))
    return "\n".join(L + ["", MEANING]) + "\n"


def main():
    rep = build()
    (E.RUN / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    text = render(rep)
    (E.RUN / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
