"""The thinking run's cut-off rows, continued: first to 8,192 generated tokens, then to 16,384.

The owner's choice, 2026-11-01 ("8k, then 16k"): the 4,096-token cap of `scripts/think27b` cut off
about half of the trained 27B's rows and four in five of the base 27B's. A higher cap does not
change a completion that stopped before 4,096 tokens, so only the cut-off rows run again, and each
continues from where it stopped instead of starting over. The rules are in
runs/think_long_2026-11-01/preregistration.md.

Same weights, prompts and sampling as the thinking run, one row at a time (its padding guard put
it there). Stage 8k continues every row the 4k cap cut, alternating the two arms in their fixed
orders; with --fresh it then goes on to the arms' next rows, generated at 8,192 tokens from the
start (addendum 1: the owner stopped the 4k run to move on to 8k). Stage 16k continues every row
still cut at 8k. No row starts after --deadline, and a rerun resumes from the rows on disk.

The loop breaker (scripts/think_loop, added 2026-11-01 after this run) stops a row whose reasoning
starts repeating itself and answers it from one copy of the repeated block. It is on for a new
--run directory and off for runs/think_long_2026-11-01, whose rows predate it.

    python scripts/think_long/extend.py plan                                          # CPU -> <run>/plan.json
    python scripts/think_long/extend.py run --stages 8192 --fresh --deadline 1793558700   # GPU: 8k, new rows too
    python scripts/think_long/extend.py run --deadline 1793622600                     # GPU: stage 8k, then 16k
"""
import argparse
import contextlib
import datetime
import importlib.util
import json
import pathlib
import statistics
import sys
import time
import zlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from evaluate import load_jsonl                                   # noqa: E402
from prompt import messages_for_case                              # noqa: E402


def load(name, path):
    """Import a module under a unique name, so folders with same-named files do not collide."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load("think27b_think27b", REPO / "scripts" / "think27b" / "think27b.py")
LB = load("think_loop_loop", REPO / "scripts" / "think_loop" / "loop.py")
RUN = REPO / "runs" / "think_long_2026-11-01"
FIRST_RUN = T.RUN                        # the thinking run: its rows and controls
FIRST_CAP = 4096                         # its pre-registered cap
CAPS = (8192, 16384)                     # generated tokens in all (reasoning plus answer), stage by stage
THINK_ARMS = ("think-trained", "think-retr8")
SEED = 33000000                          # apart from the thinking run's and the labeling pass's seeds
RC_SLOW = T.RC_SLOW


def cut(r):
    """Whether the cap stopped a completion: it ran the whole budget without a stop token."""
    return r["steps"] >= r["max_new_tokens"] and r["new_tokens"] >= r["steps"]


def arm_orders(first_run=FIRST_RUN):
    """Each thinking arm's fixed order over all the thinking run's rows."""
    meta = json.loads((first_run / "rows.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in meta["rows"]]
    return {arm: T.order_for(arm, ids) for arm in THINK_ARMS}


def first_rows(first_run=FIRST_RUN):
    """Each thinking arm's rows from the thinking run, in the arm's fixed order."""
    meta = json.loads((first_run / "rows.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in meta["rows"]]
    out = {}
    for arm in THINK_ARMS:
        path = first_run / ("rows_%s.jsonl" % arm)
        rows = {r["id"]: r for r in load_jsonl(str(path))} if path.exists() else {}
        out[arm] = [rows[i] for i in T.order_for(arm, ids) if i in rows]
    return out


def is_prefix(first_run=FIRST_RUN):
    """Whether each arm's finished rows are the start of its order (so they are a random subset)."""
    meta = json.loads((first_run / "rows.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in meta["rows"]]
    done = first_rows(first_run)
    return {arm: [r["id"] for r in rows] == T.order_for(arm, ids)[:len(rows)] for arm, rows in done.items()}


def stage_rows(run, cap, arm):
    path = run / ("rows_%d_%s.jsonl" % (cap, arm))
    return {r["id"]: r for r in load_jsonl(str(path))} if path.exists() else {}


def record_cap(r):
    """The cap a record ran under: a stage record says so; the thinking run's ran under 4,096."""
    return r.get("cap", FIRST_CAP)


def chain(i, byid, stages, arm, cap):
    """The row's records before stage `cap`, in order: its 4k completion if the thinking run reached
    it, then each earlier stage's. It ends where the row stopped or where a stage has not reached it."""
    out = [byid[arm][i]] if i in byid[arm] else []
    for c in CAPS:
        if c >= cap or (out and not cut(out[-1])):
            break
        r = stages[c][arm].get(i)
        if r is None:
            break
        out.append(r)
    return out


def work(orders, byid, stages, cap, fresh=False):
    """Per arm, in its order, the rows this stage runs: those the previous cap cut, and with `fresh`
    (stage 8k only) every row the thinking run never reached, from the start."""
    prev_cap = FIRST_CAP if cap == CAPS[0] else CAPS[CAPS.index(cap) - 1]
    out = {}
    for arm, order in orders.items():
        todo = []
        for i in order:
            if i in stages[cap][arm]:
                continue
            records = chain(i, byid, stages, arm, cap)
            if records:
                if record_cap(records[-1]) == prev_cap and cut(records[-1]):
                    todo.append(i)
            elif fresh and cap == CAPS[0]:
                todo.append(i)
        out[arm] = todo
    return out


def text_so_far(records):
    """What the row has generated before this stage, as text ("" for a row starting fresh)."""
    text = ""
    for r in records:
        if "continuation" in r:
            text += r["continuation"]
        else:
            text += r["reasoning"] + ((T.THINK_END + r["answer"]) if r["closed"] else "")
    return text


def tokens_so_far(records):
    return sum(r["new_tokens"] for r in records)


def prompt_text(tok, case, arm, train, plan, info):
    """The thinking run's prompt for this row and arm, byte for byte."""
    reg_name, _, retr = T.ARMS[arm]
    exs = [train[j] for j in plan[case["id"]]["examples"]] if retr else []
    return tok.apply_chat_template(messages_for_case(case, exs), add_generation_prompt=True, tokenize=False,
                                   **info["kwargs"][reg_name])


def seed_for(arm, row_id, cap):
    return SEED + 1000000 * CAPS.index(cap) + 100000 * list(T.ARMS).index(arm) + zlib.crc32(row_id.encode()) % 100000


def cmd_plan(args):
    first_run, run = pathlib.Path(args.first_run), pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    firsts = first_rows(first_run)
    byid = {arm: {r["id"]: r for r in rows} for arm, rows in firsts.items()}
    orders = arm_orders(first_run)
    stages = {cap: {arm: stage_rows(run, cap, arm) for arm in THINK_ARMS} for cap in CAPS}
    todo = work(orders, byid, stages, CAPS[0])
    out = {"first_run": str(first_run.relative_to(REPO)) if first_run.is_relative_to(REPO) else str(first_run),
           "prefix": is_prefix(first_run), "arms": {}}
    hours = {cap: 0.0 for cap in CAPS}       # if every continued row runs its whole budget
    for arm, rows in firsts.items():
        per_step = statistics.median(r["batch_seconds"] / max(r["steps"], 1) for r in rows) if rows else 0.0
        out["arms"][arm] = {"rows": len(rows), "cut_at_4k": sum(cut(r) for r in rows),
                            "stage_8k_to_do": len(todo[arm]), "s_per_step": round(per_step, 3),
                            "fresh_8k_done": sum(1 for r in stages[CAPS[0]][arm].values() if r.get("fresh")),
                            "never_reached": len(orders[arm]) - len(rows)}
        caps = (FIRST_CAP,) + CAPS
        for k, cap in enumerate(CAPS):
            hours[cap] += len(todo[arm]) * (cap - caps[k]) * per_step / 3600
    out["hours_at_most"] = {str(cap): round(h, 1) for cap, h in hours.items()}
    (run / "plan.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("; ".join("%s: %d rows, %d cut at 4,096, %d to continue" % (a, r["rows"], r["cut_at_4k"], r["stage_8k_to_do"])
                    for a, r in out["arms"].items())
          + "; at most %.1f h for stage 8k and %.1f h for stage 16k; prefixes %s"
          % (hours[CAPS[0]], hours[CAPS[1]], out["prefix"]), flush=True)
    return 0



def cmd_run(args):
    import torch
    first_run, run = pathlib.Path(args.first_run), pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    lb = LB.resolve(run, args.loop_break)
    print("loop breaker: %s" % (lb or "off"), flush=True)
    byid = {arm: {r["id"]: r for r in rows} for arm, rows in first_rows(first_run).items()}
    orders = arm_orders(first_run)
    flags = {r["id"]: r for r in json.loads((first_run / "rows.json").read_text(encoding="utf-8"))["rows"]}
    val = {c["id"]: c for c in load_jsonl(str(T.VAL))}
    train = {c["id"]: c for c in load_jsonl(str(T.TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(T.PLAN))}
    stages = {cap: {arm: stage_rows(run, cap, arm) for arm in THINK_ARMS} for cap in CAPS}
    tok = model = info = stop = None
    pace, slow = [], 0
    for cap in [c for c in CAPS if c in args.stages]:
        todo = work(orders, byid, stages, cap, fresh=args.fresh)
        print("stage %d: %s" % (cap, ", ".join("%s %d to do" % (a, len(t)) for a, t in todo.items())), flush=True)
        if not any(todo.values()):
            continue
        if time.time() > args.deadline:
            print("deadline already passed: stage %d not started" % cap, flush=True)
            return 0
        if model is None:
            tok, model, info = T.load_model()
            (run / ("load_%s.json" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))).write_text(
                json.dumps(info, indent=1) + "\n", encoding="utf-8")
            think_end_id = tok.convert_tokens_to_ids(T.THINK_END)
            stop = LB.stopping(tok, lb) if lb else None
            torch.cuda.reset_peak_memory_stats()
        with contextlib.ExitStack() as stack:
            files = {a: stack.enter_context((run / ("rows_%d_%s.jsonl" % (cap, a))).open("a", encoding="utf-8"))
                     for a in THINK_ARMS}
            for arm, b, ids in T.schedule(todo, 1, {a: True for a in todo}):
                if time.time() > args.deadline:
                    print("deadline reached at stage %d in %s after %d of %d rows; the report uses what is done"
                          % (cap, arm, b, len(todo[arm])), flush=True)
                    return 0
                case = val[ids[0]]
                records = chain(case["id"], byid, stages, arm, cap)
                prompt = prompt_text(tok, case, arm, train, plan, info)
                prev = text_so_far(records)
                before = tokens_so_far(records)
                budget = cap - before
                seed = seed_for(arm, case["id"], cap)
                t0 = time.time()
                outs, lens = T.generate(model, tok, [prompt + prev], T.ARMS[arm][1], budget, info, seed,
                                        stopping_criteria=stop)
                secs = time.time() - t0
                new, steps = outs[0], lens[0]
                new_text = tok.decode(new, skip_special_tokens=False)
                reasoning, answer, closed = T.split_reasoning(prev + new_text, True)
                loop = None
                if lb and not closed and len(new) < budget:     # stopped early without closing: the breaker
                    forced, loop = LB.break_loop(T.generate, model, tok, prompt, reasoning, T.ARMS[arm][1],
                                                 info, seed, lb)
                    if loop:
                        answer, closed = forced, True
                parsed, per = T.judge([case], [answer], args.timeout)
                p, r = parsed[case["id"]], per.get(case["id"], {})
                closed_before = [x for x in records if x["closed"]]
                think = (closed_before[0]["think_tokens"] if closed_before
                         else before + T.think_tokens(new, think_end_id))
                prefix_tokens = len(tok(prompt + prev)["input_ids"]) - len(tok(prompt)["input_ids"])
                rec = {"arm": arm, "id": case["id"], "cap": cap, "tier": case.get("tier"), "category": case.get("category"),
                       "in_fail": flags[case["id"]]["in_fail"], "in_hard": flags[case["id"]]["in_hard"],
                       "fresh": not records, "tokens_before": before, "prefix_tokens": prefix_tokens, "new_tokens": len(new), "steps": steps,
                       "max_new_tokens": budget, "total_tokens": before + len(new), "think_tokens": think,
                       "closed": closed, "hit_cap": steps >= budget and not closed, "answer": answer,
                       "continuation": new_text, "parse_ok": p is not None, "mode": p[0] if p else None,
                       "command": p[1] if p else None, "valid": bool(r.get("valid")), "match": bool(r.get("match")),
                       "error": r.get("error"), "seconds": round(secs, 1), "seed": seed,
                       "loop_break": lb, "forced_close": loop is not None, "loop": loop,
                       "time": datetime.datetime.now().isoformat(timespec="seconds")}
                files[arm].write(json.dumps(rec) + "\n")
                files[arm].flush()
                stages[cap][arm][case["id"]] = rec
                print("  %d %s %d/%d  %d new tokens in %.0f s (%.2f s/step)  closed %s  matched %d  peak %.1f GiB"
                      % (cap, arm, b + 1, len(todo[arm]), len(new), secs, secs / max(steps, 1), closed,
                         rec["match"], torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
                if steps >= 256:                 # per-step time is prefill-bound below this
                    per_step = secs / steps
                    if len(pace) >= 3 and per_step > 3 * statistics.median(pace):
                        slow += 1
                        if slow >= 2:
                            raise T.SlowCard("%.2f s/step against a median of %.2f: the card is paging"
                                             % (per_step, statistics.median(pace)))
                    else:
                        slow = 0
                        pace.append(per_step)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="CPU: the work, from the thinking run's rows -> <run>/plan.json")
    p.add_argument("--first-run", default=str(FIRST_RUN))
    p.add_argument("--run", default=str(RUN))
    g = sub.add_parser("run", help="GPU: stage 8k, then stage 16k, resumable, until --deadline")
    g.add_argument("--first-run", default=str(FIRST_RUN))
    g.add_argument("--run", default=str(RUN))
    g.add_argument("--deadline", type=float, required=True, help="epoch seconds; no row starts after it")
    g.add_argument("--stages", type=int, nargs="+", default=list(CAPS), choices=list(CAPS))
    g.add_argument("--fresh", action="store_true",
                   help="stage 8k goes on to the rows the thinking run never reached, generated from the start")
    g.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    g.add_argument("--loop-break", action=argparse.BooleanOptionalAction, default=None,
                   help="scripts/think_loop's loop breaker; default: on for a new --run directory, "
                        "else what the directory ran with")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    try:
        return cmd_run(args)
    except T.SlowCard as e:
        print("stopped: %s" % e, flush=True)
        return RC_SLOW


if __name__ == "__main__":
    sys.exit(main())
