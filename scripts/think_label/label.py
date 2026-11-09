#!/usr/bin/env python3
"""The base 27B, thinking, labels the hard train rows: answers verified where the 4B and the trained models have none.

By runs/think_label_2026-11-01/preregistration.md: the 795 hard rows, a report before any label is
used, and the card free again by Monday 08:00. run.sh starts it only if the thinking run's TB
reading (the base 27B with retrieved examples, thinking against not) is *helps*:
the `trigger` stage. Only the 27B thinks, as in scripts/think27b (CLAUDE.md §14; DECISIONS.md
2026-10-31): no student, results-table row or served model does.

Rows (`plan`, CPU): the 770 train rows the distilled chain's round 1 never solved in 8 samples
(runs/distill_followup_2026-10-31/train_cliff.jsonl) and the 214 no trained model solved
(approach 1's never_solved rows, runs/distill_2026-10-31/targets.jsonl): 795 rows, 189 in both.
Each row gets 8 retrieved train examples from the val plans' planner (scripts/retrieval: BM25 over
instructions, the same model's template and token budget), with the row's whole leakage component
excluded (build_splits.build_components). A train row then sees train the way a val row does: no
example shares a component with it.

Arms (`run`, GPU), from one NF4 load of the base 27B. There is no adapter: the trained 27B was
fine-tuned on these rows' targets. Both arms sample with the model card's thinking settings, as in
the thinking run, so they differ only in thinking:

  nothink-retr8  the 8 retrieved examples, thinking off (control)
  think-retr8    the 8 retrieved examples, thinking on

Batches, the padding guard, halving on OOM, the paging guard, the deadline and the resume are
scripts/think27b's, whose functions this imports. report.py checks every matched answer on the
perturbed buffers (approach 1's check) before it counts as a label.

The loop breaker (scripts/think_loop, added 2026-11-01) stops a thinking row whose reasoning
starts repeating itself and answers it from one copy of the repeated block. It is on for a new
--run directory; a directory keeps the setting it started with.

    python scripts/think_label/label.py trigger
    python scripts/think_label/label.py plan
    python scripts/think_label/label.py run --deadline 1762088400
"""
import argparse
import datetime
import importlib.util
import json
import os
import pathlib
import random
import statistics
import sys
import time
import zlib

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # see baseline.py

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))

from evaluate import load_jsonl                                   # noqa: E402
from prompt import messages_for_case                             # noqa: E402


def load(name, path):
    """Import a module by path under a unique name: several folders have same-named files."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = load("think27b_think27b", REPO / "scripts" / "think27b" / "think27b.py")
RET = load("retrieval_retrieval", REPO / "scripts" / "retrieval" / "retrieval.py")
LB = load("think_loop_loop", REPO / "scripts" / "think_loop" / "loop.py")

RUN = REPO / "runs" / "think_label_2026-11-01"
TRAIN = REPO / "data" / "splits" / "train.jsonl"
CLIFF = REPO / "runs" / "distill_followup_2026-10-31" / "train_cliff.jsonl"
TARGETS = REPO / "runs" / "distill_2026-10-31" / "targets.jsonl"
VAL_PLANS = REPO / "runs" / "retrieval_2026-10-18" / "val_plans" / "plan_manifest.json"
THINK_REPORT = T.RUN / "report.json"
K = 8
SEED = 20261102
RC_SKIP = 3

# name: registry entry for the template. The order is the run order: the quick control first, so a
# deadline can only cut the thinking arm.
ARMS = {"nothink-retr8": T.PLAIN_MODEL, "think-retr8": T.THINK_MODEL}


# ------------------------------------------------------------ the trigger

def trigger(report_path=THINK_REPORT):
    """The thinking run's TB verdict ("helps", "tied", "hurts"), or None if it has none."""
    path = pathlib.Path(report_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("verdicts", {}).get("TB")


def cmd_trigger(args):
    tb = trigger(args.report)
    if tb == "helps":
        print("TB is helps: the labeling pass runs")
        return 0
    print("TB is %s: the labeling pass does not run (pre-registered)" % (tb or "missing"))
    return RC_SKIP


# ------------------------------------------------------------ rows and plans

def select_rows(cliff_path=CLIFF, targets_path=TARGETS, train_path=TRAIN):
    """[{id, in_cliff, in_never}] in train order."""
    cliff = {r["id"] for r in load_jsonl(str(cliff_path))}
    never = {r["id"] for r in load_jsonl(str(targets_path)) if r["reason"] == "never_solved"}
    order = [c["id"] for c in load_jsonl(str(train_path))]
    return [{"id": i, "in_cliff": i in cliff, "in_never": i in never} for i in order if i in cliff | never]


class ComponentExcluded:
    """A BM25 index whose top() leaves out the current query's leakage component. The retrieval
    planner calls index.top(query, k); set `exclude` before each row."""

    def __init__(self, bm25):
        self.bm25, self.exclude = bm25, frozenset()

    def top(self, query_tokens, k):
        return self.bm25.top(query_tokens, k, exclude=self.exclude)


def plan_rows(cases, train, count, budget, k=K):
    """The retrieval planner's rows for `cases` (train rows), each with its component excluded."""
    from build_splits import build_components
    comp = {}
    for members in build_components(train):
        members = frozenset(members)
        for i in members:
            comp[i] = members
    pos = {r["id"]: i for i, r in enumerate(train)}
    index = ComponentExcluded(RET.BM25([RET.tokens(c["instruction"]) for c in train]))
    out = []
    for case in cases:
        index.exclude = comp[pos[case["id"]]]
        row = RET.plan_case(case, train, "retrieved", k, count, budget, index)
        row["component_size"] = len(index.exclude)
        out.append(row)
    return out


def cmd_plan(args):
    run = pathlib.Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    rows = select_rows()
    train = load_jsonl(str(TRAIN))
    byid = {c["id"]: c for c in train}
    val_plans = json.loads(VAL_PLANS.read_text(encoding="utf-8"))
    count, _ = RET.tokenizer_counter(val_plans["model"])        # the val plans' template and budget
    plans = plan_rows([byid[r["id"]] for r in rows], train, count, val_plans["budget_tokens"])
    RET.write_jsonl(run / "plan_retr8.jsonl", plans)
    eff = [p["effective_k"] for p in plans]
    rep = {"rows": rows, "n": len(rows), "cliff": sum(r["in_cliff"] for r in rows),
           "never": sum(r["in_never"] for r in rows), "both": sum(r["in_cliff"] and r["in_never"] for r in rows),
           "sources": {"cliff": str(CLIFF.relative_to(REPO)), "never_solved": str(TARGETS.relative_to(REPO))},
           "plan": {"model": val_plans["model"], "budget_tokens": val_plans["budget_tokens"], "k": K,
                    "effective_k_mean": round(statistics.fmean(eff), 2), "rows_short_of_k": sum(e < K for e in eff),
                    "prompt_tokens_mean": round(statistics.fmean(p["prompt_tokens"] for p in plans), 1),
                    "prompt_tokens_max": max(p["prompt_tokens"] for p in plans),
                    "component_rows_excluded": sum(p["component_size"] - 1 for p in plans),
                    "same_shape_neighbor_rows": sum(p["same_shape_neighbor"] for p in plans)}}
    (run / "rows.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print("%d rows: %d the 4B never solves, %d no trained model solves, %d both; effective k %.2f, "
          "prompt tokens mean %.0f max %d" % (rep["n"], rep["cliff"], rep["never"], rep["both"],
                                              rep["plan"]["effective_k_mean"], rep["plan"]["prompt_tokens_mean"],
                                              rep["plan"]["prompt_tokens_max"]))
    return 0


def order_for(arm, ids):
    """A fixed shuffle per arm, so rows cut by a deadline are a random subset."""
    out = list(ids)
    random.Random("%s|%s" % (SEED, arm)).shuffle(out)
    return out


def batch_seed(arm, first_id):
    """Keyed by the arm and the batch's first row, so a resumed run draws the same seeds."""
    return SEED + 100000 * list(ARMS).index(arm) + zlib.crc32(first_id.encode()) % 100000


# ------------------------------------------------------------ the GPU stage

def load_model():
    """The NF4 base 27B with no adapter, its input embedding in system RAM."""
    import torch
    from transformers import AutoTokenizer
    from baseline import load_base, load_registry, stop_token_ids
    regs = {n: load_registry(n) for n in (T.THINK_MODEL, T.PLAIN_MODEL)}
    a, b = regs[T.THINK_MODEL], regs[T.PLAIN_MODEL]
    if (a["hf_id"], a["revision"]) != (b["hf_id"], b["revision"]):
        sys.exit("%s and %s must name the same weights" % (T.THINK_MODEL, T.PLAIN_MODEL))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a["hf_id"], revision=a["revision"])
    tok.padding_side = "left"
    model = load_base(a, True)
    Q = load("qlora27b_train_27b", REPO / "scripts" / "qlora27b" / "train_27b.py")
    moved = Q.move_input_embedding_to_ram(model)
    gen = model.generation_config
    sampling = {"temperature": gen.temperature, "top_p": gen.top_p, "top_k": gen.top_k}
    if sampling != T.PREREGISTERED_SAMPLING:
        sys.exit("the model card's sampling is %s, not the pre-registered %s" % (sampling, T.PREREGISTERED_SAMPLING))
    model.eval()
    stops = set(stop_token_ids(tok, a["stop"]))
    eos = gen.eos_token_id
    stops |= set(eos if isinstance(eos, list) else [eos])
    info = {"load_s": round(time.time() - t0), "embedding_in_ram_gib": round(moved, 2),
            "allocated_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2),
            "sampling": sampling, "stop_ids": sorted(stops), "pad_id": tok.pad_token_id,
            "kwargs": {n: r.get("chat_template_kwargs") or {} for n, r in regs.items()},
            "hf_id": a["hf_id"], "revision": a["revision"], "adapter": None}
    print("loaded %s @ %s (NF4, no adapter) in %ds; %.2f GiB allocated (embedding %.2f GiB in RAM); "
          "sampling %s" % (a["hf_id"], a["revision"][:12], info["load_s"], info["allocated_gib"], moved, sampling),
          flush=True)
    return tok, model, info


def cmd_run(args):
    import torch
    run = pathlib.Path(args.run)
    rows = json.loads((run / "rows.json").read_text(encoding="utf-8"))["rows"]
    flags = {r["id"]: r for r in rows}
    train = {c["id"]: c for c in load_jsonl(str(TRAIN))}
    plan = {r["id"]: r for r in load_jsonl(str(run / "plan_retr8.jsonl"))}
    arms = [a for a in ARMS if a in args.arms]
    lb = LB.resolve(run, args.loop_break)
    print("loop breaker: %s" % (lb or "off"), flush=True)
    todo = {a: [i for i in order_for(a, flags) if i not in T.done_ids(run / ("rows_%s.jsonl" % a))] for a in arms}
    print("%s" % ", ".join("%s %d to do" % (a, len(t)) for a, t in todo.items()), flush=True)
    if not any(todo.values()):
        return 0
    if time.time() > args.deadline:
        print("deadline already passed: nothing started", flush=True)
        return 0
    tok, model, info = load_model()
    (run / ("load_%s.json" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))).write_text(
        json.dumps(info, indent=1) + "\n", encoding="utf-8")
    think_end_id = tok.convert_tokens_to_ids(T.THINK_END)
    stop = LB.stopping(tok, lb) if lb else None

    def examples(case):
        return [train[j] for j in plan[case["id"]]["examples"]]

    # The guard on the thinking arm's prompts. adapter_on=True everywhere: the plain base has no
    # adapter to disable.
    sample = [train[i] for i in order_for("guard", flags)[:args.batch_size]]
    texts = [tok.apply_chat_template(messages_for_case(c, examples(c)), add_generation_prompt=True, tokenize=False,
                                     **info["kwargs"][T.THINK_MODEL]) for c in sample]
    guard = T.padding_guard(model, tok, texts, True)
    batch_size = args.batch_size if guard <= args.guard_tol else 1
    print("guard: left-padded batch vs one row at a time, max |Δ log p| = %.4f -> batch size %d"
          % (guard, batch_size), flush=True)
    with (run / "guard.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"max_abs_dlogp": round(guard, 5), "tol": args.guard_tol, "batch_size": batch_size,
                             "time": datetime.datetime.now().isoformat(timespec="seconds")}) + "\n")
    torch.cuda.reset_peak_memory_stats()
    pace = []
    slow = 0
    for arm in arms:
        kwargs = info["kwargs"][ARMS[arm]]
        thinking = bool(kwargs.get("enable_thinking"))
        max_new = args.max_new_tokens if thinking else args.max_new_plain
        ids = todo[arm]
        with (run / ("rows_%s.jsonl" % arm)).open("a", encoding="utf-8") as fh:
            for b in range(0, len(ids), batch_size):
                if time.time() > args.deadline:
                    print("deadline reached in %s after %d of %d rows; the report uses what is done"
                          % (arm, b, len(ids)), flush=True)
                    return 0
                batch = [train[i] for i in ids[b:b + batch_size]]
                exs = [examples(c) for c in batch]
                texts = [tok.apply_chat_template(messages_for_case(c, e), add_generation_prompt=True,
                                                 tokenize=False, **kwargs) for c, e in zip(batch, exs)]
                seed = batch_seed(arm, batch[0]["id"])
                t0 = time.time()
                outs, lens = T.generate(model, tok, texts, True, max_new, info, seed,
                                        stopping_criteria=stop if thinking else None)
                secs = time.time() - t0
                steps = max(lens)
                texts_out = [tok.decode(o, skip_special_tokens=False) for o in outs]
                splits = [T.split_reasoning(t, thinking) for t in texts_out]
                loops = [None] * len(batch)
                for j, (o, (reasoning, answer, closed)) in enumerate(zip(outs, splits)):
                    if lb and thinking and not closed and len(o) < max_new:   # stopped early without closing
                        forced, loops[j] = LB.break_loop(T.generate, model, tok, texts[j], reasoning, True, info,
                                                         seed, lb)
                        if loops[j]:
                            splits[j] = (reasoning, forced, True)
                parsed, per = T.judge(batch, [s[1] for s in splits], args.timeout)
                stamp = datetime.datetime.now().isoformat(timespec="seconds")
                for c, e, o, n, (reasoning, answer, closed), loop in zip(batch, exs, outs, lens, splits, loops):
                    p, r = parsed[c["id"]], per.get(c["id"], {})
                    fh.write(json.dumps({
                        "arm": arm, "id": c["id"], "tier": c.get("tier"), "category": c.get("category"),
                        "in_cliff": flags[c["id"]]["in_cliff"], "in_never": flags[c["id"]]["in_never"],
                        "k_examples": len(e), "new_tokens": len(o),
                        "think_tokens": T.think_tokens(o, think_end_id) if thinking else 0,
                        "closed": closed, "hit_cap": n >= max_new and not closed,
                        "answer": answer, "reasoning": reasoning, "parse_ok": p is not None,
                        "mode": p[0] if p else None, "command": p[1] if p else None,
                        "valid": bool(r.get("valid")), "match": bool(r.get("match")), "error": r.get("error"),
                        "batch_seconds": round(secs, 1), "batch_size": len(batch), "steps": steps,
                        "seed": seed, "max_new_tokens": max_new, "loop_break": lb if thinking else None,
                        "forced_close": loop is not None, "loop": loop, "time": stamp}) + "\n")
                fh.flush()
                done = b + len(batch)
                print("  %s %d/%d  %d steps in %.0f s (%.2f s/step)  matched %d/%d  think tokens %s  "
                      "peak %.1f GiB" % (arm, done, len(ids), steps, secs, secs / max(steps, 1),
                                         sum(per.get(c["id"], {}).get("match", False) for c in batch), len(batch),
                                         [T.think_tokens(o, think_end_id) for o in outs] if thinking else "-",
                                         torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
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
    t = sub.add_parser("trigger", help="exit 0 iff the thinking run's TB reading is helps (else exit 3)")
    t.add_argument("--report", default=str(THINK_REPORT))
    r = sub.add_parser("plan", help="CPU: the rows and their retrieved examples -> <run>/rows.json, plan_retr8.jsonl")
    r.add_argument("--run", default=str(RUN))
    g = sub.add_parser("run", help="GPU: the two arms, resumable, until --deadline")
    g.add_argument("--run", default=str(RUN))
    g.add_argument("--deadline", type=float, required=True, help="epoch seconds; no batch starts after it")
    g.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    g.add_argument("--batch-size", type=int, default=8)
    g.add_argument("--max-new-tokens", type=int, default=4096, help="the thinking arm: reasoning + answer")
    g.add_argument("--max-new-plain", type=int, default=192, help="the control (§6 answers)")
    g.add_argument("--timeout", type=float, default=5.0, help="sandbox seconds per command")
    g.add_argument("--guard-tol", type=float, default=0.1, help="largest |Δ log p| allowed for batching")
    g.add_argument("--loop-break", action=argparse.BooleanOptionalAction, default=None,
                   help="scripts/think_loop's loop breaker on the thinking arm; default: on for a new --run "
                        "directory, else what the directory ran with")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "trigger":
        return cmd_trigger(args)
    if args.cmd == "plan":
        return cmd_plan(args)
    try:
        return cmd_run(args)
    except T.SlowCard as e:
        print("stopped: %s" % e, flush=True)
        return T.RC_SLOW


if __name__ == "__main__":
    sys.exit(main())
