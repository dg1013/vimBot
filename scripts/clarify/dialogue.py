#!/usr/bin/env python3
"""The student's turns of the clarification experiment (README.md beside it).

`--phase turn1` makes one pass over a case file with the student loaded and
records, for every case, everything that does not need the simulated user:

    direct    the §6 prompt through baseline.generate_predictions -- the
              results-table path itself, so this condition IS the baseline
    offer     the first-token distribution under OFFER_SYSTEM (§6 text plus
              the ask clause): P(ask) beside P(ex) and P(keys), and whether
              greedy decoding would ask
    noask     the greedy answer under OFFER_SYSTEM with "ask" banned at the
              first token: what the model answers when it may not ask
    question  "ask:" written into the reply, then the model's first line:
              the question it asks when it is made to ask
    placebo   turn 2 after a reply that carries no information

`--phase turn2` answers after the simulated user's reply (simulate.py).

Recording both the asked and the unasked outcome for every case is the
design: any rule for *when* to ask -- the model's own greedy choice, a
threshold on P(ask), random asking at the same rate, an oracle -- is then
scored offline by picking, row by row, which of the two outcomes it gets.

Decoding is the baseline's: greedy, batch size 1, the registry's stops, 192
new tokens for an answer; a question is cut at its first newline (64 tokens
at most). The 4B ships no generation config, so greedy applies no penalties
and the raw first-token argmax is exactly the token greedy picks.

Resumable and pausable like the difficulty probe: ids already in --out are
skipped; the stop file (or SIGTERM) finishes the case in flight, writes
PAUSED beside the stop file and exits 3.
"""
import argparse
import json
import math
import os
import pathlib
import sys
import time

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")   # as scripts/baseline.py
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl                                          # noqa: E402
from baseline import (base_quant_name, generate_predictions, load_base,  # noqa: E402
                      load_registry, resolve_base_quant, stop_token_ids)
from probe_difficulty import PAUSED_RC, PauseRequest, already_probed    # noqa: E402
from prompt import messages_for_case                                    # noqa: E402
from protocol import (ASK_PREFIX, PLACEBO_REPLY, QUESTION_MAX_TOKENS,   # noqa: E402
                      cut_question, parse_turn1, parse_turn2, turn1_messages,
                      turn2_messages)

ANSWER_MAX_TOKENS = 192          # scripts/baseline.py's default, for the same reason


def load_student(model_name, adapter, base_dtype="auto"):
    """The registry's base, plus the adapter if one is given -- baseline.py's loader."""
    from transformers import AutoTokenizer
    reg = load_registry(model_name)
    if base_dtype == "auto":
        nf4 = resolve_base_quant(adapter) if adapter else False
    else:
        nf4 = base_dtype == "nf4"
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    model = load_base(reg, nf4)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return reg, tok, model, base_quant_name(nf4)


def single_token(tok, s):
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        sys.exit("%r is %d tokens for this tokenizer; P(ask) needs one" % (s, len(ids)))
    return ids[0]


def ban_at_first(ban_ids, n_prompt):
    """A logits processor that bans ``ban_ids`` at the first generated position.

    It returns a modified *copy*. transformers hands a processor the very
    tensor it keeps as the step's raw logits, so an in-place ban leaks into
    them: the first full run (2026-10-06) measured log P(ask) = -inf on every
    case that way, and the smoke's "neither model asks" was that bug.
    """
    from transformers import LogitsProcessor

    class BanAtFirst(LogitsProcessor):
        def __call__(self, input_ids, scores):
            if input_ids.shape[1] != n_prompt:
                return scores
            scores = scores.clone()
            scores[:, list(ban_ids)] = float("-inf")
            return scores

    return BanAtFirst()


class Decoder:
    """Greedy completion of a rendered prompt, with the two hooks the
    experiment needs: a ban on given tokens at the first position, and a
    stop at the first newline."""

    def __init__(self, model, tok, reg):
        import torch
        self.torch = torch
        self.model, self.tok, self.reg = model, tok, reg
        self.stops = stop_token_ids(tok, reg.get("stop"))
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.device = next(model.parameters()).device
        self.kwargs = reg.get("chat_template_kwargs", {})
        self.ids = {s: single_token(tok, s) for s in ("ask", "Ask", "ex", "keys")}

    def render(self, messages, prefill=""):
        return self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True,
                                            **self.kwargs) + prefill

    def first_logits(self, text):
        """The next-token logits after the prompt, from a forward pass of its own.

        Measured apart from any generation, so no logits processor can touch
        it (see ban_at_first).
        """
        enc = self.tok(text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            return self.model(**enc).logits[0, -1].float()

    def complete(self, text, max_new_tokens=ANSWER_MAX_TOKENS, ban_first=(),
                 stop_at_newline=False):
        torch = self.torch
        from transformers import LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        enc = self.tok(text, return_tensors="pt").to(self.device)
        n_prompt = int(enc["input_ids"].shape[1])
        tok = self.tok

        class NewlineStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kw):
                done = (input_ids.shape[1] > n_prompt
                        and "\n" in tok.decode(input_ids[0, -1:]))
                return torch.full((input_ids.shape[0],), done, dtype=torch.bool,
                                  device=input_ids.device)

        kw = dict(max_new_tokens=max_new_tokens, do_sample=False,
                  eos_token_id=self.stops, pad_token_id=self.pad)
        if ban_first:
            kw["logits_processor"] = LogitsProcessorList([ban_at_first(ban_first, n_prompt)])
        if stop_at_newline:
            kw["stopping_criteria"] = StoppingCriteriaList([NewlineStop()])
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        latency = time.perf_counter() - t1
        gen = out[0, n_prompt:].tolist()
        first_id = gen[0] if gen else None
        if gen and gen[-1] in self.stops:
            gen = gen[:-1]
        if stop_at_newline and gen and "\n" in tok.decode(gen[-1:]):
            gen = gen[:-1]                        # the newline ends the line; not content
        return {"completion": tok.decode(gen, skip_special_tokens=True),
                "output_tokens": len(gen), "prompt_tokens": n_prompt,
                "latency_s": round(latency, 4), "first_id": first_id}

    def first_token_view(self, logits):
        """P(ask) beside P(ex), P(keys); the top token; whether greedy asks.

        ``ask_margin`` is the log-odds of asking against answering, within the
        three forms: log P(ask) - log(P(ex) + P(keys)). It ranks cases like
        P(ask) does, but in log space, so a model that almost never asks still
        ranks its cases instead of underflowing to a tie at 0.
        """
        torch = self.torch
        logp = torch.log_softmax(logits, -1)
        lp = {s: float(logp[i]) for s, i in self.ids.items()}
        ask = float(torch.logaddexp(logp[self.ids["ask"]], logp[self.ids["Ask"]]))
        answer = float(torch.logaddexp(logp[self.ids["ex"]], logp[self.ids["keys"]]))
        margin = ask - answer
        if not math.isfinite(margin):
            # A model's logits are finite: an infinite margin means the
            # measurement was tampered with (see ban_at_first). Stop the stage.
            raise RuntimeError("log-odds of asking is %r; the first-token logits are "
                               "not the model's own" % margin)
        top = int(logits.argmax())
        return {"p_ask": math.exp(ask), "p_ex": math.exp(lp["ex"]),
                "p_keys": math.exp(lp["keys"]), "ask_margin": margin,
                "p_ask_norm": 1 / (1 + math.exp(-margin)),
                "top": self.tok.decode([top]), "top_id": top,
                "chose_ask": top in (self.ids["ask"], self.ids["Ask"])}


def answer_fields(res, turn2=False):
    """An answer scored by the §6 grammar; an ``ask:`` here is a parse failure.

    ``turn2`` also records whether the failure was a second question, which
    the protocol does not allow.
    """
    parsed = parse_turn2(res["completion"])
    out = {k: res[k] for k in ("completion", "output_tokens", "prompt_tokens", "latency_s")}
    out.update(parse_ok=parsed is not None,
               mode=parsed[0] if parsed else None,
               command=parsed[1] if parsed else None)
    if turn2:
        again = parse_turn1(res["completion"])
        out["second_ask"] = bool(again and again[0] == "ask")
    return out


def turn1(dec, case):
    """Every turn-1 measurement for one case (see the module docstring)."""
    records, _ = generate_predictions(dec.model, dec.tok, [case], dec.reg,
                                      ANSWER_MAX_TOKENS)
    d = records[0]
    direct = {k: d[k] for k in ("completion", "output_tokens", "prompt_tokens",
                                "latency_s", "parse_ok", "mode", "command")}

    msgs = turn1_messages(case)
    offer = dec.first_token_view(dec.first_logits(dec.render(msgs)))
    ban = (dec.ids["ask"], dec.ids["Ask"])
    res = dec.complete(dec.render(msgs), ban_first=ban)
    noask = answer_fields(res)

    if offer["chose_ask"]:
        # Greedy would ask: keep its free completion, so the report can check
        # that the forced question below is the one it would have asked.
        free = dec.complete(dec.render(msgs))
        offer["free_completion"] = free["completion"]
        offer["first_consistent"] = free["first_id"] == offer["top_id"]
    else:
        # The ban changes nothing when greedy would not ask, so generation's
        # own first token must be the separate forward pass's argmax.
        offer["first_consistent"] = res["first_id"] == offer["top_id"]

    q = dec.complete(dec.render(msgs, prefill=ASK_PREFIX),
                     max_new_tokens=QUESTION_MAX_TOKENS, stop_at_newline=True)
    text = cut_question(q["completion"])
    question = {"text": text, "raw": q["completion"],
                # "ask:" is two tokens the model would have said itself
                "output_tokens": q["output_tokens"] + len(dec.tok.encode(
                    ASK_PREFIX, add_special_tokens=False)),
                "latency_s": q["latency_s"]}

    placebo = turn2(dec, case, text, PLACEBO_REPLY)
    return {"direct": direct, "offer": offer, "noask": noask,
            "question": question, "placebo": placebo}


def turn2(dec, case, question, reply):
    """The answer after the user's reply: a command, or a parse failure."""
    return answer_fields(dec.complete(dec.render(turn2_messages(case, question, reply))),
                         turn2=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("turn1", "turn2"), required=True)
    ap.add_argument("--model", default="qwen3.5-4b", help="registry name")
    ap.add_argument("--adapter", default=None, help="adapter dir, or 'none' for the base")
    ap.add_argument("--label", required=True, help="the checkpoint's name in the report")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--replies", help="turn2: the simulator's replies (simulate.py --out)")
    ap.add_argument("--turn1", help="turn2: this label's turn-1 file (for the questions)")
    ap.add_argument("--out", required=True, help="JSONL, appended; resumable")
    ap.add_argument("--base-dtype", choices=("auto", "bf16", "nf4"), default="auto")
    ap.add_argument("--limit", type=int, default=None, help="first N cases (smoke)")
    ap.add_argument("--stop-file", default=None)
    args = ap.parse_args()
    adapter = None if args.adapter in (None, "none") else args.adapter

    cases = load_jsonl(args.cases)
    if args.limit:
        cases = cases[:args.limit]
    replies = questions = None
    if args.phase == "turn2":
        if not (args.replies and args.turn1):
            sys.exit("--phase turn2 needs --replies and --turn1")
        replies = {r["id"]: r for r in load_jsonl(args.replies) if r["label"] == args.label}
        questions = {r["id"]: r["question"]["text"] for r in load_jsonl(args.turn1)}
        missing = [c["id"] for c in cases if c["id"] not in replies]
        if missing:
            sys.exit("%d cases have no reply for label %s (first: %s); run simulate.py first"
                     % (len(missing), args.label, missing[0]))
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = already_probed(out_path)
    todo = [c for c in cases if c["id"] not in done]
    print("clarify %s [%s]: %d cases, %d already in %s, %d to do" % (
        args.phase, args.label, len(cases), len(done), out_path, len(todo)), flush=True)
    if not todo:
        return 0

    import torch
    t0 = time.time()
    reg, tok, model, base = load_student(args.model, adapter, args.base_dtype)
    print("loaded %s @ %s (%s base)%s in %.0f s" % (
        reg["hf_id"], reg["revision"][:12], base,
        " + adapter %s" % adapter if adapter else "", time.time() - t0), flush=True)
    dec = Decoder(model, tok, reg)
    if args.phase == "turn1":
        # Every new condition decodes through Decoder, the direct one through
        # the baseline's own path. On the §6 prompt the two must agree token
        # for token, or a difference between conditions could be an artifact
        # of decoding rather than of the prompt.
        probe = todo[0]
        ref, _ = generate_predictions(model, tok, [probe], reg, ANSWER_MAX_TOKENS)
        mine = dec.complete(dec.render(messages_for_case(probe)))
        if mine["completion"] != ref[0]["completion"]:
            sys.exit("decoder self-check FAILED on %s: %r (Decoder) != %r (baseline)"
                     % (probe["id"], mine["completion"], ref[0]["completion"]))
        print("decoder self-check: identical to the baseline path on %s" % probe["id"],
              flush=True)
    pause = PauseRequest(args.stop_file)
    pause.install()

    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as fh:
        for i, case in enumerate(todo, 1):
            if args.phase == "turn1":
                row = turn1(dec, case)
            else:
                row = {"reply": replies[case["id"]]["reply"],
                       "answer": turn2(dec, case, questions[case["id"]],
                                       replies[case["id"]]["reply"])}
            row = {"id": case["id"], "label": args.label, **row}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_start) / i
                print("  %d/%d  %.1f s/case  eta %.0f min" % (
                    i, len(todo), rate, rate * (len(todo) - i) / 60), flush=True)
            if pause.check():
                marker = pause.marker()
                if marker:
                    marker.write_text("%s after %d cases\n" % (pause.requested, i))
                print("paused (%s) after %d cases" % (pause.requested, i), flush=True)
                return PAUSED_RC
            if i % 50 == 0:
                torch.cuda.empty_cache()
    print("done: %d cases in %.0f s -> %s" % (len(todo), time.time() - t_start, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
