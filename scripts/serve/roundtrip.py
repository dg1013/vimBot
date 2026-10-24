"""The question round trip (owner, 2026-10-13): answer, or ask first.

`RoundTrip.translate(instruction, buffer, cursor)` returns a command or a
question:

1. **The §6 answer**, greedy: the served row, unchanged.
2. **k answers sampled** at T = 1 (k = 8): the sample rule's evidence. The
   two requests go out together, since both are always needed.
3. **The sandbox** runs each distinct sampled command on a copy of the
   buffer through `evaluate.score`, the testing and validation path, with
   the settings file's executor timeout. This is CLAUDE.md §1's 2026-10-13
   exception: nothing is applied to the user's buffer, nothing is shown,
   and the sandbox never picks the answer.
4. **The rule.** When the most common result holds at most 5 of the 8
   samples (a sample that fails or changes nothing counts against
   agreement), it asks the clarify protocol's question: OFFER_SYSTEM, "ask:"
   prefilled, the first line. Otherwise it returns the §6 answer.

`RoundTrip.reply(dialogue_id, reply)` continues a question. Turn 2 is the
clarify dialogue (`protocol.turn2_messages`), greedy, read by the §6
grammar. One question at most; the dialogue closes after the reply.

This is the policy pre-registered and confirmed on test v2 on 2026-10-10
(+5.4, p 0.001), with the prompts it was measured on. Every result carries
its timings.
"""
import collections
import pathlib
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", REPO / "scripts" / "readings"):
    sys.path.insert(0, str(p))
from paste import pasteable                                            # noqa: E402
from prompt import messages_for_case, parse_output                     # noqa: E402
from protocol import (ASK_PREFIX, QUESTION_MAX_TOKENS, cut_question,   # noqa: E402
                      turn1_messages, turn2_messages)
import readings_protocol as R                                          # noqa: E402

ANSWER_MAX_TOKENS = 192          # dialogue.ANSWER_MAX_TOKENS, baseline.py's default


def sample_evidence(case, texts, execute, timeout):
    """What k sampled answers say: per distinct command its count and status,
    the result masses (largest first), and the most common result's count.

    `execute` is scripts/clarify/score.execute, which runs evaluate.score."""
    counts = collections.Counter()
    unparsed = 0
    for t in texts:
        p = parse_output(t)
        if p is None:
            unparsed += 1
        else:
            counts[p] += 1
    distinct = list(counts)
    answers = {"c%d" % i: {"parse_ok": True, "mode": m, "command": c}
               for i, (m, c) in enumerate(distinct)}
    executed = (execute([case], {case["id"]: answers}, timeout=timeout,
                        workers=max(1, min(8, len(answers)))) if answers else {})
    masses, failed, rows = collections.Counter(), unparsed, []
    for i, key in enumerate(distinct):
        e = executed.get((case["id"], "c%d" % i))
        status = R.reading_status(case, e)
        if status == "ok":
            masses[R.norm_key(e["buffer_after"])] += counts[key]
        else:
            failed += counts[key]
        rows.append({"mode": key[0], "command": key[1], "n": counts[key], "status": status})
    results = sorted(masses.values(), reverse=True)
    return {"k": len(texts), "distinct_commands": len(distinct), "unparsed": unparsed,
            "failed": failed, "results": results, "top": results[0] if results else 0,
            "commands": rows}


class RoundTrip:
    def __init__(self, endpoint, samples=8, max_agree=5, timeout=None, settings_path=None,
                 ttl_s=3600.0, seed=None):
        from settings import load_settings
        from score import execute
        self.ep, self.k, self.max_agree, self.seed = endpoint, samples, max_agree, seed
        # The executor timeout the way baseline.py resolves it: CLI > settings file > default.
        self.timeout = timeout if timeout is not None else \
            load_settings(settings_path)["executor"]["timeout"]
        self.execute, self.ttl = execute, ttl_s
        self._dialogues, self._lock = {}, threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4)

    # -- helpers ---------------------------------------------------------------
    @staticmethod
    def case(instruction, buffer, cursor, cid):
        return {"id": cid, "instruction": instruction, "buffer_before": buffer,
                "buffer_after": None, "cursor": [int(cursor[0]), int(cursor[1])],
                "category": "real-use"}

    def _command(self, completion, extra):
        parsed = parse_output(completion)
        if parsed is None:
            return dict(extra, type="error", message="the model's answer did not parse",
                        completion=completion)
        mode, command = parsed
        return dict(extra, type="command", mode=mode, command=command,
                    paste=pasteable(mode, command), completion=completion)

    def _sweep(self):
        now = time.time()
        with self._lock:
            for cid in [c for c, d in self._dialogues.items() if now - d["t"] > self.ttl]:
                del self._dialogues[cid]

    # -- the round trip --------------------------------------------------------
    def translate(self, instruction, buffer, cursor=(1, 1)):
        self._sweep()
        t0 = time.perf_counter()
        cid = uuid.uuid4().hex[:12]
        case = self.case(instruction, buffer, cursor, cid)
        prompt = self.ep.render(messages_for_case(case))
        direct_f = self._pool.submit(self.ep.complete, prompt, ANSWER_MAX_TOKENS)
        samples_f = self._pool.submit(self.ep.complete, prompt, ANSWER_MAX_TOKENS,
                                      temperature=1.0, n=self.k, seed=self.seed)
        direct, samples = direct_f.result(), samples_f.result()
        t1 = time.perf_counter()
        evidence = sample_evidence(case, samples["texts"], self.execute, self.timeout)
        t2 = time.perf_counter()
        timings = {"direct_s": direct["latency_s"], "samples_s": samples["latency_s"],
                   "models_wall_s": round(t1 - t0, 4), "sandbox_s": round(t2 - t1, 4)}
        rule = {"k": self.k, "max_agree": self.max_agree, "ask": evidence["top"] <= self.max_agree}
        info = {"dialogue_id": None, "rule": rule, "evidence": evidence, "timings": timings,
                "tokens": {"direct": direct["output_tokens"][0],
                           "samples": sum(samples["output_tokens"])}}
        if rule["ask"]:
            q = self.ep.complete(self.ep.render(turn1_messages(case), prefill=ASK_PREFIX),
                                 QUESTION_MAX_TOKENS, stop=["\n"])
            question = cut_question(q["texts"][0])
            timings["question_s"] = q["latency_s"]
            info["tokens"]["question"] = q["output_tokens"][0]
            if question:
                with self._lock:
                    self._dialogues[cid] = {"case": case, "question": question, "t": time.time()}
                timings["total_s"] = round(time.perf_counter() - t0, 4)
                return dict(info, type="question", question=question, dialogue_id=cid)
            rule["fallback"] = "empty question: answered directly"
        timings["total_s"] = round(time.perf_counter() - t0, 4)
        return self._command(direct["texts"][0], info)

    def reply(self, dialogue_id, reply):
        with self._lock:
            d = self._dialogues.pop(dialogue_id, None)
        if d is None:
            return {"type": "error", "message": "unknown or expired dialogue %r" % dialogue_id}
        t0 = time.perf_counter()
        r = self.ep.complete(self.ep.render(turn2_messages(d["case"], d["question"], reply)),
                             ANSWER_MAX_TOKENS)
        info = {"dialogue_id": dialogue_id, "question": d["question"], "reply": reply,
                "timings": {"turn2_s": r["latency_s"],
                            "total_s": round(time.perf_counter() - t0, 4)},
                "tokens": {"turn2": r["output_tokens"][0]}}
        return self._command(r["texts"][0], info)
