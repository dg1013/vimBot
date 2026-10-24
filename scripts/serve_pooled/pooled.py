"""The pooled pick through the two vLLM endpoints (serve_pooled.sh).

One request:
  1. 8 samples from the 4B and 8 from the 9B, at T = 1, top-p 1, requested
     concurrently (two endpoints, one card);
  2. each sample parsed by the §6 grammar, duplicates merged across models;
  3. every distinct candidate scored by the verifier (a LoRA on the 4B
     endpoint) in one batched request: logit(yes) - logit(no) after the
     yes/no restriction, read with --logprobs-mode processed_logits;
  4. the highest score wins; ties by sample count, then mode and command text.
Nothing is executed (the verifier reads text), so the pick stays inside the
§1 execution boundary. Prompts are rendered as the harness renders them
(scripts/serve/endpoint.Endpoint), so served prompts are token-identical to
the measured ones.
"""
import json
import pathlib
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "serve", REPO / "scripts" / "reranker"):
    sys.path.insert(0, str(p))

YES, NO = 9405, 2083          # the verifier's answer tokens (verifier_lib.answer_token_ids for Qwen3.5)


def post(url, body, timeout=300.0):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick(cands):
    """The winning candidate: highest score; ties by count, then mode and text."""
    return max(cands, key=lambda c: (c["score"], c["n"], c["mode"], c["command"])) if cands else None


def merge(samples):
    """[(model, mode, command)] -> distinct candidates with counts per model."""
    out = {}
    for model, mode, cmd in samples:
        k = (mode, cmd)
        if k not in out:
            out[k] = {"mode": mode, "command": cmd, "n": 0, "from": {}}
        out[k]["n"] += 1
        out[k]["from"][model] = out[k]["from"].get(model, 0) + 1
    return list(out.values())


class Pooled:
    def __init__(self, e4, e9, verifier_name="verifier", policy9_name="served9b", k=8, k9=None):
        """k samples from the 4B and k9 from the 9B (k9 None: the same k for both)."""
        self.e4, self.e9, self.k, self.k9 = e4, e9, k, k9
        self.verifier_name = verifier_name
        self.e9.name = policy9_name
        self.score_url = e4.url
        from verifier_lib import verifier_messages
        self.verifier_messages = verifier_messages
        self.pool = ThreadPoolExecutor(max_workers=2)

    def k_for(self, model):
        return self.k9 if model == "9b" and self.k9 is not None else self.k

    def _sample(self, ep, case, k=None):
        from prompt import messages_for_case
        if k is None:
            k = self.k_for("9b" if ep is self.e9 else "4b")
        return ep.complete(ep.render(messages_for_case(case)), max_tokens=192, temperature=1.0, n=k)

    def scores(self, case, cands):
        """Verifier scores of candidates, one batched request on the 4B endpoint."""
        if not cands:
            return [], 0.0
        texts = [self.e4.render(self.verifier_messages(case, c["mode"], c["command"])) for c in cands]
        t0 = time.perf_counter()
        out = post(self.score_url, {"model": self.verifier_name, "prompt": texts, "max_tokens": 1, "temperature": 0,
                                    "logprobs": 2, "allowed_token_ids": [YES, NO], "skip_special_tokens": True})
        dt = time.perf_counter() - t0
        res = []
        for ch in sorted(out["choices"], key=lambda c: c["index"]):
            top = ch["logprobs"]["top_logprobs"][0]
            yes = top.get("yes", top.get("Yes"))
            no = top.get("no", top.get("No"))
            if yes is None or no is None:
                raise ValueError("scoring returned %r, not both answer tokens" % (list(top),))
            res.append(yes - no)
        return res, dt

    def answer(self, case, models=("4b", "9b")):
        """The pooled pick (or a single model's pick) for one case, with timings."""
        from prompt import parse_output
        t0 = time.perf_counter()
        futs = {}
        if "4b" in models:
            futs["4b"] = self.pool.submit(self._sample, self.e4, case)
        if "9b" in models:
            futs["9b"] = self.pool.submit(self._sample, self.e9, case)
        outs = {m: f.result() for m, f in futs.items()}
        t_sample = time.perf_counter() - t0
        samples = []
        for m, o in outs.items():
            for text in o["texts"]:
                p = parse_output(text)
                if p:
                    samples.append((m, p[0], p[1]))
        cands = merge(samples)
        sc, t_score = self.scores(case, cands)
        for c, s in zip(cands, sc):
            c["score"] = s
        best = pick(cands)
        return {"id": case.get("id"), "mode": best["mode"] if best else None, "command": best["command"] if best else None,
                "candidates": cands, "latency_s": round(time.perf_counter() - t0, 4),
                "sample_s": round(t_sample, 4), "score_s": round(t_score, 4),
                "output_tokens": {m: sum(o["output_tokens"]) for m, o in outs.items()},
                "prompt_tokens": {m: o["prompt_tokens"] for m, o in outs.items()}}
