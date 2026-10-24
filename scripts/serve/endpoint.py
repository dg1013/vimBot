"""The served model, through vLLM's OpenAI-compatible completions endpoint.

Prompts are rendered here, the way the evaluation harness renders them
(scripts/clarify/dialogue.Decoder.render: the registry's chat template with
`add_generation_prompt` and the registry's template kwargs, then any
prefill), and sent as raw text to /v1/completions. That keeps every
served prompt token-identical to a measured one. The chat endpoint would
render a prefilled assistant turn as an *earlier* turn, without the empty
think block the generating turn gets, which is a prompt the model was never
measured on.

Stops are baseline.stop_token_ids (EOS plus the registry's single-token
stops), as in generation for the results table. Stdlib HTTP only.
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts"):
    sys.path.insert(0, str(p))


class EndpointError(RuntimeError):
    pass


class Endpoint:
    def __init__(self, base_url, served_name, reg, tok, timeout=120.0):
        from baseline import stop_token_ids
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.name, self.reg, self.tok, self.timeout = served_name, reg, tok, timeout
        self.kwargs = reg.get("chat_template_kwargs") or {}
        self.stop_ids = stop_token_ids(tok, reg.get("stop"))

    def render(self, messages, prefill=""):
        """Exactly dialogue.Decoder.render."""
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            **self.kwargs) + prefill

    def complete(self, text, max_tokens, temperature=0.0, n=1, stop=None, seed=None):
        """{"texts": [...], "output_tokens": [...], "prompt_tokens": int, "latency_s": float}.

        temperature 0 is greedy. Sampling is top_p 1 with no top-k, as in
        the GRPO probe that measured the sample rule."""
        body = {"model": self.name, "prompt": text, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": 1.0, "n": n,
                "stop_token_ids": self.stop_ids, "skip_special_tokens": True}
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(self.url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError) as exc:
            raise EndpointError("the model endpoint at %s did not answer: %s" % (self.url, exc))
        latency = time.perf_counter() - t0
        choices = sorted(out["choices"], key=lambda c: c["index"])
        texts = [c["text"] for c in choices]
        return {"texts": texts,
                "output_tokens": [len(self.tok.encode(t, add_special_tokens=False)) for t in texts],
                "prompt_tokens": out.get("usage", {}).get("prompt_tokens"),
                "latency_s": round(latency, 4)}


def load(model="qwen3.5-4b", base_url="http://127.0.0.1:8011", served_name=None):
    """An Endpoint for a registry model, its tokenizer the registry's own."""
    from transformers import AutoTokenizer
    from baseline import load_registry
    reg = load_registry(model)
    tok = AutoTokenizer.from_pretrained(reg["hf_id"], revision=reg["revision"])
    return Endpoint(base_url, served_name or "%s-served" % model, reg, tok)
