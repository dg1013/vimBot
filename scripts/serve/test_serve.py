"""Tests for scripts/serve/ (the question round trip and the MCP server).

Run:  python3 -m pytest scripts/serve          (not collected by `pytest tests/`)

No GPU and no vLLM. A scripted endpoint stands in for the model, while
the sandbox is the real one (scripts/clarify/score.execute ->
evaluate.score -> run_vim), so the rule is tested on real executions:
- prompt rendering matches the evaluation harness token for token;
- agreeing samples answer and disagreeing ones ask;
- failed samples count against agreement;
- the reply closes the dialogue;
- the sandbox gets the settings file's timeout;
- the MCP tools round-trip in process.
"""
import asyncio
import pathlib
import sys

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify", HERE):
    sys.path.insert(0, str(p))

import protocol as P                                   # noqa: E402
import roundtrip as RT                                 # noqa: E402

BUFFER = "alpha beta\n"


class ScriptedEndpoint:
    """Answers by call kind: greedy §6, the samples, the question, turn 2."""
    name = "scripted"

    def __init__(self, direct="ex: s/alpha/x/", samples=None, question=" Which word?",
                 turn2="ex: s/beta/x/"):
        self.direct, self.samples = direct, samples or ["ex: s/alpha/x/"] * 8
        self.question, self.turn2, self.calls = question, turn2, []

    def render(self, messages, prefill=""):
        """Which system prompt, how many messages, the last one, the prefill."""
        tag = "OFFER" if messages[0]["content"] == P.OFFER_SYSTEM else "SEC6"
        return "%s/%d|%s|%s" % (tag, len(messages), messages[-1]["content"], prefill)

    def complete(self, text, max_tokens, temperature=0.0, n=1, stop=None, seed=None):
        if n > 1:
            kind, texts = "samples", self.samples[:n]
        elif stop == ["\n"]:
            kind, texts = "question", [self.question]
        elif text.startswith("OFFER/4|"):               # system, user, "ask: q", the reply
            kind, texts = "turn2", [self.turn2]
        else:
            kind, texts = "direct", [self.direct]
        self.calls.append((kind, text, temperature, n))
        return {"texts": texts, "output_tokens": [3] * len(texts), "prompt_tokens": 10,
                "latency_s": 0.001}


def _rt(**kw):
    return RT.RoundTrip(ScriptedEndpoint(**kw), samples=8, max_agree=5)


def test_agreeing_samples_answer_with_the_section6_answer():
    rt = _rt(samples=["ex: s/alpha/x/"] * 6 + ["keys: cwx<Esc>"] * 2)   # one result, 8 of 8
    out = rt.translate("Rename the first word", BUFFER)
    assert out["type"] == "command" and out["command"] == "s/alpha/x/" and out["mode"] == "ex"
    assert out["paste"] == ":s/alpha/x/"
    assert out["evidence"]["top"] == 8 and out["evidence"]["distinct_commands"] == 2
    assert out["rule"] == {"k": 8, "max_agree": 5, "ask": False}
    kinds = [c[0] for c in rt.ep.calls]
    assert sorted(kinds) == ["direct", "samples"]                       # no question asked
    samples_call = next(c for c in rt.ep.calls if c[0] == "samples")
    assert samples_call[2] == 1.0 and samples_call[3] == 8              # T = 1, k = 8


def test_disagreeing_samples_ask_and_the_reply_gives_the_command():
    rt = _rt(samples=["ex: s/alpha/x/"] * 4 + ["ex: s/beta/y/"] * 4)    # 4 + 4: top 4 <= 5
    out = rt.translate("Change a word", BUFFER, (1, 1))
    assert out["type"] == "question" and out["question"] == "Which word?"
    assert out["evidence"]["results"] == [4, 4] and out["rule"]["ask"] is True
    q_call = next(c for c in rt.ep.calls if c[0] == "question")
    assert q_call[1].endswith("|ask:")                                    # the "ask:" prefill
    assert q_call[1].startswith("OFFER/2|")                               # the clarify question prompt
    direct_call = next(c for c in rt.ep.calls if c[0] == "direct")
    assert direct_call[1].startswith("SEC6/2|") and direct_call[2] == 0.0   # §6, greedy
    done = rt.reply(out["dialogue_id"], "The second one, to x.")
    assert done["type"] == "command" and done["command"] == "s/beta/x/"
    assert done["question"] == "Which word?" and done["reply"] == "The second one, to x."
    turn2 = [c for c in rt.ep.calls if c[0] == "turn2"]
    case = rt.case("Change a word", BUFFER, (1, 1), out["dialogue_id"])
    expected = rt.ep.render(P.turn2_messages(case, "Which word?", "The second one, to x."))
    assert turn2[0][1] == expected                                         # the clarify dialogue
    assert rt.reply(out["dialogue_id"], "again")["type"] == "error"       # one question, then closed


def test_failed_samples_count_against_agreement():
    rt = _rt(samples=["ex: s/zzz/q/"] * 6 + ["ex: s/alpha/x/"] * 2)     # 6 fail in vim
    out = rt.translate("Rename it", BUFFER)
    assert out["type"] == "question"
    assert out["evidence"]["failed"] == 6 and out["evidence"]["top"] == 2
    rt2 = _rt(samples=["no command"] * 8)
    assert rt2.translate("Rename it", BUFFER)["evidence"]["unparsed"] == 8


def test_an_empty_question_falls_back_to_the_answer():
    rt = _rt(samples=["ex: s/alpha/x/"] * 4 + ["ex: s/beta/y/"] * 4, question="   ")
    out = rt.translate("Change a word", BUFFER)
    assert out["type"] == "command" and out["rule"]["fallback"]


def test_the_sandbox_is_the_validation_path_with_the_settings_timeout(monkeypatch):
    from settings import load_settings
    seen = {}
    rt = _rt()

    def spy(cases, answers, timeout=5.0, workers=8):
        seen["timeout"] = timeout
        import score
        return score.execute(cases, answers, timeout=timeout, workers=workers)
    rt.execute = spy
    rt.translate("Rename the first word", BUFFER)
    assert seen["timeout"] == load_settings(None)["executor"]["timeout"]
    import score
    assert RT.RoundTrip(ScriptedEndpoint()).execute is score.execute


def test_render_is_the_harness_render(monkeypatch):
    pytest.importorskip("transformers")
    import endpoint
    from dialogue import Decoder
    ep = endpoint.load("qwen3.5-4b")

    class OneParam:
        @staticmethod
        def parameters():
            import torch
            yield torch.zeros(1)
    dec = Decoder(OneParam(), ep.tok, ep.reg)
    case = RT.RoundTrip.case("Rename the first word", BUFFER, (1, 1), "x")
    from prompt import messages_for_case
    for msgs, prefill in ((messages_for_case(case), ""), (P.turn1_messages(case), "ask:"),
                          (P.turn2_messages(case, "Which?", "That one."), "")):
        assert ep.render(msgs, prefill) == dec.render(msgs, prefill)


def test_the_mcp_tools_round_trip_in_process():
    pytest.importorskip("mcp")
    from mcp.client.client import Client
    import mcp_server
    rt = _rt(samples=["ex: s/alpha/x/"] * 4 + ["ex: s/beta/y/"] * 4)
    server = mcp_server.build_server(rt)

    async def go():
        async with Client(server) as client:
            names = sorted(t.name for t in (await client.list_tools()).tools)
            first = await client.call_tool("vimBot_translate",
                                           {"instruction": "Change a word", "buffer": BUFFER})
            q = first.structured_content
            second = await client.call_tool("vimBot_reply",
                                            {"dialogue_id": q["dialogue_id"], "reply": "the second"})
            return names, q, second.structured_content
    names, q, done = asyncio.run(go())
    assert names == ["vimBot_reply", "vimBot_translate"]
    assert q["type"] == "question" and q["question"] == "Which word?"
    assert done["type"] == "command" and done["command"] == "s/beta/x/"
