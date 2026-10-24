"""Stage 2's two-turn rollout and its ask-cost reward (README.md).

`make_rollout` is a TRL `rollout_func` (trl 1.12: `rollout_func(prompts,
trainer)`, one entry per sample -- the repeat sampler lays each prompt's G
copies side by side -- returning `prompt_ids`, `completion_ids`, `logprobs`,
an optional `env_mask`, and extra fields that reach the reward function):

1. **Turn 1, sampled** at T = 1 under the offer prompt: all of the step's
   samples in one left-padded generate call, as TRL's own path would.
2. A sample that parses as `ask: <question>` gets the simulated user's reply
   from the bank (bank.py), and **turn 2 is generated greedily** from the
   official chat rendering of the dialogue -- exactly what
   scripts/clarify/dialogue.py serves.
3. The training sequence is **turn 1 alone**. `env_mask` keeps, for an ask,
   only its first token -- the decision -- and drops the question's words;
   a direct answer keeps all its tokens. Turn 2 is environment: its outcome
   is rewarded, its tokens are not trained. Two reasons. The chat template
   renders an earlier assistant turn without the empty think block the
   generating turn gets, so turn 1 + reply + turn 2 as one sequence is a
   context the model is never served (training turn 2 is stage 1's job,
   on the right rendering). And with the reply looked up rather than
   written to the question, the question's words do not move the reward,
   so their advantage would be pure noise.

`with_ask_cost` wraps the GRPO trainer's executor reward: it scores the
FINAL command (turn 2's when the sample asked) with the unchanged §9 table,
then subtracts the cost c for asking. A policy maximizing it asks exactly
when P(the dialogue's answer is right) - P(the direct answer is right) > c.

`logprobs` is None: TRL reads it only for vLLM importance sampling, which
this repo does not use (it computes the policy's own log-probabilities).
"""
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
for p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "clarify"):
    sys.path.insert(0, str(p))
from protocol import parse_turn1, turn2_messages      # noqa: E402


def user_turn(prompt):
    """The §6 user turn of a turn-1 prompt: the key that finds its case."""
    return prompt[-1]["content"]


def ask_mask(n_tokens, asked):
    """env_mask for one sample: an ask trains its first token only."""
    return ([1] + [0] * (n_tokens - 1)) if asked and n_tokens else [1] * n_tokens


def cut_at_eos(ids, eos):
    """Generated ids up to and including the first end-of-turn token; a
    sequence that never ended keeps everything (the reward calls it
    truncated)."""
    cut = next((i for i, t in enumerate(ids) if t == eos), None)
    return list(ids) if cut is None else list(ids[:cut + 1])


def make_rollout(tok, chat_kwargs, case_by_user, bank, max_new_tokens, temperature, stats=None):
    """A TRL rollout_func; `stats` (a dict) collects the step's bank hits."""
    eos, pad = tok.eos_token_id, tok.pad_token_id

    def render(messages):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       **chat_kwargs)

    def generate(model, texts, sample):
        import torch
        side = tok.padding_side
        tok.padding_side = "left"
        try:
            enc = tok(texts, return_tensors="pt", padding=True).to(next(model.parameters()).device)
        finally:
            tok.padding_side = side
        kw = dict(max_new_tokens=max_new_tokens, eos_token_id=eos, pad_token_id=pad)
        kw.update(dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
                  if sample else dict(do_sample=False))
        with torch.no_grad():
            out = model.generate(**enc, **kw)
        return [cut_at_eos(row, eos) for row in out[:, enc["input_ids"].shape[1]:].tolist()]

    def rollout(prompts, trainer):
        from train_sft import inference_mode
        model = trainer.accelerator.unwrap_model(trainer.model)
        texts = [render(p) for p in prompts]
        prompt_ids = [tok(t)["input_ids"] for t in texts]
        n = len(prompts)
        asked, final_ids, final_text = [0] * n, [None] * n, [None] * n
        jobs, sims = [], []
        with inference_mode(model):
            completion_ids = generate(model, texts, sample=True)
            for i, (prompt, ids) in enumerate(zip(prompts, completion_ids)):
                text = tok.decode(ids, skip_special_tokens=True)
                parsed = parse_turn1(text)
                if parsed and parsed[0] == "ask":
                    case = case_by_user[user_turn(prompt)]
                    reply, sim, source = bank.reply(case["id"], parsed[1])
                    sims.append(sim)
                    asked[i] = 1
                    jobs.append((i, render(turn2_messages(case, parsed[1], reply))))
                else:
                    final_ids[i], final_text[i] = ids, text
            if jobs:
                for (i, _), ids in zip(jobs, generate(model, [t for _, t in jobs], sample=False)):
                    final_ids[i], final_text[i] = ids, tok.decode(ids, skip_special_tokens=True)
        if stats is not None:
            hits = [s for s in sims if s is not None]
            stats.update(asks=len(jobs), bank_hits=len(hits), placebo=len(sims) - len(hits),
                         mean_similarity=statistics.fmean(hits) if hits else None)
        return {"prompt_ids": prompt_ids, "completion_ids": completion_ids, "logprobs": None,
                "env_mask": [ask_mask(len(ids), a) for ids, a in zip(completion_ids, asked)],
                "asked": asked, "final_ids": final_ids, "final_text": final_text}

    return rollout


def with_ask_cost(executor_reward, cost, stats=None):
    """The executor reward on the final command, minus `cost` per question."""

    def ask_cost_reward(prompts, completions, completion_ids, **kw):
        asked = kw.pop("asked")
        final_ids = kw.pop("final_ids")
        final_text = kw.pop("final_text")
        finals = [[{"role": "assistant", "content": t}] for t in final_text]
        base = executor_reward(prompts, finals, final_ids, **kw)
        rewards = [b - cost * a for b, a in zip(base, asked)]
        log_metric = kw.get("log_metric")
        if log_metric is not None:
            n = len(rewards)
            log_metric("ask/rate", sum(asked) / n)
            for name, pick in (("asked", 1), ("direct", 0)):
                got = [b for b, a in zip(base, asked) if a == pick]
                if got:
                    log_metric("ask/base_reward_%s" % name, statistics.fmean(got))
            classes = kw.get("note_class")
            if classes:
                for cls in sorted(set(classes)):
                    idx = [i for i, c in enumerate(classes) if c == cls]
                    log_metric("ask/rate_%s" % cls, sum(asked[i] for i in idx) / len(idx))
            if stats:
                for k in ("bank_hits", "placebo"):
                    log_metric("ask/%s" % k, stats.get(k, 0) / n)
                if stats.get("mean_similarity") is not None:
                    log_metric("ask/bank_similarity", stats["mean_similarity"])
        return rewards

    ask_cost_reward.__name__ = "executor_reward"     # TRL logs rewards by function name
    return ask_cost_reward
