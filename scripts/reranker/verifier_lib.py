"""The reranker's shared pieces: the verifier prompt, candidate selection rules,
metrics, and (GPU side) model loading and yes/no scoring.

The verifier reads what the policy reads -- instruction, cursor, buffer, from
src/prompt.py's §6 user block, byte for byte -- plus one candidate command, and
answers "yes" or "no": would running the candidate make exactly the requested
edit? It never sees the wanted buffer or the gold command, and it never runs
anything: at inference it judges text only, which is what keeps it inside the
§1 execution boundary (the sandbox may decide WHETHER to ask, never WHICH
answer to give; DECISIONS.md 2026-10-13). The executor's verdicts are used to
label training examples and to measure, as everywhere else in this repo.

The §6 contract (src/prompt.py) is not changed: this is a separate prompt for
a separate model call.
"""
import math
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from prompt import render_target, render_user                      # noqa: E402

SYSTEM_TEXT = (
    "You check vim commands. You are given an editing instruction, the cursor,\n"
    "the buffer, and one candidate command. Answer yes if running the candidate\n"
    "on the buffer makes exactly the requested edit, otherwise no. Answer with\n"
    "one word."
)
YES, NO = "yes", "no"
LABEL_STATES = {"match": 1, "valid_wrong": 0, "invalid": 0}   # parse/truncated never ranked


def verifier_messages(case, mode, command):
    """System + user turn for one (case, candidate). Never reads buffer_after or
    gold_command."""
    user = render_user(case["instruction"], case["buffer_before"], case.get("cursor", (1, 1)))
    user += "\nCandidate: " + render_target(mode, command)
    return [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": user}]


# ------------------------------------------------------------ selection rules

def pick(cands, key):
    """The candidate a rule picks: max by key; ties by sample count, then greedy,
    then command text (deterministic)."""
    return max(cands, key=lambda c: (key(c), c.get("n", 0), bool(c.get("greedy")),
                                     c["cand_mode"], c["cand_command"]))


def majority_value(cands):
    """Expected correctness of majority voting by command text over the sampled
    candidates, ties split evenly (no tie-break information is used)."""
    sampled = [c for c in cands if c.get("n", 0) > 0]
    if not sampled:
        return None
    top = max(c["n"] for c in sampled)
    tied = [c for c in sampled if c["n"] == top]
    return sum(c["label"] for c in tied) / len(tied)


def group(examples):
    """{id: [candidate rows]} preserving file order."""
    out = {}
    for e in examples:
        out.setdefault(e["id"], []).append(e)
    return out


def rule_outcomes(groups, scores=None, lam=0.0):
    """Per prompt, the label each rule's pick carries. scores: {(id, mode, cmd): s}."""
    out = {}
    for cid, cands in groups.items():
        sampled = [c for c in cands if c.get("n", 0) > 0]
        greedy = [c for c in cands if c.get("greedy")]
        row = {"greedy": greedy[0]["label"] if greedy else None,
               "majority": majority_value(cands),
               "oracle_samples": int(any(c["label"] for c in sampled)) if sampled else None,
               "oracle_all": int(any(c["label"] for c in cands))}
        if scores is not None:
            def s(c):
                v = scores[(cid, c["cand_mode"], c["cand_command"])]
                return v + lam * math.log(max(c.get("n", 0), 1))
            row["verifier_samples"] = pick(sampled, s)["label"] if sampled else None
            row["verifier_all"] = pick(cands, s)["label"]
        out[cid] = row
    return out


# ------------------------------------------------------------ statistics

def auroc(pos, neg):
    """P(score of a positive > score of a negative), ties count half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2                  # mean 1-based rank of the tie block
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0)


def mcnemar_p(gained, lost):
    """Exact two-sided McNemar (binomial on the discordant pairs)."""
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a, b):
    """Gained / lost of b over a on prompts where both are defined (0/1 values)."""
    ids = [i for i in a if a[i] is not None and b.get(i) is not None]
    g = sum(1 for i in ids if b[i] and not a[i])
    l = sum(1 for i in ids if a[i] and not b[i])
    return g, l, len(ids)


# ------------------------------------------------------------ GPU side

def answer_token_ids(tok, chat_kwargs):
    """The first completion token of "yes" and of "no", found the way TRL splits
    prompt and completion: render prompt-with-generation-prompt and
    prompt+answer, take the first token after the shared prefix."""
    msgs = [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": "x"}]
    prompt = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, **chat_kwargs)
    prompt = _ids(prompt)
    out = []
    for ans in (YES, NO):
        full = _ids(tok.apply_chat_template(msgs + [{"role": "assistant", "content": ans}],
                                            tokenize=True, **chat_kwargs))
        if full[:len(prompt)] != prompt:
            raise RuntimeError("the chat template does not extend the generation prompt; "
                               "cannot locate the answer token")
        out.append(full[len(prompt)])
    if out[0] == out[1]:
        raise RuntimeError("yes and no share a first token")
    return tuple(out)


def _ids(x):
    if isinstance(x, dict) or hasattr(x, "keys"):
        x = x["input_ids"]
    if x and isinstance(x[0], list):
        x = x[0]
    return list(x)


def load_policy_merged(reg, init_adapter=None):
    """The registry base in bf16 on the GPU, with a policy adapter merged in
    when given (the verifier starts from the policy's task knowledge)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from baseline import load_base
    model = load_base(reg, False)
    if init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(init_adapter)).merge_and_unload()
    return model


def check_scoring(model, tok, rows, cases, chat_kwargs, yes_id, no_id, tol=1.0):
    """Guard: the batched, right-padded, two-row scorer must agree with plain
    full-logit scoring of each candidate alone. Returns the largest difference;
    raises if it exceeds tol. bf16 rounding of a logit near 40 is 0.25 per ulp
    (measured on the 4B smoke: 0.25 exactly); a wrong position or padding
    leaking into the recurrent state moves the score by whole units."""
    import torch
    fast, _ = score_examples(model, tok, rows, cases, chat_kwargs, yes_id, no_id, batch_size=len(rows))
    lm = _causal_lm(model)
    worst = 0.0
    for r, f in zip(rows, fast):
        ids = torch.tensor([_ids(tok.apply_chat_template(
            verifier_messages(cases[r["id"]], r["cand_mode"], r["cand_command"]),
            tokenize=True, add_generation_prompt=True, **chat_kwargs))], device=lm.device)
        with torch.no_grad():
            lg = model(input_ids=ids, use_cache=False).logits[0, -1].float()
        worst = max(worst, abs(float(lg[yes_id] - lg[no_id]) - f))
    if worst > tol:
        raise RuntimeError("batched verifier scores differ from full-logit scores by %.3f" % worst)
    return worst


def _causal_lm(model):
    """The transformers CausalLM inside a PeftModel (LoRA layers stay active)."""
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def score_examples(model, tok, rows, cases, chat_kwargs, yes_id, no_id, batch_size=16,
                   max_length=1024):
    """logit(yes) - logit(no) at the answer position for each example row.

    Only two rows of the output layer are needed, so the decoder's last hidden
    state at the answer position is multiplied by the yes/no rows of lm_head:
    a full-vocabulary logits tensor (~248k x length x batch) would cost
    gigabytes for two numbers. Right padding, so a sequence's last real token
    never attends to (or carries recurrent state from) a pad. Rows longer than
    max_length are scored on their left-truncated prompt and counted."""
    import torch
    model.eval()
    lm = _causal_lm(model)
    decoder = lm.model
    head = lm.get_output_embeddings().weight[[yes_id, no_id]]          # [2, hidden]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out, truncated = [], 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        seqs = []
        for r in batch:
            ids = _ids(tok.apply_chat_template(
                verifier_messages(cases[r["id"]], r["cand_mode"], r["cand_command"]),
                tokenize=True, add_generation_prompt=True, **chat_kwargs))
            if len(ids) > max_length:
                ids = ids[-max_length:]
                truncated += 1
            seqs.append(ids)
        width = max(len(s) for s in seqs)
        input_ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        attn = torch.zeros((len(seqs), width), dtype=torch.long)
        for i, s in enumerate(seqs):                       # right padding
            input_ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
        device = head.device
        input_ids, attn = input_ids.to(device), attn.to(device)
        with torch.no_grad():
            hidden = decoder(input_ids=input_ids, attention_mask=attn, use_cache=False).last_hidden_state
            last = attn.sum(dim=1) - 1
            h = hidden[torch.arange(len(seqs), device=device), last]          # [batch, hidden]
            two = (h.to(head.dtype) @ head.T).float()                          # [batch, 2]
        out.extend((two[:, 0] - two[:, 1]).tolist())
        del hidden, h, two
    return out, truncated
