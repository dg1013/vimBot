"""The loop breaker for the thinking 27B: a reasoning that starts repeating itself is stopped, closed and answered.

Built after the 16k stage, for future runs. Two of the 81 recorded
thinking traces (scripts/think27b, scripts/think_long) never finished. Each ended in one block of
lines said hundreds of times: hc-2179 from token 5,469 to the 16,384 cap, hc-0904 from token 6,603.
No other trace says a block more than twice in a row. A higher cap only buys more repetition, so:

1. `stopping()` watches each row as it is generated and stops it once its reasoning ends in a loop
   (`find_loop`): its last `reps` x p complete lines are one block of p <= `period_max` lines,
   said `reps` times in a row (blank lines skipped, runs of whitespace collapsed, at least
   `min_chars` non-space characters per block);
2. `break_loop()` cuts the reasoning back to one copy of the block, closes it the way the chat
   template closes reasoning (CLOSE), and generates the answer within `answer_tokens`, the
   controls' budget.

The answer is then judged like any other. A run directory records its settings in
loop_break.json (`resolve()`): a new directory takes the default, and a directory keeps what it
started with, so an arm never mixes rows with and without the breaker. A directory that already
holds rows and no record ran before the breaker existed: off.

calibrate.py replays the detector over the recorded traces (runs/think_loop_2026-11-01/).
"""
import datetime
import json
import pathlib
import re
import sys
import time

DEFAULT = {"period_max": 8, "reps": 8, "min_chars": 16, "answer_tokens": 192}
DETECTOR = ("period_max", "reps", "min_chars")
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CLOSE = "\n" + THINK_CLOSE + "\n\n"      # chat_template.jinja: '<think>\n' + reasoning + '\n</think>\n\n' + answer
WINDOW = 4096                             # the newest tokens the stopping check decodes per row
SETTINGS = "loop_break.json"


def detector(settings):
    return {k: settings[k] for k in DETECTOR}


# ------------------------------------------------------------ the detector

def lines_of(text):
    """The complete non-blank lines of `text` as (start, end, key): character offsets, and the line
    with runs of whitespace collapsed. A last line without its newline is not complete yet."""
    out = []
    for m in re.finditer(r"[^\n]*\n", text[:text.rfind("\n") + 1]):
        key = " ".join(m.group().split())
        if key:
            out.append((m.start(), m.end(), key))
    return out


def loop_in(lines, period_max=8, reps=8, min_chars=16):
    """The loop that `lines` (from lines_of) end in, or None. The smallest period wins. `start` is
    where the repetition began, `keep` where its first full block ends (character offsets)."""
    n = len(lines)
    for p in range(1, period_max + 1):
        if n < p * reps:
            break
        if any(lines[n - k][2] != lines[n - k - p][2] for k in range(1, p * (reps - 1) + 1)):
            continue
        if sum(len(line[2].replace(" ", "")) for line in lines[n - p:]) < min_chars:
            continue
        j = n - p * reps
        while j > 0 and lines[j - 1][2] == lines[j - 1 + p][2]:
            j -= 1
        return {"period": p, "reps": round((n - j) / p, 2), "start": lines[j][0], "keep": lines[j + p - 1][1],
                "block": [line[2] for line in lines[j:j + p]]}
    return None


def find_loop(text, period_max=8, reps=8, min_chars=16):
    """The loop that `text`'s complete lines end in, or None."""
    return loop_in(lines_of(text), period_max, reps, min_chars)


# ------------------------------------------------------------ during generation

def stopping(tok, settings, window=WINDOW):
    """A transformers StoppingCriteriaList that stops each row once its reasoning ends in a loop.
    A row is checked only when its newest token holds a newline. Its reasoning is the text after
    the last <think> in its newest `window` tokens, and a row whose reasoning is closed is left
    alone. It keeps no per-row state, so it serves any batch, the halves of an out-of-memory retry
    included."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    kw = detector(settings)
    newline = {}

    class LoopStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            done = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
            for j, t in enumerate(input_ids[:, -1].tolist()):
                if t not in newline:
                    newline[t] = "\n" in tok.decode([t])
                if not newline[t]:
                    continue
                text = tok.decode(input_ids[j, -window:].tolist())
                if THINK_OPEN in text:
                    text = text[text.rfind(THINK_OPEN) + len(THINK_OPEN):]
                if THINK_CLOSE not in text and find_loop(text, **kw):
                    done[j] = True
            return done

    return StoppingCriteriaList([LoopStop()])


def break_loop(generate, model, tok, prompt, reasoning, adapter_on, info, seed, settings):
    """(answer, loop) for a reasoning that ends in a loop, else (None, None). The reasoning is cut
    to one copy of the block and closed, then `generate` (scripts/think27b's) writes the answer."""
    loop = find_loop(reasoning, **detector(settings))
    if loop is None:
        return None, None
    kept = reasoning[:loop["keep"]].rstrip("\n")
    t0 = time.time()
    outs, _ = generate(model, tok, [prompt + kept + CLOSE], adapter_on, settings["answer_tokens"], info, seed)
    secs = time.time() - t0

    def count(s):
        return len(tok(s, add_special_tokens=False)["input_ids"])

    return tok.decode(outs[0]), {
        "period": loop["period"], "reps": loop["reps"], "block": loop["block"],
        "start_tokens": count(reasoning[:loop["start"]]), "kept_tokens": count(kept),
        "dropped_tokens": count(reasoning[loop["keep"]:]), "answer_tokens": len(outs[0]),
        "answer_seconds": round(secs, 1)}


# ------------------------------------------------------------ per run directory

def resolve(run, requested=None):
    """The breaker's settings for run directory `run`, or None for off. `requested` is the
    --loop-break flag: None (not given), True or False. A new directory takes the request (on by
    default) and records it; a directory keeps what it started with, and asking it for the other
    exits."""
    run = pathlib.Path(run)
    path = run / SETTINGS
    want = None if requested is False else dict(DEFAULT)
    if path.exists():
        have = json.loads(path.read_text(encoding="utf-8"))["settings"]
    elif any(run.glob("rows_*.jsonl")):
        have = None                      # rows from before the breaker existed
    else:
        run.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"settings": want, "time": datetime.datetime.now().isoformat(timespec="seconds")},
                                   indent=1) + "\n", encoding="utf-8")
        return want
    if requested is not None and (want is None) != (have is None):
        sys.exit("%s ran %s the loop breaker; a new --run directory can change that"
                 % (run, "with" if have else "without"))
    return have
