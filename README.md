# vimBot

Fine-tuning a small open LLM (large language model) to turn natural-language editing instructions
into vim commands — with an **execution-based evaluation harness** as the
foundation, and (phase 2) **reinforcement learning from verifiable rewards**,
where the reward is "the command actually produced the right buffer."

```
"delete every blank line"                      →  :g/^$/d
"change the text inside the quotes to error"   →  ci"error<Esc>
"replace cat with dog everywhere"              →  :%s/cat/dog/g
```

## Why this shape

Vim commands are a *verifiable* target: run the predicted command in a
sandboxed headless vim and diff the buffer. That one property gives us

- **honest evaluation** — semantic equivalence for free (`:1d` and `dd` both
  pass; exact-match metrics would wrongly fail one of them),
- **execution-validated synthetic data** — generated training pairs are kept
  only if they actually work,
- **a reward signal** — enabling GRPO-style RL (reinforcement learning) fine-tuning with no reward
  model and no human labels.

Taken together, this repo (repository) is a small but complete **verifiable RL
environment** — a task distribution with a controlled difficulty spread,
a sandboxed execution verifier, and a reward function with anti-gaming
terms — the same artifact class frontier labs now build and buy at scale
for post-training.

## Where it stands (2026-09-10)

| | |
|---|---|
| corpus | **7,541** validated cases, 17 categories × 3 tiers, every quota cell full |
| gold self-check | 7,541 / 7,541 — every case's own command reproduces its buffer |
| splits | train 6,551 · val 545 · test 445, frozen by id, no template or command spanning two splits |
| human verification | **445 / 445** test rows carry an owner verdict against their current text |
| baseline → SFT | qwen3.5-4b zero-shot exec-match 13.5% → **47.2%** after SFT round 1, at 30% fewer output tokens (445-row test slice, 2026-09-10) |
| size ablation | identical recipe on 0.8B / 2B / 4B / 9B: full-val exec-match 27.0 / 34.1 / 44.4 / 48.3%; the 4B stays primary (2026-09-10) |
| tests | 532 green |

Numbers and how each was measured live in `RESULTS.md`; every decision
behind them is in `DECISIONS.md`, newest first.

## Repo layout

```
src/executor.py     sandboxed headless-vim runner (stdlib only) — the oracle
src/evaluate.py     execution-match eval with per-category breakdown
src/prompt.py       the §6 model I/O contract, implemented once
src/paste.py        keys-mode output rendered as one pasteable ex line
data/combined_cases.jsonl   the corpus; data/splits/ holds the frozen splits
data/seed_cases.jsonl       14 hand-written cases; the schema reference
data/fixes/         every curation change-set, dated — the audit trail
scripts/            corpus build, split refresh, curation, review pages
```

## Quickstart

```bash
# needs vim (not vim-tiny) on PATH; python3, stdlib only
python3 src/evaluate.py --cases data/seed_cases.jsonl          # gold self-check
python3 src/evaluate.py --cases data/seed_cases.jsonl \
                        --predictions runs/your_model.jsonl    # score a model

# try a single command against a real file (the file is never modified)
python3 src/executor.py --file notes.txt --command '%s/cat/dog/g'
python3 src/executor.py --file notes.txt --command 'dd' --mode keys --line 2 \
                        > edited.txt      # the executor itself never writes files

# defaults live in configs/settings.toml; CLI flags win over it
python3 src/executor.py --show-settings          # what is actually in effect
python3 src/executor.py --settings my.toml ...   # use a different file
```

The self-check runs every case's own gold command and must score 100% —
it validates the dataset and the executor against each other. Run it in CI (continuous integration).

## Task definition

Input: a natural-language instruction, plus buffer context (`buffer_before`,
cursor position). Output: one command, in one of two modes:

- **ex** — Ex commands (`%s/…/…/g`, `g/^$/d`, `sort`, `1m$`)
- **keys** — normal-mode keystrokes with angle-bracket notation
  (`dap`, `ci"error<Esc>`, `<C-a>`)

Scope discipline (v1): single commands, single buffer. No macros (`q`),
no registers-as-API, no multi-file, no plugins, no `:w`/`:q` (file
persistence is the harness's job, so save/quit commands are *invalid* by
definition here). Freeze this scope before generating data; scope creep on
vim's surface area is the main way this project dies.

## Sandbox / threat model

Model-generated commands are untrusted input. The executor runs vim with:

- `-Z` restricted mode — `:!`, `system()`, `:shell` disabled
- `'nowrite'` set around the untrusted command — writes to arbitrary paths
  blocked in both modes (`:w /etc/x` → error → marked invalid); the
  harness's own final `wq!` re-enables writing first
- `-u NONE -i NONE -n` — no user config (configuration), viminfo, or swap
- throwaway temp cwd, deleted after each run
- hard wall-clock timeout — kills interactive waits and infinite loops
  (`:while 1 | endwhile` dies at the timeout, verified)

This targets *accidental* damage from generated commands, which is the real
risk in a data/eval loop. It is not a hardened jail against an adversarial
model (a keys-mode command could `:set write` itself); for large-scale
generation, run workers in a container as an unprivileged user anyway.

## Data plan (phase 1)

No large public NL (natural language)→vim dataset exists, which makes the dataset itself a
contribution. Pipeline:

1. **Taxonomy first.** ~15 command categories (substitute, global,
   ranges/moves, text objects, operators+motions, counts, case ops,
   indent, join, increment, …) × difficulty tiers. Generation is quota'd
   per cell so the set stays balanced.
2. **Synthetic generation.** A frontier LLM generates
   (instruction, buffer_before, command, buffer_after) tuples per cell,
   with paraphrase diversity in the instructions (terse, verbose,
   imperative, "how do I…").
3. **Execution validation.** Every tuple runs through `executor.py`; keep
   only tuples where the command reproduces `buffer_after`. Rejected
   tuples go to a quarantine file — they're free hard-negative material
   for preference pairs later.
4. **Held-out eval.** A human-verified test slice (a few hundred cases),
   plus difficulty split. VimGolf challenges and vi.stackexchange are
   *inspiration* for eval coverage — check licensing before ingesting
   anything verbatim.

Target: ~5–20k validated training pairs; a few hundred eval cases.
**Status:** 7,541 pairs and a 445-case human-verified test slice, so the
data phase has met its target; what remains is training against it.

## Training plan (phase 2)

- **Baselines before training** (this ordering is the whole discipline):
  zero-/few-shot scores for a frontier API (application programming interface) model and for the base open
  model. The fine-tune has to beat the base model and approach the
  frontier model to justify existing; the README results table leads
  with this comparison.
- **SFT (supervised fine-tuning):** QLoRA (quantized low-rank adaptation) on a small open model. Pick by hardware:
  - free/Colab T4 class → 1.5–3B (e.g. Qwen coder-class small models)
  - single 24GB (4090/A10) → 7–8B QLoRA comfortably
  - A100/H100 access → 7–8B full-parameter or larger LoRA (low-rank adaptation) sweeps
  Unsloth is the least-friction single-GPU path; HF (Hugging Face) TRL (Transformer Reinforcement Learning) if you want the
  canonical stack on the resume.
- **Model-agnostic by design:** the student model is selected from a
  config registry (`configs/models.yaml`) — prompts, eval harness,
  GRPO (Group Relative Policy Optimization) reward, and all serving clients are identical across models, so
  swapping in a different LLM is a one-entry config change that yields
  a directly comparable results-table row.
- **RL (the headline):** GRPO with reward = executes cleanly AND buffer
  matches (+ small length penalty). TRL's GRPOTrainer + `executor.py` as
  the reward function. This is RLVR (reinforcement learning from verifiable rewards) — the same recipe behind 2025-era
  reasoning models — on a task where you own the verifier end to end.
- **Error analysis between rounds:** per-category breakdown → find the
  failure cluster → targeted data. (The same loop as production intent-
  recognition tuning, which is the point of the resume story.)

## Serving & demo (phase 3)

- Merge LoRA → serve with **vLLM**; report latency/throughput.
- Quantize to **GGUF (GPT-Generated Unified Format)** so the model runs on CPU (central processing unit) via llama.cpp — enabling
  the demo: a small **vim/neovim plugin** that sends the instruction +
  buffer context to the local model and applies the result *with a
  confirm step and a repair loop* — on error or rejection, the model sees
  the diff/error and retries once.

## Definition of done

README results table (models × {valid%, exec-match%, tokens, latency,
cost/solved task}), reproducible train/eval scripts, the dataset (or its
generator), an MCP (Model Context Protocol) server exposing the model + verifier, the plugin demo
GIF. That artifact backs resume claims for: synthetic data generation
(frontier-teacher distillation), QLoRA SFT, RLVR/GRPO, execution-based
evals, RL-environment design, vLLM serving, quantization, and MCP.

## License

MIT (code). Dataset license TBD pending source audit.
