#!/usr/bin/env python3
"""Is "this instruction leaves something out" readable inside the model?

Stage 0 of training the ask decision (TRAINING.md). The measurements say a
question pays on under-specified instructions, and that no uncertainty
signal finds them: P(ask) and the disagreement among sampled answers both
predict *being wrong* but not *being fixable* (fixable rows show LESS
disagreement than the other wrong ones). If the decision is to be learned,
the information has to be in the model's reading of the prompt. A linear
probe is the cheap test: if a logistic regression on one layer's hidden
state separates the classes on held-out folds, training can find it too;
if it cannot, training the decision means teaching the model something new.

`--extract` (GPU) runs the student over each case's turn-1 prompt -- the
offer prompt, at the moment it would ask or answer -- and saves the last
token's hidden state at a few layers. `--fit` (CPU) fits an L2 logistic
regression with stratified k-fold cross-validation and reports pooled
out-of-fold AUROC per layer and regularization strength, for:

    unstated vs model            diag rows by their note's class
    unstated+ambiguous vs model  the same, with the two-reading rows
    fixable                      asking turned the unasked wrong answer
                                 right (real & not noask), val + diag

The grid's best cell is mildly optimistic (it is chosen on the same folds);
the whole grid is printed. torch only: scikit-learn is not a dependency.
"""
import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(HERE))
from evaluate import load_jsonl     # noqa: E402
from policies import auroc          # noqa: E402

LAYERS = (8, 16, 24, 32)
LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0)


def stratified_folds(labels, k, seed):
    """Fold index per example, each class spread evenly over the k folds."""
    import random
    rng = random.Random(seed)
    fold = [0] * len(labels)
    for cls in (0, 1):
        idx = [i for i, y in enumerate(labels) if y == cls]
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold[i] = j % k
    return fold


def fit_logistic(x, y, lam, steps=200):
    """L2 logistic regression by LBFGS; x standardized by the caller."""
    import torch
    w = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    b = torch.zeros(1, dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")
    yt = y.to(x.dtype)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, yt)
        loss = loss + lam * (w * w).sum() / 2
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def cv_auroc(feats, labels, lam, k=5, seed=0):
    """Pooled out-of-fold AUROC of an L2 logistic probe."""
    import torch
    x = torch.as_tensor(feats, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.long)
    fold = stratified_folds(labels, k, seed)
    scores = [0.0] * len(labels)
    for f in range(k):
        tr = [i for i in range(len(labels)) if fold[i] != f]
        te = [i for i in range(len(labels)) if fold[i] == f]
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        w, b = fit_logistic((x[tr] - mu) / sd, y[tr], lam)
        out = ((x[te] - mu) / sd) @ w + b
        for i, s in zip(te, out.tolist()):
            scores[i] = s
    return auroc(scores, labels)


def prior_auroc(groups, labels, k=5, seed=0):
    """Cross-validated AUROC of a per-group positive rate (Laplace-smoothed)."""
    fold = stratified_folds(labels, k, seed)
    scores = [0.0] * len(labels)
    for f in range(k):
        counts = {}
        for g, y, fo in zip(groups, labels, fold):
            if fo != f:
                c = counts.setdefault(g, [1, 2])
                c[0] += y
                c[1] += 1
        for i, (g, fo) in enumerate(zip(groups, fold)):
            if fo == f:
                c = counts.get(g, [1, 2])
                scores[i] = c[0] / c[1]
    return auroc(scores, labels)


def bag_of_words(text, dims=512):
    """Hashed word counts of an instruction: the text-only baseline's features."""
    import re
    import zlib
    v = [0.0] * dims
    for w in re.findall(r"[a-z]+", text.lower()):
        v[zlib.crc32(w.encode()) % dims] += 1.0
    return v


def surface_baselines(cases, labels):
    """What the probe must beat: AUROCs of signals read off the surface of
    the case, not the model. A probe that only matches them would be reading
    length or category, not under-specification."""
    out = {"shorter instruction": auroc([-len(c["instruction"]) for c in cases], labels),
           "category (cross-validated prior)": prior_auroc([c.get("category") for c in cases],
                                                            labels),
           "data source (cross-validated prior)": prior_auroc([c.get("source") for c in cases],
                                                               labels)}
    words = [bag_of_words(c["instruction"]) for c in cases]
    out["bag of words on the instruction (best λ)"] = max(
        cv_auroc(words, labels, lam) for lam in LAMBDAS)
    return out


def tasks(cases_by_set, verdicts_by_set):
    """{task: (keys, labels)} over (set, id) keys; see the module docstring."""
    out = {}
    diag = cases_by_set.get("diag", {})
    for name, pos in (("unstated vs model", {"unstated"}),
                      ("unstated+ambiguous vs model", {"unstated", "ambiguous"})):
        keys = [("diag", i) for i, c in diag.items()
                if c.get("note_class") in pos | {"model"}]
        out[name] = (keys, [int(diag[i]["note_class"] in pos) for _, i in keys])
    keys, labels = [], []
    for s, ver in verdicts_by_set.items():
        for i, v in ver.items():
            keys.append((s, i))
            labels.append(int(v["real"]["match"] and not v["noask"]["match"]))
    out["fixable (asking turns it right)"] = (keys, labels)
    return out


def extract(args, cases_by_set):
    import torch
    from dialogue import Decoder, load_student
    from protocol import turn1_messages
    reg, tok, model, base = load_student(args.model, args.adapter, "auto")
    dec = Decoder(model, tok, reg)
    feats = {}
    with torch.no_grad():
        for s, cases in cases_by_set.items():
            for i, c in cases.items():
                enc = tok(dec.render(turn1_messages(c)), return_tensors="pt").to(dec.device)
                hs = model(**enc, output_hidden_states=True).hidden_states
                feats["%s|%s" % (s, i)] = {str(L): hs[L][0, -1].float().cpu() for L in LAYERS}
    torch.save(feats, args.features)
    print("saved %d hidden states x %d layers -> %s" % (len(feats), len(LAYERS), args.features))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default=str(REPO / "runs" / "clarify_2026-10-06"))
    ap.add_argument("--label", default="served", help="whose verdicts and hidden states")
    ap.add_argument("--model", default="qwen3.5-4b")
    ap.add_argument("--adapter", default=str(REPO / "runs" / "qwen3.5-4b-grpo-gated" / "best"))
    ap.add_argument("--features", default=None, help="default $RUN/probe_<label>.pt")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    run = pathlib.Path(args.run)
    args.features = args.features or str(run / ("probe_%s.pt" % args.label))
    cases_by_set = {"val": {c["id"]: c for c in load_jsonl(str(REPO / "data" / "splits" / "val.jsonl"))},
                    "diag": {c["id"]: c for c in load_jsonl(str(run / "diag_cases.jsonl"))}}
    if args.extract:
        extract(args, cases_by_set)
    if args.fit:
        import torch
        feats = torch.load(args.features)
        verdicts = {s: {r["id"]: r for r in load_jsonl(str(run / s / ("verdicts_%s.jsonl" % args.label)))}
                    for s in ("val", "diag")}
        lines = ["| task (n, positives) | layer | " + " | ".join("λ=%g" % l for l in LAMBDAS)
                 + " | P(ask) |", "|---|---|" + "---|" * (len(LAMBDAS) + 1)]
        base_lines = ["| task | " + " | ".join(("shorter instruction", "category", "data source",
                                                 "bag of words")) + " |", "|---|---|---|---|---|"]
        for name, (keys, labels) in tasks(cases_by_set, verdicts).items():
            pa = auroc([verdicts[s][i]["ask_margin"] for s, i in keys], labels)
            for L in LAYERS:
                x = torch.stack([feats["%s|%s" % k][str(L)] for k in keys]).numpy()
                cells = ["%.3f" % cv_auroc(x, labels, lam) for lam in LAMBDAS]
                lines.append("| %s (%d, %d) | %d | %s | %.3f |" % (
                    name, len(labels), sum(labels), L, " | ".join(cells), pa))
                print(lines[-1], flush=True)
            b = surface_baselines([cases_by_set[s][i] for s, i in keys], labels)
            base_lines.append("| %s | %s |" % (name, " | ".join("%.3f" % v for v in b.values())))
            print(base_lines[-1], flush=True)
        out = run / ("probe_%s.md" % args.label)
        out.write_text("# Linear probe: is the ask decision readable from the hidden state?\n\n"
                       "Pooled out-of-fold AUROC, 5-fold stratified, L2 logistic regression on the "
                       "last token of the turn-1 prompt (`scripts/clarify/probe_ask.py`). "
                       "0.5 = chance.\n\n" + "\n".join(lines) + "\n\n"
                       "## What the probe must beat: the surface of the case\n\n"
                       "The same folds, on signals read off the case rather than the model "
                       "(the bag of words takes its best λ, so it is optimistic too):\n\n"
                       + "\n".join(base_lines) + "\n")
        print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
