#!/usr/bin/env python3
"""Soft-target KD (knowledge distillation) of the trained 27B into the 4B: train_sft.py plus a soft-target loss.

scripts/train_sft.py runs unchanged; the added loss term comes from the 27B's recorded top-20
distributions. The rules are in runs/kd27b_2026-10-31/preregistration.md.

Why. Recorded 2026-10-31 (scripts/teacher_probs), the trained 27B is peaked on its training rows
but not collapsed: on the 1,493 rows approach 1 re-targeted, the original gold -- a verified
alternative the 27B never trained on -- has its first divergent token in the teacher's top 2 on
61.8% of rows, at a median 8% of the top 20 renormalized at T = 2. A hard target puts zero there.

The loss, per answer token t (every completion position TRL trains on):

    L_t = (1 - alpha) * -log p(y_t)  +  alpha * T^2 * sum_k q_k * -log p_T(i_k)

- y_t is the hard target, the distilled file's command as in approach 1.
- i_1..i_k are the teacher's top-k token ids at t, and q = softmax(z_teacher[i] / T) over them.
- p_T = softmax(z_student / T) over the student's full vocabulary, so mass the student puts outside
  the teacher's top k is penalized through the normalization.
- T^2 keeps the soft term's gradient on the hard term's scale (Hinton et al. 2015).
- The sum is normalized as TRL normalizes its own loss: by the answer tokens in the accumulated
  batch. At alpha = 0 the loss is TRL's chunked NLL exactly (a test pins it), so a KD arm differs
  from approach 1's hard-target arm only in the soft term.

How the targets reach the loss. probs.jsonl holds, per train row, the completion token ids TRL
produces and the teacher's top 20 (ids and raw logits) at each. They ride as three extra dataset
columns through TRL's own tokenization (which keeps unknown columns), past the Trainer's column
filter (the signature list is extended), into a collator that concatenates them in batch order,
the order in which the loss reads the answer positions. On every batch the collator checks that
the answer tokens in `labels` equal the recorded ids, token for token, so a misalignment stops
the run instead of training on shifted targets.

The logits are computed only at the answer positions, in checkpointed chunks as TRL's own loss
does, from a bf16 matmul upcast afterwards (the arithmetic TRL's chunk performs under autocast).

Everything else -- data, prompt, mask, LoRA config, schedule, evaluation, checkpoint selection,
the final full-val pass -- is train_sft.py's code, so a KD arm and approach 1's arm share one
recipe.

    python scripts/kd27b/train_kd.py --model qwen3.5-4b \\
        --train runs/distill_2026-10-31/train_distilled.jsonl \\
        --out-dir runs/qwen3.5-4b-kd27b-s42 --eval-every 204 --no-eval-at-start --seed 42 --wandb
"""
import argparse
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import train_sft                                                  # noqa: E402

PROBS = REPO / "runs" / "teacher_probs_2026-10-31" / "probs.jsonl"
KD_COLUMNS = ("teacher_target", "teacher_ids", "teacher_logits")
CHUNK = 256         # answer positions per checkpointed loss chunk
KD = {"alpha": 0.5, "temperature": 2.0, "topk": 20}       # set by install()


def load_teacher(path, topk):
    """{row id: (completion ids, top-k ids, top-k logits)} from scripts/teacher_probs' probs.jsonl."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            ids = [row[:topk] for row in r["topk_ids"]]
            logits = [row[:topk] for row in r["topk_logits"]]
            if not (len(ids) == len(logits) == len(r["ids"]) == r["n"]) or any(len(x) != topk for x in ids):
                raise ValueError("row %s: %d positions, %d recorded, top-%d wanted"
                                 % (r["id"], r["n"], len(ids), topk))
            out[r["id"]] = (r["ids"], ids, logits)
    return out


def patched_build_dataset(orig, teacher):
    """train_sft.build_dataset, plus the teacher's columns for each row (by id)."""
    def build_dataset(cases, reg):
        missing = [c["id"] for c in cases if c["id"] not in teacher]
        if missing:
            raise SystemExit("%d train rows have no teacher record (first: %s)" % (len(missing), missing[0]))
        ds = orig(cases, reg)
        for k, name in enumerate(KD_COLUMNS):
            ds = ds.add_column(name, [teacher[c["id"]][k] for c in cases])
        return ds
    return build_dataset


class KDCollator:
    """TRL's collator for everything it knows, plus the teacher's arrays concatenated in batch order.
    Refuses a batch whose answer tokens are not the recorded ones."""

    def __init__(self, base):
        self.base = base

    def __call__(self, examples):
        import torch
        batch = self.base([{k: v for k, v in ex.items() if k not in KD_COLUMNS} for ex in examples])
        target = torch.tensor([t for ex in examples for t in ex["teacher_target"]], dtype=torch.long)
        labels = batch["labels"][:, 1:]
        got = labels[labels != -100]
        if got.shape != target.shape or not torch.equal(got, target):
            raise RuntimeError("the batch's answer tokens (%d) are not the teacher's recorded ones (%d): "
                               "the soft targets would be shifted" % (got.numel(), target.numel()))
        batch["teacher_ids"] = torch.tensor([t for ex in examples for t in ex["teacher_ids"]], dtype=torch.long)
        batch["teacher_logits"] = torch.tensor([t for ex in examples for t in ex["teacher_logits"]],
                                               dtype=torch.float32)
        return batch


def kd_sums(logits, target, t_ids, t_logits, alpha, temperature):
    """Sums over N answer positions: (loss, NLL, soft cross-entropy, teacher entropy, correct, student
    entropy). logits [N, V] fp32 (the student's); target [N]; t_ids, t_logits [N, k]."""
    import torch
    import torch.nn.functional as F
    lp1 = F.log_softmax(logits, -1)
    nll = -lp1.gather(1, target[:, None]).squeeze(1)
    ent = -(lp1.exp() * lp1).sum(-1)
    correct = (logits.argmax(-1) == target).float()
    if alpha:
        q = F.softmax(t_logits / temperature, -1)
        soft = -(q * F.log_softmax(logits / temperature, -1).gather(1, t_ids)).sum(-1)
        t_ent = -(q * q.clamp_min(1e-30).log()).sum(-1)
        loss = (1 - alpha) * nll + alpha * temperature ** 2 * soft
    else:
        soft = t_ent = torch.zeros_like(nll)
        loss = nll
    return tuple(x.sum() for x in (loss, nll, soft, t_ent, correct, ent))


def _chunk(h, w, target, t_ids, t_logits, alpha, temperature):
    logits = (h.to(w.dtype) @ w.t()).float()
    return kd_sums(logits, target, t_ids, t_logits, alpha, temperature)


def make_kd_trainer(SFTTrainer):
    """SFTTrainer with the KD collator, the teacher columns kept, and the KD loss."""
    import torch

    class KDTrainer(SFTTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.data_collator = KDCollator(self.data_collator)

        def _set_signature_columns_if_needed(self):
            super()._set_signature_columns_if_needed()
            self._signature_columns = list(self._signature_columns) + [
                c for c in KD_COLUMNS if c not in self._signature_columns]

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            mode = "train" if self.model.training else "eval"
            t_ids, t_logits = inputs.pop("teacher_ids"), inputs.pop("teacher_logits")
            inputs.pop("_prediction_loss_only", None)
            inner = self.accelerator.unwrap_model(model)
            if hasattr(inner, "get_base_model"):
                inner = inner.get_base_model()
            # TRL's chunked forward runs the same backbone: `base_model` skips the LM head.
            hidden = inner.base_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                      use_cache=False).last_hidden_state
            labels = inputs["labels"][:, 1:]
            valid = labels != -100
            h, y = hidden[:, :-1][valid], labels[valid]
            w = inner.get_output_embeddings().weight
            t_ids, t_logits = t_ids.to(h.device), t_logits.to(h.device)
            sums = [h.new_zeros((), dtype=torch.float32) for _ in range(6)]
            for s in range(0, max(y.numel(), 1), CHUNK):
                part = torch.utils.checkpoint.checkpoint(
                    _chunk, h[s:s + CHUNK], w, y[s:s + CHUNK], t_ids[s:s + CHUNK], t_logits[s:s + CHUNK],
                    KD["alpha"], KD["temperature"], use_reentrant=False)
                sums = [a + b for a, b in zip(sums, part)]
            loss_sum, nll, soft, t_ent, correct, ent = sums
            n = max(y.numel(), 1)
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:
                # The Trainer then leaves the loss alone, as for TRL's own loss.
                denom = num_items_in_batch.to(loss_sum.device) if torch.is_tensor(num_items_in_batch) \
                    else num_items_in_batch
            else:
                denom = n             # the Trainer divides by the accumulation steps itself
            loss = loss_sum / denom
            m = self._metrics[mode]
            m["entropy"].append(ent.item() / n)
            m["mean_token_accuracy"].append(correct.item() / n)
            m["nll"].append(nll.item() / n)
            if KD["alpha"]:
                m["kd_kl"].append((soft - t_ent).item() / n)
            if mode == "train":
                self._total_train_tokens += int(inputs["attention_mask"].sum().item())
            m["num_tokens"] = [self._total_train_tokens]
            return (loss, None) if return_outputs else loss

    return KDTrainer


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog="Every other flag goes to scripts/train_sft.py unchanged.")
    ap.add_argument("--teacher-probs", default=str(PROBS), help="scripts/teacher_probs' probs.jsonl")
    ap.add_argument("--kd-alpha", type=float, default=KD["alpha"],
                    help="weight of the soft term; 0 is approach 1's loss exactly")
    ap.add_argument("--kd-temperature", type=float, default=KD["temperature"])
    ap.add_argument("--kd-topk", type=int, default=KD["topk"], help="teacher tokens per position (<= 20)")
    return ap


def install(own):
    """Apply the KD pieces where train_sft looks them up: its own namespace for the dataset, and
    TRL's for SFTTrainer (train_sft.main imports it from `trl` at call time)."""
    import trl
    teacher = load_teacher(own.teacher_probs, own.kd_topk)
    KD.update(alpha=own.kd_alpha, temperature=own.kd_temperature, topk=own.kd_topk)
    train_sft.build_dataset = patched_build_dataset(train_sft.build_dataset, teacher)
    trl.SFTTrainer = make_kd_trainer(trl.SFTTrainer)
    return teacher


def main(argv=None):
    own, rest = build_arg_parser().parse_known_args(argv)
    if not 0.0 <= own.kd_alpha <= 1.0 or own.kd_temperature <= 0 or not 1 <= own.kd_topk <= 20:
        raise SystemExit("--kd-alpha in [0, 1], --kd-temperature > 0, --kd-topk in 1..20")
    teacher = install(own)
    sft = train_sft.build_arg_parser().parse_args(rest)
    out_dir = pathlib.Path(sft.out_dir) if sft.out_dir else REPO / "runs" / (sft.model + "-sft")
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(pathlib.Path(own.teacher_probs).read_bytes()).hexdigest()
    (out_dir / "kd_config.json").write_text(json.dumps({
        **KD, "teacher_probs": own.teacher_probs, "teacher_probs_sha256": digest,
        "teacher_rows": len(teacher), "chunk": CHUNK,
        "loss": "(1 - alpha) * NLL + alpha * T^2 * CE(teacher top-k at T, student full softmax at T)",
        "normalization": "answer tokens in the accumulated batch (TRL's)"}, indent=1) + "\n", encoding="utf-8")
    print("KD: alpha %.2f, T %.1f, top %d from %s (%d rows)"
          % (KD["alpha"], KD["temperature"], KD["topk"], own.teacher_probs, len(teacher)), flush=True)
    sys.argv = [str(REPO / "scripts" / "train_sft.py")] + rest
    return train_sft.main()


if __name__ == "__main__":
    sys.exit(main())
