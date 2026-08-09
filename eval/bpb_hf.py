"""Bits-per-byte over an ARBITRARY HF model + tokenizer (Experiment B).

Same convention as common/bpb.py (EOT-prepended docs, non-overlapping
block_size windows with the final partial window masked, byte-normalized), but
tokenizes the RAW eval text with the model's OWN tokenizer so BPB stays
comparable across architectures whose tokenizers differ (methodology #1: BPB is
tokenizer-invariant because the denominator is raw UTF-8 bytes, not tokens).

    bpb = sum_nats / (ln(2) * total_utf8_bytes)

total_bytes is the tokenizer-independent raw-text byte count, read from the
corpus meta (identical to what common/bpb.py divides by). The only thing that
changes across tokenizers is how the same raw text is segmented into tokens
(the numerator).

Validated against common/bpb.py in scripts/gate2_bpb_check.py (Gate 2).
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

LN2 = math.log(2)


def load_eval_texts(eval_dir):
    """Raw eval docs + tokenizer-independent byte count, from a prepare.py dir."""
    eval_dir = Path(eval_dir)
    meta = json.loads((eval_dir / "meta.json").read_text())
    lines = (eval_dir / "val_text.jsonl").read_text().splitlines()
    texts = [json.loads(l)["text"] for l in lines if l.strip()]
    return texts, meta["val_bytes"], meta


def tokenize_stream(tokenizer, texts, eot_id) -> np.ndarray:
    """[EOT] doc1 [EOT] doc2 ... — identical convention to data/prepare.py.

    add_special_tokens=False so only the raw BPE of each doc is used; the EOT
    document separator is prepended explicitly (as prepare.py does), so no
    tokenizer-specific auto-specials leak in.
    """
    stream = []
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if not ids:
            continue
        stream.append(eot_id)
        stream.extend(ids)
    return np.array(stream, dtype=np.int64)


@torch.no_grad()
def evaluate_bpb_stream(forward_logits, tokens: np.ndarray, total_bytes: int,
                        block_size: int, device: str, device_batch_size: int = 8,
                        autocast_dtype=None) -> dict:
    """Core windowing/CE/byte math — identical to common/bpb.evaluate_bpb but
    driven by a `forward_logits(x)->logits[B,T,V]` callable so it works for any
    HF CausalLM (model(x).logits) or our GPT (returns logits first)."""
    n = len(tokens)
    assert n >= 2, "eval stream too short"
    n_targets = n - 1
    windows = []
    for start in range(0, n_targets, block_size):
        end = min(start + block_size, n_targets)
        x = np.zeros(block_size, dtype=np.int64)
        y = np.full(block_size, -100, dtype=np.int64)
        x[: end - start] = tokens[start:end]
        y[: end - start] = tokens[start + 1 : end + 1]
        windows.append((x, y))

    total_nats = 0.0
    for i in range(0, len(windows), device_batch_size):
        batch = windows[i : i + device_batch_size]
        x = torch.from_numpy(np.stack([w[0] for w in batch])).to(device)
        y = torch.from_numpy(np.stack([w[1] for w in batch])).to(device)
        if autocast_dtype is not None:
            with torch.autocast(device_type=device.split(":")[0], dtype=autocast_dtype):
                logits = forward_logits(x)
        else:
            logits = forward_logits(x)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-100, reduction="sum")
        total_nats += loss.item()
    return {
        "bpb": total_nats / (LN2 * total_bytes),
        "nats_per_token": total_nats / n_targets,
        "n_tokens": n_targets,
        "n_bytes": total_bytes,
    }


@torch.no_grad()
def evaluate_bpb_hf(model, tokenizer, eval_dir, block_size: int, device: str,
                    device_batch_size: int = 8, autocast_dtype=None) -> dict:
    """Neutral-corpus BPB for an HF CausalLM + its tokenizer, from raw text."""
    was_training = model.training
    model.eval()
    texts, total_bytes, _ = load_eval_texts(eval_dir)
    eot = tokenizer.eos_token_id
    assert eot is not None, "tokenizer has no eos_token_id for the doc separator"
    tokens = tokenize_stream(tokenizer, texts, eot)
    out = evaluate_bpb_stream(lambda x: model(x).logits, tokens, total_bytes,
                              block_size, device, device_batch_size, autocast_dtype)
    if was_training:
        model.train()
    return out
