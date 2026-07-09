"""Bits-per-byte evaluation — methodology requirement #1.

BPB is tokenizer-invariant: total cross-entropy (in bits) a model assigns to a
fixed raw text, divided by the UTF-8 byte length of that text. This is the
primary loss metric everywhere; per-token CE is never compared across arms.

Convention (must be identical for every arm/size):
  - The eval corpus is a token stream: [EOT] doc1 [EOT] doc2 ... (EOT prepended
    before every doc, produced by data/prepare.py).
  - The model predicts every token in the stream except the very first EOT,
    in non-overlapping block_size windows (final partial window is masked).
  - bpb = sum_of_nats / (ln(2) * total_utf8_bytes_of_raw_text).
    Predicted EOT tokens contribute loss but no bytes (they encode document
    boundaries); the single unpredicted leading EOT contributes neither.
"""

import math

import numpy as np
import torch

LN2 = math.log(2)


@torch.no_grad()
def evaluate_bpb(model, tokens: np.ndarray, total_bytes: int, block_size: int,
                 device: str, device_batch_size: int = 8,
                 autocast_dtype=None) -> dict:
    """tokens: uint16/uint32 numpy stream. Returns {'bpb', 'nats_per_token', 'n_tokens'}."""
    model.eval()
    n = len(tokens)
    assert n >= 2, "eval stream too short"
    n_targets = n - 1
    windows = []
    for start in range(0, n_targets, block_size):
        end = min(start + block_size, n_targets)
        x = np.full(block_size, 0, dtype=np.int64)
        y = np.full(block_size, -1, dtype=np.int64)
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
                _, loss = model(x, y, loss_reduction="sum")
        else:
            _, loss = model(x, y, loss_reduction="sum")
        total_nats += loss.float().item()
    model.train()
    return {
        "bpb": total_nats / (LN2 * total_bytes),
        "nats_per_token": total_nats / n_targets,
        "n_tokens": n_targets,
        "n_bytes": total_bytes,
    }


def load_eval_corpus(eval_dir):
    """Load a prepared eval corpus dir (from data/prepare.py): tokens + byte count."""
    import json
    from pathlib import Path

    eval_dir = Path(eval_dir)
    meta = json.loads((eval_dir / "meta.json").read_text())
    dtype = np.uint32 if meta.get("dtype") == "uint32" else np.uint16
    tokens = np.fromfile(eval_dir / "val.bin", dtype=dtype)
    return tokens, meta["val_bytes"], meta
