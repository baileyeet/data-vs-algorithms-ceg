"""Invariance tests for the primary metric (common/bpb.py evaluate_bpb):
the BPB of a fixed token stream must not depend on eval batch size, and the
padded/masked final partial window must contribute nothing.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bpb import evaluate_bpb
from common.model_gpt2 import GPT, GPTConfig


def main():
    torch.manual_seed(0)
    model = GPT(GPTConfig(block_size=32, vocab_size=256, n_layer=2, n_head=2, n_embd=32))
    rng = np.random.default_rng(0)
    # stream deliberately NOT a multiple of block_size -> forces a partial window
    tokens = rng.integers(0, 256, size=32 * 7 + 11, dtype=np.uint16)
    nbytes = 1000

    r1 = evaluate_bpb(model, tokens, nbytes, block_size=32, device="cpu", device_batch_size=1)
    r4 = evaluate_bpb(model, tokens, nbytes, block_size=32, device="cpu", device_batch_size=4)
    r64 = evaluate_bpb(model, tokens, nbytes, block_size=32, device="cpu", device_batch_size=64)
    assert math.isclose(r1["bpb"], r4["bpb"], rel_tol=1e-6), (r1["bpb"], r4["bpb"])
    assert math.isclose(r1["bpb"], r64["bpb"], rel_tol=1e-6), (r1["bpb"], r64["bpb"])
    assert r1["n_tokens"] == len(tokens) - 1  # every token except the first is predicted

    # appending garbage AFTER the stream must change nothing if we pass the same
    # slice — and a longer stream must change the result (sanity that the test bites)
    r_same = evaluate_bpb(model, tokens.copy(), nbytes, block_size=32, device="cpu",
                          device_batch_size=8)
    # different batch shapes reorder float32 sums -> ~1e-7 jitter is expected
    assert math.isclose(r1["bpb"], r_same["bpb"], rel_tol=1e-6)
    longer = np.concatenate([tokens, rng.integers(0, 256, size=17, dtype=np.uint16)])
    r_long = evaluate_bpb(model, longer, nbytes, block_size=32, device="cpu",
                          device_batch_size=8)
    assert not math.isclose(r1["bpb"], r_long["bpb"], rel_tol=1e-6)

    # BPB definition check: bpb * ln2 * bytes == total nats == nats_per_token * n_targets
    total_nats = r1["nats_per_token"] * r1["n_tokens"]
    assert math.isclose(r1["bpb"] * math.log(2) * nbytes, total_nats, rel_tol=1e-9)

    print("ALL BPB TESTS PASSED")


if __name__ == "__main__":
    main()
