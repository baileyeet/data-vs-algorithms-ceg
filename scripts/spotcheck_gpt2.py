"""Phase 1 spot-check: load OpenAI's real GPT-2 124M weights into our GPT class
and evaluate BPB on the toy Wikipedia slice.

Validates (a) our architecture is weight-compatible with true GPT-2, and
(b) the BPB pipeline yields a sane value for a known model (English Wikipedia:
roughly ~1.0-1.3 bits/byte for GPT-2 small — far below a from-scratch toy run).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bpb import evaluate_bpb, load_eval_corpus
from common.model_gpt2 import GPT, GPTConfig


def load_hf_gpt2(device):
    from transformers import GPT2LMHeadModel

    hf = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_hf = hf.state_dict()
    model = GPT(GPTConfig(vocab_size=50257))  # true GPT-2 vocab, no padding
    sd = model.state_dict()
    # HF uses Conv1D for attn/mlp projections: transpose those weights
    transposed = ("attn.c_attn.weight", "attn.c_proj.weight",
                  "mlp.c_fc.weight", "mlp.c_proj.weight")
    for k in sd:
        hk = k
        assert hk in sd_hf, f"missing {hk}"
        src = sd_hf[hk]
        if any(hk.endswith(t) for t in transposed):
            src = src.t()
        assert src.shape == sd[k].shape, f"{k}: {src.shape} vs {sd[k].shape}"
        with torch.no_grad():
            sd[k].copy_(src)
    return model.to(device)


def main():
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    eval_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/toy_wiki_gpt2"
    tokens, nbytes, _ = load_eval_corpus(eval_dir)
    model = load_hf_gpt2(device)
    r = evaluate_bpb(model, tokens, nbytes, block_size=1024, device=device)
    print(f"real GPT-2 124M on {eval_dir}: bpb={r['bpb']:.4f} "
          f"nats/tok={r['nats_per_token']:.4f} ({r['n_tokens']:,} tokens)")
    ok = 0.8 <= r["bpb"] <= 1.5
    print("SANE" if ok else "OUT OF EXPECTED RANGE [0.8, 1.5] — investigate")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
