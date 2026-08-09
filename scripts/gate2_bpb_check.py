"""GATE 2: correctness check for eval/bpb_hf.py against common/bpb.py.

Uses ONE set of weights (HF pretrained gpt2) evaluated three ways on the frozen
neutral corpus, so any mismatch is the instrument, not the model:

  GT  = common/bpb.evaluate_bpb over the pre-tokenized gpt2 val.bin stream
        (the proven ground-truth windowing/CE/byte code), HF-gpt2 weights.
  C1  = bpb_hf.evaluate_bpb_stream over the SAME val.bin stream (isolates the
        new windowing/CE/byte math). Must equal GT to ~1e-5.
  C2  = bpb_hf.evaluate_bpb_hf: re-tokenizes the RAW val_text.jsonl with the HF
        gpt2 tokenizer and scores it (exercises the full raw-text path). Must
        equal GT to <1e-3 (only source of difference is re-tokenizing the raw
        text vs reading the pre-tokenized stream; identical gpt2 BPE => tiny).

Exit non-zero on any tolerance failure.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.bpb import evaluate_bpb, load_eval_corpus            # noqa: E402
from eval.bpb_hf import evaluate_bpb_stream, evaluate_bpb_hf     # noqa: E402

EVAL_DIR = sys.argv[1] if len(sys.argv) > 1 else "/workspace/datasets/wiki_eval"
BLOCK = 1024
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class HFWrap(torch.nn.Module):
    """Expose an HF CausalLM through our GPT's (logits, loss) interface so the
    proven common/bpb.evaluate_bpb can drive it (ignore_index=-1 like our GPT)."""
    def __init__(self, hf):
        super().__init__()
        self.hf = hf

    def forward(self, x, y=None, loss_reduction="mean"):
        logits = self.hf(x).logits
        if y is None:
            return logits, None
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                               y.reshape(-1), ignore_index=-1, reduction=loss_reduction)
        return logits, loss


def main():
    from transformers import GPT2LMHeadModel, AutoTokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEV).eval()
    tok = AutoTokenizer.from_pretrained("gpt2")

    tokens, total_bytes, meta = load_eval_corpus(EVAL_DIR)
    print(f"corpus: {len(tokens):,} gpt2 tokens, {total_bytes:,} bytes, "
          f"{meta.get('val_docs','?')} docs")

    wrap = HFWrap(model)
    gt = evaluate_bpb(wrap, tokens, total_bytes, BLOCK, DEV, device_batch_size=8)
    c1 = evaluate_bpb_stream(lambda x: model(x).logits, tokens, total_bytes,
                             BLOCK, DEV, device_batch_size=8)
    c2 = evaluate_bpb_hf(model, tok, EVAL_DIR, BLOCK, DEV, device_batch_size=8)

    print(f"\nGT (common/bpb, gpt2 val.bin)         bpb = {gt['bpb']:.8f}")
    print(f"C1 (bpb_hf stream, same val.bin)      bpb = {c1['bpb']:.8f}  "
          f"|Δ|={abs(c1['bpb']-gt['bpb']):.2e}")
    print(f"C2 (bpb_hf full, re-tokenized raw)    bpb = {c2['bpb']:.8f}  "
          f"|Δ|={abs(c2['bpb']-gt['bpb']):.2e}")
    print(f"   C2 stream tokens = {c2['n_tokens']:,} vs val.bin "
          f"{gt['n_tokens']:,} (Δ={c2['n_tokens']-gt['n_tokens']:+,})")

    ok1 = abs(c1['bpb'] - gt['bpb']) < 1e-5
    ok2 = abs(c2['bpb'] - gt['bpb']) < 1e-3
    print(f"\nC1 math-identity  (<1e-5): {'PASS' if ok1 else 'FAIL'}")
    print(f"C2 raw-text path  (<1e-3): {'PASS' if ok2 else 'FAIL'}")
    if not (ok1 and ok2):
        sys.exit("GATE 2 FAILED")
    print("\nGATE 2 PASS")


if __name__ == "__main__":
    main()
