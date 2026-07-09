"""Standalone neutral-corpus BPB evaluation for saved checkpoints.

Usage:
  python eval/neutral_bpb.py --ckpt runs/toy_a0d0/ckpt_000025.pt \
    --eval-dir datasets/toy_wiki_gpt2
  python eval/neutral_bpb.py --run-dir runs/toy_a0d0 --eval-dir datasets/toy_wiki_gpt2

The eval corpus must have been tokenized with the SAME tokenizer as the
checkpoint's training arm (the raw text is what's shared across arms).
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.bpb import evaluate_bpb, load_eval_corpus
from common.model_gpt2 import GPT, GPTConfig


def load_ckpt(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt")
    ap.add_argument("--run-dir", help="evaluate every ckpt_*.pt in a run dir")
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--device-batch-size", type=int, default=8)
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokens, nbytes, meta = load_eval_corpus(args.eval_dir)
    paths = [Path(args.ckpt)] if args.ckpt else sorted(Path(args.run_dir).glob("ckpt_*.pt"))
    for p in paths:
        model, ckpt = load_ckpt(p, device)
        r = evaluate_bpb(model, tokens, nbytes, ckpt["config"]["block_size"],
                         device, args.device_batch_size)
        print(f"{p.name}: step={ckpt['step']} tokens={ckpt['tokens']:,} "
              f"bpb={r['bpb']:.6f} nats/tok={r['nats_per_token']:.4f}")


if __name__ == "__main__":
    main()
