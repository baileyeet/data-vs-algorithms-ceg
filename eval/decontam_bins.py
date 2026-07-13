"""Decontaminate the neutral eval corpus against tokenized training .bin files.

Scans the ACTUAL training sample (the .bin token streams the models train on),
not the upstream source — exact by construction. Docs are recovered by
splitting on the EOT id and decoded with the corpus tokenizer; contaminated
eval docs (>= --max-hits shared word-level n-grams) are dropped.

Usage:
  python eval/decontam_bins.py --eval-jsonl wiki/val_text.jsonl \
    --train-bins owt/train.bin dclm/train.bin --eot 50256 \
    --out wiki/val_text.decontam.jsonl --workers 32
"""

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.decontam import ngrams

_ENC = None
_UNIVERSE = None
_N = None


def _init(universe, n):
    global _ENC, _UNIVERSE, _N
    import tiktoken

    _ENC = tiktoken.get_encoding("gpt2")
    _UNIVERSE = universe
    _N = n


def _scan_span(task):
    """Decode a token span and return the contaminated n-grams found in it."""
    path, start, count, eot = task
    toks = np.fromfile(path, dtype=np.uint16, count=count, offset=start * 2)
    found = set()
    # split on EOT into docs; spans overlap by one doc boundary at each end so
    # n-grams inside docs are never cut (cross-doc n-grams don't exist: EOT
    # separates documents in training too)
    doc = []
    for t in toks:
        if t == eot:
            if doc:
                found |= _UNIVERSE & ngrams(_ENC.decode(doc), _N)
                doc = []
        else:
            doc.append(int(t))
    if doc:
        found |= _UNIVERSE & ngrams(_ENC.decode(doc), _N)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--train-bins", nargs="+", required=True)
    ap.add_argument("--eot", type=int, default=50256)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--max-hits", type=int, default=1)
    ap.add_argument("--span-tokens", type=int, default=20_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_docs = [json.loads(l) for l in open(args.eval_jsonl)]
    doc_grams = [ngrams(d["text"], args.n) for d in eval_docs]
    universe = set().union(*doc_grams) if doc_grams else set()
    print(f"{len(eval_docs)} eval docs, {len(universe):,} distinct {args.n}-grams", flush=True)

    tasks = []
    for path in args.train_bins:
        n_tokens = Path(path).stat().st_size // 2
        # spans overlap by 4096 tokens so a doc cut at a span boundary is fully
        # contained in at least one span (docs >4096 tokens: the overlap still
        # covers any n-gram window near the cut for typical doc sizes)
        step = args.span_tokens
        for start in range(0, n_tokens, step):
            tasks.append((path, max(0, start - 4096),
                          min(step + 8192, n_tokens - max(0, start - 4096)), args.eot))
    print(f"{len(tasks)} spans across {len(args.train_bins)} bins", flush=True)

    hit = set()
    with Pool(args.workers, initializer=_init, initargs=(universe, args.n)) as pool:
        for i, found in enumerate(pool.imap_unordered(_scan_span, tasks)):
            hit |= found
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} spans, {len(hit):,} contaminated n-grams", flush=True)

    kept, dropped = [], 0
    for d, g in zip(eval_docs, doc_grams):
        if len(g & hit) >= args.max_hits:
            dropped += 1
        else:
            kept.append(d)
    with open(args.out, "w") as f:
        for d in kept:
            f.write(json.dumps(d) + "\n")
    print(f"dropped {dropped}/{len(eval_docs)} contaminated eval docs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
