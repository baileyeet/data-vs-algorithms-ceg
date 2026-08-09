"""Decontaminate the neutral eval corpus against a training corpus.

Requirement #2: CC-derived corpora often contain Wikipedia mirrors, so eval
docs sharing long n-grams with either training corpus must be dropped before
the eval set is frozen.

Method (standard n-gram overlap, cf. GPT-3/DCLM decontamination):
  1. Build the set of all word-level N-grams (default N=13) over the eval docs.
  2. Stream the training corpus; record any eval n-gram that appears.
  3. Drop eval docs with >= --max-hits contaminated n-grams (default 1).

Usage:
  python eval/decontam.py --eval-jsonl datasets/wiki_eval/val_text.jsonl \
    --train-dataset openwebtext --train-docs 8100000 \
    --out datasets/wiki_eval/val_text.decontam.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WORD_RE = re.compile(r"\w+")


def ngrams(text, n):
    words = WORD_RE.findall(text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _scan_chunk(texts, universe, n):
    found = set()
    for t in texts:
        found |= universe & ngrams(t, n)
    return found


# --- tokenized-train.bin path (scan exactly what we trained on, no HF re-stream) ---
_BIN_ENC = None


def _init_bin_worker():
    import tiktoken
    global _BIN_ENC
    _BIN_ENC = tiktoken.get_encoding("gpt2")


def _scan_chunk_bin(token_arrays, universe, n):
    """Worker: decode a chunk of per-doc GPT-2 token arrays back to text, then
    intersect their n-grams with the eval universe. Decoding happens here (not
    in the parent) so it parallelizes with the scan."""
    texts = _BIN_ENC.decode_batch([a.tolist() for a in token_arrays])
    found = set()
    for t in texts:
        found |= universe & ngrams(t, n)
    return found


def bin_token_doc_stream(path, eot, max_docs=0):
    """Yield per-doc uint16 token arrays from a prepare.py train.bin
    ([eot] doc [eot] doc ...), splitting on the eot separator. Memmapped, so
    only doc slices (not the whole 18GB) are materialized as we go."""
    arr = np.memmap(path, dtype=np.uint16, mode="r")
    eot_pos = np.flatnonzero(arr == eot)
    n = 0
    for i in range(len(eot_pos)):
        start = int(eot_pos[i]) + 1
        end = int(eot_pos[i + 1]) if i + 1 < len(eot_pos) else len(arr)
        if end > start:
            yield np.asarray(arr[start:end])
            n += 1
            if max_docs and n >= max_docs:
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--train-dataset",
                    help="name from data/prepare.py DATASETS, or a local jsonl path")
    ap.add_argument("--train-bin",
                    help="path to a prepare.py train.bin (GPT-2 BPE, eot 50256); "
                         "scans the ACTUAL tokenized training data, decoded back to "
                         "text — no HF re-stream. Mutually exclusive with --train-dataset")
    ap.add_argument("--train-docs", type=int, default=0, help="0 = whole stream")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 enables multiprocess scanning (full-scale corpora)")
    ap.add_argument("--chunk-docs", type=int, default=2000,
                    help="docs per worker task when --workers > 1")
    ap.add_argument("--max-hits", type=int, default=1,
                    help="drop eval doc if >= this many of its n-grams appear in training data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if bool(args.train_bin) == bool(args.train_dataset):
        ap.error("give exactly one of --train-bin or --train-dataset")

    eval_docs = [json.loads(l) for l in open(args.eval_jsonl)]
    doc_grams = [ngrams(d["text"], args.n) for d in eval_docs]
    universe = set().union(*doc_grams) if doc_grams else set()
    print(f"{len(eval_docs)} eval docs, {len(universe):,} distinct {args.n}-grams")

    from functools import partial
    from itertools import islice
    from multiprocessing import Pool

    def batches(it, size):
        while chunk := list(islice(it, size)):
            yield chunk

    hit = set()
    if args.train_bin:
        # scan the tokenized training data itself (decode -> n-gram), in workers
        gen = bin_token_doc_stream(args.train_bin, eot=50256, max_docs=args.train_docs)
        scanned = 0
        with Pool(args.workers, initializer=_init_bin_worker) as pool:
            for found in pool.imap_unordered(
                    partial(_scan_chunk_bin, universe=universe, n=args.n),
                    batches(gen, args.chunk_docs)):
                hit |= found
                scanned += args.chunk_docs
                if scanned % 500_000 < args.chunk_docs:
                    print(f"  scanned ~{scanned:,} training docs, "
                          f"{len(hit):,} contaminated n-grams", flush=True)
    else:
        if Path(args.train_dataset).exists():
            stream = (json.loads(l)["text"] for l in open(args.train_dataset))
        else:
            from data.prepare import DATASETS
            from datasets import load_dataset
            path, config, field = DATASETS[args.train_dataset]
            ds = load_dataset(path, config, split="train", streaming=True)
            stream = (row[field] for row in ds)

        if args.workers > 1:
            scanned = 0
            with Pool(args.workers) as pool:
                src = islice(stream, args.train_docs) if args.train_docs else stream
                for found in pool.imap_unordered(
                        partial(_scan_chunk, universe=universe, n=args.n),
                        batches(src, args.chunk_docs)):
                    hit |= found
                    scanned += args.chunk_docs
                    if scanned % 100_000 < args.chunk_docs:
                        print(f"  scanned ~{scanned:,} training docs, "
                              f"{len(hit):,} contaminated n-grams", flush=True)
        else:
            for i, text in enumerate(stream):
                if args.train_docs and i >= args.train_docs:
                    break
                hit |= universe & ngrams(text, args.n)
                if (i + 1) % 100_000 == 0:
                    print(f"  scanned {i + 1:,} training docs, {len(hit):,} contaminated n-grams")

    kept, dropped = [], 0
    for d, g in zip(eval_docs, doc_grams):
        if len(g & hit) >= args.max_hits:
            dropped += 1
        else:
            kept.append(d)
    with open(args.out, "w") as f:
        for d in kept:
            f.write(json.dumps(d) + "\n")
    print(f"dropped {dropped}/{len(eval_docs)} contaminated docs -> {args.out}")
    print("re-tokenize the cleaned jsonl with data/prepare.py --dataset jsonl for each tokenizer")


if __name__ == "__main__":
    main()
