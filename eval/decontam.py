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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WORD_RE = re.compile(r"\w+")


def ngrams(text, n):
    words = WORD_RE.findall(text.lower())
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--train-dataset", required=True,
                    help="name from data/prepare.py DATASETS, or a local jsonl path")
    ap.add_argument("--train-docs", type=int, default=0, help="0 = whole stream")
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--max-hits", type=int, default=1,
                    help="drop eval doc if >= this many of its n-grams appear in training data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    eval_docs = [json.loads(l) for l in open(args.eval_jsonl)]
    doc_grams = [ngrams(d["text"], args.n) for d in eval_docs]
    universe = set().union(*doc_grams) if doc_grams else set()
    print(f"{len(eval_docs)} eval docs, {len(universe):,} distinct {args.n}-grams")

    if Path(args.train_dataset).exists():
        stream = (json.loads(l)["text"] for l in open(args.train_dataset))
    else:
        from data.prepare import DATASETS
        from datasets import load_dataset
        path, config, field = DATASETS[args.train_dataset]
        ds = load_dataset(path, config, split="train", streaming=True)
        stream = (row[field] for row in ds)

    hit = set()
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
