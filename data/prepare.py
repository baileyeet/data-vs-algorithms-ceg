"""Download + tokenize datasets into flat .bin token shards.

Parameterized by (dataset, tokenizer) — requirement #8 says the tokenizer is
part of "algorithm", so each corpus gets tokenized once per tokenizer (4 train
variants total at full scale, reused across all model sizes).

Output layout ({out}/):
  train.bin        uint16 token stream: [EOT] doc [EOT] doc ...
  val.bin          same format, held-out docs
  val_text.jsonl   raw text of the val docs ({"text": ...} per line) — kept so
                   eval corpora can be re-tokenized identically with the other
                   tokenizer, and for decontamination
  meta.json        token/byte counts per split, tokenizer, eot id

BPB accounting: for each split we record the summed UTF-8 byte length of the
raw doc texts. Docs are never truncated mid-document; we stop adding docs once
the token budget is met, so tokens and bytes stay aligned.

Tokenizer note: modded-nanoGPT's current recipe still uses the GPT-2 BPE
tokenizer (vocab 50257, padded to 50304 in the model), so "new" is currently
an alias of gpt2 kept as a separate axis in case the new-algorithm recipe's
tokenizer diverges. The BPB metric does not depend on this coincidence.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TOKENIZERS = {"gpt2": "gpt2", "new": "gpt2"}  # name -> tiktoken encoding

DATASETS = {
    # name -> (hf path, hf config, text field)
    "openwebtext": ("Skylion007/openwebtext", None, "text"),
    "dclm": ("mlfoundations/dclm-baseline-1.0-parquet", None, "text"),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en", "text"),
    "fineweb": ("HuggingFaceFW/fineweb", "sample-10BT", "text"),
}


def get_encoder(name):
    import tiktoken

    enc = tiktoken.get_encoding(TOKENIZERS[name])
    return enc, enc.eot_token


def doc_stream(args):
    if args.dataset == "jsonl":
        def gen():
            with open(args.jsonl_path) as f:
                for line in f:
                    yield json.loads(line)["text"]
        return gen()
    if args.dataset == "txt":
        def gen():
            text = Path(args.txt_path).read_text()
            for para in text.split("\n\n"):
                if para.strip():
                    yield para
        return gen()
    from datasets import load_dataset

    path, config, field = DATASETS[args.dataset]
    ds = load_dataset(path, config, split="train", streaming=True)
    if args.shuffle_buffer:
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    return (row[field] for row in ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS) + ["jsonl", "txt"])
    ap.add_argument("--tokenizer", default="gpt2", choices=list(TOKENIZERS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-tokens", type=int, default=0, help="0 = no train split (eval-only corpus)")
    ap.add_argument("--val-tokens", type=int, default=250_000)
    ap.add_argument("--jsonl-path")
    ap.add_argument("--txt-path")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--shuffle-buffer", type=int, default=0, help="streaming shuffle buffer (docs)")
    args = ap.parse_args()

    enc, eot = get_encoder(args.tokenizer)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    splits = {}  # name -> dict(tokens=[], bytes=0, docs=0, budget=..)
    order = []
    if args.val_tokens > 0:
        order.append(("val", args.val_tokens))
    if args.train_tokens > 0:
        order.append(("train", args.train_tokens))

    val_text_f = open(out / "val_text.jsonl", "w") if args.val_tokens > 0 else None

    stream = doc_stream(args)
    for split, budget in order:
        toks, ntok, nbytes, ndocs = [], 0, 0, 0
        for text in stream:
            ids = enc.encode_ordinary(text)
            if not ids:
                continue
            toks.append(np.array([eot] + ids, dtype=np.uint16))
            ntok += len(ids) + 1
            nbytes += len(text.encode("utf-8"))
            ndocs += 1
            if split == "val" and val_text_f:
                val_text_f.write(json.dumps({"text": text}) + "\n")
            if ntok >= budget:
                break
        arr = np.concatenate(toks) if toks else np.array([], dtype=np.uint16)
        arr.tofile(out / f"{split}.bin")
        splits[split] = {"tokens": int(len(arr)), "bytes": int(nbytes), "docs": int(ndocs)}
        print(f"{split}: {len(arr):,} tokens, {nbytes:,} bytes, {ndocs:,} docs")
        if ntok < budget:
            print(f"WARNING: stream exhausted before {split} budget "
                  f"({ntok:,} < {budget:,})", file=sys.stderr)

    if val_text_f:
        val_text_f.close()

    meta = {
        "dataset": args.dataset,
        "tokenizer": args.tokenizer,
        "tiktoken_encoding": TOKENIZERS[args.tokenizer],
        "eot_id": int(eot),
        "dtype": "uint16",
        "seed": args.seed,
    }
    for s, d in splits.items():
        meta[f"{s}_tokens"] = d["tokens"]
        meta[f"{s}_bytes"] = d["bytes"]
        meta[f"{s}_docs"] = d["docs"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
