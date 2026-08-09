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
    # data-era ladder (Exp A). Both public/ungated + streamable (verified
    # 2026-08-08). NOTE the text field differs: C4 uses "text", RefinedWeb "content".
    "c4": ("allenai/c4", "en", "text"),                       # 2020, Common-Crawl
    "refinedweb": ("tiiuae/falcon-refinedweb", None, "content"),  # 2023, Common-Crawl
}


def get_encoder(name):
    import tiktoken

    enc = tiktoken.get_encoding(TOKENIZERS[name])
    return enc, enc.eot_token


_WORKER_ENC = None


def _init_worker(tok_name):
    global _WORKER_ENC
    _WORKER_ENC = get_encoder(tok_name)


def _encode_batch(texts):
    """Runs in a worker: returns (list of per-doc token arrays, per-doc byte lens).
    Empty docs return empty arrays so order/indices stay aligned with input."""
    enc, eot = _WORKER_ENC
    out_toks, out_bytes = [], []
    for ids, text in zip(enc.encode_ordinary_batch(texts), texts):
        if ids:
            out_toks.append(np.array([eot] + ids, dtype=np.uint16))
            out_bytes.append(len(text.encode("utf-8")))
        else:
            out_toks.append(np.array([], dtype=np.uint16))
            out_bytes.append(0)
    return out_toks, out_bytes


def encoded_doc_stream(stream, args):
    """Yield (token_array, n_bytes, text) per doc, in stream order.

    workers==1: serial (original behavior). workers>1: docs are batched and
    tokenized in a process pool with imap (order-preserving), so the produced
    corpus is IDENTICAL to the serial one — parallelism changes throughput only.
    """
    if args.workers <= 1:
        enc, eot = get_encoder(args.tokenizer)
        for text in stream:
            ids = enc.encode_ordinary(text)
            if not ids:
                continue
            yield np.array([eot] + ids, dtype=np.uint16), len(text.encode("utf-8")), text
        return

    from itertools import islice
    from multiprocessing import Pool

    def batches():
        while chunk := list(islice(stream, args.batch_docs)):
            yield chunk

    # imap preserves submission order; `pending` holds the texts of in-flight
    # batches (bounded by the pool's prefetch) so each result can be re-paired
    # with its input docs for val_text.jsonl.
    pending = []

    def feed():
        for chunk in batches():
            pending.append(chunk)
            yield chunk

    with Pool(args.workers, initializer=_init_worker, initargs=(args.tokenizer,)) as pool:
        for toks, nbytes in pool.imap(_encode_batch, feed(), chunksize=1):
            texts = pending.pop(0)
            for arr, nb, text in zip(toks, nbytes, texts):
                if len(arr):
                    yield arr, nb, text


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
    if args.num_stream_shards > 1:
        # disjoint-by-construction partition of the underlying data files
        ds = ds.shard(num_shards=args.num_stream_shards, index=args.stream_shard)
    # For subsetting a large corpus (e.g. 9B/18B tokens from multi-T DCLM),
    # .shuffle() on a streaming dataset randomizes BOTH the shard order and
    # docs within the buffer, so the sample isn't the corpus's first shards.
    # seed + buffer are recorded in meta.json for reproducibility.
    if args.shuffle_buffer:
        ds = ds.shuffle(seed=args.seed + args.stream_shard, buffer_size=args.shuffle_buffer)
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
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 = parallel tokenization (order-preserving; needed for full-scale prep)")
    ap.add_argument("--batch-docs", type=int, default=512, help="docs per worker task")
    ap.add_argument("--num-stream-shards", type=int, default=1,
                    help="split the HF stream into N disjoint file-shards; run one "
                         "process per shard (streaming bandwidth is the full-scale "
                         "bottleneck). Merge outputs with data/merge_shards.py")
    ap.add_argument("--stream-shard", type=int, default=0, help="this process's shard index")
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
    encoded = encoded_doc_stream(stream, args)
    for split, budget in order:
        ntok, nbytes, ndocs = 0, 0, 0
        # stream tokens to disk incrementally — full-scale splits (18B tokens
        # = 36GB) must not accumulate in RAM
        with open(out / f"{split}.bin", "wb") as bin_f:
            for arr, nb, text in encoded:
                arr.tofile(bin_f)
                ntok += len(arr)
                nbytes += nb
                ndocs += 1
                if split == "val" and val_text_f:
                    val_text_f.write(json.dumps({"text": text}) + "\n")
                if ndocs % 500_000 == 0:
                    print(f"  {split}: {ntok:,}/{budget:,} tokens "
                          f"({ndocs:,} docs)", flush=True)
                if ntok >= budget:
                    break
        splits[split] = {"tokens": int(ntok), "bytes": int(nbytes), "docs": int(ndocs)}
        print(f"{split}: {ntok:,} tokens, {nbytes:,} bytes, {ndocs:,} docs", flush=True)
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
        "shuffle_buffer": args.shuffle_buffer,
    }
    for s, d in splits.items():
        meta[f"{s}_tokens"] = d["tokens"]
        meta[f"{s}_bytes"] = d["bytes"]
        meta[f"{s}_docs"] = d["docs"]
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
