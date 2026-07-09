"""Convert our flat token .bin (data/prepare.py output) into modded-nanogpt's
sharded .bin format (256-int32 header: magic 20240520, version 1, num_tokens).

Token semantics already match: uint16 stream, docs delimited by GPT-2 EOT
(50256), which modded-nanogpt treats as BOS for batch alignment.

Usage:
  python scripts/convert_to_nanogpt_bin.py --in datasets/toy_owt_gpt2 \
    --out datasets/toy_owt_nanogpt [--shard-tokens 100000000]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

MAGIC, VERSION = 20240520, 1


def write_shard(tokens: np.ndarray, path: Path):
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-tokens", type=int, default=100_000_000)
    args = ap.parse_args()

    inp, out = Path(args.inp), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((inp / "meta.json").read_text())
    assert meta["dtype"] == "uint16" and meta["eot_id"] == 50256

    for split in ("train", "val"):
        src = inp / f"{split}.bin"
        if not src.exists():
            continue
        tokens = np.fromfile(src, dtype=np.uint16)
        n_shards = 0
        for i in range(0, len(tokens), args.shard_tokens):
            shard = tokens[i : i + args.shard_tokens]
            write_shard(shard, out / f"{split}_{n_shards:06d}.bin")
            n_shards += 1
        print(f"{split}: {len(tokens):,} tokens -> {n_shards} shard(s)")
    shutil.copy(inp / "meta.json", out / "meta.json")
    if (inp / "val_text.jsonl").exists():
        shutil.copy(inp / "val_text.jsonl", out / "val_text.jsonl")


if __name__ == "__main__":
    main()
