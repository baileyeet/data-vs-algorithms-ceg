"""Merge N sharded prepare.py output dirs (--num-stream-shards runs) into one.

train.bin parts are concatenated in shard order; meta counts are summed; the
val split (and val_text.jsonl) is taken from the one shard that produced it.

Usage: python data/merge_shards.py --parts out_s0 out_s1 ... --out merged_dir
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = [Path(p) for p in args.parts]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metas = [json.loads((p / "meta.json").read_text()) for p in parts]
    merged = dict(metas[0])
    merged["stream_shards"] = [
        {"dir": str(p), "train_tokens": m.get("train_tokens", 0),
         "seed": m.get("seed"), "shuffle_buffer": m.get("shuffle_buffer")}
        for p, m in zip(parts, metas)]

    with open(out / "train.bin", "wb") as w:
        for p, m in zip(parts, metas):
            src = p / "train.bin"
            if src.exists() and m.get("train_tokens", 0) > 0:
                with open(src, "rb") as r:
                    shutil.copyfileobj(r, w, length=64 * 1024 * 1024)
    for key in ("train_tokens", "train_bytes", "train_docs"):
        merged[key] = sum(m.get(key, 0) for m in metas)

    val_parts = [p for p, m in zip(parts, metas) if m.get("val_tokens", 0) > 0]
    assert len(val_parts) == 1, f"expected exactly one shard with a val split, got {len(val_parts)}"
    vp = val_parts[0]
    shutil.copy(vp / "val.bin", out / "val.bin")
    if (vp / "val_text.jsonl").exists():
        shutil.copy(vp / "val_text.jsonl", out / "val_text.jsonl")

    (out / "meta.json").write_text(json.dumps(merged, indent=2))
    print(f"merged {len(parts)} shards -> {out}: {merged['train_tokens']:,} train tokens, "
          f"{merged['val_tokens']:,} val tokens")


if __name__ == "__main__":
    main()
