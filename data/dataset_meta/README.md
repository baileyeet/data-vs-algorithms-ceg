# Dataset regeneration recipes

The tokenized training corpora (~100 GB) are **not** stored in git or on HF —
they are deterministically regenerable from public sources, and the DCLM-derived
shards have unreviewed redistribution terms. These `*.meta.json` files are the
exact recipe (captured from the network volume 2026-08-08) so the corpora can be
reproduced bit-for-bit on fresh hardware without re-hosting the bytes.

All runs used **GPT-2 BPE** (`tiktoken` `gpt2`, vocab 50304 / eot 50256, uint16)
and **seed 1234**. The `_gpt2` variants are the flat `train.bin`/`val.bin` used by
`train_old` (A0 arms); the `_nanogpt` variants are the same tokens re-sharded into
`train_*.bin` for `train_new` (A1 arms) via `scripts/convert_to_nanogpt_bin.py`.

Regenerate with `data/prepare.py` (see RUNBOOK.md Session 1), matching the
recorded `train_tokens` / `val_tokens` / `shuffle_buffer`:

| meta file | dataset | train_tokens | val_tokens | shuffle_buffer |
|-----------|---------|-------------|-----------|----------------|
| `owt_gpt2` / `owt_nanogpt`   | openwebtext | 9,000,000,000  | 2,000,000 | (see meta) |
| `dclm_gpt2` / `dclm_nanogpt` | dclm (18B superset; 9B arms read a prefix) | 18,000,001,876 | 3,001,876 | 60000 |
| `wiki_eval`                  | jsonl (decontaminated Wikipedia neutral-eval) | — | 1,086,611 | 0 |

The frozen `wiki_eval` corpus itself is also on HF
(`MIRIBerkeley/data-vs-algorithms-ceg/eval_corpus`, sha256 cbdd72ac…) — the
authoritative copy; regenerate only if HF is unavailable.

`stream_shards` in the DCLM/OWT metas records the per-shard seed + token counts
of the streaming tokenization (each 3B-token shard, same seed 1234) so the
concatenation order is reproducible.
