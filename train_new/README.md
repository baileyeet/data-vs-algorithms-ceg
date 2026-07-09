# New-algorithm arm: modded-nanoGPT

`modded-nanogpt/` is a clone of https://github.com/KellerJordan/modded-nanogpt
(the "new algorithm" A1 in the 2x2 grid). Recipes:

- **124M track**: the main speedrun `train_gpt.py` (Muon, untied embeddings,
  rotary, ReLU^2, FlexAttention, value embeddings, etc.)
- **355M track**: `records/medium_track/` (~5,960 steps, batch 512, bf16) —
  adjust step count to our 9B/18B token budgets, not its native budget
- **1.5B track**: the README's documented 1.5B scaling result — starting recipe
  for Tier 3
- **770M**: no first-party recipe (optional Tier 4; expect LR/warmup tuning)

## Required adaptations (Phase 2 work, tracked here)

Implemented via `train_wrapper.py` (entry point, all measurement logic) plus
`train_gpt_ceg.py` (a copy of `modded-nanogpt/train_gpt.py` with minimal
edits, every one marked `# CEG:`; the upstream clone stays pristine). Launch:

```bash
torchrun --nproc_per_node=8 train_new/train_wrapper.py --size small --arm a1d1 \
  --data-glob datasets/dclm_nanogpt/train_*.bin \
  --neutral-eval-dir datasets/wiki_gpt2 --token-budget 18000000000 \
  --out-dir runs/a1d1_small     # A1D0 adds --n-epochs 2
```

1. **Size parameterization** — `--size small` wired (124M speedrun);
   `--size medium` raises NotImplementedError until `train_gpt_medium.py`
   (355M track) is adapted the same way. DONE (small only).
2. **BPB logging** at log-spaced checkpoints — DONE. The checkpoint schedule
   comes from `common/checkpoint_schedule.checkpoint_tokens`, mapped onto the
   *variable* tokens-per-step schedule (upstream batch size grows 8→16→24
   seqs); BPB uses the same formula/windowing as `common/bpb.py` via a shim
   over the modded forward API (`model(inputs, targets, cu_seqlens, bigram,
   schedule_cfg)`), windows packed per forward with varlen cu_seqlens.
3. **Timed compute** — DONE. Upstream clock convention preserved (starts after
   kernel warmup/compile); our BPB evals + checkpoint saves pause it exactly
   where the upstream val loop did.
4. **Data** — DONE. `--data-glob` takes shards from
   `scripts/convert_to_nanogpt_bin.py`; the patched generator hard-fails past
   `--n-epochs` (no silent wrap) and reshuffles shard order + per-shard
   document traversal with a per-epoch seed (epoch 0 keeps upstream order, so
   single-epoch runs are byte-identical to upstream).
5. **Token budget override** — DONE. Step count is derived from
   `--token-budget` + the upstream stage batch schedule (fractions of total
   steps preserved; extension phase kept at the upstream 10/1380 ratio).
6. **Eval wrapper** (`eval/lm_eval_adapter.py` second `make_lm` path) — TODO;
   checkpoints save the compiled state_dict (`_orig_mod.` prefixes) plus a
   `model_args` dict for reconstruction.
7. **run_config.json parity with train_old** — DONE (`arm`, `gpu_name`,
   `torch_version`, `algorithm: "new_modded_nanogpt"`, `n_params`,
   `ckpt_steps`, `world_size`, ...); `metrics.csv` has the exact same columns
   as `train_old/train.py`.

## Validated locally (Mac, CPU)

- `py_compile` on `train_wrapper.py` and `train_gpt_ceg.py`.
- Unit tests (fake models, CPU): the BPB shim's windowing/masking matches
  `common.bpb.evaluate_bpb` to float32 precision on identical token streams;
  step-count derivation reproduces the upstream 1380/10 split exactly at the
  upstream token total (365,690,880) and preserves stage fractions at other
  budgets (18B → 67,927 scheduled + 492 extension steps); checkpoint-step
  mapping; per-epoch shuffle helpers; .bin shard header IO.

## NOT validated until the first pod session

Upstream hard-requires CUDA + torchrun at import (flash-attn kernels, Muon
distributed comms, torch.compile), so none of the following has actually run:

- The full patched training loop end-to-end (incl. warmup-reset, batch-size
  transitions, the DataExhausted graceful-stop path, DDP barrier behavior
  while rank 0 runs BPB evals — long neutral corpora could approach the NCCL
  barrier timeout; raise `--eval-windows-per-chunk` if so).
- The BPB shim against the *real* compiled model (dtype/shape plumbing of
  `cu_seqlens`/bigram inputs, eval-graph recompiles when attention window
  sizes change stage; recompiles happen while the clock is paused).
- Checkpoint save/reload of the compiled state_dict.
- Loss-EMA magnitudes (upstream train loss includes multi-token-prediction
  weighting, so `train_loss_ema` is inflated early vs plain CE — diagnostic
  only, never used for CEG measurement).
- Toy-scale caveats: `--val-batch-size` must be shrunk so the upstream
  kernel-warmup val pass fits toy val shards, and tiny step counts interact
  crudely with the fixed 300/50-step Muon momentum warmup/cooldown (upstream
  constants, deliberately untouched).

First pod session plan: single-GPU toy run (`--val-batch-size 131072
--eval-windows-per-chunk 16`, few-M token budget) before any paid full run.
