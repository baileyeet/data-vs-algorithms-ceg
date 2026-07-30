# New-algorithm arm: modded-nanoGPT

`modded-nanogpt/` is a clone of https://github.com/KellerJordan/modded-nanogpt
(the "new algorithm" A1 in the 2x2 grid). Recipes:

- **124M track**: the main speedrun `train_gpt.py` (Muon, untied embeddings,
  rotary, ReLU^2, FlexAttention, value embeddings, etc.)
- **355M track**: `records/medium_track/` (~5,960 steps, batch 512, bf16) —
  adjust step count to our 9B/18B token budgets, not its native budget
- **1.5B track**: the upstream documented 1.5B scaling result (the 2024
  ScaleUp1B recipe) — used for the Tier-3 / ScaleUp-curve A1 arms via
  `train_gpt_xl_ceg.py`

## Required adaptations (all implemented)

Implemented via `train_wrapper.py` (entry point, all measurement logic) plus
`train_gpt_ceg.py` / `train_gpt_medium_ceg.py` (copies of
`modded-nanogpt/train_gpt.py` and `train_gpt_medium.py` with minimal edits,
every one marked `# CEG:`; the upstream clone stays pristine). Launch:

```bash
torchrun --nproc_per_node=8 train_new/train_wrapper.py --size small --arm a1d1 \
  --data-glob datasets/dclm_nanogpt/train_*.bin \
  --neutral-eval-dir datasets/wiki_gpt2 --token-budget 18000000000 \
  --out-dir runs/a1d1_small     # A1D0 adds --n-epochs 2
```

1. **Size parameterization** — `--size small` (124M speedrun,
   `train_gpt_ceg.py`) and `--size medium` (355M track,
   `train_gpt_medium_ceg.py`) both wired. DONE.
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
6. **Eval wrapper** — DONE. Checkpoints save the compiled state_dict
   (`_orig_mod.` prefixes) plus a `model_args` dict for reconstruction; the
   CORE loglikelihood adapters are `eval/lm_eval_adapter_modded.py` (current-
   arch A1, with the yarn_state / split_embed reload-fidelity handling) and
   `eval/lm_eval_adapter_scaleup.py` (the plain-causal 2024-ScaleUp A1).
7. **run_config.json parity with train_old** — DONE (`arm`, `gpu_name`,
   `torch_version`, `algorithm: "new_modded_nanogpt"` /
   `"new_modded_nanogpt_medium"`, `n_params`, `ckpt_steps`, `world_size`,
   ...); `metrics.csv` has the exact same columns as `train_old/train.py`.

## Medium track (355M, `--size medium`)

`train_gpt_medium_ceg.py` is the same adaptation applied to
`modded-nanogpt/train_gpt_medium.py` (native recipe: 4,700 scheduled + 40
extension steps, 2,176,974,848 tokens, batch ramp 131,072 -> 262,144 ->
393,216 -> 524,288 tokens/step). Launch is identical apart from the size flag:

```bash
torchrun --nproc_per_node=8 train_new/train_wrapper.py --size medium --arm a1d1 \
  --data-glob datasets/dclm_nanogpt/train_*.bin \
  --neutral-eval-dir datasets/wiki_gpt2 --token-budget 18000000000 \
  --out-dir runs/a1d1_medium    # A1D0 adds --n-epochs 2
```

Structural differences from the small track, and how they were handled:

- **Batch schedule**: not stage-based (`TRAINING_STAGES.duration`) but a
  12-entry `train_bs_schedule` indexed by
  `int(12 * step/num_scheduled_iterations)` (equal 1/12 slices) plus a flat
  `train_bs_extension`. `train_wrapper.derive_step_counts_medium` /
  `simulate_step_tokens_medium` replicate that indexing exactly (including the
  float division) and solve for the smallest scheduled count whose total
  tokens >= `--token-budget`, keeping the upstream 40/4700 extension ratio.
  Because `get_bs()` indexes by step fraction, the schedule's internal
  fractions are preserved automatically at any scheduled count. The medium
  file also re-derives `args.num_iterations` (a dataclass field precomputed
  from class defaults) to keep `num_iterations = scheduled + extension`.
- **Forward signature**: `model(inputs, targets, cu_seqlens, schedule_cfg)` —
  no bigram-hash inputs — so the BPB shim uses
  `_make_eval_forward_medium` (dispatched in `run_checkpoint` by the absence
  of `get_bigram_hash`).
- **Eval loss reduction**: upstream medium's eval branch returns a *scalar
  mean* CE (small returned per-position losses), which cannot mask the final
  BPB window's padding. The one model-code CEG edit switches that eval-only
  branch to `reduction="none"` (same chunking/softcap pipeline); training loss
  math is untouched.
- **Data loader**: `BOSFinder`/`DataPreloader` instead of `Shard`; the same
  epoch semantics were added (epoch 0 byte-identical to upstream incl. the
  quickload partial-scan path, per-epoch shard + document reshuffle,
  `DataExhausted` past `--n-epochs`; the preloader thread publishes `None`
  instead of dying on iterator exhaustion, which upstream would hang on).
- **Optimizers**: three separate optimizers (`DistAdam` x2 + `NorMuon`)
  instead of the combined `NorMuonAndAdam`; `write_run_config`/`run_checkpoint`
  branch on `hasattr(training_manager, "optimizer")` to read initial/current
  LRs, and checkpoint `model_args` are pulled from `blocks[0].attn` (the
  medium GPT keeps head dims there).

## Validated locally (Mac, CPU)

- `py_compile` on `train_wrapper.py`, `train_gpt_ceg.py` and
  `train_gpt_medium_ceg.py`.
- Unit tests (fake models, CPU): the BPB shim's windowing/masking matches
  `common.bpb.evaluate_bpb` to float32 precision on identical token streams;
  step-count derivation reproduces the upstream 1380/10 split exactly at the
  upstream token total (365,690,880) and preserves stage fractions at other
  budgets (18B → 67,927 scheduled + 492 extension steps); checkpoint-step
  mapping; per-epoch shuffle helpers; .bin shard header IO.
- Medium unit tests (CPU): `simulate_step_tokens_medium` matches a verbatim
  re-implementation of the upstream `get_bs()` step-for-step at the native
  and scaled step counts; the solver reproduces the native medium recipe
  exactly (2,176,974,848 tokens → 4,700 + 40) and is minimal at our budgets
  (9B → 19,431 + 165; 18B → 38,860 + 331); checkpoint mapping ends at the
  final step; the small-track derivation results above are unchanged by the
  wrapper refactor. The patched `BOSFinder` (extracted, CPU torch): epoch 0
  is behaviorally identical to upstream (with and without quickload), and
  permuted epochs keep the loader invariants (starts are BOS positions,
  per-rank token counts exact, doc extents from natural-order successors)
  while following the per-epoch permutation deterministically.

## Validation status (all subsequently exercised)

The items below were CUDA-only and unrun at adaptation time (upstream
hard-requires CUDA + torchrun at import: flash-attn kernels, Muon distributed
comms, torch.compile). All were exercised end-to-end across the full study
(Tiers 1–3, both tracks — see `report.md`):

- The full patched training loop (warmup-reset, batch-size transitions, the
  DataExhausted graceful stop, DDP-barrier behavior during rank-0 BPB evals)
  ran clean; the BPB shim on the real compiled model, the medium
  `reduction="none"` eval branch, and NorMuon's 8-GPU param grouping all held.
- **Checkpoint save/reload was the one that bit.** Post-hoc loading of the
  compiled state_dict needed four fixes — yarn_state persistence,
  eager-vs-compiled numerics, medium split_embed restoration, and eval-packing
  parity — all resolved and validated 8/8 exact (deltas 0.0000). Full writeup:
  the "Loader saga" in `CLAUDE.md`.
- Loss-EMA magnitudes are MTP-weighted (inflated early vs plain CE) —
  diagnostic only, never used for CEG measurement.

Toy-scale caveats still apply if re-running the smoke test: shrink
`--val-batch-size` so the upstream kernel-warmup val pass fits toy val shards,
and note tiny step counts interact crudely with the fixed 300/50-step Muon
momentum warmup/cooldown (upstream constants, deliberately untouched).
