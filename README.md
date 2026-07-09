# Data vs. Algorithms: Compute-Equivalent Gain at Multiple Scales

Isolates how much of GPT-2-era → 2024-era capability progress comes from
**data quality** vs. **algorithms/architecture**, via a 2x2 grid
(old/new data x old/new algorithm) trained at three core GPT-2 sizes
(124M, 355M, 1.5B; 770M optional), measuring compute-to-reference-BPB and
Shapley-decomposing the savings in log-compute space.

| | Old data (OpenWebText) | New data (DCLM/Nemotron-CC) |
|---|---|---|
| **Old algorithm** (GPT-2 repro) | A0D0 | A0D1 |
| **New algorithm** (modded-nanoGPT) | A1D0 | A1D1 |

Token budgets are fixed across sizes (old algo: 9B, new algo: 18B) to mirror
the historical GPT-2 setup. A1D0 repeats OpenWebText for exactly 2 epochs
(explicit, reshuffled — see `common/data_loader.py`).

## Layout

- `configs/model_sizes.json` — the four size configs
- `common/` — GPT-2 model, BPB evaluator, log-spaced checkpoint schedule,
  epoch-aware data loader (shared by training + eval)
- `data/prepare.py` — download/tokenize, parameterized by (dataset, tokenizer)
- `train_old/train.py` — GPT-2 reproduction trainer (A0 arms)
- `train_new/` — modded-nanoGPT clone + adaptation notes (A1 arms)
- `eval/` — neutral-corpus BPB (`neutral_bpb.py`), decontamination
  (`decontam.py`), lm-eval/CORE adapter (`lm_eval_adapter.py`, gated by the
  requirement-#3 validity check)
- `analysis/ceg_shapley.py` — per-size interpolated compute-to-threshold +
  log-space Shapley split (plots added in Phase 5)
- `scripts/check_sizes.py` — param-count verification for all sizes

## Methodology invariants (do not "improve")

1. **BPB everywhere**, never cross-arm per-token CE (tokenizers differ in kind).
2. Reference threshold = BPB on a **fixed decontaminated Wikipedia slice**,
   identical raw text across all runs; per-dataset val BPB is logged but
   labeled same-distribution-only.
3. CORE is secondary, gated by an above-chance validity check at small sizes.
4. Compute = **timed GPU-hours** excluding warmup/compile and eval time; no
   cross-arm raw-FLOP comparisons.
5. Hyperparameters are fixed per (algorithm, size) row — never re-tuned per
   data arm.
6. Log-spaced checkpoints + log-compute interpolation for threshold crossing.
7. Shapley decomposition in log-compute space, reported as multipliers.
8. Tokenizer belongs to "algorithm": 4 tokenized corpus variants total.
9. A1D0 = exactly 2 epochs of OpenWebText, reshuffled between epochs, called
   out in the writeup as that cell's extra confound.

## Toy pipeline (Phase 1/2 validation)

```bash
V=../.venv/bin/python   # shared venv one level up
$V data/prepare.py --dataset openwebtext --tokenizer gpt2 \
  --out datasets/toy_owt_gpt2 --train-tokens 3000000 --val-tokens 100000
$V data/prepare.py --dataset wikipedia --tokenizer gpt2 \
  --out datasets/toy_wiki_gpt2 --val-tokens 150000
$V train_old/train.py --size small --data-dir datasets/toy_owt_gpt2 \
  --neutral-eval-dir datasets/toy_wiki_gpt2 --token-budget 2000000 \
  --total-batch-tokens 16384 --device-batch-size 8 --block-size 256 \
  --n-checkpoints 6 --out-dir runs/toy_a0d0
$V scripts/check_sizes.py
```

Budget/process: ~$10k cap, RunPod on-demand 8xH100 only, **no paid launch
without explicit confirmation**, per-tier cost tracking vs. the $2-5k core
estimate.
