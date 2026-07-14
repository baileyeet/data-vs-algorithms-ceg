# Data vs. Algorithms: Compute-Equivalent Gain Across Scale

2x2 grid (old/new data x old/new algorithm) trained at multiple GPT-2 sizes; compute measured in timed GPU-hours to a fixed neutral-corpus BPB threshold; savings Shapley-decomposed in log-compute space.

## 124M (small)

Reference threshold: **1.2653 BPB** (neutral corpus; = fully-trained A0D0 at this size).

| Arm | | GPU-hours to threshold | crossing |
|-----|--|------------------------|----------|
| A0D0 | old algo, old data | 5.78 | interpolated |
| A0D1 | old algo, new data | 1.56 | interpolated |
| A1D0 | new algo, old data (2-epoch OWT) | 0.27 | interpolated |
| A1D1 | new algo, new data | 0.17 | interpolated |

**Shapley split (log-compute):** data 0.892, algorithm 2.650 (sum 3.541).
**As multipliers:** data contributes a **2.44x** compute reduction, algorithm a **14.15x**; product 34.52x vs observed total **34.52x** (consistent).

![training curves](small/curves.png)

![threshold sensitivity](small/sensitivity.png)

## Methodology notes & confounds

- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia slice (tokenizer-invariant; identical raw text across all runs). Per-dataset val BPB was logged as same-distribution diagnostics only.
- Decontamination of the eval corpus (13-gram overlap against the actual tokenized training samples) removed 562/2766 candidate docs (20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages quote Wikipedia heavily — and a further 153 (5.5%) matched only DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, sha256 cbdd72ac..., identical across every arm, size, and tier.
- Compute is timed GPU-hours on the runs' actual hardware, excluding kernel-warmup/compile and eval time; raw FLOPs are never compared across arms (precision differs).
- Hyperparameters are fixed per (algorithm, size) row; verified identical between data arms with scripts/verify_row_hparams.py.
- A1D0 trains on exactly 2 reshuffled epochs of OpenWebText (the corpus has ~9B unique tokens vs the arm's 18B budget). All other arms are single-pass. This is a known, deliberate confound in that one cell; literature places safe repetition at <=4 epochs.
- Threshold crossings are interpolated in log-compute between checkpoints (log-spaced, denser early), never snapped.
- DCLM CORE scores are secondary and validity-gated (tasks near chance at small scale are excluded).
