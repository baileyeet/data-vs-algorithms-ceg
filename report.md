# Data vs. Algorithms: Compute-Equivalent Gain Across Scale

2x2 grid (old/new data x old/new algorithm) trained at multiple GPT-2 sizes; compute measured in timed GPU-hours to a fixed neutral-corpus BPB threshold; savings Shapley-decomposed in log-compute space.

**This study reports TWO separate cross-scale curves, never blended, because the 'new algorithm' is not a single fixed thing across all scales:**
- **Current-arch curve (124M, 355M):** A1 = the CURRENT modded-nanoGPT speedrun (SOTA, small-scale-tuned). No reproducible 1.5B recipe exists for it, so this curve STOPS at 355M — the 1.5B point is a disclosed gap (scaling it up needs an unvalidated invented architecture; rejected).
- **ScaleUp-arch curve (124M, 1.5B):** A1 = the 2024 ScaleUp lineage (older, plain transformer), run from its DOCUMENTED per-size recipe at each point. 355M has no documented era-appropriate recipe -> disclosed gap (no hand-derived LR).

The current-arch A1 numbers are v2-canonical (re-derived from yarn_state reruns after the loader-fidelity fixes).

## 124M (small)

Reference threshold: **1.2744 BPB** (neutral corpus; = fully-trained A0D0 at this size).

| Arm | | GPU-hours to threshold | crossing |
|-----|--|------------------------|----------|
| A0D0 | old algo, old data | 4.87 | interpolated |
| A0D1 | old algo, new data | 1.39 | interpolated |
| A1D0 | new algo, old data (OWT) | 0.23 | interpolated |
| A1D1 | new algo, new data | 0.16 | interpolated |

**Shapley split (log-compute):** data 0.800, algorithm 2.616 (sum 3.417).
**As multipliers:** data contributes a **2.23x** compute reduction, algorithm a **13.69x**; product 30.47x vs observed total **30.47x** (consistent).

![training curves](small/curves.png)

![threshold sensitivity](small/sensitivity.png)

## 354M (medium)

Reference threshold: **1.2287 BPB** (neutral corpus; = fully-trained A0D0 at this size).

| Arm | | GPU-hours to threshold | crossing |
|-----|--|------------------------|----------|
| A0D0 | old algo, old data | 13.06 | interpolated |
| A0D1 | old algo, new data | 3.86 | interpolated |
| A1D0 | new algo, old data (OWT) | 3.55 | interpolated |
| A1D1 | new algo, new data | 0.86 | interpolated |

**Shapley split (log-compute):** data 1.319, algorithm 1.401 (sum 2.720).
**As multipliers:** data contributes a **3.74x** compute reduction, algorithm a **4.06x**; product 15.18x vs observed total **15.18x** (consistent).

![training curves](medium/curves.png)

![threshold sensitivity](medium/sensitivity.png)

## Curve 1 — current-arch (124M, 355M)

The SOTA modded-nanoGPT speedrun as A1. Direct test of whether the data/algorithm split is scale-invariant for this algorithm:

![cross-scale](cross_scale.png)

| Size | data x | algorithm x | total x |
|--|--|--|--|
| small | 2.23 | 13.69 | 30.47 |
| medium | 3.74 | 4.06 | 15.18 |

**The algorithm advantage decays sharply with scale (13.1x -> 4.1x from 124M to 355M).** The 1.5B point is a disclosed GAP — no reproducible 1.5B recipe for this arch, and scaling it up requires inventing hand-tuned subsystems (U-net skip topology) with no reference and no way to validate them.

## Curve 2 — ScaleUp-arch (124M, 1.5B)

A1 = the 2024 ScaleUp lineage, run from its DOCUMENTED per-size recipe (each size a coupled bundle: dims + batch + LR + schedule — NOT the 1.5B recipe with rescaled dims, which mis-tunes). A0 = the same GPT-2 baseline. Multipliers are gpu-hours ratios to the per-size threshold.

| Scale | data x | algorithm (DCLM) | algorithm (OWT) |
|--|--|--|--|
| 124M | 3.49x | 2.35x | censored <=1x |
| 1.5B | 3.43x | 2.34x | CENSORED (<=1x; ScaleUp worse than GPT-2-XL on OWT) |

**In sharp contrast to Curve 1, the ScaleUp algorithm's advantage is essentially SCALE-INVARIANT** (~2.35x on new data at both 124M and 1.5B), and it is **data-dependent**: a real advantage on DCLM (new data), but NONE on OWT (old data) at either scale — the ScaleUp arm never crosses the threshold on OWT because GPT-2 matches/beats it there at equal budget (a genuine result, confirmed by equal-budget comparison, not undertraining).

Reading of the two curves together: the current speedrun's huge small-scale edge (13x) largely comes from small-scale-tuned tricks that do not persist, while the older ScaleUp's modest edge (Muon + rotary) is more fundamental and holds across an order of magnitude in scale. **355M is a disclosed gap** for this lineage (no documented era-appropriate recipe; no hand-derived LR).

## CORE-subset (secondary, validity-gated)

DCLM CORE is a secondary check; BPB is primary. A task is used only where the A0D0 reference clears chance by >=2 sigma at its final checkpoint. lambada is excluded for A1 arms (the modded loader has no logits path). All A1 CORE is from the v2 sweep.

- **124M**: 5 usable tasks (arc_easy, copa, hellaswag, piqa, xwinograd_en).
- **355M**: 6 usable tasks (arc_easy, boolq, copa, hellaswag, piqa, xwinograd_en).

Note: `boolq` drops at 124M (A0D0 sits at chance, 0.504) but clears at 355M — a scale effect, not an error. Arms cluster closely on CORE at these scales (limit=500, near-noisy), so no quantitative CORE-based CEG is claimed; it is a sanity gate.

## Methodology notes & confounds

- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia slice (tokenizer-invariant; identical raw text across all runs). Per-dataset val BPB was logged as same-distribution diagnostics only.
- Decontamination of the eval corpus (13-gram overlap against the actual tokenized training samples) removed 562/2766 candidate docs (20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages quote Wikipedia heavily — and a further 153 (5.5%) matched only DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, sha256 cbdd72ac..., identical across every arm, size, and tier.
- Compute is timed GPU-hours on the runs' actual hardware, excluding kernel-warmup/compile and eval time; raw FLOPs are never compared across arms (precision differs).
- Hyperparameters are fixed per (algorithm, size) row; verified identical between data arms with scripts/verify_row_hparams.py.
- Token budget is part of the algorithm bundle: the A1 (modded) arms train at 2x the upstream-native budget for each size (small ~731M, medium ~4.35B tokens), NOT a fixed cross-arm budget. The original fixed-18B design (which forced A1D0 to repeat OWT for 2 epochs) was abandoned after the recipe collapsed under ~50x schedule stretch. At the 2x-native budgets both A1 arms are single-pass — well under one epoch of their ~9B+ corpora — so there is no epoch-repetition confound in any cell.
- Threshold crossings are interpolated in log-compute between checkpoints (log-spaced, denser early), never snapped.
- Reference-threshold definition: mean neutral BPB over all checkpoints in the final 10% of A0D0's training (per size). The single-final-checkpoint variant is reported alongside as robustness. Why a range exists at 124M: the threshold sits on the threshold arm's end-of-training plateau, where its curve is flattest; with the original purely log-spaced checkpoint schedule the plateau was sampled sparsely, so vertical noise of order the same-seed rerun floor (~0.01 BPB) translated into ~20% swings in that arm's compute-to-threshold and hence ~±8% (data) / ~±16% (total) in the multipliers. Fixed going forward by adding linearly-spaced checkpoints over the final 10% of every arm's schedule; the reported range brackets the definitional freedom on the pre-fix 124M data.
- DCLM CORE scores are secondary and validity-gated (tasks near chance at small scale are excluded).
