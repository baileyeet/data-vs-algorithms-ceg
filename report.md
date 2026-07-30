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

## All configurations

Every arm on one figure: neutral BPB vs GPU-hours for all 16 setups (4 scale-points × A0/A1 old/new algorithm × D0/D1 old/new data), each panel zoomed to where its arms cross the reference BPB. The crossing GPU-hours are the raw material for the multipliers below. The ScaleUp A1D0 (new-algo/old-data) visibly never crosses at either ScaleUp scale (the censored cells).

![all 16 configurations](all_configs.png)

## Both curves at a glance (multiplier summary)

The 16 arms distilled to the compute-equivalent-gain multipliers (data / algorithm Shapley split) for both lineages across scale. The two curves are kept strictly separate (different A1 generations); each is missing the one scale where it has no validated recipe.

![multipliers vs scale](multipliers_vs_scale.png)

## Curve 1 — current-arch (124M, 355M)

The SOTA modded-nanoGPT speedrun as A1. Direct test of whether the data/algorithm split is scale-invariant for this algorithm:

| Size | data x | algorithm x | total x |
|--|--|--|--|
| small | 2.23 | 13.69 | 30.47 |
| medium | 3.74 | 4.06 | 15.18 |

**The algorithm advantage decays sharply with scale (13.1x -> 4.1x from 124M to 355M).** The 1.5B point is a disclosed GAP — no reproducible 1.5B recipe for this arch, and scaling it up requires inventing hand-tuned subsystems (U-net skip topology) with no reference and no way to validate them.

## Curve 2 — ScaleUp-arch (124M, 1.5B)

A1 = the 2024 ScaleUp lineage, run from its DOCUMENTED per-size recipe (each size a coupled bundle: dims + batch + LR + schedule — NOT the 1.5B recipe with rescaled dims, which mis-tunes). A0 = the same GPT-2 baseline. Multipliers are gpu-hours ratios to the per-size threshold.

| Scale | data x | algorithm (DCLM) | algorithm (OWT) |
|--|--|--|--|
| 124M | 3.25x | 2.9x | censored <=1x |
| 1.5B | 3.43x | 2.34x | CENSORED (<=1x; ScaleUp worse than GPT-2-XL on OWT) |

The ScaleUp algorithm's advantage on new data **declines mildly with scale (2.90x -> 2.34x, 124M -> 1.5B)** — a gentle decay, versus the current arch's steep 13.1x -> 4.1x. And it is **data-dependent**: a real advantage on DCLM (new data), but NONE on OWT (old data) at either scale — the ScaleUp arm never crosses the threshold on OWT because GPT-2 matches/beats it there at equal budget (a genuine result, confirmed by equal-budget comparison, not undertraining). The data multiplier, by contrast, is roughly stable across scale (~3.3x).

NOTE (hardware): the ScaleUp A1 arms and their GPT-2 A0 baseline were all measured on 5xH100 (the A0-124M baseline was re-run on 5 GPUs for this — an 8-vs-5 mix had distorted the 124M algo multiplier to 2.35x; the consistent value is 2.90x). GPU-hours is NOT cleanly count-invariant here (the forced batch/accum change cost ~22%).

Reading of the two curves together: **both algorithms' advantages shrink with scale, but the more aggressively small-scale-tuned current speedrun decays far faster (from a much higher base) than the older, more fundamental ScaleUp (Muon + rotary).** **355M is a disclosed gap** for the ScaleUp lineage (no documented era-appropriate recipe; no hand-derived LR).

## CORE-subset (secondary, validity-gated)

DCLM CORE is a secondary check; BPB is primary. A task is used only where the A0D0 reference clears chance by >=2 sigma at its final checkpoint. `lambada_openai` is open-vocabulary (no fixed chance) so it never enters the quantitative gate at any scale; separately, the current-arch (modded) A1 loader has no logits path so its lambada is invalid by construction, whereas the ScaleUp-arch A1 arms (1.5B, ScaleUp-124M) use a plain-causal adapter with a real logits path and a valid lambada accuracy (reported as a diagnostic only). All A1 CORE is from the v2 sweep.

- **124M**: 5 usable tasks (arc_easy, copa, hellaswag, piqa, xwinograd_en).
- **355M**: 6 usable tasks (arc_easy, boolq, copa, hellaswag, piqa, xwinograd_en).
- **1.5B**: 6 usable tasks (arc_easy, boolq, copa, hellaswag, piqa, xwinograd_en).
- **ScaleUp-124M**: 5 usable tasks (arc_easy, copa, hellaswag, piqa, xwinograd_en).

![CORE task accuracy vs model size](core_vs_scale.png)

Note: `boolq` drops at 124M / ScaleUp-124M (A0D0 sits at chance, 0.504) but clears from 355M up — a scale effect, not an error. Arms cluster closely on the gated tasks (limit=500, near-noisy), with slightly more separation at 1.5B but still within the noise floor, so no quantitative CORE-based CEG is claimed at any scale; it is a sanity gate. The ScaleUp-arch A1 lambada diagnostic rises cleanly with scale (acc 0.32 at 124M -> 0.52/0.55 at 1.5B; perplexity 55 -> 8), confirming the plain-causal adapter's logits path is sound.

## Methodology notes & confounds

- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia slice (tokenizer-invariant; identical raw text across all runs). Per-dataset val BPB was logged as same-distribution diagnostics only.
- Decontamination of the eval corpus (13-gram overlap against the actual tokenized training samples) removed 562/2766 candidate docs (20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages quote Wikipedia heavily — and a further 153 (5.5%) matched only DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, sha256 cbdd72ac..., identical across every arm, size, and tier.
- Compute is timed GPU-hours on the runs' actual hardware, excluding kernel-warmup/compile and eval time; raw FLOPs are never compared across arms (precision differs).
- Hyperparameters are fixed per (algorithm, size) row; verified identical between data arms with scripts/verify_row_hparams.py.
- Token budget is part of the algorithm bundle: the A1 (modded) arms train at 2x the upstream-native budget for each size (small ~731M, medium ~4.35B tokens), NOT a fixed cross-arm budget. The original fixed-18B design (which forced A1D0 to repeat OWT for 2 epochs) was abandoned after the recipe collapsed under ~50x schedule stretch. At the 2x-native budgets both A1 arms are single-pass — well under one epoch of their ~9B+ corpora — so there is no epoch-repetition confound in any cell.
- Threshold crossings are interpolated in log-compute between checkpoints (log-spaced, denser early), never snapped.
- Reference-threshold definition: mean neutral BPB over all checkpoints in the final 10% of A0D0's training (per size). The single-final-checkpoint variant is reported alongside as robustness. Why a range exists at 124M: the threshold sits on the threshold arm's end-of-training plateau, where its curve is flattest; with the original purely log-spaced checkpoint schedule the plateau was sampled sparsely, so vertical noise of order the same-seed rerun floor (~0.01 BPB) translated into ~20% swings in that arm's compute-to-threshold and hence ~±8% (data) / ~±16% (total) in the multipliers. Fixed going forward by adding linearly-spaced checkpoints over the final 10% of every arm's schedule; the reported range brackets the definitional freedom on the pre-fix 124M data.
- DCLM CORE scores are secondary and validity-gated (tasks near chance at small scale are excluded).
