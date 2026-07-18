# Data vs. Algorithms: Compute-Equivalent Gain Across Scale

2x2 grid (old/new data x old/new algorithm) trained at multiple GPT-2 sizes; compute measured in timed GPU-hours to a fixed neutral-corpus BPB threshold; savings Shapley-decomposed in log-compute space.

**Scope / status.** Tiers 1 (124M) and 2 (355M) are final and v2-canonical (all A1 numbers re-derived from yarn_state reruns after the loader-fidelity fixes). Tier 3 (1.5B) is in progress; its A1 arm carries a specific confound flagged below.

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

## Cross-scale result

Shapley multipliers as a function of model size — the direct test of whether the data/algorithm split is scale-invariant:

![cross-scale](cross_scale.png)

| Size | data x | algorithm x | total x |
|--|--|--|--|
| small | 2.23 | 13.69 | 30.47 |
| medium | 3.74 | 4.06 | 15.18 |

## CORE-subset (secondary, validity-gated)

DCLM CORE is a secondary check; BPB is primary. A task is used only where the A0D0 reference clears chance by >=2 sigma at its final checkpoint. lambada is excluded for A1 arms (the modded loader has no logits path). All A1 CORE is from the v2 sweep.

- **124M**: 5 usable tasks (arc_easy, copa, hellaswag, piqa, xwinograd_en).
- **355M**: 6 usable tasks (arc_easy, boolq, copa, hellaswag, piqa, xwinograd_en).

Note: `boolq` drops at 124M (A0D0 sits at chance, 0.504) but clears at 355M — a scale effect, not an error. Arms cluster closely on CORE at these scales (limit=500, near-noisy), so no quantitative CORE-based CEG is claimed; it is a sanity gate.

## Tier 3 (1.5B): the algorithm-version confound (READ FIRST)

**The 1.5B A1 arm uses a DIFFERENT, older algorithm generation than the 124M/355M A1 arms — this must front the interpretation of any 1.5B result, not sit in a footnote.** The current modded-nanoGPT speedrun (used at 124M/355M) has no first-party, reproducible 1.5B recipe: scaling its architecture to ~48 layers would require inventing several hand-tuned subsystems (notably a U-net skip topology) with no reference and no divergence signature if wrong. The only reproducible first-party 1.5B result is the 2024 'ScaleUp1B' — a PLAIN transformer (standard attention, base rotary, ReLU^2 MLP, weight tying, old Muon+AdamW; none of the current YaRN/split-embed/value-embed/skip machinery). We use that (Option A: fully reproducible) for the 1.5B A1 arm.

Consequence: the algorithm axis is NOT held fixed across scale at 1.5B. Any change in the algorithm multiplier from 355M to 1.5B is confounded with this architecture-generation change and must be read as a lower-bound-flavoured estimate of the current algorithm's scaling, not a clean same-algorithm measurement. (The A0 arm is the standard GPT-2-XL scale-up, so the data axis is unaffected.)

## Methodology notes & confounds

- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia slice (tokenizer-invariant; identical raw text across all runs). Per-dataset val BPB was logged as same-distribution diagnostics only.
- Decontamination of the eval corpus (13-gram overlap against the actual tokenized training samples) removed 562/2766 candidate docs (20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages quote Wikipedia heavily — and a further 153 (5.5%) matched only DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, sha256 cbdd72ac..., identical across every arm, size, and tier.
- Compute is timed GPU-hours on the runs' actual hardware, excluding kernel-warmup/compile and eval time; raw FLOPs are never compared across arms (precision differs).
- Hyperparameters are fixed per (algorithm, size) row; verified identical between data arms with scripts/verify_row_hparams.py.
- Token budget is part of the algorithm bundle: the A1 (modded) arms train at 2x the upstream-native budget for each size (small ~731M, medium ~4.35B tokens), NOT a fixed cross-arm budget. The original fixed-18B design (which forced A1D0 to repeat OWT for 2 epochs) was abandoned after the recipe collapsed under ~50x schedule stretch. At the 2x-native budgets both A1 arms are single-pass — well under one epoch of their ~9B+ corpora — so there is no epoch-repetition confound in any cell.
- Threshold crossings are interpolated in log-compute between checkpoints (log-spaced, denser early), never snapped.
- Reference-threshold definition: mean neutral BPB over all checkpoints in the final 10% of A0D0's training (per size). The single-final-checkpoint variant is reported alongside as robustness. Why a range exists at 124M: the threshold sits on the threshold arm's end-of-training plateau, where its curve is flattest; with the original purely log-spaced checkpoint schedule the plateau was sampled sparsely, so vertical noise of order the same-seed rerun floor (~0.01 BPB) translated into ~20% swings in that arm's compute-to-threshold and hence ~±8% (data) / ~±16% (total) in the multipliers. Fixed going forward by adding linearly-spaced checkpoints over the final 10% of every arm's schedule; the reported range brackets the definitional freedom on the pre-fix 124M data.
- DCLM CORE scores are secondary and validity-gated (tasks near chance at small scale are excluded).
