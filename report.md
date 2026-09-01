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

The 16 arms distilled to compute-equivalent-gain multipliers for both lineages across scale. The two curves are kept strictly separate (different A1 generations); each is missing the one scale where it has no validated recipe.

![multipliers vs scale](multipliers_vs_scale.png)

**The two curves are different estimators — not directly comparable.** The current-arch points are true compute-Shapley values: all four cells cross the threshold, so the data effect is averaged over both rows and the algorithm effect over both columns (in log-compute, data = ½·[(A0D0−A0D1)+(A1D0−A1D1)], algorithm = ½·[(A0D0−A1D0)+(A0D1−A1D1)]). For ScaleUp, A1D0 (ScaleUp on OpenWebText) never crosses, so one term in each average is censored and no symmetric Shapley exists; the plotted ScaleUp points are the single surviving margin — data = the A0 (GPT-2) row ratio, algorithm = the D1 (DCLM) column ratio. The censoring biases the two axes in **opposite** directions:

- **Algorithm** — the censored complement is the old-data (D0) column, where ScaleUp is *worse* than GPT-2 (≤1×). A full Shapley would average the plotted new-data margin (2.9×→2.3×) with a ≤1× term and sit **below** it, so the plotted algorithm multiplier **over-states** the balanced value.
- **Data** — the censored complement is the ScaleUp (A1) row, where ScaleUp cannot cross *at all* on old data (an effectively unbounded data multiplier). A full Shapley would sit **above** the plotted A0-row margin (3.2×→3.4×), so the plotted data multiplier **under-states** it.

So the single-margin ScaleUp numbers bound a full Shapley from opposite sides on the two axes. (Multipliers are within-hardware GPU-hour ratios — current-arch 8-GPU, ScaleUp 5-GPU — so the count overhead cancels in each ratio.)

The figure's algorithm panel also carries a THIRD estimator — the Exp B architecture lineages (Pythia, SmolLM2) as open ▽ at 1× — which are algorithm-CEG vs a matched GPT-2 (data fixed, no data-panel entry) and are all censored (≤1×, none cross). See the Exp B section below.

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

## Exp A — data-era ladder (@124M)

Follow-on to the 2x2 study: hold the algorithm axis and sweep the DATA corpus across release-years — OWT (2019), C4 (2020), RefinedWeb (2023), DCLM (2024) — to see how data-quality and algorithm contributions move with dataset vintage. Wikipedia (union-decontam) is the neutral eval, never a train corpus; CEG is GPU-hours-to-threshold vs the old-algo-OWT baseline. @124M = matched DIMENSIONS (12L/12H/768d), NOT param count (old-algo 123,689,472; current-arch 498,773,000 — the value-embed/U-net additions ARE the algorithm being measured).

Corrected 2x2 (union eval, all arms torch 2.10): threshold **1.2760 BPB**; **data 2.39×, algorithm 16.10×, total 38.49×** (vs published 2.23/13.69/30.5× — the shift is union-eval + same-seed variance, no torch component; both are torch 2.10).

| Dataset (year) | old-algo CEG | current-arch CEG | algorithm CEG at that corpus |
|--|--|--|--|
| OWT (2019) | 1.0× | 23.8× | 23.8× |
| C4 (2020) | censored | censored | censored |
| RefinedWeb (2023) | 3.3× | 37.9× | 11.6× |
| DCLM (2024) | 3.5× | 38.5× | 10.9× |

Data-quality is **NON-monotonic in release year**: C4 (2020) is CENSORED under BOTH algorithms (never reaches the OWT threshold — a WORSE training corpus than 2019 OWT), while RefinedWeb (2023) and DCLM (2024) do improve. So 'newer dataset' ≠ 'better data'. The algorithm CEG stays large across corpora (see the OWT/RefinedWeb/DCLM column).

![Exp A: CEG vs dataset release-year](era_ladder.png)

The raw training curves the CEG numbers are read from (BPB vs GPU-hours, per corpus; where each arm crosses the threshold):

![Exp A training curves per corpus](era_curves.png)

### Exp A — CORE downstream tasks (secondary)

The CEG result above is on bits-per-byte. As a downstream check we score all four era corpora on the study's CORE task subset (limit 500) at 124M — old-algorithm vs new-algorithm. (OWT and DCLM reuse the completed 2×2 study's 124M CORE; C4 and RefinedWeb come from a faithful retrain, since those era checkpoints had been lost.)

| Corpus (year) | usable tasks | old-algo mean acc | new-algo mean acc | Δ (new−old) |
|--|--|--|--|--|
| OWT (2019) | 5 | 0.528 | 0.519 | -0.009 |
| C4 (2020) | 7 | 0.503 | 0.511 | +0.008 |
| RefinedWeb (2023) | 6 | 0.554 | 0.536 | -0.019 |
| DCLM (2024) | 6 | 0.532 | 0.553 | +0.021 |

At 124M and limit=500 the old-vs-new gap is within ±0.02 (≈ stderr) at every corpus, so CORE here is a qualitative sanity check, not a quantitative CEG claim — consistent with how the core study treats CORE. The data-quality signal lives in BPB/CEG (where RefinedWeb and DCLM clearly help); downstream tasks at this scale don't resolve the algorithm difference.

![Exp A CORE downstream accuracy across data eras](core_era_ladder.png)

Per-task breakdown (old vs new algorithm across the four corpora; the two overlap within ±1 stderr on essentially every task, so read the aggregate, not any single panel):

![Exp A CORE accuracy per task](core_era_by_task.png)

## Exp B — architecture landscape (Transformer lineages vs matched GPT-2)

The completed study found a large small-scale *algorithm* CEG for the current-arch speedrun (13.7× @124M). Exp B is the direct test of whether that generalizes beyond a small-scale-optimized speedrun: it trains PUBLISHED open-model lineages — **Pythia (GPT-NeoX, 2023)** and **SmolLM2 (Llama, 2024)** — from scratch on fixed data (OpenWebText), each against a size-matched GPT-2 baseline through the identical harness, and asks whether any reaches (crosses) the GPT-2 baseline's neutral-BPB threshold. Data is held fixed → no data/algorithm 2×2; this is algorithm-CEG only.

**Result: no lineage crosses at any scale (135M–1.7B) → algorithm-CEG ≤1× everywhere (no measurable gain over a matched, properly-tuned GPT-2).** Best case is SmolLM2-135M, whose gap is within same-seed noise of parity (still not a crossing). A direct empirical 'no' to whether the current-arch small-scale advantage generalizes to these lineages. Exp B is the censored (open ▽ at 1×) markers on the algorithm panel of `multipliers_vs_scale.png` above.

Pre-registered verdict rule: |delta| within ±0.013 neutral BPB of the matched GPT-2 = parity-within-noise; ≥0.026 (2σ) = significant deficit. delta = arch tail-mean neutral BPB − its matched GPT-2 (both @512k); >0 = worse.

| Lineage | size | Δ BPB vs matched GPT-2 | algorithm-CEG |
|--|--|--|--|
| pythia | 160M | +0.082 | ≤1× (censored, no crossing) |
| pythia | 410M | +0.020 | ≤1× (censored, no crossing) |
| pythia | 1.4B | +0.030 | ≤1× (censored, no crossing) |
| smollm2 | 135M | +0.009 | ≤1× (censored, no crossing) |
| smollm2 | 360M | +0.054 *(divergence-confounded)* | ≤1× (censored, no crossing) |
| smollm2 | 1.7B | +0.120 *(divergence-confounded)* | ≤1× (censored, no crossing) |

Pythia is clean at every scale (deficit shrinks 160M→410M then stabilizes ~0.02–0.03). SmolLM2-135M is parity-within-noise (confirmed with a 2nd seed). SmolLM2-360M/1.7B are deficits but **divergence-confounded**: even at each size's documented LR they overfit OWT under the fixed ~8.87B budget (neutral BPB rises off its own minimum while own-val keeps falling), an effect that grows with size — the deficit verdict is robust (holds on the best/min BPB too) but the exact magnitude is inflated.

Methodology: like-for-like train_hf denominators at every size (a matched GPT-2 through the SAME harness — verified equivalent to the study's train_old GPT-2 to within the ±0.013 same-seed noise floor at ALL scales incl 1.4B); the undertraining regime biases toward parity (not just noise), so all arms are compared at convergence.

Raw training curves (BPB vs GPU-hours), each architecture vs its size-matched GPT-2 — the candidate curve stays at or above the GPT-2 threshold at every size:

![Exp B architecture-axis training curves](expb_arch_curves.png)

### Exp B — data axis (the new architectures across data eras)

B1 above held data fixed (OWT). This closes the grid: each new architecture (at small size) is trained from scratch on all four data-era corpora and measured against ONE fixed external bar — the size-matched GPT-2 trained on OWT (the same denominator as B1). Using one external bar (not each architecture's own OWT quality) is deliberate: it makes 'crosses the bar' mean 'beats a matched GPT-2', which is the headline. data-CEG = GPU-hours for that GPT-2-OWT baseline to reach the bar ÷ GPU-hours for the architecture-on-corpus to reach the same bar; an arm that never reaches it is censored.

| Architecture | corpus (year) | final BPB | vs matched GPT-2-OWT bar | data-CEG |
|--|--|--|--|--|
| Pythia-160M | OWT (2019) | 1.3327 | censored (loses) | — |
| Pythia-160M | C4 (2020) | 1.3272 | censored (loses) | — |
| Pythia-160M | RefinedWeb (2023) | 1.1928 | crosses (4.6×) | 4.58× |
| Pythia-160M | DCLM (2024) | 1.1788 | crosses (5.4×) | 5.36× |
| SmolLM2-135M | OWT (2019) | 1.2833 | censored (loses) | — |
| SmolLM2-135M | C4 (2020) | 1.2783 | censored (loses) | — |
| SmolLM2-135M | RefinedWeb (2023) | 1.1318 | crosses (5.4×) | 5.39× |
| SmolLM2-135M | DCLM (2024) | 1.1283 | crosses (6.4×) | 6.43× |

**Headline: better data flips both new architectures from losing to a matched GPT-2 to beating it.** On OWT (2019) and C4 (2020) both are censored — they never reach the GPT-2-OWT bar (the B1 result). On RefinedWeb (2023) and DCLM (2024) both cross it, reaching GPT-2's OWT quality with 4.6–6.4× less compute. The data lever is far larger than any architecture lever we found (in B1, no new architecture beat GPT-2 on OWT at all).

**Cross-validation of Exp A.** C4 (2020) comes out *worse* than OWT (2019) for BOTH architectures independently (both censored; final BPB higher on C4 than the crossing corpora, and OWT slightly better than C4). Exp A found exactly this non-monotonicity in dataset release-year using the original two algorithms; seeing it reproduce on two further, independent architectures (GPT-NeoX and Llama lineages) is direct cross-validation that the effect is a property of the data, not of a particular algorithm.

![Exp B data axis: CEG vs data-era per architecture](data_ladder_expB.png)

The raw training curves (BPB vs GPU-hours): on OWT/C4 both architectures stay above the GPT-2-OWT bar (censored), on RefinedWeb/DCLM they dive below it (cross):

![Exp B data-axis training curves](expb_data_curves.png)

### Exp B — CORE downstream tasks (secondary)

The BPB/CEG result above is compute-efficiency on a language-modeling bar. As a downstream-task check we ran the study's CORE suite (11 tasks, limit 500) on all 14 Exp B checkpoints and compared each architecture to its size-matched GPT-2 through the same harness. CORE is SECONDARY and noisy at limit=500; the gap is the mean per-task accuracy difference (candidate − matched GPT-2), ±1 stderr.

| Lineage | size | CORE mean-acc gap vs GPT-2 | per-task (W/T/L of 11) |
|--|--|--|--|
| pythia | 160M | -0.009 ± 0.011 (0.83σ) | 4W/2T/5L |
| pythia | 410M | -0.013 ± 0.011 (1.26σ) | 3W/3T/5L |
| pythia | 1.4B | -0.022 ± 0.011 (2.12σ) | 1W/1T/9L |
| smollm2 | 135M | +0.017 ± 0.010 (1.65σ) | 7W/2T/2L |
| smollm2 | 360M | +0.018 ± 0.010 (1.79σ) | 7W/3T/1L |
| smollm2 | 1.7B | +0.011 ± 0.010 (1.07σ) | 5W/2T/4L |

Two things stand out. **Pythia at/below parity, falling with scale** (−0.009 → −0.022, significant at 1.4B: 9 of 11 tasks lost) — CORE corroborates its BPB deficit, and more cleanly (monotone in scale). **SmolLM2 modestly ABOVE parity at every scale** (+0.011 to +0.019, ~1.7σ, winning 7 of 11 tasks at 135M and 360M) — which DISAGREES with its BPB result (parity at 135M, divergence-confounded deficit at 360M/1.7B). The most likely reading: SmolLM2's BPB penalty at larger sizes is OWT-overfitting (own-val improves while neutral BPB rises), which need not hurt downstream tasks; and even at 135M (no divergence) the Llama-family design (SwiGLU/GQA/RoPE) buys a small downstream edge that neutral BPB does not register. This is a directional, secondary signal — limit=500 CORE is noisy — not a compute-efficiency claim (on BPB/CEG neither lineage beats GPT-2).

![Exp B CORE gap vs matched GPT-2, by scale](core_expb_delta.png)

Per-task breakdown (each task is noisy at limit=500 — read the consistency across tasks, not any single panel). Exp B shows all 11 evaluated CORE-subset tasks (lambada is valid here — these are real-logits HF models — and at 135M–1.7B more tasks clear the validity gate than at the study's 124M, where the CORE figures gate down to 6). First as the gap vs matched GPT-2, then in absolute accuracy (same 3-line format as the Exp A / 2×2 CORE figures):

![Exp B CORE gap per task](core_expb_by_task.png)

![Exp B CORE absolute accuracy per task](core_expb_by_task_abs.png)

## CORE-subset (secondary, validity-gated)

DCLM CORE is a secondary check; BPB is primary. A task is used only where the A0D0 reference clears chance by >=2 sigma at its final checkpoint. `lambada_openai` is open-vocabulary (no fixed chance) so it never enters the quantitative gate at any scale; separately, the current-arch (modded) A1 loader has no logits path so its lambada is invalid by construction, whereas the ScaleUp-arch A1 arms (1.5B, ScaleUp-124M) use a plain-causal adapter with a real logits path and a valid lambada accuracy (reported as a diagnostic only). All A1 CORE is from the v2 sweep.

- **124M**: 5 usable tasks (arc_easy, copa, hellaswag, piqa, xwinograd_en).
- **355M**: 6 usable tasks (arc_easy, boolq, copa, hellaswag, piqa, xwinograd_en).
- **1.5B**: 6 usable tasks (arc_easy, boolq, copa, hellaswag, piqa, xwinograd_en).
- **ScaleUp-124M**: 5 usable tasks (arc_easy, copa, hellaswag, piqa, xwinograd_en).

![CORE task accuracy vs model size](core_vs_scale.png)

Note: `boolq` drops at 124M / ScaleUp-124M (A0D0 sits at chance, 0.504) but clears from 355M up — a scale effect, not an error. Arms cluster closely on the gated tasks (limit=500, near-noisy), with slightly more separation at 1.5B but still within the noise floor, so no quantitative CORE-based CEG is claimed at any scale; it is a sanity gate. The ScaleUp-arch A1 lambada diagnostic rises cleanly with scale (acc 0.32 at 124M -> 0.52/0.55 at 1.5B; perplexity 55 -> 8), confirming the plain-causal adapter's logits path is sound.

Across all four arms (a qualitative companion to BPB, not a second CEG claim), the only recurring hint is new-data (D1) arms edging out their old-data (D0) counterparts on arc_easy and piqa at every scale — sizable on arc_easy, within ~1–2 stderr on piqa; boolq shows no such order, so it is task-specific. Error bars (±1 stderr, limit=500) overlap widely, so this is directionally suggestive only.

![CORE accuracy across all four arms, per task](core_arms_by_task.png)

## Methodology notes & confounds

- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia slice (tokenizer-invariant; identical raw text across all runs). Per-dataset val BPB was logged as same-distribution diagnostics only.
- Decontamination of the eval corpus (13-gram overlap against the actual tokenized training samples) removed 562/2766 candidate docs (20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages quote Wikipedia heavily — and a further 153 (5.5%) matched only DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, sha256 cbdd72ac..., identical across every arm, size, and tier.
- Compute is timed GPU-hours on the runs' actual hardware, excluding kernel-warmup/compile and eval time; raw FLOPs are never compared across arms (precision differs).
- Hyperparameters are fixed per (algorithm, size) row; verified identical between data arms with scripts/verify_row_hparams.py.
- Token budget is part of the algorithm bundle: the A1 (modded) arms train at 2x the upstream-native budget for each size (small ~731M, medium ~4.35B tokens), NOT a fixed cross-arm budget. The original fixed-18B design (which forced A1D0 to repeat OWT for 2 epochs) was abandoned after the recipe collapsed under ~50x schedule stretch. At the 2x-native budgets both A1 arms are single-pass — well under one epoch of their ~9B+ corpora — so there is no epoch-repetition confound in any cell.
- Threshold crossings are interpolated in log-compute between checkpoints (log-spaced, denser early), never snapped.
- Reference-threshold definition: mean neutral BPB over all checkpoints in the final 10% of A0D0's training (per size). The single-final-checkpoint variant is reported alongside as robustness. Why a range exists at 124M: the threshold sits on the threshold arm's end-of-training plateau, where its curve is flattest; with the original purely log-spaced checkpoint schedule the plateau was sampled sparsely, so vertical noise of order the same-seed rerun floor (~0.01 BPB) translated into ~20% swings in that arm's compute-to-threshold and hence ~±8% (data) / ~±16% (total) in the multipliers. Fixed going forward by adding linearly-spaced checkpoints over the final 10% of every arm's schedule; the reported range brackets the definitional freedom on the pre-fix 124M data.
- DCLM CORE scores are secondary and validity-gated (tasks near chance at small scale are excluded).
