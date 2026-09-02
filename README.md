# Data vs. Algorithms: what drove language-model progress?

Language models improved enormously from GPT-2 (2019) to today. Two things
changed at once: the **data** got better, and the **algorithms and
architectures** got better. This project measures how much of the progress
each one caused.

We put both on one scale: compute. Instead of asking "how much better is
the loss," we ask "how much less compute do you need to reach the same
quality?" We fix a quality bar — a target score on a held-out, decontaminated
Wikipedia set. We train each configuration from scratch and record the
GPU-hours it takes to reach that bar. Better data or a better algorithm
means fewer GPU-hours. We call that ratio a **compute-equivalent gain
(CEG)**: a 2x CEG means the same quality in half the compute. We measure the
bar in bits-per-byte on the same raw text for every run, so the comparison
holds across different tokenizers, architectures, and model sizes.

Two rules for reading the numbers below. First, every comparison is against
a GPT-2 baseline with the same parameter count. When we say an architecture
is compared to "a matched GPT-2," we mean a GPT-2 of the same size, trained
through the same pipeline — so only the thing being tested differs. Second,
some runs never reach the quality bar at all. We call those **censored** and
report a bound ("≤1x, no measurable gain") instead of inventing a crossing
that didn't happen.

## The three experiments

### 1. Core 2x2 study — the foundation

We cross two axes directly: old vs. new **data**, old vs. new **algorithm**.
That gives four training runs ("arms"). Comparing their compute-to-quality
splits total progress into a data multiplier and an algorithm multiplier at
each model size — a Shapley decomposition, which averages two orderings:
add the better data first, or add the better algorithm first.

We ran this at several GPT-2 sizes, along two separate curves, because the
"new algorithm" isn't one fixed thing across scales: a current modded-nanoGPT
speedrun (small sizes) and the older 2024 "ScaleUp" recipe (which has a
documented large-scale config).

|  | Old data (OpenWebText) | New data (DCLM) | Sizes run |
|---|---|---|---|
| **Old algorithm** (GPT-2) | A0D0 | A0D1 | 124M, 355M, 1.5B |
| **New algorithm — current speedrun** | A1D0 | A1D1 | 124M, 355M |
| **New algorithm — 2024 ScaleUp** | A1D0 | A1D1 | 124M, 1.5B |

### 2. Exp A — does "newer data" mean "better data"?

We hold the algorithm fixed and walk the data corpus forward in time, all at
124M. "Worse data" has a concrete meaning here: a model of the same size,
trained on that corpus for the same compute, reaches the quality bar more
slowly — or not at all. The finding: data quality is not monotonic in
release year.

| Data corpus | Release year | Runs (each = one 124M model) |
|---|---|---|
| OpenWebText | 2019 | old-algo (the baseline) + current-arch |
| C4 | 2020 | old-algo + current-arch |
| RefinedWeb | 2023 | old-algo + current-arch |
| DCLM | 2024 | old-algo + current-arch |

### 3. Exp B — does the modern architecture advantage generalize?

The core study found a large algorithm advantage for the current speedrun
at small scale. Is that a real architectural improvement, or tricks tuned
for small-scale speedrun benchmarks? To find out, we took two published
open-model lineages — Pythia (2023) and SmolLM2 (2024) — and trained them
from scratch against a same-size GPT-2, on fixed data (OpenWebText), across
a range of sizes. The result: a clean no. Neither beats a well-tuned GPT-2
at any scale.

| Architecture | Sizes run | Baseline (same-size GPT-2) |
|---|---|---|
| GPT-2 (matched baseline) | 135M, 160M, 360M, 410M, 1.4B, 1.7B | — (it is the baseline) |
| Pythia (GPT-NeoX, 2023) | 160M, 410M, 1.4B | 160M / 410M / 1.4B GPT-2 |
| SmolLM2 (Llama, 2024) | 135M, 360M, 1.7B | 135M / 360M / 1.7B GPT-2 |

The table above holds data fixed at OpenWebText. We then trained the two
new architectures (at small size) on all four corpora too, so they get a
data multiplier like the original two architectures. That closes the data x
architecture grid; results are below.

Headline numbers for all three experiments are in "Results so far" below.
The full write-up, with figures and every caveat, is in `report.md`.

## Status: what's measured, and what's still open

The primary metric — compute-to-threshold (BPB/CEG) — is complete for all
three experiments. The CORE downstream-task metric is also complete for all
three.

**Done:**
- **Core 2x2 study** — current-arch at 124M and 355M, ScaleUp at 124M and
  1.5B, each with full BPB/CEG, the Shapley decomposition, and CORE scores.
  Two points are left out on purpose: current-arch at 1.5B (no reproducible
  recipe exists there — scaling it up would mean inventing an unvalidated
  architecture) and ScaleUp at 355M (no documented recipe for that size).
- **Exp A** (data-era ladder) — complete at 124M, including CORE downstream
  tasks across all four corpora. (C4 and RefinedWeb were retrained to
  recover lost checkpoints; the retrain reproduced the original BPB within
  ±0.008.)
- **Exp B** (architecture landscape) — complete. The architecture axis runs
  135M–1.7B: no new architecture beats a matched GPT-2 on OWT at any size.
  The data axis runs both new architectures across all four corpora at
  small size: better data flips them from losing to *beating* a matched
  GPT-2. CORE downstream tasks are scored for every B1 checkpoint. One
  wrinkle: SmolLM2 sits above its matched GPT-2 on the task suite at every
  size, even though its bits-per-byte score is only parity-or-worse — a
  small architectural edge the compute-efficiency metric misses. Pythia
  stays at or below GPT-2 on both metrics.

**Open:**
- **Next architecture tier (Mamba / Mamba-2).** Non-Transformer lineages,
  to extend Exp B past attention-based models. Needs GPUs.

Everything else — all three experiments' BPB/CEG results, plus CORE
downstream tasks for all three — is done.

## Results so far (headline numbers)

Every multiplier is a compute-reduction factor: GPU-hours-to-threshold,
baseline over candidate. "Censored" means the arm never reached the
threshold (≤1x, no measurable gain).

**Core 2x2 study — two cross-scale curves, kept separate (different A1
generations):**

| Curve | Size | Data x | Algorithm x | Total x | Threshold (BPB) |
|---|---|---|---|---|---|
| current-arch | 124M | 2.23 | 13.7 | 30.5 | 1.2744 |
| current-arch | 355M | 3.74 | 4.06 | 15.2 | 1.2287 |
| ScaleUp | 124M | 3.25 | 2.9 (DCLM); OWT censored | — | 1.2800 |
| ScaleUp | 1.5B | 3.43 | 2.34 (DCLM); OWT censored | — | 1.1879 |

Algorithm advantage falls as scale grows — sharply for current-arch, gently
for ScaleUp. The data multiplier stays roughly flat (~3x). Disclosed gaps:
current-arch at 1.5B, ScaleUp at 355M.

Note on scale labels: "124M / 355M / 1.5B" are the **GPT-2 baseline scales**
(baseline dimensions), not the candidate parameter counts. The current-arch
model has ~498.8M parameters at the 124M-baseline dimensions (its value-embed /
U-net additions are the algorithm being measured; old-algo GPT-2 = 123.7M). CEG
compares each candidate to the GPT-2 baseline of matching dimensions.

**Exp A — data-era ladder @124M** (CEG vs. the OWT GPT-2 baseline;
threshold 1.2760 — this corrected value comes from a stricter, unified
eval set built after Exp A added new corpora; see the methodology note
below and `report.md` for the full correction):

| Dataset (year) | Algorithm CEG at that corpus | Data CEG (current-arch) |
|---|---|---|
| OWT (2019) | 23.8x | 1.0x (baseline) |
| C4 (2020) | censored | censored |
| RefinedWeb (2023) | 11.6x | 1.59x |
| DCLM (2024) | 10.9x | 1.62x |

Data quality is not monotonic in release year: C4 (2020) is worse than 2019
OWT, censored under both algorithms. Corrected OWT x DCLM 2x2: data 2.39x,
algorithm 16.1x, total 38.5x.

CORE downstream tasks for Exp A (`results/core_era_ladder.png`) are a
secondary check only: at 124M and limit-500, the old-vs-new-algorithm gap
sits within stderr (±0.02) at every corpus. Treat CORE here as a sanity
check, not a quantitative claim — the data-quality signal lives in the
BPB/CEG numbers above.

**Exp B — architecture landscape.** Every architecture is compared to a
same-size GPT-2. None reach the GPT-2 bar faster — all are censored,
algorithm CEG ≤1x — so the informative number is how far short of the bar
each one lands. The table shows the bits-per-byte gap at convergence: 0
means a tie, positive means the architecture is worse.

| Architecture | ~135–160M | ~360–410M | ~1.4–1.7B |
|---|---|---|---|
| Pythia (GPT-NeoX, 2023) | +0.082 (worse) | +0.020 (2-seed) | +0.030 (worse) |
| SmolLM2 (Llama, 2024) | +0.009 (tie, within noise) | +0.054 † | +0.120 † |

No published Transformer lineage beats a matched, well-trained GPT-2
anywhere from 135M to 1.7B. The best any of them manages is a tie
(SmolLM2 at 135M). † SmolLM2's larger models overfit OpenWebText even at
their documented learning rate, which inflates these two gaps — the
deficit is real, but its exact size is confounded. This is the central Exp
B result: the current speedrun's small-scale advantage does not come from
a generally-better architecture.

CORE downstream tasks (`results/core_bpb_vs_downstream.png`,
`core_expb_by_task.png`) add a wrinkle: Pythia stays at or below its
matched GPT-2, consistent with the gap above, but SmolLM2 lands slightly
above its matched GPT-2 at every size, despite tying or losing on
bits-per-byte — a small architectural edge on downstream accuracy that the
compute-efficiency metric misses. This is a modest, noisy signal
(limit=500), not a compute-efficiency claim.

**Exp B data axis** (`results/data_replication.png`) — Pythia-160M and
SmolLM2-135M trained on all four corpora, each measured against the same
external bar (its size-matched GPT-2 trained on OWT):

| Architecture | OWT (2019) | C4 (2020) | RefinedWeb (2023) | DCLM (2024) |
|---|---|---|---|---|
| Pythia-160M | censored | censored | crosses, 4.6x | crosses, 5.4x |
| SmolLM2-135M | censored | censored | crosses, 5.4x | crosses, 6.4x |

The fixed external bar makes the headline visible: better data flips both
architectures from losing to a matched GPT-2 (OWT/C4) to beating it
(RefinedWeb/DCLM), at 4.6–6.4x less compute — a far bigger lever than any
architecture change we found (in B1, no new architecture beat GPT-2 on OWT
at all). Corpus progress is not monotonic with release date: on OWT (2019)
and C4 (2020) both architectures stay censored — neither reaches the bar —
with C4's terminal BPB marginally lower than OWT's, while RefinedWeb and
DCLM cross. This reproduces Exp A's non-monotonic-in-release-date finding
across two further, independent architectures (GPT-NeoX and Llama lineages),
rather than it being tied to any one algorithm.

## Where the artifacts live (checkpoints & data)

**Durable homes:** git holds all code, metrics, result JSONs, figures, and
`report.md`. Hugging Face holds model checkpoints. A local backup mirror
sits at `~/Desktop/era_ladder_backup/`. Pod `/workspace` volumes are
ephemeral working space only.

| Asset | Primary location | Backup / notes |
|---|---|---|
| **Core-study finals** (16 checkpoints: current-arch 124M/355M + ScaleUp 124M/1.5B) | private HF `MIRIBerkeley/data-vs-algorithms-ceg` (73.5 GB) | current-arch mirrored locally (`hf_current_arch_finals/`, 10 GB). **ScaleUp exists only on HF, and that private repo is over its storage quota — reads are currently blocked.** Fix by upgrading the HF plan; this has no other backup right now. |
| **Exp B checkpoints** (14: 6 GPT-2 denominators + 6 candidates + 2 replicate seeds) | public HF `MIRIBerkeley/data-vs-algorithms-ceg-expB` (35.7 GB) | local (`b1_cand/checkpoints/`, 36 GB); metrics.csv in git (`results/b1_metrics/`) |
| **Exp B matched GPT-2 baselines** (6, train_old) | local (`b1_checkpoints/`, 16 GB) | thresholds in git (`results/b1_baseline_thresholds.json`) |
| **Exp A era-ladder arms** | OWT/DCLM reuse the 2x2 study's 124M finals (HF + local); C4/RefinedWeb finals live on public HF `…-expB` under `exp-a-core-retrain/`, plus local | (the original C4/RW checkpoints were pod-volume-only and got lost; they were retrained. Every era-ladder final now has a durable copy.) |
| **Tokenized datasets** | pod `/workspace/datasets/` (all 4 corpora x gpt2/nanogpt tokenizers; OWT x neox/smollm2) | `owt_neox`/`owt_smollm2` mirrored locally (`b1_datasets/`, 34 GB); `wiki_eval_union` in `eval_sets.tgz`. C4/RefinedWeb/DCLM in the neox/smollm2 tokenizers (needed for the data-ladder work) are being built now. |
| **Metrics, results, figures, report, configs** | git (this repo) | — |

**The ScaleUp checkpoint issue above is worth fixing first**, not just
noting — it's the one asset in this table with no backup at all.

## Layout

- `configs/model_sizes.json` — GPT-2 size configs: the 4 study sizes
  (124M/355M/770M/1.5B) plus 6 Exp B matched-baseline sizes (b135–b1700)
- `common/` — GPT-2 model, BPB evaluator, log-spaced checkpoint schedule,
  epoch-aware data loader (shared by training and eval)
- `data/prepare.py` — downloads and tokenizes, parameterized by dataset and
  tokenizer
- `train_old/train.py` — GPT-2 reproduction trainer (A0 arms)
- `train_new/` — modded-nanoGPT clone and adaptation notes (A1 arms)
- `train_new/train_gpt_xl_ceg.py` — the 2024 ScaleUp1B plain-transformer
  trainer used for the 1.5B and ScaleUp-curve A1 arms (an older modded
  generation than the current speedrun tracks — see the report caveat)
- `train_hf/train_hf_ceg.py` — shared from-scratch trainer for HF-class
  architectures (Exp B): GPT-2, Pythia/GPT-NeoX, SmolLM2/Llama,
  Mamba/Mamba-2, with BPB over an arbitrary tokenizer (`eval/bpb_hf.py`)
- `eval/` — neutral-corpus BPB (`neutral_bpb.py`), decontamination
  (`decontam.py`), and CORE loglikelihood adapters, each gated by the
  requirement-#3 validity check: `lm_eval_adapter.py` (A0 / GPT-2),
  `lm_eval_adapter_modded.py` (current-arch A1, with yarn/split_embed reload
  fidelity), `lm_eval_adapter_scaleup.py` (plain-causal 2024-ScaleUp A1)
- `analysis/` — `threshold.py` (the one canonical neutral-BPB threshold
  function), `ceg_shapley.py` + `matched_compute.py` (compute-to-threshold
  and the log-space Shapley split), `core_gate.py` (CORE validity gate),
  `threshold_sensitivity.py`, `plots.py` (the hero `multipliers_vs_scale.png`
  — current-arch and ScaleUp only — plus the per-size and CORE figures),
  the publication-figure scripts `plot_corpus_intervention.py` (Exp A),
  `plot_training_curves_ab.py` (Exp B architecture + data replication),
  `plot_core_disagreement.py` (Exp B CORE), `plot_method_factorial.py`
  (method schematic) — each writing PNG + vector PDF; see
  `results/FIGURE_NOTES.md` for per-figure captions and provenance.
  `make_report.py` (regenerates `report.md`)
- `scripts/check_sizes.py` — param-count check for every size
- `report.md` — the full write-up: two cross-scale curves (current-arch and
  ScaleUp-arch) kept separate, plus the Exp A and Exp B sections, with every
  disclosed gap

## How we measure (methodology)

A few fixed choices keep the compute comparisons fair, so a difference in
the result reflects a difference in data or architecture — not a
difference in how we measured it.

- **Quality is bits-per-byte, not per-token loss.** Different tokenizers
  split text differently, so per-token cross-entropy isn't comparable
  across them; bits-per-byte is. We never compare per-token loss across
  architectures.
- **One fixed quality bar for everything.** The bar is bits-per-byte on a
  fixed, decontaminated slice of Wikipedia — the same raw text for every
  run, at every size. The exact bar at each size is the fully-trained
  GPT-2 baseline's score, averaged over the final 10% of its training
  (`analysis/threshold.py`).
- **Compute is timed GPU-hours**, excluding warmup, compile time, and eval
  time. We never compare raw FLOPs across architectures, since their
  numerical precision differs.
- **Hyperparameters stay fixed per (architecture, size).** We never re-tune
  them between the two data conditions, so the data comparison can't be
  contaminated by extra tuning.
- **The tokenizer counts as part of the "algorithm."** We tokenize each
  corpus once per tokenizer; changing tokenizer is an algorithm change, not
  a data change.
- **Downstream tasks (CORE) are a secondary check, not the headline.**
  They're noisy at small scale, so we gate each task on clearing chance by
  a real margin before using it, and we never make a compute-efficiency
  claim from them alone.
- **Crossings are interpolated in log-compute** between checkpoints, never
  snapped to the nearest one. The data/algorithm split is a Shapley
  decomposition done in log-compute space, reported back out as
  multipliers.

One design choice changed mid-project, worth stating plainly. The original
plan fixed the token budget across arms — 9B for the old algorithm, 18B for
the new one — which forced the new-algorithm/old-data cell to train two
epochs on OpenWebText. We abandoned that after the modded recipe broke down
under the very long schedule an 18B budget implied at small sizes. In the
final design, each modded arm trains at twice its size's native budget, in
a single pass — well under one epoch — so no cell repeats data. (Recorded
here as an explicit correction, not a silent edit; see `report.md`.)

Session procedures, the pipeline smoke test, and the cost ledger live in
`RUNBOOK.md`. The full write-up is `report.md`.
