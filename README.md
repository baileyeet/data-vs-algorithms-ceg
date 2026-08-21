# Data vs. Algorithms: Compute-Equivalent Gain at Multiple Scales

Measures how much of GPT-2-era → 2024-era language-model progress comes from
**data quality** vs. **algorithms/architecture**, by training models from scratch
and comparing *compute-to-a-fixed-quality-bar* (timed GPU-hours to a neutral-corpus
BPB threshold), with the savings Shapley-decomposed in log-compute space. The
quality bar is BPB on a fixed, decontaminated Wikipedia slice — identical raw text
across every run, so numbers are comparable across tokenizers, architectures, and scales.

The project now spans **three experiments** (full write-up in `report.md`; headline
numbers and the "what's done / what's left" status are below):

- **Core 2×2 study — the foundation.** A grid of old/new **data** × old/new
  **algorithm**, trained at GPT-2 sizes, decomposing progress into a data multiplier
  and an algorithm multiplier at each scale. Two cross-scale curves (a current
  modded-nanoGPT speedrun at 124M/355M; the 2024 "ScaleUp" lineage at 124M/1.5B),
  kept separate because the "new algorithm" isn't one fixed thing across scales.

  | | Old data (OpenWebText) | New data (DCLM/Nemotron-CC) |
  |---|---|---|
  | **Old algorithm** (GPT-2 repro) | A0D0 | A0D1 |
  | **New algorithm** (modded-nanoGPT) | A1D0 | A1D1 |

- **Exp A — data-era ladder (@124M).** Holds the algorithm axis and sweeps the data
  corpus by release-year (OWT 2019, C4 2020, RefinedWeb 2023, DCLM 2024). Finds
  data-quality is NON-monotonic in vintage (C4 is *worse* than 2019 OWT — censored).

- **Exp B — architecture landscape.** Trains published Transformer lineages
  (Pythia 2023, SmolLM2 2024) from scratch vs a size-matched GPT-2 at 135M–1.7B, on
  fixed data (OWT). **No lineage beats a matched, well-trained GPT-2 at any scale**
  (all algorithm-CEG ≤1×) — a direct "no" to whether the current-arch speedrun's
  small-scale advantage generalizes beyond small-scale-optimized tricks. Currently
  architecture-only (OWT); the **data multiplier for these archs is the planned next
  step** (see "Planned next" below).

**Note on token budgets (CORRECTED — dated, see invariant #9):** the design ORIGINALLY
fixed budgets across arms (old-algo 9B, new-algo 18B) with A1D0 repeating OWT for 2
epochs. That was **abandoned**; each A1 (modded) arm now trains at 2× its size's
upstream-native budget, **single-pass** (well under one epoch) — no epoch-repetition
confound in any cell.

## Status & completeness (what is / isn't measured)

- **Core 2×2 study:** complete at 124M + 355M (current-arch) and 124M + 1.5B
  (ScaleUp), with **BPB/CEG + Shapley + CORE**. Two disclosed gaps: current-arch
  has **no 1.5B** (no reproducible recipe; scaling needs an unvalidated invented
  arch) and ScaleUp has **no 355M** (no documented era-appropriate recipe).
- **Exp A (era ladder):** complete on **BPB/CEG** (@124M). CORE **not run** (deferred).
- **Exp B (architecture landscape):** complete on **BPB/CEG** at 135M–1.7B (all
  censored ≤1×). **No data multiplier** — Exp B holds data fixed (OWT) and varies
  only architecture, so there is no data axis to decompose (by design, not missing).
  CORE **not run** (deferred).
- **Planned next (P1, needs a GPU pod):** extend Exp B to the **full data ladder** —
  run the new architectures (Pythia, SmolLM2; + Mamba/Mamba-2) across all 4 corpora
  (OWT done, C4, RefinedWeb, DCLM) so each gets a **data multiplier**, matching the
  old two architectures' treatment. Locked to the **small size (~135–160M) only**
  (the size that overlays Exp A's 124M ladder; the data multiplier is already
  scale-stable; smallest avoids SmolLM2's size-worsening divergence). Reuses the
  existing OWT GPT-2 thresholds (b135/b160) as the fixed denominator — no new
  baselines per dataset, just tokenization + candidate training.
- **Deferred, needs a GPU pod (tracked in `report.md`):** CORE-by-task for Exp A
  **and** Exp B (downstream-task corroboration of the BPB/CEG findings); and the
  next architecture tier (Mamba / Mamba-2, "B2"). No pod is currently running.

## Results so far (headline numbers)

All multipliers are compute-reduction factors (GPU-hours-to-neutral-BPB-threshold
ratios); "censored" = the arm never reaches the threshold (≤1×, no measurable gain).

**Core 2×2 study — two cross-scale curves (kept separate; different A1 generations):**

| Curve | Size | Data × | Algorithm × | Total × | Threshold (BPB) |
|---|---|---|---|---|---|
| current-arch | 124M | 2.23 | 13.7 | 30.5 | 1.2744 |
| current-arch | 355M | 3.74 | 4.06 | 15.2 | 1.2287 |
| ScaleUp | 124M | 3.25 | 2.9 (DCLM); OWT censored | — | 1.2800 |
| ScaleUp | 1.5B | 3.43 | 2.34 (DCLM); OWT censored | — | 1.1879 |

Algorithm advantage decays with scale (steeply for current-arch, gently for ScaleUp);
data multiplier ~stable (~3×). Disclosed gaps: current-arch 1.5B, ScaleUp 355M.

**Exp A — data-era ladder @124M** (CEG vs the OWT GPT-2 baseline; threshold 1.2760):

| Dataset (year) | Algorithm CEG at that corpus | Data CEG (current-arch) |
|---|---|---|
| OWT (2019) | 23.8× | 1.0× (baseline) |
| C4 (2020) | censored | censored |
| RefinedWeb (2023) | 11.6× | 1.59× |
| DCLM (2024) | 10.9× | 1.62× |

Data-quality is NON-monotonic in release year — C4 (2020) is *worse* than 2019 OWT
(censored under both algorithms). Corrected OWT×DCLM 2×2: data 2.39×, algo 16.1×, total 38.5×.

**Exp B — architecture landscape** (algorithm-CEG vs a size-matched GPT-2, data fixed = OWT;
Δ = arch neutral BPB − matched GPT-2, >0 = worse; all arms censored ≤1×):

| Lineage | 135–160M | 360–410M | 1.4–1.7B |
|---|---|---|---|
| Pythia (GPT-NeoX, 2023) | +0.082 | +0.020 (2-seed) | +0.030 |
| SmolLM2 (Llama, 2024) | +0.009 (parity within noise) | +0.054 *(div)* | +0.120 *(div)* |

**No published Transformer lineage beats a matched, well-trained GPT-2 anywhere 135M–1.7B.**
Best case = SmolLM2-135M parity. *(div)* = divergence-confounded (SmolLM2 overfits OWT at
larger sizes, even at its documented LR). No data multiplier yet — see "Planned next".

## Where the artifacts live (checkpoints & data)

**Durable homes:** git (all code, metrics, results JSONs, figures, `report.md`);
Hugging Face (model checkpoints); a local backup mirror at
`~/Desktop/era_ladder_backup/`. Pod `/workspace` volumes are ephemeral working space.

| Asset | Primary location | Backup / notes |
|---|---|---|
| **Core-study finals** (16 ckpts: current-arch 124M/355M + scaleup 124M/1.5B) | private HF `MIRIBerkeley/data-vs-algorithms-ceg` (73.5 GB) | current-arch mirrored local (`hf_current_arch_finals/`, 10 GB). **scaleup is HF-only** and the private repo is over quota → currently read-blocked (resolve by upgrading the HF plan). |
| **Exp B checkpoints** (14: 6 GPT-2 denominators + 6 candidates + 2 replicate seeds) | public HF `MIRIBerkeley/data-vs-algorithms-ceg-expB` (35.7 GB) | local (`b1_cand/checkpoints/`, 36 GB); metrics.csv in git (`results/b1_metrics/`) |
| **Exp B matched GPT-2 baselines** (6, train_old) | local (`b1_checkpoints/`, 16 GB) | thresholds in git (`results/b1_baseline_thresholds.json`) |
| **Exp A era-ladder arms** | run metrics local (`prov/`, `run_metrics/`) + git results JSONs | final checkpoints on the pod volume only (weights not needed for the CEG result) |
| **Tokenized datasets** | pod `/workspace/datasets/` (all 4 corpora × gpt2/nanogpt; OWT × neox/smollm2) | `owt_neox`/`owt_smollm2` mirrored local (`b1_datasets/`, 34 GB); `wiki_eval_union` in `eval_sets.tgz`. **C4/RW/DCLM in neox/smollm2 tokenizers NOT yet built** (the data-ladder prereq). |
| **Metrics, results, figures, report, configs** | git (this repo) | — |

## Layout

- `configs/model_sizes.json` — the GPT-2 size configs: the 4 study sizes
  (124M/355M/770M/1.5B) plus 6 Exp B matched-baseline sizes (b135–b1700)
- `common/` — GPT-2 model, BPB evaluator, log-spaced checkpoint schedule,
  epoch-aware data loader (shared by training + eval)
- `data/prepare.py` — download/tokenize, parameterized by (dataset, tokenizer)
- `train_old/train.py` — GPT-2 reproduction trainer (A0 arms)
- `train_new/` — modded-nanoGPT clone + adaptation notes (A1 arms)
- `train_new/train_gpt_xl_ceg.py` — the 2024 ScaleUp1B plain-transformer
  trainer used for the 1.5B and ScaleUp-curve A1 arms (an OLDER modded
  generation than the current speedrun tracks — see the report caveat)
- `train_hf/train_hf_ceg.py` — shared from-scratch trainer for HF-class
  architectures (Exp B): GPT-2, Pythia/GPT-NeoX, SmolLM2/Llama, Mamba/Mamba-2,
  with BPB-over-arbitrary-tokenizer (`eval/bpb_hf.py`)
- `eval/` — neutral-corpus BPB (`neutral_bpb.py`), decontamination
  (`decontam.py`), and CORE loglikelihood adapters, all gated by the
  requirement-#3 validity check: `lm_eval_adapter.py` (A0 / GPT-2),
  `lm_eval_adapter_modded.py` (current-arch A1, with yarn/split_embed reload
  fidelity), `lm_eval_adapter_scaleup.py` (plain-causal 2024-ScaleUp A1)
- `analysis/` — `threshold.py` (canonical neutral-BPB threshold fn),
  `ceg_shapley.py` + `matched_compute.py` (compute-to-threshold + log-space
  Shapley), `core_gate.py` (CORE validity gate), `threshold_sensitivity.py`,
  `plots.py` (all figures incl. the unified `multipliers_vs_scale.png` — now
  carrying Exp B's censored lineages), `era_ladder.py`/`plot_era_ladder.py`
  (Exp A), `make_report.py` (regenerates `report.md`)
- `scripts/check_sizes.py` — param-count verification for all sizes
- `report.md` — the final write-up: two cross-scale curves (current-arch and
  ScaleUp-arch) kept separate, plus the Exp A (era-ladder) and Exp B
  (architecture landscape) sections, with all disclosed gaps

## Methodology invariants (do not "improve")

1. **BPB everywhere**, never cross-arm per-token CE (tokenizers differ in kind).
2. Reference threshold = BPB on a **fixed decontaminated Wikipedia slice**,
   identical raw text across all runs; per-dataset val BPB is logged but
   labeled same-distribution-only. (Canonical definition = mean neutral BPB
   over A0D0's final-10%-by-tokens checkpoints, `analysis/threshold.py::final_tail_threshold`.)
3. CORE is secondary, gated by an above-chance validity check at small sizes.
4. Compute = **timed GPU-hours** excluding warmup/compile and eval time; no
   cross-arm raw-FLOP comparisons.
5. Hyperparameters are fixed per (algorithm, size) row — never re-tuned per
   data arm.
6. Log-spaced checkpoints + log-compute interpolation for threshold crossing.
7. Shapley decomposition in log-compute space, reported as multipliers.
8. Tokenizer belongs to "algorithm": 4 tokenized corpus variants total.
9. ~~A1D0 = exactly 2 epochs of OpenWebText, reshuffled between epochs, called
   out in the writeup as that cell's extra confound.~~
   **CORRECTED (dated note, 2026-07):** the original design fixed token budgets
   across arms (old-algo 9B, new-algo 18B), which forced A1D0 to repeat OWT for
   exactly 2 epochs. This was **abandoned** after the modded recipe collapsed
   under the ~50× schedule stretch a fixed 18B budget implied at small sizes.
   Final design: each A1 (modded) arm trains at **2× its size's upstream-native
   budget, single-pass** (well under one epoch of its ~9B+ corpus) — so there is
   **no epoch-repetition confound in any cell**. Recorded here as an explicit
   correction (not a silent rewrite), same as every other correction in this
   project; see `report.md` methodology notes.

Session procedures, pipeline smoke test, and the cost ledger live in
`RUNBOOK.md`; the final write-up is `report.md`.
