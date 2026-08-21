# Data vs. Algorithms: Compute-Equivalent Gain at Multiple Scales

Isolates how much of GPT-2-era → 2024-era capability progress comes from
**data quality** vs. **algorithms/architecture**, via a 2x2 grid
(old/new data x old/new algorithm) trained at three core GPT-2 sizes
(124M, 355M, 1.5B), measuring compute-to-reference-BPB and
Shapley-decomposing the savings in log-compute space.

| | Old data (OpenWebText) | New data (DCLM/Nemotron-CC) |
|---|---|---|
| **Old algorithm** (GPT-2 repro) | A0D0 | A0D1 |
| **New algorithm** (modded-nanoGPT) | A1D0 | A1D1 |

**Two follow-on experiments extend the core 2x2 (both in `report.md`):**
- **Exp A — data-era ladder (@124M):** holds the algorithm axis and sweeps the
  data corpus by release-year (OWT 2019, C4 2020, RefinedWeb 2023, DCLM 2024);
  finds data-quality is NON-monotonic in vintage (C4 is censored — worse than OWT).
- **Exp B — architecture landscape:** trains published Transformer lineages
  (Pythia 2023, SmolLM2 2024) from scratch on fixed data (OWT) vs a size-matched
  GPT-2 at 135M–1.7B; **no lineage beats a matched GPT-2 at any scale** (all
  algorithm-CEG ≤1×) — a direct "no" to whether current-arch's small-scale
  advantage generalizes. Checkpoints: public HF `MIRIBerkeley/data-vs-algorithms-ceg-expB`.

**Token budgets — CORRECTED (dated note, see invariant #9):** the design ORIGINALLY
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
- **Deferred, needs a GPU pod (tracked in `report.md`):** CORE-by-task for Exp A
  **and** Exp B (downstream-task corroboration of the BPB/CEG findings); and the
  next architecture tier (Mamba / Mamba-2, "B2"). No pod is currently running.

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
