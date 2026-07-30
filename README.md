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
- `train_new/train_gpt_xl_ceg.py` — the 2024 ScaleUp1B plain-transformer
  trainer used for the 1.5B and ScaleUp-curve A1 arms (an OLDER modded
  generation than the current speedrun tracks — see the report caveat)
- `eval/` — neutral-corpus BPB (`neutral_bpb.py`), decontamination
  (`decontam.py`), and CORE loglikelihood adapters, all gated by the
  requirement-#3 validity check: `lm_eval_adapter.py` (A0 / GPT-2),
  `lm_eval_adapter_modded.py` (current-arch A1, with yarn/split_embed reload
  fidelity), `lm_eval_adapter_scaleup.py` (plain-causal 2024-ScaleUp A1)
- `analysis/` — `threshold.py` (canonical neutral-BPB threshold fn),
  `ceg_shapley.py` + `matched_compute.py` (compute-to-threshold + log-space
  Shapley), `core_gate.py` (CORE validity gate), `threshold_sensitivity.py`,
  `plots.py`, `make_report.py` (regenerates `report.md`)
- `scripts/check_sizes.py` — param-count verification for all sizes
- `report.md` — the final write-up: two cross-scale curves (current-arch and
  ScaleUp-arch), kept separate, with all disclosed gaps

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

Session procedures, pipeline smoke test, and the cost ledger live in
`RUNBOOK.md`; the final write-up is `report.md`.
