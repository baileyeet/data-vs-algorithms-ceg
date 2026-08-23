# Data vs. Algorithms: what drove language-model progress?

Language models improved enormously from GPT-2 (2019) to today. Two things changed
at once: the **data** got better (cleaner, better-filtered web corpora) and the
**algorithms/architectures** got better (new optimizers, attention variants, training
recipes). This project asks how much of the progress each one is responsible for.

The trick is to put both on a common currency: **compute**. Instead of asking "how
much better is the loss," we ask "how much *less compute* do you need to reach the same
quality?" Concretely, we fix a quality bar — a target score on a held-out,
decontaminated Wikipedia set — train each configuration from scratch, and record the
GPU-hours it takes to reach that bar. Better data or a better algorithm shows up as
needing fewer GPU-hours. We call that ratio a **compute-equivalent gain (CEG)**: a 2×
CEG means you reach the same quality in half the compute. The quality bar is measured
in bits-per-byte on the same raw text for every run, so the comparison is fair across
different tokenizers, architectures, and model sizes.

Two things to know about how to read the numbers. First, every comparison is against a
**GPT-2 baseline of the same parameter count** — when we say an architecture is
compared to "a matched GPT-2," we mean a GPT-2 with the same dimensions/parameters,
trained through the same pipeline, so the only thing that differs is the thing being
tested. Second, some configurations **never reach the quality bar** within the training
budget; we call those *censored* and report them as a bound (e.g. "≤1×, no measurable
gain") rather than inventing a crossing that didn't happen.

## The three experiments

### 1. Core 2×2 study — the foundation

Cross the two axes directly: old vs. new **data**, old vs. new **algorithm**, giving
four training runs ("arms"). Comparing their compute-to-quality lets us split the total
progress into a *data multiplier* and an *algorithm multiplier* at each model size (a
Shapley decomposition — it averages the two orderings of "add the better data first"
vs. "add the better algorithm first"). We run this at several GPT-2 sizes, along two
separate curves, because the "new algorithm" isn't one fixed thing across scales: a
current modded-nanoGPT speedrun (small sizes) and the older 2024 "ScaleUp" recipe
(which has a documented large-scale configuration).

The four arms, and the sizes each curve was run at:

|  | Old data (OpenWebText) | New data (DCLM) | Sizes run |
|---|---|---|---|
| **Old algorithm** (GPT-2) | A0D0 | A0D1 | 124M, 355M, 1.5B |
| **New algorithm — current speedrun** | A1D0 | A1D1 | 124M, 355M |
| **New algorithm — 2024 ScaleUp** | A1D0 | A1D1 | 124M, 1.5B |

### 2. Exp A — does "newer data" mean "better data"?

Hold the algorithm fixed and walk the data corpus forward in time, all at 124M. Here
"worse data" has a concrete meaning: a model of the same size, trained on that corpus
for the same compute, reaches the quality bar more slowly — or not at all. The finding
is that data quality is *not* monotonic in release year.

| Data corpus | Release year | Runs (each = one 124M model) |
|---|---|---|
| OpenWebText | 2019 | old-algo (the baseline) + current-arch |
| C4 | 2020 | old-algo + current-arch |
| RefinedWeb | 2023 | old-algo + current-arch |
| DCLM | 2024 | old-algo + current-arch |

### 3. Exp B — does the modern architecture advantage generalize?

The core study found a large algorithm advantage for the current speedrun at small
scale. Is that a genuine architectural improvement, or tricks tuned for small-scale
speedrun benchmarks? To find out, we take two *published* open-model lineages — Pythia
(2023) and SmolLM2 (2024) — and train them from scratch against a same-size GPT-2, on
fixed data (OpenWebText), across a range of sizes. The result is a clean "no": neither
beats a well-tuned GPT-2 at any scale.

| Architecture | Sizes run | Baseline (same-size GPT-2) |
|---|---|---|
| GPT-2 (matched baseline) | 135M, 160M, 360M, 410M, 1.4B, 1.7B | — (it is the baseline) |
| Pythia (GPT-NeoX, 2023) | 160M, 410M, 1.4B | 160M / 410M / 1.4B GPT-2 |
| SmolLM2 (Llama, 2024) | 135M, 360M, 1.7B | 135M / 360M / 1.7B GPT-2 |

The architecture comparison above holds data fixed at OpenWebText. We then extended Exp B
along the *data* axis too — training the two new architectures (at small size) on all four
corpora — so they get a data multiplier like the original two architectures. That closes
the data × architecture grid; results are below.

Headline numbers for all three are in "Results so far" below; the full write-up, with
figures and every caveat, is in `report.md`.

## Status: what's measured, and what's still open

The primary metric (compute-to-threshold, i.e. BPB/CEG) is complete for all three
experiments. The secondary metric (CORE, a suite of downstream tasks — see the
methodology notes) and the data axis of Exp B are the open items.

Done:
- **Core 2×2 study** — current-arch at 124M and 355M, ScaleUp at 124M and 1.5B, all
  with the full BPB/CEG + Shapley decomposition and CORE scores. Two points are
  honestly left out: current-arch at 1.5B (no reproducible recipe exists — scaling it
  up would require inventing an unvalidated architecture) and ScaleUp at 355M (no
  documented recipe for that size).
- **Exp A** (data-era ladder) — complete at 124M, including CORE downstream tasks
  across all four corpora. (C4 and RefinedWeb were retrained to recover checkpoints
  that had been lost; the retrain reproduced the original BPB to within ±0.008.)
- **Exp B** (architecture landscape) — complete. The architecture axis runs 135M–1.7B
  (no new architecture beats a matched GPT-2 on OWT at any size); the data axis runs both
  new architectures across all four corpora at small size (better data flips them from
  losing to *beating* a matched GPT-2); and CORE downstream tasks are scored for all B1
  checkpoints. CORE adds a wrinkle: SmolLM2 sits modestly *above* its matched GPT-2 on the
  task suite at every size even though its bits-per-byte is only parity-or-worse — a small
  architectural edge the compute-efficiency metric doesn't capture (Pythia stays at/below
  GPT-2 on both).

Open:
- **Next architecture tier (Mamba / Mamba-2).** Non-Transformer lineages, to extend
  Exp B beyond attention-based models. Needs GPUs.

Everything else — all three experiments' BPB/CEG results, plus CORE downstream tasks
for all three — is complete.

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

CORE downstream tasks for Exp A (`results/core_era_ladder.png`) are a secondary check:
at 124M / limit-500 the old-vs-new-algorithm gap is within ±0.02 (≈ stderr) at every
corpus, so CORE is a sanity check here, not a quantitative claim — the data-quality signal
lives in the BPB/CEG numbers above.

**Exp B — architecture landscape.** Every architecture is compared to a same-size
GPT-2. None of them reach the GPT-2 quality bar faster (all are censored, i.e. algorithm
CEG ≤ 1×), so the informative number is how far *short* of the bar each one lands. The
table shows the gap in bits-per-byte between the architecture and its matched GPT-2 at
convergence — 0 means a tie, positive means the architecture is worse:

| Architecture | ~135–160M | ~360–410M | ~1.4–1.7B |
|---|---|---|---|
| Pythia (GPT-NeoX, 2023) | +0.082 (worse) | +0.020 (2-seed) | +0.030 (worse) |
| SmolLM2 (Llama, 2024) | +0.009 (tie, within noise) | +0.054 † | +0.120 † |

**No published Transformer lineage beats a matched, well-trained GPT-2 anywhere from
135M to 1.7B** — the best any of them manages is a tie (SmolLM2 at 135M). † SmolLM2's
larger models overfit OpenWebText even at their documented learning rate, which inflates
those two gaps (the deficit is real, but the exact size is confounded). This is the
central Exp B result and the reason we conclude the current speedrun's small-scale
advantage does *not* come from a generally-better architecture.

Downstream tasks (CORE — secondary, `results/core_expb_delta.png` and
`core_expb_by_task.png`) add a wrinkle worth noting: on the task suite, Pythia stays
at-or-below its matched GPT-2 (consistent with the gap above), but **SmolLM2 lands
slightly *above* its matched GPT-2 at every size** despite the bits-per-byte tie/deficit
— a small architectural edge on downstream accuracy that the compute-efficiency metric
doesn't capture. It's a modest, noisy signal (limit=500), not a compute-efficiency claim.

**Exp B data axis** (`results/data_ladder_expB.png`) — Pythia-160M and SmolLM2-135M
trained on all four corpora, each measured against the same external bar (its size-matched
GPT-2 trained on OWT):

| Architecture | OWT (2019) | C4 (2020) | RefinedWeb (2023) | DCLM (2024) |
|---|---|---|---|---|
| Pythia-160M | censored | censored | crosses, 4.6× | crosses, 5.4× |
| SmolLM2-135M | censored | censored | crosses, 5.4× | crosses, 6.4× |

The headline the fixed external bar makes visible: **better data flips both architectures
from losing to a matched GPT-2 (OWT/C4) to beating it (RefinedWeb/DCLM), at 4.6–6.4× less
compute** — a far bigger lever than any architecture change (in B1 no new architecture beat
GPT-2 on OWT at all). And **C4 is worse than OWT for both architectures independently** —
a direct cross-validation of Exp A's non-monotonic-in-release-year finding, now reproduced
on two further architectures (GPT-NeoX and Llama lineages), so the effect is a property of
the data, not of a particular algorithm.

## Where the artifacts live (checkpoints & data)

**Durable homes:** git (all code, metrics, results JSONs, figures, `report.md`);
Hugging Face (model checkpoints); a local backup mirror at
`~/Desktop/era_ladder_backup/`. Pod `/workspace` volumes are ephemeral working space.

| Asset | Primary location | Backup / notes |
|---|---|---|
| **Core-study finals** (16 ckpts: current-arch 124M/355M + scaleup 124M/1.5B) | private HF `MIRIBerkeley/data-vs-algorithms-ceg` (73.5 GB) | current-arch mirrored local (`hf_current_arch_finals/`, 10 GB). **scaleup is HF-only** and the private repo is over quota → currently read-blocked (resolve by upgrading the HF plan). |
| **Exp B checkpoints** (14: 6 GPT-2 denominators + 6 candidates + 2 replicate seeds) | public HF `MIRIBerkeley/data-vs-algorithms-ceg-expB` (35.7 GB) | local (`b1_cand/checkpoints/`, 36 GB); metrics.csv in git (`results/b1_metrics/`) |
| **Exp B matched GPT-2 baselines** (6, train_old) | local (`b1_checkpoints/`, 16 GB) | thresholds in git (`results/b1_baseline_thresholds.json`) |
| **Exp A era-ladder arms** | OWT/DCLM = the 2×2 study's 124M finals (HF + local); C4/RefinedWeb finals on public HF `…-expB` under `exp-a-core-retrain/` + local | (the original C4/RW checkpoints were lost — pod-volume-only — and retrained; all era finals are now durably saved) |
| **Tokenized datasets** | pod `/workspace/datasets/` (all 4 corpora × gpt2/nanogpt; OWT × neox/smollm2) | `owt_neox`/`owt_smollm2` mirrored local (`b1_datasets/`, 34 GB); `wiki_eval_union` in `eval_sets.tgz`. C4/RefinedWeb/DCLM in the neox/smollm2 tokenizers (the data-ladder prereq) are being built now. |
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

## How we measure (methodology)

A handful of choices make the compute comparisons fair; they are fixed for every run so
that a difference in the result reflects a difference in data or architecture, not in
how we measured it.

- **Quality is bits-per-byte, not per-token loss.** Different tokenizers split text
  differently, so per-token cross-entropy isn't comparable across them; bits-per-byte
  is. We never compare per-token loss across architectures.
- **One fixed quality bar for everything.** The bar is bits-per-byte on a fixed,
  decontaminated slice of Wikipedia — the same raw text for every run, at every size.
  The exact bar at each size is the fully-trained GPT-2 baseline's score (averaged over
  the final 10% of its training; `analysis/threshold.py`).
- **Compute is timed GPU-hours**, excluding warmup/compile and evaluation time. We do
  not compare raw FLOPs across architectures (their numerical precision differs, so
  FLOPs aren't comparable).
- **Hyperparameters are fixed per (architecture, size).** They are never re-tuned
  between the data conditions, so the data comparison isn't contaminated by tuning.
- **The tokenizer counts as part of the "algorithm."** Each corpus is tokenized once
  per tokenizer; switching tokenizer is an algorithm change, not a data change.
- **Downstream tasks (CORE) are a secondary check**, not the headline. They are noisy
  at small scale, so we gate each task on clearing chance by a margin before using it,
  and we never make a compute-efficiency claim from them.
- **Crossings are interpolated in log-compute** between checkpoints, never snapped to
  the nearest one; the data/algorithm split is a Shapley decomposition in log-compute
  space, reported as multipliers.

One design point changed mid-project and is worth stating plainly. The original plan
fixed the token budget across arms (old-algorithm 9B, new-algorithm 18B), which forced
the new-algorithm/old-data cell to train two epochs on OpenWebText. That was abandoned
after the modded recipe broke down under the very long schedule a fixed 18B budget
implied at small sizes. In the final design each modded arm trains at twice its size's
native budget in a single pass (well under one epoch), so no cell repeats data. (Noted
here as an explicit correction rather than a silent edit; see `report.md`.)

Session procedures, the pipeline smoke test, and the cost ledger are in `RUNBOOK.md`;
the full write-up is `report.md`.
