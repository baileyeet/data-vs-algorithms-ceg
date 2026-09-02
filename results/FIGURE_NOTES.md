# Publication figure set — captions, provenance, and changelog

Prepared for the MIRI Technical Governance blog post / paper. Every figure below is
rendered to **PNG + vector PDF** (same basename) and is fully reproducible from in-repo
data only (no `~/Desktop/era_ladder_backup/` dependency). No underlying analysis number
was changed for presentation; the only edits to result files were **prose** (see changelog).

## The six canonical figures

### 1. `multipliers_vs_scale.{png,pdf}` — HERO
Script: `analysis/plots.py` (`multipliers_vs_scale`, via `python analysis/plots.py multipliers --out results/multipliers_vs_scale.png`).
Data: `results/small/ceg_newdef.json`, `results/medium/ceg_newdef.json`, `results/scaleup/ceg_124m_matrix.json`, `results/xl/ceg_1p5b_matrix.json`. Eval set: **wiki_eval**.
Caption (5 points):
1. Two lineages that share the factorial 2×2 estimator — current-arch (modded-nanoGPT speedrun) and 2024 ScaleUp — as algorithm and data compute-reduction multipliers vs GPT-2-baseline scale.
2. The algorithm advantage collapses with scale for current-arch (13.7×→4.1×, 124M→355M) and decays mildly for ScaleUp (2.9×→2.3×, 124M→1.5B); the data multiplier is comparatively stable (~2–3.7×).
3. x-axis is the **GPT-2 baseline scale** (baseline dimensions), not a candidate parameter count: the current-arch model carries ~498.8M params at the 124M-baseline dimensions (the value-embed / U-net additions are the algorithm being measured).
4. A visible **1× parity line** (no compute advantage) is drawn on both panels; log-y.
5. The two curves use independent hardware GPU-hour bases (8-GPU vs 5-GPU) and are never mixed in one ratio; Pythia/SmolLM2 are deliberately omitted (different estimator; shown in Fig. 3).

### 2. `corpus_intervention.{png,pdf}` — Exp A corpus (replaces `era_ladder.png` + `era_curves.png`)
Script: `analysis/plot_corpus_intervention.py`. Data: `results/era_orig_metrics/*/metrics.csv` (the ORIGINAL era runs) + `results/era_ladder_results.json`. Eval set: **wiki_eval_union**.
Caption:
1. Swapping the training corpus at the fixed 124M baseline scale; color follows the corpus in every panel (OWT gray, C4 amber, RefinedWeb blue, DCLM green).
2. Panel A — evidence: BPB-vs-GPU-hours for all four corpora under both recipes (old-algo solid, current-arch dashed); C4 never reaches the threshold under either recipe.
3. Panel B — corpus CEG vs the old-algo·OWT reference; Panel C — the within-recipe data lever (hold the recipe, swap OWT→corpus).
4. Corpus quality is **non-monotonic in release date**: C4 (2020) is censored under both recipes while RefinedWeb (2023) and DCLM (2024) cross strongly.
5. The data lever differs by recipe (old-algo ~3.3–3.5× vs current-arch ~1.6× on RefinedWeb/DCLM): the corpus and recipe interventions **interact**, so a single Shapley "value of data" is context-dependent (stated, not resolved into one number).

### 3. `expb_arch_curves.{png,pdf}` — Exp B architecture (redesigned in place)
Script: `analysis/plot_training_curves_ab.py` (`expb_arch`). Data: `results/b1_metrics/*/metrics.csv` + `results/b1_results.json`. Eval: OWT-trained, wiki_eval_union neutral.
Caption:
1. Six matched comparisons — each modern architecture vs a GPT-2 trained through the identical pipeline at the same size (train_hf @512k denominator); rows = lineage (Pythia/SmolLM2), columns = increasing scale.
2. Shared BPB axis across all six so the small candidate-vs-GPT-2 gaps are not visually exaggerated; lower BPB = better.
3. No candidate curve reaches its matched GPT-2 bar → **a large speedrun-style algorithmic advantage does not reproduce** across these six comparisons (best case SmolLM2-135M is within-noise parity).
4. For the two divergence-confounded SmolLM2 runs (360M, 1.7B) the best (minimum) BPB is marked, since the final BPB overshoots as the run over-fits OWT; both the tail Δ and the best-case Δ are shown.
5. This is a "does not reproduce" result, not "modern architectures are worse than GPT-2."

### 4. `data_replication.{png,pdf}` — Exp B data axis (replaces `expb_data_curves.png` + `data_ladder_expB.png`)
Script: `analysis/plot_training_curves_ab.py` (`expb_data_replication`). Data: `results/b1_metrics/` (OWT) + `results/data_ladder_metrics/` + bars from `results/data_ladder_results.json`.
Caption:
1. Does the corpus effect reproduce across architectures? Each lineage (Pythia-160M, SmolLM2-135M) trained on four corpora vs its fixed size-matched GPT-2-OWT bar.
2. Color follows the corpus entity (matches Fig. 2); censored corpora carry hollow legend markers.
3. RefinedWeb (2023) and DCLM (2024) cross the bar under **both** lineages; OWT (2019) and C4 (2020) stay censored under both.
4. The corpus effect **reproduces across the two tested stacks** (GPT-NeoX and Llama) — described as replication across tested architectures, not "architecture-independent" / "belongs to the data."
5. C4≈OWT under both lineages independently cross-validates the Exp A non-monotonicity.

### 5. `core_bpb_vs_downstream.{png,pdf}` — Exp B CORE (the single downstream figure)
Script: `analysis/plot_core_disagreement.py`. Data: `results/core_expb_summary.json` + `results/b1_results.json`.
Caption:
1. BPB compute-efficiency and downstream-task capability can disagree; both panels are oriented so up = candidate beats its size-matched GPT-2.
2. Left — compute efficiency (neutral BPB): neither lineage is above parity at any scale (the BPB headline), with the ±1σ BPB-noise band shown.
3. Right — downstream capability (CORE 11-task mean, ±1 stderr): Pythia stays at/below parity, but SmolLM2 shows a small, consistent edge.
4. So SmolLM2 never beats GPT-2 on BPB yet shows a downstream edge the BPB metric does not register — a metric disagreement, not a compute-efficiency claim.
5. CORE at limit=500 is secondary and noisy; SmolLM2 360M/1.7B BPB points are divergence-confounded (open markers, best-case less negative).

### 6. `method_factorial.{png,pdf}` — method schematic (candidate / recommended as an early "how to read" figure)
Script: `analysis/plot_method_factorial.py`. Data: `results/small/ceg_newdef.json` + `results/small/small_a0d0_dense_metrics.csv`.
Caption:
1. How a CEG is measured and decomposed, worked at the 124M current-arch point (no numbers invented — the multipliers are re-derived from the four arm GPU-hours).
2. Panel A — the primitive: train to a fixed neutral-BPB threshold and read off the GPU-hours (fewer = more compute-efficient).
3. Panel B — the 2×2 factorial: horizontal edges swap data (hold algorithm), vertical edges swap algorithm (hold data); each edge is a GPU-hours ratio.
4. The Shapley multiplier of an intervention is the geometric mean of its two edges (both orderings averaged): algorithm 13.69×, data 2.23×, total 30.5×.
5. **Recommendation:** include near the top of the blog as the "how to read the numbers" explainer (now embedded there); it is optional/appendix for the paper if space is tight.

## Numerical / methodological inconsistencies found (audit)

1. **C4 current-arch crossing — original vs retrain (RESOLVED, no numbers changed).** The in-repo
   `results/era_retrain_metrics/era_a1_c4/` CSV (a fresh CORE-recovery rerun) reaches min BPB
   **1.2730 < threshold 1.2760 → it crosses**, contradicting `era_ladder_results.json` /the report
   ("C4 censored under both algorithms"). The **original** run (`era_orig_metrics/era_a1_c4`, which
   the JSON was computed from) has min **1.2845 > 1.2760 → censored**. This is a same-seed noise-floor
   flip (0.003 below vs 0.008 above the threshold; the study's noise floor is ±0.01). **Resolution:**
   all Exp A figures plot the **originals** (verified to reproduce every JSON crossing exactly), so the
   figures match the canonical CEG values. Caveat worth a report footnote: C4-under-current-arch is a
   borderline censoring right at the noise floor, whereas C4-under-old-algo is clearly censored
   (min 1.306–1.312). The "non-monotonic in release date" finding is unaffected.
2. **Two 124M algorithm decompositions coexist (by design).** wiki_eval 2×2 (data 2.23× / algo 13.69×,
   threshold 1.2744) vs the Exp A union-eval re-score (data 2.39× / algo 16.10×, threshold 1.2760).
   Kept separate per instruction: wiki_eval is canonical for the cross-scale hero; the union-eval value
   is Exp A's own basis. Not averaged.
3. **`13.1×` typo → `13.69×`** in the Curve-1/Curve-2 prose (source `analysis/make_report.py`).
4. **`era_ladder.png` plotted non-comparable quantities** → replaced by `corpus_intervention.png`.
5. **Reproducibility gap** — Exp A/B training CSVs were partly outside the repo → the originals were
   copied in-repo (`results/era_orig_metrics/`) and all publication scripts now read in-repo only.
6. **Stale hero prose** describing Exp B "open ▽ at 1×" markers on `multipliers_vs_scale.png` was removed
   (Pythia/SmolLM2 are no longer on the hero).

## Changelog (this pass)
- `report.md` + `analysis/make_report.py`: `13.1×`→`13.69×` (×2); Exp B C4 cross-validation paragraph
  rewritten to the approved interpretation (C4 marginally lower than OWT but both censored — not "C4
  worse than OWT"); hero-triangle prose updated; figure embeds rewired to the six canonical figures;
  method figure embedded as the "how to read" explainer.
- `results/data_ladder_results.json`: `findings[1]` prose corrected (numbers unchanged).
- `analysis/plots.py`: added `_savefig` (PNG+PDF); hero redesigned (current-arch+ScaleUp only,
  GPT-2-baseline-scale axis, 1× line, param footnote); `multipliers` CLI no longer passes Exp B markers.
- `analysis/plot_training_curves_ab.py`: `expb_arch` redesigned (2×3, in-repo, shared axis, divergence
  min markers); `expb_data`/`era` replaced by `expb_data_replication` (new `data_replication.png`);
  backup-dir dependency removed.
- New scripts: `analysis/plot_corpus_intervention.py`, `analysis/plot_core_disagreement.py`,
  `analysis/plot_method_factorial.py`.
- New in-repo data: `results/era_orig_metrics/` (8 original era arm CSVs).

## Superseded / appendix (NOT deleted — awaiting your OK before removal)
- Superseded PNGs still on disk: `era_ladder.png`, `era_curves.png`, `expb_data_curves.png`,
  `data_ladder_expB.png`, `core_expb_delta.png`. Their generating scripts `analysis/plot_era_ladder.py`
  and `analysis/plot_data_ladder.py` are no longer invoked (no build driver runs them). Safe to delete
  on your confirmation.
- Appendix-only (kept, not in the blog's main flow): `all_configs.png`, per-size `curves.png`/
  `sensitivity.png`, `core_*_by_task*.png`, `core_vs_scale.png`, `core_arms_by_task.png`,
  `core_era_ladder.png`, `core_era_by_task.png`.

## Materially-changed findings
None. All CEG/BPB/CORE numbers are unchanged. Changes are presentation + prose only, plus one newly
**disclosed nuance**: C4-under-current-arch censoring sits right at the ±0.01 noise floor (original
censored by +0.008, a later rerun crossed by −0.003); C4-under-old-algo is robustly censored.
