# Publication figure set — captions, provenance, and changelog

Prepared for reuse in both the MIRI Technical Governance blog post and an academic paper.
Every figure below is rendered to **PNG + vector PDF** (same basename) and is fully
reproducible from in-repo data only (no `~/Desktop/era_ladder_backup/` dependency). No
underlying analysis number was changed for presentation — this pass is titles, labels,
legends, and in-plot text only.

**Regeneration commands** (all read only in-repo `results/` data):
```
python analysis/plots.py multipliers --out results/multipliers_vs_scale.png
python analysis/plot_corpus_intervention.py
python analysis/plot_training_curves_ab.py       # writes expb_arch_curves.png + data_replication.png
python analysis/plot_core_disagreement.py
python analysis/plot_method_factorial.py
```
Requires `matplotlib` (see `requirements.txt`, pinned `requirements-lock.txt`).

**Status of this pass:** the source edits below (titles, axis labels, legends, censoring
convention, jargon removal) are complete and committed. The actual PNG/PDF re-render and
visual QA pass (overlap/clipping/whitespace inspection) could **not** be completed in this
session — the session's network egress policy denies `pypi.org`/`files.pythonhosted.org`
(403, confirmed an organization policy, not a transient failure), so `matplotlib` could not
be installed and no plotting script could be run. **Anyone regenerating these figures should
run the commands above and visually check every panel before publishing** — this pass reasoned
about layout from the code and the previously-rendered images, but did not re-verify the
final pixels.

## Terminology used consistently across all six figures
- **GPT-2 baseline scale** — the reference dimensions (124M / 355M / 1.5B) that a candidate
  is measured against; not necessarily the candidate's own parameter count (flagged wherever
  it matters, e.g. current-arch at 124M-baseline dimensions).
- **Current training recipe** ("current-arch") = the 2024 modded-nanoGPT speedrun algorithm.
  **Old GPT-2 recipe** ("old-algo") = the 2019 GPT-2 training algorithm.
- **OpenWebText (OWT)**, **C4**, **RefinedWeb**, **DCLM** — training corpora, always labeled
  with release year.
- **Pythia (GPT-NeoX, 2023)**, **SmolLM2 (Llama, 2024)** — architecture families, always
  labeled with their real model sizes (never "small/medium/large").
- **BPB** = bits-per-byte on a fixed held-out corpus never used in training ("neutral-corpus
  BPB"); lower is better.
- **Compute-equivalent gain (CEG)** = the GPU-hours ratio to reach a fixed BPB threshold;
  reported as a "×" multiplier. **Threshold / parity** = the 1× reference (the old-recipe·OWT
  arm's converged BPB, or the size-matched GPT-2's converged BPB in Exp B).
- **Censored / did not reach threshold** — the one, single wording for a comparison that never
  crosses its reference within its compute budget. A censored point is never plotted at a
  value that could be misread as a measured 1× (parity) result: it sits below the parity line,
  as a hollow marker with a short downward tick, always paired with the phrase "did not reach
  threshold."

## The six canonical figures

### 1. `multipliers_vs_scale.{png,pdf}` — HERO
**Title:** "Compute-equivalent gain vs. GPT-2 baseline scale" (was: "Compute-equivalent-gain
multipliers vs model scale").
Script: `analysis/plots.py::multipliers_vs_scale`, via
`python analysis/plots.py multipliers --out results/multipliers_vs_scale.png`.
Data: `results/small/ceg_newdef.json`, `results/medium/ceg_newdef.json`,
`results/scaleup/ceg_124m_matrix.json`, `results/xl/ceg_1p5b_matrix.json`. Eval: **wiki_eval**.
Panels: left "Algorithm contribution", right "Data contribution" — both share the x-axis
"GPT-2 baseline scale (params)" and the y-axis "Compute-equivalent gain (×, log scale)"; both
carry a labeled 1× ("no compute advantage") reference line.
Legend: "current-arch — Shapley (avg of both orderings)" (filled circle, solid) vs.
"ScaleUp — single margin (complement cell censored)" (hollow square, dashed) — two different
estimators, never mixed into one number (see caption point 5).
**Changed:** in-plot footnote trimmed to one line (axis-reading note only); the ~499M
candidate-parameter detail and the independent-hardware-base caveat now live only in the
caption below, not baked into the image.
Caption (for reuse in blog/paper):
1. Two lineages that share the factorial 2×2 estimator — current-arch (modded-nanoGPT
   speedrun) and 2024 ScaleUp — as algorithm and data compute-equivalent-gain multipliers vs.
   GPT-2 baseline scale.
2. The algorithm advantage collapses with scale for current-arch (13.7×→4.1×, 124M→355M) and
   decays mildly for ScaleUp (2.9×→2.3×, 124M→1.5B); the data multiplier is comparatively
   stable (~2–3.7×).
3. The x-axis is the **GPT-2 baseline scale** (baseline dimensions), not a candidate parameter
   count: the current-arch model carries ~498.8M parameters at the 124M-baseline dimensions
   (the value-embed / U-net additions are the algorithm being measured).
4. A visible 1× parity line (no compute advantage) is drawn on both panels; log y-axis.
5. The two curves use independent hardware GPU-hour bases (8-GPU vs. 5-GPU) and are never
   mixed in one ratio; Pythia/SmolLM2 are deliberately omitted (different estimator, no data
   axis to decompose against — shown in Fig. 3/4 instead).
**Recommend:** (a) blog main text — yes; (b) paper main text — yes (the hero result); (c)
appendix only — no.

### 2. `corpus_intervention.{png,pdf}` — corpus intervention at fixed scale
**Title:** "Corpus compute-equivalent gain at the 124M GPT-2 baseline scale" (was: "Exp A —
corpus intervention at the 124M GPT-2 baseline scale (neutral eval: wiki_eval_union)").
Script: `analysis/plot_corpus_intervention.py`.
Data: `results/era_orig_metrics/*/metrics.csv` (the original era runs) +
`results/era_ladder_results.json`. Eval: **wiki_eval_union**.
Panels:
- A · "Compute to reach the neutral-BPB threshold" — raw BPB-vs-GPU-hours for all four
  corpora under both recipes (old GPT-2 recipe solid, current training recipe dashed).
- B · "Corpus CEG vs. old-recipe·OWT reference" — total compute-equivalent gain each corpus
  buys vs. the fixed reference arm, under each recipe.
- C · "Within-recipe corpus CEG (OWT → corpus)" (was "within-recipe data lever") — hold the
  recipe fixed, swap only the corpus; isolates the corpus-only compute-equivalent gain.
Color follows the corpus (OWT gray, C4 amber, RefinedWeb blue, DCLM green) in every panel;
recipe is a secondary encoding (linestyle in A, marker shape in B/C).
**Changed (the one real plotting-convention bug found in this pass):** in panels B/C, a
corpus that never reaches the threshold under a recipe was previously plotted with a hollow
marker sitting exactly ON the 1× parity line — indistinguishable at a glance from a measured
"no advantage" (1×) result, which is the failure mode the "censored markers must not look like
measured 1× values" convention exists to prevent. Fixed: censored markers now sit at a fixed
sub-parity position (below the 1× line) with a short downward tick and the label "did not
reach threshold," never on the reference line itself.
Also removed from the image: the floating "the corpus and training-recipe interventions
INTERACT... context-dependent" conclusion sentence and the "corpus quality is non-monotonic in
release date" conclusion — both moved to the caption below (point 5) and to `report.md`.
Caption:
1. Swapping the training corpus at the fixed 124M baseline scale, under both the old GPT-2
   recipe and the current training recipe.
2. Panel A is the underlying evidence: BPB-vs-GPU-hours curves; C4 never reaches the threshold
   under either recipe.
3. Panel B is compute-equivalent gain vs. the fixed old-recipe·OWT reference; Panel C isolates
   the corpus-only effect within each fixed recipe.
4. Corpus quality is **non-monotonic in release date**: C4 (2020) is censored under both
   recipes while RefinedWeb (2023) and DCLM (2024) cross strongly.
5. The within-recipe corpus effect differs by recipe (old GPT-2 recipe ~3.3–3.5× vs. current
   training recipe ~1.6× on RefinedWeb/DCLM) — the corpus and training-recipe interventions
   **interact**, so a single Shapley "value of the corpus" is recipe-dependent, not one number.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 3. `expb_arch_curves.{png,pdf}` — architecture comparisons
**Title:** "Bits per byte vs. GPU-hours for matched architecture comparisons" (was: "Exp B — a
large speedrun-style algorithmic advantage does NOT reproduce across six matched
comparisons" — a conclusion-style title, now removed).
Script: `analysis/plot_training_curves_ab.py::expb_arch`.
Data: `results/b1_metrics/*/metrics.csv` + `results/b1_results.json`. Eval: **wiki_eval_union**
(OWT-trained arms only).
Layout: 2×3 grid, rows = architecture family (Pythia (GPT-NeoX, 2023) / SmolLM2 (Llama,
2024)), columns = increasing scale (labeled with real sizes: 160M/410M/1.4B and
135M/360M/1.7B). Shared BPB y-axis across all six panels. Each panel: the candidate curve
(colored by family) vs. its matched GPT-2 trained through the identical pipeline (gray),
dashed reference line at the GPT-2 bar, and a compact Δ annotation.
**Changed:** the jargon term "divergence-confounded" / "(rises after; diverges)" is removed
from all rendered text. For SmolLM2-360M and SmolLM2-1.7B (whose BPB improved, then worsened
again late in training as the model overfit the training corpus), the plot now marks the best
(lowest) checkpoint with the plain-language annotation "best ckpt N.NNN (BPB rose again
after)," and reports both the final and best-checkpoint Δ.
Caption:
1. Six matched comparisons — each modern architecture vs. a GPT-2 trained through the
   identical pipeline at the same size (same trainer, tokens, and batch schedule); rows =
   architecture family, columns = increasing scale.
2. Shared BPB axis across all six so the small candidate-vs-GPT-2 gaps are not visually
   exaggerated; lower BPB is better.
3. No candidate curve reaches its matched GPT-2 bar — a large speedrun-style algorithmic
   advantage does not reproduce across these six comparisons (best case, SmolLM2-135M, is
   within-noise parity, not a crossing).
4. For SmolLM2-360M and SmolLM2-1.7B, BPB on the held-out eval corpus worsened again late in
   training even as loss on the training corpus kept improving (the model was overfitting the
   training data); the best checkpoint reached before that reversal is marked and reported
   alongside the final value.
5. This is a "does not reproduce" result about these two architecture families at these
   sizes, not a general claim that modern architectures underperform GPT-2.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 4. `data_replication.{png,pdf}` — corpus effect across architectures
**Title:** "Neutral-corpus BPB vs. GPU-hours by training corpus" (was: "Exp B — the data
effect reproduces across both tested architecture stacks").
Script: `analysis/plot_training_curves_ab.py::expb_data_replication`.
Data: `results/b1_metrics/` (OWT) + `results/data_ladder_metrics/`. Eval: **wiki_eval_union**.
Two panels, one per architecture (Pythia-160M (GPT-NeoX), SmolLM2-135M (Llama)); within each,
the same architecture trained on four corpora vs. its fixed size-matched GPT-2-OWT bar.
**Changed:** title de-conclusioned; no in-plot prose sits across the plotting area (this was
already clean before this pass — confirmed by inspection of the current PNG).
Caption:
1. Does the corpus effect reproduce across architectures? Each architecture (Pythia-160M,
   SmolLM2-135M) trained on four corpora vs. its fixed size-matched GPT-2-OWT bar.
2. Color follows the corpus (matches Fig. 2); censored corpora carry hollow legend markers.
3. RefinedWeb (2023) and DCLM (2024) cross the bar under **both** architectures; OWT (2019)
   and C4 (2020) stay censored under both.
4. The corpus effect reproduces across the two tested architecture families (GPT-NeoX and
   Llama).
5. C4 ≈ OWT under both architectures independently cross-validates the Exp A finding that
   corpus quality is non-monotonic in release year.
**Recommend:** (a) blog main text — yes; (b) paper main text — optional (strong supporting
evidence, but largely a replication of Fig. 2's finding on new architectures — a paper tight on
space could push it to an appendix); (c) appendix — acceptable alternative to (b).

### 5. `core_bpb_vs_downstream.{png,pdf}` — BPB vs. downstream capability
**Title:** "Relative BPB efficiency and CORE performance by model scale" (was: "Exp B — a
metric disagreement: BPB efficiency is not the same as downstream capability").
Script: `analysis/plot_core_disagreement.py`.
Data: `results/core_expb_summary.json` + `results/b1_results.json`.
Two panels, both oriented so **up = candidate beats its size-matched GPT-2**: left "Compute
efficiency (neutral-corpus BPB)", right "Downstream capability (CORE, 11-task mean
accuracy)". X-axis now reads "Model scale (Pythia / SmolLM2 matched pairs, params)" with real
paired sizes as tick labels (≈135–160M / ≈360–410M / ≈1.4–1.7B) — never "small/medium/large".
**Changed:** the legend entry "open = divergence-confounded (SmolLM2 360M/1.7B; best-case less
negative)" is replaced with plain language: "open = best checkpoint (SmolLM2 360M/1.7B; BPB
rose again later)." Y-axis units made explicit (bits on the left, accuracy on the right).
Caption:
1. BPB compute-efficiency and downstream-task capability can disagree; both panels are
   oriented so up = candidate beats its size-matched GPT-2 (identical pipeline).
2. Left — compute efficiency (neutral-corpus BPB, bits): neither architecture is above parity
   at any scale (the BPB headline), with the ±1σ BPB-noise band shown.
3. Right — downstream capability (CORE 11-task mean accuracy, ±1 stderr): Pythia stays at/below
   parity, but SmolLM2 shows a small, consistent edge.
4. So SmolLM2 never beats GPT-2 on BPB, yet shows a downstream edge that BPB does not register
   — a metric disagreement, not a compute-efficiency claim.
5. CORE at limit=500 is secondary and noisy; the open SmolLM2 360M/1.7B BPB points are its best
   checkpoint (BPB rose again later in training — see Fig. 3 caption).
**Recommend:** (a) blog main text — yes; (b) paper main text — yes (the metric-disagreement
finding is a distinct, citable result); (c) appendix only — no.

### 6. `method_factorial.{png,pdf}` — method figure
**Title:** "Factorial decomposition of compute-equivalent gain" (subtitle: "Worked example:
124M GPT-2 baseline scale, current training recipe"); was: "How a compute-equivalent gain is
measured and decomposed (worked at the 124M current-arch point)".
Script: `analysis/plot_method_factorial.py`.
Data: `results/small/ceg_newdef.json` + `results/small/small_a0d0_dense_metrics.csv`.
Panel A · "The primitive — compute to reach the threshold": one training curve, the threshold
line, and the crossing point. Panel B · "The 2×2 factorial and its log-space Shapley split":
the four arms as boxes (old/new algorithm × old/new data), GPU-hours-ratio edges (data edges
blue, algorithm edges orange), and the Shapley summary (geometric mean of each intervention's
two edges).
**Changed:** title split into a neutral main title + a smaller subtitle carrying the "worked
example" context (previously one long conclusion-adjacent title). Does not use "core 2×2"
anywhere — refers to "the 2×2 factorial" / "factorial experiment."
Caption:
1. How a compute-equivalent gain (CEG) is measured and decomposed, worked at the 124M
   current-training-recipe point (no numbers invented — the multipliers are re-derived from
   the four arms' GPU-hours).
2. Panel A — the primitive: train to a fixed neutral-BPB threshold and read off the GPU-hours
   (fewer = more compute-efficient).
3. Panel B — the 2×2 factorial: horizontal edges swap data (hold algorithm fixed), vertical
   edges swap algorithm (hold data fixed); each edge is a GPU-hours ratio.
4. The Shapley multiplier of an intervention is the geometric mean of its two edges (both
   orderings averaged): algorithm 13.69×, data 2.23×, total 30.5×.
5. Intended as an early "how to read the numbers" explainer figure, not a privileged or
   larger experiment — it is one worked example of the same factorial design used throughout.
**Recommend:** (a) blog main text — yes, near the top, before the results (as already
embedded in `report.md`); (b) paper main text — optional, works well as a Methods-section
figure; (c) appendix — acceptable alternative to (b) if space is tight.

## Numerical / methodological inconsistencies found (audit)

1. **C4 current-arch crossing — original vs retrain (RESOLVED, no numbers changed).** The
   in-repo `results/era_retrain_metrics/era_a1_c4/` CSV (a fresh CORE-recovery rerun) reaches
   min BPB **1.2730 < threshold 1.2760 → it crosses**, contradicting `era_ladder_results.json`
   / the report ("C4 censored under both algorithms"). The **original** run
   (`era_orig_metrics/era_a1_c4`, which the JSON was computed from) has min
   **1.2845 > 1.2760 → censored**. This is a same-seed noise-floor flip (0.003 below vs 0.008
   above the threshold; the study's noise floor is ±0.01). **Resolution:** all Exp A figures
   plot the **originals** (verified to reproduce every JSON crossing exactly), so the figures
   match the canonical CEG values. Caveat worth a report footnote: C4-under-current-recipe is
   a borderline censoring right at the noise floor, whereas C4-under-old-recipe is clearly
   censored (min 1.306–1.312). The "non-monotonic in release date" finding is unaffected.
2. **Two 124M algorithm decompositions coexist (by design).** wiki_eval 2×2 (data 2.23× /
   algo 13.69×, threshold 1.2744) vs the Exp A union-eval re-score (data 2.39× / algo 16.10×,
   threshold 1.2760). Kept separate per instruction: wiki_eval is canonical for the cross-scale
   hero; the union-eval value is the corpus-intervention figure's own basis. Not averaged.
3. No new inconsistency was found in this pass — this pass audited titles, labels, legends,
   annotation text, and one plotting-convention bug (censored markers plotted at 1×, see
   Fig. 2's changelog entry above), not the underlying analysis. No CEG/BPB/CORE number was
   touched.

## Changelog (this pass — figure-quality revision)
- **`analysis/plots.py`** (`multipliers_vs_scale`): title → "Compute-equivalent gain vs. GPT-2
  baseline scale"; y-axis → "Compute-equivalent gain (×, log scale)"; x-axis →
  "GPT-2 baseline scale (params)"; in-plot footnote trimmed from a 3-line param-count caveat
  paragraph to one factual line, with the detailed caveat moved to the caption only.
- **`analysis/plot_corpus_intervention.py`**: title de-conclusioned; panel C renamed from
  "Within-recipe data lever" to "Within-recipe corpus CEG"; **fixed the censored-marker-at-1×
  plotting bug** (see above) — censored points now render below the parity line with a
  downward tick and "did not reach threshold," never on the 1× line; removed the floating
  "INTERACT" / "non-monotonic" conclusion sentences from the in-image footnote (moved to
  caption/report); y-axis labels unified to "Compute-equivalent gain (×, log scale)" /
  "Within-recipe corpus CEG (×, log scale)"; legend relabeled "old GPT-2 recipe" / "current
  training recipe" / "hollow, below 1× = did not reach threshold".
- **`analysis/plot_training_curves_ab.py`**: `expb_arch` title → "Bits per byte vs. GPU-hours
  for matched architecture comparisons"; replaced "(rises after; diverges)" with "(BPB rose
  again after this point)" language; BPB axis label unified to "Neutral-corpus BPB (bits/byte,
  lower = better)" with units. `expb_data_replication` title → "Neutral-corpus BPB vs.
  GPU-hours by training corpus"; "lineages" → "architectures" in footnote.
- **`analysis/plot_core_disagreement.py`**: title → "Relative BPB efficiency and CORE
  performance by model scale"; legend "open = divergence-confounded..." → "open = best
  checkpoint (SmolLM2 360M/1.7B; BPB rose again later)"; y-axis units made explicit; x-axis
  label clarified to name the two paired size sequences.
- **`analysis/plot_method_factorial.py`**: title split into a neutral main title +
  a smaller "worked example" subtitle.
- **`analysis/make_report.py`** (source of `report.md`, which is fully auto-generated — hand
  edits to `report.md` do not persist): "data lever" → "corpus effect" / "within-recipe corpus
  effect" (2 occurrences); "divergence-confounded" → plain-language description of the
  best-checkpoint-vs-overfitting behavior (3 occurrences); figure alt-text updated to the new
  titles. Regenerated `report.md` from the updated script (verified byte-for-byte the direct
  output of `python analysis/make_report.py --results-dir results --out report.md`).
- No changes to any `results/*.json` or `results/*/metrics.csv` — this pass touched
  presentation code and generated markdown only.

## Superseded / appendix
- Superseded figures **live in `results/superseded/`** (retained for provenance, not
  deleted): `era_ladder.png`, `era_curves.png`, `expb_data_curves.png`, `data_ladder_expB.png`,
  `core_expb_delta.png` — see `results/superseded/README.md` for the replacement mapping.
- Appendix-only (kept at top level, not in the blog's/paper's main flow): `all_configs.png`,
  per-size `curves.png`/`sensitivity.png`, `core_*_by_task*.png`, `core_vs_scale.png`,
  `core_arms_by_task.png`, `core_era_ladder.png`, `core_era_by_task.png`.

## Materially-changed findings
None. Every CEG/BPB/CORE number is unchanged. This pass is titles, axis labels, legends,
annotation wording, and one plotting-convention fix (censored markers no longer rendered at
the 1× parity value); see the changelog above for the exhaustive list of text/code edits.

## Known limitation of this pass
The PNG/PDF outputs described above were **not re-rendered or visually re-inspected** in this
session (matplotlib install blocked by network policy — see the note at the top of this file).
The edits were reasoned through against the plotting code and the previously-rendered images
(read before this pass began), but the exact pixel output — spacing, whether the new censored-
marker position and best-checkpoint annotations land without collision, whether the shortened
footnotes still fit their reserved margins — has not been confirmed. **Run the regeneration
commands above and inspect every panel (PNG and PDF) before publishing.**
