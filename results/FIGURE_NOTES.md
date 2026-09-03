# Publication figure set — captions, provenance, and changelog

Prepared for reuse in both the MIRI Technical Governance blog post and an academic paper.
Every figure below is rendered to **PNG + vector PDF** (same basename) and is fully
reproducible from in-repo data only (no `~/Desktop/era_ladder_backup/` dependency). No
underlying analysis number was changed for presentation — every pass to date (this one
included) is titles, labels, legends, and in-plot text only.

**Regeneration commands** (all read only in-repo `results/` data):
```
python analysis/plots.py multipliers --out results/multipliers_vs_scale.png
python analysis/plot_corpus_intervention.py
python analysis/plot_training_curves_ab.py       # writes expb_arch_curves.png + data_replication.png
python analysis/plot_core_disagreement.py
python analysis/plot_method_factorial.py
```
Requires `matplotlib` (see `requirements.txt`, pinned `requirements-lock.txt`).

**Status of this pass (2026-09-03):** a prior session (`bda2398`, merged as PR #1) wrote the
source edits for a first figure-quality pass but could not install `matplotlib` in its
sandbox, so it never rendered or visually inspected the actual PNG/PDF output. This session
had a working `matplotlib` + `numpy`, so it (1) ran all five regeneration commands, (2)
visually inspected every PNG at full resolution and at 2–3× crops of every legend/annotation
cluster, (3) found and fixed three real rendering bugs plus a round of academic-audience
copy edits requested by the user, and (4) re-rendered and re-inspected until clean. No PDF
rasterizer (`pdftoppm`/`pymupdf`) is available in this container; the PDFs are not separately
rasterized and eyeballed pixel-by-pixel, but they are written by the same `fig.savefig()` call
on the same already-laid-out Matplotlib `Figure` object as the PNG (see `plots.py::_savefig`),
so the element positions are identical between the two — only antialiasing/font-hinting
differs. All six PDFs were checked for a valid `%PDF-1.4` header and `%%EOF` trailer.

## Bugs found by rendering (not visible from reading the code)

1. **`corpus_intervention.png`, panels B/C: annotation struck through by the 1× reference
   line.** The floating "did not reach threshold" label was positioned in an 11–13pt gap
   between the censored markers and the dashed 1× line; at that font size the dashes cut
   directly through the letters. **Fix:** removed the redundant floating label — the shared
   legend above the panels already states "hollow, below 1× = did not reach threshold" — and
   shrank the "1× — no compute advantage" sentence to a plain "1×" tag on the line itself.
2. **`corpus_intervention.png`, panel C: two value labels collided into "1.0×1.0×".** At the
   OWT column, both recipes are trivially 1× (swapping OWT → OWT is a no-op under either
   recipe), so the old-recipe and current-recipe labels land on the same value with only a
   0.26-unit marker offset between them — not enough horizontal room for two 4-character
   labels centered on their markers. **Fix:** labels for the two recipes now anchor
   left/right (`ha="right"`/`ha="left"`) off their own marker instead of both centering,
   so they never merge regardless of value.
3. **`core_bpb_vs_downstream.png`: rotated y-axis labels overlapped the bottom caption.**
   The original y-axis labels ("Neutral-corpus BPB advantage over matched GPT-2 (bits, ↑
   better)", similarly for CORE) were long enough, rotated 90°, that their vertical extent
   ran past the axes into the figure's bottom margin, printing directly on top of the
   caption text (visually: "**N**eutral-co..." merged with "**B**oth panels: ..." so both
   lost their leading letter). **Fix:** shortened both y-axis labels, gave the figure more
   height (5.2 → 6.0 in) and more reserved top/bottom margin, and moved the legend row down
   so it no longer collided with the suptitle either (a second overlap introduced and then
   fixed during the same edit — see below).

## Copy/terminology pass (2026-09-03, user-requested — academic-paper audience)

The user's second round of feedback, after seeing the rendered figures, asked to write for
*other researchers* rather than a general blog audience: drop parenthetical method glosses
that a paper's methods section already carries, stop restating what a labeled axis or a
legend already says, use real per-family model sizes instead of shared "small/medium/large"
buckets, and fix capitalization. Changes:

- **`plots.py` (`multipliers_vs_scale`):** legend "current-arch — Shapley (avg of both
  orderings)" / "ScaleUp — single margin (complement cell censored)" → plain entity names
  "Current training recipe" / "ScaleUp (2024)"; the estimator-difference explanation is
  reused in `report.md`'s body text (it was already there, unchanged), not the legend.
  "1× — no compute advantage" → "1×" (the y-axis is already labeled "Compute-equivalent
  gain"; a labeled reference line at 1 needs no restatement for this readership). X-axis
  "GPT-2 baseline scale (params)" → "(number of parameters)". In-plot footnote trimmed to
  one factual clause ("X-axis: dimensions of the reference GPT-2 model, not the trained
  candidate's own parameter count"); the ~499M candidate-parameter figure itself lives only
  in `report.md` and this file's caption, never in the image.
- **`plot_corpus_intervention.py`:** legend/title capitalization ("corpus"/"recipe" →
  "Corpus"/"Recipe"; "old-algo (GPT-2)"/"current-arch (modded)" → "Old GPT-2 recipe"/
  "Current training recipe", matching the B/C legend and the terminology list below); x-axis
  "Training corpus (ordered by release date)" → "Training corpus" (the years printed under
  each tick already show the ordering); "wiki_eval_union" glossed inline as "held-out,
  decontaminated Wikipedia evaluation set" on first use in this figure's caption. See bugs
  1–2 above for the overlap fixes in the same file.
- **`plot_training_curves_ab.py` (`expb_arch`):** the appended `CENSORED_DEF` boilerplate
  ("'Censored' = the arm never reaches the threshold... hollow markers") was **factually
  wrong for this figure** — none of its six panels use a hollow "did not reach threshold"
  marker; the open triangle here marks a *different* concept (best checkpoint before BPB
  rose again late in training). Replaced with an accurate footer sentence naming both
  conventions used in this figure (Δ definition + open-triangle definition). Legend
  "matched GPT-2" → "Matched GPT-2" (was the only lowercase entry in an otherwise
  title-cased legend). Added ~0.03 BPB of y-axis headroom (1.45 → 1.48) so the
  intentionally-clipped early-descent curves stop running flush into the top axis spine
  right under the panel titles.
- **`plot_training_curves_ab.py` (`expb_data_replication`):** legend title "training
  corpus" → "Training corpus"; axhline annotation "matched GPT-2-OWT bar" → "Matched
  GPT-2-OWT bar" (capitalization only; this panel had no overlap bug — confirmed clean by
  inspection both before and after this pass).
- **`plot_core_disagreement.py`:** x-axis tick labels "≈135–160M" / "≈360–410M" /
  "≈1.4–1.7B" (merging Pythia's and SmolLM2's *different* size ladders into one shared
  approximate bucket) replaced with explicit two-line labels naming both real sizes at every
  tier, e.g. "Pythia 160M / SmolLM2 135M" — never implies the two families are at the same
  scale. Removed the floating "candidate better ↑" / "worse ↓" corner annotations on both
  panels (redundant now that the meaning is stated once, compactly, in each y-axis label:
  "(bits; ↑ = candidate better)" / "(↑ = candidate better)"). Legend "open = best checkpoint
  shown (see caption)" → "Open = ..." (capitalization). See bug 3 above for the layout fix
  that was needed to make room for these changes; fixing it introduced a legend/suptitle
  overlap that was caught by re-inspection and fixed in the same edit (suptitle `y=0.985`,
  legend `bbox_to_anchor` moved down to `y=0.90`, top margin widened).
- **`plot_method_factorial.py`:** capitalized every in-panel label/annotation that starts a
  visual "sentence" (axis/cell labels "old data (OWT)" → "Old data (OWT)", etc.; "crosses
  at" → "Crosses at"; "example arm: ..." → "Example arm: ..."; the Shapley summary line
  "data = geomean(...)" / "algorithm = geomean(...)" / "total = ..." → "Data = ..." /
  "Algorithm = ..." / "Total = ..."). Glossed "wiki_eval" in the caption the same way as
  `wiki_eval_union` above.
- **`make_report.py`:** two Markdown image alt-texts didn't match their figure's actual
  title after the first pass ("How a compute-equivalent gain is measured and decomposed" /
  "multipliers vs scale" for `method_factorial.png` / `multipliers_vs_scale.png`); updated
  to the current titles for consistency. Regenerated `report.md`
  (`python analysis/make_report.py --results-dir results --out report.md`) — the diff is
  exactly those two alt-text lines, confirming no other drift.

## Terminology used consistently across all six figures
- **GPT-2 baseline scale** — the reference dimensions (124M / 355M / 1.5B) that a candidate
  is measured against; not necessarily the candidate's own parameter count (flagged wherever
  it matters, e.g. current-arch at 124M-baseline dimensions).
- **Current training recipe** ("current-arch") = the 2024 modded-nanoGPT speedrun algorithm.
  **Old GPT-2 recipe** ("old-algo") = the 2019 GPT-2 training algorithm.
- **OpenWebText (OWT)**, **C4**, **RefinedWeb**, **DCLM** — training corpora, always labeled
  with release year.
- **Pythia (GPT-NeoX, 2023)**, **SmolLM2 (Llama, 2024)** — architecture families, always
  labeled with their real model sizes (never "small/medium/large", never a shared
  approximate bucket that implies the two families sit at the same scale).
- **BPB** = bits-per-byte on a fixed held-out corpus never used in training ("neutral-corpus
  BPB"); lower is better.
- **Compute-equivalent gain (CEG)** = the GPU-hours ratio to reach a fixed BPB threshold;
  reported as a "×" multiplier. **Threshold / parity / 1×** = the reference point (the
  old-recipe·OWT arm's converged BPB, or the size-matched GPT-2's converged BPB in Exp B).
- **Censored / did not reach threshold** — the one, single wording for a comparison that never
  crosses its reference within its compute budget. A censored point is never plotted at a
  value that could be misread as a measured 1× (parity) result: it sits below the parity line,
  as a hollow marker with a short downward tick, defined once per figure (legend or caption),
  not repeated as floating in-plot prose.

## The six canonical figures

### 1. `multipliers_vs_scale.{png,pdf}` — HERO
**Title:** "Compute-equivalent gain vs. GPT-2 baseline scale."
Script: `analysis/plots.py::multipliers_vs_scale`, via
`python analysis/plots.py multipliers --out results/multipliers_vs_scale.png`.
Data: `results/small/ceg_newdef.json`, `results/medium/ceg_newdef.json`,
`results/scaleup/ceg_124m_matrix.json`, `results/xl/ceg_1p5b_matrix.json`. Eval: **wiki_eval**.
Panels: left "Algorithm contribution", right "Data contribution" — both share the x-axis
"GPT-2 baseline scale (number of parameters)" and the y-axis "Compute-equivalent gain (×,
log scale)"; both carry a "1×" reference line.
Legend: "Current training recipe" (filled circle, solid) vs. "ScaleUp (2024)" (hollow square,
dashed) — two different estimators (Shapley average vs. a single surviving margin), never
mixed into one number; the estimator distinction is explained in the caption below and in
`report.md`'s body text, not repeated in the legend.
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
4. current-arch multipliers are true Shapley values (both intervention orderings averaged,
   all four 2×2 cells cross); ScaleUp's A1D0 (ScaleUp on OpenWebText) never crosses, so its
   points are the single surviving margin, not a symmetric Shapley average — see `report.md`
   for the full estimator description.
5. The two curves use independent hardware GPU-hour bases (8-GPU vs. 5-GPU) and are never
   mixed in one ratio; Pythia/SmolLM2 are deliberately omitted (different estimator, no data
   axis to decompose against — shown in Fig. 3/4 instead).
**Recommend:** (a) blog main text — yes; (b) paper main text — yes (the hero result); (c)
appendix only — no.

### 2. `corpus_intervention.{png,pdf}` — corpus intervention at fixed scale
**Title:** "Corpus compute-equivalent gain at the 124M GPT-2 baseline scale."
Script: `analysis/plot_corpus_intervention.py`.
Data: `results/era_orig_metrics/*/metrics.csv` (the original era runs) +
`results/era_ladder_results.json`. Eval: **wiki_eval_union** (held-out, decontaminated
Wikipedia set; glossed inline in the figure's own caption on first use).
Panels:
- A · "Compute to reach the neutral-BPB threshold" — raw BPB-vs-GPU-hours for all four
  corpora under both recipes (old GPT-2 recipe solid, current training recipe dashed).
- B · "Corpus CEG vs. old GPT-2 recipe · OWT reference" — total compute-equivalent gain each
  corpus buys vs. the fixed reference arm, under each recipe.
- C · "Within-recipe corpus CEG (OWT → corpus)" — hold the recipe fixed, swap only the
  corpus; isolates the corpus-only compute-equivalent gain.
Color follows the corpus (OWT gray, C4 amber, RefinedWeb blue, DCLM green) in every panel;
recipe is a secondary encoding (linestyle in A, marker shape in B/C).
**Fixed this pass (real bugs, not just wording — see "Bugs found by rendering" above):**
(1) the floating "did not reach threshold" label was struck through by the 1× dashed line in
panels B/C; removed as redundant with the shared top legend. (2) two value labels at the OWT
column of panel C collided into "1.0×1.0×"; labels now anchor away from each other instead of
both centering on their marker.
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
   **interact**, so a single "value of the corpus" is recipe-dependent, not one number.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 3. `expb_arch_curves.{png,pdf}` — architecture comparisons
**Title:** "Bits per byte vs. GPU-hours for matched architecture comparisons."
Script: `analysis/plot_training_curves_ab.py::expb_arch`.
Data: `results/b1_metrics/*/metrics.csv` + `results/b1_results.json`. Eval: **wiki_eval_union**
(OWT-trained arms only).
Layout: 2×3 grid, rows = architecture family (Pythia (GPT-NeoX, 2023) / SmolLM2 (Llama,
2024)), columns = increasing scale (labeled with real sizes: 160M/410M/1.4B and
135M/360M/1.7B). Shared BPB y-axis across all six panels. Each panel: the candidate curve
(colored by family) vs. its matched GPT-2 trained through the identical pipeline (gray),
dashed reference line at the GPT-2 bar, and a compact Δ annotation.
**Fixed this pass:** the footer's `CENSORED_DEF` boilerplate was inaccurate for this figure
(no panel here uses a "did not reach threshold" hollow marker — the open triangle marks the
best/lowest checkpoint before BPB rose again, a different concept); replaced with an accurate
sentence. Added y-axis headroom so the deliberately-clipped early-training curves no longer
run flush against the panel titles.
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
   training data); the open-triangle marker is the best checkpoint reached before that
   reversal, reported alongside the final value.
5. This is a "does not reproduce" result about these two architecture families at these
   sizes, not a general claim that modern architectures underperform GPT-2.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 4. `data_replication.{png,pdf}` — corpus effect across architectures
**Title:** "Neutral-corpus BPB vs. GPU-hours by training corpus."
Script: `analysis/plot_training_curves_ab.py::expb_data_replication`.
Data: `results/b1_metrics/` (OWT) + `results/data_ladder_metrics/`. Eval: **wiki_eval_union**.
Two panels, one per architecture (Pythia-160M (GPT-NeoX), SmolLM2-135M (Llama)); within each,
the same architecture trained on four corpora vs. its fixed size-matched GPT-2-OWT bar. No
floating prose in the plotting area — confirmed clean by direct pixel inspection both before
and after this pass; only capitalization changed ("Training corpus" legend title, "Matched
GPT-2-OWT bar" annotation).
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
**Title:** "Relative BPB efficiency and CORE performance by model scale."
Script: `analysis/plot_core_disagreement.py`.
Data: `results/core_expb_summary.json` + `results/b1_results.json`.
Two panels, both oriented so **up = candidate beats its size-matched GPT-2**: left "Compute
efficiency (neutral-corpus BPB)", right "Downstream capability (CORE, 11-task mean
accuracy)". X-axis reads "Model scale (matched Pythia / SmolLM2 pairs)" with two-line tick
labels naming both families' real, different sizes at each tier (e.g. "Pythia 160M /
SmolLM2 135M") — never a shared "small/medium/large" or merged "≈135–160M" bucket that would
imply the two ladders are the same scale.
**Fixed this pass (real bug, not just wording):** the rotated y-axis labels were long enough
that they ran past the axes into the bottom caption, printing directly on top of it. Shortened
both labels, gave the figure more height and margin, and — since raising the legend to make
room for that surfaced a *second* collision (the legend row landed on top of the suptitle) —
fixed that too in the same edit (suptitle pinned to `y=0.985`, legend dropped to
`bbox_to_anchor y=0.90`). Also removed the floating "candidate better ↑" / "worse ↓" corner
annotations on both panels: the same information is now stated once, compactly, in each
y-axis label ("↑ = candidate better").
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
124M GPT-2 baseline scale, current training recipe").
Script: `analysis/plot_method_factorial.py`.
Data: `results/small/ceg_newdef.json` + `results/small/small_a0d0_dense_metrics.csv`.
Panel A · "The primitive — compute to reach the threshold": one training curve, the threshold
line, and the crossing point. Panel B · "The 2×2 factorial and its log-space Shapley split":
the four arms as boxes (old/new algorithm × old/new data), GPU-hours-ratio edges (data edges
blue, algorithm edges orange), and the Shapley summary (geometric mean of each intervention's
two edges). Refers to "the 2×2 factorial" / "factorial experiment" throughout — never "core
2×2" — since this is one worked example of the same design used everywhere, not a larger or
more privileged experiment.
**Changed this pass:** capitalized every in-panel label/annotation for consistency (was a mix
of sentence-case and lowercase small captions); glossed "wiki_eval" inline in the caption the
same way `corpus_intervention.png` glosses "wiki_eval_union".
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
3. No new numerical inconsistency was found in this pass or the prior one — every pass to date
   has audited titles, labels, legends, annotation text, and rendering-layout bugs (three found
   and fixed this session, listed above; one — censored markers plotted at 1× — found and
   fixed in the prior session), never the underlying analysis. No CEG/BPB/CORE number has been
   touched by any figure-quality pass.

## Materially-changed findings
None. Every CEG/BPB/CORE number is unchanged across both figure-quality passes. Both passes
combined are titles, axis labels, legends, annotation wording, tick-label granularity (real
per-family sizes instead of shared buckets), and rendering-layout fixes (overlap/clipping)
only — see the changelog sections above for the exhaustive list.

## Superseded / appendix
- Superseded figures **live in `results/superseded/`** (retained for provenance, not
  deleted): `era_ladder.png`, `era_curves.png`, `expb_data_curves.png`, `data_ladder_expB.png`,
  `core_expb_delta.png` — see `results/superseded/README.md` for the replacement mapping.
- Appendix-only (kept at top level, not in the blog's/paper's main flow): `all_configs.png`,
  per-size `curves.png`/`sensitivity.png`, `core_*_by_task*.png`, `core_vs_scale.png`,
  `core_arms_by_task.png`, `core_era_ladder.png`, `core_era_by_task.png`.

## Verification performed this pass
- All five regeneration commands run to completion with no errors or warnings, from
  unmodified in-repo data (`git status` confirms zero `results/*.json` or `metrics.csv`
  changes — only the five plotting scripts and the six regenerated image pairs).
- Every one of the six PNGs viewed at full resolution; every legend/annotation cluster
  additionally viewed at 2–3× pixel crops to check for sub-pixel-scale overlap that isn't
  visible at thumbnail size (this is how bugs 1–3 above were actually caught — none were
  visible in a first full-figure look).
- All three bugs re-verified fixed by re-rendering and re-cropping the same regions after
  the code edit.
- All six PDFs confirmed to have a valid `%PDF-1.4` header and `%%EOF` trailer; not
  separately rasterized (no `pdftoppm`/`pymupdf` available in this container) — see the
  note at the top of this file on why the PNG inspection also covers PDF layout.
- `report.md` regenerated from `make_report.py`; diff is exactly the two alt-text updates
  described above, confirming no other drift between the report and the figures.
