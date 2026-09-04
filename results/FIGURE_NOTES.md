# Publication figure set — captions, provenance, and changelog

Prepared for reuse in both the MIRI Technical Governance blog post and an academic paper.
Every figure below is rendered to **PNG + vector PDF** (same basename) and is fully
reproducible from in-repo data only (no `~/Desktop/era_ladder_backup/` dependency). No
underlying analysis number was changed for presentation — every pass to date (this one
included) is titles, labels, legends, and in-plot text only.

**Regeneration commands** (all read only in-repo `results/` data):
```
python analysis/plots.py multipliers --out results/multipliers_vs_scale.png
python analysis/plot_corpus_intervention.py      # writes corpus_bpb_curves.png +
                                                  #   corpus_ceg_total.png + corpus_ceg_within_recipe.png
python analysis/plot_training_curves_ab.py       # writes expb_arch_curves.png + data_replication.png
python analysis/plot_core_disagreement.py
python analysis/plot_method_factorial.py         # writes method_primitive.png + method_shapley_split.png
```
Requires `matplotlib` (see `requirements.txt`, pinned `requirements-lock.txt`).

**Figure count: nine, not six.** The set started as six figures, two of them multi-panel
(3-panel `corpus_intervention.png`, 2-panel `method_factorial.png`). Per user request (2026-
09-04, so each figure sizes and scales independently in a paper), both were split into one
standalone image per former panel — see the "Figure-split pass" section below. The old
combined files are retired to `results/superseded/`.

**Status of this pass (2026-09-04):** a prior session (`bda2398`, merged as PR #1) wrote the
source edits for a first figure-quality pass but could not install `matplotlib` in its
sandbox, so it never rendered or visually inspected the actual PNG/PDF output. This session
had a working `matplotlib` + `numpy`, so it went through **four rounds**, each ending with a
re-render and a full re-inspection: (1) an initial render + full-resolution visual audit that
caught 3 rendering bugs; (2) an academic-audience copy pass (dropped legend jargon, spelled
out abbreviations, fixed capitalization, real per-family size labels instead of shared
buckets); (3) a second, closer visual audit — prompted by the user spotting problems in the
rendered PNGs sent back to them — that caught 4 more bugs invisible at thumbnail size (a
legend sitting on top of data, a marker rendering as an unrecognizable blob, inconsistent
axis wording, a color-coding collision in the method schematic), plus a further minimalism
pass dropping redundant "(log scale)"/units text and cramped legend spacing; (4) a third
audit — again prompted by the user looking closely at the round-3 renders — that caught a
marker-fill/legend mismatch, a hard-to-see hollow marker, and two confusing titles, and
carried out the user's request to split the two multi-panel figures into standalone images
(see "Figure-split pass" below). Every round's bugs are listed below; no PDF rasterizer
(`pdftoppm`/`pymupdf`) is available in this container, but the PDFs are written by the same
`fig.savefig()` call on the same already-laid-out Matplotlib `Figure` object as the PNG (see
`plots.py::_savefig`), so element positions are identical between the two — only
antialiasing/font-hinting differs. All PDFs were checked for a valid `%PDF-1.4` header and
`%%EOF` trailer.

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
4. **`corpus_intervention.png`, panel A: the in-axes legend sat on top of the data.** Panel
   A's two legends ("Corpus" upper-right, "Recipe" lower-left, both `frameon=False`) were
   placed *inside* the axes, and the four corpus curves fill the plotting area densely enough
   (from top-left down to bottom-right) that all four lines ran straight through the
   transparent legend box — RefinedWeb, DCLM, and part of OWT/C4 crossed directly behind the
   "Corpus" title and every corpus name. This was caught only by cropping the region at 1.8×;
   at full-figure thumbnail size the legend just looked slightly busy, not obviously broken.
   **Fix:** removed both in-axes legends from panel A entirely and replaced all three of this
   figure's legends (two in panel A, one shared for B/C) with **one** consolidated two-row
   legend above all three panels — row 1 is corpus color, row 2 is recipe (line style *and*
   marker shape on one handle, since panel A encodes recipe as line style while B/C encode it
   as marker shape) plus the censored-marker convention. Every color/style is now defined
   exactly once for the whole figure, with zero chance of sitting behind a data line.
5. **`corpus_intervention.png`, panels B/C: the censored marker read as an unrecognizable
   blob, not a hollow circle/square.** The hollow marker for a censored (never-crossed)
   comparison had a short arrow glyph (`arrowprops`) drawn directly on top of it to suggest
   "bounded from above." At the marker's actual render size the arrowhead and the hollow
   circle/square outline merged into a shape that doesn't read as either — flagged by the user
   as looking like "random figures." **Fix:** removed the arrow overlay; the marker is now a
   plain hollow circle/square, matching the filled marker used for a real measured value
   (filled = measured, hollow = censored), with the convention stated once in the shared
   legend.
6. **`corpus_intervention.png` panel A and `method_factorial.png` panel A used different
   wording for the same two axes.** "GPU-hours (log)" vs. "GPU-hours (log scale)" (used by
   `expb_arch_curves.png`/`data_replication.png`), and "Neutral-corpus BPB (lower = better)"
   (no unit) vs. "Neutral-corpus BPB (bits/byte, lower = better)" — four figures that are all
   "BPB vs. GPU-hours" plots read as though they used different conventions. **Fix:**
   standardized both axes, on both figures, to match the wording already used by
   `expb_arch_curves.png`/`data_replication.png`: **"GPU-hours (log scale)"** /
   **"Neutral-corpus BPB (bits/byte, lower = better)"**.
7. **`method_factorial.png` panel B: two unrelated things were both color-coded blue/orange,
   reading as two overlaid diagrams.** The four arm boxes were outlined and labeled in
   `ARM_COLORS` (per-arm identity colors used for cross-figure continuity elsewhere in the
   study — a0d0 is blue, a1d0 is amber) so that they'd keep their hue across figures; the
   panel's edges are separately colored blue (data-swap edges) and orange (algorithm-swap
   edges) to carry the panel's actual message. Blue happened to label both "the A0D0 box" and
   "a data edge"; amber happened to label both "the A1D0 box" and (closely) "an algorithm
   edge" — two independent color legends sharing the same two hues, which is what read as
   confusing/superimposed. **Fix:** the four boxes are now a single neutral color (gray
   border, black bold text) — arm identity is carried by the "A0·D0"-style label text alone,
   not by color, inside this panel. Color is now used for exactly one thing: data edges are
   blue, algorithm edges are orange, full stop. Also removed a no-op
   `labels[arm] if False else labels[arm]` leftover from an earlier edit.

## Minimalism pass (2026-09-04, user-requested — trust the reader more)

A third round, after the second round's fixes were sent back to the user: two more real
layout issues, plus a push to cut in-image text the reader doesn't need repeated on every
axis once it's established once (in the caption, in the paper's methods section, or by strong
convention).

- **`corpus_intervention.png`: the two legend rows (corpus color, recipe style) sat almost on
  top of each other** — only ~0.06 of the figure height apart, so "OWT (2019)" and "Old GPT-2
  recipe" on the row below it read as one merged block rather than two legend groups. Widened
  the gap (now ~0.13), added `columnspacing`/`handletextpad` to both rows, and grew the figure
  height (5.0 → 5.4in) to make room without shrinking the three panels.
- **Removed "(log scale)" / "(× , log scale)" from every axis label that has it**
  (`multipliers_vs_scale.png`, `corpus_intervention.png` all three panels,
  `expb_arch_curves.png`, `data_replication.png`, `method_factorial.png` panel A) — the log
  nature of every one of these axes is already visually obvious from the power-of-ten tick
  labels (10⁰, 10¹, ...); stating it a second time in the axis text was one more thing for an
  academic reader to read past for no new information.
- **Removed "(bits/byte, lower = better)" from every "Neutral-corpus BPB" y-axis label** —
  BPB is a standard, already-defined metric for this readership; the axis now just reads
  "Neutral-corpus BPB". (`core_bpb_vs_downstream.png`'s y-axis labels are untouched — those
  are *delta* quantities, "BPB advantage over matched GPT-2 (bits; ...)", where the unit is
  new information, not a restatement of the metric's definition.)
- **Shortened three long two-line in-plot annotations to a bare number, matching the "1×"
  convention already used elsewhere**: `corpus_intervention.png` panel A's "Reference
  threshold 1.276\n(old GPT-2 recipe · OWT)" → "1.276"; `method_factorial.png` panel A's
  "Neutral-BPB threshold 1.274" → "1.274"; `expb_arch_curves.png`/`data_replication.png`'s
  per-panel "GPT-2 bar 1.252" / "Matched GPT-2-OWT bar 1.252" → "1.252". The full description
  of what each reference line is now lives once, in the caption below, rather than repeated
  in-image at every one of the ~14 places these threshold lines appear across the figure set.
- **`method_factorial.png`: moved the "Worked example: 124M GPT-2 baseline scale, current
  training recipe" subtitle out of the image entirely.** It's in the caption (below, and in
  `report.md`) only now; the image carries just the main title, freeing up ~6% more vertical
  room for panels A/B.
- **`method_factorial.png` panel A's "Example arm: old algorithm · old data (the baseline)"**
  → "A0·D0 (old algorithm · old data)", matching panel B's own arm-box labeling convention
  instead of using different phrasing for the same identification.

## Figure-split pass (2026-09-04, user-requested)

`corpus_intervention.png` (3 panels) and `method_factorial.png` (2 panels) were split into
five standalone figures — one PNG+PDF pair per former panel, each with its own self-contained
title, axes, and legend (previously legends were shared figure-wide, which doesn't work once
panels are separate files). Old combined files moved to `results/superseded/` (see its
README for the full mapping). This is a **layout change only** — same in-repo data, same
values.

While splitting, three more issues found by close inspection were fixed at the same time
(not new panels, just things easier to see once each panel had its own full-size render):

- **`corpus_intervention` panel B's title was confusing.** "B · Corpus CEG vs. old GPT-2
  recipe · OWT reference" packs three ideas into one compound noun phrase. The recipe
  qualifier is redundant with the marker-shape legend directly below it, so the new
  standalone title drops it: **"Corpus compute-equivalent gain vs. OWT reference."**
- **`corpus_intervention` panel C's title didn't say what it measures.** A user reading "C ·
  Within-recipe corpus CEG (OWT → corpus)" asked "is this algorithm gain or something?" — a
  fair question, since the figure never states that the algorithm/recipe is held fixed here.
  New title says so explicitly: **"Corpus-only compute-equivalent gain, training recipe held
  fixed,"** and the caption repeats it ("no algorithm effect mixed in").
- **`corpus_intervention` panel A's recipe legend didn't visibly show line style.** Flagged as
  "the dottedness isn't labeled": the legend's line-icon segments were too short at the
  default handle length to show more than one dash, so solid vs. dashed didn't read clearly
  as two different styles. Fixed with `handlelength=3.6` on that legend only, giving each
  icon enough length to show 2–3 dash repeats.

Two further bugs, unrelated to the split, were also fixed this round:

- **`expb_arch_curves.png`: the open-triangle "best checkpoint" marker was hard to read as
  hollow.** At its original size (`s=55`, no backing) the ring competed visually with the
  candidate line passing directly through the same point. Enlarged it (`s=170`) and added a
  filled surface-colored disc behind it, giving the hollow ring a clean "cutout" against the
  line so it unambiguously reads as an open (not filled) marker.
- **`data_replication.png`: censored corpora's curve markers were filled solid, contradicting
  their own hollow legend entry.** The legend already showed "OWT (censored)" / "C4
  (censored)" with a hollow marker icon, but the actual plotted line for those corpora used
  filled dots throughout — the same convention mismatch flagged by the user ("the OWT and C4
  circles look like they should be unfilled but they are filled"). Fixed: a corpus's curve
  now renders with hollow per-point markers whenever it's censored, filled when it crosses,
  matching its own legend entry exactly.
- **`expb_arch_curves.png`'s title didn't match `data_replication.png`'s**, despite both
  being "BPB vs. GPU-hours" figures on the same recipe: "Bits per byte vs. GPU-hours for
  matched architecture comparisons" vs. "Neutral-corpus BPB vs. GPU-hours by training
  corpus." Unified to **"Neutral-corpus BPB vs. GPU-hours for matched architecture
  comparisons,"** following the user's suggested template ("neutral-corpus BPB vs GPU-hours
  for/by \<thing\>").
- **`core_bpb_vs_downstream.png`: dropped the parenthetical unit/orientation text from every
  axis label** ("(bits; ↑ = candidate better)", "(↑ = candidate better)", "(matched Pythia /
  SmolLM2 pairs)") — the ↑-convention is already stated once in the caption, and the
  "matched Pythia / SmolLM2 pairs" detail is redundant with the two-line tick labels, which
  already name both families explicitly.

## Open question raised by the user, not yet acted on
**Should the appendix-only CORE figures for the original 2×2 study (not just Exp B) also be
included?** The user specifically asked about CORE for "everything we ran" — confirmed by
direct inspection, not assumption: `core_vs_scale.png` and `core_arms_by_task.png` cover the
**original 2×2 study** (current-arch, ScaleUp; the runs the study's headline result is built
on), separately from `core_era_ladder.png`/`core_era_by_task.png` (Exp A era ladder) and
`core_expb_by_task.png`/`core_expb_by_task_abs.png` (Exp B, Pythia/SmolLM2). So all three
experiments already have their own CORE figure(s) — nothing is actually missing.
Recommendation: keep them as **three separate appendix figures**, not one merged "CORE
everywhere" figure — the three cover different x-axes (model size within one family vs.
matched pairs across two families), different task sets in places, and different color
schemes tied to each experiment's own arms; forcing them into one figure would likely
re-introduce the kind of over-encoding this whole pass has been removing. `core_vs_scale.png`
(the original-study one) is reasonably close to publication-ready already but has its own
minor bug: the "chance 0.50" label sits directly on the boolq curve near 124M. None of the
six appendix CORE figures have had the terminology pass applied yet (`core_expb_by_task_abs.png`
still titles itself "Exp B ..." and uses merged "150M / 500M / 1.5B" size buckets, the same
issue fixed in the main figures). Flagging rather than fixing now — say the word and this can
be done the same way as the rest of this pass.

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

## Terminology used consistently across all nine figures
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

## The nine canonical figures

(Figures 2 and 6 are each a former multi-panel figure split into standalone images — 2a/2b/2c
and 6a/6b — per the "Figure-split pass" section above; the numbering follows the original
six-figure order so cross-references from earlier in this document still resolve.)

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

### 2a. `corpus_bpb_curves.{png,pdf}` — corpus evidence curves
**Title:** "Neutral-corpus BPB vs. GPU-hours by training corpus and recipe."
Script: `analysis/plot_corpus_intervention.py::fig_bpb_curves`.
Data: `results/era_orig_metrics/*/metrics.csv` (the original era runs) +
`results/era_ladder_results.json`. Eval: **wiki_eval_union** (held-out, decontaminated
Wikipedia set; glossed inline in the figure's own caption).
The underlying evidence: raw BPB-vs-GPU-hours curves for all four corpora (OWT gray, C4
amber, RefinedWeb blue, DCLM green) under both recipes (old GPT-2 recipe = solid line,
current training recipe = dashed). Threshold line tagged with its bare value (1.276); full
description ("old GPT-2 recipe · OWT") lives in the caption, not the image.
**Was panel A of the retired `corpus_intervention.png`** (see `results/superseded/README.md`).
Caption:
1. Neutral-corpus BPB vs. GPU-hours, all four training corpora under both the old GPT-2
   recipe and the current training recipe, at the 124M GPT-2 baseline scale.
2. Dashed threshold line (1.276) = the old-recipe·OWT arm's converged BPB; an arm that never
   reaches it is censored (its compute-to-threshold is a bound, not a value — see Figs. 2b/2c).
3. C4 never reaches the threshold under either recipe; RefinedWeb and DCLM cross under both.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 2b. `corpus_ceg_total.{png,pdf}` — total corpus compute-equivalent gain
**Title:** "Corpus compute-equivalent gain vs. OWT reference."
Script: `analysis/plot_corpus_intervention.py::fig_ceg_total`.
Same data/eval as 2a. Total compute-equivalent gain each corpus buys vs. the fixed
old-recipe·OWT reference arm (1×, by definition), under each recipe separately. Marker shape
= recipe (circle = old GPT-2, square = current training); hollow markers below the 1× line
did not reach the threshold. **Title change:** dropped "old GPT-2 recipe" as a compound
modifier (a reader found the original three-part title — "vs. old GPT-2 recipe · OWT
reference" — confusing; the recipe distinction is already carried by the marker-shape legend
directly below the title).
**Was panel B of the retired `corpus_intervention.png`.**
Caption:
1. Each corpus's total compute-equivalent gain vs. the old-recipe·OWT reference arm, at the
   124M GPT-2 baseline scale, shown separately for both recipes.
2. C4 is censored (hollow, below 1×) under both recipes; RefinedWeb and DCLM cross strongly
   under both, more so under the current training recipe (37.9×/38.5×) than the old GPT-2
   recipe (3.3×/3.5×) — the training-recipe intervention amplifies the corpus effect here.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 2c. `corpus_ceg_within_recipe.{png,pdf}` — corpus-only compute-equivalent gain
**Title:** "Corpus-only compute-equivalent gain, training recipe held fixed."
Script: `analysis/plot_corpus_intervention.py::fig_ceg_within_recipe`.
Same data/eval as 2a. Hold the recipe fixed and swap only the corpus (OWT → the labeled
corpus); isolates the corpus's own contribution, no algorithm/recipe effect mixed in.
**Title change:** the original title ("Within-recipe corpus CEG (OWT → corpus)") didn't say
what's held fixed; a reader asked "is this algorithm gain or something?" The new title
states explicitly that the training recipe is held fixed, and the caption repeats it.
**Was panel C of the retired `corpus_intervention.png`.**
Caption:
1. Each point swaps ONLY the training corpus, holding the training recipe/algorithm fixed at
   the value given by its marker shape (circle = old GPT-2, square = current training) — this
   isolates the corpus's own contribution, with no algorithm effect mixed in.
2. RefinedWeb/DCLM give a different corpus multiplier under each recipe (old GPT-2 recipe
   ~3.3–3.5× vs. current training recipe ~1.6×): the corpus and training-recipe interventions
   **interact**, so there is no single recipe-independent "value of the corpus".
3. Corpus quality is **non-monotonic in release date**: C4 (2020) buys nothing over OWT
   (2019, censored) under either recipe, while RefinedWeb (2023) and DCLM (2024) do.
**Recommend:** (a) blog main text — yes; (b) paper main text — yes; (c) appendix only — no.

### 3. `expb_arch_curves.{png,pdf}` — architecture comparisons
**Title:** "Neutral-corpus BPB vs. GPU-hours for matched architecture comparisons" (was "Bits
per byte vs. GPU-hours..." — unified with Fig. 4's title wording, see below).
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
run flush against the panel titles. **Title unified with Fig. 4** ("Bits per byte" →
"Neutral-corpus BPB", matching the user's suggested "neutral-corpus BPB vs GPU-hours
for/by \<thing\>" template — two figures showing the same kind of comparison had different
phrasing). **The open-triangle "best checkpoint" marker was hard to read as hollow** (flagged
by the user): at its original size it competed visually with the candidate line passing
directly through the same point. Enlarged (`s=55`→`170`) with a filled surface-colored disc
behind it, so the hollow ring reads as a clean cutout against the line.
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
GPT-2-OWT bar" annotation). **Fixed a real bug this round:** censored corpora's (OWT, C4)
per-point curve markers were filled solid, contradicting their own hollow legend entry
("OWT (censored)" shown with a hollow marker icon) — flagged by the user ("the OWT and C4
circles look like they should be unfilled but they are filled"). A corpus's curve now renders
with hollow per-point markers whenever it's censored, filled when it crosses, matching its
own legend entry exactly.
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
accuracy)". X-axis reads "Model scale" with two-line tick
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
y-axis label. **A later round trimmed the axis labels further** (dropped "(bits; ↑ =
candidate better)", "(↑ = candidate better)", and "(matched Pythia / SmolLM2 pairs)"
entirely) — the ↑-convention is stated once in the caption, and "matched Pythia / SmolLM2
pairs" was redundant with the tick labels, which already name both families.
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

### 6a. `method_primitive.{png,pdf}` — the CEG primitive
**Title:** "Compute to reach the neutral-BPB threshold."
Script: `analysis/plot_method_factorial.py::fig_primitive`.
Data: `results/small/ceg_newdef.json` + `results/small/small_a0d0_dense_metrics.csv`.
One training curve, the threshold line (tagged with its bare value, 1.274), and the crossing
point — the primitive behind every compute-equivalent gain in this study: train to a fixed
neutral-BPB threshold, read off the GPU-hours it took. Arm shown: A0·D0 (old algorithm · old
data, the study's baseline). **The "Worked example: 124M GPT-2 baseline scale, current
training recipe" subtitle that used to run across the top of the combined figure now lives
only in the caption** (and in `report.md`'s surrounding text) — not repeated in the image.
**Was panel A of the retired `method_factorial.png`.**
Caption:
1. The primitive behind every compute-equivalent gain: train to a fixed neutral-BPB
   threshold and read off the GPU-hours it took (fewer = more compute-efficient). Worked
   example at the 124M GPT-2 baseline scale, current training recipe, old-algorithm/old-data
   arm (A0·D0).
2. This arm crosses the (1.274) threshold at 4.87 GPU-hours — the number that feeds every
   ratio in Fig. 6b.
**Recommend:** (a) blog main text — yes, near the top, before the results; (b) paper main
text — optional, works well as a Methods-section figure; (c) appendix — acceptable
alternative to (b) if space is tight.

### 6b. `method_shapley_split.{png,pdf}` — the 2×2 factorial and Shapley split
**Title:** "The 2×2 factorial and its log-space Shapley split."
Script: `analysis/plot_method_factorial.py::fig_shapley_split`.
Same data as 6a. The four arms as neutral-gray boxes (old/new algorithm × old/new data,
identified by their "A0·D0"-style text label, not by color), GPU-hours-ratio edges (data
edges blue, algorithm edges orange — the only color used in this figure), and the Shapley
summary (geometric mean of each intervention's two edges). Refers to "the 2×2 factorial" /
"factorial experiment" throughout — never "core 2×2" — since this is one worked example of
the same design used everywhere, not a larger or more privileged experiment.
**Fixed a real bug:** the four boxes used to be outlined/labeled in the same per-arm colors
(`ARM_COLORS`) used for cross-figure continuity elsewhere in the study — which put blue on
the A0D0 box *and* on the data-edge arrows, amber on the A1D0 box *and* near-amber on the
algorithm-edge arrows, two independent color codings sharing hues. Reported by the user as
the panel looking like several images superimposed. Boxes are now neutral gray/black; color
in this figure means exactly one thing (data vs. algorithm edge). **Was panel B of the
retired `method_factorial.png`.**
Caption:
1. How a compute-equivalent gain (CEG) is decomposed, worked at the 124M current-training-
   recipe point (no numbers invented — the multipliers are re-derived from the four arms'
   GPU-hours in Fig. 6a's data).
2. Horizontal edges swap data (hold algorithm fixed), vertical edges swap algorithm (hold
   data fixed); each edge is a GPU-hours ratio.
3. The Shapley multiplier of an intervention is the geometric mean of its two edges (both
   orderings averaged): algorithm 13.69×, data 2.23×, total 30.5×.
4. Intended as an early "how to read the numbers" explainer, not a privileged or larger
   experiment — it is one worked example of the same factorial design used throughout.
**Recommend:** (a) blog main text — yes, near the top, before the results; (b) paper main
text — optional, works well as a Methods-section figure; (c) appendix — acceptable
alternative to (b) if space is tight.

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
  `core_expb_delta.png`, and (as of 2026-09-04) the retired combined `corpus_intervention.png`
  and `method_factorial.png` — see `results/superseded/README.md` for the full replacement
  mapping.
- Appendix-only (kept at top level, not in the blog's/paper's main flow): `all_configs.png`,
  per-size `curves.png`/`sensitivity.png`, `core_*_by_task*.png`, `core_vs_scale.png`,
  `core_arms_by_task.png`, `core_era_ladder.png`, `core_era_by_task.png`. None of these have
  had this figure-quality treatment applied yet — see "Open question raised by the user" above.

## Verification performed this pass
- All five regeneration commands run to completion with no errors or warnings, from
  unmodified in-repo data (`git status` confirms zero `results/*.json` or `metrics.csv`
  changes across all four rounds to date — only plotting scripts and regenerated image files).
- Every one of the (now nine) PNGs viewed at full resolution; every legend/annotation cluster
  additionally viewed at 2–4× pixel crops to check for overlap that isn't visible at
  thumbnail size — this is how the great majority of the bugs listed above were actually
  caught (none were visible in a first full-figure look). Two rounds in a row surfaced real
  bugs only after the user looked closely at the previous round's renders and reported
  specific problems the prior audit had missed (a marker convention mismatch, a hard-to-see
  hollow marker, a confusing title, an unlabeled line style) — confirming the lesson below.
- Every bug re-verified fixed by re-rendering and re-cropping the same region after its
  code edit.
- All PDFs confirmed to have a valid `%PDF-1.4` header and `%%EOF` trailer; not separately
  rasterized (no `pdftoppm`/`pymupdf` available in this container) — see the note at the top
  of this file on why the PNG inspection also covers PDF layout.
- `report.md` regenerated from `make_report.py` after every round; each diff was exactly the
  image-embed/caption text that needed to change for that round, confirming no other drift
  between the report and the figures.
- **Lesson for anyone extending this figure set:** a full-figure look at thumbnail size is
  not sufficient QA — every legend, marker cluster, and multi-line label needs its own
  close crop before a figure is called clean.
