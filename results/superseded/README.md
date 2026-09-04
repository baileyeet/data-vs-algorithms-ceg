# Superseded artifacts

This directory retains earlier renderings and result files for **provenance and history
only**. Nothing here should be used for current reporting — each item has been replaced by
a current canonical artifact. Git history additionally preserves all prior versions.

## Superseded figures (moved 2026-09-01)

| Superseded figure | Replaced by (canonical) | Why |
|---|---|---|
| `era_ladder.png` | `results/corpus_bpb_curves.png` et al. | plotted non-comparable quantities on one axis |
| `era_curves.png` | `results/corpus_bpb_curves.png` | folded into the corpus evidence figure |
| `expb_data_curves.png` | `results/data_replication.png` | folded into the combined data-replication figure |
| `data_ladder_expB.png` | `results/data_replication.png` | superseded bar rendering of the same result |
| `core_expb_delta.png` | `results/core_bpb_vs_downstream.png` | replaced by the single BPB-vs-downstream CORE figure |

## Superseded figures (moved 2026-09-04 — split into standalone images)

The two remaining 3-panel and 2-panel composite figures were split into one image per panel
at the user's request, so each panel sizes and scales independently in a paper (each is now
also a self-contained image with its own title, axes, and legend, rather than depending on
neighboring panels or a figure-level shared legend).

| Superseded figure | Replaced by (canonical, one file per former panel) |
|---|---|
| `corpus_intervention.png` (3 panels: A, B, C) | `results/corpus_bpb_curves.png` (A), `results/corpus_ceg_total.png` (B), `results/corpus_ceg_within_recipe.png` (C) |
| `method_factorial.png` (2 panels: A, B) | `results/method_primitive.png` (A), `results/method_shapley_split.png` (B) |

No analysis numbers changed in the split — same in-repo data, same values, same conventions;
only the layout (one figure per panel instead of one figure with several panels) and, per
user feedback on the combined versions, several panel titles were also clarified (`B`'s
"vs. old GPT-2 recipe · OWT reference" shortened to "vs. OWT reference"; `C`'s title now
states explicitly that the training recipe/algorithm is held fixed, since a reader asked
whether it was showing an algorithm-CEG number).

The current canonical publication figures (PNG + vector PDF) live at the top level of
`results/`; see `results/FIGURE_NOTES.md` for their captions and per-figure data provenance.

## Other superseded data files

The `*_v1_*`, `*_olddef*`, `*_legacy_*`, and `*_duplicate_*` metrics/config/JSON files here are
pre-correction run artifacts (e.g. v1 A1 arms before the yarn_state loader-fidelity reruns, and
old threshold-definition CEG JSONs). The canonical metrics and CEG JSONs are the current versions
under `results/` (and its per-size subdirectories).
