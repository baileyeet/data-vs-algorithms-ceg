# Superseded artifacts

This directory retains earlier renderings and result files for **provenance and history
only**. Nothing here should be used for current reporting — each item has been replaced by
a current canonical artifact. Git history additionally preserves all prior versions.

## Superseded figures (moved 2026-09-01)

| Superseded figure | Replaced by (canonical) | Why |
|---|---|---|
| `era_ladder.png` | `results/corpus_intervention.png` | plotted non-comparable quantities on one axis |
| `era_curves.png` | `results/corpus_intervention.png` (panel A) | folded into the combined corpus figure |
| `expb_data_curves.png` | `results/data_replication.png` | folded into the combined data-replication figure |
| `data_ladder_expB.png` | `results/data_replication.png` | superseded bar rendering of the same result |
| `core_expb_delta.png` | `results/core_bpb_vs_downstream.png` | replaced by the single BPB-vs-downstream CORE figure |

The current canonical publication figures (PNG + vector PDF) live at the top level of
`results/`; see `results/FIGURE_NOTES.md` for their captions and per-figure data provenance.

## Other superseded data files

The `*_v1_*`, `*_olddef*`, `*_legacy_*`, and `*_duplicate_*` metrics/config/JSON files here are
pre-correction run artifacts (e.g. v1 A1 arms before the yarn_state loader-fidelity reruns, and
old threshold-definition CEG JSONs). The canonical metrics and CEG JSONs are the current versions
under `results/` (and its per-size subdirectories).
