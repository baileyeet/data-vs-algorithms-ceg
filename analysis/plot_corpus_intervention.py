"""results/corpus_intervention.{png,pdf} — Exp A corpus figure (replaces the old
era_ladder.png / era_curves.png construction, which plotted non-comparable quantities).

Three panels, one story — swapping the TRAINING CORPUS (holding model size fixed at the
124M GPT-2 baseline dimensions) and its effect on compute-to-threshold:

  A. Evidence: neutral-BPB vs GPU-hours for all four corpora under both recipes
     (old-algo GPT-2 = solid, current-arch modded = dashed). An arm that never reaches
     the dashed threshold is CENSORED — its compute-to-threshold is a bound, not a value.
  B. Corpus CEG vs the reference (old-algo · OWT) baseline — the total compute reduction
     each corpus buys, under each recipe.
  C. Within-recipe data lever — hold the recipe fixed and swap OWT -> corpus; the
     compute-equivalent gain attributable to the corpus alone.

Color follows the CORPUS entity in every panel (OWT gray, C4 amber, RefinedWeb blue,
DCLM green). Recipe is a secondary encoding: linestyle in A, marker shape in B/C.

Data (all in-repo, no backup dependency):
  results/era_orig_metrics/<run>/metrics.csv  (the ORIGINAL era runs the JSON was
      computed from; the era_retrain_metrics/ CSVs are fresh CORE-recovery reruns that
      differ by same-seed noise and are NOT used here — see the figure audit note).
  results/era_ladder_results.json             (CEG values, threshold, censoring).
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, _style, _savefig

ROOT = Path(__file__).resolve().parent.parent / "results"
CORP_COLOR = {"OWT": "#8a8a86", "C4": "#eda100", "RefinedWeb": "#2a78d6", "DCLM": "#008300"}
# (corpus, year, old-algo run, current-arch run)
CORPORA = [
    ("OWT", 2019, "era_a0d0_owt", "era_a1d0_owt"),
    ("C4", 2020, "era_a0_c4", "era_a1_c4"),
    ("RefinedWeb", 2023, "era_a0_refinedweb", "era_a1_refinedweb"),
    ("DCLM", 2024, "era_a0d1_dclm", "era_a1d1_dclm"),
]
J = json.load(open(ROOT / "era_ladder_results.json"))
THR = J["corrected_threshold_neutral_bpb"]


def curve(run):
    p = ROOT / "era_orig_metrics" / run / "metrics.csv"
    rows = list(csv.DictReader(open(p)))
    return [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
            if float(r["gpu_hours"]) > 0 and r["neutral_bpb"]]


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15.5, 5.0), dpi=150)

# ---------------- Panel A: evidence curves ----------------
lo = THR
for corpus, year, old, new in CORPORA:
    col = CORP_COLOR[corpus]
    for run, ls in ((old, "-"), (new, (0, (4, 2)))):
        pts = curve(run)
        if not pts:
            continue
        hs, bs = zip(*pts)
        lo = min(lo, min(bs))
        ax_crossed = min(bs) <= THR
        axA.plot(hs, bs, color=col, lw=1.9, ls=ls,
                 marker="o", ms=2.6, alpha=0.95)
axA.axhline(THR, color=INK2, lw=1.2, ls=(0, (4, 3)))
axA.annotate(f"Reference threshold {THR:.3f}\n(old GPT-2 recipe · OWT)", xy=(0.015, THR),
             xycoords=("axes fraction", "data"), xytext=(0, 4),
             textcoords="offset points", fontsize=8, color=INK2)
axA.set_xscale("log")
axA.set_ylim(lo - 0.03, THR + 0.30)
axA.set_xlabel("GPU-hours (log scale)")
axA.set_ylabel("Neutral-corpus BPB (bits/byte, lower = better)")
axA.set_title("A · Compute to reach the neutral-BPB threshold", fontsize=10.5, loc="left")
_style(axA)
# no in-axes legend here: panel A's curves fill the whole plotting area at every corner
# (see FIGURE_NOTES.md bug list), so color/style are explained once, for the whole figure,
# in the two legend rows above all three panels — not repeated in front of the data.


# ---------------- Panels B & C: multiplier dot plots ----------------
CENSORED_Y = 0.80  # fixed below-parity slot for "did not reach threshold" markers — never
# plotted ON the 1x line itself, so a censored comparison can never be misread as a
# measured 1x (no-advantage) value.


def dotpanel(ax, getval, title, ylab):
    """getval(dataset_dict, recipe) -> (value or None). recipe in {'old','current'}."""
    xs = list(range(len(CORPORA)))
    ax.axhline(1.0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("1×", xy=(0.985, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, -3), textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=INK2)
    for i, (corpus, year, *_ ) in enumerate(CORPORA):
        d = J["datasets"][corpus]
        col = CORP_COLOR[corpus]
        for recipe, dx, mk in (("old", -0.13, "o"), ("current", 0.13, "s")):
            v = getval(d, recipe)
            if v is None:  # did not reach threshold -> hollow marker BELOW parity, never on it
                # (definition: "hollow, below 1x = did not reach threshold" — shared legend above)
                # a plain hollow marker only — an earlier version overlaid a small arrow glyph
                # on top of it, which merged into an unrecognizable blob at this marker size
                # (see FIGURE_NOTES.md bug list); the shared legend explains the convention.
                ax.scatter([i + dx], [CENSORED_Y], marker=mk, s=70, facecolors="none",
                           edgecolors=col, linewidths=1.7, zorder=4)
            else:
                ax.scatter([i + dx], [v], marker=mk, s=70, facecolors=col,
                           edgecolors=SURFACE, linewidths=1.0, zorder=4)
                # nudge left/right (not just up) so the old/current labels never collide
                # when both recipes land on the same value (e.g. the OWT reference, 1.0x)
                ha = "right" if recipe == "old" else "left"
                ax.annotate(f"{v:.1f}×", xy=(i + dx, v), xytext=(-3 if recipe == "old" else 3, 8),
                            textcoords="offset points", ha=ha, fontsize=8, color=INK)
    ax.set_yscale("log")
    ax.set_ylim(0.62, 60)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{y}" for c, y, *_ in CORPORA], fontsize=8.5)
    ax.set_xlim(-0.5, len(CORPORA) - 0.5)
    ax.set_xlabel("Training corpus")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5, loc="left")
    _style(ax)


dotpanel(axB, lambda d, r: d["old_algo" if r == "old" else "current_arch"]["ceg_vs_a0d0"],
         "B · Corpus CEG vs. old GPT-2 recipe · OWT reference",
         "Compute-equivalent gain  (×, log scale)")
dotpanel(axC, lambda d, r: d["data_ceg_old_algo" if r == "old" else "data_ceg_current_arch"],
         "C · Within-recipe corpus CEG (OWT → corpus)",
         "Within-recipe corpus CEG  (×, log scale)")

# ONE consolidated legend system for the whole figure (two rows), replacing the three
# separate per-panel legends this figure used to carry (two inside panel A, overlapping
# the curves there, plus a third for B/C) — every color/style is defined exactly once.
# Row 1: corpus color. Row 2: recipe (line style in A, marker shape in B/C, shown together
# on one handle) + the censored-marker convention.
corpus_handles = [Line2D([], [], color=CORP_COLOR[c], lw=2.4, label=f"{c} ({y})")
                  for c, y, *_ in CORPORA]
fig.legend(handles=corpus_handles, frameon=False, fontsize=8.5, labelcolor=INK2,
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955), title="Training corpus",
           title_fontsize=8.5)
recipe_handles = [
    Line2D([], [], color=INK2, lw=1.9, ls="-", marker="o", ms=7, label="Old GPT-2 recipe"),
    Line2D([], [], color=INK2, lw=1.9, ls=(0, (4, 2)), marker="s", ms=7,
           label="Current training recipe"),
    Line2D([], [], color=INK2, lw=0, marker="o", ms=8, markerfacecolor="none",
           label="Hollow, below 1× = did not reach threshold"),
]
fig.legend(handles=recipe_handles, frameon=False, fontsize=8, labelcolor=INK2,
           loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.895))

fig.suptitle("Corpus compute-equivalent gain at the 124M GPT-2 baseline scale",
             fontsize=12.5, x=0.008, ha="left", color=INK, y=0.995)
fig.text(0.008, 0.008,
         "Held-out, decontaminated Wikipedia evaluation set (“wiki_eval_union”). Panel C holds the "
         "recipe fixed and swaps only the training corpus (OWT → corpus); the two recipes give different "
         "corpus multipliers on RefinedWeb/DCLM (old GPT-2 recipe ~3.3–3.5× vs current training "
         "recipe ~1.6×) — see caption for interpretation.",
         fontsize=7, color=INK2, va="bottom", wrap=True)
fig.tight_layout(rect=(0, 0.07, 1, 0.86))
_savefig(fig, ROOT / "corpus_intervention.png")
