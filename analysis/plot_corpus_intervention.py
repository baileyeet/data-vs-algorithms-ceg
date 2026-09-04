"""results/corpus_bpb_curves.{png,pdf}, results/corpus_ceg_total.{png,pdf},
results/corpus_ceg_within_recipe.{png,pdf} — Exp A corpus figures (three standalone images;
were one 3-panel figure, split per user request so each scales independently in a paper).

Three separate figures, one story — swapping the TRAINING CORPUS (holding model size fixed at
the 124M GPT-2 baseline dimensions) and its effect on compute-to-threshold:

  1. corpus_bpb_curves: the evidence — neutral-BPB vs GPU-hours for all four corpora under
     both recipes (old GPT-2 recipe = solid, current training recipe = dashed). An arm that
     never reaches the dashed threshold is CENSORED — its compute-to-threshold is a bound,
     not a value.
  2. corpus_ceg_total: total compute-equivalent gain each corpus buys vs. the fixed
     old-recipe-OWT reference, under each recipe.
  3. corpus_ceg_within_recipe: hold the recipe fixed and swap OWT -> corpus; the
     compute-equivalent gain attributable to the corpus alone (algorithm/recipe held fixed
     in both bars compared — this is NOT an algorithm-CEG figure).

Color follows the CORPUS entity in every figure (OWT gray, C4 amber, RefinedWeb blue, DCLM
green). Recipe is a secondary encoding: linestyle in figure 1, marker shape in figures 2/3.
Each figure carries its own self-contained legend (previously one legend shared all three
panels; standalone images need their own).

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


def corpus_legend(ax_or_fig, **kw):
    handles = [Line2D([], [], color=CORP_COLOR[c], lw=2.4, label=f"{c} ({y})")
              for c, y, *_ in CORPORA]
    return ax_or_fig.legend(handles=handles, frameon=False, fontsize=8.5, labelcolor=INK2,
                            title="Training corpus", title_fontsize=8.5, **kw)


# ==================== Figure 1: evidence curves (was panel A) ====================
def fig_bpb_curves():
    fig, ax = plt.subplots(figsize=(7.2, 6.1), dpi=150)
    lo = THR
    for corpus, year, old, new in CORPORA:
        col = CORP_COLOR[corpus]
        for run, ls in ((old, "-"), (new, (0, (4, 2)))):
            pts = curve(run)
            if not pts:
                continue
            hs, bs = zip(*pts)
            lo = min(lo, min(bs))
            ax.plot(hs, bs, color=col, lw=1.9, ls=ls, marker="o", ms=2.6, alpha=0.95)
    ax.axhline(THR, color=INK2, lw=1.2, ls=(0, (4, 3)))
    # compact tag on the line itself (matches the "1x" convention on the other figures);
    # the recipe/corpus this threshold is defined on is stated once in the caption, not here
    ax.annotate(f"{THR:.3f}", xy=(0.015, THR), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(lo - 0.03, THR + 0.30)
    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Neutral-corpus BPB")
    _style(ax)

    corpus_legend(fig, loc="upper center", ncol=4, bbox_to_anchor=(0.54, 0.985),
                  columnspacing=2.0, handletextpad=0.7)
    # recipe legend: line style is the encoding here (not marker shape, as in figs 2/3), so
    # give each handle a long dash run (handlelength) - at the default handle length the
    # dash pattern only shows 1-2 dashes and doesn't read clearly as "dashed" vs "solid"
    # (flagged by the user: "the dottedness isn't labeled")
    recipe_handles = [
        Line2D([], [], color=INK2, lw=2.2, ls="-", label="Old GPT-2 recipe"),
        Line2D([], [], color=INK2, lw=2.2, ls=(0, (4, 2)), label="Current training recipe"),
    ]
    fig.legend(handles=recipe_handles, frameon=False, fontsize=8.5, labelcolor=INK2,
               loc="upper center", ncol=2, bbox_to_anchor=(0.54, 0.895),
               handlelength=3.6, columnspacing=2.2)

    fig.suptitle("Neutral-corpus BPB vs. GPU-hours by training corpus and recipe",
                 fontsize=12.5, x=0.012, ha="left", color=INK, y=0.995)
    fig.text(0.012, 0.006,
             "124M GPT-2 baseline scale. Held-out, decontaminated Wikipedia evaluation set "
             "(“wiki_eval_union”). Dashed reference line: the neutral-BPB threshold "
             f"({THR:.3f}), defined by the old GPT-2 recipe trained on OWT.",
             fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.85))
    _savefig(fig, ROOT / "corpus_bpb_curves.png")


# ==================== Figures 2 & 3: multiplier dot plots (were panels B, C) ====================
CENSORED_Y = 0.80  # fixed below-parity slot for "did not reach threshold" markers — never
# plotted ON the 1x line itself, so a censored comparison can never be misread as a
# measured 1x (no-advantage) value.


def dotpanel(ax, getval, ylab):
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
                # (definition: "hollow, below 1x = did not reach threshold" — legend below)
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
    _style(ax)


def recipe_censor_legend(fig, **kw):
    handles = [
        Line2D([], [], color=INK2, marker="o", ls="none", ms=8, label="Old GPT-2 recipe"),
        Line2D([], [], color=INK2, marker="s", ls="none", ms=8, label="Current training recipe"),
        Line2D([], [], color=INK2, marker="o", ls="none", ms=8, markerfacecolor="none",
               label="Hollow, below 1× = did not reach threshold"),
    ]
    return fig.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, **kw)


def fig_ceg_total():
    fig, ax = plt.subplots(figsize=(7.6, 6.1), dpi=150)
    dotpanel(ax, lambda d, r: d["old_algo" if r == "old" else "current_arch"]["ceg_vs_a0d0"],
             "Compute-equivalent gain (×)")
    corpus_legend(fig, loc="upper center", ncol=4, bbox_to_anchor=(0.54, 0.98),
                  columnspacing=2.0, handletextpad=0.7)
    recipe_censor_legend(fig, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 0.89),
                         columnspacing=1.6)
    # title dropped "old GPT-2 recipe" (a mouthful as a compound modifier, and already
    # carried by the marker-shape legend below) after it was flagged as confusing
    fig.suptitle("Corpus compute-equivalent gain vs. OWT reference",
                 fontsize=12.5, x=0.012, ha="left", color=INK, y=0.995)
    fig.text(0.012, 0.006,
             "124M GPT-2 baseline scale. Each corpus's total compute-equivalent gain vs. the "
             "old-recipe·OWT reference arm (1×, by definition), under each recipe separately. "
             "Marker shape = recipe (circle = old GPT-2, square = current training). Hollow "
             "markers below the 1× line did not reach the threshold within their compute "
             "budget — a bound, not a measured value.",
             fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.85))
    _savefig(fig, ROOT / "corpus_ceg_total.png")


def fig_ceg_within_recipe():
    fig, ax = plt.subplots(figsize=(7.6, 6.1), dpi=150)
    dotpanel(ax, lambda d, r: d["data_ceg_old_algo" if r == "old" else "data_ceg_current_arch"],
             "Corpus-only compute-equivalent gain (×)")
    corpus_legend(fig, loc="upper center", ncol=4, bbox_to_anchor=(0.54, 0.98),
                  columnspacing=2.0, handletextpad=0.7)
    recipe_censor_legend(fig, loc="upper center", ncol=3, bbox_to_anchor=(0.54, 0.89),
                         columnspacing=1.6)
    # title states explicitly that the training recipe/algorithm is held fixed here, since a
    # reader asked whether this panel was showing an algorithm-CEG number (it isn't — every
    # comparison on this figure swaps ONLY the corpus, never the recipe)
    fig.suptitle("Corpus-only compute-equivalent gain, training recipe held fixed",
                 fontsize=12.5, x=0.012, ha="left", color=INK, y=0.995)
    fig.text(0.012, 0.006,
             "124M GPT-2 baseline scale. Each point swaps ONLY the training corpus (OWT → the "
             "labeled corpus) while holding the training recipe/algorithm fixed at the value "
             "given by its marker shape — this isolates the corpus's own contribution, with no "
             "algorithm effect mixed in. RefinedWeb/DCLM give a different corpus multiplier "
             "under each recipe (old GPT-2 recipe ~3.3–3.5× vs. current training recipe "
             "~1.6×): the corpus and training-recipe interventions interact, so there is no "
             "single recipe-independent \"value of the corpus\".",
             fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.10, 1, 0.85))
    _savefig(fig, ROOT / "corpus_ceg_within_recipe.png")


if __name__ == "__main__":
    fig_bpb_curves()
    fig_ceg_total()
    fig_ceg_within_recipe()
