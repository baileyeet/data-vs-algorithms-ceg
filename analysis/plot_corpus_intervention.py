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
axA.annotate(f"reference threshold {THR:.3f}\n(old-algo · OWT)", xy=(0.015, THR),
             xycoords=("axes fraction", "data"), xytext=(0, 4),
             textcoords="offset points", fontsize=8, color=INK2)
axA.set_xscale("log")
axA.set_ylim(lo - 0.03, THR + 0.30)
axA.set_xlabel("GPU-hours (log)")
axA.set_ylabel("Neutral-corpus BPB  (lower = better)")
axA.set_title("A · Compute to reach the neutral-BPB threshold", fontsize=10.5, loc="left")
_style(axA)
corpus_handles = [Line2D([], [], color=CORP_COLOR[c], lw=2.4,
                         label=f"{c} ({y})") for c, y, *_ in CORPORA]
recipe_handles = [Line2D([], [], color=INK2, lw=1.9, ls="-", label="old-algo (GPT-2)"),
                  Line2D([], [], color=INK2, lw=1.9, ls=(0, (4, 2)), label="current-arch (modded)")]
leg1 = axA.legend(handles=corpus_handles, frameon=False, fontsize=8, labelcolor=INK2,
                  loc="upper right", title="corpus", title_fontsize=8)
leg1._legend_box.align = "left"
axA.add_artist(leg1)
axA.legend(handles=recipe_handles, frameon=False, fontsize=8, labelcolor=INK2,
           loc="lower left", title="recipe", title_fontsize=8)


# ---------------- Panels B & C: multiplier dot plots ----------------
def dotpanel(ax, getval, title, ylab):
    """getval(dataset_dict, recipe) -> (value or None). recipe in {'old','current'}."""
    xs = list(range(len(CORPORA)))
    ax.axhline(1.0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("1× — no advantage", xy=(0.98, 1.0), xycoords=("axes fraction", "data"),
                xytext=(0, -3), textcoords="offset points", ha="right", va="top",
                fontsize=7.5, color=INK2)
    for i, (corpus, year, *_ ) in enumerate(CORPORA):
        d = J["datasets"][corpus]
        col = CORP_COLOR[corpus]
        for recipe, dx, mk in (("old", -0.13, "o"), ("current", 0.13, "s")):
            v = getval(d, recipe)
            if v is None:  # censored -> hollow marker at the parity line + tag
                ax.scatter([i + dx], [1.0], marker=mk, s=70, facecolors="none",
                           edgecolors=col, linewidths=1.7, zorder=4)
            else:
                ax.scatter([i + dx], [v], marker=mk, s=70, facecolors=col,
                           edgecolors=SURFACE, linewidths=1.0, zorder=4)
                ax.annotate(f"{v:.1f}×", xy=(i + dx, v), xytext=(0, 8 if recipe == "old" else 8),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
    # censored note for C4 (placed ABOVE the hollow markers, clear of the x-tick label)
    ci = [i for i, (c, *_ ) in enumerate(CORPORA) if c == "C4"][0]
    ax.annotate("C4 censored\n(never crosses)", xy=(ci, 1.0), xytext=(0, 20),
                textcoords="offset points", ha="center", fontsize=7.5, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 60)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{c}\n{y}" for c, y, *_ in CORPORA], fontsize=8.5)
    ax.set_xlim(-0.5, len(CORPORA) - 0.5)
    ax.set_xlabel("Training corpus (ordered by release date)")
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=10.5, loc="left")
    _style(ax)


dotpanel(axB, lambda d, r: d["old_algo" if r == "old" else "current_arch"]["ceg_vs_a0d0"],
         "B · Corpus CEG vs reference (old-algo · OWT)",
         "Compute-reduction multiplier  (×, log)")
dotpanel(axC, lambda d, r: d["data_ceg_old_algo" if r == "old" else "data_ceg_current_arch"],
         "C · Within-recipe data lever (swap OWT → corpus)",
         "Data-only compute-reduction  (×, log)")

# shared recipe legend for B/C — figure-level, top-center (clear of all data marks)
shape_handles = [Line2D([], [], color=INK2, marker="o", ls="none", ms=8, label="old-algo (GPT-2) recipe"),
                 Line2D([], [], color=INK2, marker="s", ls="none", ms=8, label="current-arch (modded) recipe"),
                 Line2D([], [], color=INK2, marker="o", ls="none", ms=8, markerfacecolor="none",
                        label="hollow = censored (bound, not a value)")]
fig.legend(handles=shape_handles, frameon=False, fontsize=8, labelcolor=INK2,
           loc="upper center", ncol=3, bbox_to_anchor=(0.68, 0.965))

fig.suptitle("Exp A — corpus intervention at the 124M GPT-2 baseline scale (neutral eval: wiki_eval_union)",
             fontsize=12.5, x=0.008, ha="left", color=INK, y=0.995)
fig.text(0.008, 0.008,
         "Panel C isolates the corpus lever within each fixed recipe. The two recipes give different "
         "data multipliers (old-algo ~3.3–3.5× vs current-arch ~1.6× on RefinedWeb/DCLM): the corpus and "
         "training-recipe interventions INTERACT, so a Shapley attribution of “how much the data is "
         "worth” is context-dependent and not a single number. C4 (2020) never reaches the threshold "
         "under either recipe — corpus quality is non-monotonic in release date.",
         fontsize=7, color=INK2, va="bottom", wrap=True)
fig.tight_layout(rect=(0, 0.07, 1, 0.955))
_savefig(fig, ROOT / "corpus_intervention.png")
