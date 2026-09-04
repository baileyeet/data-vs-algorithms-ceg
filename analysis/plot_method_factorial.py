"""results/method_primitive.{png,pdf}, results/method_shapley_split.{png,pdf} — method
schematic (two standalone images; were one 2-panel figure, split per user request so each
scales independently in a paper).

Explains how a compute-equivalent gain (CEG) is measured and decomposed, using the 124M
current-arch point (results/small/ceg_newdef.json) as the worked example.

  1. method_primitive — "compute to threshold". For each arm we train to a fixed neutral-BPB
     threshold (the old-algo·old-data baseline's converged BPB) and read off the GPU-hours
     it took. Fewer GPU-hours = more compute-efficient.
  2. method_shapley_split — the 2x2 factorial + log-space Shapley. Four arms = {old,new
     algorithm} x {old,new data}. A horizontal edge swaps DATA (holding the algorithm
     fixed); a vertical edge swaps the ALGORITHM (holding data fixed). Each edge is a
     GPU-hours ratio; the Shapley multiplier is the geometric mean of an intervention's two
     edges (both orderings averaged).

All values in-repo; no numbers are invented here — the figure re-derives the published
multipliers from the four arm GPU-hours.
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, ARM_COLORS, _style, _savefig

ROOT = Path(__file__).resolve().parent.parent / "results"
D = json.load(open(ROOT / "small" / "ceg_newdef.json"))
GH = D["gpu_hours_to_threshold"]
THR = D["threshold_neutral_bpb"]
M = D["multipliers"]


# ==================== Figure 1: the primitive (was panel A) ====================
def fig_primitive():
    fig, ax = plt.subplots(figsize=(7.4, 6.0), dpi=150)
    rows = list(csv.DictReader(open(ROOT / "small" / "small_a0d0_dense_metrics.csv")))
    pts = [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
           if float(r["gpu_hours"]) > 0 and r["neutral_bpb"]]
    hs, bs = zip(*pts)
    ax.plot(hs, bs, color=ARM_COLORS["a0d0"], lw=2.2, marker="o", ms=3, zorder=3)
    ax.axhline(THR, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate(f"{THR:.3f}", xy=(0.02, THR), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
    cross = GH["a0d0"]
    ax.axvline(cross, color=INK2, lw=1.0, ls=":", zorder=2)
    ax.scatter([cross], [THR], s=90, color=ARM_COLORS["a0d0"], edgecolors=SURFACE,
               linewidths=1.2, zorder=5)
    ax.annotate(f"Crosses at\n{cross:.2f} GPU-h", xy=(cross, THR), xytext=(10, 34),
                textcoords="offset points", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.set_xscale("log")
    ax.set_ylim(min(bs) - 0.03, THR + 0.42)
    ax.set_xlabel("GPU-hours")
    ax.set_ylabel("Neutral-corpus BPB")
    ax.annotate("A0·D0 (old algorithm · old data)", xy=(0.02, 0.02),
                xycoords="axes fraction", va="bottom", fontsize=8, color=INK2)
    _style(ax)

    fig.suptitle("Compute to reach the neutral-BPB threshold",
                 fontsize=12.5, x=0.012, ha="left", color=INK, y=0.98)
    fig.text(0.012, 0.006,
             "The primitive behind every compute-equivalent gain: train to a fixed neutral-BPB "
             "threshold and read off the GPU-hours it took (fewer = more compute-efficient). "
             "Worked example at the 124M GPT-2 baseline scale, current training recipe, "
             "old-algorithm/old-data arm.",
             fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 0.90))
    _savefig(fig, ROOT / "method_primitive.png")


# ==================== Figure 2: 2x2 factorial + Shapley edges (was panel B) ====================
def fig_shapley_split():
    fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=150)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # cell centers: cols = data (old left, new right), rows = algorithm (old top, new bottom)
    cx = {"D0": 2.6, "D1": 7.4}
    cy = {"A0": 7.2, "A1": 2.8}
    arm_of = {("A0", "D0"): "a0d0", ("A0", "D1"): "a0d1", ("A1", "D0"): "a1d0", ("A1", "D1"): "a1d1"}
    labels = {"a0d0": "A0·D0", "a0d1": "A0·D1", "a1d0": "A1·D0", "a1d1": "A1·D1"}
    for (a, d), arm in arm_of.items():
        x, y = cx[d], cy[a]
        # box border + label are neutral ink, not the ARM_COLORS used elsewhere in the
        # report: this panel already uses blue/orange for a DIFFERENT distinction (data vs.
        # algorithm edges below), and a same-hue box border would collide with that second
        # color code (see FIGURE_NOTES.md bug list). One color language per figure: neutral
        # boxes, colored edges.
        box = FancyBboxPatch((x - 1.35, y - 0.95), 2.7, 1.9, boxstyle="round,pad=0.08",
                             fc=SURFACE, ec=INK2, lw=1.6, zorder=3)
        ax.add_patch(box)
        ax.text(x, y + 0.34, labels[arm], ha="center", va="center",
                fontsize=11, color=INK, fontweight="bold", zorder=4)
        ax.text(x, y - 0.42, f"{GH[arm]:.3f} GPU-h", ha="center", va="center",
                fontsize=9.5, color=INK2, zorder=4)
    # axis labels
    ax.text(cx["D0"], 9.5, "Old data (OWT)", ha="center", fontsize=9.5, color=INK)
    ax.text(cx["D1"], 9.5, "New data (DCLM)", ha="center", fontsize=9.5, color=INK)
    ax.text(0.2, cy["A0"], "Old\nalgorithm", ha="center", va="center", fontsize=9.5, color=INK)
    ax.text(0.2, cy["A1"], "New\nalgorithm", ha="center", va="center", fontsize=9.5, color=INK)

    def edge(x0, y0, x1, y1, ratio, color, above):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                     color=color, lw=1.8, zorder=2, shrinkA=2, shrinkB=2))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        off = 0.55 if above else -0.55
        ax.text(mx, my + off, f"×{ratio:.2f}", ha="center", va="center", fontsize=8.5,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))

    DATA_C, ALGO_C = "#2a78d6", "#eb6834"
    # data edges (horizontal): A0 row, A1 row
    edge(cx["D0"] + 1.4, cy["A0"], cx["D1"] - 1.4, cy["A0"], GH["a0d0"] / GH["a0d1"], DATA_C, True)
    edge(cx["D0"] + 1.4, cy["A1"], cx["D1"] - 1.4, cy["A1"], GH["a1d0"] / GH["a1d1"], DATA_C, False)
    # algorithm edges (vertical): D0 col, D1 col
    edge(cx["D0"], cy["A0"] - 1.0, cx["D0"], cy["A1"] + 1.0, GH["a0d0"] / GH["a1d0"], ALGO_C, True)
    edge(cx["D1"], cy["A0"] - 1.0, cx["D1"], cy["A1"] + 1.0, GH["a0d1"] / GH["a1d1"], ALGO_C, True)

    # Shapley summary box
    dr = f"Data = geomean(×{GH['a0d0']/GH['a0d1']:.2f}, ×{GH['a1d0']/GH['a1d1']:.2f}) = {M['data']:.2f}×"
    ar = f"Algorithm = geomean(×{GH['a0d0']/GH['a1d0']:.2f}, ×{GH['a0d1']/GH['a1d1']:.2f}) = {M['algorithm']:.2f}×"
    ax.text(5.0, 0.35, f"{ar}      {dr}      Total = {M['total']:.1f}×", ha="center", va="bottom",
            fontsize=8.6, color=INK,
            bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f1", ec=GRID, lw=0.8))
    ax.text(5.0, 5.0, "Algorithm edges (orange)\nhold data fixed", ha="center", va="center",
            fontsize=7.6, color=ALGO_C)
    ax.text(5.0, cy["A0"] + 1.35, "Data edges (blue) hold algorithm fixed", ha="center", va="center",
            fontsize=7.6, color=DATA_C)

    fig.suptitle("The 2×2 factorial and its log-space Shapley split",
                 fontsize=12.5, x=0.012, ha="left", color=INK, y=0.98)
    fig.text(0.012, 0.006,
             "Each edge is a GPU-hours ratio between two arms that differ in a single intervention; the Shapley multiplier for "
             "an intervention is the geometric mean of its two edges (both orderings averaged). Worked example at the 124M "
             "GPT-2 baseline scale, current training recipe. Values re-derived from the four arm GPU-hours in "
             "results/small/ceg_newdef.json (held-out Wikipedia evaluation set, “wiki_eval”).",
             fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    _savefig(fig, ROOT / "method_shapley_split.png")


if __name__ == "__main__":
    fig_primitive()
    fig_shapley_split()
