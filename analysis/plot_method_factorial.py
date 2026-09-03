"""results/method_factorial.{png,pdf} — method schematic (candidate / appendix figure).

Explains, in one image, how a compute-equivalent gain (CEG) is measured and decomposed,
using the 124M current-arch point (results/small/ceg_newdef.json) as the worked example.

  A. The primitive — "compute to threshold". For each arm we train to a fixed neutral-BPB
     threshold (the old-algo·old-data baseline's converged BPB) and read off the GPU-hours
     it took. Fewer GPU-hours = more compute-efficient.
  B. The 2x2 factorial + log-space Shapley. Four arms = {old,new algorithm} x {old,new data}.
     A horizontal edge swaps DATA (holding the algorithm fixed); a vertical edge swaps the
     ALGORITHM (holding data fixed). Each edge is a GPU-hours ratio; the Shapley multiplier
     is the geometric mean of an intervention's two edges (both orderings averaged).

All values in-repo; no numbers are invented here — the figure re-derives the published
multipliers from the four arm GPU-hours.
"""
import csv
import json
import math
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

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=150,
                               gridspec_kw={"width_ratios": [1.05, 1.25]})

# ---------------- Panel A: the primitive (compute to threshold) ----------------
rows = list(csv.DictReader(open(ROOT / "small" / "small_a0d0_dense_metrics.csv")))
pts = [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
       if float(r["gpu_hours"]) > 0 and r["neutral_bpb"]]
hs, bs = zip(*pts)
axA.plot(hs, bs, color=ARM_COLORS["a0d0"], lw=2.2, marker="o", ms=3, zorder=3)
axA.axhline(THR, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
axA.annotate(f"Neutral-BPB threshold {THR:.3f}", xy=(0.02, THR), xycoords=("axes fraction", "data"),
             xytext=(0, 4), textcoords="offset points", fontsize=8, color=INK2)
cross = GH["a0d0"]
axA.axvline(cross, color=INK2, lw=1.0, ls=":", zorder=2)
axA.scatter([cross], [THR], s=90, color=ARM_COLORS["a0d0"], edgecolors=SURFACE, linewidths=1.2, zorder=5)
axA.annotate(f"Crosses at\n{cross:.2f} GPU-h", xy=(cross, THR), xytext=(10, 34),
             textcoords="offset points", fontsize=8.5, color=INK,
             arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
axA.set_xscale("log")
axA.set_ylim(min(bs) - 0.03, THR + 0.42)
axA.set_xlabel("GPU-hours (log)")
axA.set_ylabel("Neutral-corpus BPB  (lower = better)")
axA.set_title("A · The primitive — compute to reach the threshold", fontsize=10.5, loc="left")
axA.annotate("Example arm: old algorithm · old data (the baseline)", xy=(0.02, 0.02),
             xycoords="axes fraction", va="bottom", fontsize=8, color=INK2)
_style(axA)

# ---------------- Panel B: 2x2 factorial + Shapley edges ----------------
axB.set_xlim(0, 10); axB.set_ylim(0, 10); axB.axis("off")
axB.set_title("B · The 2×2 factorial and its log-space Shapley split", fontsize=10.5, loc="left")
# cell centers: cols = data (old left, new right), rows = algorithm (old top, new bottom)
cx = {"D0": 2.6, "D1": 7.4}
cy = {"A0": 7.2, "A1": 2.8}
arm_of = {("A0", "D0"): "a0d0", ("A0", "D1"): "a0d1", ("A1", "D0"): "a1d0", ("A1", "D1"): "a1d1"}
labels = {"a0d0": "A0·D0", "a0d1": "A0·D1", "a1d0": "A1·D0", "a1d1": "A1·D1"}
for (a, d), arm in arm_of.items():
    x, y = cx[d], cy[a]
    box = FancyBboxPatch((x - 1.35, y - 0.95), 2.7, 1.9, boxstyle="round,pad=0.08",
                         fc=SURFACE, ec=ARM_COLORS[arm], lw=2.0, zorder=3)
    axB.add_patch(box)
    axB.text(x, y + 0.34, labels[arm] if False else labels[arm], ha="center", va="center",
             fontsize=11, color=ARM_COLORS[arm], fontweight="bold", zorder=4)
    axB.text(x, y - 0.42, f"{GH[arm]:.3f} GPU-h", ha="center", va="center",
             fontsize=9.5, color=INK, zorder=4)
# axis labels
axB.text(cx["D0"], 9.5, "Old data (OWT)", ha="center", fontsize=9.5, color=INK)
axB.text(cx["D1"], 9.5, "New data (DCLM)", ha="center", fontsize=9.5, color=INK)
axB.text(0.2, cy["A0"], "Old\nalgorithm", ha="center", va="center", fontsize=9.5, color=INK)
axB.text(0.2, cy["A1"], "New\nalgorithm", ha="center", va="center", fontsize=9.5, color=INK)


def edge(x0, y0, x1, y1, ratio, color, above):
    axB.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
                                  color=color, lw=1.8, zorder=2, shrinkA=2, shrinkB=2))
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    off = 0.55 if above else -0.55
    axB.text(mx, my + off, f"×{ratio:.2f}", ha="center", va="center", fontsize=8.5,
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
axB.text(5.0, 0.35, f"{ar}      {dr}      Total = {M['total']:.1f}×", ha="center", va="bottom",
         fontsize=8.6, color=INK,
         bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f1", ec=GRID, lw=0.8))
axB.text(5.0, 5.0, "Algorithm edges (orange)\nhold data fixed", ha="center", va="center",
         fontsize=7.6, color=ALGO_C)
axB.text(5.0, cy["A0"] + 1.35, "Data edges (blue) hold algorithm fixed", ha="center", va="center",
         fontsize=7.6, color=DATA_C)

fig.suptitle("Factorial decomposition of compute-equivalent gain",
             fontsize=12.5, x=0.012, ha="left", color=INK, y=0.99)
fig.text(0.012, 0.925, "Worked example: 124M GPT-2 baseline scale, current training recipe",
         fontsize=9.5, color=INK2, va="top")
fig.text(0.012, 0.006,
         "Each edge is a GPU-hours ratio between two arms that differ in a single intervention; the Shapley multiplier for "
         "an intervention is the geometric mean of its two edges (both orderings averaged). Values re-derived from the four "
         "arm GPU-hours in results/small/ceg_newdef.json (held-out Wikipedia evaluation set, “wiki_eval”).",
         fontsize=7.5, color=INK2, va="bottom", wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 0.90))
_savefig(fig, ROOT / "method_factorial.png")
