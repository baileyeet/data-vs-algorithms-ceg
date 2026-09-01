"""Render results/era_ladder.png — CEG (vs old-algo OWT baseline) vs dataset
release-year, distinct old-algo and current-arch curves. Matches plots.py style.
Censored arms (never crossed the OWT threshold) are drawn honestly as no-crossing
markers, not fabricated points.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plots import SURFACE, INK, INK2, GRID, _style

OLD_C, NEW_C = "#2a78d6", "#eb6834"  # validated categorical slots (colorblind-distinct)

L = json.load(open("results/era_ladder_results.json"))
C = json.load(open("results/era_correction_2x2.json"))
THR = L["corrected_threshold_neutral_bpb"]
order = sorted(L["datasets"], key=lambda d: L["datasets"][d]["release_year"])

fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
_style(ax)
ax.set_yscale("log")

def series(key, color, label, dx=0.0):
    xs, ys, censored = [], [], []
    for ds in order:
        e = L["datasets"][ds]
        yr = e["release_year"]
        v = e[key]["ceg_vs_a0d0"]
        if v is None:
            censored.append((yr, ds))
        else:
            xs.append(yr); ys.append(v)
    ax.plot(xs, ys, "-o", color=color, lw=2, ms=7, label=label, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    for yr, ds in censored:  # honest no-crossing marker at the floor; dx keeps the
        ax.scatter([yr + dx], [0.55], marker="v", s=90, facecolors="none",  # two algos' markers from overlapping
                   edgecolors=color, linewidths=1.8, zorder=3)
    return xs, ys

series("old_algo", OLD_C, "old-algo (GPT-2, 123.7M params)", dx=-0.4)
xs2, ys2 = series("current_arch", NEW_C, "current-arch (modded, 498.8M params)", dx=+0.4)

# baseline line at CEG=1 (old-algo OWT reference)
ax.axhline(1.0, color=GRID, lw=1, ls="--", zorder=1)
ax.annotate("old-algo OWT baseline (1×)", (2024.05, 1.0), fontsize=7.5,
            color=INK2, va="bottom", ha="right")

# C4 censored note
ax.annotate("C4 (2020): no crossing — neutral BPB stays\nabove the OWT threshold under BOTH algorithms",
            (2020, 0.62), fontsize=7.5, color=INK2, ha="center", va="bottom")

ax.set_ylim(0.28, 60)
ax.set_xlim(2018.6, 2024.6)
ax.set_xticks([2019, 2020, 2023, 2024])
ax.set_xticklabels([f"{ds}\n{L['datasets'][ds]['release_year']}" for ds in order])
ax.set_ylabel("Compute-equivalent gain vs old-algo on OWT  (×, log)")
ax.set_xlabel("Training-data corpus (release year)")
ax.set_title("Exp A — data-era ladder (@124M)", fontsize=11.5, loc="left")
ax.legend(loc="center left", frameon=False, fontsize=8.5)
fig.tight_layout()
Path("results").mkdir(exist_ok=True)
fig.savefig("results/era_ladder.png", facecolor=SURFACE, bbox_inches="tight")
print("wrote results/era_ladder.png")
