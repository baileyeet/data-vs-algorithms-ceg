"""Render results/core_era_ladder.png — Exp A CORE across data eras.

Mean CORE accuracy (gate-usable tasks) per corpus, old-algo vs new-algo, at 124M.
Secondary/qualitative: at 124M/limit-500 the arms cluster near noise, so this is a
sanity check, not a quantitative CEG claim. Reads results/core_era_ladder.json.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, _style

R = json.load(open(Path(__file__).resolve().parent.parent / "results" / "core_era_ladder.json"))
C = R["corpora"]
order = sorted(C, key=lambda k: C[k]["year"])
xs = [C[c]["year"] for c in order]
old = [C[c]["old_algo_mean_acc"] for c in order]
new = [C[c]["new_algo_mean_acc"] for c in order]

fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
_style(ax)
ax.plot(xs, old, "-o", color="#2a78d6", lw=2, ms=8, label="old algorithm (GPT-2)",
        markeredgecolor=SURFACE, markeredgewidth=1.4)
ax.plot(xs, new, "-s", color="#eb6834", lw=2, ms=8, label="new algorithm (current-arch)",
        markeredgecolor=SURFACE, markeredgewidth=1.4)
ax.set_xticks(xs)
ax.set_xticklabels([f"{c}\n{C[c]['year']}" for c in order])
ax.set_ylim(min(old + new) - 0.03, max(old + new) + 0.02)
ax.set_xlabel("Training-data corpus (release year)")
ax.set_ylabel("Mean CORE accuracy (gate-usable tasks)")
ax.set_title("Exp A — CORE downstream accuracy across data eras (124M)", fontsize=11.5, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "results" / "core_era_ladder.png"
fig.savefig(out, facecolor=SURFACE); plt.close(fig)
print("wrote", out)
