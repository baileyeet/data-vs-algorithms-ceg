"""Render results/data_ladder_expB.png — Exp B data axis.

data-CEG (vs the shared external bar = size-matched GPT-2-trained-on-OWT) as a function
of dataset release-year, one line per architecture (Pythia, SmolLM2). Arms that never
reach the GPT-2-OWT bar are drawn as hollow down-triangles at the 1x parity floor
(censored), matching plot_era_ladder's honest no-crossing convention. The headline the
one fixed external bar makes visible: on OWT/C4 both archs are censored (lose to GPT-2),
on RefinedWeb/DCLM both cross (beat GPT-2) with a 4.6-6.4x data-CEG.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, _style

R = json.load(open(Path(__file__).resolve().parent.parent / "results" / "data_ladder_results.json"))
LIN = {"pythia_160m": ("Pythia-160M (GPT-NeoX)", "#1baf7a"),
       "smollm2_135m": ("SmolLM2-135M (Llama)", "#8e44ad")}

fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=150)
_style(ax); ax.set_yscale("log")

ax.axhline(1.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
ax.annotate("1× = matched GPT-2 on OWT (the shared bar)", xy=(2024.3, 1.0),
            xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
            fontsize=8, color=INK2)

for key, (label, col) in LIN.items():
    corp = R[key]["corpora"]
    order = sorted(corp, key=lambda c: corp[c]["year"])
    xs, ys, cens = [], [], []
    for c in order:
        e = corp[c]
        if e["crosses_gpt2"]:
            xs.append(e["year"]); ys.append(e["data_ceg"])
        else:
            cens.append(e["year"])
    ax.plot(xs, ys, "-o", color=col, lw=2, ms=8, label=label, zorder=3,
            markeredgecolor=SURFACE, markeredgewidth=1.4)
    for x, y in zip(xs, ys):
        yo = 10 if key == "smollm2_135m" else -16
        ax.annotate(f"{y:.1f}×", (x, y), xytext=(0, yo), textcoords="offset points",
                    ha="center", fontsize=9, color=col)
    dx = -0.12 if key == "pythia_160m" else 0.12
    for yr in cens:
        ax.scatter([yr + dx], [0.62], marker="v", s=95, facecolors="none",
                   edgecolors=col, linewidths=1.9, zorder=3)

ax.annotate("censored — never reaches the GPT-2-OWT bar\n(loses to a matched GPT-2)",
            (2019.5, 0.62), xytext=(2020.4, 0.66), fontsize=8, color=INK2,
            va="center", ha="left")
ax.set_ylim(0.4, 9)
ax.set_xlim(2018.5, 2024.8)
ax.set_xticks([2019, 2020, 2023, 2024])
ax.set_xticklabels(["OWT\n2019", "C4\n2020", "RefinedWeb\n2023", "DCLM\n2024"])
ax.set_xlabel("Training-data corpus (release year)")
ax.set_ylabel("Data-CEG vs matched GPT-2 on OWT (log)")
ax.set_title("Exp B — data axis: better data flips new architectures from losing to beating GPT-2",
             fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
fig.text(0.012, 0.012,
         "Each architecture (small size) trained from scratch on each corpus; compared to ONE fixed "
         "external bar — the size-matched GPT-2 trained on OWT. On OWT (2019) and C4 (2020) both "
         "architectures are censored: they never reach the bar (they lose to a matched GPT-2, the "
         "Exp B / B1 result). On RefinedWeb (2023) and DCLM (2024) both cross it, reaching GPT-2's "
         "OWT quality with 4.6-6.4× less compute. C4 worse than OWT for BOTH archs independently "
         "cross-validates Exp A's non-monotonic-in-release-year finding.",
         fontsize=7.0, color=INK2, va="bottom", wrap=True)
fig.tight_layout(rect=(0, 0.08, 1, 1))
out = Path(__file__).resolve().parent.parent / "results" / "data_ladder_expB.png"
fig.savefig(out, facecolor=SURFACE); plt.close(fig)
print("wrote", out)
