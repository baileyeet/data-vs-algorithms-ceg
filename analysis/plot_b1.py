"""Render results/b1_cross_scale.png — Exp B B1 headline.

Delta (candidate neutral BPB - matched GPT-2 train_hf denominator, both @512k) vs
model scale, one curve per Transformer lineage (Pythia 2023, SmolLM2 2024). Delta=0
is parity with a matched, properly-tuned GPT-2; Delta<0 would be a genuine algorithm
advantage. NARRATIVE (Peter's Exp-B question): does the current-arch speedrun's large
small-scale algo advantage generalize to published lineages? Answer here = NO — nothing
lands in the advantage region across 135M-1.7B.

CONVENTION (matches plots.py / era_ladder.py): filled markers = clean numbers;
open markers (facecolors='none') = divergence-confounded (SmolLM2 overfit OWT at 360M/
1.7B — own-val down while neutral BPB rose). Confounded points carry a bar from their
best/min-based Delta up to the tail-mean Delta, so a reader never treats the confounded
tail number as equally trustworthy.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, _style

PYTHIA_C, SMOL_C = "#2a78d6", "#eb6834"  # validated categorical slots (colorblind-distinct)

R = json.load(open(Path(__file__).resolve().parent.parent / "results" / "b1_results.json"))
SIG = R["noise_sigma"]

fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
_style(ax)
ax.set_xscale("log")

# advantage region (Delta<0) + noise band — the point is that nothing real lands below 0
xlo, xhi = 1.1e8, 2.1e9
ax.axhspan(-0.5, 0.0, color=PYTHIA_C, alpha=0.04, zorder=0)
ax.axhline(0.0, color=INK2, lw=1.2, ls="--", zorder=1)
ax.axhspan(-SIG, SIG, color=GRID, alpha=0.6, zorder=0)
ax.annotate("algorithm advantage region\n(nothing observed here)", (1.25e8, -0.052),
            fontsize=8.5, color=PYTHIA_C, va="center", ha="left")
ax.annotate(f"parity with matched GPT-2  (±1σ={SIG})", (xhi, 0.0), xytext=(-4, 6),
            textcoords="offset points", fontsize=8, color=INK2, va="bottom", ha="right")

def lineage(key, color, label):
    d = R[key]
    order = sorted(d, key=lambda k: d[k]["params"])
    xs = [d[k]["params"] for k in order]
    ys = [d[k]["delta"] for k in order]
    ax.plot(xs, ys, "-", color=color, lw=2, zorder=2, label=label)
    for k in order:
        e = d[k]; x = e["params"]; y = e["delta"]
        if e.get("confounded"):
            # bar from best/min-based Delta up to tail-mean Delta; open marker at tail
            ax.plot([x, x], [e["delta_min"], y], color=color, lw=1.4, zorder=2)
            ax.scatter([x], [e["delta_min"]], marker="_", s=120, color=color, zorder=3)
            ax.scatter([x], [y], marker="o", s=80, facecolors="none",
                       edgecolors=color, linewidths=1.8, zorder=3)
        else:
            ax.scatter([x], [y], marker="o", s=80, color=color, zorder=3,
                       edgecolors=SURFACE, linewidths=1.4)
        ax.annotate(k, (x, y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8, color=INK2)

lineage("pythia", PYTHIA_C, "Pythia (GPT-NeoX, 2023)")
lineage("smollm2", SMOL_C, "SmolLM2 (Llama, 2024)")

# call out the two special cases explicitly
ax.annotate("SmolLM2-135M: only point at parity\n(2-seed confirmed)",
            (1.345e8, 0.0090), xytext=(1.7e8, -0.028), fontsize=7.8, color=SMOL_C,
            va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=SMOL_C, lw=0.8))
ax.annotate("open markers = divergence-confounded\n(SmolLM2 overfit OWT; bar = best→tail-mean)",
            (1.711e9, 0.1201), xytext=(2.3e8, 0.112), fontsize=7.8, color=SMOL_C,
            va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color=SMOL_C, lw=0.8, connectionstyle="arc3,rad=-0.15"))

ax.set_xlim(xlo, xhi)
ax.set_ylim(-0.075, 0.135)
ax.set_xticks([1.5e8, 3e8, 5e8, 1e9, 2e9])
ax.set_xticklabels(["150M", "300M", "500M", "1B", "2B"])
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_xlabel("Model scale (parameters, log)")
ax.set_ylabel("Δ neutral BPB  vs matched GPT-2  (>0 = deficit)")
ax.set_title("Exp B B1 — no published Transformer lineage beats a matched GPT-2 (135M–1.7B)",
             fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
fig.text(0.008, 0.008,
         "Direct test of whether current-arch's small-scale speedrun advantage generalizes: it does not. "
         "Best case = SmolLM2-135M parity; all else deficit. Converged 512k, like-for-like train_hf denominators.",
         fontsize=7.2, color=INK2, ha="left")
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = Path(__file__).resolve().parent.parent / "results" / "b1_cross_scale.png"
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
