"""results/core_bpb_vs_downstream.{png,pdf} — the ONE Exp B CORE figure for the blog.

Message: BPB compute-efficiency and downstream-task capability can DISAGREE. Both panels
are oriented so that UP = the candidate architecture beats its size-matched GPT-2
(trained through the identical pipeline); the dashed line at 0 is parity.

  Left  — compute efficiency (neutral BPB): advantage = denom_BPB - candidate_BPB (bits).
           Neither lineage is above parity at any scale (the B1 headline).
  Right — downstream capability (CORE, 11-task mean accuracy): advantage = candidate - GPT-2.
           Pythia AGREES with BPB (at/below parity); SmolLM2 DISAGREES — a small but
           consistent downstream EDGE that the BPB metric does not see.

Secondary/qualitative (CORE at limit=500 is noisy); shown with its noise band (BPB) and
±1 stderr (CORE). Data: results/core_expb_summary.json + results/b1_results.json (in-repo).
"""
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
C = json.load(open(ROOT / "core_expb_summary.json"))["pairs"]
B = json.load(open(ROOT / "b1_results.json"))
BPB_SIGMA = B["noise_sigma"]
LINCOL = {"pythia": "#1baf7a", "smollm2": "#8e44ad"}
LINLBL = {"pythia": "Pythia (GPT-NeoX, 2023)", "smollm2": "SmolLM2 (Llama, 2024)"}
# (lineage, [(tier_x, core_key, size_key)]) — tiers pair the two lineages' nearest sizes
TIERS = ["≈135–160M", "≈360–410M", "≈1.4–1.7B"]
ROWS = {
    "pythia": [(0, "pythia-160M", "160M"), (1, "pythia-410M", "410M"), (2, "pythia-1.4B", "1.4B")],
    "smollm2": [(0, "smollm2-135M", "135M"), (1, "smollm2-360M", "360M"), (2, "smollm2-1.7B", "1.7B")],
}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.2), dpi=150)

# ---- parity + noise band on both panels ----
for ax in (axL, axR):
    ax.axhline(0, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=2)
axL.axhspan(-BPB_SIGMA, BPB_SIGMA, color=GRID, alpha=0.5, zorder=0)

# ---- Left: BPB efficiency advantage (up = candidate better) ----
for lin, rows in ROWS.items():
    xs = [t for t, _, _ in rows]
    ys = [-B[lin][sz]["delta"] for _, _, sz in rows]        # advantage = -(cand - denom)
    conf = [B[lin][sz].get("confounded", False) for _, _, sz in rows]
    axL.plot(xs, ys, color=LINCOL[lin], lw=2, marker="o", ms=8, zorder=3,
             markerfacecolor=[LINCOL[lin] if not c else SURFACE for c in conf][0])
    for x, y, cf in zip(xs, ys, conf):
        axL.scatter([x], [y], s=64, zorder=4, color=LINCOL[lin],
                    facecolors=(SURFACE if cf else LINCOL[lin]), edgecolors=LINCOL[lin], linewidths=1.8)
axL.annotate("candidate better ↑", xy=(0.02, 0.98), xycoords="axes fraction", va="top",
             fontsize=8, color=INK2)
axL.annotate("worse ↓", xy=(0.02, 0.02), xycoords="axes fraction", va="bottom", fontsize=8, color=INK2)
axL.annotate("parity ±1σ (BPB noise)", xy=(0.98, BPB_SIGMA), xycoords=("axes fraction", "data"),
             ha="right", va="bottom", xytext=(0, 1), textcoords="offset points", fontsize=7.5, color=INK2)
axL.set_ylabel("BPB advantage over matched GPT-2  (bits, ↑ better)")
axL.set_title("Compute efficiency (neutral BPB)", fontsize=11, loc="left")

# ---- Right: CORE downstream advantage (up = candidate better), ±1 stderr ----
for lin, rows in ROWS.items():
    xs = [t for t, _, _ in rows]
    ys = [C[ck]["mean_delta"] for _, ck, _ in rows]
    es = [C[ck]["mean_delta_stderr"] for _, ck, _ in rows]
    axR.errorbar(xs, ys, yerr=es, color=LINCOL[lin], lw=2, marker="o", ms=8, capsize=3.5,
                 elinewidth=1.2, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3)
axR.annotate("candidate better ↑", xy=(0.02, 0.98), xycoords="axes fraction", va="top",
             fontsize=8, color=INK2)
axR.annotate("worse ↓", xy=(0.02, 0.02), xycoords="axes fraction", va="bottom", fontsize=8, color=INK2)
axR.set_ylabel("CORE accuracy advantage over matched GPT-2  (↑ better)")
axR.set_title("Downstream capability (CORE, 11-task mean)", fontsize=11, loc="left")

for ax in (axL, axR):
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(TIERS, fontsize=9)
    ax.set_xlim(-0.3, 2.3)
    ax.set_xlabel("Model scale tier")
    _style(ax)

handles = [Line2D([], [], color=LINCOL[l], lw=2, marker="o", ms=8, label=LINLBL[l])
           for l in ("pythia", "smollm2")]
handles.append(Line2D([], [], color=INK2, lw=0, marker="o", ms=8, markerfacecolor="none",
                      label="open = divergence-confounded (SmolLM2 360M/1.7B; best-case less negative)"))
fig.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="upper center",
           ncol=3, bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Exp B — a metric disagreement: BPB efficiency is not the same as downstream capability",
             fontsize=12.5, x=0.012, ha="left", color=INK)
fig.text(0.012, 0.006,
         "Both panels: ↑ = candidate beats its size-matched GPT-2 (identical pipeline). Pythia sits at/below parity on BOTH "
         "metrics. SmolLM2 never beats GPT-2 on BPB (left) yet shows a small, consistent downstream edge on CORE (right) — "
         "the two measures disagree. CORE (limit=500) is secondary and noisy; ±1 stderr shown.",
         fontsize=7.5, color=INK2, va="bottom", wrap=True)
fig.tight_layout(rect=(0, 0.05, 1, 0.9))
_savefig(fig, ROOT / "core_bpb_vs_downstream.png")
