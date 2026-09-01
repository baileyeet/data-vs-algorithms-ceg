"""Render results/core_era_by_task.png — Exp A CORE broken down BY TASK.

One panel per gate-usable CORE task; within each, old-algo vs new-algo accuracy across
the 4 data-era corpora (OWT/C4/RefinedWeb/DCLM), with ±1 stderr bars. Companion to the
aggregate core_era_ladder.png, matching the 2x2 study's core_arms_by_task convention and
Exp B's core_expb_by_task. SECONDARY / qualitative (124M, limit=500 -> noisy).

Data: OWT/DCLM from the completed 2x2 study 124M arms (results/core_finals/); C4/RefinedWeb
from the faithful retrain (results/core_era/).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, _style

ROOT = Path(__file__).resolve().parent.parent / "results"
# (corpus, year, old-algo file, new-algo file)
CORPORA = [
    ("OWT", 2019, "core_finals/small_a0d0_dense_ckpt_016925.json", "core_finals/small_a1d0_2x_v2_ckpt_002780.json"),
    ("C4", 2020, "core_era/era_a0_c4.json", "core_era/era_a1_c4.json"),
    ("RefinedWeb", 2023, "core_era/era_a0_refinedweb.json", "core_era/era_a1_refinedweb.json"),
    ("DCLM", 2024, "core_finals/small_a0d1_ckpt_016925.json", "core_finals/small_a1d1_2x_v2_ckpt_002780.json"),
]
# the study's fixed CORE-figure task set (matches core_vs_scale / core_arms_by_task)
TASKS = ["arc_easy", "hellaswag", "piqa", "copa", "xwinograd_en", "boolq"]
CHANCE = {"arc_easy": .25, "hellaswag": .25, "piqa": .5, "copa": .5, "xwinograd_en": .5, "boolq": .5}


def load(p):
    d = json.load(open(ROOT / p)); return d.get("results", d)


def acc(res, t):
    v = res.get(t, {})
    key = "acc_norm,none" if (t in ("arc_easy", "hellaswag") and "acc_norm,none" in v) else "acc,none"
    se = "acc_norm_stderr,none" if "norm" in key else "acc_stderr,none"
    return v.get(key), v.get(se)


OLD = {c: load(fo) for c, y, fo, fn in CORPORA}
NEW = {c: load(fn) for c, y, fo, fn in CORPORA}
years = [y for c, y, fo, fn in CORPORA]
labels = [c for c, y, fo, fn in CORPORA]

ncol = 3
nrow = (len(TASKS) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 3.0 * nrow), dpi=150, sharex=True)
axf = axes.ravel()
for i, task in enumerate(TASKS):
    ax = axf[i]
    ch = CHANCE[task]
    ax.axhline(ch, color=GRID, lw=1, ls=(0, (2, 2)), zorder=1)
    ax.annotate("chance", (years[0], ch), xytext=(0, 2), textcoords="offset points",
                fontsize=6.5, color=INK2, va="bottom")
    for arm, hue, mk, lbl in [(OLD, "#2a78d6", "o", "old algo (GPT-2)"),
                              (NEW, "#eb6834", "s", "new algo (current-arch)")]:
        ys, es = [], []
        for c, y, fo, fn in CORPORA:
            a, s = acc(arm[c], task)
            ys.append(a); es.append(s or 0)
        ax.errorbar(years, ys, yerr=es, color=hue, lw=1.8, marker=mk, ms=6, capsize=2.5,
                    elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1, label=lbl, zorder=3)
    ax.set_xticks(years); ax.set_xticklabels([f"{c}\n{y}" for c, y in zip(labels, years)], fontsize=7.5)
    ax.set_title(task, fontsize=10, loc="left")
    _style(ax)
for j in range(len(TASKS), len(axf)):
    axf[j].axis("off")
for r in range(nrow):
    axes[r, 0].set_ylabel("accuracy", fontsize=8.5)
axf[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")
fig.suptitle("Exp A — CORE accuracy per task, old vs new algorithm across data eras (124M)",
             fontsize=12.5, x=0.012, ha="left", color=INK)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = ROOT / "core_era_by_task.png"
fig.savefig(out, facecolor=SURFACE); plt.close(fig)
print("wrote", out)
