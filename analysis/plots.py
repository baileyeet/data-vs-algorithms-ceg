"""Phase 5 figures. All figures share one style: light surface, recessive
grid, thin marks, fixed per-arm colors (color follows the entity — an arm keeps
its hue in every figure at every size), direct labels + legend.

Figures:
  training_curves   — neutral BPB vs GPU-hours (log x), 4 arms + threshold line
  cross_scale       — data/algorithm Shapley multipliers vs model size (the
                      headline result; log x in params)
  threshold_sensitivity — multipliers vs reference-BPB threshold, per size
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# categorical slots 1-4 (validated order/palette; see dataviz reference)
ARM_COLORS = {"a0d0": "#2a78d6", "a0d1": "#1baf7a", "a1d0": "#eda100", "a1d1": "#008300"}
ARM_LABELS = {"a0d0": "A0D0 old algo · old data", "a0d1": "A0D1 old algo · new data",
              "a1d0": "A1D0 new algo · old data", "a1d1": "A1D1 new algo · new data"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MULT_COLORS = {"data": "#2a78d6", "algorithm": "#eb6834", "total": INK2}


def _style(ax):
    # log axes: major decade labels only (minor labels collide at this size)
    for axis in (ax.xaxis, ax.yaxis):
        if ax.get_xscale() == "log" and axis is ax.xaxis or \
           ax.get_yscale() == "log" and axis is ax.yaxis:
            axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def load_metrics(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "metrics.csv")))
    return [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
            if float(r["gpu_hours"]) > 0]


def training_curves(run_dirs: dict, size_label: str, out_path, threshold=None):
    """run_dirs: {'a0d0': path, ...}"""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for arm in ["a0d0", "a0d1", "a1d0", "a1d1"]:
        if arm not in run_dirs:
            continue
        pts = load_metrics(run_dirs[arm])
        hs, bs = zip(*pts)
        ax.plot(hs, bs, color=ARM_COLORS[arm], linewidth=2, marker="o",
                markersize=4, label=ARM_LABELS[arm])
        ax.annotate(arm.upper(), xy=(hs[-1], bs[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5,
                    color=INK, va="center")
    if threshold is not None:
        ax.axhline(threshold, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(f"reference BPB {threshold:.3f}", xy=(1, threshold),
                    xycoords=("axes fraction", "data"), xytext=(-4, -11),
                    textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    ax.set_xscale("log")
    ax.set_xlabel("GPU-hours (timed, log scale)")
    ax.set_ylabel("Neutral-corpus BPB")
    ax.set_title(f"Training curves — {size_label}", fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def cross_scale(results: list, out_path):
    """results: list of dicts from ceg_shapley --out-json, plus 'n_params' each."""
    results = sorted(results, key=lambda r: r["n_params"])
    xs = [r["n_params"] / 1e6 for r in results]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for key in ["data", "algorithm", "total"]:
        ys = [r["multipliers"][key] for r in results]
        style = dict(linestyle=(0, (4, 3)), linewidth=1.5) if key == "total" \
            else dict(linewidth=2)
        ax.plot(xs, ys, color=MULT_COLORS[key], marker="o", markersize=7,
                label=f"{key} contribution", **style)
        ax.annotate(f"{key} {ys[-1]:.1f}×", xy=(xs[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model size (M params, log scale)")
    ax.set_ylabel("Compute-reduction multiplier (log scale)")
    ax.set_title("Shapley split of compute-equivalent gain vs model scale",
                 fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def threshold_sensitivity(csv_path, size_label: str, out_path):
    rows = list(csv.DictReader(open(csv_path)))
    thr = [float(r["threshold_bpb"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
    for key in ["data", "algorithm"]:
        ys = [float(r[f"{key}_multiplier"]) for r in rows]
        ax.plot(thr, ys, color=MULT_COLORS[key], linewidth=2,
                label=f"{key} contribution")
        ax.annotate(f"{key}", xy=(thr[-1], ys[-1]), xytext=(6, 0),
                    textcoords="offset points", fontsize=8.5, color=INK, va="center")
    ax.invert_xaxis()  # deeper (lower BPB) targets to the right
    ax.set_yscale("log")
    ax.set_xlabel("Reference BPB threshold (deeper →)")
    ax.set_ylabel("Compute-reduction multiplier (log scale)")
    ax.set_title(f"Threshold sensitivity — {size_label}", fontsize=11, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="render one figure type")
    ap.add_argument("kind", choices=["curves", "cross_scale", "sensitivity"])
    ap.add_argument("--size-label", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--runs", nargs="*", help="arm=run_dir pairs (curves)")
    ap.add_argument("--results", nargs="*", help="ceg json files (cross_scale)")
    ap.add_argument("--csv", help="sensitivity csv")
    a = ap.parse_args()
    if a.kind == "curves":
        training_curves(dict(kv.split("=") for kv in a.runs), a.size_label,
                        a.out, a.threshold)
    elif a.kind == "cross_scale":
        cross_scale([json.loads(Path(p).read_text()) for p in a.results], a.out)
    else:
        threshold_sensitivity(a.csv, a.size_label, a.out)
    print(f"wrote {a.out}")
