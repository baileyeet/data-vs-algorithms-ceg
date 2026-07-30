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


# the two cross-scale curves are kept separate everywhere; color follows the
# lineage (an entity), never the rank. current-arch = blue, ScaleUp = orange
# (high CVD separation; from the validated categorical palette).
CURVE_COLORS = {"current-arch": "#2a78d6", "scaleup": "#eb6834"}
CURVE_LABELS = {"current-arch": "current-arch (2024 speedrun)",
                "scaleup": "ScaleUp-arch (2024)"}


def all_configs(curves: dict, out_path):
    """One figure comparing BOTH cross-scale curves across every configuration.

    curves[name] = {scales:[M params], algo:[x], data:[x], algo_note, gap_note}.
    Two panels (algorithm | data multiplier) share the model-size axis; each
    line is one lineage. Gaps (a scale with no validated recipe) are simply
    absent points; censoring/gaps are called out in the footnote, not faked.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=150,
                                   sharex=True)
    for ax, key, ttl in ((ax1, "algo", "Algorithm multiplier"),
                         (ax2, "data", "Data multiplier")):
        for name, c in curves.items():
            col = CURVE_COLORS[name]
            xs, ys = c["scales"], c[key]
            ax.plot(xs, ys, color=col, marker="o", markersize=7, linewidth=2,
                    label=CURVE_LABELS[name] if ax is ax1 else None)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}×", xy=(x, y), xytext=(0, 8),
                            textcoords="offset points", fontsize=8.5,
                            color=INK, ha="center")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([124, 355, 1536])
        ax.set_xticklabels(["124M", "355M", "1.5B"])
        ax.set_xlabel("Model size (log scale)")
        ax.set_title(ttl, fontsize=11, loc="left")
        _style(ax)
        ax.set_ylim(0.9, max(20, ax.get_ylim()[1]))
    ax1.set_ylabel("Compute-reduction multiplier (log scale)")
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper right")
    fig.suptitle("Compute-equivalent gain across all configurations",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    note = ("Each curve stops where it has no validated recipe: current-arch "
            "has no 1.5B point, ScaleUp has no 355M point (both disclosed "
            "gaps).  ScaleUp algorithm multiplier is on DCLM (new) data; on "
            "OpenWebText it is censored ≤1× at both scales (ScaleUp < GPT-2 on "
            "old data).  Multipliers are within-hardware GPU-hour ratios "
            "(current-arch 8-GPU, ScaleUp 5-GPU); the count overhead cancels in "
            "each ratio.")
    fig.text(0.012, 0.02, note, fontsize=7.4, color=INK2, va="bottom",
             wrap=True)
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def _assemble_all_configs(root: Path):
    """Build the all_configs curves dict from the canonical CEG JSONs."""
    sm = json.loads((root / "small" / "ceg_newdef.json").read_text())
    md = json.loads((root / "medium" / "ceg_newdef.json").read_text())
    su = json.loads((root / "scaleup" / "ceg_124m_matrix.json").read_text())
    xl = json.loads((root / "xl" / "ceg_1p5b_matrix.json").read_text())
    return {
        "current-arch": {
            "scales": [124, 355],
            "algo": [sm["multipliers"]["algorithm"], md["multipliers"]["algorithm"]],
            "data": [sm["multipliers"]["data"], md["multipliers"]["data"]],
        },
        "scaleup": {
            "scales": [124, 1536],
            "algo": [su["shapley"]["algo_DCLM_x"], xl["shapley_pieces"]["algo_D1col_x"]],
            "data": [su["shapley"]["data_A0row_x"], xl["shapley_pieces"]["data_A0row_x"]],
        },
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="render one figure type")
    ap.add_argument("kind", choices=["curves", "cross_scale", "sensitivity",
                                     "all_configs"])
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
    elif a.kind == "all_configs":
        all_configs(_assemble_all_configs(Path("results")), a.out)
    else:
        threshold_sensitivity(a.csv, a.size_label, a.out)
    print(f"wrote {a.out}")
