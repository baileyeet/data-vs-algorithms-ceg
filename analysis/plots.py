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
# the two curves are DIFFERENT estimators (see multipliers_vs_scale): current-
# arch = symmetric 2-ordering Shapley (all 4 cells cross); ScaleUp = a single
# computable margin (its A1D0/ScaleUp-on-OWT cell never crosses, so the
# symmetric average is undefined). Distinguish them by style, not just color.
CURVE_LABELS = {
    "current-arch": "current-arch — Shapley (avg of both orderings)",
    "scaleup": "ScaleUp — single margin (complement cell censored)"}
CURVE_STYLE = {"current-arch": dict(linestyle="-", marker="o", fill=True),
               "scaleup": dict(linestyle=(0, (5, 3)), marker="s", fill=False)}


# the 16 arms = 4 scale-points x 4 arms. Each panel is one scale-point; canonical
# per-arm metrics.csv (v2 for A1) + its neutral-BPB threshold.
SCALE_PANELS = [
    ("current-arch 124M", 1.274421, "small",
     {"a0d0": "small_a0d0_dense_metrics.csv", "a0d1": "small_a0d1_metrics.csv",
      "a1d0": "small_a1d0_2x_v2_metrics.csv", "a1d1": "small_a1d1_2x_v2_metrics.csv"}),
    ("current-arch 355M", 1.228738, "medium",
     {"a0d0": "medium_a0d0_metrics.csv", "a0d1": "medium_a0d1_metrics.csv",
      "a1d0": "medium_a1d0_v2_metrics.csv", "a1d1": "medium_a1d1_v2_metrics.csv"}),
    ("ScaleUp 124M", 1.280044, "scaleup",
     {"a0d0": "su124_a0d0_5gpu_metrics.csv", "a0d1": "su124_a0d1_5gpu_metrics.csv",
      "a1d0": "su124_a1d0_metrics.csv", "a1d1": "su124_a1d1_metrics.csv"}),
    ("ScaleUp 1.5B", 1.18792, "xl",
     {"a0d0": "xl_a0d0_metrics.csv", "a0d1": "xl_a0d1_metrics.csv",
      "a1d0": "xl_a1d0_metrics.csv", "a1d1": "xl_a1d1_metrics.csv"}),
]


def _read_curve(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    return [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
            if float(r["gpu_hours"]) > 0]


def all_configs_curves(root: Path, out_path):
    """All 16 configurations on one image: a 2x2 small-multiples of BPB-vs-
    GPU-hours, one panel per scale-point, all 4 arms per panel + the reference
    threshold. Zoomed to the threshold-crossing region. Arm color follows the
    entity (same hue in every panel). Non-crossing arms simply never reach the
    dashed line — not hidden, not faked."""
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), dpi=150)
    for ax, (label, thr, sub, arms) in zip(axes.ravel(), SCALE_PANELS):
        lo = thr
        for arm in ["a0d0", "a0d1", "a1d0", "a1d1"]:
            p = root / sub / arms[arm]
            if not p.exists():
                continue
            pts = _read_curve(p)
            if not pts:
                continue
            hs, bs = zip(*pts)
            lo = min(lo, min(bs))
            ax.plot(hs, bs, color=ARM_COLORS[arm], linewidth=1.8, marker="o",
                    markersize=3, label=ARM_LABELS[arm])
            # an arm 'crosses' iff its BPB curve reaches the threshold; the
            # ScaleUp A1D0 (ScaleUp on OWT) never does — flag it explicitly so a
            # zoomed panel can't be misread as a crop.
            crossed = min(bs) <= thr
            lbl = arm.upper() if crossed else f"{arm.upper()} — never crosses"
            ax.annotate(lbl, xy=(hs[-1], bs[-1]), xytext=(5, 0),
                        textcoords="offset points", fontsize=7.5,
                        color=(INK if crossed else ARM_COLORS[arm]),
                        fontweight=("normal" if crossed else "bold"),
                        va="center")
        ax.axhline(thr, color=INK2, linewidth=1.1, linestyle=(0, (4, 3)))
        ax.annotate(f"ref BPB {thr:.3f}", xy=(0.01, thr),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        # headroom above the threshold so a non-crossing arm's plateau AND its
        # approach are visible (it sits above the dashed line), not just the
        # crossing zone.
        ax.set_ylim(lo - 0.03, thr + 0.40)
        ax.set_xlim(right=ax.get_xlim()[1] * 3.2)  # room for end labels
        ax.set_xlabel("GPU-hours (timed, log)")
        ax.set_ylabel("Neutral BPB")
        ax.set_title(label, fontsize=10.5, loc="left")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK2,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("All 16 configurations — BPB vs GPU-hours to the reference "
                 "threshold", fontsize=12.5, x=0.012, ha="left", color=INK,
                 y=0.99)
    fig.text(0.012, 0.005,
             "Each panel is one scale-point; its 4 arms are A0/A1 (old/new "
             "algorithm) × D0/D1 (old/new data). An arm 'crosses' where its "
             "curve meets the dashed reference BPB — that GPU-hour value is the "
             "raw material for the multipliers. The ScaleUp A1D0 (new-algo / "
             "old-data) never crosses at either scale (ScaleUp < GPT-2 on "
             "OpenWebText). GPU-hours are comparable within a panel (current-arch "
             "8-GPU, ScaleUp 5-GPU).", fontsize=7.2, color=INK2, va="bottom",
             wrap=True)
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def multipliers_vs_scale(curves: dict, out_path):
    """Summary figure: BOTH cross-scale curves' CEG multipliers across scale.

    curves[name] = {scales:[M params], algo:[x], data:[x]}. Two panels
    (algorithm | data multiplier) share the model-size axis; each line is one
    lineage. This AGGREGATES the 16 arms into the Shapley multipliers — see
    all_configs_curves for the per-arm detail. Gaps are simply absent points.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=150,
                                   sharex=True)
    for ax, key, ttl in ((ax1, "algo", "Algorithm multiplier"),
                         (ax2, "data", "Data multiplier")):
        for name, c in curves.items():
            col = CURVE_COLORS[name]
            st = CURVE_STYLE[name]
            xs, ys = c["scales"], c[key]
            ax.plot(xs, ys, color=col, marker=st["marker"], markersize=7,
                    linewidth=2, linestyle=st["linestyle"],
                    markerfacecolor=(col if st["fill"] else SURFACE),
                    markeredgecolor=col, markeredgewidth=1.6,
                    label=CURVE_LABELS[name] if ax is ax1 else None)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.1f}×", xy=(x, y), xytext=(0, 9),
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
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    fig.suptitle("Compute-equivalent-gain multipliers vs model scale",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    note = ("The two lines are DIFFERENT estimators — do not read them as "
            "directly comparable magnitudes.  Current-arch (solid, filled) = the "
            "symmetric 2-ordering Shapley value (all four cells cross, so both "
            "data orderings and both algorithm orderings are defined and "
            "averaged).  ScaleUp (dashed, open) = a single computable margin: its "
            "A1D0 (ScaleUp on OpenWebText) never crosses, so the complementary "
            "marginal is censored and no symmetric Shapley exists — data = the A0 "
            "(GPT-2) row ratio, algorithm = the D1 (DCLM) column ratio.  Each "
            "curve also stops where it has no validated recipe (current-arch: no "
            "1.5B; ScaleUp: no 355M).  Ratios are within-hardware GPU-hours "
            "(current-arch 8-GPU, ScaleUp 5-GPU); the count overhead cancels.")
    fig.text(0.012, 0.02, note, fontsize=7.0, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.20, 1, 0.94))
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


# the 6 CORE tasks usable at some size; fixed distinct hues (lines are also
# direct-labeled, so color is never the sole identifier).
TASK_COLORS = {
    "arc_easy": "#2a78d6", "boolq": "#eb6834", "copa": "#1baf7a",
    "hellaswag": "#8e44ad", "piqa": "#eda100", "xwinograd_en": "#c0392b",
}


def core_vs_scale(root: Path, out_path):
    """CORE task accuracy vs model size, one line per gate-usable task.

    Plots the A0D0 (old-algo/old-data) baseline accuracy at each size — the gate
    reference. Filled marker = the task passed the 2σ>chance validity gate at
    that size; hollow = below the gate (near chance). Visualizes findings like
    'boolq comes alive at 355M+' that were previously only in the text table.
    Accuracy is a different unit from BPB, so this stays its own figure.
    """
    from matplotlib.lines import Line2D
    g = json.loads((root / "core_gate_v2.json").read_text())
    sizes = [("124M", 124), ("355M", 355), ("1.5B", 1536)]
    xs = [x for _, x in sizes]
    tasks = sorted(set(t for s, _ in sizes for t in g[s]["usable_tasks"]))
    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=150)
    for ch in (0.25, 0.50):
        ax.axhline(ch, color=GRID, linewidth=1, linestyle=(0, (1, 3)), zorder=1)
        ax.annotate(f"chance {ch:.2f}", xy=(xs[0], ch), xytext=(0, 2),
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    va="bottom")
    for t in tasks:
        col = TASK_COLORS.get(t, INK2)
        ys = [g[s]["gate"].get(t, {}).get("a0d0_acc") for s, _ in sizes]
        use = [bool(g[s]["gate"].get(t, {}).get("usable")) for s, _ in sizes]
        ax.plot(xs, ys, color=col, linewidth=1.8, zorder=3)
        for x, y, u in zip(xs, ys, use):
            ax.plot(x, y, marker="o", markersize=7, color=col, zorder=4,
                    markerfacecolor=(col if u else SURFACE),
                    markeredgecolor=col, markeredgewidth=1.6)
        ax.annotate(t, xy=(xs[-1], ys[-1]), xytext=(7, 0),
                    textcoords="offset points", fontsize=8.5, color=col,
                    va="center")
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([s for s, _ in sizes])
    ax.set_xlim(108, 3200)
    ax.set_xlabel("Model size (log scale)")
    ax.set_ylabel("CORE task accuracy — A0D0 baseline")
    ax.set_title("CORE task accuracy vs model size", fontsize=11, loc="left")
    _style(ax)
    legend = [Line2D([], [], color=INK2, marker="o", linestyle="none",
                     markersize=7, label="passed validity gate"),
              Line2D([], [], color=INK2, marker="o", linestyle="none",
                     markersize=7, markerfacecolor=SURFACE,
                     markeredgecolor=INK2, label="below gate (near chance)")]
    ax.legend(handles=legend, frameon=False, fontsize=8, labelcolor=INK2,
              loc="upper left")
    fig.text(0.012, 0.01,
             "A0D0 (old-algo / old-data) accuracy per task, limit=500 (the gate "
             "reference). Only gate-passing (filled) tasks enter the quantitative "
             "CORE table; e.g. boolq sits at chance at 124M (hollow) then clears "
             "from 355M up. Chance differs by task (0.25 four-way; 0.50 binary). "
             "Secondary metric — BPB is primary.", fontsize=7.0, color=INK2,
             va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="render one figure type")
    ap.add_argument("kind", choices=["curves", "cross_scale", "sensitivity",
                                     "all_configs", "multipliers",
                                     "core_vs_scale"])
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
        all_configs_curves(Path("results"), a.out)
    elif a.kind == "multipliers":
        multipliers_vs_scale(_assemble_all_configs(Path("results")), a.out)
    elif a.kind == "core_vs_scale":
        core_vs_scale(Path("results"), a.out)
    else:
        threshold_sensitivity(a.csv, a.size_label, a.out)
    print(f"wrote {a.out}")
