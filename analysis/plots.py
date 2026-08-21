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
            # ScaleUp A1D0 (ScaleUp on OWT) never does — bold + arm-colored end
            # label so a zoomed panel can't be misread as a crop.
            crossed = min(bs) <= thr
            ax.annotate(arm.upper(), xy=(hs[-1], bs[-1]), xytext=(5, 0),
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
        ax.set_xlabel("GPU-hours")
        ax.set_ylabel("Neutral BPB")
        ax.set_title(label, fontsize=10.5, loc="left")
        _style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5, labelcolor=INK2,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("BPB vs GPU-hours to reference threshold for all "
                 "configurations", fontsize=12.5, x=0.012, ha="left", color=INK,
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def multipliers_vs_scale(curves: dict, out_path, censored: dict = None):
    """Summary figure: cross-scale CEG multipliers across scale, ALL lineages.

    curves[name] = {scales:[M params], algo:[x], data:[x]} — numeric multiplier
    lineages (current-arch, ScaleUp). Two panels (algorithm | data) share the
    model-size axis. AGGREGATES the 16 arms into Shapley multipliers.

    censored[name] = {label, color, scales:[M params], annot:{size:txt}} — Exp B
    architecture lineages (Pythia, SmolLM2, later Mamba/Mamba2). These are a THIRD
    estimator: algorithm-CEG vs a MATCHED GPT-2 baseline, data held fixed (OWT) so
    NO data-panel entry. Every Exp B arm is CENSORED (its converged neutral BPB
    stays above the matched-GPT-2 threshold -> never crosses -> algorithm-CEG <=1x),
    so it carries no numeric multiplier; drawn as a hollow down-triangle at the 1x
    parity line on the ALGORITHM panel only. SmolLM2-135M is annotated as within
    same-seed noise of parity (still censored, not a crossing).
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
        # Exp B censored lineages: algorithm panel only, hollow v at 1x parity line
        if censored and ax is ax1:
            ax.axhline(1.0, color=GRID, lw=1.2, ls=(0, (4, 3)), zorder=1)
            ax.annotate("1× = parity with matched GPT-2 (no algorithm gain)",
                        xy=(600, 1.0), xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7.3, color=INK2)
            for lname, e in censored.items():
                ax.scatter(e["scales"], [1.0] * len(e["scales"]), marker="v",
                           s=85, facecolors="none", edgecolors=e["color"],
                           linewidths=1.8, zorder=4, label=e["label"])
                for x, txt in e.get("annot", {}).items():
                    ax.annotate(txt, xy=(x, 1.0), xytext=(-6, 12),
                                textcoords="offset points", ha="right", va="bottom",
                                fontsize=6.8, color=e["color"],
                                arrowprops=dict(arrowstyle="-", color=e["color"], lw=0.7))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([124, 355, 1536])
        ax.set_xticklabels(["124M", "355M", "1.5B"])
        ax.set_xlabel("Model size")
        ax.set_title(ttl, fontsize=11, loc="left")
        _style(ax)
        ax.set_ylim(0.82, max(20, ax.get_ylim()[1]))
    ax1.set_ylabel("Compute-reduction multiplier")
    ax1.legend(frameon=False, fontsize=7.4, labelcolor=INK2, loc="upper right")
    fig.suptitle("Compute-equivalent-gain multipliers vs model scale",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    note = ("Different estimators — NOT directly comparable.  current-arch "
            "(solid, filled) = true 2-ordering Shapley;  ScaleUp (dashed, open sq.) "
            "= single censored margin (A1D0 never crosses).  Exp B (open ▽ at 1×) = "
            "algorithm-CEG vs a MATCHED GPT-2, data fixed (no data panel); every arm "
            "is censored (converged BPB stays above the GPT-2 threshold → ≤1×, no algo "
            "gain).  SmolLM2 360M/1.7B are additionally divergence-confounded.")
    fig.text(0.012, 0.02, note, fontsize=7.2, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def core_expb_delta_vs_scale(root: Path, out_path):
    """Exp B CORE (downstream tasks) as a pairwise gap vs scale — the downstream
    analog of multipliers_vs_scale. y = mean CORE accuracy of the candidate MINUS
    its matched GPT-2 (both through the same harness), x = model size; one line per
    lineage. Parity line at 0. Unlike the core study (a 4-arm grid), Exp B is a
    pairwise candidate-vs-baseline comparison, so we plot the GAP, not absolute acc.
    Secondary to BPB; limit=500, ±1 stderr shown; reads results/core_expb_summary.json."""
    S = json.loads((root / "core_expb_summary.json").read_text())["pairs"]
    PARAMS = {"pythia-160M": 162, "pythia-410M": 405, "pythia-1.4B": 1415,
              "smollm2-135M": 135, "smollm2-360M": 362, "smollm2-1.7B": 1711}
    LIN = {"pythia": ("Pythia (GPT-NeoX, 2023)", "#1baf7a"),
           "smollm2": ("SmolLM2 (Llama, 2024)", "#8e44ad")}
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=150)
    _style(ax); ax.set_xscale("log")
    ax.axhline(0.0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("parity with matched GPT-2", xy=(1550, 0.0), xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=8, color=INK2)
    for lname, (label, col) in LIN.items():
        pts = sorted(((PARAMS[k], v) for k, v in S.items() if k.startswith(lname)))
        xs = [p for p, _ in pts]; ys = [v["mean_delta"] for _, v in pts]
        es = [v["mean_delta_stderr"] for _, v in pts]
        ax.errorbar(xs, ys, yerr=es, color=col, lw=2, marker="o", ms=8,
                    markerfacecolor=col, markeredgecolor=SURFACE, markeredgewidth=1.4,
                    capsize=3, elinewidth=1.2, label=label, zorder=3)
    ax.set_xticks([135, 400, 1500]); ax.set_xticklabels(["~135M", "~400M", "~1.5B"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(115, 2050)
    ax.set_ylim(-0.035, 0.035)
    ax.set_xlabel("Model size (log)")
    ax.set_ylabel("CORE mean-accuracy gap vs matched GPT-2")
    ax.set_title("Exp B — downstream-task (CORE) gap vs matched GPT-2, by scale (data = OWT)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")
    fig.text(0.012, 0.012,
             "y = each architecture's mean accuracy across 11 CORE tasks MINUS its size-matched "
             "GPT-2's mean, both trained on OpenWebText and scored through the same harness "
             "(limit=500 examples/task). Above 0 = the architecture beats GPT-2 on downstream "
             "tasks. Error bars are ±1 standard error of that mean gap (per-task stderrs from "
             "lm-eval, combined across the 11 tasks). SECONDARY to the BPB/CEG result. Pythia is "
             "at/below parity and falls with scale (matching its BPB deficit); SmolLM2 sits "
             "modestly above parity at every scale (~1.7σ; wins 7 of 11 tasks at small size) — a "
             "downstream edge that neutral BPB, where SmolLM2 is only parity-or-worse, misses. "
             "Per-task breakdown: core_expb_by_task.png.",
             fontsize=7.0, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(out_path, facecolor=SURFACE); plt.close(fig)


def core_expb_by_task(root: Path, out_path):
    """Exp B CORE broken down BY TASK: one panel per task, showing each lineage's
    accuracy gap vs its matched GPT-2 (candidate - GPT-2) across model size. Companion
    to the aggregate core_expb_delta figure. Data = OWT (Exp B is OWT-only); limit=500;
    ±1 stderr per point (stderr of the difference = hypot of the two arms' task stderrs).
    Reads results/core_expb_summary.json (per_task deltas)."""
    S = json.loads((root / "core_expb_summary.json").read_text())["pairs"]
    PARAMS = {"pythia-160M": 162, "pythia-410M": 405, "pythia-1.4B": 1415,
              "smollm2-135M": 135, "smollm2-360M": 362, "smollm2-1.7B": 1711}
    LIN = {"pythia": ("Pythia", "#1baf7a"), "smollm2": ("SmolLM2", "#8e44ad")}
    tasks = ["arc_easy", "arc_challenge", "openbookqa", "hellaswag", "commonsense_qa",
             "boolq", "copa", "piqa", "winogrande", "xwinograd_en", "lambada_openai"]
    ncol = 4
    nrow = (len(tasks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(12.5, 2.5 * nrow), dpi=150, sharey=True)
    axf = axes.ravel()
    for i, task in enumerate(tasks):
        ax = axf[i]
        ax.axhline(0.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for lname, (label, col) in LIN.items():
            pts = sorted(((PARAMS[k], v["per_task"].get(task, {})) for k, v in S.items()
                          if k.startswith(lname)))
            xs = [p for p, d in pts if d]
            ys = [d["delta"] for p, d in pts if d]
            ax.plot(xs, ys, color=col, marker="o", ms=5, lw=1.6, label=label, zorder=3,
                    markeredgecolor=SURFACE, markeredgewidth=0.8)
        ax.set_xscale("log"); ax.set_xticks([135, 400, 1500])
        ax.set_xticklabels(["135M", "400M", "1.5B"], fontsize=7.5)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.10, 0.10)
        ax.set_title(task, fontsize=9.5, loc="left")
        _style(ax)
    for j in range(len(tasks), len(axf)):
        axf[j].axis("off")
    axf[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
    for r in range(nrow):
        axes[r, 0].set_ylabel("Δ acc vs GPT-2", fontsize=8.5)
    fig.suptitle("Exp B — CORE accuracy gap vs matched GPT-2, per task (data = OWT)",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.008,
             "Each panel: candidate accuracy minus its size-matched GPT-2 on that task "
             "(>0 = candidate better), vs model size, per lineage. limit=500, so single "
             "tasks are noisy — read the consistency across tasks, not any one panel. "
             "SmolLM2 (purple) is positive on most tasks at most sizes; Pythia (green) "
             "scatters around/below zero. Aggregate + significance in core_expb_delta.png.",
             fontsize=7.2, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_path, facecolor=SURFACE); plt.close(fig)


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


def _assemble_expb(root: Path):
    """Exp B architecture lineages for the algorithm panel of multipliers_vs_scale.
    All arms are CENSORED (never cross their matched-GPT-2 threshold -> algorithm-CEG
    <=1x), so no numeric multiplier — we return sizes (M params) + the
    within-noise-of-parity annotation. Returns None if b1_results.json is absent."""
    p = root / "b1_results.json"
    if not p.exists():
        return None
    R = json.loads(p.read_text())
    sigma = R.get("noise_sigma", 0.013)
    spec = {"pythia": ("Pythia (GPT-NeoX) — vs matched GPT-2 (all censored ≤1×)", "#1baf7a"),
            "smollm2": ("SmolLM2 (Llama) — vs matched GPT-2 (all censored ≤1×)", "#8e44ad")}
    out = {}
    for key, (label, color) in spec.items():
        if key not in R:
            continue
        entries = R[key]
        order = sorted(entries, key=lambda s: entries[s]["params"])
        scales, annot = [], {}
        for s in order:
            pm = entries[s]["params"] / 1e6
            scales.append(pm)
            if abs(entries[s].get("delta", 1)) < sigma:
                annot[pm] = "within noise\nof parity"
        out[key] = {"label": label, "color": color, "scales": scales, "annot": annot}
    return out


# the 6 CORE tasks usable at some size; fixed distinct hues (lines are also
# direct-labeled, so color is never the sole identifier).
TASK_COLORS = {
    "arc_easy": "#2a78d6", "boolq": "#eb6834", "copa": "#1baf7a",
    "hellaswag": "#8e44ad", "piqa": "#eda100", "xwinograd_en": "#c0392b",
}


def core_vs_scale(root: Path, out_path):
    """CORE task accuracy vs model size — TWO tracks, matching the report's two
    curves and their disclosed gaps (never one spliced line).

    Plots the A0D0 (old-algo/old-data) GPT-2 baseline accuracy — the gate
    reference — per usable task, split by track:
      * current-arch: 124M -> 355M (solid; this curve has no validated 1.5B).
      * ScaleUp:      124M -> 1.5B (dashed; skips 355M, its disclosed gap).
    The 124M A0 CORE is the SAME shared GPT-2 baseline for both tracks, so they
    share that node (no false 355M->1.5B splice across tracks). Filled marker =
    passed the 2σ>chance validity gate at that size; hollow = below it. Accuracy
    is a different unit from BPB, so this stays its own figure.
    """
    from matplotlib.lines import Line2D
    g = json.loads((root / "core_gate_v2.json").read_text())
    tracks = [("current-arch", "-", [("124M", 124), ("355M", 355)]),
              ("ScaleUp", (0, (5, 3)), [("ScaleUp-124M", 124), ("1.5B", 1536)])]
    tasks = sorted(set(t for sc in ("124M", "355M", "1.5B", "ScaleUp-124M")
                       for t in g[sc]["usable_tasks"]))

    def pt(scale, task):
        ga = g[scale]["gate"].get(task, {})
        return ga.get("a0d0_acc"), bool(ga.get("usable"))

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=150)
    for ch in (0.25, 0.50):
        ax.axhline(ch, color=GRID, linewidth=1, linestyle=(0, (1, 3)), zorder=1)
        ax.annotate(f"chance {ch:.2f}", xy=(118, ch), xytext=(0, 2),
                    textcoords="offset points", fontsize=7.5, color=INK2,
                    va="bottom")
    for task in tasks:
        col = TASK_COLORS.get(task, INK2)
        for _, ls, nodes in tracks:
            pts = [pt(s, task) for s, _ in nodes]
            ax.plot([x for _, x in nodes], [p[0] for p in pts], color=col,
                    linewidth=1.8, linestyle=ls, zorder=3)
            for (_, x), (y, u) in zip(nodes, pts):
                ax.plot(x, y, marker="o", markersize=7, color=col, zorder=4,
                        markerfacecolor=(col if u else SURFACE),
                        markeredgecolor=col, markeredgewidth=1.6)
    ax.set_xscale("log")
    ax.set_xticks([124, 355, 1536]); ax.set_xticklabels(["124M", "355M", "1.5B"])
    ax.set_xlim(112, 2050)
    ax.set_ylabel("CORE task accuracy — A0D0 baseline")
    ax.set_xlabel("Model size")
    _style(ax)
    enc = [Line2D([], [], color=INK2, linestyle="-", label="current-arch (→355M)"),
           Line2D([], [], color=INK2, linestyle=(0, (5, 3)), label="ScaleUp (→1.5B)"),
           Line2D([], [], color=INK2, marker="o", linestyle="none", markersize=7,
                  label="passed gate"),
           Line2D([], [], color=INK2, marker="o", linestyle="none", markersize=7,
                  markerfacecolor=SURFACE, markeredgecolor=INK2, label="below gate")]
    ax.legend(handles=enc, frameon=False, fontsize=7.5, labelcolor=INK2,
              loc="lower right", ncol=2, columnspacing=1.2)
    tasklg = [Line2D([], [], color=TASK_COLORS[t], linewidth=2.6, label=t)
              for t in tasks]
    fig.legend(handles=tasklg, frameon=False, fontsize=8, labelcolor=INK2,
               loc="upper center", ncol=6, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("CORE task accuracy vs model size", fontsize=12.5, x=0.012,
                 ha="left", color=INK, y=0.985)
    fig.text(0.012, 0.008,
             "A0D0 (old-algo / old-data) GPT-2 baseline accuracy per task, "
             "limit=500 (the gate reference). Two tracks match the report's two "
             "curves: solid = current-arch (124M→355M, no 1.5B point); dashed = "
             "ScaleUp (124M→1.5B, skipping its 355M gap); the 124M node is the "
             "shared baseline. Filled = passed the 2σ>chance validity gate; "
             "hollow = below it (e.g. boolq sits at chance at 124M, clears from "
             "355M). Chance differs by task (0.25 four-way; 0.50 binary). "
             "Secondary metric — BPB is primary.", fontsize=7.0, color=INK2,
             va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.085, 1, 0.88))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


# (scale, arm) -> core_finals JSON stem — mirrors analysis/core_gate.py SCALES
# (A0 at ScaleUp-124M is the shared 124M baseline, so only 124M/355M/1.5B here).
_CORE_FILES = {
    ("124M", "a0d0"): "small_a0d0_dense_ckpt_016925",
    ("124M", "a0d1"): "small_a0d1_ckpt_016925",
    ("124M", "a1d0"): "small_a1d0_2x_v2_ckpt_002780",
    ("124M", "a1d1"): "small_a1d1_2x_v2_ckpt_002780",
    ("355M", "a0d0"): "medium_a0d0_ckpt_016925",
    ("355M", "a0d1"): "medium_a0d1_ckpt_016925",
    ("355M", "a1d0"): "medium_a1d0_v2_ckpt_009480",
    ("355M", "a1d1"): "medium_a1d1_v2_ckpt_009480",
    ("1.5B", "a0d0"): "xl_a0d0_ckpt_017325",
    ("1.5B", "a0d1"): "xl_a0d1_ckpt_017325",
    ("1.5B", "a1d0"): "xl_a1d0_ckpt_020343",
    ("1.5B", "a1d1"): "xl_a1d1_ckpt_020343",
}
_CORE_CHANCE = {"arc_easy": 0.25, "hellaswag": 0.25, "boolq": 0.50, "copa": 0.50,
                "piqa": 0.50, "winogrande": 0.50, "xwinograd_en": 0.50}


def core_arms_by_task(root: Path, out_path):
    """All FOUR arms' CORE accuracy per gate-usable task, with ±1 stderr bars.

    A QUALITATIVE companion to the BPB result — explicitly NOT a second
    quantitative CEG claim. One panel per gated task; within a panel the four
    arms are shown at 124M / 355M / 1.5B as points with error bars (no
    connecting lines — the arms are compared *within* each scale, and the noise
    is the point). The A1 (new-algorithm) arm is the current modded speedrun at
    124M/355M and the 2024-ScaleUp arch at 1.5B, so it is not one track across
    scales; that's fine here because nothing is connected across scale.
    """
    from matplotlib.lines import Line2D
    gate = json.loads((root / "core_gate_v2.json").read_text())
    data = {k: json.loads((root / "core_finals" / f"{v}.json").read_text())
            for k, v in _CORE_FILES.items()}
    scales = ["124M", "355M", "1.5B"]
    arms = ["a0d0", "a0d1", "a1d0", "a1d1"]
    offs = {"a0d0": -0.27, "a0d1": -0.09, "a1d0": 0.09, "a1d1": 0.27}
    tasks = sorted(set(t for sc in ("124M", "355M", "1.5B", "ScaleUp-124M")
                       for t in gate[sc]["usable_tasks"]))
    ncol = 3
    nrow = (len(tasks) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.6, 3.5 * nrow), dpi=150)
    axf = axes.ravel()
    for i, task in enumerate(tasks):
        ax = axf[i]
        ch = _CORE_CHANCE.get(task)
        if ch is not None:
            ax.axhline(ch, color=GRID, linewidth=1, linestyle=(0, (1, 3)), zorder=1)
        for si, sc in enumerate(scales):
            for arm in arms:
                d = data[(sc, arm)].get(task, {})
                acc, se = d.get("acc,none"), d.get("acc_stderr,none")
                if acc is None:
                    continue
                ax.errorbar(si + offs[arm], acc, yerr=se, marker="o",
                            markersize=5, color=ARM_COLORS[arm],
                            ecolor=ARM_COLORS[arm], elinewidth=1.1, capsize=2.5,
                            linestyle="none", zorder=3)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(scales, fontsize=8.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(task, fontsize=10, loc="left")
        _style(ax)
        if i % ncol == 0:
            ax.set_ylabel("Accuracy", fontsize=9)
    for j in range(len(tasks), len(axf)):
        axf[j].set_visible(False)
    lg = [Line2D([], [], color=ARM_COLORS[a], marker="o", linestyle="none",
                 markersize=6, label=ARM_LABELS[a]) for a in arms]
    fig.legend(handles=lg, frameon=False, fontsize=8, labelcolor=INK2,
               loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("CORE accuracy across all four arms", fontsize=12.5, x=0.012,
                 ha="left", color=INK, y=0.99)
    fig.text(0.012, 0.006,
             "Each panel is one gate-usable task; points are the four arms at "
             "124M / 355M / 1.5B with ±1 stderr bars taken directly from lm-eval "
             "(≈0.021–0.022 per task, except copa ≈0.048 — it has only 100 "
             "examples vs 500).  This is a QUALITATIVE companion to the BPB "
             "result, NOT a second CEG claim — at this noise level most arm gaps "
             "overlap within error.  The clearest recurring hint is new-data (D1) "
             "arms edging "
             "out their old-data (D0) counterparts on arc_easy and piqa at every "
             "scale (sizable on arc_easy, within ~1–2 stderr on piqa); boolq "
             "shows no such pattern, so it is task-specific, not universal.  Read "
             "as directionally suggestive only — BPB remains the primary, "
             "decisive metric.  (A1 = current modded speedrun at 124M/355M, "
             "2024-ScaleUp at 1.5B; nothing is connected across scale.)",
             fontsize=7.0, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.11 if nrow == 2 else 0.07, 1, 0.90))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="render one figure type")
    ap.add_argument("kind", choices=["curves", "cross_scale", "sensitivity",
                                     "all_configs", "multipliers",
                                     "core_vs_scale", "core_arms_by_task",
                                     "core_expb", "core_expb_by_task"])
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
        multipliers_vs_scale(_assemble_all_configs(Path("results")), a.out,
                             censored=_assemble_expb(Path("results")))
    elif a.kind == "core_vs_scale":
        core_vs_scale(Path("results"), a.out)
    elif a.kind == "core_arms_by_task":
        core_arms_by_task(Path("results"), a.out)
    elif a.kind == "core_expb":
        core_expb_delta_vs_scale(Path("results"), a.out)
    elif a.kind == "core_expb_by_task":
        core_expb_by_task(Path("results"), a.out)
    else:
        threshold_sensitivity(a.csv, a.size_label, a.out)
    print(f"wrote {a.out}")
