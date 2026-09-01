"""Raw training-curve figures (the all_configs.png analog) for Exp A and Exp B.

These show the underlying BPB-vs-GPU-hours curves that the CEG multipliers are read
off — where each arm crosses (or never reaches) its threshold. Same house style as
plots.all_configs_curves.

Outputs:
  results/expb_arch_curves.png  — Exp B architecture axis: each candidate vs its
                                   matched GPT-2-OWT threshold, one panel per lineage.
  results/expb_data_curves.png  — Exp B data axis: 6 data-ladder arms vs the shared
                                   GPT-2-OWT bar, one panel per architecture.
  results/era_curves.png        — Exp A: the 8 era arms vs the union-eval threshold,
                                   one panel per corpus (old-algo + current-arch).
"""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plots import SURFACE, INK, INK2, GRID, _style

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
BK = Path.home() / "Desktop" / "era_ladder_backup"

# one canonical definition, used on every figure that shows censored arms
CENSORED_DEF = ('“Censored” = the arm never reaches the threshold within its compute budget, '
                'so its compute-to-threshold is a bound, not a value (hollow markers).')


def curve(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    return [(float(r["gpu_hours"]), float(r["neutral_bpb"])) for r in rows
            if float(r["gpu_hours"]) > 0 and r["neutral_bpb"]]


def _panel(ax, series, thr, thr_label, title):
    """series = list of (label, csv_path, color). Draws BPB-vs-GPU-h + threshold.
    Legend-based (curves converge, so end-labels would collide); crossing arms get a
    filled-circle legend marker, non-crossing (censored) arms an open marker."""
    from matplotlib.lines import Line2D
    lo = thr; handles = []
    for label, path, color in series:
        if not Path(path).exists():
            continue
        pts = curve(path)
        if not pts:
            continue
        hs, bs = zip(*pts)
        lo = min(lo, min(bs))
        crossed = min(bs) <= thr
        ax.plot(hs, bs, color=color, lw=1.8, marker="o", ms=3)
        handles.append(Line2D([], [], color=color, lw=1.8, marker="o", ms=6,
                              markerfacecolor=(color if crossed else SURFACE),
                              label=label + ("" if crossed else "  (censored)")))
    ax.axhline(thr, color=INK2, lw=1.1, ls=(0, (4, 3)))
    ax.annotate(thr_label, xy=(0.01, thr), xycoords=("axes fraction", "data"),
                xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(lo - 0.03, thr + 0.45)
    ax.set_xlabel("GPU-hours (log)")
    ax.set_ylabel("Neutral BPB")
    ax.set_title(title, fontsize=10.5, loc="left")
    _style(ax)
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")


# ---------- Exp B architecture axis: one panel per size, candidate vs its matched GPT-2 ----------
def expb_arch():
    B1 = BK / "b1_cand" / "metrics"
    THR = json.load(open(RES / "b1_results.json"))["denominators_train_hf_512k"]
    ARCH, GPTCOL = "#8e44ad", "#2a78d6"   # candidate purple, GPT-2 blue (per panel)
    # (title, candidate csv, gpt2 csv, threshold key)
    panels = [
        ("Pythia-160M", "r512k_pythia160m", "r512k_gpt2b160", "b160"),
        ("Pythia-410M", "r512k_pythia410m", "r512k_gpt2b410", "b410"),
        ("Pythia-1.4B", "r512k_pythia1_4b", "r512k_gpt2b1400", "b1400"),
        ("SmolLM2-135M", "r512k_smollm2", "gpt2_b135_conv512k", "b135"),
        ("SmolLM2-360M", "r512k_smollm2_360m", "r512k_gpt2b360", "b360"),
        ("SmolLM2-1.7B", "r512k_smollm2_1_7b", "r512k_gpt2b1700", "b1700"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=150)
    for ax, (title, cand, gpt2, tk) in zip(axes.ravel(), panels):
        _panel(ax, [(title, f"{B1}/{cand}/metrics.csv", ARCH),
                    ("matched GPT-2", f"{B1}/{gpt2}/metrics.csv", GPTCOL)],
               THR[tk], f"GPT-2 bar {THR[tk]:.3f}", title)
    fig.suptitle("Exp B architecture axis — BPB vs GPU-hours (OWT); each arch vs its size-matched GPT-2",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.005, CENSORED_DEF, fontsize=8, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(RES / "expb_arch_curves.png", facecolor=SURFACE); plt.close(fig)
    print("wrote", RES / "expb_arch_curves.png")


# ---------- Exp B data axis (6 data-ladder arms vs GPT-2-OWT bar) ----------
def expb_data():
    LAD = RES / "data_ladder_metrics"
    B1 = BK / "b1_cand" / "metrics"
    thr_py, thr_sm = 1.251829, 1.272076  # matched GPT-2-OWT bars (b160, b135)
    PYCOL, SMCOL = "#1baf7a", "#8e44ad"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), dpi=150)
    _panel(ax1, [
        ("OWT", f"{B1}/r512k_pythia160m/metrics.csv", "#9e9e9e"),
        ("C4", f"{LAD}/dl_pythia160m_c4/metrics.csv", "#eda100"),
        ("RefinedWeb", f"{LAD}/dl_pythia160m_refinedweb/metrics.csv", "#2a78d6"),
        ("DCLM", f"{LAD}/dl_pythia160m_dclm/metrics.csv", "#008300"),
    ], thr_py, f"GPT-2-OWT bar {thr_py:.3f}", "Pythia-160M across data corpora")
    _panel(ax2, [
        ("OWT", f"{B1}/r512k_smollm2/metrics.csv", "#9e9e9e"),
        ("C4", f"{LAD}/dl_smollm2_135m_c4/metrics.csv", "#eda100"),
        ("RefinedWeb", f"{LAD}/dl_smollm2_135m_refinedweb/metrics.csv", "#2a78d6"),
        ("DCLM", f"{LAD}/dl_smollm2_135m_dclm/metrics.csv", "#008300"),
    ], thr_sm, f"GPT-2-OWT bar {thr_sm:.3f}", "SmolLM2-135M across data corpora")
    fig.suptitle("Exp B data axis — BPB vs GPU-hours; better data (RefinedWeb/DCLM) crosses the GPT-2-OWT bar",
                 fontsize=12, x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.005, CENSORED_DEF, fontsize=8, color=INK2, va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(RES / "expb_data_curves.png", facecolor=SURFACE); plt.close(fig)
    print("wrote", RES / "expb_data_curves.png")


# ---------- Exp A (8 era arms vs union-eval threshold, per corpus) ----------
def era():
    THR = json.load(open(RES / "era_correction_2x2.json"))["corrected_threshold_neutral_bpb"]
    RET = RES / "era_retrain_metrics"          # C4, RefinedWeb (retrain)
    RM = BK / "run_metrics"                     # original OWT/DCLM era arms
    OLD, NEW = "#2a78d6", "#eb6834"
    # (corpus, old-algo csv, new-algo csv)
    panels = [
        ("OWT (2019)", f"{RM}/era_a0d0_owt/metrics.csv", f"{RM}/era_a1d0_owt/metrics.csv"),
        ("C4 (2020)", f"{RET}/era_a0_c4/metrics.csv", f"{RET}/era_a1_c4/metrics.csv"),
        ("RefinedWeb (2023)", f"{RET}/era_a0_refinedweb/metrics.csv", f"{RET}/era_a1_refinedweb/metrics.csv"),
        ("DCLM (2024)", f"{RM}/era_a0d1_dclm/metrics.csv", f"{RM}/era_a1d1_dclm/metrics.csv"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), dpi=150)
    for ax, (title, pold, pnew) in zip(axes.ravel(), panels):
        _panel(ax, [("old-algo", pold, OLD), ("current-arch", pnew, NEW)],
               THR, f"threshold {THR:.3f}", title)
    fig.suptitle("Exp A — BPB vs GPU-hours per corpus (old-algo vs current-arch, 124M)",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RES / "era_curves.png", facecolor=SURFACE); plt.close(fig)
    print("wrote", RES / "era_curves.png")


if __name__ == "__main__":
    expb_arch(); expb_data(); era()
