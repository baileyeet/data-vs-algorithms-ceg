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
from plots import SURFACE, INK, INK2, GRID, _style, _savefig
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
# publication figures read ONLY in-repo data (no ~/Desktop/era_ladder_backup dependency)
B1M = RES / "b1_metrics"            # Exp B B1 arch-axis + GPT-2 denominator curves
ERAM = RES / "era_orig_metrics"     # original era arms (match the CEG JSON)

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


# ---------- Exp B architecture axis: 2x3 (rows = lineage, cols = increasing scale) ----------
def expb_arch():
    """Six matched comparisons: each modern architecture vs a GPT-2 trained through the
    IDENTICAL pipeline at the same size (the train_hf @512k denominator). Rows = lineage
    (Pythia / SmolLM2), columns = increasing scale. Shared BPB axis across all six so the
    small candidate-vs-GPT-2 gaps are NOT visually exaggerated. Lower BPB = better; the
    takeaway is that no candidate curve gets below its matched GPT-2 bar. For the two
    divergence-confounded SmolLM2 runs the best (minimum) BPB reached is marked, since the
    final BPB overshoots as the run over-fits OWT."""
    R = json.load(open(RES / "b1_results.json"))
    THR = R["denominators_train_hf_512k"]
    sig = R["noise_sigma"]
    ARCH = {"pythia": "#1baf7a", "smollm2": "#8e44ad"}   # lineage colors (match hero/data figs)
    GPTCOL = "#8a8a86"                                     # matched GPT-2 = neutral gray
    # rows of (lineage_key, size_key, title, candidate_run, gpt2_run, denom_key)
    rows = [
        [("pythia", "160M", "Pythia-160M", "r512k_pythia160m", "r512k_gpt2b160", "b160"),
         ("pythia", "410M", "Pythia-410M", "r512k_pythia410m", "r512k_gpt2b410", "b410"),
         ("pythia", "1.4B", "Pythia-1.4B", "r512k_pythia1_4b", "r512k_gpt2b1400", "b1400")],
        [("smollm2", "135M", "SmolLM2-135M", "r512k_smollm2", "gpt2_b135_conv512k", "b135"),
         ("smollm2", "360M", "SmolLM2-360M", "r512k_smollm2_360m", "r512k_gpt2b360", "b360"),
         ("smollm2", "1.7B", "SmolLM2-1.7B", "r512k_smollm2_1_7b", "r512k_gpt2b1700", "b1700")],
    ]
    # shared y-range = plateau/crossing region (early steep descent is off the top, on purpose)
    YLO, YHI = 1.15, 1.45
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), dpi=150, sharey=True)
    for r, row in enumerate(rows):
        for c, (lin, sz, title, cand, gpt2, tk) in enumerate(row):
            ax = axes[r, c]
            bar = THR[tk]
            info = R[lin][sz]
            # matched GPT-2 curve + its tail-mean bar
            for run, col, lbl in ((gpt2, GPTCOL, "matched GPT-2"), (cand, ARCH[lin], title)):
                pts = curve(f"{B1M}/{run}/metrics.csv")
                hs, bs = zip(*pts)
                ax.plot(hs, bs, color=col, lw=2.0, marker="o", ms=2.4,
                        label=lbl, zorder=3 if run == cand else 2)
            ax.axhline(bar, color=INK2, lw=1.1, ls=(0, (4, 3)), zorder=1)
            ax.annotate(f"GPT-2 bar {bar:.3f}", xy=(0.02, bar), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points", fontsize=7.5, color=INK2)
            # candidate min-BPB marker for the divergence-confounded runs
            cbs = [b for _, b in curve(f"{B1M}/{cand}/metrics.csv")]
            cmin = min(cbs)
            if info.get("confounded"):
                chs = [h for h, _ in curve(f"{B1M}/{cand}/metrics.csv")]
                hmin = chs[cbs.index(cmin)]
                ax.scatter([hmin], [cmin], marker="v", s=55, facecolors="none",
                           edgecolors=ARCH[lin], linewidths=1.6, zorder=5)
                ax.annotate(f"best {cmin:.3f}\n(rises after; diverges)", xy=(hmin, cmin),
                            xytext=(6, 10), textcoords="offset points", fontsize=7, color=ARCH[lin])
            # verdict box: delta vs bar (and min-based delta when confounded)
            d = info["delta"]
            txt = f"Δ = +{d:.3f} vs bar"
            if info.get("confounded"):
                txt += f"\n(best-case +{info['delta_min']:.3f})"
            elif abs(d) < sig:
                txt += "\n(within noise = parity)"
            ax.annotate(txt, xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                        fontsize=7.8, color=INK,
                        bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=GRID, lw=0.8))
            ax.set_xscale("log")
            ax.set_ylim(YLO, YHI)
            ax.set_title(title, fontsize=10.5, loc="left")
            if r == 1:
                ax.set_xlabel("GPU-hours (log)")
            if c == 0:
                ax.set_ylabel("Neutral BPB  (lower = better)")
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
            _style(ax)
    # row labels
    for r, name in enumerate(["Pythia (GPT-NeoX, 2023)", "SmolLM2 (Llama, 2024)"]):
        axes[r, 0].annotate(name, xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontsize=10.5, color=INK, fontweight="bold")
    fig.suptitle("Exp B — a large speedrun-style algorithmic advantage does NOT reproduce across six matched comparisons",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.006,
             "Each modern architecture vs a GPT-2 trained through the identical pipeline at the same size (OWT, 8.87B tokens, "
             "converged 512k-batch recipe). Lower BPB is better; no candidate reaches its matched GPT-2 bar. "
             + CENSORED_DEF, fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.96))
    _savefig(fig, RES / "expb_arch_curves.png")


# ---------- Exp B data-replication axis (data-ladder arms vs the GPT-2-OWT bar) ----------
def expb_data_replication():
    """Does the DATA effect reproduce across architectures? Two panels (one per lineage);
    within each, the same architecture trained on four corpora vs the fixed GPT-2-OWT bar
    (its size-matched train_hf @512k denominator). RefinedWeb/DCLM cross the bar under BOTH
    lineages, OWT/C4 stay censored under both -> the corpus effect reproduces across the two
    tested stacks (GPT-NeoX and Llama). Color follows the CORPUS entity (matches the Exp A
    corpus figure). In-repo data only."""
    LAD = RES / "data_ladder_metrics"
    thr_py, thr_sm = 1.251829, 1.272076   # matched GPT-2-OWT bars (b160, b135)
    CORP = {"OWT": "#8a8a86", "C4": "#eda100", "RefinedWeb": "#2a78d6", "DCLM": "#008300"}
    lineages = [
        ("Pythia-160M (GPT-NeoX)", thr_py, {
            "OWT": f"{B1M}/r512k_pythia160m/metrics.csv",
            "C4": f"{LAD}/dl_pythia160m_c4/metrics.csv",
            "RefinedWeb": f"{LAD}/dl_pythia160m_refinedweb/metrics.csv",
            "DCLM": f"{LAD}/dl_pythia160m_dclm/metrics.csv"}),
        ("SmolLM2-135M (Llama)", thr_sm, {
            "OWT": f"{B1M}/r512k_smollm2/metrics.csv",
            "C4": f"{LAD}/dl_smollm2_135m_c4/metrics.csv",
            "RefinedWeb": f"{LAD}/dl_smollm2_135m_refinedweb/metrics.csv",
            "DCLM": f"{LAD}/dl_smollm2_135m_dclm/metrics.csv"}),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), dpi=150, sharey=True)
    for ax, (title, bar, runs) in zip(axes, lineages):
        lo = bar
        handles = []
        for corpus in ["OWT", "C4", "RefinedWeb", "DCLM"]:
            pts = curve(runs[corpus])
            hs, bs = zip(*pts)
            lo = min(lo, min(bs))
            crossed = min(bs) <= bar
            ax.plot(hs, bs, color=CORP[corpus], lw=2.0, marker="o", ms=2.6)
            handles.append(Line2D([], [], color=CORP[corpus], lw=2.2, marker="o", ms=6,
                                  markerfacecolor=(CORP[corpus] if crossed else SURFACE),
                                  label=corpus + ("" if crossed else "  (censored)")))
        ax.axhline(bar, color=INK2, lw=1.2, ls=(0, (4, 3)))
        ax.annotate(f"matched GPT-2-OWT bar {bar:.3f}", xy=(0.02, bar),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_ylim(lo - 0.03, bar + 0.32)
        ax.set_xlabel("GPU-hours (log)")
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="upper right",
                  title="training corpus", title_fontsize=8)
        _style(ax)
    axes[0].set_ylabel("Neutral BPB  (lower = better)")
    fig.suptitle("Exp B — the data effect reproduces across both tested architecture stacks",
                 fontsize=12.5, x=0.012, ha="left", color=INK)
    fig.text(0.012, 0.006,
             "Same architecture, four corpora, vs its fixed size-matched GPT-2-OWT bar. RefinedWeb (2023) and DCLM (2024) "
             "cross under BOTH lineages; OWT (2019) and C4 (2020) stay censored under both. " + CENSORED_DEF,
             fontsize=7.5, color=INK2, va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    _savefig(fig, RES / "data_replication.png")


if __name__ == "__main__":
    expb_arch(); expb_data_replication()
