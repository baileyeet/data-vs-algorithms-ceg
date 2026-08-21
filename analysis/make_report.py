"""Assemble the final markdown report from per-size analysis outputs.

Expects a results directory laid out as (produced per RUNBOOK step 5-6):
  results/<size>/ceg.json            ceg_shapley.py --out-json (+ "n_params")
  results/<size>/sensitivity.csv     threshold_sensitivity.py output
  results/<size>/curves.png          plots.py curves
  results/<size>/sensitivity.png     plots.py sensitivity
  results/all_configs.png            plots.py all_configs (all 16 arms)
  results/multipliers_vs_scale.png   plots.py multipliers (both-curve summary)

Usage: python analysis/make_report.py --results-dir results --out report.md
"""

import argparse
import json
import math
from pathlib import Path

SIZE_ORDER = ["small", "medium", "large", "xl"]
ARMS = ["a0d0", "a0d1", "a1d0", "a1d1"]
ARM_DESC = {"a0d0": "old algo, old data", "a0d1": "old algo, new data",
            "a1d0": "new algo, old data (OWT)", "a1d1": "new algo, new data"}


def _load_ceg(d):
    # canonical = new-definition (final-10%-mean threshold), v2 for A1 arms;
    # fall back to the legacy single-final ceg.json only if newdef is absent.
    for name in ("ceg_newdef.json", "ceg.json"):
        p = d / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def size_section(size, d):
    r = _load_ceg(d)
    m, lg, C = r["multipliers"], r["shapley_log"], r["gpu_hours_to_threshold"]
    label = f"{r.get('n_params', 0)/1e6:.0f}M" if r.get("n_params") else size
    out = [f"## {label} ({size})",
           "",
           f"Reference threshold: **{r['threshold_neutral_bpb']:.4f} BPB** "
           f"(neutral corpus; = fully-trained A0D0 at this size).",
           "",
           "| Arm | | GPU-hours to threshold | crossing |",
           "|-----|--|------------------------|----------|"]
    for a in ARMS:
        out.append(f"| {a.upper()} | {ARM_DESC[a]} | {C[a]:.2f} | {r['crossing_type'][a]} |")
    prod = m["data"] * m["algorithm"]
    out += ["",
            f"**Shapley split (log-compute):** data {lg['data']:.3f}, "
            f"algorithm {lg['algorithm']:.3f} (sum {lg['total']:.3f}).",
            f"**As multipliers:** data contributes a **{m['data']:.2f}x** compute "
            f"reduction, algorithm a **{m['algorithm']:.2f}x**; product "
            f"{prod:.2f}x vs observed total **{m['total']:.2f}x** "
            f"({'consistent' if r['sanity_product_equals_total'] else 'MISMATCH — investigate'}).",
            ""]
    if (d / "curves.png").exists():
        out += [f"![training curves]({d.name}/curves.png)", ""]
    if (d / "sensitivity.png").exists():
        out += [f"![threshold sensitivity]({d.name}/sensitivity.png)", ""]
    return out, r


def _scaleup_section(root):
    """Curve 2: the ScaleUp-arch cross-scale curve (124M + 1.5B), from the two
    matrix jsons. Kept strictly separate from Curve 1 (different algorithm)."""
    p124 = root / "scaleup" / "ceg_124m_matrix.json"
    p15 = root / "xl" / "ceg_1p5b_matrix.json"
    if not (p124.exists() and p15.exists()):
        return []
    m124 = json.loads(p124.read_text())
    m15 = json.loads(p15.read_text())
    s124 = m124.get("shapley", {})
    s15 = m15.get("shapley_pieces", {})

    def algo_dclm(s):  # key differs between the two matrices
        return s.get("algo_DCLM_x") or s.get("algo_D1col_x")
    out = ["## Curve 2 — ScaleUp-arch (124M, 1.5B)", "",
           "A1 = the 2024 ScaleUp lineage, run from its DOCUMENTED per-size "
           "recipe (each size a coupled bundle: dims + batch + LR + schedule — "
           "NOT the 1.5B recipe with rescaled dims, which mis-tunes). A0 = the "
           "same GPT-2 baseline. Multipliers are gpu-hours ratios to the "
           "per-size threshold.", "",
           "| Scale | data x | algorithm (DCLM) | algorithm (OWT) |",
           "|--|--|--|--|",
           f"| 124M | {s124.get('data_A0row_x')}x | {algo_dclm(s124)}x | "
           f"{s124.get('algo_OWT', 'censored')} |",
           f"| 1.5B | {s15.get('data_A0row_x')}x | {algo_dclm(s15)}x | "
           f"{s15.get('algo_D0col', 'censored')} |",
           "",
           "The ScaleUp algorithm's advantage on new data **declines mildly with "
           "scale (2.90x -> 2.34x, 124M -> 1.5B)** — a gentle decay, versus the "
           "current arch's steep 13.1x -> 4.1x. And it is **data-dependent**: a "
           "real advantage on DCLM (new data), but NONE on OWT (old data) at "
           "either scale — the ScaleUp arm never crosses the threshold on OWT "
           "because GPT-2 matches/beats it there at equal budget (a genuine "
           "result, confirmed by equal-budget comparison, not undertraining). "
           "The data multiplier, by contrast, is roughly stable across scale "
           "(~3.3x).",
           "",
           "NOTE (hardware): the ScaleUp A1 arms and their GPT-2 A0 baseline were "
           "all measured on 5xH100 (the A0-124M baseline was re-run on 5 GPUs for "
           "this — an 8-vs-5 mix had distorted the 124M algo multiplier to 2.35x; "
           "the consistent value is 2.90x). GPU-hours is NOT cleanly count-"
           "invariant here (the forced batch/accum change cost ~22%).",
           "",
           "Reading of the two curves together: **both algorithms' advantages "
           "shrink with scale, but the more aggressively small-scale-tuned "
           "current speedrun decays far faster (from a much higher base) than the "
           "older, more fundamental ScaleUp (Muon + rotary).** **355M is a "
           "disclosed gap** for the ScaleUp lineage (no documented era-appropriate "
           "recipe; no hand-derived LR).", ""]
    return out


def _era_ladder_section(root):
    """Exp A: data-era ladder @124M. CEG (vs old-algo-OWT baseline) as a function of
    dataset release-year, for both old-algo and current-arch, plus the corrected
    OWT×DCLM 2x2. Reads era_correction_2x2.json + era_ladder_results.json."""
    cp = root / "era_correction_2x2.json"
    lp = root / "era_ladder_results.json"
    if not (cp.exists() and lp.exists()):
        return []
    C = json.loads(cp.read_text())
    L = json.loads(lp.read_text())
    m = C["cells_2x2"]["multipliers"]
    pub = C["comparison_to_published"]["published_124M"]
    pc = C["param_counts"]
    out = ["## Exp A — data-era ladder (@124M)", "",
           "Follow-on to the 2x2 study: hold the algorithm axis and sweep the DATA "
           "corpus across release-years — OWT (2019), C4 (2020), RefinedWeb (2023), "
           "DCLM (2024) — to see how data-quality and algorithm contributions move with "
           "dataset vintage. Wikipedia (union-decontam) is the neutral eval, never a "
           "train corpus; CEG is GPU-hours-to-threshold vs the old-algo-OWT baseline. "
           f"@124M = matched DIMENSIONS (12L/12H/768d), NOT param count (old-algo "
           f"{pc['old_algo']:,}; current-arch {pc['current_arch']:,} — the "
           "value-embed/U-net additions ARE the algorithm being measured).", "",
           f"Corrected 2x2 (union eval, all arms torch 2.10): threshold "
           f"**{C['corrected_threshold_neutral_bpb']:.4f} BPB**; **data {m['data']:.2f}×, "
           f"algorithm {m['algorithm']:.2f}×, total {m['total']:.2f}×** "
           f"(vs published {pub['data']}/{pub['algorithm']}/{pub['total']}× — the shift is "
           "union-eval + same-seed variance, no torch component; both are torch 2.10).", "",
           "| Dataset (year) | old-algo CEG | current-arch CEG | algorithm CEG at that corpus |",
           "|--|--|--|--|"]

    def fx(v):
        return "censored" if v is None else f"{v:.1f}×"
    for ds, e in sorted(L["datasets"].items(), key=lambda kv: kv[1]["release_year"]):
        out.append(f"| {ds} ({e['release_year']}) | {fx(e['old_algo']['ceg_vs_a0d0'])} | "
                   f"{fx(e['current_arch']['ceg_vs_a0d0'])} | {fx(e['algo_ceg_at_dataset'])} |")
    out += ["",
            "Data-quality is **NON-monotonic in release year**: C4 (2020) is CENSORED "
            "under BOTH algorithms (never reaches the OWT threshold — a WORSE training "
            "corpus than 2019 OWT), while RefinedWeb (2023) and DCLM (2024) do improve. "
            "So 'newer dataset' ≠ 'better data'. The algorithm CEG stays large across "
            "corpora (see the OWT/RefinedWeb/DCLM column).", ""]
    if (root / "era_ladder.png").exists():
        out += ["![Exp A: CEG vs dataset release-year](era_ladder.png)", ""]
    return out


def _expb_section(root):
    """Exp B: architecture landscape. Published Transformer lineages (Pythia, SmolLM2)
    vs a size-matched GPT-2, data fixed (OWT), algorithm-CEG only (no 2x2). Reads
    b1_results.json. Exp B also appears as censored markers on multipliers_vs_scale.png."""
    p = root / "b1_results.json"
    if not p.exists():
        return []
    R = json.loads(p.read_text())
    out = ["## Exp B — architecture landscape (Transformer lineages vs matched GPT-2)", "",
           "The completed study found a large small-scale *algorithm* CEG for the "
           "current-arch speedrun (13.7× @124M). Exp B is the direct test of whether that "
           "generalizes beyond a small-scale-optimized speedrun: it trains PUBLISHED "
           "open-model lineages — **Pythia (GPT-NeoX, 2023)** and **SmolLM2 (Llama, 2024)** "
           "— from scratch on fixed data (OpenWebText), each against a size-matched GPT-2 "
           "baseline through the identical harness, and asks whether any reaches (crosses) "
           "the GPT-2 baseline's neutral-BPB threshold. Data is held fixed → no "
           "data/algorithm 2×2; this is algorithm-CEG only.", "",
           "**Result: no lineage crosses at any scale (135M–1.7B) → algorithm-CEG ≤1× "
           "everywhere (no measurable gain over a matched, properly-tuned GPT-2).** Best "
           "case is SmolLM2-135M, whose gap is within same-seed noise of parity (still not "
           "a crossing). A direct empirical 'no' to whether the current-arch small-scale "
           "advantage generalizes to these lineages. Exp B is the censored (open ▽ at 1×) "
           "markers on the algorithm panel of `multipliers_vs_scale.png` above.", "",
           f"Pre-registered verdict rule: |delta| within ±{R['noise_sigma']} neutral BPB of "
           f"the matched GPT-2 = parity-within-noise; ≥{R['sig_2sigma']} (2σ) = significant "
           "deficit. delta = arch tail-mean neutral BPB − its matched GPT-2 (both @512k); "
           ">0 = worse.", "",
           "| Lineage | size | Δ BPB vs matched GPT-2 | algorithm-CEG |",
           "|--|--|--|--|"]
    for lin in ("pythia", "smollm2"):
        if lin not in R:
            continue
        for s in sorted(R[lin], key=lambda k: R[lin][k]["params"]):
            e = R[lin][s]
            conf = " *(divergence-confounded)*" if e.get("confounded") else ""
            out.append(f"| {lin} | {s} | +{e['delta']:.3f}{conf} | ≤1× (censored, no crossing) |")
    out += ["",
            "Pythia is clean at every scale (deficit shrinks 160M→410M then stabilizes "
            "~0.02–0.03). SmolLM2-135M is parity-within-noise (confirmed with a 2nd seed). "
            "SmolLM2-360M/1.7B are deficits but **divergence-confounded**: even at each "
            "size's documented LR they overfit OWT under the fixed ~8.87B budget (neutral "
            "BPB rises off its own minimum while own-val keeps falling), an effect that "
            "grows with size — the deficit verdict is robust (holds on the best/min BPB too) "
            "but the exact magnitude is inflated.", "",
            "Methodology: like-for-like train_hf denominators at every size (a matched GPT-2 "
            "through the SAME harness — verified equivalent to the study's train_old GPT-2 "
            "to within the ±0.013 same-seed noise floor at ALL scales incl 1.4B); the "
            "undertraining regime biases toward parity (not just noise), so all arms are "
            "compared at convergence.", ""]
    out += _expb_core_section(root)
    return out


def _expb_core_section(root):
    """Exp B CORE (downstream tasks): candidate vs its matched GPT-2, per lineage/scale.
    Secondary to BPB. Reads results/core_expb_summary.json. The interesting result is a
    BPB-vs-CORE tension for SmolLM2 (BPB parity/deficit, but a consistent CORE edge)."""
    p = root / "core_expb_summary.json"
    if not p.exists():
        return []
    S = json.loads(p.read_text())["pairs"]
    out = ["### Exp B — CORE downstream tasks (secondary)", "",
           "The BPB/CEG result above is compute-efficiency on a language-modeling bar. As a "
           "downstream-task check we ran the study's CORE suite (11 tasks, limit 500) on all "
           "14 Exp B checkpoints and compared each architecture to its size-matched GPT-2 "
           "through the same harness. CORE is SECONDARY and noisy at limit=500; the gap is "
           "the mean per-task accuracy difference (candidate − matched GPT-2), ±1 stderr.", "",
           "| Lineage | size | CORE mean-acc gap vs GPT-2 | per-task (W/T/L of 11) |",
           "|--|--|--|--|"]
    order = ["pythia-160M", "pythia-410M", "pythia-1.4B", "smollm2-135M", "smollm2-360M", "smollm2-1.7B"]
    for k in order:
        if k not in S:
            continue
        e = S[k]
        lin, sz = k.split("-", 1)
        out.append(f"| {lin} | {sz} | {e['mean_delta']:+.3f} ± {e['mean_delta_stderr']:.3f} "
                   f"({e['sigma']}σ) | {e['wins_ties_losses']} |")
    out += ["",
            "Two things stand out. **Pythia at/below parity, falling with scale** "
            "(−0.009 → −0.022, significant at 1.4B: 9 of 11 tasks lost) — CORE corroborates "
            "its BPB deficit, and more cleanly (monotone in scale). **SmolLM2 modestly ABOVE "
            "parity at every scale** (+0.011 to +0.019, ~1.7σ, winning 7 of 11 tasks at 135M "
            "and 360M) — which DISAGREES with its BPB result (parity at 135M, divergence-"
            "confounded deficit at 360M/1.7B). The most likely reading: SmolLM2's BPB penalty "
            "at larger sizes is OWT-overfitting (own-val improves while neutral BPB rises), "
            "which need not hurt downstream tasks; and even at 135M (no divergence) the "
            "Llama-family design (SwiGLU/GQA/RoPE) buys a small downstream edge that neutral "
            "BPB does not register. This is a directional, secondary signal — limit=500 CORE "
            "is noisy — not a compute-efficiency claim (on BPB/CEG neither lineage beats GPT-2).", ""]
    if (root / "core_expb_delta.png").exists():
        out += ["![Exp B CORE gap vs matched GPT-2, by scale](core_expb_delta.png)", ""]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="report.md")
    args = ap.parse_args()
    root = Path(args.results_dir)

    lines = ["# Data vs. Algorithms: Compute-Equivalent Gain Across Scale",
             "",
             "2x2 grid (old/new data x old/new algorithm) trained at multiple "
             "GPT-2 sizes; compute measured in timed GPU-hours to a fixed "
             "neutral-corpus BPB threshold; savings Shapley-decomposed in "
             "log-compute space.",
             "",
             "**This study reports TWO separate cross-scale curves, never "
             "blended, because the 'new algorithm' is not a single fixed thing "
             "across all scales:**",
             "- **Current-arch curve (124M, 355M):** A1 = the CURRENT "
             "modded-nanoGPT speedrun (SOTA, small-scale-tuned). No reproducible "
             "1.5B recipe exists for it, so this curve STOPS at 355M — the 1.5B "
             "point is a disclosed gap (scaling it up needs an unvalidated "
             "invented architecture; rejected).",
             "- **ScaleUp-arch curve (124M, 1.5B):** A1 = the 2024 ScaleUp "
             "lineage (older, plain transformer), run from its DOCUMENTED "
             "per-size recipe at each point. 355M has no documented "
             "era-appropriate recipe -> disclosed gap (no hand-derived LR).",
             "",
             "The current-arch A1 numbers are v2-canonical (re-derived from "
             "yarn_state reruns after the loader-fidelity fixes).",
             ""]
    results = []
    for size in SIZE_ORDER:
        d = root / size
        if (d / "ceg_newdef.json").exists() or (d / "ceg.json").exists():
            sec, r = size_section(size, d)
            lines += sec
            results.append((size, r))
    if (root / "all_configs.png").exists():
        lines += ["## All configurations", "",
                  "Every arm on one figure: neutral BPB vs GPU-hours for all 16 "
                  "setups (4 scale-points × A0/A1 old/new algorithm × D0/D1 "
                  "old/new data), each panel zoomed to where its arms cross the "
                  "reference BPB. The crossing GPU-hours are the raw material for "
                  "the multipliers below. The ScaleUp A1D0 (new-algo/old-data) "
                  "visibly never crosses at either ScaleUp scale (the censored "
                  "cells).", "",
                  "![all 16 configurations](all_configs.png)", ""]
    if (root / "multipliers_vs_scale.png").exists():
        lines += ["## Both curves at a glance (multiplier summary)", "",
                  "The 16 arms distilled to compute-equivalent-gain multipliers "
                  "for both lineages across scale. The two curves are kept "
                  "strictly separate (different A1 generations); each is missing "
                  "the one scale where it has no validated recipe.", "",
                  "![multipliers vs scale](multipliers_vs_scale.png)", "",
                  "**The two curves are different estimators — not directly "
                  "comparable.** The current-arch points are true compute-Shapley "
                  "values: all four cells cross the threshold, so the data effect "
                  "is averaged over both rows and the algorithm effect over both "
                  "columns (in log-compute, data = ½·[(A0D0−A0D1)+(A1D0−A1D1)], "
                  "algorithm = ½·[(A0D0−A1D0)+(A0D1−A1D1)]). For ScaleUp, A1D0 "
                  "(ScaleUp on OpenWebText) never crosses, so one term in each "
                  "average is censored and no symmetric Shapley exists; the "
                  "plotted ScaleUp points are the single surviving margin — data = "
                  "the A0 (GPT-2) row ratio, algorithm = the D1 (DCLM) column "
                  "ratio. The censoring biases the two axes in **opposite** "
                  "directions:", "",
                  "- **Algorithm** — the censored complement is the old-data (D0) "
                  "column, where ScaleUp is *worse* than GPT-2 (≤1×). A full "
                  "Shapley would average the plotted new-data margin (2.9×→2.3×) "
                  "with a ≤1× term and sit **below** it, so the plotted algorithm "
                  "multiplier **over-states** the balanced value.",
                  "- **Data** — the censored complement is the ScaleUp (A1) row, "
                  "where ScaleUp cannot cross *at all* on old data (an effectively "
                  "unbounded data multiplier). A full Shapley would sit **above** "
                  "the plotted A0-row margin (3.2×→3.4×), so the plotted data "
                  "multiplier **under-states** it.", "",
                  "So the single-margin ScaleUp numbers bound a full Shapley from "
                  "opposite sides on the two axes. (Multipliers are within-"
                  "hardware GPU-hour ratios — current-arch 8-GPU, ScaleUp 5-GPU — "
                  "so the count overhead cancels in each ratio.)", "",
                  "The figure's algorithm panel also carries a THIRD estimator — the "
                  "Exp B architecture lineages (Pythia, SmolLM2) as open ▽ at 1× — which "
                  "are algorithm-CEG vs a matched GPT-2 (data fixed, no data-panel entry) "
                  "and are all censored (≤1×, none cross). See the Exp B section below.", ""]
    lines += ["## Curve 1 — current-arch (124M, 355M)", "",
              "The SOTA modded-nanoGPT speedrun as A1. Direct test of whether the "
              "data/algorithm split is scale-invariant for this algorithm:", ""]
    if len(results) >= 2:
        lines += ["| Size | data x | algorithm x | total x |", "|--|--|--|--|"]
        for size, r in results:
            m = r["multipliers"]
            lines.append(f"| {size} | {m['data']:.2f} | {m['algorithm']:.2f} "
                         f"| {m['total']:.2f} |")
        lines += ["",
                  "**The algorithm advantage decays sharply with scale "
                  "(13.1x -> 4.1x from 124M to 355M).** The 1.5B point is a "
                  "disclosed GAP — no reproducible 1.5B recipe for this arch, and "
                  "scaling it up requires inventing hand-tuned subsystems (U-net "
                  "skip topology) with no reference and no way to validate them.",
                  ""]
    lines += _scaleup_section(root)
    lines += _era_ladder_section(root)
    lines += _expb_section(root)
    # CORE-subset validity gate (secondary metric), if present
    gate_p = root / "core_gate_v2.json"
    if gate_p.exists():
        g = json.loads(gate_p.read_text())
        lines += ["## CORE-subset (secondary, validity-gated)", "",
                  "DCLM CORE is a secondary check; BPB is primary. A task is used "
                  "only where the A0D0 reference clears chance by >=2 sigma at its "
                  "final checkpoint. `lambada_openai` is open-vocabulary (no fixed "
                  "chance) so it never enters the quantitative gate at any scale; "
                  "separately, the current-arch (modded) A1 loader has no logits "
                  "path so its lambada is invalid by construction, whereas the "
                  "ScaleUp-arch A1 arms (1.5B, ScaleUp-124M) use a plain-causal "
                  "adapter with a real logits path and a valid lambada accuracy "
                  "(reported as a diagnostic only). All A1 CORE is from the v2 "
                  "sweep.",
                  ""]
        for scale in ("124M", "355M", "1.5B", "ScaleUp-124M"):
            if scale in g:
                usable = g[scale]["usable_tasks"]
                lines.append(f"- **{scale}**: {len(usable)} usable tasks "
                             f"({', '.join(usable)}).")
        if (root / "core_vs_scale.png").exists():
            lines += ["",
                      "![CORE task accuracy vs model size](core_vs_scale.png)"]
        lines += ["",
                  "Note: `boolq` drops at 124M / ScaleUp-124M (A0D0 sits at chance, "
                  "0.504) but clears from 355M up — a scale effect, not an error. "
                  "Arms cluster closely on the gated tasks (limit=500, near-noisy), "
                  "with slightly more separation at 1.5B but still within the noise "
                  "floor, so no quantitative CORE-based CEG is claimed at any scale; "
                  "it is a sanity gate. The ScaleUp-arch A1 lambada diagnostic rises "
                  "cleanly with scale (acc 0.32 at 124M -> 0.52/0.55 at 1.5B; "
                  "perplexity 55 -> 8), confirming the plain-causal adapter's logits "
                  "path is sound.",
                  ""]
        if (root / "core_arms_by_task.png").exists():
            lines += ["Across all four arms (a qualitative companion to BPB, not "
                      "a second CEG claim), the only recurring hint is new-data "
                      "(D1) arms edging out their old-data (D0) counterparts on "
                      "arc_easy and piqa at every scale — sizable on arc_easy, "
                      "within ~1–2 stderr on piqa; boolq shows no such order, so "
                      "it is task-specific. Error bars (±1 stderr, limit=500) "
                      "overlap widely, so this is directionally suggestive only.",
                      "",
                      "![CORE accuracy across all four arms, per task]"
                      "(core_arms_by_task.png)", ""]

    lines += ["## Methodology notes & confounds",
              "",
              "- Loss metric is bits-per-byte on a fixed, decontaminated Wikipedia "
              "slice (tokenizer-invariant; identical raw text across all runs). "
              "Per-dataset val BPB was logged as same-distribution diagnostics only.",
              "- Decontamination of the eval corpus (13-gram overlap against the "
              "actual tokenized training samples) removed 562/2766 candidate docs "
              "(20.3%): 409 (14.8%) matched OpenWebText — Reddit-curated pages "
              "quote Wikipedia heavily — and a further 153 (5.5%) matched only "
              "DCLM-baseline (Common Crawl carries Wikipedia mirrors). Frozen "
              "corpus: 2,204 docs / 4,425,879 bytes / 1,086,611 GPT-2 tokens, "
              "sha256 cbdd72ac..., identical across every arm, size, and tier.",
              "- Compute is timed GPU-hours on the runs' actual hardware, excluding "
              "kernel-warmup/compile and eval time; raw FLOPs are never compared "
              "across arms (precision differs).",
              "- Hyperparameters are fixed per (algorithm, size) row; verified "
              "identical between data arms with scripts/verify_row_hparams.py.",
              "- Token budget is part of the algorithm bundle: the A1 (modded) "
              "arms train at 2x the upstream-native budget for each size (small "
              "~731M, medium ~4.35B tokens), NOT a fixed cross-arm budget. The "
              "original fixed-18B design (which forced A1D0 to repeat OWT for 2 "
              "epochs) was abandoned after the recipe collapsed under ~50x "
              "schedule stretch. At the 2x-native budgets both A1 arms are "
              "single-pass — well under one epoch of their ~9B+ corpora — so there "
              "is no epoch-repetition confound in any cell.",
              "- Threshold crossings are interpolated in log-compute between "
              "checkpoints (log-spaced, denser early), never snapped.",
              "- Reference-threshold definition: mean neutral BPB over all "
              "checkpoints in the final 10% of A0D0's training (per size). "
              "The single-final-checkpoint variant is reported alongside as "
              "robustness. Why a range exists at 124M: the threshold sits on "
              "the threshold arm's end-of-training plateau, where its curve "
              "is flattest; with the original purely log-spaced checkpoint "
              "schedule the plateau was sampled sparsely, so vertical noise "
              "of order the same-seed rerun floor (~0.01 BPB) translated "
              "into ~20% swings in that arm's compute-to-threshold and hence "
              "~±8% (data) / ~±16% (total) in the multipliers. Fixed going "
              "forward by adding linearly-spaced checkpoints over the final "
              "10% of every arm's schedule; the reported range brackets the "
              "definitional freedom on the pre-fix 124M data.",
              "- DCLM CORE scores are secondary and validity-gated (tasks near "
              "chance at small scale are excluded).",
              ""]
    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out} ({len(results)} size(s))")


if __name__ == "__main__":
    main()
